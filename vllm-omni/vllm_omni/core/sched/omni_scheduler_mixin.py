from __future__ import annotations

import math
import os
import time
from typing import Any

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus, StreamingUpdate

from .kv_eviction_shadow import (
    capture_request_blocks_for_shadow,
    kv_eviction_metadata_enabled,
)

logger = init_logger(__name__)


class OmniSchedulerMixin:
    """Shared scheduler helpers for omni-specific request handling."""

    @staticmethod
    def _omni_env_bool(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _omni_env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; using %d", name, raw, default)
            return default

    @staticmethod
    def _omni_env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; using %.3f", name, raw, default)
            return default

    @staticmethod
    def _omni_kv_transfer_params(request: Request) -> dict | None:
        params = getattr(request, "kv_transfer_params", None)
        return params if isinstance(params, dict) else None

    def _omni_is_scheduler_prefetch(self, request: Request) -> bool:
        params = self._omni_kv_transfer_params(request)
        return bool(params and params.get("omni_scheduler_prefetch"))

    def _omni_gpu_pin_state(self) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        leases = getattr(self, "_omni_gpu_pin_leases", None)
        if leases is None:
            leases = {}
            self._omni_gpu_pin_leases = leases
        stats = getattr(self, "_omni_gpu_pin_stats", None)
        if stats is None:
            stats = {
                "attempted": 0,
                "pinned": 0,
                "released": 0,
                "skipped": 0,
                "release_blocks": 0,
                "pinned_blocks": 0,
            }
            self._omni_gpu_pin_stats = stats
        return leases, stats

    @staticmethod
    def _omni_gpu_pin_key_from_params(params: dict[str, Any]) -> str | None:
        session_id = params.get("omni_prefetch_session_id") or params.get(
            "omni_session_id"
        )
        prefix_hash = params.get("omni_prefetch_committed_prefix_hash") or params.get(
            "omni_matched_prefix_hash"
        ) or params.get("omni_committed_prefix_hash")
        if session_id and prefix_hash:
            return f"{session_id}:{prefix_hash}"
        request_id = params.get("omni_prefetch_request_id")
        if request_id:
            return f"request:{request_id}"
        return None

    def _omni_gpu_pin_used_blocks(self) -> int:
        leases, _ = self._omni_gpu_pin_state()
        return sum(len(lease.get("blocks", ())) for lease in leases.values())

    def _omni_release_gpu_pin(self, key: str, reason: str) -> int:
        leases, stats = self._omni_gpu_pin_state()
        lease = leases.pop(key, None)
        if not lease:
            return 0
        blocks = list(lease.get("blocks", ()))
        if not blocks:
            return 0
        try:
            self.kv_cache_manager.block_pool.free_blocks(blocks)
        except Exception:
            logger.exception("[OmniGpuPin] failed to release key=%s reason=%s", key, reason)
            return 0
        stats["released"] += 1
        stats["release_blocks"] += len(blocks)
        logger.info(
            "[OmniGpuPin] released key=%s blocks=%d reason=%s active_leases=%d",
            key,
            len(blocks),
            reason,
            len(leases),
        )
        return len(blocks)

    def _omni_release_expired_gpu_pins(self) -> int:
        leases, _ = self._omni_gpu_pin_state()
        now = time.monotonic()
        expired = [
            key
            for key, lease in leases.items()
            if float(lease.get("expires_monotonic", 0.0)) <= now
        ]
        released = 0
        for key in expired:
            released += self._omni_release_gpu_pin(key, "ttl_expired")
        return released

    def _omni_collect_request_gpu_blocks(self, request_id: str) -> list[Any]:
        try:
            kv_blocks = self.kv_cache_manager.get_blocks(request_id)
        except Exception:
            logger.exception("[OmniGpuPin] failed to get KV blocks for %s", request_id)
            return []
        out: list[Any] = []
        for group in getattr(kv_blocks, "blocks", ()) or ():
            for block in group:
                if getattr(block, "is_null", False):
                    continue
                if getattr(block, "block_hash", None) is None:
                    continue
                out.append(block)
        return out

    def _omni_maybe_release_live_gpu_pin(self, request: Request) -> None:
        params = self._omni_kv_transfer_params(request)
        if not params or not params.get("omni_prefetch_hit_used"):
            return
        key = self._omni_gpu_pin_key_from_params(params)
        if key:
            self._omni_release_gpu_pin(key, "live_finish")

    def _omni_maybe_pin_prefetch_request(self, request: Request) -> None:
        params = self._omni_kv_transfer_params(request)
        if not params:
            return
        if not params.get("omni_scheduler_prefetch"):
            self._omni_maybe_release_live_gpu_pin(request)
            return
        if not self._omni_env_bool("OMNI_PREFETCH_GPU_PIN", False):
            return
        if not params.get("omni_prefetch_gpu_pin"):
            return
        if (
            getattr(request, "status", None) == RequestStatus.FINISHED_ABORTED
            and not self._omni_env_bool("OMNI_PREFETCH_GPU_PIN_ABORTED", False)
        ):
            leases, stats = self._omni_gpu_pin_state()
            stats["skipped"] += 1
            return

        key = self._omni_gpu_pin_key_from_params(params)
        if key is None:
            _, stats = self._omni_gpu_pin_state()
            stats["skipped"] += 1
            return

        leases, stats = self._omni_gpu_pin_state()
        stats["attempted"] += 1
        if key in leases:
            self._omni_release_gpu_pin(key, "replace")

        blocks = self._omni_collect_request_gpu_blocks(request.request_id)
        if not blocks:
            stats["skipped"] += 1
            logger.info(
                "[OmniGpuPin] skipped key=%s request=%s reason=no_cached_blocks",
                key,
                request.request_id,
            )
            return

        max_blocks = self._omni_env_int("OMNI_PREFETCH_GPU_PIN_MAX_BLOCKS", 4096)
        used_blocks = self._omni_gpu_pin_used_blocks()
        if max_blocks > 0 and used_blocks + len(blocks) > max_blocks:
            stats["skipped"] += 1
            logger.info(
                "[OmniGpuPin] skipped key=%s request=%s reason=budget blocks=%d "
                "used=%d max=%d",
                key,
                request.request_id,
                len(blocks),
                used_blocks,
                max_blocks,
            )
            return

        ttl_ms = params.get("omni_prefetch_gpu_pin_ttl_ms")
        try:
            ttl_ms_f = float(ttl_ms)
        except (TypeError, ValueError):
            ttl_ms_f = float("nan")
        if not math.isfinite(ttl_ms_f):
            ttl_ms_f = self._omni_env_float("OMNI_PREFETCH_GPU_PIN_TTL_MS", 5000.0)
        if not math.isfinite(ttl_ms_f):
            # Infinite expiry would retain GPU block references indefinitely.
            ttl_ms_f = 5000.0
        ttl_s = max(0.001, ttl_ms_f / 1000.0)

        try:
            self.kv_cache_manager.block_pool.touch(blocks)
        except Exception:
            stats["skipped"] += 1
            logger.exception(
                "[OmniGpuPin] failed to pin key=%s request=%s", key, request.request_id
            )
            return

        leases[key] = {
            "request_id": request.request_id,
            "session_id": params.get("omni_prefetch_session_id"),
            "prefix_hash": params.get("omni_prefetch_committed_prefix_hash"),
            "scope": params.get("omni_prefetch_prefetch_scope"),
            "blocks": blocks,
            "created_monotonic": time.monotonic(),
            "expires_monotonic": time.monotonic() + ttl_s,
        }
        stats["pinned"] += 1
        stats["pinned_blocks"] += len(blocks)
        logger.info(
            "[OmniGpuPin] pinned key=%s request=%s blocks=%d ttl_ms=%.1f "
            "active_leases=%d active_blocks=%d",
            key,
            request.request_id,
            len(blocks),
            ttl_ms_f,
            len(leases),
            self._omni_gpu_pin_used_blocks(),
        )

    def _omni_maybe_capture_kv_evict_shadow(self, request: Request) -> None:
        if not kv_eviction_metadata_enabled():
            return
        if bool(getattr(request, "_omni_kv_evict_shadow_captured", False)):
            return
        try:
            kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
        except Exception:
            logger.exception(
                "[OmniKVEvictShadow] failed to get KV blocks for %s",
                request.request_id,
            )
            return
        block_groups = getattr(kv_blocks, "blocks", ()) or ()
        if not any(block_groups):
            return
        block_size = getattr(self.kv_cache_manager, "block_size", None)
        if block_size is None:
            block_size = getattr(getattr(self, "cache_config", None), "block_size", None)
        if block_size is None:
            block_size = getattr(
                getattr(self, "scheduler_config", None),
                "block_size",
                None,
            )
        capture_request_blocks_for_shadow(
            self.kv_cache_manager.block_pool,
            request,
            block_groups,
            block_size,
        )
        try:
            setattr(request, "_omni_kv_evict_shadow_captured", True)
        except Exception:
            pass

    def _omni_live_pressure_count(self) -> int:
        """Count live requests contending with scheduler-native preload."""
        count = 0
        for queue_name in ("waiting", "skipped_waiting"):
            queue = getattr(self, queue_name, None)
            if queue:
                count += sum(
                    1 for req in queue if not self._omni_is_scheduler_prefetch(req)
                )
        running = getattr(self, "running", None) or []
        count += sum(1 for req in running if not self._omni_is_scheduler_prefetch(req))
        return count

    def _omni_cancel_prefetch_under_live_pressure(self) -> int:
        """Abort scheduler-native preload when live work is present.

        This is intentionally conservative. The request was already admitted by
        the API-side joint policy, but live/barge-in traffic can arrive after
        admission. Once that happens, the scheduler is the right layer to drop
        background preload before it competes for token or KV-transfer budget.
        """
        if not self._omni_env_bool("OMNI_SCHED_NATIVE_PREFETCH", False):
            return 0
        if not self._omni_env_bool("OMNI_SCHED_NATIVE_PREFETCH_CANCEL", True):
            return 0

        max_live = self._omni_env_int("OMNI_SCHED_NATIVE_PREFETCH_MAX_LIVE", 0)
        live_count = self._omni_live_pressure_count()
        if live_count <= max_live:
            return 0

        candidates: list[Request] = []
        for queue_name in ("waiting", "skipped_waiting"):
            queue = getattr(self, queue_name, None)
            if queue:
                candidates.extend(
                    req for req in list(queue) if self._omni_is_scheduler_prefetch(req)
                )
        running = getattr(self, "running", None) or []
        candidates.extend(
            req for req in list(running) if self._omni_is_scheduler_prefetch(req)
        )
        if not candidates:
            return 0

        request_ids = [req.request_id for req in candidates]
        try:
            self.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
        except Exception:
            logger.exception(
                "[OmniPrefetchScheduler] failed to cancel prefetch ids=%s",
                request_ids,
            )
            return 0

        total = int(getattr(self, "_omni_scheduler_prefetch_cancelled", 0)) + len(
            request_ids
        )
        self._omni_scheduler_prefetch_cancelled = total
        logger.info(
            "[OmniPrefetchScheduler] cancelled %d prefetch request(s) under "
            "live_pressure=%d max_live=%d total_cancelled=%d",
            len(request_ids),
            live_count,
            max_live,
            total,
        )
        return len(request_ids)

    def _replace_session_with_streaming_update(
        self,
        session: Request,
        update: StreamingUpdate,
    ) -> None:
        """For streaming input: Replace an existing streaming session payload with the latest update."""
        session._output_token_ids.clear()
        session._all_token_ids.clear()
        new_prompt = update.prompt_token_ids or ()
        session._all_token_ids.extend(new_prompt)
        session.num_computed_tokens = 0
        session.prompt_token_ids = update.prompt_token_ids or ()
        session.additional_information = update.additional_information or None
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)
