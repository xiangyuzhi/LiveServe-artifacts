import atexit
import importlib.util
import json
import os
import sys
import time
from collections import OrderedDict
from functools import cached_property
from pathlib import Path
from threading import Lock
from typing import Any

from aenum import extend_enum
from vllm.config import ModelConfig as _OriginalModelConfig
from vllm.inputs import TokensPrompt as _OriginalTokensPrompt
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding as _OriginalMRotaryEmbedding,
)
from vllm.v1.engine import EngineCoreOutput as _OriginalEngineCoreOutput
from vllm.v1.engine import EngineCoreOutputs as _OriginalEngineCoreOutputs
from vllm.v1.engine import EngineCoreRequest as _OriginalEngineCoreRequest
from vllm.v1.request import Request as _OriginalRequest
from vllm.v1.request import RequestStatus
from vllm.v1.request import StreamingUpdate as _OriginalStreamingUpdate

import vllm_omni.logger  # noqa: F401
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs, OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.layers.rotary_embedding import OmniMRotaryEmbedding
from vllm_omni.request import OmniRequest, OmniStreamingUpdate

# =============================================================================
# Patch ModelConfig.is_mm_prefix_lm to support omni-specific models
# =============================================================================
# WHY: HunyuanImage-3.0 requires bidirectional attention for image tokens
# (cond_token_attn_type: "joint_full" in config.json). vLLM gates this on
# is_mm_prefix_lm, which checks an internal MM_PREFIX_LM_MODELS list that
# does not include "hunyuan_image_3_moe" (the upstream HF model_type).
#
# WHY NOT model-level: is_mm_prefix_lm is checked in vLLM core (scheduler,
# attention backend selection) before model code runs — no model-level hook.
#
# SCOPE: Only affects model_type in _OMNI_MM_PREFIX_LM_MODELS (currently
# just "hunyuan_image_3_moe"). All other models fall through to the
# original vLLM implementation unchanged.
#
# FRAGILITY: Relies on is_mm_prefix_lm being a cached_property on
# ModelConfig. The __dict__ access + __set_name__ dance works around a
# pydantic dataclass issue in vllm 0.19.0+. If vLLM changes
# is_mm_prefix_lm to a regular method or removes it, this will break.
#
# TODO: Upstream a configurable MM_PREFIX_LM_MODELS or a model_config flag
# so this patch can be removed.
_OMNI_MM_PREFIX_LM_MODELS = ("hunyuan_image_3_moe",)
# Access via __dict__ to avoid triggering cached_property.__get__ which fails
# with "Cannot use cached_property instance without calling __set_name__" in
# pydantic dataclasses (vllm 0.19.0+).
_cp = _OriginalModelConfig.__dict__["is_mm_prefix_lm"]
_original_is_mm_prefix_lm = _cp.func if hasattr(_cp, "func") else _cp.fget


def _patched_is_mm_prefix_lm(self):
    if _original_is_mm_prefix_lm(self):
        return True
    model_type = getattr(self.hf_config, "model_type", "")
    return model_type in _OMNI_MM_PREFIX_LM_MODELS


_patched_cp = cached_property(_patched_is_mm_prefix_lm)
_patched_cp.__set_name__(_OriginalModelConfig, "is_mm_prefix_lm")
_OriginalModelConfig.is_mm_prefix_lm = _patched_cp

# Sanity check: verify the patch is active. If vLLM changes the descriptor
# type or __set_name__ semantics, this will fail loudly at import time
# rather than silently falling back to unpatched behavior.
_installed = _OriginalModelConfig.__dict__.get("is_mm_prefix_lm")
assert _installed is _patched_cp, (
    "is_mm_prefix_lm patch failed to install — bidirectional attention "
    "for HunyuanImage3 will not work. Check vLLM ModelConfig changes."
)

# =============================================================================
# Patch GlmImageTextConfig to expose mrope_section in rope_parameters
# =============================================================================
# GLM-Image uses M-RoPE with mrope_section: [8, 12, 12], but transformers'
# implementation doesn't expose it in rope_parameters. vLLM's uses_mrope
# detection relies on "mrope_section" being present in rope_parameters.
# This patch ensures proper M-RoPE detection for GLM-Image.
try:
    from transformers.models.glm_image.configuration_glm_image import GlmImageTextConfig

    _original_glm_image_text_config_init = GlmImageTextConfig.__init__

    def _patched_glm_image_text_config_init(self, *args, **kwargs):
        _original_glm_image_text_config_init(self, *args, **kwargs)
        # Ensure rope_parameters exists and contains mrope_section
        if self.rope_parameters is None:
            self.rope_parameters = {}
        if isinstance(self.rope_parameters, dict) and "mrope_section" not in self.rope_parameters:
            # GLM-Image uses mrope_section: [8, 12, 12] for T/H/W dimensions
            self.rope_parameters["mrope_section"] = [8, 12, 12]

    GlmImageTextConfig.__init__ = _patched_glm_image_text_config_init
except ImportError:
    # GlmImageTextConfig not available, skip patching
    pass

# Extend RequestStatus enum with omni-specific statuses
if not hasattr(RequestStatus, "WAITING_FOR_CHUNK"):
    # The value - 1 is intentionally chosen to ensure it is treated
    # as a non-finished state and remains compatible with existing comparisons.
    extend_enum(RequestStatus, "WAITING_FOR_CHUNK", -1)

# Snapshot sys.modules: `hasattr` below can trigger lazy submodule imports
# (e.g. transformers' `_LazyModule.__getattr__`), which mutate sys.modules
# during iteration and raise `dictionary changed size during iteration`.
for module_name, module in list(sys.modules.items()):
    # only do patch on module of vllm, pass others
    if "vllm" not in module_name:
        continue
    if hasattr(module, "EngineCoreOutput") and module.EngineCoreOutput == _OriginalEngineCoreOutput:
        module.EngineCoreOutput = OmniEngineCoreOutput
    if hasattr(module, "EngineCoreOutputs") and module.EngineCoreOutputs == _OriginalEngineCoreOutputs:
        module.EngineCoreOutputs = OmniEngineCoreOutputs
    if hasattr(module, "TokensPrompt") and module.TokensPrompt == _OriginalTokensPrompt:
        module.TokensPrompt = OmniTokensPrompt
    if hasattr(module, "MRotaryEmbedding") and module.MRotaryEmbedding == _OriginalMRotaryEmbedding:
        module.MRotaryEmbedding = OmniMRotaryEmbedding
    if hasattr(module, "Request") and module.Request == _OriginalRequest:
        module.Request = OmniRequest
    if hasattr(module, "StreamingUpdate") and module.StreamingUpdate == _OriginalStreamingUpdate:
        module.StreamingUpdate = OmniStreamingUpdate
    if hasattr(module, "EngineCoreRequest") and module.EngineCoreRequest == _OriginalEngineCoreRequest:
        module.EngineCoreRequest = OmniEngineCoreRequest

# =============================================================================
# Patch Scheduler/AsyncScheduler._update_after_schedule against abort race
# =============================================================================
# WHY: Under high concurrency + early-abort load (c=128, abort@t0+0.3s), the
# vLLM v1 scheduler races with engine-core abort message processing:
#
#   step_fn → scheduler.schedule()
#       └─ build scheduler_output (req_id list snapshot)
#       └─ _update_after_schedule(scheduler_output)            <-- iterates
#               └─ request = self.requests[req_id]             <-- KeyError
#
# Between the two phases, an abort RPC delivered via M2.3 path can pop the
# req from self.requests, leaving the snapshot dict referring to a dead key.
# Reproduced 6× in m2_6_keyerr_stress (track_a_multi_replica_…).
#
# Repro stack (verbatim, 4 GPU multi-replica, async_scheduler):
#   File "vllm/v1/core/sched/scheduler.py", line 999
#     request = self.requests[req_id]
#   KeyError: 'chatcmpl-ibench-fixed_time-…'
#
# The race is harmless to correctness — the aborted request will never
# produce more output anyway — but it crashes the StageEngineCoreProc
# (fatal in run_busy_loop) and takes the whole stage offline. After the
# first wave of crashes, all subsequent rounds 100% fail.
#
# WHY NOT fix in vllm-omni scheduler subclass (omni_ar_scheduler):
# 1. The bug is in the parent Scheduler.schedule() codepath, not in the
#    subclass override; fixing only the subclass leaves AsyncScheduler
#    (used by stage0 thinker) exposed.
# 2. The same race exists for any vLLM v1 deployment that supports abort,
#    so a defensive `.get()` is upstream-correct.
#
# SCOPE: Only the inner `self.requests[req_id]` lookup is replaced with
# a `.get()`. All counters/structured-output flags are skipped for
# already-removed reqs (they are about to be GC'd anyway).
#
# FRAGILITY: Relies on the method signature `_update_after_schedule(self,
# scheduler_output)` being stable in vLLM. If upstream renames or refactors
# this method, the patch will silently no-op (we assert the original is
# present at install time so it fails loudly instead).
#
# TODO: Upstream PR adding `req = self.requests.get(req_id); if req is None:
# continue` to vllm/v1/core/sched/scheduler.py and async_scheduler.py.
from vllm.v1.core.sched.async_scheduler import (  # noqa: E402
    AsyncScheduler as _OriginalAsyncScheduler,
)
from vllm.v1.core.sched.scheduler import Scheduler as _OriginalScheduler  # noqa: E402

_orig_sched_update_after_schedule = _OriginalScheduler._update_after_schedule
_orig_async_update_after_schedule = _OriginalAsyncScheduler._update_after_schedule

assert _orig_sched_update_after_schedule is not None, (
    "Scheduler._update_after_schedule missing — abort-race patch cannot install."
)
assert _orig_async_update_after_schedule is not None, (
    "AsyncScheduler._update_after_schedule missing — abort-race patch cannot install."
)

# Process-local counter so we don't spam the log per-step.
_abort_race_skip_count = {"n": 0}


def _patched_sched_update_after_schedule(self, scheduler_output):
    # Mirror upstream Scheduler._update_after_schedule, but tolerate
    # req_ids that have been popped between schedule() and now by an
    # in-flight abort RPC.
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens
    skipped_ids: list[str] = []
    skipped_tokens = 0
    for req_id, num_scheduled_token in num_scheduled_tokens.items():
        request = self.requests.get(req_id)
        if request is None:
            skipped_ids.append(req_id)
            skipped_tokens += num_scheduled_token
            continue
        request.num_computed_tokens += num_scheduled_token
        request.is_prefill_chunk = request.num_computed_tokens < (
            request.num_tokens + request.num_output_placeholders
        )
        scheduler_output.has_structured_output_requests |= (
            request.use_structured_output and not request.is_prefill_chunk
        )
    if skipped_ids:
        # Cull dead req_ids from scheduler_output and decrement
        # total_num_scheduled_tokens accordingly.  Two reasons:
        # 1. Downstream worker code (gpu_ar_model_runner.execute_model) does
        #    `tokens = [num_scheduled_tokens[i] for i in input_batch.req_ids]`
        #    and `num_scheduled_tokens_np.max()`. We must keep
        #    num_scheduled_tokens consistent with input_batch (which has
        #    already removed the aborted reqs via finished_req_ids).
        # 2. core.step_with_batch_queue uses
        #    `model_executed = total_num_scheduled_tokens > 0` to decide
        #    whether to call sample_tokens(); if total stays positive but
        #    the actual batch is empty, sample_tokens() returns None and
        #    triggers `RuntimeError("unexpected error")` at core.py:526.
        for rid in skipped_ids:
            num_scheduled_tokens.pop(rid, None)
            if hasattr(scheduler_output, "scheduled_spec_decode_tokens"):
                scheduler_output.scheduled_spec_decode_tokens.pop(rid, None)
        try:
            scheduler_output.total_num_scheduled_tokens -= skipped_tokens
            if scheduler_output.total_num_scheduled_tokens < 0:
                scheduler_output.total_num_scheduled_tokens = 0
        except Exception:
            # total_num_scheduled_tokens is a normal int field on
            # SchedulerOutput; if a future vLLM version makes it a
            # property, the patch becomes a no-op for the count but
            # the dict cull above still keeps things consistent.
            pass
        _abort_race_skip_count["n"] += len(skipped_ids)
        # Log at most once per ~100 skips to avoid log flooding.
        if _abort_race_skip_count["n"] % 100 < len(skipped_ids):
            try:
                from vllm.logger import init_logger as _ilog

                _ilog(__name__).warning(
                    "[abort-race-patch] scheduler culled %d req_id(s) already "
                    "aborted between schedule() and the post-schedule update "
                    "(cumulative=%d, last=%s, freed_tokens=%d).",
                    len(skipped_ids),
                    _abort_race_skip_count["n"],
                    skipped_ids[-1],
                    skipped_tokens,
                )
            except Exception:
                pass
    self.finished_req_ids = set()


def _patched_async_update_after_schedule(self, scheduler_output):
    # Run the (patched) parent first to advance counters & cull dead ids.
    _patched_sched_update_after_schedule(self, scheduler_output)
    # Then do AsyncScheduler-specific spec-decode placeholder bookkeeping,
    # also defensive against late aborts.
    spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
    for req_id in scheduler_output.num_scheduled_tokens:
        request = self.requests.get(req_id)
        if request is None:
            continue
        if request.is_prefill_chunk:
            continue
        scheduler_output.pending_structured_output_tokens |= (
            request.use_structured_output and request.num_output_placeholders > 0
        )
        cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
        request.num_output_placeholders += 1 + cur_num_spec_tokens
        request.spec_token_ids = self._spec_token_placeholders


_OriginalScheduler._update_after_schedule = _patched_sched_update_after_schedule
_OriginalAsyncScheduler._update_after_schedule = _patched_async_update_after_schedule

# Sanity check at import time.
assert _OriginalScheduler._update_after_schedule is _patched_sched_update_after_schedule, (
    "Scheduler._update_after_schedule abort-race patch failed to install."
)
assert (
    _OriginalAsyncScheduler._update_after_schedule is _patched_async_update_after_schedule
), "AsyncScheduler._update_after_schedule abort-race patch failed to install."


# =============================================================================
# Patch OffloadingConnectorScheduler to support interactive fast bypass
# =============================================================================
# WHY: In interactive KV prefetch experiments, most live turns are cold misses or
# admission-skipped sessions. Those live requests should not pay the CPU-offload
# lookup/store tax unless the prefetch registry has a known warm prefix.
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler as _OriginalOffloadingConnectorScheduler,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        RequestOffloadState as _OriginalRequestOffloadState,
    )
except Exception:  # pragma: no cover - connector may be absent in some builds.
    _OriginalOffloadingConnectorScheduler = None
    _OriginalRequestOffloadState = None


def _omni_should_skip_offloading_connector(request: Any) -> bool:
    params = getattr(request, "kv_transfer_params", None)
    return isinstance(params, dict) and bool(params.get("omni_skip_offloading_connector"))


if (
    _OriginalOffloadingConnectorScheduler is not None
    and not getattr(
        _OriginalOffloadingConnectorScheduler.get_num_new_matched_tokens,
        "_omni_fast_bypass_patch",
        False,
    )
):
    _original_offloading_get_num_new_matched_tokens = (
        _OriginalOffloadingConnectorScheduler.get_num_new_matched_tokens
    )

    def _patched_offloading_get_num_new_matched_tokens(
        self,
        request,
        num_computed_tokens: int,
    ):
        if _omni_should_skip_offloading_connector(request):
            req_status = self._req_status.get(request.request_id)
            if req_status is None:
                req_status = _OriginalRequestOffloadState(config=self.config, req=request)
                self._req_status[request.request_id] = req_status
            req_status.update_offload_keys()
            req_status.num_locally_computed_tokens = num_computed_tokens
            req_status.advance_stored_idx(request.num_tokens)
            return 0, False
        return _original_offloading_get_num_new_matched_tokens(
            self,
            request,
            num_computed_tokens,
        )

    _patched_offloading_get_num_new_matched_tokens._omni_fast_bypass_patch = True
    _OriginalOffloadingConnectorScheduler.get_num_new_matched_tokens = (
        _patched_offloading_get_num_new_matched_tokens
    )


# =============================================================================
# Optional KV LRU diagnostics for cache-hit profiling
# =============================================================================
# The hook is disabled by default. When enabled, it records process-local
# counters and a bounded ghost set of recently evicted block hashes so replay
# summaries can distinguish ordinary prefix misses from recent LRU evictions.

def _omni_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _omni_env_policy_enabled(name: str) -> bool:
    raw = os.environ.get(name, "")
    return raw.strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
        "none",
        "lru",
        "actual_lru",
    }


def _omni_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _omni_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


if _omni_env_bool("OMNI_KV_LRU_STATS", False):
    try:
        from vllm.v1.core.block_pool import BlockPool as _OriginalBlockPool
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
    except Exception:  # pragma: no cover - vLLM core can vary across installs.
        _OriginalBlockPool = None
        make_block_hash_with_group_id = None

    _omni_kv_lru_lock = Lock()
    _omni_kv_lru_ghost_size = max(1, _omni_env_int("OMNI_KV_LRU_GHOST_SIZE", 65536))
    _omni_kv_lru_flush_interval = max(
        0.0, _omni_env_float("OMNI_KV_LRU_FLUSH_INTERVAL_SEC", 1.0)
    )
    _omni_kv_lru_lookup_dedup_sec = max(
        0.0, _omni_env_float("OMNI_KV_LRU_STATS_LOOKUP_DEDUP_SEC", 0.0)
    )
    _omni_kv_lru_lookup_dedup_size = max(
        1, _omni_env_int("OMNI_KV_LRU_STATS_LOOKUP_DEDUP_SIZE", 65536)
    )
    _omni_kv_lru_request_stats_enabled = _omni_env_bool(
        "OMNI_KV_LRU_REQUEST_STATS", False
    )
    _omni_kv_lru_request_stats_size = max(
        1, _omni_env_int("OMNI_KV_LRU_REQUEST_STATS_SIZE", 65536)
    )
    _omni_kv_lru_stats_path = os.environ.get("OMNI_KV_LRU_STATS_PATH")
    _omni_kv_lru_events_path = os.environ.get("OMNI_KV_LRU_STATS_EVENTS_PATH")
    _omni_kv_lru_request_stats_path = os.environ.get(
        "OMNI_KV_LRU_REQUEST_STATS_PATH"
    )
    _omni_kv_lru_last_flush = {"t": 0.0}
    _omni_kv_lru_ghost: OrderedDict[str, dict[str, Any]] = OrderedDict()
    _omni_kv_lru_lookup_seen: OrderedDict[str, float] = OrderedDict()
    _omni_kv_lru_request_stats: OrderedDict[str, dict[str, Any]] = OrderedDict()
    _omni_kv_lru_request_stats_dirty: set[str] = set()
    _omni_kv_lru_counters: dict[str, int] = {
        "gpu_prefix_block_lookup_total": 0,
        "gpu_prefix_block_lookup_hit": 0,
        "gpu_prefix_block_lookup_miss": 0,
        "gpu_prefix_block_lookup_recent_eviction_miss": 0,
        "gpu_prefix_block_lookup_dedup_skipped_total": 0,
        "gpu_lru_eviction_blocks_total": 0,
        "gpu_prefix_block_touch_total": 0,
        "gpu_prefix_block_rescued_from_free_queue_total": 0,
        "cpu_offload_prefix_block_lookup_total": 0,
        "cpu_offload_prefix_block_lookup_hit": 0,
        "cpu_offload_prefix_block_lookup_miss": 0,
        "cpu_offload_prefix_block_lookup_recent_eviction_miss": 0,
        "cpu_offload_prefix_block_lookup_dedup_skipped_total": 0,
        "cpu_offload_lru_eviction_blocks_total": 0,
        "cpu_offload_prefix_block_touch_total": 0,
        "cpu_offload_prefix_block_rescued_from_free_queue_total": 0,
    }

    def _omni_kv_lru_install_current_request_meta_patch() -> None:
        helper_path = Path(__file__).parent / "core" / "sched" / "kv_eviction_shadow.py"
        spec = importlib.util.spec_from_file_location(
            "vllm_omni_kv_lru_current_meta_patch",
            helper_path,
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        install = getattr(module, "install_current_request_meta_patch", None)
        if callable(install):
            install()

    if _omni_kv_lru_events_path or _omni_kv_lru_request_stats_enabled:
        _omni_kv_lru_install_current_request_meta_patch()

    def _omni_kv_lru_pool_kind(pool: Any) -> str:
        return "cpu" if getattr(pool, "_omni_kv_lru_pool_kind", "") == "cpu" else "gpu"

    def _omni_kv_lru_counter(kind: str, suffix: str) -> str:
        if kind == "cpu":
            return f"cpu_offload_{suffix}"
        return f"gpu_{suffix}"

    def _omni_kv_lru_key(kind: str, block_hash: Any) -> str:
        return f"{kind}:{repr(block_hash)}"

    def _omni_kv_lru_current_request_meta(pool: Any) -> dict[str, Any]:
        meta = getattr(pool, "_omni_kv_evict_current_request_meta", None)
        return meta if isinstance(meta, dict) else {}

    def _omni_kv_lru_lookup_key(
        kind: str,
        block_hash: Any,
        group_ids: list[int],
    ) -> str:
        return f"{kind}:{repr(block_hash)}:{','.join(str(g) for g in group_ids)}"

    def _omni_kv_lru_inc(key: str, amount: int = 1) -> None:
        if amount <= 0:
            return
        _omni_kv_lru_counters[key] = _omni_kv_lru_counters.get(key, 0) + amount

    def _omni_kv_lru_request_id(meta: dict[str, Any]) -> str | None:
        value = meta.get("request_id")
        if value is None:
            return None
        request_id = str(value)
        return request_id or None

    def _omni_kv_lru_inc_request(
        meta: dict[str, Any],
        values: dict[str, int],
    ) -> None:
        if not _omni_kv_lru_request_stats_enabled:
            return
        request_id = _omni_kv_lru_request_id(meta)
        if not request_id:
            return
        row = _omni_kv_lru_request_stats.get(request_id)
        now = time.time()
        if row is None:
            row = {
                "request_id": request_id,
                "first_ts": now,
            }
            for meta_key in (
                "omni_session_id",
                "omni_committed_prefix_hash",
                "omni_prefix_token_estimate",
                "omni_prefetch_scope",
            ):
                meta_value = meta.get(meta_key)
                if meta_value is not None:
                    row[meta_key] = meta_value
            _omni_kv_lru_request_stats[request_id] = row
        row["last_ts"] = now
        for key, amount in values.items():
            if amount <= 0:
                continue
            row[key] = int(row.get(key, 0) or 0) + int(amount)
        _omni_kv_lru_request_stats_dirty.add(request_id)
        _omni_kv_lru_request_stats.move_to_end(request_id)
        while len(_omni_kv_lru_request_stats) > _omni_kv_lru_request_stats_size:
            old_request_id, _ = _omni_kv_lru_request_stats.popitem(last=False)
            _omni_kv_lru_request_stats_dirty.discard(old_request_id)

    def _omni_kv_lru_get_request_stats(request_id: Any) -> dict[str, Any] | None:
        if not _omni_kv_lru_request_stats_enabled or request_id is None:
            return None
        with _omni_kv_lru_lock:
            stats = _omni_kv_lru_request_stats.get(str(request_id))
            return dict(stats) if isinstance(stats, dict) else None

    try:
        import builtins as _omni_kv_lru_builtins

        _omni_kv_lru_builtins._omni_kv_lru_get_request_stats = (
            _omni_kv_lru_get_request_stats
        )
    except Exception:
        pass

    def _omni_kv_lru_write_event(row: dict[str, Any]) -> None:
        if not _omni_kv_lru_events_path:
            return
        event = dict(row)
        event.setdefault("ts", time.time())
        event.setdefault("pid", os.getpid())
        event.setdefault("ppid", os.getppid())
        try:
            parent = os.path.dirname(_omni_kv_lru_events_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(_omni_kv_lru_events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        except Exception:
            pass

    def _omni_kv_lru_flush_request_stats(force: bool = False) -> None:
        if (
            not _omni_kv_lru_request_stats_enabled
            or not _omni_kv_lru_request_stats_path
        ):
            return
        if force:
            request_ids = list(_omni_kv_lru_request_stats)
        else:
            request_ids = list(_omni_kv_lru_request_stats_dirty)
        if not request_ids:
            return
        _omni_kv_lru_request_stats_dirty.difference_update(request_ids)
        now = time.time()
        rows = []
        for request_id in request_ids:
            stats = _omni_kv_lru_request_stats.get(request_id)
            if not isinstance(stats, dict):
                continue
            row = dict(stats)
            row.setdefault("request_id", request_id)
            row["kind"] = "kv_lru_request_stats"
            row["ts"] = now
            row["pid"] = os.getpid()
            row["ppid"] = os.getppid()
            rows.append(row)
        if not rows:
            return
        try:
            parent = os.path.dirname(_omni_kv_lru_request_stats_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(
                _omni_kv_lru_request_stats_path,
                "a",
                encoding="utf-8",
            ) as f:
                for row in rows:
                    f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        except Exception:
            pass

    def _omni_kv_lru_remember(kind: str, block_hash: Any, block_id: int | None) -> None:
        key = _omni_kv_lru_key(kind, block_hash)
        _omni_kv_lru_ghost[key] = {
            "ts": time.time(),
            "block_id": block_id,
        }
        _omni_kv_lru_ghost.move_to_end(key)
        while len(_omni_kv_lru_ghost) > _omni_kv_lru_ghost_size:
            _omni_kv_lru_ghost.popitem(last=False)

    def _omni_kv_lru_recent(kind: str, block_hash: Any) -> bool:
        key = _omni_kv_lru_key(kind, block_hash)
        if key not in _omni_kv_lru_ghost:
            return False
        _omni_kv_lru_ghost.move_to_end(key)
        return True

    def _omni_kv_lru_skip_repeated_lookup(
        kind: str,
        block_hash: Any,
        group_ids: list[int],
        now: float,
    ) -> bool:
        if _omni_kv_lru_lookup_dedup_sec <= 0.0:
            return False
        key = _omni_kv_lru_lookup_key(kind, block_hash, group_ids)
        last = _omni_kv_lru_lookup_seen.get(key)
        if last is not None and now - last < _omni_kv_lru_lookup_dedup_sec:
            _omni_kv_lru_lookup_seen[key] = now
            _omni_kv_lru_lookup_seen.move_to_end(key)
            _omni_kv_lru_inc(
                _omni_kv_lru_counter(
                    kind,
                    "prefix_block_lookup_dedup_skipped_total",
                ),
                len(group_ids),
            )
            return True
        _omni_kv_lru_lookup_seen[key] = now
        _omni_kv_lru_lookup_seen.move_to_end(key)
        while len(_omni_kv_lru_lookup_seen) > _omni_kv_lru_lookup_dedup_size:
            _omni_kv_lru_lookup_seen.popitem(last=False)
        return False

    def _omni_kv_lru_pool_extra_counters(pool: Any | None) -> dict[str, int | float]:
        if pool is None:
            return {}
        kind = _omni_kv_lru_pool_kind(pool)
        out: dict[str, int | float] = {}
        for attr, prefix in (
            ("_omni_kv_evict_shadow_stats", "evict_shadow"),
            ("_omni_kv_evict_enforce_stats", "evict_enforce"),
            ("_omni_kv_evict_index_state", "evict_index"),
        ):
            values = getattr(pool, attr, None)
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                out[f"{kind}_{prefix}_{key}"] = value
        return out

    def _omni_kv_lru_flush(force: bool = False, pool: Any | None = None) -> None:
        if not _omni_kv_lru_stats_path and not _omni_kv_lru_request_stats_path:
            return
        now = time.time()
        if (
            not force
            and _omni_kv_lru_flush_interval > 0
            and now - _omni_kv_lru_last_flush["t"] < _omni_kv_lru_flush_interval
        ):
            return
        _omni_kv_lru_last_flush["t"] = now
        if _omni_kv_lru_stats_path:
            counters = dict(_omni_kv_lru_counters)
            counters.update(_omni_kv_lru_pool_extra_counters(pool))
            row = {
                "kind": "gpu_lru_stats",
                "ts": now,
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "ghost_size": len(_omni_kv_lru_ghost),
                "ghost_capacity": _omni_kv_lru_ghost_size,
                "lookup_dedup_sec": _omni_kv_lru_lookup_dedup_sec,
                "lookup_dedup_size": _omni_kv_lru_lookup_dedup_size,
                "lookup_dedup_entries": len(_omni_kv_lru_lookup_seen),
                "counters": counters,
            }
            try:
                parent = os.path.dirname(_omni_kv_lru_stats_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(_omni_kv_lru_stats_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
            except Exception:
                pass
        _omni_kv_lru_flush_request_stats(force=force)

    if (
        _OriginalBlockPool is not None
        and make_block_hash_with_group_id is not None
        and not getattr(
            _OriginalBlockPool.get_cached_block,
            "_omni_kv_lru_stats_patch",
            False,
        )
    ):
        _original_block_pool_get_cached_block = _OriginalBlockPool.get_cached_block
        _original_block_pool_maybe_evict = _OriginalBlockPool._maybe_evict_cached_block
        _original_block_pool_touch = _OriginalBlockPool.touch

        def _patched_block_pool_get_cached_block(
            self,
            block_hash,
            kv_cache_group_ids,
        ):
            group_ids = list(kv_cache_group_ids)
            result = _original_block_pool_get_cached_block(
                self,
                block_hash,
                group_ids,
            )
            try:
                kind = _omni_kv_lru_pool_kind(self)
                total = len(group_ids)
                hits = 0
                recent = 0
                with _omni_kv_lru_lock:
                    now = time.time()
                    if _omni_kv_lru_skip_repeated_lookup(
                        kind,
                        block_hash,
                        group_ids,
                        now,
                    ):
                        request_meta = _omni_kv_lru_current_request_meta(self)
                        _omni_kv_lru_inc_request(
                            request_meta,
                            {
                                _omni_kv_lru_counter(
                                    kind,
                                    "prefix_block_lookup_dedup_skipped_total",
                                ): total,
                            },
                        )
                        _omni_kv_lru_write_event({
                            "kind": "kv_lru_lookup",
                            "pool_kind": kind,
                            "block_hash": repr(block_hash),
                            "group_ids": group_ids,
                            "total": total,
                            "hits": 0,
                            "misses": 0,
                            "recent_eviction_misses": 0,
                            "dedup_skipped": True,
                            "request": request_meta,
                        })
                        _omni_kv_lru_flush(pool=self)
                        return result
                    for group_id in group_ids:
                        block_hash_with_group_id = make_block_hash_with_group_id(
                            block_hash,
                            group_id,
                        )
                        block = self.cached_block_hash_to_block.get_one_block(
                            block_hash_with_group_id
                        )
                        if block is None:
                            if _omni_kv_lru_recent(kind, block_hash_with_group_id):
                                recent += 1
                        else:
                            hits += 1
                    misses = max(0, total - hits)
                    _omni_kv_lru_inc(
                        _omni_kv_lru_counter(kind, "prefix_block_lookup_total"),
                        total,
                    )
                    _omni_kv_lru_inc(
                        _omni_kv_lru_counter(kind, "prefix_block_lookup_hit"),
                        hits,
                    )
                    _omni_kv_lru_inc(
                        _omni_kv_lru_counter(kind, "prefix_block_lookup_miss"),
                        misses,
                    )
                    _omni_kv_lru_inc(
                        _omni_kv_lru_counter(
                            kind,
                            "prefix_block_lookup_recent_eviction_miss",
                        ),
                        recent,
                    )
                    request_meta = _omni_kv_lru_current_request_meta(self)
                    _omni_kv_lru_inc_request(
                        request_meta,
                        {
                            _omni_kv_lru_counter(
                                kind, "prefix_block_lookup_total"
                            ): total,
                            _omni_kv_lru_counter(
                                kind, "prefix_block_lookup_hit"
                            ): hits,
                            _omni_kv_lru_counter(
                                kind, "prefix_block_lookup_miss"
                            ): misses,
                            _omni_kv_lru_counter(
                                kind,
                                "prefix_block_lookup_recent_eviction_miss",
                            ): recent,
                        },
                    )
                    _omni_kv_lru_write_event({
                        "kind": "kv_lru_lookup",
                        "pool_kind": kind,
                        "block_hash": repr(block_hash),
                        "group_ids": group_ids,
                        "total": total,
                        "hits": hits,
                        "misses": misses,
                        "recent_eviction_misses": recent,
                        "dedup_skipped": False,
                        "request": request_meta,
                    })
                    _omni_kv_lru_flush(pool=self)
            except Exception:
                pass
            return result

        def _patched_block_pool_maybe_evict_cached_block(self, block):
            block_hash = getattr(block, "block_hash", None)
            block_id = getattr(block, "block_id", None)
            kind = _omni_kv_lru_pool_kind(self)
            evicted = _original_block_pool_maybe_evict(self, block)
            if evicted and block_hash is not None:
                try:
                    with _omni_kv_lru_lock:
                        _omni_kv_lru_inc(
                            _omni_kv_lru_counter(
                                kind,
                                "lru_eviction_blocks_total",
                            )
                        )
                        _omni_kv_lru_inc_request(
                            _omni_kv_lru_current_request_meta(self),
                            {
                                _omni_kv_lru_counter(
                                    kind,
                                    "lru_eviction_blocks_total",
                                ): 1,
                            },
                        )
                        _omni_kv_lru_remember(kind, block_hash, block_id)
                        _omni_kv_lru_write_event({
                            "kind": "kv_lru_eviction",
                            "pool_kind": kind,
                            "block_hash": repr(block_hash),
                            "block_id": block_id,
                        })
                        _omni_kv_lru_flush(pool=self)
                except Exception:
                    pass
            return evicted

        def _patched_block_pool_touch(self, blocks):
            blocks_list = list(blocks)
            kind = _omni_kv_lru_pool_kind(self)
            rescued = sum(
                1
                for block in blocks_list
                if getattr(block, "ref_cnt", 0) == 0
                and not getattr(block, "is_null", False)
            )
            result = _original_block_pool_touch(self, blocks_list)
            try:
                with _omni_kv_lru_lock:
                    _omni_kv_lru_inc(
                        _omni_kv_lru_counter(kind, "prefix_block_touch_total"),
                        len(blocks_list),
                    )
                    _omni_kv_lru_inc(
                        _omni_kv_lru_counter(
                            kind,
                            "prefix_block_rescued_from_free_queue_total",
                        ),
                        rescued,
                    )
                    request_meta = _omni_kv_lru_current_request_meta(self)
                    _omni_kv_lru_inc_request(
                        request_meta,
                        {
                            _omni_kv_lru_counter(
                                kind,
                                "prefix_block_touch_total",
                            ): len(blocks_list),
                            _omni_kv_lru_counter(
                                kind,
                                "prefix_block_rescued_from_free_queue_total",
                            ): rescued,
                        },
                    )
                    _omni_kv_lru_write_event({
                        "kind": "kv_lru_touch",
                        "pool_kind": kind,
                        "blocks": len(blocks_list),
                        "rescued": rescued,
                        "request": request_meta,
                    })
                    _omni_kv_lru_flush(pool=self)
            except Exception:
                pass
            return result

        _patched_block_pool_get_cached_block._omni_kv_lru_stats_patch = True
        _OriginalBlockPool.get_cached_block = _patched_block_pool_get_cached_block
        _OriginalBlockPool._maybe_evict_cached_block = (
            _patched_block_pool_maybe_evict_cached_block
        )
        _OriginalBlockPool.touch = _patched_block_pool_touch

    try:
        from vllm.v1.simple_kv_offload.manager import (
            SimpleCPUOffloadScheduler as _OriginalSimpleCPUOffloadScheduler,
        )
    except Exception:  # pragma: no cover - optional connector may be absent.
        _OriginalSimpleCPUOffloadScheduler = None

    if (
        _OriginalSimpleCPUOffloadScheduler is not None
        and not getattr(
            _OriginalSimpleCPUOffloadScheduler.__init__,
            "_omni_kv_lru_stats_patch",
            False,
        )
    ):
        _original_simple_cpu_offload_init = _OriginalSimpleCPUOffloadScheduler.__init__
        _original_simple_cpu_bind_gpu_block_pool = (
            _OriginalSimpleCPUOffloadScheduler.bind_gpu_block_pool
        )

        def _patched_simple_cpu_offload_init(self, *args, **kwargs):
            _original_simple_cpu_offload_init(self, *args, **kwargs)
            try:
                self.cpu_block_pool._omni_kv_lru_pool_kind = "cpu"
            except Exception:
                pass

        def _patched_simple_cpu_bind_gpu_block_pool(self, gpu_block_pool):
            try:
                gpu_block_pool._omni_kv_lru_pool_kind = "gpu"
            except Exception:
                pass
            return _original_simple_cpu_bind_gpu_block_pool(self, gpu_block_pool)

        _patched_simple_cpu_offload_init._omni_kv_lru_stats_patch = True
        _OriginalSimpleCPUOffloadScheduler.__init__ = _patched_simple_cpu_offload_init
        _OriginalSimpleCPUOffloadScheduler.bind_gpu_block_pool = (
            _patched_simple_cpu_bind_gpu_block_pool
        )

    _omni_kv_lru_flush(force=True)
    atexit.register(lambda: _omni_kv_lru_flush(force=True))


# =============================================================================
# Env-gated interaction-aware KV eviction shadow/enforcement hooks
# =============================================================================
def _install_kv_eviction_shadow_patch() -> None:
    helper_path = Path(__file__).parent / "core" / "sched" / "kv_eviction_shadow.py"
    spec = importlib.util.spec_from_file_location(
        "vllm_omni_kv_eviction_shadow_patch",
        helper_path,
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_block_pool_shadow_patch()
    module.install_kv_cache_manager_shadow_patch()


if (
    _omni_env_bool("OMNI_KV_EVICT_SHADOW", False)
    or _omni_env_policy_enabled("OMNI_KV_EVICT_ENFORCE_POLICY")
    or _omni_env_bool("OMNI_KV_EVICT_RECORD_ALLOC_SHAPE", False)
):
    _install_kv_eviction_shadow_patch()
