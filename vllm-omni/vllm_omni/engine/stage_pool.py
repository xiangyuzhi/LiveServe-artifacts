"""Unified stage-local runtime abstraction for vLLM-Omni."""

from __future__ import annotations

import asyncio
import os
import time as _time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreOutputs

from vllm_omni.metrics.stats import StageRequestStats as StageRequestMetrics
from vllm_omni.metrics.stats import StageStats
from vllm_omni.metrics.utils import count_tokens_from_outputs
from vllm_omni.profiler.interaction_trace import emit
from vllm_omni.profiler.interaction_trace import enabled as trace_enabled

if TYPE_CHECKING:
    from vllm_omni.engine.orchestrator import OrchestratorRequestState

logger = init_logger(__name__)


@dataclass
class _ReplicaMetrics:
    """Per-replica metrics accumulators owned by a stage pool."""

    batch_seq: int = 0
    agg_total_tokens: int = 0
    agg_total_gen_time_ms: float = 0.0


@dataclass(frozen=True)
class _PrefetchCandidate:
    """A completed route candidate, not a worker KV residency lease."""

    key: tuple[str, str]
    version: int
    source_request_id: str
    replica_id: int
    client: Any
    client_generation: int
    expires: float = 0.0


class StagePool:
    """Replicas of one logical stage with RR + affinity selection."""

    def __init__(
        self,
        stage_id: int,
        clients: Any | list[Any],
        *,
        output_processor: Any = None,
        stage_vllm_config: Any = None,
    ) -> None:
        if isinstance(clients, list):
            normalized_clients = list(clients)
        else:
            normalized_clients = [clients]

        if not normalized_clients:
            raise ValueError(f"StagePool for stage {stage_id} has no replicas")
        self.stage_id = stage_id
        self.clients: list[Any] = normalized_clients
        self._output_processor = output_processor
        self._stage_vllm_config = stage_vllm_config
        self._next_replica_id = 0
        self._request_bindings: dict[str, int] = {}
        self._mm_hash_bindings: dict[str, int] = {}
        self._replica_metrics: list[_ReplicaMetrics] = [_ReplicaMetrics() for _ in self.clients]
        # These are routing hints, never evidence of resident KV or a cache hit.
        self._prefix_route_enabled = os.getenv("OMNI_PREFETCH_REPLICA_AFFINITY", "0").lower() in {
            "1", "true", "yes", "on"
        }
        self._stage_active_load_enabled = os.getenv("OMNI_PREFETCH_STAGE_ACTIVE_LOAD", "0").lower() in {
            "1", "true", "yes", "on"
        }
        self._track_stage_activity = self._prefix_route_enabled or self._stage_active_load_enabled or trace_enabled()
        self._active_stage_requests: dict[str, int] = {}
        self._activity_revisions: dict[str, int] = {}
        self._activity_sequence = 0
        self._prefetch_route_requests: dict[str, _PrefetchCandidate] = {}
        self._prefix_routes: dict[tuple[str, str], _PrefetchCandidate] = {}
        self._prefetch_versions: dict[tuple[str, str], int] = {}
        self._prefetch_sequence = 0
        self._route_clients = list(self.clients)
        self._client_generations = [0] * len(self.clients)
        self._retired_route_replicas: set[int] = set()
        self._prefix_route_ttl_s = self._route_option("OMNI_PREFETCH_ROUTE_TTL_SEC", 30.0, minimum=0.0)
        self._prefix_route_max_extra_active = int(
            self._route_option("OMNI_PREFETCH_ROUTE_MAX_EXTRA_ACTIVE", 0.0, minimum=0.0)
        )

    @staticmethod
    def _route_option(name: str, default: float, *, minimum: float) -> float:
        try:
            value = float(os.getenv(name, str(default)))
            return max(minimum, value) if value < float("inf") else default
        except ValueError:
            return default

    def active_request_snapshot(self) -> dict[str, int]:
        """Stage work still outstanding, separate from long-lived route bindings.

        This is an orchestrator-level admission hint, not a GPU running count.
        """
        return dict(self._active_stage_requests)

    def _set_activity(self, request_id: str, replica_id: int | None, *, active: bool, reason: str) -> int | None:
        if not self._track_stage_activity:
            return None
        self._activity_sequence += 1
        revision = self._activity_sequence
        if active:
            assert replica_id is not None
            self._active_stage_requests[request_id] = replica_id
            self._activity_revisions[request_id] = revision
        else:
            self._active_stage_requests.pop(request_id, None)
            self._activity_revisions.pop(request_id, None)
        if trace_enabled():
            emit("stage_activity_transition", request_id=request_id, stage_id=self.stage_id,
                 replica_id=replica_id, active=active, revision=revision, reason=reason,
                 outstanding_stage_requests=len(self._active_stage_requests),
                 semantics="submitted_stage_work_not_GPU_running_or_drain")
        return revision

    def _invalidate_candidate(self, key: tuple[str, str], reason: str) -> None:
        candidate = self._prefix_routes.pop(key, None)
        if candidate is not None and trace_enabled():
            emit("prefetch_candidate_invalidated", request_id=candidate.source_request_id,
                 stage_id=self.stage_id, replica_id=candidate.replica_id,
                 candidate_version=candidate.version, reason=reason,
                 client_generation=candidate.client_generation, cache_residency_confirmed=None)

    def _sync_client_generation(self, replica_id: int) -> int:
        # Observes replacement of the client handle only. An internal worker
        # restart under the same handle still requires an engine epoch protocol.
        if self._route_clients[replica_id] is not self.clients[replica_id]:
            self._route_clients[replica_id] = self.clients[replica_id]
            self._client_generations[replica_id] += 1
            self._retired_route_replicas.discard(replica_id)
            for key, candidate in list(self._prefix_routes.items()):
                if candidate.replica_id == replica_id:
                    self._invalidate_candidate(key, "client_replaced")
        return self._client_generations[replica_id]

    def _begin_prefetch_candidate(self, request_id: str, key: tuple[str, str], replica_id: int) -> None:
        generation = self._sync_client_generation(replica_id)
        self._prefetch_sequence += 1
        version = self._prefetch_sequence
        self._invalidate_candidate(key, "superseded")
        if len(self._prefetch_versions) >= 4096 and key not in self._prefetch_versions:
            old_key = next(iter(self._prefetch_versions))
            self._prefetch_versions.pop(old_key)
            self._invalidate_candidate(old_key, "capacity")
        self._prefetch_versions[key] = version
        self._prefetch_route_requests[request_id] = _PrefetchCandidate(
            key, version, request_id, replica_id, self.clients[replica_id], generation)

    @staticmethod
    def _prefix_route_key(params: Any) -> tuple[tuple[str, str] | None, bool]:
        extra = getattr(params, "extra_args", None) or {}
        kv = extra.get("kv_transfer_params") or {}
        is_prefetch = bool(kv.get("omni_scheduler_prefetch"))
        session = kv.get("omni_prefetch_session_id") if is_prefetch else kv.get("omni_session_id")
        prefix = (
            kv.get("omni_prefetch_committed_prefix_hash") if is_prefetch
            else kv.get("omni_matched_prefix_hash") or kv.get("omni_committed_prefix_hash")
        )
        if isinstance(session, str) and session and isinstance(prefix, str) and prefix:
            return (session, prefix), is_prefetch
        return None, is_prefetch

    @staticmethod
    def _successful_prefetch_completion(output: Any) -> bool:
        """A terminal output can also represent failed KV load or an abort.

        CompletionOutput carries that status in its finish_reason, rather than
        RequestOutput.error. Unknown/empty completions do not establish a route
        hint. Ordinary stage activity cleanup does not depend on this check.
        """
        if output is None or not getattr(output, "finished", False) or getattr(output, "error", None) is not None:
            return False
        completions = getattr(output, "outputs", None)
        return bool(completions) and all(
            getattr(completion, "finish_reason", None) in ("stop", "length")
            for completion in completions
        )

    def mark_stage_finished(self, request_id: str, replica_id: int, *, completion_output: Any = None) -> None:
        """Retire activity; publish pending prefetch hints only on success."""
        if self._track_stage_activity and self.get_bound_replica_id(request_id) != replica_id:
            return
        self._set_activity(request_id, replica_id, active=False, reason="stage_finished")
        pending = self._prefetch_route_requests.pop(request_id, None)
        if (pending is not None and self._successful_prefetch_completion(completion_output)
                and pending.client is self.clients[replica_id]
                and pending.client_generation == self._sync_client_generation(replica_id)
                and self._prefetch_versions.get(pending.key) == pending.version
                and self.get_bound_replica_id(request_id) == replica_id
                and replica_id not in self._retired_route_replicas):
            key = pending.key
            now = _time.monotonic()
            # Bounded storage; stale routes cannot survive TTL/client replacement.
            for old_key, candidate in list(self._prefix_routes.items()):
                if candidate.expires <= now or self.clients[candidate.replica_id] is not candidate.client:
                    self._invalidate_candidate(old_key, "expired_or_replaced")
            self._prefix_routes[key] = _PrefetchCandidate(
                key, pending.version, request_id, replica_id, pending.client,
                pending.client_generation, now + self._prefix_route_ttl_s)
            if trace_enabled():
                emit("prefetch_route_recorded", request_id=request_id, stage_id=self.stage_id,
                     replica_id=replica_id, ttl_s=self._prefix_route_ttl_s, residency_confirmed=False,
                     candidate_version=pending.version, candidate_state="completed_hint",
                     client_generation=pending.client_generation, cache_reuse_confirmed=None)
        elif pending is not None and trace_enabled():
            emit("prefetch_candidate_invalidated", request_id=request_id, stage_id=self.stage_id,
                 replica_id=replica_id, candidate_version=pending.version,
                 client_generation=pending.client_generation,
                 reason="completion_unconfirmed" if not self._successful_prefetch_completion(completion_output)
                 else "superseded_or_client_invalid", cache_residency_confirmed=None)
        if trace_enabled():
            emit("stage_finished", request_id=request_id, stage_id=self.stage_id, replica_id=replica_id)

    def _route_candidate(self, key: tuple[str, str] | None) -> tuple[int | None, str]:
        if key is None:
            return None, "no_prefix_key"
        route = self._prefix_routes.get(key)
        if route is None:
            return None, "no_completed_prefetch_route"
        replica_id = route.replica_id
        generation = self._sync_client_generation(replica_id)
        if (route.expires <= _time.monotonic() or self.clients[replica_id] is not route.client
                or generation != route.client_generation
                or replica_id in self._retired_route_replicas):
            self._invalidate_candidate(key, "expired_or_replaced")
            return None, "expired_or_replaced"
        loads = [0] * self.num_replicas
        for active_replica in self._active_stage_requests.values():
            loads[active_replica] += 1
        least_loaded = min(range(self.num_replicas), key=loads.__getitem__)
        if loads[replica_id] > loads[least_loaded] + self._prefix_route_max_extra_active:
            return least_loaded, "route_load_fallback"
        return replica_id, "completed_prefetch_route_hint"

    # ---- Stage-level properties ----

    @property
    def num_replicas(self) -> int:
        return len(self.clients)

    @property
    def stage_type(self) -> str | None:
        return getattr(self.stage_client, "stage_type", None)

    @property
    def final_output(self) -> bool:
        return bool(getattr(self.clients[0], "final_output", False))

    @property
    def stage_client(self) -> Any:
        return self.clients[0]

    @property
    def stage_vllm_config(self) -> Any:
        return self._stage_vllm_config

    @property
    def output_processor(self) -> Any:
        return self._output_processor

    # ---- Route binding lifecycle ----

    def get_bound_replica_id(self, request_id: str) -> int | None:
        """Return the currently bound replica id for *request_id* if present."""
        return self._request_bindings.get(request_id)

    def get_bound_client(self, request_id: str) -> Any | None:
        """Return the currently bound client for *request_id* if present."""
        replica_id = self.get_bound_replica_id(request_id)
        if replica_id is None:
            return None
        return self.clients[replica_id]

    def release_binding(self, request_id: str) -> None:
        """Drop the route binding for *request_id* in this stage."""
        replica_id = self._request_bindings.pop(request_id, None)
        if self._track_stage_activity:
            self._set_activity(request_id, replica_id, active=False, reason="binding_released")
            pending = self._prefetch_route_requests.pop(request_id, None)
            if pending is not None and trace_enabled():
                emit("prefetch_candidate_invalidated", request_id=request_id, stage_id=self.stage_id,
                     replica_id=pending.replica_id, candidate_version=pending.version,
                     client_generation=pending.client_generation, reason="released_before_completion",
                     cache_residency_confirmed=None)

    def release_bindings(self, request_ids: list[str]) -> None:
        """Drop route bindings for the given request ids in this stage."""
        for request_id in request_ids:
            self.release_binding(request_id)

    def select_replica_id(
        self,
        request_id: str,
        *,
        affinity_request_id: str | None = None,
        affinity_keys: list[str] | None = None,
        prefix_route_key: tuple[str, str] | None = None,
    ) -> int:
        """Pick a replica id for *request_id* and cache the choice."""
        cached = self.get_bound_replica_id(request_id)
        if cached is not None:
            return cached

        chosen = self.get_bound_replica_id(affinity_request_id) if affinity_request_id is not None else None
        route_reason = "existing_affinity" if chosen is not None else "round_robin"
        if chosen is None and self._prefix_route_enabled and self.stage_id == 0:
            chosen, route_reason = self._route_candidate(prefix_route_key)
        if chosen is None and affinity_keys:
            for key in affinity_keys:
                chosen = self._mm_hash_bindings.get(key)
                if chosen is not None:
                    route_reason = "multimodal_affinity"
                    break
        if chosen is None:
            if self.num_replicas == 1:
                chosen = 0
            else:
                chosen = self._next_replica_id
                self._next_replica_id = (self._next_replica_id + 1) % self.num_replicas

        self._request_bindings[request_id] = chosen
        self._set_activity(request_id, chosen, active=True, reason="initial_submit")
        if trace_enabled():
            candidate = self._prefix_routes.get(prefix_route_key)
            emit("route", request_id=request_id, stage_id=self.stage_id, replica_id=chosen,
                 route_reason=route_reason, residency_confirmed=False,
                 candidate_version=candidate.version if candidate else None,
                 candidate_source_request_id=candidate.source_request_id if candidate else None,
                 client_generation=candidate.client_generation if candidate else None,
                 cache_reuse_confirmed=None,
                 outstanding_stage_requests=len(self._active_stage_requests))
        if affinity_keys:
            for key in affinity_keys:
                self._mm_hash_bindings.setdefault(key, chosen)
        return chosen

    @staticmethod
    def _extract_mm_affinity_keys(request: Any) -> list[str]:
        features = getattr(request, "mm_features", None) or []
        keys: list[str] = []
        seen: set[str] = set()
        for feature in features:
            key = getattr(feature, "mm_hash", None) or getattr(feature, "identifier", None)
            if key is None:
                continue
            key_str = str(key)
            if key_str in seen:
                continue
            seen.add(key_str)
            keys.append(key_str)
        return keys

    # ---- Metrics ----

    def build_stage_metrics(
        self,
        request_outputs: list[Any],
        *,
        submit_ts: float,
        replica_id: int,
    ) -> StageRequestMetrics:
        """Build stage metrics for outputs produced on one replica."""
        now = _time.time()
        stage_gen_time_ms = (now - submit_ts) * 1000.0

        num_tokens_out = count_tokens_from_outputs(request_outputs)
        num_tokens_in = 0
        if self.stage_id == 0:
            for ro in request_outputs:
                ptids = getattr(ro, "prompt_token_ids", None)
                if ptids is not None:
                    num_tokens_in += len(ptids)

        metrics = self._replica_metrics[replica_id]
        metrics.batch_seq += 1
        batch_id = metrics.batch_seq
        metrics.agg_total_tokens += num_tokens_out
        metrics.agg_total_gen_time_ms += stage_gen_time_ms

        return StageRequestMetrics(
            num_tokens_in=num_tokens_in,
            num_tokens_out=num_tokens_out,
            stage_gen_time_ms=stage_gen_time_ms,
            batch_id=batch_id,
            batch_size=1,
            rx_decode_time_ms=0.0,
            rx_transfer_bytes=0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(
                total_token=metrics.agg_total_tokens,
                total_gen_time_ms=metrics.agg_total_gen_time_ms,
            ),
        )

    # ---- Stage-local admission ----

    async def submit_initial(
        self,
        request_id: str,
        req_state: OrchestratorRequestState,
        request: Any,
        *,
        prompt_text: Any = None,
        affinity_request_id: str | None = None,
        submit_kwargs: dict[str, Any] | None = None,
        params_override: Any = None,
    ) -> int:
        """Submit a stage-entry request into this pool."""
        params = params_override if params_override is not None else req_state.sampling_params_list[self.stage_id]
        submit_kwargs = dict(submit_kwargs or {})
        mm_affinity_keys = self._extract_mm_affinity_keys(request)
        prefix_key, is_prefetch = (None, False)
        if self._prefix_route_enabled and self.stage_id == 0:
            prefix_key, is_prefetch = self._prefix_route_key(params)
        if self.stage_type == "diffusion":
            replica_id = self.select_replica_id(
                request_id,
                affinity_request_id=affinity_request_id,
                affinity_keys=mm_affinity_keys,
                prefix_route_key=prefix_key,
            )
            client = self.clients[replica_id]
            if is_prefetch and prefix_key is not None:
                self._begin_prefetch_candidate(request_id, prefix_key, replica_id)
            try:
                if isinstance(request, list):
                    await client.add_batch_request_async(request_id, request, params, **submit_kwargs)
                else:
                    await client.add_request_async(request_id, request, params, **submit_kwargs)
            except (Exception, asyncio.CancelledError):
                if self._prefix_route_enabled or self._stage_active_load_enabled:
                    self.release_binding(request_id)
                raise
            return replica_id

        replica_id = self.select_replica_id(
            request_id,
            affinity_request_id=affinity_request_id,
            affinity_keys=mm_affinity_keys,
            prefix_route_key=prefix_key,
        )
        if is_prefetch and prefix_key is not None:
            self._begin_prefetch_candidate(request_id, prefix_key, replica_id)
        try:
            self.output_processor.add_request(
                request=request,
                prompt=prompt_text,
                parent_req=None,
                request_index=0,
                queue=None,
            )
        except Exception:
            self.release_binding(request_id)
            raise

        try:
            if trace_enabled():
                emit("stage_enqueue", request_id=request_id, stage_id=self.stage_id, replica_id=replica_id)
            await self.clients[replica_id].add_request_async(request, **submit_kwargs)
        except (Exception, asyncio.CancelledError) as error:
            if isinstance(error, asyncio.CancelledError) and not (
                self._prefix_route_enabled or self._stage_active_load_enabled
            ):
                raise
            self.release_binding(request_id)
            rollback = getattr(self.output_processor, "remove_request", None)
            if callable(rollback):
                try:
                    rollback(request_id)
                except Exception as rollback_error:
                    logger.warning(
                        "[StagePool] Failed to rollback output processor state for req=%s stage-%s: %s",
                        request_id,
                        self.stage_id,
                        rollback_error,
                    )
            raise
        return replica_id

    async def submit_update(
        self,
        request_id: str,
        req_state: OrchestratorRequestState,
        request: Any,
        *,
        prompt_text: Any = None,
    ) -> int:
        """Submit a streaming update to an already admitted request."""
        params = req_state.sampling_params_list[self.stage_id]
        previous_activity = self._active_stage_requests.get(request_id)
        replica_id = self.get_bound_replica_id(request_id)
        previous_binding = replica_id
        if replica_id is None:
            replica_id = self.select_replica_id(request_id)
        revision = self._set_activity(request_id, replica_id, active=True, reason="update_submit")
        if trace_enabled():
            emit("stage_update_enqueue", request_id=request_id, stage_id=self.stage_id, replica_id=replica_id)

        try:
            if self.stage_type == "diffusion":
                await self.clients[replica_id].add_request_async(request_id, request, params)
            else:
                self.output_processor.add_request(
                    request=request, prompt=prompt_text, parent_req=None, request_index=0, queue=None)
                await self.clients[replica_id].add_request_async(request)
        except (Exception, asyncio.CancelledError):
            # A completion/release/new update during the await supersedes this
            # submission. Never restore stale activity or erase newer work.
            if revision is not None and self._activity_revisions.get(request_id) == revision:
                if previous_binding is None:
                    if self._prefix_route_enabled or self._stage_active_load_enabled:
                        self.release_binding(request_id)
                    else:
                        # Tracing alone must not change legacy binding behavior.
                        self._set_activity(request_id, replica_id, active=False, reason="update_rollback")
                else:
                    self._set_activity(request_id, replica_id, active=previous_activity is not None,
                                       reason="update_rollback")
            raise
        return replica_id

    # ---- Stage-local polling ----

    async def _poll_stage_raw(self, client: Any) -> EngineCoreOutputs | None:
        """Pull raw EngineCoreOutputs from a stage replica without processing."""
        outputs = await client.get_output_async()
        if not outputs.outputs:
            return None
        return outputs

    async def process_llm_raw_outputs(
        self,
        replica_id: int,
        raw_outputs: EngineCoreOutputs,
    ) -> list[Any]:
        """Run the shared LLM output processor on one raw poll result."""
        client = self.clients[replica_id]
        processor = self.output_processor
        if trace_enabled():
            for output in raw_outputs.outputs:
                stats = getattr(output, "prefill_stats", None)
                if stats is not None:
                    emit("cache_accounting_observed", request_id=output.request_id,
                         stage_id=self.stage_id, replica_id=replica_id,
                         evidence_source="engine_core_prefill_stats",
                         **{name: getattr(stats, name, None) for name in (
                             "num_prompt_tokens", "num_computed_tokens", "num_cached_tokens",
                             "num_local_cached_tokens", "num_external_cached_tokens")},
                         prefetch_attribution_confirmed=None, cache_residency_confirmed=None)
        processed = processor.process_outputs(
            raw_outputs.outputs,
            raw_outputs.timestamp,
            None,
        )

        if processed.reqs_to_abort:
            await client.abort_requests_async(processed.reqs_to_abort)

        if raw_outputs.scheduler_stats is not None:
            processor.update_scheduler_stats(raw_outputs.scheduler_stats)

        return processed.request_outputs

    async def poll_llm_raw_output(
        self,
        replica_id: int,
        *,
        timeout_s: float = 0.001,
    ) -> EngineCoreOutputs | None:
        """Poll raw EngineCore outputs from one LLM replica once."""
        client = self.clients[replica_id]
        try:
            return await asyncio.wait_for(
                self._poll_stage_raw(client),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "[StagePool] _poll_stage_raw failed for stage-%s replica-%s",
                self.stage_id,
                replica_id,
            )
            raise

    def poll_diffusion_output(self, replica_id: int) -> Any | None:
        """Drain one ready diffusion output from the given replica if present."""
        return self.clients[replica_id].get_diffusion_output_nowait()

    # ---- Stage-local control plane ----

    async def abort_requests(self, request_ids: list[str]) -> None:
        """Abort the given requests in this stage pool.

        Request-bound abort routing stays inside the pool because route affinity
        (`request_id -> replica_id`) is pool-owned.
        """
        if not request_ids:
            return

        request_ids_by_replica: dict[int, list[str]] = {}
        for request_id in request_ids:
            replica_id = self.get_bound_replica_id(request_id)
            if replica_id is None:
                logger.debug("[StagePool] abort: no binding for req=%s in stage-%s", request_id, self.stage_id)
                continue
            request_ids_by_replica.setdefault(replica_id, []).append(request_id)

        for replica_id, replica_request_ids in request_ids_by_replica.items():
            await self.clients[replica_id].abort_requests_async(replica_request_ids)

        # Clean up OutputProcessor state (e.g. mm_accumulated tensors) that
        # would otherwise leak — aborted requests never produce a final
        # EngineCoreOutput, so process_outputs() never fires its cleanup path.
        all_aborted = [rid for ids in request_ids_by_replica.values() for rid in ids]
        if all_aborted and self._output_processor is not None:
            self._output_processor.abort_requests(all_aborted, internal=True)

    async def collective_rpc(
        self,
        replica_id: int,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Dispatch a stage-scoped control-plane RPC to one physical route."""
        kwargs = dict(kwargs or {})
        client = self.clients[replica_id]
        try:
            if hasattr(client, "collective_rpc_async"):
                return await client.collective_rpc_async(
                    method=method,
                    timeout=timeout,
                    args=args,
                    kwargs=kwargs,
                )
            return {
                "supported": False,
                "todo": True,
                "reason": f"{client.__class__.__name__}.collective_rpc_async is not implemented yet",
            }
        except Exception as exc:
            logger.exception(
                "[StagePool] collective_rpc failed: stage=%s replica=%s method=%s",
                self.stage_id,
                replica_id,
                method,
            )
            return {
                "supported": False,
                "error": str(exc),
            }

    def shutdown_replica(self, replica_id: int) -> None:
        """Shutdown one backend handle in this stage pool."""
        client = self.clients[replica_id]
        self._retired_route_replicas.add(replica_id)
        self._client_generations[replica_id] += 1
        for request_id, active_replica in list(self._active_stage_requests.items()):
            if active_replica == replica_id:
                self._set_activity(request_id, replica_id, active=False, reason="replica_shutdown")
        for key, candidate in list(self._prefix_routes.items()):
            if candidate.replica_id == replica_id:
                self._invalidate_candidate(key, "replica_shutdown")
        try:
            client.shutdown()
            logger.info(
                "[StagePool] Stage %d replica %d shut down",
                self.stage_id,
                replica_id,
            )
        except Exception as e:
            logger.warning(
                "[StagePool] Failed to shutdown stage %d replica %d: %s",
                self.stage_id,
                replica_id,
                e,
            )
