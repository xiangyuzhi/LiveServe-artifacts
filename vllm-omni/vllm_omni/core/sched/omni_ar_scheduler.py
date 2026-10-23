from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from time import monotonic_ns, time
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler as AsyncVLLMScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.core.sched.lead_governor_config import LeadGovernorParams
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.output import OmniSchedulerOutput
from vllm_omni.core.sched.talker_lead_governor import TalkerLeadGovernor
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.profiler.interaction_trace import emit as trace_emit
from vllm_omni.profiler.interaction_trace import enabled as trace_enabled

logger = init_logger(__name__)


@dataclass
class KVCacheTransferData:
    request_id: str
    layer_blocks: dict[str, Any]
    block_ids: list[int]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OmniARScheduler(OmniSchedulerMixin, VLLMScheduler):
    """Synchronous AutoRegressive scheduler for vLLM-Omni. This class is also
    used as a base class for the OmniARAsyncScheduler and holds most of the
    core scheduling logic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track requests that need KV cache transfer when finished
        # Value is {"seq_len": int, "block_ids": list[int]}
        self.requests_needing_kv_transfer: dict[str, dict[str, Any]] = {}

        # Track requests waiting for KV transfer (blocks not freed yet)
        self.waiting_for_transfer_free: set[str] = set()

        # Track ACTIVE transfers (submitted to runner but not yet acked via kv_extracted_req_ids)
        self.active_kv_transfers: set[str] = set()

        # Requests marked for deferred stop: keep running until KV extraction
        # completes so that kv_ready can be emitted while the request is still
        # alive.  Stopped on the first scheduler step after extraction ack.
        self.pending_stop_after_extraction: set[str] = set()

        # [Omni] Pre-parse KV transfer criteria
        self.kv_transfer_criteria = self._get_kv_transfer_criteria()

        # Track requests that have already triggered prefill transfer to avoid duplicates
        self.transfer_triggered_requests: set[str] = set()

        # Cache per-request flag to avoid repeated deserialization of additional_information
        self._omits_kv_transfer_cache: dict[str, bool] = {}
        model_config = self.vllm_config.model_config
        self._interaction_job_seq = 0
        self._interaction_trace_context = {
            "stage_id": getattr(model_config, "stage_id", None),
            "replica_id": getattr(model_config, "replica_id", None),
        }
        self.chunk_transfer_adapter = None
        if getattr(model_config, "async_chunk", False):
            self.chunk_transfer_adapter = OmniChunkTransferAdapter(self.vllm_config)
        # Snapshot prompt length for each streaming input update
        self._new_prompt_len_snapshot: dict[str, int] = {}
        self._bp_arrival_history: deque[tuple[float, str]] = deque()
        self._bp_arrival_history_seen: set[str] = set()

        # Lead-bounded decode scheduling governor. The default public policy
        # applies this to the talker (stage 1). The same state machine can also
        # be used on the thinker (stage 0) to create decode-only gaps for
        # waiting prefill work, while a KV-aware policy gate avoids deferring
        # requests whose long context would otherwise linger in GPU memory.
        import os as _os
        _lead_max: int | None = None
        _lead_lo: int | None = None
        _hold_max: int = 100
        _speaker_rate: int = 25
        _min_output_tokens_before_defer: int = 0
        self._bp_governor_role: str | None = None
        _stage_id_raw = getattr(model_config, "stage_id", None)
        try:
            _stage_id = int(_stage_id_raw)
        except (TypeError, ValueError):
            _stage_id = None

        def _parse_stage_ids(name: str, default: str) -> set[int]:
            parsed: set[int] = set()
            for part in _os.environ.get(name, default).split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    parsed.add(int(part))
                except ValueError:
                    logger.warning("Invalid %s item=%r, ignoring", name, part)
            return parsed

        def _parse_bp_roles(name: str, default: str) -> set[str]:
            parsed: set[str] = set()
            for part in _os.environ.get(name, default).split(","):
                role = part.strip().lower()
                if not role:
                    continue
                if role in ("all", "*"):
                    parsed.update(("thinker", "talker"))
                elif role in ("thinker", "talker"):
                    parsed.add(role)
                else:
                    logger.warning("Invalid %s item=%r, ignoring", name, part)
            return parsed

        _thinker_stage_ids = _parse_stage_ids("OMNI_THINKER_STAGE_IDS", "0")
        _talker_stage_ids: set[int] = set()
        _talker_stage_ids = _parse_stage_ids("OMNI_TALKER_STAGE_IDS", "1")
        _is_thinker_stage = _stage_id in _thinker_stage_ids
        _is_talker_stage = _stage_id in _talker_stage_ids

        # KV-aware policy knobs. These are harmless when the governor is
        # disabled. By default the KV-aware gate is thinker-only: thinker KV is
        # the main memory bottleneck, while talker should keep the direct
        # lead-bounded defer path unless OMNI_BP_KV_AWARE_ROLES explicitly
        # includes it.
        self._bp_kv_aware = self._omni_env_bool("OMNI_BP_KV_AWARE", False)
        self._bp_kv_aware_roles = _parse_bp_roles("OMNI_BP_KV_AWARE_ROLES", "thinker")
        self._bp_long_context_tokens = self._omni_env_int("OMNI_BP_LONG_CONTEXT_TOKENS", 2048)
        self._bp_kv_usage_high = self._omni_env_float("OMNI_BP_KV_USAGE_HIGH", 0.005)
        self._bp_session_kv_gib_high = self._omni_env_float("OMNI_BP_SESSION_KV_GIB_HIGH", 0.25)
        self._bp_thinker_session_kv_gib_high = self._omni_env_float(
            "OMNI_THINKER_SESSION_KV_GIB_HIGH", self._bp_session_kv_gib_high
        )
        self._bp_talker_session_kv_gib_high = self._omni_env_float(
            "OMNI_TALKER_SESSION_KV_GIB_HIGH", self._bp_session_kv_gib_high
        )
        self._bp_req_kv_gib_high = self._omni_env_float("OMNI_BP_REQ_KV_GIB_HIGH", 0.0)
        self._bp_thinker_req_kv_gib_high = self._omni_env_float(
            "OMNI_THINKER_REQ_KV_GIB_HIGH", self._bp_req_kv_gib_high
        )
        self._bp_talker_req_kv_gib_high = self._omni_env_float(
            "OMNI_TALKER_REQ_KV_GIB_HIGH", self._bp_req_kv_gib_high
        )
        self._bp_kv_pressure_suppress_defer = self._omni_env_bool(
            "OMNI_BP_KV_PRESSURE_SUPPRESS_DEFER", False
        )
        self._bp_thinker_kv_pressure_suppress_defer = self._omni_env_bool(
            "OMNI_THINKER_KV_PRESSURE_SUPPRESS_DEFER",
            self._bp_kv_pressure_suppress_defer,
        )
        self._bp_talker_kv_pressure_suppress_defer = self._omni_env_bool(
            "OMNI_TALKER_KV_PRESSURE_SUPPRESS_DEFER",
            self._bp_kv_pressure_suppress_defer,
        )
        self._bp_kv_bytes_per_token = self._omni_env_int("OMNI_BP_KV_BYTES_PER_TOKEN", 98304)
        self._bp_resume_on_empty_backlog = self._omni_env_bool(
            "OMNI_BP_RESUME_ON_EMPTY_BACKLOG", False
        )
        self._bp_empty_backlog_grace_steps = self._omni_env_int(
            "OMNI_BP_EMPTY_BACKLOG_GRACE_STEPS", 0
        )
        self._bp_first_chunk_protect_ms = max(
            0, self._omni_env_int("OMNI_BP_FIRST_CHUNK_PROTECT_MS", 0)
        )
        self._bp_prefill_priority = self._omni_env_bool(
            "OMNI_BP_PREFILL_PRIORITY", False
        )
        self._bp_thinker_prefill_priority = self._omni_env_bool(
            "OMNI_THINKER_PREFILL_PRIORITY", self._bp_prefill_priority
        )
        self._bp_talker_prefill_priority = self._omni_env_bool(
            "OMNI_TALKER_PREFILL_PRIORITY", self._bp_prefill_priority
        )
        self._bp_prefill_priority_window_steps = self._omni_env_int(
            "OMNI_BP_PREFILL_PRIORITY_WINDOW_STEPS", 0
        )
        self._bp_thinker_prefill_priority_window_steps = self._omni_env_int(
            "OMNI_THINKER_PREFILL_PRIORITY_WINDOW_STEPS",
            self._bp_prefill_priority_window_steps,
        )
        self._bp_talker_prefill_priority_window_steps = self._omni_env_int(
            "OMNI_TALKER_PREFILL_PRIORITY_WINDOW_STEPS",
            self._bp_prefill_priority_window_steps,
        )
        self._bp_include_chunk_waiting_in_pending = self._omni_env_bool(
            "OMNI_BP_INCLUDE_CHUNK_WAITING_IN_PENDING", False
        )
        self._bp_thinker_include_chunk_waiting_in_pending = self._omni_env_bool(
            "OMNI_THINKER_INCLUDE_CHUNK_WAITING_IN_PENDING",
            self._bp_include_chunk_waiting_in_pending,
        )
        self._bp_talker_include_chunk_waiting_in_pending = self._omni_env_bool(
            "OMNI_TALKER_INCLUDE_CHUNK_WAITING_IN_PENDING",
            self._bp_include_chunk_waiting_in_pending,
        )
        self._bp_min_pending_count = self._omni_env_int(
            "OMNI_BP_MIN_PENDING_COUNT", 1
        )
        self._bp_thinker_min_pending_count = self._omni_env_int(
            "OMNI_THINKER_MIN_PENDING_COUNT", self._bp_min_pending_count
        )
        self._bp_talker_min_pending_count = self._omni_env_int(
            "OMNI_TALKER_MIN_PENDING_COUNT", self._bp_min_pending_count
        )
        self._bp_min_backlog_active_steps = self._omni_env_int(
            "OMNI_BP_MIN_BACKLOG_ACTIVE_STEPS", 1
        )
        self._bp_thinker_min_backlog_active_steps = self._omni_env_int(
            "OMNI_THINKER_MIN_BACKLOG_ACTIVE_STEPS",
            self._bp_min_backlog_active_steps,
        )
        self._bp_talker_min_backlog_active_steps = self._omni_env_int(
            "OMNI_TALKER_MIN_BACKLOG_ACTIVE_STEPS",
            self._bp_min_backlog_active_steps,
        )
        self._bp_min_running_count = self._omni_env_int(
            "OMNI_BP_MIN_RUNNING_COUNT", 1
        )
        self._bp_thinker_min_running_count = self._omni_env_int(
            "OMNI_THINKER_MIN_RUNNING_COUNT", self._bp_min_running_count
        )
        self._bp_talker_min_running_count = self._omni_env_int(
            "OMNI_TALKER_MIN_RUNNING_COUNT", self._bp_min_running_count
        )
        self._bp_min_pressure_active_calls = self._omni_env_int(
            "OMNI_BP_MIN_PRESSURE_ACTIVE_CALLS", 0
        )
        self._bp_thinker_min_pressure_active_calls = self._omni_env_int(
            "OMNI_THINKER_MIN_PRESSURE_ACTIVE_CALLS",
            self._bp_min_pressure_active_calls,
        )
        self._bp_talker_min_pressure_active_calls = self._omni_env_int(
            "OMNI_TALKER_MIN_PRESSURE_ACTIVE_CALLS",
            self._bp_min_pressure_active_calls,
        )
        self._bp_pressure_bypass_pending_count = self._omni_env_int(
            "OMNI_BP_PRESSURE_BYPASS_PENDING_COUNT", 0
        )
        self._bp_thinker_pressure_bypass_pending_count = self._omni_env_int(
            "OMNI_THINKER_PRESSURE_BYPASS_PENDING_COUNT",
            self._bp_pressure_bypass_pending_count,
        )
        self._bp_talker_pressure_bypass_pending_count = self._omni_env_int(
            "OMNI_TALKER_PRESSURE_BYPASS_PENDING_COUNT",
            self._bp_pressure_bypass_pending_count,
        )
        self._bp_pressure_bypass_pending_age_ms = self._omni_env_float(
            "OMNI_BP_PRESSURE_BYPASS_PENDING_AGE_MS", 0.0
        )
        self._bp_thinker_pressure_bypass_pending_age_ms = self._omni_env_float(
            "OMNI_THINKER_PRESSURE_BYPASS_PENDING_AGE_MS",
            self._bp_pressure_bypass_pending_age_ms,
        )
        self._bp_talker_pressure_bypass_pending_age_ms = self._omni_env_float(
            "OMNI_TALKER_PRESSURE_BYPASS_PENDING_AGE_MS",
            self._bp_pressure_bypass_pending_age_ms,
        )
        self._bp_pressure_fast_warmup_pending_count = self._omni_env_int(
            "OMNI_BP_PRESSURE_FAST_WARMUP_PENDING_COUNT", 0
        )
        self._bp_thinker_pressure_fast_warmup_pending_count = self._omni_env_int(
            "OMNI_THINKER_PRESSURE_FAST_WARMUP_PENDING_COUNT",
            self._bp_pressure_fast_warmup_pending_count,
        )
        self._bp_talker_pressure_fast_warmup_pending_count = self._omni_env_int(
            "OMNI_TALKER_PRESSURE_FAST_WARMUP_PENDING_COUNT",
            self._bp_pressure_fast_warmup_pending_count,
        )
        self._bp_pressure_fast_warmup_active_calls = self._omni_env_int(
            "OMNI_BP_PRESSURE_FAST_WARMUP_ACTIVE_CALLS", 0
        )
        self._bp_thinker_pressure_fast_warmup_active_calls = self._omni_env_int(
            "OMNI_THINKER_PRESSURE_FAST_WARMUP_ACTIVE_CALLS",
            self._bp_pressure_fast_warmup_active_calls,
        )
        self._bp_talker_pressure_fast_warmup_active_calls = self._omni_env_int(
            "OMNI_TALKER_PRESSURE_FAST_WARMUP_ACTIVE_CALLS",
            self._bp_pressure_fast_warmup_active_calls,
        )
        self._bp_pressure_fast_warmup_pending_age_ms = self._omni_env_float(
            "OMNI_BP_PRESSURE_FAST_WARMUP_PENDING_AGE_MS", 0.0
        )
        self._bp_thinker_pressure_fast_warmup_pending_age_ms = self._omni_env_float(
            "OMNI_THINKER_PRESSURE_FAST_WARMUP_PENDING_AGE_MS",
            self._bp_pressure_fast_warmup_pending_age_ms,
        )
        self._bp_talker_pressure_fast_warmup_pending_age_ms = self._omni_env_float(
            "OMNI_TALKER_PRESSURE_FAST_WARMUP_PENDING_AGE_MS",
            self._bp_pressure_fast_warmup_pending_age_ms,
        )
        self._bp_pressure_fast_warmup_recent_arrivals = self._omni_env_int(
            "OMNI_BP_PRESSURE_FAST_WARMUP_RECENT_ARRIVALS", 0
        )
        self._bp_thinker_pressure_fast_warmup_recent_arrivals = self._omni_env_int(
            "OMNI_THINKER_PRESSURE_FAST_WARMUP_RECENT_ARRIVALS",
            self._bp_pressure_fast_warmup_recent_arrivals,
        )
        self._bp_talker_pressure_fast_warmup_recent_arrivals = self._omni_env_int(
            "OMNI_TALKER_PRESSURE_FAST_WARMUP_RECENT_ARRIVALS",
            self._bp_pressure_fast_warmup_recent_arrivals,
        )
        self._bp_pressure_fast_warmup_arrival_window_ms = self._omni_env_float(
            "OMNI_BP_PRESSURE_FAST_WARMUP_ARRIVAL_WINDOW_MS", 0.0
        )
        self._bp_thinker_pressure_fast_warmup_arrival_window_ms = (
            self._omni_env_float(
                "OMNI_THINKER_PRESSURE_FAST_WARMUP_ARRIVAL_WINDOW_MS",
                self._bp_pressure_fast_warmup_arrival_window_ms,
            )
        )
        self._bp_talker_pressure_fast_warmup_arrival_window_ms = (
            self._omni_env_float(
                "OMNI_TALKER_PRESSURE_FAST_WARMUP_ARRIVAL_WINDOW_MS",
                self._bp_pressure_fast_warmup_arrival_window_ms,
            )
        )
        self._bp_pressure_fast_warmup_include_running_arrivals = self._omni_env_bool(
            "OMNI_BP_PRESSURE_FAST_WARMUP_INCLUDE_RUNNING_ARRIVALS", False
        )
        self._bp_thinker_pressure_fast_warmup_include_running_arrivals = (
            self._omni_env_bool(
                "OMNI_THINKER_PRESSURE_FAST_WARMUP_INCLUDE_RUNNING_ARRIVALS",
                self._bp_pressure_fast_warmup_include_running_arrivals,
            )
        )
        self._bp_talker_pressure_fast_warmup_include_running_arrivals = (
            self._omni_env_bool(
                "OMNI_TALKER_PRESSURE_FAST_WARMUP_INCLUDE_RUNNING_ARRIVALS",
                self._bp_pressure_fast_warmup_include_running_arrivals,
            )
        )
        self._bp_pressure_fast_warmup_arrival_history = self._omni_env_bool(
            "OMNI_BP_PRESSURE_FAST_WARMUP_ARRIVAL_HISTORY", False
        )
        self._bp_thinker_pressure_fast_warmup_arrival_history = self._omni_env_bool(
            "OMNI_THINKER_PRESSURE_FAST_WARMUP_ARRIVAL_HISTORY",
            self._bp_pressure_fast_warmup_arrival_history,
        )
        self._bp_talker_pressure_fast_warmup_arrival_history = self._omni_env_bool(
            "OMNI_TALKER_PRESSURE_FAST_WARMUP_ARRIVAL_HISTORY",
            self._bp_pressure_fast_warmup_arrival_history,
        )
        self._bp_pressure_fast_warmup_pending_growth = self._omni_env_int(
            "OMNI_BP_PRESSURE_FAST_WARMUP_PENDING_GROWTH", 0
        )
        self._bp_thinker_pressure_fast_warmup_pending_growth = self._omni_env_int(
            "OMNI_THINKER_PRESSURE_FAST_WARMUP_PENDING_GROWTH",
            self._bp_pressure_fast_warmup_pending_growth,
        )
        self._bp_talker_pressure_fast_warmup_pending_growth = self._omni_env_int(
            "OMNI_TALKER_PRESSURE_FAST_WARMUP_PENDING_GROWTH",
            self._bp_pressure_fast_warmup_pending_growth,
        )
        self._bp_pressure_fast_warmup_growth_hold_calls = self._omni_env_int(
            "OMNI_BP_PRESSURE_FAST_WARMUP_GROWTH_HOLD_CALLS", 0
        )
        self._bp_thinker_pressure_fast_warmup_growth_hold_calls = (
            self._omni_env_int(
                "OMNI_THINKER_PRESSURE_FAST_WARMUP_GROWTH_HOLD_CALLS",
                self._bp_pressure_fast_warmup_growth_hold_calls,
            )
        )
        self._bp_talker_pressure_fast_warmup_growth_hold_calls = self._omni_env_int(
            "OMNI_TALKER_PRESSURE_FAST_WARMUP_GROWTH_HOLD_CALLS",
            self._bp_pressure_fast_warmup_growth_hold_calls,
        )
        self._bp_pressure_fast_warmup_require_all_signals = self._omni_env_bool(
            "OMNI_BP_PRESSURE_FAST_WARMUP_REQUIRE_ALL_SIGNALS", False
        )
        self._bp_thinker_pressure_fast_warmup_require_all_signals = (
            self._omni_env_bool(
                "OMNI_THINKER_PRESSURE_FAST_WARMUP_REQUIRE_ALL_SIGNALS",
                self._bp_pressure_fast_warmup_require_all_signals,
            )
        )
        self._bp_talker_pressure_fast_warmup_require_all_signals = self._omni_env_bool(
            "OMNI_TALKER_PRESSURE_FAST_WARMUP_REQUIRE_ALL_SIGNALS",
            self._bp_pressure_fast_warmup_require_all_signals,
        )
        self._bp_max_new_defer_per_call = self._omni_env_int(
            "OMNI_BP_MAX_NEW_DEFER_PER_CALL", 0
        )
        self._bp_thinker_max_new_defer_per_call = self._omni_env_int(
            "OMNI_THINKER_MAX_NEW_DEFER_PER_CALL",
            self._bp_max_new_defer_per_call,
        )
        self._bp_talker_max_new_defer_per_call = self._omni_env_int(
            "OMNI_TALKER_MAX_NEW_DEFER_PER_CALL",
            self._bp_max_new_defer_per_call,
        )
        self._bp_max_defer_episodes_per_window = self._omni_env_int(
            "OMNI_BP_MAX_DEFER_EPISODES_PER_WINDOW", 0
        )
        self._bp_thinker_max_defer_episodes_per_window = self._omni_env_int(
            "OMNI_THINKER_MAX_DEFER_EPISODES_PER_WINDOW",
            self._bp_max_defer_episodes_per_window,
        )
        self._bp_talker_max_defer_episodes_per_window = self._omni_env_int(
            "OMNI_TALKER_MAX_DEFER_EPISODES_PER_WINDOW",
            self._bp_max_defer_episodes_per_window,
        )
        self._bp_defer_budget_empty_reset_steps = self._omni_env_int(
            "OMNI_BP_DEFER_BUDGET_EMPTY_RESET_STEPS", 0
        )
        self._bp_thinker_defer_budget_empty_reset_steps = self._omni_env_int(
            "OMNI_THINKER_DEFER_BUDGET_EMPTY_RESET_STEPS",
            self._bp_defer_budget_empty_reset_steps,
        )
        self._bp_talker_defer_budget_empty_reset_steps = self._omni_env_int(
            "OMNI_TALKER_DEFER_BUDGET_EMPTY_RESET_STEPS",
            self._bp_defer_budget_empty_reset_steps,
        )
        self._bp_defer_cooldown_steps = self._omni_env_int(
            "OMNI_BP_DEFER_COOLDOWN_STEPS", 0
        )
        self._bp_thinker_defer_cooldown_steps = self._omni_env_int(
            "OMNI_THINKER_DEFER_COOLDOWN_STEPS",
            self._bp_defer_cooldown_steps,
        )
        self._bp_talker_defer_cooldown_steps = self._omni_env_int(
            "OMNI_TALKER_DEFER_COOLDOWN_STEPS",
            self._bp_defer_cooldown_steps,
        )
        self._bp_waiting_short_max_output_tokens = self._omni_env_int(
            "OMNI_BP_WAITING_SHORT_MAX_OUTPUT_TOKENS", 0
        )
        self._bp_thinker_waiting_short_max_output_tokens = self._omni_env_int(
            "OMNI_THINKER_WAITING_SHORT_MAX_OUTPUT_TOKENS",
            self._bp_waiting_short_max_output_tokens,
        )
        self._bp_talker_waiting_short_max_output_tokens = self._omni_env_int(
            "OMNI_TALKER_WAITING_SHORT_MAX_OUTPUT_TOKENS",
            self._bp_waiting_short_max_output_tokens,
        )
        self._bp_waiting_short_min_count = self._omni_env_int(
            "OMNI_BP_WAITING_SHORT_MIN_COUNT", 0
        )
        self._bp_thinker_waiting_short_min_count = self._omni_env_int(
            "OMNI_THINKER_WAITING_SHORT_MIN_COUNT",
            self._bp_waiting_short_min_count,
        )
        self._bp_talker_waiting_short_min_count = self._omni_env_int(
            "OMNI_TALKER_WAITING_SHORT_MIN_COUNT",
            self._bp_waiting_short_min_count,
        )
        self._bp_waiting_short_latch_calls = self._omni_env_int(
            "OMNI_BP_WAITING_SHORT_LATCH_CALLS", 0
        )
        self._bp_thinker_waiting_short_latch_calls = self._omni_env_int(
            "OMNI_THINKER_WAITING_SHORT_LATCH_CALLS",
            self._bp_waiting_short_latch_calls,
        )
        self._bp_talker_waiting_short_latch_calls = self._omni_env_int(
            "OMNI_TALKER_WAITING_SHORT_LATCH_CALLS",
            self._bp_waiting_short_latch_calls,
        )
        self._bp_waiting_short_entry_only = self._omni_env_bool(
            "OMNI_BP_WAITING_SHORT_ENTRY_ONLY", False
        )
        self._bp_thinker_waiting_short_entry_only = self._omni_env_bool(
            "OMNI_THINKER_WAITING_SHORT_ENTRY_ONLY",
            self._bp_waiting_short_entry_only,
        )
        self._bp_talker_waiting_short_entry_only = self._omni_env_bool(
            "OMNI_TALKER_WAITING_SHORT_ENTRY_ONLY",
            self._bp_waiting_short_entry_only,
        )
        self._bp_skip_global_suppressed_defer = self._omni_env_bool(
            "OMNI_BP_SKIP_GLOBAL_SUPPRESSED_DEFER", False
        )
        self._bp_thinker_skip_global_suppressed_defer = self._omni_env_bool(
            "OMNI_THINKER_SKIP_GLOBAL_SUPPRESSED_DEFER",
            self._bp_skip_global_suppressed_defer,
        )
        self._bp_talker_skip_global_suppressed_defer = self._omni_env_bool(
            "OMNI_TALKER_SKIP_GLOBAL_SUPPRESSED_DEFER",
            self._bp_skip_global_suppressed_defer,
        )
        self._bp_min_pending_age_ms = self._omni_env_float(
            "OMNI_BP_MIN_PENDING_AGE_MS", 0.0
        )
        self._bp_thinker_min_pending_age_ms = self._omni_env_float(
            "OMNI_THINKER_MIN_PENDING_AGE_MS", self._bp_min_pending_age_ms
        )
        self._bp_talker_min_pending_age_ms = self._omni_env_float(
            "OMNI_TALKER_MIN_PENDING_AGE_MS", self._bp_min_pending_age_ms
        )
        self._bp_pending_age_roles = _parse_bp_roles("OMNI_BP_PENDING_AGE_ROLES", "thinker")
        self._bp_talker_disable_on_video_mix = self._omni_env_bool(
            "OMNI_TALKER_DISABLE_ON_VIDEO_MIX", False
        )
        self._bp_talker_video_mix_allow_pending_count = self._omni_env_int(
            "OMNI_TALKER_VIDEO_MIX_ALLOW_PENDING_COUNT", 0
        )
        self._bp_talker_video_mix_allow_pending_age_ms = self._omni_env_float(
            "OMNI_TALKER_VIDEO_MIX_ALLOW_PENDING_AGE_MS", 0.0
        )
        _marker_raw = _os.environ.get(
            "OMNI_TALKER_VIDEO_MIX_MARKERS",
            "mixed,video,streamingbench",
        )
        self._bp_talker_video_mix_markers: tuple[str, ...] = tuple(
            marker.strip().lower()
            for marker in _marker_raw.split(",")
            if marker.strip()
        )

        _thinker_env_val = _os.environ.get("OMNI_THINKER_LEAD_MAX")
        _talker_env_val = _os.environ.get("OMNI_TALKER_LEAD_MAX")

        if _is_thinker_stage and _thinker_env_val is not None:
            self._bp_governor_role = "thinker"
            try:
                _lead_max = int(_thinker_env_val)
            except ValueError:
                init_logger(__name__).warning(
                    "Invalid OMNI_THINKER_LEAD_MAX=%r, ignoring", _thinker_env_val
                )
            _env_lo = _os.environ.get("OMNI_THINKER_LEAD_LO")
            if _env_lo is not None:
                try:
                    _lead_lo = int(_env_lo)
                except ValueError:
                    init_logger(__name__).warning(
                        "Invalid OMNI_THINKER_LEAD_LO=%r, ignoring", _env_lo
                    )
            _env_hold = _os.environ.get("OMNI_THINKER_DEFER_HOLD_MAX")
            if _env_hold is not None:
                try:
                    _hold_max = max(1, int(_env_hold))
                except ValueError:
                    init_logger(__name__).warning(
                        "Invalid OMNI_THINKER_DEFER_HOLD_MAX=%r, ignoring", _env_hold
                    )
            try:
                _speaker_rate = max(1, int(_os.environ.get("OMNI_THINKER_DECODE_RATE", "20")))
            except ValueError:
                init_logger(__name__).warning(
                    "Invalid OMNI_THINKER_DECODE_RATE=%r, using 20",
                    _os.environ.get("OMNI_THINKER_DECODE_RATE"),
                )
                _speaker_rate = 20
            try:
                _min_output_tokens_before_defer = max(
                    0,
                    int(_os.environ.get("OMNI_THINKER_MIN_OUTPUT_TOKENS_BEFORE_DEFER", "0")),
                )
            except ValueError:
                init_logger(__name__).warning(
                    "Invalid OMNI_THINKER_MIN_OUTPUT_TOKENS_BEFORE_DEFER=%r, using 0",
                    _os.environ.get("OMNI_THINKER_MIN_OUTPUT_TOKENS_BEFORE_DEFER"),
                )
                _min_output_tokens_before_defer = 0
        elif _is_talker_stage and _talker_env_val is not None:
            self._bp_governor_role = "talker"
            try:
                _lead_max = int(_talker_env_val)
            except ValueError:
                init_logger(__name__).warning(
                    "Invalid OMNI_TALKER_LEAD_MAX=%r, ignoring", _talker_env_val
                )
            _env_lo = _os.environ.get("OMNI_TALKER_LEAD_LO")
            if _env_lo is not None:
                try:
                    _lead_lo = int(_env_lo)
                except ValueError:
                    init_logger(__name__).warning(
                        "Invalid OMNI_TALKER_LEAD_LO=%r, ignoring", _env_lo
                    )
            _env_hold = _os.environ.get("OMNI_TALKER_DEFER_HOLD_MAX")
            if _env_hold is not None:
                try:
                    _hold_max = max(1, int(_env_hold))
                except ValueError:
                    init_logger(__name__).warning(
                        "Invalid OMNI_TALKER_DEFER_HOLD_MAX=%r, ignoring", _env_hold
                    )
            try:
                _speaker_rate = max(1, int(_os.environ.get("OMNI_TALKER_SPEAKER_RATE", "25")))
            except ValueError:
                init_logger(__name__).warning(
                    "Invalid OMNI_TALKER_SPEAKER_RATE=%r, using 25",
                    _os.environ.get("OMNI_TALKER_SPEAKER_RATE"),
                )
                _speaker_rate = 25
            try:
                _min_output_tokens_before_defer = max(
                    0,
                    int(_os.environ.get("OMNI_TALKER_MIN_OUTPUT_TOKENS_BEFORE_DEFER", "0")),
                )
            except ValueError:
                init_logger(__name__).warning(
                    "Invalid OMNI_TALKER_MIN_OUTPUT_TOKENS_BEFORE_DEFER=%r, using 0",
                    _os.environ.get("OMNI_TALKER_MIN_OUTPUT_TOKENS_BEFORE_DEFER"),
                )
                _min_output_tokens_before_defer = 0
        if _is_talker_stage and _lead_max is None:
            self._bp_governor_role = "talker"
            _omni_kv = getattr(model_config, "omni_kv_config", None)
            if isinstance(_omni_kv, dict):
                _lead_max = _omni_kv.get("talker_lead_max")
            elif _omni_kv is not None:
                _lead_max = getattr(_omni_kv, "talker_lead_max", None)
        _min_pending_age_ms = self._bp_min_pending_age_ms
        if self._bp_governor_role == "thinker":
            _min_pending_age_ms = self._bp_thinker_min_pending_age_ms
        elif self._bp_governor_role == "talker":
            _min_pending_age_ms = self._bp_talker_min_pending_age_ms
        _min_pending_age_s = (
            _min_pending_age_ms / 1000.0
            if self._bp_governor_role in self._bp_pending_age_roles
            else 0.0
        )
        self._bp_role_min_pending_age_s = max(0.0, float(_min_pending_age_s))
        _min_pending_count = self._bp_min_pending_count
        if self._bp_governor_role == "thinker":
            _min_pending_count = self._bp_thinker_min_pending_count
        elif self._bp_governor_role == "talker":
            _min_pending_count = self._bp_talker_min_pending_count
        self._bp_role_min_pending_count = max(1, int(_min_pending_count))
        _min_backlog_active_steps = self._bp_min_backlog_active_steps
        if self._bp_governor_role == "thinker":
            _min_backlog_active_steps = self._bp_thinker_min_backlog_active_steps
        elif self._bp_governor_role == "talker":
            _min_backlog_active_steps = self._bp_talker_min_backlog_active_steps
        self._bp_role_min_backlog_active_steps = max(
            1, int(_min_backlog_active_steps)
        )
        _min_running_count = self._bp_min_running_count
        if self._bp_governor_role == "thinker":
            _min_running_count = self._bp_thinker_min_running_count
        elif self._bp_governor_role == "talker":
            _min_running_count = self._bp_talker_min_running_count
        self._bp_role_min_running_count = max(1, int(_min_running_count))
        _min_pressure_active_calls = self._bp_min_pressure_active_calls
        if self._bp_governor_role == "thinker":
            _min_pressure_active_calls = self._bp_thinker_min_pressure_active_calls
        elif self._bp_governor_role == "talker":
            _min_pressure_active_calls = self._bp_talker_min_pressure_active_calls
        self._bp_role_min_pressure_active_calls = max(
            0, int(_min_pressure_active_calls)
        )
        _session_kv_gib_high = self._bp_session_kv_gib_high
        if self._bp_governor_role == "thinker":
            _session_kv_gib_high = self._bp_thinker_session_kv_gib_high
        elif self._bp_governor_role == "talker":
            _session_kv_gib_high = self._bp_talker_session_kv_gib_high
        self._bp_role_session_kv_gib_high = max(0.0, float(_session_kv_gib_high))
        _req_kv_gib_high = self._bp_req_kv_gib_high
        if self._bp_governor_role == "thinker":
            _req_kv_gib_high = self._bp_thinker_req_kv_gib_high
        elif self._bp_governor_role == "talker":
            _req_kv_gib_high = self._bp_talker_req_kv_gib_high
        self._bp_role_req_kv_gib_high = max(0.0, float(_req_kv_gib_high))
        _kv_pressure_suppress_defer = self._bp_kv_pressure_suppress_defer
        if self._bp_governor_role == "thinker":
            _kv_pressure_suppress_defer = (
                self._bp_thinker_kv_pressure_suppress_defer
            )
        elif self._bp_governor_role == "talker":
            _kv_pressure_suppress_defer = (
                self._bp_talker_kv_pressure_suppress_defer
            )
        self._bp_role_kv_pressure_suppress_defer = bool(
            _kv_pressure_suppress_defer
        )
        _pressure_bypass_pending_count = self._bp_pressure_bypass_pending_count
        if self._bp_governor_role == "thinker":
            _pressure_bypass_pending_count = (
                self._bp_thinker_pressure_bypass_pending_count
            )
        elif self._bp_governor_role == "talker":
            _pressure_bypass_pending_count = (
                self._bp_talker_pressure_bypass_pending_count
            )
        self._bp_role_pressure_bypass_pending_count = max(
            0, int(_pressure_bypass_pending_count)
        )
        _pressure_bypass_pending_age_ms = self._bp_pressure_bypass_pending_age_ms
        if self._bp_governor_role == "thinker":
            _pressure_bypass_pending_age_ms = (
                self._bp_thinker_pressure_bypass_pending_age_ms
            )
        elif self._bp_governor_role == "talker":
            _pressure_bypass_pending_age_ms = (
                self._bp_talker_pressure_bypass_pending_age_ms
            )
        self._bp_role_pressure_bypass_pending_age_s = max(
            0.0, float(_pressure_bypass_pending_age_ms) / 1000.0
        )
        _pressure_fast_warmup_pending_count = (
            self._bp_pressure_fast_warmup_pending_count
        )
        if self._bp_governor_role == "thinker":
            _pressure_fast_warmup_pending_count = (
                self._bp_thinker_pressure_fast_warmup_pending_count
            )
        elif self._bp_governor_role == "talker":
            _pressure_fast_warmup_pending_count = (
                self._bp_talker_pressure_fast_warmup_pending_count
            )
        self._bp_role_pressure_fast_warmup_pending_count = max(
            0, int(_pressure_fast_warmup_pending_count)
        )
        _pressure_fast_warmup_active_calls = (
            self._bp_pressure_fast_warmup_active_calls
        )
        if self._bp_governor_role == "thinker":
            _pressure_fast_warmup_active_calls = (
                self._bp_thinker_pressure_fast_warmup_active_calls
            )
        elif self._bp_governor_role == "talker":
            _pressure_fast_warmup_active_calls = (
                self._bp_talker_pressure_fast_warmup_active_calls
            )
        self._bp_role_pressure_fast_warmup_active_calls = max(
            0, int(_pressure_fast_warmup_active_calls)
        )
        _pressure_fast_warmup_pending_age_ms = (
            self._bp_pressure_fast_warmup_pending_age_ms
        )
        if self._bp_governor_role == "thinker":
            _pressure_fast_warmup_pending_age_ms = (
                self._bp_thinker_pressure_fast_warmup_pending_age_ms
            )
        elif self._bp_governor_role == "talker":
            _pressure_fast_warmup_pending_age_ms = (
                self._bp_talker_pressure_fast_warmup_pending_age_ms
            )
        self._bp_role_pressure_fast_warmup_pending_age_s = max(
            0.0, float(_pressure_fast_warmup_pending_age_ms) / 1000.0
        )
        _pressure_fast_warmup_recent_arrivals = (
            self._bp_pressure_fast_warmup_recent_arrivals
        )
        if self._bp_governor_role == "thinker":
            _pressure_fast_warmup_recent_arrivals = (
                self._bp_thinker_pressure_fast_warmup_recent_arrivals
            )
        elif self._bp_governor_role == "talker":
            _pressure_fast_warmup_recent_arrivals = (
                self._bp_talker_pressure_fast_warmup_recent_arrivals
            )
        self._bp_role_pressure_fast_warmup_recent_arrivals = max(
            0, int(_pressure_fast_warmup_recent_arrivals)
        )
        _pressure_fast_warmup_arrival_window_ms = (
            self._bp_pressure_fast_warmup_arrival_window_ms
        )
        if self._bp_governor_role == "thinker":
            _pressure_fast_warmup_arrival_window_ms = (
                self._bp_thinker_pressure_fast_warmup_arrival_window_ms
            )
        elif self._bp_governor_role == "talker":
            _pressure_fast_warmup_arrival_window_ms = (
                self._bp_talker_pressure_fast_warmup_arrival_window_ms
            )
        self._bp_role_pressure_fast_warmup_arrival_window_s = max(
            0.0, float(_pressure_fast_warmup_arrival_window_ms) / 1000.0
        )
        _pressure_fast_warmup_include_running_arrivals = (
            self._bp_pressure_fast_warmup_include_running_arrivals
        )
        if self._bp_governor_role == "thinker":
            _pressure_fast_warmup_include_running_arrivals = (
                self._bp_thinker_pressure_fast_warmup_include_running_arrivals
            )
        elif self._bp_governor_role == "talker":
            _pressure_fast_warmup_include_running_arrivals = (
                self._bp_talker_pressure_fast_warmup_include_running_arrivals
            )
        self._bp_role_pressure_fast_warmup_include_running_arrivals = bool(
            _pressure_fast_warmup_include_running_arrivals
        )
        _pressure_fast_warmup_arrival_history = (
            self._bp_pressure_fast_warmup_arrival_history
        )
        if self._bp_governor_role == "thinker":
            _pressure_fast_warmup_arrival_history = (
                self._bp_thinker_pressure_fast_warmup_arrival_history
            )
        elif self._bp_governor_role == "talker":
            _pressure_fast_warmup_arrival_history = (
                self._bp_talker_pressure_fast_warmup_arrival_history
            )
        self._bp_role_pressure_fast_warmup_arrival_history = bool(
            _pressure_fast_warmup_arrival_history
        )
        _pressure_fast_warmup_pending_growth = (
            self._bp_pressure_fast_warmup_pending_growth
        )
        if self._bp_governor_role == "thinker":
            _pressure_fast_warmup_pending_growth = (
                self._bp_thinker_pressure_fast_warmup_pending_growth
            )
        elif self._bp_governor_role == "talker":
            _pressure_fast_warmup_pending_growth = (
                self._bp_talker_pressure_fast_warmup_pending_growth
            )
        self._bp_role_pressure_fast_warmup_pending_growth = max(
            0, int(_pressure_fast_warmup_pending_growth)
        )
        _pressure_fast_warmup_growth_hold_calls = (
            self._bp_pressure_fast_warmup_growth_hold_calls
        )
        if self._bp_governor_role == "thinker":
            _pressure_fast_warmup_growth_hold_calls = (
                self._bp_thinker_pressure_fast_warmup_growth_hold_calls
            )
        elif self._bp_governor_role == "talker":
            _pressure_fast_warmup_growth_hold_calls = (
                self._bp_talker_pressure_fast_warmup_growth_hold_calls
            )
        self._bp_role_pressure_fast_warmup_growth_hold_calls = max(
            0, int(_pressure_fast_warmup_growth_hold_calls)
        )
        _pressure_fast_warmup_require_all_signals = (
            self._bp_pressure_fast_warmup_require_all_signals
        )
        if self._bp_governor_role == "thinker":
            _pressure_fast_warmup_require_all_signals = (
                self._bp_thinker_pressure_fast_warmup_require_all_signals
            )
        elif self._bp_governor_role == "talker":
            _pressure_fast_warmup_require_all_signals = (
                self._bp_talker_pressure_fast_warmup_require_all_signals
            )
        self._bp_role_pressure_fast_warmup_require_all_signals = bool(
            _pressure_fast_warmup_require_all_signals
        )
        _max_new_defer_per_call = self._bp_max_new_defer_per_call
        if self._bp_governor_role == "thinker":
            _max_new_defer_per_call = self._bp_thinker_max_new_defer_per_call
        elif self._bp_governor_role == "talker":
            _max_new_defer_per_call = self._bp_talker_max_new_defer_per_call
        self._bp_role_max_new_defer_per_call = max(
            0, int(_max_new_defer_per_call)
        )
        _max_defer_episodes_per_window = (
            self._bp_max_defer_episodes_per_window
        )
        if self._bp_governor_role == "thinker":
            _max_defer_episodes_per_window = (
                self._bp_thinker_max_defer_episodes_per_window
            )
        elif self._bp_governor_role == "talker":
            _max_defer_episodes_per_window = (
                self._bp_talker_max_defer_episodes_per_window
            )
        self._bp_role_max_defer_episodes_per_window = max(
            0, int(_max_defer_episodes_per_window)
        )
        _defer_budget_empty_reset_steps = (
            self._bp_defer_budget_empty_reset_steps
        )
        if self._bp_governor_role == "thinker":
            _defer_budget_empty_reset_steps = (
                self._bp_thinker_defer_budget_empty_reset_steps
            )
        elif self._bp_governor_role == "talker":
            _defer_budget_empty_reset_steps = (
                self._bp_talker_defer_budget_empty_reset_steps
            )
        self._bp_role_defer_budget_empty_reset_steps = max(
            0, int(_defer_budget_empty_reset_steps)
        )
        _defer_cooldown_steps = self._bp_defer_cooldown_steps
        if self._bp_governor_role == "thinker":
            _defer_cooldown_steps = self._bp_thinker_defer_cooldown_steps
        elif self._bp_governor_role == "talker":
            _defer_cooldown_steps = self._bp_talker_defer_cooldown_steps
        self._bp_role_defer_cooldown_steps = max(
            0, int(_defer_cooldown_steps)
        )
        _prefill_priority = self._bp_prefill_priority
        if self._bp_governor_role == "thinker":
            _prefill_priority = self._bp_thinker_prefill_priority
        elif self._bp_governor_role == "talker":
            _prefill_priority = self._bp_talker_prefill_priority
        self._bp_role_prefill_priority = bool(_prefill_priority)
        _prefill_priority_window_steps = self._bp_prefill_priority_window_steps
        if self._bp_governor_role == "thinker":
            _prefill_priority_window_steps = (
                self._bp_thinker_prefill_priority_window_steps
            )
        elif self._bp_governor_role == "talker":
            _prefill_priority_window_steps = (
                self._bp_talker_prefill_priority_window_steps
            )
        self._bp_role_prefill_priority_window_steps = max(
            0, int(_prefill_priority_window_steps)
        )
        _include_chunk_waiting = self._bp_include_chunk_waiting_in_pending
        if self._bp_governor_role == "thinker":
            _include_chunk_waiting = (
                self._bp_thinker_include_chunk_waiting_in_pending
            )
        elif self._bp_governor_role == "talker":
            _include_chunk_waiting = (
                self._bp_talker_include_chunk_waiting_in_pending
            )
        self._bp_role_include_chunk_waiting_in_pending = bool(
            _include_chunk_waiting
        )
        _waiting_short_max_output_tokens = (
            self._bp_waiting_short_max_output_tokens
        )
        if self._bp_governor_role == "thinker":
            _waiting_short_max_output_tokens = (
                self._bp_thinker_waiting_short_max_output_tokens
            )
        elif self._bp_governor_role == "talker":
            _waiting_short_max_output_tokens = (
                self._bp_talker_waiting_short_max_output_tokens
            )
        self._bp_role_waiting_short_max_output_tokens = max(
            0, int(_waiting_short_max_output_tokens)
        )
        _waiting_short_min_count = self._bp_waiting_short_min_count
        if self._bp_governor_role == "thinker":
            _waiting_short_min_count = self._bp_thinker_waiting_short_min_count
        elif self._bp_governor_role == "talker":
            _waiting_short_min_count = self._bp_talker_waiting_short_min_count
        self._bp_role_waiting_short_min_count = max(
            0, int(_waiting_short_min_count)
        )
        _waiting_short_latch_calls = self._bp_waiting_short_latch_calls
        if self._bp_governor_role == "thinker":
            _waiting_short_latch_calls = (
                self._bp_thinker_waiting_short_latch_calls
            )
        elif self._bp_governor_role == "talker":
            _waiting_short_latch_calls = (
                self._bp_talker_waiting_short_latch_calls
            )
        self._bp_role_waiting_short_latch_calls = max(
            0, int(_waiting_short_latch_calls)
        )
        self._bp_waiting_short_latch_remaining = 0
        _waiting_short_entry_only = self._bp_waiting_short_entry_only
        if self._bp_governor_role == "thinker":
            _waiting_short_entry_only = (
                self._bp_thinker_waiting_short_entry_only
            )
        elif self._bp_governor_role == "talker":
            _waiting_short_entry_only = (
                self._bp_talker_waiting_short_entry_only
            )
        self._bp_role_waiting_short_entry_only = bool(
            _waiting_short_entry_only
        )
        _skip_global_suppressed_defer = self._bp_skip_global_suppressed_defer
        if self._bp_governor_role == "thinker":
            _skip_global_suppressed_defer = (
                self._bp_thinker_skip_global_suppressed_defer
            )
        elif self._bp_governor_role == "talker":
            _skip_global_suppressed_defer = (
                self._bp_talker_skip_global_suppressed_defer
            )
        self._bp_role_skip_global_suppressed_defer = bool(
            _skip_global_suppressed_defer
        )
        self._lead_governor_params = LeadGovernorParams(
            L_max=_lead_max,
            L_lo=_lead_lo,
            hold_max=_hold_max,
            speaker_rate=_speaker_rate,
            min_pending_count=self._bp_role_min_pending_count,
            min_backlog_active_steps=self._bp_role_min_backlog_active_steps,
            min_running_count=self._bp_role_min_running_count,
            min_pressure_active_calls=self._bp_role_min_pressure_active_calls,
            pressure_bypass_pending_count=(
                self._bp_role_pressure_bypass_pending_count
            ),
            pressure_bypass_pending_age_s=(
                self._bp_role_pressure_bypass_pending_age_s
            ),
            pressure_fast_warmup_pending_count=(
                self._bp_role_pressure_fast_warmup_pending_count
            ),
            pressure_fast_warmup_active_calls=(
                self._bp_role_pressure_fast_warmup_active_calls
            ),
            pressure_fast_warmup_pending_age_s=(
                self._bp_role_pressure_fast_warmup_pending_age_s
            ),
            pressure_fast_warmup_recent_arrivals=(
                self._bp_role_pressure_fast_warmup_recent_arrivals
            ),
            pressure_fast_warmup_pending_growth=(
                self._bp_role_pressure_fast_warmup_pending_growth
            ),
            pressure_fast_warmup_growth_hold_calls=(
                self._bp_role_pressure_fast_warmup_growth_hold_calls
            ),
            pressure_fast_warmup_require_all_signals=(
                self._bp_role_pressure_fast_warmup_require_all_signals
            ),
            resume_on_empty_backlog=self._bp_resume_on_empty_backlog,
            empty_backlog_grace_steps=self._bp_empty_backlog_grace_steps,
            min_output_tokens_before_defer=_min_output_tokens_before_defer,
            min_pending_age_s=_min_pending_age_s,
            max_new_defer_per_call=self._bp_role_max_new_defer_per_call,
            max_defer_episodes_per_window=(
                self._bp_role_max_defer_episodes_per_window
            ),
            defer_budget_empty_reset_steps=(
                self._bp_role_defer_budget_empty_reset_steps
            ),
            defer_cooldown_steps=self._bp_role_defer_cooldown_steps,
            prefill_priority_window_steps=(
                self._bp_role_prefill_priority_window_steps
            ),
        )
        self._talker_lead_governor: TalkerLeadGovernor = TalkerLeadGovernor(
            **self._lead_governor_params.as_governor_kwargs()
        )
        if self._talker_lead_governor.enabled:
            init_logger(__name__).info(
                "[scheduler adaptive] TalkerLeadGovernor enabled: role=%s L_hi=%d L_lo=%d "
                "hold_max=%d rate=%d min_pending=%d min_backlog_steps=%d "
                "min_running=%d min_pressure_calls=%d "
                "pressure_bypass_pending=%d pressure_bypass_age_ms=%.3f "
                "pressure_fast_pending=%d pressure_fast_calls=%d "
                "pressure_fast_age_ms=%.3f "
                "pressure_fast_recent=%d pressure_fast_recent_window_ms=%.3f "
                "pressure_fast_include_running_arrivals=%s "
                "pressure_fast_arrival_history=%s "
                "pressure_fast_growth=%d pressure_fast_growth_hold=%d "
                "pressure_fast_require_all=%s "
                "max_new_defer_per_call=%d "
                "max_defer_episodes_per_window=%d "
                "defer_budget_empty_reset_steps=%d "
                "defer_cooldown_steps=%d "
                "prefill_priority=%s prefill_priority_window_steps=%d "
                "include_chunk_waiting_pending=%s "
                "waiting_short_max_output_tokens=%d "
                "waiting_short_min_count=%d "
                "waiting_short_latch_calls=%d "
                "waiting_short_entry_only=%s "
                "skip_global_suppressed_defer=%s "
                "min_pending_age_ms=%.3f pending_age_roles=%s "
                "resume_empty_backlog=%s empty_backlog_grace=%d min_output_tokens=%d "
                "kv_aware=%s kv_aware_roles=%s long_ctx_tokens=%d "
                "kv_usage_high=%.6f session_kv_gib_high=%.3f req_kv_gib_high=%.3f "
                "role_session_kv_gib_high=%.3f role_req_kv_gib_high=%.3f "
                "kv_pressure_suppress_defer=%s "
                "talker_disable_on_video_mix=%s video_mix_markers=%s "
                "video_mix_allow_pending=%d video_mix_allow_age_ms=%.3f "
                "(stage_id=%s)",
                self._bp_governor_role or "unknown",
                self._talker_lead_governor.L_hi,
                self._talker_lead_governor.L_lo,
                self._talker_lead_governor.hold_max,
                _speaker_rate,
                self._bp_role_min_pending_count,
                self._bp_role_min_backlog_active_steps,
                self._bp_role_min_running_count,
                self._bp_role_min_pressure_active_calls,
                self._bp_role_pressure_bypass_pending_count,
                self._bp_role_pressure_bypass_pending_age_s * 1000.0,
                self._bp_role_pressure_fast_warmup_pending_count,
                self._bp_role_pressure_fast_warmup_active_calls,
                self._bp_role_pressure_fast_warmup_pending_age_s * 1000.0,
                self._bp_role_pressure_fast_warmup_recent_arrivals,
                self._bp_role_pressure_fast_warmup_arrival_window_s * 1000.0,
                self._bp_role_pressure_fast_warmup_include_running_arrivals,
                self._bp_role_pressure_fast_warmup_arrival_history,
                self._bp_role_pressure_fast_warmup_pending_growth,
                self._bp_role_pressure_fast_warmup_growth_hold_calls,
                self._bp_role_pressure_fast_warmup_require_all_signals,
                self._bp_role_max_new_defer_per_call,
                self._bp_role_max_defer_episodes_per_window,
                self._bp_role_defer_budget_empty_reset_steps,
                self._bp_role_defer_cooldown_steps,
                self._bp_role_prefill_priority,
                self._bp_role_prefill_priority_window_steps,
                self._bp_role_include_chunk_waiting_in_pending,
                self._bp_role_waiting_short_max_output_tokens,
                self._bp_role_waiting_short_min_count,
                self._bp_role_waiting_short_latch_calls,
                self._bp_role_waiting_short_entry_only,
                self._bp_role_skip_global_suppressed_defer,
                _min_pending_age_s * 1000.0,
                sorted(self._bp_pending_age_roles),
                self._bp_resume_on_empty_backlog,
                self._bp_empty_backlog_grace_steps,
                _min_output_tokens_before_defer,
                self._bp_kv_aware,
                sorted(self._bp_kv_aware_roles),
                self._bp_long_context_tokens,
                self._bp_kv_usage_high,
                self._bp_session_kv_gib_high,
                self._bp_req_kv_gib_high,
                self._bp_role_session_kv_gib_high,
                self._bp_role_req_kv_gib_high,
                self._bp_role_kv_pressure_suppress_defer,
                self._bp_talker_disable_on_video_mix,
                self._bp_talker_video_mix_markers,
                self._bp_talker_video_mix_allow_pending_count,
                self._bp_talker_video_mix_allow_pending_age_ms,
                getattr(model_config, "stage_id", "?"),
            )
        elif (
            (_talker_env_val is not None and _talker_env_val not in ("", "0"))
            and not _is_talker_stage
        ):
            logger.info(
                "[scheduler adaptive] Talker governor disabled on non-talker stage_id=%s "
                "(talker_stage_ids=%s)",
                _stage_id_raw,
                sorted(_talker_stage_ids),
            )
        elif (
            (_thinker_env_val is not None and _thinker_env_val not in ("", "0"))
            and not _is_thinker_stage
        ):
            logger.info(
                "[scheduler adaptive] Thinker governor disabled on non-thinker stage_id=%s "
                "(thinker_stage_ids=%s)",
                _stage_id_raw,
                sorted(_thinker_stage_ids),
            )

    def _get_confirmed_num_computed_tokens(self, request: Request) -> int:
        """num_computed_tokens minus async placeholders (KV actually on GPU)."""
        # Output placeholders are zero when async scheduling isn't used
        return request.num_computed_tokens - request.num_output_placeholders

    def _bp_block_size(self) -> int | None:
        try:
            coord = self.kv_cache_manager.coordinator
            block_size = int(getattr(coord, "block_size", 0) or 0)
            if block_size > 0:
                return block_size
            managers = getattr(coord, "single_type_managers", ()) or ()
            if managers:
                block_size = int(getattr(managers[0], "block_size", 0) or 0)
                if block_size > 0:
                    return block_size
        except Exception:
            pass
        try:
            block_size = int(getattr(self.cache_config, "block_size", 0) or 0)
            if block_size > 0:
                return block_size
        except Exception:
            pass
        return None

    def _bp_allocated_blocks_by_req(self) -> dict[str, int]:
        out: dict[str, int] = {}
        try:
            managers = self.kv_cache_manager.coordinator.single_type_managers
        except Exception:
            return out
        try:
            for manager in managers:
                req_to_blocks = getattr(manager, "req_to_blocks", {}) or {}
                for rid, blocks in req_to_blocks.items():
                    out[str(rid)] = max(out.get(str(rid), 0), len(blocks))
        except Exception:
            return out
        return out

    def _bp_pressure_context(self) -> dict[str, Any]:
        allocated_blocks = self._bp_allocated_blocks_by_req()
        block_size = self._bp_block_size()
        bytes_per_token = max(1, int(getattr(self, "_bp_kv_bytes_per_token", 98304)))
        session_kv_gib = None
        if block_size:
            session_tokens = sum(allocated_blocks.values()) * block_size
            session_kv_gib = session_tokens * bytes_per_token / (1024**3)
        kv_usage = None
        try:
            kv_usage = float(self.kv_cache_manager.usage())
        except Exception:
            pass
        pressure_high = False
        usage_high = float(getattr(self, "_bp_kv_usage_high", 0.0) or 0.0)
        session_high = float(
            getattr(
                self,
                "_bp_role_session_kv_gib_high",
                getattr(self, "_bp_session_kv_gib_high", 0.0),
            )
            or 0.0
        )
        if kv_usage is not None and usage_high > 0.0 and kv_usage >= usage_high:
            pressure_high = True
        if session_kv_gib is not None and session_high > 0.0 and session_kv_gib >= session_high:
            pressure_high = True
        return {
            "allocated_blocks": allocated_blocks,
            "block_size": block_size,
            "bytes_per_token": bytes_per_token,
            "kv_usage": kv_usage,
            "session_kv_gib": session_kv_gib,
            "pressure_high": pressure_high,
        }

    def _bp_pending_age_s(self) -> float | None:
        """Return the oldest waiting request age in seconds, if observable."""
        try:
            queue = self.waiting
        except Exception:
            return None
        try:
            now_wall = time()
            oldest_age: float | None = None
            for req in queue:
                arrival_time = getattr(req, "arrival_time", None)
                if arrival_time is None:
                    continue
                try:
                    age = now_wall - float(arrival_time)
                except (TypeError, ValueError):
                    continue
                if age < 0.0:
                    continue
                oldest_age = age if oldest_age is None else max(oldest_age, age)
            return oldest_age
        except Exception:
            return None

    def _bp_request_needs_prefill(self, request: Request) -> bool:
        """True while a request still needs prompt/prefill compute."""
        try:
            prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or 0)
            if prompt_tokens <= 0:
                return False
            confirmed_computed = self._get_confirmed_num_computed_tokens(request)
            return confirmed_computed < prompt_tokens
        except Exception:
            return False

    def _bp_prefill_pressure(self) -> tuple[int | None, float | None]:
        """Count waiting/running requests whose prefill is not complete."""
        try:
            queues = (self.waiting, self.running)
        except Exception:
            return None, None
        try:
            now_wall = time()
            seen: set[str] = set()
            count = 0
            oldest_age: float | None = None
            for queue in queues:
                for req in queue:
                    req_id = getattr(req, "request_id", None)
                    if req_id is not None:
                        req_key = str(req_id)
                        if req_key in seen:
                            continue
                        seen.add(req_key)
                    if not self._bp_request_needs_prefill(req):
                        continue
                    count += 1
                    arrival_time = getattr(req, "arrival_time", None)
                    if arrival_time is None:
                        continue
                    try:
                        age = now_wall - float(arrival_time)
                    except (TypeError, ValueError):
                        continue
                    if age < 0.0:
                        continue
                    oldest_age = age if oldest_age is None else max(oldest_age, age)
            return count, oldest_age
        except Exception:
            return None, None

    def _bp_chunk_waiting_pressure(self) -> tuple[int, float | None]:
        """Count async-chunk requests hidden from waiting/running queues."""
        adapter = getattr(self, "chunk_transfer_adapter", None)
        if adapter is None:
            return 0, None
        pressure = getattr(adapter, "waiting_for_chunk_pressure", None)
        if pressure is None:
            return 0, None
        try:
            count, age = pressure()
            return max(0, int(count)), age
        except Exception:
            return 0, None

    @staticmethod
    def _bp_add_pending_pressure(
        pending_count: int | None,
        pending_age_s: float | None,
        extra_count: int,
        extra_age_s: float | None,
    ) -> tuple[int | None, float | None]:
        if extra_count <= 0:
            return pending_count, pending_age_s
        merged_count = extra_count if pending_count is None else pending_count + extra_count
        if extra_age_s is None:
            return merged_count, pending_age_s
        if pending_age_s is None:
            return merged_count, extra_age_s
        return merged_count, max(pending_age_s, extra_age_s)

    def _bp_recent_waiting_arrivals(
        self,
        window_s: float,
        include_running: bool = False,
        use_history: bool = False,
    ) -> int | None:
        """Return recent arrivals still visible in scheduler queues."""
        if window_s <= 0.0:
            return None
        try:
            now_wall = time()
            if use_history:
                queue_names = ("waiting", "running")
                for queue_name in queue_names:
                    try:
                        queue = getattr(self, queue_name)
                    except Exception:
                        continue
                    for req in queue:
                        req_id = getattr(req, "request_id", None)
                        if req_id is None:
                            continue
                        req_key = str(req_id)
                        if req_key in self._bp_arrival_history_seen:
                            continue
                        self._bp_arrival_history_seen.add(req_key)
                        arrival_time = getattr(req, "arrival_time", None)
                        try:
                            arrival_ts = (
                                float(arrival_time)
                                if arrival_time is not None
                                else now_wall
                            )
                        except (TypeError, ValueError):
                            arrival_ts = now_wall
                        if 0.0 <= now_wall - arrival_ts <= window_s:
                            self._bp_arrival_history.append((arrival_ts, req_key))
                while (
                    self._bp_arrival_history
                    and now_wall - self._bp_arrival_history[0][0] > window_s
                ):
                    self._bp_arrival_history.popleft()
                return len(self._bp_arrival_history)

            recent = 0
            seen: set[str] = set()
            queue_names = ("waiting", "running") if include_running else ("waiting",)
            for queue_name in queue_names:
                try:
                    queue = getattr(self, queue_name)
                except Exception:
                    continue
                for req in queue:
                    req_id = getattr(req, "request_id", None)
                    if req_id is not None:
                        req_key = str(req_id)
                        if req_key in seen:
                            continue
                        seen.add(req_key)
                    arrival_time = getattr(req, "arrival_time", None)
                    if arrival_time is None:
                        continue
                    try:
                        age = now_wall - float(arrival_time)
                    except (TypeError, ValueError):
                        continue
                    if 0.0 <= age <= window_s:
                        recent += 1
            return recent
        except Exception:
            return None

    def _bp_request_max_output_tokens(self, request: Any) -> int | None:
        """Best-effort max-output estimate for admission-shape gates."""
        try:
            sampling_params = getattr(request, "sampling_params", None)
        except Exception:
            sampling_params = None
        candidates: list[Any] = []
        if sampling_params is not None:
            for attr in ("max_tokens", "max_new_tokens", "output_tokens"):
                try:
                    candidates.append(getattr(sampling_params, attr, None))
                except Exception:
                    pass
            try:
                extra_args = getattr(sampling_params, "extra_args", None) or {}
                if isinstance(extra_args, dict):
                    for key in (
                        "max_tokens",
                        "max_new_tokens",
                        "output_tokens",
                        "force_output_len",
                        "forced_output_len",
                    ):
                        candidates.append(extra_args.get(key))
            except Exception:
                pass
        for value in candidates:
            if value is None:
                continue
            try:
                out = int(value)
            except (TypeError, ValueError):
                continue
            if out > 0:
                return out
        return None

    def _bp_waiting_short_count(self, max_output_tokens: int) -> int | None:
        """Count waiting requests whose configured max output looks short."""
        if max_output_tokens <= 0:
            return None
        try:
            queue = self.waiting
        except Exception:
            return None
        count = 0
        try:
            for req in queue:
                max_tokens = self._bp_request_max_output_tokens(req)
                if max_tokens is not None and max_tokens <= max_output_tokens:
                    count += 1
            return count
        except Exception:
            return None

    def _bp_waiting_short_allows_defer(self) -> bool:
        max_output_tokens = int(
            getattr(self, "_bp_role_waiting_short_max_output_tokens", 0) or 0
        )
        min_count = int(
            getattr(self, "_bp_role_waiting_short_min_count", 0) or 0
        )
        if max_output_tokens <= 0 or min_count <= 0:
            return True
        short_count = self._bp_waiting_short_count(max_output_tokens)
        latch_calls = max(
            0, int(getattr(self, "_bp_role_waiting_short_latch_calls", 0) or 0)
        )
        if short_count is not None and short_count >= min_count:
            if latch_calls > 0:
                self._bp_waiting_short_latch_remaining = latch_calls
            return True
        if latch_calls > 0:
            remaining = max(
                0,
                int(getattr(self, "_bp_waiting_short_latch_remaining", 0) or 0),
            )
            if remaining > 0:
                self._bp_waiting_short_latch_remaining = remaining - 1
                return True
        return False

    def _bp_request_identifier_text(self, request: Any) -> str:
        parts: list[str] = []
        for attr in ("request_id", "external_req_id", "session_id"):
            try:
                value = getattr(request, attr, None)
            except Exception:
                value = None
            if value:
                parts.append(str(value).lower())
        return " ".join(parts)

    def _bp_request_has_video_mix_marker(self, request: Any) -> bool:
        markers = getattr(self, "_bp_talker_video_mix_markers", ()) or ()
        if not markers:
            return False
        ident = self._bp_request_identifier_text(request)
        if ident and any(marker in ident for marker in markers):
            return True

        # Downstream talker requests may lose encoder features, so IDs are the
        # primary signal. This conservative fallback only treats explicit
        # visual multimodal feature labels as video/mix pressure.
        try:
            mm_features = getattr(request, "mm_features", None) or ()
        except Exception:
            mm_features = ()
        for feature in mm_features:
            try:
                feature_text = (
                    f"{type(feature).__name__} {getattr(feature, 'modality', '')} "
                    f"{getattr(feature, 'media_type', '')}"
                ).lower()
            except Exception:
                continue
            if "video" in feature_text or "image" in feature_text:
                return True
        return False

    def _bp_video_mix_active(self) -> bool:
        if not bool(getattr(self, "_bp_talker_disable_on_video_mix", False)):
            return False
        for queue_name in ("waiting", "running"):
            try:
                queue = getattr(self, queue_name)
            except Exception:
                continue
            try:
                for req in queue:
                    if self._bp_request_has_video_mix_marker(req):
                        return True
            except Exception:
                continue
        return False

    def _bp_talker_video_mix_allows_defer(
        self, pending_count: int | None, pending_age_s: float | None
    ) -> bool:
        """Allow talker BP in video/mix windows only under real queue pressure."""
        allow_pending = max(
            0, int(getattr(self, "_bp_talker_video_mix_allow_pending_count", 0) or 0)
        )
        allow_age_s = max(
            0.0,
            float(getattr(self, "_bp_talker_video_mix_allow_pending_age_ms", 0.0) or 0.0)
            / 1000.0,
        )
        if allow_pending <= 0 and allow_age_s <= 0.0:
            return False
        if allow_pending > 0 and pending_count is not None:
            try:
                if int(pending_count) >= allow_pending:
                    return True
            except (TypeError, ValueError):
                pass
        if allow_age_s > 0.0 and pending_age_s is not None:
            try:
                if float(pending_age_s) >= allow_age_s:
                    return True
            except (TypeError, ValueError):
                pass
        return False

    def _bp_defer_allowed(self, request: Request, ctx: dict[str, Any]) -> bool:
        if not bool(getattr(self, "_bp_kv_aware", True)):
            return True

        req_id = str(getattr(request, "request_id", ""))
        block_size = ctx.get("block_size")
        allocated_blocks = int((ctx.get("allocated_blocks") or {}).get(req_id, 0) or 0)
        allocated_tokens = allocated_blocks * int(block_size or 0)

        prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or 0)
        output_tokens = int(getattr(request, "num_output_tokens", 0) or 0)
        request_tokens = max(allocated_tokens, prompt_tokens + output_tokens, prompt_tokens)

        bytes_per_token = int(ctx.get("bytes_per_token") or getattr(self, "_bp_kv_bytes_per_token", 98304))
        # Allocated blocks can lag the logical prompt size at the exact moment
        # a defer decision is made. Use the larger logical/allocated estimate so
        # medium-long prompts are not missed by the request-KV gate.
        req_kv_tokens = max(allocated_tokens, request_tokens)
        req_kv_gib = (
            req_kv_tokens * bytes_per_token / (1024**3)
            if req_kv_tokens > 0
            else 0.0
        )

        req_kv_high = float(
            getattr(
                self,
                "_bp_role_req_kv_gib_high",
                getattr(self, "_bp_req_kv_gib_high", 0.0),
            )
            or 0.0
        )
        if req_kv_high > 0.0 and req_kv_gib >= req_kv_high:
            return False

        long_context_tokens = int(getattr(self, "_bp_long_context_tokens", 0) or 0)
        is_long_context = long_context_tokens > 0 and request_tokens >= long_context_tokens
        if is_long_context and bool(ctx.get("pressure_high", False)):
            return False

        return True

    def _get_kv_transfer_criteria(self) -> dict | None:
        # Note: vllm_config is available in Scheduler after super().__init__
        if not hasattr(self, "vllm_config"):
            return None

        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if omni_kv_config:
            if isinstance(omni_kv_config, dict):
                return omni_kv_config.get("kv_transfer_criteria", None)
            else:
                return getattr(omni_kv_config, "kv_transfer_criteria", None)
        return None

    def _request_omits_kv_transfer_to_next_stage(self, request: Request) -> bool:
        """True when orchestrator will not run stage 1+ for this request (e.g. text-only).

        The result is cached per request to avoid repeated deserialization of
        additional_information on every scheduler tick.
        """
        rid = request.request_id
        cached = self._omits_kv_transfer_cache.get(rid)
        if cached is not None:
            return cached

        payload = getattr(request, "additional_information", None)
        if payload is None:
            result = False
        else:
            info = deserialize_additional_information(payload)
            result = info.get("omni_final_stage_id") == 0

        self._omits_kv_transfer_cache[rid] = result
        return result

    def _process_kv_transfer_trigger(self, request: Request, new_token_ids: list[int]) -> bool:
        """
        Check triggers and process side effects (marking transfer).
        Returns True if request should be STOPPED.
        Returns False if request should continue (even if transfer was triggered).
        """
        if not self.kv_transfer_criteria:
            return False

        # Text-only requests finalize at stage 0; do not prefill-stop for DiT KV.
        if self._request_omits_kv_transfer_to_next_stage(request):
            return False

        if request.request_id in self.waiting_for_transfer_free:
            return False

        criteria_type = self.kv_transfer_criteria.get("type")
        stop_decode_on_trigger = self.kv_transfer_criteria.get("stop_after_transfer", True)

        if request.request_id in self.transfer_triggered_requests:
            # Deferred stop: once KV extraction is complete (no longer in
            # active_kv_transfers), stop the request.  This guarantees the
            # kv_ready signal was emitted while the request was still alive.
            if (
                request.request_id in self.pending_stop_after_extraction
                and request.request_id not in self.active_kv_transfers
            ):
                self.pending_stop_after_extraction.discard(request.request_id)
                request.status = RequestStatus.FINISHED_STOPPED
                return True
            return False

        # seq_len for KV transfer must exclude async placeholders.
        confirmed_computed = self._get_confirmed_num_computed_tokens(request)

        if criteria_type == "prefill_finished":
            if confirmed_computed >= request.num_prompt_tokens:
                self.transfer_triggered_requests.add(request.request_id)
                self._mark_request_for_kv_transfer(request.request_id, confirmed_computed)
                actually_queued = request.request_id in self.requests_needing_kv_transfer

                if stop_decode_on_trigger and actually_queued:
                    # Defer the stop until KV extraction completes so that
                    # the kv_ready signal can be emitted while the request
                    # is still alive.  The request will be stopped on the
                    # next scheduler step after extraction ack arrives.
                    self.pending_stop_after_extraction.add(request.request_id)

                return False

        elif criteria_type == "special_token":
            target_token_id = self.kv_transfer_criteria.get("token_id")
            if target_token_id is not None and target_token_id in new_token_ids:
                self.transfer_triggered_requests.add(request.request_id)

                try:
                    idx = new_token_ids.index(target_token_id)
                    tokens_to_exclude = len(new_token_ids) - (idx + 1)
                    snapshot_len = confirmed_computed - tokens_to_exclude
                except ValueError:
                    snapshot_len = confirmed_computed

                self._mark_request_for_kv_transfer(request.request_id, snapshot_len)
                actually_queued = request.request_id in self.requests_needing_kv_transfer

                if stop_decode_on_trigger and actually_queued:
                    self.pending_stop_after_extraction.add(request.request_id)

                return False

        return False

    def _bp_first_chunk_protection(self) -> tuple[set[str], float | None]:
        """Bound empty-backlog protection by actual ready/publish observations.

        process_pending_chunks must run first: a request waiting for upstream
        input must not keep other decoders paused. The adapter's first-ready
        deadline is never refreshed by subsequent chunks.
        """
        adapter = self.chunk_transfer_adapter
        budget_ms = getattr(self, "_bp_first_chunk_protect_ms", 0)
        if (budget_ms <= 0 or not self._bp_resume_on_empty_backlog
                or self._bp_governor_role != "talker" or adapter is None
                or not adapter.first_chunk_observation_enabled):
            return set(), None
        ready_ids = [request.request_id for request in (*self.waiting, *self.running)
                     if request.status != RequestStatus.WAITING_FOR_CHUNK and not request.is_finished()]
        return adapter.first_chunk_pending_ready(ready_ids, monotonic_ns(), budget_ms * 1_000_000), budget_ms / 1000

    def schedule(self) -> SchedulerOutput:  # type: ignore[override]
        trace_started_ns = monotonic_ns() if trace_enabled() else 0
        if trace_enabled():
            self._interaction_job_seq += 1
            trace_job_id = f"scheduler-{self._interaction_job_seq}"
            trace_emit(
                "scheduler_begin", **self._interaction_trace_context,
                job_id=trace_job_id, running=len(self.running), waiting=len(self.waiting),
                governor_enabled=self._talker_lead_governor.enabled,
            )
        self._omni_cancel_prefetch_under_live_pressure()
        self._omni_release_expired_gpu_pins()

        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.process_pending_chunks(self.waiting, self.running)

        # Lead-bounded decode scheduling: temporarily remove
        # requests whose generated-token lead exceeds L_hi from ``running`` so
        # that the base scheduler can prioritize waiting work. KV-aware policy
        # is thinker-only by default because thinker KV dominates memory.
        deferred_running: list = []
        first_chunk_protected: set[str] = set()
        first_chunk_time_limit_s: float | None = None
        if self._talker_lead_governor.enabled and self.running:
            try:
                first_chunk_protected, first_chunk_time_limit_s = self._bp_first_chunk_protection()
                # Pending counts the filtered waiting queue, not a newcomer
                # already admitted to running. Bounded first-codec protection
                # may continue an existing defer while that ready newcomer
                # produces its first chunk; other empty-backlog work resumes.
                _pending = len(self.waiting) if hasattr(self, "waiting") else None
                _pending_age = self._bp_pending_age_s()
                if self._bp_role_prefill_priority:
                    _prefill_pending, _prefill_age = self._bp_prefill_pressure()
                    if _prefill_pending is not None:
                        _pending = _prefill_pending
                        _pending_age = _prefill_age
                if self._bp_role_include_chunk_waiting_in_pending:
                    _chunk_pending, _chunk_age = self._bp_chunk_waiting_pressure()
                    _pending, _pending_age = self._bp_add_pending_pressure(
                        _pending, _pending_age, _chunk_pending, _chunk_age
                    )
                _recent_pending_arrivals = self._bp_recent_waiting_arrivals(
                    self._bp_role_pressure_fast_warmup_arrival_window_s,
                    include_running=(
                        self._bp_role_pressure_fast_warmup_include_running_arrivals
                    ),
                    use_history=(
                        self._bp_role_pressure_fast_warmup_arrival_history
                    ),
                )
                _defer_allowed = None
                _defer_entry_allowed = None
                _kv_defer_allowed = None
                _kv_pressure_suppresses_defer = False
                if (
                    self._bp_kv_aware
                    and self._bp_governor_role in self._bp_kv_aware_roles
                ):
                    _bp_ctx = self._bp_pressure_context()
                    _kv_defer_allowed = lambda req: self._bp_defer_allowed(req, _bp_ctx)
                    _kv_pressure_suppresses_defer = (
                        self._bp_role_kv_pressure_suppress_defer
                        and bool(_bp_ctx.get("pressure_high", False))
                    )
                _waiting_short_gate_enabled = (
                    self._bp_role_waiting_short_max_output_tokens > 0
                    and self._bp_role_waiting_short_min_count > 0
                )
                _waiting_short_allows_defer = (
                    self._bp_waiting_short_allows_defer()
                    if _waiting_short_gate_enabled
                    else True
                )
                _waiting_short_blocks_defer = (
                    _waiting_short_gate_enabled
                    and not _waiting_short_allows_defer
                )
                _waiting_short_force_suppresses = (
                    _waiting_short_blocks_defer
                    and not self._bp_role_waiting_short_entry_only
                )
                _video_mix_active = (
                    self._bp_governor_role == "talker"
                    and self._bp_video_mix_active()
                )
                _suppress_talker_video_mix = (
                    _video_mix_active
                    and not self._bp_talker_video_mix_allows_defer(
                        _pending, _pending_age
                    )
                )
                _global_defer_suppressed = (
                    _waiting_short_force_suppresses
                    or _suppress_talker_video_mix
                    or _kv_pressure_suppresses_defer
                )
                if (
                    _waiting_short_blocks_defer
                    and self._bp_role_waiting_short_entry_only
                ):
                    _defer_entry_allowed = lambda _req: False
                if (
                    _kv_defer_allowed is not None
                    or _suppress_talker_video_mix
                    or _waiting_short_force_suppresses
                    or first_chunk_protected
                ):
                    def _defer_allowed(req):
                        if req.request_id in first_chunk_protected:
                            return False
                        if _kv_pressure_suppresses_defer:
                            return False
                        if _kv_defer_allowed is not None and not _kv_defer_allowed(req):
                            return False
                        if _waiting_short_force_suppresses:
                            return False
                        if _suppress_talker_video_mix:
                            return False
                        return True
                if (
                    _global_defer_suppressed
                    and self._bp_role_skip_global_suppressed_defer
                    and not self._talker_lead_governor.has_deferred
                ):
                    self._talker_lead_governor.observe_global_suppressed_skip(
                        pending_count=_pending
                    )
                    deferred_running = []
                else:
                    deferred_running = self._talker_lead_governor.filter_running(
                        self.running,
                        pending_count=_pending,
                        pending_age_s=_pending_age,
                        recent_pending_arrivals=_recent_pending_arrivals,
                        defer_allowed=_defer_allowed,
                        defer_entry_allowed=_defer_entry_allowed,
                        empty_backlog_protected=bool(first_chunk_protected),
                        defer_time_limit_s=first_chunk_time_limit_s,
                    )
                if deferred_running:
                    if trace_enabled():
                        for request in deferred_running:
                            trace_emit(
                                "scheduler_defer", **self._interaction_trace_context,
                                request_id=request.request_id,
                                external_req_id=getattr(request, "external_req_id", None),
                                job_id=trace_job_id, reason="lead_governor",
                                pending_count=_pending, pending_age_s=_pending_age,
                                role=self._bp_governor_role,
                                generated_tokens=request.num_output_tokens,
                                first_chunk_protection_count=len(first_chunk_protected),
                                first_chunk_protect_ms=(first_chunk_time_limit_s or 0) * 1000,
                            )
                    deferred_set = {r.request_id for r in deferred_running}
                    self.running = [r for r in self.running if r.request_id not in deferred_set]
            except Exception:
                init_logger(__name__).exception(
                    "[scheduler adaptive] lead_governor.filter_running failed; bypassing"
                )
                deferred_running = []
        elif self._talker_lead_governor.enabled:
            try:
                _pending = len(self.waiting) if hasattr(self, "waiting") else None
                if self._bp_role_prefill_priority:
                    _prefill_pending, _ = self._bp_prefill_pressure()
                    if _prefill_pending is not None:
                        _pending = _prefill_pending
                if self._bp_role_include_chunk_waiting_in_pending:
                    _chunk_pending, _ = self._bp_chunk_waiting_pressure()
                    _pending, _ = self._bp_add_pending_pressure(
                        _pending, None, _chunk_pending, None
                    )
                self._talker_lead_governor.observe_idle(pending_count=_pending)
            except Exception:
                init_logger(__name__).exception(
                    "[scheduler adaptive] lead_governor.observe_idle failed; bypassing"
                )

        try:
            scheduler_output = super().schedule()
        finally:
            if self.chunk_transfer_adapter:
                # Add request waiting for chunk to the waiting and running queue
                self.chunk_transfer_adapter.restore_queues(self.waiting, self.running)
            # Reinstate deferred requests so they will be considered
            # again on the next scheduler call. Order is preserved (append at
            # the tail; vLLM's running queue is FIFO-ish but order does not
            # affect correctness, only fairness across deferred reqs).
            if deferred_running:
                self.running.extend(deferred_running)
        try:
            # Late import to avoid circulars in some launch modes
            from .output import OmniNewRequestData

            # Rewrap base NewRequestData entries with OmniNewRequestData,
            # enriching with request-level payloads
            new_list = []
            for nr in scheduler_output.scheduled_new_reqs:
                req_id = getattr(nr, "req_id", None)
                request = self.requests.get(req_id) if req_id else None
                # Build omni entry preserving all base fields
                omni_nr = OmniNewRequestData(
                    req_id=nr.req_id,
                    external_req_id=(getattr(request, "external_req_id", None) if request else None),
                    prompt_token_ids=nr.prompt_token_ids,
                    mm_features=nr.mm_features,
                    sampling_params=nr.sampling_params,
                    pooling_params=nr.pooling_params,
                    block_ids=nr.block_ids,
                    num_computed_tokens=nr.num_computed_tokens,
                    lora_request=nr.lora_request,
                    # Enrich with omni payloads from the live request object
                    prompt_embeds=(getattr(request, "prompt_embeds", None) if request else None),
                    additional_information=(getattr(request, "additional_information", None) if request else None),
                )
                new_list.append(omni_nr)

            scheduler_output.scheduled_new_reqs = new_list  # type: ignore[assignment]
            if self.chunk_transfer_adapter:
                self.chunk_transfer_adapter.postprocess_scheduler_output(scheduler_output, self.requests)
            # Add information about requests needing KV cache transfer
            finished_reqs = self.get_finished_requests_needing_kv_transfer()
        except Exception:
            # If anything goes wrong, leave the original output unchanged
            init_logger(__name__).exception("Failed to wrap scheduled_new_reqs with OmniNewRequestData")
            finished_reqs = {}

        # Wrap in omni scheduler output to carry transfer metadata.
        base_fields = SchedulerOutput.__dataclass_fields__.keys()
        base_data = {name: getattr(scheduler_output, name) for name in base_fields}
        result = OmniSchedulerOutput(
            **base_data,
            finished_requests_needing_kv_transfer=finished_reqs,
        )
        if trace_enabled():
            # Local CPU correlation only; this does not assert GPU start/end.
            result._interaction_job_id = trace_job_id
            for req_id, count in scheduler_output.num_scheduled_tokens.items():
                request = self.requests.get(req_id)
                # The base scheduler has already advanced this counter for
                # selected work. It is not a pre-schedule cache-hit count.
                trace_emit(
                    "scheduler_select", **self._interaction_trace_context,
                    request_id=req_id, external_req_id=getattr(request, "external_req_id", None),
                    job_id=trace_job_id, num_scheduled_tokens=count,
                    num_computed_tokens=getattr(request, "num_computed_tokens", None),
                )
            adapter = self.chunk_transfer_adapter
            trace_emit(
                "scheduler_end", **self._interaction_trace_context,
                job_id=trace_job_id, elapsed_ns=monotonic_ns() - trace_started_ns,
                running=len(self.running), waiting=len(self.waiting), deferred=len(deferred_running),
                selected=len(scheduler_output.num_scheduled_tokens),
                first_chunk_protection_count=len(first_chunk_protected),
                pending_chunk_loads=len(adapter._pending_load_reqs) if adapter else 0,
                pending_chunk_saves=len(adapter._pending_save_reqs) if adapter else 0,
            )
        return result

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        if trace_enabled():
            for req_id, count in num_scheduled_tokens.items():
                request = self.requests.get(req_id)
                trace_emit(
                    "scheduler_output_observed", **self._interaction_trace_context,
                    request_id=req_id, external_req_id=getattr(request, "external_req_id", None),
                    job_id=getattr(scheduler_output, "_interaction_job_id", None),
                    num_scheduled_tokens=count,
                )
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats: CUDAGraphStat | None = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # Pre-process KV extraction acks so that the per-request loop below
        # can see up-to-date active_kv_transfers state and emit kv_ready
        # signals while requests are still alive (before any deferred stop).
        kv_extracted_ids = getattr(model_runner_output, "kv_extracted_req_ids", None)
        if kv_extracted_ids:
            for req_id in kv_extracted_ids:
                try:
                    self.active_kv_transfers.discard(req_id)
                    req = self.requests.get(req_id)
                    if req is not None and not req.is_finished():
                        outputs[req.client_index].append(
                            EngineCoreOutput(
                                request_id=req_id,
                                new_token_ids=[],
                                kv_transfer_params={"kv_ready": True},
                            )
                        )
                except Exception:
                    init_logger(__name__).exception("Failed to pre-process KV extraction for %s", req_id)

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # Skip requests that were recovered from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or async scheduling).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            stopped = False
            is_segment_finished = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status
            finish_reason = None
            routed_experts = None

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            # If criteria returns True, it means we must STOP the request.
            # If criteria returns False, it might have triggered a background
            # transfer (e.g. prefill finished / special token) but continues decoding.
            if not stopped and self._process_kv_transfer_trigger(request, new_token_ids):
                stopped = True

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                if not struct_output_request.grammar.accept_tokens(req_id, new_token_ids):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. Terminating request.",
                        new_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            routed_experts_dict = getattr(model_runner_output, "routed_experts_dict", None)
            if routed_experts_dict is not None and req_id in routed_experts_dict:
                routed_experts = routed_experts_dict[req_id]

            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                is_segment_finished = request.is_finished() and request.resumable
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                elif status_before_stop == RequestStatus.WAITING_FOR_CHUNK:
                    # In async chunk mode, request may be in either queue.
                    # Remove from both to avoid stale queue entries.
                    stopped_running_reqs.add(request)
                    stopped_preempted_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None and request.sampling_params.logprobs is not None and logprobs:
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params or stopped:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=request.take_prefill_stats(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                        is_segment_finished=is_segment_finished,
                        new_prompt_len_snapshot=self._new_prompt_len_snapshot.get(req_id, None),
                    )
                )
                if self.chunk_transfer_adapter is not None:
                    self.chunk_transfer_adapter.save_async(pooler_output, request)
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

        # [Main] Handle failed KV load requests
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )
                if self.chunk_transfer_adapter is not None:
                    self.chunk_transfer_adapter.cleanup_receiver(
                        request.request_id,
                    )

        # [Omni] Cleanup state for finished requests
        for req in stopped_running_reqs:
            if req.request_id not in self.waiting_for_transfer_free:
                if req.request_id in self.transfer_triggered_requests:
                    self.transfer_triggered_requests.remove(req.request_id)
                if req.request_id in self.active_kv_transfers:
                    self.active_kv_transfers.remove(req.request_id)
                self.pending_stop_after_extraction.discard(req.request_id)

        # Same for preempted
        for req in stopped_preempted_reqs:
            if req.request_id not in self.waiting_for_transfer_free:
                if req.request_id in self.transfer_triggered_requests:
                    self.transfer_triggered_requests.remove(req.request_id)
                if req.request_id in self.active_kv_transfers:
                    self.active_kv_transfers.remove(req.request_id)
                self.pending_stop_after_extraction.discard(req.request_id)

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(finished_requests=finished_set)
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        # Free blocks that were held for transfer (kv_ready and
        # active_kv_transfers updates already done before the per-request loop).
        if kv_extracted_ids:
            for req_id in kv_extracted_ids:
                try:
                    if req_id in self.waiting_for_transfer_free:
                        req = self.requests.get(req_id)
                        if req:
                            self.kv_cache_manager.free(req)
                            if req_id in self.requests:
                                del self.requests[req_id]
                            if req_id in self.transfer_triggered_requests:
                                self.transfer_triggered_requests.remove(req_id)
                            self.active_kv_transfers.discard(req_id)
                            self.pending_stop_after_extraction.discard(req_id)
                            logger.debug(f"Freed blocks for {req_id} after transfer extraction")
                        self.waiting_for_transfer_free.remove(req_id)
                except Exception:
                    init_logger(__name__).exception("Failed to free blocks for %s after transfer", req_id)

        return engine_core_outputs

    def finish_requests(self, request_ids: Any, finished_status: RequestStatus) -> list[tuple[str, int]]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            Tuple of (req_id, client_index) for requests that were aborted. Will not
            include any that were already finished.
        """

        # The adapter and base scheduler both consume IDs. Snapshot single-pass
        # iterators even with observation disabled, so adapter cleanup cannot
        # exhaust the targets before the base scheduler cancels them.
        if (
            trace_enabled()
            or getattr(self.chunk_transfer_adapter, "first_chunk_observation_enabled", False)
            or (self.chunk_transfer_adapter is not None and isinstance(request_ids, Iterator))
        ):
            request_ids = (
                tuple(self.requests) if request_ids is None
                else (request_ids,) if isinstance(request_ids, str)
                else tuple(request_ids)
            )
        trace_rows = []
        if trace_enabled():
            for req_id in request_ids:
                request = self.requests.get(req_id)
                row = {"request_id": req_id, "external_req_id": getattr(request, "external_req_id", None),
                       "finished_status": str(getattr(finished_status, "name", finished_status)),
                       "was_known": request is not None}
                trace_rows.append(row)
                trace_emit("scheduler_finish_requested", **self._interaction_trace_context, **row)
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.finish_requests(request_ids, finished_status, self.requests)

        result = super().finish_requests(request_ids, finished_status)
        for row in trace_rows:
            trace_emit("scheduler_finish_returned", **self._interaction_trace_context, **row,
                       remains_known=row["request_id"] in self.requests)
        return result

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        """
        Override: Only extend prompt at stage 0, and replace
        the existing session with the next streaming update at other stages.

        Discards the last sampled output token from the prior input chunk at stage 0.
        """
        req_id = session.request_id
        self._new_prompt_len_snapshot[req_id] = len(update.prompt_token_ids)
        if self.vllm_config.model_config.stage_id != 0:
            self._replace_session_with_streaming_update(session, update)

        else:
            super()._update_request_as_session(session, update)

    def _free_request(self, request: Request, delay_free_blocks: bool = False) -> dict[str, Any] | None:
        # TODO(wzliu)! for offline mode, we should not end process until all data is transferred
        """Mark a request as finished and free its resources."""
        assert request.is_finished()
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter._forget_first_chunk(request.request_id)

        self._omni_maybe_capture_kv_evict_shadow(request)
        self._omits_kv_transfer_cache.pop(request.request_id, None)

        # Clear lead governor state for finished/aborted
        # request. Safe to call when governor is disabled (returns None) or
        # when the request was never tracked.
        try:
            self._talker_lead_governor.on_request_finished(request.request_id)
        except Exception:
            init_logger(__name__).exception(
                "[scheduler adaptive] lead_governor.on_request_finished failed for %s",
                request.request_id,
            )

        # 1. Standard cleanup parts from base _free_request
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        self._new_prompt_len_snapshot.pop(request_id, None)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        # 2. Omni Specific: Check if we need to transfer KV
        if self._should_transfer_kv_for_request(request_id):
            already_triggered = request_id in self.transfer_triggered_requests
            is_active = request_id in self.active_kv_transfers

            if already_triggered:
                if is_active:
                    # It triggered but hasn't finished yet. We MUST wait.
                    logger.debug(f"[Omni] Request {request_id} finished but transfer is still ACTIVE. Waiting.")
                    self.waiting_for_transfer_free.add(request_id)
                    kv_xfer_params = None
                    return kv_xfer_params
                elif request_id in self.waiting_for_transfer_free:
                    # Blocks held until KV extraction completes in a future step.
                    return None
                else:
                    logger.debug(
                        f"[Omni] Request {request_id} finished and transfer no longer ACTIVE (extracted/acked). "
                        "Freeing immediately."
                    )
            else:
                self.waiting_for_transfer_free.add(request_id)
                confirmed_computed = self._get_confirmed_num_computed_tokens(request)
                self._mark_request_for_kv_transfer(request_id, confirmed_computed)
                # Return KV transfer metadata so it propagates to RequestOutput
                if request_id in self.requests_needing_kv_transfer:
                    transfer_data = self.requests_needing_kv_transfer[request_id]
                    kv_xfer_params = {
                        "past_key_values": transfer_data["block_ids"],
                        "kv_metadata": {"seq_len": transfer_data["seq_len"], "block_ids": transfer_data["block_ids"]},
                    }
                    # Also update request.additional_information for good measure
                    add_info = getattr(request, "additional_information", None)
                    # If additional_information is an AdditionalInformationPayload-like object,
                    # unpack it into a plain dict.
                    if (
                        add_info is not None
                        and hasattr(add_info, "entries")
                        and isinstance(getattr(add_info, "entries"), dict)
                    ):
                        request.additional_information = deserialize_additional_information(add_info)
                        add_info = request.additional_information
                    if add_info is None:
                        request.additional_information = {}
                        add_info = request.additional_information
                    if isinstance(add_info, dict):
                        add_info.update(kv_xfer_params)

                return kv_xfer_params

        # 3. Standard Freeing
        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        # Helper to match base class structure if not directly available
        # VLLMScheduler has _free_blocks
        self._omni_maybe_pin_prefetch_request(request)
        self._omni_maybe_capture_kv_evict_shadow(request)
        super()._free_blocks(request)

    def _mark_request_for_kv_transfer(self, req_id: str, seq_len: int) -> None:
        """Mark a request as needing KV cache transfer when it finishes."""
        # Avoid duplicate marking (if already pending in queue)
        if req_id in self.requests_needing_kv_transfer:
            return

        if self._should_transfer_kv_for_request(req_id):
            # [Omni] Get block IDs from KVCacheManager
            try:
                block_ids_tuple = self.kv_cache_manager.get_block_ids(req_id)
                if block_ids_tuple and len(block_ids_tuple) > 0:
                    block_ids = block_ids_tuple[0]

                    # [Omni] Fix: Truncate blocks to match seq_len snapshot
                    # We need to know block_size. Usually in self.cache_config.block_size
                    # Note: vllm_config might not be directly available, check scheduler_config or cache_config
                    if hasattr(self, "cache_config") and hasattr(self.cache_config, "block_size"):
                        block_size = self.cache_config.block_size
                    elif hasattr(self, "scheduler_config") and hasattr(
                        self.scheduler_config, "block_size"
                    ):  # Some versions
                        block_size = self.scheduler_config.block_size
                    else:
                        raise ValueError("Block size not found in cache_config or scheduler_config")

                    # ceil(seq_len / block_size)
                    num_blocks = (seq_len + block_size - 1) // block_size
                    if len(block_ids) > num_blocks:
                        logger.debug(
                            f"[Omni] Truncating blocks for {req_id} from {len(block_ids)} "
                            f"to {num_blocks} (seq_len={seq_len})"
                        )
                        block_ids = block_ids[:num_blocks]

                else:
                    block_ids = []
            except Exception as e:
                init_logger(__name__).warning(f"Failed to get block IDs for {req_id}: {e}")
                block_ids = []

            self.requests_needing_kv_transfer[req_id] = {"seq_len": seq_len, "block_ids": block_ids}
            logger.debug(f"Marked request {req_id} for KV cache transfer (len={seq_len}, blocks={len(block_ids)})")

    def _should_transfer_kv_for_request(self, req_id: str) -> bool:
        """Determine if a request should trigger KV cache transfer."""
        need_send = False
        # Try to read from vLLM Config (where YAML config is typically loaded)
        # Check for omni_kv_config attribute
        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if omni_kv_config:
            # omni_kv_config could be an object or a dict
            if isinstance(omni_kv_config, dict):
                need_send = omni_kv_config.get("need_send_cache", False)
            else:
                need_send = getattr(omni_kv_config, "need_send_cache", False)
        if not need_send:
            return False
        request = self.requests.get(req_id)
        if request is not None and self._request_omits_kv_transfer_to_next_stage(request):
            return False
        return True

    def has_requests(self) -> bool:
        """Check if there are any requests to process, including KV transfers."""
        # [Omni] Also check for pending KV transfers
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_requests()

    def has_finished_requests(self) -> bool:
        """Check if there are any finished requests (including those needing KV transfer)."""
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_finished_requests()

    def has_unfinished_requests(self) -> bool:
        """Check if there are any unfinished requests (including those needing KV transfer)."""
        # [Omni] Also check for pending KV transfers to ensure the engine loop continues
        # MUST verify waiting_for_transfer_free and active_kv_transfers
        # Otherwise engine loop might exit before transfer Ack is received.
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_unfinished_requests()

    def get_finished_requests_needing_kv_transfer(self) -> dict[str, dict]:
        """Get and clear the list of requests needing KV cache transfer.
        Returns dict: {req_id: {"seq_len": int, "block_ids": list[int]}}
        """
        requests = self.requests_needing_kv_transfer.copy()

        # Mark these requests as ACTIVE (sent to runner)
        self.active_kv_transfers.update(requests.keys())

        self.requests_needing_kv_transfer.clear()
        return requests


class OmniARAsyncScheduler(OmniARScheduler, AsyncVLLMScheduler):
    """Asynchronous AutoRegressive scheduler."""
