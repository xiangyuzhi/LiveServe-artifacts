from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import (
    EngineCoreEventType,
    EngineCoreOutput,
    EngineCoreOutputs,
)
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.output import OmniCachedRequestData, OmniNewRequestData
from vllm_omni.core.sched.talker_lead_governor import TalkerLeadGovernor
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.profiler.interaction_trace import emit
from vllm_omni.profiler.interaction_trace import enabled as trace_enabled

logger = init_logger(__name__)


class OmniGenerationScheduler(OmniSchedulerMixin, VLLMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_config = self.vllm_config.model_config
        self.chunk_transfer_adapter = None
        if getattr(model_config, "async_chunk", False):
            self.chunk_transfer_adapter = OmniChunkTransferAdapter(self.vllm_config)
        self._init_generation_lead_governor()

    @staticmethod
    def _pooler_scalar_bool(value: Any) -> bool | None:
        """Decode a Ming pooler scalar metadata value into bool|None.

        Ming-flash-omni talker emits per-step scalar control flags
        (``incremental``, ``finished``) as 1-element uint8 tensors. They are
        not drainable multimodal payloads and must not be torch.cat()'d.
        """
        if value is None:
            return None
        try:
            if hasattr(value, "numel") and value.numel() == 1:
                return bool(value.item())
            if isinstance(value, bool):
                return value
        except Exception:
            return None
        return None

    @staticmethod
    def _pooler_output_finished(pooler_output: Any) -> bool | None:
        """Read Ming talker's ``finished`` scalar from pooler output.

        Returns True/False when the talker explicitly signals chunk finality
        for the OMNI_MING_INCREMENTAL streaming path; returns None when the
        flag is absent (non-Ming or non-streaming path).
        """
        if not isinstance(pooler_output, dict) or "finished" not in pooler_output:
            return None
        return OmniGenerationScheduler._pooler_scalar_bool(pooler_output.get("finished"))

    @staticmethod
    def _append_generation_placeholder(request: Request) -> None:
        """Append a placeholder token so an incremental Ming chunk can be
        scheduled in the next step without violating the diffusion fast-path
        invariant ``num_computed_tokens >= len(prompt_token_ids)``.

        A later async chunk can replace the request prompt with a shorter
        token list while the generation scheduler still remembers the previous
        computed position. In that case appending exactly one token is not
        enough: the prompt must be extended past ``num_computed_tokens`` or the
        next schedule step can produce a negative token count.
        """
        try:
            computed = int(getattr(request, "num_computed_tokens", 0) or 0)
        except (TypeError, ValueError):
            computed = 0

        target_len = max(len(request.prompt_token_ids) + 1, computed + 1)
        missing = max(0, target_len - len(request.prompt_token_ids))
        if missing:
            request.prompt_token_ids.extend([0] * missing)
        try:
            all_token_ids = request._all_token_ids  # type: ignore[attr-defined]
            all_missing = max(0, target_len - len(all_token_ids))
            if all_missing:
                all_token_ids.extend([0] * all_missing)
        except Exception:
            pass

    @staticmethod
    def _parse_stage_ids_env(name: str, default: str) -> set[int]:
        import os

        parsed: set[int] = set()
        for part in os.environ.get(name, default).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                parsed.add(int(part))
            except ValueError:
                logger.warning("Invalid %s item=%r, ignoring", name, part)
        return parsed

    @staticmethod
    def _optional_env_int(name: str) -> int | None:
        import os

        raw = os.environ.get(name)
        if raw is None or raw == "":
            return None
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid %s=%r, ignoring", name, raw)
            return None

    def _init_generation_lead_governor(self) -> None:
        """Enable lead-bounded scheduling for generation-stage talkers.

        AR talkers naturally expose ``num_output_tokens``. Generation-stage
        talkers such as Ming emit audio through pooler_output, so this scheduler
        records ``request.omni_lead_units`` in update_from_output and lets the
        same governor consume that counter.
        """
        model_config = self.vllm_config.model_config
        self._bp_governor_role: str | None = None
        self._generation_lead_speaker_rate = max(1, self._omni_env_int("OMNI_TALKER_SPEAKER_RATE", 25))
        self._talker_lead_governor = TalkerLeadGovernor(None)

        stage_id_raw = getattr(model_config, "stage_id", None)
        try:
            stage_id = int(stage_id_raw)
        except (TypeError, ValueError):
            stage_id = None
        talker_stage_ids = self._parse_stage_ids_env("OMNI_TALKER_STAGE_IDS", "1")
        if stage_id not in talker_stage_ids:
            env_val = self._optional_env_int("OMNI_TALKER_LEAD_MAX")
            if env_val not in (None, 0):
                logger.info(
                    "[scheduler adaptive] Generation talker governor disabled on non-talker stage_id=%s "
                    "(talker_stage_ids=%s)",
                    stage_id_raw,
                    sorted(talker_stage_ids),
                )
            return

        lead_max = self._optional_env_int("OMNI_TALKER_LEAD_MAX")
        if lead_max is None:
            omni_kv = getattr(model_config, "omni_kv_config", None)
            if isinstance(omni_kv, dict):
                lead_max = omni_kv.get("talker_lead_max")
            elif omni_kv is not None:
                lead_max = getattr(omni_kv, "talker_lead_max", None)
            try:
                lead_max = int(lead_max) if lead_max is not None else None
            except (TypeError, ValueError):
                lead_max = None

        lead_lo = self._optional_env_int("OMNI_TALKER_LEAD_LO")
        hold_max = max(1, self._omni_env_int("OMNI_TALKER_DEFER_HOLD_MAX", 100))
        min_pending_count = max(1, self._omni_env_int("OMNI_TALKER_MIN_PENDING_COUNT", 1))
        min_backlog_active_steps = max(1, self._omni_env_int("OMNI_TALKER_MIN_BACKLOG_ACTIVE_STEPS", 1))
        min_running_count = max(1, self._omni_env_int("OMNI_TALKER_MIN_RUNNING_COUNT", 1))
        min_pressure_active_calls = max(0, self._omni_env_int("OMNI_TALKER_MIN_PRESSURE_ACTIVE_CALLS", 0))
        pressure_bypass_pending_count = max(
            0,
            self._omni_env_int("OMNI_TALKER_PRESSURE_BYPASS_PENDING_COUNT", 0),
        )
        pressure_bypass_pending_age_s = max(
            0.0,
            self._omni_env_float("OMNI_TALKER_PRESSURE_BYPASS_PENDING_AGE_MS", 0.0) / 1000.0,
        )
        min_pending_age_s = max(
            0.0,
            self._omni_env_float("OMNI_TALKER_MIN_PENDING_AGE_MS", 0.0) / 1000.0,
        )
        min_output_tokens_before_defer = max(
            0,
            self._omni_env_int("OMNI_TALKER_MIN_OUTPUT_TOKENS_BEFORE_DEFER", 0),
        )

        self._bp_governor_role = "talker"
        self._talker_lead_governor = TalkerLeadGovernor(
            L_max=lead_max,
            L_lo=lead_lo,
            hold_max=hold_max,
            speaker_rate=self._generation_lead_speaker_rate,
            min_pending_count=min_pending_count,
            min_backlog_active_steps=min_backlog_active_steps,
            min_running_count=min_running_count,
            min_pressure_active_calls=min_pressure_active_calls,
            pressure_bypass_pending_count=pressure_bypass_pending_count,
            pressure_bypass_pending_age_s=pressure_bypass_pending_age_s,
            resume_on_empty_backlog=self._omni_env_bool("OMNI_BP_RESUME_ON_EMPTY_BACKLOG", False),
            empty_backlog_grace_steps=max(0, self._omni_env_int("OMNI_BP_EMPTY_BACKLOG_GRACE_STEPS", 0)),
            min_output_tokens_before_defer=min_output_tokens_before_defer,
            min_pending_age_s=min_pending_age_s,
        )
        if self._talker_lead_governor.enabled:
            logger.info(
                "[scheduler adaptive] Generation TalkerLeadGovernor enabled: role=talker L_hi=%d L_lo=%d "
                "hold_max=%d rate=%d min_pending=%d min_backlog_steps=%d "
                "min_running=%d min_pressure_calls=%d stage_id=%s",
                self._talker_lead_governor.L_hi,
                self._talker_lead_governor.L_lo,
                self._talker_lead_governor.hold_max,
                self._generation_lead_speaker_rate,
                min_pending_count,
                min_backlog_active_steps,
                min_running_count,
                min_pressure_active_calls,
                stage_id_raw,
            )

    def _bp_pending_age_s(self) -> float | None:
        try:
            now_wall = time.time()
            oldest_age: float | None = None
            for req in self.waiting:
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

    def _defer_lead_bounded_running(self) -> list[Request]:
        governor = getattr(self, "_talker_lead_governor", None)
        if governor is None or not governor.enabled or not self.running:
            return []
        try:
            pending = len(self.waiting) if hasattr(self, "waiting") else None
            deferred = governor.filter_running(
                self.running,
                pending_count=pending,
                pending_age_s=self._bp_pending_age_s(),
            )
            if deferred:
                deferred_ids = {request.request_id for request in deferred}
                self.running = [
                    request for request in self.running if request.request_id not in deferred_ids
                ]
            return deferred
        except Exception:
            logger.exception("[scheduler adaptive] generation lead_governor.filter_running failed; bypassing")
            return []

    def _restore_deferred_running(self, deferred_running: list[Request]) -> None:
        if deferred_running:
            self.running.extend(deferred_running)

    def _pooler_audio_seconds(self, pooler_output: Any) -> float:
        if not isinstance(pooler_output, dict):
            return 0.0
        audio = pooler_output.get("audio")
        if audio is None:
            return 0.0
        try:
            samples = int(audio.shape[-1])
        except Exception:
            try:
                samples = len(audio)
            except Exception:
                return 0.0
        if samples <= 0:
            return 0.0

        sr = pooler_output.get("sr", None)
        try:
            if hasattr(sr, "numel") and sr.numel() == 1:
                sr_value = float(sr.item())
            else:
                sr_value = float(sr)
        except (TypeError, ValueError):
            sr_value = 0.0
        if sr_value <= 0.0:
            return 0.0
        return samples / sr_value

    def _record_pooler_lead_units(self, request: Request, pooler_output: Any) -> None:
        seconds = self._pooler_audio_seconds(pooler_output)
        if seconds <= 0.0:
            return
        rate = max(1, int(getattr(self, "_generation_lead_speaker_rate", 25) or 25))
        units = max(1, int(round(seconds * rate)))
        try:
            current = float(getattr(request, "omni_lead_units", 0.0) or 0.0)
        except (TypeError, ValueError):
            current = 0.0
        setattr(request, "omni_lead_units", current + units)

    def schedule(self) -> SchedulerOutput:
        """Diffusion fast path:
        - Feed all input tokens of the request at once
          (if 0, allocate 1 placeholder token).
        - If the token budget cannot be satisfied at once, fall back to the
          default vLLM scheduling.
        """

        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        scheduled_new_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_running_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        cached_prompt_token_ids: dict[str, list[int]] = {}
        cached_additional_information: dict[str, dict | None] = {}

        # Temporary queue: preserve waiting order, do not disturb non-diffusion requests
        skipped_waiting_requests = create_request_queue(self.policy)
        req_index = 0
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.process_pending_chunks(self.waiting, self.running)
        deferred_running = self._defer_lead_bounded_running()

        # OMNI: Track requests that are already finished (e.g., marked by connector)
        # These should be removed from running and not scheduled
        already_finished_reqs: set[Request] = set()
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            # OMNI: Skip requests that are not in self.requests
            if request.request_id not in self.requests or (
                self.chunk_transfer_adapter is None and request.status == RequestStatus.FINISHED_STOPPED
            ):
                already_finished_reqs.add(request)
                req_index += 1
                continue

            num_computed_tokens = request.num_computed_tokens
            required_tokens = len(request.prompt_token_ids) - num_computed_tokens
            # async_chunk: don't schedule placeholder tokens when no new chunk is available.
            if required_tokens <= 0:
                if (
                    self.chunk_transfer_adapter is not None
                    and request.request_id in self.chunk_transfer_adapter.finished_requests
                ):
                    # Upstream may finish with no terminal tokens; append one pad token so we can emit FINISHED.
                    if len(request.prompt_token_ids) <= num_computed_tokens:
                        self._append_generation_placeholder(request)
                    required_tokens = len(request.prompt_token_ids) - num_computed_tokens
                    if required_tokens <= 0:
                        req_index += 1
                        continue
                else:
                    req_index += 1
                    continue
            num_new_tokens = min(required_tokens, token_budget)
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )
            if new_blocks is None:
                # Allocation failed (e.g., VRAM pressure); stop fast path and
                # fall back to default scheduling
                # Put the current request back to the head of the waiting queue
                # Note: the original queue order is preserved
                break
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            cached_prompt_token_ids[request.request_id] = request.prompt_token_ids
            cached_additional_information[request.request_id] = getattr(request, "additional_information", None)
            token_budget -= num_new_tokens
            scheduled_running_reqs.append(request)
            req_index += 1

        # OMNI: Remove already finished requests from running queue
        if already_finished_reqs:
            self.running = remove_all(self.running, already_finished_reqs)

        # Fast path selection and scheduling (treat all as diffusion requests,
        # independent of pooling_params)
        while (
            self.waiting
            and token_budget > 0
            and len(self.running) < self.max_num_running_reqs
            and self._pause_state == PauseState.UNPAUSED
        ):
            request = self.waiting.peek_request()
            # OMNI: Skip requests that are not in self.requests
            if request.request_id not in self.requests or (
                self.chunk_transfer_adapter is None and request.status == RequestStatus.FINISHED_STOPPED
            ):
                # Pop the finished request from waiting queue and don't schedule it
                self.waiting.pop_request()
                continue

            # async_chunk: wait for the first upstream chunk (don't start with placeholders).
            if self.chunk_transfer_adapter is not None and len(request.prompt_token_ids) == 0:
                if request.request_id in self.chunk_transfer_adapter.finished_requests:
                    request.prompt_token_ids.append(0)
                    try:
                        request._all_token_ids.append(0)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                else:
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

            # async_chunk requests can temporarily move from running back to
            # waiting (for example, capacity limiting after chunk polling). If a
            # request has already advanced, resuming it as a brand-new request
            # makes vLLM schedule the whole synthetic prompt from position
            # num_computed_tokens and can index past max_model_len. Treat it as
            # cached/resumed and schedule only the remaining placeholder/chunk.
            try:
                num_computed_tokens = int(getattr(request, "num_computed_tokens", 0) or 0)
            except (TypeError, ValueError):
                num_computed_tokens = 0
            if self.chunk_transfer_adapter is not None and num_computed_tokens > 0:
                required_tokens = len(request.prompt_token_ids) - num_computed_tokens
                if required_tokens <= 0:
                    if request.request_id in self.chunk_transfer_adapter.finished_requests:
                        self._append_generation_placeholder(request)
                        required_tokens = len(request.prompt_token_ids) - num_computed_tokens
                    if required_tokens <= 0:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                num_new_tokens = min(required_tokens, token_budget)
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )
                if new_blocks is None:
                    break

                request = self.waiting.pop_request()
                request.status = RequestStatus.RUNNING
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)

                # Resumed requests use ``new_block_ids`` as a full replacement
                # for the worker-side block table. ``allocate_slots`` only
                # returns newly allocated blocks, which can be empty when the
                # next token fits in an existing block, so use the current full
                # KV block list for the resumed payload.
                req_to_new_blocks[request.request_id] = self.kv_cache_manager.get_blocks(
                    request.request_id
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens
                cached_prompt_token_ids[request.request_id] = request.prompt_token_ids
                cached_additional_information[request.request_id] = getattr(request, "additional_information", None)
                token_budget -= num_new_tokens
                scheduled_resumed_reqs.append(request)
                continue

            # Uniformly treat as diffusion. A feature flag can be added later
            # via config or request tag.

            # Allocate all input tokens for the request in one shot
            # (allocate 1 placeholder if zero)
            required_tokens = max(len(request.prompt_token_ids), 1)
            num_new_tokens = min(required_tokens, token_budget)
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )
            if new_blocks is None:
                # Allocation failed (e.g., VRAM pressure); stop fast path and
                # fall back to default scheduling
                # Put the current request back to the head of the waiting queue
                # Note: the original queue order is preserved
                break

            # Officially schedule this request
            request = self.waiting.pop_request()
            self.running.append(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)

            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            scheduled_new_reqs.append(request)

        # Return skipped waiting requests
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # If fast path scheduled none, fall back to the original scheduling
        if not num_scheduled_tokens:
            if self.chunk_transfer_adapter:
                # Don't fall back: base scheduler doesn't handle async_chunk
                # requests with empty prompt_token_ids.
                self.chunk_transfer_adapter.restore_queues(self.waiting, self.running)
            else:
                self._restore_deferred_running(deferred_running)
                res = super().schedule()
                return res

        # Compute common prefix blocks (aligned with v1)
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(any_request.request_id)

        # Assemble SchedulerOutput (align with v0.14.0)
        if self.use_v2_model_runner:
            # No resumed reqs in fast path; pass prefill_token_ids for new reqs.
            new_reqs_data = [
                OmniNewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    getattr(req, "_all_token_ids", None),
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                OmniNewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids())
                for req in scheduled_new_reqs
            ]
        # No running/resumed reqs scheduled in our fast path
        cached_reqs_data = self._make_cached_request_data(
            running_reqs=scheduled_running_reqs,
            resumed_reqs=scheduled_resumed_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            spec_decode_tokens=scheduled_spec_decode_tokens,
            req_to_new_blocks=req_to_new_blocks,
        )
        cached_new_block_ids = cached_reqs_data.new_block_ids
        if cached_reqs_data.resumed_req_ids:
            cached_new_block_ids = list(cached_new_block_ids)
            for idx, req_id in enumerate(cached_reqs_data.req_ids):
                if req_id not in cached_reqs_data.resumed_req_ids:
                    continue
                if cached_new_block_ids[idx] is not None:
                    continue
                blocks = req_to_new_blocks.get(req_id)
                if blocks is None:
                    blocks = self.kv_cache_manager.get_blocks(req_id)
                block_ids = blocks.get_block_ids(allow_none=False)
                cached_new_block_ids[idx] = block_ids if block_ids is not None else tuple()

        cached_reqs_data = OmniCachedRequestData(
            req_ids=cached_reqs_data.req_ids,
            resumed_req_ids=cached_reqs_data.resumed_req_ids,
            new_token_ids=cached_reqs_data.new_token_ids,
            all_token_ids=cached_reqs_data.all_token_ids,
            new_block_ids=cached_new_block_ids,
            num_computed_tokens=cached_reqs_data.num_computed_tokens,
            num_output_tokens=cached_reqs_data.num_output_tokens,
            prompt_token_ids=cached_prompt_token_ids,
            additional_information=cached_additional_information,
        )

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())

        # Record the request ids scheduled in this step (v0.14.0 behavior).
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None) if self.needs_kv_cache_zeroing else None
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            preempted_req_ids=set(),
            new_block_ids_to_zero=new_block_ids_to_zero,
        )

        # KVTransfer: package metadata
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta
        # EC Connector: package metadata
        if self.ec_connector is not None:
            ec_meta = self.ec_connector.build_connector_meta(scheduler_output)
            scheduler_output.ec_connector_metadata = ec_meta

        # Update internal state (advance num_computed_tokens, free encoder inputs,
        # etc.)
        self._update_after_schedule(scheduler_output)

        try:
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
                self.chunk_transfer_adapter.postprocess_scheduler_output(scheduler_output)

        except Exception:
            # If anything goes wrong, leave the original output unchanged
            logger.exception("Failed to wrap scheduled_new_reqs with OmniNewRequestData")
        finally:
            # Ensure chunk-waiting requests are restored even on error,
            # otherwise they are permanently orphaned in the adapter's
            # internal deques and never scheduled again.
            if self.chunk_transfer_adapter:
                self.chunk_transfer_adapter.restore_queues(self.waiting, self.running)
            self._restore_deferred_running(deferred_running)

        return scheduler_output

    def finish_requests(self, request_ids, finished_status: RequestStatus) -> list[tuple[str, int]]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            Tuple of (req_id, client_index) for requests that were aborted. Will not
            include any that were already finished.
        """

        tracing = trace_enabled()
        if tracing:
            # Consume one-shot iterables only once; both the adapter and engine
            # must see the same finish set. A finish ACK is not a GPU fence.
            request_ids = (
                list(self.requests)
                if request_ids is None
                else [request_ids]
                if isinstance(request_ids, str)
                else list(request_ids)
            )
            before = {rid: self.requests.get(rid) for rid in request_ids}
            stage_id = self.vllm_config.model_config.stage_id
            status = getattr(finished_status, "name", str(finished_status))
            for rid, request in before.items():
                emit(
                    "scheduler_finish_requested",
                    request_id=rid,
                    external_req_id=getattr(request, "external_req_id", None),
                    stage_id=stage_id,
                    finished_status=status,
                    was_known=request is not None,
                )

        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.finish_requests(request_ids, finished_status, self.requests)

        result = super().finish_requests(request_ids, finished_status)
        if tracing:
            for rid, request in before.items():
                emit(
                    "scheduler_finish_returned",
                    request_id=rid,
                    external_req_id=getattr(request, "external_req_id", None),
                    stage_id=stage_id,
                    finished_status=status,
                    remains_known=rid in self.requests,
                )
        return result

    """
    Scheduler for the diffusion model.
    This scheduler is modified to stop the request immediately for the diffusion model.
    This is because the diffusion model can generate the final image/audio in one step.
    Note: This is just a minimal modification to the original scheduler,
    and there should be some further efforts to optimize the scheduler.
    The original scheduler is still used for the AR model.
    """

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: OmniModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Update the scheduler state based on the model runner output.

        This method is modified to stop the request immediately for the diffusion model.
        """
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        if self.chunk_transfer_adapter is not None:
            self.chunk_transfer_adapter.trace_completed_chunks(num_scheduled_tokens)
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
        # Merge connector-side stats (align with v0.14.0)
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and getattr(kv_connector_output, "invalid_block_ids", None):
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

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
                # Request may already be finished (e.g., aborted during
                # execution / pipeline parallelism / async scheduling).
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
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            status_before_stop = request.status
            finish_reason = None
            routed_experts = None
            self._record_pooler_lead_units(request, pooler_output)

            # Ming OMNI_MING_INCREMENTAL: the talker signals chunk-by-chunk
            # progress via pooler_output["finished"] (uint8 scalar). When the
            # talker reports ``finished=False`` the request still has more
            # chunks to emit, so we must keep it scheduled by appending a
            # placeholder token instead of falling through to FINISHED_STOPPED.
            pooler_finished = self._pooler_output_finished(pooler_output)

            # Diffusion request: completes in one step unless the model reports
            # that it emitted an incremental chunk and still has more work.
            if pooler_finished is False:
                if request.num_computed_tokens >= len(request.prompt_token_ids):
                    self._append_generation_placeholder(request)
            elif (
                request.status == RequestStatus.FINISHED_STOPPED
                or pooler_finished is True
                or (self.chunk_transfer_adapter is None and request.num_computed_tokens >= request.num_prompt_tokens)
                or (
                    self.chunk_transfer_adapter is not None
                    and request.request_id in self.chunk_transfer_adapter.finished_requests
                    and request.num_computed_tokens >= len(request.prompt_token_ids)
                )
            ):
                request.status = RequestStatus.FINISHED_STOPPED
                # Optional: set a stop_reason for front-end clarity
                # (does not affect protocol)
                request.stop_reason = request.stop_reason  # or "generation_done"
                stopped = True

            routed_experts_dict = getattr(model_runner_output, "routed_experts_dict", None)
            if routed_experts_dict is not None and req_id in routed_experts_dict:
                routed_experts = routed_experts_dict[req_id]

            if stopped:
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)
                    if self.chunk_transfer_adapter is not None:
                        self.chunk_transfer_adapter.cleanup(
                            request.request_id,
                            getattr(request, "external_req_id", None),
                        )
                if status_before_stop == RequestStatus.WAITING_FOR_CHUNK:
                    stopped_running_reqs.add(request)
                    stopped_preempted_reqs.add(request)
                else:
                    stopped_running_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None and request.sampling_params.logprobs is not None and logprobs:
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                # NOTE: structured_output_request should not be None if
                # use_structured_output, we have check above, so safe to ignore
                # type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]  # noqa: E501
                    req_id, new_token_ids
                )

            # spec_token_ids comes from the model runner output
            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params or stopped:
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
                    )
                )
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

        # Handle failed KV load requests
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
                    self.chunk_transfer_adapter.cleanup(
                        request.request_id,
                        getattr(request, "external_req_id", None),
                    )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # Collect and publish KV cache events (align with v0.14.0)
        events = self.kv_cache_manager.take_events()
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
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

        return engine_core_outputs

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        """
        Override: Just replace the existing session with the next streaming update.

        Do not expend prompt id using update.
        """
        self._replace_session_with_streaming_update(session, update)
