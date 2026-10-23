# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import json
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any

import torch
from vllm.v1.request import Request, RequestStatus

from vllm_omni.data_entry_keys import unflatten_payload
from vllm_omni.profiler.interaction_trace import emit as trace_emit
from vllm_omni.profiler.interaction_trace import enabled as trace_enabled

from ..factory import OmniConnectorFactory
from ..utils.config import ConnectorSpec
from ..utils.logging import get_connector_logger
from .base import OmniTransferAdapterBase

logger = get_connector_logger(__name__)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _bool_value(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _float_value(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_or_cfg(env_name: str, cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    value = os.environ.get(env_name)
    if value is not None and str(value).strip() != "":
        return value
    return cfg.get(key, default)


class OmniChunkTransferAdapter(OmniTransferAdapterBase):
    """Chunk-level transfer adapter for Omni connector pipelines.

    This class coordinates per-request chunk exchange between adjacent stages,
    and implements asynchronous get/put of chunks via background threads.
    It tracks per-request chunk indices for put/get, and accumulates
    payloads across chunks (concatenating tensors/lists in AR mode). It also
    caches prompt token ids and additional information for scheduler use.

    Scheduler integration is handled via WAITING_FOR_CHUNK transitions:
    requests are moved to waiting for chunk deque while polling, then restored
    to waiting/running queues once a chunk arrives. The requests will finish
    loading chunk util detecting the payload "finished" flag.

    The base class owns background recv/save loops; load/save only enqueue
    work and return immediately.
    """

    def __init__(self, vllm_config: Any):
        model_config = vllm_config.model_config
        self.scheduler_max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.connector = self.create_connector(model_config)
        super().__init__(model_config)
        connector_cfg = getattr(self.connector, "config", {}) or {}
        connector_cfg = connector_cfg.get("extra", connector_cfg) if isinstance(connector_cfg, dict) else {}
        self.model_mode = getattr(model_config, "worker_type", None) or "ar"
        self.stage2_trace_path = str(
            os.environ.get("OMNI_CODEC_STAGE2_TRACE_PATH", "")
            or connector_cfg.get("codec_stage2_trace_path", "")
            or ""
        ).strip()
        self.stage2_priority_policy = str(
            os.environ.get("OMNI_CODE2WAV_PRIORITY_POLICY", "")
            or connector_cfg.get("codec_stage2_priority_policy", "none")
            or "none"
        ).strip().lower()
        self.stage2_priority_first_chunks = max(
            0,
            _int_value(
                _env_or_cfg(
                    "OMNI_CODE2WAV_PRIORITY_FIRST_CHUNKS",
                    connector_cfg,
                    "codec_stage2_priority_first_chunks",
                    1,
                ),
                1,
            ),
        )
        first_window_policies = {"first", "first_chunk", "first_chunks"}
        hrrn_policies = {"hrrn", "first_hrrn", "balanced_hrrn"}
        self.stage2_priority_enabled = (
            self.stage2_priority_policy in (first_window_policies | hrrn_policies)
            and self.stage2_priority_first_chunks > 0
            and self.model_mode != "ar"
        )
        self.stage2_priority_uses_hrrn = self.stage2_priority_policy in hrrn_policies
        self.stage2_priority_first_bias_ms = max(
            0.0,
            _float_value(
                _env_or_cfg(
                    "OMNI_CODE2WAV_PRIORITY_FIRST_BIAS_MS",
                    connector_cfg,
                    "codec_stage2_priority_first_bias_ms",
                    None,
                ),
                250.0,
            ),
        )
        self.stage2_priority_service_floor_ms = max(
            1.0,
            _float_value(
                _env_or_cfg(
                    "OMNI_CODE2WAV_PRIORITY_SERVICE_FLOOR_MS",
                    connector_cfg,
                    "codec_stage2_priority_service_floor_ms",
                    None,
                ),
                20.0,
            ),
        )
        self.stage2_priority_demote_post_first = _bool_value(
            _env_or_cfg(
                "OMNI_CODE2WAV_PRIORITY_DEMOTE_POST_FIRST",
                connector_cfg,
                "codec_stage2_priority_demote_post_first",
                None,
            ),
            default=self.stage2_priority_policy in first_window_policies,
        )
        self.stage2_trace_enabled = bool(self.stage2_trace_path) and self.model_mode != "ar"
        # State specific to Chunk management
        self.custom_process_next_stage_input_func = None
        custom_process_next_stage_input_func = getattr(model_config, "custom_process_next_stage_input_func", None)
        if custom_process_next_stage_input_func:
            module_path, func_name = custom_process_next_stage_input_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_process_next_stage_input_func = getattr(module, func_name)
        # Observe Qwen's actual codec payload publication, never token counts.
        # Scheduler and save thread share only this small locked metadata map.
        self.first_chunk_observation_enabled = (
            _float_value(os.environ.get("OMNI_BP_FIRST_CHUNK_PROTECT_MS"), 0.0) > 0
            and self.model_mode == "ar"
            and custom_process_next_stage_input_func ==
            "vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav_async_chunk"
        )
        self._first_chunk_lock = threading.Lock()
        self._first_chunk_generation_seq = 0
        self._first_chunk_state: dict[str, dict[str, int | None]] = {}
        # mapping for request id and chunk id
        self.put_req_chunk: dict[str, int] = defaultdict(int)
        self.get_req_chunk: dict[str, int] = defaultdict(int)
        self.finished_requests: set[str] = set()
        self.request_payload = {}
        self.code_prompt_token_ids: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.request_ids_mapping: dict[str, str] = {}

        self.waiting_for_chunk_waiting_requests: deque[Any] = deque()
        self.waiting_for_chunk_running_requests: deque[Any] = deque()
        self.requests_with_ready_chunks = set()
        self.requests_origin_status = {}
        self.ready_chunk_info: dict[str, dict[str, Any]] = {}
        self.inflight_chunk_info: dict[str, dict[str, Any]] = {}

    def _observe_first_chunk_ready(self, request: Request) -> None:
        if not getattr(self, "first_chunk_observation_enabled", False):
            return
        with self._first_chunk_lock:
            if request.request_id not in self._first_chunk_state:
                self._first_chunk_generation_seq += 1
                self._first_chunk_state[request.request_id] = {
                    "generation": self._first_chunk_generation_seq,
                    "first_ready_ns": time.monotonic_ns(), "published_ns": None,
                }

    def _first_chunk_generation(self, request_id: str) -> int:
        with self._first_chunk_lock:
            return (self._first_chunk_state.get(request_id) or {}).get("generation", 0)

    def first_chunk_pending_ready(self, request_ids: list[str], now_ns: int, budget_ns: int) -> set[str]:
        """Snapshot only ready/running IDs supplied by the scheduler.

        A later input chunk cannot restart the deadline. Missing observations
        fail open, and missing input chunks are excluded by the caller.
        """
        if not getattr(self, "first_chunk_observation_enabled", False):
            return set()
        with self._first_chunk_lock:
            return {req_id for req_id in request_ids
                    if (state := self._first_chunk_state.get(req_id)) is not None
                    and state["published_ns"] is None
                    and 0 <= now_ns - state["first_ready_ns"] < budget_ns}

    def _forget_first_chunk(self, request_id: str, generation: int | None = None) -> None:
        if not getattr(self, "first_chunk_observation_enabled", False):
            return
        with self._first_chunk_lock:
            state = self._first_chunk_state.get(request_id)
            if state is not None and (generation is None or state["generation"] == generation):
                self._first_chunk_state.pop(request_id, None)

    def _observe_first_chunk_published(self, task: dict, payload: dict, chunk_id: int) -> None:
        """Called only after put succeeds, with no tensor copy or device sync."""
        if not getattr(self, "first_chunk_observation_enabled", False):
            return
        codes = payload.get("codes", {}).get("audio")
        # The supported Qwen processor returns flattened Python codec IDs.
        # An empty terminal marker is not a decodable codec chunk.
        if not isinstance(codes, (list, tuple)) or not codes:
            return
        if not isinstance(payload.get("meta", {}).get("codec_chunk_frames"), int):
            return
        if payload["meta"]["codec_chunk_frames"] <= 0:
            return
        request = task["request"]
        with self._first_chunk_lock:
            state = self._first_chunk_state.get(request.request_id)
            if (state is None or state["generation"] != task.get("_first_chunk_generation")
                    or state["published_ns"] is not None):
                return
            state["published_ns"] = time.monotonic_ns()
            observation = dict(state)
        if trace_enabled():
            trace_emit("connector_first_chunk_published", **self._interaction_trace_context,
                       request_id=request.request_id, external_req_id=request.external_req_id,
                       chunk_id=chunk_id, **observation)

    def _trace_stage2_event(self, event: str, row: dict[str, Any]) -> None:
        if not self.stage2_trace_enabled:
            return
        try:
            trace_row = {
                "event": event,
                "ts": time.time(),
                "stage_id": self.connector.stage_id,
                **row,
            }
            parent = os.path.dirname(self.stage2_trace_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.stage2_trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_jsonable(trace_row), ensure_ascii=False, sort_keys=True) + "\n")
        except Exception:
            logger.debug("Failed to write Code2Wav stage2 trace", exc_info=True)

    @staticmethod
    def _meta_from_request(request: Request) -> dict[str, Any]:
        info = getattr(request, "additional_information", None)
        if not isinstance(info, dict):
            return {}
        meta = info.get("meta")
        return meta if isinstance(meta, dict) else {}

    def _chunk_id_for_request(self, request: Request) -> int:
        info = self.ready_chunk_info.get(request.request_id) or {}
        meta = self._meta_from_request(request)
        raw_chunk_id = info.get("chunk_id", meta.get("recv_chunk_id", 1 << 30))
        try:
            return int(raw_chunk_id)
        except (TypeError, ValueError):
            return 1 << 30

    def _is_first_window_request(self, request: Request) -> bool:
        return (
            self.stage2_priority_enabled
            and self._chunk_id_for_request(request) < self.stage2_priority_first_chunks
        )

    def _chunk_service_estimate_ms(self, request: Request) -> float:
        info = self.ready_chunk_info.get(request.request_id) or {}
        meta = self._meta_from_request(request)
        payload_meta = info.get("meta") if isinstance(info.get("meta"), dict) else {}
        candidates = (
            payload_meta.get("codec_window_frames"),
            payload_meta.get("codec_chunk_frames"),
            meta.get("codec_window_frames"),
            meta.get("codec_chunk_frames"),
            info.get("size_bytes"),
        )
        for value in candidates:
            estimate = _float_value(value, 0.0)
            if estimate > 0:
                if value == info.get("size_bytes"):
                    estimate = estimate / 64.0
                return max(self.stage2_priority_service_floor_ms, estimate)
        return self.stage2_priority_service_floor_ms

    def _chunk_priority_key(self, request: Request) -> tuple[Any, ...]:
        info = self.ready_chunk_info.get(request.request_id) or {}
        chunk_id = self._chunk_id_for_request(request)
        is_first_window = self._is_first_window_request(request)
        arrival = getattr(request, "arrival_time", 0.0) or 0.0
        try:
            arrival_f = float(arrival)
        except (TypeError, ValueError):
            arrival_f = 0.0
        if self.stage2_priority_uses_hrrn:
            now = time.time()
            ready_ts = info.get("ready_ts") or info.get("recv_ts") or arrival_f
            try:
                wait_ms = max(0.0, (now - float(ready_ts)) * 1000.0)
            except (TypeError, ValueError):
                wait_ms = 0.0
            if is_first_window:
                wait_ms += self.stage2_priority_first_bias_ms
            service_est_ms = self._chunk_service_estimate_ms(request)
            response_ratio = 1.0 + wait_ms / service_est_ms
            return (-response_ratio, 0 if is_first_window else 1, arrival_f, chunk_id, request.request_id)
        return (0 if is_first_window else 1, chunk_id, arrival_f, request.request_id)

    def _prioritize_ready_chunks(self, waiting_queue: Any, running_queue: list[Request]) -> None:
        if not self.stage2_priority_enabled or not self.requests_with_ready_chunks:
            return

        ready_waiting_first: list[Request] = []
        if self.stage2_priority_demote_post_first:
            try:
                ready_waiting_first = [
                    request
                    for request in list(waiting_queue)
                    if request.request_id in self.requests_with_ready_chunks
                    and self._is_first_window_request(request)
                ]
            except Exception:
                logger.debug("Failed to inspect waiting code2wav chunks", exc_info=True)

        if self.stage2_priority_demote_post_first and ready_waiting_first:
            demoted: list[Request] = []
            kept: list[Request] = []
            for request in running_queue:
                if (
                    request.request_id in self.requests_with_ready_chunks
                    and not self._is_first_window_request(request)
                ):
                    demoted.append(request)
                else:
                    kept.append(request)
            if demoted:
                running_queue[:] = kept
                for request in demoted:
                    request.status = RequestStatus.PREEMPTED
                    waiting_queue.add_request(request)
                    self._trace_stage2_event(
                        "demote_post_first",
                        {
                            **(self.ready_chunk_info.get(request.request_id) or {}),
                            "priority_key": self._chunk_priority_key(request),
                            "waiting_first_ready": len(ready_waiting_first),
                        },
                    )

        try:
            running_queue.sort(key=self._chunk_priority_key)
        except Exception:
            logger.debug("Failed to prioritize running code2wav chunks", exc_info=True)

        try:
            ready_waiting = [
                request
                for request in list(waiting_queue)
                if request.request_id in self.requests_with_ready_chunks
            ]
            if not ready_waiting:
                return
            waiting_queue.remove_requests(ready_waiting)
            for request in reversed(sorted(ready_waiting, key=self._chunk_priority_key)):
                waiting_queue.prepend_request(request)
        except Exception:
            logger.debug("Failed to prioritize waiting code2wav chunks", exc_info=True)

    @classmethod
    def create_connector(cls, model_config: Any):
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is None:
            connector_config = {}
        elif not isinstance(connector_config, dict):
            connector_config = {
                "name": getattr(connector_config, "name", None),
                "extra": getattr(connector_config, "extra", {}),
            }

        connector_specs = ConnectorSpec(
            name=connector_config.get("name", "SharedMemoryConnector"),
            extra=connector_config.get("extra", {}),
        )
        return OmniConnectorFactory.create_connector(connector_specs)

    def load_async(self, request: Request):
        """Register a request for asynchronous chunk retrieval.

        This method does not read from the connector directly. It records
        request metadata and enqueues the request id for the background
        receive loop to poll.

        Stage-0 has no upstream producer, so this call is a no-op there.

        Args:
            request: The request object needing data.
        """
        stage_id = self.connector.stage_id

        if stage_id == 0:
            return
        if not hasattr(request, "additional_information"):
            request.additional_information = None
        self._cancelled_load_reqs.discard(request.request_id)
        if trace_enabled():
            trace_emit(
                "connector_load_enqueue", **self._interaction_trace_context,
                request_id=request.request_id, external_req_id=request.external_req_id,
                chunk_id=self.get_req_chunk[request.request_id],
                queue_depth=len(self._pending_load_reqs) + 1,
            )
        self._pending_load_reqs.append(request)
        with self._recv_cond:
            self._recv_cond.notify()

    def save_async(
        self,
        pooling_output: torch.Tensor | None = None,
        request: Request | None = None,
    ):
        """Build and enqueue one chunk for asynchronous sending.

        Payload extraction happens in ``_send_single_request`` on the
        background save_loop thread.

        Args:
            pooling_output: Partial pooling output dictionary
            request: Request object
        """
        # Only processors declaring a snapshot hook opt in. In particular the
        # Qwen Thinker bridge must not read a later mutation of the scheduler's
        # request when the background save thread eventually consumes this job.
        snapshot = getattr(self.custom_process_next_stage_input_func, "snapshot_request", None)
        queued_request = snapshot(request) if snapshot is not None else request
        task = {
            "pooling_output": pooling_output,
            "request": queued_request,
            "is_finished": queued_request.bridge_finished if snapshot is not None else request.is_finished(),
        }
        if getattr(self, "first_chunk_observation_enabled", False):
            task["_first_chunk_generation"] = self._first_chunk_generation(request.request_id)
        if trace_enabled():
            self._interaction_job_seq += 1
            task["_interaction_job_id"] = f"connector-{self._interaction_job_seq}"
            task["_interaction_enqueued_ns"] = time.monotonic_ns()
            trace_emit(
                "connector_enqueue", **self._interaction_trace_context,
                request_id=request.request_id, external_req_id=request.external_req_id,
                job_id=task["_interaction_job_id"], queue_depth=len(self._pending_save_reqs) + 1,
                is_finished=task["is_finished"],
            )
        self._pending_save_reqs.append(task)
        with self._save_cond:
            self._save_cond.notify()

    def _poll_single_request(self, request: Request):
        stage_id = self.connector.stage_id
        target_stage_id = stage_id - 1
        req_id = request.request_id
        chunk_id = self.get_req_chunk[req_id]
        external_req_id = self.request_ids_mapping.get(req_id, req_id)
        connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"
        get_started_ns = time.monotonic_ns() if trace_enabled() else 0

        # Use timeout=0 for non-blocking poll
        try:
            result = self.connector.get(
                str(target_stage_id),
                str(stage_id),
                connector_get_key,
            )
        except Exception as e:
            logger.error(f"SharedMemoryConnector get failed for req {connector_get_key}: {e}")
            return False

        if result is None:
            return False
        payload_data, size = result
        if trace_enabled():
            trace_emit(
                "connector_recv", **self._interaction_trace_context,
                request_id=req_id, external_req_id=external_req_id, chunk_id=chunk_id,
                connector_key=connector_get_key, size_bytes=size,
                get_elapsed_ns=time.monotonic_ns() - get_started_ns,
            )

        if payload_data:
            # Update connector state
            self.get_req_chunk[req_id] += 1

            meta = payload_data.get("meta", {})
            recv_ts = time.time()
            chunk_info = {
                "request_id": req_id,
                "external_req_id": external_req_id,
                "chunk_id": chunk_id,
                "connector_get_key": connector_get_key,
                "size_bytes": size,
                "recv_ts": recv_ts,
                "meta": dict(meta) if isinstance(meta, dict) else {},
            }
            if self.model_mode == "ar":
                merged_payload = self._update_request_payload(external_req_id, payload_data)
                request.additional_information = merged_payload
                if meta.get("finished"):
                    self.finished_requests.add(req_id)
            else:
                if meta.get("finished"):
                    self.finished_requests.add(req_id)

                new_ids = payload_data.get("codes", {}).get("audio", [])
                request.prompt_token_ids = new_ids
                prev_info = getattr(request, "additional_information", None)
                info = dict(prev_info) if isinstance(prev_info, dict) else {}
                for key, value in payload_data.items():
                    if key == "codes":
                        continue
                    if isinstance(value, dict):
                        existing_sub = info.get(key)
                        merged_sub = dict(existing_sub) if isinstance(existing_sub, dict) else {}
                        for sk, sv in value.items():
                            if key == "meta" and sk == "finished":
                                continue
                            merged_sub[sk] = sv
                        info[key] = merged_sub
                        continue
                    info[key] = value
                meta_info = info.setdefault("meta", {})
                if isinstance(meta_info, dict):
                    meta_info["recv_chunk_id"] = chunk_id
                    meta_info["recv_ts"] = recv_ts
                    meta_info["connector_get_key"] = connector_get_key
                request.additional_information = info
                request.num_computed_tokens = 0

                # Empty chunk with more data expected: keep polling.
                if not new_ids and not meta.get("finished"):
                    return True

            # Mark as finished for consumption
            self._finished_load_reqs.add(req_id)
            self.ready_chunk_info[req_id] = chunk_info
            self._trace_stage2_event("recv", chunk_info)
            logger.debug(f"[Stage-{stage_id}] Received one chunk for key {connector_get_key}")
            return True

        return False

    def _update_request_payload(self, req_id: str, payload_data: dict[str, Any]) -> dict[str, Any]:
        """Update the stored payload for *req_id* with the latest chunk."""
        if req_id not in self.request_payload:
            self.request_payload[req_id] = payload_data
            return payload_data
        origin = self.request_payload[req_id]
        raw_ok = payload_data.get("meta", {}).pop("override_keys", [])
        override_keys = {tuple(k) if isinstance(k, list) else k for k in raw_ok}

        for type_key, new_val in payload_data.items():
            if not isinstance(new_val, dict):
                continue
            origin_sub = origin.get(type_key)
            if not isinstance(origin_sub, dict):
                continue
            for qual, value in new_val.items():
                if type_key == "meta" and qual == "finished":
                    continue
                if (type_key, qual) in override_keys:
                    continue
                if isinstance(value, torch.Tensor) and qual in origin_sub:
                    new_val[qual] = torch.cat([origin_sub[qual], value], dim=0)
                elif isinstance(value, list) and qual in origin_sub:
                    new_val[qual] = origin_sub[qual] + value

        self.request_payload[req_id] = payload_data
        return payload_data

    def _send_single_request(self, task: dict):
        raw_po = task["pooling_output"]
        pooling_output = unflatten_payload(raw_po) if isinstance(raw_po, dict) else raw_po
        request = task["request"]
        is_finished = task["is_finished"]
        stage_id = self.connector.stage_id
        next_stage_id = stage_id + 1
        external_req_id = request.external_req_id
        chunk_id = self.put_req_chunk[external_req_id]
        connector_put_key = f"{external_req_id}_{stage_id}_{chunk_id}"
        extraction_started_ns = time.monotonic_ns() if trace_enabled() else 0
        # Process payload in save_loop thread
        payload_data = None
        if self.custom_process_next_stage_input_func:
            try:
                payload_data = self.custom_process_next_stage_input_func(
                    transfer_manager=self,
                    pooling_output=pooling_output,
                    request=request,
                    is_finished=is_finished,
                )

            except Exception as e:
                logger.error(f"Failed to use custom_process_input_func for payload extraction: {e}")

        if not payload_data:
            if trace_enabled():
                trace_emit(
                    "connector_payload_empty", **self._interaction_trace_context,
                    request_id=request.request_id, external_req_id=external_req_id,
                    job_id=task.get("_interaction_job_id"),
                    extraction_elapsed_ns=time.monotonic_ns() - extraction_started_ns,
                )
            return

        if trace_enabled():
            trace_emit(
                "connector_send_begin", **self._interaction_trace_context,
                request_id=request.request_id, external_req_id=external_req_id,
                chunk_id=chunk_id, job_id=task.get("_interaction_job_id"),
                connector_key=connector_put_key,
                extraction_elapsed_ns=time.monotonic_ns() - extraction_started_ns,
            )
        put_started_ns = time.monotonic_ns() if trace_enabled() else 0
        success, size, metadata = self.connector.put(
            from_stage=str(stage_id),
            to_stage=str(next_stage_id),
            put_key=connector_put_key,
            data=payload_data,
        )
        if trace_enabled():
            trace_emit(
                "connector_send_end", **self._interaction_trace_context,
                request_id=request.request_id, external_req_id=external_req_id,
                chunk_id=chunk_id, job_id=task.get("_interaction_job_id"),
                connector_key=connector_put_key, size_bytes=size, success=success,
                put_elapsed_ns=time.monotonic_ns() - put_started_ns,
            )

        if success:
            self._observe_first_chunk_published(task, payload_data, chunk_id)
            self.put_req_chunk[external_req_id] += 1
            logger.debug(f"[Stage-{stage_id}] Sent {connector_put_key}")
            finished_flag = payload_data.get("meta", {}).get("finished", payload_data.get("finished"))
            is_payload_finished = False
            if isinstance(finished_flag, torch.Tensor):
                is_payload_finished = finished_flag.numel() == 1 and bool(finished_flag.item())
            elif finished_flag is not None:
                is_payload_finished = bool(finished_flag)

            # Reclaim per-request async state only after the terminal payload
            # has been sent successfully. This avoids cleanup->save races.
            if is_payload_finished:
                self.cleanup(request.request_id, external_req_id,
                             first_chunk_generation=task.get("_first_chunk_generation", 0))

        if is_finished:
            self.code_prompt_token_ids.pop(external_req_id, None)
            cached_ic = getattr(self, "_cached_ic", None)
            if cached_ic is not None:
                cached_ic.pop(external_req_id, None)

    ########################################################################
    # Cleanup
    ########################################################################

    def cleanup_receiver(self, request_id: str, *, first_chunk_generation: int | None = None) -> None:
        """Reclaim receiver-side per-request state (keyed by internal id).

        Safe to call from the scheduler even when ``save_async()`` has
        enqueued work that the background thread has not yet processed,
        because it only touches receiver-side dictionaries.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        self._forget_first_chunk(request_id, first_chunk_generation)
        self.finished_requests.discard(request_id)
        self.get_req_chunk.pop(request_id, None)
        self.requests_with_ready_chunks.discard(request_id)
        self.ready_chunk_info.pop(request_id, None)
        self.inflight_chunk_info.pop(request_id, None)
        self.request_ids_mapping.pop(request_id, None)
        self.requests_origin_status.pop(request_id, None)

        self._cancelled_load_reqs.add(request_id)
        self._finished_load_reqs.discard(request_id)

    def cleanup_sender(self, external_req_id: str) -> None:
        """Reclaim sender-side per-request state (keyed by external id).

        Must only be called after the terminal chunk has actually been
        sent (i.e. from ``_send_single_request``), not before.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        self.put_req_chunk.pop(external_req_id, None)
        self.request_payload.pop(external_req_id, None)
        self.code_prompt_token_ids.pop(external_req_id, None)
        for attr_name in (
            "code_prompt_emitted_frames",
            "code_prompt_emitted_chunks",
            "code_prompt_first_frame_ts",
            "code_prompt_latest_frame_ts",
        ):
            state = getattr(self, attr_name, None)
            if isinstance(state, dict):
                state.pop(external_req_id, None)

        cached_ic = getattr(self, "_cached_ic", None)
        if cached_ic is not None:
            cached_ic.pop(external_req_id, None)

    def cleanup(
        self,
        request_id: str,
        external_req_id: str | None = None,
        *, first_chunk_generation: int | None = None,
    ) -> None:
        """Reclaim all per-request state after a request finishes.

        Idempotent: calling with an already-cleaned or unknown id is safe.

        Args:
            request_id: Internal request id (receive / scheduler side key).
            external_req_id: External request id (send / payload side key).
                When *None*, looked up from ``request_ids_mapping``.
        """
        if external_req_id is None:
            external_req_id = self.request_ids_mapping.get(request_id, request_id)

        self.cleanup_receiver(request_id, first_chunk_generation=first_chunk_generation)
        self.cleanup_sender(external_req_id)

    ########################################################################
    # Schedule Helper
    ########################################################################

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
    ) -> None:
        """
        Process pending chunks for waiting and running queues.
        """
        if self.connector.stage_id == 0:
            return
        self._process_chunk_queue(
            waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, self._finished_load_reqs
        )
        self._process_chunk_queue(
            running_queue, self.waiting_for_chunk_running_requests, RequestStatus.RUNNING, self._finished_load_reqs
        )
        self._prioritize_ready_chunks(waiting_queue, running_queue)
        while len(running_queue) > self.scheduler_max_num_seqs:
            request = running_queue.pop()
            request.status = RequestStatus.PREEMPTED
            waiting_queue.prepend_requests([request])

    def restore_queues(self, waiting_queue: Any, running_queue: list[Request]) -> None:
        """
        Restore requests waiting for chunk to the waiting and running queues.
        """
        # Add request waiting for chunk to the waiting and running queue
        for request in self.waiting_for_chunk_waiting_requests:
            waiting_queue.add_request(request)
        self.waiting_for_chunk_waiting_requests = deque()

        if self.waiting_for_chunk_running_requests:
            running_queue.extend(self.waiting_for_chunk_running_requests)
        self.waiting_for_chunk_running_requests = deque()

    def waiting_for_chunk_pressure(self) -> tuple[int, float | None]:
        """Return queued WAITING_FOR_CHUNK count and oldest arrival age.

        The scheduler historically treated downstream requests waiting for
        stage input as backlog pressure.  The current async-chunk path removes
        those requests from scheduler queues before the pressure policy runs,
        so this diagnostic hook lets experiments opt back into counting them
        without changing default scheduling behavior.
        """
        now = time.time()
        count = 0
        oldest_age: float | None = None
        for queue in (
            self.waiting_for_chunk_waiting_requests,
            self.waiting_for_chunk_running_requests,
        ):
            for request in queue:
                count += 1
                arrival_time = getattr(request, "arrival_time", None)
                if arrival_time is None:
                    continue
                try:
                    age = now - float(arrival_time)
                except (TypeError, ValueError):
                    continue
                if age < 0:
                    continue
                oldest_age = age if oldest_age is None else max(oldest_age, age)
        return count, oldest_age

    def postprocess_scheduler_output(
        self,
        scheduler_output: Any,
        requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Add additional info for cached requests and
        clean up ready chunks from scheduler output.
        """
        if requests is not None:
            self.attach_cached_additional_information(scheduler_output, requests)
        self._clear_chunk_ready(scheduler_output)

    @staticmethod
    def attach_cached_additional_information(scheduler_output: Any, requests: dict[str, Request]) -> None:
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if not cached_reqs:
            return
        if not hasattr(cached_reqs, "additional_information"):
            cached_reqs.additional_information = {}
        for req_id in cached_reqs.req_ids:
            request = requests.get(req_id) if req_id else None
            additional_info = getattr(request, "additional_information", None) if request else None
            cached_reqs.additional_information[req_id] = additional_info

    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        finished_load_reqs: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    # Requests that have loaded chunk from last round
                    # of schedule, but have not scheduled
                    continue
                if request.request_id in self.finished_requests:
                    continue
                # Requests that waiting for chunk
                self.load_async(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in finished_load_reqs:
                    request.status = target_status
                    finished_load_reqs.remove(request.request_id)
                    self.requests_with_ready_chunks.add(request.request_id)
                    self._observe_first_chunk_ready(request)
                    info = self.ready_chunk_info.get(request.request_id, {})
                    info["ready_ts"] = time.time()
                    info["target_status"] = str(target_status)
                    self.ready_chunk_info[request.request_id] = info
                    if trace_enabled():
                        trace_emit(
                            "next_stage_ready", **self._interaction_trace_context,
                            request_id=request.request_id, external_req_id=request.external_req_id,
                            chunk_id=info.get("chunk_id"), size_bytes=info.get("size_bytes"),
                        )
                    self._trace_stage2_event(
                        "ready",
                        {
                            **info,
                            "priority_key": self._chunk_priority_key(request),
                            "priority_policy": self.stage2_priority_policy,
                            "priority_first_chunks": self.stage2_priority_first_chunks,
                        },
                    )
                    continue
            queue.remove(request)
            self.requests_origin_status[request.request_id] = target_status
            waiting_for_chunk_list.append(request)

    def _clear_chunk_ready(self, scheduler_output: Any) -> None:
        scheduled_req_ids: list[str] = []
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                scheduled_req_ids.append(req_data.req_id)

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                scheduled_req_ids.append(req_id)

        scheduled_ts = time.time()
        for req_id in scheduled_req_ids:
            if req_id in self.requests_with_ready_chunks:
                self.requests_with_ready_chunks.remove(req_id)
            info = self.ready_chunk_info.pop(req_id, None)
            if info is None:
                continue
            info["scheduled_ts"] = scheduled_ts
            recv_ts = info.get("recv_ts")
            ready_ts = info.get("ready_ts")
            if isinstance(recv_ts, (int, float)):
                info["recv_to_schedule_ms"] = (scheduled_ts - recv_ts) * 1000.0
            if isinstance(ready_ts, (int, float)):
                info["ready_to_schedule_ms"] = (scheduled_ts - ready_ts) * 1000.0
            self.inflight_chunk_info[req_id] = info
            if trace_enabled():
                trace_emit(
                    "chunk_scheduled", **self._interaction_trace_context,
                    request_id=req_id, external_req_id=info.get("external_req_id"),
                    chunk_id=info.get("chunk_id"), size_bytes=info.get("size_bytes"),
                )
            self._trace_stage2_event("scheduled", info)

    def trace_completed_chunks(self, num_scheduled_tokens: dict[str, int]) -> None:
        if not self.stage2_trace_enabled and not trace_enabled():
            return
        complete_ts = time.time()
        for req_id, num_tokens in num_scheduled_tokens.items():
            info = self.inflight_chunk_info.pop(req_id, None)
            if info is None:
                continue
            scheduled_ts = info.get("scheduled_ts")
            if isinstance(scheduled_ts, (int, float)):
                info["schedule_to_complete_ms"] = (complete_ts - scheduled_ts) * 1000.0
            info["complete_ts"] = complete_ts
            info["num_scheduled_tokens"] = int(num_tokens)
            if trace_enabled():
                trace_emit(
                    "chunk_output_observed", **self._interaction_trace_context,
                    request_id=req_id, external_req_id=info.get("external_req_id"),
                    chunk_id=info.get("chunk_id"), num_scheduled_tokens=int(num_tokens),
                )
            self._trace_stage2_event("complete", info)

    def finish_requests(
        self, request_ids: Any, finished_status: RequestStatus, requests: dict[str, Request] | None = None
    ) -> list[tuple[str, int]]:
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = requests.keys()

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            self._forget_first_chunk(req_id)
            request = requests.get(req_id) if requests else None
            if request is None or request.is_finished():
                # Invalid request ID.
                continue
            if req_id in self.requests_origin_status:
                request.status = self.requests_origin_status.pop(req_id)
