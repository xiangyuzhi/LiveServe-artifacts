# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Stage input processor for Qwen3 Omni MoE: Thinker → Talker transition."""

import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.platforms import current_platform

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    extract_language_from_prompt,
    extract_language_from_request,
    extract_speaker_from_prompt,
    extract_speaker_from_request,
)

logger = logging.getLogger(__name__)

# Pooling output layer keys: "0" = word embedding, "24" = accept_hidden_layer
_EMBED_LAYER_KEY = "0"
_HIDDEN_LAYER_KEY = "24"


def _append_code2wav_trace(row: dict[str, Any], env_name: str, failure_message: str) -> None:
    trace_path = os.environ.get(env_name, "").strip()
    if not trace_path:
        return
    try:
        parent = os.path.dirname(trace_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        logger.debug(failure_message, exc_info=True)


def _maybe_trace_code2wav_chunk(row: dict[str, Any]) -> None:
    _append_code2wav_trace(row, "OMNI_CODEC_CHUNK_TRACE_PATH", "Failed to write Code2Wav chunk trace")


def _maybe_trace_code2wav_process(row: dict[str, Any]) -> None:
    _append_code2wav_trace(row, "OMNI_CODEC_PROCESS_TRACE_PATH", "Failed to write Code2Wav process trace")


def _pop_code2wav_process_state(transfer_manager: Any, request_id: str) -> None:
    for attr_name in (
        "code_prompt_emitted_frames",
        "code_prompt_emitted_chunks",
        "code_prompt_first_frame_ts",
        "code_prompt_latest_frame_ts",
    ):
        state = getattr(transfer_manager, attr_name, None)
        if isinstance(state, dict):
            state.pop(request_id, None)


def _codec_process_trace_enabled() -> bool:
    return bool(os.environ.get("OMNI_CODEC_PROCESS_TRACE_PATH", "").strip())


def _codec_process_trace_verbose() -> bool:
    return _env_flag("OMNI_CODEC_PROCESS_TRACE_VERBOSE")


def _cfg_bool(cfg: dict[str, Any], name: str, default: bool = False) -> bool:
    value = cfg.get(name, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _layer_tensor(layers: dict[Any, Any], key: str) -> torch.Tensor | None:
    """Fetch layer tensor with tolerant key lookup (str/int)."""
    if not isinstance(layers, dict):
        return None
    key_int = int(key)
    val = layers.get(key_int)
    if val is None:
        val = layers.get(key)
    return val if isinstance(val, torch.Tensor) else None


def _compute_talker_prompt_ids_length(info: OmniPayload, device: torch.device | str = "cuda") -> int:
    im_start_token_id = 151644
    system_token_id = 8948
    user_token_id = 872
    assistant_token_id = 77091

    ids = info.get("ids", {})
    thinker_sequences = torch.tensor(ids["all"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    input_ids = torch.tensor(ids["prompt"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    im_start_indexes = torch.cat(
        [
            torch.nonzero(input_ids[0] == im_start_token_id).squeeze(1),
            torch.tensor([thinker_sequences.shape[-1]], device=input_ids.device, dtype=input_ids.dtype),
        ],
        dim=0,
    )

    sum_user_len = 0
    assistant_len = 0
    for i in range(len(im_start_indexes) - 1):
        s = int(im_start_indexes[i].item())
        e = int(im_start_indexes[i + 1].item())
        role = int(input_ids[0, s + 1].item())
        if role == system_token_id:
            continue
        elif role == user_token_id:
            sum_user_len += e - s
        elif role == assistant_token_id and i == len(im_start_indexes) - 2:
            assistant_len += 9  # 3 + 4 + 1 + 1
        else:
            pass

    return sum_user_len + assistant_len


# =========================
# Common helpers
# =========================


def _ensure_list(x):
    """Convert ConstantList / tensor-like to Python list."""
    if hasattr(x, "_x"):
        return list(x._x)
    elif not isinstance(x, list):
        return x
    return list(x)


# =========================
# PD disaggregation helpers
# =========================


def _get_prefill_multimodal_output(
    request_id: str,
    streaming_context: Any | None,
) -> dict[str, Any] | None:
    bridge_states = getattr(streaming_context, "bridge_states", None)
    if not isinstance(bridge_states, dict):
        return None
    by_req = bridge_states.get("pd_prefill_multimodal_output_by_req")
    if not isinstance(by_req, dict):
        return None
    prefill_mm = by_req.get(request_id)
    return prefill_mm if isinstance(prefill_mm, dict) else None


def _merge_pd_embeddings(
    decode_emb: torch.Tensor,
    decode_hid: torch.Tensor,
    prefill_mm: dict[str, Any],
    device: torch.device,
    expected_total: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge prefill prompt embeddings with decode generated embeddings.

    In PD mode the prefill engine processes the prompt and the decode engine
    generates tokens starting from position 1.  This function concatenates
    them, removing the overlapping token(s):

        merged = prefill[:P] + decode[overlap:]

    where overlap = P + D - expected_total.
    """
    try:
        p_layers = prefill_mm.get("hidden_states", {}).get("layers", {})
        p_emb = p_layers[int(_EMBED_LAYER_KEY)].detach().to(device=device, dtype=torch.float)
        p_hid = p_layers[int(_HIDDEN_LAYER_KEY)].detach().to(device=device, dtype=torch.float)
    except (KeyError, AttributeError, TypeError) as exc:
        available_keys = list(prefill_mm.keys()) if isinstance(prefill_mm, dict) else type(prefill_mm).__name__
        logger.error(
            "_merge_pd_embeddings: failed to extract prefill embeddings (%s). "
            "Expected keys %r and %r, got: %s. "
            "Falling back to decode-only embeddings – talker user-segment will be degraded.",
            exc,
            _EMBED_LAYER_KEY,
            _HIDDEN_LAYER_KEY,
            available_keys,
        )
        return decode_emb, decode_hid

    if p_emb.shape[0] == 0 or decode_emb.shape[0] == 0:
        return decode_emb, decode_hid

    raw_total = p_emb.shape[0] + decode_emb.shape[0]
    overlap = max(0, raw_total - expected_total) if expected_total is not None else 0

    merged_emb = torch.cat([p_emb, decode_emb[overlap:]], dim=0)
    merged_hid = torch.cat([p_hid, decode_hid[overlap:]], dim=0)
    return merged_emb, merged_hid


def _resolve_tts_token_embedding(
    key: str,
    *,
    thinker_mm: dict[str, Any],
    prefill_mm: dict[str, Any] | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Return TTS BOS/EOS/PAD embedding tensors for the talker projection path.

    Values are taken from the current thinker (decode) ``multimodal_output``; in
    PD mode, missing keys may be filled from the paired prefill stage output.
    """
    val = thinker_mm.get("embed", {}).get(key)
    if val is None and prefill_mm is not None:
        val = prefill_mm.get("embed", {}).get(key)
    return val.detach().to(device=device, dtype=torch.float) if val is not None else None


# =========================
# Streaming input helpers
# =========================


@dataclass
class _Thinker2TalkerStreamingState:
    last_prompt_len: int = 0
    last_output_len: int = 0
    merged_sequences: list[int] = field(default_factory=list)


@dataclass
class _Qwen3OmniStreamingState:
    thinker2talker: _Thinker2TalkerStreamingState = field(default_factory=_Thinker2TalkerStreamingState)
    talker2code2wav_last_seq_len: int = 0


def _get_qwen3_streaming_state(
    request_id: str,
    streaming_context: Any | None,
) -> _Qwen3OmniStreamingState:
    bridge_states = getattr(streaming_context, "bridge_states", None)
    per_model_state = bridge_states.setdefault("qwen3_omni", {})
    state = per_model_state.get(request_id)
    if state is None:
        state = _Qwen3OmniStreamingState()
        per_model_state[request_id] = state
    return state


def _get_streaming_talker_tokens(
    request_id: str,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    new_prompt_len_snapshot: int | None = None,
    streaming_context: Any | None = None,
    *,
    clear_state: bool = False,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return streaming token slices and merged token views for thinker->talker.
       e.g. For the second streaming input request:
       merged_sequences: [input_prompt 1, output_tokens 1[:-1], input_prompt 2, output_tokens 2]
      thinker_input_ids: [input_prompt 1, output_tokens 1[:-1], input_prompt 2]
    Returns:
        inc_prompt: prompt token delta for this segment.
        inc_output: output token delta for this segment.
        merged_sequences: full thinker_sequences to send downstream.
        thinker_input_ids: full thinker_input_ids paired with merged_sequences.
    """
    state = _get_qwen3_streaming_state(request_id, streaming_context).thinker2talker
    if new_prompt_len_snapshot:
        prompt_token_ids = prompt_token_ids[:-new_prompt_len_snapshot]
    cur_prompt_len = len(prompt_token_ids)
    cur_output_len = len(output_token_ids)

    inc_prompt = prompt_token_ids[state.last_prompt_len :]
    inc_output = output_token_ids[state.last_output_len :]
    delta_sequences = inc_prompt + inc_output
    cached_sequences = state.merged_sequences

    merged_sequences = cached_sequences + delta_sequences
    thinker_input_ids = cached_sequences + inc_prompt

    # Persist history for next segment. Drop the latest sampled token to keep
    # thinker_input_ids / thinker_sequences alignment with next-step append.
    cached_sequences.extend(delta_sequences[:-1])

    state.last_prompt_len = cur_prompt_len
    state.last_output_len = cur_output_len

    if clear_state:
        state.last_prompt_len = 0
        state.last_output_len = 0
        state.merged_sequences.clear()

    return inc_prompt, inc_output, merged_sequences, thinker_input_ids


def _get_streaming_codec_delta_len(
    cur_seq_len: int,
    request_id: str,
    talker_output: Any,
    streaming_context: Any | None = None,
) -> int:
    """Return newly added seq_len for talker->code2wav in streaming mode."""
    state = _get_qwen3_streaming_state(request_id, streaming_context)
    prev_seq_len = state.talker2code2wav_last_seq_len
    seq_len = cur_seq_len - prev_seq_len
    state.talker2code2wav_last_seq_len = cur_seq_len + 1
    if bool(getattr(talker_output, "finished", False)):
        # Final segment: clear history to avoid cross-session carry-over.
        state.talker2code2wav_last_seq_len = 0
    return seq_len


# =========================
# Thinker -> Talker
# =========================


def _snapshot_qwen_thinker_request(request: Any) -> SimpleNamespace:
    """Freeze scheduler token identity at enqueue, before the request can advance."""
    prompt = tuple(_ensure_list(request.prompt_token_ids))
    output = tuple(_ensure_list(request.output_token_ids))
    all_ids = tuple(_ensure_list(request.all_token_ids))
    if all_ids != prompt + output:
        raise ValueError("Qwen Thinker enqueue token snapshot is inconsistent")
    return SimpleNamespace(
        request_id=request.request_id,
        external_req_id=request.external_req_id,
        prompt_token_ids=prompt,
        output_token_ids=output,
        all_token_ids=all_ids,
        bridge_snapshot_version=1,
        bridge_finished=request.is_finished(),
        bridge_speaker=extract_speaker_from_request(request),
        bridge_language=copy.deepcopy(extract_language_from_request(request)),
    )


def thinker2talker_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Assemble position-labelled Thinker features, then stream generated rows.

    The first transfer needs the whole prompt and the first *computed* generated
    token. A sampled token in ids.all does not yet have its own hidden features.
    Prefix hits/recomputation can overlap earlier positions; coordinates and the
    frozen token identity distinguish that from a new incremental chunk.
    """
    if not isinstance(pooling_output, dict):
        return None
    thinker_hs = pooling_output.get("hidden_states", {})
    layers = thinker_hs.get("layers", {}) if isinstance(thinker_hs, dict) else {}
    emb = _layer_tensor(layers, _EMBED_LAYER_KEY)
    hid = _layer_tensor(layers, _HIDDEN_LAYER_KEY)
    if emb is None or hid is None:
        return None
    if getattr(request, "bridge_snapshot_version", None) != 1:
        raise ValueError("Qwen Thinker transfer requires enqueue-time token snapshot")
    if request.bridge_finished != is_finished:
        raise ValueError("Qwen Thinker terminal flag differs from enqueue snapshot")
    span = pooling_output.get("meta", {}).get("qwen_thinker_feature_span")
    if (
        not isinstance(span, torch.Tensor)
        or span.device.type != "cpu"
        or span.dtype != torch.int64
        or span.shape != (2,)
    ):
        raise ValueError("Qwen Thinker transfer requires an exact CPU int64 feature span")
    start, end = span.tolist()
    if (
        start < 0
        or end <= start
        or emb.ndim != 2
        or hid.ndim != 2
        or emb.shape[0] != hid.shape[0]
        or end - start != emb.shape[0]
    ):
        raise ValueError("Qwen Thinker feature span/row count mismatch")
    all_ids, prompt_ids = tuple(request.all_token_ids), tuple(request.prompt_token_ids)
    if end > len(all_ids) or all_ids != prompt_ids + tuple(request.output_token_ids):
        raise ValueError("Qwen Thinker features exceed their frozen token snapshot")
    request_id = request.external_req_id
    state = transfer_manager.request_payload.get(request_id)
    if state is None:
        state = {
            "bridge_state_version": 1,
            "end": 0,
            "tokens": (),
            "prompt": prompt_ids,
            "chunks": [],
            "initial_emitted": False,
        }
        transfer_manager.request_payload[request_id] = state
    if state.get("bridge_state_version") != 1 or state["prompt"] != prompt_ids:
        raise ValueError("Qwen Thinker request/prompt identity changed during transfer")
    previous = state["end"]
    if start > previous or all_ids[:previous] != state["tokens"]:
        raise ValueError("Qwen Thinker feature gap or overlapping token identity mismatch")
    last_emitted = state.get("last_emitted_chunk")
    if last_emitted is not None and transfer_manager.put_req_chunk[request_id] <= last_emitted:
        raise ValueError("Qwen Thinker previous transfer was not acknowledged")
    skip = max(0, previous - start)
    new_emb, new_hid = emb[skip:].detach().cpu(), hid[skip:].detach().cpu()
    if end > previous:
        state["end"], state["tokens"] = end, all_ids[:end]
    meta = {"finished": torch.tensor(is_finished, dtype=torch.bool)}
    if not state["initial_emitted"]:
        if new_emb.shape[0]:
            state["chunks"].append((new_emb, new_hid))
        normal_ready = state["end"] >= len(prompt_ids) + 1
        terminal_prompt_only = is_finished and state["end"] == len(prompt_ids)
        if not normal_ready and not terminal_prompt_only:
            if is_finished:
                raise ValueError("Thinker ended before full prompt feature coverage")
            return None
        full_emb = torch.cat([part[0] for part in state["chunks"]], dim=0)
        full_hid = torch.cat([part[1] for part in state["chunks"]], dim=0)
        if full_emb.shape[0] != state["end"] or full_hid.shape[0] != state["end"]:
            raise ValueError("Qwen Thinker initial feature coverage is not contiguous")
        special = pooling_output.get("embed", {})
        payload = {
            "embed": {
                "prefill": full_emb,
                "cached_decode": full_emb[len(prompt_ids) :],
                **{name: special[name].detach().cpu() for name in ("tts_bos", "tts_eos", "tts_pad")},
            },
            "hidden_states": {"output": full_hid},
            "ids": {"all": list(all_ids), "prompt": list(prompt_ids), "output": list(request.output_token_ids)},
            "meta": meta,
        }
        meta["talker_prefill_tokens"] = _compute_talker_prompt_ids_length(payload, device="cpu")
        # Preserve the pre-existing terminal-at-prefill path (e.g. max_tokens=1).
        # Its genuine prompt has a complete assistant header, and the unchanged
        # model supplies its existing no-first-text bootstrap. No missing
        # prompt/assistant features are padded and no g0 feature is invented.
        meta["first_generated_embedding_present"] = normal_ready
        state["initial_emitted"] = True
        state["chunks"] = []
    else:
        if not new_emb.shape[0] and not is_finished:
            return None
        meta["override_keys"] = [("embed", "decode"), ("ids", "output")]
        payload = {"meta": meta, "ids": {"output": list(request.output_token_ids)}}
        if new_emb.shape[0]:
            payload["embed"] = {"decode": new_emb}
    state["last_emitted_chunk"] = transfer_manager.put_req_chunk[request_id]
    if request.bridge_speaker is not None:
        payload["speaker"] = request.bridge_speaker
    if request.bridge_language is not None:
        payload["language"] = request.bridge_language
    return payload


thinker2talker_async_chunk.snapshot_request = _snapshot_qwen_thinker_request


def thinker2talker(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """
    Process thinker outputs to create talker inputs.

    Workflow:
    1. Extract thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information

    In PD disaggregation mode, merges prefill-stage prompt embeddings with
    decode-stage generated embeddings before handing off to the talker.

    Args:
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for talker stage
    """
    thinker_outputs = source_outputs
    talker_inputs: list[OmniTokensPrompt] = []

    device = torch.device(current_platform.device_type)

    # Process each thinker output
    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        req_id = str(getattr(thinker_output, "request_id", f"idx-{i}"))
        prompt_token_ids = _ensure_list(thinker_output.prompt_token_ids)
        output_ids = _ensure_list(output.cumulative_token_ids)
        is_streaming_session = bool(getattr(streaming_context, "enabled", False))
        if is_streaming_session:
            prompt_token_ids, output_ids, thinker_sequences, thinker_input_ids = _get_streaming_talker_tokens(
                req_id,
                prompt_token_ids,
                output_ids,
                getattr(streaming_context, "new_prompt_len_snapshot", None),
                streaming_context,
                clear_state=bool(getattr(thinker_output, "finished", False)),
            )
        else:
            thinker_sequences = prompt_token_ids + output_ids
            thinker_input_ids = prompt_token_ids
        new_seq_length = len(prompt_token_ids + output_ids) - 1
        thinker_mm_raw = getattr(output, "multimodal_output", None)
        if not isinstance(thinker_mm_raw, dict):
            logger.debug("thinker2talker: skip req=%s due to empty multimodal_output", req_id)
            continue
        thinker_mm: OmniPayload = thinker_mm_raw
        mm_hs = thinker_mm.get("hidden_states", {})
        mm_layers = mm_hs.get("layers", {}) if isinstance(mm_hs, dict) else {}
        emb_layer = _layer_tensor(mm_layers, _EMBED_LAYER_KEY)
        hid_layer = _layer_tensor(mm_layers, _HIDDEN_LAYER_KEY)
        if emb_layer is None or hid_layer is None:
            logger.debug("thinker2talker: skip req=%s due to missing hidden-state layers", req_id)
            continue
        thinker_emb = emb_layer.detach().to(device=device, dtype=torch.float)[-new_seq_length:]
        thinker_hid = hid_layer.detach().to(device=device, dtype=torch.float)[-new_seq_length:]

        prefill_mm: dict[str, Any] | None = None
        prefill_mm = _get_prefill_multimodal_output(req_id, streaming_context)

        if prefill_mm is not None:
            expected_total = len(prompt_token_ids) + len(output_ids)
            try:
                thinker_emb, thinker_hid = _merge_pd_embeddings(
                    thinker_emb, thinker_hid, prefill_mm, device, expected_total=expected_total
                )
            except Exception as exc:
                logger.warning("[PD] Could not merge prefill embeddings: %s", exc)

        payload: OmniPayload = {
            "embed": {
                "prefill": thinker_emb,
                "tts_bos": _resolve_tts_token_embedding(
                    "tts_bos", thinker_mm=thinker_mm, prefill_mm=prefill_mm, device=device
                ),
                "tts_eos": _resolve_tts_token_embedding(
                    "tts_eos", thinker_mm=thinker_mm, prefill_mm=prefill_mm, device=device
                ),
                "tts_pad": _resolve_tts_token_embedding(
                    "tts_pad", thinker_mm=thinker_mm, prefill_mm=prefill_mm, device=device
                ),
            },
            "hidden_states": {
                "output": thinker_hid,
            },
            "ids": {
                "all": thinker_sequences,
                "prompt": thinker_input_ids,
            },
        }
        info = payload
        speaker = extract_speaker_from_prompt(prompt, index=i)
        if speaker is not None:
            info["speaker"] = speaker
        language = extract_language_from_prompt(prompt, index=i)
        if language is not None:
            info["language"] = language

        prompt_len = _compute_talker_prompt_ids_length(payload, device=device)

        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


# =========================
# Talker -> Code2Wav
# =========================


def _qwen3_omni_audio_code_frames(audio_codes: Any) -> list[list[int]]:
    """Normalize talker audio codes into frame-major codec frames.

    The Qwen3 Omni talker normally emits codebook-major tensors shaped
    [num_quantizers, num_frames].  Under load, one scheduler step can contain
    more than one new frame; keeping the whole flattened tensor as a single
    "frame" makes later chunks ragged.  Store fixed-width frames instead.
    """
    if audio_codes is None:
        return []
    if isinstance(audio_codes, torch.Tensor):
        if audio_codes.numel() == 0:
            return []
        code_tensor = audio_codes.detach().to(device="cpu", dtype=torch.long)
    else:
        code_tensor = torch.as_tensor(audio_codes, dtype=torch.long)
        if code_tensor.numel() == 0:
            return []

    if not bool(code_tensor.any().item()):
        return []

    if code_tensor.ndim == 1:
        flat = code_tensor.reshape(-1)
        if flat.numel() % 16 == 0:
            frames = flat.reshape(-1, 16)
        elif flat.numel() % 8 == 0:
            frames = flat.reshape(-1, 8)
        else:
            frames = flat.reshape(1, -1)
    elif code_tensor.ndim == 2:
        rows, cols = int(code_tensor.shape[0]), int(code_tensor.shape[1])
        if rows in (8, 16):
            frames = code_tensor.transpose(0, 1).contiguous()
        elif cols in (8, 16):
            frames = code_tensor.contiguous()
        elif rows <= 32 and cols > rows:
            frames = code_tensor.transpose(0, 1).contiguous()
        else:
            frames = code_tensor.contiguous()
    else:
        flattened = code_tensor.reshape(-1)
        if flattened.numel() % 16 == 0:
            frames = flattened.reshape(-1, 16)
        elif flattened.numel() % 8 == 0:
            frames = flattened.reshape(-1, 8)
        else:
            frames = flattened.reshape(1, -1)

    return [frame.tolist() for frame in frames if bool(frame.any().item())]


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
):
    """
    Pooling version.
    """
    request_id = request.external_req_id
    request_is_finished = getattr(request, "is_finished", None)
    finished = bool(is_finished or (request_is_finished() if callable(request_is_finished) else False))
    process_trace_enabled = _codec_process_trace_enabled()
    process_trace_verbose = _codec_process_trace_verbose() if process_trace_enabled else False
    terminal_emitted = None
    if process_trace_enabled:
        terminal_emitted = getattr(
            transfer_manager,
            "code_prompt_terminal_emitted",
            None,
        )
        if terminal_emitted is None:
            terminal_emitted = set()
            setattr(transfer_manager, "code_prompt_terminal_emitted", terminal_emitted)
        if request_id in terminal_emitted:
            if process_trace_verbose:
                _maybe_trace_code2wav_process(
                    {
                        "event": "skip_after_terminal_emit",
                        "ts": time.time(),
                        "request_id": request_id,
                        "finished": finished,
                    }
                )
            return None
    if not isinstance(pooling_output, dict):
        if not finished:
            return None
        code_frames: list[list[int]] = []
    else:
        talker_codes = pooling_output.get("codes", {})
        if not isinstance(talker_codes, dict):
            if not finished:
                return None
            code_frames = []
        else:
            code_frames = _qwen3_omni_audio_code_frames(talker_codes.get("audio"))

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size_config = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    first_chunk_size_config = int(
        cfg.get("codec_first_chunk_frames", chunk_size_config)
    )
    first_chunk_deadline_ms_config = max(
        0.0,
        float(cfg.get("codec_first_chunk_deadline_ms", 0) or 0),
    )
    first_chunk_deadline_min_frames_config = max(
        0,
        int(cfg.get("codec_first_chunk_deadline_min_frames", 0) or 0),
    )
    warmup_chunk_size_config = int(
        cfg.get("codec_warmup_chunk_frames", first_chunk_size_config)
    )
    warmup_chunk_count_config = max(0, int(cfg.get("codec_warmup_chunks", 0)))
    warmup_left_context_size_config = int(
        cfg.get("codec_warmup_left_context_frames", left_context_size_config)
    )
    coalesce_chunk_size_config = int(cfg.get("codec_coalesce_chunk_frames", 0) or 0)
    coalesce_after_chunks_config = max(
        0,
        int(cfg.get("codec_coalesce_after_chunks", 0) or 0),
    )
    coalesce_min_pending_config = int(
        cfg.get("codec_coalesce_min_pending_frames", coalesce_chunk_size_config)
        or 0
    )
    coalesce_left_context_size_config = int(
        cfg.get("codec_coalesce_left_context_frames", left_context_size_config)
    )
    coalesce_hold_until_min_pending = _cfg_bool(
        cfg,
        "codec_coalesce_hold_until_min_pending",
        False,
    )
    coalesce_emit_pending_frames = _cfg_bool(
        cfg,
        "codec_coalesce_emit_pending_frames",
        False,
    )
    first_chunk_size_config = max(1, first_chunk_size_config)
    chunk_size_config = max(1, chunk_size_config)
    warmup_chunk_size_config = max(1, warmup_chunk_size_config)
    coalesce_chunk_size_config = max(0, coalesce_chunk_size_config)
    coalesce_min_pending_config = max(0, coalesce_min_pending_config)
    first_chunk_deadline_min_frames_config = min(
        first_chunk_size_config,
        first_chunk_deadline_min_frames_config,
    )

    first_frame_ts_by_request = getattr(
        transfer_manager,
        "code_prompt_first_frame_ts",
        None,
    )
    if first_frame_ts_by_request is None:
        first_frame_ts_by_request = {}
        setattr(transfer_manager, "code_prompt_first_frame_ts", first_frame_ts_by_request)
    latest_frame_ts_by_request = getattr(
        transfer_manager,
        "code_prompt_latest_frame_ts",
        None,
    )
    if latest_frame_ts_by_request is None:
        latest_frame_ts_by_request = {}
        setattr(transfer_manager, "code_prompt_latest_frame_ts", latest_frame_ts_by_request)

    request_frames = transfer_manager.code_prompt_token_ids[request_id]
    length_before_append = len(request_frames)
    if code_frames:
        append_ts = time.time()
        request_frames.extend(code_frames)
        if length_before_append <= 0:
            first_frame_ts_by_request[request_id] = append_ts
        latest_frame_ts_by_request[request_id] = append_ts
        if process_trace_verbose or length_before_append <= 0:
            _maybe_trace_code2wav_process(
                {
                    "event": "frames_appended",
                    "ts": append_ts,
                    "request_id": request_id,
                    "finished": finished,
                    "frames_appended": len(code_frames),
                    "length_before": length_before_append,
                    "length_after": len(request_frames),
                    "was_first_frame": length_before_append <= 0,
                    "first_frame_ts": first_frame_ts_by_request.get(request_id),
                    "latest_frame_ts": latest_frame_ts_by_request.get(request_id),
                }
            )
    length = len(request_frames)

    if length <= 0:
        if finished:
            finish_ts = time.time()
            if isinstance(terminal_emitted, set):
                terminal_emitted.add(request_id)
            _maybe_trace_code2wav_process(
                {
                    "event": "finish_no_frames",
                    "ts": finish_ts,
                    "request_id": request_id,
                    "finished": True,
                    "length": 0,
                }
            )
            _pop_code2wav_process_state(transfer_manager, request_id)
            return {
                "codes": {"audio": []},
                "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
            }
        return None

    emitted_by_request = getattr(
        transfer_manager,
        "code_prompt_emitted_frames",
        None,
    )
    if emitted_by_request is None:
        emitted_by_request = {}
        setattr(transfer_manager, "code_prompt_emitted_frames", emitted_by_request)
    emitted_frames = int(emitted_by_request.get(request_id, 0) or 0)
    emitted_chunks_by_request = getattr(
        transfer_manager,
        "code_prompt_emitted_chunks",
        None,
    )
    if emitted_chunks_by_request is None:
        emitted_chunks_by_request = {}
        setattr(
            transfer_manager,
            "code_prompt_emitted_chunks",
            emitted_chunks_by_request,
        )
    emitted_chunks = int(emitted_chunks_by_request.get(request_id, 0) or 0)
    new_frames = max(0, length - emitted_frames)
    target_context_size = None
    first_chunk_deadline_triggered = False
    first_chunk_deadline_wait_ms = None
    if warmup_chunk_count_config > 0 and emitted_chunks < warmup_chunk_count_config:
        target_chunk_size = warmup_chunk_size_config
        active_left_context_size_config = warmup_left_context_size_config
        cadence_mode = "warmup"
    else:
        coalesce_ready = (
            coalesce_chunk_size_config > 0
            and coalesce_min_pending_config > 0
            and emitted_chunks >= coalesce_after_chunks_config
        )
        if coalesce_ready and new_frames >= coalesce_min_pending_config:
            target_chunk_size = coalesce_chunk_size_config
            target_context_size = (
                new_frames if coalesce_emit_pending_frames else target_chunk_size
            )
            active_left_context_size_config = coalesce_left_context_size_config
            cadence_mode = "coalesce"
        elif coalesce_ready and coalesce_hold_until_min_pending and not finished:
            gate_ts = time.time()
            first_frame_ts = first_frame_ts_by_request.get(request_id)
            latest_frame_ts = latest_frame_ts_by_request.get(request_id)
            if process_trace_verbose or emitted_chunks <= 0:
                _maybe_trace_code2wav_process(
                    {
                        "event": "gate_coalesce_hold",
                        "ts": gate_ts,
                        "request_id": request_id,
                        "finished": finished,
                        "length": length,
                        "emitted_frames_before": emitted_frames,
                        "emitted_chunks_before": emitted_chunks,
                        "new_frames": new_frames,
                        "target_chunk_frames": coalesce_chunk_size_config,
                        "coalesce_min_pending_frames": coalesce_min_pending_config,
                        "first_frame_ts": first_frame_ts,
                        "latest_frame_ts": latest_frame_ts,
                        "first_frame_to_gate_ms": (
                            (gate_ts - first_frame_ts) * 1000.0
                            if isinstance(first_frame_ts, (int, float))
                            else None
                        ),
                    }
                )
            return None
        else:
            target_chunk_size = (
                first_chunk_size_config if emitted_frames <= 0 else chunk_size_config
            )
            active_left_context_size_config = left_context_size_config
            cadence_mode = "base_first" if emitted_frames <= 0 else "base"

    if (
        first_chunk_deadline_ms_config > 0.0
        and first_chunk_deadline_min_frames_config > 0
        and emitted_frames <= 0
        and cadence_mode == "base_first"
        and new_frames >= first_chunk_deadline_min_frames_config
        and new_frames < target_chunk_size
        and not finished
    ):
        first_frame_ts = first_frame_ts_by_request.get(request_id)
        if isinstance(first_frame_ts, (int, float)):
            now_ts = time.time()
            first_chunk_deadline_wait_ms = (now_ts - first_frame_ts) * 1000.0
            if first_chunk_deadline_wait_ms >= first_chunk_deadline_ms_config:
                first_chunk_deadline_triggered = True
                target_chunk_size = first_chunk_deadline_min_frames_config
                cadence_mode = "base_first_deadline"

    if new_frames <= 0:
        if finished:
            finish_ts = time.time()
            if isinstance(terminal_emitted, set):
                terminal_emitted.add(request_id)
            _maybe_trace_code2wav_process(
                {
                    "event": "finish_no_new_frames",
                    "ts": finish_ts,
                    "request_id": request_id,
                    "finished": True,
                    "length": length,
                    "emitted_frames_before": emitted_frames,
                    "emitted_chunks_before": emitted_chunks,
                    "new_frames": new_frames,
                    "first_frame_ts": first_frame_ts_by_request.get(request_id),
                    "latest_frame_ts": latest_frame_ts_by_request.get(request_id),
                }
            )
            emitted_by_request.pop(request_id, None)
            emitted_chunks_by_request.pop(request_id, None)
            _pop_code2wav_process_state(transfer_manager, request_id)
            return {
                "codes": {"audio": []},
                "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
            }
        return None

    if new_frames < target_chunk_size and not finished:
        gate_ts = time.time()
        first_frame_ts = first_frame_ts_by_request.get(request_id)
        latest_frame_ts = latest_frame_ts_by_request.get(request_id)
        if process_trace_verbose or emitted_chunks <= 0:
            _maybe_trace_code2wav_process(
                {
                    "event": "gate_wait",
                    "ts": gate_ts,
                    "request_id": request_id,
                    "finished": finished,
                    "length": length,
                    "emitted_frames_before": emitted_frames,
                    "emitted_chunks_before": emitted_chunks,
                    "new_frames": new_frames,
                    "target_chunk_frames": target_chunk_size,
                    "cadence_mode": cadence_mode,
                    "first_chunk_deadline_ms": first_chunk_deadline_ms_config,
                    "first_chunk_deadline_min_frames": first_chunk_deadline_min_frames_config,
                    "first_chunk_deadline_wait_ms": first_chunk_deadline_wait_ms,
                    "first_chunk_deadline_triggered": first_chunk_deadline_triggered,
                    "first_frame_ts": first_frame_ts,
                    "latest_frame_ts": latest_frame_ts,
                    "first_frame_to_gate_ms": (
                        (gate_ts - first_frame_ts) * 1000.0
                        if isinstance(first_frame_ts, (int, float))
                        else None
                    ),
                    "latest_frame_to_gate_ms": (
                        (gate_ts - latest_frame_ts) * 1000.0
                        if isinstance(latest_frame_ts, (int, float))
                        else None
                    ),
                }
            )
        return None

    context_length = new_frames if finished else (target_context_size or target_chunk_size)
    context_length = min(context_length, new_frames)
    segment_start = emitted_frames
    segment_end = min(length, segment_start + context_length)

    # ensure left context does not exceed available emitted history
    left_context_size = max(
        0,
        min(segment_start, active_left_context_size_config),
    )
    window_start = max(0, segment_start - left_context_size)
    window_frames = transfer_manager.code_prompt_token_ids[request_id][window_start:segment_end]
    num_quantizers = len(window_frames[0])
    num_frames = len(window_frames)

    codes = [window_frames[f][q] for q in range(num_quantizers) for f in range(num_frames)]
    emitted_by_request[request_id] = segment_end
    emitted_chunks_by_request[request_id] = emitted_chunks + 1
    emit_ts = time.time()
    first_frame_ts = first_frame_ts_by_request.get(request_id)
    latest_frame_ts = latest_frame_ts_by_request.get(request_id)
    process_trace_row = {
        "event": "emit",
        "ts": emit_ts,
        "request_id": request_id,
        "cadence_mode": cadence_mode,
        "finished": finished,
        "length": length,
        "emitted_frames_before": emitted_frames,
        "emitted_frames_after": segment_end,
        "emitted_chunks_before": emitted_chunks,
        "new_frames": new_frames,
        "target_chunk_frames": target_chunk_size,
        "context_frames": context_length,
        "left_context_frames": left_context_size,
        "window_frames": num_frames,
        "first_chunk_deadline_ms": first_chunk_deadline_ms_config,
        "first_chunk_deadline_min_frames": first_chunk_deadline_min_frames_config,
        "first_chunk_deadline_wait_ms": first_chunk_deadline_wait_ms,
        "first_chunk_deadline_triggered": first_chunk_deadline_triggered,
        "first_frame_ts": first_frame_ts,
        "latest_frame_ts": latest_frame_ts,
        "first_frame_to_emit_ms": (
            (emit_ts - first_frame_ts) * 1000.0
            if isinstance(first_frame_ts, (int, float))
            else None
        ),
        "latest_frame_to_emit_ms": (
            (emit_ts - latest_frame_ts) * 1000.0
            if isinstance(latest_frame_ts, (int, float))
            else None
        ),
    }
    if process_trace_verbose or emitted_chunks <= 0:
        _maybe_trace_code2wav_process(process_trace_row)
    _maybe_trace_code2wav_chunk(
        {
            "ts": emit_ts,
            "request_id": request_id,
            "cadence_mode": cadence_mode,
            "finished": finished,
            "length": length,
            "emitted_frames_before": emitted_frames,
            "emitted_frames_after": segment_end,
            "emitted_chunks_before": emitted_chunks,
            "new_frames": new_frames,
            "target_chunk_frames": target_chunk_size,
            "context_frames": context_length,
            "left_context_frames": left_context_size,
            "window_frames": num_frames,
            "first_chunk_deadline_ms": first_chunk_deadline_ms_config,
            "first_chunk_deadline_min_frames": first_chunk_deadline_min_frames_config,
            "first_chunk_deadline_wait_ms": first_chunk_deadline_wait_ms,
            "first_chunk_deadline_triggered": first_chunk_deadline_triggered,
        }
    )
    if finished:
        if isinstance(terminal_emitted, set):
            terminal_emitted.add(request_id)
        emitted_by_request.pop(request_id, None)
        emitted_chunks_by_request.pop(request_id, None)
        _pop_code2wav_process_state(transfer_manager, request_id)

    return {
        "codes": {"audio": codes},
        "meta": {
            "left_context_size": left_context_size,
            "codec_chunk_frames": context_length,
            "codec_cadence_mode": cadence_mode,
            "codec_emitted_frames_before": emitted_frames,
            "codec_emitted_frames_after": segment_end,
            "codec_emitted_chunks_before": emitted_chunks,
            "codec_new_frames": new_frames,
            "codec_target_chunk_frames": target_chunk_size,
            "codec_window_frames": num_frames,
            "codec_first_chunk_deadline_ms": first_chunk_deadline_ms_config,
            "codec_first_chunk_deadline_min_frames": first_chunk_deadline_min_frames_config,
            "codec_first_chunk_deadline_wait_ms": first_chunk_deadline_wait_ms,
            "codec_first_chunk_deadline_triggered": first_chunk_deadline_triggered,
            "finished": torch.tensor(finished, dtype=torch.bool),
        },
    }


def talker2code2wav(
    source_outputs: list[Any],
    _prompt: OmniTokensPrompt | TextPrompt | None = None,
    _requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """
    Process talker outputs to create code2wav inputs.

    Workflow:
    1. Extract talker's codec code outputs (8-layer RVQ codes)
    2. Flatten codes for code2wav input
    3. Package for code2wav stage

    Args:
    Returns:
        List of OmniTokensPrompt for code2wav stage
    """
    talker_outputs = source_outputs
    code2wav_inputs: list[OmniTokensPrompt] = []
    # Process each talker output
    for i, talker_output in enumerate(talker_outputs):
        output = talker_output.outputs[0]
        req_id = str(getattr(talker_output, "request_id", f"idx-{i}"))
        cur_seq_len = len(output.cumulative_token_ids) - 1
        seq_len = cur_seq_len
        is_streaming_session = bool(getattr(streaming_context, "enabled", False))
        if is_streaming_session:
            seq_len = _get_streaming_codec_delta_len(cur_seq_len, req_id, talker_output, streaming_context)
        mm_raw = getattr(output, "multimodal_output", None)
        if not isinstance(mm_raw, dict):
            logger.debug("talker2code2wav: skip req=%s due to empty multimodal_output", req_id)
            continue
        mm: OmniPayload = mm_raw
        if "codes" not in mm or not isinstance(mm.get("codes"), dict) or "audio" not in mm["codes"]:
            logger.debug("talker2code2wav: skip req=%s due to missing codes.audio", req_id)
            continue
        # Extract codec codes from talker output
        # Expected shape: [8, seq_len] (8-layer RVQ codes)
        codec_codes = (
            mm["codes"]["audio"][-seq_len:].to(torch.long).transpose(0, 1).cpu().to(torch.long).reshape(-1).tolist()
        )  # 16, seq_len
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
