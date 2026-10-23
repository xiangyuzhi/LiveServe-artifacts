# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
"""Stage input processors for Ming-flash-omni-2.0 multi-stage pipeline."""

from __future__ import annotations

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)

_MING_ASYNC_BOUNDARIES = frozenset("。！？.!?\n")


def _validate_stage_inputs(source_outputs):
    """Validate source stage outputs and return them as a list."""
    if not source_outputs:
        raise ValueError("source_outputs cannot be empty")

    if not isinstance(source_outputs, list):
        source_outputs = [source_outputs]

    return source_outputs


def _first_scalar(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return value[0]
    return default if value is None else value


def _as_token_list(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        return [int(token) for token in list(value) if int(token) >= 0]
    except Exception:
        return []


def _get_async_state(transfer_manager: Any, request_id: str) -> dict[str, Any]:
    states = getattr(transfer_manager, "_ming_async_states", None)
    if states is None:
        states = {}
        setattr(transfer_manager, "_ming_async_states", states)
    return states.setdefault(request_id, {"emitted_text": ""})


def _get_async_tokenizer(transfer_manager: Any):
    tokenizer = getattr(transfer_manager, "_ming_async_tokenizer", None)
    if tokenizer is not None:
        return tokenizer

    from transformers import AutoTokenizer

    config = getattr(transfer_manager, "config", None)
    tokenizer_name = getattr(config, "tokenizer", None) or getattr(config, "model", None)
    if tokenizer_name is None:
        raise RuntimeError("Cannot resolve tokenizer for Ming async_chunk bridge")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    setattr(transfer_manager, "_ming_async_tokenizer", tokenizer)
    return tokenizer


def _decode_generated_text(transfer_manager: Any, request: Any) -> str:
    token_ids = _as_token_list(getattr(request, "output_token_ids", None))
    if not token_ids:
        return ""
    tokenizer = _get_async_tokenizer(transfer_manager)
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _extract_request_additional_info(request: Any) -> dict[str, Any]:
    try:
        return deserialize_additional_information(getattr(request, "additional_information", None))
    except Exception:
        logger.debug("Failed to deserialize Ming async request additional_information", exc_info=True)
        return {}


def _build_talker_info(
    text: str,
    additional_info: dict[str, Any],
    *,
    request_id: str | None = None,
    async_chunk: bool = True,
    eof: bool = False,
    source_finished: bool = False,
) -> dict[str, Any]:
    spk_emb = additional_info.get("spk_emb", None)
    if isinstance(spk_emb, list) and spk_emb and not hasattr(spk_emb[0], "device"):
        spk_emb = torch.tensor(spk_emb, dtype=torch.float32).unsqueeze(0)

    info = {
        "ming_task": "omni",
        "text": text,
        "spk_emb": spk_emb,
        "voice_name": _first_scalar(additional_info.get("voice_name"), "DB30"),
        "prompt_text": _first_scalar(additional_info.get("prompt_text"), None),
        "prompt_wav_lat": additional_info.get("prompt_wav_lat", None),
        "prompt_wav_emb": additional_info.get("prompt_wav_emb", None),
        "max_text_length": int(_first_scalar(additional_info.get("max_text_length"), 50)),
    }

    if request_id:
        info["request_id"] = str(request_id)

    for optional_key in (
        "ming_incremental",
        "stream_decode",
        "use_static_cache",
        "max_steps",
        "max_decode_steps",
    ):
        if optional_key in additional_info:
            info[optional_key] = _first_scalar(additional_info.get(optional_key))

    if async_chunk:
        info["ming_async_chunk"] = True
        info["ming_async_eof"] = bool(eof)
        info["ming_async_source_finished"] = bool(source_finished or eof)

    return info


def thinker2talker_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Emit Ming thinker text deltas to the talker under pipeline async_chunk.

    Ming's talker is text-driven, unlike Qwen3-Omni's hidden-state bridge.
    This bridge decodes the thinker output tokens, emits text deltas at sentence
    boundaries, and uses one dummy token to wake the generation-stage scheduler
    for each chunk.
    """
    del pooling_output

    request_id = str(getattr(request, "external_req_id", None) or getattr(request, "request_id", ""))
    finished = bool(is_finished or request.is_finished())
    state = _get_async_state(transfer_manager, request_id)

    decoded_text = _decode_generated_text(transfer_manager, request)
    emitted_text = str(state.get("emitted_text", ""))
    if decoded_text.startswith(emitted_text):
        delta = decoded_text[len(emitted_text) :]
    else:
        delta = decoded_text

    should_emit = bool(delta.strip()) and (finished or delta.rstrip()[-1:] in _MING_ASYNC_BOUNDARIES)
    eof = False
    if not should_emit:
        if not finished:
            return None
        if delta.strip():
            should_emit = True
        else:
            eof = True
            delta = ""

    if should_emit:
        state["emitted_text"] = decoded_text

    additional_info = _extract_request_additional_info(request)
    payload = _build_talker_info(
        delta,
        additional_info,
        request_id=request_id,
        async_chunk=True,
        eof=eof,
        source_finished=finished,
    )
    payload["codes"] = {"audio": [0]}
    payload["meta"] = {"finished": torch.tensor(finished, dtype=torch.bool)}

    logger.debug(
        "Ming async_chunk bridge req=%s finished=%s eof=%s text_chars=%d",
        request_id,
        finished,
        eof,
        len(delta),
    )
    return payload


def thinker2talker(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Build talker stage inputs from thinker stage outputs.

    Extracts the generated text from thinker output and constructs
    a talker input prompt with the text and any speaker/instruction info
    from the original request.
    """
    source_outputs = _validate_stage_inputs(source_outputs)

    if not isinstance(prompt, list):
        prompt = [prompt]

    talker_inputs: list[OmniTokensPrompt] = []
    for i, source_output in enumerate(source_outputs):
        output = source_output.outputs[0]

        # Get the generated text from thinker
        generated_text = output.text if hasattr(output, "text") and output.text else ""

        # Extract additional information from the original prompt
        original_prompt = prompt[i] if i < len(prompt) else None
        additional_info = {}
        if original_prompt is not None and hasattr(original_prompt, "additional_information"):
            additional_info = original_prompt.additional_information or {}

        request_id = (
            additional_info.get("request_id")
            or getattr(source_output, "external_req_id", None)
            or getattr(source_output, "request_id", None)
        )

        # Omni speech path mirrors upstream `omni_audio_generation`:
        # - `prompt` is hardcoded, `instruction` is forced to None,
        #   cfg/sigma/temperature inherit the `tts_job` defaults (the
        #   upstream API does NOT expose these knobs).
        # - Voice cloning is preset-only via `voice_name` (default
        #   'DB30'); `get_prompt_emb` is called with
        #   `use_spk_emb=True, use_zero_spk_emb=False`, so when no
        #   preset resolves upstream simply passes `spk_emb=None`
        #   through to `tts_job` rather than substituting a zero
        #   vector.
        # The bridge only plumbs the request-specific fields; the
        # talker `forward()` enforces the per-task defaults from
        # `ming_task="omni"` so any stray caller overrides are ignored.
        # Voice presets are resolved by voice_name in the talker's
        # forward() from its registered_prompts cache.
        talker_info = _build_talker_info(
            generated_text,
            additional_info,
            request_id=str(request_id) if request_id else None,
            async_chunk=False,
        )

        # Use dummy token IDs (talker builds its own embeddings from text)
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=talker_info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs
