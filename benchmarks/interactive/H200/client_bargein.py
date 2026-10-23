#!/usr/bin/env python3
"""
client_bargein.py - Lightweight barge-in model for trace replay.

Purpose
-------
Step 1 of the four-step roadmap (see interactive_workload.md section 11/12)
requires trace_replay_driver.py to issue **audio-mode** requests and inject the
same barge-in model as the single-turn benchmark, so the multi-turn baseline
data is directly comparable to LiveServe single-turn experiments.

This module contains a minimal, self-contained barge-in client:

  - BargeInPolicy (policy=`none` / `fixed_time`)
  - _check_barge_in + _compute_virtual_played_frames (3-pointer waste model)
  - consume_sse_audio (parses text + audio chunks from the SSE stream and
    triggers barge-in via aiohttp connection close)
  - build_audio_payload (Qwen3-Omni audio-mode payload with the same
    THINKER/TALKER/CODE2WAV sampling params)

VAD policy is intentionally **not** included here — it requires the
client_vad.py worker and per-request audio fixtures from LiveServe; we will
add it in Step 1.5 once `fixed_time` is validated end-to-end.
"""

from __future__ import annotations

import base64
import asyncio
import dataclasses
import importlib.util
import io
import json
import random
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

_HELPER_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm-omni"
    / "vllm_omni"
    / "benchmarks"
    / "audio_continuity.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_liveserve_audio_continuity", _HELPER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load audio continuity helper: {_HELPER_PATH}")
_audio_continuity = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _audio_continuity
_SPEC.loader.exec_module(_audio_continuity)
compute_continuity_stats_from_durations = (
    _audio_continuity.compute_continuity_stats_from_durations
)
get_audio_continuity_threshold_s = (
    _audio_continuity.get_audio_continuity_threshold_s
)
normalize_audio_chunk_durations = (
    _audio_continuity.normalize_audio_chunk_durations
)

# ---------------------------------------------------------------------------
# Constants kept in sync with the single-turn barge-in benchmark.
# ---------------------------------------------------------------------------

# Code2wav packs talker output into fixed-size chunks before emitting an
# audio SSE event. Must match `chunk_size` in
# `vllm_omni/model_executor/stage_input_processors/qwen3_omni.py`.
DEFAULT_CODEC_CHUNK_FRAMES = 25
DEFAULT_TALKER_STAGE_ID = 1
SPEAKER_RATE: float = 25.0  # frame/s, Qwen3-Omni codec rate

# Sampling params copied to guarantee server-side behavior parity with
# single-turn experiments.
_THINKER_SP = {
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 1,
    "seed": 42,
    "repetition_penalty": 1.05,
    "stop_token_ids": [151645],
}
_TALKER_SP = {
    "temperature": 0.9,
    "top_k": 50,
    "max_tokens": 4096,
    "seed": 42,
    "detokenize": False,
    "repetition_penalty": 1.05,
    "stop_token_ids": [2150],
}
_CODE2WAV_SP = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_tokens": 65536,
    "seed": 42,
    "detokenize": True,
    "repetition_penalty": 1.1,
}


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def build_audio_payload(model: str, messages: list[dict], request_id: str,
                        max_tokens: int) -> dict:
    """Build an audio-mode payload for Qwen3-Omni.

    `messages` should already be in the OpenAI ChatCompletion format
    (role + content). This wrapper attaches modalities=[text,audio]
    and the per-stage sampling params used by the barge-in benchmark.
    """
    thinker = {**_THINKER_SP, "max_tokens": max(int(max_tokens), 1)}
    return {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "request_id": request_id,
        "modalities": ["text", "audio"],
        "sampling_params_list": [thinker, _TALKER_SP, _CODE2WAV_SP],
    }


def build_text_payload(model: str, messages: list[dict], request_id: str,
                       max_tokens: int) -> dict:
    """Build a text-only payload (back-compat with Phase 3/4 runs)."""
    return {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "request_id": request_id,
        "modalities": ["text"],
        "max_tokens": max(int(max_tokens), 1),
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }


def parse_wav_duration(audio_bytes: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Barge-in policy
# ---------------------------------------------------------------------------

POLICIES = ("none", "fixed_time")


@dataclass
class BargeInPolicy:
    """Barge-in policy supporting {none, fixed_time}.

    VAD is deferred to a future extension.
    """

    policy: str = "none"

    # fixed_time
    after_seconds: float = 2.0
    jitter_sec: float = 0.5
    # "fixed" / "uniform" / "truncexp" / "samples"
    after_distribution: str = "fixed"
    after_min: float = 0.3
    after_max: float = 5.0
    after_mean: float = 1.5
    after_samples: tuple[float, ...] = field(default_factory=tuple)
    # "t0" relative to issue time, "ttfp" relative to first audio chunk
    time_anchor: str = "ttfp"

    # Per-request Bernoulli probability of injecting barge-in.
    interrupt_prob: float = 1.0

    # Virtual-speaker accounting (see _compute_virtual_played_frames)
    virtual_play_kind: str = "time"  # default: real-time virtual playback
    virtual_play_chunks: int = 0
    virtual_play_chunk_offset: float = 0.5
    virtual_play_ratio: float = 0.0
    frame_rate: float = 25.0

    def _virtual_play_dict(self) -> dict:
        return {
            "kind": self.virtual_play_kind,
            "frame_rate": self.frame_rate,
            "chunks": self.virtual_play_chunks,
            "chunk_offset": self.virtual_play_chunk_offset,
            "ratio": self.virtual_play_ratio,
        }

    def resolve_for_request(self, rng: random.Random) -> dict:
        """Sample a per-request barge-in threshold dict."""
        vp = self._virtual_play_dict()
        if self.policy == "none":
            return {"enabled": False, "virtual_play": vp}
        if rng.random() > self.interrupt_prob:
            return {"enabled": False, "skipped_by_prob": True,
                    "virtual_play": vp}
        if self.policy == "fixed_time":
            if self.after_distribution == "uniform":
                t = rng.uniform(self.after_min, self.after_max)
            elif self.after_distribution == "truncexp":
                t = self.after_mean
                for _ in range(32):
                    t = rng.expovariate(1.0 / max(self.after_mean, 1e-6))
                    if self.after_min <= t <= self.after_max:
                        break
                else:
                    t = max(self.after_min,
                            min(self.after_max, self.after_mean))
            elif self.after_distribution == "samples":
                if not self.after_samples:
                    raise ValueError("after_distribution='samples' requires after_samples")
                t = rng.choice(self.after_samples)
            else:  # fixed + jitter
                j = (rng.uniform(-self.jitter_sec, self.jitter_sec)
                     if self.jitter_sec > 0 else 0.0)
                t = self.after_seconds + j
            return {
                "enabled": True, "kind": "time",
                "threshold_sec": max(0.0, t),
                "sampled_after_sec": t,
                "after_distribution": self.after_distribution,
                "time_anchor": self.time_anchor,
                "virtual_play": vp,
            }
        raise ValueError(f"Unknown policy: {self.policy!r} "
                         f"(supported: {POLICIES})")


def _check_barge_in(thresh: dict, now_sec: float, audio_chunks: int,
                     audio_ttfp_sec: float | None) -> bool:
    """Return True if barge-in should fire now."""
    if not thresh.get("enabled"):
        return False
    kind = thresh.get("kind")
    if kind == "time":
        anchor = thresh.get("time_anchor", "t0")
        if anchor == "ttfp":
            if audio_ttfp_sec is None:
                return False  # haven't heard first audio chunk yet
            return (now_sec - audio_ttfp_sec) >= thresh["threshold_sec"]
        return now_sec >= thresh["threshold_sec"]
    return False


def _compute_virtual_played_frames(vp: dict | None, r_cut_frames: int,
                                   now_sec: float,
                                   audio_ttfp_sec: float | None,
                                   codec_chunk_frames: int,
                                   ) -> tuple[int, float | None]:
    """Compute the virtual-speaker pointer p_cut at barge-in.

    Returns (p_cut, playback_started_sec) for the three-pointer model
    (g >= r >= p).
    """
    if r_cut_frames <= 0:
        return 0, None
    if not vp or vp.get("kind", "none") == "none":
        return r_cut_frames, audio_ttfp_sec
    kind = vp.get("kind")
    frame_rate = float(vp.get("frame_rate") or 25.0)
    if kind == "time":
        if audio_ttfp_sec is None:
            return r_cut_frames, None
        elapsed = max(0.0, now_sec - audio_ttfp_sec)
        p = int(elapsed * frame_rate)
        return max(0, min(p, r_cut_frames)), audio_ttfp_sec
    # We don't ship chunks/ratio kinds in Step 1
    return r_cut_frames, audio_ttfp_sec


def _metric_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


# ---------------------------------------------------------------------------
# Streaming result container
# ---------------------------------------------------------------------------

@dataclass
class StreamResult:
    """Per-request streaming metrics, audio-mode aware."""

    # Connectivity
    text_ttft_sec: float | None = None
    audio_ttfp_sec: float | None = None
    text_chars: int = 0
    text_content: str = ""
    audio_chunks: int = 0
    last_audio_chunk_sec: float | None = None
    audio_duration_sec: float | None = None  # normalized decoded WAV seconds
    audio_first_recv_sec: float | None = None
    audio_last_recv_sec: float | None = None
    audio_first_duration_sec: float | None = None
    audio_stream_span_sec: float | None = None
    audio_duration_delta_sec: float | None = None
    audio_stream_rtf: float | None = None
    audio_post_ttfp_rtf: float | None = None
    audio_underrun_sec: float | None = None
    audio_underrun_event_count: int | None = None
    audio_continuity_ok: bool | None = None
    audio_continuity_threshold_sec: float | None = None
    audio_continuity_evaluated: bool = False
    audio_continuity_excluded_reason: str = ""
    stream_end_sec: float | None = None
    audio_timeline: list[dict[str, float | int]] = field(default_factory=list)

    # Server-side talker progress (from SSE metrics)
    talker_frames_generated: int | None = None
    kv_prefetch_hit_used: bool | None = None
    kv_prefetch_lookup_state: str | None = None
    kv_prefetch_lookup_key_kind: str | None = None
    kv_prefetch_warm_scope: str | None = None
    kv_prefetch_gpu_pin_requested: bool | None = None
    kv_skip_offload: bool | None = None
    kv_live_abort_aborted: int | None = None
    kv_lru_request_stats: dict[str, int | float | str] = field(
        default_factory=dict
    )
    server_stage0_to_first_final_output_ms: float | None = None
    server_final_stage_submit_to_first_output_ms: float | None = None
    server_first_final_output_stage_id: int | None = None
    server_queue_wait_ms: float | None = None
    server_preprocess_ms: float | None = None
    server_ar2diffusion_ms: float | None = None
    server_final_stage_input_process_ms: float | None = None
    server_ar2code2wav_ms: float | None = None
    server_stage0_gen_ms: float | None = None
    server_stage1_gen_ms: float | None = None
    server_stage2_gen_ms: float | None = None
    server_stage0_submit_to_first_output_ms: float | None = None
    server_stage1_submit_to_first_output_ms: float | None = None
    server_stage2_submit_to_first_output_ms: float | None = None
    server_stage0_submit_to_finished_ms: float | None = None
    server_stage1_submit_to_finished_ms: float | None = None
    server_stage2_submit_to_finished_ms: float | None = None
    server_stage0_to_stage1_process_inputs_ms: float | None = None
    server_stage1_to_stage2_process_inputs_ms: float | None = None
    server_stage_durations: dict[str, float] = field(default_factory=dict)

    # Interrupt outcome
    interrupted: bool = False
    interrupt_reason: str = ""
    interrupt_stage: str = ""  # "text" / "audio" / "none"
    interrupt_triggered_at_sec: float | None = None
    audio_chunks_at_interrupt: int = 0

    # Three-pointer waste accounting
    talker_frames_generated_at_interrupt: int | None = None
    talker_frames_received_at_interrupt: int | None = None
    talker_frames_played_virtual: int | None = None
    wasted_talker_frames_total: int | None = None
    wasted_talker_frames_inaudible: int | None = None
    wasted_talker_frames_pipeline: int | None = None
    waste_ratio_total: float | None = None
    waste_ratio_inaudible: float | None = None
    waste_ratio_pipeline: float | None = None
    waste_method: str = ""  # "server_progress" / "client_only"

    # Misc
    usage: dict | None = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if len(d.get("text_content", "")) > 200:
            d["text_content"] = d["text_content"][:200] + "..."
        if not d.get("audio_timeline"):
            d.pop("audio_timeline", None)
        if not d.get("server_stage_durations"):
            d.pop("server_stage_durations", None)
        if not d.get("kv_lru_request_stats"):
            d.pop("kv_lru_request_stats", None)
        return d


# ---------------------------------------------------------------------------
# SSE consumer with optional barge-in
# ---------------------------------------------------------------------------

# 64 MB SSE line cap — audio chunks under PD audio fix can be ~2 MB / chunk.
_SSE_LINE_LIMIT = 64 * 1024 * 1024


async def consume_sse_audio(
    resp: aiohttp.ClientResponse,
    t0: float,
    thresh: dict,
    *,
    codec_chunk_frames: int = DEFAULT_CODEC_CHUNK_FRAMES,
    talker_stage_id: int = DEFAULT_TALKER_STAGE_ID,
    audio_mode: bool = True,
    record_audio_timeline: bool = False,
    audio_continuity_threshold_sec: float | None = None,
    audio_continuity_chunk_mode: str = "fragment",
    continuity_include_interrupted: bool = False,
) -> StreamResult:
    """Consume an SSE stream, parse text + audio chunks, fire barge-in
    when `_check_barge_in` returns True.

    This path is trimmed to Step 1 needs: no lead sampler, no VAD, and no
    ratio/chunks policies.
    """
    r = StreamResult()
    continuity_threshold_sec = (
        get_audio_continuity_threshold_s()
        if audio_continuity_threshold_sec is None
        else float(audio_continuity_threshold_sec)
    )
    r.audio_continuity_threshold_sec = continuity_threshold_sec
    audio_chunk_arrival_times_s: list[float] = []
    audio_chunk_durations_s: list[float] = []

    while True:
        try:
            line = await resp.content.readline(max_line_length=_SSE_LINE_LIMIT)
        except (asyncio.TimeoutError, TimeoutError):
            raise
        except (aiohttp.ClientPayloadError, aiohttp.ClientConnectionError):
            break
        except Exception:
            break
        if not line:
            break

        now = time.perf_counter() - t0

        if _check_barge_in(thresh, now, r.audio_chunks, r.audio_ttfp_sec):
            r.interrupted = True
            r.interrupt_triggered_at_sec = now
            r.interrupt_stage = "audio" if r.audio_chunks > 0 else "text"
            r.audio_chunks_at_interrupt = r.audio_chunks

            r_cut = r.audio_chunks * codec_chunk_frames
            r.talker_frames_received_at_interrupt = r_cut

            vp = thresh.get("virtual_play") if isinstance(thresh, dict) else None
            p_cut, _ = _compute_virtual_played_frames(
                vp, r_cut, now, r.audio_ttfp_sec, codec_chunk_frames)
            r.talker_frames_played_virtual = p_cut
            r.wasted_talker_frames_inaudible = max(0, r_cut - p_cut)
            if r_cut > 0:
                r.waste_ratio_inaudible = (
                    r.wasted_talker_frames_inaudible / r_cut)

            if r.talker_frames_generated is not None:
                g_cut = r.talker_frames_generated
                r.talker_frames_generated_at_interrupt = g_cut
                r.wasted_talker_frames_total = max(0, g_cut - p_cut)
                r.wasted_talker_frames_pipeline = max(0, g_cut - r_cut)
                if g_cut > 0:
                    r.waste_ratio_total = (
                        r.wasted_talker_frames_total / g_cut)
                    r.waste_ratio_pipeline = (
                        r.wasted_talker_frames_pipeline / g_cut)
                r.waste_method = "server_progress"
            else:
                r.wasted_talker_frames_total = r.wasted_talker_frames_inaudible
                r.waste_ratio_total = r.waste_ratio_inaudible
                r.waste_method = "client_only"

            kind = thresh.get("kind")
            if kind == "time":
                r.interrupt_reason = (
                    f"fixed_time t={now:.2f}s "
                    f"thr={thresh.get('threshold_sec', 0.0):.2f}s")
            else:
                r.interrupt_reason = f"kind={kind}"
            break

        decoded = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not decoded.startswith("data: "):
            continue
        chunk_str = decoded[6:]
        if chunk_str == "[DONE]":
            break
        try:
            obj = json.loads(chunk_str)
        except json.JSONDecodeError:
            continue

        if obj.get("usage"):
            r.usage = obj["usage"]

        modality = obj.get("modality", "text")

        # Talker progress hook — every audio chunk / stage_progress carries
        # `metrics.num_tokens_out_cum` for the talker stage.
        m = obj.get("metrics") or {}
        if m:
            omni_kv = m.get("omni_kv")
            if isinstance(omni_kv, dict):
                hit_used = omni_kv.get("prefetch_hit_used")
                if isinstance(hit_used, bool):
                    r.kv_prefetch_hit_used = hit_used
                lookup_state = omni_kv.get("lookup_state")
                if isinstance(lookup_state, str):
                    r.kv_prefetch_lookup_state = lookup_state
                lookup_key_kind = omni_kv.get("lookup_key_kind")
                if isinstance(lookup_key_kind, str):
                    r.kv_prefetch_lookup_key_kind = lookup_key_kind
                warm_scope = omni_kv.get("warm_scope")
                if isinstance(warm_scope, str):
                    r.kv_prefetch_warm_scope = warm_scope
                gpu_pin_requested = omni_kv.get("gpu_pin_requested")
                if isinstance(gpu_pin_requested, bool):
                    r.kv_prefetch_gpu_pin_requested = gpu_pin_requested
                skip_offload = omni_kv.get("skip_offload")
                if isinstance(skip_offload, bool):
                    r.kv_skip_offload = skip_offload
                live_abort = omni_kv.get("live_abort")
                if isinstance(live_abort, dict):
                    aborted = live_abort.get("aborted")
                    if isinstance(aborted, int):
                        r.kv_live_abort_aborted = aborted
            omni_kv_lru = m.get("omni_kv_lru")
            if isinstance(omni_kv_lru, dict):
                clean_kv_lru: dict[str, int | float | str] = {}
                for source_key, source_value in omni_kv_lru.items():
                    if isinstance(source_value, bool):
                        continue
                    if isinstance(source_value, (int, float, str)):
                        clean_kv_lru[str(source_key)] = source_value
                if clean_kv_lru:
                    r.kv_lru_request_stats.update(clean_kv_lru)
            stage_durations = m.get("stage_durations")
            if isinstance(stage_durations, dict):
                clean_stage_durations: dict[str, float] = {}
                for source_key, source_value in stage_durations.items():
                    value = _metric_float(source_value)
                    if value is not None:
                        clean_stage_durations[str(source_key)] = value
                if clean_stage_durations:
                    r.server_stage_durations.update(clean_stage_durations)
                for source_key, attr in (
                    (
                        "stage0_submit_to_first_final_output_ms",
                        "server_stage0_to_first_final_output_ms",
                    ),
                    (
                        "final_stage_submit_to_first_output_ms",
                        "server_final_stage_submit_to_first_output_ms",
                    ),
                    ("queue_wait_ms", "server_queue_wait_ms"),
                    ("preprocess_ms", "server_preprocess_ms"),
                    ("ar2diffusion_ms", "server_ar2diffusion_ms"),
                    (
                        "final_stage_input_process_ms",
                        "server_final_stage_input_process_ms",
                    ),
                    ("ar2code2wav_ms", "server_ar2code2wav_ms"),
                    ("stage_0_gen_ms", "server_stage0_gen_ms"),
                    ("stage_1_gen_ms", "server_stage1_gen_ms"),
                    ("stage_2_gen_ms", "server_stage2_gen_ms"),
                    (
                        "stage0_submit_to_first_output_ms",
                        "server_stage0_submit_to_first_output_ms",
                    ),
                    (
                        "stage1_submit_to_first_output_ms",
                        "server_stage1_submit_to_first_output_ms",
                    ),
                    (
                        "stage2_submit_to_first_output_ms",
                        "server_stage2_submit_to_first_output_ms",
                    ),
                    (
                        "stage0_submit_to_finished_ms",
                        "server_stage0_submit_to_finished_ms",
                    ),
                    (
                        "stage1_submit_to_finished_ms",
                        "server_stage1_submit_to_finished_ms",
                    ),
                    (
                        "stage2_submit_to_finished_ms",
                        "server_stage2_submit_to_finished_ms",
                    ),
                    (
                        "stage0_to_stage1_process_inputs_ms",
                        "server_stage0_to_stage1_process_inputs_ms",
                    ),
                    (
                        "stage1_to_stage2_process_inputs_ms",
                        "server_stage1_to_stage2_process_inputs_ms",
                    ),
                ):
                    value = _metric_float(clean_stage_durations.get(source_key))
                    if value is not None:
                        setattr(r, attr, value)
                stage_value = _metric_float(
                    clean_stage_durations.get("first_final_output_stage_id")
                )
                if stage_value is not None:
                    r.server_first_final_output_stage_id = int(stage_value)
            try:
                stage_id = int(m.get("stage_id", -1))
            except (TypeError, ValueError):
                stage_id = -1
            if stage_id == talker_stage_id:
                cum = m.get("num_tokens_out_cum")
                if isinstance(cum, int):
                    r.talker_frames_generated = cum

        if modality == "stage_progress":
            continue

        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content") or ""

        if modality == "text" and content:
            if r.text_ttft_sec is None:
                r.text_ttft_sec = now
            r.text_chars += len(content)
            r.text_content += content
        elif modality == "audio" and content:
            if r.audio_ttfp_sec is None:
                r.audio_ttfp_sec = now
                r.audio_first_recv_sec = now
            try:
                audio_bytes = base64.b64decode(content)
            except Exception:
                continue
            r.audio_chunks += 1
            r.audio_last_recv_sec = now
            dur = parse_wav_duration(audio_bytes)
            if dur is not None:
                r.last_audio_chunk_sec = dur
                audio_chunk_arrival_times_s.append(now)
                audio_chunk_durations_s.append(dur)
                if r.audio_first_duration_sec is None:
                    r.audio_first_duration_sec = dur
                if record_audio_timeline:
                    r.audio_timeline.append({
                        "chunk": r.audio_chunks,
                        "recv_sec": now,
                        "audio_duration_sec": dur,
                        "bytes": len(audio_bytes),
                    })

        # NOTE: We *intentionally* do NOT break on choices[0].finish_reason.
        # In Qwen3-Omni audio mode the text and audio modalities are emitted
        # as parallel SSE streams. The text stream's finish_reason="stop"
        # arrives well before talker has produced any audio chunks; breaking
        # on it loses ~93% of audio under c=4 (verified in step1/v2). The
        # authoritative termination signals are the explicit `data: [DONE]`
        # marker (handled above) and the server closing the connection
        # (handled by the `if not line: break` at top of the loop).

    r.stream_end_sec = time.perf_counter() - t0
    normalized_audio_chunk_durations_s = normalize_audio_chunk_durations(
        audio_chunk_durations_s,
        mode=audio_continuity_chunk_mode,
    )
    if normalized_audio_chunk_durations_s:
        r.audio_duration_sec = sum(normalized_audio_chunk_durations_s)
    if (r.audio_ttfp_sec is not None
            and r.audio_duration_sec is not None
            and r.audio_duration_sec > 0):
        r.audio_post_ttfp_rtf = (
            max(0.0, r.stream_end_sec - r.audio_ttfp_sec)
            / r.audio_duration_sec
        )
    if (r.audio_first_recv_sec is not None
            and r.audio_last_recv_sec is not None
            and r.audio_first_duration_sec is not None
            and r.audio_duration_sec is not None):
        r.audio_stream_span_sec = max(
            0.0, r.audio_last_recv_sec - r.audio_first_recv_sec)
        r.audio_duration_delta_sec = r.audio_duration_sec
        if r.audio_duration_delta_sec > 0:
            r.audio_stream_rtf = (
                r.audio_stream_span_sec / r.audio_duration_delta_sec)

    if not audio_mode:
        r.audio_continuity_excluded_reason = "audio_disabled"
    elif r.interrupted and not continuity_include_interrupted:
        r.audio_continuity_excluded_reason = "interrupted"
    elif not audio_chunk_arrival_times_s:
        r.audio_continuity_excluded_reason = "no_audio_chunks"
    else:
        stats = compute_continuity_stats_from_durations(
            audio_chunk_arrival_times_s,
            normalized_audio_chunk_durations_s,
            threshold_s=continuity_threshold_sec,
            chunk_duration_mode="fragment",
        )
        r.audio_underrun_sec = stats.audio_underrun_s
        r.audio_underrun_event_count = stats.audio_underrun_event_count
        r.audio_continuity_ok = stats.audio_continuity_ok
        r.audio_continuity_evaluated = True

    return r


async def close_response(resp: aiohttp.ClientResponse) -> None:
    """Force the underlying connection close — this is what triggers the
    server-side abort path under streaming SSE."""
    try:
        resp.release()
    except Exception:
        pass
    try:
        resp.close()
    except Exception:
        pass
