#!/usr/bin/env python3
"""Controlled ShareGPT driver for H200 reproduction runs.

This complements vLLM's generic `bench serve` path with a smaller, predictable
client that records per-request JSONL and summary metrics. It is intentionally
single-host and OpenAI-chat-omni specific.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import importlib.util
import io
import json
import os
import random
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO_ROOT / "data" / "ShareGPT_Chinese_English_90k.json"
DEFAULT_URL = "http://127.0.0.1:8666/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

_BARGE_IN_HELPER_PATH = (
    REPO_ROOT / "benchmarks" / "interactive" / "H200" / "client_bargein.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_liveserve_sharegpt_bargein_helper",
    _BARGE_IN_HELPER_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load barge-in helper: {_BARGE_IN_HELPER_PATH}")
_barge_in_helper = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _barge_in_helper
_SPEC.loader.exec_module(_barge_in_helper)

BargeInPolicy = _barge_in_helper.BargeInPolicy
DEFAULT_CODEC_CHUNK_FRAMES = _barge_in_helper.DEFAULT_CODEC_CHUNK_FRAMES
DEFAULT_TALKER_STAGE_ID = _barge_in_helper.DEFAULT_TALKER_STAGE_ID
compute_continuity_stats_from_durations = (
    _barge_in_helper.compute_continuity_stats_from_durations
)
get_audio_continuity_threshold_s = (
    _barge_in_helper.get_audio_continuity_threshold_s
)
normalize_audio_chunk_durations = (
    _barge_in_helper.normalize_audio_chunk_durations
)
_check_barge_in = _barge_in_helper._check_barge_in
_compute_virtual_played_frames = _barge_in_helper._compute_virtual_played_frames

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, "
    "Alibaba Group, capable of perceiving auditory and visual inputs, "
    "as well as generating text and speech."
)

_DUMMY_VOCAB = [
    "the", "a", "is", "of", "to", "in", "and", "for", "on", "it",
    "that", "this", "with", "as", "was", "are", "be", "at", "by", "an",
    "from", "or", "but", "not", "have", "had", "has", "do", "did", "will",
    "can", "may", "if", "so", "no", "up", "out", "all", "one", "two",
    "new", "old", "big", "get", "go", "see", "say", "use", "way", "day",
]

SHAREGPT_LENGTH_PROFILE_DEFAULTS = {
    "shareGPT-len1024": {"input_mean": 512, "output_mean": 1024},
    "shareGPT-len2048": {"input_mean": 512, "output_mean": 2048},
    "shareGPT-len4096": {"input_mean": 512, "output_mean": 4096},
}

HETERO_PROFILE = "shareGPT-hetero-longshort"


@dataclass(frozen=True)
class WorkItem:
    prompt: str
    max_tokens: int
    prompt_target_tokens: int | None = None
    output_target_tokens: int | None = None
    length_class: str = "default"

THINKER_SP = {
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 1,
    "seed": 42,
    "repetition_penalty": 1.05,
    "stop_token_ids": [151645],
}
TALKER_SP = {
    "temperature": 0.9,
    "top_k": 50,
    "max_tokens": 4096,
    "seed": 42,
    "detokenize": False,
    "repetition_penalty": 1.05,
    "stop_token_ids": [2150],
}
CODE2WAV_SP = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_tokens": 65536,
    "seed": 42,
    "detokenize": True,
    "repetition_penalty": 1.1,
}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def stats(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "mean": (sum(values) / len(values)) if values else None,
    }

def sample_truncated_normal(
    rng: random.Random,
    mean: float,
    std: float,
    min_value: int,
    max_value: int | None = None,
) -> int:
    if std <= 0:
        value = int(round(mean))
    else:
        value = int(round(mean))
        for _ in range(100):
            sampled = int(round(rng.gauss(mean, std)))
            if sampled >= min_value and (max_value is None or sampled <= max_value):
                value = sampled
                break
        else:
            sampled = int(round(rng.gauss(mean, std)))
            value = sampled
    value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def generate_dummy_text(token_count: int, rng: random.Random) -> str:
    if token_count <= 0:
        token_count = 1
    words = [rng.choice(_DUMMY_VOCAB) for _ in range(token_count)]
    words[-1] = f"req{rng.randint(0, 99999)}"
    return " ".join(words)


def estimate_text_tokens(text: str) -> int:
    """Best-effort token estimate for streamed text before final usage arrives."""
    stripped = text.strip()
    if not stripped:
        return 0
    words = stripped.split()
    if len(words) > 1:
        return len(words)
    return max(1, int(round(len(stripped) / 4.0)))


def resolve_length_profile(args: argparse.Namespace) -> None:
    if args.length_profile == HETERO_PROFILE:
        args.prompt_source = "synthetic"
        if args.input_tokens_mean is None:
            args.input_tokens_mean = args.hetero_short_input_tokens
        if args.output_tokens_mean is None:
            args.output_tokens_mean = args.hetero_short_output_tokens
        return
    defaults = SHAREGPT_LENGTH_PROFILE_DEFAULTS.get(args.length_profile)
    if defaults is None:
        return
    if args.prompt_source == "dataset":
        args.prompt_source = "synthetic"
    if args.input_tokens_mean is None:
        args.input_tokens_mean = defaults["input_mean"]
    if args.output_tokens_mean is None:
        args.output_tokens_mean = defaults["output_mean"]


def load_arrival_offsets(path: Path, count: int, time_scale: float) -> list[float]:
    offsets: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = row.get("t_sec", row.get("submit_offset_s", row.get("arrival_time_s")))
            if raw is None:
                continue
            try:
                offsets.append(float(raw) * time_scale)
            except (TypeError, ValueError):
                continue
            if len(offsets) >= count:
                break
    if len(offsets) < count:
        raise ValueError(
            f"arrival manifest {path} has {len(offsets)} usable offsets, "
            f"requested {count}"
        )
    base = min(offsets) if offsets else 0.0
    return [max(0.0, t - base) for t in offsets[:count]]


def get_nested_value(obj: Any, key: str) -> Any:
    cur = obj
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def load_interrupt_after_samples(path: Path, key: str, time_scale: float) -> tuple[float, ...]:
    samples: list[float] = []
    fallback_keys = (key, "audio_duration_s", "audio_duration_sec", "duration_sec")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw: Any = None
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                for candidate in fallback_keys:
                    raw = get_nested_value(obj, candidate)
                    if raw is not None:
                        break
            elif isinstance(obj, (int, float, str)):
                raw = obj
            else:
                raw = line.split()[0] if line.split() else None
            try:
                value = float(raw) * time_scale
            except (TypeError, ValueError):
                continue
            if value > 0.0:
                samples.append(value)
    if not samples:
        raise ValueError(f"no positive interrupt-after samples loaded from {path}")
    return tuple(samples)


def build_work_items(args: argparse.Namespace) -> list[WorkItem]:
    rng = random.Random(args.seed)
    if args.length_profile == HETERO_PROFILE:
        items: list[WorkItem] = []
        long_count = min(max(0, args.hetero_long_count), args.num_requests)
        for idx in range(args.num_requests):
            if idx < long_count:
                input_tokens = args.hetero_long_input_tokens
                output_tokens = args.hetero_long_output_tokens
                length_class = "long"
            else:
                input_tokens = args.hetero_short_input_tokens
                output_tokens = args.hetero_short_output_tokens
                length_class = "short"
            prompt = generate_dummy_text(input_tokens, rng)
            items.append(
                WorkItem(
                    prompt=prompt,
                    max_tokens=max(1, int(output_tokens)),
                    prompt_target_tokens=input_tokens,
                    output_target_tokens=output_tokens,
                    length_class=length_class,
                )
            )
        return items

    if args.prompt_source == "synthetic":
        if args.input_tokens_mean is None:
            raise ValueError("--prompt-source=synthetic requires --input-tokens-mean")
        prompts: list[tuple[str, int | None]] = []
        for _ in range(args.num_requests):
            target = sample_truncated_normal(
                rng,
                float(args.input_tokens_mean),
                float(args.input_tokens_std),
                args.min_input_tokens,
                args.max_input_tokens if args.max_input_tokens > 0 else None,
            )
            prompts.append((generate_dummy_text(target, rng), target))
    else:
        loaded = load_prompts(
            args.dataset,
            args.seed,
            args.num_requests,
            args.max_prompt_chars,
        )
        if len(loaded) < args.num_requests:
            raise RuntimeError(
                f"only loaded {len(loaded)} prompts, requested {args.num_requests}"
            )
        prompts = [(prompt, None) for prompt in loaded]

    items: list[WorkItem] = []
    for prompt, prompt_target in prompts:
        output_target = None
        max_tokens = args.max_tokens
        if args.output_tokens_mean is not None:
            output_target = sample_truncated_normal(
                rng,
                float(args.output_tokens_mean),
                float(args.output_tokens_std),
                args.min_output_tokens,
                args.max_output_tokens if args.max_output_tokens > 0 else None,
            )
            max_tokens = output_target
        items.append(
            WorkItem(
                prompt=prompt,
                max_tokens=max(1, int(max_tokens)),
                prompt_target_tokens=prompt_target,
                output_target_tokens=output_target,
                length_class="default",
            )
        )
    return items


def ms_stats_to_sec(value: dict[str, float | None]) -> dict[str, float | None]:
    return {
        k: (None if v is None else float(v) / 1000.0)
        for k, v in value.items()
    }


def fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def fmt_float(value: float | None, ndigits: int = 3) -> str:
    return "-" if value is None else f"{value:.{ndigits}f}"


def load_prompts(path: Path, seed: int, count: int, max_prompt_chars: int) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = [e for e in data if "conversations" in e and len(e["conversations"]) >= 2]
    random.seed(seed)
    random.shuffle(rows)

    prompts: list[str] = []
    for row in rows:
        prompt = str(row["conversations"][0].get("value", ""))
        if not prompt:
            continue
        if max_prompt_chars > 0:
            prompt = prompt[:max_prompt_chars]
        prompts.append(prompt)
        if len(prompts) >= count:
            break
    return prompts


def wav_duration(audio_bytes: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return None


def build_payload(prompt: str, max_tokens: int, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.mode == "text":
        payload.update(
            {
                "modalities": ["text"],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
            }
        )
        if args.force_output_len:
            payload["min_tokens"] = max_tokens
            payload["ignore_eos"] = True
    else:
        thinker = {**THINKER_SP, "max_tokens": max_tokens}
        if args.force_output_len:
            thinker["min_tokens"] = max_tokens
            thinker["ignore_eos"] = True
        payload["modalities"] = ["text", "audio"]
        payload["sampling_params_list"] = [thinker, TALKER_SP, CODE2WAV_SP]
    return payload

def build_interrupt_threshold(idx: int, args: argparse.Namespace) -> dict[str, Any]:
    policy = BargeInPolicy(
        policy=args.policy,
        after_seconds=args.interrupt_after_seconds,
        jitter_sec=args.interrupt_jitter_sec,
        after_distribution=args.interrupt_after_distribution,
        after_min=args.interrupt_after_min,
        after_max=args.interrupt_after_max,
        after_mean=args.interrupt_after_mean,
        after_samples=getattr(args, "interrupt_after_samples_values", ()),
        time_anchor=args.interrupt_time_anchor,
        interrupt_prob=args.interrupt_prob,
    )
    # Per-index deterministic sampling keeps baseline/scheduler request decisions
    # aligned while remaining thread-safe under concurrent execution.
    rng = random.Random(args.seed + 1_000_003 * (idx + 1))
    return policy.resolve_for_request(rng)


def compute_posthoc_playback_waste(
    thresh: dict[str, Any],
    *,
    audio_duration_s: float,
    audio_ttfp_s: float | None,
    audio_chunks: int,
    talker_frames_generated: int | None,
) -> dict[str, Any] | None:
    """Estimate playback waste when the HTTP stream ends before barge-in.

    The live interrupt path can only close an active stream.  With paper-style
    cut times sampled from output audio duration, a fast server may finish
    streaming before playback reaches the sampled cut.  This helper records the
    generated-but-unplayed audio implied by that cut without changing the live
    `interrupted` flag.
    """
    if not thresh.get("enabled") or thresh.get("kind") != "time":
        return None
    if audio_duration_s <= 0.0:
        return None
    try:
        threshold_s = float(thresh.get("threshold_sec"))
    except (TypeError, ValueError):
        return None

    anchor = thresh.get("time_anchor", "t0")
    if anchor == "ttfp":
        if audio_ttfp_s is None:
            return None
        playback_cut_s = threshold_s
        trigger_at_s = audio_ttfp_s + threshold_s
    else:
        playback_cut_s = threshold_s - (audio_ttfp_s or 0.0)
        trigger_at_s = threshold_s

    if playback_cut_s < 0.0 or playback_cut_s >= audio_duration_s:
        return None

    total_frames = talker_frames_generated
    if total_frames is None or total_frames <= 0:
        total_frames = audio_chunks * DEFAULT_CODEC_CHUNK_FRAMES
    if total_frames <= 0:
        return None

    unplayed_s = max(0.0, audio_duration_s - playback_cut_s)
    waste_ratio = unplayed_s / audio_duration_s
    return {
        "posthoc_playback_interrupt": True,
        "posthoc_interrupt_triggered_at_sec": trigger_at_s,
        "posthoc_audio_played_sec": playback_cut_s,
        "posthoc_audio_unplayed_sec": unplayed_s,
        "posthoc_wasted_talker_frames_total": int(round(float(total_frames) * waste_ratio)),
        "posthoc_waste_ratio_audio": waste_ratio,
        "posthoc_waste_method": "playback_duration",
    }


def send_one(
    idx: int,
    item: WorkItem,
    args: argparse.Namespace,
    submit_offset_s: float | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    prompt = item.prompt
    thresh = build_interrupt_threshold(idx, args)
    prompt = item.prompt
    continuity_threshold_sec = (
        get_audio_continuity_threshold_s()
        if args.audio_continuity_threshold_sec is None
        else float(args.audio_continuity_threshold_sec)
    )
    row: dict[str, Any] = {
        "idx": idx,
        "mode": args.mode,
        "client_submit_offset_s": submit_offset_s,
        "prompt_chars": len(prompt),
        "prompt_target_tokens": item.prompt_target_tokens,
        "output_target_tokens": item.output_target_tokens,
        "length_class": item.length_class,
        "max_tokens": item.max_tokens,
        "ok": False,
        "error": None,
        "status_code": None,
        "elapsed_ms": None,
        "ttft_ms": None,
        "audio_ttfp_ms": None,
        "text_chars": 0,
        "audio_chunks": 0,
        "audio_duration_s": 0.0,
        "audio_stream_span_s": None,
        "audio_stream_rtf": None,
        "audio_post_ttfp_rtf": None,
        "audio_underrun_sec": None,
        "audio_underrun_event_count": None,
        "audio_continuity_ok": None,
        "audio_continuity_evaluated": False,
        "audio_continuity_excluded_reason": "",
        "interrupted": False,
        "interrupt_reason": "",
        "interrupt_stage": "none",
        "interrupt_triggered_at_sec": None,
        "audio_chunks_at_interrupt": 0,
        "skipped_by_prob": bool(thresh.get("skipped_by_prob", False)),
        "talker_frames_generated": None,
        "talker_frames_generated_at_interrupt": None,
        "talker_frames_received_at_interrupt": None,
        "talker_frames_played_virtual": None,
        "wasted_talker_frames_total": None,
        "wasted_talker_frames_inaudible": None,
        "wasted_talker_frames_pipeline": None,
        "waste_ratio_total": None,
        "waste_ratio_inaudible": None,
        "waste_ratio_pipeline": None,
        "waste_method": "",
        "posthoc_playback_interrupt": False,
        "posthoc_interrupt_triggered_at_sec": None,
        "posthoc_audio_played_sec": None,
        "posthoc_audio_unplayed_sec": None,
        "posthoc_wasted_talker_frames_total": None,
        "posthoc_waste_ratio_audio": None,
        "posthoc_waste_method": "",
        "text_tokens_estimated_at_interrupt": None,
        "completion_tokens_before_interrupt": None,
        "completion_tokens_after_interrupt": None,
        "wasted_completion_tokens": None,
        "generated_completion_tokens_total": None,
        "waste_ratio_tokens": None,
        "token_waste_method": "",
        "usage": None,
    }

    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": "", "https": ""}

    try:
        resp = session.post(
            args.url,
            json=build_payload(prompt, item.max_tokens, args),
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=args.timeout,
            proxies={"http": "", "https": ""},
        )
        row["status_code"] = resp.status_code
        if resp.status_code != 200:
            row["error"] = resp.text[:1000]
            row["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
            return row

        text_chars = 0
        text_parts: list[str] = []
        text_tokens_at_interrupt: int | None = None
        audio_chunks = 0
        audio_chunk_arrival_times_s: list[float] = []
        audio_chunk_durations_s: list[float] = []
        audio_first_recv_s: float | None = None
        audio_last_recv_s: float | None = None
        usage = None
        ttft = None
        audio_ttfp = None
        talker_frames_generated: int | None = None
        interrupted = False

        for line in resp.iter_lines():
            if not line:
                continue
            now = time.perf_counter() - t0
            anchor_ttfp = audio_ttfp if args.mode == "audio" else ttft
            if _check_barge_in(thresh, now, audio_chunks, anchor_ttfp):
                interrupted = True
                row["interrupted"] = True
                row["interrupt_triggered_at_sec"] = now
                row["interrupt_stage"] = "audio" if audio_chunks > 0 else "text"
                row["audio_chunks_at_interrupt"] = audio_chunks
                text_tokens_at_interrupt = estimate_text_tokens("".join(text_parts))
                row["text_tokens_estimated_at_interrupt"] = text_tokens_at_interrupt
                row["completion_tokens_before_interrupt"] = text_tokens_at_interrupt

                r_cut = audio_chunks * DEFAULT_CODEC_CHUNK_FRAMES
                row["talker_frames_received_at_interrupt"] = r_cut
                p_cut, _ = _compute_virtual_played_frames(
                    thresh.get("virtual_play"),
                    r_cut,
                    now,
                    audio_ttfp,
                    DEFAULT_CODEC_CHUNK_FRAMES,
                )
                row["talker_frames_played_virtual"] = p_cut
                row["wasted_talker_frames_inaudible"] = max(0, r_cut - p_cut)
                if r_cut > 0:
                    row["waste_ratio_inaudible"] = (
                        row["wasted_talker_frames_inaudible"] / r_cut
                    )
                if talker_frames_generated is not None:
                    g_cut = talker_frames_generated
                    row["talker_frames_generated_at_interrupt"] = g_cut
                    row["wasted_talker_frames_total"] = max(0, g_cut - p_cut)
                    row["wasted_talker_frames_pipeline"] = max(0, g_cut - r_cut)
                    if g_cut > 0:
                        row["waste_ratio_total"] = (
                            row["wasted_talker_frames_total"] / g_cut
                        )
                        row["waste_ratio_pipeline"] = (
                            row["wasted_talker_frames_pipeline"] / g_cut
                        )
                    row["waste_method"] = "server_progress"
                else:
                    row["wasted_talker_frames_total"] = row[
                        "wasted_talker_frames_inaudible"
                    ]
                    row["waste_ratio_total"] = row["waste_ratio_inaudible"]
                    row["waste_method"] = "client_only"
                row["interrupt_reason"] = (
                    f"fixed_time t={now:.2f}s "
                    f"thr={thresh.get('threshold_sec', 0.0):.2f}s"
                )
                resp.close()
                break

            decoded = line.decode("utf-8", errors="replace")
            if not decoded.startswith("data: "):
                continue
            chunk = decoded[6:]
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]

            metrics = obj.get("metrics") or {}
            if metrics:
                try:
                    stage_id = int(metrics.get("stage_id", -1))
                except (TypeError, ValueError):
                    stage_id = -1
                if stage_id == DEFAULT_TALKER_STAGE_ID:
                    cum = metrics.get("num_tokens_out_cum")
                    if isinstance(cum, int):
                        talker_frames_generated = cum

            modality = obj.get("modality", "text")
            if modality == "stage_progress":
                continue

            choices = obj.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content") or ""

            if modality == "text" and content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text_chars += len(content)
                text_parts.append(content)
            elif modality == "audio" and content:
                if audio_ttfp is None:
                    audio_ttfp = time.perf_counter() - t0
                    audio_first_recv_s = audio_ttfp
                audio_chunks += 1
                audio_last_recv_s = time.perf_counter() - t0
                try:
                    dur = wav_duration(base64.b64decode(content))
                    if dur:
                        audio_chunk_arrival_times_s.append(audio_last_recv_s)
                        audio_chunk_durations_s.append(dur)
                except Exception:
                    pass

        elapsed = time.perf_counter() - t0
        normalized_audio_chunk_durations_s = normalize_audio_chunk_durations(
            audio_chunk_durations_s,
            mode=args.audio_continuity_chunk_mode,
        )
        audio_duration_s = sum(normalized_audio_chunk_durations_s)
        audio_stream_span_s = None
        audio_stream_rtf = None
        audio_post_ttfp_rtf = None
        if audio_first_recv_s is not None and audio_last_recv_s is not None:
            audio_stream_span_s = max(0.0, audio_last_recv_s - audio_first_recv_s)
        if audio_duration_s > 0:
            if audio_stream_span_s is not None:
                audio_stream_rtf = audio_stream_span_s / audio_duration_s
            if audio_ttfp is not None:
                audio_post_ttfp_rtf = max(0.0, elapsed - audio_ttfp) / audio_duration_s

        if args.posthoc_playback_interrupt_waste and not interrupted:
            posthoc = compute_posthoc_playback_waste(
                thresh,
                audio_duration_s=audio_duration_s,
                audio_ttfp_s=audio_ttfp,
                audio_chunks=audio_chunks,
                talker_frames_generated=talker_frames_generated,
            )
            if posthoc:
                row.update(posthoc)

        if args.mode != "audio":
            row["audio_continuity_excluded_reason"] = "audio_disabled"
        elif interrupted and not args.continuity_include_interrupted:
            row["audio_continuity_excluded_reason"] = "interrupted"
        elif not audio_chunk_arrival_times_s:
            row["audio_continuity_excluded_reason"] = "no_audio_chunks"
        else:
            continuity = compute_continuity_stats_from_durations(
                audio_chunk_arrival_times_s,
                normalized_audio_chunk_durations_s,
                threshold_s=continuity_threshold_sec,
                chunk_duration_mode="fragment",
            )
            row["audio_underrun_sec"] = continuity.audio_underrun_s
            row["audio_underrun_event_count"] = continuity.audio_underrun_event_count
            row["audio_continuity_ok"] = continuity.audio_continuity_ok
            row["audio_continuity_evaluated"] = True

        generated_completion_tokens: int | None = None
        token_waste_method = ""
        if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
            try:
                generated_completion_tokens = int(usage.get("completion_tokens") or 0)
                token_waste_method = "usage"
            except (TypeError, ValueError):
                generated_completion_tokens = None
        if generated_completion_tokens is None:
            generated_completion_tokens = estimate_text_tokens("".join(text_parts))
            token_waste_method = "stream_text_estimate"
        row["generated_completion_tokens_total"] = generated_completion_tokens
        if interrupted:
            before = text_tokens_at_interrupt
            if before is None:
                before = estimate_text_tokens("".join(text_parts))
                row["completion_tokens_before_interrupt"] = before
            wasted_tokens = max(0, generated_completion_tokens - before)
            row["completion_tokens_after_interrupt"] = wasted_tokens
            row["wasted_completion_tokens"] = wasted_tokens
            row["waste_ratio_tokens"] = (
                wasted_tokens / generated_completion_tokens
                if generated_completion_tokens > 0 else None
            )
            row["token_waste_method"] = token_waste_method
        else:
            row["completion_tokens_before_interrupt"] = generated_completion_tokens
            row["completion_tokens_after_interrupt"] = 0
            row["wasted_completion_tokens"] = 0
            row["waste_ratio_tokens"] = 0.0 if generated_completion_tokens > 0 else None
            row["token_waste_method"] = token_waste_method

        row.update(
            {
                "ok": True,
                "elapsed_ms": elapsed * 1000.0,
                "ttft_ms": (ttft * 1000.0) if ttft is not None else None,
                "audio_ttfp_ms": (audio_ttfp * 1000.0) if audio_ttfp is not None else None,
                "text_chars": text_chars,
                "audio_chunks": audio_chunks,
                "audio_duration_s": audio_duration_s,
                "audio_stream_span_s": audio_stream_span_s,
                "audio_stream_rtf": audio_stream_rtf,
                "audio_post_ttfp_rtf": audio_post_ttfp_rtf,
                "talker_frames_generated": talker_frames_generated,
                "usage": usage,
            }
        )
    except Exception as exc:  # noqa: BLE001 - benchmark rows should capture failures.
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    finally:
        session.close()
    return row


def summarize(
    rows: list[dict[str, Any]],
    wall_s: float,
    args: argparse.Namespace,
    *,
    actual_wall_s: float | None = None,
) -> dict[str, Any]:
    measurement_rows = [
        r for r in rows if bool(r.get("counted_in_measurement", True))
    ]
    ok_rows = [
        r for r in measurement_rows if r.get("ok") and not r.get("error")
    ]
    error_rows = [
        r for r in measurement_rows if not (r.get("ok") and not r.get("error"))
    ]
    interrupted_rows = [r for r in ok_rows if r.get("interrupted")]
    posthoc_playback_interrupt_rows = [
        r for r in ok_rows if r.get("posthoc_playback_interrupt")
    ]
    completed_without_interrupt = [r for r in ok_rows if not r.get("interrupted")]
    completion_tokens = [
        int((r.get("usage") or {}).get("completion_tokens") or 0) for r in ok_rows
    ]
    generated_completion_tokens = [
        int(r.get("generated_completion_tokens_total") or 0) for r in ok_rows
    ]
    wasted_completion_tokens = [
        int(r.get("wasted_completion_tokens") or 0) for r in ok_rows
    ]
    prompt_tokens = [
        int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in ok_rows
    ]
    target_input_tokens = [
        float(r["prompt_target_tokens"])
        for r in measurement_rows
        if r.get("prompt_target_tokens") is not None
    ]
    target_output_tokens = [
        float(r["output_target_tokens"])
        for r in measurement_rows
        if r.get("output_target_tokens") is not None
    ]
    audio_duration_s = [float(r.get("audio_duration_s") or 0.0) for r in ok_rows]
    audio_e2e_rtf = [
        (float(r["elapsed_ms"]) / 1000.0) / float(r["audio_duration_s"])
        for r in ok_rows
        if r.get("elapsed_ms") is not None and float(r.get("audio_duration_s") or 0.0) > 0
    ]
    audio_stream_rtf = [
        float(r["audio_stream_rtf"])
        for r in ok_rows
        if r.get("audio_stream_rtf") is not None
    ]
    audio_post_ttfp_rtf = [
        float(r["audio_post_ttfp_rtf"])
        for r in ok_rows
        if r.get("audio_post_ttfp_rtf") is not None
    ]
    continuity_rows = [r for r in ok_rows if r.get("audio_continuity_evaluated")]
    continuity_ok = sum(1 for r in continuity_rows if r.get("audio_continuity_ok") is True)
    continuity_excluded: dict[str, int] = {}
    for row in ok_rows:
        if row.get("audio_continuity_evaluated"):
            continue
        reason = row.get("audio_continuity_excluded_reason") or "not_evaluated"
        continuity_excluded[reason] = continuity_excluded.get(reason, 0) + 1
    interrupt_stages: dict[str, int] = {}
    for row in interrupted_rows:
        stage = row.get("interrupt_stage") or "unknown"
        interrupt_stages[stage] = interrupt_stages.get(stage, 0) + 1
    e2e_ms = stats([float(r["elapsed_ms"]) for r in ok_rows if r.get("elapsed_ms") is not None])
    ttft_ms = stats([float(r["ttft_ms"]) for r in ok_rows if r.get("ttft_ms") is not None])
    audio_ttfp_ms = stats(
        [float(r["audio_ttfp_ms"]) for r in ok_rows if r.get("audio_ttfp_ms") is not None]
    )
    audio_e2e_rtf_stats = stats(audio_e2e_rtf)
    audio_stream_rtf_stats = stats(audio_stream_rtf)
    audio_post_ttfp_rtf_stats = stats(audio_post_ttfp_rtf)
    waste_total = stats([
        float(r["wasted_talker_frames_total"])
        for r in interrupted_rows
        if r.get("wasted_talker_frames_total") is not None
    ])
    waste_inaudible = stats([
        float(r["wasted_talker_frames_inaudible"])
        for r in interrupted_rows
        if r.get("wasted_talker_frames_inaudible") is not None
    ])
    posthoc_waste_total = stats([
        float(r["posthoc_wasted_talker_frames_total"])
        for r in posthoc_playback_interrupt_rows
        if r.get("posthoc_wasted_talker_frames_total") is not None
    ])
    posthoc_audio_unplayed_sec = stats([
        float(r["posthoc_audio_unplayed_sec"])
        for r in posthoc_playback_interrupt_rows
        if r.get("posthoc_audio_unplayed_sec") is not None
    ])
    posthoc_waste_ratio_audio = stats([
        float(r["posthoc_waste_ratio_audio"])
        for r in posthoc_playback_interrupt_rows
        if r.get("posthoc_waste_ratio_audio") is not None
    ])
    underrun_sec = stats([
        float(r["audio_underrun_sec"])
        for r in continuity_rows
        if r.get("audio_underrun_sec") is not None
    ])
    underrun_events = stats([
        float(r["audio_underrun_event_count"])
        for r in continuity_rows
        if r.get("audio_underrun_event_count") is not None
    ])
    by_length_class: dict[str, Any] = {}
    for length_class in sorted(
        {str(r.get("length_class") or "default") for r in measurement_rows}
    ):
        class_rows = [
            r for r in measurement_rows
            if str(r.get("length_class") or "default") == length_class
        ]
        class_ok = [r for r in class_rows if r.get("ok") and not r.get("error")]
        if not class_rows:
            continue
        class_completion_tokens = [
            int((r.get("usage") or {}).get("completion_tokens") or 0)
            for r in class_ok
        ]
        class_audio_duration_s = [
            float(r.get("audio_duration_s") or 0.0) for r in class_ok
        ]
        by_length_class[length_class] = {
            "requests": len(class_rows),
            "completed_ok": len(class_ok),
            "errors": len(class_rows) - len(class_ok),
            "completion_tokens": sum(class_completion_tokens),
            "audio_duration_s": sum(class_audio_duration_s),
            "latency": {
                "e2e_ms": stats([
                    float(r["elapsed_ms"])
                    for r in class_ok
                    if r.get("elapsed_ms") is not None
                ]),
                "ttft_ms": stats([
                    float(r["ttft_ms"])
                    for r in class_ok
                    if r.get("ttft_ms") is not None
                ]),
                "audio_ttfp_ms": stats([
                    float(r["audio_ttfp_ms"])
                    for r in class_ok
                    if r.get("audio_ttfp_ms") is not None
                ]),
            },
            "lengths": {
                "target_input_tokens": stats([
                    float(r["prompt_target_tokens"])
                    for r in class_rows
                    if r.get("prompt_target_tokens") is not None
                ]),
                "target_output_tokens": stats([
                    float(r["output_target_tokens"])
                    for r in class_rows
                    if r.get("output_target_tokens") is not None
                ]),
                "usage_completion_tokens": stats([
                    float((r.get("usage") or {}).get("completion_tokens") or 0)
                    for r in class_ok
                ]),
            },
        }

    return {
        "args": {
            "mode": args.mode,
            "concurrency": args.concurrency,
            "num_requests": args.num_requests,
            "max_tokens": args.max_tokens,
            "max_prompt_chars": args.max_prompt_chars,
            "length_profile": args.length_profile,
            "prompt_source": args.prompt_source,
            "input_tokens_mean": args.input_tokens_mean,
            "input_tokens_std": args.input_tokens_std,
            "output_tokens_mean": args.output_tokens_mean,
            "output_tokens_std": args.output_tokens_std,
            "hetero_long_count": args.hetero_long_count,
            "hetero_short_input_tokens": args.hetero_short_input_tokens,
            "hetero_long_input_tokens": args.hetero_long_input_tokens,
            "hetero_short_output_tokens": args.hetero_short_output_tokens,
            "hetero_long_output_tokens": args.hetero_long_output_tokens,
            "arrival_process": args.arrival_process,
            "request_rate": args.request_rate,
            "arrival_manifest": str(args.arrival_manifest) if args.arrival_manifest else None,
            "arrival_time_scale": args.arrival_time_scale,
            "staged_short_delay_sec": args.staged_short_delay_sec,
            "warmup_sec": args.warmup_sec,
            "run_duration_sec": args.run_duration_sec,
            "cooldown_sec": args.cooldown_sec,
            "max_inflight": args.max_inflight,
            "force_output_len": args.force_output_len,
            "seed": args.seed,
            "url": args.url,
            "model": args.model,
            "dataset": str(args.dataset),
            "tag": args.tag,
            "policy": args.policy,
            "interrupt_prob": args.interrupt_prob,
            "interrupt_after_seconds": args.interrupt_after_seconds,
            "interrupt_jitter_sec": args.interrupt_jitter_sec,
            "interrupt_after_distribution": args.interrupt_after_distribution,
            "interrupt_after_min": args.interrupt_after_min,
            "interrupt_after_max": args.interrupt_after_max,
            "interrupt_after_mean": args.interrupt_after_mean,
            "interrupt_after_samples": (
                str(args.interrupt_after_samples)
                if args.interrupt_after_samples is not None
                else None
            ),
            "interrupt_after_samples_count": len(args.interrupt_after_samples_values),
            "posthoc_playback_interrupt_waste": args.posthoc_playback_interrupt_waste,
            "interrupt_time_anchor": args.interrupt_time_anchor,
            "audio_continuity_threshold_sec": args.audio_continuity_threshold_sec,
            "audio_continuity_chunk_mode": args.audio_continuity_chunk_mode,
            "continuity_include_interrupted": args.continuity_include_interrupted,
        },
        "meta": {
            "mode": args.mode,
            "concurrency": args.concurrency,
            "num_requests": args.num_requests,
            "max_tokens": args.max_tokens,
            "max_prompt_chars": args.max_prompt_chars,
            "length_profile": args.length_profile,
            "prompt_source": args.prompt_source,
            "arrival_process": args.arrival_process,
            "request_rate": args.request_rate,
            "arrival_manifest": str(args.arrival_manifest) if args.arrival_manifest else None,
            "arrival_time_scale": args.arrival_time_scale,
            "staged_short_delay_sec": args.staged_short_delay_sec,
            "warmup_sec": args.warmup_sec,
            "run_duration_sec": args.run_duration_sec,
            "cooldown_sec": args.cooldown_sec,
            "max_inflight": args.max_inflight,
            "seed": args.seed,
            "url": args.url,
            "model": args.model,
            "dataset": str(args.dataset),
            "tag": args.tag,
        },
        "measurement": {
            "rows_total": len(rows),
            "rows_counted": len(measurement_rows),
            "rows_excluded": len(rows) - len(measurement_rows),
            "wall_time_s": wall_s,
            "actual_client_wall_time_s": actual_wall_s,
            "arrival_process": args.arrival_process,
            "warmup_sec": args.warmup_sec,
            "run_duration_sec": args.run_duration_sec,
            "cooldown_sec": args.cooldown_sec,
        },
        "totals": {
            "completed_ok": len(ok_rows),
            "errors": len(error_rows),
            "interrupted": len(interrupted_rows),
            "posthoc_playback_interrupted": len(posthoc_playback_interrupt_rows),
            "completed_without_interrupt": len(completed_without_interrupt),
            "wall_time_sec": wall_s,
            "wall_time_s": wall_s,
            "throughput_rps": (len(ok_rows) / wall_s) if wall_s > 0 else None,
            "useful_qps": (
                len(completed_without_interrupt) / wall_s if wall_s > 0 else None
            ),
            "completion_tokens": sum(completion_tokens),
            "generated_completion_tokens_total": sum(generated_completion_tokens),
            "wasted_completion_tokens_total": sum(wasted_completion_tokens),
            "waste_ratio_tokens": (
                sum(wasted_completion_tokens) / sum(generated_completion_tokens)
                if sum(generated_completion_tokens) > 0 else None
            ),
            "output_token_throughput": (sum(completion_tokens) / wall_s) if wall_s > 0 else None,
            "audio_duration_s": sum(audio_duration_s),
            "audio_duration_throughput": (sum(audio_duration_s) / wall_s) if wall_s > 0 else None,
        },
        "lengths": {
            "target_input_tokens": stats(target_input_tokens),
            "target_output_tokens": stats(target_output_tokens),
            "usage_prompt_tokens": stats([float(x) for x in prompt_tokens]),
            "usage_completion_tokens": stats([float(x) for x in completion_tokens]),
            "by_length_class": by_length_class,
        },
        "latency": {
            "e2e_ms": e2e_ms,
            "ttft_ms": ttft_ms,
            "audio_ttfp_ms": audio_ttfp_ms,
            "audio_rtf": audio_e2e_rtf_stats,
            "ttfp_sec": ms_stats_to_sec(ttft_ms),
            "audio_ttfp_sec": ms_stats_to_sec(audio_ttfp_ms),
            "e2e_sec": ms_stats_to_sec(e2e_ms),
        },
        "interrupt": {
            "stages": interrupt_stages,
            "wasted_talker_frames_total": waste_total,
            "wasted_talker_frames_inaudible": waste_inaudible,
            "posthoc_wasted_talker_frames_total": posthoc_waste_total,
            "posthoc_audio_unplayed_sec": posthoc_audio_unplayed_sec,
            "posthoc_waste_ratio_audio": posthoc_waste_ratio_audio,
            "wasted_tokens_total": sum(wasted_completion_tokens),
            "generated_tokens_total": sum(generated_completion_tokens),
            "waste_ratio_tokens": (
                sum(wasted_completion_tokens) / sum(generated_completion_tokens)
                if sum(generated_completion_tokens) > 0 else None
            ),
            "waste_ratio_tokens_by_request": stats([
                float(r["waste_ratio_tokens"])
                for r in ok_rows
                if r.get("waste_ratio_tokens") is not None
            ]),
        },
        "audio": {
            "audio_duration_sec": stats(audio_duration_s),
            "audio_e2e_rtf": audio_e2e_rtf_stats,
            "audio_post_ttfp_rtf": audio_post_ttfp_rtf_stats,
            "audio_stream_rtf": audio_stream_rtf_stats,
            "n_with_audio": len([
                r for r in ok_rows if r.get("audio_ttfp_ms") is not None
            ]),
            "n_with_stream_rtf": len(audio_stream_rtf),
            "streaming_continuity": {
                "threshold_sec": args.audio_continuity_threshold_sec,
                "chunk_mode": args.audio_continuity_chunk_mode,
                "include_interrupted": args.continuity_include_interrupted,
                "n_total_ok": len(ok_rows),
                "n_evaluated": len(continuity_rows),
                "n_ok": continuity_ok,
                "ok_rate": (
                    continuity_ok / len(continuity_rows)
                    if continuity_rows else None
                ),
                "audio_underrun_sec": underrun_sec,
                "audio_underrun_event_count": underrun_events,
                "goodput_rps": (
                    continuity_ok / wall_s if wall_s > 0 else None
                ),
                "excluded_reasons": continuity_excluded,
            },
        },
        "errors": [
            {"idx": r.get("idx"), "status_code": r.get("status_code"), "error": r.get("error")}
            for r in error_rows[:20]
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mode", choices=("text", "audio"), default="text")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num-requests", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-prompt-chars", type=int, default=500)
    ap.add_argument(
        "--length-profile",
        choices=("sharegpt", *SHAREGPT_LENGTH_PROFILE_DEFAULTS.keys(), HETERO_PROFILE),
        default="sharegpt",
        help="Preset synthetic length profile. shareGPT-len* uses normal "
             "input/output targets with means 512/<suffix> tokens; "
             "shareGPT-hetero-longshort sends a long-prefix cohort first, "
             "then short requests to expose talker slot blocking.",
    )
    ap.add_argument(
        "--prompt-source",
        choices=("dataset", "synthetic"),
        default="dataset",
        help="Use ShareGPT prompts from --dataset or synthetic prompts with "
             "controlled approximate token counts.",
    )
    ap.add_argument("--input-tokens-mean", type=int, default=None)
    ap.add_argument("--input-tokens-std", type=int, default=128)
    ap.add_argument("--output-tokens-mean", type=int, default=None)
    ap.add_argument("--output-tokens-std", type=int, default=256)
    ap.add_argument("--hetero-long-count", type=int, default=4)
    ap.add_argument("--hetero-short-input-tokens", type=int, default=96)
    ap.add_argument("--hetero-long-input-tokens", type=int, default=512)
    ap.add_argument("--hetero-short-output-tokens", type=int, default=64)
    ap.add_argument("--hetero-long-output-tokens", type=int, default=448)
    ap.add_argument("--min-input-tokens", type=int, default=1)
    ap.add_argument("--max-input-tokens", type=int, default=0,
                    help="0 means no upper clamp.")
    ap.add_argument("--min-output-tokens", type=int, default=1)
    ap.add_argument("--max-output-tokens", type=int, default=0,
                    help="0 means no upper clamp.")
    ap.add_argument("--force-output-len", action="store_true",
                    help="Set min_tokens=max_tokens and ignore_eos=true so "
                         "the sampled output target is reached more closely.")
    ap.add_argument(
        "--arrival-process",
        choices=("closed_loop", "poisson", "staged", "manifest", "open_c_bound"),
        default="closed_loop",
        help="closed_loop keeps the historical fixed-concurrency behavior; "
             "poisson issues requests with exponential inter-arrival times; "
             "staged sends the long-prefix cohort first, then the rest after "
             "--staged-short-delay-sec; manifest replays t_sec/submit_offset_s "
             "timestamps from --arrival-manifest; open_c_bound keeps up to "
             "--concurrency requests in flight for a warmup plus measured "
             "duration and reports throughput over --run-duration-sec.",
    )
    ap.add_argument("--request-rate", type=float, default=0.0,
                    help="Poisson arrival rate in requests/sec.")
    ap.add_argument("--arrival-manifest", type=Path, default=None,
                    help="JSONL manifest containing t_sec/submit_offset_s/arrival_time_s offsets.")
    ap.add_argument("--arrival-time-scale", type=float, default=1.0,
                    help="Scale manifest offsets before replaying; values <1 speed up arrivals.")
    ap.add_argument("--staged-short-delay-sec", type=float, default=1.0,
                    help="Delay before submitting non-prefix requests in "
                         "--arrival-process=staged.")
    ap.add_argument("--warmup-sec", type=float, default=0.0,
                    help="Warmup duration for --arrival-process=open_c_bound.")
    ap.add_argument("--run-duration-sec", type=float, default=0.0,
                    help="Measured duration for --arrival-process=open_c_bound.")
    ap.add_argument("--cooldown-sec", type=float, default=0.0,
                    help="Cooldown wait budget recorded for open_c_bound runs.")
    ap.add_argument("--max-inflight", type=int, default=0,
                    help="Worker cap for poisson mode. Defaults to --concurrency.")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="sharegpt")
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--policy", choices=("none", "fixed_time"), default="none")
    ap.add_argument("--barge-in-prob", "--interrupt-prob",
                    dest="interrupt_prob", type=float, default=0.0)
    ap.add_argument("--barge-in-after-seconds", "--interrupt-after-seconds",
                    dest="interrupt_after_seconds", type=float, default=2.0)
    ap.add_argument("--barge-in-jitter-sec", "--interrupt-jitter-sec",
                    dest="interrupt_jitter_sec", type=float, default=0.5)
    ap.add_argument(
        "--barge-in-after-distribution",
        "--interrupt-after-distribution",
        dest="interrupt_after_distribution",
        choices=("fixed", "uniform", "truncexp", "samples"),
        default="fixed",
        help="Distribution for fixed_time thresholds. samples reads values from "
             "--barge-in-after-samples and is used for paper-style duration sampling.",
    )
    ap.add_argument("--barge-in-after-min", "--interrupt-after-min",
                    dest="interrupt_after_min", type=float, default=0.3)
    ap.add_argument("--barge-in-after-max", "--interrupt-after-max",
                    dest="interrupt_after_max", type=float, default=5.0)
    ap.add_argument("--barge-in-after-mean", "--interrupt-after-mean",
                    dest="interrupt_after_mean", type=float, default=1.5)
    ap.add_argument("--barge-in-after-samples", "--interrupt-after-samples",
                    dest="interrupt_after_samples", type=Path, default=None,
                    help="JSONL/plain file of post-TTFP barge-in delays in seconds.")
    ap.add_argument("--barge-in-after-sample-key", "--interrupt-after-sample-key",
                    dest="interrupt_after_sample_key", default="audio_duration_s",
                    help="Dot key to read from JSONL sample rows.")
    ap.add_argument("--barge-in-after-sample-time-scale",
                    "--interrupt-after-sample-time-scale",
                    dest="interrupt_after_sample_time_scale",
                    type=float, default=1.0,
                    help="Scale sampled barge-in delays; values <1 cut earlier.")
    ap.add_argument("--posthoc-playback-barge-in-waste",
                    "--posthoc-playback-interrupt-waste",
                    dest="posthoc_playback_interrupt_waste",
                    action="store_true",
                    help="Record generated-but-unplayed audio implied by a "
                         "paper-style playback cut that occurs after the HTTP "
                         "stream has already completed. This does not mark the "
                         "request as a live barge-in.")
    ap.add_argument("--barge-in-time-anchor", "--interrupt-time-anchor",
                    dest="interrupt_time_anchor",
                    choices=("t0", "ttfp"), default="ttfp")
    ap.add_argument("--audio-continuity-threshold-sec", type=float, default=0.1)
    ap.add_argument("--audio-continuity-chunk-mode", default="fragment")
    ap.add_argument("--continuity-include-interrupted", action="store_true")
    args = ap.parse_args()
    resolve_length_profile(args)
    if args.arrival_process == "poisson" and args.request_rate <= 0:
        print("--arrival-process=poisson requires --request-rate > 0", file=sys.stderr)
        return 2
    if args.arrival_process == "open_c_bound":
        if args.run_duration_sec <= 0:
            print("--arrival-process=open_c_bound requires --run-duration-sec > 0",
                  file=sys.stderr)
            return 2
        if args.warmup_sec < 0 or args.cooldown_sec < 0:
            print("--warmup-sec and --cooldown-sec must be >= 0", file=sys.stderr)
            return 2
    if args.arrival_process == "manifest":
        if args.arrival_manifest is None:
            print("--arrival-process=manifest requires --arrival-manifest", file=sys.stderr)
            return 2
        if args.arrival_time_scale <= 0:
            print("--arrival-time-scale must be > 0", file=sys.stderr)
            return 2
    if args.max_inflight <= 0:
        args.max_inflight = args.concurrency
    args.interrupt_after_samples_values = ()
    if args.interrupt_after_distribution == "samples":
        if args.interrupt_after_samples is None:
            print("--barge-in-after-distribution=samples requires "
                  "--barge-in-after-samples", file=sys.stderr)
            return 2
        if args.interrupt_after_sample_time_scale <= 0:
            print("--barge-in-after-sample-time-scale must be > 0", file=sys.stderr)
            return 2
        try:
            args.interrupt_after_samples_values = load_interrupt_after_samples(
                args.interrupt_after_samples,
                args.interrupt_after_sample_key,
                args.interrupt_after_sample_time_scale,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    try:
        work_items = build_work_items(args)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    args.result_dir.mkdir(parents=True, exist_ok=True)
    meta_args = vars(args).copy()
    meta_args.update({
        "dataset": str(args.dataset),
        "result_dir": str(args.result_dir),
        "arrival_manifest": str(args.arrival_manifest) if args.arrival_manifest else None,
        "barge_in_prob": args.interrupt_prob,
        "barge_in_after_seconds": args.interrupt_after_seconds,
        "barge_in_jitter_sec": args.interrupt_jitter_sec,
        "barge_in_after_distribution": args.interrupt_after_distribution,
        "barge_in_after_samples": (
            str(args.interrupt_after_samples)
            if args.interrupt_after_samples is not None
            else None
        ),
        "barge_in_after_samples_count": len(args.interrupt_after_samples_values),
        "barge_in_time_anchor": args.interrupt_time_anchor,
        "posthoc_playback_barge_in_waste": args.posthoc_playback_interrupt_waste,
        "interrupt_after_samples": (
            str(args.interrupt_after_samples)
            if args.interrupt_after_samples is not None
            else None
        ),
        "interrupt_after_samples_count": len(args.interrupt_after_samples_values),
    })
    meta_args.pop("interrupt_after_samples_values", None)
    (args.result_dir / "meta.json").write_text(
        json.dumps(meta_args, indent=2),
        encoding="utf-8",
    )

    print(
        f"[sharegpt_driver] tag={args.tag} mode={args.mode} "
        f"c={args.concurrency} n={args.num_requests} max_tokens={args.max_tokens} "
        f"profile={args.length_profile} arrival={args.arrival_process}"
    )

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    done_count = 0
    partial_path = args.result_dir / "partial_per_request.jsonl"
    partial_path.write_text("", encoding="utf-8")

    def counted_in_measurement(row: dict[str, Any]) -> bool:
        if args.arrival_process != "open_c_bound":
            return True
        try:
            submit_offset = float(row.get("client_submit_offset_s"))
        except (TypeError, ValueError):
            return False
        measure_start = float(args.warmup_sec)
        measure_end = measure_start + float(args.run_duration_sec)
        return measure_start <= submit_offset < measure_end

    def collect(fut: futures.Future[dict[str, Any]]) -> None:
        nonlocal done_count
        row = fut.result()
        row["client_collect_offset_s"] = time.time() - t0
        row["counted_in_measurement"] = counted_in_measurement(row)
        rows.append(row)
        with partial_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        done_count += 1
        status = "ok" if row.get("ok") and not row.get("error") else "ERR"
        elapsed = float(row.get("elapsed_ms") or 0.0) / 1000.0
        print(
            f"  [{done_count}/{len(work_items)}] "
            f"idx={row.get('idx')} {status} elapsed={elapsed:.2f}s"
        )

    if args.arrival_process == "closed_loop":
        with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            pending = [
                pool.submit(send_one, i, item, args, time.time() - t0)
                for i, item in enumerate(work_items)
            ]
            for fut in futures.as_completed(pending):
                collect(fut)
    elif args.arrival_process == "poisson":
        arrival_rng = random.Random(args.seed + 17)
        with futures.ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
            pending: set[futures.Future[dict[str, Any]]] = set()
            for i, item in enumerate(work_items):
                if i > 0:
                    time.sleep(arrival_rng.expovariate(args.request_rate))
                pending.add(pool.submit(send_one, i, item, args, time.time() - t0))
                ready = {fut for fut in pending if fut.done()}
                for fut in ready:
                    pending.remove(fut)
                    collect(fut)
            for fut in futures.as_completed(pending):
                collect(fut)
    elif args.arrival_process == "manifest":
        assert args.arrival_manifest is not None
        offsets = load_arrival_offsets(
            args.arrival_manifest,
            len(work_items),
            args.arrival_time_scale,
        )
        with futures.ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
            pending: set[futures.Future[dict[str, Any]]] = set()
            for i, item in enumerate(work_items):
                target = offsets[i]
                now_offset = time.time() - t0
                if target > now_offset:
                    time.sleep(target - now_offset)
                pending.add(pool.submit(send_one, i, item, args, time.time() - t0))
                ready = {fut for fut in pending if fut.done()}
                for fut in ready:
                    pending.remove(fut)
                    collect(fut)
            for fut in futures.as_completed(pending):
                collect(fut)
    elif args.arrival_process == "open_c_bound":
        submit_until_s = float(args.warmup_sec) + float(args.run_duration_sec)
        with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            pending: set[futures.Future[dict[str, Any]]] = set()
            next_idx = 0

            def submit_next() -> None:
                nonlocal next_idx
                item = work_items[next_idx % len(work_items)]
                pending.add(
                    pool.submit(send_one, next_idx, item, args, time.time() - t0)
                )
                next_idx += 1

            while len(pending) < args.concurrency:
                submit_next()

            while pending:
                done, _ = futures.wait(
                    pending,
                    timeout=0.1,
                    return_when=futures.FIRST_COMPLETED,
                )
                for fut in done:
                    pending.remove(fut)
                    collect(fut)

                now_offset = time.time() - t0
                while now_offset < submit_until_s and len(pending) < args.concurrency:
                    submit_next()
                    now_offset = time.time() - t0
    else:
        prefix_count = min(max(0, args.hetero_long_count), len(work_items))
        with futures.ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
            pending: set[futures.Future[dict[str, Any]]] = set()
            for i, item in enumerate(work_items[:prefix_count]):
                pending.add(pool.submit(send_one, i, item, args, time.time() - t0))
            if args.staged_short_delay_sec > 0:
                time.sleep(args.staged_short_delay_sec)
            for i, item in enumerate(work_items[prefix_count:], start=prefix_count):
                pending.add(pool.submit(send_one, i, item, args, time.time() - t0))
            for fut in futures.as_completed(pending):
                collect(fut)

    wall_s = time.time() - t0
    rows.sort(key=lambda r: int(r["idx"]))
    with (args.result_dir / "per_request.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_wall_s = (
        float(args.run_duration_sec)
        if args.arrival_process == "open_c_bound"
        else wall_s
    )
    summary = summarize(rows, summary_wall_s, args, actual_wall_s=wall_s)
    (args.result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    t = summary["totals"]
    lat = summary["latency"]
    print(
        "[sharegpt_driver] done "
        f"ok={t['completed_ok']} err={t['errors']} barge_in={t['interrupted']} "
        f"wall={t['wall_time_s']:.2f}s "
        f"rps={t['throughput_rps']:.3f} tok/s={t['output_token_throughput']:.2f}"
    )
    print(
        "  e2e p50/p90="
        f"{fmt_ms(lat['e2e_ms']['p50'])}/{fmt_ms(lat['e2e_ms']['p90'])} ms; "
        "ttft p50/p90="
        f"{fmt_ms(lat['ttft_ms']['p50'])}/{fmt_ms(lat['ttft_ms']['p90'])} ms"
    )
    if args.mode == "audio":
        audio = summary["audio"]
        continuity = audio["streaming_continuity"]
        print(
            "  audio_ttfp p50/p90="
            f"{fmt_ms(lat['audio_ttfp_ms']['p50'])}/{fmt_ms(lat['audio_ttfp_ms']['p90'])} ms; "
            f"audio_duration/s={t['audio_duration_throughput']:.3f}"
        )
        print(
            "  audio_stream_rtf p50/p90="
            f"{fmt_float(audio['audio_stream_rtf']['p50'])}/"
            f"{fmt_float(audio['audio_stream_rtf']['p90'])}; "
            f"continuity={continuity['ok_rate']}"
        )
    return 0 if t["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
