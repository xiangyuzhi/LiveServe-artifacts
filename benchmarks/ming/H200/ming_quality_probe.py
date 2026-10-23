#!/usr/bin/env python3
"""Ming streaming quality probe.

This probe is intentionally small and conservative. It sends fixed prompts,
records the full streamed text, routes audio chunks even when they are carried
in ``delta.content`` with ``modality=audio``, and reports obvious corruption
signals such as audio base64 leaking into text, replacement characters, control
characters, or empty/very short answers.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import json
import statistics
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp


DEFAULT_URL = "http://127.0.0.1:8098/v1/chat/completions"
DEFAULT_MODEL = "/workspace/models/Jonathan1909-Ming-flash-omni-2.0"
SYSTEM_PROMPT = "你是一个友好的AI助手。\n\ndetailed thinking off"

PROMPTS: list[tuple[str, str]] = [
    ("杭州西湖", "请用四句自然的中文介绍杭州西湖，每句不超过二十个字。"),
    ("龙井茶", "请用四句清晰的中文介绍龙井茶，避免编号，语气适合语音播报。"),
    ("良渚文化", "请用四句中文介绍良渚文化，包含年代、地点和代表性文物。"),
    ("京杭大运河", "请用四句口语化中文介绍京杭大运河，句子要短。"),
    ("深圳科技产业", "请用四句中文介绍深圳科技产业，保持事实清楚。"),
    ("黄山云海", "请用四句中文描写黄山云海，避免夸张玄幻。"),
    ("成都火锅", "请用四句中文介绍成都火锅，兼顾风味和饮食文化。"),
    ("西安城墙", "请用四句中文介绍西安城墙，包含历史和游览信息。"),
]

TOPIC_ALIASES: dict[str, tuple[str, ...]] = {
    "深圳科技产业": ("深圳科技产业", "深圳的科技产业", "深圳是中国科技创新"),
}


@dataclass
class ProbeResult:
    request_index: int
    topic: str
    prompt: str
    status: int = 0
    error: str | None = None
    first_text_sec: float | None = None
    first_audio_sec: float | None = None
    e2e_sec: float | None = None
    events: int = 0
    text_events: int = 0
    audio_events: int = 0
    audio_bytes: int = 0
    audio_from_content_events: int = 0
    unparsed_audio_events: int = 0
    full_text: str = ""
    flags: dict[str, int | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["text_chars"] = len(self.full_text)
        data["text_preview"] = self.full_text[:240]
        return data


def round_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v) for v in obj]
    return obj


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def metric_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "max": max(values) if values else None,
    }


def decode_audio_candidate(value: Any, *, force_audio: bool = False) -> bytes:
    if not isinstance(value, str):
        return b""
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception:
        return b""
    if force_audio and raw:
        return raw
    if raw.startswith((b"RIFF", b"fLaC", b"OggS")) or len(raw) > 4096:
        return raw
    return b""


def build_payload(prompt: str, args: argparse.Namespace, index: int) -> dict[str, Any]:
    stage0: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "repetition_penalty": 1.05,
        "detokenize": True,
        "ignore_eos": False,
    }
    if args.seed is not None:
        stage0["seed"] = args.seed + index

    return {
        "model": args.model,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": prompt},
        ],
        "modalities": ["text", "audio"],
        "stream": True,
        "sampling_params_list": [
            stage0,
            {"temperature": 0.0, "max_tokens": 1},
        ],
    }


def analyze_text(topic: str, text: str) -> dict[str, int | bool]:
    replacement_chars = text.count("\ufffd")
    control_chars = sum(
        1
        for ch in text
        if unicodedata.category(ch)[0] == "C" and ch not in "\n\t\r"
    )
    audio_markers = sum(text.count(marker) for marker in ("UklGR", "RIFF", "data:audio", "<audio"))
    special_tokens = sum(text.count(marker) for marker in ("<|", "<tool", "</s>", "[ERROR]"))
    cjk_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    aliases = TOPIC_ALIASES.get(topic, (topic,))
    topic_present = any(alias in text for alias in aliases)
    return {
        "replacement_chars": replacement_chars,
        "control_chars": control_chars,
        "audio_markers_in_text": audio_markers,
        "special_tokens": special_tokens,
        "topic_missing": not topic_present,
        "too_short": cjk_chars < 20,
        "empty_text": not text.strip(),
    }


async def run_one(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    index: int,
    topic: str,
    prompt: str,
    args: argparse.Namespace,
) -> ProbeResult:
    result = ProbeResult(request_index=index, topic=topic, prompt=prompt)
    async with sem:
        t0 = time.perf_counter()
        try:
            async with session.post(args.url, json=build_payload(prompt, args, index)) as resp:
                result.status = resp.status
                if resp.status >= 400:
                    result.error = await resp.text()
                    result.e2e_sec = time.perf_counter() - t0
                    result.flags = analyze_text(topic, result.full_text)
                    return result

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    result.events += 1
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    modality = obj.get("modality")
                    choice = (obj.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    modality = delta.get("modality") or modality

                    audio_data = None
                    audio_obj = delta.get("audio")
                    if isinstance(audio_obj, dict):
                        audio_data = audio_obj.get("data")
                    elif isinstance(audio_obj, str):
                        audio_data = audio_obj

                    content = delta.get("content")
                    force_audio = modality == "audio" or audio_data is not None
                    raw_audio = decode_audio_candidate(audio_data, force_audio=force_audio)
                    audio_from_content = False
                    if not raw_audio:
                        raw_audio = decode_audio_candidate(content, force_audio=force_audio)
                        audio_from_content = bool(raw_audio)

                    if raw_audio:
                        result.audio_events += 1
                        result.audio_bytes += len(raw_audio)
                        if audio_from_content:
                            result.audio_from_content_events += 1
                        if result.first_audio_sec is None:
                            result.first_audio_sec = time.perf_counter() - t0
                        continue

                    if modality == "audio":
                        result.unparsed_audio_events += 1
                        continue

                    if isinstance(content, str) and content:
                        result.text_events += 1
                        result.full_text += content
                        if result.first_text_sec is None:
                            result.first_text_sec = time.perf_counter() - t0

                result.e2e_sec = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
            result.e2e_sec = time.perf_counter() - t0
        result.flags = analyze_text(topic, result.full_text)
        return result


async def run_probe(args: argparse.Namespace) -> list[ProbeResult]:
    prompts = PROMPTS[: args.num_requests]
    timeout = aiohttp.ClientTimeout(
        total=args.request_timeout,
        connect=args.connect_timeout,
        sock_connect=args.connect_timeout,
    )
    connector = aiohttp.TCPConnector(limit=max(args.concurrency * 2, 16))
    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        read_bufsize=16 * 1024 * 1024,
        max_line_size=16 * 1024 * 1024,
    ) as session:
        tasks = [
            asyncio.create_task(run_one(session, sem, i, topic, prompt, args))
            for i, (topic, prompt) in enumerate(prompts)
        ]
        results: list[ProbeResult] = []
        for fut in asyncio.as_completed(tasks):
            result = await fut
            results.append(result)
            ok = sum(1 for r in results if r.status == 200 and not r.error)
            print(f"progress {len(results)}/{len(prompts)} ok={ok}", flush=True)
        results.sort(key=lambda r: r.request_index)
        return results


def build_summary(args: argparse.Namespace, results: list[ProbeResult], elapsed: float) -> dict[str, Any]:
    ok = [r for r in results if r.status == 200 and not r.error]
    failed = [r for r in results if r not in ok]

    def values(name: str) -> list[float]:
        vals: list[float] = []
        for result in ok:
            val = getattr(result, name)
            if isinstance(val, (int, float)):
                vals.append(float(val))
        return vals

    flag_names = [
        "replacement_chars",
        "control_chars",
        "audio_markers_in_text",
        "special_tokens",
        "topic_missing",
        "too_short",
        "empty_text",
    ]
    flag_totals = {
        name: sum(int(r.flags.get(name, 0) or 0) for r in ok)
        for name in flag_names
    }

    return round_floats(
        {
            "label": args.label,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "url": args.url,
            "model": args.model,
            "num_requests": len(results),
            "concurrency": args.concurrency,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "ok": len(ok),
            "failed": len(failed),
            "elapsed_sec": elapsed,
            "throughput_rps": len(ok) / elapsed if elapsed > 0 else None,
            "first_text_sec": metric_stats(values("first_text_sec")),
            "first_audio_sec": metric_stats(values("first_audio_sec")),
            "e2e_sec": metric_stats(values("e2e_sec")),
            "text_chars": metric_stats([len(r.full_text) for r in ok]),
            "audio_events": metric_stats(values("audio_events")),
            "audio_bytes": metric_stats(values("audio_bytes")),
            "audio_from_content_events_total": sum(r.audio_from_content_events for r in ok),
            "unparsed_audio_events_total": sum(r.unparsed_audio_events for r in ok),
            "flag_totals": flag_totals,
            "errors": [
                {
                    "request_index": r.request_index,
                    "status": r.status,
                    "error": (r.error or "")[:400],
                }
                for r in failed[:10]
            ],
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--label", default="ming_quality_probe")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=len(PROMPTS))
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=420.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    results = asyncio.run(run_probe(args))
    elapsed = time.perf_counter() - t0
    summary = build_summary(args, results, elapsed)

    jsonl_path = args.output_dir / f"{args.label}.jsonl"
    summary_path = args.output_dir / f"{args.label}.summary.json"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(round_floats(result.to_dict()), ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
