#!/usr/bin/env python3
"""Ming-flash-omni streaming benchmark for H200 experiments.

The existing Qwen3-Omni drivers use a three-stage sampling_params_list. Ming's
omni speech pipeline is two-stage, so this driver keeps the request shape close
to the Ming recipe and records first-text / first-audio streaming metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import json
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp


DEFAULT_URL = "http://127.0.0.1:8098/v1/chat/completions"
DEFAULT_MODEL = "/workspace/models/Jonathan1909-Ming-flash-omni-2.0"
SYSTEM_PROMPT = "你是一个友好的AI助手。\n\ndetailed thinking off"

PROMPT_TOPICS = [
    "杭州西湖",
    "龙井茶",
    "良渚文化",
    "京杭大运河",
    "西溪湿地",
    "中国高铁",
    "苏州园林",
    "上海外滩",
    "敦煌莫高窟",
    "成都火锅",
    "广州早茶",
    "深圳科技产业",
    "北京中轴线",
    "黄山云海",
    "桂林山水",
    "西安城墙",
]


@dataclass
class RequestResult:
    request_index: int
    prompt: str
    status: int = 0
    error: str | None = None
    issued_at: float = 0.0
    connect_sec: float | None = None
    first_text_sec: float | None = None
    first_audio_sec: float | None = None
    e2e_sec: float | None = None
    events: int = 0
    text_chars: int = 0
    text_preview: str = ""
    audio_chunks: int = 0
    audio_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


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
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def round_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v) for v in obj]
    return obj


def decode_audio_candidate(value: Any) -> bytes:
    if not isinstance(value, str):
        return b""
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception:
        return b""
    if raw.startswith((b"RIFF", b"fLaC", b"OggS")) or len(raw) > 4096:
        return raw
    return b""


def build_prompt(index: int, rng: random.Random, target_sentences: int) -> str:
    topic = PROMPT_TOPICS[index % len(PROMPT_TOPICS)]
    style = rng.choice(["清晰", "自然", "口语化", "适合播客开场", "适合语音助手"])
    return (
        f"请用{target_sentences}句{style}的中文介绍{topic}，"
        "每句保持简短但信息完整。"
    )


def build_payload(prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {"role": "user", "content": prompt},
        ],
        "modalities": ["text", "audio"],
        "stream": True,
        "sampling_params_list": [
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "repetition_penalty": 1.05,
                "detokenize": True,
                "ignore_eos": False,
            },
            {"temperature": 0.0, "max_tokens": 1},
        ],
    }


async def run_one(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    index: int,
    prompt: str,
    args: argparse.Namespace,
) -> RequestResult:
    result = RequestResult(request_index=index, prompt=prompt)
    async with sem:
        t0 = time.perf_counter()
        result.issued_at = time.time()
        try:
            async with session.post(args.url, json=build_payload(prompt, args)) as resp:
                result.status = resp.status
                result.connect_sec = time.perf_counter() - t0
                if resp.status >= 400:
                    result.error = await resp.text()
                    result.e2e_sec = time.perf_counter() - t0
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
                    choice = (obj.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}

                    audio_data = None
                    audio_obj = delta.get("audio")
                    if isinstance(audio_obj, dict):
                        audio_data = audio_obj.get("data")
                    elif isinstance(audio_obj, str):
                        audio_data = audio_obj

                    content = delta.get("content")
                    raw_audio = decode_audio_candidate(audio_data)
                    if not raw_audio:
                        raw_audio = decode_audio_candidate(content)
                    if raw_audio:
                        result.audio_chunks += 1
                        result.audio_bytes += len(raw_audio)
                        if result.first_audio_sec is None:
                            result.first_audio_sec = time.perf_counter() - t0
                        continue

                    if isinstance(content, str) and content:
                        result.text_chars += len(content)
                        if len(result.text_preview) < 160:
                            result.text_preview = (result.text_preview + content)[:160]
                        if result.first_text_sec is None:
                            result.first_text_sec = time.perf_counter() - t0

                result.e2e_sec = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001 - benchmark should record all failures.
            result.error = f"{type(exc).__name__}: {exc}"
            result.e2e_sec = time.perf_counter() - t0
        return result


async def run_benchmark(args: argparse.Namespace) -> list[RequestResult]:
    rng = random.Random(args.seed)
    prompts = [
        build_prompt(i, rng, args.target_sentences)
        for i in range(args.num_requests)
    ]
    timeout = aiohttp.ClientTimeout(
        total=args.request_timeout,
        connect=args.connect_timeout,
        sock_connect=args.connect_timeout,
    )
    connector = aiohttp.TCPConnector(limit=max(args.concurrency * 2, 32))
    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        read_bufsize=16 * 1024 * 1024,
        max_line_size=16 * 1024 * 1024,
    ) as session:
        tasks = [
            asyncio.create_task(run_one(session, sem, i, prompt, args))
            for i, prompt in enumerate(prompts)
        ]
        results: list[RequestResult] = []
        for fut in asyncio.as_completed(tasks):
            result = await fut
            results.append(result)
            if len(results) % args.progress_every == 0 or len(results) == args.num_requests:
                ok = sum(1 for r in results if r.status == 200 and not r.error)
                print(f"progress {len(results)}/{args.num_requests} ok={ok}", flush=True)
        results.sort(key=lambda r: r.request_index)
        return results


def build_summary(args: argparse.Namespace, results: list[RequestResult], elapsed: float) -> dict[str, Any]:
    ok_results = [r for r in results if r.status == 200 and not r.error]
    failed = [r for r in results if r not in ok_results]

    def values(name: str) -> list[float]:
        vals: list[float] = []
        for result in ok_results:
            val = getattr(result, name)
            if isinstance(val, (int, float)):
                vals.append(float(val))
        return vals

    summary = {
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "model": args.model,
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "target_sentences": args.target_sentences,
        "ok": len(ok_results),
        "failed": len(failed),
        "elapsed_sec": elapsed,
        "throughput_rps": len(ok_results) / elapsed if elapsed > 0 else None,
        "first_text_sec": metric_stats(values("first_text_sec")),
        "first_audio_sec": metric_stats(values("first_audio_sec")),
        "e2e_sec": metric_stats(values("e2e_sec")),
        "events": metric_stats(values("events")),
        "text_chars": metric_stats(values("text_chars")),
        "audio_chunks": metric_stats(values("audio_chunks")),
        "audio_bytes": metric_stats(values("audio_bytes")),
        "errors": [
            {
                "request_index": r.request_index,
                "status": r.status,
                "error": (r.error or "")[:400],
            }
            for r in failed[:10]
        ],
    }
    return round_floats(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--label", default="ming_stream")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--target-sentences", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--progress-every", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    results = asyncio.run(run_benchmark(args))
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
