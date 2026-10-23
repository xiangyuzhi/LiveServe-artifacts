#!/usr/bin/env python3
"""Ming-flash-omni c-bound workload driver.

This is the Ming-side adapter for the Qwen c-bound reproduction scripts.  It
keeps the output layout and summary keys that the existing Qwen plotters expect,
but sends Ming's two-stage text+audio request shape and parses Ming's streaming
audio variants.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp


DEFAULT_URL = "http://127.0.0.1:8098/v1/chat/completions"
DEFAULT_MODEL = "/workspace/models/Jonathan1909-Ming-flash-omni-2.0"
DEFAULT_TRACE = (
    Path(__file__).resolve().parents[2]
    / "interactive"
    / "H200"
    / "trace"
    / "total_traces_part1.txt"
)
DEFAULT_SHAREGPT_DATASET = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "ShareGPT_Chinese_English_90k.json"
)
SYSTEM_PROMPT = "你是一个友好的AI助手。\n\ndetailed thinking off"

TOPICS = [
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
]

SYNTHETIC_WORDS = [
    "城市",
    "历史",
    "文化",
    "科技",
    "风景",
    "交通",
    "体验",
    "传统",
    "现代",
    "社区",
    "服务",
    "旅行",
]


@dataclass
class WorkItem:
    request_index: int
    request_id: str
    workload: str
    kind: str
    prompt: str
    messages: list[dict[str, Any]]
    max_tokens: int
    scheduled_offset_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestResult:
    request_index: int
    request_id: str
    workload: str
    kind: str
    prompt: str
    scheduled_offset_sec: float
    status: int = 0
    error: str | None = None
    issued_at: float = 0.0
    queue_delay_sec: float | None = None
    connect_sec: float | None = None
    first_text_sec: float | None = None
    first_audio_sec: float | None = None
    e2e_sec: float | None = None
    events: int = 0
    text_chars: int = 0
    text_preview: str = ""
    audio_chunks: int = 0
    audio_bytes: int = 0
    stopped_after_first_audio: bool = False
    usage: dict[str, Any] | None = None
    talker_frames_generated: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def percentile(values: list[float], q: float) -> float | None:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(clean) - 1)
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def metric_stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean) if clean else None,
        "p50": percentile(clean, 50),
        "p90": percentile(clean, 90),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "max": max(clean) if clean else None,
    }


def scale_stats(stats: dict[str, float | int | None], factor: float) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {}
    for key, value in stats.items():
        if key == "count" or value is None:
            out[key] = value
        else:
            out[key] = float(value) * factor
    return out


def round_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v) for v in obj]
    return obj


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


def synthetic_text(token_budget: int, salt: str, prefix: str) -> str:
    n = max(4, min(int(token_budget), 256))
    seed = sum(ord(ch) for ch in salt) % len(SYNTHETIC_WORDS)
    words = [SYNTHETIC_WORDS[(seed + i) % len(SYNTHETIC_WORDS)] for i in range(n)]
    return f"{prefix}{' '.join(words)}。"


def text_hash(messages: list[dict[str, Any]]) -> str:
    blob = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def to_ming_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            out.append(
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": content}],
                }
            )
        else:
            out.append(message)
    return out


def base_messages(prompt: str, history: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        *(history or []),
        {"role": "user", "content": prompt},
    ]


def build_payload(item: WorkItem, args: argparse.Namespace) -> dict[str, Any]:
    stage0: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": max(1, int(item.max_tokens)),
        "repetition_penalty": 1.05,
        "detokenize": True,
        "ignore_eos": False,
    }
    if args.seed is not None:
        stage0["seed"] = int(args.seed) + int(item.request_index)
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": to_ming_messages(item.messages),
        "modalities": ["text", "audio"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "request_id": item.request_id,
        "sampling_params_list": [
            stage0,
            {"temperature": 0.0, "max_tokens": 1},
        ],
    }
    payload.update(item.metadata)
    return payload


def load_sharegpt_prompts(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    prompts: list[str] = []
    for row in data if isinstance(data, list) else []:
        conversations = row.get("conversations") or row.get("messages") or []
        for turn in conversations:
            role = turn.get("from") or turn.get("role")
            if role in {"human", "user"}:
                text = str(turn.get("value") or turn.get("content") or "").strip()
                if text:
                    prompts.append(text[:1200])
                    break
        if len(prompts) >= limit:
            break
    return prompts


def build_sharegpt_items(args: argparse.Namespace) -> list[WorkItem]:
    rng = random.Random(args.seed)
    prompts = load_sharegpt_prompts(args.dataset, args.num_requests)
    while len(prompts) < args.num_requests:
        idx = len(prompts)
        topic = TOPICS[idx % len(TOPICS)]
        style = rng.choice(["清晰", "自然", "口语化", "适合语音助手"])
        prompts.append(f"请用{args.target_sentences}句{style}的中文介绍{topic}。")

    items: list[WorkItem] = []
    for idx, prompt in enumerate(prompts[: args.num_requests]):
        request_id = f"ming-sharegpt-{args.run_id}-{idx}"
        items.append(
            WorkItem(
                request_index=idx,
                request_id=request_id,
                workload="sharegpt",
                kind="text",
                prompt=prompt,
                messages=base_messages(prompt),
                max_tokens=args.max_tokens,
                metadata={"omni_workload": "ming_sharegpt_origin"},
            )
        )
    return items


def load_trace_rows(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        _ = header
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                user_id = parts[0]
                ts = int(float(parts[1]))
                query_len = int(float(parts[2]))
                response_len = int(float(parts[3]))
                round_index = int(float(parts[4]))
            except ValueError:
                continue
            if args.time_start is not None and ts < args.time_start:
                continue
            if args.time_limit is not None and ts >= args.time_limit:
                continue
            rows.append(
                {
                    "user_id": user_id,
                    "timestamp": ts,
                    "query_length": query_len,
                    "response_length": response_len,
                    "round_index": round_index,
                }
            )
            if len(rows) >= max(args.num_requests * 4, args.num_requests):
                break
    rows.sort(key=lambda row: (row["timestamp"], row["user_id"], row["round_index"]))
    return rows[: args.num_requests]


def build_interactive_items(args: argparse.Namespace) -> list[WorkItem]:
    rows = load_trace_rows(args.trace_path, args)
    if not rows:
        rows = [
            {
                "user_id": f"synthetic-{idx // 2}",
                "timestamp": idx,
                "query_length": 24 + idx,
                "response_length": args.max_tokens,
                "round_index": idx % 2,
            }
            for idx in range(args.num_requests)
        ]

    first_ts = min(int(row["timestamp"]) for row in rows)
    histories: dict[str, list[dict[str, str]]] = {}
    history_tokens: dict[str, int] = {}
    items: list[WorkItem] = []
    for idx, row in enumerate(rows):
        sid = f"ming-interactive-{row['user_id']}"
        history = histories.setdefault(sid, [])
        hist_tokens_before = history_tokens.get(sid, 0)
        query_len = max(1, int(row["query_length"]))
        response_len = max(1, min(int(row["response_length"]), args.max_tokens))
        prompt = synthetic_text(
            query_len,
            f"{row['user_id']}-{row['round_index']}",
            "请用自然中文回答当前用户问题：",
        )
        messages = base_messages(prompt, history)
        prefix_messages = messages[:-1]
        metadata: dict[str, Any] = {
            "omni_workload": "ming_interactive_retained",
            "omni_session_id": sid,
            "omni_round_index": row["round_index"],
            "omni_trace_timestamp": row["timestamp"],
            "omni_query_length": query_len,
            "omni_response_length": response_len,
            "omni_history_tokens_before": hist_tokens_before,
            "omni_history_tokens_after": hist_tokens_before + query_len + response_len,
            "omni_total_input_tokens": hist_tokens_before + query_len,
        }
        if prefix_messages:
            metadata["committed_prefix_hash"] = text_hash(prefix_messages)
            metadata["omni_prefix_token_estimate"] = hist_tokens_before
        scheduled = max(0.0, (int(row["timestamp"]) - first_ts) / max(args.speedup, 1e-6))
        items.append(
            WorkItem(
                request_index=idx,
                request_id=f"ming-interactive-{args.run_id}-{idx}",
                workload="interactive",
                kind="text",
                prompt=prompt,
                messages=messages,
                max_tokens=response_len,
                scheduled_offset_sec=scheduled,
                metadata=metadata,
            )
        )
        history.extend(
            [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": synthetic_text(response_len, f"resp-{idx}", "上轮回答摘要："),
                },
            ]
        )
        histories[sid] = history[-2 * max(1, args.retained_turns) :]
        history_tokens[sid] = hist_tokens_before + query_len + response_len
    return items


def build_mix_items(args: argparse.Namespace) -> list[WorkItem]:
    base = build_interactive_items(args)
    rng = random.Random(args.seed + 17)
    items: list[WorkItem] = []
    video_share = min(max(args.mix_video_share, 0.0), 1.0)
    video_count = int(round(len(base) * video_share))
    video_count = min(video_count, len(base))
    video_indices = {
        min(len(base) - 1, int((slot + 0.5) * len(base) / max(video_count, 1)))
        for slot in range(video_count)
    }
    for idx, item in enumerate(base):
        kind = "video_surrogate" if idx in video_indices else "text"
        if kind == "video_surrogate":
            topic = TOPICS[(idx + 5) % len(TOPICS)]
            prompt = (
                "这是混合 text/video workload 的视频事件替代请求。"
                f"请根据一段关于{topic}的假想视频，用三句中文给出口语化说明。"
            )
            item = dataclasses.replace(
                item,
                kind=kind,
                prompt=prompt,
                messages=base_messages(prompt, item.messages[1:-1]),
                metadata={
                    **item.metadata,
                    "omni_workload": "ming_mix_retained",
                    "omni_mix_kind": "video_surrogate",
                    "omni_video_surrogate": True,
                },
            )
        else:
            item.metadata["omni_workload"] = "ming_mix_retained"
            item.metadata["omni_mix_kind"] = "text"
        jitter = rng.uniform(0.0, max(args.mix_arrival_jitter_sec, 0.0))
        items.append(dataclasses.replace(item, workload="mix", scheduled_offset_sec=item.scheduled_offset_sec + jitter))
    return items[: args.num_requests]


def build_items(args: argparse.Namespace) -> list[WorkItem]:
    if args.workload == "sharegpt":
        return build_sharegpt_items(args)
    if args.workload == "interactive":
        return build_interactive_items(args)
    if args.workload == "mix":
        return build_mix_items(args)
    raise ValueError(f"unknown workload: {args.workload}")


async def run_one(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    item: WorkItem,
    args: argparse.Namespace,
    run_started: float,
) -> RequestResult:
    now = time.perf_counter()
    target = run_started + max(0.0, item.scheduled_offset_sec)
    if target > now:
        await asyncio.sleep(target - now)

    async with sem:
        t0 = time.perf_counter()
        queue_delay = max(0.0, t0 - target)
        result = RequestResult(
            request_index=item.request_index,
            request_id=item.request_id,
            workload=item.workload,
            kind=item.kind,
            prompt=item.prompt,
            scheduled_offset_sec=item.scheduled_offset_sec,
            queue_delay_sec=queue_delay,
        )
        result.issued_at = time.time()
        try:
            async with session.post(args.url, json=build_payload(item, args)) as resp:
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
                    if obj.get("usage"):
                        result.usage = obj["usage"]
                    metrics = obj.get("metrics") or {}
                    if metrics:
                        try:
                            stage_id = int(metrics.get("stage_id", -1))
                        except (TypeError, ValueError):
                            stage_id = -1
                        if stage_id == args.talker_stage_id:
                            cum = metrics.get("num_tokens_out_cum")
                            if isinstance(cum, int):
                                result.talker_frames_generated = cum

                    choice = (obj.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    modality = delta.get("modality") or obj.get("modality")

                    audio_data = None
                    audio_obj = delta.get("audio")
                    if isinstance(audio_obj, dict):
                        audio_data = audio_obj.get("data")
                    elif isinstance(audio_obj, str):
                        audio_data = audio_obj

                    content = delta.get("content")
                    force_audio = modality == "audio" or audio_data is not None
                    raw_audio = decode_audio_candidate(audio_data, force_audio=force_audio)
                    if not raw_audio:
                        raw_audio = decode_audio_candidate(content, force_audio=force_audio)
                    if raw_audio:
                        result.audio_chunks += 1
                        result.audio_bytes += len(raw_audio)
                        if result.first_audio_sec is None:
                            result.first_audio_sec = time.perf_counter() - t0
                            if args.stop_after_first_audio:
                                result.stopped_after_first_audio = True
                                break
                        continue

                    if isinstance(content, str) and content and modality != "stage_progress":
                        result.text_chars += len(content)
                        if len(result.text_preview) < 240:
                            result.text_preview = (result.text_preview + content)[:240]
                        if result.first_text_sec is None:
                            result.first_text_sec = time.perf_counter() - t0

                result.e2e_sec = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture failures.
            result.error = f"{type(exc).__name__}: {exc}"
            result.e2e_sec = time.perf_counter() - t0
        return result


async def run_benchmark(args: argparse.Namespace) -> list[RequestResult]:
    items = build_items(args)
    timeout = aiohttp.ClientTimeout(
        total=args.request_timeout,
        connect=args.connect_timeout,
        sock_connect=args.connect_timeout,
    )
    connector = aiohttp.TCPConnector(limit=max(args.max_inflight * 2, 32))
    sem = asyncio.Semaphore(args.max_inflight)
    run_started = time.perf_counter()
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        read_bufsize=16 * 1024 * 1024,
        max_line_size=16 * 1024 * 1024,
    ) as session:
        tasks = [
            asyncio.create_task(run_one(session, sem, item, args, run_started))
            for item in items
        ]
        results: list[RequestResult] = []
        for fut in asyncio.as_completed(tasks):
            result = await fut
            results.append(result)
            if len(results) % args.progress_every == 0 or len(results) == len(items):
                ok = sum(1 for r in results if r.status == 200 and not r.error)
                print(f"progress {len(results)}/{len(items)} ok={ok}", flush=True)
        results.sort(key=lambda r: r.request_index)
        return results


def build_summary(args: argparse.Namespace, results: list[RequestResult], elapsed: float) -> dict[str, Any]:
    ok_results = [r for r in results if r.status == 200 and not r.error]
    failed = [r for r in results if r not in ok_results]
    has_video_surrogate = any(r.kind == "video_surrogate" for r in results)

    def values(name: str, rows: list[RequestResult] | None = None) -> list[float]:
        vals: list[float] = []
        for result in rows or ok_results:
            val = getattr(result, name)
            if isinstance(val, (int, float)):
                vals.append(float(val))
        return vals

    audio_ttfp = metric_stats(values("first_audio_sec"))
    text_ttft = metric_stats(values("first_text_sec"))
    e2e = metric_stats(values("e2e_sec"))
    throughput = len(ok_results) / elapsed if elapsed > 0 else None
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({r.kind for r in ok_results}):
        kind_rows = [r for r in ok_results if r.kind == kind]
        by_kind[kind] = {
            "count": len(kind_rows),
            "audio_ttfp_sec": metric_stats(values("first_audio_sec", kind_rows)),
            "e2e_sec": metric_stats(values("e2e_sec", kind_rows)),
        }

    summary: dict[str, Any] = {
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workload": args.workload,
        "url": args.url,
        "model": args.model,
        "num_requests": len(results),
        "max_inflight": args.max_inflight,
        "concurrency": args.max_inflight,
        "max_tokens": args.max_tokens,
        "counts": {
            "total": len(results),
            "completed_ok": len(ok_results),
            "failed": len(failed),
            "interrupted": 0,
            "completed_normally": len(ok_results),
        },
        "totals": {
            "completed_ok": len(ok_results),
            "failed": len(failed),
            "throughput_rps": throughput,
            "useful_qps": throughput,
            "completed_qps": throughput,
            "live_completed_qps": throughput,
            "wall_s": elapsed,
        },
        "latency": {
            "text_ttft_sec": text_ttft,
            "ttft_sec": text_ttft,
            "audio_ttfp_sec": audio_ttfp,
            "audio_ttfp_ms": scale_stats(audio_ttfp, 1000.0),
            "e2e_sec": e2e,
        },
        "audio": {
            "chunks": metric_stats(values("audio_chunks")),
            "bytes": metric_stats(values("audio_bytes")),
        },
        "events": metric_stats(values("events")),
        "text_chars": metric_stats(values("text_chars")),
        "by_kind": by_kind,
        "errors": [
            {
                "request_index": r.request_index,
                "request_id": r.request_id,
                "status": r.status,
                "error": (r.error or "")[:400],
            }
            for r in failed[:10]
        ],
        "ming_adapter": {
            "schema": "qwen_cbound_plot_compatible",
            "video_endpoint": "not_used" if args.workload == "mix" else "n/a",
            "mix_video_surrogate": has_video_surrogate,
            "stop_after_first_audio": args.stop_after_first_audio,
        },
    }
    if args.workload == "mix":
        summary["latency"]["audio_ttfp_sec"] = {
            "all": audio_ttfp,
            **{
                kind: data["audio_ttfp_sec"]
                for kind, data in by_kind.items()
            },
        }
    return round_floats(summary)


def write_outputs(args: argparse.Namespace, results: list[RequestResult], summary: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_request.jsonl").open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(round_floats(result.to_dict()), ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    meta = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "meta.json").write_text(
        json.dumps(round_floats(meta), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("sharegpt", "interactive", "mix"), required=True)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--label", default="ming_cbound")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--max-inflight", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--target-sentences", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=420.0)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_SHAREGPT_DATASET)
    parser.add_argument("--trace-path", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--time-start", type=int, default=6780)
    parser.add_argument("--time-limit", type=int, default=7080)
    parser.add_argument("--speedup", type=float, default=100.0)
    parser.add_argument("--retained-turns", type=int, default=4)
    parser.add_argument("--mix-video-share", type=float, default=0.0)
    parser.add_argument("--mix-arrival-jitter-sec", type=float, default=0.0)
    parser.add_argument("--talker-stage-id", type=int, default=1)
    parser.add_argument("--stop-after-first-audio", action="store_true")
    args = parser.parse_args()
    if args.num_requests <= 0:
        raise SystemExit("--num-requests must be positive")
    if args.max_inflight <= 0:
        raise SystemExit("--max-inflight must be positive")
    return args


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    results = asyncio.run(run_benchmark(args))
    elapsed = time.perf_counter() - t0
    summary = build_summary(args, results, elapsed)
    write_outputs(args, results, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["counts"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
