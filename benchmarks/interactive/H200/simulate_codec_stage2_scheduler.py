#!/usr/bin/env python3
"""Offline simulator for Code2Wav stage2 chunk scheduling policies.

The input is the JSONL emitted by ``OMNI_CODEC_STAGE2_TRACE_PATH``.  Each
completed chunk contributes a ready timestamp and an observed service time
(``schedule_to_complete_ms``).  The simulator replays those chunks on a fixed
number of identical stage2 workers and evaluates alternative queue ordering
policies.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Chunk:
    seq: int
    request_id: str
    chunk_id: int
    ready_ms: float
    service_ms: float
    service_est_ms: float
    mode: str
    window_frames: int
    observed_wait_ms: float | None


@dataclass(frozen=True)
class Policy:
    name: str
    first_bias_ms: float = 0.0
    first_deadline_ms: float = 0.0
    post_deadline_ms: float = 0.0
    service_floor_ms: float = 20.0

    @property
    def label(self) -> str:
        parts = [self.name]
        if self.first_bias_ms:
            parts.append(f"fb{self.first_bias_ms:g}")
        if self.first_deadline_ms or self.post_deadline_ms:
            parts.append(f"fd{self.first_deadline_ms:g}_pd{self.post_deadline_ms:g}")
        if self.service_floor_ms != 20.0:
            parts.append(f"sf{self.service_floor_ms:g}")
        return "_".join(parts)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _pct(values: Iterable[float], q: float) -> float | None:
    xs = sorted(v for v in values if _is_num(v))
    if not xs:
        return None
    idx = (len(xs) - 1) * q / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)


def _fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.3f}{suffix}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_chunks(path: Path) -> list[Chunk]:
    rows = _load_jsonl(path)
    complete_rows = [row for row in rows if row.get("event") == "complete"]
    if not complete_rows:
        raise SystemExit(f"no complete events found in {path}")

    min_ready = min(
        float(row.get("ready_ts") or row.get("recv_ts") or row.get("ts"))
        for row in complete_rows
    )
    chunks: list[Chunk] = []
    for seq, row in enumerate(sorted(complete_rows, key=lambda r: (r.get("ready_ts") or r.get("ts"), r.get("request_id", ""), r.get("chunk_id", 0)))):
        ready_ts = row.get("ready_ts") or row.get("recv_ts") or row.get("ts")
        service_ms = row.get("schedule_to_complete_ms")
        if not _is_num(ready_ts) or not _is_num(service_ms):
            continue
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        chunk_id = int(row.get("chunk_id") or 0)
        window_frames = int(meta.get("codec_window_frames") or meta.get("codec_chunk_frames") or 1)
        # Relative estimate only.  HRRN needs a service denominator; the actual
        # replay still uses observed service_ms to preserve the measured load.
        service_est_ms = max(1.0, float(window_frames))
        chunks.append(
            Chunk(
                seq=seq,
                request_id=str(row.get("request_id") or ""),
                chunk_id=chunk_id,
                ready_ms=(float(ready_ts) - min_ready) * 1000.0,
                service_ms=max(0.001, float(service_ms)),
                service_est_ms=service_est_ms,
                mode=str(meta.get("codec_cadence_mode") or ""),
                window_frames=window_frames,
                observed_wait_ms=(
                    float(row["ready_to_schedule_ms"])
                    if _is_num(row.get("ready_to_schedule_ms"))
                    else None
                ),
            )
        )
    return chunks


def _choose_ready(
    ready: list[Chunk],
    *,
    now_ms: float,
    policy: Policy,
    first_chunks: int,
) -> int:
    def is_first(chunk: Chunk) -> bool:
        return chunk.chunk_id < first_chunks

    if policy.name == "fcfs":
        return min(range(len(ready)), key=lambda i: (ready[i].ready_ms, ready[i].seq))
    if policy.name == "spt":
        return min(range(len(ready)), key=lambda i: (ready[i].service_ms, ready[i].ready_ms, ready[i].seq))
    if policy.name == "chunk":
        return min(range(len(ready)), key=lambda i: (ready[i].chunk_id, ready[i].ready_ms, ready[i].seq))
    if policy.name == "first_strict":
        return min(
            range(len(ready)),
            key=lambda i: (0 if is_first(ready[i]) else 1, ready[i].chunk_id, ready[i].ready_ms, ready[i].seq),
        )
    if policy.name == "deadline":
        def deadline_key(i: int) -> tuple[float, int, float, int]:
            chunk = ready[i]
            deadline = policy.first_deadline_ms if is_first(chunk) else policy.post_deadline_ms
            if deadline <= 0:
                deadline = 0.0 if is_first(chunk) else 1_000_000.0
            return (chunk.ready_ms + deadline, 0 if is_first(chunk) else 1, chunk.ready_ms, chunk.seq)

        return min(range(len(ready)), key=deadline_key)
    if policy.name == "hrrn":
        def hrrn_key(i: int) -> tuple[float, int, float, int]:
            chunk = ready[i]
            wait_ms = max(0.0, now_ms - chunk.ready_ms)
            if is_first(chunk):
                wait_ms += policy.first_bias_ms
            service = max(policy.service_floor_ms, chunk.service_est_ms)
            response_ratio = 1.0 + wait_ms / service
            return (-response_ratio, 0 if is_first(chunk) else 1, chunk.ready_ms, chunk.seq)

        return min(range(len(ready)), key=hrrn_key)
    raise ValueError(f"unsupported policy {policy.name!r}")


def simulate(
    chunks: list[Chunk],
    *,
    servers: int,
    first_chunks: int,
    policy: Policy,
) -> dict[str, Any]:
    arrivals = sorted(chunks, key=lambda c: (c.ready_ms, c.seq))
    server_heap: list[float] = [0.0 for _ in range(servers)]
    heapq.heapify(server_heap)
    ready: list[Chunk] = []
    idx = 0
    waits: list[tuple[Chunk, float, float]] = []

    while idx < len(arrivals) or ready:
        free_ms = heapq.heappop(server_heap)
        if not ready and idx < len(arrivals) and arrivals[idx].ready_ms > free_ms:
            free_ms = arrivals[idx].ready_ms
        while idx < len(arrivals) and arrivals[idx].ready_ms <= free_ms:
            ready.append(arrivals[idx])
            idx += 1
        if not ready:
            heapq.heappush(server_heap, free_ms)
            continue
        choice = _choose_ready(ready, now_ms=free_ms, policy=policy, first_chunks=first_chunks)
        chunk = ready.pop(choice)
        start_ms = max(free_ms, chunk.ready_ms)
        wait_ms = start_ms - chunk.ready_ms
        finish_ms = start_ms + chunk.service_ms
        waits.append((chunk, wait_ms, finish_ms))
        heapq.heappush(server_heap, finish_ms)

    def group(is_first: bool) -> list[float]:
        return [wait for chunk, wait, _ in waits if (chunk.chunk_id < first_chunks) == is_first]

    def completion_group(is_first: bool) -> list[float]:
        return [wait + chunk.service_ms for chunk, wait, _ in waits if (chunk.chunk_id < first_chunks) == is_first]

    all_wait = [wait for _, wait, _ in waits]
    first_wait = group(True)
    post_wait = group(False)
    first_complete = completion_group(True)
    post_complete = completion_group(False)
    makespan = max((finish for _, _, finish in waits), default=0.0)
    return {
        "policy": policy.label,
        "n": len(waits),
        "makespan_ms": makespan,
        "first_n": len(first_wait),
        "post_n": len(post_wait),
        "wait_p50": _pct(all_wait, 50),
        "wait_p90": _pct(all_wait, 90),
        "wait_p95": _pct(all_wait, 95),
        "wait_max": max(all_wait) if all_wait else None,
        "first_wait_p50": _pct(first_wait, 50),
        "first_wait_p90": _pct(first_wait, 90),
        "first_wait_p95": _pct(first_wait, 95),
        "first_wait_max": max(first_wait) if first_wait else None,
        "first_complete_p90": _pct(first_complete, 90),
        "first_complete_p95": _pct(first_complete, 95),
        "post_wait_p50": _pct(post_wait, 50),
        "post_wait_p90": _pct(post_wait, 90),
        "post_wait_p95": _pct(post_wait, 95),
        "post_wait_max": max(post_wait) if post_wait else None,
        "post_complete_p90": _pct(post_complete, 90),
        "post_complete_p95": _pct(post_complete, 95),
        "post_starved_1s": sum(1 for v in post_wait if v > 1000.0),
        "post_starved_5s": sum(1 for v in post_wait if v > 5000.0),
        "first_starved_1s": sum(1 for v in first_wait if v > 1000.0),
        "first_starved_5s": sum(1 for v in first_wait if v > 5000.0),
    }


def default_policies() -> list[Policy]:
    policies = [
        Policy("fcfs"),
        Policy("spt"),
        Policy("chunk"),
        Policy("first_strict"),
        Policy("deadline", first_deadline_ms=250.0, post_deadline_ms=1000.0),
        Policy("deadline", first_deadline_ms=500.0, post_deadline_ms=1000.0),
    ]
    for bias in (0.0, 50.0, 100.0, 200.0, 300.0, 500.0, 800.0, 1000.0):
        policies.append(Policy("hrrn", first_bias_ms=bias, service_floor_ms=20.0))
    return policies


def _observed_summary(chunks: list[Chunk], first_chunks: int) -> dict[str, Any]:
    rows = [
        (chunk, chunk.observed_wait_ms)
        for chunk in chunks
        if chunk.observed_wait_ms is not None
    ]
    all_wait = [float(wait) for _, wait in rows]
    first = [float(wait) for chunk, wait in rows if chunk.chunk_id < first_chunks]
    post = [float(wait) for chunk, wait in rows if chunk.chunk_id >= first_chunks]
    return {
        "policy": "observed_trace",
        "n": len(rows),
        "makespan_ms": None,
        "first_n": len(first),
        "post_n": len(post),
        "wait_p50": _pct(all_wait, 50),
        "wait_p90": _pct(all_wait, 90),
        "wait_p95": _pct(all_wait, 95),
        "wait_max": max(all_wait) if all_wait else None,
        "first_wait_p50": _pct(first, 50),
        "first_wait_p90": _pct(first, 90),
        "first_wait_p95": _pct(first, 95),
        "first_wait_max": max(first) if first else None,
        "first_complete_p90": None,
        "first_complete_p95": None,
        "post_wait_p50": _pct(post, 50),
        "post_wait_p90": _pct(post, 90),
        "post_wait_p95": _pct(post, 95),
        "post_wait_max": max(post) if post else None,
        "post_complete_p90": None,
        "post_complete_p95": None,
        "post_starved_1s": sum(1 for v in post if v > 1000.0),
        "post_starved_5s": sum(1 for v in post if v > 5000.0),
        "first_starved_1s": sum(1 for v in first if v > 1000.0),
        "first_starved_5s": sum(1 for v in first if v > 5000.0),
    }


def _table(results: list[dict[str, Any]]) -> list[str]:
    cols = [
        ("policy", ""),
        ("first_wait_p90", "ms"),
        ("first_wait_p95", "ms"),
        ("first_wait_max", "ms"),
        ("post_wait_p90", "ms"),
        ("post_wait_p95", "ms"),
        ("post_wait_max", "ms"),
        ("post_starved_1s", ""),
        ("post_starved_5s", ""),
        ("makespan_ms", "ms"),
    ]
    lines = [
        "| " + " | ".join(name for name, _ in cols) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|",
    ]
    for row in results:
        vals = []
        for name, suffix in cols:
            val = row.get(name)
            if name == "policy":
                vals.append(f"`{val}`")
            elif isinstance(val, int):
                vals.append(str(val))
            elif _is_num(val):
                vals.append(_fmt(float(val), suffix))
            else:
                vals.append("-")
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def render_report(
    chunks: list[Chunk],
    results: list[dict[str, Any]],
    *,
    input_path: Path,
    servers: int,
    first_chunks: int,
) -> str:
    sorted_results = sorted(
        results,
        key=lambda row: (
            float("inf") if row.get("first_wait_p90") is None else row["first_wait_p90"],
            float("inf") if row.get("post_wait_p95") is None else row["post_wait_p95"],
        ),
    )
    lines = [
        "# Code2Wav Stage2 Scheduler Simulation",
        "",
        f"Input: `{input_path}`",
        f"Chunks: {len(chunks)}",
        f"Servers: {servers}",
        f"First-window chunks: chunk_id < {first_chunks}",
        "",
        "The replay uses measured `ready_ts` as queue arrival time and measured "
        "`schedule_to_complete_ms` as non-preemptive service time. It estimates "
        "stage2 ready-queue behavior only; upstream chunk readiness feedback must "
        "still be validated on GPU.",
        "",
        "## Policy Sweep",
        "",
    ]
    lines.extend(_table(sorted_results))
    lines.append("")

    candidates = [
        row
        for row in sorted_results
        if row["policy"] != "observed_trace"
        and (float("inf") if row.get("first_wait_p90") is None else row["first_wait_p90"]) <= 300.0
        and (float("inf") if row.get("post_wait_p95") is None else row["post_wait_p95"]) <= 1000.0
        and row.get("post_starved_5s") == 0
    ]
    lines.append("## Candidate Filter")
    if candidates:
        for row in candidates[:8]:
            lines.append(
                f"- `{row['policy']}`: first_wait_p90={_fmt(row.get('first_wait_p90'), 'ms')}, "
                f"post_wait_p95={_fmt(row.get('post_wait_p95'), 'ms')}, "
                f"post_starved_5s={row.get('post_starved_5s')}"
            )
    else:
        lines.append("- No policy met first_wait_p90<=300ms, post_wait_p95<=1000ms, post_starved_5s=0.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--servers", type=int, default=4)
    parser.add_argument("--first-chunks", type=int, default=1)
    args = parser.parse_args()

    chunks = load_chunks(args.input)
    first_chunks = max(1, args.first_chunks)
    results = [_observed_summary(chunks, first_chunks)]
    for policy in default_policies():
        results.append(
            simulate(
                chunks,
                servers=max(1, args.servers),
                first_chunks=first_chunks,
                policy=policy,
            )
        )
    report = render_report(
        chunks,
        results,
        input_path=args.input,
        servers=max(1, args.servers),
        first_chunks=first_chunks,
    )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
