#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRACE_DIR = Path(__file__).parent / "trace"


@dataclass
class Entry:
    user_id: str
    timestamp: int
    query_length: int
    response_length: int
    round_index: int
    next_gap_sec: int | None = None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[idx]


def describe(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    xs = [float(v) for v in values]
    return {
        "n": len(xs),
        "min": min(xs),
        "p50": percentile(xs, 0.50),
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
        "max": max(xs),
        "mean": statistics.mean(xs),
    }


def load_entries(trace_dir: Path, parts: list[int]) -> list[Entry]:
    entries: list[Entry] = []
    for part in parts:
        path = trace_dir / f"total_traces_part{part}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                fields = line.split()
                if len(fields) < 5:
                    continue
                entries.append(
                    Entry(
                        user_id=fields[0],
                        timestamp=int(fields[1]),
                        query_length=int(fields[2]),
                        response_length=int(fields[3]),
                        round_index=int(fields[4]),
                    )
                )
    entries.sort(key=lambda e: (e.timestamp, e.user_id, e.round_index))
    next_by_user: dict[str, Entry] = {}
    for entry in reversed(entries):
        nxt = next_by_user.get(entry.user_id)
        entry.next_gap_sec = None if nxt is None else nxt.timestamp - entry.timestamp
        next_by_user[entry.user_id] = entry
    return entries


def summarize_window(
    entries: list[Entry],
    *,
    start: int,
    limit: int,
    kv_bytes_per_token: int,
    block_size_tokens: int,
    stage0_cache_tokens: int,
) -> dict[str, Any]:
    window = [e for e in entries if start <= e.timestamp < limit]
    by_user: dict[str, list[Entry]] = defaultdict(list)
    for entry in window:
        by_user[entry.user_id].append(entry)
    for user_entries in by_user.values():
        user_entries.sort(key=lambda e: (e.timestamp, e.round_index))

    same_run_returns = 0
    same_run_gaps: list[int] = []
    window_history_after: list[int] = []
    retained_tokens_peak = 0
    pending_tokens_peak = 0
    idle_pending_tokens_peak = 0
    waiting_sessions_peak = 0

    max_round_by_user = {
        uid: max(e.round_index for e in user_entries)
        for uid, user_entries in by_user.items()
        if user_entries
    }
    history_tokens_by_user = {uid: 0 for uid in by_user}
    events_by_time: dict[int, list[Entry]] = defaultdict(list)
    for uid, user_entries in by_user.items():
        for prev, cur in zip(user_entries, user_entries[1:]):
            same_run_returns += 1
            same_run_gaps.append(cur.timestamp - prev.timestamp)
        for entry in user_entries:
            events_by_time[entry.timestamp].append(entry)

    last_round_seen: dict[str, int] = {}
    for ts in sorted(events_by_time):
        current_users = {entry.user_id for entry in events_by_time[ts]}
        for entry in sorted(events_by_time[ts], key=lambda e: (e.user_id, e.round_index)):
            if entry.round_index == 0:
                history_tokens_by_user[entry.user_id] = 0
            history_tokens_by_user[entry.user_id] += (
                int(entry.query_length) + int(entry.response_length)
            )
            last_round_seen[entry.user_id] = entry.round_index
            window_history_after.append(history_tokens_by_user[entry.user_id])

        retained_now = sum(history_tokens_by_user.values())
        pending_now = 0
        idle_pending_now = 0
        waiting_now = 0
        for uid, tokens in history_tokens_by_user.items():
            if tokens <= 0:
                continue
            last_round = last_round_seen.get(uid)
            if last_round is None:
                continue
            if last_round < max_round_by_user.get(uid, last_round):
                pending_now += tokens
                waiting_now += 1
                if uid not in current_users:
                    idle_pending_now += tokens

        retained_tokens_peak = max(retained_tokens_peak, retained_now)
        pending_tokens_peak = max(pending_tokens_peak, pending_now)
        idle_pending_tokens_peak = max(idle_pending_tokens_peak, idle_pending_now)
        waiting_sessions_peak = max(waiting_sessions_peak, waiting_now)

    def kv_gib(tokens: int) -> float:
        return round(float(tokens) * float(kv_bytes_per_token) / (1024.0**3), 3)

    def blocks(tokens: int) -> int:
        return int(math.ceil(float(tokens) / float(max(1, block_size_tokens))))

    entries_count = len(window)
    stage0_cache_frac = (
        retained_tokens_peak / stage0_cache_tokens
        if stage0_cache_tokens > 0 else None
    )
    pending_stage0_cache_frac = (
        pending_tokens_peak / stage0_cache_tokens
        if stage0_cache_tokens > 0 else None
    )
    idle_pending_stage0_cache_frac = (
        idle_pending_tokens_peak / stage0_cache_tokens
        if stage0_cache_tokens > 0 else None
    )
    return {
        "start": start,
        "limit": limit,
        "window_sec": limit - start,
        "entries": entries_count,
        "sessions": len(by_user),
        "same_run_returns": same_run_returns,
        "same_run_return_ratio": same_run_returns / entries_count if entries_count else 0.0,
        "same_run_gap_sec": describe(same_run_gaps),
        "turns_per_session": describe([len(v) for v in by_user.values()]),
        "window_history_tokens_after": describe(window_history_after),
        "retained_history_tokens_peak": retained_tokens_peak,
        "pending_retained_history_tokens_peak": pending_tokens_peak,
        "idle_pending_retained_history_tokens_peak": idle_pending_tokens_peak,
        "retained_history_blocks_peak_est": blocks(retained_tokens_peak),
        "pending_retained_history_blocks_peak_est": blocks(pending_tokens_peak),
        "idle_pending_retained_history_blocks_peak_est": blocks(idle_pending_tokens_peak),
        "retained_history_kv_gib_peak_est": kv_gib(retained_tokens_peak),
        "pending_retained_history_kv_gib_peak_est": kv_gib(pending_tokens_peak),
        "idle_pending_retained_history_kv_gib_peak_est": kv_gib(idle_pending_tokens_peak),
        "waiting_sessions_peak": waiting_sessions_peak,
        "stage0_cache_frac_peak_est": stage0_cache_frac,
        "pending_stage0_cache_frac_peak_est": pending_stage0_cache_frac,
        "idle_pending_stage0_cache_frac_peak_est": idle_pending_stage0_cache_frac,
    }


def write_markdown(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Interactive Eviction Window Scan",
        "",
        f"- parts: {config['parts']}",
        f"- window_sec: {config['window_sec']}",
        f"- step_sec: {config['step_sec']}",
        f"- stage0_cache_tokens: {config['stage0_cache_tokens']}",
        f"- block_size_tokens: {config['block_size_tokens']}",
        "",
        "| rank | window | entries | sessions | returns | return_ratio | "
        "hist_p90 | retained_peak_tokens | idle_pending_tokens | "
        "idle_pending_kv_gib | cache_frac |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows, 1):
        hist_p90 = row["window_history_tokens_after"]["p90"]
        cache_frac = row.get("idle_pending_stage0_cache_frac_peak_est")
        lines.append(
            f"| {idx} | [{row['start']},{row['limit']}) | "
            f"{row['entries']} | {row['sessions']} | {row['same_run_returns']} | "
            f"{row['same_run_return_ratio']:.3f} | "
            f"{hist_p90:.0f} | "
            f"{row['retained_history_tokens_peak']} | "
            f"{row['idle_pending_retained_history_tokens_peak']} | "
            f"{row['idle_pending_retained_history_kv_gib_peak_est']:.3f} | "
            f"{cache_frac:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--parts", type=int, nargs="+", default=[1])
    parser.add_argument("--window-sec", type=int, default=300)
    parser.add_argument("--step-sec", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--time-start", type=int, default=None)
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--kv-bytes-per-token",
        type=int,
        default=int(os.environ.get("OMNI_SESSION_KV_BYTES_PER_TOKEN", "98304")),
    )
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument(
        "--stage0-cache-tokens",
        type=int,
        default=0,
        help="Optional observed stage-0 cache capacity tokens for pressure fraction.",
    )
    args = parser.parse_args()

    entries = load_entries(args.trace_dir, args.parts)
    if not entries:
        raise SystemExit("no entries loaded")
    global_start = entries[0].timestamp if args.time_start is None else args.time_start
    global_limit = entries[-1].timestamp + 1 if args.time_limit is None else args.time_limit

    rows: list[dict[str, Any]] = []
    start = global_start
    while start + args.window_sec <= global_limit:
        rows.append(
            summarize_window(
                entries,
                start=start,
                limit=start + args.window_sec,
                kv_bytes_per_token=args.kv_bytes_per_token,
                block_size_tokens=args.block_size_tokens,
                stage0_cache_tokens=args.stage0_cache_tokens,
            )
        )
        start += max(1, args.step_sec)

    rows.sort(
        key=lambda row: (
            row["idle_pending_retained_history_tokens_peak"],
            row["same_run_returns"],
            row["window_history_tokens_after"]["p90"] or 0,
        ),
        reverse=True,
    )
    top_rows = rows[: max(1, args.top_k)]

    config = {
        "trace_dir": str(args.trace_dir),
        "parts": args.parts,
        "window_sec": args.window_sec,
        "step_sec": args.step_sec,
        "time_start": global_start,
        "time_limit": global_limit,
        "kv_bytes_per_token": args.kv_bytes_per_token,
        "block_size_tokens": args.block_size_tokens,
        "stage0_cache_tokens": args.stage0_cache_tokens,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "window_scan.json").write_text(
        json.dumps({"config": config, "top_windows": top_rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (args.output_dir / "window_scan.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "start",
                "limit",
                "window_sec",
                "entries",
                "sessions",
                "same_run_returns",
                "same_run_return_ratio",
                "history_p90",
                "retained_history_tokens_peak",
                "idle_pending_retained_history_tokens_peak",
                "idle_pending_retained_history_kv_gib_peak_est",
                "idle_pending_stage0_cache_frac_peak_est",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(top_rows, 1):
            writer.writerow(
                {
                    "rank": rank,
                    "start": row["start"],
                    "limit": row["limit"],
                    "window_sec": row["window_sec"],
                    "entries": row["entries"],
                    "sessions": row["sessions"],
                    "same_run_returns": row["same_run_returns"],
                    "same_run_return_ratio": row["same_run_return_ratio"],
                    "history_p90": row["window_history_tokens_after"]["p90"],
                    "retained_history_tokens_peak": row["retained_history_tokens_peak"],
                    "idle_pending_retained_history_tokens_peak": (
                        row["idle_pending_retained_history_tokens_peak"]
                    ),
                    "idle_pending_retained_history_kv_gib_peak_est": (
                        row["idle_pending_retained_history_kv_gib_peak_est"]
                    ),
                    "idle_pending_stage0_cache_frac_peak_est": (
                        row["idle_pending_stage0_cache_frac_peak_est"]
                    ),
                }
            )
    write_markdown(args.output_dir / "window_scan.md", config, top_rows)
    print(args.output_dir / "window_scan.json")
    print(args.output_dir / "window_scan.md")
    print(args.output_dir / "window_scan.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
