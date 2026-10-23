#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from simulate_interactive_kv_policies import (
    TRACE_DIR,
    cap_real_multiturn_full_sessions,
    cap_real_multiturn_full_sessions_peak,
    cap_real_multiturn_peak,
    cap_session_balanced,
    load_entries,
    reannotate_next_use_within_selected,
    select_window,
    simulate_policy,
)


def parse_int_list(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def cheap_window_row(all_entries: list[Any], *, start: int, window_sec: int) -> dict[str, Any]:
    limit = start + window_sec
    window = select_window(all_entries, start, limit)
    counts: dict[str, int] = {}
    token_total = 0
    for entry in window:
        counts[entry.user_id] = counts.get(entry.user_id, 0) + 1
        token_total += int(entry.query_length) + int(entry.response_length)
    same_run_returns = sum(max(0, count - 1) for count in counts.values())
    return {
        "start": start,
        "limit": limit,
        "window_sec": window_sec,
        "entries": len(window),
        "sessions": len(counts),
        "same_run_returns": same_run_returns,
        "token_total": token_total,
        "score": same_run_returns * 100000 + token_total,
    }


def cap_entries(
    entries: list[Any],
    *,
    max_requests: int,
    policy: str,
    min_turns: int,
) -> list[Any]:
    if max_requests <= 0 or len(entries) <= max_requests:
        return entries
    if policy == "session_balanced":
        return cap_session_balanced(entries, max_requests, min_turns)
    if policy == "real_multiturn_peak":
        return cap_real_multiturn_peak(entries, max_requests, min_turns)
    if policy == "real_multiturn_full_sessions":
        return cap_real_multiturn_full_sessions(entries, max_requests, min_turns)
    if policy == "real_multiturn_full_sessions_peak":
        return cap_real_multiturn_full_sessions_peak(entries, max_requests, min_turns)
    if policy == "first":
        return entries[:max_requests]
    raise ValueError(f"unknown cap policy: {policy}")


def eval_combo(
    all_entries: list[Any],
    *,
    start: int,
    window_sec: int,
    max_requests: int,
    max_requests_policy: str,
    min_turns: int,
    capacity_tokens: int,
    block_size_tokens: int,
    policy: str,
    next_use_scope: str,
) -> dict[str, Any]:
    limit = start + window_sec
    window = select_window(all_entries, start, limit)
    selected = cap_entries(
        window,
        max_requests=max_requests,
        policy=max_requests_policy,
        min_turns=min_turns,
    )
    selected = [replace(entry) for entry in selected]
    if next_use_scope == "selected":
        reannotate_next_use_within_selected(selected)

    capacity_blocks = max(1, capacity_tokens // block_size_tokens)
    lru = simulate_policy(
        selected,
        policy="lru",
        capacity_blocks=capacity_blocks,
        block_size_tokens=block_size_tokens,
    )
    candidate = simulate_policy(
        selected,
        policy=policy,
        capacity_blocks=capacity_blocks,
        block_size_tokens=block_size_tokens,
    )
    lru_miss = int(lru["miss_blocks"])
    candidate_miss = int(candidate["miss_blocks"])
    lru_evictions = int(lru["evictions"])
    candidate_evictions = int(candidate["evictions"])
    return {
        "start": start,
        "limit": limit,
        "window_sec": window_sec,
        "entries_before_cap": len(window),
        "requests": len(selected),
        "sessions": len({entry.user_id for entry in selected}),
        "max_requests": max_requests,
        "max_requests_policy": max_requests_policy,
        "capacity_tokens": capacity_tokens,
        "capacity_blocks": capacity_blocks,
        "lru_hit_ratio": float(lru["hit_ratio"]),
        "policy_hit_ratio": float(candidate["hit_ratio"]),
        "lru_miss_blocks": lru_miss,
        "policy_miss_blocks": candidate_miss,
        "miss_blocks_saved": lru_miss - candidate_miss,
        "lru_evictions": lru_evictions,
        "policy_evictions": candidate_evictions,
        "evictions_saved": lru_evictions - candidate_evictions,
        "lru_useful_evictions": int(lru["useful_evictions"]),
        "policy_useful_evictions": int(candidate["useful_evictions"]),
        "lru_requests_with_history_blocks": int(lru["requests_with_history_blocks"]),
        "policy_requests_with_history_blocks": int(candidate["requests_with_history_blocks"]),
    }


def write_markdown(path: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    lines = [
        "# Eviction Micro Workload Candidates",
        "",
        f"- trace_dir: {config['trace_dir']}",
        f"- scan_range: [{config['time_start']}, {config['time_limit']})",
        f"- policy: {config['policy']}",
        f"- cap_policy: {config['max_requests_policy']}",
        "",
        "| rank | window | cap_tokens | requests | sessions | lru_miss | "
        "policy_miss | lru_evict | policy_evict | hit_delta |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        hit_delta = row["policy_hit_ratio"] - row["lru_hit_ratio"]
        lines.append(
            f"| {rank} | [{row['start']},{row['limit']}) | "
            f"{row['capacity_tokens']} | {row['requests']} | {row['sessions']} | "
            f"{row['lru_miss_blocks']} | {row['policy_miss_blocks']} | "
            f"{row['lru_evictions']} | {row['policy_evictions']} | "
            f"{hit_delta:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--parts", type=int, nargs="+", default=[1])
    parser.add_argument("--time-start", type=int, default=None)
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--window-sec", type=parse_int_list, default=[90, 120, 180])
    parser.add_argument("--step-sec", type=int, default=30)
    parser.add_argument(
        "--max-requests",
        type=parse_int_list,
        default=[1000, 1200, 1500, 1800, 2200],
    )
    parser.add_argument(
        "--max-requests-policy",
        choices=[
            "first",
            "session_balanced",
            "real_multiturn_peak",
            "real_multiturn_full_sessions",
            "real_multiturn_full_sessions_peak",
        ],
        default="session_balanced",
    )
    parser.add_argument("--session-balanced-min-turns", type=int, default=3)
    parser.add_argument(
        "--capacity-tokens",
        type=parse_int_list,
        default=[80000, 100000, 153072],
    )
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument(
        "--policy",
        default="expected_gap_bucket_response_tail_aware",
    )
    parser.add_argument(
        "--next-use-scope",
        choices=["full_trace", "selected"],
        default="selected",
    )
    parser.add_argument("--min-lru-evictions", type=int, default=500)
    parser.add_argument("--min-lru-misses", type=int, default=100)
    parser.add_argument(
        "--window-top-k",
        type=int,
        default=5,
        help="Cheap prefilter count per window size before running simulator. "
             "Use 0 for exhaustive scan.",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    all_entries = load_entries(args.trace_dir, args.parts)
    if not all_entries:
        raise SystemExit("no entries loaded")
    scan_start = all_entries[0].timestamp if args.time_start is None else args.time_start
    scan_limit = all_entries[-1].timestamp + 1 if args.time_limit is None else args.time_limit

    candidate_windows: list[dict[str, Any]] = []
    for window_sec in args.window_sec:
        window_rows: list[dict[str, Any]] = []
        start = scan_start
        while start + window_sec <= scan_limit:
            window_rows.append(
                cheap_window_row(all_entries, start=start, window_sec=window_sec)
            )
            start += max(1, args.step_sec)
        window_rows.sort(
            key=lambda row: (
                row["score"],
                row["same_run_returns"],
                row["entries"],
            ),
            reverse=True,
        )
        if args.window_top_k > 0:
            window_rows = window_rows[: args.window_top_k]
        candidate_windows.extend(window_rows)

    rows: list[dict[str, Any]] = []
    for window in candidate_windows:
        for capacity_tokens in args.capacity_tokens:
            for max_requests in args.max_requests:
                rows.append(
                    eval_combo(
                        all_entries,
                        start=int(window["start"]),
                        window_sec=int(window["window_sec"]),
                        max_requests=max_requests,
                        max_requests_policy=args.max_requests_policy,
                        min_turns=args.session_balanced_min_turns,
                        capacity_tokens=capacity_tokens,
                        block_size_tokens=args.block_size_tokens,
                        policy=args.policy,
                        next_use_scope=args.next_use_scope,
                    )
                )

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        eligible = (
            row["lru_evictions"] >= args.min_lru_evictions
            and row["lru_miss_blocks"] >= args.min_lru_misses
        )
        return (
            0 if eligible else 1,
            row["requests"],
            row["window_sec"],
            row["capacity_tokens"],
            -row["miss_blocks_saved"],
            -row["lru_evictions"],
            row["start"],
        )

    rows.sort(key=sort_key)
    top_rows = rows[: max(1, args.top_k)]

    config = {
        "trace_dir": str(args.trace_dir),
        "parts": args.parts,
        "time_start": scan_start,
        "time_limit": scan_limit,
        "window_sec": args.window_sec,
        "step_sec": args.step_sec,
        "max_requests": args.max_requests,
        "max_requests_policy": args.max_requests_policy,
        "session_balanced_min_turns": args.session_balanced_min_turns,
        "capacity_tokens": args.capacity_tokens,
        "block_size_tokens": args.block_size_tokens,
        "policy": args.policy,
        "next_use_scope": args.next_use_scope,
        "min_lru_evictions": args.min_lru_evictions,
        "min_lru_misses": args.min_lru_misses,
        "window_top_k": args.window_top_k,
        "candidate_windows": candidate_windows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "micro_workload_candidates.json").write_text(
        json.dumps({"config": config, "top_candidates": top_rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fieldnames = list(top_rows[0].keys()) if top_rows else []
    with (args.output_dir / "micro_workload_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top_rows)
    write_markdown(args.output_dir / "micro_workload_candidates.md", top_rows, config)
    print(args.output_dir / "micro_workload_candidates.json")
    print(args.output_dir / "micro_workload_candidates.md")
    print(args.output_dir / "micro_workload_candidates.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
