#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import simulate_interactive_kv_policies as sim


def parse_int_list(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


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


def cap_entries(
    entries: list[sim.Entry],
    *,
    max_requests: int,
    policy: str,
    min_turns: int,
) -> list[sim.Entry]:
    if max_requests <= 0 or len(entries) <= max_requests:
        return entries
    if policy == "first":
        return entries[:max_requests]
    if policy == "session_balanced":
        return sim.cap_session_balanced(entries, max_requests, min_turns)
    if policy == "real_multiturn_peak":
        return sim.cap_real_multiturn_peak(entries, max_requests, min_turns)
    raise ValueError(f"unknown cap policy: {policy}")


def summarize_window(
    entries: list[sim.Entry],
    *,
    start: int,
    limit: int,
    block_size_tokens: int,
) -> dict[str, Any]:
    window = sim.select_window(entries, start, limit)
    by_user: dict[str, list[sim.Entry]] = defaultdict(list)
    for entry in window:
        by_user[entry.user_id].append(entry)
    for user_entries in by_user.values():
        user_entries.sort(key=lambda e: (e.timestamp, e.round_index))

    same_run_returns = 0
    same_run_gaps: list[int] = []
    new_session_count = 0
    history_after_values: list[int] = []
    demand_blocks_values: list[int] = []
    demand_request_count = 0
    total_tokens = 0
    response_tokens = 0
    pending_tokens_peak = 0
    idle_pending_tokens_peak = 0
    waiting_sessions_peak = 0

    max_round_by_user = {
        uid: max(entry.round_index for entry in user_entries)
        for uid, user_entries in by_user.items()
        if user_entries
    }
    history_by_user: dict[str, int] = defaultdict(int)
    last_round_seen: dict[str, int] = {}
    events_by_time: dict[int, list[sim.Entry]] = defaultdict(list)

    for uid, user_entries in by_user.items():
        same_run_returns += max(0, len(user_entries) - 1)
        for prev, cur in zip(user_entries, user_entries[1:]):
            same_run_gaps.append(cur.timestamp - prev.timestamp)
        for entry in user_entries:
            events_by_time[entry.timestamp].append(entry)

    for ts in sorted(events_by_time):
        current_users = {entry.user_id for entry in events_by_time[ts]}
        for entry in sorted(events_by_time[ts], key=lambda e: (e.user_id, e.round_index)):
            if entry.round_index == 0:
                history_by_user[entry.user_id] = 0
                new_session_count += 1
            history_before = int(history_by_user.get(entry.user_id, 0))
            demand_blocks = history_before // block_size_tokens
            if demand_blocks > 0:
                demand_request_count += 1
            demand_blocks_values.append(demand_blocks)
            total_tokens += int(entry.query_length) + int(entry.response_length)
            response_tokens += int(entry.response_length)
            history_after = history_before + int(entry.query_length) + int(entry.response_length)
            history_by_user[entry.user_id] = history_after
            last_round_seen[entry.user_id] = entry.round_index
            history_after_values.append(history_after)

        pending_now = 0
        idle_pending_now = 0
        waiting_now = 0
        for uid, tokens in history_by_user.items():
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
        pending_tokens_peak = max(pending_tokens_peak, pending_now)
        idle_pending_tokens_peak = max(idle_pending_tokens_peak, idle_pending_now)
        waiting_sessions_peak = max(waiting_sessions_peak, waiting_now)

    entries_count = len(window)
    sessions = len(by_user)
    return {
        "start": start,
        "limit": limit,
        "window_sec": limit - start,
        "entries": entries_count,
        "sessions": sessions,
        "new_session_count": new_session_count,
        "new_session_ratio": new_session_count / entries_count if entries_count else 0.0,
        "same_run_returns": same_run_returns,
        "same_run_return_ratio": same_run_returns / entries_count if entries_count else 0.0,
        "same_run_gap_sec": describe(same_run_gaps),
        "total_tokens": total_tokens,
        "response_tokens": response_tokens,
        "history_after": describe(history_after_values),
        "demand_blocks": describe(demand_blocks_values),
        "demand_request_ratio": demand_request_count / entries_count if entries_count else 0.0,
        "pending_retained_tokens_peak": pending_tokens_peak,
        "idle_pending_retained_tokens_peak": idle_pending_tokens_peak,
        "waiting_sessions_peak": waiting_sessions_peak,
    }


def row_scores(row: dict[str, Any]) -> dict[str, float]:
    entries = max(1, int(row["entries"]))
    history_p90 = float(row["history_after"]["p90"] or 0.0)
    demand_p90 = float(row["demand_blocks"]["p90"] or 0.0)
    cold_score = (
        float(row["new_session_ratio"]) * float(row["entries"])
        - 0.35 * float(row["same_run_returns"])
        - 0.05 * demand_p90
    )
    reuse_score = (
        float(row["same_run_returns"]) * (1.0 + float(row["demand_request_ratio"]))
        + 0.01 * history_p90
    )
    pressure_score = (
        float(row["idle_pending_retained_tokens_peak"])
        + 16.0 * demand_p90
        + 30.0 * float(row["same_run_returns"])
    )
    compact_score = (
        float(row["same_run_returns"]) / entries
        + min(1.0, history_p90 / 2048.0)
        + min(1.0, float(row["idle_pending_retained_tokens_peak"]) / 12000.0)
    )
    return {
        "cold_score": cold_score,
        "reuse_score": reuse_score,
        "pressure_score": pressure_score,
        "compact_score": compact_score,
    }


def window_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["start"]), int(row["limit"])


def scan_windows(
    entries: list[sim.Entry],
    *,
    time_start: int,
    time_limit: int,
    window_sec_values: list[int],
    step_sec: int,
    block_size_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window_sec in window_sec_values:
        start = time_start
        while start + window_sec <= time_limit:
            row = summarize_window(
                entries,
                start=start,
                limit=start + window_sec,
                block_size_tokens=block_size_tokens,
            )
            row.update(row_scores(row))
            rows.append(row)
            start += max(1, step_sec)
    return rows


def evaluate_slice(
    entries: list[sim.Entry],
    row: dict[str, Any],
    *,
    max_requests: int,
    max_requests_policy: str,
    min_turns: int,
    capacity_tokens: int,
    block_size_tokens: int,
    policy: str,
) -> dict[str, Any]:
    window = sim.select_window(entries, int(row["start"]), int(row["limit"]))
    selected = cap_entries(
        window,
        max_requests=max_requests,
        policy=max_requests_policy,
        min_turns=min_turns,
    )
    selected = [replace(entry) for entry in selected]
    sim.reannotate_next_use_within_selected(selected)
    capacity_blocks = max(1, capacity_tokens // block_size_tokens)
    lru = sim.simulate_policy(
        selected,
        policy="lru",
        capacity_blocks=capacity_blocks,
        block_size_tokens=block_size_tokens,
    )
    candidate = sim.simulate_policy(
        selected,
        policy=policy,
        capacity_blocks=capacity_blocks,
        block_size_tokens=block_size_tokens,
    )
    lru_hit_ratio = float(lru["hit_ratio"])
    policy_hit_ratio = float(candidate["hit_ratio"])
    lru_miss = int(lru["miss_blocks"])
    policy_miss = int(candidate["miss_blocks"])
    out = dict(row)
    out.update(
        {
            "max_requests": max_requests,
            "max_requests_policy": max_requests_policy,
            "requests_after_cap": len(selected),
            "capacity_tokens": capacity_tokens,
            "capacity_blocks": capacity_blocks,
            "lru_hit_ratio": lru_hit_ratio,
            "policy_hit_ratio": policy_hit_ratio,
            "hit_ratio_delta": policy_hit_ratio - lru_hit_ratio,
            "lru_miss_blocks": lru_miss,
            "policy_miss_blocks": policy_miss,
            "miss_blocks_saved": lru_miss - policy_miss,
            "miss_saved_per_request": (
                (lru_miss - policy_miss) / max(1, len(selected))
            ),
            "lru_evictions": int(lru["evictions"]),
            "policy_evictions": int(candidate["evictions"]),
            "evictions_saved": int(lru["evictions"]) - int(candidate["evictions"]),
            "lru_useful_evictions": int(lru["useful_evictions"]),
            "policy_useful_evictions": int(candidate["useful_evictions"]),
        }
    )
    return out


def pick_top(
    rows: list[dict[str, Any]],
    *,
    score_name: str,
    top_k: int,
    min_entries: int,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if int(row["entries"]) >= min_entries]
    candidates.sort(
        key=lambda row: (
            float(row[score_name]),
            int(row["same_run_returns"]),
            int(row["entries"]),
        ),
        reverse=True,
    )
    return candidates[: max(1, top_k)]


def command_for(
    row: dict[str, Any],
    *,
    policy: str,
    gap_bucket_sec: float,
    fast_mode: str,
    variants: str,
) -> str:
    env = {
        "FAST_MODE": fast_mode,
        "VARIANTS": variants,
        "TIME_START": int(row["start"]),
        "TIME_LIMIT": int(row["limit"]),
        "MAX_REQUESTS": int(row["max_requests"]),
        "MAX_REQUESTS_POLICY": str(row["max_requests_policy"]),
        "EVICT_POLICY": policy,
        "EVICT_GAP_BUCKET_SEC": gap_bucket_sec,
        "EVICT_MAX_CACHED": 1,
        "EVICT_PARTIAL_OVER_CAP": 1,
        "EVICT_PROFILE_NO_CACHED_SCAN_BLOCKS": 0,
        "RUN_WORKLOAD_ANALYSIS": 0,
    }
    parts: list[str] = []
    for key, value in env.items():
        if isinstance(value, str) and " " in value:
            parts.append(f'{key}="{value}"')
        else:
            parts.append(f"{key}={value}")
    parts.append("benchmarks/interactive/H200/run_eviction_fast_iter_gpu67.sh")
    return " ".join(parts)


def add_slice_commands(
    selected: list[dict[str, Any]],
    *,
    policy: str,
    gap_bucket_sec: float,
) -> None:
    for row in selected:
        kind = str(row["slice"])
        if kind in {"nano_confirm", "micro_check", "low_confirm"}:
            fast_mode = {
                "nano_confirm": "nanokv_stress600",
                "micro_check": "microkv_session1200",
                "low_confirm": "lowkv_session2500",
            }[kind]
            variants = "lru_control eviction_only"
        else:
            fast_mode = "tinykv_quick300"
            variants = "eviction_only"
        row["gpu_command"] = command_for(
            row,
            policy=policy,
            gap_bucket_sec=gap_bucket_sec,
            fast_mode=fast_mode,
            variants=variants,
        )
        row["analyze_only_command"] = "ANALYZE_ONLY=1 " + row["gpu_command"]


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                out[f"{key}_{sub_key}"] = sub_value
        elif key.endswith("_command"):
            continue
        else:
            out[key] = value
    return out


def write_markdown(
    path: Path,
    *,
    config: dict[str, Any],
    selected: list[dict[str, Any]],
) -> None:
    lines = [
        "# Eviction Iteration Slices",
        "",
        f"- trace_dir: {config['trace_dir']}",
        f"- parts: {config['parts']}",
        f"- scan_range: [{config['time_start']}, {config['time_limit']})",
        f"- policy: {config['policy']}",
        f"- gap_bucket_sec: {config['gap_bucket_sec']}",
        f"- capacity_tokens: {config['capacity_tokens']}",
        "",
        "These slices keep the original interactive trace and only narrow the",
        "time/request budget for faster policy iteration. Use them as screens;",
        "do not claim final performance from these probes.",
        "",
        "## Selected Slices",
        "",
        "| slice | purpose | window | requests | entries | returns | new_ratio | "
        "hist_p90 | lru_miss | policy_miss | hit_delta | miss_saved/req |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['slice']} | {row['purpose']} | "
            f"[{row['start']},{row['limit']}) | {row['requests_after_cap']} | "
            f"{row['entries']} | {row['same_run_returns']} | "
            f"{float(row['new_session_ratio']):.3f} | "
            f"{float(row['history_after']['p90'] or 0.0):.0f} | "
            f"{row['lru_miss_blocks']} | {row['policy_miss_blocks']} | "
            f"{float(row['hit_ratio_delta']):.4f} | "
            f"{float(row['miss_saved_per_request']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Commands",
            "",
            "Default order: run `reuse_tiny_probe`, then `pressure_tiny_probe`, then",
            "`nano_confirm` only if the tiny probes improve cache counters and p90.",
            "",
        ]
    )
    for row in selected:
        lines.extend(
            [
                f"### {row['slice']}",
                "",
                "Analyze only:",
                "",
                "```bash",
                row["analyze_only_command"],
                "```",
                "",
                "GPU probe:",
                "",
                "```bash",
                row["gpu_command"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=sim.TRACE_DIR)
    parser.add_argument("--parts", type=int, nargs="+", default=[1])
    parser.add_argument("--time-start", type=int, default=None)
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--window-sec", type=parse_int_list, default=[45, 60, 90, 120])
    parser.add_argument("--step-sec", type=int, default=15)
    parser.add_argument("--candidate-top-k", type=int, default=24)
    parser.add_argument("--max-requests", type=int, default=300)
    parser.add_argument(
        "--max-requests-policy",
        choices=["first", "session_balanced", "real_multiturn_peak"],
        default="session_balanced",
    )
    parser.add_argument("--session-balanced-min-turns", type=int, default=3)
    parser.add_argument("--capacity-tokens", type=int, default=12000)
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument(
        "--policy",
        default="expected_gap_bucket_response_tail_aware",
    )
    parser.add_argument("--gap-bucket-sec", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sim.GAP_BUCKET_SEC = float(args.gap_bucket_sec)
    entries = sim.load_entries(args.trace_dir, args.parts)
    if not entries:
        raise SystemExit("no entries loaded")
    time_start = entries[0].timestamp if args.time_start is None else args.time_start
    time_limit = entries[-1].timestamp + 1 if args.time_limit is None else args.time_limit

    rows = scan_windows(
        entries,
        time_start=time_start,
        time_limit=time_limit,
        window_sec_values=args.window_sec,
        step_sec=args.step_sec,
        block_size_tokens=args.block_size_tokens,
    )
    candidate_keys: set[tuple[int, int]] = set()
    candidate_rows: list[dict[str, Any]] = []
    for score_name in ("cold_score", "reuse_score", "pressure_score", "compact_score"):
        for row in pick_top(
            rows,
            score_name=score_name,
            top_k=args.candidate_top_k,
            min_entries=50,
        ):
            key = (int(row["start"]), int(row["limit"]))
            if key not in candidate_keys:
                candidate_keys.add(key)
                candidate_rows.append(row)

    evaluated = [
        evaluate_slice(
            entries,
            row,
            max_requests=args.max_requests,
            max_requests_policy=args.max_requests_policy,
            min_turns=args.session_balanced_min_turns,
            capacity_tokens=args.capacity_tokens,
            block_size_tokens=args.block_size_tokens,
            policy=args.policy,
        )
        for row in candidate_rows
    ]

    def best_by(sort_key: Any, predicate: Any | None = None) -> dict[str, Any]:
        candidates = [
            row for row in evaluated
            if predicate is None or predicate(row)
        ]
        if not candidates:
            candidates = evaluated
        if not candidates:
            raise SystemExit("no candidate slices")
        return dict(sorted(candidates, key=sort_key)[0])

    cold = best_by(
        lambda row: (
            -float(row["cold_score"]),
            -int(row["lru_evictions"]),
            -int(row["entries"]),
        ),
        predicate=lambda row: (
            float(row["new_session_ratio"]) >= 0.18
            and int(row["lru_evictions"]) >= 100
        ),
    )
    cold.update(
        {
            "slice": "cold_heavy_guard",
            "purpose": "cold-heavy window that still has enough eviction signal",
        }
    )

    reuse = best_by(
        lambda row: (
            -float(row["miss_saved_per_request"]),
            -float(row["hit_ratio_delta"]),
            -int(row["same_run_returns"]),
            int(row["requests_after_cap"]),
        )
    )
    reuse.update(
        {
            "slice": "reuse_tiny_probe",
            "purpose": "smallest useful prefix-reuse screen for most policy ideas",
        }
    )
    reuse_key = window_key(reuse)

    pressure = best_by(
        lambda row: (
            -float(row["pressure_score"]),
            -int(row["lru_evictions"]),
            -int(row["lru_miss_blocks"]),
            int(row["requests_after_cap"]),
        ),
        predicate=lambda row: window_key(row) != reuse_key,
    )
    pressure.update(
        {
            "slice": "pressure_tiny_probe",
            "purpose": "stronger tiny-cache replacement pressure before nano promotion",
        }
    )

    known_tiny = evaluate_slice(
        entries,
        {
            "start": 6951,
            "limit": 7041,
            "window_sec": 90,
            **row_scores(summarize_window(entries, start=6951, limit=7041, block_size_tokens=args.block_size_tokens)),
            **summarize_window(entries, start=6951, limit=7041, block_size_tokens=args.block_size_tokens),
        },
        max_requests=args.max_requests,
        max_requests_policy=args.max_requests_policy,
        min_turns=args.session_balanced_min_turns,
        capacity_tokens=args.capacity_tokens,
        block_size_tokens=args.block_size_tokens,
        policy=args.policy,
    )
    known_tiny.update(
        {
            "slice": "known_tiny_control",
            "purpose": "keeps continuity with previous tiny calibration runs",
        }
    )

    nano = evaluate_slice(
        entries,
        {
            "start": 6675,
            "limit": 6855,
            "window_sec": 180,
            **row_scores(summarize_window(entries, start=6675, limit=6855, block_size_tokens=args.block_size_tokens)),
            **summarize_window(entries, start=6675, limit=6855, block_size_tokens=args.block_size_tokens),
        },
        max_requests=600,
        max_requests_policy=args.max_requests_policy,
        min_turns=args.session_balanced_min_turns,
        capacity_tokens=40000,
        block_size_tokens=args.block_size_tokens,
        policy=args.policy,
    )
    nano.update(
        {
            "slice": "nano_confirm",
            "purpose": "first GPU promotion lane after tiny probes pass",
        }
    )

    selected = [reuse, pressure, cold, known_tiny, nano]
    add_slice_commands(selected, policy=args.policy, gap_bucket_sec=args.gap_bucket_sec)

    config = {
        "trace_dir": str(args.trace_dir),
        "parts": args.parts,
        "time_start": time_start,
        "time_limit": time_limit,
        "window_sec": args.window_sec,
        "step_sec": args.step_sec,
        "candidate_top_k": args.candidate_top_k,
        "max_requests": args.max_requests,
        "max_requests_policy": args.max_requests_policy,
        "session_balanced_min_turns": args.session_balanced_min_turns,
        "capacity_tokens": args.capacity_tokens,
        "block_size_tokens": args.block_size_tokens,
        "policy": args.policy,
        "gap_bucket_sec": args.gap_bucket_sec,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "iteration_slices.json"
    csv_path = args.output_dir / "iteration_slice_candidates.csv"
    md_path = args.output_dir / "iteration_slices.md"
    json_path.write_text(
        json.dumps(
            {
                "config": config,
                "selected_slices": selected,
                "evaluated_candidates": evaluated,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    flat_rows = [flatten_for_csv(row) for row in evaluated]
    fieldnames = sorted({key for row in flat_rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    write_markdown(md_path, config=config, selected=selected)
    print(json_path)
    print(md_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
