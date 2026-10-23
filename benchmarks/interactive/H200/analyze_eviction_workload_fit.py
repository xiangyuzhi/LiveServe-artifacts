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
from select_eviction_micro_workload import cap_entries


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
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
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    xs = [float(value) for value in values]
    return {
        "n": len(xs),
        "p50": percentile(xs, 0.50),
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
        "max": max(xs),
        "mean": statistics.mean(xs),
    }


def selected_entries(
    entries: list[sim.Entry],
    *,
    time_start: int,
    time_limit: int,
    max_requests: int,
    cap_policy: str,
    min_turns: int,
) -> list[sim.Entry]:
    window = sim.select_window(entries, time_start, time_limit)
    selected = cap_entries(
        window,
        max_requests=max_requests,
        policy=cap_policy,
        min_turns=min_turns,
    )
    selected = [replace(entry) for entry in selected]
    sim.reannotate_next_use_within_selected(selected)
    return selected


def summarize_workload(entries: list[sim.Entry]) -> dict[str, Any]:
    by_user: dict[str, list[sim.Entry]] = defaultdict(list)
    for entry in entries:
        by_user[entry.user_id].append(entry)
    for user_entries in by_user.values():
        user_entries.sort(key=lambda item: (item.timestamp, item.round_index))

    turns = [len(user_entries) for user_entries in by_user.values()]
    round_indexes = [entry.round_index for entry in entries]
    query_lengths = [entry.query_length for entry in entries]
    response_lengths = [entry.response_length for entry in entries]
    next_gaps = [
        entry.next_gap_sec
        for entry in entries
        if entry.next_gap_sec is not None
    ]

    history_after: list[int] = []
    for user_entries in by_user.values():
        total = 0
        for entry in user_entries:
            total += int(entry.query_length) + int(entry.response_length)
            history_after.append(total)

    span = (
        max(entry.timestamp for entry in entries) - min(entry.timestamp for entry in entries)
        if len(entries) >= 2
        else 0
    )
    sessions = len(by_user)
    requests = len(entries)
    return {
        "requests": requests,
        "sessions": sessions,
        "same_run_return_ratio": (requests - sessions) / requests if requests else 0.0,
        "terminal_ratio": sessions / requests if requests else 0.0,
        "arrival_span_sec": span,
        "trace_request_rate": requests / span if span > 0 else None,
        "turns_per_session": describe(turns),
        "round_index": describe(round_indexes),
        "query_length": describe(query_lengths),
        "response_length": describe(response_lengths),
        "window_history_tokens_after": describe(history_after),
        "next_gap_sec_nonterminal": describe([gap for gap in next_gaps if gap is not None]),
    }


def safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def relative_error(value: Any, reference: Any) -> float:
    lhs = safe_float(value)
    rhs = safe_float(reference)
    if lhs is None or rhs is None or rhs == 0.0:
        return 0.0
    return abs(lhs - rhs) / abs(rhs)


def workload_fit_score(summary: dict[str, Any], reference: dict[str, Any]) -> float:
    turns = summary["turns_per_session"]
    ref_turns = reference["turns_per_session"]
    round_index = summary["round_index"]
    ref_round = reference["round_index"]
    return 100.0 * (
        0.18 * relative_error(turns.get("mean"), ref_turns.get("mean"))
        + 0.18 * relative_error(turns.get("p50"), ref_turns.get("p50"))
        + 0.18 * relative_error(turns.get("p90"), ref_turns.get("p90"))
        + 0.10 * relative_error(turns.get("p99"), ref_turns.get("p99"))
        + 0.12 * relative_error(round_index.get("p90"), ref_round.get("p90"))
        + 0.12 * relative_error(summary.get("same_run_return_ratio"), reference.get("same_run_return_ratio"))
        + 0.06 * relative_error(summary["response_length"].get("p90"), reference["response_length"].get("p90"))
        + 0.06 * relative_error(summary["query_length"].get("p90"), reference["query_length"].get("p90"))
    )


def simulate_cache_surface(
    entries: list[sim.Entry],
    *,
    capacity_tokens: int,
    block_size_tokens: int,
    policy: str,
) -> dict[str, Any]:
    capacity_blocks = max(1, capacity_tokens // block_size_tokens)
    lru = sim.simulate_policy(
        entries,
        policy="lru",
        capacity_blocks=capacity_blocks,
        block_size_tokens=block_size_tokens,
    )
    candidate = sim.simulate_policy(
        entries,
        policy=policy,
        capacity_blocks=capacity_blocks,
        block_size_tokens=block_size_tokens,
    )
    requests = max(1, len(entries))
    return {
        "capacity_tokens": capacity_tokens,
        "capacity_blocks": capacity_blocks,
        "lru_hit_ratio": float(lru["hit_ratio"]),
        "policy_hit_ratio": float(candidate["hit_ratio"]),
        "lru_miss_blocks": int(lru["miss_blocks"]),
        "policy_miss_blocks": int(candidate["miss_blocks"]),
        "miss_blocks_saved": int(lru["miss_blocks"]) - int(candidate["miss_blocks"]),
        "miss_saved_per_request": (
            int(lru["miss_blocks"]) - int(candidate["miss_blocks"])
        ) / requests,
        "lru_evictions": int(lru["evictions"]),
        "policy_evictions": int(candidate["evictions"]),
    }


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    summary = row["summary"]
    turns = summary["turns_per_session"]
    round_index = summary["round_index"]
    cache = row.get("cache_surface") or {}
    return {
        "cap_policy": row["cap_policy"],
        "max_requests": row["max_requests"],
        "requests": summary["requests"],
        "sessions": summary["sessions"],
        "mean_turns": turns.get("mean"),
        "p50_turns": turns.get("p50"),
        "p90_turns": turns.get("p90"),
        "p99_turns": turns.get("p99"),
        "round_p90": round_index.get("p90"),
        "same_run_return_ratio": summary["same_run_return_ratio"],
        "fit_score": row["fit_score"],
        "capacity_tokens": cache.get("capacity_tokens"),
        "lru_miss_blocks": cache.get("lru_miss_blocks"),
        "policy_miss_blocks": cache.get("policy_miss_blocks"),
        "miss_blocks_saved": cache.get("miss_blocks_saved"),
        "miss_saved_per_request": cache.get("miss_saved_per_request"),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.{digits}f}"
    if value is None:
        return "-"
    return str(value)


def write_markdown(path: Path, config: dict[str, Any], reference: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    ref_turns = reference["turns_per_session"]
    lines = [
        "# Eviction Workload Fit",
        "",
        f"- trace_dir: {config['trace_dir']}",
        f"- parts: {config['parts']}",
        f"- window: [{config['time_start']}, {config['time_limit']})",
        f"- policy_for_cache_surface: {config['policy']}",
        "",
        "## Reference",
        "",
        f"- requests: {reference['requests']}",
        f"- sessions: {reference['sessions']}",
        f"- turns/session mean/p50/p90/p99/max: "
        f"{fmt(ref_turns['mean'])}/{fmt(ref_turns['p50'])}/"
        f"{fmt(ref_turns['p90'])}/{fmt(ref_turns['p99'])}/{fmt(ref_turns['max'])}",
        f"- same-run return ratio: {fmt(reference['same_run_return_ratio'])}",
        "",
        "## Candidates",
        "",
        "| rank | cap policy | cap | requests | sessions | turns mean | turns p50 | "
        "turns p90 | same-return | fit score | lru miss | policy miss | miss saved/req |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        flat = flatten_row(row)
        lines.append(
            f"| {rank} | {flat['cap_policy']} | {flat['max_requests']} | "
            f"{flat['requests']} | {flat['sessions']} | "
            f"{fmt(flat['mean_turns'])} | {fmt(flat['p50_turns'])} | "
            f"{fmt(flat['p90_turns'])} | {fmt(flat['same_run_return_ratio'])} | "
            f"{fmt(flat['fit_score'])} | {fmt(flat['lru_miss_blocks'])} | "
            f"{fmt(flat['policy_miss_blocks'])} | "
            f"{fmt(flat['miss_saved_per_request'])} |"
        )
    lines.extend(
        [
            "",
            "Lower `fit score` means closer to the reference workload distribution. "
            "Use this table as a guardrail: fast tuning cells may be smaller, but "
            "promotion should happen on the best low-score full-session lane.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=sim.TRACE_DIR)
    parser.add_argument("--parts", type=int, nargs="+", default=[1])
    parser.add_argument("--time-start", type=int, default=0)
    parser.add_argument("--time-limit", type=int, default=20000)
    parser.add_argument(
        "--max-requests",
        type=parse_int_list,
        default=[1200, 2400, 3600, 5000, 7200, 10000],
    )
    parser.add_argument(
        "--cap-policy",
        choices=[
            "first",
            "session_balanced",
            "real_multiturn_peak",
            "real_multiturn_full_sessions",
            "real_multiturn_full_sessions_peak",
        ],
        action="append",
        default=None,
    )
    parser.add_argument("--session-balanced-min-turns", type=int, default=2)
    parser.add_argument("--capacity-tokens", type=int, default=80000)
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument("--policy", default="expected_gap_tail_aware")
    parser.add_argument("--skip-cache-sim", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cap_policies = args.cap_policy or ["real_multiturn_full_sessions"]
    all_entries = sim.load_entries(args.trace_dir, args.parts)
    reference_entries = selected_entries(
        all_entries,
        time_start=args.time_start,
        time_limit=args.time_limit,
        max_requests=0,
        cap_policy="first",
        min_turns=args.session_balanced_min_turns,
    )
    reference = summarize_workload(reference_entries)

    rows: list[dict[str, Any]] = []
    for cap_policy in cap_policies:
        for max_requests in args.max_requests:
            selected = selected_entries(
                all_entries,
                time_start=args.time_start,
                time_limit=args.time_limit,
                max_requests=max_requests,
                cap_policy=cap_policy,
                min_turns=args.session_balanced_min_turns,
            )
            summary = summarize_workload(selected)
            row = {
                "cap_policy": cap_policy,
                "max_requests": max_requests,
                "summary": summary,
                "fit_score": workload_fit_score(summary, reference),
            }
            if not args.skip_cache_sim:
                row["cache_surface"] = simulate_cache_surface(
                    selected,
                    capacity_tokens=args.capacity_tokens,
                    block_size_tokens=args.block_size_tokens,
                    policy=args.policy,
                )
            rows.append(row)

    rows.sort(
        key=lambda row: (
            float(row["fit_score"]),
            -int(row["summary"]["requests"]),
            row["cap_policy"],
        )
    )
    config = {
        "trace_dir": str(args.trace_dir),
        "parts": args.parts,
        "time_start": args.time_start,
        "time_limit": args.time_limit,
        "max_requests": args.max_requests,
        "cap_policy": cap_policies,
        "session_balanced_min_turns": args.session_balanced_min_turns,
        "capacity_tokens": args.capacity_tokens,
        "block_size_tokens": args.block_size_tokens,
        "policy": args.policy,
        "skip_cache_sim": args.skip_cache_sim,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "workload_fit.json"
    csv_path = args.output_dir / "workload_fit.csv"
    md_path = args.output_dir / "workload_fit.md"
    json_path.write_text(
        json.dumps(
            {"config": config, "reference": reference, "candidates": rows},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    flat_rows = [flatten_row(row) for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    write_markdown(md_path, config, reference, rows)
    print(json_path)
    print(csv_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
