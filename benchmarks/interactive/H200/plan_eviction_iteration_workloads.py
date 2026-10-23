#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import simulate_interactive_kv_policies as sim
from select_eviction_micro_workload import (
    cap_entries,
    cheap_window_row,
    eval_combo,
)


@dataclass(frozen=True)
class Lane:
    name: str
    purpose: str
    fast_mode: str
    variants: str
    window_sec: tuple[int, ...]
    step_sec: int
    max_requests: tuple[int, ...]
    capacity_tokens: tuple[int, ...]
    min_lru_evictions: int
    min_lru_misses: int
    window_top_k: int
    top_k: int


LANES = (
    Lane(
        name="tiny_screen",
        purpose="cheapest GPU screen for rejecting weak policy ideas",
        fast_mode="tinykv_quick300",
        variants="eviction_only",
        window_sec=(90,),
        step_sec=15,
        max_requests=(300,),
        capacity_tokens=(12000,),
        min_lru_evictions=150,
        min_lru_misses=80,
        window_top_k=8,
        top_k=5,
    ),
    Lane(
        name="nano_screen",
        purpose="short inner-loop probe for policy/cap/profile changes",
        fast_mode="nanokv_stress600",
        variants="eviction_only",
        window_sec=(120, 180),
        step_sec=15,
        max_requests=(600,),
        capacity_tokens=(40000,),
        min_lru_evictions=180,
        min_lru_misses=40,
        window_top_k=6,
        top_k=5,
    ),
    Lane(
        name="nano_pressure_metrics",
        purpose="matched nano replay that scores only the pressure-active tail",
        fast_mode="nanokv_pressure600",
        variants="lru_control eviction_only",
        window_sec=(180,),
        step_sec=15,
        max_requests=(600,),
        capacity_tokens=(40000,),
        min_lru_evictions=180,
        min_lru_misses=40,
        window_top_k=6,
        top_k=3,
    ),
    Lane(
        name="micro_check",
        purpose="matched LRU-control recheck after a nano candidate improves",
        fast_mode="microkv_session1200",
        variants="lru_control eviction_only",
        window_sec=(120, 180),
        step_sec=30,
        max_requests=(1200,),
        capacity_tokens=(80000,),
        min_lru_evictions=500,
        min_lru_misses=100,
        window_top_k=5,
        top_k=5,
    ),
    Lane(
        name="low_confirm",
        purpose="stronger pressure confirmation before claiming diagnostic gain",
        fast_mode="lowkv_session2500",
        variants="lru_control eviction_only",
        window_sec=(180,),
        step_sec=30,
        max_requests=(2500,),
        capacity_tokens=(153072,),
        min_lru_evictions=2500,
        min_lru_misses=300,
        window_top_k=5,
        top_k=5,
    ),
    Lane(
        name="standard_guard",
        purpose="standard-capacity overhead guard, not a policy-ranking cell",
        fast_mode="standard_guard60",
        variants="baseline eviction_only",
        window_sec=(60,),
        step_sec=15,
        max_requests=(0,),
        capacity_tokens=(687568,),
        min_lru_evictions=0,
        min_lru_misses=0,
        window_top_k=8,
        top_k=5,
    ),
    Lane(
        name="standard_pressure",
        purpose="standard-capacity full pressure check; expensive promotion gate",
        fast_mode="standard_pressure600",
        variants="lru_control eviction_only",
        window_sec=(600,),
        step_sec=60,
        max_requests=(0,),
        capacity_tokens=(687568,),
        min_lru_evictions=50000,
        min_lru_misses=10000,
        window_top_k=3,
        top_k=3,
    ),
)

KNOWN_CONTROL_WINDOWS = {
    "tiny_screen": ((6951, 7041),),
    "nano_screen": ((6675, 6855),),
    "nano_pressure_metrics": ((6675, 6855),),
    "micro_check": ((6996, 7176),),
    "low_confirm": ((6996, 7176),),
    "standard_guard": ((6780, 6840),),
    "standard_pressure": ((6600, 7200),),
}


def candidate_windows(
    entries: list[Any],
    *,
    time_start: int,
    time_limit: int,
    window_sec: int,
    step_sec: int,
    top_k: int,
) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    step = max(1, step_sec)
    anchors = {time_start, time_start - (time_start % step)}
    for anchor in sorted(anchors):
        start = anchor
        while start + window_sec <= time_limit:
            if start >= time_start:
                row = cheap_window_row(entries, start=start, window_sec=window_sec)
                rows_by_key[(int(row["start"]), int(row["limit"]))] = row
            start += step
    rows = list(rows_by_key.values())
    rows.sort(
        key=lambda row: (
            row["same_run_returns"],
            row["token_total"],
            row["entries"],
        ),
        reverse=True,
    )
    return rows[: max(1, top_k)]


def row_efficiency(row: dict[str, Any]) -> float:
    requests = max(1, int(row["requests"]))
    return float(row["miss_blocks_saved"]) / float(requests)


def rank_lane_rows(rows: list[dict[str, Any]], lane: Lane) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        eligible = (
            int(row["lru_evictions"]) >= lane.min_lru_evictions
            and int(row["lru_miss_blocks"]) >= lane.min_lru_misses
        )
        hit_delta = float(row["policy_hit_ratio"]) - float(row["lru_hit_ratio"])
        return (
            0 if eligible else 1,
            0 if row.get("known_control_window") else 1,
            int(row["requests"]),
            int(row["window_sec"]),
            -row_efficiency(row),
            -int(row["miss_blocks_saved"]),
            -hit_delta,
            int(row["start"]),
        )

    ranked = sorted(rows, key=key)
    for idx, row in enumerate(ranked, 1):
        row["lane_rank"] = idx
        row["eligible"] = (
            int(row["lru_evictions"]) >= lane.min_lru_evictions
            and int(row["lru_miss_blocks"]) >= lane.min_lru_misses
        )
        row["miss_saved_per_request"] = row_efficiency(row)
    return ranked[: max(1, lane.top_k)]


def evaluate_lane(
    entries: list[Any],
    lane: Lane,
    *,
    time_start: int,
    time_limit: int,
    policy: str,
    next_use_scope: str,
    max_requests_policy: str,
    min_turns: int,
    block_size_tokens: int,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for window_sec in lane.window_sec:
        windows.extend(
            candidate_windows(
                entries,
                time_start=time_start,
                time_limit=time_limit,
                window_sec=window_sec,
                step_sec=lane.step_sec,
                top_k=lane.window_top_k,
            )
        )

    rows: list[dict[str, Any]] = []
    for window in windows:
        for capacity_tokens in lane.capacity_tokens:
            for max_requests in lane.max_requests:
                row = eval_combo(
                    entries,
                    start=int(window["start"]),
                    window_sec=int(window["window_sec"]),
                    max_requests=max_requests,
                    max_requests_policy=max_requests_policy,
                    min_turns=min_turns,
                    capacity_tokens=capacity_tokens,
                    block_size_tokens=block_size_tokens,
                    policy=policy,
                    next_use_scope=next_use_scope,
                )
                row["lane"] = lane.name
                row["purpose"] = lane.purpose
                row["fast_mode"] = lane.fast_mode
                row["variants"] = lane.variants
                row["known_control_window"] = False
                rows.append(row)
    for start, limit in KNOWN_CONTROL_WINDOWS.get(lane.name, ()):
        window_sec = int(limit) - int(start)
        if start < time_start or limit > time_limit:
            continue
        if window_sec not in lane.window_sec:
            continue
        for capacity_tokens in lane.capacity_tokens:
            for max_requests in lane.max_requests:
                row = eval_combo(
                    entries,
                    start=int(start),
                    window_sec=window_sec,
                    max_requests=max_requests,
                    max_requests_policy=max_requests_policy,
                    min_turns=min_turns,
                    capacity_tokens=capacity_tokens,
                    block_size_tokens=block_size_tokens,
                    policy=policy,
                    next_use_scope=next_use_scope,
                )
                row["lane"] = lane.name
                row["purpose"] = lane.purpose
                row["fast_mode"] = lane.fast_mode
                row["variants"] = lane.variants
                row["known_control_window"] = True
                rows.append(row)
    return rank_lane_rows(rows, lane)


def command_for(row: dict[str, Any], *, policy: str, gap_bucket_sec: float) -> str:
    env = {
        "FAST_MODE": row["fast_mode"],
        "VARIANTS": row["variants"],
        "TIME_START": row["start"],
        "TIME_LIMIT": row["limit"],
        "MAX_REQUESTS": row["max_requests"],
        "MAX_REQUESTS_POLICY": row["max_requests_policy"],
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


def write_markdown(
    path: Path,
    *,
    config: dict[str, Any],
    lane_rows: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Eviction Iteration Workload Plan",
        "",
        f"- trace_dir: {config['trace_dir']}",
        f"- parts: {config['parts']}",
        f"- scan_range: [{config['time_start']}, {config['time_limit']})",
        f"- policy: {config['policy']}",
        f"- gap_bucket_sec: {config['gap_bucket_sec']}",
        f"- next_use_scope: {config['next_use_scope']}",
        "",
        "## Recommended Lanes",
        "",
        "| lane | purpose | window | requests | capacity | lru_miss | "
        "policy_miss | lru_evict | policy_evict | known | miss_saved/req |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for lane in LANES:
        rows = lane_rows.get(lane.name, [])
        if not rows:
            continue
        row = rows[0]
        lines.append(
            f"| {lane.name} | {lane.purpose} | [{row['start']},{row['limit']}) | "
            f"{row['requests']} | {row['capacity_tokens']} | "
            f"{row['lru_miss_blocks']} | {row['policy_miss_blocks']} | "
            f"{row['lru_evictions']} | {row['policy_evictions']} | "
            f"{row.get('known_control_window', False)} | "
            f"{row['miss_saved_per_request']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## GPU Commands",
            "",
            "Run only the first command for inner-loop probes. Promote to the next "
            "lane only after the previous lane improves cache stats and latency "
            "without hurting continuity.",
            "",
            "Promotion gates:",
            "",
            "- `tiny_screen` is a rejection screen only; any apparent win must repeat "
            "on `nano_screen` before it changes the working best policy.",
            "- Promote from `nano_screen` to `nano_pressure_metrics` only when full-window "
            "p90 latency or throughput improves and LRU recent-miss/cache-hit counters "
            "do not regress.",
            "- Promote from `nano_pressure_metrics` to `micro_check` only when the "
            "pressure-active tail also improves; this prevents early/cold no-pressure "
            "requests from creating false positives.",
            "- Promote from `micro_check` to `low_confirm` only when the candidate beats "
            "the current best micro result on at least two of rps, goodput, audio p90, "
            "and e2e p90 without continuity loss.",
            "- Use `standard_guard` only as an overhead/regression guard; the available "
            "standard-capacity window has little replacement pressure.",
            "- Use `standard_pressure` only after a candidate has survived the low-KV "
            "lanes; it is the shortest standard-capacity window that reproduces "
            "replacement pressure in the current trace scan.",
            "",
        ]
    )
    for lane in LANES:
        rows = lane_rows.get(lane.name, [])
        if not rows:
            continue
        row = rows[0]
        lines.extend(
            [
                f"### {lane.name}",
                "",
                "```bash",
                command_for(
                    row,
                    policy=str(config["policy"]),
                    gap_bucket_sec=float(config["gap_bucket_sec"]),
                ),
                "```",
                "",
            ]
        )

    lines.extend(["## Ranked Candidates", ""])
    for lane in LANES:
        rows = lane_rows.get(lane.name, [])
        if not rows:
            continue
        lines.extend(
            [
                f"### {lane.name}",
                "",
                "| rank | eligible | known | window | requests | lru_miss | policy_miss | "
                "lru_evict | policy_evict | hit_delta | miss_saved/req |",
                "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            hit_delta = float(row["policy_hit_ratio"]) - float(row["lru_hit_ratio"])
            lines.append(
                f"| {row['lane_rank']} | {row['eligible']} | "
                f"{row.get('known_control_window', False)} | "
                f"[{row['start']},{row['limit']}) | {row['requests']} | "
                f"{row['lru_miss_blocks']} | {row['policy_miss_blocks']} | "
                f"{row['lru_evictions']} | {row['policy_evictions']} | "
                f"{hit_delta:.4f} | {row['miss_saved_per_request']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=sim.TRACE_DIR)
    parser.add_argument("--parts", type=int, nargs="+", default=[1])
    parser.add_argument("--time-start", type=int, default=None)
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument(
        "--policy",
        default="expected_gap_bucket_response_tail_aware",
    )
    parser.add_argument("--gap-bucket-sec", type=float, default=30.0)
    parser.add_argument(
        "--next-use-scope",
        choices=["full_trace", "selected"],
        default="selected",
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
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sim.GAP_BUCKET_SEC = float(args.gap_bucket_sec)
    entries = sim.load_entries(args.trace_dir, args.parts)
    if not entries:
        raise SystemExit("no entries loaded")
    time_start = entries[0].timestamp if args.time_start is None else args.time_start
    time_limit = entries[-1].timestamp + 1 if args.time_limit is None else args.time_limit

    lane_rows: dict[str, list[dict[str, Any]]] = {}
    for lane in LANES:
        lane_rows[lane.name] = evaluate_lane(
            entries,
            lane,
            time_start=time_start,
            time_limit=time_limit,
            policy=args.policy,
            next_use_scope=args.next_use_scope,
            max_requests_policy=args.max_requests_policy,
            min_turns=args.session_balanced_min_turns,
            block_size_tokens=args.block_size_tokens,
        )

    config = {
        "trace_dir": str(args.trace_dir),
        "parts": args.parts,
        "time_start": time_start,
        "time_limit": time_limit,
        "policy": args.policy,
        "gap_bucket_sec": args.gap_bucket_sec,
        "next_use_scope": args.next_use_scope,
        "max_requests_policy": args.max_requests_policy,
        "session_balanced_min_turns": args.session_balanced_min_turns,
        "block_size_tokens": args.block_size_tokens,
    }
    output = {
        "config": config,
        "lanes": lane_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "iteration_workload_plan.json"
    md_path = args.output_dir / "iteration_workload_plan.md"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, config=config, lane_rows=lane_rows)
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
