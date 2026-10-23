#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRACE_DIR = Path(__file__).parent / "trace"

RESPONSE_GAP_SEC = {
    "<=16": 18.0,
    "17-32": 30.0,
    "33-64": 45.0,
    "65-128": 68.0,
    "129-256": 112.0,
    ">256": 191.0,
}


@dataclass
class Entry:
    user_id: str
    timestamp: int
    query_length: int
    response_length: int
    round_index: int
    prev_gap_sec: int | None = None
    next_gap_sec: int | None = None
    expected_next_use_ts: int | None = None
    history_tokens_before: int = 0
    history_tokens_after: int = 0
    window_history_tokens_before: int = 0
    window_history_tokens_after: int = 0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[idx]


def describe(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None}
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


def response_bucket(response_length: int) -> str:
    if response_length <= 16:
        return "<=16"
    if response_length <= 32:
        return "17-32"
    if response_length <= 64:
        return "33-64"
    if response_length <= 128:
        return "65-128"
    if response_length <= 256:
        return "129-256"
    return ">256"


def load_entries(trace_dir: Path, parts: list[int]) -> list[Entry]:
    entries: list[Entry] = []
    for part in parts:
        path = trace_dir / f"total_traces_part{part}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                line = line.strip()
                if not line:
                    continue
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
    annotate_reuse(entries)
    annotate_history_tokens(entries)
    return entries


def annotate_reuse(entries: list[Entry]) -> None:
    prev_by_user: dict[str, Entry] = {}
    for entry in entries:
        prev = prev_by_user.get(entry.user_id)
        entry.prev_gap_sec = None if prev is None else entry.timestamp - prev.timestamp
        prev_by_user[entry.user_id] = entry

    next_by_user: dict[str, Entry] = {}
    for entry in reversed(entries):
        nxt = next_by_user.get(entry.user_id)
        if nxt is None:
            entry.next_gap_sec = None
            entry.expected_next_use_ts = None
        else:
            entry.next_gap_sec = nxt.timestamp - entry.timestamp
            entry.expected_next_use_ts = nxt.timestamp
        next_by_user[entry.user_id] = entry


def annotate_history_tokens(entries: list[Entry]) -> None:
    cumulative_by_user: dict[str, int] = defaultdict(int)
    for entry in entries:
        if entry.round_index == 0:
            cumulative_by_user[entry.user_id] = 0
        before = cumulative_by_user[entry.user_id]
        after = before + entry.query_length + entry.response_length
        entry.history_tokens_before = before
        entry.history_tokens_after = after
        cumulative_by_user[entry.user_id] = after


def filter_window(entries: list[Entry], start: int, limit: int) -> list[Entry]:
    window = [e for e in entries if start <= e.timestamp < limit]
    annotate_window_history_tokens(window)
    return window


def annotate_window_history_tokens(entries: list[Entry]) -> None:
    cumulative_by_user: dict[str, int] = defaultdict(int)
    for entry in sorted(entries, key=lambda e: (e.timestamp, e.user_id, e.round_index)):
        if entry.round_index == 0:
            cumulative_by_user[entry.user_id] = 0
        before = cumulative_by_user[entry.user_id]
        after = before + entry.query_length + entry.response_length
        entry.window_history_tokens_before = before
        entry.window_history_tokens_after = after
        cumulative_by_user[entry.user_id] = after


def terminal_rank(entry: Entry) -> int:
    return 1 if entry.next_gap_sec is None else 0


def bucket_gap(entry: Entry) -> float:
    if entry.next_gap_sec is None:
        return float("inf")
    return RESPONSE_GAP_SEC[response_bucket(entry.response_length)]


def true_gap(entry: Entry) -> float:
    if entry.next_gap_sec is None:
        return float("inf")
    return float(entry.next_gap_sec)


def cost_score(entry: Entry, gap: float) -> float:
    if math.isinf(gap):
        return float("inf")
    return gap * max(1.0, float(entry.window_history_tokens_after))


def policy_scores(entry: Entry) -> dict[str, float]:
    return {
        "response_len": float("inf") if entry.next_gap_sec is None else float(entry.response_length),
        "bucket_gap": bucket_gap(entry),
        "true_gap": true_gap(entry),
        "bucket_gap_history_cost": cost_score(entry, bucket_gap(entry)),
        "true_gap_history_cost": cost_score(entry, true_gap(entry)),
    }


def top_slice(entries: list[Entry], key_name: str, fraction: float) -> dict[str, Any]:
    if not entries:
        return {}
    count = max(1, int(math.ceil(len(entries) * fraction)))
    ranked = sorted(entries, key=lambda e: policy_scores(e)[key_name], reverse=True)[:count]
    nonterminal = [e for e in ranked if e.next_gap_sec is not None]
    gaps = [e.next_gap_sec for e in nonterminal if e.next_gap_sec is not None]
    return {
        "count": count,
        "terminal_ratio": sum(1 for e in ranked if e.next_gap_sec is None) / count,
        "window_history_tokens_after": describe([e.window_history_tokens_after for e in ranked]),
        "full_trace_history_tokens_after": describe([e.history_tokens_after for e in ranked]),
        "response_length": describe([e.response_length for e in ranked]),
        "next_gap_sec_nonterminal": describe(gaps),
    }


def analyze(entries: list[Entry]) -> dict[str, Any]:
    by_user: dict[str, list[Entry]] = defaultdict(list)
    for entry in entries:
        by_user[entry.user_id].append(entry)

    window_local_returns: list[Entry] = []
    window_local_gaps: list[int] = []
    for user_entries in by_user.values():
        user_entries.sort(key=lambda e: (e.timestamp, e.round_index))
        for prev, curr in zip(user_entries, user_entries[1:]):
            window_local_returns.append(curr)
            window_local_gaps.append(curr.timestamp - prev.timestamp)

    metadata_prev_turns = [e for e in entries if e.prev_gap_sec is not None]
    next_gap_nonterminal = [e.next_gap_sec for e in entries if e.next_gap_sec is not None]
    terminal = [e for e in entries if e.next_gap_sec is None]
    response_buckets = Counter(response_bucket(e.response_length) for e in entries)

    bucket_gap_errors = []
    for entry in entries:
        if entry.next_gap_sec is None:
            continue
        bucket_gap_errors.append(bucket_gap(entry) - float(entry.next_gap_sec))

    return {
        "entries": len(entries),
        "sessions": len(by_user),
        "window_local_returns": len(window_local_returns),
        "window_local_return_ratio": len(window_local_returns) / len(entries) if entries else 0.0,
        "window_local_gap_sec": describe(window_local_gaps),
        "metadata_prev_turns": len(metadata_prev_turns),
        "metadata_prev_turn_ratio": len(metadata_prev_turns) / len(entries) if entries else 0.0,
        "terminal_turns": len(terminal),
        "terminal_ratio": len(terminal) / len(entries) if entries else 0.0,
        "turns_per_session": describe([len(v) for v in by_user.values()]),
        "round_index": describe([e.round_index for e in entries]),
        "query_length": describe([e.query_length for e in entries]),
        "response_length": describe([e.response_length for e in entries]),
        "full_trace_history_tokens_before": describe([e.history_tokens_before for e in entries]),
        "full_trace_history_tokens_after": describe([e.history_tokens_after for e in entries]),
        "window_history_tokens_before": describe([e.window_history_tokens_before for e in entries]),
        "window_history_tokens_after": describe([e.window_history_tokens_after for e in entries]),
        "next_gap_sec_nonterminal": describe([g for g in next_gap_nonterminal if g is not None]),
        "prev_gap_sec_metadata": describe([e.prev_gap_sec for e in metadata_prev_turns if e.prev_gap_sec is not None]),
        "response_buckets": dict(sorted(response_buckets.items())),
        "bucket_gap_minus_true_gap_nonterminal": describe(bucket_gap_errors),
        "top_10pct_by_policy": {
            name: top_slice(entries, name, 0.10)
            for name in (
                "response_len",
                "bucket_gap",
                "true_gap",
                "bucket_gap_history_cost",
                "true_gap_history_cost",
            )
        },
    }


def write_markdown(path: Path, config: dict[str, Any], summary: dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            if math.isinf(value):
                return "inf"
            return f"{value:.3f}"
        return str(value)

    lines = [
        "# Interactive Eviction Workload Analysis",
        "",
        f"- parts: {config['parts']}",
        f"- window: [{config['time_start']}, {config['time_limit']})",
        f"- entries: {summary['entries']}",
        f"- sessions: {summary['sessions']}",
        f"- window-local returning turns: {summary['window_local_returns']} ({summary['window_local_return_ratio']:.3f})",
        f"- metadata previous-turn coverage: {summary['metadata_prev_turns']} ({summary['metadata_prev_turn_ratio']:.3f})",
        f"- terminal turns: {summary['terminal_turns']} ({summary['terminal_ratio']:.3f})",
        "",
        "## Key Distributions",
        "",
        "| metric | n | p50 | p90 | p95 | p99 | max | mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "turns_per_session",
        "round_index",
        "query_length",
        "response_length",
        "full_trace_history_tokens_before",
        "full_trace_history_tokens_after",
        "window_history_tokens_before",
        "window_history_tokens_after",
        "next_gap_sec_nonterminal",
        "window_local_gap_sec",
        "prev_gap_sec_metadata",
        "bucket_gap_minus_true_gap_nonterminal",
    ):
        d = summary[name]
        lines.append(
            f"| {name} | {fmt(d['n'])} | {fmt(d['p50'])} | {fmt(d['p90'])} | "
            f"{fmt(d['p95'])} | {fmt(d['p99'])} | {fmt(d['max'])} | {fmt(d['mean'])} |"
        )

    lines.extend(["", "## Top 10% Candidate Slices", ""])
    for name, d in summary["top_10pct_by_policy"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- terminal_ratio: {d['terminal_ratio']:.3f}",
                f"- window_history_after_p90: {fmt(d['window_history_tokens_after']['p90'])}",
                f"- full_trace_history_after_p90: {fmt(d['full_trace_history_tokens_after']['p90'])}",
                f"- response_p90: {fmt(d['response_length']['p90'])}",
                f"- next_gap_p50_nonterminal: {fmt(d['next_gap_sec_nonterminal']['p50'])}",
                f"- next_gap_p90_nonterminal: {fmt(d['next_gap_sec_nonterminal']['p90'])}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--parts", type=int, nargs="+", default=[1])
    parser.add_argument("--time-start", type=int, default=6780)
    parser.add_argument("--time-limit", type=int, default=6840)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    entries = load_entries(args.trace_dir, args.parts)
    window = filter_window(entries, args.time_start, args.time_limit)
    summary = analyze(window)
    config = {
        "trace_dir": str(args.trace_dir),
        "parts": args.parts,
        "time_start": args.time_start,
        "time_limit": args.time_limit,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "workload_analysis.json").write_text(
        json.dumps({"config": config, "summary": summary}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "workload_analysis.md", config, summary)
    print(args.output_dir / "workload_analysis.json")
    print(args.output_dir / "workload_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
