#!/usr/bin/env python3
"""Search online-feasible gates using both request deltas and applied events.

`search_eviction_admission_gates.py` splices per-request latency rows.  That is
useful, but an online eviction gate is evaluated at allocation/enforcement time,
so it may block every real replacement even when the request-level splice looks
positive.  This helper adds the missing event-level view from
`kv_eviction_shadow.jsonl`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Gate:
    label: str
    func: Callable[[dict[str, Any]], bool]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def trace_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("user_id"),
        row.get("round_index"),
        row.get("timestamp"),
        row.get("query_length"),
        row.get("response_length"),
    )


def read_requests(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("error") is not None or row.get("status") != 200:
            continue
        rows[trace_key(row)] = row
    by_user: dict[Any, list[dict[str, Any]]] = {}
    for row in rows.values():
        by_user.setdefault(row.get("user_id"), []).append(row)
    for user_rows in by_user.values():
        user_rows.sort(
            key=lambda row: (
                finite(row.get("timestamp"), 0.0) or 0.0,
                finite(row.get("round_index"), 0.0) or 0.0,
            )
        )
        for idx, row in enumerate(user_rows):
            if idx + 1 < len(user_rows):
                cur_ts = finite(row.get("timestamp"), 0.0) or 0.0
                next_ts = finite(user_rows[idx + 1].get("timestamp"), 0.0) or 0.0
                row["_expected_gap"] = max(0.0, next_ts - cur_ts)
                row["_is_terminal"] = False
            else:
                row["_expected_gap"] = float("inf")
                row["_is_terminal"] = True
    return rows


def finite(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[idx]


def pct_reduction(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return (baseline - candidate) / baseline * 100.0


def request_features(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": finite(row.get("timestamp"), 0.0) or 0.0,
        "round_index": finite(row.get("round_index"), 0.0) or 0.0,
        "response_length": finite(row.get("response_length"), 0.0) or 0.0,
        "query_length": finite(row.get("query_length"), 0.0) or 0.0,
        "prefix_tokens": finite(row.get("kv_prefix_token_estimate"), 0.0) or 0.0,
        "audio_duration": finite(row.get("audio_duration_sec"), 0.0) or 0.0,
        "expected_gap": finite(row.get("_expected_gap"), 0.0) or 0.0,
        "is_terminal": bool(row.get("_is_terminal")),
    }


def event_features(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": finite(meta.get("omni_trace_timestamp"), 0.0) or 0.0,
        "round_index": finite(meta.get("omni_round_index"), 0.0) or 0.0,
        "response_length": finite(meta.get("omni_response_length"), 0.0) or 0.0,
        "query_length": finite(meta.get("omni_query_length"), 0.0) or 0.0,
        "prefix_tokens": finite(meta.get("omni_prefix_token_estimate"), 0.0) or 0.0,
        "expected_gap": (
            float("inf")
            if meta.get("omni_is_terminal_turn")
            else (finite(meta.get("omni_expected_next_gap_sec"), 0.0) or 0.0)
        ),
        "is_terminal": bool(meta.get("omni_is_terminal_turn")),
    }


def make_gates() -> list[Gate]:
    gates: list[Gate] = [Gate("all", lambda row: True)]
    cuts = {
        "timestamp": [6400, 6600, 6800, 6900, 7040, 7120, 7200, 7400],
        "round_index": [1, 3, 5, 10, 15, 20],
        "response_length": [16, 32, 48, 64, 96, 128],
        "query_length": [16, 32, 64, 96, 128],
        "prefix_tokens": [1, 64, 128, 192, 256, 512, 1024],
        "expected_gap": [30, 45, 60, 90, 120, 180],
    }
    for feature, values in cuts.items():
        for cut in values:
            gates.append(
                Gate(
                    f"{feature}>={cut:g}",
                    lambda row, f=feature, c=cut: float(row.get(f, 0.0)) >= c,
                )
            )
            gates.append(
                Gate(
                    f"{feature}<{cut:g}",
                    lambda row, f=feature, c=cut: float(row.get(f, 0.0)) < c,
                )
            )

    ranges = [(6600, 6900), (6900, 7040), (7040, 7120), (7120, 7743)]
    for lo, hi in ranges:
        gates.append(
            Gate(
                f"{lo:g}<=timestamp<{hi:g}",
                lambda row, lo=lo, hi=hi: lo <= float(row.get("timestamp", 0.0)) < hi,
            )
        )
        gates.append(
            Gate(
                f"not({lo:g}<=timestamp<{hi:g})",
                lambda row, lo=lo, hi=hi: not (
                    lo <= float(row.get("timestamp", 0.0)) < hi
                ),
            )
        )

    # Small set of conjunctions that map directly to existing env gates.
    for ts in [6900, 7040, 7120, 7200]:
        for resp in [32, 48, 64, 96]:
            gates.append(
                Gate(
                    f"timestamp>={ts:g}&response_length>={resp:g}",
                    lambda row, ts=ts, resp=resp: float(row.get("timestamp", 0.0)) >= ts
                    and float(row.get("response_length", 0.0)) >= resp,
                )
            )
    for ts in [6900, 7040, 7120]:
        for rnd in [15, 20]:
            gates.append(
                Gate(
                    f"timestamp>={ts:g}&round_index>={rnd:g}",
                    lambda row, ts=ts, rnd=rnd: float(row.get("timestamp", 0.0)) >= ts
                    and float(row.get("round_index", 0.0)) >= rnd,
                )
            )
    return gates


def find_current_request(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        current = obj.get("current_request")
        if isinstance(current, dict):
            return current
        extra = obj.get("extra")
        if isinstance(extra, dict):
            current = extra.get("current_request")
            if isinstance(current, dict):
                return current
        for value in obj.values():
            found = find_current_request(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_current_request(value)
            if found:
                return found
    return {}


def read_applied_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        if row.get("reason") != "applied" and row.get("applied") is not True:
            continue
        current = find_current_request(row)
        if current:
            events.append(event_features(current))
    return events


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    audio = [v for row in rows if (v := finite(row.get("audio_ttfp_sec"))) is not None]
    e2e = [v for row in rows if (v := finite(row.get("e2e_sec"))) is not None]
    ttft = [v for row in rows if (v := finite(row.get("ttfp_sec"))) is not None]
    continuity = [
        bool(row.get("audio_continuity_ok"))
        for row in rows
        if row.get("audio_continuity_evaluated") is True
    ]
    return {
        "audio_p90": percentile(audio, 0.90),
        "e2e_p90": percentile(e2e, 0.90),
        "ttft_p90": percentile(ttft, 0.90),
        "continuity": sum(continuity) / len(continuity) if continuity else None,
    }


def evaluate_gate(
    gate: Gate,
    baseline: dict[tuple[Any, ...], dict[str, Any]],
    candidate: dict[tuple[Any, ...], dict[str, Any]],
    keys: list[tuple[Any, ...]],
    baseline_metrics: dict[str, float | None],
    applied_events: list[dict[str, Any]],
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    candidate_requests = 0
    for key in keys:
        brow = baseline[key]
        if gate.func(request_features(brow)):
            selected.append(candidate[key])
            candidate_requests += 1
        else:
            selected.append(brow)
    metrics = summarize_rows(selected)
    covered_events = sum(1 for event in applied_events if gate.func(event))
    total_events = len(applied_events)
    audio_gain = pct_reduction(metrics["audio_p90"], baseline_metrics["audio_p90"])
    e2e_gain = pct_reduction(metrics["e2e_p90"], baseline_metrics["e2e_p90"])
    ttft_gain = pct_reduction(metrics["ttft_p90"], baseline_metrics["ttft_p90"])
    continuity_gain = (
        (metrics["continuity"] - baseline_metrics["continuity"]) * 100.0
        if isinstance(metrics["continuity"], (int, float))
        and isinstance(baseline_metrics["continuity"], (int, float))
        else None
    )
    event_coverage = covered_events / total_events if total_events else 0.0
    score = (
        0.45 * float(audio_gain or 0.0)
        + 0.20 * float(e2e_gain or 0.0)
        + 0.05 * float(ttft_gain or 0.0)
        + 0.10 * float(continuity_gain or 0.0)
        + 0.20 * (event_coverage * 100.0)
    )
    return {
        "gate": gate.label,
        "candidate_requests": candidate_requests,
        "applied_events_covered": covered_events,
        "applied_events_total": total_events,
        "event_coverage": event_coverage,
        **metrics,
        "audio_gain_pct": audio_gain,
        "e2e_gain_pct": e2e_gain,
        "ttft_gain_pct": ttft_gain,
        "continuity_gain_pp": continuity_gain,
        "score": score,
    }


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "event_gate_search.json"
    csv_path = output_dir / "event_gate_search.csv"
    md_path = output_dir / "event_gate_search.md"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Eviction Event Gate Search",
        "",
        "This ranks online-feasible gates using request-level splice metrics plus "
        "coverage of actual `applied` replacement events from `kv_eviction_shadow.jsonl`.",
        "",
        "| rank | gate | reqs | applied coverage | audio p90 | audio gain | e2e gain | ttft gain | score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows[:40], 1):
        lines.append(
            f"| {idx} | `{row['gate']}` | {row['candidate_requests']} | "
            f"{row['applied_events_covered']}/{row['applied_events_total']} "
            f"({row['event_coverage'] * 100:.1f}%) | "
            f"{row['audio_p90']:.3f}s | {row['audio_gain_pct']:+.3f}% | "
            f"{row['e2e_gain_pct']:+.3f}% | {row['ttft_gain_pct']:+.3f}% | "
            f"{row['score']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(json_path)
    print(csv_path)
    print(md_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = read_requests(args.baseline)
    candidate = read_requests(args.candidate)
    keys = [key for key in baseline.keys() if key in candidate]
    baseline_metrics = summarize_rows([baseline[key] for key in keys])
    applied_events = read_applied_events(args.shadow)
    rows = [
        evaluate_gate(gate, baseline, candidate, keys, baseline_metrics, applied_events)
        for gate in make_gates()
    ]
    rows.sort(
        key=lambda row: (
            row["score"],
            row["audio_gain_pct"] or -999.0,
            row["event_coverage"],
        ),
        reverse=True,
    )
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
