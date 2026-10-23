#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SUMMARY_KEYS = (
    "response_length",
    "expected_gap_sec_finite",
    "gap_bucket_finite",
    "block_depth_ratio",
    "lost_tokens_if_prefix_break",
    "suffix_blocks_including",
    "history_tokens_after",
)

GAP_THRESHOLDS = (15.0, 30.0, 60.0)
DELTA_THRESHOLDS = (32.0, 64.0, 128.0, 192.0)


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


def summary_metric(summary: dict[str, Any], key: str, qkey: str = "p50") -> float | None:
    value = summary.get(key)
    if not isinstance(value, dict):
        return None
    raw = value.get(qkey)
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def add_summary_values(
    out: dict[str, list[float]],
    prefix: str,
    summary: dict[str, Any],
) -> None:
    for key in SUMMARY_KEYS:
        for qkey in ("p50", "p90", "max"):
            value = summary_metric(summary, key, qkey)
            if value is not None:
                out[f"{prefix}.{key}.{qkey}"].append(value)
    terminal = summary.get("terminal")
    count = summary.get("count")
    if isinstance(terminal, (int, float)):
        out[f"{prefix}.terminal_count"].append(float(terminal))
    if isinstance(count, (int, float)) and count > 0 and isinstance(terminal, (int, float)):
        out[f"{prefix}.terminal_ratio"].append(float(terminal) / float(count))


def add_exchange_flags(
    out: dict[str, list[float]],
    actual: dict[str, Any],
    selected: dict[str, Any],
) -> None:
    actual_gap = summary_metric(actual, "expected_gap_sec_finite", "p50")
    actual_response = summary_metric(actual, "response_length", "p50")
    actual_history = summary_metric(actual, "history_tokens_after", "p50")
    actual_lost = summary_metric(actual, "lost_tokens_if_prefix_break", "p50")
    selected_response = summary_metric(selected, "response_length", "p50")
    selected_history = summary_metric(selected, "history_tokens_after", "p50")
    selected_lost = summary_metric(selected, "lost_tokens_if_prefix_break", "p50")

    def add_flag(name: str, passed: bool) -> None:
        out[f"flag.{name}"].append(1.0 if passed else 0.0)

    if actual_gap is not None:
        for threshold in GAP_THRESHOLDS:
            add_flag(f"actual_gap_le_{int(threshold)}", actual_gap <= threshold)
            add_flag(f"actual_gap_gt_{int(threshold)}", actual_gap > threshold)
    if actual_response is not None:
        for threshold in (16.0, 32.0, 64.0):
            add_flag(
                f"actual_response_le_{int(threshold)}",
                actual_response <= threshold,
            )
    response_delta = None
    history_delta = None
    lost_delta = None
    if actual_response is not None and selected_response is not None:
        response_delta = selected_response - actual_response
        for threshold in DELTA_THRESHOLDS:
            add_flag(f"response_delta_ge_{int(threshold)}", response_delta >= threshold)
    if actual_history is not None and selected_history is not None:
        history_delta = selected_history - actual_history
        for threshold in DELTA_THRESHOLDS:
            add_flag(f"history_delta_ge_{int(threshold)}", history_delta >= threshold)
    if actual_lost is not None and selected_lost is not None:
        lost_delta = selected_lost - actual_lost
        for threshold in DELTA_THRESHOLDS:
            add_flag(f"lost_delta_ge_{int(threshold)}", lost_delta >= threshold)

    if response_delta is not None and history_delta is not None:
        add_flag(
            "response_ge64_and_history_ge128",
            response_delta >= 64.0 and history_delta >= 128.0,
        )
        add_flag(
            "response_ge64_or_history_ge192",
            response_delta >= 64.0 or history_delta >= 192.0,
        )
    if response_delta is not None and lost_delta is not None:
        add_flag(
            "response_ge64_and_lost_ge32",
            response_delta >= 64.0 and lost_delta >= 32.0,
        )
    if actual_gap is not None and actual_response is not None:
        add_flag(
            "actual_very_near_low_response",
            actual_gap <= 15.0 and actual_response <= 16.0,
        )
        add_flag(
            "actual_near_low_response",
            actual_gap <= 30.0 and actual_response <= 32.0,
        )


def analyze_shadow(path: Path) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    numeric: dict[str, list[float]] = defaultdict(list)
    policy: str | None = None
    events = 0
    applied = 0

    for row in iter_jsonl(path):
        events += 1
        if policy is None and row.get("policy"):
            policy = str(row.get("policy"))
        reason = str(row.get("reason") or "none")
        reasons[reason] += 1
        is_applied = row.get("applied") is True or reason == "applied"
        if is_applied:
            applied += 1
        for key in (
            "cached_victim_count",
            "replacement_count",
            "selected_count",
            "num_requested_blocks",
            "elapsed_ns",
            "candidate_count",
            "metadata_ratio",
            "attempts_since_last_cached_victim",
            "attempts_since_last_applied",
            "no_cached_victim_burst",
            "recent_cached_victim_attempts_64",
            "recent_applied_attempts_64",
            "recent_no_cached_victim_attempts_64",
            "cached_victim_density_64",
            "applied_density_64",
            "no_cached_victim_density_64",
        ):
            value = row.get(key)
            if isinstance(value, (int, float)):
                numeric[key].append(float(value))
        pressure_phase = row.get("pressure_phase")
        if isinstance(pressure_phase, str) and pressure_phase:
            numeric[f"pressure_phase.{pressure_phase}"].append(1.0)
        if not is_applied:
            continue

        actual = row.get("actual_lru_victim_summary")
        selected = row.get("selected_victim_summary")
        if isinstance(actual, dict):
            add_summary_values(numeric, "actual", actual)
        if isinstance(selected, dict):
            add_summary_values(numeric, "selected", selected)
        if isinstance(actual, dict) and isinstance(selected, dict):
            for key in SUMMARY_KEYS:
                a = summary_metric(actual, key, "p50")
                s = summary_metric(selected, key, "p50")
                if a is not None and s is not None:
                    numeric[f"delta.selected_minus_actual.{key}.p50"].append(s - a)
            add_exchange_flags(numeric, actual, selected)

    return {
        "path": str(path),
        "policy": policy,
        "events": events,
        "applied": applied,
        "applied_ratio": applied / events if events else 0.0,
        "reasons": reasons.most_common(),
        "metrics": {key: describe(values) for key, values in sorted(numeric.items())},
    }


def find_shadow_path(root_or_path: Path) -> Path:
    if root_or_path.is_file():
        return root_or_path
    matches = sorted(root_or_path.glob("*/_probe/kv_eviction_shadow.jsonl"))
    if not matches:
        matches = sorted(root_or_path.glob("**/kv_eviction_shadow.jsonl"))
    if not matches:
        raise FileNotFoundError(f"no kv_eviction_shadow.jsonl under {root_or_path}")
    return matches[0]


def parse_run(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), Path(path)
    path = Path(raw)
    return path.name, path


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def metric(summary: dict[str, Any], name: str, qkey: str = "p50") -> Any:
    return summary.get("metrics", {}).get(name, {}).get(qkey)


def write_markdown(path: Path, runs: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Eviction Shadow Profile Analysis",
        "",
        "| run | policy | events | applied | applied_ratio | actual_gap_p50 | "
        "actual_resp_p50 | actual_hist_p50 | selected_gap_p50 | "
        "selected_resp_p50 | selected_hist_p50 | selected_terminal |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in runs.items():
        lines.append(
            f"| {label} | {summary.get('policy')} | {summary['events']} | "
            f"{summary['applied']} | {summary['applied_ratio']:.3f} | "
            f"{fmt(metric(summary, 'actual.expected_gap_sec_finite.p50'))} | "
            f"{fmt(metric(summary, 'actual.response_length.p50'))} | "
            f"{fmt(metric(summary, 'actual.history_tokens_after.p50'))} | "
            f"{fmt(metric(summary, 'selected.expected_gap_sec_finite.p50'))} | "
            f"{fmt(metric(summary, 'selected.response_length.p50'))} | "
            f"{fmt(metric(summary, 'selected.history_tokens_after.p50'))} | "
            f"{fmt(metric(summary, 'selected.terminal_ratio'))} |"
        )

    lines.extend(["", "## Selected Metric Details", ""])
    detail_metrics = (
        "actual.expected_gap_sec_finite.p50",
        "actual.response_length.p50",
        "actual.history_tokens_after.p50",
        "actual.lost_tokens_if_prefix_break.p50",
        "selected.response_length.p50",
        "selected.history_tokens_after.p50",
        "selected.lost_tokens_if_prefix_break.p50",
        "delta.selected_minus_actual.response_length.p50",
        "delta.selected_minus_actual.history_tokens_after.p50",
        "delta.selected_minus_actual.lost_tokens_if_prefix_break.p50",
        "cached_victim_count",
        "replacement_count",
        "no_cached_victim_burst",
        "attempts_since_last_cached_victim",
        "attempts_since_last_applied",
        "cached_victim_density_64",
        "applied_density_64",
        "no_cached_victim_density_64",
    )
    lines.extend(
        [
            "| run | metric | n | p50 | p90 | p99 | mean |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, summary in runs.items():
        for name in detail_metrics:
            d = summary.get("metrics", {}).get(name)
            if not d or d.get("n", 0) == 0:
                continue
            lines.append(
                f"| {label} | {name} | {d['n']} | {fmt(d['p50'])} | "
                f"{fmt(d['p90'])} | {fmt(d['p99'])} | {fmt(d['mean'])} |"
            )
    lines.extend(
        [
            "",
            "## Exchange Flag Rates",
            "",
            "| run | flag | n | pass_rate |",
            "|---|---|---:|---:|",
        ]
    )
    for label, summary in runs.items():
        for name, d in sorted(summary.get("metrics", {}).items()):
            if not name.startswith("flag."):
                continue
            if not d or d.get("n", 0) == 0:
                continue
            lines.append(
                f"| {label} | {name.removeprefix('flag.')} | "
                f"{d['n']} | {fmt(d['mean'])} |"
            )
    lines.extend(
        [
            "",
            "## Pressure Phase Rates",
            "",
            "| run | phase | n | event_rate |",
            "|---|---|---:|---:|",
        ]
    )
    for label, summary in runs.items():
        total_events = max(1, int(summary.get("events") or 0))
        for name, d in sorted(summary.get("metrics", {}).items()):
            if not name.startswith("pressure_phase."):
                continue
            n = int(d.get("n") or 0)
            lines.append(
                f"| {label} | {name.removeprefix('pressure_phase.')} | "
                f"{n} | {n / total_events:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run root, shadow JSONL path, or label=path. Repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summaries: dict[str, dict[str, Any]] = {}
    for raw in args.run:
        label, path = parse_run(raw)
        shadow_path = find_shadow_path(path)
        summaries[label] = analyze_shadow(shadow_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "eviction_shadow_profile_analysis.json"
    md_path = args.output_dir / "eviction_shadow_profile_analysis.md"
    json_path.write_text(
        json.dumps({"runs": summaries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(md_path, summaries)
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
