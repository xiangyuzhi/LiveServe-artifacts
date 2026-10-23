#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def parse_run(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(raw)
    return path.name, path


def find_shadow_path(root_or_path: Path) -> Path:
    if root_or_path.is_file():
        return root_or_path
    patterns = (
        "eviction_only/_probe/kv_eviction_shadow.jsonl",
        "*/_probe/kv_eviction_shadow.jsonl",
        "**/kv_eviction_shadow.jsonl",
    )
    for pattern in patterns:
        matches = sorted(root_or_path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no kv_eviction_shadow.jsonl under {root_or_path}")


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


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def summary_metric(summary: dict[str, Any], name: str, qkey: str = "p50") -> float | None:
    value = summary.get(name)
    if not isinstance(value, dict):
        return None
    return finite_number(value.get(qkey))


def logged_number(value: Any) -> float | None:
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    return finite_number(value)


def pair_passes(pair: dict[str, Any], *, margin_sec: float, terminal_ok: bool) -> bool:
    actual_gap = logged_number(pair.get("actual_gap_sec"))
    selected_gap = logged_number(pair.get("selected_gap_sec"))
    if actual_gap is None or math.isinf(actual_gap):
        return False
    if selected_gap is None:
        return False
    if math.isinf(selected_gap):
        return bool(terminal_ok and pair.get("selected_terminal") is True)
    return selected_gap - actual_gap >= margin_sec


def exact_retained_count_from_pair_details(
    pair_details: dict[str, Any],
    *,
    margin_sec: float,
    terminal_ok: bool,
) -> int | None:
    values = pair_details.get("values")
    count = int(pair_details.get("count") or 0)
    if not isinstance(values, list):
        return None
    if bool(pair_details.get("truncated")) or len(values) < count:
        return None
    retained = 0
    for pair in values[:count]:
        if not isinstance(pair, dict):
            break
        if pair_passes(pair, margin_sec=margin_sec, terminal_ok=terminal_ok):
            retained += 1
            continue
        break
    return retained


def estimate_margin(
    path: Path,
    *,
    margin_sec: float,
    terminal_ok: bool,
) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    applied_events = 0
    total_replacements = 0
    retained_events = 0
    retained_replacements = 0
    terminal_pass_events = 0
    finite_margin_pass_events = 0
    observed_pair_detail_events = 0
    observed_pair_count = 0
    observed_pair_pass_count = 0
    exact_pair_complete_events = 0
    exact_retained_events = 0
    exact_retained_replacements = 0
    rejected_examples: list[dict[str, Any]] = []

    for row in iter_jsonl(path):
        reason = str(row.get("reason") or "none")
        reasons[reason] += 1
        if row.get("applied") is not True and reason != "applied":
            continue

        applied_events += 1
        replacement_count = int(row.get("replacement_count") or 0)
        total_replacements += replacement_count
        pair_details = row.get("gap_margin_pair_details")
        if isinstance(pair_details, dict):
            observed_pair_detail_events += 1
            observed_pair_count += int(pair_details.get("count") or 0)
            observed_pair_pass_count += int(pair_details.get("pass_count") or 0)
            exact_retained_count = exact_retained_count_from_pair_details(
                pair_details,
                margin_sec=margin_sec,
                terminal_ok=terminal_ok,
            )
            if exact_retained_count is not None:
                exact_pair_complete_events += 1
                if exact_retained_count > 0:
                    exact_retained_events += 1
                    exact_retained_replacements += exact_retained_count
        actual = row.get("actual_lru_victim_summary") or {}
        selected = row.get("selected_victim_summary") or {}
        if not isinstance(actual, dict) or not isinstance(selected, dict):
            continue

        actual_gap = summary_metric(actual, "expected_gap_sec_finite")
        selected_gap = summary_metric(selected, "expected_gap_sec_finite")
        selected_count = int(selected.get("count") or 0)
        selected_terminal = int(selected.get("terminal") or 0)
        pass_terminal = terminal_ok and selected_count > 0 and selected_terminal > 0
        pass_finite_margin = (
            actual_gap is not None
            and selected_gap is not None
            and selected_gap - actual_gap >= margin_sec
        )

        if pass_terminal or pass_finite_margin:
            retained_events += 1
            retained_replacements += replacement_count
            terminal_pass_events += int(pass_terminal)
            finite_margin_pass_events += int(pass_finite_margin)
            continue

        if len(rejected_examples) < 8:
            rejected_examples.append(
                {
                    "actual_gap_p50": actual_gap,
                    "selected_gap_p50": selected_gap,
                    "selected_terminal": selected_terminal,
                    "selected_count": selected_count,
                    "replacement_count": replacement_count,
                }
            )

    return {
        "path": str(path),
        "margin_sec": margin_sec,
        "terminal_ok": terminal_ok,
        "events": sum(reasons.values()),
        "reason_counts": dict(reasons.most_common()),
        "applied_events": applied_events,
        "total_replacements": total_replacements,
        "estimated_retained_events": retained_events,
        "estimated_retained_replacements": retained_replacements,
        "estimated_retained_event_ratio": (
            retained_events / applied_events if applied_events else 0.0
        ),
        "estimated_retained_replacement_ratio": (
            retained_replacements / total_replacements if total_replacements else 0.0
        ),
        "terminal_pass_events": terminal_pass_events,
        "finite_margin_pass_events": finite_margin_pass_events,
        "observed_pair_detail_events": observed_pair_detail_events,
        "observed_pair_count": observed_pair_count,
        "observed_pair_pass_count": observed_pair_pass_count,
        "exact_pair_complete_events": exact_pair_complete_events,
        "exact_retained_events": exact_retained_events,
        "exact_retained_replacements": exact_retained_replacements,
        "exact_retained_event_ratio": (
            exact_retained_events / exact_pair_complete_events
            if exact_pair_complete_events else None
        ),
        "exact_retained_replacement_ratio": (
            exact_retained_replacements / observed_pair_count
            if observed_pair_count else None
        ),
        "first_rejected_examples": rejected_examples,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Gap Margin Shadow Estimate",
        "",
        "This is an event-level approximation from logged victim summaries. It is useful for ranking margin candidates, but a matched GPU run is still decisive because old logs do not preserve per-block selected/actual pairs.",
        "",
        "| run | margin | terminal ok | applied events | retained events | retained event ratio | total replacements | retained replacements | terminal-pass events | finite-margin events | observed pair details | observed pair pass/count | exact retained events | exact retained replacements |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, margins in result["runs"].items():
        for row in margins:
            lines.append(
                f"| {label} | {fmt(row['margin_sec'])} | {row['terminal_ok']} | "
                f"{row['applied_events']} | {row['estimated_retained_events']} | "
                f"{fmt(row['estimated_retained_event_ratio'])} | "
                f"{row['total_replacements']} | {row['estimated_retained_replacements']} | "
                f"{row['terminal_pass_events']} | {row['finite_margin_pass_events']} | "
                f"{row['observed_pair_detail_events']} | "
                f"{row['observed_pair_pass_count']}/{row['observed_pair_count']} | "
                f"{row['exact_retained_events']} | "
                f"{row['exact_retained_replacements']} |"
            )
    lines.extend(["", "## First Rejected Examples", ""])
    for label, margins in result["runs"].items():
        lines.append(f"### {label}")
        lines.append("")
        for row in margins:
            examples = row.get("first_rejected_examples") or []
            if not examples:
                continue
            lines.append(f"- margin `{fmt(row['margin_sec'])}`")
            for example in examples[:4]:
                lines.append(
                    "  - "
                    f"actual_gap={fmt(example.get('actual_gap_p50'))}, "
                    f"selected_gap={fmt(example.get('selected_gap_p50'))}, "
                    f"selected_terminal={example.get('selected_terminal')}, "
                    f"replacement_count={example.get('replacement_count')}"
                )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="label=run_root or shadow jsonl")
    parser.add_argument("--margins", type=float, nargs="+", default=[30.0, 45.0, 60.0])
    parser.add_argument("--terminal-ok", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    terminal_ok = bool(args.terminal_ok)
    runs: dict[str, list[dict[str, Any]]] = {}
    for raw in args.run:
        label, run_path = parse_run(raw)
        shadow_path = find_shadow_path(run_path)
        runs[label] = [
            estimate_margin(
                shadow_path,
                margin_sec=margin,
                terminal_ok=terminal_ok,
            )
            for margin in args.margins
        ]

    result = {
        "config": {
            "margins": args.margins,
            "terminal_ok": terminal_ok,
            "method": "event-level approximation from victim summaries; selected terminal or selected_gap_p50 - actual_gap_p50 >= margin",
            "exact_pair_method": "when complete gap_margin_pair_details are present, recompute retained prefix pairs using selected_gap_sec/actual_gap_sec for the requested margin",
        },
        "runs": runs,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "gap_margin_shadow_estimate.json"
    md_path = args.output_dir / "gap_margin_shadow_estimate.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, result)
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
