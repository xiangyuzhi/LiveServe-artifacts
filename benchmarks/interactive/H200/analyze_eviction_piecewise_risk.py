#!/usr/bin/env python3
"""Summarize victim-outcome risk buckets for eviction score tuning."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _score_value(row: dict[str, str]) -> float | None:
    raw = row.get("selected_score_tuple")
    if not raw:
        return None
    try:
        score = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(score, list) or len(score) < 3:
        return None
    return _float(score[2])


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * q)
    return ordered[max(0, min(len(ordered) - 1, idx))]


def _summarize_group(rows: list[dict[str, str]]) -> dict[str, Any]:
    victims = len(rows)
    future_misses = sum(_int(row.get("future_lookup_misses")) for row in rows)
    future_miss_victims = sum(
        1 for row in rows if _int(row.get("future_lookup_misses")) > 0
    )
    scores = [score for row in rows if (score := _score_value(row)) is not None]
    histories = [
        value
        for row in rows
        if (value := _float(row.get("victim_history_tokens_after"))) is not None
    ]
    gaps = [
        value
        for row in rows
        if (value := _float(row.get("victim_expected_next_gap_sec"))) is not None
    ]
    return {
        "victims": victims,
        "future_misses": future_misses,
        "future_miss_victims": future_miss_victims,
        "future_miss_victim_rate": future_miss_victims / victims if victims else 0.0,
        "score_p25": _percentile(scores, 0.25),
        "score_p50": median(scores) if scores else None,
        "score_p75": _percentile(scores, 0.75),
        "history_p50": median(histories) if histories else None,
        "gap_p50": median(gaps) if gaps else None,
    }


def _bucket_table(
    rows: list[dict[str, str]],
    keys: tuple[str, ...],
    min_victims: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    out = []
    for key, group_rows in grouped.items():
        if len(group_rows) < min_victims:
            continue
        summary = _summarize_group(group_rows)
        for key_name, value in zip(keys, key):
            summary[key_name] = value
        out.append(summary)
    return sorted(out, key=lambda row: (row["future_misses"], row["victims"]), reverse=True)


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _recommend_penalties(
    cross_rows: list[dict[str, Any]],
    scale: float,
    cap: float,
) -> dict[str, float]:
    def excess(row: dict[str, Any], reference: float) -> float:
        return max(0.0, float(row["future_miss_victim_rate"]) - reference)

    refs: dict[str, float] = {}
    for row in cross_rows:
        if row.get("history_bucket") == "hist_128_256":
            refs[str(row.get("gap_bucket"))] = float(row["future_miss_victim_rate"])
    global_ref = min(
        (float(row["future_miss_victim_rate"]) for row in cross_rows),
        default=0.0,
    )

    penalties = {
        "piecewise_hist_lt64_penalty": 0.0,
        "piecewise_hist_64_128_penalty": 0.0,
        "piecewise_gap_15_30_penalty": 0.0,
        "piecewise_gap_30_60_penalty": 0.0,
    }
    for row in cross_rows:
        gap_bucket = str(row.get("gap_bucket"))
        history_bucket = str(row.get("history_bucket"))
        if gap_bucket not in {"gap_15_30", "gap_30_60"}:
            continue
        reference = refs.get(gap_bucket, global_ref)
        suggested = min(cap, scale * excess(row, reference))
        if history_bucket == "hist_lt64":
            penalties["piecewise_hist_lt64_penalty"] = max(
                penalties["piecewise_hist_lt64_penalty"],
                suggested,
            )
        elif history_bucket == "hist_64_128":
            penalties["piecewise_hist_64_128_penalty"] = max(
                penalties["piecewise_hist_64_128_penalty"],
                suggested,
            )
        if history_bucket in {"hist_lt64", "hist_64_128"}:
            if gap_bucket == "gap_15_30":
                penalties["piecewise_gap_15_30_penalty"] = max(
                    penalties["piecewise_gap_15_30_penalty"],
                    suggested * 0.25,
                )
            elif gap_bucket == "gap_30_60":
                penalties["piecewise_gap_30_60_penalty"] = max(
                    penalties["piecewise_gap_30_60_penalty"],
                    suggested * 0.25,
                )
    return {key: round(value, 2) for key, value in penalties.items()}


def _write_markdown(
    output: Path,
    run_summaries: list[dict[str, Any]],
    cross_tables: dict[str, list[dict[str, Any]]],
    recommendations: dict[str, float],
) -> None:
    lines = ["# Eviction Piecewise Risk Analysis", ""]
    lines.append("## Runs")
    lines.append("")
    lines.append("| run | victims | future misses | victim miss rate |")
    lines.append("|---|---:|---:|---:|")
    for summary in run_summaries:
        lines.append(
            "| {name} | {victims} | {future_misses} | {rate} |".format(
                name=summary["name"],
                victims=summary["victims"],
                future_misses=summary["future_misses"],
                rate=_format_float(summary["future_miss_victim_rate"]),
            )
        )
    lines.append("")
    lines.append("## Suggested Starting Penalties")
    lines.append("")
    for key, value in recommendations.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    for run_name, rows in cross_tables.items():
        lines.append(f"## Cross Buckets: {run_name}")
        lines.append("")
        lines.append(
            "| history | gap | victims | miss victims | miss rate | future misses | score p50 | history p50 | gap p50 |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows[:20]:
            lines.append(
                "| {history} | {gap} | {victims} | {miss_victims} | {rate} | {misses} | {score} | {hist} | {gap_p50} |".format(
                    history=row.get("history_bucket", ""),
                    gap=row.get("gap_bucket", ""),
                    victims=row["victims"],
                    miss_victims=row["future_miss_victims"],
                    rate=_format_float(row["future_miss_victim_rate"]),
                    misses=row["future_misses"],
                    score=_format_float(row["score_p50"]),
                    hist=_format_float(row["history_p50"], 1),
                    gap_p50=_format_float(row["gap_p50"], 1),
                )
            )
        lines.append("")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Run input as NAME=PATH_TO_victim_outcomes.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-victims", type=int, default=20)
    parser.add_argument("--penalty-scale", type=float, default=120.0)
    parser.add_argument("--penalty-cap", type=float, default=32.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict[str, Any]] = []
    cross_tables: dict[str, list[dict[str, Any]]] = {}
    first_cross: list[dict[str, Any]] | None = None
    for item in args.input:
        if "=" not in item:
            raise SystemExit(f"--input must be NAME=PATH, got {item!r}")
        name, path_str = item.split("=", 1)
        rows = _read_rows(Path(path_str))
        summary = _summarize_group(rows)
        summary["name"] = name
        run_summaries.append(summary)
        cross = _bucket_table(rows, ("history_bucket", "gap_bucket"), args.min_victims)
        cross_tables[name] = cross
        if first_cross is None:
            first_cross = cross

    recommendations = _recommend_penalties(
        first_cross or [],
        scale=args.penalty_scale,
        cap=args.penalty_cap,
    )

    summary = {
        "runs": run_summaries,
        "recommendations": recommendations,
        "cross_tables": cross_tables,
        "config": {
            "min_victims": args.min_victims,
            "penalty_scale": args.penalty_scale,
            "penalty_cap": args.penalty_cap,
        },
    }
    (output_dir / "piecewise_risk_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(
        output_dir / "piecewise_risk_summary.md",
        run_summaries,
        cross_tables,
        recommendations,
    )
    print(output_dir / "piecewise_risk_summary.json")
    print(output_dir / "piecewise_risk_summary.md")


if __name__ == "__main__":
    main()
