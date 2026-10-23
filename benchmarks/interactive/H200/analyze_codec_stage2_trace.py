#!/usr/bin/env python3
"""Summarize Code2Wav stage2 chunk trace JSONL."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _pct(values: list[float], q: float) -> float | None:
    xs = sorted(v for v in values if _is_num(v))
    if not xs:
        return None
    idx = (len(xs) - 1) * q / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)


def _fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.3f}{suffix}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _metric_table(rows: list[dict[str, Any]], keys: list[str]) -> list[str]:
    lines = [
        "| metric | n | p50 | p90 | p95 | max | mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in keys:
        vals = [float(row[key]) for row in rows if _is_num(row.get(key))]
        mean = sum(vals) / len(vals) if vals else None
        max_v = max(vals) if vals else None
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{key}`",
                    str(len(vals)),
                    _fmt(_pct(vals, 50), "ms"),
                    _fmt(_pct(vals, 90), "ms"),
                    _fmt(_pct(vals, 95), "ms"),
                    _fmt(max_v, "ms"),
                    _fmt(mean, "ms"),
                ]
            )
            + " |"
        )
    return lines


def summarize(rows: list[dict[str, Any]], *, first_chunks: int) -> str:
    events = Counter(str(row.get("event")) for row in rows)
    chunk_ids = Counter(row.get("chunk_id") for row in rows if row.get("chunk_id") is not None)
    modes = Counter(
        str((row.get("meta") or {}).get("codec_cadence_mode"))
        for row in rows
        if isinstance(row.get("meta"), dict) and (row.get("meta") or {}).get("codec_cadence_mode") is not None
    )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("event") not in {"scheduled", "complete"}:
            continue
        try:
            chunk_id = int(row.get("chunk_id"))
        except (TypeError, ValueError):
            chunk_id = 1 << 30
        group = "first_window" if chunk_id < first_chunks else "post_first"
        groups[group].append(row)

    lines: list[str] = ["# Code2Wav Stage2 Trace Summary", ""]
    lines.extend(
        [
            f"Rows: {len(rows)}",
            "",
            f"Events: `{dict(events)}`",
            "",
            f"Top chunk ids: `{dict(chunk_ids.most_common(12))}`",
            "",
            f"Cadence modes: `{dict(modes)}`",
            "",
        ]
    )

    scheduled = [row for row in rows if row.get("event") == "scheduled"]
    complete = [row for row in rows if row.get("event") == "complete"]
    lines.append("## All Scheduled")
    lines.extend(_metric_table(scheduled, ["recv_to_schedule_ms", "ready_to_schedule_ms"]))
    lines.append("")
    lines.append("## All Complete")
    lines.extend(_metric_table(complete, ["schedule_to_complete_ms"]))
    lines.append("")

    for name in ("first_window", "post_first"):
        lines.append(f"## {name}")
        scheduled_group = [row for row in groups.get(name, []) if row.get("event") == "scheduled"]
        complete_group = [row for row in groups.get(name, []) if row.get("event") == "complete"]
        lines.extend(_metric_table(scheduled_group, ["recv_to_schedule_ms", "ready_to_schedule_ms"]))
        lines.append("")
        lines.extend(_metric_table(complete_group, ["schedule_to_complete_ms"]))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--first-chunks", type=int, default=1)
    args = parser.parse_args()

    rows = _load_jsonl(args.input)
    report = summarize(rows, first_chunks=max(1, args.first_chunks))
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
