#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_run(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(raw)
    return path.name, path


def find_shadow(root_or_path: Path) -> Path:
    if root_or_path.is_file():
        return root_or_path
    for pattern in (
        "eviction_only/_probe/kv_eviction_shadow.jsonl",
        "*/_probe/kv_eviction_shadow.jsonl",
        "**/kv_eviction_shadow.jsonl",
    ):
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


def p50(summary: dict[str, Any] | None, key: str) -> float | None:
    value = (summary or {}).get(key)
    if not isinstance(value, dict):
        return None
    raw = value.get("p50")
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        return float(raw)
    return None


def terminal(summary: dict[str, Any] | None) -> bool:
    if not isinstance(summary, dict):
        return False
    count = summary.get("count")
    term = summary.get("terminal")
    return isinstance(count, (int, float)) and count > 0 and term == count


def read_applied(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        if row.get("reason") != "applied":
            continue
        actual = row.get("actual_lru_victim_summary")
        selected = row.get("selected_victim_summary")
        actual_lost = p50(actual, "lost_tokens_if_prefix_break")
        selected_lost = p50(selected, "lost_tokens_if_prefix_break")
        if actual_lost is None or selected_lost is None:
            continue
        actual_gap = p50(actual, "expected_gap_sec_finite")
        selected_gap = p50(selected, "expected_gap_sec_finite")
        gap_margin = (
            selected_gap - actual_gap
            if selected_gap is not None and actual_gap is not None
            else None
        )
        rows.append(
            {
                "lost_delta_tokens": selected_lost - actual_lost,
                "gap_margin_sec": gap_margin,
                "selected_terminal": terminal(selected),
            }
        )
    return rows


def estimate(rows: list[dict[str, Any]], delta: float, margin: float) -> dict[str, Any]:
    guarded = 0
    for row in rows:
        if row["selected_terminal"]:
            continue
        if row["lost_delta_tokens"] < delta:
            continue
        gap_margin = row["gap_margin_sec"]
        if margin <= 0.0 or gap_margin is None or gap_margin < margin:
            guarded += 1
    total = len(rows)
    return {
        "max_lost_delta_tokens": delta,
        "lost_delta_min_gap_margin_sec": margin,
        "guarded_applied": guarded,
        "applied": total,
        "guarded_ratio": guarded / total if total else 0.0,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Prefix-Loss Guard Estimate",
        "",
        "Estimates use applied enforcement events only. They are screening signals, not GPU performance evidence.",
        "",
        "| run | applied events | max lost delta | min gap margin | guarded | guarded ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        for row in run["estimates"]:
            lines.append(
                f"| `{run['label']}` | {row['applied']} | "
                f"{row['max_lost_delta_tokens']:.0f} | "
                f"{row['lost_delta_min_gap_margin_sec']:.0f} | "
                f"{row['guarded_applied']} | {row['guarded_ratio']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--lost-delta", type=float, nargs="+", default=[128.0, 192.0])
    parser.add_argument("--gap-margin", type=float, nargs="+", default=[90.0, 120.0])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = []
    for raw in args.run:
        label, root = parse_run(raw)
        path = find_shadow(root)
        rows = read_applied(path)
        runs.append(
            {
                "label": label,
                "path": str(path),
                "applied_events": len(rows),
                "estimates": [
                    estimate(rows, delta, margin)
                    for delta in args.lost_delta
                    for margin in args.gap_margin
                ],
            }
        )
    result = {"runs": runs}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "prefix_loss_guard_estimate.json"
    md_path = args.output_dir / "prefix_loss_guard_estimate.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, result)
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
