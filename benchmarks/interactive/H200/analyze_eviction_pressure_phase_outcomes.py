#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for seq, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_seq"] = seq
            yield row


def find_probe_file(root_or_path: Path, name: str) -> Path:
    if root_or_path.is_file():
        return root_or_path
    direct = root_or_path / "_probe" / name
    if direct.exists():
        return direct
    for pattern in (f"*/_probe/{name}", f"**/{name}"):
        matches = sorted(root_or_path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no {name} under {root_or_path}")


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str) and value.strip():
        try:
            out = float(value)
        except ValueError:
            return None
        return out if math.isfinite(out) else None
    return None


def as_int(value: Any) -> int | None:
    out = as_float(value)
    return int(out) if out is not None else None


def bucket(value: float | None, cuts: list[float], prefix: str) -> str:
    if value is None:
        return f"{prefix}_missing"
    prev = None
    for cut in cuts:
        if value < cut:
            if prev is None:
                return f"{prefix}_lt{cut:g}"
            return f"{prefix}_{prev:g}_{cut:g}"
        prev = cut
    return f"{prefix}_{cuts[-1]:g}_plus"


def safe_rate(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def read_shadow_events(path: Path) -> dict[int, dict[str, Any]]:
    events: dict[int, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("event") == "allocation_pressure_enforce":
            events[int(row["_seq"])] = row
    return events


def add_group(
    groups: dict[tuple[str, str], dict[str, Any]],
    group: str,
    value: str,
    future_misses: int,
) -> None:
    row = groups[(group, value)]
    row["group"] = group
    row["value"] = value
    row["victims"] += 1
    row["miss_victims"] += int(future_misses > 0)
    row["future_misses"] += future_misses


def analyze(shadow_path: Path, victim_csv: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = read_shadow_events(shadow_path)
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "group": "",
            "value": "",
            "victims": 0,
            "miss_victims": 0,
            "future_misses": 0,
        }
    )
    total_victims = 0
    matched_events: set[int] = set()
    with victim_csv.open("r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            event_seq = as_int(row.get("event_seq"))
            if event_seq is None:
                continue
            event = events.get(event_seq)
            if not event:
                continue
            total_victims += 1
            matched_events.add(event_seq)
            future_misses = as_int(row.get("future_lookup_misses")) or 0
            current = event.get("current_request")
            current = current if isinstance(current, dict) else {}
            add_group(
                groups,
                "pressure_phase",
                str(event.get("pressure_phase") or "missing"),
                future_misses,
            )
            add_group(
                groups,
                "recent_applied_attempts_64",
                bucket(
                    as_float(event.get("recent_applied_attempts_64")),
                    [8, 16, 32, 48, 56, 64],
                    "applied64",
                ),
                future_misses,
            )
            add_group(
                groups,
                "recent_cached_victim_attempts_64",
                bucket(
                    as_float(event.get("recent_cached_victim_attempts_64")),
                    [8, 16, 32, 48, 56, 64],
                    "cached64",
                ),
                future_misses,
            )
            add_group(
                groups,
                "recent_no_cached_victim_attempts_64",
                bucket(
                    as_float(event.get("recent_no_cached_victim_attempts_64")),
                    [8, 16, 32, 48, 56, 64],
                    "nocached64",
                ),
                future_misses,
            )
            add_group(
                groups,
                "current_round_index",
                bucket(
                    as_float(current.get("omni_round_index")),
                    [4, 8, 16, 32, 64],
                    "round",
                ),
                future_misses,
            )
            add_group(
                groups,
                "current_expected_gap_sec",
                bucket(
                    as_float(current.get("omni_expected_next_gap_sec")),
                    [15, 30, 60, 120],
                    "current_gap",
                ),
                future_misses,
            )
            add_group(
                groups,
                "current_response_length",
                bucket(
                    as_float(current.get("omni_response_length")),
                    [16, 32, 64, 128],
                    "current_resp",
                ),
                future_misses,
            )

    rows = []
    for row in groups.values():
        victims = int(row["victims"])
        miss_victims = int(row["miss_victims"])
        future_misses = int(row["future_misses"])
        row = dict(row)
        row["miss_victim_rate"] = safe_rate(miss_victims, victims)
        row["future_misses_per_victim"] = safe_rate(future_misses, victims)
        row["victim_share"] = safe_rate(victims, total_victims)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row["group"]),
            -int(row["future_misses"]),
            str(row["value"]),
        )
    )
    summary = {
        "shadow_path": str(shadow_path),
        "victim_csv": str(victim_csv),
        "shadow_events": len(events),
        "matched_applied_events": len(matched_events),
        "total_victims": total_victims,
        "top_future_groups": sorted(
            rows,
            key=lambda row: int(row["future_misses"]),
            reverse=True,
        )[:20],
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "group",
        "value",
        "victims",
        "victim_share",
        "miss_victims",
        "miss_victim_rate",
        "future_misses",
        "future_misses_per_victim",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Eviction Pressure Phase Outcomes",
        "",
        f"- shadow events: `{summary['shadow_events']}`",
        f"- matched applied events: `{summary['matched_applied_events']}`",
        f"- total selected victims: `{summary['total_victims']}`",
        "",
        "## Top Future-Miss Groups",
        "",
        "| group | value | victims | victim share | miss victims | miss rate | future misses | future/victim |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["top_future_groups"]:
        lines.append(
            f"| {row['group']} | {row['value']} | {row['victims']} | "
            f"{row['victim_share']:.3f} | {row['miss_victims']} | "
            f"{row['miss_victim_rate']:.3f} | {row['future_misses']} | "
            f"{row['future_misses_per_victim']:.3f} |"
        )
    for group in sorted({str(row["group"]) for row in rows}):
        lines.extend(
            [
                "",
                f"## {group}",
                "",
                "| value | victims | miss rate | future misses | future/victim |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in [r for r in rows if r["group"] == group]:
            lines.append(
                f"| {row['value']} | {row['victims']} | "
                f"{row['miss_victim_rate']:.3f} | {row['future_misses']} | "
                f"{row['future_misses_per_victim']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow", required=True, help="run root or kv_eviction_shadow.jsonl")
    parser.add_argument("--victims", required=True, help="victim_outcomes.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    shadow_path = find_probe_file(Path(args.shadow), "kv_eviction_shadow.jsonl")
    victim_csv = Path(args.victims)
    rows, summary = analyze(shadow_path, victim_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pressure_phase_outcomes.csv", rows)
    (args.output_dir / "pressure_phase_outcomes.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "groups": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_markdown(
        args.output_dir / "pressure_phase_outcomes.md",
        rows,
        summary,
    )
    print(args.output_dir / "pressure_phase_outcomes.json")
    print(args.output_dir / "pressure_phase_outcomes.csv")
    print(args.output_dir / "pressure_phase_outcomes.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
