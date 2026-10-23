#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_surface(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
        path = path.strip()
    else:
        path = raw.strip()
        label = Path(path).parent.name or Path(path).stem
    if not label or not path:
        raise argparse.ArgumentTypeError(f"expected label=path, got {raw!r}")
    return label, Path(path)


def surface_json_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "simulation.json"
    if candidate.exists():
        return candidate
    candidate = path / "_workload" / "simulation" / "simulation.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"no simulation.json under {path}")


def read_surface(path: Path) -> dict[str, Any]:
    return json.loads(surface_json_path(path).read_text(encoding="utf-8"))


def finite(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def tail_risk(row: dict[str, Any], useful_weight: float, eviction_weight: float) -> float:
    return (
        finite(row.get("miss_blocks"))
        + useful_weight * finite(row.get("useful_evictions"))
        + eviction_weight * finite(row.get("evictions"))
    )


def annotate_rows(
    label: str,
    data: dict[str, Any],
    *,
    useful_weight: float,
    eviction_weight: float,
    full_hit_weight: float,
    min_lru_tail_risk: float,
) -> list[dict[str, Any]]:
    by_capacity: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in data.get("results", []):
        by_capacity[int(row.get("capacity_tokens") or 0)].append(row)

    annotated: list[dict[str, Any]] = []
    for capacity, rows in sorted(by_capacity.items()):
        lru = next((row for row in rows if row.get("policy") == "lru"), None)
        if lru is None:
            continue
        lru_risk = tail_risk(lru, useful_weight, eviction_weight)
        lru_miss_blocks = finite(lru.get("miss_blocks"))
        lru_full_hit = finite(lru.get("request_full_hit_rate"))
        requests_with_history = finite(lru.get("requests_with_history_blocks"))
        for row in rows:
            row_risk = tail_risk(row, useful_weight, eviction_weight)
            full_hit_gain_requests = (
                finite(row.get("request_full_hit_rate")) - lru_full_hit
            ) * requests_with_history
            miss_blocks = finite(row.get("miss_blocks"))
            lru_miss_blocks = finite(lru.get("miss_blocks"))
            miss_reduction_pct = None
            if lru_miss_blocks > 0:
                miss_reduction_pct = 100.0 * (lru_miss_blocks - miss_blocks) / lru_miss_blocks
            lru_risk_reduction_pct = None
            if lru_risk > 0:
                lru_risk_reduction_pct = 100.0 * (lru_risk - row_risk) / lru_risk
            relative_score = (
                (lru_risk - row_risk)
                + full_hit_weight * full_hit_gain_requests
            )
            annotated.append(
                {
                    "surface": label,
                    "capacity_tokens": capacity,
                    "policy": row.get("policy"),
                    "effective_policy": row.get("effective_policy", row.get("policy")),
                    "hit_ratio": row.get("hit_ratio"),
                    "full_hit_rate": row.get("request_full_hit_rate"),
                    "miss_blocks": row.get("miss_blocks"),
                    "evictions": row.get("evictions"),
                    "useful_evictions": row.get("useful_evictions"),
                    "hit_gain_blocks": finite(row.get("hit_blocks")) - finite(lru.get("hit_blocks")),
                    "miss_reduction_pct": miss_reduction_pct,
                    "lru_tail_risk_score": lru_risk,
                    "lru_miss_blocks": lru_miss_blocks,
                    "tail_risk_score": row_risk,
                    "tail_risk_reduction_pct": lru_risk_reduction_pct,
                    "full_hit_gain_requests": full_hit_gain_requests,
                    "relative_score": relative_score,
                    "has_pressure": lru_risk >= min_lru_tail_risk,
                    "is_lru": row.get("policy") == "lru",
                }
            )
    annotated.sort(
        key=lambda row: (
            row["surface"],
            row["capacity_tokens"],
            -finite(row["relative_score"]),
            str(row["policy"]),
        )
    )
    return annotated


def summarize_policy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("is_lru"):
            continue
        by_key[(str(row["surface"]), str(row["policy"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (surface, policy), group in sorted(by_key.items()):
        scores = [finite(row.get("relative_score")) for row in group]
        pressure_group = [row for row in group if row.get("has_pressure")]
        pressure_scores = [finite(row.get("relative_score")) for row in pressure_group]
        summary.append(
            {
                "surface": surface,
                "policy": policy,
                "capacities": len(group),
                "pressure_capacities": len(pressure_group),
                "avg_relative_score": sum(scores) / len(scores) if scores else None,
                "min_relative_score": min(scores) if scores else None,
                "avg_pressure_relative_score": (
                    sum(pressure_scores) / len(pressure_scores)
                    if pressure_scores else None
                ),
                "min_pressure_relative_score": (
                    min(pressure_scores) if pressure_scores else None
                ),
                "avg_hit_ratio": mean([row.get("hit_ratio") for row in group]),
                "avg_full_hit_rate": mean([row.get("full_hit_rate") for row in group]),
                "avg_tail_risk_reduction_pct": mean(
                    [row.get("tail_risk_reduction_pct") for row in group]
                ),
                "avg_pressure_tail_risk_reduction_pct": mean(
                    [row.get("tail_risk_reduction_pct") for row in pressure_group]
                ),
                "avg_miss_reduction_pct": mean(
                    [row.get("miss_reduction_pct") for row in group]
                ),
            }
        )
    summary.sort(
        key=lambda row: (
            row["surface"],
            -finite(row.get("avg_pressure_relative_score")),
            -finite(row.get("avg_relative_score")),
            str(row["policy"]),
        )
    )
    return summary


def mean(values: list[Any]) -> float | None:
    xs = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return sum(xs) / len(xs) if xs else None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    *,
    top_k: int,
    useful_weight: float,
    eviction_weight: float,
    full_hit_weight: float,
) -> None:
    lines = [
        "# Eviction Policy Surface Ranking",
        "",
        "This is an offline ranking signal, not GPU performance evidence.",
        "",
        "Scoring:",
        "",
        "```text",
        "tail_risk = miss_blocks",
        f"          + {useful_weight:g} * useful_evictions",
        f"          + {eviction_weight:g} * evictions",
        "relative_score = LRU_tail_risk - policy_tail_risk",
        f"                 + {full_hit_weight:g} * full_hit_gain_requests",
        "```",
        "",
        "Higher `relative_score` is better. The useful-eviction term is a proxy for",
        "future reuse damage; the eviction term is a small churn/overhead penalty.",
        "",
        "## Robust Policy Summary",
        "",
        "| surface | policy | pressure caps | avg pressure score | min pressure score | avg score | avg hit | avg full-hit | avg pressure tail-risk reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['surface']} | `{row['policy']}` | "
            f"{row['pressure_capacities']}/{row['capacities']} | "
            f"{fmt(row['avg_pressure_relative_score'])} | "
            f"{fmt(row['min_pressure_relative_score'])} | "
            f"{fmt(row['avg_relative_score'])} | "
            f"{fmt(row['avg_hit_ratio'], 4)} | {fmt(row['avg_full_hit_rate'], 4)} | "
            f"{fmt(row['avg_pressure_tail_risk_reduction_pct'])}% |"
        )

    lines.extend(["", "## Top Rows By Capacity", ""])
    surfaces = sorted({str(row["surface"]) for row in rows})
    for surface in surfaces:
        lines.extend([f"### {surface}", ""])
        capacities = sorted({int(row["capacity_tokens"]) for row in rows if row["surface"] == surface})
        for capacity in capacities:
            top_rows = [
                row for row in rows
                if row["surface"] == surface and int(row["capacity_tokens"]) == capacity
            ][:top_k]
            lines.extend(
                [
                    f"#### {capacity} tokens",
                    "",
                    "| policy | score | hit | full-hit | miss blocks | evictions | useful evictions | tail-risk reduction |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in top_rows:
                lines.append(
                    f"| `{row['policy']}` | {fmt(row['relative_score'])} | "
                    f"{fmt(row['hit_ratio'], 4)} | {fmt(row['full_hit_rate'], 4)} | "
                    f"{fmt(row['miss_blocks'])} | {fmt(row['evictions'])} | "
                    f"{fmt(row['useful_evictions'])} | "
                    f"{fmt(row['tail_risk_reduction_pct'])}% |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", action="append", type=parse_surface, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--useful-eviction-weight", type=float, default=0.25)
    parser.add_argument("--eviction-weight", type=float, default=0.02)
    parser.add_argument("--full-hit-weight", type=float, default=4.0)
    parser.add_argument(
        "--min-lru-tail-risk",
        type=float,
        default=1.0,
        help="Capacities with lower LRU tail-risk are excluded from pressure summaries.",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    source_paths: dict[str, str] = {}
    for label, path in args.surface:
        json_path = surface_json_path(path)
        source_paths[label] = str(json_path)
        rows.extend(
            annotate_rows(
                label,
                read_surface(json_path),
                    useful_weight=args.useful_eviction_weight,
                    eviction_weight=args.eviction_weight,
                    full_hit_weight=args.full_hit_weight,
                    min_lru_tail_risk=args.min_lru_tail_risk,
                )
            )
    summary = summarize_policy(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "policy_surface_ranking.json"
    md_path = args.output_dir / "policy_surface_ranking.md"
    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "source_paths": source_paths,
                    "useful_eviction_weight": args.useful_eviction_weight,
                    "eviction_weight": args.eviction_weight,
                    "full_hit_weight": args.full_hit_weight,
                    "min_lru_tail_risk": args.min_lru_tail_risk,
                },
                "summary": summary,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_markdown(
        md_path,
        rows,
        summary,
        top_k=max(1, args.top_k),
        useful_weight=args.useful_eviction_weight,
        eviction_weight=args.eviction_weight,
        full_hit_weight=args.full_hit_weight,
    )
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
