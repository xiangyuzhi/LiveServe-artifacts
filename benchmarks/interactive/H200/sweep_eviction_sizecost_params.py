#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import simulate_interactive_kv_policies as sim


@dataclass(frozen=True)
class Surface:
    name: str
    time_start: int
    time_limit: int
    capacities: tuple[int, ...]
    max_requests: int = 0
    max_requests_policy: str = "first"
    session_balanced_min_turns: int = 3


SURFACES = {
    "micro1200": Surface(
        name="micro1200",
        time_start=6996,
        time_limit=7176,
        capacities=(86000,),
        max_requests=1200,
        max_requests_policy="session_balanced",
        session_balanced_min_turns=3,
    ),
    "w300": Surface(
        name="w300",
        time_start=6846,
        time_limit=7146,
        capacities=(86000, 172000, 344000),
    ),
    "w600": Surface(
        name="w600",
        time_start=6726,
        time_limit=7326,
        capacities=(172000, 344000, 687568),
    ),
}


CONTROL_POLICIES = (
    "expected_gap_tail_aware",
    "expected_gap_cost_aware",
    "expected_gap_bucket_response_tail_aware",
)


def parse_float_list(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return values


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


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


def apply_size_params(
    *,
    min_gap: float,
    history_weight: float,
    response_weight: float,
    loss_weight: float,
) -> None:
    sim.SIZE_BONUS_MIN_GAP_SEC = min_gap
    sim.SIZE_HISTORY_WEIGHT = history_weight
    sim.SIZE_RESPONSE_WEIGHT = response_weight
    sim.SIZE_LOSS_WEIGHT = loss_weight


def select_surface_entries(surface: Surface) -> list[sim.Entry]:
    entries = sim.select_window(
        sim.load_entries(sim.TRACE_DIR, [1]),
        surface.time_start,
        surface.time_limit,
    )
    if surface.max_requests > 0 and len(entries) > surface.max_requests:
        if surface.max_requests_policy == "session_balanced":
            entries = sim.cap_session_balanced(
                entries,
                surface.max_requests,
                surface.session_balanced_min_turns,
            )
        elif surface.max_requests_policy == "real_multiturn_peak":
            entries = sim.cap_real_multiturn_peak(
                entries,
                surface.max_requests,
                surface.session_balanced_min_turns,
            )
        else:
            entries = entries[: surface.max_requests]
    return entries


def annotate_result(
    *,
    surface: Surface,
    capacity_tokens: int,
    label: str,
    policy: str,
    row: dict[str, Any],
    lru: dict[str, Any],
    useful_weight: float,
    eviction_weight: float,
    full_hit_weight: float,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lru_risk = tail_risk(lru, useful_weight, eviction_weight)
    row_risk = tail_risk(row, useful_weight, eviction_weight)
    requests_with_history = finite(lru.get("requests_with_history_blocks"))
    full_hit_gain_requests = (
        finite(row.get("request_full_hit_rate"))
        - finite(lru.get("request_full_hit_rate"))
    ) * requests_with_history
    relative_score = (
        lru_risk
        - row_risk
        + full_hit_weight * full_hit_gain_requests
    )
    out = {
        "surface": surface.name,
        "capacity_tokens": capacity_tokens,
        "label": label,
        "policy": policy,
        "relative_score": relative_score,
        "tail_risk": row_risk,
        "lru_tail_risk": lru_risk,
        "tail_risk_reduction_pct": (
            100.0 * (lru_risk - row_risk) / lru_risk if lru_risk > 0 else None
        ),
        "full_hit_gain_requests": full_hit_gain_requests,
        "hit_ratio": row.get("hit_ratio"),
        "full_hit_rate": row.get("request_full_hit_rate"),
        "miss_blocks": row.get("miss_blocks"),
        "evictions": row.get("evictions"),
        "useful_evictions": row.get("useful_evictions"),
    }
    if params:
        out.update(params)
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), []).append(row)
    summary = []
    for label, group in by_label.items():
        scores = [finite(row.get("relative_score")) for row in group]
        summary.append(
            {
                "label": label,
                "policy": group[0].get("policy"),
                "surfaces": len({row["surface"] for row in group}),
                "capacities": len(group),
                "avg_score": statistics.mean(scores) if scores else None,
                "min_score": min(scores) if scores else None,
                "avg_hit_ratio": statistics.mean(
                    finite(row.get("hit_ratio")) for row in group
                ),
                "avg_full_hit_rate": statistics.mean(
                    finite(row.get("full_hit_rate")) for row in group
                ),
                "avg_tail_risk_reduction_pct": statistics.mean(
                    finite(row.get("tail_risk_reduction_pct")) for row in group
                ),
                "size_bonus_min_gap_sec": group[0].get("size_bonus_min_gap_sec"),
                "size_history_weight": group[0].get("size_history_weight"),
                "size_response_weight": group[0].get("size_response_weight"),
                "size_loss_weight": group[0].get("size_loss_weight"),
            }
        )
    summary.sort(
        key=lambda row: (
            finite(row.get("avg_score")),
            finite(row.get("min_score")),
            finite(row.get("avg_hit_ratio")),
        ),
        reverse=True,
    )
    return summary


def write_markdown(
    path: Path,
    *,
    summary: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    top_k: int,
) -> None:
    lines = [
        "# Size-Cost Eviction Parameter Sweep",
        "",
        "This is an offline simulator sweep, not GPU performance evidence.",
        "",
        "## Top Variants",
        "",
        "| rank | label | avg score | min score | avg hit | avg full-hit | avg tail-risk reduction | params |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(summary[:top_k], 1):
        params = (
            f"min_gap={fmt(row.get('size_bonus_min_gap_sec'), 1)}, "
            f"hist={fmt(row.get('size_history_weight'), 1)}, "
            f"resp={fmt(row.get('size_response_weight'), 1)}, "
            f"loss={fmt(row.get('size_loss_weight'), 1)}"
            if row.get("policy") == "expected_gap_size_cost_aware"
            else "control"
        )
        lines.append(
            f"| {rank} | `{row['label']}` | {fmt(row.get('avg_score'))} | "
            f"{fmt(row.get('min_score'))} | {fmt(row.get('avg_hit_ratio'), 4)} | "
            f"{fmt(row.get('avg_full_hit_rate'), 4)} | "
            f"{fmt(row.get('avg_tail_risk_reduction_pct'))}% | {params} |"
        )

    lines.extend(["", "## Best Rows By Surface", ""])
    for surface in sorted({str(row["surface"]) for row in rows}):
        lines.extend([f"### {surface}", ""])
        capacities = sorted(
            int(row["capacity_tokens"]) for row in rows if row["surface"] == surface
        )
        for capacity in sorted(set(capacities)):
            selected = [
                row for row in rows
                if row["surface"] == surface and int(row["capacity_tokens"]) == capacity
            ]
            selected.sort(
                key=lambda row: finite(row.get("relative_score")),
                reverse=True,
            )
            lines.extend(
                [
                    f"#### {capacity} tokens",
                    "",
                    "| rank | label | score | hit | full-hit | miss | evictions | useful evictions |",
                    "|---:|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for rank, row in enumerate(selected[: min(top_k, 12)], 1):
                lines.append(
                    f"| {rank} | `{row['label']}` | {fmt(row.get('relative_score'))} | "
                    f"{fmt(row.get('hit_ratio'), 4)} | "
                    f"{fmt(row.get('full_hit_rate'), 4)} | "
                    f"{fmt(row.get('miss_blocks'))} | "
                    f"{fmt(row.get('evictions'))} | "
                    f"{fmt(row.get('useful_evictions'))} |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--surface",
        action="append",
        choices=sorted(SURFACES),
        help="Surface to include; default includes all built-in surfaces.",
    )
    parser.add_argument("--min-gaps", type=parse_float_list, default=parse_float_list("30,45,60"))
    parser.add_argument("--history-weights", type=parse_float_list, default=parse_float_list("4,8,12"))
    parser.add_argument("--response-weights", type=parse_float_list, default=parse_float_list("0,2,4"))
    parser.add_argument("--loss-weights", type=parse_float_list, default=parse_float_list("0.5,1,2"))
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument("--useful-eviction-weight", type=float, default=0.25)
    parser.add_argument("--eviction-weight", type=float, default=0.02)
    parser.add_argument("--full-hit-weight", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selected_surfaces = [SURFACES[name] for name in (args.surface or sorted(SURFACES))]
    variants = []
    seen = set()
    for min_gap in args.min_gaps:
        for history_weight in args.history_weights:
            for response_weight in args.response_weights:
                for loss_weight in args.loss_weights:
                    key = (min_gap, history_weight, response_weight, loss_weight)
                    if key in seen:
                        continue
                    seen.add(key)
                    variants.append(key)

    rows: list[dict[str, Any]] = []
    for surface in selected_surfaces:
        entries = select_surface_entries(surface)
        for capacity_tokens in surface.capacities:
            if not args.quiet:
                print(
                    (
                        f"[sizecost-sweep] surface={surface.name} "
                        f"capacity={capacity_tokens} variants={len(variants)}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            capacity_blocks = max(1, capacity_tokens // args.block_size_tokens)
            lru = sim.simulate_policy(
                entries,
                policy="lru",
                capacity_blocks=capacity_blocks,
                block_size_tokens=args.block_size_tokens,
            )
            if args.include_controls:
                for policy in CONTROL_POLICIES:
                    control = sim.simulate_policy(
                        entries,
                        policy=policy,
                        capacity_blocks=capacity_blocks,
                        block_size_tokens=args.block_size_tokens,
                    )
                    rows.append(
                        annotate_result(
                            surface=surface,
                            capacity_tokens=capacity_tokens,
                            label=policy,
                            policy=policy,
                            row=control,
                            lru=lru,
                            useful_weight=args.useful_eviction_weight,
                            eviction_weight=args.eviction_weight,
                            full_hit_weight=args.full_hit_weight,
                        )
                    )
            for min_gap, history_weight, response_weight, loss_weight in variants:
                apply_size_params(
                    min_gap=min_gap,
                    history_weight=history_weight,
                    response_weight=response_weight,
                    loss_weight=loss_weight,
                )
                result = sim.simulate_policy(
                    entries,
                    policy="expected_gap_size_cost_aware",
                    capacity_blocks=capacity_blocks,
                    block_size_tokens=args.block_size_tokens,
                )
                label = (
                    f"sizecost_m{min_gap:g}_h{history_weight:g}_"
                    f"r{response_weight:g}_l{loss_weight:g}"
                )
                rows.append(
                    annotate_result(
                        surface=surface,
                        capacity_tokens=capacity_tokens,
                        label=label,
                        policy="expected_gap_size_cost_aware",
                        row=result,
                        lru=lru,
                        useful_weight=args.useful_eviction_weight,
                        eviction_weight=args.eviction_weight,
                        full_hit_weight=args.full_hit_weight,
                        params={
                            "size_bonus_min_gap_sec": min_gap,
                            "size_history_weight": history_weight,
                            "size_response_weight": response_weight,
                            "size_loss_weight": loss_weight,
                        },
                    )
                )

    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "surfaces": [surface.name for surface in selected_surfaces],
            "variants": len(variants),
            "min_gaps": args.min_gaps,
            "history_weights": args.history_weights,
            "response_weights": args.response_weights,
            "loss_weights": args.loss_weights,
            "include_controls": args.include_controls,
            "block_size_tokens": args.block_size_tokens,
            "useful_eviction_weight": args.useful_eviction_weight,
            "eviction_weight": args.eviction_weight,
            "full_hit_weight": args.full_hit_weight,
        },
        "summary": summary,
        "rows": rows,
    }
    (args.output_dir / "sizecost_param_sweep.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fields = [
        "surface",
        "capacity_tokens",
        "label",
        "policy",
        "relative_score",
        "tail_risk_reduction_pct",
        "full_hit_gain_requests",
        "hit_ratio",
        "full_hit_rate",
        "miss_blocks",
        "evictions",
        "useful_evictions",
        "size_bonus_min_gap_sec",
        "size_history_weight",
        "size_response_weight",
        "size_loss_weight",
    ]
    with (args.output_dir / "sizecost_param_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    write_markdown(
        args.output_dir / "sizecost_param_sweep.md",
        summary=summary,
        rows=rows,
        top_k=max(1, args.top_k),
    )
    print(args.output_dir / "sizecost_param_sweep.json")
    print(args.output_dir / "sizecost_param_sweep.csv")
    print(args.output_dir / "sizecost_param_sweep.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
