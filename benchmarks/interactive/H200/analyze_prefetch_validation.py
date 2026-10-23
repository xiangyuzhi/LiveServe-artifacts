#!/usr/bin/env python3
"""Summarize prefetch validation runs.

The report joins replay summaries, prefetch rows, live per-request lookup state,
KV LRU counters, and optional paired deltas against an LRU baseline.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def find_cell(root: Path, variant: str) -> Path | None:
    base = root / variant
    if not base.exists():
        return None
    cells = sorted(
        p for p in base.iterdir()
        if p.is_dir() and (p / "summary.json").exists()
    )
    if not cells:
        return None
    if len(cells) > 1:
        cells.sort(key=lambda p: (p / "summary.json").stat().st_mtime, reverse=True)
    return cells[0]


def q(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[idx]


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def measurement_window(summary: dict[str, Any]) -> tuple[float | None, float | None]:
    measurement = summary.get("measurement") or {}
    start = measurement.get("metrics_start_trace_time")
    end = measurement.get("metrics_end_trace_time")
    try:
        start_f = float(start) if start not in (None, "") else None
    except (TypeError, ValueError):
        start_f = None
    try:
        end_f = float(end) if end not in (None, "") else None
    except (TypeError, ValueError):
        end_f = None
    return start_f, end_f


def in_window(row: dict[str, Any], start: float | None, end: float | None) -> bool:
    ts = row.get("timestamp")
    if not isinstance(ts, (int, float)):
        return False
    if start is not None and float(ts) < start:
        return False
    if end is not None and float(ts) >= end:
        return False
    return True


def counter_to_dict(counter: collections.Counter[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in counter.most_common():
        out[str(key)] = int(value)
    return out


def metric_delta(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    base_by_key = {
        (row.get("user_id"), row.get("round_index"), row.get("timestamp")): row
        for row in baseline
        if not row.get("error")
    }
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in candidate:
        if row.get("error"):
            continue
        key = (row.get("user_id"), row.get("round_index"), row.get("timestamp"))
        base = base_by_key.get(key)
        if base is not None:
            pairs.append((base, row))

    def summarize_pairs(
        selected_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> dict[str, Any]:
        return {
            "ttfp_sec": summarize_delta(selected_pairs, "ttfp_sec"),
            "e2e_sec": summarize_delta(selected_pairs, "e2e_sec"),
            "tokens_per_sec": summarize_delta(
                selected_pairs, "tokens_per_sec", higher_better=True
            ),
            "output_tokens": summarize_delta(selected_pairs, "output_tokens"),
        }

    def summarize_delta(
        selected_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
        metric: str,
        *,
        higher_better: bool = False,
    ) -> dict[str, Any]:
        deltas: list[float] = []
        for base, cand in selected_pairs:
            b_val = base.get(metric)
            c_val = cand.get(metric)
            if isinstance(b_val, (int, float)) and isinstance(c_val, (int, float)):
                deltas.append(float(c_val) - float(b_val))
        if not deltas:
            return {}
        if higher_better:
            improved = sum(1 for value in deltas if value > 0)
        else:
            improved = sum(1 for value in deltas if value < 0)
        return {
            "n": len(deltas),
            "mean": mean(deltas),
            "p50": q(deltas, 0.50),
            "p90": q(deltas, 0.90),
            "improved_rate": improved / len(deltas),
        }

    by_hit: dict[str, dict[str, Any]] = {}
    for hit_value in (True, False):
        sub_pairs = [
            (base, cand)
            for base, cand in pairs
            if bool(cand.get("kv_prefetch_hit_used")) == hit_value
        ]
        by_hit[str(hit_value).lower()] = {
            "pairs": len(sub_pairs),
            "all": summarize_pairs(sub_pairs) if sub_pairs else {},
        }

    return {
        "pairs": len(pairs),
        "all": summarize_pairs(pairs),
        "by_candidate_hit": by_hit,
    }


def variant_summary(cell: Path, start: float | None, end: float | None) -> dict[str, Any]:
    summary = read_json(cell / "summary.json")
    rows = read_jsonl(cell / "per_request.jsonl")
    window_rows = [row for row in rows if in_window(row, start, end)]
    ok_window = [row for row in window_rows if not row.get("error")]
    prefetch_rows = read_jsonl(cell / "prefetch.jsonl")
    prefetch_window = [row for row in prefetch_rows if in_window(row, start, end)]
    errors = [row for row in rows if row.get("error")]

    live_states = collections.Counter(
        row.get("kv_prefetch_lookup_state") for row in ok_window
    )
    live_scopes = collections.Counter(
        row.get("kv_prefetch_warm_scope") for row in ok_window
        if row.get("kv_prefetch_warm_scope")
    )
    prefetch_skip = collections.Counter(
        row.get("skip_reason") for row in prefetch_rows if row.get("skip_reason")
    )
    prefetch_window_skip = collections.Counter(
        row.get("skip_reason") for row in prefetch_window if row.get("skip_reason")
    )

    return {
        "cell": str(cell),
        "summary": {
            "totals": summary.get("totals"),
            "latency": summary.get("latency"),
            "prefetch": summary.get("prefetch"),
            "kv_lru": {
                key: (summary.get("kv_lru") or {}).get(key)
                for key in (
                    "gpu_evictions",
                    "gpu_lookup_misses",
                    "recent_eviction_misses",
                    "recent_eviction_miss_ratio_blocks",
                )
            },
            "kv_hit_ratio": (summary.get("kv_hit_ratio") or {}).get("prefetch_proxy"),
        },
        "rows": {
            "all": len(rows),
            "errors": len(errors),
            "window": len(window_rows),
            "window_ok": len(ok_window),
            "window_errors": len(window_rows) - len(ok_window),
            "window_live_lookup_states": counter_to_dict(live_states),
            "window_live_warm_scopes": counter_to_dict(live_scopes),
            "window_warm_hits": sum(1 for row in ok_window if row.get("kv_prefetch_hit_used")),
        },
        "prefetch_rows": {
            "all": len(prefetch_rows),
            "ok": sum(1 for row in prefetch_rows if row.get("ok")),
            "window": len(prefetch_window),
            "window_ok": sum(1 for row in prefetch_window if row.get("ok")),
            "skip_reasons": counter_to_dict(prefetch_skip),
            "window_skip_reasons": counter_to_dict(prefetch_window_skip),
            "record_kind": counter_to_dict(collections.Counter(row.get("record_kind") for row in prefetch_rows)),
            "window_record_kind": counter_to_dict(collections.Counter(row.get("record_kind") for row in prefetch_window)),
        },
        "errors": [
            {
                "user_id": row.get("user_id"),
                "round_index": row.get("round_index"),
                "timestamp": row.get("timestamp"),
                "error": row.get("error"),
            }
            for row in errors[:20]
        ],
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = [
        "# Prefetch Validation",
        "",
        f"- run_root: `{report['run_root']}`",
        f"- baseline_root: `{report.get('baseline_root') or ''}`",
        f"- window: `{report['window']['start']}` to `{report['window']['end']}`",
        "",
        "## Variants",
        "",
        "| variant | ok_window | errors_all | TTFT p90 | e2e p90 | warm_hits_window | prefetch_ok_window | gpu_miss | gpu_evict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in report["variants"].items():
        latency = ((data.get("summary") or {}).get("latency") or {})
        kv_lru = ((data.get("summary") or {}).get("kv_lru") or {})
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    fmt((data.get("rows") or {}).get("window_ok"), 0),
                    fmt((data.get("rows") or {}).get("errors"), 0),
                    fmt(((latency.get("ttfp_sec") or {}).get("p90")), 4),
                    fmt(((latency.get("e2e_sec") or {}).get("p90")), 4),
                    fmt((data.get("rows") or {}).get("window_warm_hits"), 0),
                    fmt((data.get("prefetch_rows") or {}).get("window_ok"), 0),
                    fmt(kv_lru.get("gpu_lookup_misses"), 0),
                    fmt(kv_lru.get("gpu_evictions"), 0),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Prefetch Skips", ""])
    for name, data in report["variants"].items():
        prefetch = data.get("prefetch_rows") or {}
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- all skip reasons: `{prefetch.get('skip_reasons')}`")
        lines.append(f"- window skip reasons: `{prefetch.get('window_skip_reasons')}`")
        lines.append("")

    paired = report.get("paired_delta")
    if paired:
        lines.extend(["## Paired Delta", ""])
        lines.append(f"- pairs: `{paired.get('pairs')}`")
        for group_name, group in (paired.get("all") or {}).items():
            lines.append(
                f"- {group_name}: mean `{fmt(group.get('mean'), 6)}`, "
                f"p50 `{fmt(group.get('p50'), 6)}`, "
                f"p90 `{fmt(group.get('p90'), 6)}`, "
                f"improved `{fmt(group.get('improved_rate'), 3)}`"
            )
        lines.append("")
        by_hit = paired.get("by_candidate_hit") or {}
        for hit_name, hit_data in by_hit.items():
            lines.append(f"### candidate_hit={hit_name}")
            lines.append("")
            if hit_data.get("pairs") == 0:
                lines.append("- pairs: `0`")
                lines.append("")
                continue
            for group_name, group in (hit_data.get("all") or {}).items():
                lines.append(
                    f"- {group_name}: mean `{fmt(group.get('mean'), 6)}`, "
                    f"p50 `{fmt(group.get('p50'), 6)}`, "
                    f"p90 `{fmt(group.get('p90'), 6)}`, "
                    f"improved `{fmt(group.get('improved_rate'), 3)}`"
                )
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--baseline-variant", default="lru_control")
    parser.add_argument("--candidate-variant", default="prefetch_only")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    run_root = args.run_root
    baseline_root = args.baseline_root or run_root
    candidate_cell = find_cell(run_root, args.candidate_variant)
    if candidate_cell is None:
        raise FileNotFoundError(f"no cell for {args.candidate_variant} under {run_root}")
    candidate_summary = read_json(candidate_cell / "summary.json")
    start, end = measurement_window(candidate_summary)

    variants: dict[str, Any] = {
        args.candidate_variant: variant_summary(candidate_cell, start, end)
    }
    baseline_cell = find_cell(baseline_root, args.baseline_variant)
    if baseline_cell is not None:
        variants[args.baseline_variant] = variant_summary(baseline_cell, start, end)

    paired_delta = None
    if baseline_cell is not None:
        base_rows = [
            row for row in read_jsonl(baseline_cell / "per_request.jsonl")
            if in_window(row, start, end)
        ]
        cand_rows = [
            row for row in read_jsonl(candidate_cell / "per_request.jsonl")
            if in_window(row, start, end)
        ]
        paired_delta = metric_delta(base_rows, cand_rows)

    report = {
        "run_root": str(run_root),
        "baseline_root": str(baseline_root) if baseline_cell is not None else None,
        "window": {"start": start, "end": end},
        "candidate_variant": args.candidate_variant,
        "baseline_variant": args.baseline_variant if baseline_cell is not None else None,
        "variants": variants,
        "paired_delta": paired_delta,
    }

    output_dir = args.output_dir or (run_root / "_analysis" / "prefetch_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "prefetch_validation.json"
    md_path = output_dir / "prefetch_validation.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, md_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
