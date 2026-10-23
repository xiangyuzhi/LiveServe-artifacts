#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


WEIGHTS = {
    "audio_ttfp": 0.35,
    "continuity": 0.20,
    "e2e": 0.15,
    "ttft": 0.05,
}


def parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.name, path
    label, path = raw.split("=", 1)
    return label.strip(), Path(path.strip())


def find_per_request(path: Path) -> Path:
    if path.is_file():
        return path
    patterns = [
        "lru_control/*/per_request.jsonl",
        "eviction_only/*/per_request.jsonl",
        "*/*/per_request.jsonl",
        "**/per_request.jsonl",
    ]
    for pattern in patterns:
        matches = sorted(path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no per_request.jsonl under {path}")


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
        if row.get("error") is not None:
            continue
        if row.get("status") not in (None, 200):
            continue
        rows[trace_key(row)] = row
    return rows


def finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


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


def metric(row: dict[str, Any], name: str, default: float = math.inf) -> float:
    value = finite(row.get(name))
    return default if value is None else value


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audio = [value for row in rows if (value := finite(row.get("audio_ttfp_sec"))) is not None]
    e2e = [value for row in rows if (value := finite(row.get("e2e_sec"))) is not None]
    ttft = [value for row in rows if (value := finite(row.get("ttfp_sec"))) is not None]
    continuity = [
        1.0 if row.get("audio_continuity_ok") else 0.0
        for row in rows
        if row.get("audio_continuity_evaluated") is True
    ]
    return {
        "requests": len(rows),
        "audio_ttfp_p50": percentile(audio, 0.50),
        "audio_ttfp_p90": percentile(audio, 0.90),
        "audio_ttfp_p99": percentile(audio, 0.99),
        "e2e_p90": percentile(e2e, 0.90),
        "ttft_p90": percentile(ttft, 0.90),
        "continuity": sum(continuity) / len(continuity) if continuity else None,
    }


def deltas(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    continuity = None
    if isinstance(metrics.get("continuity"), (int, float)) and isinstance(baseline.get("continuity"), (int, float)):
        continuity = (metrics["continuity"] - baseline["continuity"]) * 100.0
    return {
        "audio_ttfp_p90_reduction_pct": pct_reduction(metrics.get("audio_ttfp_p90"), baseline.get("audio_ttfp_p90")),
        "e2e_p90_reduction_pct": pct_reduction(metrics.get("e2e_p90"), baseline.get("e2e_p90")),
        "ttft_p90_reduction_pct": pct_reduction(metrics.get("ttft_p90"), baseline.get("ttft_p90")),
        "continuity_delta_pp": continuity,
    }


def per_request_score(base: dict[str, Any], row: dict[str, Any]) -> float:
    def reduction(name: str) -> float:
        b = finite(base.get(name))
        c = finite(row.get(name))
        if b is None or c is None or b == 0.0:
            return 0.0
        return (b - c) / b * 100.0

    continuity_gain = 0.0
    if base.get("audio_continuity_evaluated") is True and row.get("audio_continuity_evaluated") is True:
        continuity_gain = (
            (1.0 if row.get("audio_continuity_ok") else 0.0)
            - (1.0 if base.get("audio_continuity_ok") else 0.0)
        ) * 100.0
    return (
        WEIGHTS["audio_ttfp"] * reduction("audio_ttfp_sec")
        + WEIGHTS["continuity"] * continuity_gain
        + WEIGHTS["e2e"] * reduction("e2e_sec")
        + WEIGHTS["ttft"] * reduction("ttfp_sec")
    )


def select_rows(
    labels: list[str],
    data: dict[str, dict[tuple[Any, ...], dict[str, Any]]],
    common_keys: list[tuple[Any, ...]],
    *,
    selector: str,
    baseline_label: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    chosen: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for key in common_keys:
        if selector == "audio_min":
            label = min(labels, key=lambda item: metric(data[item][key], "audio_ttfp_sec"))
        elif selector == "e2e_min":
            label = min(labels, key=lambda item: metric(data[item][key], "e2e_sec"))
        elif selector == "audio_priority":
            base = data[baseline_label][key]
            label = max(labels, key=lambda item: per_request_score(base, data[item][key]))
        else:
            raise ValueError(f"unknown selector: {selector}")
        chosen.append(data[label][key])
        counts[label] += 1
    return chosen, counts


def audio_duration_bucket(row: dict[str, Any]) -> str:
    duration = finite(row.get("audio_duration_sec")) or 0.0
    if duration < 1.0:
        return "<1s"
    if duration < 4.0:
        return "1-4s"
    if duration < 8.0:
        return "4-8s"
    if duration < 16.0:
        return "8-16s"
    return ">=16s"


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.{digits}f}"
    if value is None:
        return "-"
    return str(value)


def flat_metric_row(label: str, metrics: dict[str, Any], delta: dict[str, Any], counts: Counter[str] | None = None) -> dict[str, Any]:
    row = {"label": label, **metrics, **delta}
    if counts is not None:
        row["choice_counts"] = dict(counts)
    return row


def write_markdown(
    path: Path,
    *,
    config: dict[str, Any],
    baseline_metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Eviction Oracle Upper Bound",
        "",
        f"- baseline: {config['baseline_label']}",
        f"- candidates: {', '.join(config['candidate_labels'])}",
        f"- common requests: {config['common_requests']}",
        "",
        "This is an offline splice over completed per-request rows. It does not estimate QPS, "
        "because scheduler interaction changes when a selector is implemented online.",
        "",
        "## Overall",
        "",
        "| label | requests | audio p50 | audio p90 | audio p99 | audio p90 gain | continuity | continuity gain | e2e p90 | e2e gain | ttft p90 | ttft gain | choices |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        choices = row.get("choice_counts")
        if isinstance(choices, dict):
            choice_text = ", ".join(f"{key}:{value}" for key, value in choices.items())
        else:
            choice_text = "-"
        lines.append(
            f"| {row['label']} | {row['requests']} | "
            f"{fmt(row.get('audio_ttfp_p50'))} | {fmt(row.get('audio_ttfp_p90'))} | "
            f"{fmt(row.get('audio_ttfp_p99'))} | {fmt(row.get('audio_ttfp_p90_reduction_pct'))}% | "
            f"{fmt(row.get('continuity'))} | {fmt(row.get('continuity_delta_pp'))}pp | "
            f"{fmt(row.get('e2e_p90'))} | {fmt(row.get('e2e_p90_reduction_pct'))}% | "
            f"{fmt(row.get('ttft_p90'))} | {fmt(row.get('ttft_p90_reduction_pct'))}% | "
            f"{choice_text} |"
        )
    lines.extend(
        [
            "",
            "## Audio-Duration Buckets",
            "",
            "| selector | bucket | requests | baseline audio p90 | selector audio p90 | audio p90 gain | baseline e2e p90 | selector e2e p90 | e2e gain |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bucket_rows:
        lines.append(
            f"| {row['selector']} | {row['bucket']} | {row['requests']} | "
            f"{fmt(row.get('baseline_audio_p90'))} | {fmt(row.get('selector_audio_p90'))} | "
            f"{fmt(row.get('audio_p90_reduction_pct'))}% | "
            f"{fmt(row.get('baseline_e2e_p90'))} | {fmt(row.get('selector_e2e_p90'))} | "
            f"{fmt(row.get('e2e_p90_reduction_pct'))}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="label=/path/to/run or per_request.jsonl")
    parser.add_argument("--candidate", action="append", required=True, help="label=/path/to/run or per_request.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_label, baseline_path = parse_labeled_path(args.baseline)
    candidate_specs = [parse_labeled_path(raw) for raw in args.candidate]
    labels = [baseline_label] + [label for label, _ in candidate_specs]
    paths = {baseline_label: find_per_request(baseline_path)}
    paths.update({label: find_per_request(path) for label, path in candidate_specs})
    data = {label: read_requests(path) for label, path in paths.items()}
    common_keys = sorted(set.intersection(*(set(rows.keys()) for rows in data.values())), key=lambda key: (key[2], key[0], key[1]))
    if not common_keys:
        raise SystemExit("no common requests across variants")

    baseline_rows = [data[baseline_label][key] for key in common_keys]
    baseline_metrics = summarize(baseline_rows)
    metric_rows: list[dict[str, Any]] = [
        flat_metric_row(
            baseline_label,
            baseline_metrics,
            deltas(baseline_metrics, baseline_metrics),
        )
    ]
    for label in labels[1:]:
        rows = [data[label][key] for key in common_keys]
        metrics = summarize(rows)
        metric_rows.append(flat_metric_row(label, metrics, deltas(metrics, baseline_metrics)))

    selector_rows: dict[str, list[dict[str, Any]]] = {}
    selector_counts: dict[str, Counter[str]] = {}
    for selector in ("audio_min", "e2e_min", "audio_priority"):
        rows, counts = select_rows(labels, data, common_keys, selector=selector, baseline_label=baseline_label)
        selector_rows[selector] = rows
        selector_counts[selector] = counts
        metrics = summarize(rows)
        metric_rows.append(
            flat_metric_row(
                f"oracle:{selector}",
                metrics,
                deltas(metrics, baseline_metrics),
                counts,
            )
        )

    bucket_rows: list[dict[str, Any]] = []
    for selector, rows in selector_rows.items():
        buckets = sorted({audio_duration_bucket(row) for row in baseline_rows})
        for bucket in buckets:
            idxs = [
                idx
                for idx, row in enumerate(baseline_rows)
                if audio_duration_bucket(row) == bucket
            ]
            base_bucket = [baseline_rows[idx] for idx in idxs]
            selector_bucket = [rows[idx] for idx in idxs]
            base_metrics = summarize(base_bucket)
            selector_metrics = summarize(selector_bucket)
            bucket_rows.append(
                {
                    "selector": selector,
                    "bucket": bucket,
                    "requests": len(idxs),
                    "baseline_audio_p90": base_metrics.get("audio_ttfp_p90"),
                    "selector_audio_p90": selector_metrics.get("audio_ttfp_p90"),
                    "audio_p90_reduction_pct": pct_reduction(
                        selector_metrics.get("audio_ttfp_p90"),
                        base_metrics.get("audio_ttfp_p90"),
                    ),
                    "baseline_e2e_p90": base_metrics.get("e2e_p90"),
                    "selector_e2e_p90": selector_metrics.get("e2e_p90"),
                    "e2e_p90_reduction_pct": pct_reduction(
                        selector_metrics.get("e2e_p90"),
                        base_metrics.get("e2e_p90"),
                    ),
                }
            )

    config = {
        "baseline_label": baseline_label,
        "candidate_labels": labels[1:],
        "paths": {label: str(path) for label, path in paths.items()},
        "common_requests": len(common_keys),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "oracle_upper_bound.json"
    csv_path = args.output_dir / "oracle_upper_bound.csv"
    bucket_csv_path = args.output_dir / "oracle_upper_bound_buckets.csv"
    md_path = args.output_dir / "oracle_upper_bound.md"
    json_path.write_text(
        json.dumps(
            {
                "config": config,
                "metrics": metric_rows,
                "bucket_metrics": bucket_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "label",
            "requests",
            "audio_ttfp_p50",
            "audio_ttfp_p90",
            "audio_ttfp_p99",
            "audio_ttfp_p90_reduction_pct",
            "continuity",
            "continuity_delta_pp",
            "e2e_p90",
            "e2e_p90_reduction_pct",
            "ttft_p90",
            "ttft_p90_reduction_pct",
            "choice_counts",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metric_rows)
    with bucket_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bucket_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bucket_rows)
    write_markdown(
        md_path,
        config=config,
        baseline_metrics=baseline_metrics,
        rows=metric_rows,
        bucket_rows=bucket_rows,
    )
    print(json_path)
    print(csv_path)
    print(bucket_csv_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
