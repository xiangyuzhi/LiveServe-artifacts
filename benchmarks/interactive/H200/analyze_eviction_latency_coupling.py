#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSpec:
    label: str
    root: Path


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(math.ceil(q * len(values))) - 1))
    return values[idx]


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "mean": sum(values) / len(values) if values else None,
    }


def parse_run(raw: str) -> RunSpec:
    if "=" not in raw:
        path = Path(raw)
        return RunSpec(path.name, path)
    label, path = raw.split("=", 1)
    return RunSpec(label.strip(), Path(path.strip()))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def find_variant_roots(root: Path) -> list[Path]:
    variants: list[Path] = []
    for child in sorted(root.iterdir() if root.exists() else []):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if (child / "_probe" / "kv_eviction_shadow.jsonl").exists():
            variants.append(child)
    return variants


def find_per_request(variant_root: Path) -> Path | None:
    matches = sorted(variant_root.glob("*/*per_request.jsonl"))
    if not matches:
        matches = sorted(variant_root.glob("*/per_request.jsonl"))
    return matches[0] if matches else None


def finite_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def event_features(event: dict[str, Any]) -> dict[str, float]:
    applied = 1.0 if bool(event.get("applied")) else 0.0
    replacement = finite_float(event, "replacement_count")
    if replacement is None:
        replacement = finite_float(event, "selected_cached_victims")
    if replacement is None:
        selected_block_ids = event.get("selected_block_ids")
        if isinstance(selected_block_ids, list):
            replacement = float(len(selected_block_ids))
    replacement = replacement or 0.0
    cached = finite_float(event, "cached_victim_count") or 0.0
    requested = finite_float(event, "num_requested_blocks") or 0.0
    return {
        "events": 1.0,
        "applied": applied,
        "fallback": 0.0 if applied else 1.0,
        "replacement_count": replacement,
        "cached_victim_count": cached,
        "num_requested_blocks": requested,
    }


def sum_features(events: list[dict[str, Any]]) -> dict[str, float]:
    totals = {
        "events": 0.0,
        "applied": 0.0,
        "fallback": 0.0,
        "replacement_count": 0.0,
        "cached_victim_count": 0.0,
        "num_requested_blocks": 0.0,
    }
    for event in events:
        for key, value in event_features(event).items():
            totals[key] += value
    return totals


def bucket_id(value: float, bucket_sec: float) -> int:
    return int(math.floor(max(0.0, value) / max(0.001, bucket_sec)))


def analyze_variant(
    label: str,
    variant_root: Path,
    *,
    bucket_sec: float,
    pre_window_sec: float,
) -> dict[str, Any]:
    per_request_path = find_per_request(variant_root)
    event_path = variant_root / "_probe" / "kv_eviction_shadow.jsonl"
    requests_raw = read_jsonl(per_request_path) if per_request_path else []
    events_raw = read_jsonl(event_path)

    requests = [
        row for row in requests_raw
        if row.get("error") is None
        and finite_float(row, "issued_at") is not None
        and finite_float(row, "e2e_sec") is not None
    ]
    event_ts_values = [
        float(row["ts"]) for row in events_raw
        if isinstance(row.get("ts"), (int, float))
    ]
    request_start_values = [float(row["issued_at"]) for row in requests]
    if not requests or not event_ts_values:
        return {
            "label": label,
            "variant": variant_root.name,
            "per_request_path": str(per_request_path) if per_request_path else None,
            "event_path": str(event_path),
            "requests": len(requests),
            "events": len(events_raw),
            "error": "missing requests or eviction events",
        }

    request_t0 = min(request_start_values)
    event_t0 = min(event_ts_values)
    events: list[dict[str, Any]] = []
    for row in events_raw:
        ts = finite_float(row, "ts")
        if ts is None:
            continue
        copied = dict(row)
        copied["_rel_ts"] = ts - event_t0
        events.append(copied)
    events.sort(key=lambda row: row["_rel_ts"])

    audio_values = [
        value for row in requests
        if (value := finite_float(row, "audio_ttfp_sec")) is not None
    ]
    e2e_values = [
        value for row in requests
        if (value := finite_float(row, "e2e_sec")) is not None
    ]
    audio_p90 = percentile(audio_values, 0.90)
    e2e_p90 = percentile(e2e_values, 0.90)

    event_by_bucket: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        event_by_bucket.setdefault(
            bucket_id(float(event["_rel_ts"]), bucket_sec),
            [],
        ).append(event)

    request_rows: list[dict[str, Any]] = []
    for row in requests:
        start_rel = float(row["issued_at"]) - request_t0
        e2e = finite_float(row, "e2e_sec") or 0.0
        end_rel = start_rel + e2e
        pre_start = max(0.0, start_rel - pre_window_sec)
        in_flight = [
            event for event in events
            if start_rel <= float(event["_rel_ts"]) <= end_rel
        ]
        pre = [
            event for event in events
            if pre_start <= float(event["_rel_ts"]) < start_rel
        ]
        request_rows.append(
            {
                "request_id": row.get("request_id"),
                "start_rel": start_rel,
                "end_rel": end_rel,
                "audio_ttfp_sec": finite_float(row, "audio_ttfp_sec"),
                "e2e_sec": e2e,
                "response_length": row.get("response_length"),
                "query_length": row.get("query_length"),
                "kv_prefix_token_estimate": row.get("kv_prefix_token_estimate"),
                "returning_prefix": (
                    isinstance(row.get("kv_prefix_token_estimate"), (int, float))
                    and float(row.get("kv_prefix_token_estimate") or 0.0) > 0.0
                ),
                "post_cold_10s": start_rel >= 10.0,
                "post_cold_20s": start_rel >= 20.0,
                "audio_tail_p90": (
                    audio_p90 is not None
                    and (finite_float(row, "audio_ttfp_sec") or -1.0) >= audio_p90
                ),
                "e2e_tail_p90": e2e_p90 is not None and e2e >= e2e_p90,
                "inflight_events": sum_features(in_flight),
                "pre_events": sum_features(pre),
            }
        )

    def group_stats(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        inflight_applied = [row["inflight_events"]["applied"] for row in rows]
        pre_applied = [row["pre_events"]["applied"] for row in rows]
        inflight_events = [row["inflight_events"]["events"] for row in rows]
        pre_events = [row["pre_events"]["events"] for row in rows]
        return {
            "name": name,
            "requests": len(rows),
            "audio_ttfp_sec": summary([
                value for row in rows
                if (value := row.get("audio_ttfp_sec")) is not None
            ]),
            "e2e_sec": summary([float(row["e2e_sec"]) for row in rows]),
            "inflight_applied": summary(inflight_applied),
            "pre_applied": summary(pre_applied),
            "inflight_events": summary(inflight_events),
            "pre_events": summary(pre_events),
        }

    groups = [
        group_stats("all", request_rows),
        group_stats(
            "audio_p90_tail",
            [row for row in request_rows if row["audio_tail_p90"]],
        ),
        group_stats(
            "audio_non_tail",
            [row for row in request_rows if not row["audio_tail_p90"]],
        ),
        group_stats(
            "e2e_p90_tail",
            [row for row in request_rows if row["e2e_tail_p90"]],
        ),
        group_stats(
            "e2e_non_tail",
            [row for row in request_rows if not row["e2e_tail_p90"]],
        ),
        group_stats(
            "returning_prefix",
            [row for row in request_rows if row["returning_prefix"]],
        ),
        group_stats(
            "post_cold_10s",
            [row for row in request_rows if row["post_cold_10s"]],
        ),
        group_stats(
            "post_cold_20s",
            [row for row in request_rows if row["post_cold_20s"]],
        ),
    ]

    bucket_rows: list[dict[str, Any]] = []
    request_by_bucket: dict[int, list[dict[str, Any]]] = {}
    for row in request_rows:
        request_by_bucket.setdefault(
            bucket_id(float(row["start_rel"]), bucket_sec),
            [],
        ).append(row)
    all_bucket_ids = sorted(set(event_by_bucket) | set(request_by_bucket))
    for idx in all_bucket_ids:
        bucket_events = event_by_bucket.get(idx, [])
        bucket_requests = request_by_bucket.get(idx, [])
        reasons = Counter(str(event.get("reason")) for event in bucket_events)
        features = sum_features(bucket_events)
        bucket_rows.append(
            {
                "bucket": idx,
                "start_sec": idx * bucket_sec,
                "limit_sec": (idx + 1) * bucket_sec,
                "requests": len(bucket_requests),
                "audio_p90": percentile([
                    value for row in bucket_requests
                    if (value := row.get("audio_ttfp_sec")) is not None
                ], 0.90),
                "e2e_p90": percentile([
                    float(row["e2e_sec"]) for row in bucket_requests
                ], 0.90),
                "events": int(features["events"]),
                "applied": int(features["applied"]),
                "fallback": int(features["fallback"]),
                "replacement_count": features["replacement_count"],
                "cached_victim_count": features["cached_victim_count"],
                "top_reasons": reasons.most_common(4),
            }
        )

    return {
        "label": label,
        "variant": variant_root.name,
        "per_request_path": str(per_request_path),
        "event_path": str(event_path),
        "clock_alignment": {
            "method": "relative_first_record",
            "request_t0_perf_counter": request_t0,
            "event_t0_epoch": event_t0,
            "approximate": True,
        },
        "config": {
            "bucket_sec": bucket_sec,
            "pre_window_sec": pre_window_sec,
        },
        "requests": len(request_rows),
        "events": len(events),
        "audio_p90": audio_p90,
        "e2e_p90": e2e_p90,
        "event_reasons": Counter(str(event.get("reason")) for event in events).most_common(),
        "groups": groups,
        "buckets": bucket_rows,
    }


def write_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Eviction Latency Coupling",
        "",
        "Request timestamps and eviction-event timestamps use different clocks, so "
        "this analysis aligns each stream by its first record. Treat bucket-level "
        "patterns as approximate, not per-request causal proof.",
        "",
        "## Tail Group Summary",
        "",
        "| run | variant | group | requests | audio_p90 | e2e_p90 | "
        "inflight_applied_mean | pre_applied_mean | inflight_events_mean |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        if "error" in result:
            lines.append(
                f"| {result['label']} | {result['variant']} | error: "
                f"{result['error']} | {result.get('requests', 0)} | | | | | |"
            )
            continue
        for group in result["groups"]:
            lines.append(
                f"| {result['label']} | {result['variant']} | {group['name']} | "
                f"{group['requests']} | "
                f"{(group['audio_ttfp_sec']['p90'] or 0):.3f} | "
                f"{(group['e2e_sec']['p90'] or 0):.3f} | "
                f"{(group['inflight_applied']['mean'] or 0):.3f} | "
                f"{(group['pre_applied']['mean'] or 0):.3f} | "
                f"{(group['inflight_events']['mean'] or 0):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Busiest Event Buckets",
            "",
            "| run | variant | sec | requests | audio_p90 | e2e_p90 | "
            "events | applied | replacement_count | top_reasons |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for result in results:
        if "error" in result:
            continue
        buckets = sorted(
            result["buckets"],
            key=lambda row: (row["applied"], row["events"], row["requests"]),
            reverse=True,
        )[:12]
        for row in buckets:
            reasons = ", ".join(f"{name}:{count}" for name, count in row["top_reasons"])
            lines.append(
                f"| {result['label']} | {result['variant']} | "
                f"{row['start_sec']:.0f}-{row['limit_sec']:.0f} | "
                f"{row['requests']} | "
                f"{(row['audio_p90'] or 0):.3f} | "
                f"{(row['e2e_p90'] or 0):.3f} | "
                f"{row['events']} | {row['applied']} | "
                f"{row['replacement_count']:.1f} | {reasons} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="label=/path/to/run_root or /path/to/run_root",
    )
    parser.add_argument("--bucket-sec", type=float, default=5.0)
    parser.add_argument("--pre-window-sec", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    for spec in [parse_run(raw) for raw in args.run]:
        variants = find_variant_roots(spec.root)
        if not variants:
            results.append(
                {
                    "label": spec.label,
                    "variant": "",
                    "requests": 0,
                    "events": 0,
                    "error": "no variant roots with kv_eviction_shadow.jsonl",
                }
            )
            continue
        for variant_root in variants:
            results.append(
                analyze_variant(
                    spec.label,
                    variant_root,
                    bucket_sec=args.bucket_sec,
                    pre_window_sec=args.pre_window_sec,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "eviction_latency_coupling.json"
    md_path = args.output_dir / "eviction_latency_coupling.md"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, results)
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
