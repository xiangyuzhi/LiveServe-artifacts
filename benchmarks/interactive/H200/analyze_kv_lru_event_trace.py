#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class RunSpec:
    label: str
    root: Path


def parse_run(raw: str) -> RunSpec:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return RunSpec(label.strip(), Path(path.strip()))
    path = Path(raw)
    return RunSpec(path.name, path)


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def find_variant_roots(root: Path) -> list[Path]:
    if (root / "_probe" / "kv_lru_events.jsonl").exists():
        return [root]
    variants: list[Path] = []
    for child in sorted(root.iterdir() if root.exists() else []):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if (child / "_probe" / "kv_lru_events.jsonl").exists():
            variants.append(child)
    return variants


def first_match(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_per_request(variant_root: Path) -> Path | None:
    return first_match(
        variant_root,
        [
            "per_request.jsonl",
            "*/per_request.jsonl",
            "*/*/per_request.jsonl",
            "**/per_request.jsonl",
        ],
    )


def find_summary(variant_root: Path) -> Path | None:
    return first_match(
        variant_root,
        [
            "summary.json",
            "*/summary.json",
            "*/*/summary.json",
            "**/summary.json",
        ],
    )


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isfinite(out):
            return out
    return None


def finite_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    try:
        return int(str(value))
    except Exception:
        return None


def numeric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = finite_float(row.get(key))
    return default if value is None else value


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[idx]


def describe(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def normalize_request_id(value: Any) -> str | None:
    if value is None:
        return None
    out = str(value)
    for prefix in ("chatcmpl-", "cmpl-"):
        if out.startswith(prefix):
            out = out[len(prefix):]
    return out or None


def trace_key_from_request(row: dict[str, Any]) -> tuple[str, int, float] | None:
    user_id = row.get("user_id")
    round_index = finite_int(row.get("round_index"))
    timestamp = finite_float(row.get("timestamp"))
    if user_id is None or round_index is None or timestamp is None:
        return None
    return (str(user_id), round_index, timestamp)


def trace_key_from_meta(meta: dict[str, Any]) -> tuple[str, int, float] | None:
    session_id = meta.get("omni_session_id")
    round_index = finite_int(meta.get("omni_round_index"))
    timestamp = finite_float(meta.get("omni_trace_timestamp"))
    if session_id is None or round_index is None or timestamp is None:
        return None
    return (str(session_id), round_index, timestamp)


def bucket_value(value: float, cuts: list[float], labels: list[str]) -> str:
    for cut, label in zip(cuts, labels):
        if value < cut:
            return label
    return labels[-1]


def request_static_groups(row: dict[str, Any]) -> dict[str, str]:
    prefix = numeric(row, "kv_prefix_token_estimate", 0.0)
    response = numeric(row, "response_length", 0.0)
    audio = numeric(row, "audio_duration_sec", 0.0)
    round_index = numeric(row, "round_index", 0.0)
    timestamp = numeric(row, "timestamp", 0.0)
    return {
        "all": "all",
        "prefix": "returning" if prefix > 0 else "cold",
        "prefix_bucket": bucket_value(
            prefix,
            [1, 64, 128, 256],
            ["p0", "p1_63", "p64_127", "p128_255", "p256_plus"],
        ),
        "response_bucket": bucket_value(
            response,
            [16, 32, 64, 128],
            ["r_lt16", "r16_31", "r32_63", "r64_127", "r128_plus"],
        ),
        "audio_duration_bucket": bucket_value(
            audio,
            [1, 4, 16, 64],
            ["audio_lt1", "audio1_4", "audio4_16", "audio16_64", "audio64_plus"],
        ),
        "round_bucket": bucket_value(
            round_index,
            [1, 3, 10, 30],
            ["round0", "round1_2", "round3_9", "round10_29", "round30_plus"],
        ),
        "timestamp_bucket": bucket_value(
            timestamp,
            [6900, 7040, 7120],
            ["ts_lt6900", "ts6900_7039", "ts7040_7119", "ts7120_plus"],
        ),
    }


def empty_req_stats(meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "meta": dict(meta or {}),
        "lookup_events": 0,
        "lookup_total": 0,
        "lookup_hits": 0,
        "lookup_misses": 0,
        "recent_eviction_misses": 0,
        "dedup_skipped_events": 0,
        "touch_events": 0,
        "touch_blocks": 0,
        "rescued_blocks": 0,
    }


def add_event_to_stats(stats: dict[str, Any], event: dict[str, Any]) -> None:
    kind = event.get("kind")
    if kind == "kv_lru_lookup":
        if event.get("dedup_skipped") is True:
            stats["dedup_skipped_events"] += 1
        else:
            stats["lookup_events"] += 1
            stats["lookup_total"] += int(event.get("total") or 0)
            stats["lookup_hits"] += int(event.get("hits") or 0)
            stats["lookup_misses"] += int(event.get("misses") or 0)
            stats["recent_eviction_misses"] += int(
                event.get("recent_eviction_misses") or 0
            )
    elif kind == "kv_lru_touch":
        stats["touch_events"] += 1
        stats["touch_blocks"] += int(event.get("blocks") or 0)
        stats["rescued_blocks"] += int(event.get("rescued") or 0)


def load_requests(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {"by_id": {}, "by_trace": {}}
    rows = [
        row for row in iter_jsonl(path)
        if isinstance(row, dict) and row.get("error") is None
    ]
    by_id: dict[str, dict[str, Any]] = {}
    by_trace: dict[tuple[str, int, float], dict[str, Any]] = {}
    for row in rows:
        request_id = normalize_request_id(row.get("request_id"))
        if request_id:
            by_id[request_id] = row
        trace_key = trace_key_from_request(row)
        if trace_key is not None:
            by_trace[trace_key] = row
    return rows, {"by_id": by_id, "by_trace": by_trace}


def event_request_keys(event: dict[str, Any]) -> tuple[str | None, tuple[str, int, float] | None, dict[str, Any]]:
    meta = event.get("request")
    if not isinstance(meta, dict):
        meta = {}
    request_id = normalize_request_id(meta.get("request_id"))
    return request_id, trace_key_from_meta(meta), meta


def group_metrics(
    label: str,
    rows: list[dict[str, Any]],
    *,
    wall_time: float | None,
) -> dict[str, Any]:
    audio = [
        value for row in rows
        if (value := finite_float(row.get("audio_ttfp_sec"))) is not None
    ]
    ttft = [
        value for row in rows
        if (value := finite_float(row.get("ttfp_sec"))) is not None
    ]
    e2e = [
        value for row in rows
        if (value := finite_float(row.get("e2e_sec"))) is not None
    ]
    continuity = [
        bool(row.get("audio_continuity_ok")) for row in rows
        if row.get("audio_continuity_evaluated") is True
    ]
    lookup_total = sum(int(row.get("lookup_total") or 0) for row in rows)
    lookup_hits = sum(int(row.get("lookup_hits") or 0) for row in rows)
    lookup_misses = sum(int(row.get("lookup_misses") or 0) for row in rows)
    recent = sum(int(row.get("recent_eviction_misses") or 0) for row in rows)
    n = len(rows)
    hit_depth_ratios = []
    for row in rows:
        total = int(row.get("lookup_total") or 0)
        hits = int(row.get("lookup_hits") or 0)
        if total > 0:
            hit_depth_ratios.append(hits / total)
    return {
        "group": label,
        "n": n,
        "qps_est": (n / wall_time) if wall_time and wall_time > 0 else None,
        "audio_ttfp_mean": sum(audio) / len(audio) if audio else None,
        "audio_ttfp_p50": percentile(audio, 0.50),
        "audio_ttfp_p90": percentile(audio, 0.90),
        "ttft_p90": percentile(ttft, 0.90),
        "e2e_mean": sum(e2e) / len(e2e) if e2e else None,
        "e2e_p90": percentile(e2e, 0.90),
        "continuity_ok_rate": (
            sum(continuity) / len(continuity) if continuity else None
        ),
        "continuity_n": len(continuity),
        "lookup_total": lookup_total,
        "lookup_total_mean": lookup_total / n if n else None,
        "lookup_hit_mean": lookup_hits / n if n else None,
        "lookup_miss_mean": lookup_misses / n if n else None,
        "recent_miss_mean": recent / n if n else None,
        "lookup_hit_ratio": lookup_hits / lookup_total if lookup_total else None,
        "lookup_miss_ratio": lookup_misses / lookup_total if lookup_total else None,
        "recent_miss_ratio": recent / lookup_total if lookup_total else None,
        "hit_depth_ratio_p50": percentile(hit_depth_ratios, 0.50),
        "hit_depth_ratio_p90": percentile(hit_depth_ratios, 0.90),
        "recent_request_rate": (
            sum(int(row.get("recent_eviction_misses") or 0) > 0 for row in rows) / n
            if n else None
        ),
        "lookup_hits": lookup_hits,
        "lookup_misses": lookup_misses,
        "recent_eviction_misses": recent,
    }


def analyze_variant(run_label: str, variant_root: Path) -> dict[str, Any]:
    event_path = variant_root / "_probe" / "kv_lru_events.jsonl"
    per_request_path = find_per_request(variant_root)
    summary_path = find_summary(variant_root)
    summary = read_json(summary_path) if summary_path else {}
    totals = summary.get("totals") if isinstance(summary.get("totals"), dict) else {}
    wall_time = finite_float(totals.get("wall_time_sec"))

    requests, request_indexes = load_requests(per_request_path)
    req_stats_by_key: dict[tuple[str, Any], dict[str, Any]] = {}
    event_counts: dict[str, int] = defaultdict(int)
    no_request_events = 0
    unmatched_stats = 0

    for event in iter_jsonl(event_path):
        if not isinstance(event, dict):
            continue
        kind = str(event.get("kind") or "unknown")
        event_counts[kind] += 1
        if kind == "kv_lru_eviction":
            continue
        request_id, trace_key, meta = event_request_keys(event)
        key: tuple[str, Any] | None = None
        if request_id:
            key = ("id", request_id)
        elif trace_key is not None:
            key = ("trace", trace_key)
        if key is None:
            no_request_events += 1
            continue
        stats = req_stats_by_key.setdefault(key, empty_req_stats(meta))
        if not stats["meta"] and meta:
            stats["meta"] = dict(meta)
        add_event_to_stats(stats, event)

    rows: list[dict[str, Any]] = []
    matched_keys: set[tuple[str, Any]] = set()
    by_id: dict[str, dict[str, Any]] = request_indexes["by_id"]
    by_trace: dict[tuple[str, int, float], dict[str, Any]] = request_indexes["by_trace"]
    for key, stats in req_stats_by_key.items():
        row = None
        if key[0] == "id":
            row = by_id.get(str(key[1]))
            if row is None:
                trace_key = trace_key_from_meta(stats.get("meta") or {})
                row = by_trace.get(trace_key) if trace_key is not None else None
        elif key[0] == "trace":
            row = by_trace.get(key[1])
        if row is None:
            unmatched_stats += 1
            continue
        merged = dict(row)
        merged.update({k: v for k, v in stats.items() if k != "meta"})
        rows.append(merged)
        matched_keys.add(key)

    request_ids_with_stats = {
        normalize_request_id(row.get("request_id")) for row in rows
    }
    request_ids_with_stats.discard(None)
    trace_keys_with_stats = {
        trace_key_from_request(row) for row in rows
    }
    trace_keys_with_stats.discard(None)
    for row in requests:
        request_id = normalize_request_id(row.get("request_id"))
        trace_key = trace_key_from_request(row)
        if request_id in request_ids_with_stats or trace_key in trace_keys_with_stats:
            continue
        merged = dict(row)
        merged.update(empty_req_stats())
        rows.append(merged)

    audio_values = [
        value for row in rows
        if (value := finite_float(row.get("audio_ttfp_sec"))) is not None
    ]
    e2e_values = [
        value for row in rows
        if (value := finite_float(row.get("e2e_sec"))) is not None
    ]
    audio_p90 = percentile(audio_values, 0.90)
    e2e_p90 = percentile(e2e_values, 0.90)

    predicates: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("all", lambda row: True),
        ("lookup_any", lambda row: int(row.get("lookup_total") or 0) > 0),
        ("lookup_none", lambda row: int(row.get("lookup_total") or 0) == 0),
        ("lookup_miss_any", lambda row: int(row.get("lookup_misses") or 0) > 0),
        (
            "recent_eviction_miss_any",
            lambda row: int(row.get("recent_eviction_misses") or 0) > 0,
        ),
        (
            "lookup_miss_without_recent",
            lambda row: int(row.get("lookup_misses") or 0) > 0
            and int(row.get("recent_eviction_misses") or 0) == 0,
        ),
        (
            "audio_ttfp_p90_tail",
            lambda row: audio_p90 is not None
            and (finite_float(row.get("audio_ttfp_sec")) or -1.0) >= audio_p90,
        ),
        (
            "e2e_p90_tail",
            lambda row: e2e_p90 is not None
            and (finite_float(row.get("e2e_sec")) or -1.0) >= e2e_p90,
        ),
    ]
    for group_type in [
        "prefix",
        "prefix_bucket",
        "response_bucket",
        "audio_duration_bucket",
        "round_bucket",
        "timestamp_bucket",
    ]:
        values = sorted({request_static_groups(row)[group_type] for row in rows})
        for value in values:
            predicates.append(
                (
                    f"{group_type}:{value}",
                    lambda row, group_type=group_type, value=value: (
                        request_static_groups(row)[group_type] == value
                    ),
                )
            )

    group_rows = [
        group_metrics(name, [row for row in rows if predicate(row)], wall_time=wall_time)
        for name, predicate in predicates
    ]

    top_rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("recent_eviction_misses") or 0),
            int(row.get("lookup_misses") or 0),
            finite_float(row.get("audio_ttfp_sec")) or 0.0,
        ),
        reverse=True,
    )

    return {
        "run": run_label,
        "variant": variant_root.name,
        "variant_root": str(variant_root),
        "event_path": str(event_path),
        "per_request_path": str(per_request_path) if per_request_path else None,
        "summary_path": str(summary_path) if summary_path else None,
        "request_rows": len(requests),
        "matched_request_rows": len(rows),
        "req_stats_keys": len(req_stats_by_key),
        "unmatched_stats": unmatched_stats,
        "no_request_events": no_request_events,
        "event_counts": dict(sorted(event_counts.items())),
        "summary_totals": totals,
        "latency": {
            "audio_ttfp_sec": describe(audio_values),
            "e2e_sec": describe(e2e_values),
        },
        "groups": group_rows,
        "top_requests": top_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_group(run_row: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    out = {
        "run": run_row["run"],
        "variant": run_row["variant"],
    }
    out.update(group)
    return out


def flatten_top_request(run_row: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": run_row["run"],
        "variant": run_row["variant"],
        "request_id": row.get("request_id"),
        "user_id": row.get("user_id"),
        "round_index": row.get("round_index"),
        "timestamp": row.get("timestamp"),
        "audio_ttfp_sec": row.get("audio_ttfp_sec"),
        "ttft_sec": row.get("ttfp_sec"),
        "e2e_sec": row.get("e2e_sec"),
        "audio_continuity_ok": row.get("audio_continuity_ok"),
        "kv_prefix_token_estimate": row.get("kv_prefix_token_estimate"),
        "query_length": row.get("query_length"),
        "response_length": row.get("response_length"),
        "audio_duration_sec": row.get("audio_duration_sec"),
        "lookup_events": row.get("lookup_events"),
        "lookup_total": row.get("lookup_total"),
        "lookup_hits": row.get("lookup_hits"),
        "lookup_misses": row.get("lookup_misses"),
        "recent_eviction_misses": row.get("recent_eviction_misses"),
        "touch_events": row.get("touch_events"),
        "touch_blocks": row.get("touch_blocks"),
        "rescued_blocks": row.get("rescued_blocks"),
    }


def markdown_report(results: list[dict[str, Any]], top_n: int) -> str:
    lines = [
        "# KV LRU Event Trace Analysis",
        "",
        "TTFP here means `audio_ttfp_sec`. `ttft_p90` is the thinker/text first-token latency from `ttfp_sec`.",
        "",
        "## Runs",
        "",
        "| run | variant | requests | stats keys | unmatched | events | audio p90 | e2e p90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        event_count = sum(int(v) for v in result["event_counts"].values())
        lines.append(
            f"| {result['run']} | {result['variant']} | "
            f"{result['matched_request_rows']} | {result['req_stats_keys']} | "
            f"{result['unmatched_stats']} | {event_count} | "
            f"{fmt(result['latency']['audio_ttfp_sec']['p90'])} | "
            f"{fmt(result['latency']['e2e_sec']['p90'])} |"
        )

    lines.extend([
        "",
        "## Key Groups",
        "",
        "| run | variant | group | n | qps est | audio p90 | ttft p90 | e2e p90 | continuity | lookup/request | hit/request | miss/request | hit ratio | hit-depth p50 | recent/request | recent req rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    important = {
        "all",
        "lookup_any",
        "lookup_miss_any",
        "recent_eviction_miss_any",
        "lookup_miss_without_recent",
        "audio_ttfp_p90_tail",
        "e2e_p90_tail",
        "prefix:cold",
        "prefix:returning",
        "audio_duration_bucket:audio_lt1",
        "audio_duration_bucket:audio16_64",
        "audio_duration_bucket:audio64_plus",
    }
    for result in results:
        for group in result["groups"]:
            if group["group"] not in important:
                continue
            lines.append(
                f"| {result['run']} | {result['variant']} | {group['group']} | "
                f"{group['n']} | {fmt(group['qps_est'], 4)} | "
                f"{fmt(group['audio_ttfp_p90'])} | {fmt(group['ttft_p90'])} | "
                f"{fmt(group['e2e_p90'])} | {fmt(group['continuity_ok_rate'], 4)} | "
                f"{fmt(group['lookup_total_mean'])} | {fmt(group['lookup_hit_mean'])} | "
                f"{fmt(group['lookup_miss_mean'])} | {fmt(group['lookup_hit_ratio'], 4)} | "
                f"{fmt(group['hit_depth_ratio_p50'], 4)} | {fmt(group['recent_miss_mean'])} | "
                f"{fmt(group['recent_request_rate'], 4)} |"
            )

    lines.extend([
        "",
        f"## Top {top_n} Requests By Recent Eviction Misses",
        "",
        "| run | variant | request | trace | audio | ttft | e2e | prefix | response | lookups | misses | recent | continuity |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for result in results:
        for row in result["top_requests"][:top_n]:
            lines.append(
                f"| {result['run']} | {result['variant']} | {row.get('request_id')} | "
                f"{row.get('user_id')}/r{row.get('round_index')}/t{row.get('timestamp')} | "
                f"{fmt(finite_float(row.get('audio_ttfp_sec')))} | "
                f"{fmt(finite_float(row.get('ttfp_sec')))} | "
                f"{fmt(finite_float(row.get('e2e_sec')))} | "
                f"{row.get('kv_prefix_token_estimate')} | {row.get('response_length')} | "
                f"{row.get('lookup_total')} | {row.get('lookup_misses')} | "
                f"{row.get('recent_eviction_misses')} | {row.get('audio_continuity_ok')} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="label=/path/to/run_root or variant root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in [parse_run(raw) for raw in args.run]:
        variants = find_variant_roots(spec.root)
        if not variants:
            missing.append(f"{spec.label}: no _probe/kv_lru_events.jsonl under {spec.root}")
            continue
        for variant_root in variants:
            results.append(analyze_variant(spec.label, variant_root))

    group_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    for result in results:
        group_rows.extend(flatten_group(result, group) for group in result["groups"])
        top_rows.extend(
            flatten_top_request(result, row)
            for row in result["top_requests"][: max(1, args.top_n)]
        )

    group_fields = [
        "run",
        "variant",
        "group",
        "n",
        "qps_est",
        "audio_ttfp_mean",
        "audio_ttfp_p50",
        "audio_ttfp_p90",
        "ttft_p90",
        "e2e_mean",
        "e2e_p90",
        "continuity_ok_rate",
        "continuity_n",
        "lookup_total",
        "lookup_total_mean",
        "lookup_hit_mean",
        "lookup_miss_mean",
        "recent_miss_mean",
        "lookup_hit_ratio",
        "lookup_miss_ratio",
        "recent_miss_ratio",
        "hit_depth_ratio_p50",
        "hit_depth_ratio_p90",
        "recent_request_rate",
        "lookup_hits",
        "lookup_misses",
        "recent_eviction_misses",
    ]
    top_fields = [
        "run",
        "variant",
        "request_id",
        "user_id",
        "round_index",
        "timestamp",
        "audio_ttfp_sec",
        "ttft_sec",
        "e2e_sec",
        "audio_continuity_ok",
        "kv_prefix_token_estimate",
        "query_length",
        "response_length",
        "audio_duration_sec",
        "lookup_events",
        "lookup_total",
        "lookup_hits",
        "lookup_misses",
        "recent_eviction_misses",
        "touch_events",
        "touch_blocks",
        "rescued_blocks",
    ]
    write_csv(output_dir / "kv_lru_event_trace_groups.csv", group_rows, group_fields)
    write_csv(output_dir / "kv_lru_event_trace_top_requests.csv", top_rows, top_fields)

    compact_results = []
    for result in results:
        copied = dict(result)
        copied["top_requests"] = [
            flatten_top_request(result, row)
            for row in result["top_requests"][: max(1, args.top_n)]
        ]
        compact_results.append(copied)
    with (output_dir / "kv_lru_event_trace_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"results": compact_results, "missing": missing}, f, indent=2, sort_keys=True)

    report = markdown_report(results, max(1, args.top_n))
    if missing:
        report += "\n## Missing Inputs\n\n" + "\n".join(f"- {item}" for item in missing) + "\n"
    (output_dir / "kv_lru_event_trace_analysis.md").write_text(report, encoding="utf-8")
    print(output_dir / "kv_lru_event_trace_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
