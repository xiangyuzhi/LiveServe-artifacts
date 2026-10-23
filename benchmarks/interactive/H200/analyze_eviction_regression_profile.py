#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Callable


Row = dict[str, Any]

LATENCY_DELTA_METRICS = (
    "e2e_sec_delta",
    "ttfp_sec_delta",
    "tokens_per_sec_delta",
)

SERVER_STAGE_SOURCE_METRICS = (
    "server_queue_wait_ms",
    "server_stage0_submit_to_first_output_ms",
    "server_stage0_submit_to_finished_ms",
    "server_stage1_submit_to_finished_ms",
    "server_stage2_submit_to_finished_ms",
    "server_stage0_to_stage1_process_inputs_ms",
    "server_stage1_to_stage2_process_inputs_ms",
)

SERVER_STAGE_DELTA_METRICS = tuple(f"{metric}_delta" for metric in SERVER_STAGE_SOURCE_METRICS)

KV_EVENT_STAT_FIELDS = (
    "lookup_events",
    "lookup_total",
    "lookup_hits",
    "lookup_misses",
    "recent_eviction_misses",
    "dedup_skipped_events",
    "lru_eviction_blocks",
    "touch_events",
    "touch_blocks",
    "rescued_blocks",
)

KV_EVENT_DELTA_FIELDS = (
    "lookup_total",
    "lookup_hits",
    "lookup_misses",
    "recent_eviction_misses",
    "lru_eviction_blocks",
    "touch_blocks",
    "rescued_blocks",
)


def parse_run(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(raw)
    return path.name, path


def find_per_request(root_or_path: Path) -> Path:
    if root_or_path.is_file():
        return root_or_path
    for pattern in (
        "eviction_only/*/per_request.jsonl",
        "lru_control/*/per_request.jsonl",
        "*/*/per_request.jsonl",
        "**/per_request.jsonl",
    ):
        matches = sorted(root_or_path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no per_request.jsonl under {root_or_path}")


def find_kv_event_path(root_or_path: Path) -> Path | None:
    if root_or_path.is_file():
        return root_or_path if root_or_path.name == "kv_lru_events.jsonl" else None
    for pattern in (
        "_probe/kv_lru_events.jsonl",
        "lru_control/_probe/kv_lru_events.jsonl",
        "eviction_only/_probe/kv_lru_events.jsonl",
        "*/_probe/kv_lru_events.jsonl",
        "**/kv_lru_events.jsonl",
    ):
        matches = sorted(root_or_path.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_kv_request_stats_path(root_or_path: Path) -> Path | None:
    if root_or_path.is_file():
        return root_or_path if root_or_path.name == "kv_lru_request_stats.jsonl" else None
    for pattern in (
        "_probe/kv_lru_request_stats.jsonl",
        "lru_control/_probe/kv_lru_request_stats.jsonl",
        "eviction_only/_probe/kv_lru_request_stats.jsonl",
        "*/_probe/kv_lru_request_stats.jsonl",
        "**/kv_lru_request_stats.jsonl",
    ):
        matches = sorted(root_or_path.glob(pattern))
        if matches:
            return matches[0]
    return None


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


def finite(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else default
    try:
        out = float(str(value))
    except Exception:
        return default
    return out if math.isfinite(out) else default


def numeric(row: Row, key: str, default: float | None = None) -> float | None:
    return finite(row.get(key), default)


def percentile(values: list[float], q: float) -> float | None:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return None
    idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[idx]


def describe(values: list[float]) -> dict[str, Any]:
    xs = [v for v in values if math.isfinite(v)]
    return {
        "n": len(xs),
        "mean": mean(xs) if xs else None,
        "p10": percentile(xs, 0.10),
        "p50": percentile(xs, 0.50),
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
        "min": min(xs) if xs else None,
        "max": max(xs) if xs else None,
    }


def trace_key(row: Row) -> tuple[Any, ...]:
    entry = row.get("trace_entry_index")
    if entry is not None:
        return ("trace_entry_index", int(entry))
    return (
        str(row.get("user_id")),
        int(row.get("round_index") or 0),
        int(row.get("timestamp") or 0),
        int(row.get("query_length") or 0),
        int(row.get("response_length") or 0),
    )


def event_trace_key(meta: Row) -> tuple[Any, ...] | None:
    try:
        return (
            str(meta["omni_session_id"]),
            int(meta["omni_round_index"]),
            int(float(meta["omni_trace_timestamp"])),
            int(meta.get("omni_query_length") or 0),
            int(meta.get("omni_response_length") or 0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def normalize_request_id(value: Any) -> str | None:
    if value is None:
        return None
    out = str(value)
    for prefix in ("chatcmpl-", "cmpl-"):
        if out.startswith(prefix):
            out = out[len(prefix):]
    return out or None


def read_requests(path: Path) -> dict[tuple[Any, ...], Row]:
    out: dict[tuple[Any, ...], Row] = {}
    for row in iter_jsonl(path):
        if not isinstance(row, dict):
            continue
        if row.get("error") is not None:
            continue
        if row.get("status") not in (None, 200):
            continue
        out[trace_key(row)] = row
    return out


def empty_event_stats() -> Row:
    return {field: 0 for field in KV_EVENT_STAT_FIELDS}


def add_event_to_stats(stats: Row, event: Row) -> None:
    kind = event.get("kind")
    if kind == "kv_lru_lookup":
        if event.get("dedup_skipped") is True:
            stats["dedup_skipped_events"] = int(stats.get("dedup_skipped_events") or 0) + 1
            return
        stats["lookup_events"] = int(stats.get("lookup_events") or 0) + 1
        stats["lookup_total"] = int(stats.get("lookup_total") or 0) + int(event.get("total") or 0)
        stats["lookup_hits"] = int(stats.get("lookup_hits") or 0) + int(event.get("hits") or 0)
        stats["lookup_misses"] = int(stats.get("lookup_misses") or 0) + int(event.get("misses") or 0)
        stats["recent_eviction_misses"] = int(stats.get("recent_eviction_misses") or 0) + int(
            event.get("recent_eviction_misses") or 0
        )
    elif kind == "kv_lru_touch":
        stats["touch_events"] = int(stats.get("touch_events") or 0) + 1
        stats["touch_blocks"] = int(stats.get("touch_blocks") or 0) + int(event.get("blocks") or 0)
        stats["rescued_blocks"] = int(stats.get("rescued_blocks") or 0) + int(event.get("rescued") or 0)
    elif kind == "kv_lru_eviction":
        stats["lru_eviction_blocks"] = int(stats.get("lru_eviction_blocks") or 0) + 1


def add_request_stats_snapshot(stats: Row, snapshot: Row) -> None:
    field_map = {
        "lookup_total": "gpu_prefix_block_lookup_total",
        "lookup_hits": "gpu_prefix_block_lookup_hit",
        "lookup_misses": "gpu_prefix_block_lookup_miss",
        "recent_eviction_misses": "gpu_prefix_block_lookup_recent_eviction_miss",
        "dedup_skipped_events": "gpu_prefix_block_lookup_dedup_skipped_total",
        "lru_eviction_blocks": "gpu_lru_eviction_blocks_total",
        "touch_blocks": "gpu_prefix_block_touch_total",
        "rescued_blocks": "gpu_prefix_block_rescued_from_free_queue_total",
    }
    for target, source in field_map.items():
        stats[target] = int(stats.get(target) or 0) + int(snapshot.get(source) or 0)


def event_trace_key_from_request(row: Row) -> tuple[Any, ...]:
    return (
        str(row.get("user_id")),
        int(row.get("round_index") or 0),
        int(row.get("timestamp") or 0),
        int(row.get("query_length") or 0),
        int(row.get("response_length") or 0),
    )


def read_event_stats_for_requests(
    event_path: Path | None,
    requests: dict[tuple[Any, ...], Row],
) -> tuple[dict[tuple[Any, ...], Row], Row]:
    meta: Row = {
        "event_path": str(event_path) if event_path else None,
        "event_source": "events",
        "event_rows": 0,
        "event_rows_with_request": 0,
        "event_rows_matched": 0,
        "event_rows_unmatched": 0,
    }
    if event_path is None or not event_path.exists():
        return {}, meta

    by_id: dict[str, tuple[Any, ...]] = {}
    by_trace: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for key, row in requests.items():
        request_id = normalize_request_id(row.get("request_id"))
        if request_id:
            by_id[request_id] = key
        by_trace[event_trace_key_from_request(row)] = key

    out: dict[tuple[Any, ...], Row] = {}
    for event in iter_jsonl(event_path):
        if not isinstance(event, dict):
            continue
        meta["event_rows"] += 1
        if event.get("kind") not in {"kv_lru_lookup", "kv_lru_touch", "kv_lru_eviction"}:
            continue
        request_meta = event.get("request")
        if not isinstance(request_meta, dict):
            meta["event_rows_unmatched"] += 1
            continue
        meta["event_rows_with_request"] += 1
        key = None
        request_id = normalize_request_id(request_meta.get("request_id"))
        if request_id:
            key = by_id.get(request_id)
        if key is None:
            trace = event_trace_key(request_meta)
            if trace is not None:
                key = by_trace.get(trace)
        if key is None:
            meta["event_rows_unmatched"] += 1
            continue
        stats = out.setdefault(key, empty_event_stats())
        add_event_to_stats(stats, event)
        meta["event_rows_matched"] += 1
    return out, meta


def read_request_stats_for_requests(
    stats_path: Path | None,
    requests: dict[tuple[Any, ...], Row],
) -> tuple[dict[tuple[Any, ...], Row], Row]:
    meta: Row = {
        "event_path": str(stats_path) if stats_path else None,
        "event_source": "request_stats",
        "event_rows": 0,
        "event_rows_with_request": 0,
        "event_rows_matched": 0,
        "event_rows_unmatched": 0,
        "event_latest_slices": 0,
    }
    if stats_path is None or not stats_path.exists():
        return {}, meta

    by_id: dict[str, tuple[Any, ...]] = {}
    for key, row in requests.items():
        request_id = normalize_request_id(row.get("request_id"))
        if request_id:
            by_id[request_id] = key

    latest: dict[tuple[tuple[Any, ...], str], Row] = {}
    latest_ts: dict[tuple[tuple[Any, ...], str], float] = {}
    for snapshot in iter_jsonl(stats_path):
        if not isinstance(snapshot, dict):
            continue
        meta["event_rows"] += 1
        if snapshot.get("kind") not in (None, "kv_lru_request_stats"):
            continue
        request_id = normalize_request_id(snapshot.get("request_id"))
        if not request_id:
            meta["event_rows_unmatched"] += 1
            continue
        meta["event_rows_with_request"] += 1
        key = by_id.get(request_id)
        if key is None:
            meta["event_rows_unmatched"] += 1
            continue
        proc = str(snapshot.get("pid") or "unknown")
        latest_key = (key, proc)
        ts = finite(snapshot.get("ts"), 0.0) or 0.0
        if ts >= latest_ts.get(latest_key, -1.0):
            latest_ts[latest_key] = ts
            latest[latest_key] = snapshot
        meta["event_rows_matched"] += 1

    out: dict[tuple[Any, ...], Row] = {}
    for (key, _proc), snapshot in latest.items():
        stats = out.setdefault(key, empty_event_stats())
        add_request_stats_snapshot(stats, snapshot)
        meta["event_latest_slices"] += 1
    return out, meta


def attach_event_stats(row: Row, prefix: str, stats: Row) -> None:
    for field in KV_EVENT_STAT_FIELDS:
        row[f"{prefix}_kv_{field}"] = int(stats.get(field) or 0)
    total = int(stats.get("lookup_total") or 0)
    hits = int(stats.get("lookup_hits") or 0)
    row[f"{prefix}_kv_lookup_hit_ratio"] = hits / total if total > 0 else None


def bucket_value(value: float, cuts: list[float], labels: list[str]) -> str:
    for cut, label in zip(cuts, labels):
        if value < cut:
            return label
    return labels[-1]


def phase_bucket(seconds: float) -> str:
    return bucket_value(
        seconds,
        [10, 20, 40, 80, 120, 180, 240],
        ["0_10", "10_20", "20_40", "40_80", "80_120", "120_180", "180_240", "240_plus"],
    )


def add_delta(row: Row, left: Row, right: Row, metric: str) -> None:
    lv = numeric(left, metric)
    rv = numeric(right, metric)
    if lv is None or rv is None:
        return
    row[f"left_{metric}"] = lv
    row[f"right_{metric}"] = rv
    row[f"{metric}_delta"] = rv - lv


def pair_rows(
    left_path: Path,
    right_path: Path,
    *,
    left_event_path: Path | None = None,
    right_event_path: Path | None = None,
    left_request_stats_path: Path | None = None,
    right_request_stats_path: Path | None = None,
) -> tuple[list[Row], dict[str, Any]]:
    left = read_requests(left_path)
    right = read_requests(right_path)
    if left_request_stats_path is not None and left_request_stats_path.exists():
        left_events, left_event_meta = read_request_stats_for_requests(
            left_request_stats_path,
            left,
        )
    else:
        left_events, left_event_meta = read_event_stats_for_requests(left_event_path, left)
    if right_request_stats_path is not None and right_request_stats_path.exists():
        right_events, right_event_meta = read_request_stats_for_requests(
            right_request_stats_path,
            right,
        )
    else:
        right_events, right_event_meta = read_event_stats_for_requests(right_event_path, right)
    left_event_enabled = bool(
        left_events
        or (left_event_path is not None and left_event_path.exists())
        or (left_request_stats_path is not None and left_request_stats_path.exists())
    )
    right_event_enabled = bool(
        right_events
        or (right_event_path is not None and right_event_path.exists())
        or (right_request_stats_path is not None and right_request_stats_path.exists())
    )
    common = sorted(set(left) & set(right))
    left_issue0 = min((numeric(left[k], "issued_at") or 0.0 for k in common), default=0.0)
    right_issue0 = min((numeric(right[k], "issued_at") or 0.0 for k in common), default=0.0)
    trace0 = min(
        (numeric(left[k], "timestamp", numeric(right[k], "timestamp", 0.0)) or 0.0 for k in common),
        default=0.0,
    )

    rows: list[Row] = []
    for key in common:
        lrow = left[key]
        rrow = right[key]
        timestamp = numeric(lrow, "timestamp", numeric(rrow, "timestamp", trace0)) or trace0
        issued_left = numeric(lrow, "issued_at", left_issue0) or left_issue0
        issued_right = numeric(rrow, "issued_at", right_issue0) or right_issue0
        prefix = numeric(rrow, "kv_prefix_token_estimate", numeric(lrow, "kv_prefix_token_estimate", 0.0)) or 0.0
        query = numeric(rrow, "query_length", numeric(lrow, "query_length", 0.0)) or 0.0
        response = numeric(rrow, "response_length", numeric(lrow, "response_length", 0.0)) or 0.0
        input_tokens = (
            numeric(rrow, "input_tokens_total")
            or numeric(lrow, "input_tokens_total")
            or prefix + query
        )
        row: Row = {
            "key": repr(key),
            "trace_entry_index": rrow.get("trace_entry_index", lrow.get("trace_entry_index")),
            "user_id": rrow.get("user_id", lrow.get("user_id")),
            "round_index": numeric(rrow, "round_index", numeric(lrow, "round_index", 0.0)) or 0.0,
            "timestamp": timestamp,
            "trace_rel_sec": timestamp - trace0,
            "left_issue_rel_sec": issued_left - left_issue0,
            "right_issue_rel_sec": issued_right - right_issue0,
            "query_length": query,
            "response_length": response,
            "input_tokens_total": input_tokens,
            "prefix_tokens": prefix,
            "has_prefix": 1 if prefix > 0 else 0,
            "output_tokens": numeric(rrow, "output_tokens", numeric(lrow, "output_tokens")),
            "tokens_per_sec": numeric(rrow, "tokens_per_sec"),
            "left_request_id": lrow.get("request_id"),
            "right_request_id": rrow.get("request_id"),
        }
        add_delta(row, lrow, rrow, "output_tokens")
        for metric in (
            "e2e_sec",
            "ttfp_sec",
            "tokens_per_sec",
            *SERVER_STAGE_SOURCE_METRICS,
        ):
            add_delta(row, lrow, rrow, metric)
        if left_event_enabled:
            attach_event_stats(row, "left", left_events.get(key, empty_event_stats()))
        if right_event_enabled:
            attach_event_stats(row, "right", right_events.get(key, empty_event_stats()))
        if left_event_enabled and right_event_enabled:
            for field in KV_EVENT_DELTA_FIELDS:
                row[f"kv_{field}_delta"] = (
                    int(row.get(f"right_kv_{field}") or 0)
                    - int(row.get(f"left_kv_{field}") or 0)
                )
            left_hit_ratio = numeric(row, "left_kv_lookup_hit_ratio")
            right_hit_ratio = numeric(row, "right_kv_lookup_hit_ratio")
            if left_hit_ratio is not None and right_hit_ratio is not None:
                row["kv_lookup_hit_ratio_delta"] = right_hit_ratio - left_hit_ratio
        rows.append(row)
    meta = {
        "left_requests": len(left),
        "right_requests": len(right),
        "common_requests": len(common),
        "left_issue0": left_issue0,
        "right_issue0": right_issue0,
        "trace0": trace0,
        "left_event_meta": left_event_meta,
        "right_event_meta": right_event_meta,
    }
    return rows, meta


def groups_for(row: Row) -> dict[str, str]:
    return {
        "all": "all",
        "response_bucket": bucket_value(
            float(row["response_length"]),
            [16, 32, 64, 128],
            ["r_lt16", "r16_31", "r32_63", "r64_127", "r128_plus"],
        ),
        "query_bucket": bucket_value(
            float(row["query_length"]),
            [16, 32, 64, 128],
            ["q_lt16", "q16_31", "q32_63", "q64_127", "q128_plus"],
        ),
        "prefix_bucket": bucket_value(
            float(row["prefix_tokens"]),
            [1, 64, 128, 256, 512, 768, 1024],
            ["p0", "p1_63", "p64_127", "p128_255", "p256_511", "p512_767", "p768_1023", "p1024_plus"],
        ),
        "round_bucket": bucket_value(
            float(row["round_index"]),
            [1, 3, 10, 30],
            ["round0", "round1_2", "round3_9", "round10_29", "round30_plus"],
        ),
        "trace_phase": phase_bucket(float(row["trace_rel_sec"])),
        "right_issue_phase": phase_bucket(float(row["right_issue_rel_sec"])),
    }


def summarize_subset(rows: list[Row], metric: str) -> dict[str, Any]:
    values = [float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float))]
    summary = describe(values)
    if values:
        summary.update(
            {
                "improved_rate": sum(v < 0.0 for v in values) / len(values),
                "regress_gt1_rate": sum(v > 1.0 for v in values) / len(values),
                "regress_gt3_rate": sum(v > 3.0 for v in values) / len(values),
                "improve_gt1_rate": sum(v < -1.0 for v in values) / len(values),
                "improve_gt3_rate": sum(v < -3.0 for v in values) / len(values),
            }
        )
    return summary


def bucket_profiles(rows: list[Row]) -> list[Row]:
    buckets: dict[tuple[str, str], list[Row]] = {}
    for row in rows:
        for group, value in groups_for(row).items():
            buckets.setdefault((group, value), []).append(row)
    out: list[Row] = []
    for (group, value), subset in sorted(buckets.items()):
        e2e = summarize_subset(subset, "e2e_sec_delta")
        ttfp = summarize_subset(subset, "ttfp_sec_delta")
        out.append(
            {
                "group": group,
                "value": value,
                "n": len(subset),
                "e2e_mean": e2e.get("mean"),
                "e2e_p50": e2e.get("p50"),
                "e2e_p90": e2e.get("p90"),
                "e2e_p95": e2e.get("p95"),
                "e2e_improved_rate": e2e.get("improved_rate"),
                "e2e_regress_gt1_rate": e2e.get("regress_gt1_rate"),
                "e2e_regress_gt3_rate": e2e.get("regress_gt3_rate"),
                "ttfp_mean": ttfp.get("mean"),
                "ttfp_p90": ttfp.get("p90"),
            }
        )
    return out


def feature_profiles(rows: list[Row]) -> list[Row]:
    specs: list[tuple[str, Callable[[Row], bool]]] = [
        ("all", lambda r: True),
        ("e2e_regress_gt1", lambda r: float(r.get("e2e_sec_delta", 0.0)) > 1.0),
        ("e2e_regress_gt3", lambda r: float(r.get("e2e_sec_delta", 0.0)) > 3.0),
        ("e2e_improve_gt1", lambda r: float(r.get("e2e_sec_delta", 0.0)) < -1.0),
        ("e2e_improve_gt3", lambda r: float(r.get("e2e_sec_delta", 0.0)) < -3.0),
    ]
    features = [
        "query_length",
        "response_length",
        "input_tokens_total",
        "prefix_tokens",
        "round_index",
        "timestamp",
        "trace_rel_sec",
        "right_issue_rel_sec",
        "e2e_sec_delta",
        "ttfp_sec_delta",
        "tokens_per_sec_delta",
        "output_tokens_delta",
        "server_queue_wait_ms_delta",
        "server_stage0_submit_to_first_output_ms_delta",
        "server_stage0_submit_to_finished_ms_delta",
        "right_kv_lookup_total",
        "right_kv_lookup_misses",
        "right_kv_recent_eviction_misses",
        "right_kv_lru_eviction_blocks",
        "right_kv_lookup_hit_ratio",
        "kv_lookup_misses_delta",
        "kv_recent_eviction_misses_delta",
        "kv_lru_eviction_blocks_delta",
    ]
    out: list[Row] = []
    total = max(1, len(rows))
    for label, pred in specs:
        subset = [row for row in rows if pred(row)]
        profile: Row = {"subset": label, "n": len(subset), "rate": len(subset) / total}
        for feature in features:
            values = [float(row[feature]) for row in subset if isinstance(row.get(feature), (int, float))]
            desc = describe(values)
            for key in ("mean", "p50", "p90", "p95"):
                profile[f"{feature}_{key}"] = desc.get(key)
        out.append(profile)
    return out


def rolling_windows(rows: list[Row], field: str, window_sec: float, min_count: int) -> list[Row]:
    if window_sec <= 0:
        return []
    ordered = sorted(
        [row for row in rows if isinstance(row.get(field), (int, float))],
        key=lambda row: float(row[field]),
    )
    out: list[Row] = []
    start = 0
    for end, row in enumerate(ordered):
        current = float(row[field])
        while start <= end and current - float(ordered[start][field]) >= window_sec:
            start += 1
        subset = ordered[start : end + 1]
        if len(subset) < min_count:
            continue
        e2e = summarize_subset(subset, "e2e_sec_delta")
        ttfp = summarize_subset(subset, "ttfp_sec_delta")
        queue = summarize_subset(subset, "server_queue_wait_ms_delta")
        stage0_first = summarize_subset(subset, "server_stage0_submit_to_first_output_ms_delta")
        stage0_finish = summarize_subset(subset, "server_stage0_submit_to_finished_ms_delta")
        out.append(
            {
                "field": field,
                "window_start": float(subset[0][field]),
                "window_end": current,
                "n": len(subset),
                "e2e_mean": e2e.get("mean"),
                "e2e_p90": e2e.get("p90"),
                "e2e_regress_gt1_rate": e2e.get("regress_gt1_rate"),
                "e2e_regress_gt3_rate": e2e.get("regress_gt3_rate"),
                "e2e_improve_gt1_rate": e2e.get("improve_gt1_rate"),
                "ttfp_mean": ttfp.get("mean"),
                "ttfp_p90": ttfp.get("p90"),
                "queue_p90": queue.get("p90"),
                "stage0_first_p90": stage0_first.get("p90"),
                "stage0_finish_p90": stage0_finish.get("p90"),
            }
        )
    # Deduplicate very similar suffix windows by start/end bucket.
    dedup: dict[tuple[str, int, int], Row] = {}
    for item in out:
        key = (
            str(item["field"]),
            int(float(item["window_start"]) // max(1.0, window_sec / 2.0)),
            int(float(item["window_end"]) // max(1.0, window_sec / 2.0)),
        )
        previous = dedup.get(key)
        if previous is None or float(item.get("e2e_mean") or 0.0) > float(previous.get("e2e_mean") or 0.0):
            dedup[key] = item
    return list(dedup.values())


def kv_event_overall(rows: list[Row]) -> Row:
    out: Row = {}
    for side in ("left", "right"):
        for field in KV_EVENT_STAT_FIELDS:
            out[f"{side}_{field}"] = sum(int(row.get(f"{side}_kv_{field}") or 0) for row in rows)
        total = int(out.get(f"{side}_lookup_total") or 0)
        hits = int(out.get(f"{side}_lookup_hits") or 0)
        misses = int(out.get(f"{side}_lookup_misses") or 0)
        recent = int(out.get(f"{side}_recent_eviction_misses") or 0)
        out[f"{side}_lookup_hit_ratio"] = hits / total if total > 0 else None
        out[f"{side}_lookup_miss_ratio"] = misses / total if total > 0 else None
        out[f"{side}_recent_eviction_miss_ratio"] = recent / total if total > 0 else None
        out[f"{side}_requests_with_lookup_miss"] = sum(
            int(row.get(f"{side}_kv_lookup_misses") or 0) > 0 for row in rows
        )
        out[f"{side}_requests_with_recent_eviction_miss"] = sum(
            int(row.get(f"{side}_kv_recent_eviction_misses") or 0) > 0 for row in rows
        )
    for field in KV_EVENT_DELTA_FIELDS:
        out[f"{field}_delta"] = int(out.get(f"right_{field}") or 0) - int(out.get(f"left_{field}") or 0)
    return out


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(
    result: dict[str, Any],
    worst: list[Row],
    best: list[Row],
    bucket_rows: list[Row],
    feature_rows: list[Row],
    window_rows: list[Row],
) -> str:
    overall = result["overall"]
    lines = [
        "# Eviction Regression Profile",
        "",
        f"- left: `{result['left_label']}`",
        f"- right: `{result['right_label']}`",
        "- deltas are `right - left`; positive means right is slower.",
        f"- common requests: `{result['common_requests']}`",
        "",
        "## Overall",
        "",
        "| metric | mean | p50 | p90 | p95 | p99 | improved | regress>1s | regress>3s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in ("e2e_sec_delta", "ttfp_sec_delta", "tokens_per_sec_delta"):
        item = overall.get(metric, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    metric,
                    fmt(item.get("mean")),
                    fmt(item.get("p50")),
                    fmt(item.get("p90")),
                    fmt(item.get("p95")),
                    fmt(item.get("p99")),
                    fmt(item.get("improved_rate")),
                    fmt(item.get("regress_gt1_rate")),
                    fmt(item.get("regress_gt3_rate")),
                ]
            )
            + " |"
        )

    available_stage_metrics = [
        metric
        for metric in SERVER_STAGE_DELTA_METRICS
        if overall.get(metric, {}).get("n")
    ]
    if available_stage_metrics:
        lines += [
            "",
            "## Server Stage Delta Overall",
            "",
            "| metric | mean ms | p50 ms | p90 ms | p95 ms | p99 ms | improved | regress>1000ms | regress>3000ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for metric in available_stage_metrics:
            item = overall.get(metric, {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        metric,
                        fmt(item.get("mean")),
                        fmt(item.get("p50")),
                        fmt(item.get("p90")),
                        fmt(item.get("p95")),
                        fmt(item.get("p99")),
                        fmt(item.get("improved_rate")),
                        fmt(item.get("regress_gt1000ms_rate")),
                        fmt(item.get("regress_gt3000ms_rate")),
                    ]
                )
                + " |"
            )

    kv_event = result.get("kv_event_overall")
    if isinstance(kv_event, dict) and (
        int(kv_event.get("left_lookup_total") or 0) > 0
        or int(kv_event.get("right_lookup_total") or 0) > 0
    ):
        lines += [
            "",
            "## KV Overall",
            "",
            f"- left source: `{(result.get('left_event_meta') or {}).get('event_source', 'events')}` "
            f"`{(result.get('left_event_meta') or {}).get('event_path')}`",
            f"- right source: `{(result.get('right_event_meta') or {}).get('event_source', 'events')}` "
            f"`{(result.get('right_event_meta') or {}).get('event_path')}`",
            "",
            "| side | lookup total | hits | misses | recent misses | evictions | hit ratio | requests with miss | requests with recent miss | touch blocks | rescued blocks |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for side in ("left", "right"):
            lines.append(
                "| "
                + " | ".join(
                    [
                        side,
                        fmt(kv_event.get(f"{side}_lookup_total")),
                        fmt(kv_event.get(f"{side}_lookup_hits")),
                        fmt(kv_event.get(f"{side}_lookup_misses")),
                        fmt(kv_event.get(f"{side}_recent_eviction_misses")),
                        fmt(kv_event.get(f"{side}_lru_eviction_blocks")),
                        fmt(kv_event.get(f"{side}_lookup_hit_ratio")),
                        fmt(kv_event.get(f"{side}_requests_with_lookup_miss")),
                        fmt(kv_event.get(f"{side}_requests_with_recent_eviction_miss")),
                        fmt(kv_event.get(f"{side}_touch_blocks")),
                        fmt(kv_event.get(f"{side}_rescued_blocks")),
                    ]
                )
                + " |"
            )
        lines += [
            "",
            "| delta right-left | lookup total | hits | misses | recent misses | evictions | touch blocks | rescued blocks |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            "| delta | "
            + " | ".join(
                [
                    fmt(kv_event.get("lookup_total_delta")),
                    fmt(kv_event.get("lookup_hits_delta")),
                    fmt(kv_event.get("lookup_misses_delta")),
                    fmt(kv_event.get("recent_eviction_misses_delta")),
                    fmt(kv_event.get("lru_eviction_blocks_delta")),
                    fmt(kv_event.get("touch_blocks_delta")),
                    fmt(kv_event.get("rescued_blocks_delta")),
                ]
            )
            + " |",
        ]

    lines += [
        "",
        "## Feature Profile",
        "",
        "| subset | n | rate | response p50 | response p90 | prefix p50 | prefix p90 | ttfp delta p50 | ttfp delta p90 | queue delta p90 | stage0 first delta p90 | stage0 finish delta p90 | right kv miss p90 | right recent miss p90 | kv miss delta p90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in feature_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["subset"]),
                    str(row["n"]),
                    fmt(row["rate"]),
                    fmt(row.get("response_length_p50")),
                    fmt(row.get("response_length_p90")),
                    fmt(row.get("prefix_tokens_p50")),
                    fmt(row.get("prefix_tokens_p90")),
                    fmt(row.get("ttfp_sec_delta_p50")),
                    fmt(row.get("ttfp_sec_delta_p90")),
                    fmt(row.get("server_queue_wait_ms_delta_p90")),
                    fmt(row.get("server_stage0_submit_to_first_output_ms_delta_p90")),
                    fmt(row.get("server_stage0_submit_to_finished_ms_delta_p90")),
                    fmt(row.get("right_kv_lookup_misses_p90")),
                    fmt(row.get("right_kv_recent_eviction_misses_p90")),
                    fmt(row.get("kv_lookup_misses_delta_p90")),
                ]
            )
            + " |"
        )

    top_groups = sorted(
        [row for row in bucket_rows if row["group"] != "all" and int(row["n"]) >= 50],
        key=lambda row: float(row.get("e2e_mean") or -1e9),
        reverse=True,
    )[:16]
    lines += [
        "",
        "## Worst Buckets For Right",
        "",
        "| group | value | n | e2e mean | e2e p90 | regress>1s | regress>3s | ttfp p90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_groups:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group"]),
                    str(row["value"]),
                    str(row["n"]),
                    fmt(row.get("e2e_mean")),
                    fmt(row.get("e2e_p90")),
                    fmt(row.get("e2e_regress_gt1_rate")),
                    fmt(row.get("e2e_regress_gt3_rate")),
                    fmt(row.get("ttfp_p90")),
                ]
            )
            + " |"
        )

    top_windows = sorted(
        window_rows,
        key=lambda row: (float(row.get("e2e_mean") or -1e9), float(row.get("e2e_regress_gt1_rate") or 0.0)),
        reverse=True,
    )[:12]
    lines += [
        "",
        "## Worst Rolling Windows",
        "",
        "| field | start | end | n | e2e mean | e2e p90 | regress>1s | regress>3s | ttfp p90 | queue p90 ms | stage0 first p90 ms | stage0 finish p90 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_windows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["field"]),
                    fmt(row.get("window_start")),
                    fmt(row.get("window_end")),
                    str(row["n"]),
                    fmt(row.get("e2e_mean")),
                    fmt(row.get("e2e_p90")),
                    fmt(row.get("e2e_regress_gt1_rate")),
                    fmt(row.get("e2e_regress_gt3_rate")),
                    fmt(row.get("ttfp_p90")),
                    fmt(row.get("queue_p90")),
                    fmt(row.get("stage0_first_p90")),
                    fmt(row.get("stage0_finish_p90")),
                ]
            )
            + " |"
        )

    def request_table(title: str, rows: list[Row]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| idx | user | e2e delta | ttfp delta | output delta | response | prefix | round | ts | left e2e | right e2e | right kv miss | right recent miss |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("trace_entry_index")),
                        str(row.get("user_id")),
                        fmt(row.get("e2e_sec_delta")),
                        fmt(row.get("ttfp_sec_delta")),
                        fmt(row.get("output_tokens_delta")),
                        fmt(row.get("response_length")),
                        fmt(row.get("prefix_tokens")),
                        fmt(row.get("round_index")),
                        fmt(row.get("timestamp")),
                        fmt(row.get("left_e2e_sec")),
                        fmt(row.get("right_e2e_sec")),
                        fmt(row.get("right_kv_lookup_misses")),
                        fmt(row.get("right_kv_recent_eviction_misses")),
                    ]
                )
                + " |"
            )

    request_table("Worst Requests For Right", worst)
    request_table("Best Requests For Right", best)
    lines.append("")
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    left_label, left_raw = parse_run(args.left)
    right_label, right_raw = parse_run(args.right)
    left_path = find_per_request(left_raw)
    right_path = find_per_request(right_raw)
    left_event_path = args.left_events or find_kv_event_path(left_raw)
    right_event_path = args.right_events or find_kv_event_path(right_raw)
    left_request_stats_path = args.left_request_stats or find_kv_request_stats_path(left_raw)
    right_request_stats_path = args.right_request_stats or find_kv_request_stats_path(right_raw)
    rows, meta = pair_rows(
        left_path,
        right_path,
        left_event_path=left_event_path,
        right_event_path=right_event_path,
        left_request_stats_path=left_request_stats_path,
        right_request_stats_path=right_request_stats_path,
    )
    for row in rows:
        if "tokens_per_sec_delta" in row:
            # Higher tokens/sec is better, so invert for "right slower" summaries.
            row["tokens_per_sec_delta"] = -float(row["tokens_per_sec_delta"])

    overall = {
        metric: summarize_subset(rows, metric)
        for metric in (*LATENCY_DELTA_METRICS, *SERVER_STAGE_DELTA_METRICS)
    }
    for metric in SERVER_STAGE_DELTA_METRICS:
        values = [
            float(row[metric])
            for row in rows
            if isinstance(row.get(metric), (int, float)) and math.isfinite(float(row[metric]))
        ]
        if not values:
            continue
        overall[metric]["regress_gt1000ms_rate"] = sum(v > 1000.0 for v in values) / len(values)
        overall[metric]["regress_gt3000ms_rate"] = sum(v > 3000.0 for v in values) / len(values)
    bucket_rows = bucket_profiles(rows)
    feature_rows = feature_profiles(rows)
    window_rows = rolling_windows(rows, "trace_rel_sec", args.window_sec, args.min_window_count)
    window_rows += rolling_windows(rows, "right_issue_rel_sec", args.window_sec, args.min_window_count)
    worst = sorted(rows, key=lambda row: float(row.get("e2e_sec_delta") or -1e9), reverse=True)[: args.top_n]
    best = sorted(rows, key=lambda row: float(row.get("e2e_sec_delta") or 1e9))[: args.top_n]
    return {
        "left_label": left_label,
        "right_label": right_label,
        "left_path": str(left_path),
        "right_path": str(right_path),
        "left_event_path": str(left_event_path) if left_event_path else None,
        "right_event_path": str(right_event_path) if right_event_path else None,
        **meta,
        "overall": overall,
        "kv_event_overall": kv_event_overall(rows),
        "rows": rows,
        "bucket_rows": bucket_rows,
        "feature_rows": feature_rows,
        "window_rows": window_rows,
        "worst_rows": worst,
        "best_rows": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile paired eviction latency regressions.")
    parser.add_argument("--left", required=True, help="label=path or path for baseline/left run")
    parser.add_argument("--right", required=True, help="label=path or path for candidate/right run")
    parser.add_argument("--left-events", type=Path, help="optional left kv_lru_events.jsonl")
    parser.add_argument("--right-events", type=Path, help="optional right kv_lru_events.jsonl")
    parser.add_argument("--left-request-stats", type=Path, help="optional left kv_lru_request_stats.jsonl")
    parser.add_argument("--right-request-stats", type=Path, help="optional right kv_lru_request_stats.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--min-window-count", type=int, default=50)
    args = parser.parse_args()

    result = analyze(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paired_requests.csv", result["rows"])
    write_csv(output_dir / "bucket_profile.csv", result["bucket_rows"])
    write_csv(output_dir / "feature_profile.csv", result["feature_rows"])
    write_csv(output_dir / "rolling_windows.csv", result["window_rows"])
    write_csv(output_dir / "worst_requests.csv", result["worst_rows"])
    write_csv(output_dir / "best_requests.csv", result["best_rows"])

    json_result = {
        key: value
        for key, value in result.items()
        if key not in {"rows", "bucket_rows", "feature_rows", "window_rows", "worst_rows", "best_rows"}
    }
    (output_dir / "regression_profile.json").write_text(
        json.dumps(json_result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown = render_markdown(
        json_result,
        result["worst_rows"],
        result["best_rows"],
        result["bucket_rows"],
        result["feature_rows"],
        result["window_rows"],
    )
    (output_dir / "regression_profile.md").write_text(markdown, encoding="utf-8")
    print(output_dir / "regression_profile.md")


if __name__ == "__main__":
    main()
