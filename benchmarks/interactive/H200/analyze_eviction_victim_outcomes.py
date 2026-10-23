#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
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
    for pattern in (
        f"*/_probe/{name}",
        f"**/{name}",
    ):
        matches = sorted(root_or_path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no {name} under {root_or_path}")


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, str) and value.strip():
        try:
            out = float(value)
        except ValueError:
            return None
        return out if math.isfinite(out) else None
    return None


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[idx]


def decode_bytes_repr(raw: Any) -> bytes | None:
    if not isinstance(raw, str) or not raw.startswith("b"):
        return None
    try:
        value = ast.literal_eval(raw)
    except Exception:
        return None
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    return None


def block_hash_lookup_keys(raw: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(raw, str) and raw:
        keys.append(raw)
    value = decode_bytes_repr(raw)
    if value is not None:
        keys.append(repr(value))
        if len(value) > 4:
            # vLLM eviction events store group-id-suffixed hashes, while lookup
            # events store the base hash plus group_ids separately.
            keys.append(repr(value[:-4]))
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def numeric_bucket(value: float | None, cuts: list[float], prefix: str) -> str:
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


def read_applied_enforcements(shadow_path: Path) -> list[dict[str, Any]]:
    direct: list[dict[str, Any]] = []
    shadow_applied: list[dict[str, Any]] = []
    for row in iter_jsonl(shadow_path):
        event = row.get("event")
        if event == "allocation_pressure_enforce":
            ids = row.get("selected_block_ids")
            if (row.get("applied") is True or row.get("reason") == "applied") and isinstance(ids, list):
                direct.append(row)
        elif event == "allocation_pressure_shadow":
            ids = row.get("enforced_block_ids")
            if row.get("enforce_applied") is True and isinstance(ids, list):
                row["ts"] = row.get("ts") or row.get("wall_time")
                row["policy"] = row.get("enforced_policy") or row.get("enforce_policy_requested")
                row["reason"] = "applied"
                row["selected_block_ids"] = ids
                row["selected_victims"] = row.get("enforced_victims")
                shadow_applied.append(row)
    return direct if direct else shadow_applied


def victim_records_by_id(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    sources: list[Any] = [
        row.get("enforced_victims"),
        row.get("selected_victims"),
    ]
    detail = row.get("selected_victim_records")
    if isinstance(detail, dict):
        sources.append(detail.get("records"))
    out: dict[int, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            block_id = as_int(item.get("block_id"))
            if block_id is not None:
                out.setdefault(block_id, item)
    return out


def score_values(row: dict[str, Any]) -> list[Any]:
    raw = row.get("selected_score_tuples")
    if isinstance(raw, dict) and isinstance(raw.get("values"), list):
        return raw["values"]
    return []


def read_lru_events(events_path: Path):
    evictions_by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    lookups_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for row in iter_jsonl(events_path):
        kind = row.get("kind")
        counts[str(kind)] += 1
        ts = as_float(row.get("ts"))
        if ts is None:
            continue
        row["ts"] = ts
        if kind == "kv_lru_eviction":
            block_id = as_int(row.get("block_id"))
            if block_id is None:
                continue
            row["_lookup_keys"] = block_hash_lookup_keys(row.get("block_hash"))
            evictions_by_block[block_id].append(row)
        elif kind == "kv_lru_lookup":
            for key in block_hash_lookup_keys(row.get("block_hash")):
                lookups_by_hash[key].append(row)
    for rows in evictions_by_block.values():
        rows.sort(key=lambda item: float(item["ts"]))
    for rows in lookups_by_hash.values():
        rows.sort(key=lambda item: float(item["ts"]))
    return evictions_by_block, lookups_by_hash, counts


def find_matching_eviction(
    rows: list[dict[str, Any]],
    enforce_ts: float,
    pre_window_sec: float,
    post_window_sec: float,
) -> dict[str, Any] | None:
    if not rows:
        return None
    ts_values = [float(row["ts"]) for row in rows]
    lo = bisect_left(ts_values, enforce_ts - pre_window_sec)
    hi = bisect_right(ts_values, enforce_ts + post_window_sec)
    candidates = rows[lo:hi]
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs(float(row["ts"]) - enforce_ts))


def future_lookups(
    lookup_rows: dict[str, list[dict[str, Any]]],
    keys: list[str],
    evict_ts: float,
    lookup_window_sec: float,
) -> list[dict[str, Any]]:
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for key in keys:
        rows = lookup_rows.get(key) or []
        ts_values = [float(row["ts"]) for row in rows]
        start = bisect_right(ts_values, evict_ts)
        end = bisect_right(ts_values, evict_ts + lookup_window_sec)
        for row in rows[start:end]:
            seq = int(row.get("_seq", -1))
            if seq in seen:
                continue
            seen.add(seq)
            out.append(row)
    out.sort(key=lambda row: float(row["ts"]))
    return out


def sum_field(rows: list[dict[str, Any]], name: str) -> int:
    total = 0
    for row in rows:
        value = as_int(row.get(name))
        if value is not None:
            total += value
    return total


def first_delta(rows: list[dict[str, Any]], base_ts: float, pred) -> float | None:
    for row in rows:
        if pred(row):
            return float(row["ts"]) - base_ts
    return None


def event_request_field(row: dict[str, Any], key: str) -> Any:
    current = row.get("current_request")
    if isinstance(current, dict):
        return current.get(key)
    return None


def record_field(record: dict[str, Any] | None, key: str) -> Any:
    if isinstance(record, dict):
        return record.get(key)
    return None


def build_victim_rows(
    enforcements: list[dict[str, Any]],
    evictions_by_block: dict[int, list[dict[str, Any]]],
    lookups_by_hash: dict[str, list[dict[str, Any]]],
    *,
    pre_window_sec: float,
    post_window_sec: float,
    lookup_window_sec: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event_idx, event in enumerate(enforcements):
        enforce_ts = as_float(event.get("ts"))
        if enforce_ts is None:
            continue
        ids = event.get("selected_block_ids")
        if not isinstance(ids, list):
            continue
        records = victim_records_by_id(event)
        scores = score_values(event)
        current = event.get("current_request") if isinstance(event.get("current_request"), dict) else {}
        for victim_idx, raw_block_id in enumerate(ids):
            block_id = as_int(raw_block_id)
            if block_id is None:
                continue
            record = records.get(block_id)
            matched = find_matching_eviction(
                evictions_by_block.get(block_id, []),
                enforce_ts,
                pre_window_sec,
                post_window_sec,
            )
            lookup_rows: list[dict[str, Any]] = []
            evict_ts = None
            hash_keys: list[str] = []
            if matched is not None:
                evict_ts = float(matched["ts"])
                hash_keys = matched.get("_lookup_keys") or block_hash_lookup_keys(
                    matched.get("block_hash")
                )
                lookup_rows = future_lookups(
                    lookups_by_hash,
                    hash_keys,
                    evict_ts,
                    lookup_window_sec,
                )
            miss_events = [
                row for row in lookup_rows if (as_int(row.get("misses")) or 0) > 0
            ]
            recent_events = [
                row
                for row in lookup_rows
                if (as_int(row.get("recent_eviction_misses")) or 0) > 0
            ]
            lookup_misses = sum_field(lookup_rows, "misses")
            recent_misses = sum_field(lookup_rows, "recent_eviction_misses")
            within = {}
            for window in (5.0, 30.0, 120.0):
                subset = [
                    row
                    for row in lookup_rows
                    if evict_ts is not None and float(row["ts"]) - evict_ts <= window
                ]
                label = str(int(window))
                within[f"lookup_events_{label}s"] = len(subset)
                within[f"lookup_misses_{label}s"] = sum_field(subset, "misses")
                within[f"recent_eviction_misses_{label}s"] = sum_field(
                    subset,
                    "recent_eviction_misses",
                )
            expected_gap = as_float(record_field(record, "omni_expected_next_gap_sec"))
            response_length = as_float(record_field(record, "omni_response_length"))
            history_tokens = as_float(record_field(record, "omni_history_tokens_after"))
            row = {
                "event_idx": event_idx,
                "event_seq": event.get("_seq"),
                "policy": event.get("policy"),
                "reason": event.get("reason"),
                "enforce_ts": enforce_ts,
                "victim_idx": victim_idx,
                "block_id": block_id,
                "eviction_matched": matched is not None,
                "eviction_ts": evict_ts,
                "eviction_delta_to_enforce_sec": (
                    evict_ts - enforce_ts if evict_ts is not None else None
                ),
                "eviction_block_hash": matched.get("block_hash") if matched else None,
                "lookup_key_count": len(hash_keys),
                "future_lookup_events": len(lookup_rows),
                "future_lookup_total": sum_field(lookup_rows, "total"),
                "future_lookup_hits": sum_field(lookup_rows, "hits"),
                "future_lookup_misses": lookup_misses,
                "future_miss_events": len(miss_events),
                "future_recent_eviction_misses": recent_misses,
                "future_recent_eviction_miss_events": len(recent_events),
                "first_lookup_delta_sec": first_delta(
                    lookup_rows,
                    evict_ts or enforce_ts,
                    lambda _: True,
                ),
                "first_miss_delta_sec": first_delta(
                    lookup_rows,
                    evict_ts or enforce_ts,
                    lambda item: (as_int(item.get("misses")) or 0) > 0,
                ),
                "first_recent_eviction_miss_delta_sec": first_delta(
                    lookup_rows,
                    evict_ts or enforce_ts,
                    lambda item: (as_int(item.get("recent_eviction_misses")) or 0) > 0,
                ),
                "selected_score_tuple": (
                    json.dumps(scores[victim_idx], sort_keys=True)
                    if victim_idx < len(scores)
                    else None
                ),
                "current_session_id": current.get("omni_session_id"),
                "current_round_index": current.get("omni_round_index"),
                "current_trace_timestamp": current.get("omni_trace_timestamp"),
                "current_query_length": current.get("omni_query_length"),
                "current_response_length": current.get("omni_response_length"),
                "current_expected_next_gap_sec": current.get("omni_expected_next_gap_sec"),
                "victim_session_id": record_field(record, "omni_session_id"),
                "victim_round_index": record_field(record, "omni_round_index"),
                "victim_trace_timestamp": record_field(record, "omni_trace_timestamp"),
                "victim_expected_next_gap_sec": expected_gap,
                "victim_expected_next_use_ts": record_field(record, "omni_expected_next_use_ts"),
                "victim_response_length": response_length,
                "victim_query_length": record_field(record, "omni_query_length"),
                "victim_history_tokens_after": history_tokens,
                "victim_prefix_token_estimate": record_field(record, "omni_prefix_token_estimate"),
                "victim_is_terminal_turn": record_field(record, "omni_is_terminal_turn"),
                "victim_depth_ratio": record_field(record, "block_depth_ratio"),
                "victim_lost_tokens_if_prefix_break": record_field(
                    record,
                    "estimated_lost_tokens_if_prefix_break",
                ),
                "victim_suffix_blocks_including": record_field(
                    record,
                    "suffix_blocks_including",
                ),
                "victim_freed_age_sec": record_field(record, "freed_age_sec"),
                "victim_metadata_present": bool(record),
                "gap_bucket": numeric_bucket(expected_gap, [15, 30, 60, 120, 300], "gap"),
                "response_bucket": numeric_bucket(response_length, [16, 32, 64, 128], "resp"),
                "history_bucket": numeric_bucket(history_tokens, [64, 128, 256, 512, 1024], "hist"),
                **within,
            }
            out.append(row)
    return out


def summarize_victims(rows: list[dict[str, Any]], event_counts: Counter[str]) -> dict[str, Any]:
    matched = [row for row in rows if row.get("eviction_matched")]
    future = [row for row in rows if (as_int(row.get("future_lookup_misses")) or 0) > 0]
    recent = [
        row
        for row in rows
        if (as_int(row.get("future_recent_eviction_misses")) or 0) > 0
    ]
    first_miss = [
        value
        for row in rows
        if (value := as_float(row.get("first_miss_delta_sec"))) is not None
    ]
    return {
        "selected_victims": len(rows),
        "matched_evictions": len(matched),
        "unmatched_evictions": len(rows) - len(matched),
        "victims_with_future_lookup_miss": len(future),
        "victims_with_recent_eviction_miss": len(recent),
        "future_lookup_misses": sum(as_int(row.get("future_lookup_misses")) or 0 for row in rows),
        "future_recent_eviction_misses": sum(
            as_int(row.get("future_recent_eviction_misses")) or 0 for row in rows
        ),
        "future_miss_victim_rate": len(future) / len(rows) if rows else None,
        "recent_miss_victim_rate": len(recent) / len(rows) if rows else None,
        "first_miss_delta_p50_sec": percentile(first_miss, 0.50),
        "first_miss_delta_p90_sec": percentile(first_miss, 0.90),
        "lru_event_counts": dict(event_counts),
    }


def group_victims(rows: list[dict[str, Any]], group_field: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    values = sorted({str(row.get(group_field)) for row in rows})
    for value in values:
        subset = [row for row in rows if str(row.get(group_field)) == value]
        first_miss = [
            delta
            for row in subset
            if (delta := as_float(row.get("first_miss_delta_sec"))) is not None
        ]
        future_miss_victims = sum(
            1 for row in subset if (as_int(row.get("future_lookup_misses")) or 0) > 0
        )
        recent_miss_victims = sum(
            1
            for row in subset
            if (as_int(row.get("future_recent_eviction_misses")) or 0) > 0
        )
        out.append(
            {
                "group": group_field,
                "value": value,
                "victims": len(subset),
                "matched_evictions": sum(1 for row in subset if row.get("eviction_matched")),
                "future_miss_victims": future_miss_victims,
                "future_miss_victim_rate": future_miss_victims / len(subset) if subset else None,
                "recent_miss_victims": recent_miss_victims,
                "recent_miss_victim_rate": recent_miss_victims / len(subset) if subset else None,
                "future_lookup_misses": sum(
                    as_int(row.get("future_lookup_misses")) or 0 for row in subset
                ),
                "recent_eviction_misses": sum(
                    as_int(row.get("future_recent_eviction_misses")) or 0
                    for row in subset
                ),
                "first_miss_delta_p50_sec": percentile(first_miss, 0.50),
                "first_miss_delta_p90_sec": percentile(first_miss, 0.90),
            }
        )
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    *,
    shadow_path: Path,
    events_path: Path,
    enforcements: list[dict[str, Any]],
    summary: dict[str, Any],
    groups: list[dict[str, Any]],
) -> None:
    lines = [
        "# Eviction Victim Outcomes",
        "",
        f"- shadow log: `{shadow_path}`",
        f"- lru events: `{events_path}`",
        f"- applied enforcement events: `{len(enforcements)}`",
        f"- selected victims: `{summary['selected_victims']}`",
        f"- matched evictions: `{summary['matched_evictions']}`",
        f"- victims with future lookup miss: `{summary['victims_with_future_lookup_miss']}` "
        f"({fmt(summary.get('future_miss_victim_rate'))})",
        f"- victims with recent-eviction miss: `{summary['victims_with_recent_eviction_miss']}` "
        f"({fmt(summary.get('recent_miss_victim_rate'))})",
        f"- future lookup misses attributed to selected victims: `{summary['future_lookup_misses']}`",
        f"- recent eviction misses attributed to selected victims: `{summary['future_recent_eviction_misses']}`",
        f"- first miss delta p50/p90 sec: `{fmt(summary.get('first_miss_delta_p50_sec'))}` / "
        f"`{fmt(summary.get('first_miss_delta_p90_sec'))}`",
        "",
        "## Top Groups",
        "",
        "| group | value | victims | matched | future miss victims | future miss rate | recent miss victims | recent miss rate | future misses | recent misses | first miss p50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(
        groups,
        key=lambda row: (
            as_float(row.get("future_miss_victim_rate")) or 0.0,
            as_int(row.get("victims")) or 0,
        ),
        reverse=True,
    )
    for row in ranked[:40]:
        lines.append(
            f"| {row['group']} | {row['value']} | {row['victims']} | "
            f"{row['matched_evictions']} | {row['future_miss_victims']} | "
            f"{fmt(row.get('future_miss_victim_rate'))} | {row['recent_miss_victims']} | "
            f"{fmt(row.get('recent_miss_victim_rate'))} | {row['future_lookup_misses']} | "
            f"{row['recent_eviction_misses']} | {fmt(row.get('first_miss_delta_p50_sec'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow", required=True, help="run root or kv_eviction_shadow.jsonl")
    parser.add_argument("--events", required=True, help="run root or kv_lru_events.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pre-window-sec", type=float, default=2.0)
    parser.add_argument("--post-window-sec", type=float, default=0.5)
    parser.add_argument("--lookup-window-sec", type=float, default=600.0)
    args = parser.parse_args()

    shadow_path = find_probe_file(Path(args.shadow), "kv_eviction_shadow.jsonl")
    events_path = find_probe_file(Path(args.events), "kv_lru_events.jsonl")
    enforcements = read_applied_enforcements(shadow_path)
    evictions_by_block, lookups_by_hash, event_counts = read_lru_events(events_path)
    victim_rows = build_victim_rows(
        enforcements,
        evictions_by_block,
        lookups_by_hash,
        pre_window_sec=max(0.0, args.pre_window_sec),
        post_window_sec=max(0.0, args.post_window_sec),
        lookup_window_sec=max(0.0, args.lookup_window_sec),
    )
    summary = summarize_victims(victim_rows, event_counts)
    groups: list[dict[str, Any]] = []
    for field in (
        "policy",
        "gap_bucket",
        "response_bucket",
        "history_bucket",
        "victim_is_terminal_turn",
        "victim_metadata_present",
        "current_round_index",
    ):
        groups.extend(group_victims(victim_rows, field))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    victim_csv = args.output_dir / "victim_outcomes.csv"
    group_csv = args.output_dir / "victim_outcome_groups.csv"
    json_path = args.output_dir / "victim_outcome_summary.json"
    md_path = args.output_dir / "victim_outcomes.md"
    write_csv(victim_csv, victim_rows)
    write_csv(group_csv, groups)
    json_path.write_text(
        json.dumps(
            {
                "shadow_path": str(shadow_path),
                "events_path": str(events_path),
                "config": {
                    "pre_window_sec": args.pre_window_sec,
                    "post_window_sec": args.post_window_sec,
                    "lookup_window_sec": args.lookup_window_sec,
                },
                "applied_enforcement_events": len(enforcements),
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_markdown(
        md_path,
        shadow_path=shadow_path,
        events_path=events_path,
        enforcements=enforcements,
        summary=summary,
        groups=groups,
    )
    print(json_path)
    print(victim_csv)
    print(group_csv)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
