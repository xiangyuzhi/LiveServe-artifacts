#!/usr/bin/env python3
"""Join Code2Wav emit trace with stage2 trace.

The chunk trace records when the stage1/talker side emitted a Code2Wav chunk.
The stage2 trace records when that same chunk was received, became scheduler
ready, was scheduled, and completed.  With per-request timing rows, this script
also infers the missing stage2-submit-to-emit segment for chunk 0.

This is offline-only and does not require GPU resources.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CHATCMPL_PREFIX = "chatcmpl-"


@dataclass(frozen=True)
class RunInput:
    label: str
    per_request: Path | None
    chunk_trace: Path
    stage2_trace: Path


def finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def normalize_request_id(value: Any) -> str:
    request_id = str(value or "")
    if request_id.startswith(CHATCMPL_PREFIX):
        return request_id[len(CHATCMPL_PREFIX) :]
    return request_id


def pct(values: Iterable[float], q: float) -> float | None:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return None
    idx = (len(xs) - 1) * q / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return {
            "n": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": len(xs),
        "min": xs[0],
        "p50": pct(xs, 50),
        "p90": pct(xs, 90),
        "p95": pct(xs, 95),
        "p99": pct(xs, 99),
        "max": xs[-1],
        "mean": sum(xs) / len(xs),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows, bad


def value_ms(row: dict[str, Any], key: str) -> float | None:
    value = finite(row.get(key))
    if value is not None:
        return value * 1000.0 if key.endswith("_sec") else value
    stage_durations = row.get("server_stage_durations")
    if isinstance(stage_durations, dict):
        mapped = {
            "server_stage2_submit_to_first_output_ms": "stage2_submit_to_first_output_ms",
            "server_final_stage_submit_to_first_output_ms": "final_stage_submit_to_first_output_ms",
            "server_stage0_to_first_final_output_ms": "stage0_submit_to_first_final_output_ms",
        }.get(key)
        if mapped:
            return finite(stage_durations.get(mapped))
    return None


def find_per_request(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.rglob("per_request.jsonl"))
    if not matches:
        raise FileNotFoundError(f"no per_request.jsonl under {path}")
    if len(matches) > 1:
        raise ValueError(f"multiple per_request.jsonl files under {path}; pass one file")
    return matches[0]


def trace_request_id(row: dict[str, Any]) -> str:
    return normalize_request_id(row.get("request_id") or row.get("external_req_id"))


def int_value(value: Any) -> int | None:
    number = finite(value)
    if number is None:
        return None
    return int(number)


def chunk_id(row: dict[str, Any]) -> int | None:
    value = int_value(row.get("chunk_id"))
    if value is not None:
        return value
    meta = row.get("meta")
    if isinstance(meta, dict):
        return int_value(meta.get("codec_emitted_chunks_before"))
    return None


def emit_chunk_id(row: dict[str, Any]) -> int | None:
    return int_value(row.get("emitted_chunks_before"))


def event_ts(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = finite(row.get(key))
        if value is not None:
            return value
    return None


def ms_delta(end: float | None, start: float | None) -> float | None:
    if end is None or start is None:
        return None
    return (end - start) * 1000.0


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def parse_pair(spec: str) -> RunInput:
    parts = spec.split("::")
    if len(parts) == 4:
        label, per_request, chunk_trace, stage2_trace = parts
        return RunInput(
            label=label,
            per_request=find_per_request(Path(per_request)),
            chunk_trace=Path(chunk_trace),
            stage2_trace=Path(stage2_trace),
        )
    if len(parts) == 3:
        label, chunk_trace, stage2_trace = parts
        return RunInput(
            label=label,
            per_request=None,
            chunk_trace=Path(chunk_trace),
            stage2_trace=Path(stage2_trace),
        )
    raise ValueError(
        "--pair must be LABEL::PER_REQUEST::CHUNK_TRACE::STAGE2_TRACE "
        "or LABEL::CHUNK_TRACE::STAGE2_TRACE"
    )


def load_per_request(path: Path | None) -> tuple[dict[str, dict[str, Any]], int, int]:
    if path is None:
        return {}, 0, 0
    rows, bad = load_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        request_id = normalize_request_id(row.get("request_id"))
        if request_id:
            out[request_id] = row
    return out, len(rows), bad


def index_emit_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], dict[str, Any]]:
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    duplicates = 0
    missing_key = 0
    for row in rows:
        request_id = trace_request_id(row)
        cid = emit_chunk_id(row)
        if not request_id or cid is None:
            missing_key += 1
            continue
        key = (request_id, cid)
        if by_key.get(key):
            duplicates += 1
        by_key[key].append(row)
    for values in by_key.values():
        values.sort(key=lambda row: event_ts(row, "ts") or 0.0)
    return by_key, {
        "rows": len(rows),
        "indexed": len(by_key),
        "duplicates": duplicates,
        "missing_key": missing_key,
    }


def codec_meta_present(meta: dict[str, Any]) -> bool:
    return any(
        key in meta
        for key in (
            "codec_emitted_chunks_before",
            "codec_emitted_frames_before",
            "codec_emitted_frames_after",
            "codec_window_frames",
            "codec_new_frames",
            "codec_target_chunk_frames",
        )
    )


def metadata_matches(stage2_row: dict[str, Any], emit_row: dict[str, Any]) -> bool:
    meta = stage2_row.get("meta") if isinstance(stage2_row.get("meta"), dict) else {}
    checks = (
        ("codec_emitted_frames_before", "emitted_frames_before"),
        ("codec_emitted_frames_after", "emitted_frames_after"),
        ("codec_new_frames", "new_frames"),
        ("codec_target_chunk_frames", "target_chunk_frames"),
        ("codec_window_frames", "window_frames"),
    )
    for stage2_key, emit_key in checks:
        stage2_value = int_value(meta.get(stage2_key))
        emit_value = int_value(emit_row.get(emit_key))
        if stage2_value is not None and emit_value is not None and stage2_value != emit_value:
            return False
    if "finished" in meta and "finished" in emit_row:
        if bool(meta.get("finished")) != bool(emit_row.get("finished")):
            return False
    return True


def pick_emit_row(
    stage2_row: dict[str, Any],
    emit_index: dict[tuple[str, int], list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str]:
    request_id = trace_request_id(stage2_row)
    cid = chunk_id(stage2_row)
    if not request_id or cid is None:
        return None, "missing_stage2_key"
    meta = stage2_row.get("meta") if isinstance(stage2_row.get("meta"), dict) else {}
    if not codec_meta_present(meta):
        return None, "stage2_no_codec_meta"
    candidates = emit_index.get((request_id, cid), [])
    if not candidates:
        return None, "no_emit_trace"
    candidates = [row for row in candidates if metadata_matches(stage2_row, row)]
    if not candidates:
        return None, "no_metadata_match"
    recv_ts = event_ts(stage2_row, "recv_ts", "ts")
    if recv_ts is None:
        return candidates[0], "matched_no_recv_ts"
    before = [
        row
        for row in candidates
        if (event_ts(row, "ts") is not None and event_ts(row, "ts") <= recv_ts)
    ]
    if before:
        return max(before, key=lambda row: event_ts(row, "ts") or float("-inf")), "matched"
    return min(candidates, key=lambda row: abs((event_ts(row, "ts") or recv_ts) - recv_ts)), "matched_after_recv"


def first_complete_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("event") != "complete":
            continue
        request_id = trace_request_id(row)
        cid = chunk_id(row)
        if not request_id or cid is None:
            continue
        key = (request_id, cid)
        row_ts = event_ts(row, "complete_ts", "ts") or 0.0
        existing = best.get(key)
        existing_ts = event_ts(existing, "complete_ts", "ts") if existing is not None else None
        if existing is None or existing_ts is None or row_ts < existing_ts:
            best[key] = row
    return sorted(best.values(), key=lambda row: event_ts(row, "complete_ts", "ts") or 0.0)


def build_join_row(
    *,
    label: str,
    stage2_row: dict[str, Any],
    emit_row: dict[str, Any] | None,
    emit_match_reason: str,
    per_request: dict[str, Any] | None,
) -> dict[str, Any]:
    request_id = trace_request_id(stage2_row)
    cid = chunk_id(stage2_row)
    meta = stage2_row.get("meta") if isinstance(stage2_row.get("meta"), dict) else {}

    emit_ts = event_ts(emit_row, "ts") if emit_row is not None else None
    recv_ts = event_ts(stage2_row, "recv_ts")
    ready_ts = event_ts(stage2_row, "ready_ts")
    scheduled_ts = event_ts(stage2_row, "scheduled_ts")
    complete_ts = event_ts(stage2_row, "complete_ts", "ts")

    emit_to_recv_ms = ms_delta(recv_ts, emit_ts)
    emit_to_ready_ms = ms_delta(ready_ts, emit_ts)
    emit_to_schedule_ms = ms_delta(scheduled_ts, emit_ts)
    emit_to_complete_ms = ms_delta(complete_ts, emit_ts)
    recv_to_ready_ms = ms_delta(ready_ts, recv_ts)
    ready_to_schedule_ms = finite(stage2_row.get("ready_to_schedule_ms"))
    recv_to_schedule_ms = finite(stage2_row.get("recv_to_schedule_ms"))
    schedule_to_complete_ms = finite(stage2_row.get("schedule_to_complete_ms"))
    recv_to_complete_ms = None
    if recv_to_schedule_ms is not None and schedule_to_complete_ms is not None:
        recv_to_complete_ms = recv_to_schedule_ms + schedule_to_complete_ms

    stage2_total_ms = None
    audio_ttfp_ms = None
    if per_request is not None:
        audio_ttfp_ms = value_ms(per_request, "audio_ttfp_sec")
        stage2_total_ms = value_ms(
            per_request,
            "server_stage2_submit_to_first_output_ms",
        )
        if stage2_total_ms is None:
            stage2_total_ms = value_ms(
                per_request,
                "server_final_stage_submit_to_first_output_ms",
            )

    inferred_submit_to_emit_ms = None
    if stage2_total_ms is not None and emit_to_complete_ms is not None:
        inferred_submit_to_emit_ms = stage2_total_ms - emit_to_complete_ms

    inferred_submit_to_recv_ms = None
    if stage2_total_ms is not None and recv_to_complete_ms is not None:
        inferred_submit_to_recv_ms = stage2_total_ms - recv_to_complete_ms

    matched = emit_row is not None
    return {
        "label": label,
        "request_id": request_id,
        "chunk_id": cid,
        "is_chunk0": cid == 0,
        "matched_emit": matched,
        "emit_match_reason": emit_match_reason,
        "missing_emit_reason": "" if matched else emit_match_reason,
        "emit_ts": emit_ts,
        "recv_ts": recv_ts,
        "ready_ts": ready_ts,
        "scheduled_ts": scheduled_ts,
        "complete_ts": complete_ts,
        "stage2_total_ms": stage2_total_ms,
        "audio_ttfp_ms": audio_ttfp_ms,
        "inferred_stage2_submit_to_codec_emit_ms": inferred_submit_to_emit_ms,
        "inferred_stage2_submit_to_chunk0_recv_ms": inferred_submit_to_recv_ms,
        "codec_emit_to_stage2_recv_ms": emit_to_recv_ms,
        "codec_emit_to_ready_ms": emit_to_ready_ms,
        "codec_emit_to_schedule_ms": emit_to_schedule_ms,
        "codec_emit_to_complete_ms": emit_to_complete_ms,
        "stage2_recv_to_ready_ms": recv_to_ready_ms,
        "stage2_ready_to_schedule_ms": ready_to_schedule_ms,
        "stage2_recv_to_schedule_ms": recv_to_schedule_ms,
        "stage2_schedule_to_complete_ms": schedule_to_complete_ms,
        "stage2_recv_to_complete_ms": recv_to_complete_ms,
        "submit_to_emit_ratio": safe_ratio(inferred_submit_to_emit_ms, stage2_total_ms),
        "emit_to_recv_ratio": safe_ratio(emit_to_recv_ms, stage2_total_ms),
        "recv_to_complete_ratio": safe_ratio(recv_to_complete_ms, stage2_total_ms),
        "codec_cadence_mode": meta.get("codec_cadence_mode")
        or (emit_row or {}).get("cadence_mode"),
        "codec_window_frames": finite(meta.get("codec_window_frames"))
        or finite((emit_row or {}).get("window_frames")),
        "codec_chunk_frames": finite(meta.get("codec_chunk_frames"))
        or finite((emit_row or {}).get("context_frames")),
        "codec_target_chunk_frames": finite(meta.get("codec_target_chunk_frames"))
        or finite((emit_row or {}).get("target_chunk_frames")),
        "codec_new_frames": finite(meta.get("codec_new_frames"))
        or finite((emit_row or {}).get("new_frames")),
        "codec_emitted_frames_before": finite(meta.get("codec_emitted_frames_before"))
        or finite((emit_row or {}).get("emitted_frames_before")),
        "codec_emitted_frames_after": finite(meta.get("codec_emitted_frames_after"))
        or finite((emit_row or {}).get("emitted_frames_after")),
        "codec_finished": meta.get("finished")
        if "finished" in meta
        else (emit_row or {}).get("finished"),
        "audio_duration_sec": finite((per_request or {}).get("audio_duration_sec")),
        "audio_chunks": finite((per_request or {}).get("audio_chunks")),
        "audio_underrun_sec": finite((per_request or {}).get("audio_underrun_sec")),
        "audio_post_ttfp_rtf": finite((per_request or {}).get("audio_post_ttfp_rtf")),
        "talker_frames_generated": finite(
            (per_request or {}).get("talker_frames_generated")
        ),
    }


METRICS = [
    "stage2_total_ms",
    "audio_ttfp_ms",
    "inferred_stage2_submit_to_codec_emit_ms",
    "inferred_stage2_submit_to_chunk0_recv_ms",
    "codec_emit_to_stage2_recv_ms",
    "stage2_recv_to_ready_ms",
    "stage2_ready_to_schedule_ms",
    "stage2_schedule_to_complete_ms",
    "stage2_recv_to_complete_ms",
    "codec_emit_to_complete_ms",
    "submit_to_emit_ratio",
    "emit_to_recv_ratio",
    "recv_to_complete_ratio",
]


def metric_table(rows: list[dict[str, Any]], metrics: list[str] = METRICS) -> list[str]:
    lines = [
        "| metric | n | p50 | p90 | p95 | p99 | max | mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        stats = describe(
            float(row[metric])
            for row in rows
            if isinstance(row.get(metric), (int, float))
        )
        suffix = "" if metric.endswith("_ratio") else "ms"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{metric}`",
                    str(stats["n"]),
                    fmt(stats["p50"]) + suffix,
                    fmt(stats["p90"]) + suffix,
                    fmt(stats["p95"]) + suffix,
                    fmt(stats["p99"]) + suffix,
                    fmt(stats["max"]) + suffix,
                    fmt(stats["mean"]) + suffix,
                ]
            )
            + " |"
        )
    return lines


def summarize_group(
    name: str,
    rows: list[dict[str, Any]],
) -> str:
    def p90(key: str) -> str:
        stats = describe(float(row[key]) for row in rows if isinstance(row.get(key), (int, float)))
        return fmt(stats["p90"]) + ("ms" if not key.endswith("_ratio") else "")

    return (
        f"| `{name}` | {len(rows)} | "
        f"{sum(1 for row in rows if row.get('matched_emit'))} | "
        f"{p90('stage2_total_ms')} | "
        f"{p90('inferred_stage2_submit_to_codec_emit_ms')} | "
        f"{p90('codec_emit_to_stage2_recv_ms')} | "
        f"{p90('stage2_recv_to_ready_ms')} | "
        f"{p90('stage2_ready_to_schedule_ms')} | "
        f"{p90('stage2_schedule_to_complete_ms')} | "
        f"{p90('submit_to_emit_ratio')} | "
        f"{p90('emit_to_recv_ratio')} |"
    )


def tail_table(rows: list[dict[str, Any]], key: str, limit: int) -> list[str]:
    ranked = sorted(
        [row for row in rows if isinstance(row.get(key), (int, float))],
        key=lambda row: float(row[key]),
        reverse=True,
    )[:limit]
    lines = [
        "| request_id | chunk | matched | stage2 total | submit->emit | emit->recv | recv->ready | ready->schedule | schedule->complete | audio ttfp | audio sec | mode/window |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('request_id')}`",
                    fmt(row.get("chunk_id"), 0),
                    "yes" if row.get("matched_emit") else "no",
                    fmt(row.get("stage2_total_ms")),
                    fmt(row.get("inferred_stage2_submit_to_codec_emit_ms")),
                    fmt(row.get("codec_emit_to_stage2_recv_ms")),
                    fmt(row.get("stage2_recv_to_ready_ms")),
                    fmt(row.get("stage2_ready_to_schedule_ms")),
                    fmt(row.get("stage2_schedule_to_complete_ms")),
                    fmt(row.get("audio_ttfp_ms")),
                    fmt(row.get("audio_duration_sec")),
                    f"{row.get('codec_cadence_mode')}/{fmt(row.get('codec_window_frames'), 0)}",
                ]
            )
            + " |"
        )
    return lines


def analyze_run(run: RunInput) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    per_rows, per_count, per_bad = load_per_request(run.per_request)
    chunk_rows, chunk_bad = load_jsonl(run.chunk_trace)
    stage2_rows, stage2_bad = load_jsonl(run.stage2_trace)
    emit_index, emit_stats = index_emit_rows(chunk_rows)
    complete_rows = first_complete_rows(stage2_rows)

    joined: list[dict[str, Any]] = []
    missing_per_request = 0
    for stage2_row in complete_rows:
        request_id = trace_request_id(stage2_row)
        cid = chunk_id(stage2_row)
        key = (request_id, int(cid)) if request_id and cid is not None else None
        if key is None:
            emit_row = None
            emit_match_reason = "missing_stage2_key"
        else:
            emit_row, emit_match_reason = pick_emit_row(stage2_row, emit_index)
        per_request = per_rows.get(request_id)
        if per_rows and per_request is None:
            missing_per_request += 1
        joined.append(
            build_join_row(
                label=run.label,
                stage2_row=stage2_row,
                emit_row=emit_row,
                emit_match_reason=emit_match_reason,
                per_request=per_request,
            )
        )

    matched_count = sum(1 for row in joined if row["matched_emit"])
    chunk0 = [row for row in joined if row.get("chunk_id") == 0]
    chunk0_matched = [row for row in chunk0 if row.get("matched_emit")]
    chunk0_unmatched = [row for row in chunk0 if not row.get("matched_emit")]
    positive_audio = [
        row
        for row in chunk0
        if isinstance(row.get("audio_duration_sec"), (int, float))
        and float(row["audio_duration_sec"]) > 0.0
    ]
    zero_duration = [
        row
        for row in chunk0
        if isinstance(row.get("audio_duration_sec"), (int, float))
        and float(row["audio_duration_sec"]) <= 0.0
    ]

    events = Counter(str(row.get("event")) for row in stage2_rows)
    cadence = Counter(str(row.get("codec_cadence_mode")) for row in joined)
    finished = Counter(str(row.get("codec_finished")) for row in joined)
    missing_reasons = Counter(
        str(row.get("missing_emit_reason"))
        for row in joined
        if not row.get("matched_emit")
    )
    negative_counts = Counter()
    for row in joined:
        for key in (
            "inferred_stage2_submit_to_codec_emit_ms",
            "codec_emit_to_stage2_recv_ms",
            "inferred_stage2_submit_to_chunk0_recv_ms",
        ):
            value = row.get(key)
            if isinstance(value, (int, float)) and value < -1e-3:
                negative_counts[key] += 1

    lines: list[str] = [f"## {run.label}", ""]
    if run.per_request is not None:
        lines.append(f"Per-request: `{run.per_request}`")
        lines.append("")
    lines.append(f"Chunk trace: `{run.chunk_trace}`")
    lines.append("")
    lines.append(f"Stage2 trace: `{run.stage2_trace}`")
    lines.append("")
    lines.append(
        "Rows: "
        f"per_request={per_count} "
        f"chunk_trace={len(chunk_rows)} "
        f"stage2_trace={len(stage2_rows)} "
        f"stage2_complete_chunks={len(complete_rows)} "
        f"joined={len(joined)} "
        f"matched_emit={matched_count} "
        f"chunk0={len(chunk0)} "
        f"chunk0_matched={len(chunk0_matched)} "
        f"chunk0_unmatched={len(chunk0_unmatched)}"
    )
    lines.append("")
    lines.append(f"Bad JSON: per_request={per_bad} chunk_trace={chunk_bad} stage2_trace={stage2_bad}")
    lines.append("")
    lines.append(f"Stage2 events: `{dict(events)}`")
    lines.append("")
    lines.append(f"Emit index stats: `{emit_stats}`")
    lines.append("")
    lines.append(f"Missing per-request rows for stage2 chunks: `{missing_per_request}`")
    lines.append("")
    lines.append(f"Cadence modes among completed chunks: `{dict(cadence)}`")
    lines.append("")
    lines.append(f"Finished flags among completed chunks: `{dict(finished)}`")
    lines.append("")
    lines.append(f"Missing emit reasons: `{dict(missing_reasons)}`")
    lines.append("")
    lines.append(f"Negative inferred counts: `{dict(negative_counts)}`")
    lines.append("")

    lines.append("### Chunk0 Latency Split")
    lines.append("")
    lines.extend(metric_table(chunk0))
    lines.append("")
    lines.append("### Chunk0 Matched-Only Latency Split")
    lines.append("")
    lines.extend(metric_table(chunk0_matched))
    lines.append("")

    lines.append("### Grouped Chunk0 Split")
    lines.append("")
    lines.append(
        "| group | n | matched | stage2 p90 | submit->emit p90 | emit->recv p90 | recv->ready p90 | ready wait p90 | service p90 | submit->emit ratio p90 | emit->recv ratio p90 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    groups = {
        "all": chunk0,
        "matched": chunk0_matched,
        "unmatched": chunk0_unmatched,
        "positive_audio": positive_audio,
        "zero_duration": zero_duration,
    }
    for name, group_rows in groups.items():
        lines.append(summarize_group(name, group_rows))
    lines.append("")

    lines.append("### All Completed Chunks Transfer")
    lines.append("")
    lines.extend(
        metric_table(
            joined,
            [
                "codec_emit_to_stage2_recv_ms",
                "stage2_recv_to_ready_ms",
                "stage2_ready_to_schedule_ms",
                "stage2_schedule_to_complete_ms",
                "stage2_recv_to_complete_ms",
            ],
        )
    )
    lines.append("")

    lines.append("### Tail By Chunk0 Stage2 Total")
    lines.append("")
    lines.extend(tail_table(chunk0, "stage2_total_ms", 12))
    lines.append("")

    lines.append("### Tail By Chunk0 Submit->Emit")
    lines.append("")
    lines.extend(tail_table(chunk0, "inferred_stage2_submit_to_codec_emit_ms", 12))
    lines.append("")

    lines.append("### Tail By Chunk0 Emit->Recv")
    lines.append("")
    lines.extend(tail_table(chunk0, "codec_emit_to_stage2_recv_ms", 12))
    lines.append("")

    summary = {
        "label": run.label,
        "per_request": str(run.per_request) if run.per_request else None,
        "chunk_trace": str(run.chunk_trace),
        "stage2_trace": str(run.stage2_trace),
        "per_request_rows": per_count,
        "chunk_trace_rows": len(chunk_rows),
        "stage2_trace_rows": len(stage2_rows),
        "stage2_complete_chunks": len(complete_rows),
        "joined_rows": len(joined),
        "matched_emit_rows": matched_count,
        "chunk0_rows": len(chunk0),
        "chunk0_matched_rows": len(chunk0_matched),
        "chunk0_unmatched_rows": len(chunk0_unmatched),
        "stage2_bad_json": stage2_bad,
        "chunk_bad_json": chunk_bad,
        "per_bad_json": per_bad,
        "emit_index_stats": emit_stats,
        "events": dict(events),
        "negative_counts": dict(negative_counts),
        "missing_emit_reasons": dict(missing_reasons),
        "chunk0": {key: describe(row[key] for row in chunk0 if isinstance(row.get(key), (int, float))) for key in METRICS},
        "chunk0_matched": {key: describe(row[key] for row in chunk0_matched if isinstance(row.get(key), (int, float))) for key in METRICS},
        "positive_audio": {key: describe(row[key] for row in positive_audio if isinstance(row.get(key), (int, float))) for key in METRICS},
        "zero_duration": {key: describe(row[key] for row in zero_duration if isinstance(row.get(key), (int, float))) for key in METRICS},
    }
    return "\n".join(lines).rstrip() + "\n", joined, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        for group in ("chunk0", "chunk0_matched", "positive_audio", "zero_duration"):
            group_stats = summary.get(group)
            if not isinstance(group_stats, dict):
                continue
            for metric, stats in group_stats.items():
                if not isinstance(stats, dict):
                    continue
                row = {
                    "label": summary["label"],
                    "group": group,
                    "metric": metric,
                    **stats,
                }
                rows.append(row)
    write_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help=(
            "LABEL::PER_REQUEST::CHUNK_TRACE::STAGE2_TRACE, or "
            "LABEL::CHUNK_TRACE::STAGE2_TRACE. Repeatable."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.pair:
        parser.error("at least one --pair is required")

    runs = [parse_pair(spec) for spec in args.pair]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Code2Wav Emit -> Stage2 Attribution",
        "",
        "This report joins Code2Wav chunk emission trace with stage2 recv/ready/schedule/complete trace.",
        "When per-request timing is available, chunk0 `stage2 submit -> codec emit` is inferred as server stage2 first-output time minus observed emit-to-complete time.",
        "",
    ]
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run in runs:
        report, rows, summary = analyze_run(run)
        report_lines.append(report)
        all_rows.extend(rows)
        summaries.append(summary)

    report_md = "\n".join(report_lines).rstrip() + "\n"
    (args.output_dir / "codec_emit_to_stage2.md").write_text(
        report_md,
        encoding="utf-8",
    )
    (args.output_dir / "codec_emit_to_stage2.json").write_text(
        json.dumps({"runs": summaries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "codec_emit_to_stage2_rows.csv", all_rows)
    write_summary_csv(args.output_dir / "codec_emit_to_stage2_summary.csv", summaries)
    print(report_md)


if __name__ == "__main__":
    main()
