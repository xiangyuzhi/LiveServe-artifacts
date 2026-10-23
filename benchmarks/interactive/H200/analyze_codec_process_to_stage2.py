#!/usr/bin/env python3
"""Join Code2Wav process trace with stage2 first-output timing.

This analyzer focuses on chunk0.  It splits the previously inferred
`stage2 submit -> codec emit` segment into:

  stage2 submit -> first codec frame observed by the stage1 payload builder
  first codec frame -> Code2Wav chunk emit
  emit -> stage2 recv/ready/schedule/complete

It is offline-only and does not require GPU resources.
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

from analyze_codec_emit_to_stage2 import (
    describe,
    event_ts,
    finite,
    first_complete_rows,
    fmt,
    index_emit_rows,
    load_jsonl,
    load_per_request,
    ms_delta,
    pick_emit_row,
    trace_request_id,
    value_ms,
)


@dataclass(frozen=True)
class RunInput:
    label: str
    per_request: Path
    process_trace: Path
    chunk_trace: Path
    stage2_trace: Path


METRICS = [
    "stage2_total_ms",
    "audio_ttfp_ms",
    "inferred_stage2_submit_to_first_code_frame_ms",
    "first_code_frame_to_codec_emit_ms",
    "inferred_stage2_submit_to_codec_emit_ms",
    "codec_emit_to_stage2_recv_ms",
    "stage2_recv_to_ready_ms",
    "stage2_ready_to_schedule_ms",
    "stage2_schedule_to_complete_ms",
    "stage2_recv_to_complete_ms",
    "first_code_frame_to_complete_ms",
    "codec_emit_to_complete_ms",
    "submit_to_first_code_frame_ratio",
    "first_code_frame_to_emit_ratio",
    "emit_to_recv_ratio",
]


def parse_pair(spec: str) -> RunInput:
    parts = spec.split("::")
    if len(parts) != 5:
        raise ValueError(
            "--pair must be LABEL::PER_REQUEST::PROCESS_TRACE::CHUNK_TRACE::STAGE2_TRACE"
        )
    label, per_request, process_trace, chunk_trace, stage2_trace = parts
    return RunInput(
        label=label,
        per_request=Path(per_request),
        process_trace=Path(process_trace),
        chunk_trace=Path(chunk_trace),
        stage2_trace=Path(stage2_trace),
    )


def int_value(value: Any) -> int | None:
    number = finite(value)
    if number is None:
        return None
    return int(number)


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def process_chunk_id(row: dict[str, Any]) -> int | None:
    for key in ("emitted_chunks_before", "chunk_id"):
        value = int_value(row.get(key))
        if value is not None:
            return value
    return None


def index_process_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_frame_by_request: dict[str, dict[str, Any]] = {}
    emit_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    gate_counts_by_key: Counter[tuple[str, int]] = Counter()
    frames_appended_by_request: Counter[str] = Counter()
    event_counts = Counter(str(row.get("event")) for row in rows)

    for row in rows:
        request_id = trace_request_id(row)
        if not request_id:
            continue
        event = str(row.get("event") or "")
        if event == "frames_appended":
            frames_appended_by_request[request_id] += 1
            if bool(row.get("was_first_frame")) or request_id not in first_frame_by_request:
                existing = first_frame_by_request.get(request_id)
                row_ts = event_ts(row, "ts") or 0.0
                existing_ts = event_ts(existing, "ts") if existing is not None else None
                if existing is None or existing_ts is None or row_ts < existing_ts:
                    first_frame_by_request[request_id] = row
        elif event == "emit":
            cid = process_chunk_id(row)
            if cid is not None:
                emit_by_key[(request_id, cid)].append(row)
        elif event in {"gate_wait", "gate_coalesce_hold"}:
            cid = process_chunk_id(row)
            if cid is not None:
                gate_counts_by_key[(request_id, cid)] += 1

    for values in emit_by_key.values():
        values.sort(key=lambda row: event_ts(row, "ts") or 0.0)

    return {
        "first_frame_by_request": first_frame_by_request,
        "emit_by_key": emit_by_key,
        "gate_counts_by_key": gate_counts_by_key,
        "frames_appended_by_request": frames_appended_by_request,
        "event_counts": event_counts,
    }


def pick_process_emit(
    request_id: str,
    chunk_id: int,
    process_index: dict[str, Any],
    emit_ts: float | None,
) -> dict[str, Any] | None:
    rows = process_index["emit_by_key"].get((request_id, chunk_id), [])
    if not rows:
        return None
    if emit_ts is None:
        return rows[0]
    return min(rows, key=lambda row: abs((event_ts(row, "ts") or emit_ts) - emit_ts))


def build_row(
    *,
    label: str,
    stage2_row: dict[str, Any],
    emit_row: dict[str, Any] | None,
    emit_match_reason: str,
    process_index: dict[str, Any],
    per_request: dict[str, Any] | None,
) -> dict[str, Any]:
    request_id = trace_request_id(stage2_row)
    cid = int_value(stage2_row.get("chunk_id"))
    if cid is None:
        meta = stage2_row.get("meta")
        if isinstance(meta, dict):
            cid = int_value(meta.get("codec_emitted_chunks_before"))

    emit_ts = event_ts(emit_row, "ts") if emit_row is not None else None
    process_emit = pick_process_emit(request_id, int(cid or 0), process_index, emit_ts)
    first_frame_row = process_index["first_frame_by_request"].get(request_id)
    first_frame_ts = event_ts(first_frame_row, "ts") if first_frame_row else None
    if first_frame_ts is None and process_emit is not None:
        first_frame_ts = finite(process_emit.get("first_frame_ts"))
    latest_frame_ts = finite(process_emit.get("latest_frame_ts")) if process_emit is not None else None

    recv_ts = event_ts(stage2_row, "recv_ts")
    ready_ts = event_ts(stage2_row, "ready_ts")
    scheduled_ts = event_ts(stage2_row, "scheduled_ts")
    complete_ts = event_ts(stage2_row, "complete_ts", "ts")

    emit_to_recv_ms = ms_delta(recv_ts, emit_ts)
    emit_to_complete_ms = ms_delta(complete_ts, emit_ts)
    first_to_emit_ms = ms_delta(emit_ts, first_frame_ts)
    first_to_complete_ms = ms_delta(complete_ts, first_frame_ts)
    recv_to_ready_ms = ms_delta(ready_ts, recv_ts)
    ready_to_schedule_ms = finite(stage2_row.get("ready_to_schedule_ms"))
    recv_to_schedule_ms = finite(stage2_row.get("recv_to_schedule_ms"))
    schedule_to_complete_ms = finite(stage2_row.get("schedule_to_complete_ms"))
    recv_to_complete_ms = None
    if recv_to_schedule_ms is not None and schedule_to_complete_ms is not None:
        recv_to_complete_ms = recv_to_schedule_ms + schedule_to_complete_ms

    stage2_total_ms = None
    audio_ttfp_ms = None
    audio_duration_sec = None
    if per_request is not None:
        audio_ttfp_ms = value_ms(per_request, "audio_ttfp_sec")
        audio_duration_sec = finite(per_request.get("audio_duration_sec"))
        stage2_total_ms = value_ms(per_request, "server_stage2_submit_to_first_output_ms")
        if stage2_total_ms is None:
            stage2_total_ms = value_ms(per_request, "server_final_stage_submit_to_first_output_ms")

    submit_to_first_ms = None
    if stage2_total_ms is not None and first_to_complete_ms is not None:
        submit_to_first_ms = stage2_total_ms - first_to_complete_ms
    submit_to_emit_ms = None
    if stage2_total_ms is not None and emit_to_complete_ms is not None:
        submit_to_emit_ms = stage2_total_ms - emit_to_complete_ms

    gate_counts = process_index["gate_counts_by_key"]
    frames_appended = process_index["frames_appended_by_request"]
    process_matched = process_emit is not None
    first_frame_matched = first_frame_ts is not None
    return {
        "label": label,
        "request_id": request_id,
        "chunk_id": cid,
        "matched_emit": emit_row is not None,
        "emit_match_reason": emit_match_reason,
        "matched_process_emit": process_matched,
        "matched_first_code_frame": first_frame_matched,
        "missing_process_reason": ""
        if process_matched and first_frame_matched
        else (
            "missing_process_emit"
            if not process_matched
            else "missing_first_code_frame"
        ),
        "stage2_total_ms": stage2_total_ms,
        "audio_ttfp_ms": audio_ttfp_ms,
        "inferred_stage2_submit_to_first_code_frame_ms": submit_to_first_ms,
        "first_code_frame_to_codec_emit_ms": first_to_emit_ms,
        "inferred_stage2_submit_to_codec_emit_ms": submit_to_emit_ms,
        "codec_emit_to_stage2_recv_ms": emit_to_recv_ms,
        "stage2_recv_to_ready_ms": recv_to_ready_ms,
        "stage2_ready_to_schedule_ms": ready_to_schedule_ms,
        "stage2_schedule_to_complete_ms": schedule_to_complete_ms,
        "stage2_recv_to_complete_ms": recv_to_complete_ms,
        "first_code_frame_to_complete_ms": first_to_complete_ms,
        "codec_emit_to_complete_ms": emit_to_complete_ms,
        "submit_to_first_code_frame_ratio": safe_ratio(submit_to_first_ms, stage2_total_ms),
        "first_code_frame_to_emit_ratio": safe_ratio(first_to_emit_ms, stage2_total_ms),
        "emit_to_recv_ratio": safe_ratio(emit_to_recv_ms, stage2_total_ms),
        "process_gate_wait_events_before_emit": int(gate_counts.get((request_id, int(cid or 0)), 0)),
        "process_frames_appended_events": int(frames_appended.get(request_id, 0)),
        "first_frame_batch_frames": finite((first_frame_row or {}).get("frames_appended")),
        "codec_target_chunk_frames": finite((process_emit or {}).get("target_chunk_frames"))
        if process_emit is not None
        else None,
        "codec_window_frames": finite((process_emit or {}).get("window_frames"))
        if process_emit is not None
        else None,
        "codec_cadence_mode": (process_emit or {}).get("cadence_mode"),
        "first_frame_ts": first_frame_ts,
        "latest_frame_ts": latest_frame_ts,
        "emit_ts": emit_ts,
        "recv_ts": recv_ts,
        "complete_ts": complete_ts,
        "audio_duration_sec": audio_duration_sec,
    }


def metric_table(rows: list[dict[str, Any]], metrics: Iterable[str] = METRICS) -> list[str]:
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


def summary_row(name: str, rows: list[dict[str, Any]]) -> str:
    def p90(metric: str) -> str:
        stats = describe(float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float)))
        if metric.endswith("_ratio"):
            return fmt(stats["p90"])
        return fmt(stats["p90"]) + "ms"

    return (
        f"| `{name}` | {len(rows)} | "
        f"{sum(1 for row in rows if row.get('matched_first_code_frame'))} | "
        f"{p90('stage2_total_ms')} | "
        f"{p90('inferred_stage2_submit_to_first_code_frame_ms')} | "
        f"{p90('first_code_frame_to_codec_emit_ms')} | "
        f"{p90('inferred_stage2_submit_to_codec_emit_ms')} | "
        f"{p90('codec_emit_to_stage2_recv_ms')} | "
        f"{p90('stage2_ready_to_schedule_ms')} | "
        f"{p90('stage2_schedule_to_complete_ms')} | "
        f"{p90('submit_to_first_code_frame_ratio')} | "
        f"{p90('first_code_frame_to_emit_ratio')} |"
    )


def tail_table(rows: list[dict[str, Any]], key: str, limit: int = 12) -> list[str]:
    ranked = sorted(
        [row for row in rows if isinstance(row.get(key), (int, float))],
        key=lambda row: float(row[key]),
        reverse=True,
    )[:limit]
    lines = [
        "| request_id | stage2 total | submit->first frame | first frame->emit | submit->emit | emit->recv | ready wait | service | audio ttfp | audio sec | gates | frames events | target/window |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('request_id')}`",
                    fmt(row.get("stage2_total_ms")),
                    fmt(row.get("inferred_stage2_submit_to_first_code_frame_ms")),
                    fmt(row.get("first_code_frame_to_codec_emit_ms")),
                    fmt(row.get("inferred_stage2_submit_to_codec_emit_ms")),
                    fmt(row.get("codec_emit_to_stage2_recv_ms")),
                    fmt(row.get("stage2_ready_to_schedule_ms")),
                    fmt(row.get("stage2_schedule_to_complete_ms")),
                    fmt(row.get("audio_ttfp_ms")),
                    fmt(row.get("audio_duration_sec")),
                    fmt(row.get("process_gate_wait_events_before_emit"), 0),
                    fmt(row.get("process_frames_appended_events"), 0),
                    f"{fmt(row.get('codec_target_chunk_frames'), 0)}/{fmt(row.get('codec_window_frames'), 0)}",
                ]
            )
            + " |"
        )
    return lines


def analyze_run(run: RunInput) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    per_rows, per_count, per_bad = load_per_request(run.per_request)
    process_rows, process_bad = load_jsonl(run.process_trace)
    chunk_rows, chunk_bad = load_jsonl(run.chunk_trace)
    stage2_rows, stage2_bad = load_jsonl(run.stage2_trace)
    process_index = index_process_rows(process_rows)
    emit_index, emit_stats = index_emit_rows(chunk_rows)

    joined: list[dict[str, Any]] = []
    missing_per_request = 0
    for stage2_row in first_complete_rows(stage2_rows):
        cid = int_value(stage2_row.get("chunk_id"))
        if cid is None:
            meta = stage2_row.get("meta")
            if isinstance(meta, dict):
                cid = int_value(meta.get("codec_emitted_chunks_before"))
        if cid != 0:
            continue
        request_id = trace_request_id(stage2_row)
        emit_row, emit_match_reason = pick_emit_row(stage2_row, emit_index)
        per_request = per_rows.get(request_id)
        if per_request is None:
            missing_per_request += 1
        joined.append(
            build_row(
                label=run.label,
                stage2_row=stage2_row,
                emit_row=emit_row,
                emit_match_reason=emit_match_reason,
                process_index=process_index,
                per_request=per_request,
            )
        )

    positive_audio = [
        row
        for row in joined
        if isinstance(row.get("audio_duration_sec"), (int, float))
        and float(row["audio_duration_sec"]) > 0.0
    ]
    zero_duration = [
        row
        for row in joined
        if isinstance(row.get("audio_duration_sec"), (int, float))
        and float(row["audio_duration_sec"]) <= 0.0
    ]
    matched_first = [row for row in joined if row.get("matched_first_code_frame")]
    missing_process = Counter(
        str(row.get("missing_process_reason"))
        for row in joined
        if row.get("missing_process_reason")
    )
    negative_counts = Counter()
    for row in joined:
        for key in (
            "inferred_stage2_submit_to_first_code_frame_ms",
            "first_code_frame_to_codec_emit_ms",
            "inferred_stage2_submit_to_codec_emit_ms",
        ):
            value = row.get(key)
            if isinstance(value, (int, float)) and value < -1e-3:
                negative_counts[key] += 1

    lines = [f"## {run.label}", ""]
    lines.append(f"Per-request: `{run.per_request}`")
    lines.append("")
    lines.append(f"Process trace: `{run.process_trace}`")
    lines.append("")
    lines.append(f"Chunk trace: `{run.chunk_trace}`")
    lines.append("")
    lines.append(f"Stage2 trace: `{run.stage2_trace}`")
    lines.append("")
    lines.append(
        "Rows: "
        f"per_request={per_count} "
        f"process_trace={len(process_rows)} "
        f"chunk_trace={len(chunk_rows)} "
        f"stage2_trace={len(stage2_rows)} "
        f"chunk0_joined={len(joined)} "
        f"matched_first_frame={len(matched_first)}"
    )
    lines.append("")
    lines.append(
        "Bad JSON: "
        f"per_request={per_bad} process_trace={process_bad} "
        f"chunk_trace={chunk_bad} stage2_trace={stage2_bad}"
    )
    lines.append("")
    lines.append(f"Process events: `{dict(process_index['event_counts'])}`")
    lines.append("")
    lines.append(f"Emit index stats: `{emit_stats}`")
    lines.append("")
    lines.append(f"Missing per-request rows: `{missing_per_request}`")
    lines.append("")
    lines.append(f"Missing process reasons: `{dict(missing_process)}`")
    lines.append("")
    lines.append(f"Negative inferred counts: `{dict(negative_counts)}`")
    lines.append("")

    lines.append("### Chunk0 Process Split")
    lines.append("")
    lines.extend(metric_table(joined))
    lines.append("")

    lines.append("### Grouped Chunk0 Split")
    lines.append("")
    lines.append(
        "| group | n | first frame matched | stage2 p90 | submit->first frame p90 | first frame->emit p90 | submit->emit p90 | emit->recv p90 | ready wait p90 | service p90 | submit->first ratio p90 | first->emit ratio p90 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    groups = {
        "all": joined,
        "matched_first": matched_first,
        "positive_audio": positive_audio,
        "zero_duration": zero_duration,
    }
    for name, rows in groups.items():
        lines.append(summary_row(name, rows))
    lines.append("")

    lines.append("### Tail By Submit->First Code Frame")
    lines.append("")
    lines.extend(tail_table(joined, "inferred_stage2_submit_to_first_code_frame_ms"))
    lines.append("")
    lines.append("### Tail By First Code Frame->Emit")
    lines.append("")
    lines.extend(tail_table(joined, "first_code_frame_to_codec_emit_ms"))
    lines.append("")

    summary = {
        "label": run.label,
        "per_request": str(run.per_request),
        "process_trace": str(run.process_trace),
        "chunk_trace": str(run.chunk_trace),
        "stage2_trace": str(run.stage2_trace),
        "per_request_rows": per_count,
        "process_trace_rows": len(process_rows),
        "chunk_trace_rows": len(chunk_rows),
        "stage2_trace_rows": len(stage2_rows),
        "chunk0_joined_rows": len(joined),
        "matched_first_frame_rows": len(matched_first),
        "process_events": dict(process_index["event_counts"]),
        "missing_process_reasons": dict(missing_process),
        "negative_counts": dict(negative_counts),
        "groups": {
            name: {
                key: describe(row[key] for row in rows if isinstance(row.get(key), (int, float)))
                for key in METRICS
            }
            for name, rows in groups.items()
        },
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
        groups = summary.get("groups")
        if not isinstance(groups, dict):
            continue
        for group_name, metrics in groups.items():
            if not isinstance(metrics, dict):
                continue
            for metric, stats in metrics.items():
                if isinstance(stats, dict):
                    rows.append(
                        {
                            "label": summary["label"],
                            "group": group_name,
                            "metric": metric,
                            **stats,
                        }
                    )
    write_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="LABEL::PER_REQUEST::PROCESS_TRACE::CHUNK_TRACE::STAGE2_TRACE. Repeatable.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.pair:
        parser.error("at least one --pair is required")

    runs = [parse_pair(spec) for spec in args.pair]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Code2Wav Process -> Stage2 Attribution",
        "",
        "This report joins Code2Wav process trace, chunk emit trace, stage2 trace, and per-request server timings.",
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
    (args.output_dir / "codec_process_to_stage2.md").write_text(report_md, encoding="utf-8")
    (args.output_dir / "codec_process_to_stage2.json").write_text(
        json.dumps({"runs": summaries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "codec_process_to_stage2_rows.csv", all_rows)
    write_summary_csv(args.output_dir / "codec_process_to_stage2_summary.csv", summaries)
    print(report_md)


if __name__ == "__main__":
    main()
