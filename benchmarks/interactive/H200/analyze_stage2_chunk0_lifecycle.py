#!/usr/bin/env python3
"""Join per-request rows with Code2Wav stage2 trace for chunk-0 timing.

The server timing SSE fields tell us how long the final stage took from submit
to first output.  The stage2 trace tells us when the first Code2Wav chunk was
received, became scheduler-ready, was scheduled, and completed.  Joining them by
request id gives an offline split of first-audio latency into:

* inferred stage2 submit -> chunk0 recv/ready
* ready queue wait
* model/decode service after scheduling

This is intentionally offline-only and does not require GPU resources.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CHATCMPL_PREFIX = "chatcmpl-"


@dataclass(frozen=True)
class RunInput:
    label: str
    per_request: Path
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


def chunk_id(row: dict[str, Any]) -> int | None:
    value = finite(row.get("chunk_id"))
    if value is None:
        return None
    return int(value)


def first_complete_by_request(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("event") != "complete" or chunk_id(row) != 0:
            continue
        request_id = trace_request_id(row)
        if not request_id:
            continue
        existing = out.get(request_id)
        row_ts = finite(row.get("complete_ts")) or finite(row.get("ts")) or 0.0
        existing_ts = (
            finite(existing.get("complete_ts")) or finite(existing.get("ts")) or float("inf")
            if existing is not None
            else float("inf")
        )
        if existing is None or row_ts < existing_ts:
            out[request_id] = row
    return out


def ms_delta(row: dict[str, Any], end_key: str, start_key: str) -> float | None:
    end = finite(row.get(end_key))
    start = finite(row.get(start_key))
    if end is None or start is None:
        return None
    return (end - start) * 1000.0


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def build_lifecycle_row(per_row: dict[str, Any], trace_row: dict[str, Any]) -> dict[str, Any]:
    meta = trace_row.get("meta") if isinstance(trace_row.get("meta"), dict) else {}
    request_id = normalize_request_id(per_row.get("request_id"))
    audio_ttfp_ms = value_ms(per_row, "audio_ttfp_sec")
    stage2_total_ms = value_ms(per_row, "server_stage2_submit_to_first_output_ms")
    if stage2_total_ms is None:
        stage2_total_ms = value_ms(per_row, "server_final_stage_submit_to_first_output_ms")

    recv_to_ready_ms = ms_delta(trace_row, "ready_ts", "recv_ts")
    ready_to_schedule_ms = finite(trace_row.get("ready_to_schedule_ms"))
    recv_to_schedule_ms = finite(trace_row.get("recv_to_schedule_ms"))
    schedule_to_complete_ms = finite(trace_row.get("schedule_to_complete_ms"))

    inferred_submit_to_ready_ms = None
    inferred_submit_to_recv_ms = None
    if stage2_total_ms is not None and schedule_to_complete_ms is not None:
        if ready_to_schedule_ms is not None:
            inferred_submit_to_ready_ms = (
                stage2_total_ms - ready_to_schedule_ms - schedule_to_complete_ms
            )
        if recv_to_schedule_ms is not None:
            inferred_submit_to_recv_ms = (
                stage2_total_ms - recv_to_schedule_ms - schedule_to_complete_ms
            )

    chunk0_recv_to_complete_ms = None
    if recv_to_schedule_ms is not None and schedule_to_complete_ms is not None:
        chunk0_recv_to_complete_ms = recv_to_schedule_ms + schedule_to_complete_ms

    chunk0_ready_to_complete_ms = None
    if ready_to_schedule_ms is not None and schedule_to_complete_ms is not None:
        chunk0_ready_to_complete_ms = ready_to_schedule_ms + schedule_to_complete_ms

    return {
        "request_id": request_id,
        "user_id": per_row.get("user_id"),
        "round_index": per_row.get("round_index"),
        "timestamp": per_row.get("timestamp"),
        "trace_entry_index": per_row.get("trace_entry_index"),
        "audio_ttfp_ms": audio_ttfp_ms,
        "server_stage2_submit_to_first_output_ms": stage2_total_ms,
        "server_stage0_to_first_final_output_ms": value_ms(
            per_row, "server_stage0_to_first_final_output_ms"
        ),
        "chunk0_recv_to_ready_ms": recv_to_ready_ms,
        "chunk0_ready_to_schedule_ms": ready_to_schedule_ms,
        "chunk0_recv_to_schedule_ms": recv_to_schedule_ms,
        "chunk0_schedule_to_complete_ms": schedule_to_complete_ms,
        "chunk0_recv_to_complete_ms": chunk0_recv_to_complete_ms,
        "chunk0_ready_to_complete_ms": chunk0_ready_to_complete_ms,
        "inferred_stage2_submit_to_chunk0_recv_ms": inferred_submit_to_recv_ms,
        "inferred_stage2_submit_to_chunk0_ready_ms": inferred_submit_to_ready_ms,
        "inferred_submit_to_recv_ratio": safe_ratio(
            inferred_submit_to_recv_ms, stage2_total_ms
        ),
        "inferred_submit_to_ready_ratio": safe_ratio(
            inferred_submit_to_ready_ms, stage2_total_ms
        ),
        "ready_to_schedule_ratio": safe_ratio(ready_to_schedule_ms, stage2_total_ms),
        "schedule_to_complete_ratio": safe_ratio(schedule_to_complete_ms, stage2_total_ms),
        "recv_to_complete_ratio": safe_ratio(chunk0_recv_to_complete_ms, stage2_total_ms),
        "codec_cadence_mode": meta.get("codec_cadence_mode"),
        "codec_window_frames": finite(meta.get("codec_window_frames")),
        "codec_chunk_frames": finite(meta.get("codec_chunk_frames")),
        "codec_target_chunk_frames": finite(meta.get("codec_target_chunk_frames")),
        "codec_new_frames": finite(meta.get("codec_new_frames")),
        "codec_finished": meta.get("finished"),
        "audio_duration_sec": finite(per_row.get("audio_duration_sec")),
        "audio_chunks": finite(per_row.get("audio_chunks")),
        "audio_underrun_sec": finite(per_row.get("audio_underrun_sec")),
        "audio_post_ttfp_rtf": finite(per_row.get("audio_post_ttfp_rtf")),
        "talker_frames_generated": finite(per_row.get("talker_frames_generated")),
    }


METRICS = [
    "audio_ttfp_ms",
    "server_stage2_submit_to_first_output_ms",
    "chunk0_recv_to_ready_ms",
    "chunk0_ready_to_schedule_ms",
    "chunk0_schedule_to_complete_ms",
    "chunk0_recv_to_schedule_ms",
    "chunk0_ready_to_complete_ms",
    "chunk0_recv_to_complete_ms",
    "inferred_stage2_submit_to_chunk0_recv_ms",
    "inferred_stage2_submit_to_chunk0_ready_ms",
    "inferred_submit_to_recv_ratio",
    "inferred_submit_to_ready_ratio",
    "ready_to_schedule_ratio",
    "schedule_to_complete_ratio",
    "recv_to_complete_ratio",
]


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    positive_audio = [
        row
        for row in rows
        if (finite(row.get("audio_duration_sec")) or 0.0) > 0.0
    ]
    zero_duration = [
        row
        for row in rows
        if (finite(row.get("audio_duration_sec")) or 0.0) <= 0.0
    ]
    return {
        "all": rows,
        "positive_audio": positive_audio,
        "zero_duration": zero_duration,
    }


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: describe(
            value
            for row in rows
            if (value := finite(row.get(name))) is not None
        )
        for name in METRICS
    }


def summarize_run(run: RunInput, top_k: int) -> dict[str, Any]:
    per_rows_all, per_bad = load_jsonl(run.per_request)
    trace_rows, trace_bad = load_jsonl(run.stage2_trace)
    trace_chunk0 = first_complete_by_request(trace_rows)

    audio_rows = [
        row
        for row in per_rows_all
        if row.get("error") is None
        and value_ms(row, "audio_ttfp_sec") is not None
        and value_ms(row, "server_stage2_submit_to_first_output_ms") is not None
    ]

    joined: list[dict[str, Any]] = []
    missing_trace: list[str] = []
    for per_row in audio_rows:
        request_id = normalize_request_id(per_row.get("request_id"))
        trace_row = trace_chunk0.get(request_id)
        if trace_row is None:
            missing_trace.append(request_id)
            continue
        joined.append(build_lifecycle_row(per_row, trace_row))

    groups = group_rows(joined)
    metrics = summarize_metrics(joined)
    group_metrics = {name: summarize_metrics(group) for name, group in groups.items()}
    negative_counts = {
        name: sum(1 for row in joined if (value := finite(row.get(name))) is not None and value < 0.0)
        for name in (
            "inferred_stage2_submit_to_chunk0_recv_ms",
            "inferred_stage2_submit_to_chunk0_ready_ms",
        )
    }

    tail_by_total = sorted(
        joined,
        key=lambda row: finite(row.get("server_stage2_submit_to_first_output_ms"))
        or float("-inf"),
        reverse=True,
    )[:top_k]
    tail_by_ready_wait = sorted(
        joined,
        key=lambda row: finite(row.get("chunk0_ready_to_schedule_ms"))
        or float("-inf"),
        reverse=True,
    )[:top_k]

    return {
        "label": run.label,
        "per_request": str(run.per_request),
        "stage2_trace": str(run.stage2_trace),
        "per_request_rows": len(per_rows_all),
        "per_request_bad_json": per_bad,
        "stage2_trace_rows": len(trace_rows),
        "stage2_trace_bad_json": trace_bad,
        "audio_stage2_rows": len(audio_rows),
        "trace_chunk0_complete_ids": len(trace_chunk0),
        "joined_rows": len(joined),
        "missing_trace_rows": len(missing_trace),
        "missing_trace_examples": missing_trace[:20],
        "negative_counts": negative_counts,
        "metrics": metrics,
        "group_counts": {name: len(group) for name, group in groups.items()},
        "group_metrics": group_metrics,
        "tail_by_stage2_total": tail_by_total,
        "tail_by_ready_wait": tail_by_ready_wait,
        "rows": joined,
    }


def metric_table(metrics: dict[str, dict[str, Any]], names: list[str], *, ratio: bool = False) -> list[str]:
    suffix = "" if ratio else "ms"
    lines = [
        "| metric | n | p50 | p90 | p95 | p99 | max | mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        desc = metrics.get(name, {})
        lines.append(
            f"| `{name}` | {desc.get('n', 0)} | "
            f"{fmt(desc.get('p50'))}{suffix if desc.get('p50') is not None else ''} | "
            f"{fmt(desc.get('p90'))}{suffix if desc.get('p90') is not None else ''} | "
            f"{fmt(desc.get('p95'))}{suffix if desc.get('p95') is not None else ''} | "
            f"{fmt(desc.get('p99'))}{suffix if desc.get('p99') is not None else ''} | "
            f"{fmt(desc.get('max'))}{suffix if desc.get('max') is not None else ''} | "
            f"{fmt(desc.get('mean'))}{suffix if desc.get('mean') is not None else ''} |"
        )
    return lines


def tail_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| request_id | stage2 total | inferred submit->recv | recv->ready | ready->schedule | schedule->complete | audio ttfp | audio sec | chunks | mode/window |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        mode = row.get("codec_cadence_mode")
        window = fmt(row.get("codec_window_frames"), 0)
        lines.append(
            f"| `{row.get('request_id')}` | "
            f"{fmt(row.get('server_stage2_submit_to_first_output_ms'))} | "
            f"{fmt(row.get('inferred_stage2_submit_to_chunk0_recv_ms'))} | "
            f"{fmt(row.get('chunk0_recv_to_ready_ms'))} | "
            f"{fmt(row.get('chunk0_ready_to_schedule_ms'))} | "
            f"{fmt(row.get('chunk0_schedule_to_complete_ms'))} | "
            f"{fmt(row.get('audio_ttfp_ms'))} | "
            f"{fmt(row.get('audio_duration_sec'))} | "
            f"{fmt(row.get('audio_chunks'), 0)} | "
            f"{mode}/{window} |"
        )
    return lines


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = ["# Stage2 Chunk0 Lifecycle Attribution", ""]
    lines.append(
        "This report joins `per_request.jsonl` with `codec_stage2_trace.jsonl` by request id."
    )
    lines.append(
        "The `submit->recv/ready` numbers are inferred as `server_stage2_submit_to_first_output_ms` minus observed trace sub-intervals."
    )
    lines.append("")

    for run in report["runs"]:
        lines.append(f"## {run['label']}")
        lines.append("")
        lines.append(f"Per-request: `{run['per_request']}`")
        lines.append("")
        lines.append(f"Stage2 trace: `{run['stage2_trace']}`")
        lines.append("")
        lines.append(
            "Rows: "
            f"per_request={run['per_request_rows']} "
            f"audio_stage2={run['audio_stage2_rows']} "
            f"trace_chunk0_complete={run['trace_chunk0_complete_ids']} "
            f"joined={run['joined_rows']} "
            f"missing_trace={run['missing_trace_rows']} "
            f"trace_bad_json={run['stage2_trace_bad_json']}"
        )
        lines.append("")
        lines.append(f"Negative inferred counts: `{run['negative_counts']}`")
        lines.append("")
        lines.append("### Latency Split")
        lines.append("")
        lines.extend(
            metric_table(
                run["metrics"],
                [
                    "audio_ttfp_ms",
                    "server_stage2_submit_to_first_output_ms",
                    "inferred_stage2_submit_to_chunk0_recv_ms",
                    "chunk0_recv_to_ready_ms",
                    "chunk0_ready_to_schedule_ms",
                    "chunk0_schedule_to_complete_ms",
                    "chunk0_recv_to_complete_ms",
                    "inferred_stage2_submit_to_chunk0_ready_ms",
                ],
            )
        )
        lines.append("")
        lines.append("### Relative Contribution")
        lines.append("")
        lines.extend(
            metric_table(
                run["metrics"],
                [
                    "inferred_submit_to_recv_ratio",
                    "inferred_submit_to_ready_ratio",
                    "ready_to_schedule_ratio",
                    "schedule_to_complete_ratio",
                    "recv_to_complete_ratio",
                ],
                ratio=True,
            )
        )
        lines.append("")
        lines.append("### Grouped Split")
        lines.append("")
        lines.append(
            "| group | n | stage2 p90 | inferred submit->recv p90 | ready wait p90 | service p90 | submit->recv ratio p50 | recv->complete ratio p90 |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for group_name in ("all", "positive_audio", "zero_duration"):
            group = run["group_metrics"].get(group_name, {})
            stage2 = group.get("server_stage2_submit_to_first_output_ms", {})
            submit_recv = group.get("inferred_stage2_submit_to_chunk0_recv_ms", {})
            ready_wait = group.get("chunk0_ready_to_schedule_ms", {})
            service = group.get("chunk0_schedule_to_complete_ms", {})
            submit_ratio = group.get("inferred_submit_to_recv_ratio", {})
            recv_complete_ratio = group.get("recv_to_complete_ratio", {})
            lines.append(
                f"| `{group_name}` | {run['group_counts'].get(group_name, 0)} | "
                f"{fmt(stage2.get('p90'))}ms | "
                f"{fmt(submit_recv.get('p90'))}ms | "
                f"{fmt(ready_wait.get('p90'))}ms | "
                f"{fmt(service.get('p90'))}ms | "
                f"{fmt(submit_ratio.get('p50'))} | "
                f"{fmt(recv_complete_ratio.get('p90'))} |"
            )
        lines.append("")
        lines.append("### Tail By Stage2 Total")
        lines.append("")
        lines.extend(tail_table(run["tail_by_stage2_total"]))
        lines.append("")
        lines.append("### Tail By Ready Wait")
        lines.append("")
        lines.extend(tail_table(run["tail_by_ready_wait"]))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_rows_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "label",
        "request_id",
        "user_id",
        "round_index",
        "timestamp",
        "trace_entry_index",
        *METRICS,
        "codec_cadence_mode",
        "codec_window_frames",
        "codec_chunk_frames",
        "codec_target_chunk_frames",
        "codec_new_frames",
        "codec_finished",
        "audio_duration_sec",
        "audio_chunks",
        "audio_underrun_sec",
        "audio_post_ttfp_rtf",
        "talker_frames_generated",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in report["runs"]:
            for row in run["rows"]:
                writer.writerow({"label": run["label"], **{k: row.get(k) for k in fieldnames if k != "label"}})


def write_summary_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = ["label", "group", "metric", "n", "p50", "p90", "p95", "p99", "max", "mean"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in report["runs"]:
            for group_name, metrics in run["group_metrics"].items():
                for metric, desc in metrics.items():
                    writer.writerow(
                        {
                            "label": run["label"],
                            "group": group_name,
                            "metric": metric,
                            "n": desc.get("n"),
                            "p50": desc.get("p50"),
                            "p90": desc.get("p90"),
                            "p95": desc.get("p95"),
                            "p99": desc.get("p99"),
                            "max": desc.get("max"),
                            "mean": desc.get("mean"),
                        }
                    )


def parse_pair(raw: str) -> RunInput:
    parts = raw.split("::")
    if len(parts) != 3:
        raise ValueError("pair must be LABEL::PER_REQUEST::STAGE2_TRACE")
    label, per_request, stage2_trace = parts
    return RunInput(
        label=label,
        per_request=find_per_request(Path(per_request).expanduser().resolve()),
        stage2_trace=Path(stage2_trace).expanduser().resolve(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        help="LABEL::PER_REQUEST_OR_ROOT::CODEC_STAGE2_TRACE_JSONL. Repeatable.",
    )
    parser.add_argument("--label", help="single-run label")
    parser.add_argument("--per-request", type=Path, help="single-run per_request.jsonl or root")
    parser.add_argument("--stage2-trace", type=Path, help="single-run codec_stage2_trace.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs: list[RunInput] = []
    if args.pair:
        runs.extend(parse_pair(raw) for raw in args.pair)
    if args.per_request or args.stage2_trace:
        if not (args.per_request and args.stage2_trace):
            raise SystemExit("--per-request and --stage2-trace must be provided together")
        runs.append(
            RunInput(
                label=args.label or args.per_request.parent.name,
                per_request=find_per_request(args.per_request.expanduser().resolve()),
                stage2_trace=args.stage2_trace.expanduser().resolve(),
            )
        )
    if not runs:
        raise SystemExit("provide --pair or --per-request/--stage2-trace")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"runs": [summarize_run(run, top_k=max(1, args.top_k)) for run in runs]}

    json_path = args.output_dir / "stage2_chunk0_lifecycle.json"
    md_path = args.output_dir / "stage2_chunk0_lifecycle.md"
    rows_csv_path = args.output_dir / "stage2_chunk0_lifecycle_rows.csv"
    summary_csv_path = args.output_dir / "stage2_chunk0_lifecycle_summary.csv"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, md_path)
    write_rows_csv(report, rows_csv_path)
    write_summary_csv(report, summary_csv_path)

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {rows_csv_path}")
    print(f"wrote {summary_csv_path}")


if __name__ == "__main__":
    main()
