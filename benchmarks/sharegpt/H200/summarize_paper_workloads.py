#!/usr/bin/env python3
"""Summarize ShareGPT paper workload result directories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def nested(data: dict[str, Any], path: list[str]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct_delta(value: Any, base: Any, *, lower_is_better: bool = False) -> float | None:
    if value is None or base is None:
        return None
    try:
        value_f = float(value)
        base_f = float(base)
    except (TypeError, ValueError):
        return None
    if base_f == 0:
        return None
    if lower_is_better:
        return (base_f - value_f) / base_f * 100.0
    return (value_f - base_f) / base_f * 100.0


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/**/summary.json")):
        if any(part.startswith("_") for part in summary_path.relative_to(root).parts):
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        case_dir = summary_path.parent
        variant = case_dir.parent.name
        case = case_dir.name
        totals = data.get("totals") or {}
        ok = totals.get("completed_ok")
        interrupted = totals.get("interrupted")
        interrupt_rate = None
        if ok is not None and interrupted is not None:
            try:
                interrupt_rate = float(interrupted) / float(ok) if float(ok) > 0 else None
            except (TypeError, ValueError):
                interrupt_rate = None
        rows.append(
            {
                "variant": variant,
                "case": case,
                "ok": ok,
                "errors": totals.get("errors"),
                "interrupted": interrupted,
                "interrupt_rate": interrupt_rate,
                "posthoc_playback_interrupted": totals.get("posthoc_playback_interrupted"),
                "throughput_rps": totals.get("throughput_rps"),
                "useful_qps": totals.get("useful_qps"),
                "audio_ttfp_p90_ms": nested(data, ["latency", "audio_ttfp_ms", "p90"]),
                "e2e_p90_ms": nested(data, ["latency", "e2e_ms", "p90"]),
                "audio_stream_rtf_p90": nested(data, ["audio", "audio_stream_rtf", "p90"]),
                "continuity_ok_rate": nested(data, ["audio", "streaming_continuity", "ok_rate"]),
                "waste_ratio_tokens": nested(data, ["interrupt", "waste_ratio_tokens"]),
                "waste_talker_frames_total_p90": nested(
                    data, ["interrupt", "wasted_talker_frames_total", "p90"]
                ),
                "waste_talker_frames_inaudible_p90": nested(
                    data, ["interrupt", "wasted_talker_frames_inaudible", "p90"]
                ),
                "posthoc_waste_talker_frames_total_p90": nested(
                    data, ["interrupt", "posthoc_wasted_talker_frames_total", "p90"]
                ),
                "posthoc_audio_unplayed_p90_sec": nested(
                    data, ["interrupt", "posthoc_audio_unplayed_sec", "p90"]
                ),
                "posthoc_waste_ratio_audio_p90": nested(
                    data, ["interrupt", "posthoc_waste_ratio_audio", "p90"]
                ),
                "summary": str(summary_path),
            }
        )
    return rows


def add_baseline_deltas(rows: list[dict[str, Any]], baseline_variant: str) -> None:
    baselines = {
        row["case"]: row for row in rows if row.get("variant") == baseline_variant
    }
    for row in rows:
        base = baselines.get(row["case"])
        if base is None:
            continue
        row["audio_ttfp_p90_delta_pct"] = pct_delta(
            row.get("audio_ttfp_p90_ms"),
            base.get("audio_ttfp_p90_ms"),
            lower_is_better=True,
        )
        row["e2e_p90_delta_pct"] = pct_delta(
            row.get("e2e_p90_ms"),
            base.get("e2e_p90_ms"),
            lower_is_better=True,
        )
        row["throughput_rps_delta_pct"] = pct_delta(
            row.get("throughput_rps"),
            base.get("throughput_rps"),
        )
        row["useful_qps_delta_pct"] = pct_delta(
            row.get("useful_qps"),
            base.get("useful_qps"),
        )
        cont = row.get("continuity_ok_rate")
        base_cont = base.get("continuity_ok_rate")
        try:
            row["continuity_delta"] = (
                None if cont is None or base_cont is None else float(cont) - float(base_cont)
            )
        except (TypeError, ValueError):
            row["continuity_delta"] = None


def add_validity(rows: list[dict[str, Any]], *, min_ok: int, max_errors: int) -> None:
    for row in rows:
        try:
            ok = int(row.get("ok") or 0)
            errors = int(row.get("errors") or 0)
        except (TypeError, ValueError):
            row["valid"] = False
            continue
        row["valid"] = ok >= min_ok and errors <= max_errors


def print_markdown(rows: list[dict[str, Any]]) -> None:
    has_deltas = any("audio_ttfp_p90_delta_pct" in row for row in rows)
    headers = [
        "variant",
        "case",
        "valid",
        "ok",
        "err",
        "int",
        "int.rate",
        "posthoc.int",
        "rps",
        "useful",
        "ttfp.p90 ms",
        "e2e.p90 ms",
        "rtf.p90",
        "cont",
        "waste.frames.p90",
        "posthoc.frames.p90",
    ]
    if has_deltas:
        headers.extend([
            "ttfp delta %",
            "rps delta %",
            "useful delta %",
            "e2e delta %",
            "cont delta",
        ])
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = [
            row["variant"],
            row["case"],
            "yes" if row.get("valid") else "no",
            fmt(row["ok"], 0),
            fmt(row["errors"], 0),
            fmt(row["interrupted"], 0),
            fmt(row["interrupt_rate"]),
            fmt(row["posthoc_playback_interrupted"], 0),
            fmt(row["throughput_rps"]),
            fmt(row["useful_qps"]),
            fmt(row["audio_ttfp_p90_ms"], 1),
            fmt(row["e2e_p90_ms"], 1),
            fmt(row["audio_stream_rtf_p90"]),
            fmt(row["continuity_ok_rate"]),
            fmt(row["waste_talker_frames_total_p90"], 1),
            fmt(row["posthoc_waste_talker_frames_total_p90"], 1),
        ]
        if has_deltas:
            cells.extend([
                fmt(row.get("audio_ttfp_p90_delta_pct"), 1),
                fmt(row.get("throughput_rps_delta_pct"), 1),
                fmt(row.get("useful_qps_delta_pct"), 1),
                fmt(row.get("e2e_p90_delta_pct"), 1),
                fmt(row.get("continuity_delta"), 3),
            ])
        print("| " + " | ".join(str(cell) for cell in cells) + " |")


def print_legacy_markdown(rows: list[dict[str, Any]]) -> None:
    headers = [
        "variant",
        "case",
        "ok",
        "err",
        "int",
        "rps",
        "useful",
        "ttfp.p90 ms",
        "e2e.p90 ms",
        "rtf.p90",
        "cont",
        "waste",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print(
            "| {variant} | {case} | {ok} | {errors} | {interrupted} | "
            "{throughput_rps} | {useful_qps} | {audio_ttfp_p90_ms} | "
            "{e2e_p90_ms} | {audio_stream_rtf_p90} | {continuity_ok_rate} | "
            "{waste_ratio_tokens} |".format(
                variant=row["variant"],
                case=row["case"],
                ok=fmt(row["ok"], 0),
                errors=fmt(row["errors"], 0),
                interrupted=fmt(row["interrupted"], 0),
                throughput_rps=fmt(row["throughput_rps"]),
                useful_qps=fmt(row["useful_qps"]),
                audio_ttfp_p90_ms=fmt(row["audio_ttfp_p90_ms"], 1),
                e2e_p90_ms=fmt(row["e2e_p90_ms"], 1),
                audio_stream_rtf_p90=fmt(row["audio_stream_rtf_p90"]),
                continuity_ok_rate=fmt(row["continuity_ok_rate"]),
                waste_ratio_tokens=fmt(row["waste_ratio_tokens"]),
            )
        )


def print_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "variant",
        "case",
        "valid",
        "ok",
        "errors",
        "interrupted",
        "interrupt_rate",
        "posthoc_playback_interrupted",
        "throughput_rps",
        "useful_qps",
        "audio_ttfp_p90_ms",
        "e2e_p90_ms",
        "audio_stream_rtf_p90",
        "continuity_ok_rate",
        "waste_ratio_tokens",
        "waste_talker_frames_total_p90",
        "waste_talker_frames_inaudible_p90",
        "posthoc_waste_talker_frames_total_p90",
        "posthoc_audio_unplayed_p90_sec",
        "posthoc_waste_ratio_audio_p90",
        "summary",
    ]
    delta_fields = [
        "audio_ttfp_p90_delta_pct",
        "throughput_rps_delta_pct",
        "useful_qps_delta_pct",
        "e2e_p90_delta_pct",
        "continuity_delta",
    ]
    if any(any(key in row for key in delta_fields) for row in rows):
        fieldnames.extend(delta_fields)
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument(
        "--baseline-variant",
        default=None,
        help="If set, add per-case deltas against this variant. Positive delta "
             "means improvement for latency and throughput metrics.",
    )
    parser.add_argument("--min-ok", type=int, default=1)
    parser.add_argument("--max-errors", type=int, default=0)
    parser.add_argument(
        "--legacy-markdown",
        action="store_true",
        help="Keep the original compact markdown columns.",
    )
    args = parser.parse_args()
    rows = load_rows(args.result_root)
    add_validity(rows, min_ok=args.min_ok, max_errors=args.max_errors)
    if args.baseline_variant:
        add_baseline_deltas(rows, args.baseline_variant)
    if args.format == "csv":
        print_csv(rows)
    elif args.legacy_markdown:
        print_legacy_markdown(rows)
    else:
        print_markdown(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
