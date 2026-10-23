#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(math.ceil(q * len(values))) - 1))
    return values[idx]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def find_one(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def parse_system_snapshot(path: Path) -> dict[str, Any]:
    load1: float | None = None
    matching_processes: list[str] = []
    in_processes = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("loadavg="):
            fields = line.removeprefix("loadavg=").split()
            if fields:
                try:
                    load1 = float(fields[0])
                except ValueError:
                    load1 = None
            continue
        if line == "matching_processes=":
            in_processes = True
            continue
        if line.endswith("="):
            in_processes = False
            continue
        if in_processes and line.strip():
            stripped = line.strip()
            if not is_self_quality_process(stripped):
                matching_processes.append(stripped)
    return {
        "path": str(path),
        "name": path.name,
        "load1": load1,
        "matching_process_count": len(matching_processes),
        "matching_processes": matching_processes[:20],
    }


def is_self_quality_process(line: str) -> bool:
    return any(
        token in line
        for token in (
            "run_original_baseline_vs_eviction_gpu45.sh",
            "run_original_baseline_vs_eviction_gpu67.sh",
            "run_original_baseline_vs_eviction_gpuall8.sh",
            "run_eviction_fast_iter_gpu67.sh",
            "run_eviction_fast_iter_gpuall8.sh",
            "pgrep -af",
        )
    )


def summarize_run(root: Path, slow_threshold_sec: float) -> dict[str, Any]:
    summary_path = find_one(root, "*/*/summary.json")
    request_path = find_one(root, "*/*/per_request.jsonl")
    if summary_path is None:
        raise FileNotFoundError(f"no summary.json under {root}")
    if request_path is None:
        raise FileNotFoundError(f"no per_request.jsonl under {root}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_jsonl(request_path)
    ok_rows = [row for row in rows if row.get("error") is None]
    e2e_values = [
        float(row["e2e_sec"]) for row in ok_rows
        if isinstance(row.get("e2e_sec"), (int, float))
    ]
    audio_values = [
        float(row["audio_ttfp_sec"]) for row in ok_rows
        if isinstance(row.get("audio_ttfp_sec"), (int, float))
    ]
    continuity_values = [
        bool(row.get("audio_continuity_ok")) for row in ok_rows
        if row.get("audio_continuity_evaluated") is True
    ]
    snapshots = [
        parse_system_snapshot(path)
        for path in sorted(root.glob("system_snapshot_*.txt"))
    ]
    return {
        "root": str(root),
        "summary_path": str(summary_path),
        "request_path": str(request_path),
        "requests": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "throughput_rps": summary.get("totals", {}).get("throughput_rps"),
        "goodput_rps": summary.get("audio", {}).get("streaming_continuity", {}).get("goodput_rps"),
        "summary_audio_p90": summary.get("latency", {}).get("audio_ttfp_sec", {}).get("p90"),
        "summary_e2e_p90": summary.get("latency", {}).get("e2e_sec", {}).get("p90"),
        "e2e_p50": percentile(e2e_values, 0.50),
        "e2e_p90": percentile(e2e_values, 0.90),
        "e2e_p99": percentile(e2e_values, 0.99),
        "audio_p90": percentile(audio_values, 0.90),
        "slow_e2e_rate": (
            sum(value > slow_threshold_sec for value in e2e_values) / len(e2e_values)
            if e2e_values else None
        ),
        "continuity_ok_rate": (
            sum(continuity_values) / len(continuity_values)
            if continuity_values else None
        ),
        "system_snapshots": snapshots,
    }


def reject_if(
    failures: list[str],
    condition: bool,
    message: str,
) -> None:
    if condition:
        failures.append(message)


def evaluate(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    max_load1 = max(
        (
            snap["load1"] for snap in summary["system_snapshots"]
            if isinstance(snap.get("load1"), (int, float))
        ),
        default=None,
    )
    max_matching_processes = max(
        (int(snap.get("matching_process_count") or 0) for snap in summary["system_snapshots"]),
        default=0,
    )
    reject_if(
        failures,
        summary["requests"] < args.min_requests,
        f"requests {summary['requests']} < {args.min_requests}",
    )
    reject_if(
        failures,
        (summary["e2e_p90"] or float("inf")) > args.max_e2e_p90,
        f"e2e_p90 {summary['e2e_p90']} > {args.max_e2e_p90}",
    )
    reject_if(
        failures,
        (summary["audio_p90"] or float("inf")) > args.max_audio_p90,
        f"audio_p90 {summary['audio_p90']} > {args.max_audio_p90}",
    )
    reject_if(
        failures,
        (summary["slow_e2e_rate"] or 0.0) > args.max_slow_rate,
        f"slow_e2e_rate {summary['slow_e2e_rate']} > {args.max_slow_rate}",
    )
    reject_if(
        failures,
        (summary["continuity_ok_rate"] or 0.0) < args.min_continuity,
        f"continuity {summary['continuity_ok_rate']} < {args.min_continuity}",
    )
    if args.max_snapshot_load1 is not None and max_load1 is not None:
        reject_if(
            failures,
            max_load1 > args.max_snapshot_load1,
            f"max snapshot load1 {max_load1} > {args.max_snapshot_load1}",
        )
    if args.max_matching_processes is not None:
        reject_if(
            failures,
            max_matching_processes > args.max_matching_processes,
            (
                f"max matching process count {max_matching_processes} > "
                f"{args.max_matching_processes}"
            ),
        )
    return {
        "passed": not failures,
        "failures": failures,
        "max_snapshot_load1": max_load1,
        "max_matching_processes": max_matching_processes,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(
    path: Path,
    summary: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    lines = [
        "# Eviction Sanity Evaluation",
        "",
        f"- root: `{summary['root']}`",
        f"- decision: `{'PASS' if evaluation['passed'] else 'REJECT'}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| requests | {summary['requests']} |",
        f"| errors | {summary['errors']} |",
        f"| rps | {fmt(summary['throughput_rps'])} |",
        f"| goodput | {fmt(summary['goodput_rps'])} |",
        f"| audio_p90 | {fmt(summary['audio_p90'])} |",
        f"| e2e_p50 | {fmt(summary['e2e_p50'])} |",
        f"| e2e_p90 | {fmt(summary['e2e_p90'])} |",
        f"| e2e_p99 | {fmt(summary['e2e_p99'])} |",
        f"| slow_e2e_rate | {fmt(summary['slow_e2e_rate'])} |",
        f"| continuity | {fmt(summary['continuity_ok_rate'])} |",
        f"| max_snapshot_load1 | {fmt(evaluation['max_snapshot_load1'])} |",
        f"| max_matching_processes | {evaluation['max_matching_processes']} |",
    ]
    if evaluation["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in evaluation["failures"]:
            lines.append(f"- {failure}")
    if summary["system_snapshots"]:
        lines.extend(["", "## System Snapshots", ""])
        for snap in summary["system_snapshots"]:
            lines.append(
                f"- {snap['name']}: load1={fmt(snap['load1'])}, "
                f"matching_processes={snap['matching_process_count']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slow-threshold-sec", type=float, default=10.0)
    parser.add_argument("--min-requests", type=int, default=600)
    parser.add_argument("--max-e2e-p90", type=float, default=3.0)
    parser.add_argument("--max-audio-p90", type=float, default=3.0)
    parser.add_argument("--max-slow-rate", type=float, default=0.02)
    parser.add_argument("--min-continuity", type=float, default=0.99)
    parser.add_argument("--max-snapshot-load1", type=float, default=8.0)
    parser.add_argument("--max-matching-processes", type=int, default=0)
    parser.add_argument("--fail-on-reject", action="store_true")
    args = parser.parse_args()

    summary = summarize_run(args.run_root, args.slow_threshold_sec)
    evaluation = evaluate(summary, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "eviction_sanity_evaluation.json"
    md_path = args.output_dir / "eviction_sanity_evaluation.md"
    json_path.write_text(
        json.dumps({"summary": summary, "evaluation": evaluation}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(md_path, summary, evaluation)
    print(json_path)
    print(md_path)
    if args.fail_on_reject and not evaluation["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
