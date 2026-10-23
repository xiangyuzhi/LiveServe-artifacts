#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSpec:
    label: str
    root: Path


def parse_run(raw: str) -> RunSpec:
    if "=" not in raw:
        path = Path(raw)
        return RunSpec(path.name, path)
    label, path = raw.split("=", 1)
    return RunSpec(label.strip(), Path(path.strip()))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[idx]


def describe(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "mean": sum(values) / len(values) if values else None,
        "max": max(values) if values else None,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def find_one(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_system_snapshots(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("system_snapshot_*.txt")):
        load1 = None
        process_count = 0
        in_processes = False
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("loadavg="):
                fields = line.removeprefix("loadavg=").split()
                if fields:
                    try:
                        load1 = float(fields[0])
                    except ValueError:
                        load1 = None
            elif line == "matching_processes=":
                in_processes = True
            elif line.endswith("="):
                in_processes = False
            elif in_processes and line.strip():
                process_count += 1
        rows.append({
            "name": path.name,
            "load1": load1,
            "matching_process_count": process_count,
        })
    return rows


def summarize_run(spec: RunSpec, slow_threshold_sec: float) -> dict[str, Any]:
    summary_path = find_one(spec.root, "*/*/summary.json")
    request_path = find_one(spec.root, "*/*/per_request.jsonl")
    shadow_path = find_one(spec.root, "*/_probe/kv_eviction_shadow.jsonl")
    summary = load_json(summary_path)
    request_rows = read_jsonl(request_path) if request_path else []
    ok_rows = [row for row in request_rows if row.get("error") is None]
    e2e = [
        float(row["e2e_sec"]) for row in ok_rows
        if isinstance(row.get("e2e_sec"), (int, float))
    ]
    audio = [
        float(row["audio_ttfp_sec"]) for row in ok_rows
        if isinstance(row.get("audio_ttfp_sec"), (int, float))
    ]
    continuity = [
        bool(row.get("audio_continuity_ok")) for row in ok_rows
        if row.get("audio_continuity_evaluated") is True
    ]
    reasons: dict[str, int] = {}
    if shadow_path and shadow_path.exists():
        for row in read_jsonl(shadow_path):
            reason = str(row.get("reason") or "none")
            reasons[reason] = reasons.get(reason, 0) + 1
    kv = summary.get("kv_lru", {}) if isinstance(summary, dict) else {}
    return {
        "label": spec.label,
        "root": str(spec.root),
        "summary_path": str(summary_path) if summary_path else None,
        "request_path": str(request_path) if request_path else None,
        "requests": len(ok_rows),
        "errors": len(request_rows) - len(ok_rows),
        "throughput_rps": summary.get("totals", {}).get("throughput_rps"),
        "goodput_rps": summary.get("audio", {}).get("streaming_continuity", {}).get("goodput_rps"),
        "summary_audio_p90": summary.get("latency", {}).get("audio_ttfp_sec", {}).get("p90"),
        "summary_e2e_p90": summary.get("latency", {}).get("e2e_sec", {}).get("p90"),
        "per_request_audio_ttfp_sec": describe(audio),
        "per_request_e2e_sec": describe(e2e),
        "slow_e2e_rate": (
            sum(value > slow_threshold_sec for value in e2e) / len(e2e)
            if e2e else None
        ),
        "continuity_ok_rate": (
            sum(continuity) / len(continuity) if continuity else None
        ),
        "kv_evictions": kv.get("gpu_evictions"),
        "kv_lookup_misses": kv.get("gpu_lookup_misses"),
        "kv_recent_eviction_misses": kv.get("recent_eviction_misses"),
        "shadow_reasons": sorted(reasons.items()),
        "system_snapshots": summarize_system_snapshots(spec.root),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]], slow_threshold_sec: float) -> None:
    lines = [
        "# Eviction Run Stability",
        "",
        f"- slow_e2e_threshold_sec: {slow_threshold_sec}",
        "",
        "| run | requests | rps | goodput | audio_p90 | e2e_p50 | e2e_p90 | e2e_p99 | slow_rate | continuity | evict | recent_miss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        e2e = row["per_request_e2e_sec"]
        lines.append(
            f"| {row['label']} | {row['requests']} | "
            f"{fmt(row['throughput_rps'])} | {fmt(row['goodput_rps'])} | "
            f"{fmt(row['summary_audio_p90'])} | {fmt(e2e['p50'])} | "
            f"{fmt(e2e['p90'])} | {fmt(e2e['p99'])} | "
            f"{fmt(row['slow_e2e_rate'])} | {fmt(row['continuity_ok_rate'])} | "
            f"{fmt(row['kv_evictions'])} | {fmt(row['kv_recent_eviction_misses'])} |"
        )
    lines.extend(["", "## System Snapshots", ""])
    for row in rows:
        lines.append(f"### {row['label']}")
        if not row["system_snapshots"]:
            lines.append("")
            lines.append("- no system snapshots recorded")
            lines.append("")
            continue
        for snap in row["system_snapshots"]:
            lines.append(
                f"- {snap['name']}: load1={fmt(snap['load1'])}, "
                f"matching_processes={snap['matching_process_count']}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slow-threshold-sec", type=float, default=10.0)
    args = parser.parse_args()

    rows = [
        summarize_run(parse_run(raw), args.slow_threshold_sec)
        for raw in args.run
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "eviction_run_stability.json"
    md_path = args.output_dir / "eviction_run_stability.md"
    json_path.write_text(json.dumps({"runs": rows}, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, rows, args.slow_threshold_sec)
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
