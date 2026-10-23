#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROGRESS_RE = re.compile(
    r"\[(?P<done>\d+)/(?P<total>\d+)\]\s+"
    r"elapsed=(?P<elapsed>[0-9.]+)s\s+"
    r"rps=(?P<rps>[0-9.]+)\s+"
    r"inflight=(?P<inflight>\d+)\s+"
    r"errors=(?P<errors>\d+)"
)
LOAD_RE = re.compile(
    r"Same-run returns:\s+(?P<returns>\d+).*"
    r"ratio=(?P<ratio>[0-9.]+).*"
    r"gap_p50=(?P<gap_p50>[^,\s]+).*"
    r"gap_max=(?P<gap_max>[^)\s]+)"
)
LIFECYCLE_RE = re.compile(
    r"Session lifecycle:\s+sessions=(?P<sessions>\d+)\s+"
    r"open_peak=(?P<open_peak>\d+)\s+"
    r"waiting_peak=(?P<waiting_peak>\d+)\s+"
    r"idle_pending_kv_peak_est=(?P<idle_kv>[0-9.]+)GiB"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_summary(path: Path) -> dict[str, Any]:
    data = load_json(path)
    return {
        "path": str(path),
        "ok": data.get("totals", {}).get("completed_ok"),
        "errors": data.get("totals", {}).get("errors"),
        "wall_sec": data.get("totals", {}).get("wall_time_sec"),
        "rps": data.get("totals", {}).get("throughput_rps"),
        "audio_ttfp_p50": data.get("latency", {}).get("audio_ttfp_sec", {}).get("p50"),
        "audio_ttfp_p90": data.get("latency", {}).get("audio_ttfp_sec", {}).get("p90"),
        "e2e_p50": data.get("latency", {}).get("e2e_sec", {}).get("p50"),
        "e2e_p90": data.get("latency", {}).get("e2e_sec", {}).get("p90"),
        "continuity": data.get("audio", {}).get("streaming_continuity", {}).get("ok_rate"),
        "goodput": data.get("audio", {}).get("streaming_continuity", {}).get("goodput_rps"),
    }


def summarize_log(path: Path) -> dict[str, Any]:
    last_progress: dict[str, Any] | None = None
    load_shape: dict[str, Any] = {}
    lifecycle: dict[str, Any] = {}
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = PROGRESS_RE.search(line)
        if m:
            last_progress = {
                "done": int(m.group("done")),
                "total": int(m.group("total")),
                "elapsed_sec": float(m.group("elapsed")),
                "rps": float(m.group("rps")),
                "inflight": int(m.group("inflight")),
                "errors": int(m.group("errors")),
            }
            continue
        m = LOAD_RE.search(line)
        if m:
            load_shape = {
                "same_run_returns": int(m.group("returns")),
                "same_run_return_ratio": float(m.group("ratio")),
                "same_run_gap_p50_sec": _maybe_number(m.group("gap_p50")),
                "same_run_gap_max_sec": _maybe_number(m.group("gap_max")),
            }
            continue
        m = LIFECYCLE_RE.search(line)
        if m:
            lifecycle = {
                "sessions": int(m.group("sessions")),
                "open_peak": int(m.group("open_peak")),
                "waiting_peak": int(m.group("waiting_peak")),
                "idle_pending_kv_gib": float(m.group("idle_kv")),
            }
    return {
        "path": str(path),
        "last_progress": last_progress,
        "load_shape": load_shape,
        "session_lifecycle": lifecycle,
    }


def _maybe_number(value: str) -> int | float | str | None:
    if value in {"None", "none", "null"}:
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


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


def summarize_shadow(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"path": str(path), "exists": path.exists(), "events": 0}
    reasons: Counter[str] = Counter()
    events = 0
    applied = 0
    max_cached_victims = 0
    max_live_cached = 0
    max_scan_cached = 0
    min_free_blocks: int | None = None
    max_free_blocks: int | None = None
    for row in iter_jsonl(path):
        events += 1
        reason = str(row.get("reason") or "none")
        reasons[reason] += 1
        if row.get("applied") is True or reason == "applied":
            applied += 1
        max_cached_victims = max(max_cached_victims, int(row.get("cached_victim_count") or 0))
        max_live_cached = max(
            max_live_cached,
            int(row.get("profile_index_live_cached_blocks") or 0),
        )
        max_scan_cached = max(max_scan_cached, int(row.get("profile_cached_blocks") or 0))
        free_blocks = row.get("profile_free_blocks")
        if isinstance(free_blocks, int):
            min_free_blocks = free_blocks if min_free_blocks is None else min(min_free_blocks, free_blocks)
            max_free_blocks = free_blocks if max_free_blocks is None else max(max_free_blocks, free_blocks)
    return {
        "path": str(path),
        "exists": True,
        "events": events,
        "applied": applied,
        "reasons": reasons.most_common(),
        "max_cached_victims": max_cached_victims,
        "max_live_cached": max_live_cached,
        "max_scan_cached": max_scan_cached,
        "min_free_blocks": min_free_blocks,
        "max_free_blocks": max_free_blocks,
    }


def summarize_lru(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"path": str(path), "exists": path.exists(), "records": 0}
    latest_by_pid: dict[int, dict[str, Any]] = {}
    records = 0
    for row in iter_jsonl(path):
        records += 1
        pid = row.get("pid")
        ts = row.get("ts", 0)
        if not isinstance(pid, int):
            continue
        prev = latest_by_pid.get(pid)
        if prev is None or float(ts or 0) >= float(prev.get("ts") or 0):
            latest_by_pid[pid] = row

    totals: defaultdict[str, int] = defaultdict(int)
    for row in latest_by_pid.values():
        counters = row.get("counters") or {}
        for key, value in counters.items():
            if isinstance(value, int):
                totals[key] += value
    return {
        "path": str(path),
        "exists": True,
        "records": records,
        "pids": len(latest_by_pid),
        "gpu_lru_eviction_blocks_total": totals.get("gpu_lru_eviction_blocks_total", 0),
        "gpu_prefix_hit": totals.get("gpu_prefix_block_lookup_hit", 0),
        "gpu_prefix_miss": totals.get("gpu_prefix_block_lookup_miss", 0),
        "gpu_prefix_rescued": totals.get("gpu_prefix_block_rescued_from_free_queue_total", 0),
        "gpu_prefix_touch": totals.get("gpu_prefix_block_touch_total", 0),
    }


def summarize_system_snapshot(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    loadavg: str | None = None
    load1: float | None = None
    matching_processes: list[str] = []
    in_processes = False
    for line in text.splitlines():
        if line.startswith("loadavg="):
            loadavg = line.removeprefix("loadavg=").strip()
            fields = loadavg.split()
            if fields:
                try:
                    load1 = float(fields[0])
                except ValueError:
                    load1 = None
            continue
        if line == "matching_processes=":
            in_processes = True
            continue
        if line.endswith("=") and line != "matching_processes=":
            in_processes = False
        if in_processes and line.strip():
            matching_processes.append(line.strip())
    return {
        "path": str(path),
        "loadavg": loadavg,
        "load1": load1,
        "matching_process_count": len(matching_processes),
        "matching_processes": matching_processes[:20],
    }


def summarize_run(root: Path) -> dict[str, Any]:
    summaries = [summarize_summary(path) for path in sorted(root.glob("*/*/summary.json"))]
    logs = [summarize_log(path) for path in sorted((root / "_logs").glob("*.log")) if path.name != "run.log"]
    shadows = [
        summarize_shadow(path)
        for path in sorted(root.glob("*/_probe/kv_eviction_shadow.jsonl"))
    ]
    lru = [
        summarize_lru(path)
        for path in sorted(root.glob("*/_probe/kv_lru_stats.jsonl"))
    ]
    system_snapshots = [
        summarize_system_snapshot(path)
        for path in sorted(root.glob("system_snapshot_*.txt"))
    ]
    config_path = root / "run_config.json"
    return {
        "root": str(root),
        "run_config": load_json(config_path) if config_path.exists() else None,
        "system_snapshots": system_snapshots,
        "completed_summaries": summaries,
        "partial_logs": logs,
        "kv_eviction_shadow": shadows,
        "kv_lru_stats": lru,
    }


def print_text(summary: dict[str, Any]) -> None:
    print(f"root: {summary['root']}")
    config = summary.get("run_config") or {}
    if config:
        print(
            "config: "
            f"variants={config.get('variants')} "
            f"c={config.get('concurrency_list')} "
            f"window=[{config.get('time_start')},{config.get('time_limit')}) "
            f"policy={config.get('evict_policy')} "
            f"max_requests={config.get('max_requests', 0)}"
        )
    if summary["completed_summaries"]:
        print("completed:")
        for row in summary["completed_summaries"]:
            print(
                f"- {Path(row['path']).parent.name}: ok={row['ok']} err={row['errors']} "
                f"rps={row['rps']} goodput={row['goodput']} "
                f"audio_p90={row['audio_ttfp_p90']} e2e_p90={row['e2e_p90']} "
                f"continuity={row['continuity']}"
            )
    if summary.get("system_snapshots"):
        print("system snapshots:")
        for row in summary["system_snapshots"]:
            print(
                f"- {Path(row['path']).name}: load1={row['load1']} "
                f"matching_processes={row['matching_process_count']}"
            )
    if summary["partial_logs"]:
        print("logs:")
        for row in summary["partial_logs"]:
            progress = row.get("last_progress")
            shape = row.get("load_shape") or {}
            lifecycle = row.get("session_lifecycle") or {}
            print(
                f"- {Path(row['path']).name}: progress={progress} "
                f"returns={shape.get('same_run_returns')} "
                f"return_ratio={shape.get('same_run_return_ratio')} "
                f"idle_kv_gib={lifecycle.get('idle_pending_kv_gib')}"
            )
    if summary["kv_eviction_shadow"]:
        print("kv_eviction_shadow:")
        for row in summary["kv_eviction_shadow"]:
            print(
                f"- {Path(row['path']).parent.parent.name}: events={row['events']} "
                f"applied={row.get('applied')} reasons={row.get('reasons')} "
                f"max_live_cached={row.get('max_live_cached')} "
                f"max_cached_victims={row.get('max_cached_victims')} "
                f"min_free_blocks={row.get('min_free_blocks')}"
            )
    if summary["kv_lru_stats"]:
        print("kv_lru_stats:")
        for row in summary["kv_lru_stats"]:
            print(
                f"- {Path(row['path']).parent.parent.name}: "
                f"lru_evict={row.get('gpu_lru_eviction_blocks_total')} "
                f"hit={row.get('gpu_prefix_hit')} miss={row.get('gpu_prefix_miss')} "
                f"rescued={row.get('gpu_prefix_rescued')}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = summarize_run(args.root)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
