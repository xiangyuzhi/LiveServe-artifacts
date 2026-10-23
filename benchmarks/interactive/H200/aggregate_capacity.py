#!/usr/bin/env python3
"""Aggregate H200/L20X capacity sweep cells.

Usage:
    python aggregate_capacity.py results/H200/capacity2gpu/<run_id>
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def pct(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    xs = sorted(values)
    n = len(xs)
    return {
        "p50": xs[n // 2],
        "p90": xs[int(n * 0.90)],
        "p95": xs[int(n * 0.95)],
        "p99": xs[int(n * 0.99)],
        "mean": statistics.mean(xs),
    }


def get_path(node: dict, path: str, default=None):
    cur = node
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def ms(value):
    return f"{value * 1000:.0f}" if isinstance(value, (int, float)) else "-"


def num(value, digits=2):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def load_rtf_proxy(path: Path) -> dict[str, float]:
    per_req = path / "per_request.jsonl"
    if not per_req.exists():
        return {}
    values = []
    with per_req.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") is not None or row.get("interrupted"):
                continue
            e2e = row.get("e2e_sec")
            audio_dur = row.get("audio_duration_sec")
            if isinstance(e2e, (int, float)) and isinstance(audio_dur, (int, float)) and audio_dur > 0:
                values.append(e2e / audio_dur)
    return pct(values)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: aggregate_capacity.py <result_root>", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).resolve()
    if not root.exists():
        print(f"missing result root: {root}", file=sys.stderr)
        return 2

    rows = []
    for summary_path in root.rglob("summary.json"):
        cell_dir = summary_path.parent
        with summary_path.open() as f:
            summary = json.load(f)
        args = summary.get("args", {})
        totals = summary.get("totals", {})
        latency = summary.get("latency", {})
        sanity = summary.get("sanity_check", {})
        rtf_proxy = load_rtf_proxy(cell_dir)
        rows.append({
            "tag": cell_dir.name,
            "c": args.get("max_inflight", -1),
            "reqs": totals.get("completed_ok", 0),
            "err": totals.get("errors", 0),
            "wall": totals.get("wall_time_sec", 0),
            "rps": totals.get("throughput_rps", 0),
            "useful": totals.get("useful_qps", 0),
            "audio_p50": get_path(latency, "audio_ttfp_sec.p50"),
            "audio_p90": get_path(latency, "audio_ttfp_sec.p90"),
            "e2e_p50": get_path(latency, "e2e_sec.p50"),
            "e2e_p90": get_path(latency, "e2e_sec.p90"),
            "rtf_p50": rtf_proxy.get("p50"),
            "rtf_p90": rtf_proxy.get("p90"),
            "sanity": sanity.get("ratio_expected_over_actual"),
            "sanity_ok": sanity.get("within_tolerance"),
        })

    rows.sort(key=lambda r: (r["c"], r["tag"]))

    header = (
        f"{'tag':<34} {'c':>4} {'ok':>5} {'err':>3} {'wall_s':>7} "
        f"{'rps':>6} {'useful':>7} {'aud50':>7} {'aud90':>7} "
        f"{'e2e50':>7} {'e2e90':>7} {'rtf50':>7} {'rtf90':>7} {'sanity':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        sanity = "-"
        if isinstance(r["sanity"], (int, float)):
            sanity = f"{r['sanity']:.2f}/{'OK' if r['sanity_ok'] else 'WRN'}"
        print(
            f"{r['tag']:<34} {r['c']:>4} {r['reqs']:>5} {r['err']:>3} "
            f"{r['wall']:>7.1f} {r['rps']:>6.2f} {r['useful']:>7.2f} "
            f"{ms(r['audio_p50']):>7} {ms(r['audio_p90']):>7} "
            f"{ms(r['e2e_p50']):>7} {ms(r['e2e_p90']):>7} "
            f"{num(r['rtf_p50']):>7} {num(r['rtf_p90']):>7} {sanity:>8}"
        )

    print()
    print("Notes:")
    print("  rtf50/rtf90 are e2e/audio_duration proxies for non-interrupted OK requests.")
    print("  They include startup/wait time, so use them as cliff indicators, not exact talker RTF.")
    print("  p_int=0 means useful_qps should equal throughput_rps unless errors appear.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
