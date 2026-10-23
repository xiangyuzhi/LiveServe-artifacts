#!/usr/bin/env python3
"""
diff_summary.py - Compare a fresh sweep's summary.json against a reference run.

Usage:
    python scripts/diff_summary.py \
        --runs results/H200/<run_id> \
        --reference results/reference \
        --tolerance 0.15

Behavior:
    - Recursively finds <runs>/**/<cell>/summary.json
    - For each cell that ALSO exists under <reference>/<cell>/summary.json, compares
      the headline metrics from the trace_replay_driver schema:
        totals.useful_qps         totals.throughput_rps      totals.interrupted
        latency.audio_ttfp_sec.{p50,p90}  (converted to ms)
        latency.e2e_sec.{p50,p90}         (converted to ms)
        interrupt.wasted_talker_frames_total.{p50,p90}    (barge-in waste frames; missing -> 0 for OFF cells)
    - Reports each metric with absolute delta, relative delta, and PASS/WARN flag
      based on --tolerance (fraction; default 0.15 = +/-15%).
    - Exit 0 if all compared metrics pass; non-zero if any WARN.

Cells in --runs but not --reference, or vice versa, are listed but not failed.
Designed as a portable cross-host validation step for current H200 reproduction
runs against a frozen reference directory.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

# Each entry: (display_name, json_path, scale, fmt, missing_default)
#   scale: multiplier (e.g. 1000.0 to convert seconds to milliseconds)
#   fmt:   format spec for the value column
#   missing_default: value to use when the path is absent (None = skip metric)
_METRICS = [
    # (name,                       path,                                            scale,   fmt,      missing_default)
    ("useful_qps",                 "totals.useful_qps",                              1.0,     "{:.4f}", None),
    ("wall_rps",                   "totals.throughput_rps",                          1.0,     "{:.4f}", None),
    ("audio_ttfp_p50_ms",          "latency.audio_ttfp_sec.p50",                     1000.0,  "{:.1f}", None),
    ("audio_ttfp_p90_ms",          "latency.audio_ttfp_sec.p90",                     1000.0,  "{:.1f}", None),
    ("audio_underrun_p90_ms",      "audio.streaming_continuity.audio_underrun_sec.p90", 1000.0, "{:.1f}", None),
    ("audio_underrun_p99_ms",      "audio.streaming_continuity.audio_underrun_sec.p99", 1000.0, "{:.1f}", None),
    ("audio_continuity_ok_rate",   "audio.streaming_continuity.ok_rate",             1.0,     "{:.4f}", None),
    ("audio_continuity_goodput",   "audio.streaming_continuity.goodput_rps",         1.0,     "{:.4f}", None),
    ("e2e_p50_ms",                 "latency.e2e_sec.p50",                            1000.0,  "{:.1f}", None),
    ("e2e_p90_ms",                 "latency.e2e_sec.p90",                            1000.0,  "{:.1f}", None),
    ("wasted_talker_frames_p50",   "interrupt.wasted_talker_frames_total.p50",       1.0,     "{:.0f}", 0.0),
    ("wasted_talker_frames_p90",   "interrupt.wasted_talker_frames_total.p90",       1.0,     "{:.0f}", 0.0),
    ("interrupted",                "totals.interrupted",                             1.0,     "{:.0f}", 0.0),
    ("wall_time_sec",              "totals.wall_time_sec",                           1.0,     "{:.1f}", None),
]


def _dotget(node: Any, path: str) -> Optional[float]:
    """Walk dot-separated path through nested dicts; return float or None if any segment missing."""
    cur = node
    for seg in path.split("."):
        if not isinstance(cur, dict) or seg not in cur:
            return None
        cur = cur[seg]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def _load(p: Path) -> Optional[dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] could not parse {p}: {e}", file=sys.stderr)
        return None


def _find_cells(root: Path) -> dict[str, Path]:
    """Return {cell_tag: summary.json path} for every <cell>/summary.json under root.
    cell_tag is the parent directory name (so step{1,2}/baseline_c8_p0/summary.json -> 'baseline_c8_p0').
    """
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for sj in root.rglob("summary.json"):
        out[sj.parent.name] = sj
    return out


def _delta(new: float, ref: float) -> tuple[float, float]:
    """Return (abs_delta, rel_delta). rel is signed (new-ref)/|ref|. NaN if ref==0."""
    abs_d = new - ref
    if ref == 0:
        return abs_d, math.nan
    return abs_d, abs_d / abs(ref)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--runs", required=True, type=Path, help="Dir containing fresh sweep <cell>/summary.json files")
    ap.add_argument("--reference", required=True, type=Path, help="Dir containing reference <cell>/summary.json files")
    ap.add_argument("--tolerance", type=float, default=0.15, help="Fraction tolerance, e.g. 0.15 = +/-15%%")
    args = ap.parse_args()

    runs = _find_cells(args.runs)
    refs = _find_cells(args.reference)

    common = sorted(set(runs) & set(refs))
    only_runs = sorted(set(runs) - set(refs))
    only_refs = sorted(set(refs) - set(runs))

    if only_runs:
        print(f"[info] cells present in --runs but not --reference (no comparison): {only_runs}")
    if only_refs:
        print(f"[info] cells present in --reference but not --runs (not yet reproduced): {only_refs}")

    if not common:
        print("[fatal] no overlapping cells between --runs and --reference; nothing to compare.", file=sys.stderr)
        return 2

    any_warn = False
    print(f"\n# diff_summary - tolerance +/-{args.tolerance * 100:.0f}%\n")

    for tag in common:
        new = _load(runs[tag])
        ref = _load(refs[tag])
        if new is None or ref is None:
            continue
        print(f"## {tag}")
        print(f"|  metric                       |        run |        ref |   delta | delta% | flag    |")
        print(f"|-------------------------------|-----------:|-----------:|--------:|------:|:--------|")
        for name, path, scale, fmt, miss_default in _METRICS:
            a_raw = _dotget(new, path)
            b_raw = _dotget(ref, path)
            if a_raw is None and miss_default is not None:
                a_raw = miss_default
            if b_raw is None and miss_default is not None:
                b_raw = miss_default
            if a_raw is None or b_raw is None:
                continue
            a = a_raw * scale
            b = b_raw * scale
            abs_d, rel_d = _delta(a, b)
            if math.isnan(rel_d):
                # ref==0 (e.g. no barge-in baseline). Pass iff new is also 0 (or near-0).
                flag = "PASS" if a == 0 else "WARN"
                rel_str = "n/a"
                if flag == "WARN":
                    any_warn = True
            elif abs(rel_d) <= args.tolerance:
                flag = "PASS"
                rel_str = f"{rel_d * 100:+.1f}%"
            else:
                flag = "WARN"
                rel_str = f"{rel_d * 100:+.1f}%"
                any_warn = True
            print(f"| {name:<29} | {fmt.format(a):>10} | {fmt.format(b):>10} | {abs_d:+8.3g} | {rel_str:>5} | {flag} |")
        print()

    return 1 if any_warn else 0


if __name__ == "__main__":
    sys.exit(main())
