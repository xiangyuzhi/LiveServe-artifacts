#!/usr/bin/env python3
"""Step-1 (audio-mode multi-turn baseline) aggregator.

Walks every  results/H200/step1/<tag>/summary.json  produced by
trace_replay_driver.py and prints a single comparison table with
TTFP / audio_TTFP / E2E / sanity-check / coverage metrics.

Usage:
    python3 aggregate_step1.py [<root>]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(__file__).resolve().parents[3] / "results" / "H200" / "step1"

cols = (
    f"{'tag':<28} {'c':>3} {'win_s':>5} {'aud':>3} {'reqs':>5} "
    f"{'rps':>6} {'ttfp_p50':>9} {'aud_p50':>9} {'aud_p90':>9} "
    f"{'e2e_p50':>9} {'e2e_p90':>9} {'cov':>6} {'sanity':>11}"
)
print(cols)
print("-" * len(cols))

rows = []
for sub in sorted(ROOT.iterdir()):
    sp = sub / "summary.json"
    mp = sub / "meta.json"
    if not sp.exists():
        continue
    s = json.load(open(sp))
    m = json.load(open(mp)) if mp.exists() else {}
    args = s.get("args", {})
    lat = s.get("latency", {})
    audio = s.get("audio", {})
    sc = s.get("sanity_check", {})
    win = (m.get("trace_time_range_sec") or
           ((args.get("time_limit") or 0) - (m.get("args", {}).get("time_start") or 0)))
    aud = "Y" if args.get("with_audio") else "N"
    reqs = s["totals"]["completed_ok"]
    n_aud = audio.get("n_with_audio", 0)
    cov = (n_aud / reqs) if reqs else 0
    ttfp = lat.get("ttfp_sec") or {}
    a_ttfp = lat.get("audio_ttfp_sec") or {}
    e2e = lat.get("e2e_sec") or {}

    def ms(x):
        return f"{x*1000:.0f}ms" if isinstance(x, (int, float)) else "—"

    sanity_str = (
        f"r={sc['ratio_expected_over_actual']:.2f}/{'OK' if sc.get('within_tolerance') else 'WRN'}"
        if sc else "—"
    )
    print(
        f"{sub.name:<28} {args.get('max_inflight'):>3d} {int(win or 0):>5d} "
        f"{aud:>3} {reqs:>5d} {s['totals']['throughput_rps']:>6.2f} "
        f"{ms(ttfp.get('p50')):>9} {ms(a_ttfp.get('p50')):>9} {ms(a_ttfp.get('p90')):>9} "
        f"{ms(e2e.get('p50')):>9} {ms(e2e.get('p90')):>9} "
        f"{cov*100:>5.1f}% {sanity_str:>11}"
    )
    rows.append((sub.name, s, m))

print()
print("Legend:")
print("  cov     = fraction of OK requests with audio_ttfp recorded "
      "(should be ~100% with the fixed SSE consumer)")
print("  sanity  = expected_e2e/actual_e2e where expected = wall*c/n. "
      "OK if 0.7 <= ratio <= 1.3 (server-bound healthy)")
