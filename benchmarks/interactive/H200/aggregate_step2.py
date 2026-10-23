#!/usr/bin/env python3
"""Step-2 scheduler lead-control sweep aggregator.

Walks every  results/H200/step2/<tag>/summary.json  produced by
trace_replay_driver.py + scheduler_run_one_cell.sh, plus the no-policy
references from results/H200/step1/, and prints a
single comparison table that highlights the interaction-aware scheduler effect.

Usage:
    python3 aggregate_step2.py
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
STEP2 = REPO_ROOT / "results" / "H200" / "step2"
STEP1 = REPO_ROOT / "results" / "H200" / "step1"

def cell(root: Path, *tags: str) -> Path:
    """Return the first existing cell path, falling back to the preferred tag."""
    for tag in tags:
        p = root / tag
        if (p / "summary.json").exists():
            return p
    return root / tags[0]

# Reference rows pulled from Step 1 for context.
REFS = [
    ("no-policy       (c=8 p_bi=0.0)", cell(STEP1, "baseline_c8_p0", "audio_c8_60s_prob0"),  "—", "—"),
    ("naive-barge-in  (c=8 p_bi=1.0)", cell(STEP1, "naive_c8_p10", "audio_c8_60s_prob10"), "—", "—"),
]

# Scheduler cells in widening-then-tightening order for readability.
SCHEDULER_CELLS = [
    ("scheduler_wide  (50/25 p_bi=1.0)", cell(STEP2, "scheduler_wide_p10", "scheduler_wide"), 50, 25),
    ("scheduler_mid   (30/15 p_bi=1.0)", cell(STEP2, "scheduler_mid"),        30,      15),
    ("scheduler_tight (20/10 p_bi=1.0)", cell(STEP2, "scheduler_tight"),      20,      10),
    ("scheduler_vtight(12/6  p_bi=1.0)", cell(STEP2, "scheduler_very_tight"), 12,       6),
]

# Phase 2: prob sweep on wide(50/25) — verify useful_qps recovery.
SCHEDULER_PROB = [
    ("scheduler_wide  (50/25 p_bi=0.3)", cell(STEP2, "scheduler_wide_p03"),   50,      25),
    ("scheduler_wide  (50/25 p_bi=0.5)", cell(STEP2, "scheduler_wide_p05"),   50,      25),
    ("scheduler_wide  (50/25 p_bi=0.7)", cell(STEP2, "scheduler_wide_p07"),   50,      25),
]

def fmt_ms(x):
    return f"{x*1000:.0f}" if isinstance(x, (int, float)) else "—"

def load_row(label, path, L_hi, L_lo):
    sp = path / "summary.json"
    if not sp.exists():
        return None
    s = json.load(open(sp))
    t = s["totals"]; lat = s["latency"]; intr = s.get("interrupt", {})
    sc = s.get("sanity_check", {})
    a_ttfp = lat.get("audio_ttfp_sec") or {}
    e2e    = lat.get("e2e_sec") or {}
    waste  = intr.get("wasted_talker_frames_total") or {}
    return {
        "label":     label,
        "L_hi":      L_hi,
        "L_lo":      L_lo,
        "reqs":      t["completed_ok"],
        "interr":    t.get("interrupted", 0),
        "wall_rps":  t["throughput_rps"],
        "useful":    t.get("useful_qps", float("nan")),
        "ttfp_p50":  a_ttfp.get("p50"),
        "ttfp_p90":  a_ttfp.get("p90"),
        "e2e_p50":   e2e.get("p50"),
        "e2e_p90":   e2e.get("p90"),
        "wast_p50":  waste.get("p50"),
        "wast_p90":  waste.get("p90"),
        "wast_max":  waste.get("max"),
        "sanity":    sc.get("ratio_expected_over_actual", float("nan")),
    }

rows = []
for label, path, L_hi, L_lo in REFS + SCHEDULER_CELLS:
    r = load_row(label, path, L_hi, L_lo)
    if r is not None:
        rows.append(r)

# Phase 2 (prob sweep on wide) — only included if data exists.
phase2_rows = []
for label, path, L_hi, L_lo in SCHEDULER_PROB:
    r = load_row(label, path, L_hi, L_lo)
    if r is not None:
        phase2_rows.append(r)
rows.extend(phase2_rows)

# ---- main table ----
hdr = (
    f"{'cell':<32} {'L_hi':>5} {'L_lo':>5} {'reqs':>5} {'intr':>5} "
    f"{'rps':>6} {'useful':>7} {'aud_p50':>8} {'aud_p90':>8} "
    f"{'e2e_p50':>8} {'e2e_p90':>8} {'wast50':>7} {'wast90':>7} {'wmax':>5} {'sanity':>7}"
)
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(
        f"{r['label']:<32} {r['L_hi']:>5} {r['L_lo']:>5} {r['reqs']:>5d} {r['interr']:>5d} "
        f"{r['wall_rps']:>6.2f} {r['useful']:>7.2f} "
        f"{fmt_ms(r['ttfp_p50']):>7}ms {fmt_ms(r['ttfp_p90']):>7}ms "
        f"{fmt_ms(r['e2e_p50']):>7}ms {fmt_ms(r['e2e_p90']):>7}ms "
        f"{r['wast_p50'] or 0:>7.0f} {r['wast_p90'] or 0:>7.0f} {r['wast_max'] or 0:>5.0f} "
        f"{r['sanity']:>7.3f}"
    )

# ---- delta-vs-naive barge-in table ----
print()
print("=== Delta vs naive barge-in baseline (positive = scheduler better, except wasted/ttfp/e2e where lower is better) ===")
ref = next((r for r in rows if "naive-barge-in" in r["label"]), None)
if ref is None:
    print("  (skip: naive barge-in baseline not found)")
else:
    hdr2 = (
        f"{'cell':<32} {'Δrps':>7} {'Δuseful':>8} {'Δaud_p90':>9} "
        f"{'Δe2e_p90':>9} {'Δwast_p90':>10}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for r in rows:
        if not r["label"].startswith("scheduler"):
            continue
        def pct(a, b):
            if b in (None, 0) or a is None:
                return "—"
            return f"{(a-b)/b*100:+.0f}%"
        print(
            f"{r['label']:<32} "
            f"{pct(r['wall_rps'], ref['wall_rps']):>7} "
            f"{pct(r['useful'],   ref['useful']):>8} "
            f"{pct(r['ttfp_p90'], ref['ttfp_p90']):>9} "
            f"{pct(r['e2e_p90'],  ref['e2e_p90']):>9} "
            f"{pct(r['wast_p90'], ref['wast_p90']):>10}"
        )

# ---- Phase 2: useful_qps recovery vs OFF baseline ----
if phase2_rows:
    print()
    print("=== Phase 2: useful_qps recovery vs no-policy baseline at varying p_bi ===")
    bl = next((r for r in rows if "no-policy" in r["label"]), None)
    wide_p10 = next((r for r in rows if "scheduler_wide" in r["label"] and "p_bi=1.0" in r["label"]), None)
    if bl:
        print(f"Question: at moderate p_bi, does scheduler wide(50/25) recover useful_qps >= no-policy baseline ({bl['useful']:.2f})?")
    else:
        print("Question: at moderate p_bi, does scheduler wide(50/25) recover useful_qps >= no-policy baseline?")

    hdr3 = (
        f"{'cell':<32} {'rps':>6} {'useful':>7} {'Δuseful_vs_BL':>14} "
        f"{'aud_p90':>8} {'e2e_p90':>8} {'wast_p90':>9}"
    )
    print(hdr3)
    print("-" * len(hdr3))

    # Anchor rows: no-policy baseline (p_bi=0) and scheduler wide @ p_bi=1.0
    if bl:
        print(
            f"{'(p_bi=0.0) no-policy':<32} {bl['wall_rps']:>6.2f} {bl['useful']:>7.2f} "
            f"{'baseline':>14} "
            f"{fmt_ms(bl['ttfp_p90']):>6}ms {fmt_ms(bl['e2e_p90']):>6}ms {bl['wast_p90'] or 0:>9.0f}"
        )
    for r in phase2_rows:
        delta = ((r['useful'] - bl['useful']) / bl['useful'] * 100) if bl else float('nan')
        print(
            f"{r['label']:<32} {r['wall_rps']:>6.2f} {r['useful']:>7.2f} "
            f"{delta:>+13.0f}% "
            f"{fmt_ms(r['ttfp_p90']):>6}ms {fmt_ms(r['e2e_p90']):>6}ms {r['wast_p90'] or 0:>9.0f}"
        )
    if wide_p10:
        delta = ((wide_p10['useful'] - bl['useful']) / bl['useful'] * 100) if bl else float('nan')
        print(
            f"{'(p_bi=1.0) scheduler_wide':<32} {wide_p10['wall_rps']:>6.2f} {wide_p10['useful']:>7.2f} "
            f"{delta:>+13.0f}% "
            f"{fmt_ms(wide_p10['ttfp_p90']):>6}ms {fmt_ms(wide_p10['e2e_p90']):>6}ms {wide_p10['wast_p90'] or 0:>9.0f}"
        )

print()
print("Legend:")
print("  L_hi/L_lo      = scheduler lead thresholds (frames; 1 frame=40ms @25Hz)")
print("  rps            = wall_rps  (entries / wall_time)")
print("  useful         = useful_qps  (completed without barge-in / wall_time)")
print("  aud_p50/p90    = audio_ttfp percentiles  (server-side TTSP)")
print("  e2e_p50/p90    = end-to-end percentiles  (until done or barge-in-final chunk)")
print("  wast50/p90/max = wasted_talker_frames_total percentiles  (frames decoded after barge-in cutoff)")
print("  sanity         = expected_e2e / actual_e2e  (~1.0 = healthy server-bound)")
