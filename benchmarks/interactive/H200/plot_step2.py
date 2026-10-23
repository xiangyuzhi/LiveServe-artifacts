#!/usr/bin/env python3
"""plot_step2.py — Step 2 scheduler sweep visualization

Reads results/H200/step2 scheduler summaries plus Step 1 no-policy baselines,
and writes figures to analysis/figures/step2/.

  Fig 1 — L_hi vs throughput trio:
    wall_rps / useful_qps / barge-in rate vs L_hi (frames),
    with no-policy baseline + naive barge-in horizontal references.
    Answers: "is there a useful_qps cliff as L_hi tightens?"

  Fig 2 — L_hi vs latency trio:
    audio_ttfp p50 / audio_ttfp p90 / E2E p90 vs L_hi.
    Answers: "what is the latency cost of relaxing L_hi?"

  Fig 3 — L_hi vs wasted_talker_frames:
    wasted_p50 / wasted_p90 / wasted_max vs L_hi (log scale on Y),
    with naive barge-in p90=206 reference + ideal_line (L_hi*1.2+12).
    Answers: "does waste track L_hi linearly?"

  Fig 4 — Pareto: wall_rps  vs  wasted_p90  scatter+annotate:
    All scheduler cells + no-policy baselines, with quadrant labels.
    Answers: "what is the optimal operating point?"

Run:  python3 plot_step2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
S2 = REPO_ROOT / "results" / "H200" / "step2"
S1 = REPO_ROOT / "results" / "H200" / "step1"
OUT = ROOT / "analysis" / "figures" / "step2"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

C_SCHEDULER = "#4C78A8"
C_OFF_BL = "#54A24B"   # no-policy baseline healthy reference
C_OFF_NAIVE = "#E45756" # naive barge-in pathology
C_AUX = "#B279A2"      # purple (auxiliary)
C_GOLD = "#F58518"     # gold for ideal line

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
SCHEDULER_CELLS = [
    # (tag aliases, L_hi, L_lo, label)
    (("scheduler_very_tight",), 12, 6, "very_tight\n(12/6)"),
    (("scheduler_tight",), 20, 10, "tight\n(20/10)"),
    (("scheduler_mid",), 30, 15, "mid\n(30/15)"),
    (("scheduler_wide_p10", "scheduler_wide"), 50, 25, "wide\n(50/25)"),
]

def load(p: Path) -> dict | None:
    sp = p / "summary.json"
    if not sp.exists():
        print(f"[WARN] missing {sp}")
        return None
    return json.load(open(sp))

def load_cell(base: Path, *tags: str) -> dict | None:
    for tag in tags:
        s = load(base / tag)
        if s is not None:
            return s
    return None

def extract(s: dict) -> dict:
    t = s["totals"]; lat = s["latency"]; intr = s.get("interrupt", {})
    a_ttfp = lat.get("audio_ttfp_sec") or {}
    e2e    = lat.get("e2e_sec") or {}
    waste  = intr.get("wasted_talker_frames_total") or {}
    return {
        "wall_rps":  t["throughput_rps"],
        "useful":   t.get("useful_qps", float("nan")),
        "reqs":     t["completed_ok"],
        "interr":   t.get("interrupted", 0),
        "intr_rate": (t.get("interrupted", 0) / t["completed_ok"]) if t["completed_ok"] else 0.0,
        "ttfp_p50": (a_ttfp.get("p50") or 0) * 1000,  # ms
        "ttfp_p90": (a_ttfp.get("p90") or 0) * 1000,
        "e2e_p50":  (e2e.get("p50") or 0) * 1000,
        "e2e_p90":  (e2e.get("p90") or 0) * 1000,
        "wast_p50": waste.get("p50") or 0,
        "wast_p90": waste.get("p90") or 0,
        "wast_max": waste.get("max") or 0,
    }

# Load scheduler cells, sorted by L_hi ascending for axis order
data = []
for tags, L_hi, L_lo, label in SCHEDULER_CELLS:
    s = load_cell(S2, *tags)
    if s is None:
        continue
    d = extract(s)
    d.update({"tag": tags[0], "L_hi": L_hi, "L_lo": L_lo, "label": label})
    data.append(d)
data.sort(key=lambda d: d["L_hi"])
L_HIS = [d["L_hi"] for d in data]
LABELS = [d["label"] for d in data]

# References (Step 1)
ref_baseline = extract(load_cell(S1, "baseline_c8_p0", "audio_c8_60s_prob0"))   # c=8 prob=0
ref_naive    = extract(load_cell(S1, "naive_c8_p10", "audio_c8_60s_prob10"))  # c=8 prob=1.0

print(f"[load] {len(data)} scheduler cells: {[d['L_hi'] for d in data]}")
print(f"[load] no-policy baseline: rps={ref_baseline['wall_rps']:.2f} useful={ref_baseline['useful']:.2f}")
print(f"[load] naive barge-in    : rps={ref_naive['wall_rps']:.2f} useful={ref_naive['useful']:.2f} "
      f"wast_p90={ref_naive['wast_p90']:.0f}")

# ---------------------------------------------------------------------------
# Fig 1 — Throughput trio
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.0))

x = L_HIS
ax.plot(x, [d["wall_rps"] for d in data], "o-", color=C_SCHEDULER, lw=2, ms=8, label="wall_rps")
ax.plot(x, [d["useful"]   for d in data], "s-", color=C_AUX,     lw=2, ms=8, label="useful_qps")

# no-policy references
ax.axhline(ref_baseline["wall_rps"], color=C_OFF_BL,   ls="--", lw=1.4,
           label=f"no-policy baseline rps={ref_baseline['wall_rps']:.2f}")
ax.axhline(ref_naive["wall_rps"],    color=C_OFF_NAIVE, ls="--", lw=1.4,
           label=f"naive barge-in rps={ref_naive['wall_rps']:.2f}")
ax.axhline(ref_naive["useful"],      color=C_OFF_NAIVE, ls=":",  lw=1.4,
           label=f"naive barge-in useful={ref_naive['useful']:.2f}")

# Cell labels above wall_rps points
for d in data:
    ax.annotate(d["label"].replace("\n", " "),
                xy=(d["L_hi"], d["wall_rps"]), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=9, color=C_SCHEDULER)

ax.set_xlabel("L_hi (talker_lead_max, frames @25Hz; 1f=40ms)")
ax.set_ylabel("requests / sec")
ax.set_title("Fig 1 · scheduler throughput sweep (c=8, p_bi=1.0, audio)")
ax.set_xticks(L_HIS)
ax.legend(loc="lower right", framealpha=0.95)
ax.set_ylim(0, max(d["wall_rps"] for d in data) * 1.25)
plt.tight_layout()
plt.savefig(OUT / "fig1_throughput_vs_lhi.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2 — Latency trio
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.0))
ax.plot(x, [d["ttfp_p50"] for d in data], "o-", color=C_SCHEDULER, lw=2, ms=8, label="audio_ttfp p50")
ax.plot(x, [d["ttfp_p90"] for d in data], "s-", color=C_GOLD,  lw=2, ms=8, label="audio_ttfp p90")
ax.plot(x, [d["e2e_p90"]  for d in data], "^-", color=C_AUX,   lw=2, ms=8, label="E2E p90")

ax.axhline(ref_naive["ttfp_p90"], color=C_OFF_NAIVE, ls="--", lw=1.4,
           label=f"naive barge-in ttfp_p90={ref_naive['ttfp_p90']:.0f}ms")
ax.axhline(ref_naive["e2e_p90"], color=C_OFF_NAIVE, ls=":", lw=1.4,
           label=f"naive barge-in e2e_p90={ref_naive['e2e_p90']:.0f}ms")

for d in data:
    ax.annotate(d["label"].replace("\n", " "),
                xy=(d["L_hi"], d["ttfp_p50"]), xytext=(0, -16),
                textcoords="offset points", ha="center", fontsize=9, color=C_SCHEDULER)

ax.set_xlabel("L_hi (frames)")
ax.set_ylabel("latency (ms)")
ax.set_title("Fig 2 · scheduler latency sweep - TTFP / E2E vs L_hi")
ax.set_xticks(L_HIS)
ax.legend(loc="upper left", framealpha=0.95)
ax.set_ylim(0, max(ref_naive["e2e_p90"], max(d["e2e_p90"] for d in data)) * 1.1)
plt.tight_layout()
plt.savefig(OUT / "fig2_latency_vs_lhi.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 3 — Wasted talker frames
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.0))
ax.plot(x, [d["wast_p50"] for d in data], "o-", color=C_SCHEDULER, lw=2, ms=8, label="wasted_p50")
ax.plot(x, [d["wast_p90"] for d in data], "s-", color=C_GOLD, lw=2, ms=8, label="wasted_p90")
ax.plot(x, [d["wast_max"] for d in data], "^--", color=C_AUX, lw=1.5, ms=7, label="wasted_max")

# Ideal-line: empirical fit wasted_p90 ≈ L_hi*1.2 + 12 (good fit for tight/mid; over-estimates wide)
xs = np.array([min(L_HIS), max(L_HIS)])
ax.plot(xs, xs * 1.2 + 12, ls=":", color="#888", lw=1.2,
        label="empirical fit  ≈1.2·L_hi+12")

# naive barge-in references
ax.axhline(ref_naive["wast_p50"], color=C_OFF_NAIVE, ls="--", lw=1.4,
           label=f"naive barge-in p50={ref_naive['wast_p50']:.0f}")
ax.axhline(ref_naive["wast_p90"], color=C_OFF_NAIVE, ls=":", lw=1.4,
           label=f"naive barge-in p90={ref_naive['wast_p90']:.0f}")

for d in data:
    ax.annotate(d["label"].replace("\n", " "),
                xy=(d["L_hi"], d["wast_p90"]), xytext=(8, 0),
                textcoords="offset points", ha="left", fontsize=9, color=C_SCHEDULER)

ax.set_xlabel("L_hi (frames)")
ax.set_ylabel("wasted talker frames per barge-in")
ax.set_title("Fig 3 · Wasted talker frames vs L_hi (lower = better)")
ax.set_xticks(L_HIS)
ax.set_yscale("log")
ax.legend(loc="upper left", framealpha=0.95, ncol=2)
plt.tight_layout()
plt.savefig(OUT / "fig3_wasted_vs_lhi.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 4 — Pareto: wall_rps vs wasted_p90
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))

# All scheduler points
xs_b = [d["wast_p90"] for d in data]
ys_b = [d["wall_rps"] for d in data]
ax.scatter(xs_b, ys_b, s=140, color=C_SCHEDULER, zorder=3, label="scheduler sweep")
# Per-cell annotation offsets to avoid overlap (12,20,30,50 ascending)
ann_offsets = [(-12, -22), (10, -18), (10, 14), (10, -18)]
for d, off in zip(data, ann_offsets):
    ha = "right" if off[0] < 0 else "left"
    ax.annotate(d["label"].replace("\n", " "),
                xy=(d["wast_p90"], d["wall_rps"]), xytext=off,
                textcoords="offset points", fontsize=9, color=C_SCHEDULER, ha=ha)

# no-policy baseline (no barge-in -> wast_p90 ~ 0)
ax.scatter([0.5], [ref_baseline["wall_rps"]], s=140, color=C_OFF_BL, zorder=3,
           label=f"no-policy baseline (no barge-in)  rps={ref_baseline['wall_rps']:.2f}")
ax.annotate(" no-policy baseline", xy=(0.5, ref_baseline["wall_rps"]),
            xytext=(8, -3), textcoords="offset points", fontsize=9, color=C_OFF_BL)

# naive barge-in (with barge-in but no lead control)
ax.scatter([ref_naive["wast_p90"]], [ref_naive["wall_rps"]], s=140, color=C_OFF_NAIVE,
           zorder=3, label=f"naive barge-in  rps={ref_naive['wall_rps']:.2f}")
ax.annotate(" naive barge-in (no lead control)",
            xy=(ref_naive["wast_p90"], ref_naive["wall_rps"]),
            xytext=(8, -3), textcoords="offset points", fontsize=9, color=C_OFF_NAIVE)

ax.set_xlabel("wasted_talker_frames p90 (lower is better)")
ax.set_ylabel("wall_rps (higher is better)")
ax.set_title("Fig 4 · Pareto frontier — throughput vs waste")
ax.set_xscale("log")
ax.set_xlim(0.3, 1000)
ax.set_ylim(0, max(ys_b) * 1.25)
ax.legend(loc="lower right", framealpha=0.95)

# Quadrant guide: ideal corner top-left
ax.text(0.32, max(ys_b) * 1.15,
        "← ideal corner\n  (low waste, high rps)",
        fontsize=10, color="#444", style="italic", ha="left", va="top")
# Highlight the chosen sweet spot (wide) with a star
wide = next(d for d in data if d["L_hi"] == 50)
ax.scatter([wide["wast_p90"]], [wide["wall_rps"]], s=350, marker="*",
           color="#EBC647", edgecolor="#A47900", lw=1.5, zorder=4,
           label="best operating point")
ax.legend(loc="lower right", framealpha=0.95)
plt.tight_layout()
plt.savefig(OUT / "fig4_pareto_rps_vs_waste.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 5 - Phase 2: scheduler wide(50/25) vs barge-in probability (only if data exists)
# ---------------------------------------------------------------------------
PROB_CELLS = [
    # (path tag, prob)
    ("scheduler_wide_p03", 0.3),
    ("scheduler_wide_p05", 0.5),
    ("scheduler_wide_p07", 0.7),
]
prob_data = []
for tag, prob in PROB_CELLS:
    s = load(S2 / tag)
    if s is None:
        continue
    d = extract(s); d["prob"] = prob; d["tag"] = tag
    prob_data.append(d)

if prob_data:
    # Anchors: p_bi=0 from no-policy baseline; p_bi=1 from scheduler wide.
    wide_p10 = extract(load_cell(S2, "scheduler_wide_p10", "scheduler_wide"))
    probs = [0.0] + [d["prob"] for d in prob_data] + [1.0]
    rps   = [ref_baseline["wall_rps"]] + [d["wall_rps"] for d in prob_data] + [wide_p10["wall_rps"]]
    use   = [ref_baseline["useful"]]   + [d["useful"]   for d in prob_data] + [wide_p10["useful"]]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(probs, rps, "o-", color=C_SCHEDULER, lw=2, ms=8, label="scheduler wide(50/25) wall_rps")
    ax.plot(probs, use, "s-", color=C_AUX,  lw=2, ms=8, label="scheduler wide(50/25) useful_qps")
    ax.axhline(ref_baseline["wall_rps"], color=C_OFF_BL, ls="--", lw=1.4,
               label=f"no-policy baseline (rps={ref_baseline['wall_rps']:.2f})")
    ax.axhline(ref_naive["useful"], color=C_OFF_NAIVE, ls=":", lw=1.4,
               label=f"naive barge-in useful={ref_naive['useful']:.2f}")
    # Annotate the p_bi=0 anchor specially.
    ax.annotate("(p_bi=0: no-policy baseline)",
                xy=(0, ref_baseline["wall_rps"]),
                xytext=(0.05, ref_baseline["wall_rps"] + 0.4),
                fontsize=9, color="#666", style="italic")
    for d in prob_data:
        ax.annotate(f"p={d['prob']:.1f}",
                    xy=(d["prob"], d["wall_rps"]), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9, color=C_SCHEDULER)
    ax.set_xlabel("p_bi (P[barge-in | request])")
    ax.set_ylabel("requests / sec")
    ax.set_title("Fig 5 · scheduler wide(50/25) - useful_qps recovery vs p_bi")
    ax.set_xticks(probs)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.set_ylim(0, max(max(rps), max(use)) * 1.20)
    plt.tight_layout()
    plt.savefig(OUT / "fig5_prob_sweep.png")
    plt.close(fig)
    print(f"[fig5] prob sweep with {len(prob_data)} data points: {[d['prob'] for d in prob_data]}")
else:
    print("[fig5] skipped (no Phase 2 data yet under results/step2/scheduler_wide_p*)")

print(f"[done] figs -> {OUT}/")
for p in sorted(OUT.glob("fig*.png")):
    print(f"  {p.name:<40} {p.stat().st_size / 1024:>6.1f} KB")
