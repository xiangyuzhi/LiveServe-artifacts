#!/usr/bin/env python3
"""Step-1 baseline visualization for interactive audio/barge-in experiments.

Reads results/step1/*/summary.json and writes four figures under
analysis/figures/step1:
  Fig 1: wall throughput versus configured barge-in probability.
  Fig 2: wall throughput versus useful throughput.
  Fig 3: E2E p50/p90 split by completed versus interrupted requests.
  Fig 4: wasted talker frames for c=4 and c=8.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no DISPLAY needed
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results" / "step1"
OUT = ROOT / "analysis" / "figures" / "step1"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cell registry — (tag, c, prob, label, color)
# ---------------------------------------------------------------------------
CELLS: list[tuple[str, int, float, str, str]] = [
    ("v2_audio_c4_60s_v2",   4, 0.0, "c=4 baseline",  "#4C78A8"),
    ("audio_c4_60s_prob03",  4, 0.3, "c=4 prob=0.3",  "#4C78A8"),
    ("audio_c4_60s_prob07",  4, 0.7, "c=4 prob=0.7",  "#4C78A8"),
    ("audio_c4_60s_prob10",  4, 1.0, "c=4 prob=1.0",  "#4C78A8"),
    ("audio_c8_60s_prob0",   8, 0.0, "c=8 baseline",  "#F58518"),
    ("audio_c8_60s_prob10",  8, 1.0, "c=8 prob=1.0",  "#F58518"),
]

C4_COLOR = "#4C78A8"
C8_COLOR = "#F58518"
GOAL_COLOR = "#54A24B"  # scheduler target

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


def load_summary(tag: str) -> dict[str, Any] | None:
    p = RES / tag / "summary.json"
    if not p.exists():
        print(f"[WARN] missing {p}")
        return None
    with open(p) as f:
        return json.load(f)


def get_summary_data() -> list[dict[str, Any]]:
    """Resolve all CELLS into a flat dict list with derived fields."""
    out = []
    for tag, c, prob, label, color in CELLS:
        s = load_summary(tag)
        if s is None:
            continue
        totals = s.get("totals", {})
        latency = s.get("latency", {})
        intr = s.get("interrupt", {})
        out.append({
            "tag": tag, "c": c, "prob": prob, "label": label, "color": color,
            "rps": totals.get("throughput_rps", 0.0),
            "useful_qps": totals.get("useful_qps", totals.get("throughput_rps", 0.0)),
            "completed": totals.get("completed_ok", 0),
            "interrupted": totals.get("interrupted", 0),
            "wall_time_sec": totals.get("wall_time_sec", 0.0),
            "ttfp_p50_ms": latency.get("ttfp_sec", {}).get("p50", 0) * 1000,
            "audio_ttfp_p50_ms": latency.get("audio_ttfp_sec", {}).get("p50", 0) * 1000,
            "audio_ttfp_p90_ms": latency.get("audio_ttfp_sec", {}).get("p90", 0) * 1000,
            "e2e_p50_ms": latency.get("e2e_sec", {}).get("p50", 0) * 1000,
            "e2e_p90_ms": latency.get("e2e_sec", {}).get("p90", 0) * 1000,
            "waste_total_p50": intr.get("wasted_talker_frames_total", {}).get("p50", 0),
            "waste_total_p90": intr.get("wasted_talker_frames_total", {}).get("p90", 0),
            "waste_inaud_p50": intr.get("wasted_talker_frames_inaudible", {}).get("p50", 0),
            "waste_inaud_p90": intr.get("wasted_talker_frames_inaudible", {}).get("p90", 0),
        })
    return out


# ---------------------------------------------------------------------------
# Fig 1: wall throughput versus configured barge-in probability.
# ---------------------------------------------------------------------------
def fig1_rps_vs_prob(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    c4 = sorted([r for r in rows if r["c"] == 4], key=lambda x: x["prob"])
    c8 = sorted([r for r in rows if r["c"] == 8], key=lambda x: x["prob"])

    ax.plot([r["prob"] for r in c4], [r["rps"] for r in c4],
            marker="o", linewidth=2.5, markersize=10, color=C4_COLOR,
            label="c=4 (max_inflight)")
    ax.plot([r["prob"] for r in c8], [r["rps"] for r in c8],
            marker="s", linewidth=2.5, markersize=10, color=C8_COLOR,
            label="c=8 (max_inflight)")

    # Value labels.
    for r in c4 + c8:
        ax.annotate(f"{r['rps']:.2f}", xy=(r["prob"], r["rps"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9)

    # scheduler target
    ax.axhline(1.40, color=GOAL_COLOR, linestyle="--", linewidth=1.3, alpha=0.7,
               label="scheduler wall_rps target (>= 1.40)")

    ax.set_xlabel("barge-in probability (configured)")
    ax.set_ylabel("wall throughput (req/s)")
    ax.set_title("Fig 1 - Barge-in yield curve (wall throughput vs probability)\n"
                 "audio mode, anchor=ttfp+2.0+/-0.5s, 60s trace window")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(0.5, 1.7)
    ax.legend(loc="lower right")

    # Highlight key deltas.
    if len(c4) == 4 and len(c8) == 2:
        delta_c4 = (c4[-1]["rps"] - c4[0]["rps"]) / c4[0]["rps"] * 100
        delta_c8 = (c8[-1]["rps"] - c8[0]["rps"]) / c8[0]["rps"] * 100
        ax.annotate(f"+{delta_c4:.0f}% (c=4)",
                    xy=(0.5, (c4[0]["rps"] + c4[-1]["rps"]) / 2),
                    fontsize=10, color=C4_COLOR, fontweight="bold")
        ax.annotate(f"+{delta_c8:.0f}% (c=8)",
                    xy=(0.5, (c8[0]["rps"] + c8[-1]["rps"]) / 2 + 0.05),
                    fontsize=10, color=C8_COLOR, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "fig1_rps_vs_prob.png")
    plt.close(fig)
    print(f"[OK] {OUT / 'fig1_rps_vs_prob.png'}")


# ---------------------------------------------------------------------------
# Fig 2: wall throughput versus useful throughput.
# ---------------------------------------------------------------------------
def fig2_wall_vs_useful(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    c4 = sorted([r for r in rows if r["c"] == 4], key=lambda x: x["prob"])
    c8 = sorted([r for r in rows if r["c"] == 8], key=lambda x: x["prob"])

    # c=4 pair.
    ax.plot([r["prob"] for r in c4], [r["rps"] for r in c4],
            marker="o", color=C4_COLOR, linewidth=2.2, markersize=9,
            label="c=4 wall_rps")
    ax.plot([r["prob"] for r in c4], [r["useful_qps"] for r in c4],
            marker="o", color=C4_COLOR, linewidth=2.2, markersize=9,
            linestyle="--", alpha=0.6, label="c=4 useful_qps")

    # c=8 pair.
    ax.plot([r["prob"] for r in c8], [r["rps"] for r in c8],
            marker="s", color=C8_COLOR, linewidth=2.2, markersize=9,
            label="c=8 wall_rps")
    ax.plot([r["prob"] for r in c8], [r["useful_qps"] for r in c8],
            marker="s", color=C8_COLOR, linewidth=2.2, markersize=9,
            linestyle="--", alpha=0.6, label="c=8 useful_qps")

    # Shaded gaps between wall throughput and useful throughput.
    if len(c4) >= 2:
        xs = [r["prob"] for r in c4]
        wall = [r["rps"] for r in c4]
        useful = [r["useful_qps"] for r in c4]
        ax.fill_between(xs, useful, wall, color=C4_COLOR, alpha=0.10)
    if len(c8) >= 2:
        xs = [r["prob"] for r in c8]
        wall = [r["rps"] for r in c8]
        useful = [r["useful_qps"] for r in c8]
        ax.fill_between(xs, useful, wall, color=C8_COLOR, alpha=0.10)

    # scheduler target
    ax.axhline(1.10, color=GOAL_COLOR, linestyle="--", linewidth=1.3, alpha=0.7,
               label="scheduler useful_qps target (>= 1.10)")

    ax.set_xlabel("barge-in probability (configured)")
    ax.set_ylabel("throughput (req/s)")
    ax.set_title("Fig 2 - Wall vs Useful Throughput Divergence\n"
                 "solid=wall_rps (counts interrupted/wasted reqs), "
                 "dashed=useful_qps (only fully completed)")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(0.3, 1.7)
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_wall_vs_useful.png")
    plt.close(fig)
    print(f"[OK] {OUT / 'fig2_wall_vs_useful.png'}")


# ---------------------------------------------------------------------------
# Fig 3: E2E p50/p90 split by interrupted versus completed requests.
# ---------------------------------------------------------------------------
def fig3_e2e_split(rows: list[dict]) -> None:
    """Recompute percentiles from per_request.jsonl by interrupt status."""
    cells_to_plot = [
        ("v2_audio_c4_60s_v2",   "c=4\nbase"),
        ("audio_c4_60s_prob10",  "c=4\nprob=1.0"),
        ("audio_c8_60s_prob0",   "c=8\nbase"),
        ("audio_c8_60s_prob10",  "c=8\nprob=1.0"),
    ]

    def _percentile(arr, p):
        if not arr:
            return 0.0
        s = sorted(arr)
        idx = int(len(s) * p)
        return s[min(idx, len(s) - 1)]

    groups: list[dict] = []
    for tag, label in cells_to_plot:
        path = RES / tag / "per_request.jsonl"
        if not path.exists():
            print(f"[WARN] missing {path}")
            continue
        e2e_all, e2e_int, e2e_notint = [], [], []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error"):
                    continue
                e2e = r.get("e2e_sec")
                if e2e is None:
                    continue
                e2e_all.append(e2e * 1000)
                if r.get("interrupted"):
                    e2e_int.append(e2e * 1000)
                else:
                    e2e_notint.append(e2e * 1000)
        groups.append({
            "tag": tag, "label": label,
            "all_p50": _percentile(e2e_all, 0.5),
            "all_p90": _percentile(e2e_all, 0.9),
            "int_p50": _percentile(e2e_int, 0.5) if e2e_int else 0,
            "int_p90": _percentile(e2e_int, 0.9) if e2e_int else 0,
            "not_p50": _percentile(e2e_notint, 0.5) if e2e_notint else 0,
            "not_p90": _percentile(e2e_notint, 0.9) if e2e_notint else 0,
            "n_int": len(e2e_int), "n_not": len(e2e_notint),
        })

    fig, (ax50, ax90) = plt.subplots(1, 2, figsize=(13, 5))
    n = len(groups)
    x = np.arange(n)
    width = 0.27

    def draw(ax, key_all, key_int, key_not, title):
        ax.bar(x - width, [g[key_all] for g in groups], width,
               label="overall", color="#888", alpha=0.85)
        ax.bar(x,         [g[key_int] for g in groups], width,
               label="interrupted", color="#E45756", alpha=0.85)
        ax.bar(x + width, [g[key_not] for g in groups], width,
               label="not interrupted", color="#54A24B", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([g["label"] for g in groups])
        ax.set_ylabel("E2E latency (ms)")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=9)
        # bar value labels
        for i, g in enumerate(groups):
            for off, k in [(-width, key_all), (0, key_int), (width, key_not)]:
                v = g[k]
                if v > 0:
                    ax.annotate(f"{int(v)}", xy=(x[i] + off, v),
                                xytext=(0, 2), textcoords="offset points",
                                ha="center", fontsize=8)

    draw(ax50, "all_p50", "int_p50", "not_p50",
         "Fig 3a — E2E latency p50 (overall vs split)")
    draw(ax90, "all_p90", "int_p90", "not_p90",
         "Fig 3b — E2E latency p90 (overall vs split)")
    fig.suptitle(
        "Fig 3 - Interrupt as implicit priority? (split by interrupted vs not)",
        fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_e2e_split.png")
    plt.close(fig)
    print(f"[OK] {OUT / 'fig3_e2e_split.png'}")


# ---------------------------------------------------------------------------
# Fig 4: wasted_talker_frames p50/p90 for c=4 and c=8.
# ---------------------------------------------------------------------------
def fig4_waste(rows: list[dict]) -> None:
    interrupted_rows = [r for r in rows if r["prob"] > 0 and r["waste_total_p50"] > 0]
    if not interrupted_rows:
        print("[WARN] no interrupted cells for fig4")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5))
    n = len(interrupted_rows)
    x = np.arange(n)
    width = 0.20

    labels = [f"{r['label']}" for r in interrupted_rows]
    colors_total_p50 = "#5277A5"
    colors_total_p90 = "#3A5A82"
    colors_inaud_p50 = "#F2A14E"
    colors_inaud_p90 = "#C76C16"

    ax.bar(x - 1.5 * width, [r["waste_total_p50"] for r in interrupted_rows],
           width, label="total p50", color=colors_total_p50)
    ax.bar(x - 0.5 * width, [r["waste_total_p90"] for r in interrupted_rows],
           width, label="total p90", color=colors_total_p90)
    ax.bar(x + 0.5 * width, [r["waste_inaud_p50"] for r in interrupted_rows],
           width, label="inaudible p50", color=colors_inaud_p50)
    ax.bar(x + 1.5 * width, [r["waste_inaud_p90"] for r in interrupted_rows],
           width, label="inaudible p90", color=colors_inaud_p90)

    # value labels
    for i, r in enumerate(interrupted_rows):
        for off, key in [(-1.5 * width, "waste_total_p50"),
                         (-0.5 * width, "waste_total_p90"),
                         (0.5 * width,  "waste_inaud_p50"),
                         (1.5 * width,  "waste_inaud_p90")]:
            v = r[key]
            if v > 0:
                ax.annotate(f"{int(v)}", xy=(x[i] + off, v),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=8)

    # scheduler target
    ax.axhline(50, color=GOAL_COLOR, linestyle="--", linewidth=1.3, alpha=0.7,
               label="scheduler wasted p90 target (<= 50 frames)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("wasted talker frames (1 frame = 40ms audio)")
    ax.set_title(
        "Fig 4 - Wasted Talker Frames per Interrupt (lower = better)\n"
        "1 frame ~ 40ms TTS audio; total = generated-played, "
        "inaudible = received-played")
    ax.legend(loc="upper left", fontsize=9)

    # Secondary axis in seconds.
    ax2 = ax.twinx()
    ymin, ymax = ax.get_ylim()
    ax2.set_ylim(ymin / 25, ymax / 25)
    ax2.set_ylabel("equivalent audio time (s)")
    ax2.grid(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig4_wasted_frames.png")
    plt.close(fig)
    print(f"[OK] {OUT / 'fig4_wasted_frames.png'}")


# ---------------------------------------------------------------------------
def main():
    rows = get_summary_data()
    if not rows:
        print("[ERROR] no summary data found", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] loaded {len(rows)} cells, output dir = {OUT}")
    fig1_rps_vs_prob(rows)
    fig2_wall_vs_useful(rows)
    fig3_e2e_split(rows)
    fig4_waste(rows)
    print("\n[DONE] 4 figures written to:", OUT)


if __name__ == "__main__":
    main()
