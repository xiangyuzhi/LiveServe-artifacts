#!/usr/bin/env python3
"""Plot current H200 step-1 interactive results.

This is the H200 counterpart of the H20 step-1 plotting script, adapted to the
current H200 cells:

  - results/H200/step1/baseline_c8_p0
  - results/H200/step1/naive_c8_p10

Outputs stay under benchmarks/interactive/H200/analysis/figures/step1_current.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
RES = REPO_ROOT / "results" / "H200" / "step1"
OUT = ROOT / "analysis" / "figures" / "step1_current"
OUT.mkdir(parents=True, exist_ok=True)

CELLS = [
    ("baseline_c8_p0", 0.0, "OFF baseline\np=0"),
    ("naive_c8_p10", 1.0, "OFF naive\np=1"),
]

C_BLUE = "#4C78A8"
C_ORANGE = "#F58518"
C_RED = "#E45756"
C_GREEN = "#54A24B"
C_PURPLE = "#B279A2"

plt.rcParams.update(
    {
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
    }
)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def load_summary(tag: str) -> dict[str, Any]:
    path = RES / tag / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_rows(tag: str) -> list[dict[str, Any]]:
    path = RES / tag / "per_request.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_path(obj: dict[str, Any], dotted: str, default: float = 0.0) -> float:
    cur: Any = obj
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return float(cur) if isinstance(cur, (int, float)) else default


def collect() -> list[dict[str, Any]]:
    rows = []
    for tag, prob, label in CELLS:
        s = load_summary(tag)
        totals = s["totals"]
        lat = s["latency"]
        intr = s.get("interrupt", {})
        rows.append(
            {
                "tag": tag,
                "prob": prob,
                "label": label,
                "ok": totals["completed_ok"],
                "err": totals["errors"],
                "interrupted": totals.get("interrupted", 0),
                "wall_rps": totals["throughput_rps"],
                "useful_qps": totals.get("useful_qps", totals["throughput_rps"]),
                "ttfp_p90_ms": get_path(lat, "ttfp_sec.p90") * 1000,
                "audio_ttfp_p50_ms": get_path(lat, "audio_ttfp_sec.p50") * 1000,
                "audio_ttfp_p90_ms": get_path(lat, "audio_ttfp_sec.p90") * 1000,
                "e2e_p50_ms": get_path(lat, "e2e_sec.p50") * 1000,
                "e2e_p90_ms": get_path(lat, "e2e_sec.p90") * 1000,
                "waste_p50": get_path(intr, "wasted_talker_frames_total.p50"),
                "waste_p90": get_path(intr, "wasted_talker_frames_total.p90"),
            }
        )
    return rows


def fig1_throughput(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = [r["prob"] for r in rows]
    ax.plot(x, [r["wall_rps"] for r in rows], "o-", color=C_BLUE, lw=2.4, ms=9, label="wall_rps")
    ax.plot(x, [r["useful_qps"] for r in rows], "s--", color=C_PURPLE, lw=2.0, ms=8, label="useful_qps")
    for r in rows:
        ax.annotate(
            f"{r['wall_rps']:.2f}/{r['useful_qps']:.2f}",
            xy=(r["prob"], r["wall_rps"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("interrupt_prob")
    ax.set_ylabel("requests / sec")
    ax.set_title("H200 Step-1: OFF baseline vs naive interrupt throughput")
    ax.set_xlim(-0.08, 1.08)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_step1_wall_vs_useful.png")
    plt.close(fig)


def fig2_latency(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    width = 0.25
    ax.bar(x - width, [r["ttfp_p90_ms"] for r in rows], width, color=C_GREEN, label="text TTFT p90")
    ax.bar(x, [r["audio_ttfp_p90_ms"] for r in rows], width, color=C_ORANGE, label="audio TTFP p90")
    ax.bar(x + width, [r["e2e_p90_ms"] for r in rows], width, color=C_BLUE, label="E2E p90")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("latency (ms)")
    ax.set_title("H200 Step-1: latency impact of naive interrupt")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_step1_latency_bars.png")
    plt.close(fig)


def fig3_e2e_split() -> None:
    groups = []
    for tag, _prob, label in CELLS:
        all_e2e: list[float] = []
        int_e2e: list[float] = []
        not_e2e: list[float] = []
        for r in load_rows(tag):
            if r.get("error") is not None or r.get("e2e_sec") is None:
                continue
            value = float(r["e2e_sec"]) * 1000
            all_e2e.append(value)
            if r.get("interrupted"):
                int_e2e.append(value)
            else:
                not_e2e.append(value)
        groups.append(
            {
                "label": label,
                "all_p90": pct(all_e2e, 0.90),
                "int_p90": pct(int_e2e, 0.90),
                "not_p90": pct(not_e2e, 0.90),
            }
        )

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(groups))
    width = 0.25
    ax.bar(x - width, [g["all_p90"] for g in groups], width, color=C_BLUE, label="all p90")
    ax.bar(x, [g["int_p90"] for g in groups], width, color=C_RED, label="interrupted p90")
    ax.bar(x + width, [g["not_p90"] for g in groups], width, color=C_GREEN, label="not interrupted p90")
    ax.set_xticks(x)
    ax.set_xticklabels([g["label"] for g in groups])
    ax.set_ylabel("E2E latency p90 (ms)")
    ax.set_title("H200 Step-1: E2E p90 split by interrupt status")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_step1_e2e_split.png")
    plt.close(fig)


def fig4_waste(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    width = 0.35
    ax.bar(x - width / 2, [r["waste_p50"] for r in rows], width, color=C_BLUE, label="wasted frames p50")
    ax.bar(x + width / 2, [r["waste_p90"] for r in rows], width, color=C_ORANGE, label="wasted frames p90")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("talker frames")
    ax.set_title("H200 Step-1: naive interrupt wasted talker frames")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_step1_wasted_frames.png")
    plt.close(fig)


def main() -> int:
    rows = collect()
    fig1_throughput(rows)
    fig2_latency(rows)
    fig3_e2e_split()
    fig4_waste(rows)
    for path in sorted(OUT.glob("*.png")):
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
