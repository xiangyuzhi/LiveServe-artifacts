#!/usr/bin/env python3
"""Plot current H200 high-concurrency capacity sweeps.

Input roots are under results/H200/capacity{2,3,4,6}gpu. Figures are written to
benchmarks/interactive/H200/analysis/figures/capacity.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
RES_ROOT = REPO_ROOT / "results" / "H200"
OUT = ROOT / "analysis" / "figures" / "capacity"
OUT.mkdir(parents=True, exist_ok=True)

RUN_ROOTS = {
    2: RES_ROOT / "capacity2gpu" / "highc_2gpu_20260523_150215",
    3: RES_ROOT / "capacity3gpu" / "highc_3gpu_20260523_163829",
    4: RES_ROOT / "capacity4gpu" / "highc_4gpu_20260523_173952",
    6: RES_ROOT / "capacity6gpu" / "highc_6gpu_clean_20260523_190350",
}

COLORS = {
    2: "#4C78A8",
    3: "#F58518",
    4: "#54A24B",
    6: "#E45756",
}

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


def get_path(obj: dict[str, Any], dotted: str, default: float = 0.0) -> float:
    cur: Any = obj
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return float(cur) if isinstance(cur, (int, float)) else default


def infer_c(cell_dir: Path, summary: dict[str, Any]) -> int:
    value = summary.get("args", {}).get("max_inflight")
    if isinstance(value, int):
        return value
    m = re.search(r"_c(\d+)_", cell_dir.name)
    if not m:
        raise ValueError(f"cannot infer concurrency from {cell_dir}")
    return int(m.group(1))


def load_rows() -> list[dict[str, Any]]:
    rows = []
    for gpus, root in RUN_ROOTS.items():
        if not root.exists():
            print(f"[WARN] missing {root}")
            continue
        for summary_path in sorted(root.rglob("summary.json")):
            cell = summary_path.parent
            summary = json.loads(summary_path.read_text())
            rows.append(
                {
                    "gpus": gpus,
                    "c": infer_c(cell, summary),
                    "tag": cell.name,
                    "ok": int(summary.get("totals", {}).get("completed_ok", 0)),
                    "err": int(summary.get("totals", {}).get("errors", 0)),
                    "rps": get_path(summary, "totals.throughput_rps"),
                    "useful": get_path(summary, "totals.useful_qps"),
                    "audio_p50_ms": get_path(summary, "latency.audio_ttfp_sec.p50") * 1000,
                    "audio_p90_ms": get_path(summary, "latency.audio_ttfp_sec.p90") * 1000,
                    "e2e_p90_ms": get_path(summary, "latency.e2e_sec.p90") * 1000,
                }
            )
    rows.sort(key=lambda r: (r["gpus"], r["c"]))
    return rows


def by_gpu(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["gpus"], []).append(row)
    return out


def fig1_throughput(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for gpus, group in by_gpu(rows).items():
        ax.plot(
            [r["c"] for r in group],
            [r["useful"] for r in group],
            "o-",
            color=COLORS[gpus],
            lw=2.3,
            ms=7,
            label=f"{gpus}GPU",
        )
    ax.set_xlabel("max_inflight")
    ax.set_ylabel("useful_qps")
    ax.set_title("H200 capacity sweep: useful throughput vs concurrency")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_capacity_useful_qps.png")
    plt.close(fig)


def fig2_audio_latency(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for gpus, group in by_gpu(rows).items():
        ax.plot(
            [r["c"] for r in group],
            [r["audio_p90_ms"] / 1000 for r in group],
            "s-",
            color=COLORS[gpus],
            lw=2.3,
            ms=7,
            label=f"{gpus}GPU",
        )
    ax.set_xlabel("max_inflight")
    ax.set_ylabel("audio TTFP p90 (s)")
    ax.set_title("H200 capacity sweep: first-audio latency cliff")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_capacity_audio_ttfp_p90.png")
    plt.close(fig)


def fig3_pareto(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    for gpus, group in by_gpu(rows).items():
        ax.scatter(
            [r["audio_p90_ms"] / 1000 for r in group],
            [r["useful"] for r in group],
            s=85,
            color=COLORS[gpus],
            label=f"{gpus}GPU",
            alpha=0.9,
        )
        for r in group:
            if r["c"] in {16, 32, 48, 96, 128, 160, 192}:
                ax.annotate(f"c={r['c']}", (r["audio_p90_ms"] / 1000, r["useful"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("audio TTFP p90 (s, lower is better)")
    ax.set_ylabel("useful_qps (higher is better)")
    ax.set_title("H200 capacity Pareto: throughput vs first-audio latency")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_capacity_pareto.png")
    plt.close(fig)


def fig4_best_by_gpu(rows: list[dict[str, Any]]) -> None:
    best = []
    for gpus, group in by_gpu(rows).items():
        best.append(max(group, key=lambda r: r["useful"]))
    best.sort(key=lambda r: r["gpus"])

    labels = [f"{r['gpus']}GPU\nc={r['c']}" for r in best]
    x = np.arange(len(best))
    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    bars = ax1.bar(x, [r["useful"] for r in best], color=[COLORS[r["gpus"]] for r in best], alpha=0.85)
    ax1.set_ylabel("best useful_qps")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("H200 capacity best point by GPU count")
    for rect, row in zip(bars, best):
        ax1.annotate(f"{row['useful']:.2f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, [r["audio_p90_ms"] / 1000 for r in best], "o--", color="#333333", lw=1.8, label="audio p90 at best")
    ax2.set_ylabel("audio TTFP p90 at best (s)")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_capacity_best_by_gpu.png")
    plt.close(fig)


def main() -> int:
    rows = load_rows()
    if not rows:
        raise SystemExit("no capacity rows found")
    fig1_throughput(rows)
    fig2_audio_latency(rows)
    fig3_pareto(rows)
    fig4_best_by_gpu(rows)
    for path in sorted(OUT.glob("*.png")):
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
