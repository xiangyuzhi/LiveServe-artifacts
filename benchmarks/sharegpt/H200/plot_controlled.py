#!/usr/bin/env python3
"""Plot current H200 controlled ShareGPT results.

Figures are written to benchmarks/sharegpt/H200/analysis/figures/controlled so
they cannot be confused with H20 ShareGPT artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
RES = REPO_ROOT / "results" / "H200" / "sharegpt"
OUT = ROOT / "analysis" / "figures" / "controlled"
OUT.mkdir(parents=True, exist_ok=True)

BASELINE = RES / "controlled_baseline_2gpu_20260523_2050"
SCHEDULER = RES / "controlled_scheduler_wide_2gpu_20260523_2100"

C_BASE = "#4C78A8"
C_SCHEDULER = "#F58518"
C_AUX = "#54A24B"
C_RED = "#E45756"

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


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads((path / "summary.json").read_text())


def load_series(root: Path, prefix: str) -> list[dict[str, Any]]:
    rows = []
    for cell in sorted(root.glob(f"{prefix}_c*_n*_tok*")):
        s = load_summary(cell)
        meta = s["meta"]
        rows.append(
            {
                "cell": cell.name,
                "c": int(meta["concurrency"]),
                "ok": int(s["totals"]["completed_ok"]),
                "err": int(s["totals"]["errors"]),
                "rps": float(s["totals"]["throughput_rps"]),
                "tokps": float(s["totals"].get("output_token_throughput") or 0.0),
                "audio_durps": float(s["totals"].get("audio_duration_throughput") or 0.0),
                "e2e_p50": float(s["latency"]["e2e_ms"]["p50"] or 0.0),
                "e2e_p90": float(s["latency"]["e2e_ms"]["p90"] or 0.0),
                "ttft_p90": float(s["latency"]["ttft_ms"]["p90"] or 0.0),
                "audio_ttfp_p90": float(s["latency"]["audio_ttfp_ms"]["p90"] or 0.0),
            }
        )
    rows.sort(key=lambda r: r["c"])
    return rows


def fig1_text_scaling(text: list[dict[str, Any]]) -> None:
    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    x = [r["c"] for r in text]
    ax1.plot(x, [r["rps"] for r in text], "o-", color=C_BASE, lw=2.3, ms=8, label="requests/s")
    ax1.set_xlabel("concurrency")
    ax1.set_ylabel("requests / sec", color=C_BASE)
    ax1.tick_params(axis="y", labelcolor=C_BASE)
    ax2 = ax1.twinx()
    ax2.plot(x, [r["tokps"] for r in text], "s--", color=C_AUX, lw=2.0, ms=7, label="output tokens/s")
    ax2.set_ylabel("output tokens / sec", color=C_AUX)
    ax2.tick_params(axis="y", labelcolor=C_AUX)
    ax1.set_title("H200 ShareGPT text baseline scaling")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_sharegpt_text_scaling.png")
    plt.close(fig)


def fig2_audio_throughput(base: list[dict[str, Any]], scheduler: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot([r["c"] for r in base], [r["rps"] for r in base], "o-", color=C_BASE, lw=2.3, ms=8, label="baseline rps")
    ax.plot([r["c"] for r in scheduler], [r["rps"] for r in scheduler], "o-", color=C_SCHEDULER, lw=2.3, ms=8, label="scheduler wide rps")
    ax.set_xlabel("concurrency")
    ax.set_ylabel("requests / sec")
    ax.set_title("H200 ShareGPT audio: baseline vs scheduler throughput")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_sharegpt_audio_rps.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot([r["c"] for r in base], [r["audio_durps"] for r in base], "s-", color=C_BASE, lw=2.3, ms=8, label="baseline audio duration/s")
    ax.plot([r["c"] for r in scheduler], [r["audio_durps"] for r in scheduler], "s-", color=C_SCHEDULER, lw=2.3, ms=8, label="scheduler wide audio duration/s")
    ax.set_xlabel("concurrency")
    ax.set_ylabel("generated audio duration / wall second")
    ax.set_title("H200 ShareGPT audio: audio-duration throughput")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_sharegpt_audio_duration_throughput.png")
    plt.close(fig)


def fig3_audio_latency(base: list[dict[str, Any]], scheduler: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot([r["c"] for r in base], [r["audio_ttfp_p90"] for r in base], "o-", color=C_BASE, lw=2.3, ms=8, label="baseline audio TTFP p90")
    ax.plot([r["c"] for r in scheduler], [r["audio_ttfp_p90"] for r in scheduler], "o-", color=C_SCHEDULER, lw=2.3, ms=8, label="scheduler wide audio TTFP p90")
    ax.set_xlabel("concurrency")
    ax.set_ylabel("latency (ms)")
    ax.set_title("H200 ShareGPT audio: first audio packet latency")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_sharegpt_audio_ttfp_p90.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot([r["c"] for r in base], [r["e2e_p90"] for r in base], "^-", color=C_BASE, lw=2.3, ms=8, label="baseline E2E p90")
    ax.plot([r["c"] for r in scheduler], [r["e2e_p90"] for r in scheduler], "^-", color=C_SCHEDULER, lw=2.3, ms=8, label="scheduler wide E2E p90")
    ax.set_xlabel("concurrency")
    ax.set_ylabel("latency (ms)")
    ax.set_title("H200 ShareGPT audio: end-to-end latency")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_sharegpt_audio_e2e_p90.png")
    plt.close(fig)


def fig4_pareto(base: list[dict[str, Any]], scheduler: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for rows, color, label in [(base, C_BASE, "baseline"), (scheduler, C_SCHEDULER, "scheduler wide")]:
        ax.scatter([r["e2e_p90"] / 1000 for r in rows], [r["rps"] for r in rows], s=95, color=color, label=label)
        for r in rows:
            ax.annotate(f"c={r['c']}", (r["e2e_p90"] / 1000, r["rps"]), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("E2E p90 (s, lower is better)")
    ax.set_ylabel("requests / sec (higher is better)")
    ax.set_title("H200 ShareGPT audio Pareto: throughput vs E2E p90")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_sharegpt_audio_pareto.png")
    plt.close(fig)


def main() -> int:
    text = load_series(BASELINE, "text")
    audio_base = load_series(BASELINE, "audio")
    audio_scheduler = load_series(SCHEDULER, "audio")
    if not text or not audio_base or not audio_scheduler:
        raise SystemExit("missing controlled ShareGPT rows")
    fig1_text_scaling(text)
    fig2_audio_throughput(audio_base, audio_scheduler)
    fig3_audio_latency(audio_base, audio_scheduler)
    fig4_pareto(audio_base, audio_scheduler)
    for path in sorted(OUT.glob("*.png")):
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
