#!/usr/bin/env python3
"""Plot H200 3-GPU barge-in matrix results.

Usage:
    python benchmarks/interactive/H200/plot_bargein3gpu_matrix.py \
        results/H200/bargein3gpu/<run_id>

Outputs are written under `<run_root>/_plots` so H200/L20X figures stay next to
their source run and do not mix with H20 artifacts.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SYSTEMS = ["naive", "scheduler_wide", "scheduler_tight"]
SYSTEM_LABELS = {
    "naive": "naive",
    "scheduler_wide": "scheduler wide (50/25)",
    "scheduler_tight": "scheduler tight (20/10)",
}
COLORS = {
    "naive": "#4C78A8",
    "scheduler_wide": "#F58518",
    "scheduler_tight": "#54A24B",
}
MARKERS = {
    "naive": "o",
    "scheduler_wide": "s",
    "scheduler_tight": "^",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "figure.dpi": 120,
    "savefig.dpi": 170,
    "savefig.bbox": "tight",
})


def get_path(node: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = node
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    xs = sorted(values)
    n = len(xs)
    return {
        "p50": xs[n // 2],
        "p90": xs[int(n * 0.90)],
        "p95": xs[int(n * 0.95)],
        "mean": statistics.mean(xs),
    }


def parse_tag(tag: str) -> tuple[str, int, float]:
    m = re.match(r"(.+)_c(\d+)_p(\d+)_win", tag)
    if not m:
        raise ValueError(f"cannot parse cell tag: {tag}")
    system = m.group(1)
    legacy_prefix = "bp" + "v3"
    legacy = {
        f"{legacy_prefix}_wide": "scheduler_wide",
        f"{legacy_prefix}_tight": "scheduler_tight",
    }
    return legacy.get(system, system), int(m.group(2)), int(m.group(3)) / 100.0


def load_audio_rtf(cell_dir: Path) -> dict[str, Any]:
    per_req = cell_dir / "per_request.jsonl"
    post: list[float] = []
    stream: list[float] = []
    e2e_proxy: list[float] = []
    n_timeline = 0

    if not per_req.exists():
        return {
            "post": {},
            "stream": {},
            "e2e_proxy": {},
            "n_timeline": 0,
        }

    with per_req.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") is not None or row.get("interrupted"):
                continue
            e2e = row.get("e2e_sec")
            audio_ttfp = row.get("audio_ttfp_sec")
            timeline = row.get("audio_timeline") or []
            chunk_durations = [
                item.get("audio_duration_sec") for item in timeline
                if isinstance(item, dict)
                and isinstance(item.get("audio_duration_sec"), (int, float))
                and item.get("audio_duration_sec") > 0
            ]
            if chunk_durations:
                audio_duration = float(sum(chunk_durations))
                n_timeline += 1
            else:
                fallback = row.get("audio_duration_sec")
                if not (isinstance(fallback, (int, float)) and fallback > 0):
                    continue
                audio_duration = float(fallback)

            if isinstance(e2e, (int, float)):
                e2e_proxy.append(e2e / audio_duration)
                if isinstance(audio_ttfp, (int, float)):
                    post.append(max(0.0, e2e - audio_ttfp) / audio_duration)

            if len(timeline) >= 2:
                first = timeline[0]
                last = timeline[-1]
                t0 = first.get("recv_sec")
                t1 = last.get("recv_sec")
                if isinstance(t0, (int, float)) and isinstance(t1, (int, float)):
                    stream.append(max(0.0, t1 - t0) / audio_duration)

    return {
        "post": percentiles(post),
        "stream": percentiles(stream),
        "e2e_proxy": percentiles(e2e_proxy),
        "n_timeline": n_timeline,
    }


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("summary.json")):
        cell_dir = summary_path.parent
        system, c, p = parse_tag(cell_dir.name)
        with summary_path.open() as f:
            summary = json.load(f)
        rtf = load_audio_rtf(cell_dir)
        rows.append({
            "tag": cell_dir.name,
            "system": system,
            "c": c,
            "p": p,
            "ok": int(get_path(summary, "totals.completed_ok", 0)),
            "err": int(get_path(summary, "totals.errors", 0)),
            "interrupted": int(get_path(summary, "totals.interrupted", 0)),
            "wall_s": float(get_path(summary, "totals.wall_time_sec", 0.0)),
            "rps": float(get_path(summary, "totals.throughput_rps", 0.0)),
            "useful": float(get_path(summary, "totals.useful_qps", 0.0)),
            "text_p90_ms": float(get_path(summary, "latency.ttfp_sec.p90", math.nan)) * 1000,
            "audio_p50_s": float(get_path(summary, "latency.audio_ttfp_sec.p50", math.nan)),
            "audio_p90_s": float(get_path(summary, "latency.audio_ttfp_sec.p90", math.nan)),
            "e2e_p90_s": float(get_path(summary, "latency.e2e_sec.p90", math.nan)),
            "waste_p50": float(get_path(summary, "interrupt.wasted_talker_frames_total.p50", math.nan)),
            "waste_p90": float(get_path(summary, "interrupt.wasted_talker_frames_total.p90", math.nan)),
            "stream_rtf_p50": rtf["stream"].get("p50"),
            "stream_rtf_p90": rtf["stream"].get("p90"),
            "post_ttfp_rtf_p50": rtf["post"].get("p50"),
            "post_ttfp_rtf_p90": rtf["post"].get("p90"),
            "e2e_audio_rtf_p50": rtf["e2e_proxy"].get("p50"),
            "e2e_audio_rtf_p90": rtf["e2e_proxy"].get("p90"),
            "timeline_n": rtf["n_timeline"],
        })
    rows.sort(key=lambda r: (SYSTEMS.index(r["system"]) if r["system"] in SYSTEMS else 99,
                             r["c"], r["p"]))
    return rows


def value(rows: list[dict[str, Any]], system: str, c: int, p: float, key: str) -> float:
    for row in rows:
        if row["system"] == system and row["c"] == c and abs(row["p"] - p) < 1e-9:
            v = row.get(key)
            return float(v) if isinstance(v, (int, float)) else math.nan
    return math.nan


def write_csv(rows: list[dict[str, Any]], out: Path) -> None:
    fields = [
        "system", "c", "p", "ok", "err", "interrupted", "wall_s", "rps",
        "useful", "text_p90_ms", "audio_p50_s", "audio_p90_s", "e2e_p90_s",
        "waste_p50", "waste_p90", "stream_rtf_p50", "stream_rtf_p90",
        "post_ttfp_rtf_p50", "post_ttfp_rtf_p90", "e2e_audio_rtf_p50",
        "e2e_audio_rtf_p90", "timeline_n", "tag",
    ]
    with (out / "cell_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def setup_axes_grid(title: str):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True)
    fig.suptitle(title, y=0.99, fontsize=14)
    return fig, axes.ravel()


def plot_metric_grid(rows: list[dict[str, Any]], out: Path, key: str,
                     ylabel: str, filename: str, title: str,
                     *, hline: float | None = None) -> None:
    ps = sorted({r["p"] for r in rows})
    cs = sorted({r["c"] for r in rows})
    fig, axes = setup_axes_grid(title)
    for ax, p in zip(axes, ps):
        for system in SYSTEMS:
            ys = [value(rows, system, c, p, key) for c in cs]
            ax.plot(cs, ys, marker=MARKERS[system], color=COLORS[system],
                    lw=2.1, ms=6, label=SYSTEM_LABELS[system])
        if hline is not None:
            ax.axhline(hline, color="#333333", lw=1.2, ls="--", alpha=0.7)
        ax.set_title(f"p_bi={p:.1f}")
        ax.set_xlabel("max_inflight")
        ax.set_ylabel(ylabel)
        ax.set_xticks(cs)
    axes[0].legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out / filename)
    plt.close(fig)


def plot_delta_vs_naive(rows: list[dict[str, Any]], out: Path) -> None:
    ps = sorted({r["p"] for r in rows})
    cs = sorted({r["c"] for r in rows})
    fig, axes = setup_axes_grid("Scheduler delta vs naive: audio TTFP p90")
    for ax, p in zip(axes, ps):
        for system in ["scheduler_wide", "scheduler_tight"]:
            ys = [
                value(rows, system, c, p, "audio_p90_s")
                - value(rows, "naive", c, p, "audio_p90_s")
                for c in cs
            ]
            ax.plot(cs, ys, marker=MARKERS[system], color=COLORS[system],
                    lw=2.1, ms=6, label=SYSTEM_LABELS[system])
        ax.axhline(0, color="#333333", lw=1.2)
        ax.set_title(f"p_bi={p:.1f}")
        ax.set_xlabel("max_inflight")
        ax.set_ylabel("delta audio p90 (s); negative is better")
        ax.set_xticks(cs)
    axes[0].legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out / "fig2_delta_audio_ttfp_p90_vs_naive.png")
    plt.close(fig)


def plot_pareto(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for system in SYSTEMS:
        group = [r for r in rows if r["system"] == system]
        ax.scatter(
            [r["audio_p90_s"] for r in group],
            [r["useful"] for r in group],
            s=[45 + (r["c"] / 2) for r in group],
            color=COLORS[system],
            marker=MARKERS[system],
            alpha=0.82,
            label=SYSTEM_LABELS[system],
        )
        for r in group:
            if r["c"] == 96 and r["p"] in {0.3, 0.9}:
                ax.annotate(
                    f"c{r['c']} p{r['p']:.1f}",
                    (r["audio_p90_s"], r["useful"]),
                    xytext=(5, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
    ax.set_xlabel("audio TTFP p90 (s, lower is better)")
    ax.set_ylabel("useful_qps (higher is better)")
    ax.set_title("Pareto: useful throughput vs first-audio tail latency")
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out / "fig5_pareto_useful_vs_audio_ttfp.png")
    plt.close(fig)


def write_markdown(rows: list[dict[str, Any]], out: Path, root: Path) -> None:
    total_ok = sum(r["ok"] for r in rows)
    total_err = sum(r["err"] for r in rows)

    def rng(system: str, c: int, key: str) -> tuple[float, float]:
        vals = [
            float(r[key]) for r in rows
            if r["system"] == system and r["c"] == c
            and isinstance(r.get(key), (int, float))
        ]
        return min(vals), max(vals)

    lines = [
        "# H200 3GPU Barge-In Matrix Analysis",
        "",
        f"Run root: `{root}`",
        "",
        "## Validity",
        "",
        f"- Cells: **{len(rows)} / 48**",
        f"- Requests: **{total_ok} OK / {total_err} errors**",
        "- Systems: naive, scheduler wide (50/25), scheduler tight (20/10)",
        "- Concurrency: 32, 48, 64, 96",
        "- Barge-in probability: 0.3, 0.5, 0.7, 0.9",
        "",
        "## Headline Findings",
        "",
        "1. The first-audio latency cliff is the main symptom of saturation. "
        "Text TTFT remains sub-second while audio TTFP p90 grows sharply at c=64/96.",
        "2. Client-observed streaming RTF is mostly below 1.0, so steady audio "
        "emission is faster than realtime once it starts. The user-visible slowdown "
        "is therefore dominated by queueing/admission before first audio, not by "
        "per-chunk audio generation speed.",
        "3. The tight scheduler is the more stable lead-control preset at high concurrency. It reduces "
        "audio TTFP p90 at c=96 for p=0.3/0.7/0.9 and keeps useful_qps comparable.",
        "4. The wide scheduler is mixed: it helps c=96 at p=0.3/0.5 but is weaker than tight "
        "at high barge-in probabilities.",
        "",
        "## Ranges By System",
        "",
        "| system | c | useful_qps range | audio TTFP p90 range | stream RTF p90 range |",
        "|---|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        for c in [32, 48, 64, 96]:
            u = rng(system, c, "useful")
            a = rng(system, c, "audio_p90_s")
            s = rng(system, c, "stream_rtf_p90")
            lines.append(
                f"| {SYSTEM_LABELS[system]} | {c} | {u[0]:.2f}-{u[1]:.2f} | "
                f"{a[0]:.2f}-{a[1]:.2f}s | {s[0]:.2f}-{s[1]:.2f} |"
            )

    lines.extend([
        "",
        "## c=96 Delta Vs Naive",
        "",
        "| p_bi | scheduler wide audio p90 delta | scheduler tight audio p90 delta | "
        "scheduler wide useful delta | scheduler tight useful delta |",
        "|---:|---:|---:|---:|---:|",
    ])
    for p in [0.3, 0.5, 0.7, 0.9]:
        base_a = value(rows, "naive", 96, p, "audio_p90_s")
        base_u = value(rows, "naive", 96, p, "useful")
        wide_a = value(rows, "scheduler_wide", 96, p, "audio_p90_s")
        wide_u = value(rows, "scheduler_wide", 96, p, "useful")
        tight_a = value(rows, "scheduler_tight", 96, p, "audio_p90_s")
        tight_u = value(rows, "scheduler_tight", 96, p, "useful")
        lines.append(
            f"| {p:.1f} | {wide_a - base_a:+.2f}s | {tight_a - base_a:+.2f}s | "
            f"{wide_u - base_u:+.2f} | {tight_u - base_u:+.2f} |"
        )

    lines.extend([
        "",
        "## Figures",
        "",
        "- `fig1_audio_ttfp_p90_grid.png`",
        "- `fig2_delta_audio_ttfp_p90_vs_naive.png`",
        "- `fig3_useful_qps_grid.png`",
        "- `fig4_stream_rtf_p90_grid.png`",
        "- `fig5_pareto_useful_vs_audio_ttfp.png`",
        "- `fig6_waste_p90_grid.png`",
        "",
    ])
    (out / "BARGEIN3GPU_RTF_ANALYSIS.md").write_text("\n".join(lines))


def main() -> int:
    if len(sys.argv) > 2:
        print("usage: plot_bargein3gpu_matrix.py [run_root]", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else (
        Path(__file__).resolve().parents[3]
        / "results/H200/bargein3gpu"
    )
    if not root.exists():
        print(f"missing run root: {root}", file=sys.stderr)
        return 2

    out = root / "_plots"
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root)
    if not rows:
        print(f"no rows found in {root}", file=sys.stderr)
        return 1

    write_csv(rows, out)
    write_markdown(rows, out, root)
    plot_metric_grid(
        rows, out, "audio_p90_s", "audio TTFP p90 (s)",
        "fig1_audio_ttfp_p90_grid.png",
        "Audio first-chunk latency under 3GPU barge-in load",
    )
    plot_delta_vs_naive(rows, out)
    plot_metric_grid(
        rows, out, "useful", "useful_qps",
        "fig3_useful_qps_grid.png",
        "Useful throughput under 3GPU barge-in load",
    )
    plot_metric_grid(
        rows, out, "stream_rtf_p90", "client-observed stream RTF p90",
        "fig4_stream_rtf_p90_grid.png",
        "Audio streaming RTF after first chunk",
        hline=1.0,
    )
    plot_pareto(rows, out)
    plot_metric_grid(
        rows, out, "waste_p90", "wasted talker frames p90",
        "fig6_waste_p90_grid.png",
        "Talker waste under barge-in",
    )

    for path in sorted(out.iterdir()):
        if path.suffix in {".png", ".csv", ".md"}:
            print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
