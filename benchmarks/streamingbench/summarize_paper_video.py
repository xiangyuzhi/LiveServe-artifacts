#!/usr/bin/env python3
"""Summarize and plot paper-style StreamingBench video matrix runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


NAME_RE = re.compile(r"g(?P<gpu>\d+)_(?P<variant>[A-Za-z0-9_]+)_(?P<mode>baseline|bargein|barge_in|interrupt)_c(?P<c>\d+)$")
VARIANT_ORDER = {
    "baseline_no_offload": 0,
    "baseline": 1,
    "liveserve_scheduler": 1,
    "scheduler": 1,
    "joint": 2,
    "joint_g1": 2,
    "liveserve_g1": 2,
    "liveserve_allopt": 2,
    "thinker2": 3,
}
COLORS = {
    "baseline_no_offload": "#8c8c8c",
    "baseline": "#2f6fed",
    "liveserve_scheduler": "#e15b31",
    "scheduler": "#e15b31",
    "joint": "#7b3fc7",
    "joint_g1": "#7b3fc7",
    "liveserve_g1": "#7b3fc7",
    "liveserve_allopt": "#7b3fc7",
    "thinker2": "#2f9e44",
}


def variants_in_order(rows: list[dict[str, Any]]) -> list[str]:
    variants = sorted(
        {r["variant"] for r in rows},
        key=lambda v: (VARIANT_ORDER.get(v, 99), v),
    )
    return variants


def get_path(node: dict[str, Any], path: str) -> Any:
    cur: Any = node
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def load_rows(result_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(result_root.glob("g*_*/summary.json")):
        match = NAME_RE.match(summary_path.parent.name)
        if not match:
            continue
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        row: dict[str, Any] = {
            "name": summary_path.parent.name,
            "gpu": int(match.group("gpu")),
            "variant": match.group("variant"),
            "mode": "bargein" if match.group("mode") in ("bargein", "barge_in", "interrupt") else match.group("mode"),
            "concurrency": int(match.group("c")),
            "requests": get_path(summary, "totals.requests"),
            "ok": get_path(summary, "totals.ok"),
            "errors": get_path(summary, "totals.errors"),
            "qps": get_path(summary, "totals.throughput_qps"),
            "accuracy": get_path(summary, "totals.accuracy"),
            "interrupted": get_path(summary, "totals.interrupted"),
            "text_ttft_p50": get_path(summary, "latency.text_ttft_sec.p50"),
            "audio_ttfp_p50": get_path(summary, "latency.audio_ttfp_sec.p50"),
            "audio_ttfp_p90": get_path(summary, "latency.audio_ttfp_sec.p90"),
            "e2e_p50": get_path(summary, "latency.e2e_sec.p50"),
            "e2e_p90": get_path(summary, "latency.e2e_sec.p90"),
            "stream_rtf_p50": get_path(summary, "audio.stream_rtf.p50"),
            "stream_rtf_p95": get_path(summary, "audio.stream_rtf.p95"),
            "rtf_pass_rate": get_path(summary, "audio.rtf_pass_rate"),
            "audio_chunks_mean": get_path(summary, "audio.chunks.mean"),
            "short_chunk_rate": get_path(summary, "audio.short_chunk_rate"),
            "audio_duration_mean": get_path(summary, "audio.duration_sec.mean"),
            "audio_duration_p50": get_path(summary, "audio.duration_sec.p50"),
            "continuity_evaluated": get_path(summary, "audio.streaming_continuity.n_evaluated"),
            "continuity_ok_rate": get_path(summary, "audio.streaming_continuity.ok_rate"),
            "continuity_goodput_rps": get_path(summary, "audio.streaming_continuity.goodput_rps"),
            "audio_underrun_mean": get_path(summary, "audio.streaming_continuity.audio_underrun_sec.mean"),
            "audio_underrun_p90": get_path(summary, "audio.streaming_continuity.audio_underrun_sec.p90"),
            "audio_underrun_p95": get_path(summary, "audio.streaming_continuity.audio_underrun_sec.p95"),
            "audio_underrun_events_mean": get_path(
                summary,
                "audio.streaming_continuity.audio_underrun_event_count.mean",
            ),
            "text_chars_mean": get_path(summary, "text.response_chars.mean"),
            "text_chars_p50": get_path(summary, "text.response_chars.p50"),
            "client_question_p50": get_path(summary, "client.client_question_sec.p50"),
            "frame_extract_p50": get_path(summary, "client.frame_extract_sec.p50"),
            "frame_send_p50": get_path(summary, "client.frame_send_sec.p50"),
            "query_collect_p50": get_path(summary, "client.query_collect_sec.p50"),
            "pre_int_audio_mean": get_path(summary, "interrupt.pre_interrupt_audio_duration_sec.mean"),
            "pre_int_audio_p50": get_path(summary, "interrupt.pre_interrupt_audio_duration_sec.p50"),
            "pre_int_text_mean": get_path(summary, "interrupt.pre_interrupt_text_chars.mean"),
            "kv_metadata_requests": get_path(summary, "kv.metadata_enabled_requests"),
            "kv_committed_prefix_requests": get_path(summary, "kv.with_committed_prefix"),
            "prefetch_attempts": get_path(summary, "kv.prefetch.attempts"),
            "prefetch_ok": get_path(summary, "kv.prefetch.ok"),
            "prefetch_failed_or_skipped": get_path(summary, "kv.prefetch.failed_or_skipped"),
            "prefetch_wall_p50": get_path(summary, "kv.prefetch.wall_sec.p50"),
            "dir": str(summary_path.parent),
        }
        rows.append(row)
    rows.sort(
        key=lambda r: (
            r["gpu"],
            r["mode"],
            r["concurrency"],
            VARIANT_ORDER.get(r["variant"], 99),
            r["variant"],
        )
    )
    return rows


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    fields = [
        "name",
        "gpu",
        "variant",
        "mode",
        "concurrency",
        "requests",
        "ok",
        "errors",
        "qps",
        "accuracy",
        "interrupted",
        "text_ttft_p50",
        "audio_ttfp_p50",
        "audio_ttfp_p90",
        "e2e_p50",
        "e2e_p90",
        "stream_rtf_p50",
        "stream_rtf_p95",
        "rtf_pass_rate",
        "audio_chunks_mean",
        "short_chunk_rate",
        "audio_duration_mean",
        "audio_duration_p50",
        "continuity_evaluated",
        "continuity_ok_rate",
        "continuity_goodput_rps",
        "audio_underrun_mean",
        "audio_underrun_p90",
        "audio_underrun_p95",
        "audio_underrun_events_mean",
        "text_chars_mean",
        "text_chars_p50",
        "client_question_p50",
        "frame_extract_p50",
        "frame_send_p50",
        "query_collect_p50",
        "pre_int_audio_mean",
        "pre_int_audio_p50",
        "pre_int_text_mean",
        "kv_metadata_requests",
        "kv_committed_prefix_requests",
        "prefetch_attempts",
        "prefetch_ok",
        "prefetch_failed_or_skipped",
        "prefetch_wall_p50",
        "dir",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], out_path: Path) -> None:
    headers = [
        "gpu",
        "variant",
        "mode",
        "c",
        "ok/req",
        "qps",
        "audio p50",
        "e2e p50",
        "audio dur mean",
        "cont ok",
        "underrun p95",
        "short chunk",
        "text chars mean",
        "rtf p95",
        "rtf pass",
        "pre-barge audio mean",
        "preload ok/att",
        "client p50",
    ]
    lines = [
        "# Paper StreamingBench Video Matrix Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        cells = [
            row["gpu"],
            row["variant"],
            row["mode"],
            row["concurrency"],
            f"{row['ok']}/{row['requests']}",
            fmt(row["qps"]),
            fmt(row["audio_ttfp_p50"]),
            fmt(row["e2e_p50"]),
            fmt(row["audio_duration_mean"]),
            fmt(row["continuity_ok_rate"]),
            fmt(row["audio_underrun_p95"]),
            fmt(row["short_chunk_rate"]),
            fmt(row["text_chars_mean"]),
            fmt(row["stream_rtf_p95"]),
            fmt(row["rtf_pass_rate"]),
            fmt(row["pre_int_audio_mean"]),
            f"{row.get('prefetch_ok')}/{row.get('prefetch_attempts')}",
            fmt(row["client_question_p50"]),
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_metric(rows: list[dict[str, Any]], metric: str, ylabel: str, title: str, out_path: Path) -> None:
    mode_order = {"baseline": 0, "bargein": 1, "interrupt": 1}
    modes = sorted({r["mode"] for r in rows}, key=lambda m: mode_order.get(m, 99))
    if not modes:
        return
    fig, axes_obj = plt.subplots(1, len(modes), figsize=(6.2 * len(modes), 4.4), sharey=False)
    axes = [axes_obj] if len(modes) == 1 else list(axes_obj)
    markers = {2: "o", 3: "s"}
    for ax, mode in zip(axes, modes, strict=True):
        subset = [r for r in rows if r["mode"] == mode]
        for gpu in sorted({r["gpu"] for r in subset}):
            for variant in variants_in_order(subset):
                pts = [r for r in subset if r["gpu"] == gpu and r["variant"] == variant and r.get(metric) is not None]
                pts.sort(key=lambda r: r["concurrency"])
                if not pts:
                    continue
                xs = [r["concurrency"] for r in pts]
                ys = [float(r[metric]) for r in pts]
                ax.plot(
                    xs,
                    ys,
                    marker=markers.get(gpu, "o"),
                    color=COLORS.get(variant),
                    linestyle="-" if gpu == 3 else "--",
                    label=f"{gpu}GPU {variant}",
                )
        ax.set_title(mode)
        ax.set_xlabel("concurrency")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overview(rows: list[dict[str, Any]], out_path: Path) -> None:
    interrupt_rows = [r for r in rows if r["mode"] in ("bargein", "interrupt")]
    if not interrupt_rows:
        return
    panels = [
        ("qps", "Throughput", "questions / second"),
        ("audio_ttfp_p50", "Audio TTFP p50", "seconds"),
        ("e2e_p50", "E2E p50", "seconds"),
        ("stream_rtf_p95", "Stream RTF p95", "RTF"),
    ]
    markers = {2: "o", 3: "s"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for ax, (metric, panel_title, ylabel) in zip(axes.ravel(), panels, strict=True):
        for gpu in sorted({r["gpu"] for r in interrupt_rows}):
            for variant in variants_in_order(interrupt_rows):
                pts = [
                    r for r in interrupt_rows
                    if r["gpu"] == gpu and r["variant"] == variant and r.get(metric) is not None
                ]
                pts.sort(key=lambda r: r["concurrency"])
                if not pts:
                    continue
                ax.plot(
                    [r["concurrency"] for r in pts],
                    [float(r[metric]) for r in pts],
                    marker=markers.get(gpu, "o"),
                    color=COLORS.get(variant),
                    linestyle="-" if gpu == 3 else "--",
                    label=f"{gpu}GPU {variant}",
                )
        ax.set_title(panel_title)
        ax.set_xlabel("concurrency")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(4, len(labels)),
            fontsize=9,
        )
    fig.suptitle("StreamingBench Multi-turn Video Barge-in Sweep", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_rps_ttfp(rows: list[dict[str, Any]], metric: str, metric_label: str, out_path: Path) -> None:
    plot_rows = [
        r for r in rows
        if r["mode"] in ("bargein", "interrupt") and r.get("qps") is not None and r.get(metric) is not None
    ]
    if not plot_rows:
        plot_rows = [
            r for r in rows
            if r.get("qps") is not None and r.get(metric) is not None
        ]
    if not plot_rows:
        return
    markers = {2: "o", 3: "s"}
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for gpu in sorted({r["gpu"] for r in plot_rows}):
        for variant in variants_in_order(plot_rows):
            pts = [
                r for r in plot_rows
                if r["gpu"] == gpu and r["variant"] == variant
            ]
            pts.sort(key=lambda r: r["concurrency"])
            if not pts:
                continue
            xs = [float(r[metric]) for r in pts]
            ys = [float(r["qps"]) for r in pts]
            ax.plot(
                xs,
                ys,
                marker=markers.get(gpu, "o"),
                color=COLORS.get(variant),
                linestyle="-" if gpu == 3 else "--",
                label=f"{gpu}GPU {variant}",
            )
            for row, x, y in zip(pts, xs, ys, strict=True):
                ax.annotate(
                    f"c={row['concurrency']}",
                    (x, y),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
    ax.set_xlabel(f"{metric_label} (s, lower is better)")
    ax.set_ylabel("requests / second (higher is better)")
    ax.set_title(f"StreamingBench Video Pareto: RPS vs {metric_label}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    out_dir = Path(args.out_dir) if args.out_dir else result_root / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(result_root)
    if not rows:
        raise SystemExit(f"no paper summary rows found under {result_root}")

    write_csv(rows, out_dir / "paper_summary.csv")
    write_markdown(rows, out_dir / "paper_summary.md")
    plot_metric(rows, "audio_ttfp_p50", "seconds", "Audio Time to First Packet", out_dir / "fig1_audio_ttfp_p50.png")
    plot_metric(rows, "qps", "questions / second", "Throughput", out_dir / "fig2_throughput.png")
    plot_metric(rows, "stream_rtf_p95", "RTF", "Streaming Audio RTF p95", out_dir / "fig3_stream_rtf_p95.png")
    plot_metric(rows, "pre_int_audio_mean", "seconds", "Pre-barge-in Audio Waste", out_dir / "fig4_bargein_waste.png")
    plot_metric(rows, "audio_duration_mean", "seconds", "Generated Audio Duration", out_dir / "fig5_audio_duration.png")
    plot_metric(rows, "short_chunk_rate", "fraction", "One-chunk Response Rate", out_dir / "fig6_short_chunk_rate.png")
    plot_metric(rows, "continuity_ok_rate", "fraction", "Audio Continuity OK Rate", out_dir / "fig7_continuity_ok_rate.png")
    plot_metric(rows, "audio_underrun_p95", "seconds", "Audio Underrun p95", out_dir / "fig8_audio_underrun_p95.png")
    plot_overview(rows, out_dir / "fig0_video_bargein_overview.png")
    plot_rps_ttfp(rows, "audio_ttfp_p50", "Audio TTFP p50", out_dir / "fig9_rps_vs_audio_ttfp_p50.png")
    plot_rps_ttfp(rows, "audio_ttfp_p90", "Audio TTFP p90", out_dir / "fig10_rps_vs_audio_ttfp_p90.png")
    print(f"wrote paper summary to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
