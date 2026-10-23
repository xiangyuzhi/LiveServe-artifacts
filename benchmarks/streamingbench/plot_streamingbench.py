#!/usr/bin/env python3
"""Plot StreamingBench LiveServe result directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def get_path(node: dict[str, Any], path: str) -> Any:
    cur: Any = node
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def load_summaries(paths: list[Path]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for path in paths:
        summary_path = path / "summary.json" if path.is_dir() else path
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as f:
            rows.append((summary_path.parent.name, json.load(f)))
    return rows


def bar_plot(labels: list[str], series: list[tuple[str, list[float | None]]], title: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4.2))
    width = 0.8 / max(1, len(series))
    xs = list(range(len(labels)))
    for idx, (name, values) in enumerate(series):
        vals = [0.0 if v is None else float(v) for v in values]
        offs = [x - 0.4 + width / 2 + idx * width for x in xs]
        ax.bar(offs, vals, width=width, label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", nargs="+", help="Result directories or summary.json files.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    rows = load_summaries([Path(p) for p in args.result_dirs])
    if not rows:
        raise SystemExit("no summary.json files found")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.result_dirs[0]).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [name for name, _summary in rows]

    bar_plot(
        labels,
        [
            ("e2e p50 ms", [(get_path(s, "latency.e2e_sec.p50") or 0) * 1000 for _, s in rows],
            ),
            ("e2e p90 ms", [(get_path(s, "latency.e2e_sec.p90") or 0) * 1000 for _, s in rows],
            ),
        ],
        "StreamingBench Video Latency",
        "milliseconds",
        out_dir / "fig1_video_latency.png",
    )
    bar_plot(
        labels,
        [
            ("audio RTF p50", [get_path(s, "audio.stream_rtf.p50") for _, s in rows]),
            ("audio RTF p90", [get_path(s, "audio.stream_rtf.p90") for _, s in rows]),
        ],
        "StreamingBench Audio Stream RTF",
        "RTF",
        out_dir / "fig2_video_audio_rtf.png",
    )
    bar_plot(
        labels,
        [
            ("pre-barge text chars p50", [get_path(s, "interrupt.pre_interrupt_text_chars.p50") for _, s in rows]),
            ("pre-barge audio sec p50", [get_path(s, "interrupt.pre_interrupt_audio_duration_sec.p50") for _, s in rows]),
        ],
        "Barge-in Waste Proxy",
        "chars / seconds",
        out_dir / "fig3_video_bargein_waste.png",
    )
    bar_plot(
        labels,
        [
            ("throughput qps", [get_path(s, "totals.throughput_qps") for _, s in rows]),
            ("accuracy", [get_path(s, "totals.accuracy") for _, s in rows]),
        ],
        "Throughput and Accuracy",
        "qps / ratio",
        out_dir / "fig4_video_throughput_accuracy.png",
    )
    print(f"wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
