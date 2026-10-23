#!/usr/bin/env python3
"""Plot closed-loop ShareGPT barge-in baseline results.

Reads 18 cases (c=1/4/8 x barge-in probability 0/0.1/0.3/0.5/0.7/0.9)
from a benchmark run dir and produces a 2x3 figure plus individual PNGs.

Each subplot:
    x-axis: barge-in probability
    y-axis: a metric
    3 lines: c=1, c=4, c=8

Usage:
    python plot_b2_cl_bargein.py [RUN_DIR]
    (defaults to the latest bench_b2_cl_bargein_* dir under logs/bargein_C;
    historical bench_b2_clint_* directories are still accepted)
"""
import json
import os
import sys
import glob
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# LOG_ROOT defaults to the local barge-in logs dir.
# The old env var is still accepted for historical result directories.
_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.environ.get(
    "LIVESERVE_SHAREGPT_BARGEIN_LOG_ROOT",
    os.environ.get(
        "LIVESERVE_SHAREGPT_INTERRUPT_LOG_ROOT",
        os.path.join(_HERE, "logs", "bargein_C"),
    ),
)


def find_latest_run(root: str) -> str:
    cands = []
    for pattern in ("bench_b2_cl_bargein_*", "bench_b2_clint_*"):
        cands.extend(glob.glob(os.path.join(root, pattern)))
    if not cands:
        raise SystemExit(
            f"no bench_b2_cl_bargein_* or bench_b2_clint_* dir under {root}"
        )
    return max(cands, key=os.path.getmtime)


CASE_RE = re.compile(r"c(\d+)_n(\d+)_pint(\d+p\d+)$")


def parse_case_dir(name: str):
    m = CASE_RE.match(name)
    if not m:
        return None
    c = int(m.group(1))
    n = int(m.group(2))
    p_str = m.group(3).replace("p", ".")
    p_int = float(p_str)
    return c, n, p_int


def load_metrics(case_dir: str):
    sp = os.path.join(case_dir, "summary.json")
    if not os.path.exists(sp):
        return None
    s = json.load(open(sp))
    pr_path = os.path.join(case_dir, "per_request.jsonl")
    n_total = 0
    n_intr = 0
    if os.path.exists(pr_path):
        with open(pr_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                n_total += 1
                if r.get("interrupted"):
                    n_intr += 1
    actual_rate = (n_intr / n_total) if n_total else 0.0

    ttfp = (s.get("latency") or {}).get("audio_ttfp_ms") or {}
    rtf = s.get("audio_rtf") or {}
    e2e = s.get("e2e_sec") or {}
    th = s.get("throughput") or {}

    return {
        "ttfp_p50": ttfp.get("p50"),
        "ttfp_p90": ttfp.get("p90"),
        "ttfp_mean": ttfp.get("mean"),
        "rtf_p50": rtf.get("p50"),
        "rtf_p90": rtf.get("p90"),
        "rtf_mean": rtf.get("mean"),
        "e2e_p50": e2e.get("p50"),
        "qps": th.get("requests_per_sec"),
        "n_completed": (s.get("counts") or {}).get("completed_normally"),
        "actual_rate": actual_rate,
    }


def collect(run_dir: str):
    """returns: dict[c] -> list[(p_int, metrics)] sorted by p_int."""
    by_c = defaultdict(list)
    for entry in sorted(os.listdir(run_dir)):
        full = os.path.join(run_dir, entry)
        if not os.path.isdir(full):
            continue
        parsed = parse_case_dir(entry)
        if parsed is None:
            continue
        c, n, p_int = parsed
        m = load_metrics(full)
        if m is None:
            continue
        by_c[c].append((p_int, m))
    for c in by_c:
        by_c[c].sort(key=lambda kv: kv[0])
    return by_c


METRIC_PLOTS = [
    ("ttfp_p50", "TTFP.p50 (ms)", "TTFP p50 vs barge-in prob"),
    ("ttfp_p90", "TTFP.p90 (ms)", "TTFP p90 vs barge-in prob"),
    ("rtf_p50",  "RTF.p50",       "Audio RTF p50 vs barge-in prob"),
    ("rtf_p90",  "RTF.p90",       "Audio RTF p90 vs barge-in prob"),
    ("e2e_p50",  "E2E.p50 (s)",   "E2E p50 vs barge-in prob"),
    ("qps",      "QPS (req/s)",   "Throughput vs barge-in prob"),
]

C_COLORS = {1: "#1f77b4", 4: "#ff7f0e", 8: "#d62728"}
C_MARKERS = {1: "o", 4: "s", 8: "^"}


def plot_one(ax, by_c, key, ylabel, title):
    for c in sorted(by_c.keys()):
        rows = by_c[c]
        xs = [p for p, _ in rows]
        ys = [m[key] for _, m in rows]
        ax.plot(
            xs, ys,
            label=f"c={c}",
            color=C_COLORS.get(c, None),
            marker=C_MARKERS.get(c, "o"),
            linewidth=2.0,
            markersize=7,
        )
        for x, y in zip(xs, ys):
            if y is None:
                continue
            ax.annotate(
                f"{y:.2f}" if isinstance(y, float) and abs(y) < 10 else f"{y:.0f}",
                xy=(x, y), xytext=(0, 6), textcoords="offset points",
                fontsize=7, ha="center", color=C_COLORS.get(c, "black"), alpha=0.7,
            )
    ax.set_xlabel("barge-in probability  $p_{bi}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run(LOG_ROOT)
    print(f"[plot] run_dir = {run_dir}")
    by_c = collect(run_dir)
    if not by_c:
        raise SystemExit("no metrics collected")

    # ---- print a quick text table for sanity check ----
    print(f"\n{'c':>3} {'p_bi':>6} {'done':>5} {'bi%':>6} "
          f"{'TTFP.p50':>9} {'TTFP.p90':>9} "
          f"{'RTF.p50':>8} {'RTF.p90':>8} "
          f"{'E2E.p50':>8} {'qps':>6}")
    for c in sorted(by_c):
        for p, m in by_c[c]:
            print(f"{c:>3d} {p:>6.2f} {m['n_completed']:>5d} "
                  f"{m['actual_rate']*100:>5.0f}% "
                  f"{m['ttfp_p50']:>9.0f} {m['ttfp_p90']:>9.0f} "
                  f"{m['rtf_p50']:>8.3f} {m['rtf_p90']:>8.3f} "
                  f"{m['e2e_p50']:>8.2f} {m['qps']:>6.3f}")

    # ---- combined 2x3 figure ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (key, ylabel, title) in zip(axes.flat, METRIC_PLOTS):
        plot_one(ax, by_c, key, ylabel, title)
    fig.suptitle(
        f"closed-loop ShareGPT barge-in (2 GPU)\n{os.path.basename(run_dir)}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_combined = os.path.join(run_dir, "plot_b2_cl_bargein_overview.png")
    fig.savefig(out_combined, dpi=140)
    print(f"[plot] saved {out_combined}")

    # ---- individual PNGs (handy for slides) ----
    for key, ylabel, title in METRIC_PLOTS:
        f, ax = plt.subplots(figsize=(7, 5))
        plot_one(ax, by_c, key, ylabel, title)
        f.tight_layout()
        out = os.path.join(run_dir, f"plot_b2_cl_bargein_{key}.png")
        f.savefig(out, dpi=140)
        plt.close(f)
        print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
