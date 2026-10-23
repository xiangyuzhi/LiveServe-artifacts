#!/usr/bin/env python3
"""Aggregate H200 3GPU barge-in matrix cells.

Usage:
    python aggregate_bargein3gpu.py results/H200/bargein3gpu/<run_id>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


TAG_RE = re.compile(
    r"^(?P<system>naive|scheduler_wide|scheduler_tight)_c(?P<c>\d+)_p(?P<p>\d+)_"
)


def get_path(node: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = node
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def ms(value: Any) -> str:
    return f"{value * 1000:.0f}" if isinstance(value, (int, float)) else "-"


def num(value: Any, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def parse_tag(tag: str) -> tuple[str, int, float]:
    match = TAG_RE.search(tag)
    if not match:
        return tag, -1, -1.0
    return match.group("system"), int(match.group("c")), int(match.group("p")) / 100.0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: aggregate_bargein3gpu.py <result_root>", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).resolve()
    if not root.exists():
        print(f"missing result root: {root}", file=sys.stderr)
        return 2

    rows = []
    for summary_path in root.rglob("summary.json"):
        tag = summary_path.parent.name
        system, c, prob = parse_tag(tag)
        with summary_path.open() as f:
            s = json.load(f)
        totals = s.get("totals", {})
        latency = s.get("latency", {})
        interrupt = s.get("interrupt", {})
        sanity = s.get("sanity_check", {})
        rows.append(
            {
                "tag": tag,
                "system": system,
                "c": c,
                "prob": prob,
                "ok": totals.get("completed_ok", 0),
                "err": totals.get("errors", 0),
                "interrupted": totals.get("interrupted", 0),
                "wall": totals.get("wall_time_sec", 0),
                "rps": totals.get("throughput_rps", 0),
                "useful": totals.get("useful_qps", 0),
                "text_p90": get_path(latency, "ttfp_sec.p90"),
                "audio_p90": get_path(latency, "audio_ttfp_sec.p90"),
                "e2e_p90": get_path(latency, "e2e_sec.p90"),
                "waste_p50": get_path(interrupt, "wasted_talker_frames_total.p50"),
                "waste_p90": get_path(interrupt, "wasted_talker_frames_total.p90"),
                "sanity": sanity.get("ratio_expected_over_actual"),
                "sanity_ok": sanity.get("within_tolerance"),
            }
        )

    order = {
        "naive": 0,
        "scheduler_wide": 1,
        "scheduler_tight": 2,
    }
    rows.sort(key=lambda r: (order.get(r["system"], 99), r["c"], r["prob"], r["tag"]))

    header = (
        f"{'system':<10} {'c':>4} {'p':>4} {'ok':>5} {'err':>3} {'int':>5} "
        f"{'wall_s':>7} {'rps':>6} {'useful':>7} {'txt90':>7} {'aud90':>7} "
        f"{'e2e90':>7} {'w50':>5} {'w90':>5} {'sanity':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        sanity = "-"
        if isinstance(r["sanity"], (int, float)):
            sanity = f"{r['sanity']:.2f}/{'OK' if r['sanity_ok'] else 'WRN'}"
        print(
            f"{r['system']:<10} {r['c']:>4} {r['prob']:>4.1f} {r['ok']:>5} {r['err']:>3} "
            f"{r['interrupted']:>5} {r['wall']:>7.1f} {r['rps']:>6.2f} "
            f"{r['useful']:>7.2f} {ms(r['text_p90']):>7} {ms(r['audio_p90']):>7} "
            f"{ms(r['e2e_p90']):>7} {num(r['waste_p50'], 0):>5} "
            f"{num(r['waste_p90'], 0):>5} {sanity:>8}"
        )

    print(f"\nRows completed: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
