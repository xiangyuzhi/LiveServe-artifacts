#!/usr/bin/env python3
"""Aggregate client-observable audio RTF metrics for H200 trace cells.

The main trace replay summary reports end-to-end latency. This helper focuses
on audio production speed:

* post_ttfp_rtf = (request_end - first_audio_chunk_time) / audio_duration
* stream_rtf    = (last_audio_chunk_time - first_audio_chunk_time)
                  / sum(audio_chunk_durations)

`stream_rtf` requires cells produced with `--record-audio-timeline` or newer
per-request fields. Older cells still get `post_ttfp_rtf` and the legacy
`e2e/audio_duration` proxy.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


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


def fmt(value: Any, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def ms(value: Any) -> str:
    return f"{value * 1000:.0f}" if isinstance(value, (int, float)) else "-"


def get_path(node: dict[str, Any], path: str, default=None):
    cur: Any = node
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_tag(tag: str) -> tuple[str, int, float]:
    m = re.match(r"(.+)_c(\d+)_p(\d+)_win", tag)
    if not m:
        return tag, -1, -1.0
    return m.group(1), int(m.group(2)), int(m.group(3)) / 100.0


def load_rtf_metrics(cell_dir: Path) -> dict[str, Any]:
    per_req = cell_dir / "per_request.jsonl"
    legacy_proxy: list[float] = []
    post_ttfp: list[float] = []
    stream: list[float] = []
    n_ok = 0
    n_non_interrupted = 0

    if not per_req.exists():
        return {
            "n_ok": 0,
            "n_non_interrupted": 0,
            "legacy": {},
            "post": {},
            "stream": {},
        }

    with per_req.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") is not None:
                continue
            n_ok += 1
            if row.get("interrupted"):
                continue
            n_non_interrupted += 1

            e2e = row.get("e2e_sec")
            audio_ttfp = row.get("audio_ttfp_sec")
            audio_dur = row.get("audio_duration_sec")
            timeline = row.get("audio_timeline") or []
            chunk_durations = [
                item.get("audio_duration_sec") for item in timeline
                if isinstance(item, dict)
                and isinstance(item.get("audio_duration_sec"), (int, float))
                and item.get("audio_duration_sec") > 0
            ]
            # Newer Qwen3-Omni streams emit each chunk as an independent WAV
            # fragment, so summing timeline durations is the best client-side
            # audio-duration estimate. Older cells without timeline fall back
            # to the legacy per-request duration field.
            if chunk_durations:
                audio_dur_effective = sum(chunk_durations)
            elif isinstance(audio_dur, (int, float)) and audio_dur > 0:
                audio_dur_effective = audio_dur
            else:
                continue

            if isinstance(e2e, (int, float)):
                legacy_proxy.append(e2e / audio_dur_effective)

            if (isinstance(e2e, (int, float))
                    and isinstance(audio_ttfp, (int, float))):
                post_ttfp.append(
                    max(0.0, e2e - audio_ttfp) / audio_dur_effective)

            if len(timeline) >= 2:
                first = timeline[0]
                last = timeline[-1]
                t0 = first.get("recv_sec")
                t1 = last.get("recv_sec")
                if all(isinstance(x, (int, float)) for x in (t0, t1)):
                    stream.append(
                        max(0.0, t1 - t0) / audio_dur_effective)

    return {
        "n_ok": n_ok,
        "n_non_interrupted": n_non_interrupted,
        "legacy": percentiles(legacy_proxy),
        "post": percentiles(post_ttfp),
        "stream": percentiles(stream),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: aggregate_audio_rtf.py <result_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    if not root.exists():
        print(f"missing result root: {root}", file=sys.stderr)
        return 2

    rows = []
    for summary_path in root.rglob("summary.json"):
        cell_dir = summary_path.parent
        with summary_path.open() as f:
            summary = json.load(f)
        system, c, p = parse_tag(cell_dir.name)
        rtf = load_rtf_metrics(cell_dir)
        rows.append({
            "tag": cell_dir.name,
            "system": system,
            "c": c,
            "p": p,
            "ok": get_path(summary, "totals.completed_ok", 0),
            "err": get_path(summary, "totals.errors", 0),
            "aud50": get_path(summary, "latency.audio_ttfp_sec.p50"),
            "aud90": get_path(summary, "latency.audio_ttfp_sec.p90"),
            "legacy": rtf["legacy"],
            "post": rtf["post"],
            "stream": rtf["stream"],
            "n_non_interrupted": rtf["n_non_interrupted"],
        })

    rows.sort(key=lambda r: (r["system"], r["c"], r["p"], r["tag"]))
    header = (
        f"{'system':<12} {'c':>4} {'p':>4} {'ok':>5} {'err':>3} "
        f"{'nonint':>6} {'aud50':>7} {'aud90':>7} "
        f"{'post50':>7} {'post90':>7} {'stream50':>8} {'stream90':>8} "
        f"{'e2e/dur50':>9} {'e2e/dur90':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['system']:<12} {r['c']:>4} {r['p']:>4.1f} "
            f"{r['ok']:>5} {r['err']:>3} {r['n_non_interrupted']:>6} "
            f"{ms(r['aud50']):>7} {ms(r['aud90']):>7} "
            f"{fmt(r['post'].get('p50')):>7} {fmt(r['post'].get('p90')):>7} "
            f"{fmt(r['stream'].get('p50')):>8} {fmt(r['stream'].get('p90')):>8} "
            f"{fmt(r['legacy'].get('p50')):>9} {fmt(r['legacy'].get('p90')):>9}"
        )

    print()
    print("Notes:")
    print("  post_rtf excludes pre-first-audio waiting but includes final SSE/drain time.")
    print("  stream_rtf uses first/last audio chunk receive times and needs --record-audio-timeline.")
    print("  e2e/dur is the old end-to-end proxy and includes queueing + TTFP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
