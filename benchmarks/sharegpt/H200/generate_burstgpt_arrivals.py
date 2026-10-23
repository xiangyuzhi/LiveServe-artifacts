#!/usr/bin/env python3
"""Generate BurstGPT arrival manifests for ShareGPT audio experiments.

The ShareGPT driver consumes a lightweight JSONL manifest with ``t_sec``
offsets.  BurstGPT timestamps in the public trace are second-resolution and may
contain many requests with the same timestamp, so this tool optionally spreads
same-second ties deterministically before scaling the selected burst.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BURSTGPT_PATH = Path("/workspace/xyz/test/BurstGPT/data/BurstGPT_1.csv")


def passes_log_type(row: dict[str, str], log_type: str) -> bool:
    value = row.get("Log Type", "")
    if log_type == "all":
        return True
    if log_type == "api":
        return value == "API log"
    if log_type == "conversation":
        return value == "Conversation log"
    return False


def parse_float(row: dict[str, str], key: str) -> float:
    return float(str(row.get(key, "")).strip())


def parse_int(row: dict[str, str], key: str) -> int:
    return int(float(str(row.get(key, "")).strip()))


def find_peak_window(path: Path, log_type: str, window_sec: int) -> int:
    counts: Counter[int] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if not passes_log_type(row, log_type):
                continue
            try:
                ts = int(parse_float(row, "Timestamp"))
            except Exception:
                continue
            counts[(ts // window_sec) * window_sec] += 1
    if not counts:
        raise SystemExit("no BurstGPT rows available for peak-window selection")
    return counts.most_common(1)[0][0]


def find_random_active_window(
    path: Path,
    log_type: str,
    window_sec: int,
    rng: random.Random,
) -> int:
    counts: Counter[int] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if not passes_log_type(row, log_type):
                continue
            try:
                ts = int(parse_float(row, "Timestamp"))
            except Exception:
                continue
            counts[(ts // window_sec) * window_sec] += 1
    starts = sorted(start for start, count in counts.items() if count > 0)
    if not starts:
        raise SystemExit("no active BurstGPT windows found")
    return rng.choice(starts)


def load_rows(
    path: Path,
    *,
    log_type: str,
    window_start: int,
    window_sec: int,
) -> list[dict[str, Any]]:
    window_end = window_start + window_sec
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for trace_index, row in enumerate(csv.DictReader(f)):
            if not passes_log_type(row, log_type):
                continue
            try:
                timestamp = parse_float(row, "Timestamp")
                request_tokens = parse_int(row, "Request tokens")
                response_tokens = parse_int(row, "Response tokens")
                total_tokens = parse_int(row, "Total tokens")
            except Exception:
                continue
            if timestamp < window_start or timestamp >= window_end:
                continue
            rows.append(
                {
                    "trace_index": trace_index,
                    "timestamp": timestamp,
                    "request_tokens": request_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens,
                    "model": row.get("Model", ""),
                    "log_type": row.get("Log Type", ""),
                }
            )
    rows.sort(key=lambda item: (float(item["timestamp"]), int(item["trace_index"])))
    return rows


def add_smoothed_timestamps(rows: list[dict[str, Any]], *, spread_ties: bool) -> None:
    if not rows:
        return
    by_ts: dict[float, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_ts[float(row["timestamp"])].append(idx)
    for timestamp, indices in by_ts.items():
        if spread_ties and len(indices) > 1:
            for rank, idx in enumerate(indices):
                rows[idx]["smoothed_timestamp"] = timestamp + (rank + 0.5) / len(indices)
        else:
            for idx in indices:
                rows[idx]["smoothed_timestamp"] = timestamp


def select_slice(
    rows: list[dict[str, Any]],
    *,
    num_requests: int,
    mode: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if len(rows) <= num_requests:
        return list(rows)
    if mode == "first":
        return list(rows[:num_requests])
    if mode == "random":
        start = rng.randint(0, len(rows) - num_requests)
        return list(rows[start : start + num_requests])
    if mode == "uniform":
        if num_requests == 1:
            return [rows[0]]
        step = (len(rows) - 1) / float(num_requests - 1)
        return [rows[int(round(i * step))] for i in range(num_requests)]
    if mode != "densest":
        raise SystemExit(f"unsupported slice mode: {mode}")

    best_start = 0
    best_span = math.inf
    for start in range(0, len(rows) - num_requests + 1):
        end = start + num_requests - 1
        span = float(rows[end]["smoothed_timestamp"]) - float(rows[start]["smoothed_timestamp"])
        if span < best_span:
            best_span = span
            best_start = start
    return list(rows[best_start : best_start + num_requests])


def peak_rate(rows: list[dict[str, Any]], bin_sec: float) -> float:
    if bin_sec <= 0:
        raise SystemExit("--peak-bin-sec must be > 0")
    if not rows:
        return 0.0
    first = min(float(row["smoothed_timestamp"]) for row in rows)
    counts: Counter[int] = Counter()
    for row in rows:
        rel = max(0.0, float(row["smoothed_timestamp"]) - first)
        counts[int(rel // bin_sec)] += 1
    return max(counts.values()) / bin_sec if counts else 0.0


def compute_scale(
    rows: list[dict[str, Any]],
    *,
    scale: float | None,
    target_rps: float | None,
    target_peak_rps: float | None,
    peak_bin_sec: float,
) -> tuple[float, dict[str, Any]]:
    specified = sum(value is not None for value in (scale, target_rps, target_peak_rps))
    if specified > 1:
        raise SystemExit("use only one of --scale, --target-rps, or --target-peak-rps")
    first = min(float(row["smoothed_timestamp"]) for row in rows)
    last = max(float(row["smoothed_timestamp"]) for row in rows)
    raw_duration = max(1e-9, last - first)
    raw_rps = len(rows) / raw_duration
    raw_peak = peak_rate(rows, peak_bin_sec)

    if target_rps is not None:
        if target_rps <= 0:
            raise SystemExit("--target-rps must be > 0")
        actual_scale = target_rps / raw_rps
        scale_source = "target_rps"
    elif target_peak_rps is not None:
        if target_peak_rps <= 0:
            raise SystemExit("--target-peak-rps must be > 0")
        if raw_peak <= 0:
            raise SystemExit("cannot compute peak scale from empty rows")
        actual_scale = target_peak_rps / raw_peak
        scale_source = "target_peak_rps"
    else:
        actual_scale = 1.0 if scale is None else scale
        scale_source = "scale"

    if actual_scale <= 0:
        raise SystemExit("arrival scale must be > 0")
    return actual_scale, {
        "raw_duration_sec": raw_duration,
        "raw_rps": raw_rps,
        "raw_peak_rps": raw_peak,
        "peak_bin_sec": peak_bin_sec,
        "scale_source": scale_source,
    }


def write_manifest(
    rows: list[dict[str, Any]],
    *,
    output: Path,
    scale: float,
    window_start: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    first = min(float(row["smoothed_timestamp"]) for row in rows)
    with output.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            raw_offset = max(0.0, float(row["smoothed_timestamp"]) - first)
            t_sec = raw_offset / scale
            out = {
                "idx": idx,
                "t_sec": t_sec,
                "arrival_time_s": t_sec,
                "raw_timestamp": row["timestamp"],
                "smoothed_timestamp": row["smoothed_timestamp"],
                "raw_window_start": window_start,
                "trace_index": row["trace_index"],
                "request_tokens": row["request_tokens"],
                "response_tokens": row["response_tokens"],
                "total_tokens": row["total_tokens"],
                "model": row["model"],
                "log_type": row["log_type"],
            }
            f.write(json.dumps(out, sort_keys=True) + "\n")


def scaled_peak_rate(
    rows: list[dict[str, Any]],
    *,
    scale: float,
    peak_bin_sec: float,
) -> float:
    if peak_bin_sec <= 0:
        raise SystemExit("--peak-bin-sec must be > 0")
    if not rows:
        return 0.0
    first = min(float(row["smoothed_timestamp"]) for row in rows)
    counts: Counter[int] = Counter()
    for row in rows:
        raw_offset = max(0.0, float(row["smoothed_timestamp"]) - first)
        t_sec = raw_offset / scale
        counts[int(t_sec // peak_bin_sec)] += 1
    return max(counts.values()) / peak_bin_sec if counts else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--burstgpt-csv", type=Path, default=DEFAULT_BURSTGPT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta-output", type=Path, default=None)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--log-type", choices=["all", "api", "conversation"], default="api")
    parser.add_argument(
        "--window-mode",
        choices=["peak", "explicit", "random-active"],
        default="peak",
    )
    parser.add_argument("--window-start", type=int, default=None)
    parser.add_argument("--window-sec", type=int, default=300)
    parser.add_argument(
        "--slice-mode",
        choices=["densest", "first", "uniform", "random"],
        default="densest",
    )
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--target-rps", type=float, default=None)
    parser.add_argument("--target-peak-rps", type=float, default=None)
    parser.add_argument("--peak-bin-sec", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-spread-ties",
        action="store_true",
        help="Keep identical second-resolution BurstGPT timestamps co-located.",
    )
    args = parser.parse_args()

    if args.num_requests <= 0:
        raise SystemExit("--num-requests must be > 0")
    if args.window_sec <= 0:
        raise SystemExit("--window-sec must be > 0")

    rng = random.Random(args.seed)
    if args.window_mode == "peak":
        window_start = find_peak_window(args.burstgpt_csv, args.log_type, args.window_sec)
    elif args.window_mode == "random-active":
        window_start = find_random_active_window(
            args.burstgpt_csv,
            args.log_type,
            args.window_sec,
            rng,
        )
    else:
        if args.window_start is None:
            raise SystemExit("--window-start is required when --window-mode=explicit")
        window_start = args.window_start

    raw_rows = load_rows(
        args.burstgpt_csv,
        log_type=args.log_type,
        window_start=window_start,
        window_sec=args.window_sec,
    )
    if not raw_rows:
        raise SystemExit("no BurstGPT rows selected")

    add_smoothed_timestamps(raw_rows, spread_ties=not args.no_spread_ties)
    rows = select_slice(
        raw_rows,
        num_requests=args.num_requests,
        mode=args.slice_mode,
        rng=rng,
    )
    if len(rows) < args.num_requests:
        raise SystemExit(
            f"selected only {len(rows)} rows from BurstGPT window; "
            f"requested {args.num_requests}"
        )

    scale, scale_meta = compute_scale(
        rows,
        scale=args.scale,
        target_rps=args.target_rps,
        target_peak_rps=args.target_peak_rps,
        peak_bin_sec=args.peak_bin_sec,
    )
    write_manifest(rows, output=args.output, scale=scale, window_start=window_start)

    first_offset = min(float(row["smoothed_timestamp"]) for row in rows)
    last_offset = max(float(row["smoothed_timestamp"]) for row in rows)
    scaled_duration = max(0.0, (last_offset - first_offset) / scale)
    scaled_peak = scaled_peak_rate(rows, scale=scale, peak_bin_sec=args.peak_bin_sec)
    meta = {
        "burstgpt_csv": str(args.burstgpt_csv),
        "output": str(args.output),
        "num_requests": len(rows),
        "log_type": args.log_type,
        "window_mode": args.window_mode,
        "window_start": window_start,
        "window_sec": args.window_sec,
        "slice_mode": args.slice_mode,
        "spread_ties": not args.no_spread_ties,
        "scale": scale,
        "target_rps": args.target_rps,
        "target_peak_rps": args.target_peak_rps,
        "scaled_duration_sec": scaled_duration,
        "scaled_avg_rps": len(rows) / scaled_duration if scaled_duration > 0 else None,
        "scaled_peak_rps": scaled_peak,
        **scale_meta,
    }
    meta_output = args.meta_output or args.output.with_suffix(args.output.suffix + ".meta.json")
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    meta_output.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "[generate_burstgpt_arrivals] "
        f"wrote {len(rows)} rows to {args.output}; "
        f"window_start={window_start} scale={scale:.6g} "
        f"scaled_duration={scaled_duration:.3f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
