#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_run(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(raw)
    return path.name, path


def find_per_request(root_or_path: Path) -> Path:
    if root_or_path.is_file():
        return root_or_path
    patterns = [
        "eviction_only/*/per_request.jsonl",
        "lru_control/*/per_request.jsonl",
        "*/*/per_request.jsonl",
        "**/per_request.jsonl",
    ]
    for pattern in patterns:
        matches = sorted(root_or_path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no per_request.jsonl under {root_or_path}")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def trace_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("user_id"),
        row.get("round_index"),
        row.get("timestamp"),
        row.get("query_length"),
        row.get("response_length"),
    )


def numeric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[idx]


def bucket_value(value: float, cuts: list[float], labels: list[str]) -> str:
    for cut, label in zip(cuts, labels):
        if value < cut:
            return label
    return labels[-1]


def phase_bucket(seconds: float) -> str:
    return bucket_value(
        seconds,
        [10, 20, 40, 80, 120],
        ["0_10", "10_20", "20_40", "40_80", "80_120", "120_plus"],
    )


def request_groups(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_issue_rel: float,
    right_issue_rel: float,
    trace_rel: float,
) -> dict[str, str]:
    response = numeric(left, "response_length", numeric(right, "response_length"))
    query = numeric(left, "query_length", numeric(right, "query_length"))
    prefix = numeric(left, "kv_prefix_token_estimate", 0.0)
    timestamp = numeric(left, "timestamp", numeric(right, "timestamp"))
    round_index = numeric(left, "round_index", numeric(right, "round_index"))
    audio_duration = numeric(left, "audio_duration_sec", numeric(right, "audio_duration_sec"))
    return {
        "all": "all",
        "prefix": "returning" if prefix > 0 else "cold",
        "prefix_bucket": bucket_value(
            prefix,
            [1, 64, 128, 256],
            ["p0", "p1_63", "p64_127", "p128_255", "p256_plus"],
        ),
        "response_bucket": bucket_value(
            response,
            [16, 32, 64, 128],
            ["r_lt16", "r16_31", "r32_63", "r64_127", "r128_plus"],
        ),
        "query_bucket": bucket_value(
            query,
            [16, 32, 64, 128],
            ["q_lt16", "q16_31", "q32_63", "q64_127", "q128_plus"],
        ),
        "round_bucket": bucket_value(
            round_index,
            [1, 3, 10, 30],
            ["round0", "round1_2", "round3_9", "round10_29", "round30_plus"],
        ),
        "timestamp_bucket": bucket_value(
            timestamp,
            [6900, 7040, 7120],
            ["ts_lt6900", "ts6900_7039", "ts7040_7119", "ts7120_plus"],
        ),
        "trace_phase": phase_bucket(trace_rel),
        "left_issue_phase": phase_bucket(left_issue_rel),
        "right_issue_phase": phase_bucket(right_issue_rel),
        "audio_duration_bucket": bucket_value(
            audio_duration,
            [1, 4, 16, 64],
            ["audio_lt1", "audio1_4", "audio4_16", "audio16_64", "audio64_plus"],
        ),
    }


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "improved_rate": None,
        }
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p10": percentile(values, 0.10),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "improved_rate": sum(value < 0.0 for value in values) / len(values),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def read_requests(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("error") is not None:
            continue
        rows[trace_key(row)] = row
    return rows


def analyze(left_path: Path, right_path: Path, left_label: str, right_label: str) -> dict[str, Any]:
    left = read_requests(left_path)
    right = read_requests(right_path)
    common = sorted(set(left) & set(right))
    left_issue_t0 = min(
        (numeric(left[key], "issued_at") for key in common),
        default=0.0,
    )
    right_issue_t0 = min(
        (numeric(right[key], "issued_at") for key in common),
        default=0.0,
    )
    trace_t0 = min(
        (numeric(left[key], "timestamp", numeric(right[key], "timestamp")) for key in common),
        default=0.0,
    )
    group_values: dict[tuple[str, str], dict[str, list[float]]] = {}
    rows: list[dict[str, Any]] = []
    for key in common:
        lrow = left[key]
        rrow = right[key]
        left_issue_rel = numeric(lrow, "issued_at") - left_issue_t0
        right_issue_rel = numeric(rrow, "issued_at") - right_issue_t0
        trace_rel = numeric(lrow, "timestamp", numeric(rrow, "timestamp")) - trace_t0
        e2e_delta = numeric(rrow, "e2e_sec") - numeric(lrow, "e2e_sec")
        audio_delta = numeric(rrow, "audio_ttfp_sec") - numeric(lrow, "audio_ttfp_sec")
        row = {
            "key": key,
            "e2e_delta": e2e_delta,
            "audio_delta": audio_delta,
            "left_e2e": numeric(lrow, "e2e_sec"),
            "right_e2e": numeric(rrow, "e2e_sec"),
            "left_audio": numeric(lrow, "audio_ttfp_sec"),
            "right_audio": numeric(rrow, "audio_ttfp_sec"),
            "response_length": numeric(lrow, "response_length", numeric(rrow, "response_length")),
            "query_length": numeric(lrow, "query_length", numeric(rrow, "query_length")),
            "prefix_tokens": numeric(lrow, "kv_prefix_token_estimate", 0.0),
            "left_issue_rel": left_issue_rel,
            "right_issue_rel": right_issue_rel,
            "trace_rel": trace_rel,
        }
        rows.append(row)
        for group_name, group_value in request_groups(
            lrow,
            rrow,
            left_issue_rel=left_issue_rel,
            right_issue_rel=right_issue_rel,
            trace_rel=trace_rel,
        ).items():
            slot = group_values.setdefault((group_name, group_value), {"e2e": [], "audio": []})
            slot["e2e"].append(e2e_delta)
            slot["audio"].append(audio_delta)

    groups: list[dict[str, Any]] = []
    for (group_name, group_value), values in sorted(group_values.items()):
        e2e = summarize(values["e2e"])
        audio = summarize(values["audio"])
        groups.append(
            {
                "group": group_name,
                "value": group_value,
                "n": e2e["n"],
                "e2e_delta": e2e,
                "audio_delta": audio,
            }
        )

    return {
        "left_label": left_label,
        "right_label": right_label,
        "left_path": str(left_path),
        "right_path": str(right_path),
        "left_requests": len(left),
        "right_requests": len(right),
        "common_requests": len(common),
        "overall": {
            "e2e_delta": summarize([row["e2e_delta"] for row in rows]),
            "audio_delta": summarize([row["audio_delta"] for row in rows]),
        },
        "groups": groups,
        "best_groups_for_right": sorted(
            groups,
            key=lambda item: (
                item["e2e_delta"]["mean"] if item["e2e_delta"]["mean"] is not None else float("inf")
            ),
        )[:20],
        "worst_groups_for_right": sorted(
            groups,
            key=lambda item: (
                item["e2e_delta"]["mean"] if item["e2e_delta"]["mean"] is not None else float("-inf")
            ),
            reverse=True,
        )[:20],
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Eviction Policy Pair Deltas",
        "",
        f"- left: `{result['left_label']}`",
        f"- right: `{result['right_label']}`",
        "- deltas are `right - left`; negative means right is faster.",
        f"- common requests: `{result['common_requests']}`",
        "",
        "| metric | mean | p10 | p50 | p90 | improved_rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric_name in ("e2e_delta", "audio_delta"):
        summary = result["overall"][metric_name]
        lines.append(
            f"| {metric_name} | {fmt(summary['mean'])} | {fmt(summary['p10'])} | "
            f"{fmt(summary['p50'])} | {fmt(summary['p90'])} | "
            f"{fmt(summary['improved_rate'])} |"
        )
    for title, key in (
        ("Best Groups For Right", "best_groups_for_right"),
        ("Worst Groups For Right", "worst_groups_for_right"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| group | value | n | e2e_mean | e2e_p50 | e2e_p90 | improved_rate | audio_mean |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in result[key]:
            e2e = item["e2e_delta"]
            audio = item["audio_delta"]
            lines.append(
                f"| {item['group']} | {item['value']} | {item['n']} | "
                f"{fmt(e2e['mean'])} | {fmt(e2e['p50'])} | {fmt(e2e['p90'])} | "
                f"{fmt(e2e['improved_rate'])} | {fmt(audio['mean'])} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, help="label=run_root or per_request.jsonl")
    parser.add_argument("--right", required=True, help="label=run_root or per_request.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    left_label, left_root = parse_run(args.left)
    right_label, right_root = parse_run(args.right)
    result = analyze(
        find_per_request(left_root),
        find_per_request(right_root),
        left_label,
        right_label,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "policy_pair_deltas.json"
    md_path = args.output_dir / "policy_pair_deltas.md"
    csv_path = args.output_dir / "policy_pair_groups.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, result)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "value",
                "n",
                "e2e_mean",
                "e2e_p50",
                "e2e_p90",
                "e2e_improved_rate",
                "audio_mean",
                "audio_p50",
                "audio_p90",
                "audio_improved_rate",
            ],
        )
        writer.writeheader()
        for item in result["groups"]:
            e2e = item["e2e_delta"]
            audio = item["audio_delta"]
            writer.writerow(
                {
                    "group": item["group"],
                    "value": item["value"],
                    "n": item["n"],
                    "e2e_mean": e2e["mean"],
                    "e2e_p50": e2e["p50"],
                    "e2e_p90": e2e["p90"],
                    "e2e_improved_rate": e2e["improved_rate"],
                    "audio_mean": audio["mean"],
                    "audio_p50": audio["p50"],
                    "audio_p90": audio["p90"],
                    "audio_improved_rate": audio["improved_rate"],
                }
            )
    print(json_path)
    print(md_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
