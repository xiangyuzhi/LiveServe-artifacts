#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_EXPECTED_POLICIES = {
    "tiny": "expected_gap_bucket_response_tail_aware",
    "tiny300": "expected_gap_bucket_response_tail_aware",
    "nano": "expected_gap_cost_aware",
    "nano600": "expected_gap_cost_aware",
    "micro": "expected_gap_cost_aware",
    "micro1200": "expected_gap_cost_aware",
    "low": "expected_gap_bucket_response_tail_aware",
    "low2500": "expected_gap_bucket_response_tail_aware",
}


def parse_mapping(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"expected label=value, got {raw!r}")
    label, value = raw.split("=", 1)
    label = label.strip()
    value = value.strip()
    if not label or not value:
        raise argparse.ArgumentTypeError(f"expected label=value, got {raw!r}")
    return label, value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def find_summary(root: Path, variant: str | None) -> Path:
    if root.is_file() and root.name == "summary.json":
        return root
    patterns: list[str] = []
    if variant:
        patterns.extend([f"{variant}/*/summary.json", f"*/{variant}*/summary.json"])
    patterns.append("*/*/summary.json")
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no summary.json under {root}")


def find_shadow(root: Path) -> Path | None:
    if root.is_file() and root.name == "kv_eviction_shadow.jsonl":
        return root
    patterns = [
        "eviction_only/_probe/kv_eviction_shadow.jsonl",
        "*/_probe/kv_eviction_shadow.jsonl",
        "**/kv_eviction_shadow.jsonl",
    ]
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def metrics_from_summary(path: Path) -> dict[str, Any]:
    summary = read_json(path)
    return {
        "summary_path": str(path),
        "requests": summary.get("totals", {}).get("completed_ok"),
        "errors": summary.get("totals", {}).get("errors"),
        "rps": summary.get("totals", {}).get("throughput_rps"),
        "goodput": summary.get("audio", {}).get("streaming_continuity", {}).get("goodput_rps"),
        "audio_p90": summary.get("latency", {}).get("audio_ttfp_sec", {}).get("p90"),
        "e2e_p90": summary.get("latency", {}).get("e2e_sec", {}).get("p90"),
        "continuity": summary.get("audio", {}).get("streaming_continuity", {}).get("ok_rate"),
        "lru_evictions": summary.get("kv_lru", {}).get("gpu_evictions"),
        "recent_miss": summary.get("kv_lru", {}).get("recent_eviction_misses"),
    }


def pct_gain(new: Any, old: Any, *, lower_is_better: bool = False) -> float | None:
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)):
        return None
    if old == 0:
        return None
    value = (old - new) / old if lower_is_better else (new - old) / old
    return 100.0 * value


def parse_system_snapshot(path: Path) -> dict[str, Any]:
    load1: float | None = None
    matching_processes: list[str] = []
    in_processes = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("loadavg="):
            fields = line.removeprefix("loadavg=").split()
            if fields:
                try:
                    load1 = float(fields[0])
                except ValueError:
                    load1 = None
            continue
        if line == "matching_processes=":
            in_processes = True
            continue
        if line.endswith("="):
            in_processes = False
            continue
        if in_processes and line.strip():
            stripped = line.strip()
            if not is_self_quality_process(stripped):
                matching_processes.append(stripped)
    return {
        "path": str(path),
        "load1": load1,
        "matching_process_count": len(matching_processes),
        "matching_processes": matching_processes[:20],
    }


def is_self_quality_process(line: str) -> bool:
    return any(
        token in line
        for token in (
            "run_original_baseline_vs_eviction_gpu45.sh",
            "run_original_baseline_vs_eviction_gpu67.sh",
            "run_original_baseline_vs_eviction_gpuall8.sh",
            "run_eviction_fast_iter_gpu67.sh",
            "run_eviction_fast_iter_gpuall8.sh",
            "pgrep -af",
        )
    )


def summarize_quality(root: Path) -> dict[str, Any]:
    snapshots = [parse_system_snapshot(path) for path in sorted(root.glob("system_snapshot_*.txt"))]
    max_load1 = max(
        (
            snap["load1"] for snap in snapshots
            if isinstance(snap.get("load1"), (int, float))
        ),
        default=None,
    )
    max_matching = max(
        (int(snap.get("matching_process_count") or 0) for snap in snapshots),
        default=None if not snapshots else 0,
    )
    return {
        "snapshots": snapshots,
        "max_load1": max_load1,
        "max_matching_processes": max_matching,
    }


def summarize_shadow_policy(root: Path) -> dict[str, Any]:
    path = find_shadow(root)
    if path is None:
        return {"path": None, "events": 0, "policy_counts": {}}
    counts: Counter[str] = Counter()
    events = 0
    applied = 0
    for row in iter_jsonl(path):
        events += 1
        policy = row.get("policy")
        if isinstance(policy, str) and policy:
            counts[policy] += 1
        if row.get("applied") is True or row.get("reason") == "applied":
            applied += 1
    return {
        "path": str(path),
        "events": events,
        "applied": applied,
        "policy_counts": dict(counts.most_common()),
    }


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def evaluate_lane(
    label: str,
    run_root: Path,
    baseline_root: Path,
    expected_policy: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate_summary_path = find_summary(run_root, "eviction_only")
    baseline_summary_path = find_summary(baseline_root, "lru_control")
    candidate = metrics_from_summary(candidate_summary_path)
    baseline = metrics_from_summary(baseline_summary_path)
    quality = summarize_quality(run_root)
    shadow = summarize_shadow_policy(run_root)
    run_config_path = run_root / "run_config.json"
    run_config = read_json(run_config_path) if run_config_path.exists() else {}

    goodput_gain = pct_gain(candidate["goodput"], baseline["goodput"])
    audio_reduction = pct_gain(candidate["audio_p90"], baseline["audio_p90"], lower_is_better=True)
    e2e_reduction = pct_gain(candidate["e2e_p90"], baseline["e2e_p90"], lower_is_better=True)
    continuity_delta_pp = None
    if finite(candidate["continuity"]) and finite(baseline["continuity"]):
        continuity_delta_pp = 100.0 * (float(candidate["continuity"]) - float(baseline["continuity"]))
    score_parts = [x for x in (goodput_gain, audio_reduction, e2e_reduction) if x is not None]
    score = sum(score_parts) / len(score_parts) if score_parts else None

    failures: list[str] = []
    if candidate.get("errors") not in (0, None):
        failures.append(f"candidate errors={candidate.get('errors')}")
    if (candidate.get("requests") or 0) < args.min_requests:
        failures.append(f"candidate requests {candidate.get('requests')} < {args.min_requests}")
    if goodput_gain is None or goodput_gain < args.min_lane_goodput_gain:
        failures.append(f"goodput gain {fmt(goodput_gain)} < {args.min_lane_goodput_gain}")
    if audio_reduction is None or audio_reduction < args.min_lane_audio_p90_reduction:
        failures.append(f"audio p90 reduction {fmt(audio_reduction)} < {args.min_lane_audio_p90_reduction}")
    if e2e_reduction is None or e2e_reduction < args.min_lane_e2e_p90_reduction:
        failures.append(f"e2e p90 reduction {fmt(e2e_reduction)} < {args.min_lane_e2e_p90_reduction}")
    if continuity_delta_pp is None or continuity_delta_pp < args.min_continuity_delta_pp:
        failures.append(f"continuity delta {fmt(continuity_delta_pp)}pp < {args.min_continuity_delta_pp}pp")

    policy_counts = shadow.get("policy_counts") or {}
    if args.require_shadow_policy:
        if not policy_counts:
            failures.append("missing shadow policy events")
        elif expected_policy not in policy_counts:
            failures.append(f"expected effective policy {expected_policy!r} not observed in shadow log")
    if not args.allow_non_adaptive:
        evict_policy = run_config.get("evict_policy")
        if evict_policy != "expected_gap_capacity_adaptive_tail_aware":
            failures.append(f"run_config evict_policy={evict_policy!r} is not adaptive")
    if args.require_quality_snapshots and not quality["snapshots"]:
        failures.append("missing system snapshots")
    if quality["max_load1"] is not None and quality["max_load1"] > args.max_snapshot_load1:
        failures.append(f"max snapshot load1 {quality['max_load1']} > {args.max_snapshot_load1}")
    if (
        quality["max_matching_processes"] is not None
        and quality["max_matching_processes"] > args.max_matching_processes
    ):
        failures.append(
            f"max matching processes {quality['max_matching_processes']} > {args.max_matching_processes}"
        )

    return {
        "label": label,
        "run_root": str(run_root),
        "baseline_root": str(baseline_root),
        "expected_policy": expected_policy,
        "candidate": candidate,
        "baseline": baseline,
        "quality": quality,
        "shadow": shadow,
        "deltas": {
            "goodput_gain_pct": goodput_gain,
            "audio_p90_reduction_pct": audio_reduction,
            "e2e_p90_reduction_pct": e2e_reduction,
            "continuity_delta_pp": continuity_delta_pp,
            "three_metric_score_pct": score,
        },
        "passed": not failures,
        "failures": failures,
    }


def write_markdown(path: Path, lanes: list[dict[str, Any]], overall: dict[str, Any]) -> None:
    lines = [
        "# Adaptive Eviction Validation",
        "",
        f"- decision: `{'PASS' if overall['passed'] else 'REJECT'}`",
        f"- average three-metric score: `{fmt(overall['average_three_metric_score_pct'])}%`",
        "",
        "| lane | expected policy | requests | goodput gain | audio p90 reduction | e2e p90 reduction | continuity delta | score | decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for lane in lanes:
        candidate = lane["candidate"]
        deltas = lane["deltas"]
        lines.append(
            f"| {lane['label']} | `{lane['expected_policy']}` | "
            f"{candidate.get('requests')} | "
            f"{fmt(deltas['goodput_gain_pct'])}% | "
            f"{fmt(deltas['audio_p90_reduction_pct'])}% | "
            f"{fmt(deltas['e2e_p90_reduction_pct'])}% | "
            f"{fmt(deltas['continuity_delta_pp'])}pp | "
            f"{fmt(deltas['three_metric_score_pct'])}% | "
            f"`{'PASS' if lane['passed'] else 'REJECT'}` |"
        )
    failures = [(lane["label"], failure) for lane in lanes for failure in lane["failures"]]
    failures.extend(("overall", failure) for failure in overall["failures"])
    if failures:
        lines.extend(["", "## Failures", ""])
        for label, failure in failures:
            lines.append(f"- `{label}`: {failure}")
    lines.extend(["", "## Shadow Policies", ""])
    for lane in lanes:
        lines.append(
            f"- `{lane['label']}`: {lane['shadow'].get('policy_counts') or {}} "
            f"from `{lane['shadow'].get('path')}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", action="append", type=parse_mapping, required=True)
    parser.add_argument("--baseline", action="append", type=parse_mapping, required=True)
    parser.add_argument("--expected-policy", action="append", type=parse_mapping, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-requests", type=int, default=300)
    parser.add_argument("--min-lane-goodput-gain", type=float, default=-5.0)
    parser.add_argument("--min-lane-audio-p90-reduction", type=float, default=10.0)
    parser.add_argument("--min-lane-e2e-p90-reduction", type=float, default=10.0)
    parser.add_argument("--min-continuity-delta-pp", type=float, default=-1.0)
    parser.add_argument("--min-average-score", type=float, default=30.0)
    parser.add_argument("--max-snapshot-load1", type=float, default=8.0)
    parser.add_argument("--max-matching-processes", type=int, default=0)
    parser.add_argument("--require-quality-snapshots", action="store_true")
    parser.add_argument("--require-shadow-policy", action="store_true")
    parser.add_argument("--allow-non-adaptive", action="store_true")
    parser.add_argument("--fail-on-reject", action="store_true")
    args = parser.parse_args()

    baselines = {label: Path(path) for label, path in args.baseline}
    expected_overrides = {label: policy for label, policy in args.expected_policy}
    lanes: list[dict[str, Any]] = []
    for label, raw_root in args.lane:
        if label not in baselines:
            raise SystemExit(f"missing --baseline for lane {label!r}")
        expected = expected_overrides.get(label) or DEFAULT_EXPECTED_POLICIES.get(label)
        if expected is None:
            raise SystemExit(f"missing --expected-policy for unknown lane {label!r}")
        lanes.append(
            evaluate_lane(
                label,
                Path(raw_root),
                baselines[label],
                expected,
                args,
            )
        )

    scores = [
        lane["deltas"]["three_metric_score_pct"]
        for lane in lanes
        if lane["deltas"]["three_metric_score_pct"] is not None
    ]
    average_score = sum(scores) / len(scores) if scores else None
    overall_failures: list[str] = []
    if average_score is None or average_score < args.min_average_score:
        overall_failures.append(f"average score {fmt(average_score)} < {args.min_average_score}")
    if any(not lane["passed"] for lane in lanes):
        overall_failures.append("one or more lanes rejected")
    overall = {
        "passed": not overall_failures,
        "average_three_metric_score_pct": average_score,
        "failures": overall_failures,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "adaptive_eviction_validation.json"
    md_path = args.output_dir / "adaptive_eviction_validation.md"
    json_path.write_text(
        json.dumps({"lanes": lanes, "overall": overall}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(md_path, lanes, overall)
    print(json_path)
    print(md_path)
    if args.fail_on_reject and not overall["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
