#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


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


def find_summary(root: Path, variant: str) -> Path:
    patterns = [
        f"{variant}/*/summary.json",
        f"*/{variant}*/summary.json",
    ]
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no {variant} summary.json under {root}")


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
        "e2e_p99": summary.get("latency", {}).get("e2e_sec", {}).get("p99"),
        "continuity": summary.get("audio", {}).get("streaming_continuity", {}).get("ok_rate"),
        "lru_evictions": summary.get("kv_lru", {}).get("gpu_evictions"),
        "recent_miss": summary.get("kv_lru", {}).get("recent_eviction_misses"),
    }


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def pct_gain(new: Any, old: Any, *, lower_is_better: bool = False) -> float | None:
    if not finite(new) or not finite(old) or float(old) == 0.0:
        return None
    value = (float(old) - float(new)) / float(old) if lower_is_better else (
        float(new) - float(old)
    ) / float(old)
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
            snap["load1"]
            for snap in snapshots
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


def summarize_shadow(root: Path) -> dict[str, Any]:
    path = root / "eviction_only" / "_probe" / "kv_eviction_shadow.jsonl"
    if not path.exists():
        matches = sorted(root.glob("**/kv_eviction_shadow.jsonl"))
        path = matches[0] if matches else path
    if not path.exists():
        return {"path": None, "events": 0, "applied": 0, "policy_counts": {}}
    counts: Counter[str] = Counter()
    events = 0
    applied = 0
    replacements = 0
    reasons: Counter[str] = Counter()
    for row in iter_jsonl(path):
        events += 1
        policy = row.get("policy")
        if isinstance(policy, str) and policy:
            counts[policy] += 1
        reason = row.get("reason")
        if isinstance(reason, str):
            reasons[reason] += 1
        if row.get("applied") is True or reason == "applied":
            applied += 1
            replacements += int(row.get("replacement_count") or 0)
    return {
        "path": str(path),
        "events": events,
        "applied": applied,
        "replacements": replacements,
        "policy_counts": dict(counts.most_common()),
        "reason_counts": dict(reasons.most_common(8)),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def evaluate_run(label: str, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    candidate = metrics_from_summary(find_summary(root, "eviction_only"))
    baseline = metrics_from_summary(find_summary(root, "lru_control"))
    run_config_path = root / "run_config.json"
    run_config = read_json(run_config_path) if run_config_path.exists() else {}
    quality = summarize_quality(root)
    shadow = summarize_shadow(root)

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
    if goodput_gain is None or goodput_gain < args.min_goodput_gain:
        failures.append(f"goodput gain {fmt(goodput_gain)} < {args.min_goodput_gain}")
    if audio_reduction is None or audio_reduction < args.min_audio_p90_reduction:
        failures.append(f"audio p90 reduction {fmt(audio_reduction)} < {args.min_audio_p90_reduction}")
    if e2e_reduction is None or e2e_reduction < args.min_e2e_p90_reduction:
        failures.append(f"e2e p90 reduction {fmt(e2e_reduction)} < {args.min_e2e_p90_reduction}")
    if continuity_delta_pp is None or continuity_delta_pp < args.min_continuity_delta_pp:
        failures.append(
            f"continuity delta {fmt(continuity_delta_pp)}pp < {args.min_continuity_delta_pp}pp"
        )
    if args.expected_policy:
        policy_counts = shadow.get("policy_counts") or {}
        if args.expected_policy not in policy_counts:
            failures.append(f"expected shadow policy {args.expected_policy!r} not observed")
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
        "root": str(root),
        "variant_order": run_config.get("variants"),
        "policy": run_config.get("evict_policy"),
        "baseline": baseline,
        "candidate": candidate,
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


def write_markdown(path: Path, runs: list[dict[str, Any]], overall: dict[str, Any]) -> None:
    lines = [
        "# Eviction Order Robustness",
        "",
        f"- decision: `{'PASS' if overall['passed'] else 'REJECT'}`",
        f"- min score: `{fmt(overall['min_score_pct'])}%`",
        f"- average score: `{fmt(overall['average_score_pct'])}%`",
        "",
        "| run | order | goodput gain | audio p90 reduction | e2e p90 reduction | continuity delta | score | decision |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        deltas = run["deltas"]
        lines.append(
            f"| {run['label']} | `{run.get('variant_order')}` | "
            f"{fmt(deltas['goodput_gain_pct'])}% | "
            f"{fmt(deltas['audio_p90_reduction_pct'])}% | "
            f"{fmt(deltas['e2e_p90_reduction_pct'])}% | "
            f"{fmt(deltas['continuity_delta_pp'])}pp | "
            f"{fmt(deltas['three_metric_score_pct'])}% | "
            f"`{'PASS' if run['passed'] else 'REJECT'}` |"
        )
    failures = [(run["label"], failure) for run in runs for failure in run["failures"]]
    failures.extend(("overall", failure) for failure in overall["failures"])
    if failures:
        lines.extend(["", "## Failures", ""])
        for label, failure in failures:
            lines.append(f"- `{label}`: {failure}")
    lines.extend(["", "## Candidate Metrics", ""])
    lines.append("| run | goodput | audio p90 | e2e p90 | e2e p99 | continuity | evictions | recent miss | shadow applied/repl |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        candidate = run["candidate"]
        shadow = run["shadow"]
        lines.append(
            f"| {run['label']} | {fmt(candidate['goodput'])} | "
            f"{fmt(candidate['audio_p90'])} | {fmt(candidate['e2e_p90'])} | "
            f"{fmt(candidate['e2e_p99'])} | {fmt(candidate['continuity'])} | "
            f"{fmt(candidate['lru_evictions'])} | {fmt(candidate['recent_miss'])} | "
            f"{fmt(shadow.get('applied'))}/{fmt(shadow.get('replacements'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_mapping, required=True)
    parser.add_argument("--expected-policy", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-requests", type=int, default=600)
    parser.add_argument("--min-goodput-gain", type=float, default=-5.0)
    parser.add_argument("--min-audio-p90-reduction", type=float, default=10.0)
    parser.add_argument("--min-e2e-p90-reduction", type=float, default=10.0)
    parser.add_argument("--min-continuity-delta-pp", type=float, default=-1.0)
    parser.add_argument("--min-score", type=float, default=30.0)
    parser.add_argument("--max-snapshot-load1", type=float, default=8.0)
    parser.add_argument("--max-matching-processes", type=int, default=0)
    parser.add_argument("--require-quality-snapshots", action="store_true")
    parser.add_argument("--fail-on-reject", action="store_true")
    args = parser.parse_args()

    runs = [
        evaluate_run(label, Path(raw_root), args)
        for label, raw_root in args.run
    ]
    scores = [
        run["deltas"]["three_metric_score_pct"]
        for run in runs
        if run["deltas"]["three_metric_score_pct"] is not None
    ]
    min_score = min(scores) if scores else None
    average_score = sum(scores) / len(scores) if scores else None
    failures: list[str] = []
    if min_score is None or min_score < args.min_score:
        failures.append(f"min score {fmt(min_score)} < {args.min_score}")
    if any(not run["passed"] for run in runs):
        failures.append("one or more runs rejected")
    overall = {
        "passed": not failures,
        "min_score_pct": min_score,
        "average_score_pct": average_score,
        "failures": failures,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "eviction_order_robustness.json"
    md_path = args.output_dir / "eviction_order_robustness.md"
    json_path.write_text(
        json.dumps({"runs": runs, "overall": overall}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(md_path, runs, overall)
    print(json_path)
    print(md_path)
    if args.fail_on_reject and not overall["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
