#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXP_ROOT = Path("/workspace/xyz/exp_tmp_dir")


@dataclass(frozen=True)
class LaneSpec:
    name: str
    capacity_blocks: int
    baseline_root: str


@dataclass(frozen=True)
class CandidateSpec:
    lane: str
    label: str
    root: str
    notes: str = ""


DEFAULT_LANES = (
    LaneSpec(
        name="tiny300",
        capacity_blocks=750,
        baseline_root="interactive_original_gpu67_eviction_tinykv_bucket_cap1_20260613_021545",
    ),
    LaneSpec(
        name="nano600",
        capacity_blocks=2500,
        baseline_root="interactive_original_gpu67_eviction_nanokv_stress600_partialmax1_20260612_233019",
    ),
    LaneSpec(
        name="micro1200",
        capacity_blocks=5000,
        baseline_root="interactive_original_gpu67_eviction_microkv_session1200_partialmax1_20260612_225802",
    ),
    LaneSpec(
        name="low2500",
        capacity_blocks=9567,
        baseline_root="interactive_original_gpu67_eviction_lowkv_bucketresp_w30_w180_session_balanced2500_20260612_202231",
    ),
)


DEFAULT_CANDIDATES = (
    CandidateSpec(
        "tiny300",
        "bucket_cap1",
        "interactive_original_gpu67_eviction_tinykv_bucket_cap1_20260613_021545",
        "current robust tiny default",
    ),
    CandidateSpec(
        "tiny300",
        "near_response_cost",
        "interactive_original_gpu67_eviction_tinykv_near_response_cost_cap1_20260613_022730",
    ),
    CandidateSpec(
        "tiny300",
        "bucket_dynmax4",
        "interactive_original_gpu67_eviction_tinykv_bucket_dynmax4_20260613_023351",
    ),
    CandidateSpec(
        "tiny300",
        "response_cost_guard30",
        "interactive_original_gpu67_eviction_tinykv_respcost_guard30_20260613_024002",
    ),
    CandidateSpec(
        "tiny300",
        "cost_response_cap1",
        "interactive_original_gpu67_eviction_tinykv_cost_response_cap1_20260613_030612",
    ),
    CandidateSpec(
        "tiny300",
        "loss_tail_cap1",
        "interactive_original_gpu67_eviction_tinykv_loss_tail_cap1_20260613_032248",
    ),
    CandidateSpec(
        "tiny300",
        "bucket_phaseprof",
        "interactive_original_gpu67_eviction_tinykv_bucket_phaseprof_20260613_033220",
    ),
    CandidateSpec(
        "tiny300",
        "cost_phaseprof",
        "interactive_original_gpu67_eviction_tinykv_cost_phaseprof_20260613_033819",
    ),
    CandidateSpec(
        "nano600",
        "bucket_cap1",
        "interactive_original_gpu67_eviction_nanokv_stress600_partialmax1_20260612_233019",
        "matched LRU control root",
    ),
    CandidateSpec(
        "nano600",
        "partialmax2",
        "interactive_original_gpu67_eviction_nanokv_stress600_partialmax2_20260612_234510",
    ),
    CandidateSpec(
        "nano600",
        "partialmax2_gap60",
        "interactive_original_gpu67_eviction_nanokv_stress600_partialmax2_gap60_20260613_001034",
    ),
    CandidateSpec(
        "nano600",
        "dynamicmax2_gap30_matched_wait",
        "interactive_original_gpu67_eviction_nanokv_dynmax2_gap30_matched_wait_20260613_072102",
        "clean same-command matched nano rerun",
    ),
    CandidateSpec(
        "nano600",
        "cost_aware_cap1",
        "interactive_original_gpu67_eviction_nanokv_stress600_costaware_cap1_20260613_002704",
    ),
    CandidateSpec(
        "nano600",
        "bucket_cost_cap1",
        "interactive_original_gpu67_eviction_nanokv_stress600_bucketcost_cap1_20260613_005546",
    ),
    CandidateSpec(
        "nano600",
        "bucket_response_w10",
        "interactive_original_gpu67_eviction_nanokv_stress600_bucketresp_w10_cap1_20260613_010533",
    ),
    CandidateSpec(
        "nano600",
        "bucket_response_w20",
        "interactive_original_gpu67_eviction_nanokv_stress600_bucketresp_w20_cap1_20260613_011306",
    ),
    CandidateSpec(
        "nano600",
        "near_cost_cap1",
        "interactive_original_gpu67_eviction_nanokv_stress600_nearcost_cap1_20260613_013343",
    ),
    CandidateSpec(
        "nano600",
        "bucket_tail_cap1",
        "interactive_original_gpu67_eviction_nanokv_stress600_buckettail_cap1_20260613_014139",
    ),
    CandidateSpec(
        "nano600",
        "bucket_response_cost_cap1",
        "interactive_original_gpu67_eviction_nanokv_stress600_bucketrespcost_cap1_20260613_015335",
        "current nano adaptive branch",
    ),
    CandidateSpec(
        "nano600",
        "bucket_dynmax4",
        "interactive_original_gpu67_eviction_nanokv_stress600_bucket_dynmax4_20260613_024913",
    ),
    CandidateSpec(
        "nano600",
        "cost_response_cap1",
        "interactive_original_gpu67_eviction_nanokv_cost_response_cap1_20260613_043516",
    ),
    CandidateSpec(
        "micro1200",
        "bucket_cap1",
        "interactive_original_gpu67_eviction_microkv_session1200_partialmax1_20260612_225802",
        "matched LRU control root",
    ),
    CandidateSpec(
        "micro1200",
        "partialmax2",
        "interactive_original_gpu67_eviction_microkv_session1200_partialmax2_20260612_231811",
    ),
    CandidateSpec(
        "micro1200",
        "partialmax2_gap60",
        "interactive_original_gpu67_eviction_microkv_session1200_partialmax2_gap60_20260613_001822",
    ),
    CandidateSpec(
        "micro1200",
        "cost_aware_cap1",
        "interactive_original_gpu67_eviction_microkv_session1200_costaware_cap1_20260613_003427",
        "current micro adaptive branch",
    ),
    CandidateSpec(
        "micro1200",
        "bucket_response_cost_cap1",
        "interactive_original_gpu67_eviction_microkv_session1200_bucketrespcost_cap1_20260613_020120",
    ),
    CandidateSpec(
        "micro1200",
        "bucket_phaseprof",
        "interactive_original_gpu67_eviction_microkv_bucket_phaseprof_20260613_040953",
    ),
    CandidateSpec(
        "micro1200",
        "bucket_guard60",
        "interactive_original_gpu67_eviction_microkv_bucket_guard60_20260613_042230",
    ),
    CandidateSpec(
        "low2500",
        "bucket_w30",
        "interactive_original_gpu67_eviction_lowkv_bucketresp_w30_w180_session_balanced2500_20260612_202231",
        "matched LRU control root",
    ),
    CandidateSpec(
        "low2500",
        "bucket_w60",
        "interactive_original_gpu67_eviction_lowkv_bucketresp_w60_w180_session_balanced2500_20260612_205646",
    ),
    CandidateSpec(
        "low2500",
        "bucket_cap4",
        "interactive_original_gpu67_eviction_lowkv_bucketresp_w30_maxcached4_w180_session2500_20260612_215335",
    ),
    CandidateSpec(
        "low2500",
        "bucket_cap1_noscan",
        "interactive_original_gpu67_eviction_lowkv_bucketresp_w30_maxcached1_noscan_w180_session2500_20260612_220622",
    ),
    CandidateSpec(
        "low2500",
        "bucket_partial_cap1",
        "interactive_original_gpu67_eviction_lowkv_bucketresp_w30_partialmax1_noscan_w180_session2500_20260612_222223",
        "current low adaptive branch",
    ),
    CandidateSpec(
        "low2500",
        "cost_aware_cap1",
        "interactive_original_gpu67_eviction_lowkv_session2500_costaware_cap1_20260613_004220",
    ),
)


def resolve_root(root: str, exp_root: Path) -> Path:
    path = Path(root)
    if path.is_absolute():
        return path
    return exp_root / root


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
    if root.is_file() and root.name == "summary.json":
        return root
    patterns = [
        f"{variant}/*/summary.json",
        f"*/{variant}*/summary.json",
        "*/*/summary.json",
    ]
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no {variant} summary under {root}")


def find_shadow(root: Path) -> Path | None:
    for pattern in (
        "eviction_only/_probe/kv_eviction_shadow.jsonl",
        "*/_probe/kv_eviction_shadow.jsonl",
        "**/kv_eviction_shadow.jsonl",
    ):
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def metric(summary: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = summary
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def metrics_from_summary(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return {
        "summary_path": str(path),
        "requests": metric(data, ("totals", "completed_ok")),
        "errors": metric(data, ("totals", "errors")),
        "rps": metric(data, ("totals", "throughput_rps")),
        "goodput": metric(data, ("audio", "streaming_continuity", "goodput_rps")),
        "audio_p90": metric(data, ("latency", "audio_ttfp_sec", "p90")),
        "e2e_p90": metric(data, ("latency", "e2e_sec", "p90")),
        "continuity": metric(data, ("audio", "streaming_continuity", "ok_rate")),
        "kv_evictions": metric(data, ("kv_lru", "gpu_evictions")),
        "lookup_miss": metric(data, ("kv_lru", "gpu_lookup_misses")),
        "recent_miss": metric(data, ("kv_lru", "recent_eviction_misses")),
    }


def pct(new: Any, old: Any, *, lower_is_better: bool = False) -> float | None:
    if not finite(new) or not finite(old) or float(old) == 0.0:
        return None
    if lower_is_better:
        return 100.0 * (float(old) - float(new)) / float(old)
    return 100.0 * (float(new) - float(old)) / float(old)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(xs) if xs else None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize_shadow(root: Path) -> dict[str, Any]:
    path = find_shadow(root)
    if path is None:
        return {"path": None, "events": 0, "applied": 0, "policy_counts": {}}
    policies: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    events = 0
    applied = 0
    replacement_counts: list[float] = []
    for row in iter_jsonl(path):
        events += 1
        policy = row.get("policy")
        if isinstance(policy, str) and policy:
            policies[policy] += 1
        reason = str(row.get("reason") or "none")
        reasons[reason] += 1
        if row.get("applied") is True or reason == "applied":
            applied += 1
        replacement_count = row.get("replacement_count")
        if finite(replacement_count):
            replacement_counts.append(float(replacement_count))
    return {
        "path": str(path),
        "events": events,
        "applied": applied,
        "policy_counts": dict(policies.most_common()),
        "reason_counts": dict(reasons.most_common()),
        "replacement_count_p90": percentile(replacement_counts, 0.90),
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[idx]


def run_config(root: Path) -> dict[str, Any]:
    path = root / "run_config.json"
    return read_json(path) if path.exists() else {}


def evaluate_candidate(
    lane: LaneSpec,
    candidate: CandidateSpec,
    *,
    exp_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    root = resolve_root(candidate.root, exp_root)
    baseline_root = resolve_root(lane.baseline_root, exp_root)
    cand_summary = metrics_from_summary(find_summary(root, "eviction_only"))
    base_summary = metrics_from_summary(find_summary(baseline_root, "lru_control"))
    config = run_config(root)
    shadow = summarize_shadow(root)

    deltas = {
        "rps_gain_pct": pct(cand_summary["rps"], base_summary["rps"]),
        "goodput_gain_pct": pct(cand_summary["goodput"], base_summary["goodput"]),
        "audio_p90_reduction_pct": pct(
            cand_summary["audio_p90"],
            base_summary["audio_p90"],
            lower_is_better=True,
        ),
        "e2e_p90_reduction_pct": pct(
            cand_summary["e2e_p90"],
            base_summary["e2e_p90"],
            lower_is_better=True,
        ),
        "continuity_delta_pp": (
            100.0 * (float(cand_summary["continuity"]) - float(base_summary["continuity"]))
            if finite(cand_summary["continuity"]) and finite(base_summary["continuity"])
            else None
        ),
        "eviction_reduction_pct": pct(
            cand_summary["kv_evictions"],
            base_summary["kv_evictions"],
            lower_is_better=True,
        ),
        "recent_miss_reduction_pct": pct(
            cand_summary["recent_miss"],
            base_summary["recent_miss"],
            lower_is_better=True,
        ),
    }
    deltas["three_metric_score_pct"] = mean(
        [
            deltas["goodput_gain_pct"],
            deltas["audio_p90_reduction_pct"],
            deltas["e2e_p90_reduction_pct"],
        ]
    )
    cache_score = deltas["recent_miss_reduction_pct"]
    deltas["cache_weighted_score_pct"] = (
        None
        if deltas["three_metric_score_pct"] is None
        else float(deltas["three_metric_score_pct"]) + 0.25 * float(cache_score or 0.0)
    )

    failures: list[str] = []
    if cand_summary["errors"] not in (0, None):
        failures.append(f"errors={cand_summary['errors']}")
    if cand_summary["requests"] != base_summary["requests"]:
        failures.append(f"request mismatch candidate={cand_summary['requests']} baseline={base_summary['requests']}")
    if deltas["goodput_gain_pct"] is None or deltas["goodput_gain_pct"] < args.min_goodput_gain_pct:
        failures.append(f"goodput gain {fmt(deltas['goodput_gain_pct'])}% < {args.min_goodput_gain_pct}%")
    if deltas["audio_p90_reduction_pct"] is None or deltas["audio_p90_reduction_pct"] < args.min_audio_p90_reduction_pct:
        failures.append(f"audio p90 reduction {fmt(deltas['audio_p90_reduction_pct'])}% < {args.min_audio_p90_reduction_pct}%")
    if deltas["e2e_p90_reduction_pct"] is None or deltas["e2e_p90_reduction_pct"] < args.min_e2e_p90_reduction_pct:
        failures.append(f"e2e p90 reduction {fmt(deltas['e2e_p90_reduction_pct'])}% < {args.min_e2e_p90_reduction_pct}%")
    if deltas["continuity_delta_pp"] is None or deltas["continuity_delta_pp"] < args.min_continuity_delta_pp:
        failures.append(f"continuity delta {fmt(deltas['continuity_delta_pp'])}pp < {args.min_continuity_delta_pp}pp")
    repl_p90 = shadow.get("replacement_count_p90")
    if finite(repl_p90) and float(repl_p90) > args.max_replacement_count_p90:
        failures.append(
            f"replacement p90 {fmt(repl_p90)} > {args.max_replacement_count_p90}"
        )
    evict_reduction = deltas["eviction_reduction_pct"]
    if finite(evict_reduction) and float(evict_reduction) < args.min_eviction_reduction_pct:
        failures.append(
            f"eviction reduction {fmt(evict_reduction)}% < {args.min_eviction_reduction_pct}%"
        )

    return {
        "lane": lane.name,
        "capacity_blocks": lane.capacity_blocks,
        "label": candidate.label,
        "root": str(root),
        "baseline_root": str(baseline_root),
        "policy": config.get("evict_policy"),
        "gap_bucket_sec": config.get("evict_gap_bucket_sec"),
        "max_cached": config.get("evict_max_cached"),
        "partial_over_cap": config.get("evict_partial_over_cap"),
        "notes": candidate.notes,
        "candidate": cand_summary,
        "baseline": base_summary,
        "deltas": deltas,
        "shadow": shadow,
        "promotable": not failures,
        "failures": failures,
    }


def tie_break_key(row: dict[str, Any]) -> tuple[Any, ...]:
    deltas = row["deltas"]
    return (
        -float(deltas.get("recent_miss_reduction_pct") or -1e9),
        -float(deltas.get("eviction_reduction_pct") or -1e9),
        -float(deltas.get("goodput_gain_pct") or -1e9),
        -float(deltas.get("three_metric_score_pct") or -1e9),
        str(row["label"]),
    )


def select_lanes(rows: list[dict[str, Any]], epsilon: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_lane: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_lane.setdefault(row["lane"], []).append(row)
    for lane in sorted(by_lane, key=lambda name: by_lane[name][0]["capacity_blocks"]):
        lane_rows = by_lane[lane]
        promotable = [row for row in lane_rows if row["promotable"]]
        pool = promotable if promotable else lane_rows
        best_score = max(
            float(row["deltas"].get("three_metric_score_pct") or -1e9)
            for row in pool
        )
        close = [
            row for row in pool
            if float(row["deltas"].get("three_metric_score_pct") or -1e9)
            >= best_score - epsilon
        ]
        winner = dict(sorted(close, key=tie_break_key)[0])
        ranked = sorted(
            lane_rows,
            key=lambda row: (
                0 if row["promotable"] else 1,
                -float(row["deltas"].get("three_metric_score_pct") or -1e9),
                tie_break_key(row),
            ),
        )
        winner["ranked_candidates"] = [
            {
                "label": row["label"],
                "policy": row["policy"],
                "promotable": row["promotable"],
                "three_metric_score_pct": row["deltas"].get("three_metric_score_pct"),
                "goodput_gain_pct": row["deltas"].get("goodput_gain_pct"),
                "audio_p90_reduction_pct": row["deltas"].get("audio_p90_reduction_pct"),
                "e2e_p90_reduction_pct": row["deltas"].get("e2e_p90_reduction_pct"),
                "recent_miss_reduction_pct": row["deltas"].get("recent_miss_reduction_pct"),
                "failures": row["failures"],
            }
            for row in ranked
        ]
        selected.append(winner)
    return selected


def threshold_between(left_capacity: int, right_capacity: int) -> int:
    candidates = [1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000]
    for value in candidates:
        if left_capacity < value < right_capacity:
            return value
    return int((left_capacity + right_capacity) // 2)


def adaptive_segments(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not selected:
        return []
    ordered = sorted(selected, key=lambda row: int(row["capacity_blocks"]))
    segments: list[dict[str, Any]] = []
    current = {
        "lanes": [ordered[0]["lane"]],
        "max_capacity_seen": ordered[0]["capacity_blocks"],
        "policy": ordered[0]["policy"],
        "label": ordered[0]["label"],
    }
    for prev, row in zip(ordered, ordered[1:]):
        if row["policy"] == current["policy"]:
            current["lanes"].append(row["lane"])
            current["max_capacity_seen"] = row["capacity_blocks"]
            continue
        current["recommended_max_blocks"] = threshold_between(
            int(prev["capacity_blocks"]),
            int(row["capacity_blocks"]),
        )
        segments.append(current)
        current = {
            "lanes": [row["lane"]],
            "max_capacity_seen": row["capacity_blocks"],
            "policy": row["policy"],
            "label": row["label"],
        }
    current["recommended_max_blocks"] = None
    segments.append(current)
    return segments


def write_markdown(path: Path, rows: list[dict[str, Any]], selected: list[dict[str, Any]], segments: list[dict[str, Any]]) -> None:
    avg_score = mean([row["deltas"].get("three_metric_score_pct") for row in selected])
    avg_goodput = mean([row["deltas"].get("goodput_gain_pct") for row in selected])
    avg_audio = mean([row["deltas"].get("audio_p90_reduction_pct") for row in selected])
    avg_e2e = mean([row["deltas"].get("e2e_p90_reduction_pct") for row in selected])
    lines = [
        "# Adaptive Eviction Policy Selection",
        "",
        "This report uses clean historical runs only; polluted retests are not part of the default candidate set.",
        "",
        "## Selected Lanes",
        "",
        "| lane | blocks | selected label | policy | score | goodput | audio p90 | e2e p90 | continuity | recent miss |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        d = row["deltas"]
        lines.append(
            f"| {row['lane']} | {row['capacity_blocks']} | `{row['label']}` | `{row['policy']}` | "
            f"{fmt(d.get('three_metric_score_pct'))}% | {fmt(d.get('goodput_gain_pct'))}% | "
            f"{fmt(d.get('audio_p90_reduction_pct'))}% | {fmt(d.get('e2e_p90_reduction_pct'))}% | "
            f"{fmt(d.get('continuity_delta_pp'))}pp | {fmt(d.get('recent_miss_reduction_pct'))}% |"
        )
    lines.extend(
        [
            "",
            f"- average three-metric score: `{fmt(avg_score)}%`",
            f"- average goodput gain: `{fmt(avg_goodput)}%`",
            f"- average audio p90 reduction: `{fmt(avg_audio)}%`",
            f"- average e2e p90 reduction: `{fmt(avg_e2e)}%`",
            "",
            "## Recommended Capacity Branches",
            "",
            "| branch lanes | max blocks | policy | representative winner |",
            "|---|---:|---|---|",
        ]
    )
    for segment in segments:
        max_blocks = segment["recommended_max_blocks"]
        lines.append(
            f"| {', '.join(segment['lanes'])} | {fmt(max_blocks, 0)} | "
            f"`{segment['policy']}` | `{segment['label']}` |"
        )
    lines.extend(
        [
            "",
            "## Candidate Detail",
            "",
            "| lane | label | policy | promotable | score | goodput | audio p90 | e2e p90 | continuity | evict | recent miss | applied | repl p90 | failures |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(
        rows,
        key=lambda r: (
            r["capacity_blocks"],
            0 if r["promotable"] else 1,
            -float(r["deltas"].get("three_metric_score_pct") or -1e9),
            tie_break_key(r),
        ),
    ):
        d = row["deltas"]
        shadow = row["shadow"]
        failures = "; ".join(row["failures"])
        lines.append(
            f"| {row['lane']} | `{row['label']}` | `{row['policy']}` | {row['promotable']} | "
            f"{fmt(d.get('three_metric_score_pct'))}% | {fmt(d.get('goodput_gain_pct'))}% | "
            f"{fmt(d.get('audio_p90_reduction_pct'))}% | {fmt(d.get('e2e_p90_reduction_pct'))}% | "
            f"{fmt(d.get('continuity_delta_pp'))}pp | {fmt(d.get('eviction_reduction_pct'))}% | "
            f"{fmt(d.get('recent_miss_reduction_pct'))}% | {shadow.get('applied')} | "
            f"{fmt(shadow.get('replacement_count_p90'))} | {failures} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_candidate(raw: str) -> CandidateSpec:
    parts = raw.split("=", 1)
    if len(parts) != 2 or ":" not in parts[0]:
        raise argparse.ArgumentTypeError("candidate must be lane:label=root")
    lane, label = parts[0].split(":", 1)
    return CandidateSpec(lane.strip(), label.strip(), parts[1].strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", type=Path, default=EXP_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        help="Append a candidate as lane:label=root. Use with --replace-default-candidates to disable built-ins.",
    )
    parser.add_argument("--replace-default-candidates", action="store_true")
    parser.add_argument("--score-epsilon", type=float, default=1.0)
    parser.add_argument("--min-goodput-gain-pct", type=float, default=-1.0)
    parser.add_argument("--min-audio-p90-reduction-pct", type=float, default=0.0)
    parser.add_argument("--min-e2e-p90-reduction-pct", type=float, default=0.0)
    parser.add_argument("--min-continuity-delta-pp", type=float, default=-0.5)
    parser.add_argument("--max-replacement-count-p90", type=float, default=1.0)
    parser.add_argument("--min-eviction-reduction-pct", type=float, default=-5.0)
    args = parser.parse_args()

    lanes = {lane.name: lane for lane in DEFAULT_LANES}
    candidates = [] if args.replace_default_candidates else list(DEFAULT_CANDIDATES)
    candidates.extend(args.candidate or [])
    rows = [
        evaluate_candidate(lanes[candidate.lane], candidate, exp_root=args.exp_root, args=args)
        for candidate in candidates
        if candidate.lane in lanes
    ]
    selected = select_lanes(rows, args.score_epsilon)
    segments = adaptive_segments(selected)
    result = {
        "selection_args": {
            "score_epsilon": args.score_epsilon,
            "min_goodput_gain_pct": args.min_goodput_gain_pct,
            "min_audio_p90_reduction_pct": args.min_audio_p90_reduction_pct,
            "min_e2e_p90_reduction_pct": args.min_e2e_p90_reduction_pct,
            "min_continuity_delta_pp": args.min_continuity_delta_pp,
            "max_replacement_count_p90": args.max_replacement_count_p90,
            "min_eviction_reduction_pct": args.min_eviction_reduction_pct,
        },
        "lanes": [lane.__dict__ for lane in DEFAULT_LANES],
        "selected": selected,
        "segments": segments,
        "candidates": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "adaptive_policy_selection.json"
    md_path = args.output_dir / "adaptive_policy_selection.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, rows, selected, segments)
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
