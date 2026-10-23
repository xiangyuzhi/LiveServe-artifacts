#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RESPONSE_GAP_SEC = {
    "<=16": 18.0,
    "17-32": 30.0,
    "33-64": 45.0,
    "65-128": 68.0,
    "129-256": 112.0,
    ">256": 191.0,
}


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return -float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def response_bucket(response_length: int | None) -> str:
    if response_length is None:
        return "33-64"
    if response_length <= 16:
        return "<=16"
    if response_length <= 32:
        return "17-32"
    if response_length <= 64:
        return "33-64"
    if response_length <= 128:
        return "65-128"
    if response_length <= 256:
        return "129-256"
    return ">256"


def predicted_gap_sec(record: dict[str, Any]) -> float:
    if record.get("omni_is_terminal_turn"):
        return float("inf")
    response_length = as_int(record.get("omni_response_length"))
    return RESPONSE_GAP_SEC[response_bucket(response_length)]


def expected_or_predicted_gap_sec(record: dict[str, Any]) -> float:
    if record.get("omni_is_terminal_turn"):
        return float("inf")
    expected_gap = as_float(record.get("omni_expected_next_gap_sec"))
    if expected_gap is not None and expected_gap >= 0.0:
        return expected_gap
    return predicted_gap_sec(record)


def prefetch_protected(record: dict[str, Any]) -> bool:
    if record.get("omni_scheduler_prefetch") or record.get("omni_prefetch_gpu_pin"):
        return True
    if record.get("omni_prefetch_hit_used"):
        return True
    lookup_state = str(record.get("omni_prefetch_lookup_state") or "").lower()
    return lookup_state in {
        "prefetch_hit",
        "ready",
        "pending",
        "loading",
        "warm",
        "warm_fallback",
        "hit",
    }


def cost_loss_penalty(lost_tokens: int, weight: float) -> float:
    return max(0.0, weight) * math.log1p(max(0, lost_tokens))


def near_reuse_history_penalty(
    gap: float,
    terminal: bool,
    history_tokens: int,
    gap_sec: float,
    history_target: int,
    weight: float,
    terminal_ok: bool,
) -> float:
    if weight <= 0.0 or gap_sec <= 0.0 or history_target <= 0:
        return 0.0
    if terminal and terminal_ok:
        return 0.0
    if math.isinf(gap) or gap >= gap_sec:
        return 0.0
    history_deficit = max(0.0, history_target - max(0, history_tokens))
    if history_deficit <= 0.0:
        return 0.0
    history_ratio = history_deficit / max(1.0, float(history_target))
    gap_ratio = (gap_sec - max(0.0, gap)) / max(gap_sec, 1e-9)
    return max(0.0, weight) * history_ratio * gap_ratio


def piecewise_risk_penalty(
    gap: float,
    terminal: bool,
    history_tokens: int,
    spec: dict[str, Any],
) -> float:
    if terminal and bool(spec.get("terminal_ok", True)):
        return 0.0
    if math.isinf(gap):
        return 0.0
    max_gap = float(spec.get("max_gap", 60.0))
    if max_gap > 0.0 and gap >= max_gap:
        return 0.0

    history_tokens = max(0, history_tokens)
    penalty = 0.0
    low_history = history_tokens < 128
    if history_tokens < 64:
        penalty += max(0.0, float(spec.get("hist_lt64", 0.0)))
    elif low_history:
        penalty += max(0.0, float(spec.get("hist_64_128", 0.0)))

    if low_history:
        if 15.0 <= gap < 30.0:
            penalty += max(0.0, float(spec.get("gap_15_30", 0.0)))
        elif 30.0 <= gap < 60.0:
            penalty += max(0.0, float(spec.get("gap_30_60", 0.0)))
    return penalty


def score_expected_gap_cost_aware(
    record: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[Any, ...]:
    response_length = as_int(record.get("omni_response_length")) or 0
    history_tokens = as_int(record.get("omni_history_tokens_after")) or 0
    terminal = bool(record.get("omni_is_terminal_turn"))
    depth_ratio = as_float(record.get("block_depth_ratio")) or 0.0
    suffix_blocks = as_int(record.get("suffix_blocks_including")) or 0
    lost_tokens = as_int(record.get("estimated_lost_tokens_if_prefix_break"))
    if lost_tokens is None:
        lost_tokens = history_tokens
    gap = expected_or_predicted_gap_sec(record)
    loss_penalty = cost_loss_penalty(
        lost_tokens,
        float(spec.get("cost_loss_weight", 1.0)),
    )
    near_penalty = near_reuse_history_penalty(
        gap,
        terminal,
        history_tokens,
        float(spec.get("near_gap", 0.0)),
        int(spec.get("near_history_target", 0)),
        float(spec.get("near_weight", 0.0)),
        bool(spec.get("near_terminal_ok", True)),
    )
    risk_penalty = piecewise_risk_penalty(gap, terminal, history_tokens, spec)
    score = (
        float("inf")
        if math.isinf(gap)
        else gap - loss_penalty - near_penalty - risk_penalty
    )
    return (
        0 if prefetch_protected(record) else 1,
        1 if terminal else 0,
        score,
        gap,
        depth_ratio,
        -lost_tokens,
        -suffix_blocks,
        response_length,
    )


def score_lists_show_improvement(
    selected_scores: list[tuple[Any, ...]],
    actual_scores: list[tuple[Any, ...]],
) -> bool:
    if not selected_scores or len(selected_scores) != len(actual_scores):
        return False
    for selected, actual in zip(
        sorted(selected_scores, reverse=True),
        sorted(actual_scores, reverse=True),
    ):
        if selected > actual:
            return True
        if selected < actual:
            return False
    return False


def records_from_container(container: Any) -> list[dict[str, Any]]:
    if isinstance(container, dict):
        records = container.get("records")
        return records if isinstance(records, list) else []
    if isinstance(container, list):
        return container
    return []


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for record in records:
        block_id = as_int(record.get("block_id"))
        if block_id is None or block_id in seen:
            continue
        seen.add(block_id)
        out.append(record)
    return out


def bucket_history(record: dict[str, Any]) -> str:
    history = as_int(record.get("omni_history_tokens_after")) or 0
    if history < 64:
        return "hist_lt64"
    if history < 128:
        return "hist_64_128"
    if history < 256:
        return "hist_128_256"
    if history < 512:
        return "hist_256_512"
    return "hist_512_plus"


def bucket_gap(record: dict[str, Any]) -> str:
    gap = expected_or_predicted_gap_sec(record)
    if math.isinf(gap):
        return "gap_missing"
    if gap < 15:
        return "gap_lt15"
    if gap < 30:
        return "gap_15_30"
    if gap < 60:
        return "gap_30_60"
    if gap < 120:
        return "gap_60_120"
    if gap < 300:
        return "gap_120_300"
    return "gap_300_plus"


def parse_spec(raw: str) -> dict[str, Any]:
    name, _, payload = raw.partition(":")
    if not name or not payload:
        raise ValueError(f"Invalid --spec {raw!r}; expected name:k=v,...")
    spec: dict[str, Any] = {
        "name": name,
        "cost_loss_weight": 1.0,
        "max_gap": 60.0,
        "hist_lt64": 0.0,
        "hist_64_128": 0.0,
        "gap_15_30": 0.0,
        "gap_30_60": 0.0,
        "terminal_ok": True,
        "near_gap": 0.0,
        "near_history_target": 0,
        "near_weight": 0.0,
        "near_terminal_ok": True,
    }
    for item in payload.split(","):
        if not item:
            continue
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid spec item {item!r} in {raw!r}")
        key = key.strip().replace("-", "_")
        value = value.strip()
        if key in {"terminal_ok", "near_terminal_ok"}:
            spec[key] = value.lower() in {"1", "true", "yes", "on"}
        elif key in {"near_history_target"}:
            spec[key] = int(value)
        else:
            spec[key] = float(value)
    return spec


def load_specs(raw_specs: list[str]) -> list[dict[str, Any]]:
    if raw_specs:
        return [parse_spec(raw) for raw in raw_specs]
    return [
        parse_spec(
            "baseline:hist_lt64=0,hist_64_128=0,gap_15_30=0,gap_30_60=0"
        ),
        parse_spec(
            "piecewise_v1:hist_lt64=31.3,hist_64_128=12.0,"
            "gap_15_30=7.0,gap_30_60=7.8,max_gap=60"
        ),
        parse_spec(
            "piecewise_v2:hist_lt64=24.0,hist_64_128=24.0,"
            "gap_15_30=7.0,gap_30_60=10.0,max_gap=60"
        ),
    ]


def update_bucket_counts(
    prefix: str,
    records: list[dict[str, Any]],
    counter: Counter[str],
) -> None:
    for record in records:
        counter[f"{prefix}:history:{bucket_history(record)}"] += 1
        counter[f"{prefix}:gap:{bucket_gap(record)}"] += 1


def replay_event(
    event: dict[str, Any],
    specs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]] | None:
    candidate_detail = event.get("candidate_detail")
    if not isinstance(candidate_detail, dict):
        return None
    candidates = dedupe_records(records_from_container(candidate_detail))
    actual = dedupe_records(records_from_container(event.get("actual_lru_victim_records")))
    if not actual:
        actual = dedupe_records(records_from_container(event.get("actual_lru_victims")))
    if not candidates or not actual:
        return None
    replacement_count = as_int(event.get("replacement_count")) or min(
        len(actual),
        as_int(event.get("cached_victim_count")) or len(actual),
    )
    replacement_count = min(replacement_count, len(actual), len(candidates))
    if replacement_count <= 0:
        return None

    observed_ids = {
        block_id
        for block_id in (as_int(item) for item in event.get("selected_block_ids") or [])
        if block_id is not None
    }
    result: dict[str, dict[str, Any]] = {}
    for spec in specs:
        scored_candidates = [
            (score_expected_gap_cost_aware(record, spec), record)
            for record in candidates
        ]
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [record for _, record in scored_candidates[:replacement_count]]
        selected_scores = [score for score, _ in scored_candidates[:replacement_count]]
        actual_scores = [
            score_expected_gap_cost_aware(record, spec)
            for record in actual[:replacement_count]
        ]
        selected_ids = {
            block_id
            for block_id in (as_int(record.get("block_id")) for record in selected)
            if block_id is not None
        }
        result[str(spec["name"])] = {
            "passes_improvement": score_lists_show_improvement(
                selected_scores,
                actual_scores,
            ),
            "selected_records": selected,
            "changed_from_observed": selected_ids != observed_ids,
            "overlap_with_observed": len(selected_ids & observed_ids),
            "selected_count": len(selected_ids),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay expected_gap_cost_aware ranking on logged candidates.",
    )
    parser.add_argument("--input", required=True, help="kv_eviction_shadow.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help=(
            "Ranking spec as name:k=v,... . Keys include hist_lt64, "
            "hist_64_128, gap_15_30, gap_30_60, max_gap, cost_loss_weight."
        ),
    )
    args = parser.parse_args()

    specs = load_specs(args.spec)
    stats: dict[str, dict[str, Any]] = {
        str(spec["name"]): {
            "events": 0,
            "passes_improvement": 0,
            "changed_from_observed": 0,
            "observed_overlap_total": 0,
            "selected_total": 0,
            "buckets": Counter(),
        }
        for spec in specs
    }
    global_counts = Counter()

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            global_counts["events"] += 1
            detail = event.get("candidate_detail")
            if isinstance(detail, dict):
                global_counts["events_with_candidate_detail"] += 1
                if detail.get("truncated"):
                    global_counts["truncated_candidate_detail"] += 1
            replayed = replay_event(event, specs)
            if replayed is None:
                continue
            global_counts["replayed_events"] += 1
            for name, result in replayed.items():
                row = stats[name]
                row["events"] += 1
                row["passes_improvement"] += int(result["passes_improvement"])
                row["changed_from_observed"] += int(result["changed_from_observed"])
                row["observed_overlap_total"] += int(result["overlap_with_observed"])
                row["selected_total"] += int(result["selected_count"])
                update_bucket_counts(
                    "selected",
                    result["selected_records"],
                    row["buckets"],
                )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input": args.input,
        "global": dict(global_counts),
        "specs": specs,
        "runs": {
            name: {
                key: (dict(value) if isinstance(value, Counter) else value)
                for key, value in row.items()
            }
            for name, row in stats.items()
        },
    }
    (out_dir / "candidate_replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with open(out_dir / "candidate_replay_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "spec",
                "events",
                "passes_improvement",
                "pass_rate",
                "changed_from_observed",
                "changed_rate",
                "avg_observed_overlap",
                "selected_total",
            ]
        )
        for name, row in stats.items():
            events = int(row["events"])
            selected_total = int(row["selected_total"])
            writer.writerow(
                [
                    name,
                    events,
                    row["passes_improvement"],
                    (row["passes_improvement"] / events if events else None),
                    row["changed_from_observed"],
                    (row["changed_from_observed"] / events if events else None),
                    (
                        row["observed_overlap_total"] / selected_total
                        if selected_total
                        else None
                    ),
                    selected_total,
                ]
            )

    with open(out_dir / "candidate_replay_buckets.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["spec", "bucket", "count"])
        for name, row in stats.items():
            for bucket, count in sorted(row["buckets"].items()):
                writer.writerow([name, bucket, count])

    md_lines = [
        "# Candidate Ranking Replay",
        "",
        f"- input: `{args.input}`",
        f"- events: `{global_counts['events']}`",
        f"- events with candidate detail: `{global_counts['events_with_candidate_detail']}`",
        f"- replayed events: `{global_counts['replayed_events']}`",
        f"- truncated candidate detail events: `{global_counts['truncated_candidate_detail']}`",
        "",
        "| spec | events | pass rate | changed rate | observed overlap | selected blocks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in stats.items():
        events = int(row["events"])
        selected_total = int(row["selected_total"])
        pass_rate = row["passes_improvement"] / events if events else 0.0
        changed_rate = row["changed_from_observed"] / events if events else 0.0
        overlap = row["observed_overlap_total"] / selected_total if selected_total else 0.0
        md_lines.append(
            f"| {name} | {events} | {pass_rate:.3f} | {changed_rate:.3f} | "
            f"{overlap:.3f} | {selected_total} |"
        )
    md_lines.append("")
    md_lines.append("## Bucket Counts")
    md_lines.append("")
    for name, row in stats.items():
        md_lines.append(f"### {name}")
        md_lines.append("")
        md_lines.append("| bucket | count |")
        md_lines.append("|---|---:|")
        for bucket, count in sorted(row["buckets"].items()):
            md_lines.append(f"| {bucket} | {count} |")
        md_lines.append("")
    (out_dir / "candidate_replay_summary.md").write_text(
        "\n".join(md_lines),
        encoding="utf-8",
    )

    print(out_dir / "candidate_replay_summary.json")
    print(out_dir / "candidate_replay_summary.csv")
    print(out_dir / "candidate_replay_buckets.csv")
    print(out_dir / "candidate_replay_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
