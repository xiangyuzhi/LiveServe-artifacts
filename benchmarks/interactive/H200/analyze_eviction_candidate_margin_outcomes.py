#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from replay_eviction_candidate_ranking import (
    as_float,
    as_int,
    bucket_gap,
    bucket_history,
    dedupe_records,
    expected_or_predicted_gap_sec,
    records_from_container,
    score_lists_show_improvement,
)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for seq, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_seq"] = seq
            yield row


def find_probe_file(root_or_path: Path, name: str) -> Path:
    if root_or_path.is_file():
        return root_or_path
    direct = root_or_path / "_probe" / name
    if direct.exists():
        return direct
    for pattern in (f"*/_probe/{name}", f"**/{name}"):
        matches = sorted(root_or_path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no {name} under {root_or_path}")


def safe_rate(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[idx]


def score_values(container: Any) -> list[tuple[Any, ...]]:
    if isinstance(container, dict) and isinstance(container.get("values"), list):
        out: list[tuple[Any, ...]] = []
        for item in container["values"]:
            if isinstance(item, list):
                out.append(tuple(as_float(value) if value in {"inf", "-inf"} else value for value in item))
            elif isinstance(item, tuple):
                out.append(item)
        return out
    return []


def numeric_score(score: tuple[Any, ...] | None) -> float | None:
    if not score or len(score) < 3:
        return None
    return as_float(score[2])


def score_gap(score: tuple[Any, ...] | None) -> float | None:
    if not score or len(score) < 4:
        return None
    return as_float(score[3])


def score_lost_tokens(score: tuple[Any, ...] | None) -> int | None:
    if not score or len(score) < 6:
        return None
    value = as_float(score[5])
    return int(-value) if value is not None else None


def records_by_id(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for record in records:
        block_id = as_int(record.get("block_id"))
        if block_id is not None:
            out.setdefault(block_id, record)
    return out


def selected_records(event: dict[str, Any]) -> list[dict[str, Any]]:
    direct = dedupe_records(records_from_container(event.get("selected_victim_records")))
    if direct:
        return direct

    detail_records = records_from_container(event.get("candidate_detail"))
    selected_from_detail = [
        record
        for record in detail_records
        if record.get("candidate_detail_source") == "selected"
    ]
    if selected_from_detail:
        return dedupe_records(selected_from_detail)

    ids = {
        block_id
        for block_id in (as_int(item) for item in event.get("selected_block_ids") or [])
        if block_id is not None
    }
    if ids:
        return dedupe_records(
            [
                record
                for record in detail_records
                if as_int(record.get("block_id")) in ids
            ]
        )
    return []


def actual_records(event: dict[str, Any]) -> list[dict[str, Any]]:
    direct = dedupe_records(records_from_container(event.get("actual_lru_victim_records")))
    if direct:
        return direct
    return dedupe_records(records_from_container(event.get("actual_lru_victims")))


def attach_scores(
    records: list[dict[str, Any]],
    scores: list[tuple[Any, ...]],
) -> list[tuple[dict[str, Any], tuple[Any, ...] | None]]:
    out: list[tuple[dict[str, Any], tuple[Any, ...] | None]] = []
    for idx, record in enumerate(records):
        out.append((record, scores[idx] if idx < len(scores) else None))
    return out


def read_victim_outcomes(path: Path | None) -> dict[tuple[int, int], dict[str, Any]]:
    if path is None:
        return {}
    out: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            event_seq = as_int(row.get("event_seq"))
            block_id = as_int(row.get("block_id"))
            if event_seq is None or block_id is None:
                continue
            out[(event_seq, block_id)] = row
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.3f}"
    return str(value)


def score_margin_bucket(value: float | None) -> str:
    if value is None:
        return "margin_missing"
    if math.isinf(value):
        return "margin_inf" if value > 0 else "margin_neg_inf"
    if value < 0:
        return "margin_lt0"
    if value < 10:
        return "margin_0_10"
    if value < 30:
        return "margin_10_30"
    if value < 60:
        return "margin_30_60"
    return "margin_60_plus"


def simple_bucket(value: float | None, cuts: list[float], prefix: str) -> str:
    if value is None:
        return f"{prefix}_missing"
    if math.isinf(value):
        return f"{prefix}_inf"
    prev = None
    for cut in cuts:
        if value < cut:
            if prev is None:
                return f"{prefix}_lt{cut:g}"
            return f"{prefix}_{prev:g}_{cut:g}"
        prev = cut
    return f"{prefix}_{cuts[-1]:g}_plus"


def victim_future(row: dict[str, Any] | None) -> tuple[int, int]:
    if not row:
        return 0, 0
    misses = as_int(row.get("future_lookup_misses")) or 0
    return int(misses > 0), misses


def analyze(
    shadow_path: Path,
    victim_outcomes: dict[tuple[int, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    event_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    group_acc: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "group": "",
            "value": "",
            "events": 0,
            "selected_victims": 0,
            "future_miss_victims": 0,
            "future_misses": 0,
        }
    )
    counts: dict[str, int] = defaultdict(int)
    margins: list[float] = []

    for event in iter_jsonl(shadow_path):
        if event.get("event") != "allocation_pressure_enforce":
            continue
        counts["enforce_events"] += 1
        detail = event.get("candidate_detail")
        if isinstance(detail, dict):
            counts["events_with_candidate_detail"] += 1
            if detail.get("truncated"):
                counts["events_with_truncated_candidate_detail"] += 1
        else:
            continue

        event_seq = int(event["_seq"])
        selected = selected_records(event)
        actual = actual_records(event)
        selected_scores = score_values(event.get("selected_score_tuples"))
        actual_scores = score_values(event.get("actual_score_tuples"))
        if not selected or not actual:
            counts["events_missing_records"] += 1
            continue

        selected_pairs = attach_scores(selected, selected_scores)
        actual_pairs = attach_scores(actual, actual_scores)
        selected_numeric_scores = [
            value
            for _, score in selected_pairs
            if (value := numeric_score(score)) is not None
        ]
        actual_numeric_scores = [
            value
            for _, score in actual_pairs
            if (value := numeric_score(score)) is not None
        ]
        selected_min_score = min(selected_numeric_scores) if selected_numeric_scores else None
        actual_max_score = max(actual_numeric_scores) if actual_numeric_scores else None
        numeric_margin = (
            selected_min_score - actual_max_score
            if selected_min_score is not None and actual_max_score is not None
            else None
        )
        if numeric_margin is not None and math.isfinite(numeric_margin):
            margins.append(numeric_margin)

        selected_ids = {
            block_id
            for block_id in (as_int(record.get("block_id")) for record in selected)
            if block_id is not None
        }
        actual_ids = {
            block_id
            for block_id in (as_int(record.get("block_id")) for record in actual)
            if block_id is not None
        }
        selected_future_miss_victims = 0
        selected_future_misses = 0
        selected_terminal = sum(1 for record in selected if bool(record.get("omni_is_terminal_turn")))
        actual_terminal = sum(1 for record in actual if bool(record.get("omni_is_terminal_turn")))
        selected_lost = sum(
            as_int(record.get("estimated_lost_tokens_if_prefix_break"))
            or as_int(record.get("omni_history_tokens_after"))
            or 0
            for record in selected
        )
        actual_lost = sum(
            as_int(record.get("estimated_lost_tokens_if_prefix_break"))
            or as_int(record.get("omni_history_tokens_after"))
            or 0
            for record in actual
        )

        for victim_idx, (record, score) in enumerate(selected_pairs):
            block_id = as_int(record.get("block_id"))
            outcome = (
                victim_outcomes.get((event_seq, block_id))
                if block_id is not None
                else None
            )
            future_miss_victim, future_misses = victim_future(outcome)
            selected_future_miss_victims += future_miss_victim
            selected_future_misses += future_misses
            terminal = bool(record.get("omni_is_terminal_turn"))
            gap = expected_or_predicted_gap_sec(record)
            score_num = numeric_score(score)
            row = {
                "event_seq": event_seq,
                "victim_idx": victim_idx,
                "block_id": block_id,
                "reason": event.get("reason"),
                "applied": event.get("applied"),
                "pressure_phase": event.get("pressure_phase"),
                "margin_bucket": score_margin_bucket(numeric_margin),
                "event_numeric_margin": numeric_margin,
                "selected_score": score_num,
                "selected_gap_from_score": score_gap(score),
                "selected_gap": gap,
                "selected_gap_bucket": bucket_gap(record),
                "selected_history_bucket": bucket_history(record),
                "selected_response_bucket": simple_bucket(
                    as_float(record.get("omni_response_length")),
                    [16, 32, 64, 128],
                    "resp",
                ),
                "selected_terminal": terminal,
                "selected_history_tokens_after": record.get("omni_history_tokens_after"),
                "selected_lost_tokens": (
                    score_lost_tokens(score)
                    or as_int(record.get("estimated_lost_tokens_if_prefix_break"))
                ),
                "selected_suffix_blocks_including": record.get("suffix_blocks_including"),
                "future_miss_victim": future_miss_victim,
                "future_lookup_misses": future_misses,
            }
            selected_rows.append(row)

            group_values = {
                "margin_bucket": row["margin_bucket"],
                "selected_terminal": str(terminal),
                "selected_gap_bucket": row["selected_gap_bucket"],
                "selected_history_bucket": row["selected_history_bucket"],
                "selected_response_bucket": row["selected_response_bucket"],
                "pressure_phase": str(event.get("pressure_phase") or "missing"),
            }
            for group, value in group_values.items():
                acc = group_acc[(group, str(value))]
                acc["group"] = group
                acc["value"] = str(value)
                acc["selected_victims"] += 1
                acc["future_miss_victims"] += future_miss_victim
                acc["future_misses"] += future_misses

        event_row = {
            "event_seq": event_seq,
            "reason": event.get("reason"),
            "applied": event.get("applied"),
            "policy": event.get("policy"),
            "pressure_phase": event.get("pressure_phase"),
            "replacement_count": event.get("replacement_count"),
            "candidate_count": event.get("candidate_count"),
            "candidate_detail_logged": detail.get("logged") if isinstance(detail, dict) else None,
            "candidate_detail_count": detail.get("count") if isinstance(detail, dict) else None,
            "candidate_detail_truncated": detail.get("truncated") if isinstance(detail, dict) else None,
            "selected_count": len(selected),
            "actual_count": len(actual),
            "selected_actual_overlap": len(selected_ids & actual_ids),
            "selected_terminal": selected_terminal,
            "actual_terminal": actual_terminal,
            "selected_nonterminal": len(selected) - selected_terminal,
            "actual_nonterminal": len(actual) - actual_terminal,
            "selected_lost_tokens": selected_lost,
            "actual_lost_tokens": actual_lost,
            "selected_lost_minus_actual_lost": selected_lost - actual_lost,
            "selected_min_score": selected_min_score,
            "actual_max_score": actual_max_score,
            "numeric_margin": numeric_margin,
            "margin_bucket": score_margin_bucket(numeric_margin),
            "score_tuple_improves": score_lists_show_improvement(
                selected_scores[: len(actual_scores)],
                actual_scores[: len(selected_scores)],
            ),
            "selected_future_miss_victims": selected_future_miss_victims,
            "selected_future_misses": selected_future_misses,
            "recent_no_cached_victim_attempts_64": event.get("recent_no_cached_victim_attempts_64"),
            "recent_cached_victim_attempts_64": event.get("recent_cached_victim_attempts_64"),
            "recent_applied_attempts_64": event.get("recent_applied_attempts_64"),
        }
        event_rows.append(event_row)
        group_values = {
            "event_margin_bucket": event_row["margin_bucket"],
            "event_selected_terminal_count": str(selected_terminal),
            "event_actual_terminal_count": str(actual_terminal),
            "event_selected_lost_minus_actual_lost": simple_bucket(
                as_float(event_row["selected_lost_minus_actual_lost"]),
                [-128, -64, 0, 64, 128],
                "lost_delta",
            ),
        }
        for group, value in group_values.items():
            acc = group_acc[(group, str(value))]
            acc["group"] = group
            acc["value"] = str(value)
            acc["events"] += 1
            acc["selected_victims"] += len(selected)
            acc["future_miss_victims"] += selected_future_miss_victims
            acc["future_misses"] += selected_future_misses

    groups = []
    for row in group_acc.values():
        out = dict(row)
        victims = int(out["selected_victims"])
        out["future_miss_victim_rate"] = safe_rate(out["future_miss_victims"], victims)
        out["future_misses_per_victim"] = safe_rate(out["future_misses"], victims)
        groups.append(out)
    groups.sort(
        key=lambda row: (
            str(row["group"]),
            -int(row["future_misses"]),
            -int(row["selected_victims"]),
            str(row["value"]),
        )
    )
    summary = {
        "shadow_path": str(shadow_path),
        "counts": dict(counts),
        "analyzed_events": len(event_rows),
        "selected_victims": len(selected_rows),
        "events_with_future_miss": sum(
            1 for row in event_rows if (as_int(row.get("selected_future_misses")) or 0) > 0
        ),
        "future_miss_victims": sum(
            as_int(row.get("future_miss_victim")) or 0 for row in selected_rows
        ),
        "future_misses": sum(
            as_int(row.get("future_lookup_misses")) or 0 for row in selected_rows
        ),
        "selected_terminal_victims": sum(
            1 for row in selected_rows if str(row.get("selected_terminal")) == "True"
        ),
        "selected_nonterminal_victims": sum(
            1 for row in selected_rows if str(row.get("selected_terminal")) != "True"
        ),
        "finite_margin_p50": percentile(margins, 0.50),
        "finite_margin_p10": percentile(margins, 0.10),
        "finite_margin_p90": percentile(margins, 0.90),
    }
    return event_rows, selected_rows, groups, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    *,
    summary: dict[str, Any],
    groups: list[dict[str, Any]],
) -> None:
    lines = [
        "# Eviction Candidate Margin Outcomes",
        "",
        f"- shadow log: `{summary['shadow_path']}`",
        f"- analyzed events: `{summary['analyzed_events']}`",
        f"- selected victims: `{summary['selected_victims']}`",
        f"- future miss victims: `{summary['future_miss_victims']}`",
        f"- future misses: `{summary['future_misses']}`",
        f"- selected terminal/nonterminal victims: "
        f"`{summary['selected_terminal_victims']}` / `{summary['selected_nonterminal_victims']}`",
        f"- finite numeric margin p10/p50/p90: "
        f"`{fmt(summary.get('finite_margin_p10'))}` / "
        f"`{fmt(summary.get('finite_margin_p50'))}` / "
        f"`{fmt(summary.get('finite_margin_p90'))}`",
        f"- candidate detail truncated events: "
        f"`{summary['counts'].get('events_with_truncated_candidate_detail', 0)}`",
        "",
        "## Top Outcome Groups",
        "",
        "| group | value | events | victims | future miss victims | miss rate | future misses | future/victim |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(
        groups,
        key=lambda row: (
            int(row.get("future_misses") or 0),
            int(row.get("selected_victims") or 0),
        ),
        reverse=True,
    )
    for row in ranked[:48]:
        lines.append(
            f"| {row['group']} | {row['value']} | {row['events']} | "
            f"{row['selected_victims']} | {row['future_miss_victims']} | "
            f"{float(row['future_miss_victim_rate']):.3f} | "
            f"{row['future_misses']} | "
            f"{float(row['future_misses_per_victim']):.3f} |"
        )

    for group in sorted({str(row["group"]) for row in groups}):
        lines.extend(
            [
                "",
                f"## {group}",
                "",
                "| value | events | victims | miss rate | future misses | future/victim |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in [item for item in groups if item["group"] == group]:
            lines.append(
                f"| {row['value']} | {row['events']} | {row['selected_victims']} | "
                f"{float(row['future_miss_victim_rate']):.3f} | "
                f"{row['future_misses']} | "
                f"{float(row['future_misses_per_victim']):.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow", required=True, help="run root or kv_eviction_shadow.jsonl")
    parser.add_argument("--victims", help="victim_outcomes.csv from analyze_eviction_victim_outcomes.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    shadow_path = find_probe_file(Path(args.shadow), "kv_eviction_shadow.jsonl")
    victim_outcomes = read_victim_outcomes(Path(args.victims) if args.victims else None)
    event_rows, selected_rows, groups, summary = analyze(shadow_path, victim_outcomes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "candidate_margin_events.csv", event_rows)
    write_csv(args.output_dir / "candidate_margin_selected_victims.csv", selected_rows)
    write_csv(args.output_dir / "candidate_margin_groups.csv", groups)
    (args.output_dir / "candidate_margin_summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "groups": groups,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_markdown(
        args.output_dir / "candidate_margin_summary.md",
        summary=summary,
        groups=groups,
    )
    print(args.output_dir / "candidate_margin_summary.json")
    print(args.output_dir / "candidate_margin_events.csv")
    print(args.output_dir / "candidate_margin_selected_victims.csv")
    print(args.output_dir / "candidate_margin_groups.csv")
    print(args.output_dir / "candidate_margin_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
