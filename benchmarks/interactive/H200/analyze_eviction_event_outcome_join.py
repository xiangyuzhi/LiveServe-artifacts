#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


WEIGHTS = {
    "audio_ttfp": 0.35,
    "continuity": 0.20,
    "e2e": 0.15,
    "ttft": 0.05,
}


@dataclass(frozen=True)
class Gate:
    label: str
    func: Callable[[dict[str, Any]], bool]


def parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(raw)
    return path.name, path


def find_per_request(root_or_path: Path) -> Path:
    if root_or_path.is_file():
        return root_or_path
    patterns = (
        "lru_control/*/per_request.jsonl",
        "eviction_only/*/per_request.jsonl",
        "*/*/per_request.jsonl",
        "**/per_request.jsonl",
    )
    for pattern in patterns:
        matches = sorted(root_or_path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no per_request.jsonl under {root_or_path}")


def find_shadow_path(root_or_path: Path) -> Path:
    if root_or_path.is_file():
        return root_or_path
    patterns = (
        "eviction_only/_probe/kv_eviction_shadow.jsonl",
        "*/_probe/kv_eviction_shadow.jsonl",
        "**/kv_eviction_shadow.jsonl",
    )
    for pattern in patterns:
        matches = sorted(root_or_path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no kv_eviction_shadow.jsonl under {root_or_path}")


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


def finite(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[idx]


def pct_reduction(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return (baseline - candidate) / baseline * 100.0


def trace_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("user_id"),
        row.get("round_index"),
        row.get("timestamp"),
        row.get("query_length"),
        row.get("response_length"),
    )


def shadow_trace_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    current = row.get("current_request")
    if not isinstance(current, dict):
        return None
    return (
        current.get("omni_session_id"),
        current.get("omni_round_index"),
        current.get("omni_trace_timestamp"),
        current.get("omni_query_length"),
        current.get("omni_response_length"),
    )


def read_requests(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("error") is not None:
            continue
        rows[trace_key(row)] = row
    return rows


def summary_metric(
    summary: dict[str, Any] | None,
    name: str,
    qkey: str = "p50",
) -> float | None:
    if not isinstance(summary, dict):
        return None
    raw = summary.get(name)
    if not isinstance(raw, dict):
        return None
    return finite(raw.get(qkey))


def terminal_ratio(summary: dict[str, Any] | None) -> float | None:
    if not isinstance(summary, dict):
        return None
    count = finite(summary.get("count"))
    terminal = finite(summary.get("terminal"))
    if count is None or terminal is None or count <= 0:
        return None
    return terminal / count


def numeric_bucket(value: float | None, cuts: list[float], prefix: str) -> str:
    if value is None:
        return f"{prefix}_missing"
    prev = None
    for cut in cuts:
        if value < cut:
            if prev is None:
                return f"{prefix}_lt{cut:g}"
            return f"{prefix}_{prev:g}_{cut:g}"
        prev = cut
    return f"{prefix}_{cuts[-1]:g}_plus"


def add_stat(stats: dict[str, Any], name: str, value: float | None) -> None:
    if value is None:
        return
    values = stats.setdefault(f"_{name}_values", [])
    values.append(float(value))


def finish_stat(stats: dict[str, Any], name: str) -> None:
    values = stats.pop(f"_{name}_values", [])
    if not values:
        stats[f"{name}_min"] = None
        stats[f"{name}_mean"] = None
        stats[f"{name}_max"] = None
        return
    stats[f"{name}_min"] = min(values)
    stats[f"{name}_mean"] = sum(values) / len(values)
    stats[f"{name}_max"] = max(values)


def read_shadow_events(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = shadow_trace_key(row)
        if key is None:
            continue
        current = row.get("current_request") or {}
        stats = by_key.setdefault(
            key,
            {
                "shadow_event_count": 0,
                "shadow_applied_events": 0,
                "shadow_blocked_events": 0,
                "shadow_replacement_count": 0,
                "shadow_elapsed_ns_total": 0.0,
                "shadow_elapsed_ns_max": 0.0,
                "shadow_min_reason": "",
                "current_prefix_tokens": finite(
                    current.get("omni_prefix_token_estimate"), 0.0
                ),
                "current_round_index": finite(current.get("omni_round_index"), 0.0),
                "current_trace_timestamp": finite(
                    current.get("omni_trace_timestamp"), 0.0
                ),
                "current_query_length": finite(current.get("omni_query_length"), 0.0),
                "current_response_length": finite(
                    current.get("omni_response_length"), 0.0
                ),
                "current_history_tokens_after": finite(
                    current.get("omni_history_tokens_after"), 0.0
                ),
                "current_expected_next_gap_sec": finite(
                    current.get("omni_expected_next_gap_sec")
                ),
                "current_is_terminal_turn": bool(
                    current.get("omni_is_terminal_turn")
                ),
            },
        )
        stats["shadow_event_count"] += 1
        applied = row.get("applied") is True or row.get("reason") == "applied"
        if applied:
            stats["shadow_applied_events"] += 1
            stats["shadow_replacement_count"] += int(row.get("replacement_count") or 0)
        else:
            stats["shadow_blocked_events"] += 1
            if not stats["shadow_min_reason"]:
                stats["shadow_min_reason"] = str(row.get("reason") or "")
        elapsed = finite(row.get("elapsed_ns"), 0.0) or 0.0
        stats["shadow_elapsed_ns_total"] += elapsed
        stats["shadow_elapsed_ns_max"] = max(stats["shadow_elapsed_ns_max"], elapsed)
        stats["shadow_last_pressure_phase"] = row.get("pressure_phase")
        stats["shadow_last_reason"] = row.get("reason")
        stats["shadow_recent_applied_attempts_64"] = finite(
            row.get("recent_applied_attempts_64"), 0.0
        )
        stats["shadow_recent_cached_victim_attempts_64"] = finite(
            row.get("recent_cached_victim_attempts_64"), 0.0
        )
        stats["shadow_recent_no_cached_victim_attempts_64"] = finite(
            row.get("recent_no_cached_victim_attempts_64"), 0.0
        )
        stats["shadow_applied_density_64"] = finite(row.get("applied_density_64"), 0.0)
        stats["shadow_cached_victim_density_64"] = finite(
            row.get("cached_victim_density_64"), 0.0
        )
        stats["shadow_no_cached_victim_density_64"] = finite(
            row.get("no_cached_victim_density_64"), 0.0
        )
        actual = row.get("actual_lru_victim_summary")
        selected = row.get("selected_victim_summary")
        add_stat(stats, "actual_gap", summary_metric(actual, "expected_gap_sec_finite"))
        add_stat(stats, "selected_gap", summary_metric(selected, "expected_gap_sec_finite"))
        add_stat(stats, "actual_response", summary_metric(actual, "response_length"))
        add_stat(stats, "selected_response", summary_metric(selected, "response_length"))
        add_stat(stats, "actual_history", summary_metric(actual, "history_tokens_after"))
        add_stat(stats, "selected_history", summary_metric(selected, "history_tokens_after"))
        add_stat(stats, "actual_lost", summary_metric(actual, "lost_tokens_if_prefix_break"))
        add_stat(stats, "selected_lost", summary_metric(selected, "lost_tokens_if_prefix_break"))
        actual_term = terminal_ratio(actual)
        selected_term = terminal_ratio(selected)
        add_stat(stats, "actual_terminal_ratio", actual_term)
        add_stat(stats, "selected_terminal_ratio", selected_term)
        if selected_term is not None and selected_term >= 0.5:
            stats["shadow_selected_terminal_events"] = int(
                stats.get("shadow_selected_terminal_events", 0)
            ) + 1
        if summary_metric(selected, "expected_gap_sec_finite") is not None:
            stats["shadow_selected_finite_gap_events"] = int(
                stats.get("shadow_selected_finite_gap_events", 0)
            ) + 1
        pair_details = row.get("gap_margin_pair_details")
        if isinstance(pair_details, dict):
            add_stat(
                stats,
                "gap_margin_pass_ratio",
                (
                    (finite(pair_details.get("pass_count"), 0.0) or 0.0)
                    / (finite(pair_details.get("count"), 0.0) or 1.0)
                ),
            )
    for stats in by_key.values():
        for name in (
            "actual_gap",
            "selected_gap",
            "actual_response",
            "selected_response",
            "actual_history",
            "selected_history",
            "actual_lost",
            "selected_lost",
            "actual_terminal_ratio",
            "selected_terminal_ratio",
            "gap_margin_pass_ratio",
        ):
            finish_stat(stats, name)
        applied = int(stats.get("shadow_applied_events", 0))
        events = int(stats.get("shadow_event_count", 0))
        stats["shadow_has_event"] = events > 0
        stats["shadow_has_applied"] = applied > 0
        stats["shadow_selected_terminal_event_ratio"] = (
            int(stats.get("shadow_selected_terminal_events", 0)) / applied
            if applied
            else 0.0
        )
        stats["shadow_selected_finite_gap_event_ratio"] = (
            int(stats.get("shadow_selected_finite_gap_events", 0)) / applied
            if applied
            else 0.0
        )
    return by_key


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audio = [value for row in rows if (value := finite(row.get("audio_ttfp_sec"))) is not None]
    e2e = [value for row in rows if (value := finite(row.get("e2e_sec"))) is not None]
    ttft = [value for row in rows if (value := finite(row.get("ttfp_sec"))) is not None]
    continuity = [
        bool(row.get("audio_continuity_ok"))
        for row in rows
        if row.get("audio_continuity_evaluated") is True
    ]
    return {
        "requests": len(rows),
        "audio_ttfp_p50": percentile(audio, 0.50),
        "audio_ttfp_p90": percentile(audio, 0.90),
        "audio_ttfp_p99": percentile(audio, 0.99),
        "e2e_p50": percentile(e2e, 0.50),
        "e2e_p90": percentile(e2e, 0.90),
        "e2e_p99": percentile(e2e, 0.99),
        "ttft_p90": percentile(ttft, 0.90),
        "continuity": sum(continuity) / len(continuity) if continuity else None,
    }


def deltas(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    out = {
        "audio_ttfp_p90_reduction_pct": pct_reduction(
            metrics.get("audio_ttfp_p90"), baseline.get("audio_ttfp_p90")
        ),
        "e2e_p90_reduction_pct": pct_reduction(
            metrics.get("e2e_p90"), baseline.get("e2e_p90")
        ),
        "ttft_p90_reduction_pct": pct_reduction(
            metrics.get("ttft_p90"), baseline.get("ttft_p90")
        ),
        "continuity_delta_pp": (
            (metrics.get("continuity") - baseline.get("continuity")) * 100.0
            if isinstance(metrics.get("continuity"), (int, float))
            and isinstance(baseline.get("continuity"), (int, float))
            else None
        ),
    }
    values = [
        out.get("audio_ttfp_p90_reduction_pct"),
        out.get("continuity_delta_pp"),
        out.get("e2e_p90_reduction_pct"),
        out.get("ttft_p90_reduction_pct"),
    ]
    out["no_qps_score"] = (
        WEIGHTS["audio_ttfp"] * float(values[0])
        + WEIGHTS["continuity"] * float(values[1])
        + WEIGHTS["e2e"] * float(values[2])
        + WEIGHTS["ttft"] * float(values[3])
        if all(isinstance(value, (int, float)) for value in values)
        else None
    )
    return out


def build_join_rows(
    baseline: dict[tuple[Any, ...], dict[str, Any]],
    candidate: dict[tuple[Any, ...], dict[str, Any]],
    events: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    common = sorted(set(baseline) & set(candidate), key=lambda key: (key[2], key[0], key[1]))
    min_ts = min((finite(key[2], 0.0) or 0.0 for key in common), default=0.0)
    rows: list[dict[str, Any]] = []
    for key in common:
        base = baseline[key]
        cand = candidate[key]
        event = events.get(key, {})
        row = {
            "user_id": key[0],
            "round_index": finite(key[1], 0.0) or 0.0,
            "timestamp": finite(key[2], 0.0) or 0.0,
            "trace_rel_sec": (finite(key[2], 0.0) or 0.0) - min_ts,
            "query_length": finite(key[3], 0.0) or 0.0,
            "response_length": finite(key[4], 0.0) or 0.0,
            "input_tokens_total": finite(cand.get("input_tokens_total"), 0.0) or 0.0,
            "prefix_tokens": finite(cand.get("kv_prefix_token_estimate"), 0.0) or 0.0,
            "baseline_audio_ttfp_sec": finite(base.get("audio_ttfp_sec")),
            "candidate_audio_ttfp_sec": finite(cand.get("audio_ttfp_sec")),
            "baseline_e2e_sec": finite(base.get("e2e_sec")),
            "candidate_e2e_sec": finite(cand.get("e2e_sec")),
            "baseline_ttft_sec": finite(base.get("ttfp_sec")),
            "candidate_ttft_sec": finite(cand.get("ttfp_sec")),
            "baseline_continuity_ok": base.get("audio_continuity_ok"),
            "candidate_continuity_ok": cand.get("audio_continuity_ok"),
            "audio_delta_sec": (
                finite(cand.get("audio_ttfp_sec")) - finite(base.get("audio_ttfp_sec"))
                if finite(cand.get("audio_ttfp_sec")) is not None
                and finite(base.get("audio_ttfp_sec")) is not None
                else None
            ),
            "e2e_delta_sec": (
                finite(cand.get("e2e_sec")) - finite(base.get("e2e_sec"))
                if finite(cand.get("e2e_sec")) is not None
                and finite(base.get("e2e_sec")) is not None
                else None
            ),
            "ttft_delta_sec": (
                finite(cand.get("ttfp_sec")) - finite(base.get("ttfp_sec"))
                if finite(cand.get("ttfp_sec")) is not None
                and finite(base.get("ttfp_sec")) is not None
                else None
            ),
        }
        row.update(event)
        row.setdefault("shadow_event_count", 0)
        row.setdefault("shadow_applied_events", 0)
        row.setdefault("shadow_blocked_events", 0)
        row.setdefault("shadow_replacement_count", 0)
        row.setdefault("shadow_has_event", False)
        row.setdefault("shadow_has_applied", False)
        row["prefix_class"] = "cold" if float(row["prefix_tokens"]) <= 0.0 else "returning"
        row["round_bucket"] = numeric_bucket(float(row["round_index"]), [1, 3, 10, 20, 30], "round")
        row["trace_phase"] = numeric_bucket(float(row["trace_rel_sec"]), [10, 20, 40, 80, 120], "trace")
        row["response_bucket"] = numeric_bucket(float(row["response_length"]), [16, 32, 64, 128], "resp")
        row["query_bucket"] = numeric_bucket(float(row["query_length"]), [16, 32, 64, 128], "query")
        row["event_count_bucket"] = numeric_bucket(
            float(row.get("shadow_applied_events") or 0.0),
            [1, 2, 4, 8],
            "applied",
        )
        row["selected_kind"] = "none"
        if row.get("shadow_applied_events"):
            term_ratio = float(row.get("shadow_selected_terminal_event_ratio") or 0.0)
            finite_ratio = float(row.get("shadow_selected_finite_gap_event_ratio") or 0.0)
            if term_ratio >= 0.8:
                row["selected_kind"] = "terminal"
            elif finite_ratio >= 0.8:
                row["selected_kind"] = "finite"
            else:
                row["selected_kind"] = "mixed"
        rows.append(row)
    return rows


def group_rows(rows: list[dict[str, Any]], group_name: str) -> list[dict[str, Any]]:
    group_values = sorted({str(row.get(group_name)) for row in rows})
    out: list[dict[str, Any]] = []
    for group_value in group_values:
        subset = [row for row in rows if str(row.get(group_name)) == group_value]
        e2e = [v for row in subset if (v := finite(row.get("e2e_delta_sec"))) is not None]
        audio = [v for row in subset if (v := finite(row.get("audio_delta_sec"))) is not None]
        out.append(
            {
                "group": group_name,
                "value": group_value,
                "requests": len(subset),
                "applied_events": sum(int(row.get("shadow_applied_events") or 0) for row in subset),
                "e2e_mean_delta": sum(e2e) / len(e2e) if e2e else None,
                "e2e_p50_delta": percentile(e2e, 0.50),
                "e2e_p90_delta": percentile(e2e, 0.90),
                "e2e_improved_rate": (
                    sum(1 for value in e2e if value < 0.0) / len(e2e) if e2e else None
                ),
                "audio_mean_delta": sum(audio) / len(audio) if audio else None,
                "audio_p90_delta": percentile(audio, 0.90),
            }
        )
    return out


def add_threshold_gates(gates: list[Gate], feature: str, cuts: list[float]) -> None:
    for cut in cuts:
        gates.append(
            Gate(
                f"{feature}>={cut:g}",
                lambda row, f=feature, c=cut: float(row.get(f) or 0.0) >= c,
            )
        )
        gates.append(
            Gate(
                f"{feature}<{cut:g}",
                lambda row, f=feature, c=cut: float(row.get(f) or 0.0) < c,
            )
        )


def make_gates() -> list[Gate]:
    gates: list[Gate] = [
        Gate("all_candidate", lambda row: True),
        Gate("has_shadow_event", lambda row: bool(row.get("shadow_has_event"))),
        Gate("has_applied_event", lambda row: bool(row.get("shadow_has_applied"))),
        Gate("no_applied_event", lambda row: not bool(row.get("shadow_has_applied"))),
        Gate("cold_request", lambda row: float(row.get("prefix_tokens") or 0.0) <= 0.0),
        Gate("returning_request", lambda row: float(row.get("prefix_tokens") or 0.0) > 0.0),
        Gate("selected_terminal_majority", lambda row: float(row.get("shadow_selected_terminal_event_ratio") or 0.0) >= 0.5),
        Gate("selected_finite_majority", lambda row: float(row.get("shadow_selected_finite_gap_event_ratio") or 0.0) >= 0.5),
    ]
    thresholds = {
        "trace_rel_sec": [10, 20, 40, 80, 120],
        "round_index": [1, 3, 10, 20, 30],
        "query_length": [16, 32, 64, 128],
        "response_length": [16, 32, 64, 128],
        "input_tokens_total": [32, 64, 96, 128, 192, 256],
        "prefix_tokens": [1, 32, 64, 96, 128, 192, 256],
        "shadow_applied_events": [1, 2, 3, 4, 8],
        "shadow_elapsed_ns_total": [100000.0, 500000.0, 1000000.0, 5000000.0],
        "actual_gap_min": [15, 30, 45, 60, 90, 120],
        "actual_gap_mean": [15, 30, 45, 60, 90, 120],
        "selected_gap_mean": [45, 60, 90, 120, 180],
        "actual_response_mean": [16, 32, 64, 128],
        "selected_response_mean": [16, 32, 64, 128],
        "shadow_selected_terminal_event_ratio": [0.25, 0.5, 0.75, 1.0],
        "shadow_selected_finite_gap_event_ratio": [0.25, 0.5, 0.75, 1.0],
        "shadow_cached_victim_density_64": [0.05, 0.1, 0.2, 0.5],
        "shadow_no_cached_victim_density_64": [0.5, 0.8, 0.9],
    }
    for feature, cuts in thresholds.items():
        add_threshold_gates(gates, feature, cuts)
    combo_specs = [
        ("returning_and_terminal", lambda row: float(row.get("prefix_tokens") or 0.0) > 0.0 and float(row.get("shadow_selected_terminal_event_ratio") or 0.0) >= 0.5),
        ("returning_and_finite", lambda row: float(row.get("prefix_tokens") or 0.0) > 0.0 and float(row.get("shadow_selected_finite_gap_event_ratio") or 0.0) >= 0.5),
        ("cold_and_terminal", lambda row: float(row.get("prefix_tokens") or 0.0) <= 0.0 and float(row.get("shadow_selected_terminal_event_ratio") or 0.0) >= 0.5),
        ("cold_and_finite", lambda row: float(row.get("prefix_tokens") or 0.0) <= 0.0 and float(row.get("shadow_selected_finite_gap_event_ratio") or 0.0) >= 0.5),
        ("not_early_and_terminal", lambda row: float(row.get("trace_rel_sec") or 0.0) >= 20.0 and float(row.get("shadow_selected_terminal_event_ratio") or 0.0) >= 0.5),
        ("not_early_and_finite", lambda row: float(row.get("trace_rel_sec") or 0.0) >= 20.0 and float(row.get("shadow_selected_finite_gap_event_ratio") or 0.0) >= 0.5),
        ("cold_burst_skip", lambda row: not (float(row.get("prefix_tokens") or 0.0) <= 0.0 and float(row.get("trace_rel_sec") or 0.0) < 20.0)),
    ]
    gates.extend(Gate(label, func) for label, func in combo_specs)
    return gates


def evaluate_gate(
    gate: Gate,
    baseline_rows: dict[tuple[Any, ...], dict[str, Any]],
    candidate_rows: dict[tuple[Any, ...], dict[str, Any]],
    joined_rows: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    candidate_count = 0
    applied_events = 0
    for joined in joined_rows:
        key = (
            joined["user_id"],
            int(joined["round_index"]),
            int(joined["timestamp"]),
            int(joined["query_length"]),
            int(joined["response_length"]),
        )
        if gate.func(joined):
            selected.append(candidate_rows[key])
            candidate_count += 1
            applied_events += int(joined.get("shadow_applied_events") or 0)
        else:
            selected.append(baseline_rows[key])
    metrics = summarize(selected)
    delta = deltas(metrics, baseline_metrics)
    return {
        "selector": gate.label,
        "candidate_requests": candidate_count,
        "baseline_requests": len(joined_rows) - candidate_count,
        "candidate_applied_events": applied_events,
        **metrics,
        **delta,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(
    path: Path,
    *,
    labels: dict[str, str],
    config: dict[str, Any],
    metrics: list[dict[str, Any]],
    group_rows_out: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Eviction Event Outcome Join",
        "",
        "This joins shadow/enforcement events to paired per-request outcomes using",
        "`(user_id, round_index, timestamp, query_length, response_length)`. Request",
        "IDs are not used because server and client suffixes differ. Selector rows",
        "are offline splices: if a gate matches, use the candidate request row;",
        "otherwise use LRU. This is a screen for implementable event gates, not a",
        "runtime guarantee.",
        "",
        f"- baseline: `{labels['baseline']}`",
        f"- candidate: `{labels['candidate']}`",
        f"- common requests: `{config['common_requests']}`",
        f"- shadow matched requests: `{config['shadow_matched_requests']}`",
        f"- shadow applied events: `{config['shadow_applied_events']}`",
        "",
        "## Overall",
        "",
        "| label | requests | audio p90 | e2e p90 | ttft p90 | continuity | audio gain | e2e gain | score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['label']} | {row['requests']} | {fmt(row.get('audio_ttfp_p90'))} | "
            f"{fmt(row.get('e2e_p90'))} | {fmt(row.get('ttft_p90'))} | "
            f"{fmt(row.get('continuity'))} | {fmt(row.get('audio_ttfp_p90_reduction_pct'))}% | "
            f"{fmt(row.get('e2e_p90_reduction_pct'))}% | {fmt(row.get('no_qps_score'))} |"
        )
    lines.extend(
        [
            "",
            "## Top Event Selectors",
            "",
            "| rank | selector | candidate reqs | baseline reqs | selected applied events | audio p90 | audio gain | e2e p90 | e2e gain | ttft gain | continuity gain | score |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(gate_rows[:30], 1):
        lines.append(
            f"| {idx} | `{row['selector']}` | {row['candidate_requests']} | "
            f"{row['baseline_requests']} | {row['candidate_applied_events']} | "
            f"{fmt(row.get('audio_ttfp_p90'))} | {fmt(row.get('audio_ttfp_p90_reduction_pct'))}% | "
            f"{fmt(row.get('e2e_p90'))} | {fmt(row.get('e2e_p90_reduction_pct'))}% | "
            f"{fmt(row.get('ttft_p90_reduction_pct'))}% | {fmt(row.get('continuity_delta_pp'))}pp | "
            f"{fmt(row.get('no_qps_score'))} |"
        )
    lines.extend(
        [
            "",
            "## Outcome Groups",
            "",
            "| group | value | requests | applied events | e2e mean delta | e2e p50 delta | e2e p90 delta | improved rate | audio mean delta | audio p90 delta |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in group_rows_out:
        lines.append(
            f"| {row['group']} | {row['value']} | {row['requests']} | "
            f"{row['applied_events']} | {fmt(row.get('e2e_mean_delta'))} | "
            f"{fmt(row.get('e2e_p50_delta'))} | {fmt(row.get('e2e_p90_delta'))} | "
            f"{fmt(row.get('e2e_improved_rate'))} | {fmt(row.get('audio_mean_delta'))} | "
            f"{fmt(row.get('audio_p90_delta'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="label=/path/to/run or per_request.jsonl")
    parser.add_argument("--candidate", required=True, help="label=/path/to/run or per_request.jsonl")
    parser.add_argument("--shadow", required=True, help="run root or kv_eviction_shadow.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_label, baseline_root = parse_labeled_path(args.baseline)
    candidate_label, candidate_root = parse_labeled_path(args.candidate)
    baseline_path = find_per_request(baseline_root)
    candidate_path = find_per_request(candidate_root)
    shadow_path = find_shadow_path(Path(args.shadow))
    baseline_rows = read_requests(baseline_path)
    candidate_rows = read_requests(candidate_path)
    event_rows = read_shadow_events(shadow_path)
    joined_rows = build_join_rows(baseline_rows, candidate_rows, event_rows)
    common_keys = {
        (
            row["user_id"],
            int(row["round_index"]),
            int(row["timestamp"]),
            int(row["query_length"]),
            int(row["response_length"]),
        )
        for row in joined_rows
    }
    baseline_common = [baseline_rows[key] for key in common_keys]
    candidate_common = [candidate_rows[key] for key in common_keys]
    baseline_metrics = summarize(baseline_common)
    candidate_metrics = summarize(candidate_common)
    metric_rows = [
        {"label": baseline_label, **baseline_metrics, **deltas(baseline_metrics, baseline_metrics)},
        {"label": candidate_label, **candidate_metrics, **deltas(candidate_metrics, baseline_metrics)},
    ]
    groups = []
    for name in (
        "prefix_class",
        "trace_phase",
        "round_bucket",
        "response_bucket",
        "query_bucket",
        "event_count_bucket",
        "selected_kind",
        "shadow_last_pressure_phase",
        "shadow_last_reason",
    ):
        groups.extend(group_rows(joined_rows, name))
    groups.sort(
        key=lambda row: (
            row["group"],
            -abs(float(row.get("e2e_mean_delta") or 0.0)),
            row["value"],
        )
    )
    gates = [
        evaluate_gate(gate, baseline_rows, candidate_rows, joined_rows, baseline_metrics)
        for gate in make_gates()
    ]
    gates.sort(
        key=lambda row: (
            float(row.get("no_qps_score") or float("-inf")),
            float(row.get("audio_ttfp_p90_reduction_pct") or float("-inf")),
            float(row.get("e2e_p90_reduction_pct") or float("-inf")),
        ),
        reverse=True,
    )
    config = {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "shadow_path": str(shadow_path),
        "common_requests": len(joined_rows),
        "shadow_matched_requests": sum(1 for row in joined_rows if row.get("shadow_has_event")),
        "shadow_applied_events": sum(int(row.get("shadow_applied_events") or 0) for row in joined_rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": {"baseline": baseline_label, "candidate": candidate_label},
        "config": config,
        "metrics": metric_rows,
        "groups": groups,
        "gates": gates,
        "joined_rows": joined_rows,
    }
    json_path = args.output_dir / "event_outcome_join.json"
    groups_path = args.output_dir / "event_outcome_groups.csv"
    gates_path = args.output_dir / "event_selector_search.csv"
    joined_path = args.output_dir / "event_request_join.csv"
    md_path = args.output_dir / "event_outcome_join.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with groups_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "group",
            "value",
            "requests",
            "applied_events",
            "e2e_mean_delta",
            "e2e_p50_delta",
            "e2e_p90_delta",
            "e2e_improved_rate",
            "audio_mean_delta",
            "audio_p90_delta",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in groups)
    with gates_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "selector",
            "candidate_requests",
            "baseline_requests",
            "candidate_applied_events",
            "audio_ttfp_p90",
            "audio_ttfp_p90_reduction_pct",
            "e2e_p90",
            "e2e_p90_reduction_pct",
            "ttft_p90",
            "ttft_p90_reduction_pct",
            "continuity",
            "continuity_delta_pp",
            "no_qps_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in gates)
    with joined_path.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({key for row in joined_rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(joined_rows)
    write_markdown(
        md_path,
        labels={"baseline": baseline_label, "candidate": candidate_label},
        config=config,
        metrics=metric_rows,
        group_rows_out=groups[:120],
        gate_rows=gates,
    )
    print(json_path)
    print(groups_path)
    print(gates_path)
    print(joined_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
