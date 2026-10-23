#!/usr/bin/env python3
"""Analyze pressure-window p90-tail predictors.

This is an offline diagnostic for deciding whether the next small validation
workload should target eviction, scheduler/prefill pressure, or audio-tail
behavior. It keeps feature classes explicit:

- admission: known at request admission in the replay setup;
- trace_oracle: known from the trace but not generally known online;
- observed: measured after the request finishes;
- kv_event: request-attributed KV lookup counters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Feature:
    name: str
    category: str
    func: Callable[[dict[str, Any]], float | None]


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


def find_per_request(path: Path) -> Path:
    if path.is_file():
        return path
    for pattern in (
        "lru_control/*/per_request.jsonl",
        "eviction_only/*/per_request.jsonl",
        "*/*/per_request.jsonl",
        "**/per_request.jsonl",
    ):
        matches = sorted(path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no per_request.jsonl under {path}")


def finite(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else default
    try:
        out = float(str(value))
    except Exception:
        return default
    return out if math.isfinite(out) else default


def numeric(row: dict[str, Any], key: str, default: float | None = None) -> float | None:
    return finite(row.get(key), default)


def percentile(values: list[float], q: float) -> float | None:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return None
    idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[idx]


def describe(values: list[float]) -> dict[str, Any]:
    xs = [v for v in values if math.isfinite(v)]
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs) if xs else None,
        "p50": percentile(xs, 0.50),
        "p90": percentile(xs, 0.90),
        "max": max(xs) if xs else None,
    }


def normalize_request_id(value: Any) -> str | None:
    if value is None:
        return None
    out = str(value)
    for prefix in ("chatcmpl-", "cmpl-"):
        if out.startswith(prefix):
            out = out[len(prefix):]
    return out or None


def trace_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("user_id")),
        int(row.get("round_index") or 0),
        int(row.get("timestamp") or 0),
        int(row.get("query_length") or 0),
        int(row.get("response_length") or 0),
    )


def read_requests(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        if not isinstance(row, dict):
            continue
        if row.get("error") is not None:
            continue
        if row.get("status") not in (None, 200):
            continue
        copied = dict(row)
        copied["_request_id_norm"] = normalize_request_id(row.get("request_id"))
        copied["_trace_key"] = trace_key(row)
        copied.setdefault("lookup_total", 0)
        copied.setdefault("lookup_hits", 0)
        copied.setdefault("lookup_misses", 0)
        copied.setdefault("recent_eviction_misses", 0)
        rows.append(copied)
    return rows


def attach_kv_events(rows: list[dict[str, Any]], path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"enabled": False}
    by_id = {
        str(row["_request_id_norm"]): row
        for row in rows
        if row.get("_request_id_norm")
    }
    counts: dict[str, int] = {}
    unmatched = 0
    for event in iter_jsonl(path):
        if not isinstance(event, dict):
            continue
        kind = str(event.get("kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
        if kind != "kv_lru_lookup":
            continue
        if event.get("dedup_skipped") is True:
            continue
        meta = event.get("request") if isinstance(event.get("request"), dict) else {}
        row = by_id.get(str(normalize_request_id(meta.get("request_id"))))
        if row is None:
            unmatched += 1
            continue
        row["lookup_total"] = int(row.get("lookup_total") or 0) + int(event.get("total") or 0)
        row["lookup_hits"] = int(row.get("lookup_hits") or 0) + int(event.get("hits") or 0)
        row["lookup_misses"] = int(row.get("lookup_misses") or 0) + int(event.get("misses") or 0)
        row["recent_eviction_misses"] = int(row.get("recent_eviction_misses") or 0) + int(
            event.get("recent_eviction_misses") or 0
        )
    return {
        "enabled": True,
        "path": str(path),
        "event_counts": counts,
        "unmatched_lookup_events": unmatched,
    }


def add_derived_fields(rows: list[dict[str, Any]], pressure_start: float | None) -> float:
    timestamps = [numeric(row, "timestamp") for row in rows]
    min_ts = min(v for v in timestamps if v is not None)
    origin = min_ts if pressure_start is None else pressure_start
    for row in rows:
        ts = numeric(row, "timestamp", origin) or origin
        prefix = numeric(row, "kv_prefix_token_estimate", 0.0) or 0.0
        query = numeric(row, "query_length", 0.0) or 0.0
        response = numeric(row, "response_length", 0.0) or 0.0
        input_tokens = numeric(row, "input_tokens_total")
        if input_tokens is None or input_tokens <= 0:
            input_tokens = prefix + query
        row["_trace_rel_sec"] = ts - origin
        row["_input_tokens_effective"] = input_tokens
        row["_log_prefix_tokens"] = math.log1p(max(0.0, prefix))
        row["_log_input_tokens"] = math.log1p(max(0.0, input_tokens))
        row["_response_per_input"] = response / max(1.0, input_tokens)
        row["_query_per_input"] = query / max(1.0, input_tokens)
        row["_has_prefix"] = 1.0 if prefix > 0 else 0.0
        row["_lookup_miss_without_recent"] = (
            1.0
            if int(row.get("lookup_misses") or 0) > 0
            and int(row.get("recent_eviction_misses") or 0) == 0
            else 0.0
        )
    return origin


def features() -> list[Feature]:
    return [
        Feature("trace_rel_sec", "admission", lambda r: numeric(r, "_trace_rel_sec")),
        Feature("timestamp", "admission", lambda r: numeric(r, "timestamp")),
        Feature("round_index", "admission", lambda r: numeric(r, "round_index")),
        Feature("prefix_tokens", "admission", lambda r: numeric(r, "kv_prefix_token_estimate")),
        Feature("has_prefix", "admission", lambda r: numeric(r, "_has_prefix")),
        Feature("query_length", "admission", lambda r: numeric(r, "query_length")),
        Feature("input_tokens", "admission", lambda r: numeric(r, "_input_tokens_effective")),
        Feature("log_prefix_tokens", "admission", lambda r: numeric(r, "_log_prefix_tokens")),
        Feature("log_input_tokens", "admission", lambda r: numeric(r, "_log_input_tokens")),
        Feature("query_per_input", "admission", lambda r: numeric(r, "_query_per_input")),
        Feature("response_length", "trace_oracle", lambda r: numeric(r, "response_length")),
        Feature("response_per_input", "trace_oracle", lambda r: numeric(r, "_response_per_input")),
        Feature("audio_duration_sec", "trace_oracle", lambda r: numeric(r, "audio_duration_sec")),
        Feature("ttft_sec", "observed", lambda r: numeric(r, "ttfp_sec")),
        Feature("lookup_total", "kv_event", lambda r: numeric(r, "lookup_total")),
        Feature("lookup_misses", "kv_event", lambda r: numeric(r, "lookup_misses")),
        Feature("recent_eviction_misses", "kv_event", lambda r: numeric(r, "recent_eviction_misses")),
        Feature("lookup_miss_without_recent", "kv_event", lambda r: numeric(r, "_lookup_miss_without_recent")),
    ]


def label_rows(rows: list[dict[str, Any]], metric: str, q: float) -> tuple[float, set[int]]:
    values = [numeric(row, metric) for row in rows]
    threshold = percentile([v for v in values if v is not None], q)
    if threshold is None:
        return float("nan"), set()
    tail = {
        idx
        for idx, row in enumerate(rows)
        if (value := numeric(row, metric)) is not None and value >= threshold
    }
    return threshold, tail


def gate_stats(
    rows: list[dict[str, Any]],
    tail: set[int],
    feature: Feature,
    op: str,
    cut: float,
    target: str,
) -> dict[str, Any] | None:
    selected: list[int] = []
    for idx, row in enumerate(rows):
        value = feature.func(row)
        if value is None or not math.isfinite(value):
            continue
        if (op == ">=" and value >= cut) or (op == "<" and value < cut):
            selected.append(idx)
    if not selected:
        return None
    tail_total = len(tail)
    covered = sum(1 for idx in selected if idx in tail)
    precision = covered / len(selected) if selected else 0.0
    recall = covered / tail_total if tail_total else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    selected_values = [numeric(rows[idx], target) for idx in selected]
    selected_values = [v for v in selected_values if v is not None]
    return {
        "target": target,
        "feature": feature.name,
        "category": feature.category,
        "op": op,
        "cut": cut,
        "selected": len(selected),
        "tail_covered": covered,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "selected_target_p90": percentile(selected_values, 0.90),
    }


def evaluate_feature_gates(rows: list[dict[str, Any]], target: str, tail: set[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feature in features():
        values = [feature.func(row) for row in rows]
        values = [v for v in values if v is not None and math.isfinite(v)]
        if len(set(values)) < 2:
            continue
        cuts = sorted(
            {
                percentile(values, q)
                for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
                if percentile(values, q) is not None
            }
        )
        for cut in cuts:
            for op in (">=", "<"):
                row = gate_stats(rows, tail, feature, op, float(cut), target)
                if row is not None:
                    out.append(row)
    out.sort(key=lambda r: (r["f1"], r["recall"], r["precision"]), reverse=True)
    return out


def feature_summaries(rows: list[dict[str, Any]], target: str, tail: set[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feature in features():
        tail_values: list[float] = []
        non_tail_values: list[float] = []
        for idx, row in enumerate(rows):
            value = feature.func(row)
            if value is None or not math.isfinite(value):
                continue
            if idx in tail:
                tail_values.append(value)
            else:
                non_tail_values.append(value)
        if not tail_values or not non_tail_values:
            continue
        tail_desc = describe(tail_values)
        non_desc = describe(non_tail_values)
        out.append(
            {
                "target": target,
                "feature": feature.name,
                "category": feature.category,
                "tail_mean": tail_desc["mean"],
                "non_tail_mean": non_desc["mean"],
                "tail_p50": tail_desc["p50"],
                "non_tail_p50": non_desc["p50"],
                "tail_p90": tail_desc["p90"],
                "non_tail_p90": non_desc["p90"],
                "mean_delta": (
                    tail_desc["mean"] - non_desc["mean"]
                    if tail_desc["mean"] is not None and non_desc["mean"] is not None
                    else None
                ),
            }
        )
    out.sort(key=lambda r: abs(r["mean_delta"] or 0.0), reverse=True)
    return out


def pair_candidate(
    baseline_rows: list[dict[str, Any]],
    candidate_path: Path | None,
    tail_sets: dict[str, set[int]],
) -> dict[str, Any] | None:
    if candidate_path is None:
        return None
    candidate_rows = read_requests(find_per_request(candidate_path))
    candidate_by_key = {row["_trace_key"]: row for row in candidate_rows}
    pairs: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for idx, row in enumerate(baseline_rows):
        candidate = candidate_by_key.get(row["_trace_key"])
        if candidate is not None:
            pairs.append((idx, row, candidate))
    groups: dict[str, dict[str, Any]] = {}
    for label, members in {"all": set(range(len(baseline_rows))), **tail_sets}.items():
        selected = [(b, c) for idx, b, c in pairs if idx in members]
        if not selected:
            continue
        audio_delta = [
            (numeric(c, "audio_ttfp_sec", 0.0) or 0.0)
            - (numeric(b, "audio_ttfp_sec", 0.0) or 0.0)
            for b, c in selected
        ]
        e2e_delta = [
            (numeric(c, "e2e_sec", 0.0) or 0.0)
            - (numeric(b, "e2e_sec", 0.0) or 0.0)
            for b, c in selected
        ]
        groups[label] = {
            "pairs": len(selected),
            "audio_delta_mean": sum(audio_delta) / len(audio_delta),
            "audio_delta_p50": percentile(audio_delta, 0.50),
            "audio_delta_p90": percentile(audio_delta, 0.90),
            "e2e_delta_mean": sum(e2e_delta) / len(e2e_delta),
            "e2e_delta_p50": percentile(e2e_delta, 0.50),
            "e2e_delta_p90": percentile(e2e_delta, 0.90),
            "e2e_improved_rate": sum(1 for v in e2e_delta if v < 0.0) / len(e2e_delta),
        }
    return {"candidate_path": str(candidate_path), "groups": groups}


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    *,
    baseline_path: Path,
    rows: list[dict[str, Any]],
    thresholds: dict[str, float],
    gates: dict[str, list[dict[str, Any]]],
    summaries: dict[str, list[dict[str, Any]]],
    top_requests: dict[str, list[dict[str, Any]]],
    kv_info: dict[str, Any],
    candidate: dict[str, Any] | None,
) -> None:
    lines = [
        "# Pressure Tail Predictor Analysis",
        "",
        f"- baseline: `{baseline_path}`",
        f"- analyzed requests: `{len(rows)}`",
        f"- kv events: `{'yes' if kv_info.get('enabled') else 'no'}`",
        "",
        "## Tail Thresholds",
        "",
        "| target | p90 threshold | tail requests |",
        "|---|---:|---:|",
    ]
    for target, threshold in thresholds.items():
        lines.append(f"| `{target}` | {fmt(threshold)} | {len(top_requests[target])} |")

    for target in ("e2e_sec", "audio_ttfp_sec"):
        lines.extend(
            [
                "",
                f"## Top Gates For `{target}` Tail",
                "",
                "| rank | category | feature | gate | selected | covered | precision | recall | f1 |",
                "|---:|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for idx, row in enumerate(gates[target][:20], 1):
            lines.append(
                f"| {idx} | {row['category']} | `{row['feature']}` | "
                f"`{row['feature']}{row['op']}{fmt(row['cut'])}` | "
                f"{row['selected']} | {row['tail_covered']} | "
                f"{fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['f1'])} |"
            )
        lines.extend(
            [
                "",
                f"## Largest Feature Deltas For `{target}` Tail",
                "",
                "| rank | category | feature | tail mean | non-tail mean | delta |",
                "|---:|---|---|---:|---:|---:|",
            ]
        )
        for idx, row in enumerate(summaries[target][:12], 1):
            lines.append(
                f"| {idx} | {row['category']} | `{row['feature']}` | "
                f"{fmt(row['tail_mean'])} | {fmt(row['non_tail_mean'])} | "
                f"{fmt(row['mean_delta'])} |"
            )

    if candidate:
        lines.extend(["", "## Candidate Effect On Tail", ""])
        lines.append("| group | pairs | e2e mean delta | e2e p90 delta | improved rate |")
        lines.append("|---|---:|---:|---:|---:|")
        for group, row in candidate["groups"].items():
            lines.append(
                f"| `{group}` | {row['pairs']} | {fmt(row['e2e_delta_mean'])} | "
                f"{fmt(row['e2e_delta_p90'])} | {fmt(row['e2e_improved_rate'])} |"
            )

    if kv_info.get("enabled"):
        lines.extend(["", "## KV Event Input", "", "```json"])
        lines.append(json.dumps(kv_info, indent=2, sort_keys=True))
        lines.append("```")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--kv-events", type=Path)
    parser.add_argument("--metrics-start-trace-time", type=float)
    parser.add_argument("--metrics-end-trace-time", type=float)
    parser.add_argument("--tail-quantile", type=float, default=0.90)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    baseline_path = find_per_request(args.baseline)
    all_rows = read_requests(baseline_path)
    kv_info = attach_kv_events(all_rows, args.kv_events)
    origin = add_derived_fields(all_rows, args.metrics_start_trace_time)
    rows = [
        row
        for row in all_rows
        if (
            (args.metrics_start_trace_time is None or (numeric(row, "timestamp", -math.inf) or -math.inf) >= args.metrics_start_trace_time)
            and (args.metrics_end_trace_time is None or (numeric(row, "timestamp", math.inf) or math.inf) < args.metrics_end_trace_time)
        )
    ]
    if not rows:
        raise SystemExit("no requests in selected metrics window")
    add_derived_fields(rows, origin)

    thresholds: dict[str, float] = {}
    tail_sets: dict[str, set[int]] = {}
    gates: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, list[dict[str, Any]]] = {}
    top_requests: dict[str, list[dict[str, Any]]] = {}
    for target in ("e2e_sec", "audio_ttfp_sec"):
        threshold, tail = label_rows(rows, target, args.tail_quantile)
        thresholds[target] = threshold
        tail_sets[f"{target}_tail"] = tail
        gates[target] = evaluate_feature_gates(rows, target, tail)
        summaries[target] = feature_summaries(rows, target, tail)
        top_requests[target] = [
            {
                "request_id": row.get("request_id"),
                "user_id": row.get("user_id"),
                "round_index": row.get("round_index"),
                "timestamp": row.get("timestamp"),
                "e2e_sec": row.get("e2e_sec"),
                "audio_ttfp_sec": row.get("audio_ttfp_sec"),
                "ttfp_sec": row.get("ttfp_sec"),
                "audio_duration_sec": row.get("audio_duration_sec"),
                "kv_prefix_token_estimate": row.get("kv_prefix_token_estimate"),
                "query_length": row.get("query_length"),
                "response_length": row.get("response_length"),
                "lookup_total": row.get("lookup_total"),
                "lookup_misses": row.get("lookup_misses"),
                "recent_eviction_misses": row.get("recent_eviction_misses"),
            }
            for idx, row in sorted(
                enumerate(rows),
                key=lambda item: numeric(item[1], target, -math.inf) or -math.inf,
                reverse=True,
            )
            if idx in tail
        ]

    candidate = pair_candidate(rows, args.candidate, tail_sets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pressure_tail_feature_gates.csv", gates["e2e_sec"] + gates["audio_ttfp_sec"])
    write_csv(args.output_dir / "pressure_tail_feature_summary.csv", summaries["e2e_sec"] + summaries["audio_ttfp_sec"])
    write_csv(args.output_dir / "pressure_tail_top_requests.csv", top_requests["e2e_sec"] + top_requests["audio_ttfp_sec"])
    payload = {
        "baseline_path": str(baseline_path),
        "requests": len(rows),
        "metrics_start_trace_time": args.metrics_start_trace_time,
        "metrics_end_trace_time": args.metrics_end_trace_time,
        "tail_quantile": args.tail_quantile,
        "thresholds": thresholds,
        "kv_events": kv_info,
        "top_gates": {target: values[:30] for target, values in gates.items()},
        "feature_summaries": {target: values[:30] for target, values in summaries.items()},
        "candidate": candidate,
    }
    (args.output_dir / "pressure_tail_predictors.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        args.output_dir / "pressure_tail_predictors.md",
        baseline_path=baseline_path,
        rows=rows,
        thresholds=thresholds,
        gates=gates,
        summaries=summaries,
        top_requests=top_requests,
        kv_info=kv_info,
        candidate=candidate,
    )
    print(args.output_dir / "pressure_tail_predictors.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
