#!/usr/bin/env python3
"""Build an offline workload/bottleneck selection report.

This report is intentionally coarse grained. It joins replay summaries,
per-request rows, KV LRU counters, prefetch-hit markers, and paired deltas to
answer whether a run is useful for:

- KV/cache replacement or prefetch work;
- scheduler/first-token work;
- audio-facing tail work.

It does not need GPU access and should be run before spending more GPU time on
candidate sweeps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Cell:
    source: str
    variant: str
    label: str
    per_request_path: Path
    summary_path: Path | None
    run_root: Path
    metrics_start_trace_time: float | None
    metrics_end_trace_time: float | None


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


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


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


def percentile(values: list[float], q: float) -> float | None:
    xs = sorted(value for value in values if math.isfinite(value))
    if not xs:
        return None
    idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[idx]


def describe(values: list[float]) -> dict[str, Any]:
    xs = [value for value in values if math.isfinite(value)]
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs) if xs else None,
        "p50": percentile(xs, 0.50),
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
        "max": max(xs) if xs else None,
    }


def pct_delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return (candidate - baseline) / baseline * 100.0


def pct_reduction(candidate: float | None, baseline: float | None) -> float | None:
    delta = pct_delta(candidate, baseline)
    return -delta if delta is not None else None


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
        rows.append(copied)
    return rows


def filter_rows_for_cell(cell: Cell, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if cell.metrics_start_trace_time is None and cell.metrics_end_trace_time is None:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        ts = finite(row.get("timestamp"))
        if ts is None:
            continue
        if cell.metrics_start_trace_time is not None and ts < cell.metrics_start_trace_time:
            continue
        if cell.metrics_end_trace_time is not None and ts >= cell.metrics_end_trace_time:
            continue
        out.append(row)
    return out


def infer_cell(per_request: Path, source: str) -> Cell:
    summary = per_request.with_name("summary.json")
    summary_path = summary if summary.exists() else None

    # Expected layout:
    #   run_root/variant/cell_name/per_request.jsonl
    cell_dir = per_request.parent
    variant_dir = cell_dir.parent
    run_root = variant_dir.parent
    variant = variant_dir.name
    cell_name = cell_dir.name
    if variant in {".", ""} or run_root == variant_dir:
        variant = cell_name
        run_root = cell_dir.parent
    run_config = read_json(run_root / "run_config.json")
    metrics_start = finite(run_config.get("metrics_start_trace_time"))
    metrics_end = finite(run_config.get("metrics_end_trace_time"))
    label = f"{source}:{variant}/{cell_name}"
    return Cell(
        source=source,
        variant=variant,
        label=label,
        per_request_path=per_request,
        summary_path=summary_path,
        run_root=run_root,
        metrics_start_trace_time=metrics_start,
        metrics_end_trace_time=metrics_end,
    )


def parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.name, path
    label, path = raw.split("=", 1)
    return label.strip(), Path(path.strip())


def discover_cells(source: str, path: Path) -> list[Cell]:
    if path.is_file():
        name = path.name
        if name == "per_request.jsonl":
            return [infer_cell(path, source)]
        if name == "summary.json" and path.with_name("per_request.jsonl").exists():
            return [infer_cell(path.with_name("per_request.jsonl"), source)]
        raise FileNotFoundError(f"unsupported file input: {path}")

    candidates: list[Path] = []
    direct = path / "per_request.jsonl"
    if direct.exists():
        candidates.append(direct)
    candidates.extend(sorted(path.glob("*/*/per_request.jsonl")))
    candidates.extend(sorted(path.glob("*/per_request.jsonl")))
    seen: set[Path] = set()
    cells: list[Cell] = []
    for per_request in candidates:
        resolved = per_request.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        cells.append(infer_cell(per_request, source))
    if not cells:
        raise FileNotFoundError(f"no per_request.jsonl under {path}")
    return cells


def metric_from_summary(summary: dict[str, Any], metric: str, stat: str) -> float | None:
    latency = summary.get("latency") if isinstance(summary.get("latency"), dict) else {}
    block = latency.get(metric) if isinstance(latency.get(metric), dict) else {}
    return finite(block.get(stat))


def audio_metric_from_summary(summary: dict[str, Any], metric: str, stat: str) -> float | None:
    audio = summary.get("audio") if isinstance(summary.get("audio"), dict) else {}
    block = audio.get(metric) if isinstance(audio.get(metric), dict) else {}
    return finite(block.get(stat))


def totals_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    totals = summary.get("totals") if isinstance(summary.get("totals"), dict) else {}
    audio = summary.get("audio") if isinstance(summary.get("audio"), dict) else {}
    continuity = (
        audio.get("streaming_continuity")
        if isinstance(audio.get("streaming_continuity"), dict)
        else {}
    )
    kv = summary.get("kv_lru") if isinstance(summary.get("kv_lru"), dict) else {}
    return {
        "requests": int(totals.get("completed_ok") or totals.get("entries_dispatched") or 0),
        "throughput_rps": finite(totals.get("throughput_rps")),
        "useful_qps": finite(totals.get("useful_qps")),
        "goodput_rps": finite(continuity.get("goodput_rps")),
        "continuity": finite(continuity.get("ok_rate")),
        "kv_enabled": bool(kv.get("enabled")),
        "gpu_evictions": finite(kv.get("gpu_evictions"), 0.0),
        "gpu_lookup_misses": finite(kv.get("gpu_lookup_misses"), 0.0),
        "recent_eviction_misses": finite(kv.get("recent_eviction_misses"), 0.0),
        "recent_eviction_miss_ratio_blocks": finite(kv.get("recent_eviction_miss_ratio_blocks")),
    }


def row_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [value for row in rows if (value := finite(row.get(key))) is not None]


def derived_value(row: dict[str, Any], key: str) -> float | None:
    if key == "prompt_tokens":
        prompt = finite(row.get("usage_prompt_tokens"))
        if prompt is not None:
            return prompt
        prefix = finite(row.get("kv_prefix_token_estimate"), 0.0) or 0.0
        query = finite(row.get("query_length"), 0.0) or 0.0
        total = finite(row.get("input_tokens_total"))
        return total if total is not None and total > 0 else prefix + query
    if key == "output_tokens":
        output = finite(row.get("output_tokens"))
        return output if output is not None else finite(row.get("usage_completion_tokens"))
    return finite(row.get(key))


def summarize_tail(rows: list[dict[str, Any]], metric: str, q: float = 0.90) -> dict[str, Any]:
    values = row_values(rows, metric)
    threshold = percentile(values, q)
    if threshold is None:
        return {"metric": metric, "threshold": None, "tail_n": 0}
    tail = [row for row in rows if (finite(row.get(metric)) or -math.inf) >= threshold]
    n = len(tail)

    def share(pred) -> float | None:
        if not tail:
            return None
        return sum(1 for row in tail if pred(row)) / len(tail)

    return {
        "metric": metric,
        "threshold": threshold,
        "tail_n": n,
        "tail_share": n / max(1, len(rows)),
        "tail_ttft_p50": percentile(row_values(tail, "ttfp_sec"), 0.50),
        "tail_audio_ttfp_p50": percentile(row_values(tail, "audio_ttfp_sec"), 0.50),
        "tail_e2e_p50": percentile(row_values(tail, "e2e_sec"), 0.50),
        "tail_audio_duration_p50": percentile(row_values(tail, "audio_duration_sec"), 0.50),
        "tail_output_tokens_p50": percentile(
            [v for row in tail if (v := derived_value(row, "output_tokens")) is not None],
            0.50,
        ),
        "share_audio_lt_0p5": share(
            lambda r: (derived_value(r, "audio_duration_sec") or math.inf) < 0.5
        ),
        "share_audio_ge_1p5": share(
            lambda r: (derived_value(r, "audio_duration_sec") or -math.inf) >= 1.5
        ),
        "share_output_ge_32": share(
            lambda r: (derived_value(r, "output_tokens") or -math.inf) >= 32
        ),
        "share_round_lt_5": share(
            lambda r: (derived_value(r, "round_index") or math.inf) < 5
        ),
        "share_round_ge_30": share(
            lambda r: (derived_value(r, "round_index") or -math.inf) >= 30
        ),
        "share_prefetch_hit": share(lambda r: r.get("kv_prefetch_hit_used") is True),
    }


def summarize_cell(cell: Cell) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows_all = read_requests(cell.per_request_path)
    rows = filter_rows_for_cell(cell, rows_all)
    summary = read_json(cell.summary_path)
    totals = totals_from_summary(summary)
    row_n = len(rows)
    if totals["requests"] <= 0:
        totals["requests"] = row_n

    prompt_values = [v for row in rows if (v := derived_value(row, "prompt_tokens")) is not None]
    output_values = [v for row in rows if (v := derived_value(row, "output_tokens")) is not None]
    prefix_values = [v for row in rows if (v := finite(row.get("kv_prefix_token_estimate"))) is not None]
    warm_hits = sum(1 for row in rows if row.get("kv_prefetch_hit_used") is True)
    warm_total = sum(1 for row in rows if row.get("kv_prefetch_lookup_state") in {"warm", "cold"})

    cell_row = {
        "label": cell.label,
        "source": cell.source,
        "variant": cell.variant,
        "per_request_path": str(cell.per_request_path),
        "summary_path": str(cell.summary_path) if cell.summary_path else None,
        "metrics_start_trace_time": cell.metrics_start_trace_time,
        "metrics_end_trace_time": cell.metrics_end_trace_time,
        "requests": totals["requests"],
        "row_requests": row_n,
        "throughput_rps": totals["throughput_rps"],
        "useful_qps": totals["useful_qps"],
        "goodput_rps": totals["goodput_rps"],
        "continuity": totals["continuity"],
        "ttft_p50": metric_from_summary(summary, "ttfp_sec", "p50")
        or percentile(row_values(rows, "ttfp_sec"), 0.50),
        "ttft_p90": metric_from_summary(summary, "ttfp_sec", "p90")
        or percentile(row_values(rows, "ttfp_sec"), 0.90),
        "audio_ttfp_p50": metric_from_summary(summary, "audio_ttfp_sec", "p50")
        or percentile(row_values(rows, "audio_ttfp_sec"), 0.50),
        "audio_ttfp_p90": metric_from_summary(summary, "audio_ttfp_sec", "p90")
        or percentile(row_values(rows, "audio_ttfp_sec"), 0.90),
        "e2e_p50": metric_from_summary(summary, "e2e_sec", "p50")
        or percentile(row_values(rows, "e2e_sec"), 0.50),
        "e2e_p90": metric_from_summary(summary, "e2e_sec", "p90")
        or percentile(row_values(rows, "e2e_sec"), 0.90),
        "tokens_per_sec_p50": metric_from_summary(summary, "tokens_per_sec", "p50")
        or percentile(row_values(rows, "tokens_per_sec"), 0.50),
        "audio_duration_p90": audio_metric_from_summary(summary, "audio_duration_sec", "p90")
        or percentile(row_values(rows, "audio_duration_sec"), 0.90),
        "audio_post_ttfp_rtf_p99": audio_metric_from_summary(summary, "audio_post_ttfp_rtf", "p99"),
        "audio_stream_rtf_p99": audio_metric_from_summary(summary, "audio_stream_rtf", "p99"),
        "prompt_p90": percentile(prompt_values, 0.90),
        "output_p90": percentile(output_values, 0.90),
        "prefix_p90": percentile(prefix_values, 0.90),
        "prefetch_warm_hits": warm_hits,
        "prefetch_lookup_rows": warm_total,
        "prefetch_hit_rate": warm_hits / warm_total if warm_total else None,
        **{key: totals[key] for key in (
            "kv_enabled",
            "gpu_evictions",
            "gpu_lookup_misses",
            "recent_eviction_misses",
            "recent_eviction_miss_ratio_blocks",
        )},
    }
    cell_row.update(classify_cell(cell_row))

    tail_rows = [
        {"label": cell.label, **summarize_tail(rows, metric)}
        for metric in ("ttfp_sec", "audio_ttfp_sec", "e2e_sec")
    ]
    return cell_row, tail_rows


def classify_cell(row: dict[str, Any]) -> dict[str, Any]:
    tags: list[str] = []
    notes: list[str] = []
    ttft = finite(row.get("ttft_p90"))
    audio = finite(row.get("audio_ttfp_p90"))
    e2e = finite(row.get("e2e_p90"))
    audio_rtf = finite(row.get("audio_post_ttfp_rtf_p99"))
    kv_miss = finite(row.get("gpu_lookup_misses"), 0.0) or 0.0
    recent = finite(row.get("recent_eviction_misses"), 0.0) or 0.0
    requests = max(1.0, finite(row.get("requests"), 1.0) or 1.0)

    if audio is not None and ttft is not None and audio - ttft > max(1.0, 2.0 * ttft):
        tags.append("audio_first_packet_tail")
        notes.append("audio TTFP p90 is far above text TTFT p90")
    if audio is not None and audio_rtf is not None and audio_rtf < 0.8:
        tags.append("not_audio_stream_rtf_tail")
        notes.append("post-TTFP audio stream RTF is low at p99")
    if audio is None and ttft is not None and e2e is not None and e2e - ttft > 0.3:
        tags.append("decode_or_output_tail")
        notes.append("text-only e2e tail is mostly after first token")
    if kv_miss / requests > 0.5 and recent <= 0:
        tags.append("kv_lookup_not_recent_eviction")
        notes.append("lookup misses exist but recent-eviction misses are absent")
    if recent > 0:
        tags.append("kv_recent_eviction_pressure")
        notes.append("recent eviction misses are present")
    if not tags:
        tags.append("mixed_or_low_signal")
    return {"bottleneck_tags": ",".join(tags), "bottleneck_notes": "; ".join(notes)}


def keyed_rows(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {row["_trace_key"]: row for row in rows}


def pair_rows(base_rows: list[dict[str, Any]], cand_rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidate_by_key = keyed_rows(cand_rows)
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in base_rows:
        candidate = candidate_by_key.get(row["_trace_key"])
        if candidate is not None:
            out.append((row, candidate))
    return out


def delta_values(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    metric: str,
) -> list[float]:
    values: list[float] = []
    for base, cand in pairs:
        base_value = derived_value(base, metric) if metric in {"output_tokens", "prompt_tokens"} else finite(base.get(metric))
        cand_value = derived_value(cand, metric) if metric in {"output_tokens", "prompt_tokens"} else finite(cand.get(metric))
        if base_value is None or cand_value is None:
            continue
        values.append(cand_value - base_value)
    return values


def pair_group(
    pair_name: str,
    group: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    out: dict[str, Any] = {"pair": pair_name, "group": group, "pairs": len(pairs)}
    for metric in ("ttfp_sec", "audio_ttfp_sec", "e2e_sec", "audio_duration_sec", "output_tokens", "prompt_tokens"):
        values = delta_values(pairs, metric)
        desc = describe(values)
        prefix = metric.replace("_sec", "").replace("_tokens", "")
        out[f"{prefix}_delta_mean"] = desc["mean"]
        out[f"{prefix}_delta_p50"] = desc["p50"]
        out[f"{prefix}_delta_p90"] = desc["p90"]
        out[f"{prefix}_delta_p95"] = desc["p95"]
        out[f"{prefix}_faster_rate"] = (
            sum(1 for value in values if value < 0.0) / len(values) if values else None
        )
    return out


def tail_threshold(rows: list[dict[str, Any]], metric: str, q: float = 0.90) -> float | None:
    return percentile(row_values(rows, metric), q)


def compare_cells(pair_name: str, base: Cell, cand: Cell) -> dict[str, Any]:
    base_rows = filter_rows_for_cell(base, read_requests(base.per_request_path))
    cand_rows = filter_rows_for_cell(cand, read_requests(cand.per_request_path))
    pairs = pair_rows(base_rows, cand_rows)

    group_defs: list[tuple[str, Any]] = [
        ("all", lambda b, c: True),
        ("candidate_prefetch_hit", lambda b, c: c.get("kv_prefetch_hit_used") is True),
        ("audio_dur_lt_0p5", lambda b, c: (derived_value(b, "audio_duration_sec") or math.inf) < 0.5),
        ("audio_dur_ge_1p5", lambda b, c: (derived_value(b, "audio_duration_sec") or -math.inf) >= 1.5),
        ("prompt_ge_768", lambda b, c: (derived_value(b, "prompt_tokens") or -math.inf) >= 768),
        ("round_lt_5", lambda b, c: (derived_value(b, "round_index") or math.inf) < 5),
        ("round_ge_30", lambda b, c: (derived_value(b, "round_index") or -math.inf) >= 30),
    ]
    for metric in ("ttfp_sec", "audio_ttfp_sec", "e2e_sec"):
        threshold = tail_threshold(base_rows, metric)
        if threshold is not None:
            group_defs.append(
                (
                    f"base_{metric}_p90_tail",
                    lambda b, c, metric=metric, threshold=threshold: (
                        finite(b.get(metric), -math.inf) or -math.inf
                    )
                    >= threshold,
                )
            )

    group_rows = []
    for group, pred in group_defs:
        selected = [(b, c) for b, c in pairs if pred(b, c)]
        if selected:
            group_rows.append(pair_group(pair_name, group, selected))

    base_summary = read_json(base.summary_path)
    cand_summary = read_json(cand.summary_path)
    base_totals = totals_from_summary(base_summary)
    cand_totals = totals_from_summary(cand_summary)
    all_row = group_rows[0] if group_rows else {"pairs": 0}
    kv_miss_delta = pct_delta(cand_totals.get("gpu_lookup_misses"), base_totals.get("gpu_lookup_misses"))
    eviction_delta = pct_delta(cand_totals.get("gpu_evictions"), base_totals.get("gpu_evictions"))
    decision, reason = classify_pair(all_row, kv_miss_delta, eviction_delta)
    summary = {
        "pair": pair_name,
        "baseline": base.label,
        "candidate": cand.label,
        "paired_requests": len(pairs),
        "baseline_requests": len(base_rows),
        "candidate_requests": len(cand_rows),
        "ttft_delta_p50": all_row.get("ttfp_delta_p50"),
        "ttft_delta_p90": all_row.get("ttfp_delta_p90"),
        "ttft_delta_mean": all_row.get("ttfp_delta_mean"),
        "audio_delta_p50": all_row.get("audio_ttfp_delta_p50"),
        "audio_delta_p90": all_row.get("audio_ttfp_delta_p90"),
        "audio_delta_mean": all_row.get("audio_ttfp_delta_mean"),
        "e2e_delta_p50": all_row.get("e2e_delta_p50"),
        "e2e_delta_p90": all_row.get("e2e_delta_p90"),
        "e2e_delta_mean": all_row.get("e2e_delta_mean"),
        "audio_faster_rate": all_row.get("audio_ttfp_faster_rate"),
        "e2e_faster_rate": all_row.get("e2e_faster_rate"),
        "output_delta_mean": all_row.get("output_delta_mean"),
        "gpu_lookup_miss_delta_pct": kv_miss_delta,
        "gpu_eviction_delta_pct": eviction_delta,
        "recent_eviction_miss_delta": (
            (cand_totals.get("recent_eviction_misses") or 0)
            - (base_totals.get("recent_eviction_misses") or 0)
        ),
        "decision": decision,
        "decision_reason": reason,
    }
    return {"summary": summary, "groups": group_rows}


def classify_pair(
    all_row: dict[str, Any],
    kv_miss_delta_pct: float | None,
    eviction_delta_pct: float | None,
) -> tuple[str, str]:
    ttft_p50 = finite(all_row.get("ttfp_delta_p50"))
    ttft_mean = finite(all_row.get("ttfp_delta_mean"))
    audio_p90 = finite(all_row.get("audio_ttfp_delta_p90"))
    e2e_p90 = finite(all_row.get("e2e_delta_p90"))
    kv_delta = kv_miss_delta_pct if kv_miss_delta_pct is not None else 0.0
    evict_delta = eviction_delta_pct if eviction_delta_pct is not None else 0.0

    if ttft_p50 is not None and ttft_p50 > 0.02 and kv_delta >= -5.0:
        return (
            "reject_for_kv_cache",
            "paired TTFT gets slower and lookup misses do not materially drop",
        )
    if e2e_p90 is not None and e2e_p90 > 0.5 and audio_p90 is not None and audio_p90 > 0.5:
        return (
            "reject_for_tail",
            "paired audio/e2e tail deltas regress by more than 0.5s",
        )
    if kv_delta >= 0.0 and evict_delta >= 0.0:
        return (
            "not_a_kv_load_win",
            "lookup misses and evictions do not improve",
        )
    if ttft_mean is not None and ttft_mean < -0.02 and kv_delta < -5.0:
        return (
            "candidate_for_kv_validation",
            "paired TTFT and lookup misses both improve",
        )
    return (
        "mixed_requires_targeted_check",
        "signals are mixed; use a narrower workload before promotion",
    )


def auto_pairs(cells: list[Cell]) -> list[tuple[str, Cell, Cell]]:
    by_source: dict[str, list[Cell]] = {}
    for cell in cells:
        by_source.setdefault(cell.source, []).append(cell)
    pairs: list[tuple[str, Cell, Cell]] = []
    for source, group in by_source.items():
        controls = [
            cell
            for cell in group
            if cell.variant in {"lru_control", "baseline"} or "/lru_control_" in cell.label
        ]
        if not controls:
            continue
        control = sorted(controls, key=lambda c: c.label)[0]
        for cell in sorted(group, key=lambda c: c.label):
            if cell == control:
                continue
            if cell.variant in {"lru_control", "baseline"}:
                continue
            pairs.append((f"{source}:{control.variant}_vs_{cell.variant}", control, cell))
    return pairs


def parse_pair(raw: str) -> tuple[str, Path, Path]:
    if "=" not in raw:
        raise ValueError("--pair must be label=baseline_path,candidate_path")
    label, paths = raw.split("=", 1)
    parts = paths.split(",", 1)
    if len(parts) != 2:
        raise ValueError("--pair must be label=baseline_path,candidate_path")
    return label.strip(), Path(parts[0].strip()), Path(parts[1].strip())


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def recommendation(cell_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> list[str]:
    decisions = {str(row.get("decision")) for row in pair_rows}
    has_audio_first_tail = any("audio_first_packet_tail" in str(row.get("bottleneck_tags")) for row in cell_rows)
    has_kv_recent = any("kv_recent_eviction_pressure" in str(row.get("bottleneck_tags")) for row in cell_rows)
    has_pair_tail_regression = any(
        (finite(row.get("audio_delta_p90"), 0.0) or 0.0) > 0.5
        and (finite(row.get("e2e_delta_p90"), 0.0) or 0.0) > 0.5
        for row in pair_rows
    )
    has_output_confounded_pair = any(
        abs(finite(row.get("output_delta_mean"), 0.0) or 0.0) > 1.0
        and int(row.get("paired_requests") or 0) >= 1000
        for row in pair_rows
    )
    lines: list[str] = []
    if "reject_for_kv_cache" in decisions or "not_a_kv_load_win" in decisions:
        lines.append(
            "Do not promote current replacement/prefetch variants as KV-load wins; paired TTFT or KV counters do not support it."
        )
    if "reject_for_tail" in decisions or has_pair_tail_regression:
        lines.append(
            "Do not use aggregate audio/e2e p90 alone for promotion; paired tail deltas regress for a large slice."
        )
    if has_output_confounded_pair:
        lines.append(
            "Use forced-output or stricter response-length control for the next original-workload check; output-token drift is large enough to confound aggregate audio/e2e p90."
        )
    if has_audio_first_tail:
        lines.append(
            "For the original audio-facing workload, prioritize scheduler/admission or first-audio-packet attribution before another KV policy sweep."
        )
    if has_kv_recent:
        lines.append(
            "Keep the text-only forced pressure lane as the cheap KV rejection gate, but require hit/miss counters and paired TTFT to move together."
        )
    if not lines:
        lines.append("No strong single bottleneck was isolated; use a narrower target workload before GPU validation.")
    return lines


def write_markdown(
    path: Path,
    *,
    cell_rows: list[dict[str, Any]],
    tail_rows: list[dict[str, Any]],
    pair_summaries: list[dict[str, Any]],
    pair_groups: list[dict[str, Any]],
) -> None:
    lines = [
        "# Workload Bottleneck Selection",
        "",
        "Offline report from per-request rows, summaries, KV counters, prefetch states, and paired deltas.",
        "",
        "## Recommendation",
        "",
    ]
    for item in recommendation(cell_rows, pair_summaries):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Cell Summary",
            "",
            "| cell | req | ttft p90 | audio p90 | e2e p90 | qps | cont | kv miss | kv evict | recent | tags |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in cell_rows:
        lines.append(
            f"| `{row['label']}` | {row['requests']} | {fmt(row.get('ttft_p90'))} | "
            f"{fmt(row.get('audio_ttfp_p90'))} | {fmt(row.get('e2e_p90'))} | "
            f"{fmt(row.get('useful_qps'))} | {fmt(row.get('continuity'), 4)} | "
            f"{fmt(row.get('gpu_lookup_misses'), 0)} | {fmt(row.get('gpu_evictions'), 0)} | "
            f"{fmt(row.get('recent_eviction_misses'), 0)} | `{row.get('bottleneck_tags')}` |"
        )

    if pair_summaries:
        lines.extend(
            [
                "",
                "## Paired Candidate Summary",
                "",
                "| pair | pairs | ttft p50 d | ttft p90 d | audio p50 d | audio p90 d | e2e p50 d | e2e p90 d | kv miss d | evict d | decision |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in pair_summaries:
            lines.append(
                f"| `{row['pair']}` | {row['paired_requests']} | "
                f"{fmt(row.get('ttft_delta_p50'))} | {fmt(row.get('ttft_delta_p90'))} | "
                f"{fmt(row.get('audio_delta_p50'))} | {fmt(row.get('audio_delta_p90'))} | "
                f"{fmt(row.get('e2e_delta_p50'))} | {fmt(row.get('e2e_delta_p90'))} | "
                f"{fmt(row.get('gpu_lookup_miss_delta_pct'))}% | "
                f"{fmt(row.get('gpu_eviction_delta_pct'))}% | `{row.get('decision')}` |"
            )

    lines.extend(
        [
            "",
            "## Tail Composition",
            "",
            "| cell | metric | threshold | tail n | ttft p50 | audio p50 | e2e p50 | short audio | long audio | output>=32 | prefetch hit |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tail_rows:
        if row.get("tail_n", 0) <= 0:
            continue
        lines.append(
            f"| `{row['label']}` | `{row['metric']}` | {fmt(row.get('threshold'))} | "
            f"{row.get('tail_n')} | {fmt(row.get('tail_ttft_p50'))} | "
            f"{fmt(row.get('tail_audio_ttfp_p50'))} | {fmt(row.get('tail_e2e_p50'))} | "
            f"{fmt(row.get('share_audio_lt_0p5'), 4)} | {fmt(row.get('share_audio_ge_1p5'), 4)} | "
            f"{fmt(row.get('share_output_ge_32'), 4)} | {fmt(row.get('share_prefetch_hit'), 4)} |"
        )

    if pair_groups:
        lines.extend(
            [
                "",
                "## Selected Pair Slices",
                "",
                "| pair | group | n | ttft mean d | audio mean d | e2e mean d | e2e p90 d | output mean d |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        keep = {
            "all",
            "candidate_prefetch_hit",
            "base_ttfp_sec_p90_tail",
            "base_audio_ttfp_sec_p90_tail",
            "base_e2e_sec_p90_tail",
            "audio_dur_lt_0p5",
            "audio_dur_ge_1p5",
        }
        for row in pair_groups:
            if row.get("group") not in keep:
                continue
            lines.append(
                f"| `{row['pair']}` | `{row['group']}` | {row['pairs']} | "
                f"{fmt(row.get('ttfp_delta_mean'))} | "
                f"{fmt(row.get('audio_ttfp_delta_mean'))} | "
                f"{fmt(row.get('e2e_delta_mean'))} | "
                f"{fmt(row.get('e2e_delta_p90'))} | "
                f"{fmt(row.get('output_delta_mean'))} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="label=/path/to/run_root_or_per_request")
    parser.add_argument("--pair", action="append", default=[], help="label=baseline_path,candidate_path")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.run and not args.pair:
        raise SystemExit("provide at least one --run or --pair")

    cells: list[Cell] = []
    for raw in args.run:
        source, path = parse_labeled_path(raw)
        cells.extend(discover_cells(source, path))

    explicit_pairs: list[tuple[str, Cell, Cell]] = []
    for raw in args.pair:
        label, base_path, cand_path = parse_pair(raw)
        base_cells = discover_cells(f"{label}_baseline", base_path)
        cand_cells = discover_cells(f"{label}_candidate", cand_path)
        explicit_pairs.append((label, base_cells[0], cand_cells[0]))
        cells.extend([base_cells[0], cand_cells[0]])

    # De-duplicate cells by per-request path.
    by_path: dict[Path, Cell] = {}
    for cell in cells:
        by_path[cell.per_request_path.resolve()] = cell
    cells = list(by_path.values())

    cell_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for cell in sorted(cells, key=lambda c: c.label):
        cell_row, cell_tail = summarize_cell(cell)
        cell_rows.append(cell_row)
        tail_rows.extend(cell_tail)

    pair_summaries: list[dict[str, Any]] = []
    pair_groups: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    for pair_name, base, cand in auto_pairs(cells) + explicit_pairs:
        if pair_name in seen_pairs:
            continue
        seen_pairs.add(pair_name)
        comparison = compare_cells(pair_name, base, cand)
        pair_summaries.append(comparison["summary"])
        pair_groups.extend(comparison["groups"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cell_summary.csv", cell_rows)
    write_csv(args.output_dir / "tail_composition.csv", tail_rows)
    write_csv(args.output_dir / "pair_summary.csv", pair_summaries)
    write_csv(args.output_dir / "pair_slices.csv", pair_groups)
    payload = {
        "cells": cell_rows,
        "tails": tail_rows,
        "pairs": pair_summaries,
        "pair_slices": pair_groups,
        "recommendation": recommendation(cell_rows, pair_summaries),
    }
    (args.output_dir / "workload_bottleneck_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path = args.output_dir / "workload_bottleneck_selection.md"
    write_markdown(
        md_path,
        cell_rows=cell_rows,
        tail_rows=tail_rows,
        pair_summaries=pair_summaries,
        pair_groups=pair_groups,
    )
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
