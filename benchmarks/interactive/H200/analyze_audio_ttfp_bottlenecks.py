#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class LabeledPath:
    label: str
    path: Path


def parse_labeled_path(raw: str) -> LabeledPath:
    if "=" not in raw:
        path = Path(raw)
        return LabeledPath(path.name, path)
    label, path = raw.split("=", 1)
    return LabeledPath(label.strip(), Path(path.strip()))


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


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def find_first(root: Path, patterns: list[str]) -> Path | None:
    if root.is_file():
        return root
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_per_request(root: Path) -> Path:
    path = find_first(
        root,
        [
            "per_request.jsonl",
            "lru_control/*/per_request.jsonl",
            "eviction_only/*/per_request.jsonl",
            "*/*/per_request.jsonl",
            "**/per_request.jsonl",
        ],
    )
    if path is None:
        raise FileNotFoundError(f"no per_request.jsonl under {root}")
    return path


def find_summary_from_per_request(path: Path) -> Path | None:
    sibling = path.with_name("summary.json")
    if sibling.exists():
        return sibling
    return find_first(path.parent, ["summary.json", "../*/summary.json", "**/summary.json"])


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
        row.get("user_id"),
        row.get("round_index"),
        row.get("timestamp"),
        row.get("query_length"),
        row.get("response_length"),
    )


def finite(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
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


def pct_reduction(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return (baseline - candidate) / baseline * 100.0


def pct_gain(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return (candidate - baseline) / baseline * 100.0


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


def request_indexes(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_trace: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        request_id = row.get("_request_id_norm")
        if request_id:
            by_id[str(request_id)] = row
        by_trace[row["_trace_key"]] = row
    return by_id, by_trace


def row_audio(row: dict[str, Any]) -> float | None:
    return finite(row.get("audio_ttfp_sec"))


def summarize_rows(rows: list[dict[str, Any]], *, wall_time: float | None = None) -> dict[str, Any]:
    audio = [value for row in rows if (value := row_audio(row)) is not None]
    e2e = [value for row in rows if (value := finite(row.get("e2e_sec"))) is not None]
    ttft = [value for row in rows if (value := finite(row.get("ttfp_sec"))) is not None]
    continuity = [
        1.0 if row.get("audio_continuity_ok") else 0.0
        for row in rows
        if row.get("audio_continuity_evaluated") is True
    ]
    return {
        "requests": len(rows),
        "audio_ttfp_p50": percentile(audio, 0.50),
        "audio_ttfp_p90": percentile(audio, 0.90),
        "audio_ttfp_p95": percentile(audio, 0.95),
        "audio_ttfp_p99": percentile(audio, 0.99),
        "audio_ttfp_mean": sum(audio) / len(audio) if audio else None,
        "continuity": sum(continuity) / len(continuity) if continuity else None,
        "e2e_p90": percentile(e2e, 0.90),
        "ttft_p90": percentile(ttft, 0.90),
        "qps": (len(rows) / wall_time) if wall_time and wall_time > 0 else None,
    }


def summary_metrics(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = read_json(find_summary_from_per_request(path))
    totals = summary.get("totals") if isinstance(summary.get("totals"), dict) else {}
    latency = summary.get("latency") if isinstance(summary.get("latency"), dict) else {}
    audio = latency.get("audio_ttfp_sec") if isinstance(latency.get("audio_ttfp_sec"), dict) else {}
    e2e = latency.get("e2e_sec") if isinstance(latency.get("e2e_sec"), dict) else {}
    ttft = latency.get("ttfp_sec") if isinstance(latency.get("ttfp_sec"), dict) else {}
    audio_block = summary.get("audio") if isinstance(summary.get("audio"), dict) else {}
    continuity = audio_block.get("streaming_continuity") if isinstance(audio_block.get("streaming_continuity"), dict) else {}
    if summary:
        return {
            "requests": int(totals.get("completed_ok") or len(rows)),
            "audio_ttfp_p50": finite(audio.get("p50")),
            "audio_ttfp_p90": finite(audio.get("p90")),
            "audio_ttfp_p95": finite(audio.get("p95")),
            "audio_ttfp_p99": finite(audio.get("p99")),
            "audio_ttfp_mean": finite(audio.get("mean")),
            "continuity": finite(continuity.get("ok_rate")),
            "e2e_p90": finite(e2e.get("p90")),
            "ttft_p90": finite(ttft.get("p90")),
            "qps": finite(totals.get("useful_qps")) or finite(totals.get("throughput_rps")),
        }
    return summarize_rows(rows)


def metric_deltas(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    continuity_delta = None
    if isinstance(metrics.get("continuity"), (int, float)) and isinstance(baseline.get("continuity"), (int, float)):
        continuity_delta = (float(metrics["continuity"]) - float(baseline["continuity"])) * 100.0
    return {
        "audio_ttfp_p90_reduction_pct": pct_reduction(metrics.get("audio_ttfp_p90"), baseline.get("audio_ttfp_p90")),
        "qps_gain_pct": pct_gain(metrics.get("qps"), baseline.get("qps")),
        "continuity_delta_pp": continuity_delta,
        "e2e_p90_reduction_pct": pct_reduction(metrics.get("e2e_p90"), baseline.get("e2e_p90")),
        "ttft_p90_reduction_pct": pct_reduction(metrics.get("ttft_p90"), baseline.get("ttft_p90")),
    }


def bucket(value: float, cuts: list[float], labels: list[str]) -> str:
    for cut, label in zip(cuts, labels):
        if value < cut:
            return label
    return labels[-1]


def row_features(row: dict[str, Any]) -> dict[str, float]:
    prefix = finite(row.get("kv_prefix_token_estimate"), 0.0) or 0.0
    input_tokens = finite(row.get("input_tokens_total"), 0.0) or 0.0
    if input_tokens <= 0:
        input_tokens = finite(row.get("usage_prompt_tokens"), 0.0) or 0.0
    return {
        "prefix": prefix,
        "input_tokens": input_tokens,
        "round_index": finite(row.get("round_index"), 0.0) or 0.0,
        "response_len": finite(row.get("response_length"), 0.0) or 0.0,
        "query_len": finite(row.get("query_length"), 0.0) or 0.0,
        "audio_duration": finite(row.get("audio_duration_sec"), 0.0) or 0.0,
        "timestamp": finite(row.get("timestamp"), 0.0) or 0.0,
        "lookup_total": finite(row.get("lookup_total"), 0.0) or 0.0,
        "lookup_misses": finite(row.get("lookup_misses"), 0.0) or 0.0,
        "recent_eviction_misses": finite(row.get("recent_eviction_misses"), 0.0) or 0.0,
        "shape_pad_events": finite(row.get("shape_pad_events"), 0.0) or 0.0,
        "shape_expected_len_max": finite(row.get("shape_expected_len_max"), 0.0) or 0.0,
        "shape_hidden_pad_max": finite(row.get("shape_hidden_pad_max"), 0.0) or 0.0,
    }


def default_groups() -> list[tuple[str, Callable[[dict[str, Any]], bool]]]:
    return [
        ("all", lambda row: True),
        ("audio<1", lambda row: row_features(row)["audio_duration"] < 1),
        ("audio1_4", lambda row: 1 <= row_features(row)["audio_duration"] < 4),
        ("audio4_16", lambda row: 4 <= row_features(row)["audio_duration"] < 16),
        ("audio>=16", lambda row: row_features(row)["audio_duration"] >= 16),
        ("prefix0", lambda row: row_features(row)["prefix"] <= 0),
        ("prefix1_511", lambda row: 0 < row_features(row)["prefix"] < 512),
        ("prefix512_2047", lambda row: 512 <= row_features(row)["prefix"] < 2048),
        ("prefix>=2048", lambda row: row_features(row)["prefix"] >= 2048),
        ("prefix>=4096", lambda row: row_features(row)["prefix"] >= 4096),
        ("round<10", lambda row: row_features(row)["round_index"] < 10),
        ("round10_29", lambda row: 10 <= row_features(row)["round_index"] < 30),
        ("round>=30", lambda row: row_features(row)["round_index"] >= 30),
        ("response<16", lambda row: row_features(row)["response_len"] < 16),
        ("response16_31", lambda row: 16 <= row_features(row)["response_len"] < 32),
        ("response>=32", lambda row: row_features(row)["response_len"] >= 32),
        ("response>=64", lambda row: row_features(row)["response_len"] >= 64),
    ]


def compare_candidate(
    label: str,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, baseline_by_trace = request_indexes(baseline_rows)
    _, candidate_by_trace = request_indexes(candidate_rows)
    common = sorted(set(baseline_by_trace) & set(candidate_by_trace))
    pairs = [(baseline_by_trace[key], candidate_by_trace[key]) for key in common]
    group_rows: list[dict[str, Any]] = []
    for group, predicate in default_groups():
        selected = [(base, cand) for base, cand in pairs if predicate(base)]
        if not selected:
            continue
        base_rows = [base for base, _ in selected]
        cand_rows = [cand for _, cand in selected]
        base_summary = summarize_rows(base_rows)
        cand_summary = summarize_rows(cand_rows)
        deltas = metric_deltas(cand_summary, base_summary)
        audio_deltas = [
            (row_audio(cand) or 0.0) - (row_audio(base) or 0.0)
            for base, cand in selected
            if row_audio(base) is not None and row_audio(cand) is not None
        ]
        group_rows.append(
            {
                "candidate": label,
                "group": group,
                "requests": len(selected),
                "baseline_audio_p90": base_summary.get("audio_ttfp_p90"),
                "candidate_audio_p90": cand_summary.get("audio_ttfp_p90"),
                "audio_ttfp_p90_reduction_pct": deltas.get("audio_ttfp_p90_reduction_pct"),
                "baseline_continuity": base_summary.get("continuity"),
                "candidate_continuity": cand_summary.get("continuity"),
                "continuity_delta_pp": deltas.get("continuity_delta_pp"),
                "baseline_e2e_p90": base_summary.get("e2e_p90"),
                "candidate_e2e_p90": cand_summary.get("e2e_p90"),
                "e2e_p90_reduction_pct": deltas.get("e2e_p90_reduction_pct"),
                "baseline_ttft_p90": base_summary.get("ttft_p90"),
                "candidate_ttft_p90": cand_summary.get("ttft_p90"),
                "ttft_p90_reduction_pct": deltas.get("ttft_p90_reduction_pct"),
                "mean_audio_delta_sec": sum(audio_deltas) / len(audio_deltas) if audio_deltas else None,
                "audio_improve_rate": sum(delta < 0 for delta in audio_deltas) / len(audio_deltas) if audio_deltas else None,
            }
        )

    pair_rows: list[dict[str, Any]] = []
    for base, cand in pairs:
        base_audio = row_audio(base)
        cand_audio = row_audio(cand)
        if base_audio is None or cand_audio is None:
            continue
        features = row_features(base)
        pair_rows.append(
            {
                "candidate": label,
                "request_id": base.get("request_id"),
                "user_id": base.get("user_id"),
                "round_index": base.get("round_index"),
                "timestamp": base.get("timestamp"),
                "audio_delta_sec": cand_audio - base_audio,
                "baseline_audio_ttfp_sec": base_audio,
                "candidate_audio_ttfp_sec": cand_audio,
                "baseline_e2e_sec": finite(base.get("e2e_sec")),
                "candidate_e2e_sec": finite(cand.get("e2e_sec")),
                "baseline_continuity_ok": base.get("audio_continuity_ok"),
                "candidate_continuity_ok": cand.get("audio_continuity_ok"),
                "prefix_tokens": features["prefix"],
                "round_index_feature": features["round_index"],
                "response_len": features["response_len"],
                "audio_duration": features["audio_duration"],
            }
        )
    pair_rows.sort(key=lambda row: float(row["audio_delta_sec"]), reverse=True)
    return group_rows, pair_rows


def merge_kv_events(rows: list[dict[str, Any]], events_path: Path) -> None:
    by_id, _ = request_indexes(rows)
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {
        "lookup_events": 0,
        "lookup_total": 0,
        "lookup_hits": 0,
        "lookup_misses": 0,
        "recent_eviction_misses": 0,
        "touch_events": 0,
        "touch_blocks": 0,
        "rescued_blocks": 0,
    })
    for event in iter_jsonl(events_path):
        if not isinstance(event, dict):
            continue
        request = event.get("request")
        if not isinstance(request, dict):
            continue
        request_id = normalize_request_id(request.get("request_id"))
        if not request_id:
            continue
        row = stats[request_id]
        kind = event.get("kind")
        if kind == "kv_lru_lookup":
            if event.get("dedup_skipped") is True:
                continue
            row["lookup_events"] += 1
            row["lookup_total"] += int(event.get("total") or 0)
            row["lookup_hits"] += int(event.get("hits") or 0)
            row["lookup_misses"] += int(event.get("misses") or 0)
            row["recent_eviction_misses"] += int(event.get("recent_eviction_misses") or 0)
        elif kind == "kv_lru_touch":
            row["touch_events"] += 1
            row["touch_blocks"] += int(event.get("blocks") or 0)
            row["rescued_blocks"] += int(event.get("rescued") or 0)
    for request_id, stat in stats.items():
        row = by_id.get(request_id)
        if row is not None:
            row.update(stat)
    for row in rows:
        for key in (
            "lookup_events",
            "lookup_total",
            "lookup_hits",
            "lookup_misses",
            "recent_eviction_misses",
            "touch_events",
            "touch_blocks",
            "rescued_blocks",
        ):
            row.setdefault(key, 0)


PREFIX_SHAPE_RE = re.compile(
    r"\[OMNI_PREFIX_SHAPE_PAD\]\s+req=(?P<request_id>\S+).*?"
    r"expected_len=(?P<expected_len>\d+).*?"
    r"embed_len=(?P<embed_len>\d+).*?"
    r"hidden_len=(?P<hidden_len>\d+).*?"
    r"embed_pad=(?P<embed_pad>\d+).*?"
    r"hidden_pad=(?P<hidden_pad>\d+)"
)


def merge_prefix_shape_log(rows: list[dict[str, Any]], log_path: Path) -> None:
    by_id, _ = request_indexes(rows)
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {
        "shape_pad_events": 0,
        "shape_expected_len_max": 0,
        "shape_embed_pad_max": 0,
        "shape_hidden_pad_max": 0,
        "shape_embed_pad_sum": 0,
        "shape_hidden_pad_sum": 0,
        "shape_large_pad_events": 0,
    })
    if not log_path.exists():
        return
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = PREFIX_SHAPE_RE.search(line)
            if not match:
                continue
            request_id = normalize_request_id(match.group("request_id"))
            if not request_id:
                continue
            expected_len = int(match.group("expected_len"))
            embed_pad = int(match.group("embed_pad"))
            hidden_pad = int(match.group("hidden_pad"))
            row = stats[request_id]
            row["shape_pad_events"] += 1
            row["shape_expected_len_max"] = max(row["shape_expected_len_max"], expected_len)
            row["shape_embed_pad_max"] = max(row["shape_embed_pad_max"], embed_pad)
            row["shape_hidden_pad_max"] = max(row["shape_hidden_pad_max"], hidden_pad)
            row["shape_embed_pad_sum"] += embed_pad
            row["shape_hidden_pad_sum"] += hidden_pad
            if max(embed_pad, hidden_pad) >= 128:
                row["shape_large_pad_events"] += 1
    for request_id, stat in stats.items():
        row = by_id.get(request_id)
        if row is not None:
            row.update(stat)
    for row in rows:
        for key in (
            "shape_pad_events",
            "shape_expected_len_max",
            "shape_embed_pad_max",
            "shape_hidden_pad_max",
            "shape_embed_pad_sum",
            "shape_hidden_pad_sum",
            "shape_large_pad_events",
        ):
            row.setdefault(key, 0)


def profile_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audio_values = [value for row in rows if (value := row_audio(row)) is not None]
    audio_p90 = percentile(audio_values, 0.90)
    groups: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("all", lambda row: True),
        ("audio_p90_tail", lambda row: audio_p90 is not None and (row_audio(row) or -1.0) >= audio_p90),
        ("recent_miss_any", lambda row: row_features(row)["recent_eviction_misses"] > 0),
        ("recent_miss_none", lambda row: row_features(row)["recent_eviction_misses"] <= 0),
        ("lookup_miss_any", lambda row: row_features(row)["lookup_misses"] > 0),
        ("lookup_miss_none", lambda row: row_features(row)["lookup_misses"] <= 0),
        ("lookup_total>=128", lambda row: row_features(row)["lookup_total"] >= 128),
        ("lookup_total<128", lambda row: row_features(row)["lookup_total"] < 128),
        ("shape_pad_any", lambda row: row_features(row)["shape_pad_events"] > 0),
        ("shape_hidden_pad>=128", lambda row: row_features(row)["shape_hidden_pad_max"] >= 128),
        ("shape_expected>=1024", lambda row: row_features(row)["shape_expected_len_max"] >= 1024),
        ("shape_expected>=4096", lambda row: row_features(row)["shape_expected_len_max"] >= 4096),
        ("prefix>=2048", lambda row: row_features(row)["prefix"] >= 2048),
        ("prefix>=4096", lambda row: row_features(row)["prefix"] >= 4096),
        ("round>=30", lambda row: row_features(row)["round_index"] >= 30),
    ]
    out: list[dict[str, Any]] = []
    tail_rows = [
        row for row in rows
        if audio_p90 is not None and (row_audio(row) or -1.0) >= audio_p90
    ]
    tail_total = max(1, len(tail_rows))
    for label, predicate in groups:
        selected = [row for row in rows if predicate(row)]
        if not selected:
            continue
        metrics = summarize_rows(selected)
        lookups = [row_features(row)["lookup_total"] for row in selected]
        misses = [row_features(row)["lookup_misses"] for row in selected]
        recent = [row_features(row)["recent_eviction_misses"] for row in selected]
        selected_tail = [
            row for row in selected
            if audio_p90 is not None and (row_audio(row) or -1.0) >= audio_p90
        ]
        out.append(
            {
                "group": label,
                **metrics,
                "lookup_total_mean": sum(lookups) / len(lookups) if lookups else None,
                "lookup_misses_mean": sum(misses) / len(misses) if misses else None,
                "recent_misses_mean": sum(recent) / len(recent) if recent else None,
                "tail_rows": len(selected_tail),
                "tail_share": len(selected_tail) / tail_total,
                "group_share": len(selected) / max(1, len(rows)),
            }
        )
    return out


def optimistic_bound_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenarios: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("recent_miss_any", lambda row: row_features(row)["recent_eviction_misses"] > 0),
        ("lookup_miss_any", lambda row: row_features(row)["lookup_misses"] > 0),
        ("shape_hidden_pad>=128", lambda row: row_features(row)["shape_hidden_pad_max"] >= 128),
        ("shape_expected>=1024", lambda row: row_features(row)["shape_expected_len_max"] >= 1024),
        ("prefix>=2048", lambda row: row_features(row)["prefix"] >= 2048),
        ("round>=30", lambda row: row_features(row)["round_index"] >= 30),
    ]
    audio = [value for row in rows if (value := row_audio(row)) is not None]
    baseline_p90 = percentile(audio, 0.90)
    out: list[dict[str, Any]] = []
    for label, predicate in scenarios:
        feature_rows = [row for row in rows if predicate(row) and row_audio(row) is not None]
        non_feature_audio = [
            value for row in rows
            if not predicate(row) and (value := row_audio(row)) is not None
        ]
        if not feature_rows or not non_feature_audio:
            continue
        median_non_feature = percentile(non_feature_audio, 0.50) or 0.0
        zeroed: list[float] = []
        capped: list[float] = []
        for row in rows:
            value = row_audio(row)
            if value is None:
                continue
            if predicate(row):
                zeroed.append(0.0)
                capped.append(min(value, median_non_feature))
            else:
                zeroed.append(value)
                capped.append(value)
        zero_p90 = percentile(zeroed, 0.90)
        capped_p90 = percentile(capped, 0.90)
        out.append(
            {
                "scenario": label,
                "feature_requests": len(feature_rows),
                "feature_share": len(feature_rows) / max(1, len(audio)),
                "baseline_audio_p90": baseline_p90,
                "if_feature_zero_audio_p90": zero_p90,
                "if_feature_zero_reduction_pct": pct_reduction(zero_p90, baseline_p90),
                "if_feature_to_nonfeature_median_p90": capped_p90,
                "if_feature_to_nonfeature_median_reduction_pct": pct_reduction(capped_p90, baseline_p90),
                "nonfeature_audio_median": median_non_feature,
            }
        )
    return out


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = [
        "prefix",
        "input_tokens",
        "round_index",
        "response_len",
        "query_len",
        "audio_duration",
        "lookup_total",
        "lookup_misses",
        "recent_eviction_misses",
        "shape_pad_events",
        "shape_expected_len_max",
        "shape_hidden_pad_max",
    ]
    out: list[dict[str, Any]] = []
    for feature in features:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            audio = row_audio(row)
            if audio is None:
                continue
            xs.append(row_features(row)[feature])
            ys.append(audio)
        corr = pearson(rankdata(xs), rankdata(ys)) if len(set(xs)) > 1 else None
        out.append({"feature": feature, "spearman_audio_ttfp": corr, "n": len(xs)})
    out.sort(key=lambda row: abs(float(row.get("spearman_audio_ttfp") or 0.0)), reverse=True)
    return out


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(
    path: Path,
    *,
    baseline_label: str,
    baseline_metrics: dict[str, Any],
    candidate_overall: list[dict[str, Any]],
    candidate_groups: list[dict[str, Any]],
    profile_label: str | None,
    profile_groups_rows: list[dict[str, Any]],
    bound_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Audio TTFP Bottleneck Analysis",
        "",
        "Metric note: `audio_ttfp_sec` is the target TTFP. `ttfp_sec` is thinker/text TTFT.",
        "",
        "## Clean Performance",
        "",
        "| run | requests | audio p50 | audio p90 | audio gain | qps | qps gain | continuity | cont gain | e2e p90 | e2e gain | ttft p90 | ttft gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {baseline_label} | {baseline_metrics['requests']} | "
            f"{fmt(baseline_metrics.get('audio_ttfp_p50'))} | "
            f"{fmt(baseline_metrics.get('audio_ttfp_p90'))} | 0.000% | "
            f"{fmt(baseline_metrics.get('qps'), 4)} | 0.000% | "
            f"{fmt(baseline_metrics.get('continuity'), 4)} | 0.000pp | "
            f"{fmt(baseline_metrics.get('e2e_p90'))} | 0.000% | "
            f"{fmt(baseline_metrics.get('ttft_p90'))} | 0.000% |"
        ),
    ]
    for row in candidate_overall:
        lines.append(
            f"| {row['candidate']} | {row['requests']} | "
            f"{fmt(row.get('audio_ttfp_p50'))} | {fmt(row.get('audio_ttfp_p90'))} | "
            f"{fmt(row.get('audio_ttfp_p90_reduction_pct'))}% | "
            f"{fmt(row.get('qps'), 4)} | {fmt(row.get('qps_gain_pct'))}% | "
            f"{fmt(row.get('continuity'), 4)} | {fmt(row.get('continuity_delta_pp'))}pp | "
            f"{fmt(row.get('e2e_p90'))} | {fmt(row.get('e2e_p90_reduction_pct'))}% | "
            f"{fmt(row.get('ttft_p90'))} | {fmt(row.get('ttft_p90_reduction_pct'))}% |"
        )

    lines.extend([
        "",
        "## Pair Slices",
        "",
        "| candidate | group | requests | baseline audio p90 | candidate audio p90 | audio gain | cont gain | e2e gain | ttft gain | mean audio delta | improve rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in candidate_groups:
        if row["group"] not in {
            "all",
            "audio<1",
            "audio1_4",
            "audio4_16",
            "audio>=16",
            "prefix>=2048",
            "prefix>=4096",
            "round10_29",
            "round>=30",
            "response>=32",
        }:
            continue
        lines.append(
            f"| {row['candidate']} | {row['group']} | {row['requests']} | "
            f"{fmt(row.get('baseline_audio_p90'))} | {fmt(row.get('candidate_audio_p90'))} | "
            f"{fmt(row.get('audio_ttfp_p90_reduction_pct'))}% | "
            f"{fmt(row.get('continuity_delta_pp'))}pp | "
            f"{fmt(row.get('e2e_p90_reduction_pct'))}% | "
            f"{fmt(row.get('ttft_p90_reduction_pct'))}% | "
            f"{fmt(row.get('mean_audio_delta_sec'))} | "
            f"{fmt(row.get('audio_improve_rate'), 4)} |"
        )

    if profile_groups_rows:
        lines.extend([
            "",
            f"## Profile Attribution: {profile_label}",
            "",
            "Profile data can perturb latency, so use this section for attribution only, not clean performance ranking.",
            "",
            "| group | requests | share | tail share | audio p90 | continuity | e2e p90 | lookup/request | miss/request | recent/request |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in profile_groups_rows:
            lines.append(
                f"| {row['group']} | {row['requests']} | {fmt(row.get('group_share'), 4)} | "
                f"{fmt(row.get('tail_share'), 4)} | {fmt(row.get('audio_ttfp_p90'))} | "
                f"{fmt(row.get('continuity'), 4)} | {fmt(row.get('e2e_p90'))} | "
                f"{fmt(row.get('lookup_total_mean'))} | {fmt(row.get('lookup_misses_mean'))} | "
                f"{fmt(row.get('recent_misses_mean'))} |"
            )

        lines.extend([
            "",
            "## Optimistic Feature Upper Bounds",
            "",
            "The `zero` column is an impossible lower bound that sets all feature requests' audio TTFP to zero. "
            "The `median` column caps those requests at the non-feature median.",
            "",
            "| scenario | requests | share | baseline p90 | zero p90 | zero gain | median p90 | median gain | nonfeature median |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in bound_rows:
            lines.append(
                f"| {row['scenario']} | {row['feature_requests']} | "
                f"{fmt(row.get('feature_share'), 4)} | "
                f"{fmt(row.get('baseline_audio_p90'))} | "
                f"{fmt(row.get('if_feature_zero_audio_p90'))} | "
                f"{fmt(row.get('if_feature_zero_reduction_pct'))}% | "
                f"{fmt(row.get('if_feature_to_nonfeature_median_p90'))} | "
                f"{fmt(row.get('if_feature_to_nonfeature_median_reduction_pct'))}% | "
                f"{fmt(row.get('nonfeature_audio_median'))} |"
            )

        lines.extend([
            "",
            "## Correlations",
            "",
            "| feature | spearman with audio TTFP | n |",
            "|---|---:|---:|",
        ])
        for row in corr_rows[:12]:
            lines.append(
                f"| {row['feature']} | {fmt(row.get('spearman_audio_ttfp'), 4)} | {row['n']} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="label=/path/to/run or per_request.jsonl")
    parser.add_argument("--candidate", action="append", default=[], help="label=/path/to/run or per_request.jsonl")
    parser.add_argument("--profile", help="label=/path/to/profile per_request or run")
    parser.add_argument("--kv-events", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_spec = parse_labeled_path(args.baseline)
    baseline_path = find_per_request(baseline_spec.path)
    baseline_rows = read_requests(baseline_path)
    baseline_metrics = summary_metrics(baseline_path, baseline_rows)

    candidate_overall: list[dict[str, Any]] = []
    candidate_groups: list[dict[str, Any]] = []
    worst_pairs: list[dict[str, Any]] = []
    best_pairs: list[dict[str, Any]] = []
    for raw in args.candidate:
        spec = parse_labeled_path(raw)
        path = find_per_request(spec.path)
        rows = read_requests(path)
        metrics = summary_metrics(path, rows)
        deltas = metric_deltas(metrics, baseline_metrics)
        candidate_overall.append({"candidate": spec.label, **metrics, **deltas, "per_request_path": str(path)})
        group_rows, pair_rows = compare_candidate(spec.label, baseline_rows, rows)
        candidate_groups.extend(group_rows)
        worst_pairs.extend(pair_rows[: args.top_n])
        best_pairs.extend(sorted(pair_rows, key=lambda row: float(row["audio_delta_sec"]))[: args.top_n])

    profile_label = None
    profile_groups_rows: list[dict[str, Any]] = []
    bound_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    if args.profile:
        profile_spec = parse_labeled_path(args.profile)
        profile_label = profile_spec.label
        profile_path = find_per_request(profile_spec.path)
        profile_rows = read_requests(profile_path)
        if args.kv_events:
            merge_kv_events(profile_rows, args.kv_events)
        if args.server_log:
            merge_prefix_shape_log(profile_rows, args.server_log)
        profile_groups_rows = profile_groups(profile_rows)
        bound_rows = optimistic_bound_rows(profile_rows)
        corr_rows = correlations(profile_rows)

    write_csv(
        args.output_dir / "candidate_overall.csv",
        candidate_overall,
        [
            "candidate",
            "requests",
            "audio_ttfp_p50",
            "audio_ttfp_p90",
            "audio_ttfp_p90_reduction_pct",
            "qps",
            "qps_gain_pct",
            "continuity",
            "continuity_delta_pp",
            "e2e_p90",
            "e2e_p90_reduction_pct",
            "ttft_p90",
            "ttft_p90_reduction_pct",
            "per_request_path",
        ],
    )
    write_csv(
        args.output_dir / "candidate_pair_slices.csv",
        candidate_groups,
        [
            "candidate",
            "group",
            "requests",
            "baseline_audio_p90",
            "candidate_audio_p90",
            "audio_ttfp_p90_reduction_pct",
            "baseline_continuity",
            "candidate_continuity",
            "continuity_delta_pp",
            "baseline_e2e_p90",
            "candidate_e2e_p90",
            "e2e_p90_reduction_pct",
            "baseline_ttft_p90",
            "candidate_ttft_p90",
            "ttft_p90_reduction_pct",
            "mean_audio_delta_sec",
            "audio_improve_rate",
        ],
    )
    write_csv(
        args.output_dir / "candidate_audio_worst_pairs.csv",
        worst_pairs,
        [
            "candidate",
            "request_id",
            "user_id",
            "round_index",
            "timestamp",
            "audio_delta_sec",
            "baseline_audio_ttfp_sec",
            "candidate_audio_ttfp_sec",
            "baseline_e2e_sec",
            "candidate_e2e_sec",
            "prefix_tokens",
            "response_len",
            "audio_duration",
        ],
    )
    write_csv(
        args.output_dir / "candidate_audio_best_pairs.csv",
        best_pairs,
        [
            "candidate",
            "request_id",
            "user_id",
            "round_index",
            "timestamp",
            "audio_delta_sec",
            "baseline_audio_ttfp_sec",
            "candidate_audio_ttfp_sec",
            "baseline_e2e_sec",
            "candidate_e2e_sec",
            "prefix_tokens",
            "response_len",
            "audio_duration",
        ],
    )
    if profile_groups_rows:
        write_csv(
            args.output_dir / "profile_groups.csv",
            profile_groups_rows,
            [
                "group",
                "requests",
                "group_share",
                "tail_rows",
                "tail_share",
                "audio_ttfp_p50",
                "audio_ttfp_p90",
                "audio_ttfp_p99",
                "continuity",
                "e2e_p90",
                "ttft_p90",
                "lookup_total_mean",
                "lookup_misses_mean",
                "recent_misses_mean",
            ],
        )
        write_csv(
            args.output_dir / "profile_upper_bounds.csv",
            bound_rows,
            [
                "scenario",
                "feature_requests",
                "feature_share",
                "baseline_audio_p90",
                "if_feature_zero_audio_p90",
                "if_feature_zero_reduction_pct",
                "if_feature_to_nonfeature_median_p90",
                "if_feature_to_nonfeature_median_reduction_pct",
                "nonfeature_audio_median",
            ],
        )
        write_csv(
            args.output_dir / "profile_correlations.csv",
            corr_rows,
            ["feature", "spearman_audio_ttfp", "n"],
        )

    payload = {
        "baseline": {
            "label": baseline_spec.label,
            "per_request_path": str(baseline_path),
            "metrics": baseline_metrics,
        },
        "candidates": candidate_overall,
        "profile_label": profile_label,
        "profile_groups": profile_groups_rows,
        "profile_upper_bounds": bound_rows,
        "profile_correlations": corr_rows,
    }
    (args.output_dir / "audio_ttfp_bottlenecks.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(
        args.output_dir / "audio_ttfp_bottlenecks.md",
        baseline_label=baseline_spec.label,
        baseline_metrics=baseline_metrics,
        candidate_overall=candidate_overall,
        candidate_groups=candidate_groups,
        profile_label=profile_label,
        profile_groups_rows=profile_groups_rows,
        bound_rows=bound_rows,
        corr_rows=corr_rows,
    )
    print(args.output_dir / "audio_ttfp_bottlenecks.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
