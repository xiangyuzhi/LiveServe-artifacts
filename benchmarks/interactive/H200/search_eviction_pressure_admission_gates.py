#!/usr/bin/env python3
"""Search admission-time pressure gates for eviction diagnostics.

This extends search_eviction_learned_gates.py with features that are closer to
the runtime decision point: rolling trace pressure, observed runtime inflight in
the baseline run, and already-completed same-session history. It remains an
offline diagnostic: rows are paired between an LRU baseline and an eviction
candidate, then gates are evaluated by splicing candidate rows into the
baseline row set.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text


TRACE_WINDOWS_SEC = (1, 5, 10, 30, 60, 120, 300)
WALL_WINDOWS_SEC = (1, 5, 15, 30, 60)
TARGET_AUDIO_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
CV_SPLITS = 3


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
    raise FileNotFoundError(f"no per_request.jsonl found under {path}")


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
        str(row.get("user_id")),
        int(row.get("timestamp") or 0),
        int(row.get("round_index") or 0),
        int(row.get("query_length") or 0),
        int(row.get("response_length") or 0),
    )


def load_rows(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("error") is not None:
            continue
        if row.get("status") not in (None, 200):
            continue
        rows[trace_key(row)] = row
    return rows


def finite(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def percentile(values: list[float], q: float) -> float | None:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return None
    idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[idx]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audio = [finite(row, "audio_ttfp_sec", math.nan) for row in rows]
    e2e = [finite(row, "e2e_sec", math.nan) for row in rows]
    ttft = [finite(row, "ttfp_sec", math.nan) for row in rows]
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
        "e2e_p90": percentile(e2e, 0.90),
        "e2e_p95": percentile(e2e, 0.95),
        "ttft_p90": percentile(ttft, 0.90),
        "continuity": sum(continuity) / len(continuity) if continuity else None,
    }


def pct_reduction(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return (baseline - candidate) / baseline * 100.0


def score_delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    continuity_delta = None
    if isinstance(metrics.get("continuity"), (int, float)) and isinstance(
        baseline.get("continuity"), (int, float)
    ):
        continuity_delta = (metrics["continuity"] - baseline["continuity"]) * 100.0
    out = {
        "audio_ttfp_reduction_pct": pct_reduction(
            metrics.get("audio_ttfp_p90"), baseline.get("audio_ttfp_p90")
        ),
        "e2e_reduction_pct": pct_reduction(
            metrics.get("e2e_p90"), baseline.get("e2e_p90")
        ),
        "ttft_reduction_pct": pct_reduction(
            metrics.get("ttft_p90"), baseline.get("ttft_p90")
        ),
        "continuity_delta_pp": continuity_delta,
    }
    out["score"] = (
        0.35 * float(out["audio_ttfp_reduction_pct"] or 0.0)
        + 0.20 * float(out["continuity_delta_pp"] or 0.0)
        + 0.15 * float(out["e2e_reduction_pct"] or 0.0)
        + 0.05 * float(out["ttft_reduction_pct"] or 0.0)
    )
    return out


class RollingTrace:
    def __init__(self, windows: tuple[int, ...]):
        self.windows = windows
        self.queues = {
            window: deque()
            for window in windows
        }
        self.token_sums = {window: 0.0 for window in windows}
        self.prefix_sums = {window: 0.0 for window in windows}

    def add_features(self, row: dict[str, Any]) -> None:
        ts = finite(row, "timestamp")
        for window in self.windows:
            q = self.queues[window]
            while q and q[0][0] < ts - window:
                _, tokens, prefix = q.popleft()
                self.token_sums[window] -= tokens
                self.prefix_sums[window] -= prefix
            row[f"_trace_arrivals_{window}s"] = float(len(q))
            row[f"_trace_tokens_{window}s"] = float(self.token_sums[window])
            row[f"_trace_prefix_tokens_{window}s"] = float(self.prefix_sums[window])

    def push(self, row: dict[str, Any]) -> None:
        ts = finite(row, "timestamp")
        tokens = finite(row, "query_length") + finite(row, "response_length")
        prefix = finite(row, "kv_prefix_token_estimate")
        if prefix <= 0:
            prefix = max(0.0, finite(row, "input_tokens_total") - finite(row, "query_length"))
        for window in self.windows:
            self.queues[window].append((ts, tokens, prefix))
            self.token_sums[window] += tokens
            self.prefix_sums[window] += prefix


class RollingWall:
    def __init__(self, windows: tuple[int, ...]):
        self.windows = windows
        self.queues = {window: deque() for window in windows}

    def add_features(self, row: dict[str, Any]) -> None:
        issued = finite(row, "issued_at", finite(row, "timestamp"))
        for window in self.windows:
            q = self.queues[window]
            while q and q[0] < issued - window:
                q.popleft()
            row[f"_wall_issued_{window}s"] = float(len(q))

    def push(self, row: dict[str, Any]) -> None:
        issued = finite(row, "issued_at", finite(row, "timestamp"))
        for window in self.windows:
            self.queues[window].append(issued)


def enrich_features(rows: dict[tuple[Any, ...], dict[str, Any]]) -> list[str]:
    ordered = sorted(
        rows.values(),
        key=lambda row: (
            finite(row, "timestamp"),
            str(row.get("user_id")),
            finite(row, "round_index"),
        ),
    )
    if not ordered:
        return []

    min_ts = finite(ordered[0], "timestamp")
    max_ts = finite(ordered[-1], "timestamp")
    same_ts_counts = Counter(int(finite(row, "timestamp")) for row in ordered)
    same_ts_seen: Counter[int] = Counter()
    trace_roll = RollingTrace(TRACE_WINDOWS_SEC)

    per_user_audio: dict[str, list[float]] = defaultdict(list)
    per_user_audio_ttfp: dict[str, list[float]] = defaultdict(list)
    per_user_e2e: dict[str, list[float]] = defaultdict(list)
    per_user_last_ts: dict[str, float] = {}
    per_user_hist_tokens: dict[str, float] = defaultdict(float)
    global_hist_tokens = 0.0

    for idx, row in enumerate(ordered):
        uid = str(row.get("user_id"))
        ts = finite(row, "timestamp")
        round_index = finite(row, "round_index")
        prefix = finite(row, "kv_prefix_token_estimate")
        input_tokens = finite(row, "input_tokens_total")
        query = finite(row, "query_length")
        response = finite(row, "response_length")
        if input_tokens <= 0:
            input_tokens = finite(row, "usage_prompt_tokens", prefix + query)
        if prefix <= 0:
            prefix = max(0.0, input_tokens - query)
        denom = max(1.0, input_tokens)

        row["_timestamp"] = ts
        row["_window_elapsed_sec"] = ts - min_ts
        row["_window_remaining_sec"] = max_ts - ts
        row["_arrival_rank_frac"] = idx / max(1.0, len(ordered) - 1.0)
        row["_same_ts_count"] = float(same_ts_counts[int(ts)])
        row["_same_ts_rank"] = float(same_ts_seen[int(ts)])
        same_ts_seen[int(ts)] += 1
        row["_query_len"] = query
        row["_response_len"] = response
        row["_input_tokens"] = input_tokens
        row["_prefix_tokens"] = prefix
        row["_round_index"] = round_index
        row["_log_prefix_tokens"] = math.log1p(prefix)
        row["_log_input_tokens"] = math.log1p(input_tokens)
        row["_response_per_input"] = response / denom
        row["_query_per_input"] = query / denom
        row["_response_times_round"] = response * max(1.0, round_index)
        row["_prefix_per_round"] = prefix / max(1.0, round_index + 1.0)

        hist_audio = per_user_audio[uid]
        hist_audio_ttfp = per_user_audio_ttfp[uid]
        hist_e2e = per_user_e2e[uid]
        row["_hist_count"] = float(len(hist_audio))
        row["_hist_prev_audio_duration"] = hist_audio[-1] if hist_audio else 0.0
        row["_hist_max_audio_duration"] = max(hist_audio) if hist_audio else 0.0
        row["_hist_mean_audio_duration"] = (
            sum(hist_audio) / len(hist_audio) if hist_audio else 0.0
        )
        row["_hist_prev_audio_ttfp"] = hist_audio_ttfp[-1] if hist_audio_ttfp else 0.0
        row["_hist_max_audio_ttfp"] = max(hist_audio_ttfp) if hist_audio_ttfp else 0.0
        row["_hist_mean_audio_ttfp"] = (
            sum(hist_audio_ttfp) / len(hist_audio_ttfp) if hist_audio_ttfp else 0.0
        )
        row["_hist_prev_e2e"] = hist_e2e[-1] if hist_e2e else 0.0
        row["_hist_max_e2e"] = max(hist_e2e) if hist_e2e else 0.0
        row["_hist_mean_e2e"] = sum(hist_e2e) / len(hist_e2e) if hist_e2e else 0.0
        row["_user_gap_since_prev_sec"] = ts - per_user_last_ts.get(uid, ts)
        row["_user_hist_tokens_before"] = per_user_hist_tokens[uid]
        row["_global_hist_tokens_before"] = global_hist_tokens
        row["_active_users_seen"] = float(len(per_user_hist_tokens))

        trace_roll.add_features(row)

        audio = finite(row, "audio_duration_sec", math.nan)
        if math.isfinite(audio) and audio > 0:
            hist_audio.append(audio)
        audio_ttfp = finite(row, "audio_ttfp_sec", math.nan)
        if math.isfinite(audio_ttfp):
            hist_audio_ttfp.append(audio_ttfp)
        e2e = finite(row, "e2e_sec", math.nan)
        if math.isfinite(e2e):
            hist_e2e.append(e2e)
        per_user_last_ts[uid] = ts
        added_hist = query + response
        per_user_hist_tokens[uid] += added_hist
        global_hist_tokens += added_hist
        trace_roll.push(row)

    wall_order = sorted(
        ordered,
        key=lambda row: (
            finite(row, "issued_at", finite(row, "timestamp")),
            finite(row, "timestamp"),
            str(row.get("user_id")),
            finite(row, "round_index"),
        ),
    )
    active_end_heap: list[float] = []
    wall_roll = RollingWall(WALL_WINDOWS_SEC)
    for row in wall_order:
        issued = finite(row, "issued_at", finite(row, "timestamp"))
        while active_end_heap and active_end_heap[0] <= issued:
            heapq.heappop(active_end_heap)
        row["_runtime_inflight_at_issue"] = float(len(active_end_heap))
        wall_roll.add_features(row)
        duration = finite(row, "e2e_sec", 0.0)
        if duration > 0:
            heapq.heappush(active_end_heap, issued + duration)
        wall_roll.push(row)

    feature_names = sorted(
        key for key in ordered[0].keys()
        if key.startswith("_") and key != "_audio_duration"
    )
    return feature_names


def feature_matrix(
    keys: list[tuple[Any, ...]],
    rows: dict[tuple[Any, ...], dict[str, Any]],
    feature_names: list[str],
) -> np.ndarray:
    matrix: list[list[float]] = []
    for key in keys:
        row = rows[key]
        matrix.append([finite(row, name) for name in feature_names])
    return np.asarray(matrix, dtype=float)


def splice_metrics(
    keys: list[tuple[Any, ...]],
    baseline: dict[tuple[Any, ...], dict[str, Any]],
    candidate: dict[tuple[Any, ...], dict[str, Any]],
    mask: np.ndarray,
) -> dict[str, Any]:
    rows = [
        candidate[key] if bool(use_candidate) else baseline[key]
        for key, use_candidate in zip(keys, mask)
    ]
    metrics = summarize(rows)
    metrics["candidate_requests"] = int(mask.sum())
    return metrics


def model_specs() -> dict[str, Any]:
    return {
        "tree3": DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=60, random_state=11
        ),
        "rf3": RandomForestClassifier(
            n_estimators=40,
            max_depth=3,
            min_samples_leaf=50,
            random_state=11,
            n_jobs=1,
        ),
        "gb2": GradientBoostingClassifier(
            n_estimators=25, max_depth=2, learning_rate=0.06, random_state=11
        ),
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
    }


def benefit_model_specs() -> dict[str, tuple[Any, bool]]:
    return {
        "dt_reg3": (
            DecisionTreeRegressor(max_depth=3, min_samples_leaf=60, random_state=12),
            False,
        ),
        "rf_reg3": (
            RandomForestRegressor(
                n_estimators=40,
                max_depth=3,
                min_samples_leaf=50,
                random_state=12,
                n_jobs=1,
            ),
            False,
        ),
        "gb_reg2": (
            GradientBoostingRegressor(
                n_estimators=25, max_depth=2, learning_rate=0.06, random_state=12
            ),
            False,
        ),
        "dt_cls3": (
            DecisionTreeClassifier(max_depth=3, min_samples_leaf=60, random_state=12),
            True,
        ),
        "rf_cls3": (
            RandomForestClassifier(
                n_estimators=40,
                max_depth=3,
                min_samples_leaf=50,
                random_state=12,
                n_jobs=1,
            ),
            True,
        ),
        "gb_cls2": (
            GradientBoostingClassifier(
                n_estimators=25, max_depth=2, learning_rate=0.06, random_state=12
            ),
            True,
        ),
    }


def cross_validated_scores(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    classifier: bool,
) -> np.ndarray:
    scores = np.zeros(len(y), dtype=float)
    n_splits = min(CV_SPLITS, len(set(groups.tolist())))
    if n_splits < 2:
        raise ValueError("need at least two users for grouped cross-validation")
    cv = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in cv.split(x, y, groups):
        y_train = y[train_idx]
        if classifier and len(set(y_train.tolist())) < 2:
            scores[test_idx] = float(y_train[0]) if len(y_train) else 0.0
            continue
        fitted = clone(model)
        fitted.fit(x[train_idx], y_train)
        if classifier and hasattr(fitted, "predict_proba"):
            proba = fitted.predict_proba(x[test_idx])
            classes = getattr(fitted, "classes_", None)
            if classes is None and hasattr(fitted, "named_steps"):
                last_step = list(fitted.named_steps.values())[-1]
                classes = getattr(last_step, "classes_", None)
            if proba.shape[1] == 1:
                cls = classes[0] if classes is not None and len(classes) else 0
                scores[test_idx] = 1.0 if int(cls) == 1 else 0.0
            elif classes is not None and 1 in set(classes.tolist()):
                scores[test_idx] = proba[:, list(classes).index(1)]
            else:
                scores[test_idx] = proba[:, -1]
        elif classifier and hasattr(fitted, "decision_function"):
            scores[test_idx] = fitted.decision_function(x[test_idx])
        else:
            scores[test_idx] = fitted.predict(x[test_idx])
    return scores


def best_quantile_gate(
    scores: np.ndarray,
    keys: list[tuple[Any, ...]],
    baseline_rows: dict[tuple[Any, ...], dict[str, Any]],
    candidate_rows: dict[tuple[Any, ...], dict[str, Any]],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for q in np.linspace(0.05, 0.95, 37):
        cutoff = float(np.quantile(scores, q))
        mask = scores >= cutoff
        metrics = splice_metrics(keys, baseline_rows, candidate_rows, mask)
        row = {
            "score_quantile": float(q),
            "score_cutoff": cutoff,
            **metrics,
            **score_delta(metrics, baseline_metrics),
        }
        if best is None or row["score"] > best["score"]:
            best = row
    assert best is not None
    return best


def run_single_feature_gates(
    keys: list[tuple[Any, ...]],
    baseline_rows: dict[tuple[Any, ...], dict[str, Any]],
    candidate_rows: dict[tuple[Any, ...], dict[str, Any]],
    baseline_metrics: dict[str, Any],
    feature_names: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feature in feature_names:
        values = np.asarray([finite(baseline_rows[key], feature) for key in keys])
        if np.nanmax(values) == np.nanmin(values):
            continue
        cuts = sorted(set(float(np.quantile(values, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)))
        for cut in cuts:
            for op in (">=", "<"):
                if op == ">=":
                    mask = values >= cut
                else:
                    mask = values < cut
                if mask.sum() == 0 or mask.sum() == len(mask):
                    continue
                metrics = splice_metrics(keys, baseline_rows, candidate_rows, mask)
                row = {
                    "gate": f"{feature}{op}{cut:.6g}",
                    "feature": feature,
                    "op": op,
                    "cut": cut,
                    **metrics,
                    **score_delta(metrics, baseline_metrics),
                }
                out.append(row)
    out.sort(key=lambda row: row["score"], reverse=True)
    return out


def feature_importance(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    fitted = clone(model)
    fitted.fit(x, y)
    if hasattr(fitted, "feature_importances_"):
        values = fitted.feature_importances_
    elif hasattr(fitted, "named_steps") and "logisticregression" in fitted.named_steps:
        values = np.abs(fitted.named_steps["logisticregression"].coef_[0])
    else:
        return []
    rows = [
        {"feature": name, "importance": float(value)}
        for name, value in zip(feature_names, values)
    ]
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


def write_outputs(
    output_dir: Path,
    *,
    config: dict[str, Any],
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    learned_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    importances: list[dict[str, Any]],
    tree_text: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pressure_admission_gate_search.json").write_text(
        json.dumps(
            {
                "config": config,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "learned_rows": learned_rows,
                "feature_rows": feature_rows,
                "feature_importance": importances,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with (output_dir / "pressure_admission_learned.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(learned_rows[0].keys()))
        writer.writeheader()
        writer.writerows(learned_rows)
    with (output_dir / "pressure_admission_feature_gates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)

    ranked = sorted(learned_rows, key=lambda row: row["score"], reverse=True)
    lines = [
        "# Pressure Admission Gate Search",
        "",
        "Offline diagnostic with rolling trace pressure, baseline runtime "
        "inflight, and completed same-session history features.",
        "",
        f"- common requests: {config['common_requests']}",
        f"- feature count: {config['feature_count']}",
        "",
        "## Plain Runs",
        "",
        "| run | requests | audio p90 | continuity | e2e p90 | ttft p90 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| baseline | {baseline_metrics['requests']} | "
            f"{fmt(baseline_metrics['audio_ttfp_p90'])} | "
            f"{fmt(baseline_metrics['continuity'])} | "
            f"{fmt(baseline_metrics['e2e_p90'])} | "
            f"{fmt(baseline_metrics['ttft_p90'])} |"
        ),
        (
            f"| candidate | {candidate_metrics['requests']} | "
            f"{fmt(candidate_metrics['audio_ttfp_p90'])} | "
            f"{fmt(candidate_metrics['continuity'])} | "
            f"{fmt(candidate_metrics['e2e_p90'])} | "
            f"{fmt(candidate_metrics['ttft_p90'])} |"
        ),
        "",
        "## Top Learned Gates",
        "",
        "| rank | target | model | auc | candidate reqs | audio p90 | audio gain | "
        "continuity gain | e2e gain | ttft gain | score |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(ranked[:30], 1):
        lines.append(
            f"| {idx} | `{row['target']}` | `{row['model']}` | "
            f"{fmt(row.get('auc'))} | {row['candidate_requests']} | "
            f"{fmt(row['audio_ttfp_p90'])} | "
            f"{fmt(row['audio_ttfp_reduction_pct'])}% | "
            f"{fmt(row['continuity_delta_pp'])}pp | "
            f"{fmt(row['e2e_reduction_pct'])}% | "
            f"{fmt(row['ttft_reduction_pct'])}% | "
            f"{fmt(row['score'])} |"
        )
    lines.extend(
        [
            "",
            "## Top Single-feature Gates",
            "",
            "| rank | gate | candidate reqs | audio p90 | audio gain | "
            "continuity gain | e2e gain | ttft gain | score |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(feature_rows[:30], 1):
        lines.append(
            f"| {idx} | `{row['gate']}` | {row['candidate_requests']} | "
            f"{fmt(row['audio_ttfp_p90'])} | "
            f"{fmt(row['audio_ttfp_reduction_pct'])}% | "
            f"{fmt(row['continuity_delta_pp'])}pp | "
            f"{fmt(row['e2e_reduction_pct'])}% | "
            f"{fmt(row['ttft_reduction_pct'])}% | "
            f"{fmt(row['score'])} |"
        )
    lines.extend(["", "## Best-model Feature Importance", ""])
    if importances:
        lines.extend(["| rank | feature | importance |", "|---:|---|---:|"])
        for idx, row in enumerate(importances[:25], 1):
            lines.append(
                f"| {idx} | `{row['feature']}` | {fmt(row['importance'], 6)} |"
            )
    else:
        lines.append("No native feature importance was available for the best model.")
    lines.extend(["", "## Interpretable Benefit Tree", "", "```text", tree_text, "```"])
    md_path = output_dir / "pressure_admission_gate_search.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(output_dir / "pressure_admission_gate_search.json")
    print(output_dir / "pressure_admission_learned.csv")
    print(output_dir / "pressure_admission_feature_gates.csv")
    print(md_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    baseline_rows = load_rows(find_per_request(args.baseline))
    candidate_rows = load_rows(find_per_request(args.candidate))
    feature_names = enrich_features(baseline_rows)
    keys = sorted(
        set(baseline_rows) & set(candidate_rows),
        key=lambda key: (
            finite(baseline_rows[key], "timestamp"),
            str(key[0]),
            finite(baseline_rows[key], "round_index"),
        ),
    )
    if not keys:
        raise SystemExit("no common successful requests")

    x = feature_matrix(keys, baseline_rows, feature_names)
    groups = np.asarray([str(key[0]) for key in keys])
    baseline_mask = np.zeros(len(keys), dtype=bool)
    candidate_mask = np.ones(len(keys), dtype=bool)
    baseline_metrics = splice_metrics(keys, baseline_rows, candidate_rows, baseline_mask)
    candidate_metrics = splice_metrics(keys, baseline_rows, candidate_rows, candidate_mask)

    durations = np.asarray(
        [finite(baseline_rows[key], "audio_duration_sec") for key in keys],
        dtype=float,
    )
    audio_delta = np.asarray(
        [
            finite(candidate_rows[key], "audio_ttfp_sec")
            - finite(baseline_rows[key], "audio_ttfp_sec")
            for key in keys
        ],
        dtype=float,
    )
    e2e_delta = np.asarray(
        [
            finite(candidate_rows[key], "e2e_sec")
            - finite(baseline_rows[key], "e2e_sec")
            for key in keys
        ],
        dtype=float,
    )
    continuity_delta = np.asarray(
        [
            (1.0 if candidate_rows[key].get("audio_continuity_ok") else 0.0)
            - (1.0 if baseline_rows[key].get("audio_continuity_ok") else 0.0)
            for key in keys
        ],
        dtype=float,
    )
    benefit = -2.0 * audio_delta - 0.2 * e2e_delta + 0.5 * continuity_delta
    benefit_label = (benefit > 0.0).astype(int)

    learned_rows: list[dict[str, Any]] = []
    classifier_specs = model_specs()
    for threshold in TARGET_AUDIO_THRESHOLDS:
        y = (durations >= threshold).astype(int)
        if y.min() == y.max():
            continue
        for name, model in classifier_specs.items():
            scores = cross_validated_scores(model, x, y, groups, classifier=True)
            auc = roc_auc_score(y, scores)
            row = best_quantile_gate(
                scores, keys, baseline_rows, candidate_rows, baseline_metrics
            )
            row.update(
                {
                    "target": f"audio_duration>={threshold:g}",
                    "model": name,
                    "auc": float(auc),
                    "positive_rate": float(y.mean()),
                }
            )
            learned_rows.append(row)

    all_benefit_models = benefit_model_specs()
    for name, (model, classifier) in all_benefit_models.items():
        y = benefit_label if classifier else benefit
        scores = cross_validated_scores(model, x, y, groups, classifier=classifier)
        auc = roc_auc_score(benefit_label, scores)
        row = best_quantile_gate(
            scores, keys, baseline_rows, candidate_rows, baseline_metrics
        )
        row.update(
            {
                "target": "request_benefit",
                "model": name,
                "auc": float(auc),
                "positive_rate": float(benefit_label.mean()),
            }
        )
        learned_rows.append(row)

    feature_rows = run_single_feature_gates(
        keys, baseline_rows, candidate_rows, baseline_metrics, feature_names
    )
    best = max(learned_rows, key=lambda row: row["score"])
    if best["target"].startswith("audio_duration>="):
        threshold = float(best["target"].split(">=", 1)[1])
        best_y = (durations >= threshold).astype(int)
        best_model = classifier_specs[best["model"]]
    else:
        best_model, best_is_classifier = all_benefit_models[best["model"]]
        best_y = benefit_label if best_is_classifier else benefit
    importances = feature_importance(best_model, x, best_y, feature_names)
    tree = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=50, random_state=13
    ).fit(x, benefit_label)
    tree_text = export_text(tree, feature_names=feature_names)

    write_outputs(
        args.output_dir,
        config={
            "baseline": str(args.baseline),
            "candidate": str(args.candidate),
            "common_requests": len(keys),
            "feature_count": len(feature_names),
            "trace_windows_sec": list(TRACE_WINDOWS_SEC),
            "wall_windows_sec": list(WALL_WINDOWS_SEC),
            "cv_splits": CV_SPLITS,
        },
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        learned_rows=learned_rows,
        feature_rows=feature_rows,
        importances=importances,
        tree_text=tree_text,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
