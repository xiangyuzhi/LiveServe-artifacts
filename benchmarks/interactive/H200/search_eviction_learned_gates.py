#!/usr/bin/env python3
"""Search learned admission gates for eviction-policy diagnostics.

This is an offline diagnostic: it trains simple models from fields available at
request admission, then splices baseline/candidate per-request metrics using
cross-validated predictions. Strong oracle gates but weak learned gates imply
the policy needs a new admission-time feature rather than more hand thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


FEATURE_NAMES = [
    "response_length",
    "query_length",
    "prefix_tokens",
    "input_tokens",
    "round_index",
    "timestamp",
    "log_prefix_tokens",
    "log_input_tokens",
    "response_per_input",
    "query_per_input",
    "response_times_round",
    "prefix_per_round",
]


def find_per_request(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.glob("*/*/per_request.jsonl"))
    if not matches:
        matches = sorted(path.glob("**/per_request.jsonl"))
    if not matches:
        raise FileNotFoundError(f"no per_request.jsonl found under {path}")
    return matches[0]


def load_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            user_id = row.get("user_id")
            round_index = row.get("round_index")
            if user_id is None or not isinstance(round_index, int):
                continue
            rows[(str(user_id), int(round_index))] = row
    return rows


def finite(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def feature_row(row: dict[str, Any]) -> list[float]:
    prefix = finite(row, "kv_prefix_token_estimate")
    response = finite(row, "response_length")
    query = finite(row, "query_length")
    input_tokens = finite(row, "input_tokens_total")
    if input_tokens <= 0:
        input_tokens = finite(row, "usage_prompt_tokens", prefix + query)
    round_index = finite(row, "round_index")
    timestamp = finite(row, "timestamp")
    denom = max(1.0, input_tokens)
    return [
        response,
        query,
        prefix,
        input_tokens,
        round_index,
        timestamp,
        math.log1p(prefix),
        math.log1p(input_tokens),
        response / denom,
        query / denom,
        response * max(1.0, round_index),
        prefix / max(1.0, round_index + 1.0),
    ]


def percentile(values: list[float], q: float) -> float | None:
    xs = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not xs:
        return None
    return xs[min(len(xs) - 1, int((len(xs) - 1) * q))]


def splice_metrics(
    keys: list[tuple[str, int]],
    baseline: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
    mask: np.ndarray,
) -> dict[str, Any]:
    rows = [candidate[key] if bool(use_candidate) else baseline[key]
            for key, use_candidate in zip(keys, mask)]
    audio = [finite(row, "audio_ttfp_sec", math.nan) for row in rows]
    e2e = [finite(row, "e2e_sec", math.nan) for row in rows]
    ttft = [finite(row, "ttfp_sec", math.nan) for row in rows]
    continuity = sum(bool(row.get("audio_continuity_ok")) for row in rows) / max(
        1, len(rows)
    )
    return {
        "candidate_requests": int(mask.sum()),
        "audio_ttfp_p90": percentile(audio, 0.90),
        "continuity": continuity,
        "e2e_p90": percentile(e2e, 0.90),
        "ttft_p90": percentile(ttft, 0.90),
    }


def delta_metrics(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    def pct_reduction(key: str) -> float | None:
        value = metrics.get(key)
        base = baseline.get(key)
        if not isinstance(value, (int, float)) or not isinstance(base, (int, float)):
            return None
        if base == 0:
            return None
        return 100.0 * (base - value) / base

    continuity_delta = None
    if isinstance(metrics.get("continuity"), (int, float)) and isinstance(
        baseline.get("continuity"), (int, float)
    ):
        continuity_delta = 100.0 * (metrics["continuity"] - baseline["continuity"])
    score = 0.0
    for key, weight in (
        ("audio_ttfp_reduction_pct", 0.35),
        ("continuity_delta_pp", 0.20),
        ("e2e_reduction_pct", 0.15),
        ("ttft_reduction_pct", 0.05),
    ):
        pass
    out = {
        "audio_ttfp_reduction_pct": pct_reduction("audio_ttfp_p90"),
        "continuity_delta_pp": continuity_delta,
        "e2e_reduction_pct": pct_reduction("e2e_p90"),
        "ttft_reduction_pct": pct_reduction("ttft_p90"),
    }
    score = (
        0.35 * (out["audio_ttfp_reduction_pct"] or 0.0)
        + 0.20 * (out["continuity_delta_pp"] or 0.0)
        + 0.15 * (out["e2e_reduction_pct"] or 0.0)
        + 0.05 * (out["ttft_reduction_pct"] or 0.0)
    )
    out["no_qps_score"] = score
    return out


def model_specs() -> dict[str, Any]:
    return {
        "tree2": DecisionTreeClassifier(
            max_depth=2, min_samples_leaf=50, random_state=1
        ),
        "tree3": DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=50, random_state=1
        ),
        "tree4": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=40, random_state=1
        ),
        "rf3": RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=40,
            random_state=1,
            n_jobs=1,
        ),
        "gb2": GradientBoostingClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.05, random_state=1
        ),
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
    }


def benefit_model_specs() -> dict[str, Any]:
    return {
        "dt_reg2": DecisionTreeRegressor(
            max_depth=2, min_samples_leaf=60, random_state=2
        ),
        "dt_reg3": DecisionTreeRegressor(
            max_depth=3, min_samples_leaf=60, random_state=2
        ),
        "dt_reg4": DecisionTreeRegressor(
            max_depth=4, min_samples_leaf=50, random_state=2
        ),
        "rf_reg3": RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=50,
            random_state=2,
            n_jobs=1,
        ),
        "gb_reg2": GradientBoostingRegressor(
            n_estimators=60, max_depth=2, learning_rate=0.05, random_state=2
        ),
        "dt_cls3": DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=60, random_state=2
        ),
        "rf_cls3": RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=50,
            random_state=2,
            n_jobs=1,
        ),
        "gb_cls2": GradientBoostingClassifier(
            n_estimators=60, max_depth=2, learning_rate=0.05, random_state=2
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
    cv = GroupKFold(n_splits=5)
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
    keys: list[tuple[str, int]],
    baseline_rows: dict[tuple[str, int], dict[str, Any]],
    candidate_rows: dict[tuple[str, int], dict[str, Any]],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for q in np.linspace(0.05, 0.95, 37):
        cutoff = float(np.quantile(scores, q))
        mask = scores >= cutoff
        metrics = splice_metrics(keys, baseline_rows, candidate_rows, mask)
        deltas = delta_metrics(metrics, baseline_metrics)
        row = {
            "score_quantile": float(q),
            "score_cutoff": cutoff,
            **metrics,
            **deltas,
        }
        if best is None or row["no_qps_score"] > best["no_qps_score"]:
            best = row
    assert best is not None
    return best


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], tree_text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "learned_gate_search.json"
    csv_path = output_dir / "learned_gate_search.csv"
    md_path = output_dir / "learned_gate_search.md"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Learned Eviction Gate Search",
        "",
        "Offline diagnostic using 5-fold GroupKFold by user_id. Gates are "
        "chosen from cross-validated predictions, then evaluated by splicing "
        "candidate requests into the baseline run.",
        "",
        "| rank | target | model | auc | candidate reqs | audio p90 | audio gain | "
        "continuity gain | e2e gain | ttft gain | score |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(rows, key=lambda row: row["no_qps_score"], reverse=True)
    for idx, row in enumerate(ranked[:30], 1):
        lines.append(
            f"| {idx} | `{row['target']}` | `{row['model']}` | "
            f"{row.get('auc', float('nan')):.3f} | "
            f"{row['candidate_requests']} | "
            f"{row['audio_ttfp_p90']:.3f}s | "
            f"{row['audio_ttfp_reduction_pct']:+.3f}% | "
            f"{row['continuity_delta_pp']:+.3f}pp | "
            f"{row['e2e_reduction_pct']:+.3f}% | "
            f"{row['ttft_reduction_pct']:+.3f}% | "
            f"{row['no_qps_score']:.3f} |"
        )
    lines.extend(["", "## Interpretable Benefit Tree", "", "```text", tree_text, "```"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(csv_path)
    print(md_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    baseline_path = find_per_request(args.baseline)
    candidate_path = find_per_request(args.candidate)
    baseline_rows = load_rows(baseline_path)
    candidate_rows = load_rows(candidate_path)
    keys = sorted(
        set(baseline_rows) & set(candidate_rows),
        key=lambda key: (
            finite(baseline_rows[key], "timestamp"),
            key[0],
            key[1],
        ),
    )
    if not keys:
        raise SystemExit("no common (user_id, round_index) requests")

    x = np.asarray([feature_row(baseline_rows[key]) for key in keys], dtype=float)
    groups = np.asarray([key[0] for key in keys])
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
    benefit = -2.0 * audio_delta - 0.2 * e2e_delta
    benefit_label = (benefit > 0.0).astype(int)
    baseline_metrics = splice_metrics(
        keys, baseline_rows, candidate_rows, np.zeros(len(keys), dtype=bool)
    )

    rows: list[dict[str, Any]] = []
    for threshold in (1, 2, 4, 8, 16, 32):
        y = (durations >= threshold).astype(int)
        if y.min() == y.max():
            continue
        for name, model in model_specs().items():
            scores = cross_validated_scores(model, x, y, groups, classifier=True)
            auc = roc_auc_score(y, scores)
            row = best_quantile_gate(
                scores, keys, baseline_rows, candidate_rows, baseline_metrics
            )
            row.update(
                {
                    "target": f"audio_duration>={threshold}",
                    "model": name,
                    "auc": float(auc),
                    "positive_rate": float(y.mean()),
                }
            )
            rows.append(row)

    for name, model in benefit_model_specs().items():
        classifier = "cls" in name
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
        rows.append(row)

    tree = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=60, random_state=2
    ).fit(x, benefit_label)
    tree_text = export_text(tree, feature_names=FEATURE_NAMES)
    write_outputs(args.output_dir, rows, tree_text)


if __name__ == "__main__":
    main()
