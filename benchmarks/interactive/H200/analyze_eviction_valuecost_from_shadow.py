#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(raw)
    return path.name, path


def find_shadow_path(root_or_path: Path) -> Path:
    if root_or_path.is_file():
        return root_or_path
    patterns = [
        "eviction_only/_probe/kv_eviction_shadow.jsonl",
        "*/_probe/kv_eviction_shadow.jsonl",
        "**/kv_eviction_shadow.jsonl",
    ]
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


def metric(summary: dict[str, Any], name: str, qkey: str = "p50") -> float | None:
    value = summary.get(name)
    if not isinstance(value, dict):
        return None
    raw = value.get(qkey)
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        return float(raw)
    return None


def terminal_ratio(summary: dict[str, Any]) -> float:
    terminal = summary.get("terminal")
    count = summary.get("count")
    if isinstance(terminal, (int, float)) and isinstance(count, (int, float)) and count > 0:
        return float(terminal) / float(count)
    return 0.0


def victim_features(summary: dict[str, Any]) -> dict[str, Any]:
    terminal = terminal_ratio(summary) >= 0.5
    gap = metric(summary, "expected_gap_sec_finite", "p50")
    if gap is None:
        gap = math.inf if terminal else 0.0
    return {
        "gap": float(gap),
        "terminal": terminal,
        "terminal_ratio": terminal_ratio(summary),
        "history": metric(summary, "history_tokens_after", "p50") or 0.0,
        "response": metric(summary, "response_length", "p50") or 0.0,
        "lost": metric(summary, "lost_tokens_if_prefix_break", "p50") or 0.0,
    }


def size_bonus(
    *,
    gap: float,
    terminal: bool,
    history: float,
    response: float,
    min_gap: float,
    history_weight: float,
    response_weight: float,
) -> float:
    if not terminal and not math.isinf(gap) and gap < max(0.0, min_gap):
        return 0.0
    return max(0.0, history_weight) * math.log1p(max(0.0, history) / 64.0) + max(
        0.0, response_weight
    ) * math.log1p(max(0.0, response))


def value_cost_score(
    features: dict[str, Any],
    *,
    terminal_bonus: float,
    gap_cap: float,
    loss_weight: float,
    min_gap: float,
    history_weight: float,
    response_weight: float,
) -> float:
    gap = float(features["gap"])
    terminal = bool(features["terminal"])
    if terminal:
        gap_value = max(0.0, terminal_bonus)
    elif math.isinf(gap):
        gap_value = max(0.0, gap_cap) if gap_cap > 0.0 else 0.0
    else:
        gap_value = max(0.0, gap)
        if gap_cap > 0.0:
            gap_value = min(gap_value, gap_cap)
    bonus = size_bonus(
        gap=gap,
        terminal=terminal,
        history=float(features["history"]),
        response=float(features["response"]),
        min_gap=min_gap,
        history_weight=history_weight,
        response_weight=response_weight,
    )
    loss_penalty = max(0.0, loss_weight) * math.log1p(max(0.0, float(features["lost"])))
    return gap_value + bonus - loss_penalty


def load_applied_pairs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        if row.get("applied") is not True and row.get("reason") != "applied":
            continue
        actual = row.get("actual_lru_victim_summary")
        selected = row.get("selected_victim_summary")
        if not isinstance(actual, dict) or not isinstance(selected, dict):
            continue
        rows.append(
            {
                "actual": victim_features(actual),
                "selected": victim_features(selected),
                "replacement_count": row.get("replacement_count"),
                "cached_victim_count": row.get("cached_victim_count"),
                "policy": row.get("policy"),
                "pressure_phase": row.get("pressure_phase"),
            }
        )
    return rows


def protect_actual_ok(features: dict[str, Any], threshold: float) -> bool:
    if threshold <= 0.0:
        return True
    gap = float(features["gap"])
    return not math.isinf(gap) and gap <= threshold


def evaluate(
    pairs: list[dict[str, Any]],
    *,
    terminal_bonus: float,
    gap_cap: float,
    loss_weight: float,
    min_gap: float,
    history_weight: float,
    response_weight: float,
    protect_actual_max_gap: float,
) -> dict[str, Any]:
    protected = 0
    pass_score = 0
    pass_both = 0
    selected_terminal = 0
    selected_far_large = 0
    actual_near_short = 0
    margins: list[float] = []
    for pair in pairs:
        actual = pair["actual"]
        selected = pair["selected"]
        actual_score = value_cost_score(
            actual,
            terminal_bonus=terminal_bonus,
            gap_cap=gap_cap,
            loss_weight=loss_weight,
            min_gap=min_gap,
            history_weight=history_weight,
            response_weight=response_weight,
        )
        selected_score = value_cost_score(
            selected,
            terminal_bonus=terminal_bonus,
            gap_cap=gap_cap,
            loss_weight=loss_weight,
            min_gap=min_gap,
            history_weight=history_weight,
            response_weight=response_weight,
        )
        score_ok = selected_score > actual_score
        protect_ok = protect_actual_ok(actual, protect_actual_max_gap)
        if protect_ok:
            protected += 1
        if score_ok:
            pass_score += 1
        if score_ok and protect_ok:
            pass_both += 1
            if selected["terminal"]:
                selected_terminal += 1
            if (
                (math.isinf(float(selected["gap"])) or float(selected["gap"]) >= 90.0)
                and (float(selected["response"]) >= 64.0 or float(selected["history"]) >= 256.0)
            ):
                selected_far_large += 1
            if (
                not math.isinf(float(actual["gap"]))
                and float(actual["gap"]) <= 30.0
                and float(actual["response"]) <= 32.0
            ):
                actual_near_short += 1
            margins.append(selected_score - actual_score)
    total = len(pairs)
    return {
        "terminal_bonus": terminal_bonus,
        "gap_cap": gap_cap,
        "loss_weight": loss_weight,
        "min_gap": min_gap,
        "history_weight": history_weight,
        "response_weight": response_weight,
        "protect_actual_max_gap": protect_actual_max_gap,
        "events": total,
        "protect_pass": protected,
        "score_pass": pass_score,
        "pass_both": pass_both,
        "protect_pass_rate": protected / total if total else 0.0,
        "score_pass_rate": pass_score / total if total else 0.0,
        "pass_both_rate": pass_both / total if total else 0.0,
        "selected_terminal_rate": selected_terminal / pass_both if pass_both else 0.0,
        "selected_far_large_rate": selected_far_large / pass_both if pass_both else 0.0,
        "actual_near_short_rate": actual_near_short / pass_both if pass_both else 0.0,
        "score_margin_mean": sum(margins) / len(margins) if margins else None,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(path: Path, results: dict[str, list[dict[str, Any]]]) -> None:
    lines = [
        "# Value-Cost Shadow Replay",
        "",
        "This is an approximate replay over already-applied shadow exchanges. It uses",
        "per-event victim-summary p50 features, so it is a parameter screen rather",
        "than a GPU performance result.",
        "",
    ]
    for label, rows in results.items():
        lines.extend(
            [
                f"## {label}",
                "",
                "| terminal_bonus | gap_cap | pass_both_rate | score_pass_rate | "
                "protect_pass_rate | selected_terminal | selected_far_large | "
                "actual_near_short | margin_mean |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(
            rows,
            key=lambda item: (
                -float(item["pass_both_rate"]),
                -float(item["actual_near_short_rate"]),
                float(item["selected_terminal_rate"]),
            ),
        )[:12]:
            lines.append(
                f"| {fmt(row['terminal_bonus'])} | {fmt(row['gap_cap'])} | "
                f"{fmt(row['pass_both_rate'])} | {fmt(row['score_pass_rate'])} | "
                f"{fmt(row['protect_pass_rate'])} | {fmt(row['selected_terminal_rate'])} | "
                f"{fmt(row['selected_far_large_rate'])} | {fmt(row['actual_near_short_rate'])} | "
                f"{fmt(row['score_margin_mean'])} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--terminal-bonus", type=float, action="append", default=[60.0, 80.0, 100.0, 120.0, 150.0])
    parser.add_argument("--gap-cap", type=float, action="append", default=[120.0, 180.0, 240.0])
    parser.add_argument("--loss-weight", type=float, default=1.0)
    parser.add_argument("--min-gap", type=float, default=45.0)
    parser.add_argument("--history-weight", type=float, default=8.0)
    parser.add_argument("--response-weight", type=float, default=2.0)
    parser.add_argument("--protect-actual-max-gap", type=float, default=30.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict[str, Any]]] = {}
    for raw in args.run:
        label, root = parse_labeled_path(raw)
        shadow = find_shadow_path(root)
        pairs = load_applied_pairs(shadow)
        rows = []
        for terminal_bonus in args.terminal_bonus:
            for gap_cap in args.gap_cap:
                row = evaluate(
                    pairs,
                    terminal_bonus=terminal_bonus,
                    gap_cap=gap_cap,
                    loss_weight=args.loss_weight,
                    min_gap=args.min_gap,
                    history_weight=args.history_weight,
                    response_weight=args.response_weight,
                    protect_actual_max_gap=args.protect_actual_max_gap,
                )
                row["label"] = label
                row["shadow_path"] = str(shadow)
                rows.append(row)
        all_results[label] = rows

    json_path = args.output_dir / "valuecost_shadow_replay.json"
    csv_path = args.output_dir / "valuecost_shadow_replay.csv"
    md_path = args.output_dir / "valuecost_shadow_replay.md"
    json_path.write_text(json.dumps({"runs": all_results}, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = [
        "label",
        "terminal_bonus",
        "gap_cap",
        "loss_weight",
        "min_gap",
        "history_weight",
        "response_weight",
        "protect_actual_max_gap",
        "events",
        "protect_pass",
        "score_pass",
        "pass_both",
        "protect_pass_rate",
        "score_pass_rate",
        "pass_both_rate",
        "selected_terminal_rate",
        "selected_far_large_rate",
        "actual_near_short_rate",
        "score_margin_mean",
        "shadow_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rows in all_results.values():
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
    write_markdown(md_path, all_results)
    print(json_path)
    print(csv_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
