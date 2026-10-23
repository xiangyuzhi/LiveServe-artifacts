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
    patterns = [
        "eviction_only/*/per_request.jsonl",
        "lru_control/*/per_request.jsonl",
        "*/*/per_request.jsonl",
        "**/per_request.jsonl",
    ]
    for pattern in patterns:
        matches = sorted(root_or_path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no per_request.jsonl under {root_or_path}")


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
        row.get("user_id"),
        row.get("round_index"),
        row.get("timestamp"),
        row.get("query_length"),
        row.get("response_length"),
    )


def read_requests(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("error") is not None:
            continue
        rows[trace_key(row)] = row
    return rows


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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audio = [v for row in rows if (v := finite(row.get("audio_ttfp_sec"))) is not None]
    e2e = [v for row in rows if (v := finite(row.get("e2e_sec"))) is not None]
    ttft = [v for row in rows if (v := finite(row.get("ttfp_sec"))) is not None]
    continuity_values = [
        bool(row.get("audio_continuity_ok"))
        for row in rows
        if row.get("audio_continuity_evaluated") is True
    ]
    return {
        "requests": len(rows),
        "audio_ttfp_p90": percentile(audio, 0.90),
        "e2e_p90": percentile(e2e, 0.90),
        "ttft_p90": percentile(ttft, 0.90),
        "continuity": (
            sum(continuity_values) / len(continuity_values)
            if continuity_values
            else None
        ),
    }


def deltas(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    out = {
        "audio_ttfp_reduction_pct": pct_reduction(
            metrics.get("audio_ttfp_p90"),
            baseline.get("audio_ttfp_p90"),
        ),
        "e2e_reduction_pct": pct_reduction(
            metrics.get("e2e_p90"),
            baseline.get("e2e_p90"),
        ),
        "ttft_reduction_pct": pct_reduction(
            metrics.get("ttft_p90"),
            baseline.get("ttft_p90"),
        ),
        "continuity_delta_pp": (
            (metrics.get("continuity") - baseline.get("continuity")) * 100.0
            if isinstance(metrics.get("continuity"), (int, float))
            and isinstance(baseline.get("continuity"), (int, float))
            else None
        ),
    }
    values = [
        out.get("audio_ttfp_reduction_pct"),
        out.get("continuity_delta_pp"),
        out.get("e2e_reduction_pct"),
        out.get("ttft_reduction_pct"),
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


def enrich_features(rows: dict[tuple[Any, ...], dict[str, Any]]) -> None:
    min_ts = min(
        (finite(row.get("timestamp"), 0.0) or 0.0 for row in rows.values()),
        default=0.0,
    )
    for row in rows.values():
        timestamp = finite(row.get("timestamp"), 0.0) or 0.0
        row["_trace_rel_sec"] = timestamp - min_ts
        row["_prefix_tokens"] = finite(row.get("kv_prefix_token_estimate"), 0.0) or 0.0
        row["_response_len"] = finite(row.get("response_length"), 0.0) or 0.0
        row["_query_len"] = finite(row.get("query_length"), 0.0) or 0.0
        row["_input_tokens"] = finite(row.get("input_tokens_total"), 0.0) or 0.0
        row["_round_index"] = finite(row.get("round_index"), 0.0) or 0.0
        row["_audio_duration"] = finite(row.get("audio_duration_sec"), 0.0) or 0.0


def add_threshold_gates(
    gates: list[Gate],
    feature: str,
    cuts: list[float],
    *,
    oracle: bool = False,
) -> None:
    prefix = "oracle:" if oracle else ""
    for cut in cuts:
        gates.append(
            Gate(
                f"{prefix}{feature}>={cut:g}",
                lambda row, f=feature, c=cut: float(row.get(f, 0.0)) >= c,
            )
        )
        gates.append(
            Gate(
                f"{prefix}{feature}<{cut:g}",
                lambda row, f=feature, c=cut: float(row.get(f, 0.0)) < c,
            )
        )


def make_gates(include_oracle_audio: bool) -> list[Gate]:
    gates: list[Gate] = [Gate("all_a", lambda row: True)]
    feature_cuts = {
        "_response_len": [8, 12, 16, 24, 32, 40, 48, 64, 96, 128],
        "_query_len": [8, 16, 24, 32, 48, 64, 96, 128],
        "_input_tokens": [32, 64, 96, 128, 192, 256, 320],
        "_prefix_tokens": [1, 32, 64, 96, 128, 192, 256],
        "_round_index": [1, 3, 10, 20, 30, 45],
        "_trace_rel_sec": [10, 20, 40, 60, 80, 100, 120, 140],
    }
    for feature, cuts in feature_cuts.items():
        add_threshold_gates(gates, feature, cuts)

    and_features = {
        "_response_len": [8, 16, 24, 32, 48, 64],
        "_query_len": [16, 32, 64],
        "_input_tokens": [32, 64, 96, 128],
        "_prefix_tokens": [1, 32, 64, 128],
        "_trace_rel_sec": [20, 60, 100],
        "_round_index": [1, 3, 10],
    }
    pairs = [
        ("_response_len", "_prefix_tokens"),
        ("_response_len", "_input_tokens"),
        ("_query_len", "_prefix_tokens"),
        ("_query_len", "_input_tokens"),
        ("_input_tokens", "_prefix_tokens"),
        ("_trace_rel_sec", "_prefix_tokens"),
        ("_round_index", "_prefix_tokens"),
    ]
    for left, right in pairs:
        for lcut in and_features[left]:
            for rcut in and_features[right]:
                gates.append(
                    Gate(
                        f"{left}>={lcut:g}&{right}>={rcut:g}",
                        lambda row, lf=left, lc=lcut, rf=right, rc=rcut: (
                            float(row.get(lf, 0.0)) >= lc
                            and float(row.get(rf, 0.0)) >= rc
                        ),
                    )
                )
                gates.append(
                    Gate(
                        f"{left}<{lcut:g}|{right}<{rcut:g}",
                        lambda row, lf=left, lc=lcut, rf=right, rc=rcut: (
                            float(row.get(lf, 0.0)) < lc
                            or float(row.get(rf, 0.0)) < rc
                        ),
                    )
                )
    if include_oracle_audio:
        add_threshold_gates(
            gates,
            "_audio_duration",
            [0.5, 1, 2, 4, 8, 16, 32, 64],
            oracle=True,
        )
    return gates


def evaluate_gate(
    gate: Gate,
    baseline_rows: dict[tuple[Any, ...], dict[str, Any]],
    a_rows: dict[tuple[Any, ...], dict[str, Any]],
    b_rows: dict[tuple[Any, ...], dict[str, Any]],
    baseline_metrics: dict[str, Any],
    keys: list[tuple[Any, ...]],
    *,
    direction: str,
) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    a_count = 0
    for key in keys:
        feature_row = baseline_rows[key]
        passed = gate.func(feature_row)
        choose_a = passed if direction == "a_if_gate" else not passed
        if choose_a:
            selected_rows.append(a_rows[key])
            a_count += 1
        else:
            selected_rows.append(b_rows[key])
    metrics = summarize(selected_rows)
    return {
        "selector": f"{direction}:{gate.label}",
        "direction": direction,
        "gate": gate.label,
        "a_requests": a_count,
        "b_requests": len(keys) - a_count,
        **metrics,
        **deltas(metrics, baseline_metrics),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(
    path: Path,
    *,
    baseline_label: str,
    a_label: str,
    b_label: str,
    baseline_metrics: dict[str, Any],
    a_metrics: dict[str, Any],
    b_metrics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Eviction Policy Selector Search",
        "",
        "This search splices completed per-request metrics offline. Gates use fields",
        "known at request admission unless prefixed with `oracle:`. It does not prove",
        "QPS or runtime interaction; use it to decide whether a runtime policy selector",
        "is worth implementing.",
        "",
        f"- baseline: `{baseline_label}`",
        f"- policy A: `{a_label}`",
        f"- policy B: `{b_label}`",
        "",
        "## Plain Sources",
        "",
        "| source | requests | audio_p90 | continuity | e2e_p90 | ttft_p90 | no_qps_score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in [
        (baseline_label, baseline_metrics),
        (a_label, a_metrics),
        (b_label, b_metrics),
    ]:
        score = 0.0 if label == baseline_label else deltas(metrics, baseline_metrics).get("no_qps_score")
        lines.append(
            f"| `{label}` | {metrics['requests']} | "
            f"{fmt(metrics.get('audio_ttfp_p90'))} | "
            f"{fmt(metrics.get('continuity'))} | "
            f"{fmt(metrics.get('e2e_p90'))} | "
            f"{fmt(metrics.get('ttft_p90'))} | {fmt(score)} |"
        )

    feasible = [row for row in rows if not str(row["gate"]).startswith("oracle:")]
    lines.extend(
        [
            "",
            "## Top Feasible Selectors",
            "",
            "| rank | selector | A reqs | B reqs | audio_p90 | audio gain | continuity gain | e2e gain | ttft gain | no_qps_score |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(feasible[:40], 1):
        lines.append(
            f"| {rank} | `{row['selector']}` | {row['a_requests']} | "
            f"{row['b_requests']} | {fmt(row.get('audio_ttfp_p90'))} | "
            f"{fmt(row.get('audio_ttfp_reduction_pct'))}% | "
            f"{fmt(row.get('continuity_delta_pp'))}pp | "
            f"{fmt(row.get('e2e_reduction_pct'))}% | "
            f"{fmt(row.get('ttft_reduction_pct'))}% | "
            f"{fmt(row.get('no_qps_score'))} |"
        )

    oracle = [row for row in rows if str(row["gate"]).startswith("oracle:")]
    if oracle:
        lines.extend(
            [
                "",
                "## Oracle Diagnostic Selectors",
                "",
                "| rank | selector | A reqs | B reqs | audio_p90 | audio gain | continuity gain | e2e gain | ttft gain | no_qps_score |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(oracle[:20], 1):
            lines.append(
                f"| {rank} | `{row['selector']}` | {row['a_requests']} | "
                f"{row['b_requests']} | {fmt(row.get('audio_ttfp_p90'))} | "
                f"{fmt(row.get('audio_ttfp_reduction_pct'))}% | "
                f"{fmt(row.get('continuity_delta_pp'))}pp | "
                f"{fmt(row.get('e2e_reduction_pct'))}% | "
                f"{fmt(row.get('ttft_reduction_pct'))}% | "
                f"{fmt(row.get('no_qps_score'))} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="label=/path/to/run or per_request.jsonl")
    parser.add_argument("--policy-a", required=True, help="label=/path/to/run or per_request.jsonl")
    parser.add_argument("--policy-b", required=True, help="label=/path/to/run or per_request.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-oracle-audio", action="store_true")
    args = parser.parse_args()

    baseline_label, baseline_root = parse_labeled_path(args.baseline)
    a_label, a_root = parse_labeled_path(args.policy_a)
    b_label, b_root = parse_labeled_path(args.policy_b)
    baseline_path = find_per_request(baseline_root)
    a_path = find_per_request(a_root)
    b_path = find_per_request(b_root)
    baseline_rows = read_requests(baseline_path)
    a_rows = read_requests(a_path)
    b_rows = read_requests(b_path)
    common_keys = sorted(set(baseline_rows) & set(a_rows) & set(b_rows))
    if not common_keys:
        raise SystemExit("no common requests")
    enrich_features(baseline_rows)
    enrich_features(a_rows)
    enrich_features(b_rows)
    baseline_metrics = summarize([baseline_rows[key] for key in common_keys])
    a_metrics = summarize([a_rows[key] for key in common_keys])
    b_metrics = summarize([b_rows[key] for key in common_keys])

    rows: list[dict[str, Any]] = []
    for gate in make_gates(args.include_oracle_audio):
        for direction in ("a_if_gate", "a_if_not_gate"):
            rows.append(
                evaluate_gate(
                    gate,
                    baseline_rows,
                    a_rows,
                    b_rows,
                    baseline_metrics,
                    common_keys,
                    direction=direction,
                )
            )
    rows.sort(
        key=lambda row: (
            float(row.get("no_qps_score") or float("-inf")),
            float(row.get("audio_ttfp_reduction_pct") or float("-inf")),
            float(row.get("e2e_reduction_pct") or float("-inf")),
        ),
        reverse=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline_label": baseline_label,
        "policy_a_label": a_label,
        "policy_b_label": b_label,
        "baseline_path": str(baseline_path),
        "policy_a_path": str(a_path),
        "policy_b_path": str(b_path),
        "common_requests": len(common_keys),
        "baseline_metrics": baseline_metrics,
        "policy_a_metrics": a_metrics,
        "policy_b_metrics": b_metrics,
        "rows": rows,
    }
    (args.output_dir / "policy_selector_search.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fields = [
        "selector",
        "direction",
        "gate",
        "a_requests",
        "b_requests",
        "audio_ttfp_p90",
        "audio_ttfp_reduction_pct",
        "continuity",
        "continuity_delta_pp",
        "e2e_p90",
        "e2e_reduction_pct",
        "ttft_p90",
        "ttft_reduction_pct",
        "no_qps_score",
    ]
    with (args.output_dir / "policy_selector_search.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    write_markdown(
        args.output_dir / "policy_selector_search.md",
        baseline_label=baseline_label,
        a_label=a_label,
        b_label=b_label,
        baseline_metrics=baseline_metrics,
        a_metrics=a_metrics,
        b_metrics=b_metrics,
        rows=rows,
    )
    print(args.output_dir / "policy_selector_search.json")
    print(args.output_dir / "policy_selector_search.csv")
    print(args.output_dir / "policy_selector_search.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
