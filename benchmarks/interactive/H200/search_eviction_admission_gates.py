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
    if "=" not in raw:
        path = Path(raw)
        return path.name, path
    label, path = raw.split("=", 1)
    return label.strip(), Path(path.strip())


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


def pct_gain(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return (candidate - baseline) / baseline * 100.0


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
    min_ts = min((finite(row.get("timestamp"), 0.0) or 0.0 for row in rows.values()), default=0.0)
    for row in rows.values():
        timestamp = finite(row.get("timestamp"), 0.0) or 0.0
        row["_trace_rel_sec"] = timestamp - min_ts
        row["_prefix_tokens"] = finite(row.get("kv_prefix_token_estimate"), 0.0) or 0.0
        row["_response_len"] = finite(row.get("response_length"), 0.0) or 0.0
        row["_query_len"] = finite(row.get("query_length"), 0.0) or 0.0
        row["_input_tokens"] = finite(row.get("input_tokens_total"), 0.0) or 0.0
        row["_round_index"] = finite(row.get("round_index"), 0.0) or 0.0
        row["_audio_duration"] = finite(row.get("audio_duration_sec"), 0.0) or 0.0


def make_single_feature_gates(
    feature: str,
    cuts: list[float],
    *,
    oracle: bool = False,
) -> list[Gate]:
    gates: list[Gate] = []
    prefix = "oracle:" if oracle else ""
    for cut in cuts:
        gates.append(Gate(f"{prefix}{feature}>={cut:g}", lambda row, f=feature, c=cut: float(row.get(f, 0.0)) >= c))
        gates.append(Gate(f"{prefix}{feature}<{cut:g}", lambda row, f=feature, c=cut: float(row.get(f, 0.0)) < c))
    return gates


def make_gates(include_oracle_audio: bool) -> list[Gate]:
    feature_cuts = {
        "_response_len": [8, 12, 16, 24, 32, 40, 48, 64, 96, 128],
        "_query_len": [8, 16, 24, 32, 48, 64, 96, 128],
        "_input_tokens": [32, 64, 96, 128, 192, 256, 320, 512, 1024, 2048],
        "_prefix_tokens": [1, 32, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096],
        "_round_index": [1, 3, 10, 15, 20, 25, 30, 35, 45],
        "_trace_rel_sec": [10, 20, 40, 60, 80, 100, 120, 140],
    }
    gates = [Gate("all_candidate", lambda row: True)]
    for feature, cuts in feature_cuts.items():
        gates.extend(make_single_feature_gates(feature, cuts))

    response_cuts = [16, 24, 32, 48, 64]
    prefix_cuts = [1, 64, 128, 256, 512, 1024, 2048]
    round_cuts = [1, 10, 15, 20, 25, 30, 35]
    phase_cuts = [20, 60, 120]
    for response in response_cuts:
        for prefix in prefix_cuts:
            gates.append(
                Gate(
                    f"_response_len>={response}&_prefix_tokens>={prefix}",
                    lambda row, r=response, p=prefix: float(row.get("_response_len", 0.0)) >= r
                    and float(row.get("_prefix_tokens", 0.0)) >= p,
                )
            )
            gates.append(
                Gate(
                    f"_response_len<{response}|_prefix_tokens<{prefix}",
                    lambda row, r=response, p=prefix: float(row.get("_response_len", 0.0)) < r
                    or float(row.get("_prefix_tokens", 0.0)) < p,
                )
            )
    for response in response_cuts:
        for round_idx in round_cuts:
            gates.append(
                Gate(
                    f"_response_len>={response}&_round_index>={round_idx}",
                    lambda row, r=response, g=round_idx: float(row.get("_response_len", 0.0)) >= r
                    and float(row.get("_round_index", 0.0)) >= g,
                )
            )
    for prefix in [512, 1024, 2048]:
        gates.append(
            Gate(
                f"_prefix_tokens>={prefix}&_round_index>=10&_round_index<30",
                lambda row, p=prefix: float(row.get("_prefix_tokens", 0.0)) >= p
                and 10 <= float(row.get("_round_index", 0.0)) < 30,
            )
        )
        gates.append(
            Gate(
                f"_prefix_tokens<{prefix}|_round_index<10|_round_index>=30",
                lambda row, p=prefix: float(row.get("_prefix_tokens", 0.0)) < p
                or float(row.get("_round_index", 0.0)) < 10
                or float(row.get("_round_index", 0.0)) >= 30,
            )
        )
    for response in response_cuts:
        for phase in phase_cuts:
            gates.append(
                Gate(
                    f"_response_len>={response}&_trace_rel_sec>={phase}",
                    lambda row, r=response, p=phase: float(row.get("_response_len", 0.0)) >= r
                    and float(row.get("_trace_rel_sec", 0.0)) >= p,
                )
            )

    if include_oracle_audio:
        gates.extend(
            make_single_feature_gates(
                "_audio_duration",
                [0.5, 1, 2, 4, 8, 16, 32, 64],
                oracle=True,
            )
        )
    return gates


def evaluate_gate(
    gate: Gate,
    baseline_rows: dict[tuple[Any, ...], dict[str, Any]],
    candidate_rows: dict[tuple[Any, ...], dict[str, Any]],
    baseline_metrics: dict[str, Any],
    keys: list[tuple[Any, ...]],
) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    candidate_count = 0
    for key in keys:
        brow = baseline_rows[key]
        if gate.func(brow):
            selected_rows.append(candidate_rows[key])
            candidate_count += 1
        else:
            selected_rows.append(brow)
    metrics = summarize(selected_rows)
    result = {
        "gate": gate.label,
        "candidate_requests": candidate_count,
        "baseline_requests": len(keys) - candidate_count,
        **metrics,
        **deltas(metrics, baseline_metrics),
    }
    return result


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
    candidate_label: str,
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Eviction Admission Gate Search",
        "",
        "This search splices per-request metrics offline. Gates use only fields known "
        "at request admission unless prefixed with `oracle:`. It does not prove QPS; "
        "use it to choose a small number of GPU6-7 validation runs.",
        "",
        f"- baseline: `{baseline_label}`",
        f"- candidate: `{candidate_label}`",
        "",
        "## Plain Candidate",
        "",
        "| source | requests | audio_p90 | continuity | e2e_p90 | ttft_p90 | no_qps_score |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| `{baseline_label}` | {baseline_metrics['requests']} | "
            f"{fmt(baseline_metrics.get('audio_ttfp_p90'))} | "
            f"{fmt(baseline_metrics.get('continuity'))} | "
            f"{fmt(baseline_metrics.get('e2e_p90'))} | "
            f"{fmt(baseline_metrics.get('ttft_p90'))} | 0.000 |"
        ),
        (
            f"| `{candidate_label}` | {candidate_metrics['requests']} | "
            f"{fmt(candidate_metrics.get('audio_ttfp_p90'))} | "
            f"{fmt(candidate_metrics.get('continuity'))} | "
            f"{fmt(candidate_metrics.get('e2e_p90'))} | "
            f"{fmt(candidate_metrics.get('ttft_p90'))} | "
            f"{fmt(deltas(candidate_metrics, baseline_metrics).get('no_qps_score'))} |"
        ),
        "",
        "## Top Feasible Gates",
        "",
        "| rank | gate | candidate reqs | audio_p90 | audio gain | continuity gain | e2e gain | ttft gain | no_qps_score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    feasible = [row for row in rows if not str(row["gate"]).startswith("oracle:")]
    for rank, row in enumerate(feasible[:30], 1):
        lines.append(
            f"| {rank} | `{row['gate']}` | {row['candidate_requests']} | "
            f"{fmt(row.get('audio_ttfp_p90'))} | "
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
                "## Oracle Diagnostic Gates",
                "",
                "| rank | gate | candidate reqs | audio_p90 | audio gain | continuity gain | e2e gain | ttft gain | no_qps_score |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(oracle[:20], 1):
            lines.append(
                f"| {rank} | `{row['gate']}` | {row['candidate_requests']} | "
                f"{fmt(row.get('audio_ttfp_p90'))} | "
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
    parser.add_argument("--candidate", required=True, help="label=/path/to/run or per_request.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-oracle-audio", action="store_true")
    args = parser.parse_args()

    baseline_label, baseline_root = parse_labeled_path(args.baseline)
    candidate_label, candidate_root = parse_labeled_path(args.candidate)
    baseline_path = find_per_request(baseline_root)
    candidate_path = find_per_request(candidate_root)
    baseline_rows = read_requests(baseline_path)
    candidate_rows = read_requests(candidate_path)
    common_keys = sorted(set(baseline_rows) & set(candidate_rows))
    if not common_keys:
        raise SystemExit("no common requests")
    enrich_features(baseline_rows)
    enrich_features(candidate_rows)
    baseline_metrics = summarize([baseline_rows[key] for key in common_keys])
    candidate_metrics = summarize([candidate_rows[key] for key in common_keys])

    rows = [
        evaluate_gate(gate, baseline_rows, candidate_rows, baseline_metrics, common_keys)
        for gate in make_gates(args.include_oracle_audio)
    ]
    rows.sort(
        key=lambda row: (
            float(row.get("no_qps_score") or float("-inf")),
            float(row.get("audio_ttfp_reduction_pct") or float("-inf")),
            float(row.get("e2e_reduction_pct") or float("-inf")),
        ),
        reverse=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "admission_gate_search.json"
    csv_path = args.output_dir / "admission_gate_search.csv"
    md_path = args.output_dir / "admission_gate_search.md"
    payload = {
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "common_requests": len(common_keys),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_deltas": deltas(candidate_metrics, baseline_metrics),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = [
        "gate",
        "candidate_requests",
        "baseline_requests",
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
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    write_markdown(
        md_path,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        rows=rows,
    )
    print(json_path)
    print(csv_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
