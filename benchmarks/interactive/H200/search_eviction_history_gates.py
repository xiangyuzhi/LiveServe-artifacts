#!/usr/bin/env python3
"""Search online-feasible gates based on prior turns in the same session.

The current-request audio duration is an oracle feature, but prior generated
audio/timing from the same session is available in a real multi-turn service.
This diagnostic checks whether those history features are strong enough to
select when an eviction policy should run.
"""

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
        if row.get("status") not in (None, 200):
            continue
        rows[trace_key(row)] = row
    return rows


def finite(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def percentile(values: list[float], q: float) -> float | None:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return None
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
    continuity = [
        1.0 if row.get("audio_continuity_ok") else 0.0
        for row in rows
        if row.get("audio_continuity_evaluated") is True
    ]
    return {
        "requests": len(rows),
        "audio_ttfp_p90": percentile(audio, 0.90),
        "continuity": sum(continuity) / len(continuity) if continuity else None,
        "e2e_p90": percentile(e2e, 0.90),
        "ttft_p90": percentile(ttft, 0.90),
    }


def deltas(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
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
        "continuity_delta_pp": (
            (metrics.get("continuity") - baseline.get("continuity")) * 100.0
            if isinstance(metrics.get("continuity"), (int, float))
            and isinstance(baseline.get("continuity"), (int, float))
            else None
        ),
    }
    out["no_qps_score"] = (
        WEIGHTS["audio_ttfp"] * float(out["audio_ttfp_reduction_pct"] or 0.0)
        + WEIGHTS["continuity"] * float(out["continuity_delta_pp"] or 0.0)
        + WEIGHTS["e2e"] * float(out["e2e_reduction_pct"] or 0.0)
        + WEIGHTS["ttft"] * float(out["ttft_reduction_pct"] or 0.0)
    )
    return out


def add_history_features(rows: dict[tuple[Any, ...], dict[str, Any]]) -> None:
    by_user: dict[str, list[dict[str, Any]]] = {}
    for row in rows.values():
        by_user.setdefault(str(row.get("user_id")), []).append(row)
    for user_rows in by_user.values():
        user_rows.sort(
            key=lambda row: (
                finite(row.get("timestamp"), 0.0) or 0.0,
                finite(row.get("round_index"), 0.0) or 0.0,
            )
        )
        prev_audio: list[float] = []
        prev_ttfp: list[float] = []
        prev_e2e: list[float] = []
        prev_cont: list[float] = []
        for row in user_rows:
            row["_hist_count"] = len(prev_audio)
            row["_hist_prev_audio_duration"] = prev_audio[-1] if prev_audio else 0.0
            row["_hist_max_audio_duration"] = max(prev_audio) if prev_audio else 0.0
            row["_hist_mean_audio_duration"] = (
                sum(prev_audio) / len(prev_audio) if prev_audio else 0.0
            )
            row["_hist_prev_audio_ttfp"] = prev_ttfp[-1] if prev_ttfp else 0.0
            row["_hist_max_audio_ttfp"] = max(prev_ttfp) if prev_ttfp else 0.0
            row["_hist_mean_audio_ttfp"] = (
                sum(prev_ttfp) / len(prev_ttfp) if prev_ttfp else 0.0
            )
            row["_hist_prev_e2e"] = prev_e2e[-1] if prev_e2e else 0.0
            row["_hist_max_e2e"] = max(prev_e2e) if prev_e2e else 0.0
            row["_hist_mean_e2e"] = sum(prev_e2e) / len(prev_e2e) if prev_e2e else 0.0
            row["_hist_continuity_rate"] = (
                sum(prev_cont) / len(prev_cont) if prev_cont else 1.0
            )

            audio = finite(row.get("audio_duration_sec"))
            if audio is not None and audio > 0:
                prev_audio.append(audio)
            audio_ttfp = finite(row.get("audio_ttfp_sec"))
            if audio_ttfp is not None:
                prev_ttfp.append(audio_ttfp)
            e2e = finite(row.get("e2e_sec"))
            if e2e is not None:
                prev_e2e.append(e2e)
            if row.get("audio_continuity_evaluated") is True:
                prev_cont.append(1.0 if row.get("audio_continuity_ok") else 0.0)


def enrich_static_features(rows: dict[tuple[Any, ...], dict[str, Any]]) -> None:
    for row in rows.values():
        prefix = finite(row.get("kv_prefix_token_estimate"), 0.0) or 0.0
        input_tokens = finite(row.get("input_tokens_total"), 0.0) or 0.0
        if input_tokens <= 0:
            input_tokens = finite(row.get("usage_prompt_tokens"), 0.0) or 0.0
        row["_prefix_tokens"] = prefix
        row["_input_tokens"] = input_tokens
        row["_round_index"] = finite(row.get("round_index"), 0.0) or 0.0
        row["_response_len"] = finite(row.get("response_length"), 0.0) or 0.0
        row["_query_len"] = finite(row.get("query_length"), 0.0) or 0.0
        row["_timestamp"] = finite(row.get("timestamp"), 0.0) or 0.0
        row["_audio_duration"] = finite(row.get("audio_duration_sec"), 0.0) or 0.0


def single_feature_gates(feature: str, cuts: list[float], *, oracle: bool = False) -> list[Gate]:
    prefix = "oracle:" if oracle else ""
    gates: list[Gate] = []
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
    return gates


def make_gates(include_oracle: bool) -> list[Gate]:
    gates = [Gate("all_candidate", lambda row: True)]
    feature_cuts = {
        "_hist_count": [1, 2, 3, 5, 10, 20, 30],
        "_hist_prev_audio_duration": [0.5, 1, 2, 4, 8, 16, 32],
        "_hist_max_audio_duration": [1, 2, 4, 8, 16, 32, 64],
        "_hist_mean_audio_duration": [1, 2, 4, 8, 16, 32],
        "_hist_prev_audio_ttfp": [1, 2, 3, 4, 5],
        "_hist_max_audio_ttfp": [2, 3, 4, 5],
        "_hist_mean_audio_ttfp": [1, 2, 3, 4],
        "_hist_prev_e2e": [2, 4, 8, 16, 32, 64],
        "_hist_max_e2e": [4, 8, 16, 32, 64, 128],
        "_hist_mean_e2e": [2, 4, 8, 16, 32, 64],
        "_hist_continuity_rate": [0.25, 0.5, 0.75, 0.9],
        "_round_index": [1, 3, 10, 20, 30, 45],
        "_prefix_tokens": [1, 64, 128, 192, 256, 320],
        "_response_len": [8, 16, 24, 32, 48, 64, 96, 128],
    }
    for feature, cuts in feature_cuts.items():
        gates.extend(single_feature_gates(feature, cuts))

    for hist_cut in [1, 2, 4, 8, 16]:
        for prefix_cut in [1, 64, 128, 192]:
            gates.append(
                Gate(
                    f"_hist_max_audio_duration>={hist_cut:g}&_prefix_tokens>={prefix_cut:g}",
                    lambda row, h=hist_cut, p=prefix_cut: (
                        float(row.get("_hist_max_audio_duration", 0.0)) >= h
                        and float(row.get("_prefix_tokens", 0.0)) >= p
                    ),
                )
            )
        for round_cut in [3, 10, 20, 30]:
            gates.append(
                Gate(
                    f"_hist_max_audio_duration>={hist_cut:g}&_round_index>={round_cut:g}",
                    lambda row, h=hist_cut, r=round_cut: (
                        float(row.get("_hist_max_audio_duration", 0.0)) >= h
                        and float(row.get("_round_index", 0.0)) >= r
                    ),
                )
            )
    for prev_cut in [1, 2, 4, 8, 16]:
        for resp_cut in [16, 32, 48, 64]:
            gates.append(
                Gate(
                    f"_hist_prev_audio_duration>={prev_cut:g}&_response_len>={resp_cut:g}",
                    lambda row, h=prev_cut, r=resp_cut: (
                        float(row.get("_hist_prev_audio_duration", 0.0)) >= h
                        and float(row.get("_response_len", 0.0)) >= r
                    ),
                )
            )

    if include_oracle:
        gates.extend(
            single_feature_gates(
                "_audio_duration",
                [0.5, 1, 2, 4, 8, 16, 32, 64],
                oracle=True,
            )
        )
    return gates


def splice_rows(
    keys: list[tuple[Any, ...]],
    baseline: dict[tuple[Any, ...], dict[str, Any]],
    candidate: dict[tuple[Any, ...], dict[str, Any]],
    gate: Gate,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    selected = 0
    for key in keys:
        base = baseline[key]
        if gate.func(base):
            rows.append(candidate[key])
            selected += 1
        else:
            rows.append(base)
    return rows, selected


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(
    path: Path,
    *,
    config: dict[str, Any],
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Eviction History Gate Search",
        "",
        "History features are computed from prior turns of the same session in the baseline run.",
        "",
        f"- baseline: {config['baseline_label']}",
        f"- candidate: {config['candidate_label']}",
        f"- common requests: {config['common_requests']}",
        "",
        "## Plain Candidate",
        "",
        "| source | requests | audio p90 | continuity | e2e p90 | ttft p90 | score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in (
        (config["baseline_label"], baseline_metrics),
        (config["candidate_label"], candidate_metrics),
    ):
        delta = deltas(metrics, baseline_metrics)
        lines.append(
            f"| {label} | {metrics['requests']} | {fmt(metrics['audio_ttfp_p90'])} | "
            f"{fmt(metrics['continuity'])} | {fmt(metrics['e2e_p90'])} | "
            f"{fmt(metrics['ttft_p90'])} | {fmt(delta['no_qps_score'])} |"
        )
    lines.extend(
        [
            "",
            "## Top Gates",
            "",
            "| rank | gate | candidate reqs | audio p90 | audio gain | continuity gain | e2e gain | ttft gain | score |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(rows[:50], 1):
        lines.append(
            f"| {idx} | `{row['gate']}` | {row['candidate_requests']} | "
            f"{fmt(row['audio_ttfp_p90'])} | {fmt(row['audio_ttfp_reduction_pct'])}% | "
            f"{fmt(row['continuity_delta_pp'])}pp | {fmt(row['e2e_reduction_pct'])}% | "
            f"{fmt(row['ttft_reduction_pct'])}% | {fmt(row['no_qps_score'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="label=/path/to/run or per_request.jsonl")
    parser.add_argument("--candidate", required=True, help="label=/path/to/run or per_request.jsonl")
    parser.add_argument("--include-oracle", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_label, baseline_path_raw = parse_labeled_path(args.baseline)
    candidate_label, candidate_path_raw = parse_labeled_path(args.candidate)
    baseline_path = find_per_request(baseline_path_raw)
    candidate_path = find_per_request(candidate_path_raw)
    baseline = read_requests(baseline_path)
    candidate = read_requests(candidate_path)
    common_keys = sorted(
        set(baseline) & set(candidate),
        key=lambda key: (
            finite(key[2], 0.0) or 0.0,
            str(key[0]),
            finite(key[1], 0.0) or 0.0,
        ),
    )
    if not common_keys:
        raise SystemExit("no common requests")

    add_history_features(baseline)
    enrich_static_features(baseline)
    add_history_features(candidate)
    enrich_static_features(candidate)

    baseline_rows = [baseline[key] for key in common_keys]
    candidate_rows = [candidate[key] for key in common_keys]
    baseline_metrics = summarize(baseline_rows)
    candidate_metrics = summarize(candidate_rows)

    out_rows: list[dict[str, Any]] = []
    for gate in make_gates(args.include_oracle):
        rows, selected = splice_rows(common_keys, baseline, candidate, gate)
        metrics = summarize(rows)
        delta = deltas(metrics, baseline_metrics)
        out_rows.append(
            {
                "gate": gate.label,
                "candidate_requests": selected,
                **metrics,
                **delta,
            }
        )
    out_rows.sort(
        key=lambda row: (
            float(row.get("no_qps_score") or -1e9),
            float(row.get("audio_ttfp_reduction_pct") or -1e9),
            float(row.get("e2e_reduction_pct") or -1e9),
        ),
        reverse=True,
    )

    config = {
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "common_requests": len(common_keys),
        "include_oracle": args.include_oracle,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "history_gate_search.json"
    csv_path = args.output_dir / "history_gate_search.csv"
    md_path = args.output_dir / "history_gate_search.md"
    json_path.write_text(
        json.dumps(
            {
                "config": config,
                "baseline_metrics": baseline_metrics,
                "candidate_metrics": candidate_metrics,
                "gates": out_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    fields = [
        "gate",
        "candidate_requests",
        "requests",
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
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    write_markdown(
        md_path,
        config=config,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        rows=out_rows,
    )
    print(json_path)
    print(csv_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
