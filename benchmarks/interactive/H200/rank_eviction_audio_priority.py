#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_WEIGHTS = {
    "audio_ttfp": 0.35,
    "qps": 0.25,
    "continuity": 0.20,
    "e2e": 0.15,
    "ttft": 0.05,
}


def parse_mapping(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"expected label=path, got {raw!r}")
    label, value = raw.split("=", 1)
    label = label.strip()
    value = value.strip()
    if not label or not value:
        raise argparse.ArgumentTypeError(f"expected label=path, got {raw!r}")
    return label, Path(value)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_summary(path: Path) -> Path:
    if path.is_file() and path.name == "summary.json":
        return path
    matches = sorted(path.glob("*/*/summary.json"))
    if not matches:
        matches = sorted(path.glob("**/summary.json"))
    if not matches:
        raise FileNotFoundError(f"no summary.json found under {path}")
    return matches[0]


def metrics_from_summary(path: Path) -> dict[str, Any]:
    summary = read_json(path)
    totals = summary.get("totals", {})
    latency = summary.get("latency", {})
    continuity = summary.get("audio", {}).get("streaming_continuity", {})
    lru = summary.get("kv_lru", {})
    return {
        "summary_path": str(path),
        "requests": totals.get("completed_ok"),
        "errors": totals.get("errors"),
        "audio_ttfp_p90": latency.get("audio_ttfp_sec", {}).get("p90"),
        "qps": totals.get("useful_qps") or totals.get("throughput_rps"),
        "continuity": continuity.get("ok_rate"),
        "e2e_p90": latency.get("e2e_sec", {}).get("p90"),
        "ttft_p90": latency.get("ttfp_sec", {}).get("p90"),
        "goodput": continuity.get("goodput_rps"),
        "lru_evictions": lru.get("gpu_evictions"),
        "recent_miss": lru.get("recent_eviction_misses"),
    }


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def pct_reduction(candidate: Any, baseline: Any) -> float | None:
    if not finite(candidate) or not finite(baseline) or baseline == 0:
        return None
    return 100.0 * (float(baseline) - float(candidate)) / float(baseline)


def pct_gain(candidate: Any, baseline: Any) -> float | None:
    if not finite(candidate) or not finite(baseline) or baseline == 0:
        return None
    return 100.0 * (float(candidate) - float(baseline)) / float(baseline)


def delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "audio_ttfp_reduction_pct": pct_reduction(
            candidate.get("audio_ttfp_p90"),
            baseline.get("audio_ttfp_p90"),
        ),
        "qps_gain_pct": pct_gain(candidate.get("qps"), baseline.get("qps")),
        "continuity_delta_pp": (
            100.0 * (float(candidate["continuity"]) - float(baseline["continuity"]))
            if finite(candidate.get("continuity")) and finite(baseline.get("continuity"))
            else None
        ),
        "e2e_reduction_pct": pct_reduction(candidate.get("e2e_p90"), baseline.get("e2e_p90")),
        "ttft_reduction_pct": pct_reduction(candidate.get("ttft_p90"), baseline.get("ttft_p90")),
        "goodput_gain_pct": pct_gain(candidate.get("goodput"), baseline.get("goodput")),
    }


def weighted_score(deltas: dict[str, Any], weights: dict[str, float]) -> float | None:
    parts = {
        "audio_ttfp": deltas.get("audio_ttfp_reduction_pct"),
        "qps": deltas.get("qps_gain_pct"),
        "continuity": deltas.get("continuity_delta_pp"),
        "e2e": deltas.get("e2e_reduction_pct"),
        "ttft": deltas.get("ttft_reduction_pct"),
    }
    if any(not finite(value) for value in parts.values()):
        return None
    return sum(float(parts[name]) * float(weights[name]) for name in parts)


def rows_from_validation(paths: list[Path]) -> list[tuple[str, Path, Path]]:
    rows: list[tuple[str, Path, Path]] = []
    for path in paths:
        data = read_json(path)
        for lane in data.get("lanes", []):
            label = str(lane.get("label") or path.parent.name)
            candidate = lane.get("candidate", {}).get("summary_path")
            baseline = lane.get("baseline", {}).get("summary_path")
            if candidate and baseline:
                rows.append((label, Path(candidate), Path(baseline)))
    return rows


def format_float(value: Any, digits: int = 3) -> str:
    if not finite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_outputs(output_dir: Path, baseline: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"baseline": baseline, "rows": rows}
    (output_dir / "audio_priority_ranking.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fields = [
        "rank",
        "label",
        "score",
        "audio_ttfp_p90",
        "audio_ttfp_reduction_pct",
        "qps",
        "qps_gain_pct",
        "continuity",
        "continuity_delta_pp",
        "e2e_p90",
        "e2e_reduction_pct",
        "ttft_p90",
        "ttft_reduction_pct",
        "goodput",
        "goodput_gain_pct",
        "lru_evictions",
        "recent_miss",
        "requests",
        "errors",
        "summary_path",
    ]
    with (output_dir / "audio_priority_ranking.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(rows, 1):
            out = {field: row.get(field) for field in fields}
            out["rank"] = idx
            writer.writerow(out)

    lines = [
        "# Eviction Audio-Priority Ranking",
        "",
        "Primary TTFP is `latency.audio_ttfp_sec.p90`. "
        "`latency.ttfp_sec.p90` is reported as TTFT.",
        "",
        "| rank | candidate | score | audio TTFP | QPS | continuity | E2E | TTFT | goodput |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows, 1):
        lines.append(
            "| {rank} | `{label}` | {score} | {audio}s / {audio_d:+.2f}% | "
            "{qps} / {qps_d:+.2f}% | {cont} / {cont_d:+.2f}pp | "
            "{e2e}s / {e2e_d:+.2f}% | {ttft}s / {ttft_d:+.2f}% | "
            "{goodput} / {goodput_d:+.2f}% |".format(
                rank=idx,
                label=row["label"],
                score=format_float(row.get("score"), 2),
                audio=format_float(row.get("audio_ttfp_p90"), 3),
                audio_d=float(row.get("audio_ttfp_reduction_pct") or 0.0),
                qps=format_float(row.get("qps"), 4),
                qps_d=float(row.get("qps_gain_pct") or 0.0),
                cont=format_float(row.get("continuity"), 4),
                cont_d=float(row.get("continuity_delta_pp") or 0.0),
                e2e=format_float(row.get("e2e_p90"), 3),
                e2e_d=float(row.get("e2e_reduction_pct") or 0.0),
                ttft=format_float(row.get("ttft_p90"), 3),
                ttft_d=float(row.get("ttft_reduction_pct") or 0.0),
                goodput=format_float(row.get("goodput"), 4),
                goodput_d=float(row.get("goodput_gain_pct") or 0.0),
            )
        )
    (output_dir / "audio_priority_ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_weights(raw: str | None) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if not raw:
        return weights
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"expected metric=weight in {part!r}")
        key, value = part.split("=", 1)
        key = key.strip()
        if key not in weights:
            raise argparse.ArgumentTypeError(f"unknown weight key {key!r}")
        weights[key] = float(value)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--candidate-summary", action="append", type=parse_mapping, default=[])
    parser.add_argument("--validation", action="append", type=Path, default=[])
    parser.add_argument("--weights", help="comma-separated metric=weight overrides")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    weights = parse_weights(args.weights)
    candidates: list[tuple[str, Path, Path | None]] = []
    candidates.extend((label, path, None) for label, path in args.candidate_summary)
    candidates.extend(rows_from_validation(args.validation))

    if not candidates:
        raise SystemExit("no candidates provided")
    if args.baseline_summary is None and any(baseline is None for _, _, baseline in candidates):
        raise SystemExit("--baseline-summary is required for --candidate-summary inputs")

    baseline_path = find_summary(args.baseline_summary) if args.baseline_summary else candidates[0][2]
    assert baseline_path is not None
    baseline_metrics = metrics_from_summary(baseline_path)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for label, candidate_path, candidate_baseline_path in candidates:
        resolved_baseline_path = candidate_baseline_path or baseline_path
        assert resolved_baseline_path is not None
        summary_path = find_summary(candidate_path)
        key = (str(summary_path), str(resolved_baseline_path))
        if key in seen:
            continue
        seen.add(key)
        local_baseline = metrics_from_summary(find_summary(resolved_baseline_path))
        metrics = metrics_from_summary(summary_path)
        deltas = delta(metrics, local_baseline)
        rows.append(
            {
                "label": label,
                "score": weighted_score(deltas, weights),
                "baseline_summary_path": str(resolved_baseline_path),
                **metrics,
                **deltas,
            }
        )

    rows.sort(key=lambda row: float(row["score"]) if finite(row.get("score")) else float("-inf"), reverse=True)
    write_outputs(args.output_dir, baseline_metrics, rows)
    print(output_dir := args.output_dir)
    print(output_dir / "audio_priority_ranking.md")


if __name__ == "__main__":
    main()
