#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_eval(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "lanes" not in data or not isinstance(data["lanes"], list):
        raise ValueError(f"{path} does not look like adaptive_eviction_validation.json")
    return data


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.parent.name, path
    name, path = raw.split("=", 1)
    return name.strip(), Path(path)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Eviction Candidate Matrix",
        "",
        f"- decision: `{'PASS' if result['passed'] else 'REJECT'}`",
        f"- average score: `{result['average_score_pct']:.3f}%`",
        f"- pass rate: `{result['pass_rate']:.3f}` ({result['passed_settings']}/{result['settings']})",
        f"- thresholds: average `>= {result['min_average_score_pct']:.3f}%`, pass-rate `>= {result['min_pass_rate']:.3f}`",
        "",
        "| setting | lane | score | passed | reason | source |",
        "|---|---|---:|---|---|---|",
    ]
    for row in result["rows"]:
        failures = "; ".join(row.get("failures") or [])
        lines.append(
            f"| {row['setting']} | {row['lane']} | {row['score_pct']:.3f}% | "
            f"{'yes' if row['passed'] else 'no'} | {failures} | `{row['source']}` |"
        )
    if result.get("failures"):
        lines.extend(["", "## Failures", ""])
        for failure in result["failures"]:
            lines.append(f"- {failure}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate several adaptive eviction validation outputs using a "
            "majority/average criterion."
        )
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate eval JSON, optionally named as label=/path/to/adaptive_eviction_validation.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-average-score", type=float, default=30.0)
    parser.add_argument("--min-pass-rate", type=float, default=0.51)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for raw in args.candidate:
        name, path = parse_named_path(raw)
        data = load_eval(path)
        for lane in data["lanes"]:
            deltas = lane.get("deltas") or {}
            score = deltas.get("three_metric_score_pct")
            if score is None:
                raise ValueError(f"{path} lane {lane.get('label')} has no score")
            rows.append(
                {
                    "setting": name,
                    "lane": lane.get("label", ""),
                    "score_pct": float(score),
                    "passed": bool(lane.get("passed")),
                    "failures": lane.get("failures") or [],
                    "source": str(path),
                    "run_root": lane.get("run_root"),
                    "baseline": lane.get("baseline"),
                    "candidate": lane.get("candidate"),
                }
            )

    if not rows:
        raise ValueError("no lanes found")
    settings = len(rows)
    passed_settings = sum(1 for row in rows if row["passed"])
    average_score = sum(row["score_pct"] for row in rows) / settings
    pass_rate = passed_settings / settings
    failures: list[str] = []
    if average_score < args.min_average_score:
        failures.append(
            f"average score {average_score:.3f}% < {args.min_average_score:.3f}%"
        )
    if pass_rate < args.min_pass_rate:
        failures.append(
            f"pass rate {pass_rate:.3f} < {args.min_pass_rate:.3f}"
        )

    result = {
        "passed": not failures,
        "settings": settings,
        "passed_settings": passed_settings,
        "pass_rate": pass_rate,
        "average_score_pct": average_score,
        "min_average_score_pct": args.min_average_score,
        "min_pass_rate": args.min_pass_rate,
        "failures": failures,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "eviction_candidate_matrix.json"
    md_path = args.output_dir / "eviction_candidate_matrix.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, result)
    print(json_path)
    print(md_path)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
