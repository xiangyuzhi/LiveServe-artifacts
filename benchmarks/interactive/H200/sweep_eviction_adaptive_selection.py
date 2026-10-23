#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import select_eviction_adaptive_policy as selector


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    score_epsilon: float
    max_replacement_count_p90: float
    min_eviction_reduction_pct: float
    min_goodput_gain_pct: float = -1.0
    min_audio_p90_reduction_pct: float = 0.0
    min_e2e_p90_reduction_pct: float = 0.0
    min_continuity_delta_pp: float = -0.5


SCENARIOS = (
    Scenario(
        name="stable_default",
        description="cap1-stable selector used for the current adaptive default",
        score_epsilon=1.0,
        max_replacement_count_p90=1.0,
        min_eviction_reduction_pct=-5.0,
    ),
    Scenario(
        name="score_only_cap1",
        description="highest score among cap1 candidates, with weak churn constraints",
        score_epsilon=0.0,
        max_replacement_count_p90=1.0,
        min_eviction_reduction_pct=-100.0,
        min_goodput_gain_pct=-5.0,
        min_continuity_delta_pp=-5.0,
    ),
    Scenario(
        name="strict_churn",
        description="requires LRU eviction reduction while keeping cap1 stability",
        score_epsilon=1.0,
        max_replacement_count_p90=1.0,
        min_eviction_reduction_pct=0.0,
    ),
    Scenario(
        name="allow_multi_repl",
        description="allows p90 replacement count 2 if other metrics pass",
        score_epsilon=1.0,
        max_replacement_count_p90=2.0,
        min_eviction_reduction_pct=-10.0,
    ),
    Scenario(
        name="latency_only_allow_multi",
        description="mostly latency-ranked; useful for seeing tempting but risky policies",
        score_epsilon=0.0,
        max_replacement_count_p90=99.0,
        min_eviction_reduction_pct=-100.0,
        min_goodput_gain_pct=-5.0,
        min_continuity_delta_pp=-5.0,
    ),
)


def args_for(scenario: Scenario, output_dir: Path, exp_root: Path) -> argparse.Namespace:
    return SimpleNamespace(
        exp_root=exp_root,
        output_dir=output_dir,
        candidate=None,
        score_epsilon=scenario.score_epsilon,
        min_goodput_gain_pct=scenario.min_goodput_gain_pct,
        min_audio_p90_reduction_pct=scenario.min_audio_p90_reduction_pct,
        min_e2e_p90_reduction_pct=scenario.min_e2e_p90_reduction_pct,
        min_continuity_delta_pp=scenario.min_continuity_delta_pp,
        max_replacement_count_p90=scenario.max_replacement_count_p90,
        min_eviction_reduction_pct=scenario.min_eviction_reduction_pct,
    )


def evaluate_scenario(
    scenario: Scenario,
    *,
    exp_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    run_args = args_for(scenario, output_dir, exp_root)
    lanes = {lane.name: lane for lane in selector.DEFAULT_LANES}
    rows = [
        selector.evaluate_candidate(lanes[candidate.lane], candidate, exp_root=exp_root, args=run_args)
        for candidate in selector.DEFAULT_CANDIDATES
        if candidate.lane in lanes
    ]
    selected = selector.select_lanes(rows, scenario.score_epsilon)
    segments = selector.adaptive_segments(selected)
    return {
        "name": scenario.name,
        "description": scenario.description,
        "thresholds": {
            "score_epsilon": scenario.score_epsilon,
            "max_replacement_count_p90": scenario.max_replacement_count_p90,
            "min_eviction_reduction_pct": scenario.min_eviction_reduction_pct,
            "min_goodput_gain_pct": scenario.min_goodput_gain_pct,
            "min_audio_p90_reduction_pct": scenario.min_audio_p90_reduction_pct,
            "min_e2e_p90_reduction_pct": scenario.min_e2e_p90_reduction_pct,
            "min_continuity_delta_pp": scenario.min_continuity_delta_pp,
        },
        "selected": selected,
        "segments": segments,
        "averages": {
            "three_metric_score_pct": selector.mean(
                [row["deltas"].get("three_metric_score_pct") for row in selected]
            ),
            "goodput_gain_pct": selector.mean(
                [row["deltas"].get("goodput_gain_pct") for row in selected]
            ),
            "audio_p90_reduction_pct": selector.mean(
                [row["deltas"].get("audio_p90_reduction_pct") for row in selected]
            ),
            "e2e_p90_reduction_pct": selector.mean(
                [row["deltas"].get("e2e_p90_reduction_pct") for row in selected]
            ),
        },
        "candidate_failures": [
            {
                "lane": row["lane"],
                "label": row["label"],
                "policy": row["policy"],
                "score": row["deltas"].get("three_metric_score_pct"),
                "failures": row["failures"],
            }
            for row in rows
            if row["failures"]
        ],
    }


def segment_signature(segments: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for segment in segments:
        max_blocks = segment.get("recommended_max_blocks")
        scope = "rest" if max_blocks is None else f"<={max_blocks}"
        parts.append(f"{scope}:{segment.get('policy')}")
    return "; ".join(parts)


def fmt(value: Any) -> str:
    return selector.fmt(value)


def write_markdown(path: Path, scenarios: list[dict[str, Any]]) -> None:
    lines = [
        "# Adaptive Selection Sensitivity",
        "",
        "This sweep reuses clean historical candidate runs and changes only the promotion/selection constraints.",
        "",
        "## Scenario Summary",
        "",
        "| scenario | avg score | avg goodput | avg audio p90 | avg e2e p90 | branch signature |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for scenario in scenarios:
        avg = scenario["averages"]
        lines.append(
            f"| `{scenario['name']}` | {fmt(avg['three_metric_score_pct'])}% | "
            f"{fmt(avg['goodput_gain_pct'])}% | {fmt(avg['audio_p90_reduction_pct'])}% | "
            f"{fmt(avg['e2e_p90_reduction_pct'])}% | "
            f"`{segment_signature(scenario['segments'])}` |"
        )
    lines.extend(["", "## Selected Lanes", ""])
    for scenario in scenarios:
        lines.extend(
            [
                f"### {scenario['name']}",
                "",
                scenario["description"],
                "",
                "| lane | blocks | label | policy | score | goodput | audio p90 | e2e p90 | recent miss |",
                "|---|---:|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in scenario["selected"]:
            d = row["deltas"]
            lines.append(
                f"| {row['lane']} | {row['capacity_blocks']} | `{row['label']}` | "
                f"`{row['policy']}` | {fmt(d.get('three_metric_score_pct'))}% | "
                f"{fmt(d.get('goodput_gain_pct'))}% | "
                f"{fmt(d.get('audio_p90_reduction_pct'))}% | "
                f"{fmt(d.get('e2e_p90_reduction_pct'))}% | "
                f"{fmt(d.get('recent_miss_reduction_pct'))}% |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- `stable_default` is the recommended selector for promotion because it avoids multi-replacement candidates and treats near-tied latency scores as a cache-churn tie-break.",
            "- `score_only_cap1` shows the latency-only temptation: tiny flips to cost-response and nano flips to bucket-response-cost, but both changes are small enough that churn/stability should decide.",
            "- `allow_multi_repl` and `latency_only_allow_multi` expose why partialmax2 is not promoted by default: it can improve nano latency, but it raises replacement multiplicity and can increase LRU churn.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", type=Path, default=selector.EXP_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        evaluate_scenario(scenario, exp_root=args.exp_root, output_dir=args.output_dir)
        for scenario in SCENARIOS
    ]
    json_path = args.output_dir / "adaptive_selection_sensitivity.json"
    md_path = args.output_dir / "adaptive_selection_sensitivity.md"
    json_path.write_text(json.dumps({"scenarios": rows}, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, rows)
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
