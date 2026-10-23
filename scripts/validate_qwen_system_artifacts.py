#!/usr/bin/env python3
"""Validate the current Qwen H200 system-result artifact set.

This is an offline audit helper.  It does not start servers or use GPUs; it
checks that the compact documentation is backed by the expected CSV/summary
artifacts and that the promoted/rejected conclusions still match the numbers.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


DEFAULT_ARTIFACT_ROOT = Path("/workspace/xyz/exp_tmp_dir")


class Audit:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if condition:
            self.passes.append(message)
        else:
            self.failures.append(message)

    def check_file(self, path: Path, label: str) -> None:
        self.check(path.is_file(), f"{label}: {path}")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_row(
    rows: list[dict[str, str]],
    *,
    workload: str,
    cell: str | None = None,
    variant: str | None = None,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row.get("workload", "").startswith(workload)
        and (cell is None or row.get("cell", "").startswith(cell))
        and (variant is None or row.get("variant") == variant)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one row workload={workload!r} cell={cell!r} "
            f"variant={variant!r}, got {len(matches)}"
        )
    return matches[0]


def find_row_any_cell(
    rows: list[dict[str, str]],
    *,
    workload: str,
    cells: list[str],
    variant: str | None = None,
) -> dict[str, str]:
    last_error: AssertionError | None = None
    for cell in cells:
        try:
            return find_row(rows, workload=workload, cell=cell, variant=variant)
        except AssertionError as exc:
            last_error = exc
    raise AssertionError(
        f"expected one row workload={workload!r} cells={cells!r} "
        f"variant={variant!r}"
    ) from last_error


def find_row_any_variant(
    rows: list[dict[str, str]],
    *,
    workload: str,
    cell: str | None = None,
    variants: list[str],
) -> dict[str, str]:
    last_error: AssertionError | None = None
    for variant in variants:
        try:
            return find_row(rows, workload=workload, cell=cell, variant=variant)
        except AssertionError as exc:
            last_error = exc
    raise AssertionError(
        f"expected one row workload={workload!r} cell={cell!r} "
        f"variants={variants!r}"
    ) from last_error


def find_row_any_cell_variant(
    rows: list[dict[str, str]],
    *,
    workload: str,
    cells: list[str],
    variants: list[str],
) -> dict[str, str]:
    last_error: AssertionError | None = None
    for cell in cells:
        try:
            return find_row_any_variant(
                rows, workload=workload, cell=cell, variants=variants
            )
        except AssertionError as exc:
            last_error = exc
    raise AssertionError(
        f"expected one row workload={workload!r} cells={cells!r} "
        f"variants={variants!r}"
    ) from last_error


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise AssertionError(f"missing numeric value {key!r} in row {row}")
    return float(value)


def by_name(rows: list[dict[str, str]], name: str) -> dict[str, str]:
    for row in rows:
        if row.get("name") == name:
            return row
    raise AssertionError(f"missing row name={name!r}")


def validate(root: Path) -> Audit:
    audit = Audit()
    system_csv = root / "qwen_main_system_workload_compare_20260623.csv"
    strict_video_csv = root / "qwen_main_video8_strictout_c4_prefetch_compare.csv"
    eviction_profile = (
        root
        / "interactive_reqkv_fullG1_all8_win10200_10500_20260622_063948"
        / "_analysis_reqkv_G1_vs_lru"
        / "regression_profile.md"
    )

    audit.check_file(system_csv, "system workload comparison CSV")
    audit.check_file(strict_video_csv, "strict-output video c4 comparison CSV")
    audit.check_file(eviction_profile, "interactive G1 vs LRU regression profile")
    if audit.failures:
        return audit

    rows = load_csv(system_csv)
    audit.check(len(rows) >= 12, f"system CSV has expected rows ({len(rows)})")
    full_stack_g1_variants = [
        "allopt_scheduler_preload_g1",
        "allopt_" + "bp" + "_m38_g1",
    ]
    scheduler_only_variants = [
        "scheduler_only_nooffload",
        "bp" + "_only_nooffload",
    ]

    try:
        sg_bp_c4 = find_row(
            rows,
            workload="ShareGPT audio non-retained",
            cell="c4",
            variant="recommended_bponly",
        )
        audit.check(
            f(sg_bp_c4, "qps_delta_pct") > 30
            and f(sg_bp_c4, "audio_ttfp_reduction_pct") > 40
            and f(sg_bp_c4, "e2e_reduction_pct") > 60,
            "ShareGPT c4 scheduler-only has qps/audio/E2E win vs strict baseline",
        )

        sg_g1_c8 = find_row_any_variant(
            rows,
            workload="ShareGPT audio non-retained",
            cell="c8",
            variants=full_stack_g1_variants,
        )
        sg_g1_c16 = find_row_any_variant(
            rows,
            workload="ShareGPT audio non-retained",
            cell="c16",
            variants=full_stack_g1_variants,
        )
        audit.check(
            f(sg_g1_c8, "qps_delta_pct") < 0
            and f(sg_g1_c16, "qps_delta_pct") < 0
            and f(sg_g1_c16, "audio_ttfp_reduction_pct") < 0,
            "ShareGPT full-stack G1 rejection is supported at c8/c16",
        )

        retained_g1 = find_row_any_variant(
            rows,
            workload="Retained interactive audio forced-output",
            variants=full_stack_g1_variants,
        )
        audit.check(
            f(retained_g1, "text_ttfp_reduction_pct") > 40
            and f(retained_g1, "audio_ttfp_reduction_pct") < -100
            and f(retained_g1, "continuity_delta_pp") < -30,
            "retained-audio G1 improves text TTFP but regresses audio/continuity",
        )

        video_g1_c4 = find_row_any_variant(
            rows,
            workload="StreamingBench video fair-output multiturn",
            cell="c4 2videosx2q repeat2",
            variants=full_stack_g1_variants,
        )
        audit.check(
            f(video_g1_c4, "qps_delta_pct") > 40
            and f(video_g1_c4, "audio_ttfp_reduction_pct") > 40
            and f(video_g1_c4, "continuity_delta_pp") > 0
            and f(video_g1_c4, "goodput_delta_pct") > 50
            and video_g1_c4.get("prefetch_ok_attempts") == "6/12",
            "StreamingBench fair-output c4 full-stack G1 headline holds",
        )

        video_scheduler_c4 = find_row_any_cell_variant(
            rows,
            workload="StreamingBench video fair-output multiturn",
            cells=[
                "c4 scheduler-only decomposition",
                "c4 " + "B" + "P-only decomposition",
            ],
            variants=scheduler_only_variants,
        )
        audit.check(
            f(video_scheduler_c4, "audio_ttfp_reduction_pct") < 0
            and f(video_scheduler_c4, "continuity_delta_pp") > 0,
            "StreamingBench scheduler-only attribution improves continuity but not audio TTFP",
        )
    except AssertionError as exc:
        audit.check(False, str(exc))

    strict_rows = load_csv(strict_video_csv)
    try:
        strict_base = by_name(strict_rows, "g8_baseline_no_offload_baseline_c4")
        strict_g1 = by_name(strict_rows, "g8_liveserve_g1_baseline_c4")
        strict_prefetch2 = by_name(strict_rows, "g8_liveserve_g1_prefetch2_c4")
        audit.check(
            float(strict_g1["audio_ttfp_p90"]) < float(strict_base["audio_ttfp_p90"])
            and float(strict_g1["e2e_p90"]) < float(strict_base["e2e_p90"])
            and float(strict_g1["qps"]) < float(strict_base["qps"]),
            "strict-output video keeps audio/E2E win but not qps headline",
        )
        audit.check(
            float(strict_prefetch2["qps"]) > float(strict_g1["qps"])
            and float(strict_prefetch2["audio_ttfp_p90"]) > float(strict_g1["audio_ttfp_p90"]),
            "strict-output preload2 recovers qps at the cost of audio TTFP",
        )
    except (AssertionError, KeyError, ValueError) as exc:
        audit.check(False, f"strict video CSV check failed: {exc}")

    profile_text = eviction_profile.read_text()
    audit.check(
        "common requests: `11724`" in profile_text
        and "| left | 46265 | 34558 | 11707 | 4501 | 174864 | 0.747 |" in profile_text
        and "| right | 60174 | 48471 | 11703 | 3546 | 161152 | 0.806 |" in profile_text,
        "interactive text-only G1 profile contains expected request/KV evidence",
    )

    expected_files = [
        root
        / "qwen_main_sharegpt8_bponly_c4816_20260622_215818"
        / "summary_sharegpt_cbound.csv",
        root
        / "qwen_main_sharegpt8_g1_fullcap_c4816_20260622_214301"
        / "summary_sharegpt_cbound.csv",
        root / "qwen_main_retained8_force_c64_threeway_compare.csv",
        root
        / "qwen_main_video8_fairout_g1_c124_20260623_072204"
        / "video8_fairout_delta.csv",
        root
        / "qwen_main_video8_strictout_g1_c124_20260623_001154"
        / "video8_strictout_delta.csv",
    ]
    for path in expected_files:
        audit.check_file(path, "supporting artifact")

    return audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="directory containing the qwen_main_* artifact trees",
    )
    args = ap.parse_args()

    audit = validate(args.artifact_root)
    for message in audit.passes:
        print(f"PASS {message}")
    for message in audit.failures:
        print(f"FAIL {message}", file=sys.stderr)
    print(f"summary: {len(audit.passes)} passed, {len(audit.failures)} failed")
    return 1 if audit.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
