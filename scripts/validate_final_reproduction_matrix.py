#!/usr/bin/env python3
"""Validate the final reproduction matrix against the repository layout."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = Path("docs/experiments/final_reproduction_matrix.md")
CHECKLIST_PATH = Path("docs/experiments/merge_readiness_checklist.md")

REQUIRED_QWEN_WORKLOADS = (
    "ShareGPT audio, arrival, and barge-in",
    "ShareGPT origin c-bound",
    "Retained interactive audio",
    "Interactive text/KV pressure",
    "StreamingBench video",
    "Mixed text/video smoke",
)

REQUIRED_MING_TOPOLOGIES = (
    "TP4 thinker + colocated talker",
    "TP2 thinker + talker DP2",
)

REQUIRED_MING_WORKLOADS = (
    "Ming c-bound driver",
    "run_ming_cbound_smoke.sh",
    "run_ming_4gpu_tuning_sweep.sh",
)

REQUIRED_COMMANDS = (
    "bash scripts/validate_main_no_gpu.sh",
    "bash scripts/validate_ming_no_gpu.sh",
)

REQUIRED_LATEST_EVIDENCE = (
    "qwen_allworkloads_gpu47_currenttree_final_20260626_163306",
    "qwen_allworkloads_gpu47_post_ming_unified_retry_20260626_140335",
    "ming_4gpu_dp2_allworkloads_post_ming4gpu_20260626_124936",
    "ming_4gpu_dp2_goalcheck_20260626_144410",
    "ming_4gpu_dp2_currenttree_smoke_20260626_155256",
    "ming_4gpu_dp2_medium_goal_20260626_170814",
)

REQUIRED_CHECKLIST_SECTIONS = (
    "## Worktree Inventory",
    "## Validation Evidence",
    "## Goal Coverage Audit",
    "## Commit Packaging Recommendation",
    "## Branch Cleanup",
    "## Merge Gates",
    "## Residual Risks",
)


class Audit:
    def __init__(self) -> None:
        self.passes: list[str] = []
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if condition:
            self.passes.append(message)
        else:
            self.failures.append(message)


def extract_code_paths(text: str) -> list[str]:
    paths: list[str] = []
    for item in re.findall(r"`([^`]+)`", text):
        if not re.search(r"\.(sh|py|md|yaml|json|csv)$", item):
            continue
        if any(token in item for token in (" ", "<", ">", "*", "$", "|")):
            continue
        paths.append(item)
    return paths


def validate(root: Path) -> Audit:
    audit = Audit()
    matrix = root / MATRIX_PATH
    audit.check(matrix.is_file(), f"{MATRIX_PATH} exists")
    if not matrix.is_file():
        return audit

    text = matrix.read_text(encoding="utf-8")
    checklist = root / CHECKLIST_PATH
    audit.check(checklist.is_file(), f"{CHECKLIST_PATH} exists")
    checklist_text = checklist.read_text(encoding="utf-8") if checklist.is_file() else ""

    for workload in REQUIRED_QWEN_WORKLOADS:
        audit.check(workload in text, f"Qwen workload listed: {workload}")
    for topology in REQUIRED_MING_TOPOLOGIES:
        audit.check(topology in text, f"Ming topology listed: {topology}")
    for workload in REQUIRED_MING_WORKLOADS:
        audit.check(workload in text, f"Ming workload/tool listed: {workload}")
    for command in REQUIRED_COMMANDS:
        audit.check(command in text, f"validation command documented: {command}")
    audit.check(
        "Last no-GPU check in the integration worktree: 2026-06-28." in text,
        "matrix records the latest no-GPU check date",
    )
    audit.check(
        "internal H20 worker notes" in text
        and "stale hardware" in text
        and "capacity labels" in text
        and "local smoke checkout paths" in text
        and "deploy/profiling comments avoid internal H20 worker notes" in checklist_text,
        "matrix and checklist document open-source hygiene guard coverage",
    )
    audit.check(
        "`benchmarks/sharegpt/H200/sharegpt_driver.py`" in text
        and "`sharegpt_driver.py`" not in text,
        "matrix uses repo-relative ShareGPT driver path",
    )
    for evidence in REQUIRED_LATEST_EVIDENCE:
        audit.check(
            evidence in text,
            f"latest matrix evidence listed: {evidence}",
        )
        audit.check(
            evidence in checklist_text,
            f"latest merge-readiness evidence listed: {evidence}",
        )
    for section in REQUIRED_CHECKLIST_SECTIONS:
        audit.check(section in checklist_text, f"merge-readiness section present: {section}")
    audit.check(
        "Ming 4GPU wrapper smoke on any four free GPUs" in checklist_text,
        "merge-readiness Ming GPU gate documents the 4GPU default path",
    )
    reduced_validator = root / "scripts/validate_reduced_smoke_artifacts.py"
    main_gate = (root / "scripts/validate_main_no_gpu.sh").read_text(encoding="utf-8")
    audit.check(
        reduced_validator.is_file()
        and "validate_reduced_smoke_artifacts.py" in text
        and "validate_reduced_smoke_artifacts.py" in main_gate
        and "SKIP_REDUCED_SMOKE_ARTIFACT_VALIDATION" in main_gate,
        "reduced smoke artifact validator is documented and wired into the main gate",
    )

    paths = sorted(set(extract_code_paths(text) + extract_code_paths(checklist_text)))
    missing: list[str] = []
    for rel in paths:
        if not (root / rel).exists():
            missing.append(rel)
    audit.check(
        not missing,
        "all matrix code-path references exist"
        + ("" if not missing else ": " + ", ".join(missing)),
    )

    gpu_lanes = ("GPU4-6", "GPU4-7", "GPU6,7", "CUDA_VISIBLE_DEVICES=4,5,6,7")
    audit.check(
        any(lane in text for lane in gpu_lanes),
        "GPU lane guidance is present for follow-up reproduction",
    )
    audit.check(
        "Do not commit raw results" in text,
        "raw result handling policy is present",
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    args = parser.parse_args()

    audit = validate(args.repo_root)
    for message in audit.passes:
        print(f"PASS {message}")
    for message in audit.failures:
        print(f"FAIL {message}", file=sys.stderr)
    print(f"summary: {len(audit.passes)} passed, {len(audit.failures)} failed")
    return 1 if audit.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
