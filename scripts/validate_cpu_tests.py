#!/usr/bin/env python3
"""Run the retained launcher/runtime CPU regressions using only the standard library."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_SUITES = (
    ("tests", "test_environment.py"),
    ("tests", "test_h200_launcher_portability.py"),
)
RUNTIME_SUITES = (
    ("tests/profile", "test_interaction_trace.py"),
    ("tests/engine", "test_prefetch_route_cpu.py"),
    ("tests/engine", "test_prefetch_completion_cpu.py"),
    ("tests/engine", "test_cancel_trace_cpu.py"),
    ("tests/core", "test_first_chunk_protection_cpu.py"),
    ("tests/engine", "test_prefetch_pin_cpu.py"),
    ("tests/worker", "test_base_profile_lazy_cuda_cpu.py"),
    ("tests/entrypoints/openai_api", "test_kv_policy_cpu.py"),
    ("tests/model_executor/stage_input_processors", "test_qwen_codec_trace_cpu.py"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if sys.version_info < (3, 11):
        parser.error("Python 3.11+ is required")
    suites = [(root / folder, name) for folder, name in ROOT_SUITES]
    suites.extend((root / "vllm-omni" / folder, name) for folder, name in RUNTIME_SUITES)
    missing = [str((folder / name).relative_to(root)) for folder, name in suites
               if not (folder / name).is_file()]
    if missing:
        parser.error("Required CPU test files are missing from this source checkout: " + ", ".join(missing))
    failures = []
    for folder, pattern in suites:
        command = [sys.executable, "-m", "unittest", "discover", "-s", str(folder), "-p", pattern, "-v"]
        print(f"[cpu-tests] {folder.relative_to(root)}/{pattern}", flush=True)
        if subprocess.run(command, cwd=root, check=False).returncode:
            failures.append(str(folder.relative_to(root) / pattern))
    if failures:
        print("FAIL CPU suites: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("PASS retained launcher/runtime CPU regressions; research tooling, model service "
          "and GPU performance were not validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
