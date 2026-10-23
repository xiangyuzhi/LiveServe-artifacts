#!/usr/bin/env python3
"""Validate no-GPU wiring for Ming integration on top of current main.

This guard is intentionally static.  It checks that the Ming-specific adapter
was added without replacing the current Qwen scheduler/G1 launchers, and that
the default Ming validation path uses the 4GPU TP2/talker-DP2 topology before
using GPUs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]


class Audit:
    def __init__(self) -> None:
        self.passes: list[str] = []
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if condition:
            self.passes.append(message)
        else:
            self.failures.append(message)


def read(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8")


def exists(root: Path, rel: str) -> bool:
    return (root / rel).is_file()


def near(text: str, anchor: str, needles: list[str], span: int = 1200) -> bool:
    idx = text.find(anchor)
    if idx < 0:
        return False
    window = text[idx : idx + span]
    return all(needle in window for needle in needles)


def validate(root: Path) -> Audit:
    audit = Audit()

    required = [
        "benchmarks/ming/H200/ming_stream_benchmark.py",
        "benchmarks/ming/H200/ming_quality_probe.py",
        "benchmarks/ming/H200/ming_cbound_driver.py",
        "profiling/scheduler_analysis/scripts/gpu_preflight_common.sh",
        "profiling/scheduler_analysis/scripts/run_ming_cbound_smoke.sh",
        "profiling/scheduler_analysis/scripts/run_ming_4gpu_tuning_sweep.sh",
        "profiling/scheduler_analysis/scripts/summarize_ming_4gpu_tuning.py",
        "scripts/validate_ming_integration_no_gpu.sh",
        "scripts/H200/start_ming_4gpu_dp2_baseline.sh",
        "scripts/H200/start_ming_4gpu_dp2_async.sh",
        "deploy/H200/ming_flash_omni_4gpu_tp2_talker_dp2.yaml",
        "deploy/H200/ming_flash_omni_4gpu_tp2_talker_dp2_async.yaml",
    ]
    missing = [rel for rel in required if not exists(root, rel)]
    audit.check(not missing, "Ming integration entrypoints are present" + ("" if not missing else ": " + ", ".join(missing)))

    ming_no_gpu_gate = read(root, "scripts/validate_ming_no_gpu.sh")
    smoke = read(root, "profiling/scheduler_analysis/scripts/run_ming_cbound_smoke.sh")
    gpu_common = read(root, "profiling/scheduler_analysis/scripts/gpu_preflight_common.sh")
    audit.check(
        all(
            token in gpu_common
            for token in [
                "GPU_MIN_FREE_GIB",
                "GPU_WAIT_FOR_FREE_SEC",
                "GPU_FREE_STABLE_SAMPLES",
                "GPU_REQUIRE_NO_COMPUTE_APPS",
                "GPU_PRECHECK_ONLY",
                "check_gpu_free_once()",
                "check_gpu_free()",
                "--query-compute-apps=pid,used_memory,process_name",
            ]
        ),
        "shared GPU preflight helper covers Ming memory and compute-app guards",
    )
    audit.check(
        all(
            token in smoke
            for token in [
                'VARIANTS="${VARIANTS:-baseline ming_async_scheduler}"',
                'WORKLOADS="${WORKLOADS:-sharegpt interactive mix}"',
                'BASELINE_DEPLOY="${MING_BASELINE_DEPLOY:-${ROOT}/deploy/H200/ming_flash_omni_4gpu_tp2_talker_dp2.yaml}"',
                'ASYNC_DEPLOY="${MING_ASYNC_DEPLOY:-${ROOT}/deploy/H200/ming_flash_omni_4gpu_tp2_talker_dp2_async.yaml}"',
                'TEXT_GPUS="${TEXT_GPUS:-4,5,6,7}"',
                'TEXT_PORT="${TEXT_PORT:-8098}"',
                'DRIVER="${MING_CB_DRIVER:-${ROOT}/benchmarks/ming/H200/ming_cbound_driver.py}"',
                'QUALITY_DRIVER="${MING_QUALITY_DRIVER:-${ROOT}/benchmarks/ming/H200/ming_quality_probe.py}"',
                'export CUDA_VISIBLE_DEVICES="${TEXT_GPUS}"',
                'export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"',
                'export VLLM_OMNI_STAGE_INIT_MAX_WORKERS="${VLLM_OMNI_STAGE_INIT_MAX_WORKERS:-1}"',
                "MING_CUDA_WHEEL_DIR",
                "MING_NVRTC_LINK_DIR",
                "apply_nvrtc_env",
                "CPATH=",
                "libnvrtc.so",
                "MING_QUALITY_PROBE",
                "run_quality_probe",
                "GPU_PREFLIGHT_LIB",
                "gpu_preflight_common.sh",
                'source "${GPU_PREFLIGHT_LIB}"',
                "gpu_precheck_only_enabled",
                "check_gpu_free",
                'env_snapshot_args >"${server_dir}/env_snapshot.txt"',
                'cat >"${server_dir}/config.json"',
                "SERVER_INIT_TIMEOUT",
                '--init-timeout "${SERVER_INIT_TIMEOUT}"',
                '--stage-init-timeout "${SERVER_STAGE_INIT_TIMEOUT}"',
            ]
        ),
        "Ming smoke runner defaults to 4GPU TP2/talker-DP2 and wires driver, NVRTC, GPU preflight, env snapshots and conservative stage init",
    )
    audit.check(
        "start_ming_4gpu_dp2_baseline.sh" in ming_no_gpu_gate
        and "start_ming_4gpu_dp2_async.sh" in ming_no_gpu_gate
        and "start_ming_tp4_4gpu_baseline.sh" in ming_no_gpu_gate
        and "start_ming_tp4_4gpu_async.sh" in ming_no_gpu_gate
        and "start_ming_tp2_3gpu" not in ming_no_gpu_gate
        and "ming_flash_omni_3gpu" not in ming_no_gpu_gate,
        "Ming no-GPU validation gate is restricted to 4GPU launchers",
    )

    root_readme = read(root, "README.md")
    docs_readme = read(root, "docs/legacy_h200_guide.md")
    ming_readme = read(root, "benchmarks/ming/README.md")
    ming_tp_doc = read(root, "docs/experiments/ming_tp_reproduction.md")
    scripts_readme = read(root, "scripts/README.md")
    scheduler_readme = read(root, "profiling/scheduler_analysis/README.md")
    audit.check(
        "Ming 4GPU TP2/talker-DP2 smoke" in docs_readme
        and "run_ming_cbound_smoke.sh" in docs_readme
        and "start_ming_4gpu_dp2_async.sh" in docs_readme
        and 'VARIANTS="baseline ming_async_scheduler ming_liveserve_g1 ming_liveserve_ttfp_best"' in docs_readme
        and 'C_LEVELS="2"' in docs_readme
        and "NUM_REQUESTS=4" in docs_readme
        and "MAX_TOKENS=64" in docs_readme
        and ("Ming TP2/" + "3GPU smoke") not in docs_readme
        and "default to the 4GPU TP2/talker-DP2 path on port 8098" in ming_readme,
        "historical Ming guide preserves its 4GPU TP2/talker-DP2 c=2 n=4 validation path",
    )
    audit.check(
        "docs/legacy_h200_guide.md" in root_readme
        and "Ming" in root_readme
        and "ming_4gpu_dp2_goal_cont_20260626_181650" in ming_tp_doc
        and "ming_liveserve_g1" in ming_tp_doc
        and "ming_liveserve_ttfp_best" in ming_tp_doc,
        "root links the historical guide and Ming evidence remains in its experiment report",
    )
    audit.check(
        "## Ming C-Bound And Tuning" in scheduler_readme
        and "TEXT_GPUS=4,5,6,7" in scheduler_readme
        and "TEXT_PORT=8098" in scheduler_readme
        and 'VARIANTS="baseline ming_async_scheduler ming_liveserve_g1 ming_liveserve_ttfp_best"' in scheduler_readme
        and 'C_LEVELS="2"' in scheduler_readme
        and "NUM_REQUESTS=4" in scheduler_readme
        and "MAX_TOKENS=64" in scheduler_readme
        and "MING_STOP_AFTER_FIRST_AUDIO=1" in scheduler_readme
        and "bash profiling/scheduler_analysis/scripts/run_ming_cbound_smoke.sh" in scheduler_readme,
        "scheduler profiling README uses the 4GPU/8098 Ming c=2 n=4 validation example",
    )
    audit.check(
        "Maintained 4GPU medium smoke through the integrated runner" in ming_tp_doc
        and "ming_4gpu_dp2_goal_cont_20260626_181650" in ming_tp_doc
        and 'VARIANTS="baseline ming_async_scheduler ming_liveserve_g1 ming_liveserve_ttfp_best"' in ming_tp_doc
        and 'C_LEVELS="2"' in ming_tp_doc
        and "NUM_REQUESTS=4" in ming_tp_doc
        and "MAX_TOKENS=64" in ming_tp_doc
        and "scripts/validate_reduced_smoke_artifacts.py" in ming_tp_doc,
        "Ming TP reproduction doc records the maintained 4GPU medium validation root",
    )
    audit.check(
        "Qwen reduced smoke plus Ming 4GPU medium smoke" in scripts_readme
        and "start_ming_4gpu_dp2_baseline.sh" in scripts_readme
        and "start_ming_4gpu_dp2_async.sh" in scripts_readme
        and "start_ming_tp4_4gpu_baseline.sh" in scripts_readme
        and "start_ming_tp4_4gpu_async.sh" in scripts_readme
        and "C_LEVELS=2" in scripts_readme
        and "NUM_REQUESTS=4" in scripts_readme
        and "MAX_TOKENS=64" in scripts_readme,
        "scripts README documents Ming 4GPU medium validation and launchers",
    )
    audit.check(
        "apply_ming_async_env" in smoke
        and near(
            smoke,
            "apply_ming_async_env_values()",
            [
                "OMNI_MING_INCREMENTAL=1",
                "OMNI_TALKER_LEAD_MAX",
                "OMNI_TALKER_DEFER_HOLD_MAX",
                "OMNI_TALKER_STAGE_IDS=1",
                "OMNI_THINKER_STAGE_IDS=0",
            ],
        )
        and near(
            smoke,
            "ming_async_scheduler|async|optimized)",
            ["apply_ming_async_env"],
            span=400,
        ),
        "Ming async variant enables incremental path and scheduler lead controls",
    )
    audit.check(
        all(
            token in smoke
            for token in [
                "scheduler_policy_common.sh",
                "eviction_policy_common.sh",
                "ming_liveserve_g1|ming_fullstack_g1|ming_allopt)",
                "ming_liveserve_ttfp_best|ming_ttfp_best|ming_liveserve_g1_ttfp)",
                "apply_ming_ttfp_tuned_env",
                "MING_TTFP_TUNED_CHUNK_STEPS",
                "MING_TTFP_TUNED_SCHEDULER_POLICY_OVERLAYS",
                "scheduler_policy_apply_and_export",
                "apply_prefetch_on_env",
                "OMNI_PREFETCH_ADMISSION=1",
                "OMNI_SCHED_NATIVE_PREFETCH=1",
                "eviction_policy_apply_g1_env",
                "OMNI_KV_EVICT_CAPTURE_ALLOCATE_SLOTS",
                "LIVE_VARIANT_FOR_PLOTS",
            ]
        ),
        "Ming full-stack variant wires scheduler policy, native prefetch, G1 eviction and plot selection",
    )

    sweep = read(root, "profiling/scheduler_analysis/scripts/run_ming_4gpu_tuning_sweep.sh")
    audit.check(
        all(
            token in sweep
            for token in [
                'TEXT_GPUS="${TEXT_GPUS:-4,5,6,7}"',
                "ming_flash_omni_4gpu_tp2_talker_dp2.yaml",
                "ming_flash_omni_4gpu_tp2_talker_dp2_async.yaml",
                'WORKLOADS="${WORKLOADS:-sharegpt interactive mix}"',
                'C_LEVELS="${C_LEVELS:-2 4}"',
                "async_nodefer_chunk1",
                "full_nodefer_chunk1",
                "expected_summary_path",
                "candidate_has_complete_results",
                "candidate.complete",
                "P_LABEL",
                "GPU_PREFLIGHT_LIB",
                "gpu_preflight_common.sh",
                'source "${GPU_PREFLIGHT_LIB}"',
                "gpu_precheck_only_enabled",
                "GPU_PRECHECK_ONLY",
                "GPU_REQUIRE_NO_COMPUTE_APPS",
                "GPU_MIN_FREE_GIB",
                "SERVER_INIT_TIMEOUT",
                '--retained-workloads "interactive,mix"',
                "summarize_ming_4gpu_tuning.py",
            ]
        ),
        "Ming 4GPU tuning sweep defaults to GPU4-7, TP2/talker-DP2 deploys, three workloads, GPU preflight and candidate summary",
    )

    summarizer = read(root, "profiling/scheduler_analysis/scripts/summarize_ming_4gpu_tuning.py")
    audit.check(
        all(
            token in summarizer
            for token in [
                "Primary metric",
                "baseline",
                "retained_p50_improve_avg",
                "--retained-workloads",
                "parse_workloads",
                "ming_async_scheduler",
                "missing_baseline_keys",
                "missing baseline summaries",
                "summary_candidates.csv",
                "summary_candidates.md",
            ]
        ),
        "Ming 4GPU tuning summarizer compares candidates against baseline and ranks retained workloads",
    )

    audit.check(
        "reset_omni_env" in smoke
        and "unset" in smoke
        and "compgen -v OMNI_" in smoke
        and "compgen -v SCHEDULER_POLICY_" in smoke,
        "Ming smoke runner clears inherited OMNI/scheduler policy env before variants",
    )

    driver = read(root, "benchmarks/ming/H200/ming_cbound_driver.py")
    stream_driver = read(root, "benchmarks/ming/H200/ming_stream_benchmark.py")
    quality_driver = read(root, "benchmarks/ming/H200/ming_quality_probe.py")
    audit.check(
        'choices=("sharegpt", "interactive", "mix")' in driver
        and "sampling_params_list" in driver
        and "committed_prefix_hash" in driver
        and "summary.json" in driver
        and "per_request.jsonl" in driver
        and "meta.json" in driver,
        "Ming c-bound driver supports Qwen-style workload lanes and output layout",
    )
    audit.check(
        'DEFAULT_URL = "http://127.0.0.1:8098/v1/chat/completions"' in driver
        and 'DEFAULT_URL = "http://127.0.0.1:8098/v1/chat/completions"' in stream_driver
        and 'DEFAULT_URL = "http://127.0.0.1:8098/v1/chat/completions"' in quality_driver,
        "Ming benchmark clients default to the 4GPU DP2 validation port",
    )

    baseline_launcher = read(root, "scripts/H200/start_ming_4gpu_dp2_baseline.sh")
    async_launcher = read(root, "scripts/H200/start_ming_4gpu_dp2_async.sh")
    audit.check(
        "compgen -e OMNI_" in baseline_launcher
        and "compgen -e SCHEDULER_POLICY_" in baseline_launcher
        and 'export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"' in baseline_launcher
        and "ming_flash_omni_4gpu_tp2_talker_dp2.yaml" in baseline_launcher
        and 'CUDA_DEVS="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"' in baseline_launcher
        and 'PORT="${LIVESERVE_PORT:-8098}"' in baseline_launcher
        and "--init-timeout" in baseline_launcher
        and "MING_CUDA_WHEEL_DIR" in baseline_launcher
        and "MING_NVRTC_LINK_DIR" in baseline_launcher
        and "libnvrtc.so" in baseline_launcher
        and "CPLUS_INCLUDE_PATH" in baseline_launcher
        and "OMNI_*/SCHEDULER_POLICY_*   = cleared" in baseline_launcher,
        "Ming 4GPU DP2 baseline launcher clears optimization env, injects NVRTC include path and uses DP2 baseline deploy",
    )
    audit.check(
        "OMNI_MING_INCREMENTAL=1" in async_launcher
        and "OMNI_TALKER_LEAD_MAX" in async_launcher
        and "OMNI_TALKER_STAGE_IDS=1" in async_launcher
        and 'export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"' in async_launcher
        and "compgen -e SCHEDULER_POLICY_" in async_launcher
        and "--init-timeout" in async_launcher
        and "ming_flash_omni_4gpu_tp2_talker_dp2_async.yaml" in async_launcher
        and 'CUDA_DEVS="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"' in async_launcher
        and 'PORT="${LIVESERVE_PORT:-8098}"' in async_launcher,
        "Ming 4GPU DP2 async launcher clears scheduler policy env, enables incremental + lead controls and uses async DP2 deploy",
    )

    baseline_4gpu_yaml = read(root, "deploy/H200/ming_flash_omni_4gpu_tp2_talker_dp2.yaml")
    async_4gpu_yaml = read(root, "deploy/H200/ming_flash_omni_4gpu_tp2_talker_dp2_async.yaml")
    audit.check(
        "pipeline: ming_flash_omni" in baseline_4gpu_yaml
        and "async_chunk: false" in baseline_4gpu_yaml
        and "tensor_parallel_size: 2" in baseline_4gpu_yaml
        and 'devices: "0,1"' in baseline_4gpu_yaml
        and 'devices: "2,3"' in baseline_4gpu_yaml
        and "num_replicas: 2" in baseline_4gpu_yaml,
        "Ming 4GPU baseline deploy uses thinker TP2 and talker DP2",
    )
    audit.check(
        "pipeline: ming_flash_omni" in async_4gpu_yaml
        and "async_chunk: true" in async_4gpu_yaml
        and "SharedMemoryConnector" in async_4gpu_yaml
        and "tensor_parallel_size: 2" in async_4gpu_yaml
        and 'devices: "0,1"' in async_4gpu_yaml
        and 'devices: "2,3"' in async_4gpu_yaml
        and "num_replicas: 2" in async_4gpu_yaml,
        "Ming 4GPU async deploy uses connector, thinker TP2 and talker DP2",
    )

    ming_gate = read(root, "scripts/validate_ming_integration_no_gpu.sh")
    audit.check(
        "validate_ming_runner_wiring.py" in ming_gate
        and "gpu_preflight_common.sh" in ming_gate
        and "run_ming_cbound_smoke.sh" in ming_gate
        and "run_ming_4gpu_tuning_sweep.sh" in ming_gate
        and "summarize_ming_4gpu_tuning.py" in ming_gate,
        "dedicated Ming no-GPU gate includes runner, sweep and summarizer checks",
    )
    audit.check(
        'source "${ROOT}/scripts/env_defaults.sh"' in ming_gate
        and 'PYBIN="$(liveserve_resolve_pybin "${ROOT}")"' in ming_gate
        and "liveserve_export_pycache_prefix" in ming_gate,
        "dedicated Ming no-GPU gate uses shared validation env defaults",
    )

    return audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    args = ap.parse_args()

    audit = validate(args.repo_root)
    for message in audit.passes:
        print(f"PASS {message}")
    for message in audit.failures:
        print(f"FAIL {message}", file=sys.stderr)
    print(f"summary: {len(audit.passes)} passed, {len(audit.failures)} failed")
    return 1 if audit.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
