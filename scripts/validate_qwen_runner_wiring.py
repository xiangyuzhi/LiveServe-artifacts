#!/usr/bin/env python3
"""Validate no-GPU wiring for current Qwen H200 benchmark variants.

The artifact validator checks result evidence.  This script checks the runner
and launcher wiring that should produce those results:

* baseline / baseline_no_offload disables scheduler, preload and custom eviction
* scheduler-only variants enable LiveServe scheduling but keep preload and custom eviction off
* liveserve_g1 enables scheduling, history preload and G1 eviction

The checks are intentionally lightweight static guards.  They catch accidental
variant-name drift and environment-cleanup regressions without starting a
server or using GPUs.
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


def check_rename_contract(audit: Audit, root: Path) -> None:
    legacy_scheduler = "b" + "pv3"
    legacy_interrupt = "inter" + "rupt"
    renamed = [
        (
            f"benchmarks/interactive/H200/aggregate_{legacy_interrupt}3gpu.py",
            "benchmarks/interactive/H200/aggregate_bargein3gpu.py",
        ),
        (
            f"benchmarks/interactive/H200/{legacy_scheduler}_run_one_cell.sh",
            "benchmarks/interactive/H200/scheduler_run_one_cell.sh",
        ),
        (
            f"benchmarks/interactive/H200/client_{legacy_interrupt}.py",
            "benchmarks/interactive/H200/client_bargein.py",
        ),
        (
            f"benchmarks/interactive/H200/plot_{legacy_interrupt}3gpu_matrix.py",
            "benchmarks/interactive/H200/plot_bargein3gpu_matrix.py",
        ),
        (
            f"benchmarks/interactive/H200/run_{legacy_interrupt}3gpu_matrix.sh",
            "benchmarks/interactive/H200/run_bargein3gpu_matrix.sh",
        ),
        (
            f"benchmarks/interactive/H200/step1_{legacy_interrupt}_chain_a.sh",
            "benchmarks/interactive/H200/step1_bargein_chain_a.sh",
        ),
        (
            f"benchmarks/interactive/H200/step1_{legacy_interrupt}_chain_b.sh",
            "benchmarks/interactive/H200/step1_bargein_chain_b.sh",
        ),
        (
            f"benchmarks/interactive/H200/step2_{legacy_scheduler}_chain.sh",
            "benchmarks/interactive/H200/step2_scheduler_chain.sh",
        ),
        (
            f"benchmarks/interactive/H200/step2_{legacy_scheduler}_chain_prob.sh",
            "benchmarks/interactive/H200/step2_scheduler_chain_prob.sh",
        ),
        (
            f"profiling/{legacy_scheduler}_analysis/README.md",
            "profiling/scheduler_analysis/README.md",
        ),
        (
            f"profiling/{legacy_scheduler}_analysis/scripts/analyze_kv_pressure_windows.py",
            "profiling/scheduler_analysis/scripts/analyze_kv_pressure_windows.py",
        ),
        (
            f"profiling/{legacy_scheduler}_analysis/scripts/bp_policy_common.sh",
            "profiling/scheduler_analysis/scripts/scheduler_policy_common.sh",
        ),
        (
            f"profiling/{legacy_scheduler}_analysis/scripts/eviction_policy_common.sh",
            "profiling/scheduler_analysis/scripts/eviction_policy_common.sh",
        ),
        (
            f"profiling/{legacy_scheduler}_analysis/scripts/plot_retained_cbound_qps_ttfp.py",
            "profiling/scheduler_analysis/scripts/plot_retained_cbound_qps_ttfp.py",
        ),
        (
            f"profiling/{legacy_scheduler}_analysis/scripts/plot_sharegpt_cbound_ttfp_qps.py",
            "profiling/scheduler_analysis/scripts/plot_sharegpt_cbound_ttfp_qps.py",
        ),
        (
            f"profiling/{legacy_scheduler}_analysis/scripts/run_all_workloads_kv_profile_smoke.sh",
            "profiling/scheduler_analysis/scripts/run_all_workloads_kv_profile_smoke.sh",
        ),
        (
            f"profiling/{legacy_scheduler}_analysis/scripts/run_retained_cbound_sweep.sh",
            "profiling/scheduler_analysis/scripts/run_retained_cbound_sweep.sh",
        ),
        (
            f"profiling/{legacy_scheduler}_analysis/scripts/run_sharegpt_origin_cbound_sweep.sh",
            "profiling/scheduler_analysis/scripts/run_sharegpt_origin_cbound_sweep.sh",
        ),
        (
            "profiling/kv_offload_prefetch/large_scale/run_4gpu_" + "bp" + "_m38.sh",
            "profiling/kv_offload_prefetch/large_scale/run_4gpu_scheduler_preload.sh",
        ),
        (
            "profiling/kv_offload_prefetch/large_scale/run_4gpu_"
            + "bp"
            + "_m38_eviction.sh",
            "profiling/kv_offload_prefetch/large_scale/run_4gpu_scheduler_preload_eviction.sh",
        ),
        (
            f"benchmarks/sharegpt/H200/{legacy_interrupt}/plot_b2_"
            + "clint.py",
            "benchmarks/sharegpt/H200/bargein/plot_b2_cl_bargein.py",
        ),
        (
            "profiling/kv_offload_prefetch/sharegpt_"
            f"{legacy_interrupt}/scripts/plot_sharegpt_{legacy_interrupt}.py",
            "profiling/kv_offload_prefetch/sharegpt_bargein/scripts/plot_sharegpt_bargein.py",
        ),
        (
            f"scripts/H200/reproduce_interactive_{legacy_scheduler}.sh",
            "scripts/H200/reproduce_interactive_scheduler.sh",
        ),
        (
            f"scripts/H200/reproduce_sharegpt_{legacy_scheduler}.sh",
            "scripts/H200/reproduce_sharegpt_scheduler.sh",
        ),
        (
            f"scripts/H200/start_server_2gpu_{legacy_scheduler}.sh",
            "scripts/H200/start_server_2gpu_scheduler.sh",
        ),
        (
            f"scripts/H200/start_server_2gpu_video_{legacy_scheduler}.sh",
            "scripts/H200/start_server_2gpu_video_scheduler.sh",
        ),
        (
            f"scripts/H200/start_server_3gpu_video_{legacy_scheduler}.sh",
            "scripts/H200/start_server_3gpu_video_scheduler.sh",
        ),
    ]

    missing_new = [new for _, new in renamed if not (root / new).is_file()]
    leftover_old = [old for old, _ in renamed if (root / old).exists()]
    audit.check(
        not missing_new and not leftover_old,
        "legacy entrypoint renames have replacements and no leftover old files"
        + (
            ""
            if not missing_new and not leftover_old
            else f": missing_new={missing_new[:8]} leftover_old={leftover_old[:8]}"
        ),
    )


def check_scheduler_config_module(audit: Audit, root: Path) -> None:
    legacy_module = "bp" + "_lead_governor_config"
    new_config = root / "vllm-omni/vllm_omni/core/sched/lead_governor_config.py"
    old_config = root / f"vllm-omni/vllm_omni/core/sched/{legacy_module}.py"
    sched_files = [
        root / "vllm-omni/vllm_omni/core/sched/omni_ar_scheduler.py",
        root / "vllm-omni/vllm_omni/core/sched/omni_generation_scheduler.py",
        root / "vllm-omni/vllm_omni/core/sched/omni_scheduler_mixin.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sched_files if path.exists())
    audit.check(
        new_config.is_file()
        and not old_config.exists()
        and "lead_governor_config import LeadGovernorParams" in text
        and legacy_module not in text,
        "vllm-omni scheduler config module is renamed and imported from public name",
    )


def near(text: str, anchor: str, needles: list[str], span: int = 900) -> bool:
    idx = text.find(anchor)
    if idx < 0:
        return False
    window = text[idx : idx + span]
    return all(needle in window for needle in needles)


def before(text: str, first: str, second: str) -> bool:
    first_idx = text.find(first)
    second_idx = text.find(second)
    return first_idx >= 0 and second_idx >= 0 and first_idx < second_idx


def check_gpu_preflight_helper(audit: Audit, gpu_common: str) -> None:
    audit.check(
        all(
            token in gpu_common
            for token in [
                "GPU_MIN_FREE_GIB",
                "GPU_WAIT_FOR_FREE_SEC",
                "GPU_FREE_STABLE_SAMPLES",
                "GPU_REQUIRE_NO_COMPUTE_APPS",
                "GPU_PRECHECK_ONLY",
                "gpu_require_no_compute_apps_enabled()",
                "gpu_precheck_only_enabled()",
                "check_gpu_free_once()",
                "check_gpu_free()",
                "--query-compute-apps=pid,used_memory,process_name",
                "already has compute app(s)",
            ]
        ),
        "shared GPU preflight helper covers memory, compute-app guard and explicit precheck mode",
    )


def check_validation_env_defaults(audit: Audit, root: Path) -> None:
    helper = read(root, "scripts/env_defaults.sh")
    audit.check(
        "liveserve_resolve_pybin()" in helper
        and ".venv-liveserve/bin/python" in helper
        and ".venv-liveserve-dev/bin/python" in helper
        and "PYBIN_FALLBACK:-python3" in helper
        and "liveserve_export_pycache_prefix()" in helper
        and "liveserve_require_command()" in helper,
        "validation env defaults prefer explicit/repo-local interpreters and a portable fallback",
    )

    for rel in [
        "scripts/validate_qwen_main_no_gpu.sh",
        "scripts/validate_ming_no_gpu.sh",
        "scripts/validate_ming_integration_no_gpu.sh",
    ]:
        text = read(root, rel)
        audit.check(
            'source "${ROOT}/scripts/env_defaults.sh"' in text
            and 'PYBIN="$(liveserve_resolve_pybin "${ROOT}")"' in text
            and 'liveserve_require_command "${PYBIN}" "python"' in text
            and "liveserve_export_pycache_prefix" in text
            and "/workspace/xyz/omni_020/bin/python" not in text,
            f"{rel}: uses shared validation env defaults instead of inline machine path",
        )


def check_common_variant_script(audit: Audit, label: str, text: str) -> None:
    audit.check(
        near(
            text,
            "baseline|baseline_no_offload|z0_strict)",
            [
                "SCHEDULER_POLICY_PRESET=baseline",
                "apply_prefetch_off_env",
                "apply_lru_env",
            ],
        ),
        f"{label}: strict baseline disables scheduler/preload/custom eviction",
    )
    audit.check(
        (
            near(
                text,
                "scheduler_legacy_l12|liveserve_scheduler)",
                [
                    "apply_prefetch_off_env",
                    "apply_lru_env",
                ],
            )
            or near(
                text,
                "liveserve_scheduler|scheduler_legacy_l12)",
                [
                    "apply_prefetch_off_env",
                    "apply_lru_env",
                ],
            )
        ),
        f"{label}: scheduler-only variant keeps preload/custom eviction off",
    )
    audit.check(
        near(
            text,
            "liveserve_g1|liveserve_allopt)",
            [
                "LIVE_SCHEDULER_POLICY_PRESET",
                "apply_prefetch_on_env",
                "apply_g1_eviction_env",
            ],
        ),
        f"{label}: liveserve_g1 enables scheduler, preload and G1 eviction",
    )
    audit.check(
        "GPU_PREFLIGHT_LIB" in text
        and "gpu_preflight_common.sh" in text
        and 'source "${GPU_PREFLIGHT_LIB}"' in text
        and "check_gpu_free" in text
        and "gpu_precheck_only_enabled" in text
        and "GPU availability preflight failed" in text
        and "GPU precheck-only failed" in text,
        f"{label}: GPU preflight/no-workload guard is wired",
    )


def check_public_wrappers(audit: Audit, root: Path) -> None:
    public_implementations = {
        "profiling/scheduler_analysis/scripts/run_sharegpt_origin_cbound_sweep.sh": [
            "profiling/scheduler_analysis/results/sharegpt_origin_cbound_",
            "SCHEDULER_POLICY_LIB",
            "scheduler_policy_apply_and_export",
            "MAINTAINED_BARGE_IN_BENCH",
            "qwen3_omni_moe_4thinker_4talker_8gpu_standardcap.yaml",
            'PLOT_BASELINE_VARIANT="${PLOT_BASELINE_VARIANT:-baseline_no_offload}"',
            'VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"',
        ],
        "profiling/scheduler_analysis/scripts/run_retained_cbound_sweep.sh": [
            "profiling/scheduler_analysis/results/retained_cbound_",
            "SCHEDULER_POLICY_LIB",
            "scheduler_policy_apply_and_export",
            "PRELOAD_LEAD_MS",
            'VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"',
        ],
        "profiling/scheduler_analysis/scripts/run_all_workloads_kv_profile_smoke.sh": [
            "profiling/scheduler_analysis/results/${RUN_ID}",
            "apply_scheduler_policy_preset",
            "profiling/scheduler_analysis/scripts/analyze_kv_pressure_windows.py",
            "GPU_PREFLIGHT_LIB",
            "check_service_gpu_preflight",
            "gpu_precheck_only_enabled",
            "is_g1_variant()",
            "LIVE_SCHEDULER_POLICY_OVERLAYS",
            "MIX_REQUEST_ID_PREFIX",
            'REQUEST_ID_PREFIX="${MIX_REQUEST_ID_PREFIX:-${mix_run_id}-}"',
            "VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}",
            "_runtime_env.txt",
        ],
    }
    for rel, needles in public_implementations.items():
        text = read(root, rel)
        audit.check(
            all(needle in text for needle in needles),
            f"{rel}: public scheduler entrypoint contains implementation",
        )

    implementation_checks = {
        "benchmarks/interactive/H200/scheduler_run_one_cell.sh": [
            "OMNI_TALKER_LEAD_MAX",
            "--barge-in-prob",
            "server_scheduler_",
        ],
        "benchmarks/interactive/H200/step2_scheduler_chain.sh": [
            "scheduler_run_one_cell.sh",
            "scheduler_wide",
        ],
        "benchmarks/interactive/H200/run_bargein3gpu_matrix.sh": [
            "results/H200/bargein3gpu",
            "--barge-in-prob",
        ],
        "scripts/H200/start_server_2gpu_scheduler.sh": [
            "scheduler lead control enabled",
            "OMNI_TALKER_LEAD_MAX",
        ],
        "scripts/H200/start_server_2gpu_video_scheduler.sh": [
            "scheduler_policy_apply_and_export",
            "profiling/scheduler_analysis/scripts/scheduler_policy_common.sh",
            "SCHEDULER_POLICY_DEFAULT_TALKER_MIN_OUTPUT_TOKENS:-48",
            "OMNI_TALKER_MIN_OUTPUT_TOKENS_BEFORE_DEFER",
        ],
        "scripts/H200/start_server_3gpu_video_scheduler.sh": [
            "scheduler_policy_apply_and_export",
            "profiling/scheduler_analysis/scripts/scheduler_policy_common.sh",
            "SCHEDULER_POLICY_DEFAULT_TALKER_MIN_OUTPUT_TOKENS:-48",
            "OMNI_TALKER_MIN_OUTPUT_TOKENS_BEFORE_DEFER",
        ],
        "scripts/H200/reproduce_interactive_scheduler.sh": [
            "scheduler_run_one_cell.sh",
            "scheduler_wide_p05",
        ],
        "scripts/H200/reproduce_sharegpt_scheduler.sh": [
            "start_server_2gpu_scheduler.sh",
            "sharegpt_benchmark.sh",
        ],
        "profiling/kv_offload_prefetch/large_scale/run_4gpu_scheduler_preload.sh": [
            "SCHEDULER_POLICY_LIB",
            "scheduler_policy_apply_and_export",
            "PRELOAD_LEAD_MS",
            "bargein_kv",
        ],
        "profiling/kv_offload_prefetch/large_scale/run_4gpu_scheduler_preload_eviction.sh": [
            "PRELOAD_MAX_INFLIGHT",
            "run_4gpu_scheduler_preload.sh",
            "OMNI_KV_EVICT_ENFORCE_POLICY",
        ],
    }
    for rel, needles in implementation_checks.items():
        text = read(root, rel)
        audit.check(
            all(needle in text for needle in needles) and "Compatibility wrapper" not in text,
            f"{rel}: public entrypoint contains scheduler/barge-in implementation",
        )

    public_policy = root / "profiling/scheduler_analysis/scripts/scheduler_policy_common.sh"
    public_evict = root / "profiling/scheduler_analysis/scripts/eviction_policy_common.sh"
    audit.check(
        public_policy.exists()
        and "scheduler_policy_apply_and_export()" in public_policy.read_text(encoding="utf-8")
        and public_evict.exists()
        and "eviction_policy_apply_g1_env()" in public_evict.read_text(encoding="utf-8"),
        "scheduler_analysis exposes public policy helper implementations",
    )


def check_sharegpt_paper_runner(audit: Audit, text: str) -> None:
    start_server_idx = text.find("start_server_once()")
    start_server_window = (
        "" if start_server_idx < 0 else text[start_server_idx : start_server_idx + 2600]
    )
    audit.check(
        near(
            text,
            "apply_common_env()",
            [
                "OMNI_KV_EVICT_ENFORCE_POLICY=none",
                "OMNI_KV_EVICT_ENFORCE_INDEX=0",
                "OMNI_PREFETCH_ADMISSION=0",
                "OMNI_PREFETCH_HISTORY_HIT_ENABLE=0",
                "OMNI_SCHED_NATIVE_PREFETCH=0",
            ],
            span=1400,
        ),
        "ShareGPT paper runner: common env keeps preload/custom eviction off",
    )
    audit.check(
        near(
            text,
            "baseline)",
            ['scheduler_policy_apply_and_export baseline ""'],
            span=300,
        ),
        "ShareGPT paper runner: baseline disables scheduler",
    )
    audit.check(
        near(
            text,
            "scheduler_legacy_l12|liveserve_scheduler)",
            ['scheduler_policy_apply_and_export scheduler "${SCHEDULER_POLICY_OVERLAYS}"'],
            span=300,
        ),
        "ShareGPT paper runner: default scheduler-only honors scheduler policy overlays",
    )
    audit.check(
        all(
            token in text
            for token in [
                'scheduler_l8|scheduler_talkerlead_8_4_75',
                'scheduler_policy_apply_and_export scheduler "talkerlead=8:4:75"',
                'scheduler_l8_h25|scheduler_talkerlead_8_4_25',
                'scheduler_policy_apply_and_export scheduler "talkerlead=8:4:25"',
                'scheduler_l12|scheduler_talkerlead_12_6_75',
                'scheduler_policy_apply_and_export scheduler "talkerlead=12:6:75"',
                'scheduler_l16|scheduler_talkerlead_16_8_75',
                'scheduler_policy_apply_and_export scheduler "talkerlead=16:8:75"',
                'scheduler_l20|scheduler_talkerlead_20_10_75',
                'scheduler_policy_apply_and_export scheduler "talkerlead=20:10:75"',
            ]
        ),
        "ShareGPT paper runner: scheduler lead-tuning variants are wired",
    )
    audit.check(
        "GPU_PREFLIGHT_LIB" in text
        and "gpu_preflight_common.sh" in text
        and 'source "${GPU_PREFLIGHT_LIB}"' in text
        and "check_gpu_free" in text
        and "GPU availability preflight failed" in text
        and before(start_server_window, 'apply_variant_env "${variant}"', "check_gpu_free")
        and before(start_server_window, "check_gpu_free", "setsid env"),
        "ShareGPT paper runner: GPU free-memory preflight/wait runs before server launch",
    )
    audit.check(
        "gpu_precheck_only_enabled" in text
        and "GPU precheck-only failed" in text
        and "RUN_ARRIVAL=0 and RUN_BARGE_IN=0; nothing to run" in text,
        "ShareGPT paper runner: explicit GPU precheck/no-workload modes are wired",
    )
    audit.check(
        "SERVER_START_ATTEMPTS" in text
        and "SERVER_START_RETRY_SLEEP" in text
        and "start_server_once()" in text
        and "retrying server start" in text,
        "ShareGPT paper runner: transient server-start failures are retried",
    )
    audit.check(
        "SERVER_READY_ENDPOINT" in text
        and "SERVER_READY_TIMEOUT_SEC" in text
        and "/v1/models" in text
        and "server_ready()" in text
        and "server lost readiness before workload" in text
        and "StageEngineCoreProc died during READY" in text,
        "ShareGPT paper runner: readiness waits for model endpoint and stage failures",
    )
    audit.check(
        "CASE_WALL_TIMEOUT_SEC" in text
        and "STOP_ON_CASE_FAILURE" in text
        and 'timeout --preserve-status "${CASE_WALL_TIMEOUT_SEC}"' in text,
        "ShareGPT paper runner: case timeout and fail-fast controls are wired",
    )
    audit.check(
        "CASE_MIN_OK" in text
        and "CASE_MAX_ERRORS" in text
        and "summary gate" in text
        and "completed_ok" in text,
        "ShareGPT paper runner: summary success gate is wired",
    )
    audit.check(
        "SKIP_VALID_CASES" in text
        and "skip_valid_cases_enabled()" in text
        and "variant_all_requested_cases_valid()" in text
        and "all requested cases already satisfy summary gate" in text
        and "existing summary gate" in text,
        "ShareGPT paper runner: valid-case resume skip is wired",
    )
    audit.check(
        "RESTART_BETWEEN_WORKLOAD_GROUPS" in text
        and "restarting ${variant} server between arrival and barge-in groups" in text
        and 'if [[ "${RUN_ARRIVAL}" == "1" && "${RESTART_BETWEEN_WORKLOAD_GROUPS}" == "1" ]]' in text,
        "ShareGPT paper runner: can restart between arrival and barge-in groups",
    )
    audit.check(
        "SERVER_DIR_SUFFIX" in text
        and "server_dir_for_label()" in text
        and "SERVER_LOG_FILE" in text
        and "_server_${SERVER_DIR_SUFFIX}" in text,
        "ShareGPT paper runner: split-phase server logs can be kept separate",
    )
    audit.check(
        'server_log_has_fatal_startup "${log_file}"' in text
        and "ValueError: Free memory" in text
        and "subprocess died unexpectedly" in text,
        "ShareGPT paper runner: startup fatal log detection is wired",
    )
    audit.check(
        "BARGE_IN_AFTER_DISTRIBUTION" in text
        and "--barge-in-after-distribution" in text
        and "--barge-in-after-samples" in text
        and "BARGE_IN_AFTER_SAMPLE_KEY" in text,
        "ShareGPT paper runner: sampled barge-in timing controls are wired",
    )
    audit.check(
        'BARGE_IN_PROBS="${BARGE_IN_PROBS:-0.0 0.3 0.7 1.0}"' in text,
        "ShareGPT paper runner: default barge-in grid follows paper setting",
    )
    audit.check(
        "BARGE_IN_AFTER_SAMPLE_TIME_SCALE" in text
        and "--barge-in-after-sample-time-scale" in text,
        "ShareGPT paper runner: sampled barge-in time-scale control is wired",
    )
    audit.check(
        "POSTHOC_PLAYBACK_BARGE_IN_WASTE" in text
        and "--posthoc-playback-barge-in-waste" in text,
        "ShareGPT paper runner: posthoc playback-waste control is wired",
    )
    audit.check(
        "run_duration_probe()" in text
        and "prepare_barge_in_samples()" in text
        and 'BARGE_IN_AFTER_SAMPLES}" == "auto"' in text
        and '"_duration_probe"' in text,
        "ShareGPT paper runner: auto duration probe for sampled barge-in is wired",
    )
    audit.check(
        "DURATION_PROBE_REUSE_FIRST_VARIANT" in text
        and "REUSED_PROBE_SERVER_VARIANT" in text
        and 'if [[ "${RUN_ARRIVAL}" == "1" ]]; then' in text
        and 'if [[ "${RUN_BARGE_IN}" == "1" ]]; then' in text
        and "reusing duration-probe server" in text
        and 'case_name="_duration_probe/${DURATION_PROBE_CASE}"' in text,
        "ShareGPT paper runner: auto duration probe can reuse the first variant server",
    )
    audit.check(
        near(
            text,
            'local burst_manifest=""',
            ['if [[ "${RUN_ARRIVAL}" == "1" ]]; then', "generate_burst_manifest", "run_variant"],
            span=1200,
        ),
        "ShareGPT paper runner: BurstGPT manifest generation is arrival-only",
    )
    audit.check(
        'BURST_SCALE_MODE="${BURST_SCALE_MODE:-target_peak}"' in text
        and 'scale_args=(--target-peak-rps "${BURST_TARGET_RPS}")' in text,
        "ShareGPT paper runner: BurstGPT default uses paper-aligned peak-rate scaling",
    )


def check_all_workload_runner(audit: Audit, text: str) -> None:
    audit.check(
        'INTERACTIVE_TIME_START="${INTERACTIVE_TIME_START:-990}"' in text
        and 'INTERACTIVE_TIME_LIMIT="${INTERACTIVE_TIME_LIMIT:-1020}"' in text,
        "all-workload runner: default interactive trace window is non-empty",
    )
    audit.check(
        "is_g1_variant()" in text
        and '[[ "${variant}" == "liveserve_g1"'
        and '[[ "${variant}" == "service_aware" || "${variant}" == adaptive*'
        and 'is_g1_variant "${variant}"' in text,
        "all-workload runner: liveserve_g1 is a first-class variant",
    )
    audit.check(
        near(
            text,
            'if is_g1_variant "${variant}"; then',
            [
                'apply_scheduler_policy_preset "${LIVE_SCHEDULER_POLICY_PRESET}"',
                'apply_scheduler_policy_overlays "${LIVE_SCHEDULER_POLICY_OVERLAYS}"',
            ],
            span=500,
        ),
        "all-workload runner: liveserve_g1 applies scheduler policy",
    )
    audit.check(
        near(
            text,
            'OMNI_PREFETCH_ADMISSION=${OMNI_PREFETCH_ADMISSION:-1}',
            [
                "OMNI_PREFETCH_FAST_BYPASS",
                "OMNI_SCHED_NATIVE_PREFETCH",
                "OMNI_KV_EVICT_ENFORCE_POLICY=${LIVE_G1_EVICT_POLICY:-expected_gap_cost_aware}",
                "OMNI_KV_EVICT_INDEX_MAX_BLOCKS_PER_REQUEST=${LIVE_G1_EVICT_INDEX_TAIL:-64}",
                "OMNI_KV_EVICT_INDEX_GROUP_BY_PREFIX=${LIVE_G1_EVICT_INDEX_GROUP_BY_PREFIX:-1}",
            ],
            span=5200,
        ),
        "all-workload runner: liveserve_g1 enables native preload and G1 eviction",
    )
    audit.check(
        near(
            text,
            "interactive_preload_args=(",
            [
                "--preload-mode omni",
                "--preload-lead-ms",
                "--preload-same-time-request-first",
            ],
            span=900,
        )
        and near(
            text,
            "streamingbench_preload_args=(",
            [
                "--enable-kv-preload",
                "--preload-scope",
                "--preload-max-tokens",
            ],
            span=900,
        )
        and 'PREFETCH_MODE="${mix_preload_mode}"' in text,
        "all-workload runner: liveserve_g1 passes preload hints to clients",
    )
    audit.check(
        "OMNI_PREFETCH_ADMISSION" in text
        and "OMNI_KV_EVICT_ENFORCE_POLICY" in text
        and "unset_args=(" in text
        and "-u OMNI_PREFETCH_ADMISSION" in text
        and "-u OMNI_KV_EVICT_ENFORCE_POLICY" in text,
        "all-workload runner: baseline clears inherited preload/eviction env",
    )
    audit.check(
        '_runtime_env.txt"' in text and 'printf \'%s\\n\' "${env_args[@]}"' in text,
        "all-workload runner: runtime env snapshot is written for audit",
    )
    audit.check(
        "RUN_SERVICES_SEQUENTIALLY" in text
        and "running text and video/mix sequentially" in text
        and near(
            text,
            'RUN_SERVICES_SEQUENTIALLY}" == "1"',
            [
                'start_service "${variant}" text',
                'run_text_workloads "${variant}"',
                "stop_services",
                'start_service "${variant}" video',
                'run_video_mix_workloads "${variant}"',
            ],
            span=3000,
        ),
        "all-workload runner: GPU-lane sequential text/video mode is wired",
    )


def check_serving_chat_interrupt_abort(audit: Audit, text: str) -> None:
    audio_stream_idx = text.find('elif final_output_type == "audio":')
    audio_stream_window = (
        "" if audio_stream_idx < 0 else text[audio_stream_idx : audio_stream_idx + 900]
    )
    cancel_idx = text.find("except asyncio.CancelledError:")
    cancel_window = "" if cancel_idx < 0 else text[cancel_idx : cancel_idx + 1200]
    create_audio_idx = text.find("def _create_audio_choice(")
    create_audio_window = (
        "" if create_audio_idx < 0 else text[create_audio_idx : create_audio_idx + 1800]
    )

    audit.check(
        "choices_data = self._create_audio_choice" in audio_stream_window
        and "if not choices_data:" in audio_stream_window
        and "continue" in audio_stream_window,
        "serving_chat: streaming audio skips empty chunks",
    )
    audit.check(
        "await self.engine_client.abort(request_id)" in cancel_window
        and "return" in cancel_window,
        "serving_chat: client disconnect is request-local abort",
    )
    audit.check(
        "audio_chunks = [" in create_audio_window
        and "candidate is not None" in create_audio_window
        and "candidate.numel() > 0" in create_audio_window
        and "if stream:" in create_audio_window
        and "return []" in create_audio_window
        and 'raise ValueError("Audio generation completed but no audio was produced.")'
        in create_audio_window,
        "serving_chat: empty audio is ignored only for streaming abort path",
    )


def validate(root: Path) -> Audit:
    audit = Audit()

    sharegpt = read(root, "profiling/scheduler_analysis/scripts/run_sharegpt_origin_cbound_sweep.sh")
    sharegpt_driver = read(root, "benchmarks/sharegpt/H200/sharegpt_driver.py")
    sharegpt_paper = read(root, "benchmarks/sharegpt/H200/run_paper_workloads_gpu67.sh")
    serving_chat = read(root, "vllm-omni/vllm_omni/entrypoints/openai/serving_chat.py")
    retained = read(root, "profiling/scheduler_analysis/scripts/run_retained_cbound_sweep.sh")
    all_workload = read(root, "profiling/scheduler_analysis/scripts/run_all_workloads_kv_profile_smoke.sh")
    streaming = read(root, "benchmarks/streamingbench/run_paper_video_matrix.sh")
    evict_common = read(root, "profiling/scheduler_analysis/scripts/eviction_policy_common.sh")
    gpu_common = read(root, "profiling/scheduler_analysis/scripts/gpu_preflight_common.sh")

    check_gpu_preflight_helper(audit, gpu_common)
    check_validation_env_defaults(audit, root)
    check_rename_contract(audit, root)
    check_scheduler_config_module(audit, root)
    check_public_wrappers(audit, root)
    check_common_variant_script(audit, "ShareGPT runner", sharegpt)
    audit.check(
        "MAINTAINED_BARGE_IN_BENCH" in sharegpt
        and "--arrival-process open_c_bound" in sharegpt
        and "--run-duration-sec" in sharegpt
        and "BARGE_IN_BENCH is not readable" in sharegpt
        and sharegpt.find("check_barge_in_bench") < sharegpt.find("local variant\n  for variant in ${VARIANTS}"),
        "ShareGPT origin runner: maintained open-c-bound driver is default and archived entrypoint is optional",
    )
    audit.check(
        'DEFAULT_TEXT_GPUS="0,1,2,3,4,5,6,7"' in sharegpt
        and 'DEFAULT_NO_OFFLOAD_DEPLOY="${ROOT}/deploy/H200/qwen3_omni_moe_4thinker_4talker_8gpu_standardcap.yaml"' in sharegpt
        and 'VARIANTS="${VARIANTS:-baseline_no_offload scheduler_legacy_l12}"' in sharegpt
        and '--baseline-variant "${PLOT_BASELINE_VARIANT}"' in sharegpt,
        "ShareGPT origin runner: paper-main default compares scheduler-only against 8GPU no-offload baseline",
    )
    audit.check(
        '"open_c_bound"' in sharegpt_driver
        and "--run-duration-sec" in sharegpt_driver
        and "counted_in_measurement" in sharegpt_driver
        and "rows_counted" in sharegpt_driver
        and "actual_client_wall_time_s" in sharegpt_driver,
        "ShareGPT driver: maintained open-c-bound measurement mode is wired",
    )
    check_sharegpt_paper_runner(audit, sharegpt_paper)
    check_serving_chat_interrupt_abort(audit, serving_chat)
    check_common_variant_script(audit, "retained runner", retained)
    check_all_workload_runner(audit, all_workload)

    audit.check(
        near(
            retained,
            "liveserve_scheduler|scheduler_legacy_l12)",
            [
                "LIVE_SCHEDULER_POLICY_PRESET",
                "apply_prefetch_off_env",
                "apply_lru_env",
            ],
        ),
        "retained runner: liveserve_scheduler is clean scheduler-only",
    )
    audit.check(
        near(
            retained,
            "preload_args_for_variant()",
            [
                "baseline_no_offload",
                "liveserve_scheduler",
                "liveserve_g1_no_preload",
                'printf \'%s\\n\' "--preload-mode" "none"',
            ],
            span=1400,
        ),
        "retained runner: non-preload variants pass --preload-mode none",
    )

    audit.check(
        near(
            streaming,
            "is_joint_variant()",
            ["liveserve_g1|liveserve_allopt", "return 0"],
        )
        and near(
            streaming,
            "evict_profile_for_variant()",
            ["liveserve_g1|liveserve_allopt", "printf '%s' g1"],
        ),
        "StreamingBench runner: liveserve_g1 routes to joint launcher with G1 profile",
    )
    audit.check(
        "start_server_2gpu_video_baseline.sh" in streaming
        and "start_server_2gpu_video_scheduler.sh" in streaming
        and 'LIVE_EVICT_PROFILE="$(evict_profile_for_variant "${variant}")"' in streaming,
        "StreamingBench runner: baseline, scheduler-only and joint launchers are distinct",
    )

    audit.check(
        "LIVE_G1_EVICT_POLICY:-expected_gap_cost_aware" in evict_common
        and "LIVE_G1_EVICT_INDEX_TAIL:-64" in evict_common
        and "LIVE_G1_EVICT_MAX_CACHED:-16" in evict_common
        and "LIVE_G1_EVICT_INDEX_GROUP_BY_PREFIX:-1" in evict_common,
        "G1 shared eviction profile uses expected policy, heap tail and group victims",
    )

    video_common = read(root, "scripts/H200/video_env_common.sh")
    audit.check(
        all(
            token in video_common
            for token in [
                "liveserve_clear_bp_runtime_env",
                "liveserve_clear_prefetch_env",
                "liveserve_clear_eviction_env",
                "liveserve_clear_video_optimization_env",
            ]
        ),
        "video env helper defines scheduler/preload/eviction cleanup",
    )

    launcher_paths = [
        "scripts/H200/start_server_2gpu_video_baseline.sh",
        "scripts/H200/start_server_2gpu_video_scheduler.sh",
        "scripts/H200/start_server_3gpu_video_baseline.sh",
        "scripts/H200/start_server_3gpu_video_scheduler.sh",
        "scripts/H200/start_server_3gpu_video_2thinker.sh",
    ]
    for rel in launcher_paths:
        text = read(root, rel)
        audit.check(
            "video_env_common.sh" in text
            and "liveserve_clear_video_optimization_env" in text,
            f"{rel}: clears inherited optimization env",
        )
        audit.check(
            'VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"' in text,
            f"{rel}: disables vLLM usage reporter by default",
        )

    for rel in [
        "scripts/H200/start_server_2gpu_video_scheduler.sh",
        "scripts/H200/start_server_3gpu_video_scheduler.sh",
    ]:
        text = read(root, rel)
        audit.check(
            before(text, "liveserve_clear_video_optimization_env", "scheduler_policy_apply_and_export")
            and 'scheduler_policy_apply_and_export "${SCHEDULER_POLICY_PRESET:-scheduler}" "${SCHEDULER_POLICY_OVERLAYS:-talkerlead=12:6:75}"'
            in text,
            f"{rel}: applies scheduler policy after cleanup",
        )
        audit.check(
            "OMNI_PREFETCH_" not in text and "OMNI_KV_EVICT_" not in text,
            f"{rel}: scheduler-only launcher does not set preload or eviction env",
        )

    joint = read(root, "scripts/H200/start_server_2gpu_video_joint.sh")
    audit.check(
        'scheduler_policy_apply_and_export \\'
        in joint
        and 'OMNI_PREFETCH_FAST_BYPASS="${OMNI_PREFETCH_FAST_BYPASS:-1}"' in joint
        and 'OMNI_SCHED_NATIVE_PREFETCH="${OMNI_SCHED_NATIVE_PREFETCH:-1}"' in joint
        and "eviction_policy_apply_g1_env" in joint,
        "video joint launcher enables scheduler, native preload and G1 eviction",
    )
    audit.check(
        'VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"' in joint,
        "video joint launcher disables vLLM usage reporter by default",
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
