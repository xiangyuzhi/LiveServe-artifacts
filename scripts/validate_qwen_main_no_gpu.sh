#!/usr/bin/env bash
# Run the current no-GPU validation gate for the H200/Qwen main branch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

source "${ROOT}/scripts/env_defaults.sh"
PYBIN="$(liveserve_resolve_pybin "${ROOT}")"
liveserve_require_command "${PYBIN}" "python"
liveserve_require_command rg "ripgrep (required by source checks)"
liveserve_export_pycache_prefix

if [ "${LIVESERVE_VALIDATE_EXTERNAL_ARTIFACTS:-0}" = "0" ]; then
  export SKIP_QWEN_ARTIFACT_VALIDATION="${SKIP_QWEN_ARTIFACT_VALIDATION:-1}"
fi

run() {
  printf '\n[validate-no-gpu] +'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

if [ "${SKIP_QWEN_ARTIFACT_VALIDATION:-0}" != "1" ]; then
  run "${PYBIN}" scripts/validate_qwen_system_artifacts.py
else
  echo "[validate-no-gpu] skipping artifact validation by SKIP_QWEN_ARTIFACT_VALIDATION=1"
fi
run "${PYBIN}" scripts/validate_qwen_runner_wiring.py
run "${PYBIN}" scripts/validate_sharegpt_paper_workloads.py
run "${PYBIN}" scripts/validate_repo_hygiene.py
run "${PYBIN}" scripts/validate_markdown_links.py
run "${PYBIN}" scripts/validate_final_reproduction_matrix.py

shell_scripts=(
  scripts/env_defaults.sh
  scripts/setup_env.sh
  scripts/prepare_sharegpt.sh
  benchmarks/sharegpt/H200/sharegpt_benchmark.sh
  benchmarks/sharegpt/H200/run_paper_workloads_gpu67.sh
  benchmarks/streamingbench/run_paper_video_matrix.sh
  benchmarks/interactive/H200/scheduler_run_one_cell.sh
  benchmarks/interactive/H200/step1_bargein_chain_a.sh
  benchmarks/interactive/H200/step1_bargein_chain_b.sh
  benchmarks/interactive/H200/step2_scheduler_chain.sh
  benchmarks/interactive/H200/step2_scheduler_chain_prob.sh
  benchmarks/interactive/H200/run_bargein3gpu_matrix.sh
  profiling/scheduler_analysis/scripts/run_sharegpt_origin_cbound_sweep.sh
  profiling/scheduler_analysis/scripts/run_retained_cbound_sweep.sh
  profiling/scheduler_analysis/scripts/run_all_workloads_kv_profile_smoke.sh
  profiling/scheduler_analysis/scripts/gpu_preflight_common.sh
  profiling/scheduler_analysis/scripts/scheduler_policy_common.sh
  profiling/scheduler_analysis/scripts/eviction_policy_common.sh
  profiling/kv_offload_prefetch/large_scale/run_4gpu_scheduler_preload.sh
  profiling/kv_offload_prefetch/large_scale/run_4gpu_scheduler_preload_eviction.sh
  profiling/kv_offload_prefetch/scripts/run_hbm_growth_4cell.sh
  profiling/kv_offload_prefetch/kv_usage_probe/scripts/run_kv_usage_4cell.sh
  profiling/kv_offload_prefetch/session_kv_probe/scripts/run_session_kv_probe_4cell.sh
  profiling/kv_offload_prefetch/prefix_shape_pad/scripts/run_prefix_shape_pad_3cell.sh
  profiling/kv_offload_prefetch/h_states_probe/run_probe_4cell.sh
  profiling/kv_offload_prefetch/multiturn_kv_reuse/driver/run_smoke_2cell.sh
  scripts/H200/video_env_common.sh
  scripts/H200/start_server_2gpu_scheduler.sh
  scripts/H200/start_server_2gpu_video_baseline.sh
  scripts/H200/start_server_2gpu_video_scheduler.sh
  scripts/H200/start_server_2gpu_video_joint.sh
  scripts/H200/start_server_3gpu_video_baseline.sh
  scripts/H200/start_server_3gpu_video_scheduler.sh
  scripts/H200/start_server_3gpu_video_2thinker.sh
  scripts/H200/reproduce_interactive_scheduler.sh
  scripts/H200/reproduce_sharegpt_scheduler.sh
)
for script in "${shell_scripts[@]}"; do
  run bash -n "${script}"
done

run bash -c '
source scripts/H200/video_env_common.sh
export OMNI_TALKER_LEAD_MAX=50
export OMNI_THINKER_LEAD_MAX=24
export OMNI_BP_KV_BYTES_PER_TOKEN=1
export OMNI_PREFETCH_ADMISSION=1
export OMNI_SCHED_NATIVE_PREFETCH=1
export OMNI_KV_EVICT_ENFORCE_POLICY=g1
export OMNI_KV_LRU_STATS=1
export OMNI_PREFIX_SHAPE_PAD=1
liveserve_clear_video_optimization_env
remaining="$(env | LC_ALL=C sort | rg "^(OMNI_TALKER|OMNI_THINKER|OMNI_BP_|OMNI_PREFETCH|OMNI_SCHED_NATIVE_PREFETCH|OMNI_KV_EVICT|OMNI_KV_LRU|OMNI_PREFIX_SHAPE_PAD)" || true)"
if [ "${remaining}" != "OMNI_PREFIX_SHAPE_PAD=1" ]; then
  echo "[validate-no-gpu] unexpected optimization env after cleanup:" >&2
  printf "%s\n" "${remaining}" >&2
  exit 1
fi
echo "PASS video_env_common cleanup leaves only OMNI_PREFIX_SHAPE_PAD"
'

run bash -c '
if rg -n "\[BP adaptive\]|BP adaptive|bp adaptive|BPv3|bpv3|BP-v3|OLBP|olbp" \
  vllm-omni/vllm_omni benchmarks docs scripts profiling deploy README.md \
  -g "!scripts/validate_qwen_main_no_gpu.sh" -S; then
  echo "[validate-no-gpu] stale scheduler/BP naming found" >&2
  exit 1
fi
echo "PASS no stale scheduler/BP user-facing naming"
'

python_files=(
  scripts/diff_summary.py
  scripts/validate_qwen_system_artifacts.py
  scripts/validate_qwen_runner_wiring.py
  scripts/validate_sharegpt_paper_workloads.py
  scripts/validate_repo_hygiene.py
  scripts/validate_markdown_links.py
  scripts/validate_final_reproduction_matrix.py
  scripts/validate_reduced_smoke_artifacts.py
  benchmarks/sharegpt/H200/download_dataset.py
  benchmarks/sharegpt/H200/generate_burstgpt_arrivals.py
  benchmarks/sharegpt/H200/single_probe.py
  benchmarks/sharegpt/H200/sharegpt_driver.py
  benchmarks/sharegpt/H200/summarize_paper_workloads.py
  benchmarks/sharegpt/H200/verify_dataset.py
  benchmarks/interactive/H200/plot_step1.py
  benchmarks/interactive/H200/client_bargein.py
  benchmarks/interactive/H200/aggregate_bargein3gpu.py
  benchmarks/interactive/H200/plot_bargein3gpu_matrix.py
  benchmarks/sharegpt/H200/bargein/plot_b2_cl_bargein.py
  vllm-omni/vllm_omni/core/sched/lead_governor_config.py
  vllm-omni/vllm_omni/core/sched/omni_scheduler_mixin.py
  vllm-omni/vllm_omni/core/sched/omni_ar_scheduler.py
  vllm-omni/vllm_omni/core/sched/omni_generation_scheduler.py
  vllm-omni/vllm_omni/core/sched/talker_lead_governor.py
  vllm-omni/vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py
  vllm-omni/vllm_omni/entrypoints/openai/serving_chat.py
  vllm-omni/vllm_omni/metrics/stats.py
  benchmarks/streamingbench/streamingbench_driver.py
  benchmarks/streamingbench/summarize_paper_video.py
  profiling/mixed_text_video/workload/generate_manifest.py
  profiling/mixed_text_video/workload/mixed_replay_driver.py
  profiling/kv_offload_prefetch/sharegpt_bargein/scripts/plot_sharegpt_bargein.py
  profiling/kv_offload_prefetch/latency_breakdown/driver/profile_multiturn_target.py
  profiling/kv_offload_prefetch/h_states_probe/driver/smoke_3turn.py
  profiling/kv_offload_prefetch/prefix_shape_pad/driver/smoke_multiturn.py
  profiling/scheduler_analysis/scripts/analyze_kv_pressure_windows.py
  profiling/scheduler_analysis/scripts/plot_retained_cbound_qps_ttfp.py
  profiling/scheduler_analysis/scripts/plot_sharegpt_cbound_ttfp_qps.py
)
run "${PYBIN}" -m py_compile "${python_files[@]}"

run git -C vllm-omni diff --check
run "${PYBIN}" - <<'PY'
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

root = Path("vllm-omni")
changed = subprocess.check_output(
    ["git", "-C", str(root), "diff", "--name-only", "HEAD", "--"],
    text=True,
)
untracked = subprocess.check_output(
    ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
    text=True,
)
failures: list[str] = []
for rel in sorted(
    set(line for line in (changed.splitlines() + untracked.splitlines()) if line)
):
    path = root / rel
    if not path.is_file():
        continue
    data = path.read_bytes()
    if b"\0" in data:
        continue
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        continue
    if data and not data.endswith(b"\n"):
        failures.append(f"{rel}: missing final newline")
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.rstrip(" \t") != line:
            failures.append(f"{rel}:{lineno}: trailing whitespace")
            break
if failures:
    print("[validate-no-gpu] vllm-omni changed-file hygiene failed:", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    raise SystemExit(1)
print("PASS vllm-omni changed/untracked text hygiene")
PY

run git diff --check

echo
echo "[validate-no-gpu] PASS"
