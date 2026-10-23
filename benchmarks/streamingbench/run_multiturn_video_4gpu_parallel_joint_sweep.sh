#!/usr/bin/env bash
# Multi-turn video sweep on two parallel 2GPU lanes.
#
# Default layout:
#   phase 1:
#     baseline_no_offload       -> GPUs 4,5 on port 8777
#     liveserve_g1              -> GPUs 6,7 on port 8778
#   phase 2:
#     baseline                  -> GPUs 4,5 on port 8777
#
# The joint lane enables the LiveServe scheduler, KV preload, and G1 eviction.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

DEFAULT_METADATA="/workspace/xyz/test/video/StreamingBench/src/data/questions_real_stream.json"
if [ ! -f "${DEFAULT_METADATA}" ]; then
  DEFAULT_METADATA="${ROOT}/example_questions.json"
fi

export RESULT_BASE="${RESULT_BASE:-${REPO_ROOT}/results/video/multiturn_video_joint_4gpu_parallel_${STAMP}}"
export STREAMINGBENCH_METADATA="${STREAMINGBENCH_METADATA:-${DEFAULT_METADATA}}"
export STREAMINGBENCH_DATA_ROOT="${STREAMINGBENCH_DATA_ROOT:-/workspace/xyz/test/video/StreamingBench/src/data}"

export GPU_COUNTS="2"
export MODES="${MODES:-bargein}"
export CONCURRENCIES_2GPU="${CONCURRENCIES_2GPU:-1 2 4 6}"
export LIVESERVE_ALLOW_GPU_0_2="${LIVESERVE_ALLOW_GPU_0_2:-0}"

export FIXED_REPEAT="${FIXED_REPEAT:-1}"
export MAX_VIDEOS="${MAX_VIDEOS:-16}"
export MAX_QUESTIONS_PER_VIDEO="${MAX_QUESTIONS_PER_VIDEO:-3}"
export STREAMINGBENCH_FOLLOWUP_TURNS_PER_QUESTION="${STREAMINGBENCH_FOLLOWUP_TURNS_PER_QUESTION:-1}"
export STREAMINGBENCH_FOLLOWUP_PROFILE="${STREAMINGBENCH_FOLLOWUP_PROFILE:-varied}"

export STREAMINGBENCH_WITH_AUDIO_OUTPUT="${STREAMINGBENCH_WITH_AUDIO_OUTPUT:-1}"
export STREAMINGBENCH_PROMPT_STYLE="${STREAMINGBENCH_PROMPT_STYLE:-detailed_answer}"
export STREAMINGBENCH_DIALOGUE_MODE="${STREAMINGBENCH_DIALOGUE_MODE:-multiturn}"
export STREAMINGBENCH_DIALOGUE_HISTORY_TURNS="${STREAMINGBENCH_DIALOGUE_HISTORY_TURNS:-3}"
export STREAMINGBENCH_DIALOGUE_HISTORY_ANSWER_CHARS="${STREAMINGBENCH_DIALOGUE_HISTORY_ANSWER_CHARS:-700}"

export STREAMINGBENCH_MAX_TOKENS="${STREAMINGBENCH_MAX_TOKENS:-512}"
export STREAMINGBENCH_FPS="${STREAMINGBENCH_FPS:-2}"
export STREAMINGBENCH_FRAMES_PER_QUERY="${STREAMINGBENCH_FRAMES_PER_QUERY:-32}"
export STREAMINGBENCH_NUM_SAMPLE_FRAMES="${STREAMINGBENCH_NUM_SAMPLE_FRAMES:-16}"
export STREAMINGBENCH_MAX_BUFFER_FRAMES="${STREAMINGBENCH_MAX_BUFFER_FRAMES:-96}"

export STREAMINGBENCH_BARGE_IN_TIME_ANCHOR="${STREAMINGBENCH_BARGE_IN_TIME_ANCHOR:-audio}"
export BARGE_IN_AFTER_SECONDS="${BARGE_IN_AFTER_SECONDS:-0.6}"
export BARGE_IN_JITTER_SEC="${BARGE_IN_JITTER_SEC:-0.0}"
export BARGE_IN_PROB="${BARGE_IN_PROB:-1.0}"
export STREAMINGBENCH_PRELOAD_LEAD_MS="${STREAMINGBENCH_PRELOAD_LEAD_MS:-${STREAMINGBENCH_PREFETCH_LEAD_MS:-5000}}"
export TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"
export POST_TEXT_AUDIO_GRACE_SEC="${POST_TEXT_AUDIO_GRACE_SEC:-90}"
export POST_BOOT_SLEEP_SEC="${POST_BOOT_SLEEP_SEC:-10}"
export POST_STOP_SLEEP_SEC="${POST_STOP_SLEEP_SEC:-10}"
export ENABLE_GPU_DMON="${ENABLE_GPU_DMON:-1}"
export GPU_DMON_INTERVAL_SEC="${GPU_DMON_INTERVAL_SEC:-1}"
export CONTINUE_ON_DRIVER_FAILURE="${CONTINUE_ON_DRIVER_FAILURE:-1}"
export RESTART_SERVER_ON_DRIVER_FAILURE="${RESTART_SERVER_ON_DRIVER_FAILURE:-1}"

export LIVESERVE_DEPLOY_YAML_2GPU="${LIVESERVE_DEPLOY_YAML_2GPU:-${REPO_ROOT}/profiling/kv_offload_prefetch/large_scale/deploy/qwen3_omni_moe_2gpu_chunk8_prefix_on_offloading.yaml}"
export LIVESERVE_DEPLOY_YAML_NO_OFFLOAD_2GPU="${LIVESERVE_DEPLOY_YAML_NO_OFFLOAD_2GPU:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_multi_replicas_2gpu_chunk8.yaml}"
unset LIVESERVE_DEPLOY_YAML

mkdir -p "${RESULT_BASE}"

echo "[video-joint-4gpu] RESULT_BASE=${RESULT_BASE}"
echo "[video-joint-4gpu] metadata=${STREAMINGBENCH_METADATA}"
echo "[video-joint-4gpu] workload=${MAX_VIDEOS} videos x ${MAX_QUESTIONS_PER_VIDEO} questions followups=${STREAMINGBENCH_FOLLOWUP_TURNS_PER_QUESTION}"
echo "[video-joint-4gpu] concurrencies=${CONCURRENCIES_2GPU} prompt=${STREAMINGBENCH_PROMPT_STYLE}"
echo "[video-joint-4gpu] preload_lead_ms=${STREAMINGBENCH_PRELOAD_LEAD_MS}"
echo "[video-joint-4gpu] baseline no-offload lane=${BASELINE_GPU_DEVICES:-4,5} port=${BASELINE_PORT:-8777}"
echo "[video-joint-4gpu] baseline offload lane=${BASELINE_OFFLOAD_GPU_DEVICES:-${BASELINE_GPU_DEVICES:-4,5}} port=${BASELINE_OFFLOAD_PORT:-${BASELINE_PORT:-8777}}"
echo "[video-joint-4gpu] joint lane=${JOINT_GPU_DEVICES:-6,7} port=${JOINT_PORT:-8778}"
echo "[video-joint-4gpu] offload deploy=${LIVESERVE_DEPLOY_YAML_2GPU}"
echo "[video-joint-4gpu] no-offload deploy=${LIVESERVE_DEPLOY_YAML_NO_OFFLOAD_2GPU}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[video-joint-4gpu] DRY_RUN=1; not starting servers"
  exit 0
fi

pids=()

cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

clear_joint_env() {
  unset SCHEDULER_POLICY_PRESET SCHEDULER_POLICY_OVERLAYS
  unset OMNI_PREFETCH_FAST_BYPASS OMNI_PREFETCH_SKIP_OFFLOAD_ON_MISS OMNI_PREFETCH_HISTORY_HIT_ENABLE
  unset OMNI_PREFETCH_REGISTRY_TTL_SEC OMNI_PREFETCH_STREAM_TIMEOUT_MS OMNI_PREFETCH_GPU_PIN
  unset OMNI_PREFETCH_GPU_PIN_TTL_MS OMNI_PREFETCH_GPU_PIN_MAX_BLOCKS OMNI_PREFETCH_PRIORITY
  unset OMNI_PREFETCH_ADMISSION OMNI_SCHED_NATIVE_PREFETCH OMNI_SCHED_NATIVE_PREFETCH_CANCEL
  unset OMNI_KV_EVICT_SHADOW OMNI_KV_EVICT_ENFORCE_POLICY OMNI_KV_EVICT_ENFORCE_INDEX
  unset OMNI_KV_EVICT_ENFORCE_INDEX_FALLBACK_SCAN OMNI_KV_EVICT_INDEX_MAX_BLOCKS_PER_REQUEST
  unset OMNI_KV_EVICT_ENFORCE_MIN_CACHED_VICTIMS_PER_ALLOC OMNI_KV_EVICT_ENFORCE_MAX_CACHED_VICTIMS_PER_ALLOC
  unset STREAMINGBENCH_ENABLE_KV_METADATA STREAMINGBENCH_ENABLE_KV_PRELOAD STREAMINGBENCH_ENABLE_KV_PREFETCH
}

launch_lane() {
  local label="$1"
  local variant="$2"
  local devices="$3"
  local port="$4"
  local log_file="${RESULT_BASE}/_lane_${label}.log"
  echo "[video-joint-4gpu] launching ${label}: variant=${variant} devices=${devices} port=${port}"
  (
    if [[ "${variant}" == "liveserve_g1" || "${variant}" == "liveserve_allopt" || "${variant}" == "joint_g1" ]]; then
      export SCHEDULER_POLICY_PRESET="${SCHEDULER_POLICY_PRESET:-scheduler}"
      export SCHEDULER_POLICY_OVERLAYS="${SCHEDULER_POLICY_OVERLAYS:-talkerlead=12:6:75}"
      export STREAMINGBENCH_ENABLE_KV_PRELOAD="${STREAMINGBENCH_ENABLE_KV_PRELOAD:-1}"
    else
      clear_joint_env
    fi
    export VARIANTS="${variant}"
    export LIVESERVE_GPU_DEVICES_2GPU="${devices}"
    export LIVESERVE_PORT="${port}"
    export REQUEST_ID_NAMESPACE="${REQUEST_ID_NAMESPACE:-${STAMP}_videojoint4gpu}_${label}"
    exec bash "${ROOT}/run_paper_video_matrix.sh"
  ) > "${log_file}" 2>&1 &
  pids+=("$!")
}

status=0
wait_lanes() {
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  pids=()
}

launch_lane "baseline_nooffload_g45" "baseline_no_offload" "${BASELINE_GPU_DEVICES:-4,5}" "${BASELINE_PORT:-8777}"
launch_lane "liveserve_g1_g67" "liveserve_g1" "${JOINT_GPU_DEVICES:-6,7}" "${JOINT_PORT:-8778}"
wait_lanes

case "${RUN_BASELINE_OFFLOAD:-1}" in
  1|true|TRUE|yes|YES|on|ON)
    launch_lane "baseline_offload_g45" "baseline" "${BASELINE_OFFLOAD_GPU_DEVICES:-${BASELINE_GPU_DEVICES:-4,5}}" "${BASELINE_OFFLOAD_PORT:-${BASELINE_PORT:-8777}}"
    wait_lanes
    ;;
esac
trap - INT TERM

"${LIVESERVE_PYBIN:-/workspace/xyz/omni_020/bin/python}" \
  "${ROOT}/summarize_paper_video.py" "${RESULT_BASE}" \
  --out-dir "${RESULT_BASE}/paper_figures" || true

if [ "${status}" != "0" ]; then
  echo "[video-joint-4gpu] one or more lanes failed; logs are under ${RESULT_BASE}/_lane_*.log" >&2
  exit "${status}"
fi

echo "[video-joint-4gpu] done: ${RESULT_BASE}"
