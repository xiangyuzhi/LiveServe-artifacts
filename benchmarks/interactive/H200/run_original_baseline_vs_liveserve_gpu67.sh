#!/usr/bin/env bash
# Baseline-vs-LiveServe interactive-original sweep on a 2-GPU pair.
#
# Workload contract:
#   - original interactive trace lengths/timing: no retained-session selection,
#     no synthetic length profile, no warm-history seeding, no prompt suffix
#   - audio mode, p_int=0, same high-concurrency capacity window used by the
#     existing H200 interactive capacity scripts
#   - all run outputs go outside the repo, under /workspace/xyz/exp_tmp_dir
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../../.." && pwd)"

TS="${TS:-$(date -u +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-/workspace/xyz/exp_tmp_dir}"
RUN_ID="${RUN_ID:-interactive_original_gpu67_${TS}}"
RESULT_ROOT="${RESULT_ROOT:-${EXP_ROOT}/${RUN_ID}}"
LOGDIR="${RESULT_ROOT}/_logs"

PYBIN="${LIVESERVE_PYBIN:-/workspace/xyz/omni_020/bin/python}"
VLLM_BIN="${LIVESERVE_VLLM_BIN:-/workspace/xyz/omni_020/bin/vllm-omni}"
MODEL="${LIVESERVE_MODEL:-/workspace/models/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695}"
SERVED_MODEL_NAME="${LIVESERVE_SERVED_MODEL_NAME:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
DRIVER="${ROOT}/trace_replay_driver.py"
SCHEDULER_POLICY_LIB="${REPO_ROOT}/profiling/scheduler_analysis/scripts/scheduler_policy_common.sh"

TEXT_GPUS="${TEXT_GPUS:-6,7}"
PORT="${PORT:-8897}"
VARIANTS="${VARIANTS:-baseline liveserve_full}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-8 16 24 32 48 64 96 128 160}"
TIME_START="${TIME_START:-6780}"
TIME_LIMIT="${TIME_LIMIT:-6840}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-300}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-1200}"
SERVER_MOE_BACKEND="${SERVER_MOE_BACKEND:-triton}"
RECORD_AUDIO_TIMELINE="${RECORD_AUDIO_TIMELINE:-0}"
ENABLE_KV_LRU_STATS="${ENABLE_KV_LRU_STATS:-1}"

BASELINE_DEPLOY_YAML="${BASELINE_DEPLOY_YAML:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_multi_replicas_2gpu.yaml}"
LIVESERVE_DEPLOY_YAML="${LIVESERVE_DEPLOY_YAML:-${REPO_ROOT}/profiling/kv_offload_prefetch/large_scale/deploy/qwen3_omni_moe_2gpu_prefix_on_offloading.yaml}"

PREFETCH_LEAD_MS="${PREFETCH_LEAD_MS:-5000}"
PREFETCH_MAX_INFLIGHT="${PREFETCH_MAX_INFLIGHT:-2}"
PREFETCH_CLIPPED_START_LIMIT="${PREFETCH_CLIPPED_START_LIMIT:-64}"
LIVE_SCHEDULER_POLICY_PRESET="${LIVE_SCHEDULER_POLICY_PRESET:-scheduler}"
LIVE_SCHEDULER_POLICY_OVERLAYS="${LIVE_SCHEDULER_POLICY_OVERLAYS:-talkerlead=12:6:75}"
LIVE_EVICT_POLICY="${LIVE_EVICT_POLICY:-response_len_tail_aware}"
LIVE_EVICT_SHADOW="${LIVE_EVICT_SHADOW:-1}"
LIVE_EVICT_INDEX="${LIVE_EVICT_INDEX:-1}"
LIVE_EVICT_INDEX_FALLBACK_SCAN="${LIVE_EVICT_INDEX_FALLBACK_SCAN:-0}"
LIVE_EVICT_INDEX_TAIL="${LIVE_EVICT_INDEX_TAIL:-24}"
LIVE_EVICT_MIN_CACHED="${LIVE_EVICT_MIN_CACHED:-1}"
LIVE_EVICT_MAX_CACHED="${LIVE_EVICT_MAX_CACHED:-1}"
LIVE_EVICT_PARTIAL_OVER_CAP="${LIVE_EVICT_PARTIAL_OVER_CAP:-0}"
LIVE_EVICT_MAX_SCAN_BLOCKS="${LIVE_EVICT_MAX_SCAN_BLOCKS:-512}"
LIVE_EVICT_CANDIDATE_FREE_RATIO="${LIVE_EVICT_CANDIDATE_FREE_RATIO:-0.95}"
LIVE_EVICT_MIN_METADATA_RATIO="${LIVE_EVICT_MIN_METADATA_RATIO:-0.75}"
LIVE_EVICT_ENFORCE_LOG="${LIVE_EVICT_ENFORCE_LOG:-0}"
LIVE_EVICT_COST_LOSS_WEIGHT="${LIVE_EVICT_COST_LOSS_WEIGHT:-1.0}"
LIVE_EVICT_NEAR_GAP_SEC="${LIVE_EVICT_NEAR_GAP_SEC:-45.0}"
LIVE_EVICT_RESPONSE_PENALTY_WEIGHT="${LIVE_EVICT_RESPONSE_PENALTY_WEIGHT:-4.0}"

SERVER_PID=""
STARTED_SERVER=0
BASE_PYTHONPATH="${PYTHONPATH:-}"

mkdir -p "${RESULT_ROOT}" "${LOGDIR}"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[gpu67-interactive $(stamp)] $*" | tee -a "${LOGDIR}/run.log"; }

fatal() {
  log "FATAL: $*"
  exit 2
}

require_two_gpu_pair() {
  case "${TEXT_GPUS}" in
    [0-9],[0-9]) ;;
    *) fatal "TEXT_GPUS must be a two-GPU pair like 6,7; got '${TEXT_GPUS}'" ;;
  esac
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "${TEXT_GPUS}" ]]; then
    fatal "CUDA_VISIBLE_DEVICES is '${CUDA_VISIBLE_DEVICES}', expected '${TEXT_GPUS}'"
  fi
}

health_url() {
  printf 'http://127.0.0.1:%s/health' "${PORT}"
}

api_url() {
  printf 'http://127.0.0.1:%s/v1/chat/completions' "${PORT}"
}

is_healthy() {
  [[ "$(curl -s -o /dev/null -w '%{http_code}' -m 2 "$(health_url)" 2>/dev/null || true)" == "200" ]]
}

stop_server() {
  if [[ "${STARTED_SERVER}" -eq 1 && -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    log "stopping server pid=${SERVER_PID}"
    kill -TERM "-${SERVER_PID}" >/dev/null 2>&1 || kill -TERM "${SERVER_PID}" >/dev/null 2>&1 || true
    for _ in $(seq 1 90); do
      if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        log "server stopped"
        SERVER_PID=""
        STARTED_SERVER=0
        return
      fi
      sleep 1
    done
    log "server still alive after 90s; killing process group"
    kill -KILL "-${SERVER_PID}" >/dev/null 2>&1 || kill -KILL "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  SERVER_PID=""
  STARTED_SERVER=0
}

cleanup() {
  stop_server
}
trap cleanup EXIT

reset_runtime_env() {
  local name
  while IFS= read -r name; do
    unset "${name}" || true
  done < <(compgen -v OMNI_)
  while IFS= read -r name; do
    unset "${name}" || true
  done < <(compgen -v SCHEDULER_POLICY_)
}

apply_common_env() {
  local variant="$1"
  export CUDA_VISIBLE_DEVICES="${TEXT_GPUS}"
  export PYTHONPATH="${REPO_ROOT}/vllm-omni${BASE_PYTHONPATH:+:${BASE_PYTHONPATH}}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
  export VLLM_VIDEO_ASYNC_CHUNK="${VLLM_VIDEO_ASYNC_CHUNK:-on}"
  export VLLM_VIDEO_AUDIO_DELTA_MODE="${VLLM_VIDEO_AUDIO_DELTA_MODE:-fast}"
  export VLLM_OMNI_STAGE_INIT_MAX_WORKERS="${VLLM_OMNI_STAGE_INIT_MAX_WORKERS:-7}"
  export MC_TRANSFER_TIMEOUT="${MC_TRANSFER_TIMEOUT:-300}"
  export OMNI_PREFIX_SHAPE_PAD=1
  export OMNI_THINKER_STAGE_IDS=0
  export OMNI_TALKER_STAGE_IDS=1
  export OMNI_BP_KV_BYTES_PER_TOKEN=98304
  export OMNI_SESSION_KV_BYTES_PER_TOKEN=98304
  if [[ "${ENABLE_KV_LRU_STATS}" == "1" ]]; then
    export OMNI_KV_LRU_STATS=1
    export OMNI_KV_LRU_STATS_PATH="${RESULT_ROOT}/${variant}/_probe/kv_lru_stats.jsonl"
    export OMNI_KV_LRU_STATS_LOOKUP_DEDUP_SEC=0
    export OMNI_KV_LRU_STATS_LOOKUP_DEDUP_SIZE=65536
  fi
}

apply_prefetch_off_env() {
  export OMNI_PREFETCH_ADMISSION=0
  export OMNI_PREFETCH_FAST_BYPASS=0
  export OMNI_PREFETCH_SKIP_OFFLOAD_ON_MISS=0
  export OMNI_PREFETCH_HISTORY_HIT_ENABLE=0
  export OMNI_PREFETCH_GPU_PIN=0
  export OMNI_SCHED_NATIVE_PREFETCH=0
  export OMNI_SCHED_NATIVE_PREFETCH_CANCEL=0
  export OMNI_PREFETCH_PRIORITY=0
}

apply_prefetch_on_env() {
  export OMNI_PREFETCH_ADMISSION=1
  export OMNI_PREFETCH_FAST_BYPASS=1
  export OMNI_PREFETCH_SKIP_OFFLOAD_ON_MISS=1
  export OMNI_PREFETCH_HISTORY_HIT_ENABLE=1
  export OMNI_PREFETCH_REGISTRY_TTL_SEC=900
  export OMNI_PREFETCH_STREAM_TIMEOUT_MS=500
  export OMNI_PREFETCH_GPU_PIN=1
  export OMNI_PREFETCH_GPU_PIN_TTL_MS=600000
  export OMNI_PREFETCH_GPU_PIN_MAX_BLOCKS=8192
  export OMNI_SCHED_NATIVE_PREFETCH=1
  export OMNI_SCHED_NATIVE_PREFETCH_CANCEL=1
  export OMNI_SCHED_NATIVE_PREFETCH_MAX_LIVE=0
  export OMNI_PREFETCH_PRIORITY=100
}

apply_lru_env() {
  export OMNI_KV_EVICT_SHADOW=0
  export OMNI_KV_EVICT_ENFORCE_POLICY=none
  export OMNI_KV_EVICT_ENFORCE_INDEX=0
  export OMNI_KV_EVICT_INDEX_MAX_BLOCKS_PER_REQUEST=0
  export OMNI_KV_EVICT_ENFORCE_MIN_CACHED_VICTIMS_PER_ALLOC=0
  export OMNI_KV_EVICT_ENFORCE_MAX_CACHED_VICTIMS_PER_ALLOC=0
}

apply_tail_aware_env() {
  export OMNI_KV_EVICT_SHADOW="${LIVE_EVICT_SHADOW}"
  export OMNI_KV_EVICT_LOG_PATH="${RESULT_ROOT}/liveserve_full/_probe/kv_eviction_shadow.jsonl"
  export OMNI_KV_EVICT_ENFORCE_POLICY="${LIVE_EVICT_POLICY}"
  export OMNI_KV_EVICT_ENFORCE_INDEX="${LIVE_EVICT_INDEX}"
  export OMNI_KV_EVICT_ENFORCE_INDEX_FALLBACK_SCAN="${LIVE_EVICT_INDEX_FALLBACK_SCAN}"
  export OMNI_KV_EVICT_MAX_SCAN_BLOCKS="${LIVE_EVICT_MAX_SCAN_BLOCKS}"
  export OMNI_KV_EVICT_INDEX_MAX_BLOCKS_PER_REQUEST="${LIVE_EVICT_INDEX_TAIL}"
  export OMNI_KV_EVICT_ENFORCE_MIN_CACHED_VICTIMS_PER_ALLOC="${LIVE_EVICT_MIN_CACHED}"
  export OMNI_KV_EVICT_ENFORCE_MAX_CACHED_VICTIMS_PER_ALLOC="${LIVE_EVICT_MAX_CACHED}"
  export OMNI_KV_EVICT_ENFORCE_PARTIAL_OVER_CAP="${LIVE_EVICT_PARTIAL_OVER_CAP}"
  export OMNI_KV_EVICT_ENFORCE_MAX_CANDIDATE_FREE_RATIO="${LIVE_EVICT_CANDIDATE_FREE_RATIO}"
  export OMNI_KV_EVICT_ENFORCE_MIN_METADATA_RATIO="${LIVE_EVICT_MIN_METADATA_RATIO}"
  export OMNI_KV_EVICT_ENFORCE_LOG="${LIVE_EVICT_ENFORCE_LOG}"
  export OMNI_KV_EVICT_COST_LOSS_WEIGHT="${LIVE_EVICT_COST_LOSS_WEIGHT}"
  export OMNI_KV_EVICT_NEAR_GAP_SEC="${LIVE_EVICT_NEAR_GAP_SEC}"
  export OMNI_KV_EVICT_RESPONSE_PENALTY_WEIGHT="${LIVE_EVICT_RESPONSE_PENALTY_WEIGHT}"
}

apply_variant_env() {
  local variant="$1"
  reset_runtime_env
  apply_common_env "${variant}"
  case "${variant}" in
    baseline)
      export SCHEDULER_POLICY_PRESET=baseline
      export SCHEDULER_POLICY_OVERLAYS=
      scheduler_policy_apply_and_export "${SCHEDULER_POLICY_PRESET}" "${SCHEDULER_POLICY_OVERLAYS}"
      apply_prefetch_off_env
      apply_lru_env
      ;;
    liveserve_full)
      export SCHEDULER_POLICY_PRESET="${LIVE_SCHEDULER_POLICY_PRESET}"
      export SCHEDULER_POLICY_OVERLAYS="${LIVE_SCHEDULER_POLICY_OVERLAYS}"
      scheduler_policy_apply_and_export "${SCHEDULER_POLICY_PRESET}" "${SCHEDULER_POLICY_OVERLAYS}"
      apply_prefetch_on_env
      apply_tail_aware_env
      ;;
    *)
      fatal "unknown variant '${variant}'"
      ;;
  esac
}

deploy_yaml_for_variant() {
  case "$1" in
    baseline) printf '%s\n' "${BASELINE_DEPLOY_YAML}" ;;
    liveserve_full) printf '%s\n' "${LIVESERVE_DEPLOY_YAML}" ;;
  esac
}

env_snapshot() {
  env | LC_ALL=C sort | grep -E '^(SCHEDULER_POLICY_|CUDA_VISIBLE_DEVICES=|HF_HUB_OFFLINE=|MC_TRANSFER_TIMEOUT=|OMNI_|PYTHONPATH=|VLLM_)' || true
}

write_run_config() {
  cat >"${RESULT_ROOT}/run_config.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "result_root": "${RESULT_ROOT}",
  "text_gpus": "${TEXT_GPUS}",
  "port": ${PORT},
  "variants": "${VARIANTS}",
  "concurrency_list": "${CONCURRENCY_LIST}",
  "workload": "interactive_original",
  "trace_parts": [1],
  "time_start": ${TIME_START},
  "time_limit": ${TIME_LIMIT},
  "length_profile": "trace",
  "with_audio": true,
  "interrupt_prob": 0.0,
  "server_moe_backend": "${SERVER_MOE_BACKEND}",
  "baseline_deploy_yaml": "${BASELINE_DEPLOY_YAML}",
  "liveserve_deploy_yaml": "${LIVESERVE_DEPLOY_YAML}",
  "served_model_name": "${SERVED_MODEL_NAME}"
}
EOF
}

start_server() {
  local variant="$1"
  local deploy_yaml
  deploy_yaml="$(deploy_yaml_for_variant "${variant}")"
  [[ -f "${deploy_yaml}" ]] || fatal "deploy yaml not found: ${deploy_yaml}"
  if is_healthy; then
    fatal "port ${PORT} is already healthy; refusing to collide"
  fi

  apply_variant_env "${variant}"
  local server_dir="${RESULT_ROOT}/${variant}/_server"
  mkdir -p "${server_dir}" "${RESULT_ROOT}/${variant}/_probe"
  env_snapshot >"${server_dir}/env_snapshot.txt"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >"${server_dir}/nvidia_smi_before.txt" || true

  local -a serve_args=(
    "${PYBIN}" "${VLLM_BIN}" serve
    "${MODEL}"
    --omni
    --port "${PORT}"
    --host 0.0.0.0
    --served-model-name "${SERVED_MODEL_NAME}"
    --stage-init-timeout 1800
    --deploy-config "${deploy_yaml}"
  )
  if [[ -n "${SERVER_MOE_BACKEND}" ]]; then
    serve_args+=(--moe-backend "${SERVER_MOE_BACKEND}")
  fi

  log "starting ${variant}: gpus=${TEXT_GPUS} port=${PORT} deploy=${deploy_yaml}"
  setsid env $(env_snapshot) "${serve_args[@]}" >"${server_dir}/server.log" 2>&1 &
  SERVER_PID=$!
  STARTED_SERVER=1
  echo "${SERVER_PID}" >"${server_dir}/server.pid"

  local ready=0
  for i in $(seq 1 1800); do
    if is_healthy; then
      ready=1
      log "${variant} server healthy after ${i}s"
      break
    fi
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      log "${variant} server died during boot; tail follows"
      tail -160 "${server_dir}/server.log" | tee -a "${LOGDIR}/run.log" || true
      return 3
    fi
    sleep 1
  done
  if [[ "${ready}" -ne 1 ]]; then
    log "${variant} server did not become healthy; tail follows"
    tail -160 "${server_dir}/server.log" | tee -a "${LOGDIR}/run.log" || true
    return 4
  fi
  log "warmup grace 30s"
  sleep 30
}

prefetch_args_for_variant() {
  case "$1" in
    baseline)
      printf '%s\n' --prefetch-mode none
      ;;
    liveserve_full)
      printf '%s\n' \
        --prefetch-mode omni \
        --prefetch-lead-ms "${PREFETCH_LEAD_MS}" \
        --prefetch-max-tokens 1 \
        --prefetch-timeout-sec 30 \
        --prefetch-max-inflight "${PREFETCH_MAX_INFLIGHT}" \
        --prefetch-history-min-remaining-lead-ms 0 \
        --current-prefill-min-remaining-lead-ms 250 \
        --prefetch-clipped-start-limit "${PREFETCH_CLIPPED_START_LIMIT}" \
        --prefetch-same-time-request-first
      ;;
  esac
}

run_cell() {
  local variant="$1"
  local c="$2"
  local tag="${variant}_c${c}_p0_win${TIME_START}_${TIME_LIMIT}"
  local outdir="${RESULT_ROOT}/${variant}/${tag}"
  local cell_log="${LOGDIR}/${tag}.log"
  local -a prefetch_args=()
  local -a timeline_args=()
  while IFS= read -r item; do
    prefetch_args+=("${item}")
  done < <(prefetch_args_for_variant "${variant}")
  if [[ "${RECORD_AUDIO_TIMELINE}" == "1" ]]; then
    timeline_args+=(--record-audio-timeline)
  fi
  mkdir -p "${outdir}"
  log "starting cell ${tag}"
  "${PYBIN}" -u "${DRIVER}" \
    --url "$(api_url)" \
    --mode smoke \
    --time-start "${TIME_START}" \
    --time-limit "${TIME_LIMIT}" \
    --max-inflight "${c}" \
    --with-audio \
    --policy fixed_time \
    --interrupt-after-seconds 2.0 \
    --interrupt-jitter-sec 0.5 \
    --interrupt-time-anchor ttfp \
    --interrupt-prob 0 \
    --model "${SERVED_MODEL_NAME}" \
    --progress-interval "${PROGRESS_INTERVAL}" \
    --request-timeout-sec "${REQUEST_TIMEOUT_SEC}" \
    "${timeline_args[@]}" \
    "${prefetch_args[@]}" \
    --result-dir "${outdir}" \
    >>"${cell_log}" 2>&1
  local rc=$?
  log "cell ${tag} exit=${rc}"
  if [[ "${rc}" -ne 0 ]]; then
    tail -160 "${cell_log}" | tee -a "${LOGDIR}/run.log" || true
  fi
  return "${rc}"
}

aggregate_results() {
  log "aggregating summaries"
  "${PYBIN}" "${ROOT}/aggregate_capacity.py" "${RESULT_ROOT}" >"${RESULT_ROOT}/capacity_table.txt" 2>"${RESULT_ROOT}/capacity_table.err" || true
  "${PYBIN}" "${ROOT}/aggregate_audio_rtf.py" "${RESULT_ROOT}" >"${RESULT_ROOT}/audio_rtf_table.txt" 2>"${RESULT_ROOT}/audio_rtf_table.err" || true
}

main() {
  require_two_gpu_pair
  [[ -x "${PYBIN}" ]] || fatal "python not executable: ${PYBIN}"
  [[ -x "${VLLM_BIN}" ]] || fatal "vllm-omni not executable: ${VLLM_BIN}"
  [[ -f "${SCHEDULER_POLICY_LIB}" ]] || fatal "missing scheduler policy helper: ${SCHEDULER_POLICY_LIB}"
  # shellcheck source=/dev/null
  source "${SCHEDULER_POLICY_LIB}"

  write_run_config
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >"${RESULT_ROOT}/nvidia_smi_start.txt" || true
  log "RESULT_ROOT=${RESULT_ROOT}"
  log "TEXT_GPUS=${TEXT_GPUS}; CUDA_VISIBLE_DEVICES will be forced to ${TEXT_GPUS}"
  log "CONCURRENCY_LIST=${CONCURRENCY_LIST}; trace window=[${TIME_START},${TIME_LIMIT})"

  local rc_total=0
  local variant c rc
  for variant in ${VARIANTS}; do
    start_server "${variant}" || return $?
    for c in ${CONCURRENCY_LIST}; do
      run_cell "${variant}" "${c}"
      rc=$?
      if [[ "${rc}" -ne 0 && "${rc_total}" -eq 0 ]]; then
        rc_total="${rc}"
      fi
      aggregate_results
    done
    stop_server
    log "cooldown 30s after ${variant}"
    sleep 30
  done
  aggregate_results
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >"${RESULT_ROOT}/nvidia_smi_end.txt" || true
  log "DONE rc=${rc_total}; results=${RESULT_ROOT}"
  return "${rc_total}"
}

main "$@"
