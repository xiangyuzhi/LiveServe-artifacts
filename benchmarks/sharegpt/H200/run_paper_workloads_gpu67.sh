#!/usr/bin/env bash
# Run the paper ShareGPT audio arrival-distribution and barge-in workloads.
#
# Defaults are sized for the currently available H200 GPUs 6-7.  Override
# TEXT_GPUS, TEXT_PORT, NUM_REQUESTS, CONCURRENCY, VARIANTS, and workload knobs
# as needed for larger sweeps.
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SCHEDULER_POLICY_LIB="${SCHEDULER_POLICY_LIB:-${ROOT}/profiling/scheduler_analysis/scripts/scheduler_policy_common.sh}"
GPU_PREFLIGHT_LIB="${GPU_PREFLIGHT_LIB:-${ROOT}/profiling/scheduler_analysis/scripts/gpu_preflight_common.sh}"

PYBIN="${LIVESERVE_PYBIN:-/workspace/xyz/omni_020/bin/python}"
VLLM_BIN="${LIVESERVE_VLLM_BIN:-/workspace/xyz/omni_020/bin/vllm-omni}"
MODEL="${LIVESERVE_MODEL:-/workspace/models/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695}"
SERVED_MODEL_NAME="${LIVESERVE_SERVED_MODEL_NAME:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
DEPLOY_YAML="${LIVESERVE_DEPLOY_YAML:-${ROOT}/deploy/H200/qwen3_omni_moe_multi_replicas_2gpu.yaml}"
DEFAULT_SHAREGPT_DATASET="${ROOT}/data/ShareGPT_Chinese_English_90k.json"
if [[ ! -f "${DEFAULT_SHAREGPT_DATASET}" && -f "/workspace/xyz/dataset/ShareGPT_Chinese_English_90k.json" ]]; then
  DEFAULT_SHAREGPT_DATASET="/workspace/xyz/dataset/ShareGPT_Chinese_English_90k.json"
fi
SHAREGPT_DATASET="${SHAREGPT_DATASET:-${LIVESERVE_SHAREGPT_DATASET:-${DEFAULT_SHAREGPT_DATASET}}}"
BURSTGPT_CSV="${BURSTGPT_CSV:-/workspace/xyz/test/BurstGPT/data/BurstGPT_1.csv}"

TEXT_GPUS="${TEXT_GPUS:-6,7}"
TEXT_PORT="${TEXT_PORT:-8896}"
SERVER_MOE_BACKEND="${SERVER_MOE_BACKEND:-triton}"
SERVER_INIT_TIMEOUT="${SERVER_INIT_TIMEOUT:-2400}"
SERVER_STAGE_INIT_TIMEOUT="${SERVER_STAGE_INIT_TIMEOUT:-1800}"
SERVER_WARMUP_SLEEP="${SERVER_WARMUP_SLEEP:-5}"
SERVER_START_ATTEMPTS="${SERVER_START_ATTEMPTS:-2}"
SERVER_START_RETRY_SLEEP="${SERVER_START_RETRY_SLEEP:-20}"
SERVER_READY_ENDPOINT="${SERVER_READY_ENDPOINT:-/v1/models}"
SERVER_READY_GRACE_SEC="${SERVER_READY_GRACE_SEC:-5}"
SERVER_READY_TIMEOUT_SEC="${SERVER_READY_TIMEOUT_SEC:-1500}"

TS="${TS:-$(date -u +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/benchmarks/sharegpt/H200/results/paper_workloads_${TS}}"
VARIANTS="${VARIANTS:-baseline scheduler_legacy_l12}"
RUN_ARRIVAL="${RUN_ARRIVAL:-1}"
RUN_BARGE_IN="${RUN_BARGE_IN:-1}"
SCHEDULER_POLICY_OVERLAYS="${SCHEDULER_POLICY_OVERLAYS:-talkerlead=12:6:75}"

NUM_REQUESTS="${NUM_REQUESTS:-32}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_INFLIGHT="${MAX_INFLIGHT:-${CONCURRENCY}}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_PROMPT_CHARS="${MAX_PROMPT_CHARS:-500}"
REQUEST_RATE="${REQUEST_RATE:-4.0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
CASE_WALL_TIMEOUT_SEC="${CASE_WALL_TIMEOUT_SEC:-0}"
STOP_ON_CASE_FAILURE="${STOP_ON_CASE_FAILURE:-0}"
CASE_MIN_OK="${CASE_MIN_OK:-${NUM_REQUESTS}}"
CASE_MAX_ERRORS="${CASE_MAX_ERRORS:-0}"
SKIP_VALID_CASES="${SKIP_VALID_CASES:-0}"
RESTART_BETWEEN_WORKLOAD_GROUPS="${RESTART_BETWEEN_WORKLOAD_GROUPS:-1}"
SERVER_DIR_SUFFIX="${SERVER_DIR_SUFFIX:-}"
SEED="${SEED:-42}"

BURST_LOG_TYPE="${BURST_LOG_TYPE:-api}"
BURST_WINDOW_MODE="${BURST_WINDOW_MODE:-peak}"
BURST_WINDOW_SEC="${BURST_WINDOW_SEC:-300}"
BURST_SLICE_MODE="${BURST_SLICE_MODE:-densest}"
BURST_SCALE_MODE="${BURST_SCALE_MODE:-target_peak}"
BURST_TARGET_RPS="${BURST_TARGET_RPS:-${REQUEST_RATE}}"
BURST_PEAK_BIN_SEC="${BURST_PEAK_BIN_SEC:-1.0}"

BARGE_IN_PROBS="${BARGE_IN_PROBS:-0.0 0.3 0.7 1.0}"
BARGE_IN_AFTER_SEC="${BARGE_IN_AFTER_SEC:-2.0}"
BARGE_IN_JITTER_SEC="${BARGE_IN_JITTER_SEC:-0.5}"
BARGE_IN_AFTER_DISTRIBUTION="${BARGE_IN_AFTER_DISTRIBUTION:-fixed}"
BARGE_IN_AFTER_MIN="${BARGE_IN_AFTER_MIN:-0.3}"
BARGE_IN_AFTER_MAX="${BARGE_IN_AFTER_MAX:-5.0}"
BARGE_IN_AFTER_MEAN="${BARGE_IN_AFTER_MEAN:-1.5}"
BARGE_IN_AFTER_SAMPLES="${BARGE_IN_AFTER_SAMPLES:-}"
BARGE_IN_AFTER_SAMPLE_KEY="${BARGE_IN_AFTER_SAMPLE_KEY:-audio_duration_s}"
BARGE_IN_AFTER_SAMPLE_TIME_SCALE="${BARGE_IN_AFTER_SAMPLE_TIME_SCALE:-1.0}"
BARGE_IN_TIME_ANCHOR="${BARGE_IN_TIME_ANCHOR:-ttfp}"
POSTHOC_PLAYBACK_BARGE_IN_WASTE="${POSTHOC_PLAYBACK_BARGE_IN_WASTE:-0}"
DURATION_PROBE_VARIANT="${DURATION_PROBE_VARIANT:-baseline}"
DURATION_PROBE_CASE="${DURATION_PROBE_CASE:-duration_probe_p0}"
DURATION_PROBE_REUSE_FIRST_VARIANT="${DURATION_PROBE_REUSE_FIRST_VARIANT:-1}"

BASE_PYTHONPATH="${PYTHONPATH:-}"
SERVER_PID=""
SERVER_LOG_FILE=""
REUSED_PROBE_SERVER_VARIANT=""

source "${SCHEDULER_POLICY_LIB}"

log() {
  printf '[sharegpt-paper %s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

source "${GPU_PREFLIGHT_LIB}"

skip_valid_cases_enabled() {
  [[ "${SKIP_VALID_CASES}" == "1" \
    || "${SKIP_VALID_CASES}" == "true" \
    || "${SKIP_VALID_CASES}" == "yes" ]]
}

server_log_has_fatal_startup() {
  local log_file="$1"
  [[ -f "${log_file}" ]] || return 1
  tail -200 "${log_file}" | rg -q \
    'ValueError: Free memory|subprocess died unexpectedly|Address already in use|EngineCore failed|StageEngineCoreProc died during READY|StageEngineCoreProc failed to start|Orchestrator thread crashed'
}

server_ready() {
  local port="$1"
  curl -fsS "http://127.0.0.1:${port}${SERVER_READY_ENDPOINT}" >/dev/null 2>&1
}

wait_http() {
  local port="$1"
  local log_file="$2"
  local timeout_sec="${SERVER_READY_TIMEOUT_SEC}"
  if (( timeout_sec < 1 )); then
    timeout_sec=1
  fi
  for _ in $(seq 1 "${timeout_sec}"); do
    if server_ready "${port}"; then
      log "server ready on port ${port} endpoint=${SERVER_READY_ENDPOINT}"
      return 0
    fi
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      log "server process exited before health; tail follows"
      tail -180 "${log_file}" || true
      return 1
    fi
    if server_log_has_fatal_startup "${log_file}"; then
      log "server startup hit a fatal condition; tail follows"
      tail -180 "${log_file}" || true
      return 1
    fi
    sleep 1
  done
  log "server failed to become ready; tail follows"
  tail -180 "${log_file}" || true
  return 1
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    log "stopping server pgid ${SERVER_PID}"
    kill -- "-${SERVER_PID}" >/dev/null 2>&1 || true
    sleep 5
    if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      kill -9 -- "-${SERVER_PID}" >/dev/null 2>&1 || true
    fi
    log "server pgid ${SERVER_PID} stopped"
  fi
  SERVER_PID=""
  SERVER_LOG_FILE=""
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
}

apply_common_env() {
  local variant="$1"
  export CUDA_VISIBLE_DEVICES="${TEXT_GPUS}"
  export PYTHONPATH="${ROOT}/vllm-omni${BASE_PYTHONPATH:+:${BASE_PYTHONPATH}}"
  export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
  export MC_TRANSFER_TIMEOUT="${MC_TRANSFER_TIMEOUT:-300}"
  export OMNI_SESSION_KV_PROBE_DIR="${RESULT_ROOT}/${variant}/_probe/text"
  export OMNI_SESSION_KV_PROBE_INTERVAL="${OMNI_SESSION_KV_PROBE_INTERVAL:-0.1}"
  export OMNI_KV_LRU_STATS="${OMNI_KV_LRU_STATS:-0}"
  export OMNI_KV_EVICT_ENFORCE_POLICY=none
  export OMNI_KV_EVICT_ENFORCE_INDEX=0
  export OMNI_PREFETCH_ADMISSION=0
  export OMNI_PREFETCH_HISTORY_HIT_ENABLE=0
  export OMNI_SCHED_NATIVE_PREFETCH=0
}

apply_variant_env() {
  local variant="$1"
  reset_runtime_env
  apply_common_env "${variant}"
  case "${variant}" in
    baseline)
      scheduler_policy_apply_and_export baseline ""
      ;;
    scheduler_legacy_l12|liveserve_scheduler)
      scheduler_policy_apply_and_export scheduler "${SCHEDULER_POLICY_OVERLAYS}"
      ;;
    scheduler_l8|scheduler_talkerlead_8_4_75)
      scheduler_policy_apply_and_export scheduler "talkerlead=8:4:75"
      ;;
    scheduler_l8_h25|scheduler_talkerlead_8_4_25)
      scheduler_policy_apply_and_export scheduler "talkerlead=8:4:25"
      ;;
    scheduler_l12|scheduler_talkerlead_12_6_75)
      scheduler_policy_apply_and_export scheduler "talkerlead=12:6:75"
      ;;
    scheduler_l16|scheduler_talkerlead_16_8_75)
      scheduler_policy_apply_and_export scheduler "talkerlead=16:8:75"
      ;;
    scheduler_l20|scheduler_talkerlead_20_10_75)
      scheduler_policy_apply_and_export scheduler "talkerlead=20:10:75"
      ;;
    *)
      log "unknown variant ${variant}; supported: baseline scheduler_legacy_l12 liveserve_scheduler scheduler_l8 scheduler_l8_h25 scheduler_l12 scheduler_l16 scheduler_l20"
      return 1
      ;;
  esac
}

env_snapshot_args() {
  env | LC_ALL=C sort | grep -E '^(SCHEDULER_POLICY_|CUDA_VISIBLE_DEVICES=|GPU_|MC_TRANSFER_TIMEOUT=|OMNI_|PYTHONPATH=|VLLM_)' || true
}

server_dir_for_label() {
  local result_label="$1"
  local dir_name="_server"
  if [[ -n "${SERVER_DIR_SUFFIX}" ]]; then
    dir_name="_server_${SERVER_DIR_SUFFIX}"
  fi
  printf '%s/%s/%s\n' "${RESULT_ROOT}" "${result_label}" "${dir_name}"
}

write_variant_meta() {
  local variant="$1"
  local result_label="${2:-${variant}}"
  local server_dir
  server_dir="$(server_dir_for_label "${result_label}")"
  mkdir -p "${server_dir}" "${RESULT_ROOT}/${result_label}/_probe/text"
  env_snapshot_args >"${server_dir}/env_snapshot.txt"
  "${PYBIN}" - "${server_dir}/config.json" <<PY
import json
from pathlib import Path

data = {
    "variant": "${variant}",
    "result_label": "${result_label}",
    "text_gpus": "${TEXT_GPUS}",
    "port": ${TEXT_PORT},
    "moe_backend": "${SERVER_MOE_BACKEND}",
    "model": "${MODEL}",
    "served_model_name": "${SERVED_MODEL_NAME}",
    "deploy_yaml": "${DEPLOY_YAML}",
    "server_start_attempts": ${SERVER_START_ATTEMPTS},
    "server_start_retry_sleep": ${SERVER_START_RETRY_SLEEP},
    "server_ready_endpoint": "${SERVER_READY_ENDPOINT}",
    "server_ready_grace_sec": ${SERVER_READY_GRACE_SEC},
    "server_ready_timeout_sec": ${SERVER_READY_TIMEOUT_SEC},
    "server_dir_suffix": "${SERVER_DIR_SUFFIX}",
    "gpu_min_free_gib": ${GPU_MIN_FREE_GIB},
    "gpu_wait_for_free_sec": ${GPU_WAIT_FOR_FREE_SEC},
    "gpu_free_stable_samples": ${GPU_FREE_STABLE_SAMPLES},
    "gpu_require_no_compute_apps": "${GPU_REQUIRE_NO_COMPUTE_APPS}",
    "dataset": "${SHAREGPT_DATASET}",
    "num_requests": ${NUM_REQUESTS},
    "concurrency": ${CONCURRENCY},
    "max_inflight": ${MAX_INFLIGHT},
    "max_tokens": ${MAX_TOKENS},
    "request_rate": ${REQUEST_RATE},
    "case_min_ok": ${CASE_MIN_OK},
    "case_max_errors": ${CASE_MAX_ERRORS},
    "skip_valid_cases": "${SKIP_VALID_CASES}",
    "restart_between_workload_groups": "${RESTART_BETWEEN_WORKLOAD_GROUPS}",
    "barge_in_probs": "${BARGE_IN_PROBS}".split(),
    "barge_in_after_distribution": "${BARGE_IN_AFTER_DISTRIBUTION}",
    "barge_in_after_samples": "${BARGE_IN_AFTER_SAMPLES}",
    "barge_in_after_sample_key": "${BARGE_IN_AFTER_SAMPLE_KEY}",
    "barge_in_after_sample_time_scale": ${BARGE_IN_AFTER_SAMPLE_TIME_SCALE},
    "posthoc_playback_barge_in_waste": "${POSTHOC_PLAYBACK_BARGE_IN_WASTE}",
    "interrupt_probs": "${BARGE_IN_PROBS}".split(),
    "interrupt_after_distribution": "${BARGE_IN_AFTER_DISTRIBUTION}",
    "interrupt_after_samples": "${BARGE_IN_AFTER_SAMPLES}",
    "interrupt_after_sample_key": "${BARGE_IN_AFTER_SAMPLE_KEY}",
    "interrupt_after_sample_time_scale": ${BARGE_IN_AFTER_SAMPLE_TIME_SCALE},
    "posthoc_playback_interrupt_waste": "${POSTHOC_PLAYBACK_BARGE_IN_WASTE}",
    "scheduler_policy_overlays": "${SCHEDULER_POLICY_OVERLAYS}",
}
Path("${server_dir}/config.json").write_text(
    json.dumps(data, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)
PY
}

start_server_once() {
  local variant="$1"
  local result_label="${2:-${variant}}"
  local server_dir
  server_dir="$(server_dir_for_label "${result_label}")"
  local log_file="${server_dir}/server.log"
  if curl -fsS "http://127.0.0.1:${TEXT_PORT}/v1/models" >/dev/null 2>&1; then
    log "port ${TEXT_PORT} already has a server; refusing to collide"
    return 1
  fi

  apply_variant_env "${variant}"
  export OMNI_SESSION_KV_PROBE_DIR="${RESULT_ROOT}/${result_label}/_probe/text"
  if ! check_gpu_free; then
    log "GPU availability preflight failed for ${TEXT_GPUS}"
    return 1
  fi
  write_variant_meta "${variant}" "${result_label}"
  mkdir -p "${server_dir}"

  local -a env_args=()
  while IFS= read -r item; do
    env_args+=("${item}")
  done < <(env_snapshot_args)

  local -a serve_args=(
    "${PYBIN}" "${VLLM_BIN}" serve
    "${MODEL}"
    --omni
    --port "${TEXT_PORT}"
    --host 0.0.0.0
    --served-model-name "${SERVED_MODEL_NAME}"
    --stage-init-timeout "${SERVER_STAGE_INIT_TIMEOUT}"
    --init-timeout "${SERVER_INIT_TIMEOUT}"
    --deploy-config "${DEPLOY_YAML}"
  )
  if [[ -n "${SERVER_MOE_BACKEND}" ]]; then
    serve_args+=(--moe-backend "${SERVER_MOE_BACKEND}")
  fi

  log "starting ${variant}: label=${result_label} gpus=${TEXT_GPUS} port=${TEXT_PORT} deploy=${DEPLOY_YAML}"
  setsid env "${env_args[@]}" "${serve_args[@]}" >"${log_file}" 2>&1 &
  SERVER_PID=$!
  SERVER_LOG_FILE="${log_file}"
  echo "${SERVER_PID}" >"${server_dir}/server.pid"
  if ! wait_http "${TEXT_PORT}" "${log_file}"; then
    return 1
  fi
  sleep "${SERVER_WARMUP_SLEEP}"
  if [[ "${SERVER_READY_GRACE_SEC}" != "0" && "${SERVER_READY_GRACE_SEC}" != "0.0" ]]; then
    sleep "${SERVER_READY_GRACE_SEC}"
  fi
  if ! server_ready "${TEXT_PORT}"; then
    log "server lost readiness before workload; tail follows"
    tail -180 "${log_file}" || true
    return 1
  fi
}

start_server() {
  local variant="$1"
  local result_label="${2:-${variant}}"
  local attempts="${SERVER_START_ATTEMPTS}"
  local attempt rc
  if (( attempts < 1 )); then
    attempts=1
  fi
  for attempt in $(seq 1 "${attempts}"); do
    if (( attempt > 1 )); then
      log "retrying server start ${variant}/${result_label}: attempt ${attempt}/${attempts}"
    fi
    set +e
    start_server_once "${variant}" "${result_label}"
    rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      return 0
    fi
    stop_server
    if (( attempt < attempts )); then
      log "server start failed rc=${rc}; sleeping ${SERVER_START_RETRY_SLEEP}s before retry"
      sleep "${SERVER_START_RETRY_SLEEP}"
    fi
  done
  return "${rc}"
}

generate_burst_manifest() {
  local manifest="${RESULT_ROOT}/_workloads/burstgpt_${NUM_REQUESTS}_${BURST_SCALE_MODE}${BURST_TARGET_RPS}.jsonl"
  local -a scale_args=()
  case "${BURST_SCALE_MODE}" in
    target_peak)
      scale_args=(--target-peak-rps "${BURST_TARGET_RPS}")
      ;;
    target_avg)
      scale_args=(--target-rps "${BURST_TARGET_RPS}")
      ;;
    raw)
      scale_args=(--scale 1.0)
      ;;
    *)
      log "unknown BURST_SCALE_MODE=${BURST_SCALE_MODE}"
      return 1
      ;;
  esac
  "${PYBIN}" "${SCRIPT_DIR}/generate_burstgpt_arrivals.py" \
    --burstgpt-csv "${BURSTGPT_CSV}" \
    --output "${manifest}" \
    --num-requests "${NUM_REQUESTS}" \
    --log-type "${BURST_LOG_TYPE}" \
    --window-mode "${BURST_WINDOW_MODE}" \
    --window-sec "${BURST_WINDOW_SEC}" \
    --slice-mode "${BURST_SLICE_MODE}" \
    --peak-bin-sec "${BURST_PEAK_BIN_SEC}" \
    --seed "${SEED}" \
    "${scale_args[@]}"
  printf '%s\n' "${manifest}"
}

summary_gate_status() {
  local summary_file="$1"
  [[ -s "${summary_file}" ]] || return 1
  "${PYBIN}" - "${summary_file}" "${CASE_MIN_OK}" "${CASE_MAX_ERRORS}" <<'PY'
import json
import sys

summary_path, min_ok_raw, max_errors_raw = sys.argv[1:4]
data = json.load(open(summary_path, "r", encoding="utf-8"))
totals = data.get("totals") or {}
ok = int(totals.get("completed_ok") or 0)
errors = int(totals.get("errors") or 0)
min_ok = int(float(min_ok_raw))
max_errors = int(float(max_errors_raw))
if ok < min_ok or errors > max_errors:
    print(f"ok={ok} min_ok={min_ok} errors={errors} max_errors={max_errors}")
    raise SystemExit(1)
print(f"ok={ok} errors={errors}")
PY
}

case_summary_valid() {
  local variant="$1"
  local name="$2"
  local summary_file="${RESULT_ROOT}/${variant}/${name}/summary.json"
  summary_gate_status "${summary_file}" >/dev/null
}

variant_all_requested_cases_valid() {
  local variant="$1"
  skip_valid_cases_enabled || return 1
  local -a case_names=()
  if [[ "${RUN_ARRIVAL}" == "1" ]]; then
    case_names+=("poisson_r${REQUEST_RATE}_p0")
    case_names+=("burstgpt_${BURST_SCALE_MODE}${BURST_TARGET_RPS}_p0")
  fi
  if [[ "${RUN_BARGE_IN}" == "1" ]]; then
    local p
    for p in ${BARGE_IN_PROBS}; do
      case_names+=("bargein_$(prob_label "${p}")")
    done
  fi
  if (( ${#case_names[@]} == 0 )); then
    return 1
  fi
  local name
  for name in "${case_names[@]}"; do
    if ! case_summary_valid "${variant}" "${name}"; then
      return 1
    fi
  done
  return 0
}

run_sharegpt_case() {
  local variant="$1"
  local name="$2"
  shift 2
  local result_dir="${RESULT_ROOT}/${variant}/${name}"
  local log_file="${result_dir}.log"
  local summary_file="${result_dir}/summary.json"
  local summary_status
  if skip_valid_cases_enabled; then
    if summary_status="$(summary_gate_status "${summary_file}")"; then
      log "skipping ${variant}/${name}: existing summary gate ${summary_status}"
      return 0
    fi
  fi
  mkdir -p "${result_dir}"
  log "case ${variant}/${name} -> ${result_dir}"
  if ! server_ready "${TEXT_PORT}"; then
    log "server not ready before ${variant}/${name}; server tail follows"
    tail -160 "${SERVER_LOG_FILE:-${RESULT_ROOT}/${variant}/_server/server.log}" || true
    return 1
  fi
  local -a driver_cmd=(
    "${PYBIN}" -u "${SCRIPT_DIR}/sharegpt_driver.py"
    --dataset "${SHAREGPT_DATASET}"
    --url "http://127.0.0.1:${TEXT_PORT}/v1/chat/completions"
    --model "${SERVED_MODEL_NAME}"
    --mode audio
    --concurrency "${CONCURRENCY}"
    --max-inflight "${MAX_INFLIGHT}"
    --num-requests "${NUM_REQUESTS}"
    --max-tokens "${MAX_TOKENS}"
    --max-prompt-chars "${MAX_PROMPT_CHARS}"
    --timeout "${TIMEOUT_SEC}"
    --seed "${SEED}"
    --tag "${variant}_${name}"
    "$@"
    --result-dir "${result_dir}"
  )
  set +e
  if [[ "${CASE_WALL_TIMEOUT_SEC}" != "0" && "${CASE_WALL_TIMEOUT_SEC}" != "0.0" ]]; then
    timeout --preserve-status "${CASE_WALL_TIMEOUT_SEC}" "${driver_cmd[@]}" >"${log_file}" 2>&1
  else
    "${driver_cmd[@]}" >"${log_file}" 2>&1
  fi
  local rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    log "FAILED ${variant}/${name} rc=${rc}; tail follows"
    tail -120 "${log_file}" || true
    if [[ "${STOP_ON_CASE_FAILURE}" == "1" ]]; then
      return "${rc}"
    fi
  fi
  if [[ ! -s "${summary_file}" ]]; then
    log "FAILED ${variant}/${name}: missing summary ${summary_file}"
    tail -120 "${log_file}" || true
    return 1
  fi
  summary_status="$(summary_gate_status "${summary_file}")" || {
    log "FAILED ${variant}/${name}: summary gate ${summary_status}"
    tail -120 "${log_file}" || true
    return 1
  }
  log "case ${variant}/${name} summary gate ${summary_status}"
  if ! server_ready "${TEXT_PORT}"; then
    log "server unhealthy after ${variant}/${name}; server tail follows"
    tail -160 "${SERVER_LOG_FILE:-${RESULT_ROOT}/${variant}/_server/server.log}" || true
    return 1
  fi
  return 0
}

run_arrival_cases() {
  local variant="$1"
  local burst_manifest="$2"
  run_sharegpt_case "${variant}" "poisson_r${REQUEST_RATE}_p0" \
    --arrival-process poisson \
    --request-rate "${REQUEST_RATE}" \
    --policy none
  run_sharegpt_case "${variant}" "burstgpt_${BURST_SCALE_MODE}${BURST_TARGET_RPS}_p0" \
    --arrival-process manifest \
    --arrival-manifest "${burst_manifest}" \
    --policy none
}

prob_label() {
  printf 'p%s' "${1//./p}"
}

run_barge_in_cases() {
  local variant="$1"
  local p p_label policy
  local -a distribution_args=(
    --barge-in-after-distribution "${BARGE_IN_AFTER_DISTRIBUTION}"
    --barge-in-after-min "${BARGE_IN_AFTER_MIN}"
    --barge-in-after-max "${BARGE_IN_AFTER_MAX}"
    --barge-in-after-mean "${BARGE_IN_AFTER_MEAN}"
    --barge-in-after-sample-key "${BARGE_IN_AFTER_SAMPLE_KEY}"
    --barge-in-after-sample-time-scale "${BARGE_IN_AFTER_SAMPLE_TIME_SCALE}"
  )
  if [[ -n "${BARGE_IN_AFTER_SAMPLES}" ]]; then
    distribution_args+=(--barge-in-after-samples "${BARGE_IN_AFTER_SAMPLES}")
  fi
  if [[ "${POSTHOC_PLAYBACK_BARGE_IN_WASTE}" == "1" ]]; then
    distribution_args+=(--posthoc-playback-barge-in-waste)
  fi
  for p in ${BARGE_IN_PROBS}; do
    p_label="$(prob_label "${p}")"
    policy="fixed_time"
    if [[ "${p}" == "0" || "${p}" == "0.0" || "${p}" == "0.00" ]]; then
      policy="none"
    fi
    run_sharegpt_case "${variant}" "bargein_${p_label}" \
      --arrival-process closed_loop \
      --policy "${policy}" \
      --barge-in-prob "${p}" \
      --barge-in-after-seconds "${BARGE_IN_AFTER_SEC}" \
      --barge-in-jitter-sec "${BARGE_IN_JITTER_SEC}" \
      "${distribution_args[@]}" \
      --barge-in-time-anchor "${BARGE_IN_TIME_ANCHOR}"
  done
}

run_duration_probe() {
  local first_variant="$1"
  local probe_label="_duration_probe"
  local result_label="${probe_label}"
  local case_variant="${probe_label}"
  local case_name="${DURATION_PROBE_CASE}"
  local keep_server=0
  if [[ "${DURATION_PROBE_REUSE_FIRST_VARIANT}" == "1" && "${DURATION_PROBE_VARIANT}" == "${first_variant}" ]]; then
    result_label="${first_variant}"
    case_variant="${first_variant}"
    case_name="_duration_probe/${DURATION_PROBE_CASE}"
    keep_server=1
  fi
  local probe_sample="${RESULT_ROOT}/${case_variant}/${case_name}/per_request.jsonl"
  log "auto duration probe: variant=${DURATION_PROBE_VARIANT} case=${case_variant}/${case_name} reuse_server=${keep_server}"
  mkdir -p "${RESULT_ROOT}/${result_label}"
  start_server "${DURATION_PROBE_VARIANT}" "${result_label}"
  run_sharegpt_case "${case_variant}" "${case_name}" \
    --arrival-process closed_loop \
    --policy none
  if [[ ! -s "${probe_sample}" ]]; then
    log "duration probe did not produce ${probe_sample}"
    stop_server
    return 1
  fi
  BARGE_IN_AFTER_SAMPLES="${probe_sample}"
  export BARGE_IN_AFTER_SAMPLES
  log "auto barge-in-after samples ${BARGE_IN_AFTER_SAMPLES}"
  if [[ "${keep_server}" == "1" ]]; then
    REUSED_PROBE_SERVER_VARIANT="${first_variant}"
  else
    stop_server
  fi
}

prepare_barge_in_samples() {
  local first_variant="$1"
  if [[ "${RUN_BARGE_IN}" != "1" || "${BARGE_IN_AFTER_DISTRIBUTION}" != "samples" ]]; then
    return 0
  fi
  if [[ "${BARGE_IN_AFTER_SAMPLES}" == "auto" ]]; then
    run_duration_probe "${first_variant}"
    return 0
  fi
  if [[ -z "${BARGE_IN_AFTER_SAMPLES}" ]]; then
    log "BARGE_IN_AFTER_DISTRIBUTION=samples requires BARGE_IN_AFTER_SAMPLES=/path or auto"
    return 1
  fi
  if [[ ! -s "${BARGE_IN_AFTER_SAMPLES}" ]]; then
    log "BARGE_IN_AFTER_SAMPLES does not exist or is empty: ${BARGE_IN_AFTER_SAMPLES}"
    return 1
  fi
}

run_variant() {
  local variant="$1"
  local burst_manifest="$2"
  mkdir -p "${RESULT_ROOT}/${variant}"
  if variant_all_requested_cases_valid "${variant}"; then
    log "skipping ${variant}: all requested cases already satisfy summary gate"
    return 0
  fi
  if [[ -n "${REUSED_PROBE_SERVER_VARIANT}" && "${variant}" == "${REUSED_PROBE_SERVER_VARIANT}" ]]; then
    log "reusing duration-probe server for ${variant}"
    if [[ "${RUN_ARRIVAL}" == "1" ]]; then
      run_arrival_cases "${variant}" "${burst_manifest}"
    fi
    if [[ "${RUN_BARGE_IN}" == "1" ]]; then
      if [[ "${RUN_ARRIVAL}" == "1" && "${RESTART_BETWEEN_WORKLOAD_GROUPS}" == "1" ]]; then
        log "restarting ${variant} server between arrival and barge-in groups"
        stop_server
        start_server "${variant}"
      fi
      run_barge_in_cases "${variant}"
    fi
    stop_server
    REUSED_PROBE_SERVER_VARIANT=""
    return 0
  fi
  start_server "${variant}"
  if [[ "${RUN_ARRIVAL}" == "1" ]]; then
    run_arrival_cases "${variant}" "${burst_manifest}"
  fi
  if [[ "${RUN_BARGE_IN}" == "1" ]]; then
    if [[ "${RUN_ARRIVAL}" == "1" && "${RESTART_BETWEEN_WORKLOAD_GROUPS}" == "1" ]]; then
      log "restarting ${variant} server between arrival and barge-in groups"
      stop_server
      start_server "${variant}"
    fi
    run_barge_in_cases "${variant}"
  fi
  stop_server
}

main() {
  mkdir -p "${RESULT_ROOT}/_workloads"
  log "result root ${RESULT_ROOT}"
  log "variants=${VARIANTS}"
  log "gpus=${TEXT_GPUS} port=${TEXT_PORT} n=${NUM_REQUESTS} c=${CONCURRENCY}"
  if gpu_precheck_only_enabled; then
    if ! check_gpu_free; then
      log "GPU precheck-only failed for ${TEXT_GPUS}"
      return 1
    fi
    log "GPU precheck-only passed for ${TEXT_GPUS}"
    return 0
  fi
  if [[ "${RUN_ARRIVAL}" != "1" && "${RUN_BARGE_IN}" != "1" ]]; then
    log "RUN_ARRIVAL=0 and RUN_BARGE_IN=0; nothing to run"
    return 0
  fi
  local burst_manifest=""
  if [[ "${RUN_ARRIVAL}" == "1" ]]; then
    burst_manifest="$(generate_burst_manifest)"
    log "burst manifest ${burst_manifest}"
  fi
  local variant
  local first_variant=""
  for variant in ${VARIANTS}; do
    first_variant="${variant}"
    break
  done
  prepare_barge_in_samples "${first_variant}"

  for variant in ${VARIANTS}; do
    run_variant "${variant}" "${burst_manifest}"
  done
  log "done ${RESULT_ROOT}"
}

main "$@"
