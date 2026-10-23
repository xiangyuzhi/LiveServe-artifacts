#!/usr/bin/env bash
# Paper-style StreamingBench video load matrix.
#
# The script intentionally defaults to the current H200 profiling lane:
#   2GPU cells -> CUDA_VISIBLE_DEVICES=4,5
#   3GPU cells -> CUDA_VISIBLE_DEVICES=4,5,6
# Override with LIVESERVE_GPU_DEVICES_2GPU / LIVESERVE_GPU_DEVICES_3GPU when
# a run must stay on a different GPU range.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
PYBIN="${LIVESERVE_PYBIN:-/workspace/xyz/omni_020/bin/python}"
VLLM_BIN="${LIVESERVE_VLLM_BIN:-/workspace/xyz/omni_020/bin/vllm-omni}"
PORT="${LIVESERVE_PORT:-8777}"
PIDFILE="/tmp/liveserve_streamingbench_paper_${PORT}.pid"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_BASE="${RESULT_BASE:-${REPO_ROOT}/results/video/paper_streamingbench_${STAMP}}"
REQUEST_ID_NAMESPACE="${REQUEST_ID_NAMESPACE:-${STAMP}}"
METADATA="${STREAMINGBENCH_METADATA:-${ROOT}/example_questions.json}"
DATA_ROOT="${STREAMINGBENCH_DATA_ROOT:-/workspace/xyz/test/video/StreamingBench/src/data}"
GPU_COUNTS="${GPU_COUNTS:-2 3}"
VARIANTS="${VARIANTS:-baseline liveserve_scheduler}"
MODES="${MODES:-baseline bargein}"
CONCURRENCIES_2GPU="${CONCURRENCIES_2GPU:-1 2 4 6 8}"
CONCURRENCIES_3GPU="${CONCURRENCIES_3GPU:-1 2 4 8 12}"
CONCURRENCIES_8GPU="${CONCURRENCIES_8GPU:-1 2 4 8 12 16}"
REPEAT_FACTOR="${REPEAT_FACTOR:-4}"
MODEL_DEFAULT="/workspace/models/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"

export LIVESERVE_MODEL="${LIVESERVE_MODEL:-${MODEL_DEFAULT}}"
export LIVESERVE_PYBIN="${PYBIN}"
export LIVESERVE_VLLM_BIN="${VLLM_BIN}"
export LIVESERVE_PORT="${PORT}"

mkdir -p "${RESULT_BASE}"

stop_server() {
  local stopped=0
  if [ -s "${PIDFILE}" ]; then
    local pid
    pid="$(cat "${PIDFILE}")"
    local pgid self_pgid
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
    if [ -z "${pgid}" ]; then
      pgid="${pid}"
    fi
    self_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]' || true)"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[paper] stopping server pid=${pid}"
      stopped=1
      if [ -n "${pgid}" ] && [ "${pgid}" != "${self_pgid}" ]; then
        kill -TERM "-${pgid}" 2>/dev/null || true
      else
        kill -TERM "${pid}" 2>/dev/null || true
      fi
      for _ in $(seq 1 90); do
        if ! kill -0 "${pid}" 2>/dev/null; then
          break
        fi
        sleep 1
      done
      if kill -0 "${pid}" 2>/dev/null; then
        if [ -n "${pgid}" ] && [ "${pgid}" != "${self_pgid}" ]; then
          kill -9 "-${pgid}" 2>/dev/null || true
        else
          kill -9 "${pid}" 2>/dev/null || true
        fi
      fi
    elif [ -n "${pgid}" ] && [ "${pgid}" != "${self_pgid}" ] && kill -0 "-${pgid}" 2>/dev/null; then
      echo "[paper] stopping orphaned server process group pgid=${pgid}"
      stopped=1
      kill -TERM "-${pgid}" 2>/dev/null || true
      for _ in $(seq 1 30); do
        if ! kill -0 "-${pgid}" 2>/dev/null; then
          break
        fi
        sleep 1
      done
      if kill -0 "-${pgid}" 2>/dev/null; then
        kill -9 "-${pgid}" 2>/dev/null || true
      fi
    fi
    rm -f "${PIDFILE}"
  fi
  if [ "${stopped}" = "1" ]; then
    sleep "${POST_STOP_SLEEP_SEC:-30}"
  fi
}

wait_health() {
  local log_file="$1"
  for i in $(seq 1 900); do
    if curl -s -o /dev/null -w '%{http_code}' -m 2 "http://127.0.0.1:${PORT}/health" | grep -q '^200$'; then
      echo "[paper] server healthy after ${i}s"
      return 0
    fi
    local pid
    pid="$(cat "${PIDFILE}")"
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[fatal] server exited during boot; tail ${log_file}" >&2
      tail -120 "${log_file}" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "[fatal] server did not become healthy; tail ${log_file}" >&2
  tail -120 "${log_file}" >&2 || true
  return 1
}

devices_for_gpu_count() {
  case "$1" in
    2) printf '%s' "${LIVESERVE_GPU_DEVICES_2GPU:-4,5}" ;;
    3) printf '%s' "${LIVESERVE_GPU_DEVICES_3GPU:-4,5,6}" ;;
    8) printf '%s' "${LIVESERVE_GPU_DEVICES_8GPU:-0,1,2,3,4,5,6,7}" ;;
    *) echo "[fatal] unsupported GPU count: $1" >&2; return 2 ;;
  esac
}

memory_used_for_device() {
  local device="$1"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v dev="${device}" '
      {
        gsub(/^[ \t]+|[ \t]+$/, "", $1)
        gsub(/^[ \t]+|[ \t]+$/, "", $2)
        if ($1 == dev) {
          print $2 + 0
          found = 1
        }
      }
      END {
        if (!found) {
          exit 1
        }
      }'
}

wait_devices_available() {
  local devices="$1"
  local threshold="${GPU_BUSY_MEM_THRESHOLD_MB:-1024}"
  local timeout="${GPU_WAIT_TIMEOUT_SEC:-1800}"
  local start
  start="$(date +%s)"
  while true; do
    local busy=()
    local dev
    for dev in ${devices//,/ }; do
      local used
      used="$(memory_used_for_device "${dev}" || echo 999999)"
      if [ "${used}" -ge "${threshold}" ]; then
        busy+=("${dev}:${used}MiB")
      fi
    done
    if [ "${#busy[@]}" -eq 0 ]; then
      sleep "${POST_DEVICE_IDLE_SLEEP_SEC:-5}"
      return 0
    fi
    local now
    now="$(date +%s)"
    if [ $((now - start)) -ge "${timeout}" ]; then
      echo "[fatal] target GPU(s) still busy after ${timeout}s: ${busy[*]}" >&2
      return 1
    fi
    echo "[paper] waiting for target GPU(s) to become idle: ${busy[*]}"
    sleep "${GPU_WAIT_POLL_SEC:-15}"
  done
}

concurrencies_for_gpu_count() {
  case "$1" in
    2) printf '%s\n' "${CONCURRENCIES_2GPU}" ;;
    3) printf '%s\n' "${CONCURRENCIES_3GPU}" ;;
    8) printf '%s\n' "${CONCURRENCIES_8GPU}" ;;
    *) return 2 ;;
  esac
}

deploy_yaml_for_cell() {
  local gpu_count="$1"
  local variant="$2"
  local generic="${LIVESERVE_DEPLOY_YAML:-}"

  case "${gpu_count}:${variant}" in
    2:baseline_no_offload)
      printf '%s' "${LIVESERVE_DEPLOY_YAML_NO_OFFLOAD_2GPU:-${LIVESERVE_DEPLOY_YAML_2GPU_NO_OFFLOAD:-${generic:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_multi_replicas_2gpu_chunk8.yaml}}}"
      ;;
    2:baseline|2:liveserve_scheduler|2:scheduler|2:joint|2:joint_g1|2:liveserve_g1|2:liveserve_allopt)
      printf '%s' "${LIVESERVE_DEPLOY_YAML_2GPU:-${generic}}"
      ;;
    3:baseline|3:liveserve_scheduler|3:scheduler)
      printf '%s' "${LIVESERVE_DEPLOY_YAML_3GPU:-${generic}}"
      ;;
    3:thinker2)
      printf '%s' "${LIVESERVE_DEPLOY_YAML_THINKER2_3GPU:-${LIVESERVE_DEPLOY_YAML_3GPU_THINKER2:-${LIVESERVE_DEPLOY_YAML_THINKER2:-${generic}}}}"
      ;;
    8:baseline|8:baseline_no_offload|8:liveserve_scheduler|8:scheduler)
      printf '%s' "${LIVESERVE_DEPLOY_YAML_NO_OFFLOAD_8GPU:-${LIVESERVE_DEPLOY_YAML_8GPU_NO_OFFLOAD:-${LIVESERVE_DEPLOY_YAML_8GPU:-${generic:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_4thinker_4talker_8gpu.yaml}}}}"
      ;;
    8:joint)
      printf '%s' "${LIVESERVE_DEPLOY_YAML_8GPU:-${generic:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_4thinker_4talker_8gpu_prefix_cache_only.yaml}}"
      ;;
    8:joint_g1|8:liveserve_g1|8:liveserve_allopt)
      printf '%s' "${LIVESERVE_DEPLOY_YAML_G1_8GPU:-${LIVESERVE_DEPLOY_YAML_8GPU_G1:-${LIVESERVE_DEPLOY_YAML_8GPU:-${generic:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_4thinker_4talker_8gpu_prefix_cache_only_standardcap_codec16_lc8.yaml}}}}"
      ;;
    *)
      printf '%s' "${generic}"
      ;;
  esac
}

is_joint_variant() {
  case "$1" in
    joint|joint_g1|liveserve_g1|liveserve_allopt)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_scheduler_variant() {
  case "$1" in
    liveserve_scheduler|scheduler)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_barge_in_mode() {
  case "$1" in
    bargein|barge_in)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

evict_profile_for_variant() {
  case "$1" in
    joint_g1|liveserve_g1|liveserve_allopt)
      printf '%s' g1
      ;;
    *)
      printf '%s' tail24
      ;;
  esac
}

start_server() {
  local gpu_count="$1"
  local variant="$2"
  local out_dir="$3"
  local devices
  devices="$(devices_for_gpu_count "${gpu_count}")"
  case ",${devices}," in
    *,0,*|*,1,*|*,2,*)
      case "${LIVESERVE_ALLOW_GPU_0_2:-0}" in
        1|true|TRUE|yes|YES|on|ON) ;;
        *)
          echo "[fatal] refusing to use GPUs 0-2; got ${devices}; set LIVESERVE_ALLOW_GPU_0_2=1 to override" >&2
          exit 2
          ;;
      esac
      ;;
  esac

  mkdir -p "${out_dir}"
  local log_file="${out_dir}/server_g${gpu_count}_${variant}.log"
  touch "${log_file}"

  local max_attempts="${SERVER_START_RETRIES:-2}"
  local attempt
  for attempt in $(seq 1 "${max_attempts}"); do
    stop_server
    wait_devices_available "${devices}"
    local deploy_yaml
    deploy_yaml="$(deploy_yaml_for_cell "${gpu_count}" "${variant}")"
    echo "[paper] starting g${gpu_count} ${variant} attempt=${attempt}/${max_attempts} on CUDA_VISIBLE_DEVICES=${devices}"
    {
      echo
      echo "[paper] ===== server start $(date +%Y-%m-%dT%H:%M:%S%z) ====="
      echo "[paper] start attempt ${attempt}/${max_attempts}"
      echo "[paper] CUDA_VISIBLE_DEVICES=${devices}"
      echo "[paper] LIVESERVE_DEPLOY_YAML=${deploy_yaml:-<script-default>}"
    } >> "${log_file}"
    if [ "${gpu_count}" = "2" ]; then
      if is_scheduler_variant "${variant}"; then
        CUDA_VISIBLE_DEVICES="${devices}" LIVESERVE_DEPLOY_YAML="${deploy_yaml}" nohup setsid bash "${REPO_ROOT}/scripts/H200/start_server_2gpu_video_scheduler.sh" >> "${log_file}" 2>&1 &
      elif is_joint_variant "${variant}"; then
        CUDA_VISIBLE_DEVICES="${devices}" LIVESERVE_DEPLOY_YAML="${deploy_yaml}" LIVE_EVICT_PROFILE="$(evict_profile_for_variant "${variant}")" nohup setsid bash "${REPO_ROOT}/scripts/H200/start_server_2gpu_video_joint.sh" >> "${log_file}" 2>&1 &
      elif [ "${variant}" = "thinker2" ]; then
        echo "[fatal] variant thinker2 requires gpu_count=3" >&2
        return 2
      else
        CUDA_VISIBLE_DEVICES="${devices}" LIVESERVE_DEPLOY_YAML="${deploy_yaml}" nohup setsid bash "${REPO_ROOT}/scripts/H200/start_server_2gpu_video_baseline.sh" >> "${log_file}" 2>&1 &
      fi
    elif [ "${gpu_count}" = "3" ]; then
      if is_scheduler_variant "${variant}"; then
        CUDA_VISIBLE_DEVICES="${devices}" LIVESERVE_DEPLOY_YAML="${deploy_yaml}" nohup setsid bash "${REPO_ROOT}/scripts/H200/start_server_3gpu_video_scheduler.sh" >> "${log_file}" 2>&1 &
      elif [ "${variant}" = "thinker2" ]; then
        CUDA_VISIBLE_DEVICES="${devices}" LIVESERVE_DEPLOY_YAML="${deploy_yaml}" nohup setsid bash "${REPO_ROOT}/scripts/H200/start_server_3gpu_video_2thinker.sh" >> "${log_file}" 2>&1 &
      else
        CUDA_VISIBLE_DEVICES="${devices}" LIVESERVE_DEPLOY_YAML="${deploy_yaml}" nohup setsid bash "${REPO_ROOT}/scripts/H200/start_server_3gpu_video_baseline.sh" >> "${log_file}" 2>&1 &
      fi
    else
      if is_scheduler_variant "${variant}"; then
        CUDA_VISIBLE_DEVICES="${devices}" LIVESERVE_ALLOW_GPU_0_2="${LIVESERVE_ALLOW_GPU_0_2:-1}" LIVESERVE_DEPLOY_YAML="${deploy_yaml}" nohup setsid bash "${REPO_ROOT}/scripts/H200/start_server_2gpu_video_scheduler.sh" >> "${log_file}" 2>&1 &
      elif is_joint_variant "${variant}"; then
        CUDA_VISIBLE_DEVICES="${devices}" LIVESERVE_ALLOW_GPU_0_2="${LIVESERVE_ALLOW_GPU_0_2:-1}" LIVESERVE_DEPLOY_YAML="${deploy_yaml}" LIVE_EVICT_PROFILE="$(evict_profile_for_variant "${variant}")" nohup setsid bash "${REPO_ROOT}/scripts/H200/start_server_2gpu_video_joint.sh" >> "${log_file}" 2>&1 &
      else
        CUDA_VISIBLE_DEVICES="${devices}" LIVESERVE_ALLOW_GPU_0_2="${LIVESERVE_ALLOW_GPU_0_2:-1}" LIVESERVE_DEPLOY_YAML="${deploy_yaml}" nohup setsid bash "${REPO_ROOT}/scripts/H200/start_server_2gpu_video_baseline.sh" >> "${log_file}" 2>&1 &
      fi
    fi
    echo $! > "${PIDFILE}"
    if wait_health "${log_file}"; then
      sleep "${POST_BOOT_SLEEP_SEC:-15}"
      return 0
    fi
    echo "[paper] server start failed for g${gpu_count} ${variant}; retrying after cleanup"
    stop_server
  done

  echo "[fatal] unable to start g${gpu_count} ${variant} after ${max_attempts} attempt(s)" >&2
  return 1
}

run_cell() {
  local gpu_count="$1"
  local variant="$2"
  local mode="$3"
  local concurrency="$4"
  local repeat
  if [ -n "${FIXED_REPEAT:-}" ]; then
    repeat="${FIXED_REPEAT}"
  else
    repeat=$((concurrency * REPEAT_FACTOR))
    if [ "${repeat}" -lt "${MIN_REPEAT:-4}" ]; then
      repeat="${MIN_REPEAT:-4}"
    fi
  fi
  local tag="g${gpu_count}_${variant}_${mode}_c${concurrency}"
  local out_dir="${RESULT_BASE}/${tag}"
  local request_id_prefix="${REQUEST_ID_NAMESPACE}_${tag}"
  local devices
  devices="$(devices_for_gpu_count "${gpu_count}")"
  mkdir -p "${out_dir}"
  echo "[paper] driver ${tag} repeat=${repeat} audio_output=${STREAMINGBENCH_WITH_AUDIO_OUTPUT:-0}"

  local barge_in_args=()
  if is_barge_in_mode "${mode}"; then
    barge_in_args=(
      --barge-in-after-seconds "${BARGE_IN_AFTER_SECONDS:-0.20}"
      --barge-in-jitter-sec "${BARGE_IN_JITTER_SEC:-0.05}"
      --barge-in-prob "${BARGE_IN_PROB:-1.0}"
      --barge-in-time-anchor "${BARGE_IN_TIME_ANCHOR:-${STREAMINGBENCH_BARGE_IN_TIME_ANCHOR:-request}}"
    )
  fi

  local audio_output_args=()
  case "${STREAMINGBENCH_WITH_AUDIO_OUTPUT:-0}" in
    1|true|TRUE|yes|YES|on|ON)
      audio_output_args=(--with-audio-output)
      ;;
  esac

  local generation_args=()
  case "${STREAMINGBENCH_IGNORE_EOS:-0}" in
    1|true|TRUE|yes|YES|on|ON)
      generation_args+=(--ignore-eos)
      ;;
  esac
  case "${STREAMINGBENCH_DISABLE_THINKER_STOP_TOKEN_IDS:-0}" in
    1|true|TRUE|yes|YES|on|ON)
      generation_args+=(--disable-thinker-stop-token-ids)
      ;;
  esac

  local continuity_threshold
  continuity_threshold="${STREAMINGBENCH_AUDIO_CONTINUITY_THRESHOLD_SEC:-${LIVESERVE_AUDIO_CONTINUITY_THRESHOLD_S:-0.1}}"
  local continuity_args=(
    --audio-continuity-threshold-sec "${continuity_threshold}"
    --audio-continuity-chunk-mode "${STREAMINGBENCH_AUDIO_CONTINUITY_CHUNK_MODE:-fragment}"
  )
  case "${STREAMINGBENCH_CONTINUITY_INCLUDE_INTERRUPTED:-0}" in
    1|true|TRUE|yes|YES|on|ON)
      continuity_args+=(--continuity-include-barge-in)
      ;;
  esac
  case "${STREAMINGBENCH_RECORD_AUDIO_TIMELINE:-0}" in
    1|true|TRUE|yes|YES|on|ON)
      continuity_args+=(--record-audio-timeline)
      ;;
  esac

  local kv_args=()
  if is_joint_variant "${variant}"; then
    case "${STREAMINGBENCH_ENABLE_KV_PRELOAD:-${STREAMINGBENCH_ENABLE_KV_PREFETCH:-1}}" in
      0|false|FALSE|no|NO|off|OFF)
        kv_args=()
        ;;
      *)
        kv_args=(
          --enable-kv-preload
          --preload-scope "${STREAMINGBENCH_PRELOAD_SCOPE:-${STREAMINGBENCH_PREFETCH_SCOPE:-history_preload}}"
          --preload-lead-ms "${STREAMINGBENCH_PRELOAD_LEAD_MS:-${STREAMINGBENCH_PREFETCH_LEAD_MS:-5000}}"
          --preload-timeout-sec "${STREAMINGBENCH_PRELOAD_TIMEOUT_SEC:-${STREAMINGBENCH_PREFETCH_TIMEOUT_SEC:-5}}"
          --preload-max-tokens "${STREAMINGBENCH_PRELOAD_MAX_TOKENS:-${STREAMINGBENCH_PREFETCH_MAX_TOKENS:-1}}"
        )
        ;;
    esac
  fi

  local monitor_pid
  monitor_pid="$(start_gpu_monitor "${devices}" "${out_dir}")"
  set +e
  "${PYBIN}" -u "${ROOT}/streamingbench_driver.py" \
    --ws-url "ws://127.0.0.1:${PORT}/v1/video/chat/stream" \
    --metadata "${METADATA}" \
    --data-root "${DATA_ROOT}" \
    --result-dir "${out_dir}" \
    --mode "${mode}" \
    --concurrency "${concurrency}" \
    --repeat "${repeat}" \
    --request-id-prefix "${request_id_prefix}" \
    --max-videos "${MAX_VIDEOS:-1}" \
    --max-questions-per-video "${MAX_QUESTIONS_PER_VIDEO:-2}" \
    --fps "${STREAMINGBENCH_FPS:-2}" \
    --frames-per-query "${STREAMINGBENCH_FRAMES_PER_QUERY:-32}" \
    --max-buffer-frames "${STREAMINGBENCH_MAX_BUFFER_FRAMES:-96}" \
    --num-sample-frames "${STREAMINGBENCH_NUM_SAMPLE_FRAMES:-16}" \
    --resize-max-side "${STREAMINGBENCH_RESIZE_MAX_SIDE:-448}" \
    --max-tokens "${STREAMINGBENCH_MAX_TOKENS:-48}" \
    --min-tokens "${STREAMINGBENCH_MIN_TOKENS:-0}" \
    "${generation_args[@]}" \
    --prompt-style "${STREAMINGBENCH_PROMPT_STYLE:-mcq_letter}" \
    --dialogue-mode "${STREAMINGBENCH_DIALOGUE_MODE:-single}" \
    --dialogue-history-turns "${STREAMINGBENCH_DIALOGUE_HISTORY_TURNS:-3}" \
    --dialogue-history-answer-chars "${STREAMINGBENCH_DIALOGUE_HISTORY_ANSWER_CHARS:-600}" \
    --followup-profile "${STREAMINGBENCH_FOLLOWUP_PROFILE:-evidence}" \
    --followup-turns-per-question "${STREAMINGBENCH_FOLLOWUP_TURNS_PER_QUESTION:-0}" \
    "${audio_output_args[@]}" \
    --stream-segments \
    --timeout-sec "${TIMEOUT_SEC:-300}" \
    --post-text-audio-grace-sec "${POST_TEXT_AUDIO_GRACE_SEC:-10}" \
    "${barge_in_args[@]}" \
    "${continuity_args[@]}" \
    "${kv_args[@]}" \
    ${DRIVER_EXTRA_ARGS:-} \
    2>&1 | tee "${out_dir}/driver.log"
  local driver_rc=${PIPESTATUS[0]}
  set -e
  stop_gpu_monitor "${monitor_pid}"
  return "${driver_rc}"
}

trap stop_server EXIT

echo "[paper] RESULT_BASE=${RESULT_BASE}"
echo "[paper] METADATA=${METADATA}"
echo "[paper] DATA_ROOT=${DATA_ROOT}"
echo "[paper] GPU_COUNTS=${GPU_COUNTS}"
echo "[paper] VARIANTS=${VARIANTS}"
echo "[paper] MODES=${MODES}"
echo "[paper] REQUEST_ID_NAMESPACE=${REQUEST_ID_NAMESPACE}"
echo "[paper] PROMPT_STYLE=${STREAMINGBENCH_PROMPT_STYLE:-mcq_letter}"
echo "[paper] DIALOGUE_MODE=${STREAMINGBENCH_DIALOGUE_MODE:-single}"
echo "[paper] FOLLOWUP_PROFILE=${STREAMINGBENCH_FOLLOWUP_PROFILE:-evidence}"
echo "[paper] PRELOAD_SCOPE=${STREAMINGBENCH_PRELOAD_SCOPE:-${STREAMINGBENCH_PREFETCH_SCOPE:-history_preload}}"
echo "[paper] PRELOAD_LEAD_MS=${STREAMINGBENCH_PRELOAD_LEAD_MS:-${STREAMINGBENCH_PREFETCH_LEAD_MS:-5000}}"
echo "[paper] PRELOAD_TIMEOUT_SEC=${STREAMINGBENCH_PRELOAD_TIMEOUT_SEC:-${STREAMINGBENCH_PREFETCH_TIMEOUT_SEC:-5}}"
echo "[paper] PRELOAD_MAX_TOKENS=${STREAMINGBENCH_PRELOAD_MAX_TOKENS:-${STREAMINGBENCH_PREFETCH_MAX_TOKENS:-1}}"
echo "[paper] OUTPUT_MAX_TOKENS=${STREAMINGBENCH_MAX_TOKENS:-48}"
echo "[paper] OUTPUT_MIN_TOKENS=${STREAMINGBENCH_MIN_TOKENS:-0}"
echo "[paper] OUTPUT_IGNORE_EOS=${STREAMINGBENCH_IGNORE_EOS:-0}"
echo "[paper] OUTPUT_DISABLE_THINKER_STOP_TOKEN_IDS=${STREAMINGBENCH_DISABLE_THINKER_STOP_TOKEN_IDS:-0}"
echo "[paper] CONTINUITY_THRESHOLD=${STREAMINGBENCH_AUDIO_CONTINUITY_THRESHOLD_SEC:-${LIVESERVE_AUDIO_CONTINUITY_THRESHOLD_S:-0.1}}"
echo "[paper] CONTINUITY_CHUNK_MODE=${STREAMINGBENCH_AUDIO_CONTINUITY_CHUNK_MODE:-fragment}"
echo "[paper] CONTINUITY_INCLUDE_INTERRUPTED=${STREAMINGBENCH_CONTINUITY_INCLUDE_INTERRUPTED:-0}"

start_gpu_monitor() {
  local devices="$1"
  local out_dir="$2"
  case "${ENABLE_GPU_DMON:-0}" in
    1|true|TRUE|yes|YES|on|ON)
      nvidia-smi dmon \
        -i "${devices}" \
        -s pucmt \
        -d "${GPU_DMON_INTERVAL_SEC:-1}" \
        -o DT > "${out_dir}/gpu_dmon.log" 2>&1 &
      echo $!
      ;;
    *)
      echo ""
      ;;
  esac
}

stop_gpu_monitor() {
  local pid="${1:-}"
  if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

for gpu_count in ${GPU_COUNTS}; do
  for variant in ${VARIANTS}; do
    if [ "${variant}" = "thinker2" ] && [ "${gpu_count}" != "3" ]; then
      echo "[paper] skipping thinker2 for g${gpu_count}; diagnostic topology requires exactly 3 GPUs"
      continue
    fi
    server_root="${RESULT_BASE}/_server_g${gpu_count}_${variant}"
    start_server "${gpu_count}" "${variant}" "${server_root}"
    server_needs_restart=0
    for mode in ${MODES}; do
      for c in $(concurrencies_for_gpu_count "${gpu_count}"); do
        if [ "${server_needs_restart}" = "1" ]; then
          echo "[paper] restarting server before next cell"
          start_server "${gpu_count}" "${variant}" "${server_root}"
          server_needs_restart=0
        fi
        if run_cell "${gpu_count}" "${variant}" "${mode}" "${c}"; then
          :
        else
          rc=$?
          tag="g${gpu_count}_${variant}_${mode}_c${c}"
          echo "[paper] driver failed for ${tag} rc=${rc}" >&2
          echo "driver failed rc=${rc}" > "${RESULT_BASE}/${tag}/cell_failure.txt"
          case "${CONTINUE_ON_DRIVER_FAILURE:-0}" in
            1|true|TRUE|yes|YES|on|ON)
              case "${RESTART_SERVER_ON_DRIVER_FAILURE:-0}" in
                1|true|TRUE|yes|YES|on|ON)
                  server_needs_restart=1
                  ;;
              esac
              ;;
            *)
              exit "${rc}"
              ;;
          esac
        fi
      done
    done
  done
done

"${PYBIN}" "${ROOT}/summarize_paper_video.py" "${RESULT_BASE}" --out-dir "${RESULT_BASE}/paper_figures" || true
echo "[paper] done: ${RESULT_BASE}"
