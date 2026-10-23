#!/usr/bin/env bash
# Run a small StreamingBench video matrix without touching the main workspace.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
PYBIN="${LIVESERVE_PYBIN:-/workspace/xyz/omni_020/bin/python}"
PORT="${LIVESERVE_PORT:-8777}"
PIDFILE="/tmp/liveserve_streamingbench_${PORT}.pid"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_BASE="${RESULT_BASE:-${REPO_ROOT}/results/video/streamingbench_${STAMP}}"
METADATA="${STREAMINGBENCH_METADATA:-${ROOT}/example_questions.json}"
DATA_ROOT="${STREAMINGBENCH_DATA_ROOT:-/workspace/xyz/test/video/StreamingBench}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6}"
MODEL_DEFAULT="/workspace/models/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"
export LIVESERVE_MODEL="${LIVESERVE_MODEL:-${MODEL_DEFAULT}}"
export LIVESERVE_PYBIN="${PYBIN}"
export LIVESERVE_VLLM_BIN="${LIVESERVE_VLLM_BIN:-/workspace/xyz/omni_020/bin/vllm-omni}"
export LIVESERVE_PORT="${PORT}"
export CUDA_VISIBLE_DEVICES

case ",${CUDA_VISIBLE_DEVICES}," in
  *,0,*|*,1,*|*,2,*)
    echo "[fatal] refusing to use GPUs 0-2; set CUDA_VISIBLE_DEVICES within 4-7" >&2
    exit 2
    ;;
esac

mkdir -p "${RESULT_BASE}"

stop_server() {
  if [ -s "${PIDFILE}" ]; then
    local pid
    pid="$(cat "${PIDFILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[matrix] stopping server pid=${pid}"
      kill -TERM "${pid}" || true
      for _ in $(seq 1 60); do
        if ! kill -0 "${pid}" 2>/dev/null; then
          break
        fi
        sleep 1
      done
      if kill -0 "${pid}" 2>/dev/null; then
        kill -9 "${pid}" || true
      fi
    fi
    rm -f "${PIDFILE}"
  fi
}

wait_health() {
  local log_file="$1"
  for i in $(seq 1 900); do
    if curl -s -o /dev/null -w '%{http_code}' -m 2 "http://127.0.0.1:${PORT}/health" | grep -q '^200$'; then
      echo "[matrix] server healthy after ${i}s"
      return 0
    fi
    local pid
    pid="$(cat "${PIDFILE}")"
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[fatal] server exited during boot; tail ${log_file}" >&2
      tail -80 "${log_file}" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "[fatal] server did not become healthy; tail ${log_file}" >&2
  tail -80 "${log_file}" >&2 || true
  return 1
}

start_server() {
  local kind="$1"
  local out_dir="$2"
  stop_server
  local log_file="${out_dir}/server_${kind}.log"
  : > "${log_file}"
  echo "[matrix] starting ${kind} server on GPUs ${CUDA_VISIBLE_DEVICES}, port ${PORT}"
  if [ "${kind}" = "scheduler" ]; then
    OMNI_TALKER_LEAD_MAX="${OMNI_TALKER_LEAD_MAX:-50}" \
    OMNI_TALKER_LEAD_LO="${OMNI_TALKER_LEAD_LO:-25}" \
    nohup bash "${REPO_ROOT}/scripts/H200/start_server_3gpu_video_scheduler.sh" >> "${log_file}" 2>&1 &
  else
    nohup bash "${REPO_ROOT}/scripts/H200/start_server_3gpu_video_baseline.sh" >> "${log_file}" 2>&1 &
  fi
  echo $! > "${PIDFILE}"
  wait_health "${log_file}"
  sleep 20
}

run_cell() {
  local tag="$1"
  local mode="$2"
  local concurrency="$3"
  local out_dir="${RESULT_BASE}/${tag}"
  mkdir -p "${out_dir}"
  echo "[matrix] driver ${tag}"
  "${PYBIN}" -u "${ROOT}/streamingbench_driver.py" \
    --ws-url "ws://127.0.0.1:${PORT}/v1/video/chat/stream" \
    --metadata "${METADATA}" \
    --data-root "${DATA_ROOT}" \
    --result-dir "${out_dir}" \
    --mode "${mode}" \
    --concurrency "${concurrency}" \
    --repeat "${REPEAT:-1}" \
    --max-videos "${MAX_VIDEOS:-4}" \
    --max-questions-per-video "${MAX_QUESTIONS_PER_VIDEO:-2}" \
    --fps "${STREAMINGBENCH_FPS:-1}" \
    --frames-per-query "${STREAMINGBENCH_FRAMES_PER_QUERY:-16}" \
    --num-sample-frames "${STREAMINGBENCH_NUM_SAMPLE_FRAMES:-8}" \
    --max-tokens "${STREAMINGBENCH_MAX_TOKENS:-24}" \
    ${WITH_AUDIO_OUTPUT:+--with-audio-output} \
    ${STREAM_SEGMENTS:+--stream-segments} \
    ${DRIVER_EXTRA_ARGS:-} \
    2>&1 | tee "${out_dir}/driver.log"
}

trap stop_server EXIT

echo "[matrix] RESULT_BASE=${RESULT_BASE}"
echo "[matrix] METADATA=${METADATA}"
echo "[matrix] DATA_ROOT=${DATA_ROOT}"

baseline_root="${RESULT_BASE}/_baseline_server"
mkdir -p "${baseline_root}"
start_server baseline "${baseline_root}"
for c in ${CONCURRENCIES}; do
  run_cell "baseline_no_bargein_c${c}" baseline "${c}"
  run_cell "baseline_bargein_c${c}" bargein "${c}"
done

scheduler_root="${RESULT_BASE}/_scheduler_server"
mkdir -p "${scheduler_root}"
start_server scheduler "${scheduler_root}"
for c in ${CONCURRENCIES}; do
  run_cell "scheduler_bargein_c${c}" bargein "${c}"
done

"${PYBIN}" "${ROOT}/plot_streamingbench.py" "${RESULT_BASE}"/*_c* --out-dir "${RESULT_BASE}/figures" || true
echo "[matrix] done: ${RESULT_BASE}"
