#!/usr/bin/env bash
# 2-GPU high-concurrency capacity sweep for the H200/L20X lane.
#
# Purpose:
#   Keep the already-validated 2-GPU deployment fixed and increase
#   max_inflight to locate the local throughput plateau / latency cliff before
#   moving to 3/4/6-GPU deployments.
#
# Default matrix:
#   governor OFF, p_int=0, audio mode, trace window [6780, 6840),
#   c in {8,16,24,32,48,64,96,128,160}.
#
# Useful overrides:
#   CONCURRENCY_LIST="8 16 32" TIME_START=6780 TIME_LIMIT=6840 bash ...
#   RESULT_BASE=/path/to/results PORT=8666 bash ...
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../../.." && pwd)"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-${REPO_ROOT}/results/H200/capacity2gpu/${RUN_ID}}"
LOGDIR="${ROOT}/logs/capacity2gpu/${RUN_ID}"
mkdir -p "${RESULT_BASE}" "${LOGDIR}"

PYBIN="${LIVESERVE_PYBIN:-python3}"
VLLM_BIN="${LIVESERVE_VLLM_BIN:-$(command -v vllm-omni || true)}"
MODEL="${LIVESERVE_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
DEPLOY_YAML="${LIVESERVE_DEPLOY_YAML:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_multi_replicas_2gpu.yaml}"
DRIVER="${ROOT}/trace_replay_driver.py"

PORT="${PORT:-8666}"
TIME_START="${TIME_START:-6780}"
TIME_LIMIT="${TIME_LIMIT:-6840}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-8 16 24 32 48 64 96 128 160}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-300}"
START_SERVER="${START_SERVER:-1}"
STOP_SERVER="${STOP_SERVER:-1}"

HEALTH_URL="http://127.0.0.1:${PORT}/health"
API_URL="http://127.0.0.1:${PORT}/v1/chat/completions"
SERVER_LOG="${LOGDIR}/server_baseline.log"
SERVER_PID=""
STARTED_SERVER=0

stamp() { date -Is; }
log() { echo "[capacity2gpu $(stamp)] $*" | tee -a "${LOGDIR}/run.log"; }

health_code() {
  curl -s -o /dev/null -w '%{http_code}' -m 2 "${HEALTH_URL}" 2>/dev/null || true
}

is_healthy() {
  [ "$(health_code)" = "200" ]
}

cleanup() {
  if [ "${STARTED_SERVER}" -eq 1 ] && [ "${STOP_SERVER}" -eq 1 ] && [ -n "${SERVER_PID}" ]; then
    log "stopping server pid=${SERVER_PID}"
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "server stopped"
        return
      fi
      sleep 1
    done
    log "server still alive after 60s; leaving pid=${SERVER_PID} for manual inspection"
  fi
}
trap cleanup EXIT

if [ -z "${VLLM_BIN}" ]; then
  log "FATAL: vllm-omni binary not found; set LIVESERVE_VLLM_BIN or activate the env"
  exit 2
fi

log "RESULT_BASE=${RESULT_BASE}"
log "DEPLOY_YAML=${DEPLOY_YAML}"
log "MODEL=${MODEL}"
log "PORT=${PORT}"
log "CONCURRENCY_LIST=${CONCURRENCY_LIST}"
log "TRACE_WINDOW=[${TIME_START}, ${TIME_LIMIT})"

if is_healthy; then
  log "reusing existing healthy server on port ${PORT}"
elif [ "${START_SERVER}" -eq 1 ]; then
  log "starting 2GPU baseline server on port ${PORT}; log=${SERVER_LOG}"
  : > "${SERVER_LOG}"

  export PYTHONPATH="${REPO_ROOT}/vllm-omni${PYTHONPATH:+:${PYTHONPATH}}"
  unset OMNI_TALKER_LEAD_MAX OMNI_TALKER_LEAD_LO

  VLLM_ATTENTION_BACKEND=FLASHINFER \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  nohup "${PYBIN}" "${VLLM_BIN}" serve \
    "${MODEL}" \
    --omni \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --stage-init-timeout 1800 \
    --deploy-config "${DEPLOY_YAML}" \
    >> "${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  STARTED_SERVER=1
  echo "${SERVER_PID}" > "${LOGDIR}/server.pid"
  log "server pid=${SERVER_PID}; waiting for health"

  HEALTHY=0
  for i in $(seq 1 900); do
    if is_healthy; then
      HEALTHY=1
      log "server healthy after ${i}s"
      break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      log "FATAL: server died during boot; tail follows"
      tail -80 "${SERVER_LOG}" | tee -a "${LOGDIR}/run.log"
      exit 3
    fi
    sleep 1
  done
  if [ "${HEALTHY}" -ne 1 ]; then
    log "FATAL: server did not become healthy within 900s"
    tail -80 "${SERVER_LOG}" | tee -a "${LOGDIR}/run.log"
    exit 4
  fi

  log "warmup grace 30s"
  sleep 30
else
  log "FATAL: no healthy server on ${PORT}, and START_SERVER=0"
  exit 5
fi

RC_TOTAL=0
for C in ${CONCURRENCY_LIST}; do
  TAG="baseline_c${C}_p0_win${TIME_START}_${TIME_LIMIT}"
  OUTDIR="${RESULT_BASE}/${TAG}"
  CELL_LOG="${LOGDIR}/${TAG}.log"
  mkdir -p "${OUTDIR}"

  log "starting cell ${TAG}"
  "${PYBIN}" -u "${DRIVER}" \
    --url "${API_URL}" \
    --mode smoke \
    --time-start "${TIME_START}" \
    --time-limit "${TIME_LIMIT}" \
    --max-inflight "${C}" \
    --with-audio \
    --policy fixed_time \
    --interrupt-after-seconds 2.0 \
    --interrupt-jitter-sec 0.5 \
    --interrupt-time-anchor ttfp \
    --interrupt-prob 0 \
    --model "${MODEL}" \
    --progress-interval "${PROGRESS_INTERVAL}" \
    --result-dir "${OUTDIR}" \
    >> "${CELL_LOG}" 2>&1
  RC=$?
  log "cell ${TAG} exit=${RC}"
  if [ "${RC}" -ne 0 ]; then
    RC_TOTAL=${RC}
    tail -80 "${CELL_LOG}" | tee -a "${LOGDIR}/run.log"
  fi
done

if [ -x "${PYBIN}" ]; then
  log "aggregate summary"
  "${PYBIN}" "${ROOT}/aggregate_capacity.py" "${RESULT_BASE}" | tee "${RESULT_BASE}/capacity_table.txt"
fi

log "DONE rc=${RC_TOTAL}; results=${RESULT_BASE}"
exit "${RC_TOTAL}"
