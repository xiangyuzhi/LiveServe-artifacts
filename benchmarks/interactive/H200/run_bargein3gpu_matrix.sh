#!/usr/bin/env bash
# 3-GPU barge-in matrix for the H200/L20X lane.
#
# Matrix:
#   systems: naive/off, scheduler wide(50/25), scheduler tight(20/10)
#   max_inflight: 32, 48, 64, 96
#   barge-in probability: 0.3, 0.5, 0.7, 0.9
#
# The script keeps one server alive per system and sweeps all c/prob cells
# before moving to the next system.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../../.." && pwd)"

RUN_ID="${RUN_ID:-matrix_3gpu_$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-${REPO_ROOT}/results/H200/bargein3gpu/${RUN_ID}}"
LOGDIR="${ROOT}/logs/bargein3gpu/${RUN_ID}"
mkdir -p "${RESULT_BASE}" "${LOGDIR}"

PYBIN="${LIVESERVE_PYBIN:-python3}"
VLLM_BIN="${LIVESERVE_VLLM_BIN:-$(command -v vllm-omni || true)}"
MODEL="${LIVESERVE_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
DEPLOY_YAML="${LIVESERVE_DEPLOY_YAML:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_multi_replicas_3gpu.yaml}"
DRIVER="${ROOT}/trace_replay_driver.py"

PORT="${PORT:-8666}"
TIME_START="${TIME_START:-6780}"
TIME_LIMIT="${TIME_LIMIT:-6840}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-32 48 64 96}"
PROB_LIST="${BARGE_IN_PROB_LIST:-${PROB_LIST:-0.3 0.5 0.7 0.9}}"
SYSTEM_LIST="${SYSTEM_LIST:-naive scheduler_wide scheduler_tight}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-300}"
STOP_SERVER="${STOP_SERVER:-1}"
RECORD_AUDIO_TIMELINE="${RECORD_AUDIO_TIMELINE:-0}"

HEALTH_URL="http://127.0.0.1:${PORT}/health"
API_URL="http://127.0.0.1:${PORT}/v1/chat/completions"
SERVER_PID=""
STARTED_SERVER=0
CURRENT_SYSTEM=""

stamp() { date -Is; }
log() { echo "[bargein3gpu $(stamp)] $*" | tee -a "${LOGDIR}/run.log"; }

prob_tag() {
  local p="$1"
  python3 - "$p" <<'PY'
import sys
p = float(sys.argv[1])
print(f"p{int(round(p * 100)):02d}")
PY
}

health_code() {
  curl -s -o /dev/null -w '%{http_code}' -m 2 "${HEALTH_URL}" 2>/dev/null || true
}

is_healthy() {
  [ "$(health_code)" = "200" ]
}

kill_existing_server() {
  local old
  old="$(pgrep -f 'vllm-omni serve' | head -1 || true)"
  if [ -z "${old}" ]; then
    return
  fi
  log "killing old vllm-omni pid=${old}"
  kill -TERM "${old}" 2>/dev/null || true
  for i in $(seq 1 90); do
    if ! kill -0 "${old}" 2>/dev/null; then
      log "old server stopped after ${i}s"
      sleep 5
      return
    fi
    sleep 1
  done
  log "force killing old server pid=${old}"
  kill -9 "${old}" 2>/dev/null || true
  sleep 8
}

stop_current_server() {
  if [ "${STARTED_SERVER}" -eq 1 ] && [ "${STOP_SERVER}" -eq 1 ] && [ -n "${SERVER_PID}" ]; then
    log "stopping ${CURRENT_SYSTEM} server pid=${SERVER_PID}"
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 120); do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "${CURRENT_SYSTEM} server stopped"
        SERVER_PID=""
        STARTED_SERVER=0
        sleep 8
        return
      fi
      sleep 1
    done
    log "server still alive after 120s; force killing pid=${SERVER_PID}"
    kill -9 "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
    STARTED_SERVER=0
    sleep 8
  fi
}

cleanup() {
  stop_current_server
}
trap cleanup EXIT

scheduler_config() {
  case "$1" in
    scheduler_wide) echo "50 25" ;;
    scheduler_tight) echo "20 10" ;;
    *) echo "" ;;
  esac
}

start_server_for_system() {
  local system="$1"
  local server_log="${LOGDIR}/server_${system}.log"
  local cfg l_hi l_lo
  CURRENT_SYSTEM="${system}"

  kill_existing_server
  : > "${server_log}"
  echo "${system}" > "${RESULT_BASE}/current_system.txt"

  export PYTHONPATH="${REPO_ROOT}/vllm-omni${PYTHONPATH:+:${PYTHONPATH}}"

  if [ "${system}" = "naive" ]; then
    unset OMNI_TALKER_LEAD_MAX OMNI_TALKER_LEAD_LO
    log "starting naive/off 3GPU server; log=${server_log}"
    VLLM_ATTENTION_BACKEND=FLASHINFER \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    nohup "${PYBIN}" "${VLLM_BIN}" serve \
      "${MODEL}" \
      --omni \
      --port "${PORT}" \
      --host 0.0.0.0 \
      --stage-init-timeout 1800 \
      --deploy-config "${DEPLOY_YAML}" \
      >> "${server_log}" 2>&1 &
  else
    cfg="$(scheduler_config "${system}")"
    if [ -z "${cfg}" ]; then
      log "FATAL: unknown system=${system}"
      exit 2
    fi
    read -r l_hi l_lo <<< "${cfg}"
    log "starting ${system} 3GPU server L_hi=${l_hi} L_lo=${l_lo}; log=${server_log}"
    OMNI_TALKER_LEAD_MAX="${l_hi}" \
    OMNI_TALKER_LEAD_LO="${l_lo}" \
    VLLM_ATTENTION_BACKEND=FLASHINFER \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    nohup "${PYBIN}" "${VLLM_BIN}" serve \
      "${MODEL}" \
      --omni \
      --port "${PORT}" \
      --host 0.0.0.0 \
      --stage-init-timeout 1800 \
      --deploy-config "${DEPLOY_YAML}" \
      >> "${server_log}" 2>&1 &
  fi

  SERVER_PID=$!
  STARTED_SERVER=1
  echo "${SERVER_PID}" > "${LOGDIR}/server_${system}.pid"
  log "server pid=${SERVER_PID}; waiting for health"

  local healthy=0
  for i in $(seq 1 1200); do
    if is_healthy; then
      healthy=1
      log "${system} server healthy after ${i}s"
      break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      log "FATAL: ${system} server died during boot; tail follows"
      tail -120 "${server_log}" | tee -a "${LOGDIR}/run.log"
      exit 3
    fi
    sleep 1
  done
  if [ "${healthy}" -ne 1 ]; then
    log "FATAL: ${system} server did not become healthy within 1200s"
    tail -120 "${server_log}" | tee -a "${LOGDIR}/run.log"
    exit 4
  fi

  if [ "${system}" != "naive" ]; then
    local gov_line
    gov_line="$(grep -m1 'TalkerLeadGovernor enabled' "${server_log}" || true)"
    if [ -z "${gov_line}" ]; then
      log "FATAL: scheduler lead control not enabled for ${system}"
      grep -m5 -E 'OMNI_TALKER|TalkerLead|governor' "${server_log}" | tee -a "${LOGDIR}/run.log" || true
      exit 5
    fi
    echo "${gov_line}" > "${RESULT_BASE}/governor_${system}.txt"
    log "lead controller confirmed: ${gov_line}"
  fi

  log "warmup grace 30s"
  sleep 30
}

run_cell() {
  local system="$1"
  local c="$2"
  local prob="$3"
  local ptag
  ptag="$(prob_tag "${prob}")"
  local tag="${system}_c${c}_${ptag}_win${TIME_START}_${TIME_LIMIT}"
  local outdir="${RESULT_BASE}/${tag}"
  local cell_log="${LOGDIR}/${tag}.log"
  local timeline_args=()
  mkdir -p "${outdir}"
  if [ "${RECORD_AUDIO_TIMELINE}" = "1" ]; then
    timeline_args+=(--record-audio-timeline)
  fi

  log "starting cell ${tag}"
  "${PYBIN}" -u "${DRIVER}" \
    --url "${API_URL}" \
    --mode smoke \
    --time-start "${TIME_START}" \
    --time-limit "${TIME_LIMIT}" \
    --max-inflight "${c}" \
    --with-audio \
    --policy fixed_time \
    --barge-in-after-seconds 2.0 \
    --barge-in-jitter-sec 0.5 \
    --barge-in-time-anchor ttfp \
    --barge-in-prob "${prob}" \
    --model "${MODEL}" \
    --progress-interval "${PROGRESS_INTERVAL}" \
    "${timeline_args[@]}" \
    --result-dir "${outdir}" \
    >> "${cell_log}" 2>&1
  local rc=$?
  log "cell ${tag} exit=${rc}"
  if [ "${rc}" -ne 0 ]; then
    tail -120 "${cell_log}" | tee -a "${LOGDIR}/run.log"
  fi
  return "${rc}"
}

main() {
  if [ -z "${VLLM_BIN}" ]; then
    log "FATAL: vllm-omni binary not found; set LIVESERVE_VLLM_BIN or activate the env"
    exit 2
  fi
  if [ ! -f "${DEPLOY_YAML}" ]; then
    log "FATAL: deploy yaml not found: ${DEPLOY_YAML}"
    exit 2
  fi

  log "RESULT_BASE=${RESULT_BASE}"
  log "DEPLOY_YAML=${DEPLOY_YAML}"
  log "MODEL=${MODEL}"
  log "PORT=${PORT}"
  log "SYSTEM_LIST=${SYSTEM_LIST}"
  log "CONCURRENCY_LIST=${CONCURRENCY_LIST}"
  log "PROB_LIST=${PROB_LIST}"
  log "RECORD_AUDIO_TIMELINE=${RECORD_AUDIO_TIMELINE}"
  log "TRACE_WINDOW=[${TIME_START}, ${TIME_LIMIT})"

  local rc_total=0
  for system in ${SYSTEM_LIST}; do
    start_server_for_system "${system}"
    for c in ${CONCURRENCY_LIST}; do
      for prob in ${PROB_LIST}; do
        run_cell "${system}" "${c}" "${prob}" || rc_total=$?
      done
    done
    stop_current_server
  done

  log "DONE rc=${rc_total}; results=${RESULT_BASE}"
  exit "${rc_total}"
}

main "$@"
