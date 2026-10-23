#!/usr/bin/env bash
# Step 2 - scheduler lead-control sweep, single-cell runner.
#
# Restarts vllm-omni with OMNI_TALKER_LEAD_MAX / OMNI_TALKER_LEAD_LO env vars,
# verifies the scheduler lead controller entered the "enabled" state, then runs
# trace_replay_driver with the same args as Step 1 chain B (c=8, prob=1.0,
# audio, fixed_time @ttfp+2s, jitter 0.5s, 60s window, seed=42).
#
# Usage:
#   ./scheduler_run_one_cell.sh <L_HI> <L_LO> <TAG> [PROB]
# e.g.
#   ./scheduler_run_one_cell.sh 30 15 scheduler_mid          # default prob=1.0
#   ./scheduler_run_one_cell.sh 50 25 scheduler_wide_p07 0.7 # prob=0.7
#
# Outputs:
#   results/step2/<TAG>/{summary,per_request,meta}.{json,jsonl}  (driver)
#   results/step2/<TAG>/server_init.log                          (server boot)
#   results/step2/<TAG>/governor_init.txt                        (lead-controller line)
#   logs/step2/<TAG>.log                                         (driver stdout)
#
# Per-cell wall: ~2 min (server restart) + ~18 min (driver) ≈ 20 min.
set -uo pipefail

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
  echo "usage: $0 <L_HI> <L_LO> <TAG> [PROB]" >&2
  echo "  e.g. $0 30 15 scheduler_mid          (prob=1.0)" >&2
  echo "       $0 50 25 scheduler_wide_p07 0.7 (prob=0.7)" >&2
  exit 2
fi

L_HI="$1"
L_LO="$2"
TAG="$3"
PROB="${4:-1.0}"

# ---- paths (resolved relative to this script's directory) ----
# This script lives at benchmarks/interactive/H200/scheduler_run_one_cell.sh.
# ROOT       → benchmarks/interactive/H200/
# REPO_ROOT  → LiveServe/
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../../.." && pwd)"
LOGDIR="${ROOT}/logs/step2"
RESBASE="${REPO_ROOT}/results/H200/step2"
OUTDIR="${RESBASE}/${TAG}"
DRV_LOG="${LOGDIR}/${TAG}.log"
SRV_LOG_NEW="/tmp/server_scheduler_${TAG}.log"
mkdir -p "${LOGDIR}" "${OUTDIR}"

# Allow overrides via env vars (useful when porting to other hosts / venvs):
#   LIVESERVE_PYBIN    — python interpreter (default: python3 from PATH, expected to be the activated .venv-liveserve)
#   LIVESERVE_VLLM_BIN — vllm-omni serve binary (default: vllm-omni from PATH, expected to be the activated venv's)
#   LIVESERVE_DEPLOY_YAML — deploy yaml (default: deploy/H200/qwen3_omni_moe_multi_replicas_2gpu.yaml)
#   LIVESERVE_MODEL    — HF model id (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)
PYTHON="${LIVESERVE_PYBIN:-python3}"
DRIVER="${ROOT}/trace_replay_driver.py"
DEPLOY_YAML="${LIVESERVE_DEPLOY_YAML:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_multi_replicas_2gpu.yaml}"
VLLM_BIN="${LIVESERVE_VLLM_BIN:-vllm-omni}"
PYBIN="${LIVESERVE_PYBIN:-python3}"
MODEL="${LIVESERVE_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"

# ---- common driver args (identical to Step 1 chain B prob=1.0) ----
COMMON=(
  --mode smoke
  --time-start 6780
  --time-limit 6840
  --max-inflight 8
  --with-audio
  --policy fixed_time
  --barge-in-after-seconds 2.0
  --barge-in-jitter-sec 0.5
  --barge-in-time-anchor ttfp
  --barge-in-prob "${PROB}"
  --model "${MODEL}"
  --progress-interval 300
)

stamp() { date -Is; }
log() { echo "[scheduler $(stamp)] $*" | tee -a "${DRV_LOG}"; }

# ---- step 1: kill running server ----
OLD_PID=$(pgrep -f 'vllm-omni serve' | head -1 || true)
if [ -n "${OLD_PID}" ]; then
  log "killing old vllm-omni pid=${OLD_PID}"
  kill -TERM "${OLD_PID}" || true
  for i in $(seq 1 30); do
    if ! kill -0 "${OLD_PID}" 2>/dev/null; then
      log "old server stopped after ${i}s"
      break
    fi
    sleep 1
  done
  if kill -0 "${OLD_PID}" 2>/dev/null; then
    log "force kill ${OLD_PID}"
    kill -9 "${OLD_PID}" || true
    sleep 3
  fi
  # extra grace for child stage processes / GPU mem release
  sleep 5
else
  log "no running vllm-omni found"
fi

# ---- step 2: launch new server with scheduler lead-control env ----
log "starting new server: L_HI=${L_HI} L_LO=${L_LO} PROB=${PROB} log=${SRV_LOG_NEW}"
: > "${SRV_LOG_NEW}"
# Server-launch convention:
#   - With .venv-liveserve activated, `vllm-omni` is on PATH and `python3 $(which vllm-omni) serve ...`
#     works on both H20 and H200. Override LIVESERVE_VLLM_BIN to a fully-qualified path if needed.
if [ -z "${LIVESERVE_VLLM_BIN:-}" ] && command -v vllm-omni >/dev/null 2>&1; then
  VLLM_BIN="$(command -v vllm-omni)"
fi
log "VLLM_BIN=${VLLM_BIN} PYBIN=${PYBIN} DEPLOY_YAML=${DEPLOY_YAML}"

# Prefer the LiveServe submodule over any editable install already present in
# the activated environment.
export PYTHONPATH="${REPO_ROOT}/vllm-omni${PYTHONPATH:+:${PYTHONPATH}}"

OMNI_TALKER_LEAD_MAX="${L_HI}" \
OMNI_TALKER_LEAD_LO="${L_LO}" \
VLLM_ATTENTION_BACKEND=FLASHINFER \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nohup "${PYBIN}" "${VLLM_BIN}" serve \
  "${MODEL}" \
  --omni \
  --port 8666 \
  --host 0.0.0.0 \
  --stage-init-timeout 1800 \
  --deploy-config "${DEPLOY_YAML}" \
  >> "${SRV_LOG_NEW}" 2>&1 &
NEW_PID=$!
log "new server pid=${NEW_PID}"

# ---- step 3: wait for health (max 5 min) ----
log "waiting for /health 200 ..."
HEALTHY=0
for i in $(seq 1 300); do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 2 http://localhost:8666/health 2>/dev/null || echo 000)
  if [ "${CODE}" = "200" ]; then
    log "server healthy after ${i}s"
    HEALTHY=1
    break
  fi
  if ! kill -0 "${NEW_PID}" 2>/dev/null; then
    log "FATAL: server pid ${NEW_PID} died during boot, see ${SRV_LOG_NEW}"
    tail -50 "${SRV_LOG_NEW}" | tee -a "${DRV_LOG}"
    exit 3
  fi
  sleep 1
done
if [ "${HEALTHY}" -ne 1 ]; then
  log "FATAL: server did not become healthy in 300s"
  tail -50 "${SRV_LOG_NEW}" | tee -a "${DRV_LOG}"
  exit 4
fi

# ---- step 4: verify scheduler lead controller enabled ----
GOV_LINE=$(grep -m1 'TalkerLeadGovernor enabled' "${SRV_LOG_NEW}" || true)
if [ -z "${GOV_LINE}" ]; then
  log "FATAL: 'TalkerLeadGovernor enabled' not found in server.log; lead control did not engage"
  grep -m5 -E 'OMNI_TALKER|TalkerLead|governor' "${SRV_LOG_NEW}" | tee -a "${DRV_LOG}"
  exit 5
fi
log "lead controller confirmed: ${GOV_LINE}"
echo "${GOV_LINE}" > "${OUTDIR}/governor_init.txt"

# archive boot log (first 1MB is plenty)
head -c 1048576 "${SRV_LOG_NEW}" > "${OUTDIR}/server_init.log"

# warmup grace — let prefill model warmup quiesce
log "warmup grace 30s ..."
sleep 30

# ---- step 5: run driver ----
log "running trace_replay_driver -> ${OUTDIR}"
${PYTHON} -u "${DRIVER}" \
    "${COMMON[@]}" \
    --result-dir "${OUTDIR}" \
    >> "${DRV_LOG}" 2>&1
RC=$?
log "driver exit=${RC}"

# tail server.log into result for debugging scheduler lead-control stats post-run
tail -c 524288 "${SRV_LOG_NEW}" > "${OUTDIR}/server_tail.log" 2>/dev/null || true

# ---- summary ----
if [ "${RC}" -eq 0 ] && [ -s "${OUTDIR}/summary.json" ]; then
  log "DONE ${TAG} OK"
  log "summary highlights:"
  # NOTE: trace_replay_driver writes a nested schema: totals.*, latency.*.{p50,p90}, interrupt.*.
  # The interrupt node name is a legacy result-schema field for barge-in metrics.
  # Latency keys are in *seconds*; convert to ms for headline display.
  ${PYTHON} -c "
import json
s = json.load(open('${OUTDIR}/summary.json'))
totals    = s.get('totals',    {})
latency   = s.get('latency',   {})
interrupt = s.get('interrupt', {})
def g(node, path, scale=1.0, default=None):
    cur = node
    for seg in path.split('.'):
        if not isinstance(cur, dict) or seg not in cur: return default
        cur = cur[seg]
    return cur * scale if isinstance(cur,(int,float)) else default
print(f'  wall_rps                 = {totals.get(\"throughput_rps\")}')
print(f'  useful_qps               = {totals.get(\"useful_qps\")}')
print(f'  barge_in_count           = {totals.get(\"interrupted\")}')
print(f'  wall_time_sec            = {totals.get(\"wall_time_sec\")}')
print(f'  audio_ttfp_p50_ms        = {g(latency, \"audio_ttfp_sec.p50\", 1000.0)}')
print(f'  audio_ttfp_p90_ms        = {g(latency, \"audio_ttfp_sec.p90\", 1000.0)}')
print(f'  e2e_p50_ms               = {g(latency, \"e2e_sec.p50\", 1000.0)}')
print(f'  e2e_p90_ms               = {g(latency, \"e2e_sec.p90\", 1000.0)}')
print(f'  wasted_talker_frames_p50 = {g(interrupt, \"wasted_talker_frames_total.p50\")}')
print(f'  wasted_talker_frames_p90 = {g(interrupt, \"wasted_talker_frames_total.p90\")}')
print(f'  sanity_ratio             = {s.get(\"sanity_check\",{}).get(\"ratio_expected_over_actual\")}')
" 2>&1 | tee -a "${DRV_LOG}"
else
  log "FAIL ${TAG} rc=${RC}"
fi

exit ${RC}
