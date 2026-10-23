#!/usr/bin/env bash
# Step 1 - barge-in sweep batch A:
#   policy=fixed_time, c=4, win=60s, prob in {0.3, 0.7, 1.0}
# All 3 cells run sequentially against the live vllm-omni server on :8666.
# Per-cell expected wall ~ 1940s (= prob=0 baseline). Total ~ 1h45min.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../../.." && pwd)"
LOGDIR="${ROOT}/logs/step1"
RESBASE="${REPO_ROOT}/results/H200/step1"
mkdir -p "${LOGDIR}" "${RESBASE}"

PYTHON="python3"
DRIVER="${ROOT}/trace_replay_driver.py"
MODEL="${LIVESERVE_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"

# Common args (must match prob=0 baseline `v2_audio_c4_60s_v2`)
COMMON=(
  --mode smoke
  --time-start 6780
  --time-limit 6840           # 60s window
  --max-inflight 4
  --with-audio
  --policy fixed_time
  --barge-in-after-seconds 2.0
  --barge-in-jitter-sec 0.5
  --barge-in-time-anchor ttfp
  --progress-interval 300
)

run_cell() {
  local prob="$1"
  local tag="audio_c4_60s_prob${prob//./}"   # prob0.3 -> prob03
  local outdir="${RESBASE}/${tag}"
  local logfile="${LOGDIR}/${tag}.log"
  echo "==========================================" | tee -a "${logfile}"
  echo "[chain] starting ${tag} at $(date -Is)"   | tee -a "${logfile}"
  echo "==========================================" | tee -a "${logfile}"
  ${PYTHON} -u "${DRIVER}" \
      "${COMMON[@]}" \
      --barge-in-prob "${prob}" \
      --model "${MODEL}" \
      --result-dir "${outdir}" \
      >> "${logfile}" 2>&1
  local rc=$?
  echo "[chain] ${tag} exit=${rc} at $(date -Is)" | tee -a "${logfile}"
  return ${rc}
}

# Run sequentially. Continue even if one fails so we still get partial data.
run_cell 0.3 ; rc1=$?
run_cell 0.7 ; rc2=$?
run_cell 1.0 ; rc3=$?

echo "[chain] DONE prob03=${rc1} prob07=${rc2} prob10=${rc3}"
