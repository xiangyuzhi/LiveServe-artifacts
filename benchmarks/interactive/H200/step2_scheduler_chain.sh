#!/usr/bin/env bash
# Step 2 - scheduler lead-control sweep chain: wide -> tight -> very-tight.
#
# Each cell: kill server, restart with new (L_HI, L_LO), verify lead control
# enabled, run trace_replay_driver (c=8, prob=1.0, audio, ttfp+2s, 60s, seed=42),
# archive summary + lead-controller init + server_init/tail logs.
#
# Per-cell wall ~12min (~3min restart + ~9-11min driver). Total ~36-40min.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${ROOT}/scheduler_run_one_cell.sh"
LOGDIR="${ROOT}/logs/step2"
mkdir -p "${LOGDIR}"
CHAIN_LOG="${LOGDIR}/chain.log"

run_cell() {
  local L_HI="$1"
  local L_LO="$2"
  local TAG="$3"
  echo "============================================================" | tee -a "${CHAIN_LOG}"
  echo "[chain $(date -Is)] starting ${TAG} (L_hi=${L_HI} L_lo=${L_LO})" | tee -a "${CHAIN_LOG}"
  echo "============================================================" | tee -a "${CHAIN_LOG}"
  bash "${RUNNER}" "${L_HI}" "${L_LO}" "${TAG}"
  local rc=$?
  echo "[chain $(date -Is)] ${TAG} exit=${rc}" | tee -a "${CHAIN_LOG}"
  return ${rc}
}

run_cell 50 25 scheduler_wide       ; rc1=$?
run_cell 20 10 scheduler_tight      ; rc2=$?
run_cell 12  6 scheduler_very_tight ; rc3=$?

echo "[chain $(date -Is)] DONE wide=${rc1} tight=${rc2} very_tight=${rc3}" | tee -a "${CHAIN_LOG}"
