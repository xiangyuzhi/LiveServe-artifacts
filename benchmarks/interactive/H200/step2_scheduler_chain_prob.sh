#!/usr/bin/env bash
# Step 2 Phase 2 - scheduler barge-in probability sweep:
#   Use best operating point from Phase 1 (wide = L_hi=50 / L_lo=25),
#   sweep barge-in probability in {0.3, 0.5, 0.7} to verify whether useful_qps
#   recovers to >= OFF baseline (1.21) at moderate barge-in rates.
#
# Goal: answer the archived caveat: at prob=1.0, the wide scheduler useful_qps=1.00 is below
# OFF baseline 1.21. Does it recover when prob<1.0?
#
# Per-cell wall ~ 14min (server restart + driver). Total ~42min.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${ROOT}/scheduler_run_one_cell.sh"
LOGDIR="${ROOT}/logs/step2"
mkdir -p "${LOGDIR}"
CHAIN_LOG="${LOGDIR}/chain_prob.log"

run_cell() {
  local L_HI="$1"
  local L_LO="$2"
  local TAG="$3"
  local PROB="$4"
  echo "============================================================" | tee -a "${CHAIN_LOG}"
  echo "[chain-prob $(date -Is)] starting ${TAG} (L_hi=${L_HI} L_lo=${L_LO} prob=${PROB})" | tee -a "${CHAIN_LOG}"
  echo "============================================================" | tee -a "${CHAIN_LOG}"
  bash "${RUNNER}" "${L_HI}" "${L_LO}" "${TAG}" "${PROB}"
  local rc=$?
  echo "[chain-prob $(date -Is)] ${TAG} exit=${rc}" | tee -a "${CHAIN_LOG}"
  return ${rc}
}

# Order: low prob -> high prob, all using wide (50/25)
run_cell 50 25 scheduler_wide_p03 0.3 ; rc1=$?
run_cell 50 25 scheduler_wide_p05 0.5 ; rc2=$?
run_cell 50 25 scheduler_wide_p07 0.7 ; rc3=$?

echo "[chain-prob $(date -Is)] DONE p03=${rc1} p05=${rc2} p07=${rc3}" | tee -a "${CHAIN_LOG}"
