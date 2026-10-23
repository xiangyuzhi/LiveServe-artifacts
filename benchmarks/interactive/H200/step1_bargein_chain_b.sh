#!/usr/bin/env bash
# Step 1 - barge-in sweep batch B:
#   policy=fixed_time, c=8, win=60s, prob in {1.0}  (core data point)
#   Optionally extend to prob=0.7 for trend (uncomment last run_cell line).
#
# Goal: answer §14.8-3 / §15.9 — does the talker-ceiling-breakthrough seen at
# c=4 (rps 0.90 -> 1.43, +59%) survive at c=8, or do we hit a thinker-prefill
# wall? The prob=0 baseline at c=8 already exists in
#   results/step1/audio_c8_60s_prob0
# (rps ~ 1.21, sweet spot per §14), so prob=1.0 here directly compares.
#
# Per-cell expected wall ~ 12-18 min (lower than c=4 baseline 1940s because
# c=8 finishes faster AND prob=1.0 cuts each request short ~2s after ttfp).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../../.." && pwd)"
LOGDIR="${ROOT}/logs/step1"
RESBASE="${REPO_ROOT}/results/H200/step1"
mkdir -p "${LOGDIR}" "${RESBASE}"

PYTHON="python3"
DRIVER="${ROOT}/trace_replay_driver.py"
MODEL="${LIVESERVE_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"

# Common args — c=8, audio mode, fixed_time @ttfp+2s, jitter 0.5s, seed=42
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
  --progress-interval 300
)

run_cell() {
  local prob="$1"
  local tag="audio_c8_60s_prob${prob//./}"   # 1.0 -> prob10
  local outdir="${RESBASE}/${tag}"
  local logfile="${LOGDIR}/${tag}.log"
  echo "==========================================" | tee -a "${logfile}"
  echo "[chain-b] starting ${tag} at $(date -Is)" | tee -a "${logfile}"
  echo "==========================================" | tee -a "${logfile}"
  ${PYTHON} -u "${DRIVER}" \
      "${COMMON[@]}" \
      --barge-in-prob "${prob}" \
      --model "${MODEL}" \
      --result-dir "${outdir}" \
      >> "${logfile}" 2>&1
  local rc=$?
  echo "[chain-b] ${tag} exit=${rc} at $(date -Is)" | tee -a "${logfile}"
  return ${rc}
}

# Core: prob=1.0 is the breakthrough comparison point
run_cell 1.0 ; rc1=$?

# Optional trend point — uncomment if rc1==0 and you want to extend
# run_cell 0.7 ; rc2=$?

echo "[chain-b] DONE prob10=${rc1}"
