#!/usr/bin/env bash
# reproduce_interactive_baseline.sh — Run cells 1+2 of the 4-cell core matrix on H200.
#
# Cells:
#   1. baseline_c8_p0  — c=8, prob=0,   60s window. (Step 1 baseline)
#   2. naive_c8_p10    — c=8, prob=1.0, 60s window. (Step 1 naive barge-in)
#
# Prerequisites:
#   - The vllm-omni server must already be running with LiveServe scheduling disabled.
#     Start it in another terminal: bash scripts/H200/start_server_2gpu_baseline.sh
#
# Result dirs:
#   results/H200/step1/baseline_c8_p0/{summary,per_request,meta}.{json,jsonl}
#   results/H200/step1/naive_c8_p10/...
#
# Wall budget: ~30–35 min on 2x H200-96G.
#
# Smoke mode (--smoke): shrinks to 10s window, c=2 → ~5 min total.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
H200_DIR="${REPO_ROOT}/benchmarks/interactive/H200"
DRIVER="${H200_DIR}/trace_replay_driver.py"
PYBIN="${LIVESERVE_PYBIN:-python3}"
MODEL="${LIVESERVE_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"

SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "[reproduce_interactive_baseline] unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [ "${SMOKE}" -eq 1 ]; then
  TIME_START=6780; TIME_LIMIT=6790; CONC=2
  echo "[reproduce] SMOKE mode (10s window, c=2)"
else
  TIME_START=6780; TIME_LIMIT=6840; CONC=8
  echo "[reproduce] FULL mode (60s window, c=8)"
fi

# Sanity: server up?
if ! curl -s -o /dev/null -w '%{http_code}' -m 2 http://localhost:8666/health | grep -q 200; then
  echo "[fatal] server not healthy on :8666 — start it first via scripts/H200/start_server_2gpu_baseline.sh" >&2
  exit 3
fi

run_cell() {
  local prob="$1"
  local tag="$2"
  local outdir="${REPO_ROOT}/results/H200/step1/${tag}"
  mkdir -p "${outdir}"
  echo
  echo "==== ${tag} (prob=${prob}) → ${outdir} ===="
  "${PYBIN}" -u "${DRIVER}" \
    --mode smoke \
    --time-start "${TIME_START}" \
    --time-limit "${TIME_LIMIT}" \
    --max-inflight "${CONC}" \
    --with-audio \
    --policy fixed_time \
    --barge-in-after-seconds 2.0 \
    --barge-in-jitter-sec 0.5 \
    --barge-in-time-anchor ttfp \
    --barge-in-prob "${prob}" \
    --model "${MODEL}" \
    --progress-interval 300 \
    --result-dir "${outdir}"
}

# Cell 1: baseline (prob=0) - no barge-in.
run_cell 0   "baseline_c8_p0"

# Cell 2: naive (prob=1.0) - always barge-in.
run_cell 1.0 "naive_c8_p10"

echo
echo "[reproduce] DONE. Cells written to results/H200/step1/{baseline_c8_p0,naive_c8_p10}/"
echo "Validate against frozen reference:"
echo "  python scripts/diff_summary.py --runs results/H200 --reference results/reference --tolerance 0.15"
