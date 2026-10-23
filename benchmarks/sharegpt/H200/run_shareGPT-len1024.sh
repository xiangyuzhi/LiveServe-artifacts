#!/usr/bin/env bash
# TokenFlow-like ShareGPT length-aligned workload.
#
# This wrapper assumes a vllm-omni server is already running. It only controls
# the client workload shape: synthetic ShareGPT-style prompts with normal
# input/output length targets centered at 512/1024 tokens and Poisson arrivals.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../../.." && pwd)"

PYBIN="${LIVESERVE_PYBIN:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8666}"
MODEL="${LIVESERVE_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
DATASET="${LIVESERVE_SHAREGPT_DATASET:-${REPO_ROOT}/data/ShareGPT_Chinese_English_90k.json}"
DRIVER="${ROOT}/sharegpt_driver.py"

RUN_ID="${RUN_ID:-shareGPT-len1024_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/results/${RUN_ID}}"
MODE="${MODE:-text}"
NUM_REQUESTS="${NUM_REQUESTS:-128}"
MAX_INFLIGHT="${MAX_INFLIGHT:-64}"
REQUEST_RATE="${REQUEST_RATE:-5}"
INPUT_MEAN="${INPUT_MEAN:-512}"
INPUT_STD="${INPUT_STD:-128}"
OUTPUT_MEAN="${OUTPUT_MEAN:-1024}"
OUTPUT_STD="${OUTPUT_STD:-256}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"
SEED="${SEED:-42}"
FORCE_OUTPUT_LEN="${FORCE_OUTPUT_LEN:-0}"
ARRIVAL_PROCESS="${ARRIVAL_PROCESS:-poisson}"
ARRIVAL_MANIFEST="${ARRIVAL_MANIFEST:-}"
ARRIVAL_TIME_SCALE="${ARRIVAL_TIME_SCALE:-1.0}"
BARGE_IN_PROB="${BARGE_IN_PROB:-${INTERRUPT_PROB:-0.0}}"
INTERRUPT_PROB="${INTERRUPT_PROB:-${BARGE_IN_PROB}}"
POLICY="${POLICY:-}"
if [[ -z "${POLICY}" ]]; then
  if [[ "${BARGE_IN_PROB}" == "0" || "${BARGE_IN_PROB}" == "0.0" || "${BARGE_IN_PROB}" == "0.00" ]]; then
    POLICY="none"
  else
    POLICY="fixed_time"
  fi
fi
BARGE_IN_AFTER_SEC="${BARGE_IN_AFTER_SEC:-${INTERRUPT_AFTER_SEC:-2.0}}"
BARGE_IN_JITTER_SEC="${BARGE_IN_JITTER_SEC:-${INTERRUPT_JITTER_SEC:-0.5}}"
BARGE_IN_TIME_ANCHOR="${BARGE_IN_TIME_ANCHOR:-${INTERRUPT_TIME_ANCHOR:-ttfp}}"

force_args=()
if [[ "${FORCE_OUTPUT_LEN}" == "1" ]]; then
  force_args+=(--force-output-len)
fi

arrival_args=(--arrival-process "${ARRIVAL_PROCESS}")
if [[ "${ARRIVAL_PROCESS}" == "poisson" ]]; then
  arrival_args+=(--request-rate "${REQUEST_RATE}")
elif [[ "${ARRIVAL_PROCESS}" == "manifest" ]]; then
  if [[ -z "${ARRIVAL_MANIFEST}" ]]; then
    echo "FATAL: ARRIVAL_PROCESS=manifest requires ARRIVAL_MANIFEST" >&2
    exit 1
  fi
  arrival_args+=(--arrival-manifest "${ARRIVAL_MANIFEST}" --arrival-time-scale "${ARRIVAL_TIME_SCALE}")
fi

interrupt_args=(--policy "${POLICY}")
if [[ "${POLICY}" != "none" ]]; then
  interrupt_args+=(
    --barge-in-prob "${BARGE_IN_PROB}"
    --barge-in-after-seconds "${BARGE_IN_AFTER_SEC}"
    --barge-in-jitter-sec "${BARGE_IN_JITTER_SEC}"
    --barge-in-time-anchor "${BARGE_IN_TIME_ANCHOR}"
  )
fi

mkdir -p "${RESULT_DIR}"

if ! curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "FATAL: server is not healthy at http://${HOST}:${PORT}/health" >&2
  exit 1
fi

echo "[shareGPT-len1024] result_dir=${RESULT_DIR}"
echo "[shareGPT-len1024] mode=${MODE} n=${NUM_REQUESTS} arrival=${ARRIVAL_PROCESS} rate=${REQUEST_RATE}/s max_inflight=${MAX_INFLIGHT}"
echo "[shareGPT-len1024] target input/output=${INPUT_MEAN}/${OUTPUT_MEAN} tokens"
echo "[shareGPT-len1024] policy=${POLICY} barge_in_prob=${BARGE_IN_PROB}"

"${PYBIN}" -u "${DRIVER}" \
  --dataset "${DATASET}" \
  --url "http://${HOST}:${PORT}/v1/chat/completions" \
  --model "${MODEL}" \
  --mode "${MODE}" \
  --num-requests "${NUM_REQUESTS}" \
  --concurrency "${MAX_INFLIGHT}" \
  --max-inflight "${MAX_INFLIGHT}" \
  --length-profile shareGPT-len1024 \
  --prompt-source synthetic \
  --input-tokens-mean "${INPUT_MEAN}" \
  --input-tokens-std "${INPUT_STD}" \
  --output-tokens-mean "${OUTPUT_MEAN}" \
  --output-tokens-std "${OUTPUT_STD}" \
  "${arrival_args[@]}" \
  --timeout "${TIMEOUT_SEC}" \
  --seed "${SEED}" \
  --tag shareGPT-len1024 \
  "${force_args[@]}" \
  "${interrupt_args[@]}" \
  --result-dir "${RESULT_DIR}"
