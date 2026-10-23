#!/usr/bin/env bash
# Interactive trace replay with TokenFlow-like length alignment.
#
# This keeps the original interactive trace timing/session structure, but
# resamples query/response lengths from normal distributions centered at
# 512/1024 tokens. The server must already be running.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYBIN="${LIVESERVE_PYBIN:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8666}"
MODEL="${LIVESERVE_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
DRIVER="${ROOT}/trace_replay_driver.py"

RUN_ID="${RUN_ID:-interactive-len1024_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/results/${RUN_ID}}"
MODE="${MODE:-smoke}"
SPEEDUP="${SPEEDUP:-10}"
MAX_INFLIGHT="${MAX_INFLIGHT:-64}"
INPUT_MEAN="${INPUT_MEAN:-512}"
INPUT_STD="${INPUT_STD:-128}"
OUTPUT_MEAN="${OUTPUT_MEAN:-1024}"
OUTPUT_STD="${OUTPUT_STD:-256}"
SEED="${SEED:-42}"
WITH_AUDIO="${WITH_AUDIO:-0}"
POLICY="${POLICY:-none}"
BARGE_IN_PROB="${BARGE_IN_PROB:-${INTERRUPT_PROB:-0.0}}"
BARGE_IN_AFTER_SEC="${BARGE_IN_AFTER_SEC:-${INTERRUPT_AFTER_SEC:-2.0}}"
BARGE_IN_JITTER_SEC="${BARGE_IN_JITTER_SEC:-${INTERRUPT_JITTER_SEC:-0.5}}"

audio_args=()
if [[ "${WITH_AUDIO}" == "1" ]]; then
  audio_args+=(--with-audio)
fi

mkdir -p "${RESULT_DIR}"

if ! curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "FATAL: server is not healthy at http://${HOST}:${PORT}/health" >&2
  exit 1
fi

echo "[interactive-len1024] result_dir=${RESULT_DIR}"
echo "[interactive-len1024] mode=${MODE} speedup=${SPEEDUP} max_inflight=${MAX_INFLIGHT}"
echo "[interactive-len1024] target query/response=${INPUT_MEAN}/${OUTPUT_MEAN} tokens"

"${PYBIN}" -u "${DRIVER}" \
  --url "http://${HOST}:${PORT}/v1/chat/completions" \
  --model "${MODEL}" \
  --mode "${MODE}" \
  --speedup "${SPEEDUP}" \
  --max-inflight "${MAX_INFLIGHT}" \
  --length-profile interactive-len1024 \
  --input-tokens-mean "${INPUT_MEAN}" \
  --input-tokens-std "${INPUT_STD}" \
  --output-tokens-mean "${OUTPUT_MEAN}" \
  --output-tokens-std "${OUTPUT_STD}" \
  --policy "${POLICY}" \
  --barge-in-prob "${BARGE_IN_PROB}" \
  --barge-in-after-seconds "${BARGE_IN_AFTER_SEC}" \
  --barge-in-jitter-sec "${BARGE_IN_JITTER_SEC}" \
  --seed "${SEED}" \
  "${audio_args[@]}" \
  --result-dir "${RESULT_DIR}"
