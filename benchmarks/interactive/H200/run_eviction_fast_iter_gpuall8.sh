#!/usr/bin/env bash
# Fast all-8-GPU entrypoint for eviction-policy iteration.
#
# Defaults use the 8-GPU micro-KV deploy so short original-distribution probes
# can reach replacement pressure even with four thinker replicas.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../../.." && pwd)"

export GPU_PAIR_LABEL="${GPU_PAIR_LABEL:-gpuall8}"
export REQUIRED_TEXT_GPUS="${REQUIRED_TEXT_GPUS:-0,1,2,3,4,5,6,7}"
export REQUIRED_TEXT_GPUS_ALT="${REQUIRED_TEXT_GPUS_ALT:-7,6,5,4,3,2,1,0}"
export TEXT_GPUS="${TEXT_GPUS:-0,1,2,3,4,5,6,7}"
export PORT="${PORT:-8898}"

export BASELINE_DEPLOY_YAML="${BASELINE_DEPLOY_YAML:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_4thinker_4talker_8gpu_microkv.yaml}"
export EVICTION_DEPLOY_YAML="${EVICTION_DEPLOY_YAML:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_4thinker_4talker_8gpu_prefix_cache_only_microkv.yaml}"
export STAGE0_CACHE_TOKENS="${STAGE0_CACHE_TOKENS:-320000}"
export FAST_MODE="${FAST_MODE:-realmtdist_micro1200}"
export CONCURRENCY_LIST="${CONCURRENCY_LIST:-128}"
export SPEEDUP="${SPEEDUP:-16.0}"
export REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-1200}"
export CELL_TIMEOUT_SEC="${CELL_TIMEOUT_SEC:-2400}"
export STREAM_STAGE_TIMINGS="${STREAM_STAGE_TIMINGS:-1}"

exec "${ROOT}/run_eviction_fast_iter_gpu67.sh" "$@"
