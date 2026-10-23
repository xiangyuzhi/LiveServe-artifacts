#!/usr/bin/env bash
# All-8-GPU entrypoint for interactive-original baseline vs eviction-only tuning.
#
# Default topology:
#   logical GPU0-3: thinker replicas
#   logical GPU4-7: talker/code2wav replicas
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../../.." && pwd)"

export GPU_PAIR_LABEL="${GPU_PAIR_LABEL:-gpuall8}"
export REQUIRED_TEXT_GPUS="${REQUIRED_TEXT_GPUS:-0,1,2,3,4,5,6,7}"
export REQUIRED_TEXT_GPUS_ALT="${REQUIRED_TEXT_GPUS_ALT:-7,6,5,4,3,2,1,0}"
export TEXT_GPUS="${TEXT_GPUS:-0,1,2,3,4,5,6,7}"
export PORT="${PORT:-8898}"

export BASELINE_DEPLOY_YAML="${BASELINE_DEPLOY_YAML:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_4thinker_4talker_8gpu.yaml}"
export EVICTION_DEPLOY_YAML="${EVICTION_DEPLOY_YAML:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_4thinker_4talker_8gpu_prefix_cache_only.yaml}"

exec "${ROOT}/run_original_baseline_vs_eviction_gpu45.sh" "$@"
