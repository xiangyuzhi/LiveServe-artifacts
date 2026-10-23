#!/usr/bin/env bash
# GPU6-7 entrypoint for interactive-original baseline vs eviction-only tuning.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_PAIR_LABEL="${GPU_PAIR_LABEL:-gpu67}"
export REQUIRED_TEXT_GPUS="${REQUIRED_TEXT_GPUS:-6,7}"
export REQUIRED_TEXT_GPUS_ALT="${REQUIRED_TEXT_GPUS_ALT:-7,6}"
export TEXT_GPUS="${TEXT_GPUS:-6,7}"
export PORT="${PORT:-8897}"

exec "${ROOT}/run_original_baseline_vs_eviction_gpu45.sh" "$@"
