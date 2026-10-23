#!/usr/bin/env bash
# prepare_sharegpt.sh — Download + verify the ShareGPT-Chinese-English-90k dataset.
#
# Wraps benchmarks/sharegpt/H200/{download_dataset,verify_dataset}.py with
# repo-relative defaults. Honors LIVESERVE_DATA_ROOT and LIVESERVE_SHAREGPT_DATASET
# env vars (see benchmarks/sharegpt/H200/README.md).
#
# Usage:
#   bash scripts/prepare_sharegpt.sh
#
# Output: ${LIVESERVE_DATA_ROOT:-<repo>/data}/ShareGPT_Chinese_English_90k.json (~120 MB)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${LIVESERVE_PYBIN:-python3}"

SCRIPT_DIR="${REPO_ROOT}/benchmarks/sharegpt/H200"
echo "[prepare_sharegpt] using ${SCRIPT_DIR}/download_dataset.py"

cd "${SCRIPT_DIR}"
"${PY}" download_dataset.py
echo "[prepare_sharegpt] verifying ..."
"${PY}" verify_dataset.py
echo "[prepare_sharegpt] DONE."
