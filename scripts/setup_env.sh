#!/usr/bin/env bash
# setup_env.sh - Create a serving environment or a lightweight CPU development environment.
#
# Default: create/reuse .venv-liveserve, install the pinned runtime in editable
# mode plus benchmark dependencies, and verify imports and the serving CLI.
# --dev: create/reuse .venv-liveserve-dev and install only requirements/dev.txt.
# Both modes require Python 3.11+; select it with LIVESERVE_PYTHON.
#
# Usage:
#   bash scripts/setup_env.sh                # idempotent setup
#   bash scripts/setup_env.sh --dev          # CPU checks only; no CUDA/model packages
#   bash scripts/setup_env.sh --recreate     # recreate the selected environment
#
# Full runtime setup downloads large CUDA wheels; CPU setup does not.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${LIVESERVE_PYTHON:-python3}"
RECREATE=0
DEV_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --recreate) RECREATE=1 ;;
    --dev) DEV_ONLY=1 ;;
    -h|--help)  sed -n '2,/^set -/p' "$0" | sed '$d'; exit 0 ;;
    *)          echo "[setup_env] unknown arg: $arg" >&2; exit 2 ;;
  esac
done

"${PYTHON_BIN}" -c 'import sys; sys.exit("LiveServe tools require Python 3.11+; set LIVESERVE_PYTHON to a compatible interpreter.") if sys.version_info < (3, 11) else None'
if [ "${DEV_ONLY}" -eq 1 ]; then
  VENV="${REPO_ROOT}/.venv-liveserve-dev"
else
  VENV="${REPO_ROOT}/.venv-liveserve"
fi

if [ "${RECREATE}" -eq 1 ] && [ -d "${VENV}" ]; then
  echo "[setup_env] --recreate: removing ${VENV}"
  rm -rf "${VENV}"
fi

if [ ! -d "${VENV}" ]; then
  echo "[setup_env] creating venv at ${VENV} (interpreter: ${PYTHON_BIN})"
  "${PYTHON_BIN}" -m venv "${VENV}"
else
  echo "[setup_env] reusing existing venv at ${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if [ "${DEV_ONLY}" -eq 1 ]; then
  python -m pip install -r "${REPO_ROOT}/requirements/dev.txt"
  python -c 'import aiohttp, numpy, psutil, requests, soundfile, yaml'
  echo "[setup_env] CPU environment ready. Run:"
  echo "    LIVESERVE_PYBIN=${VENV}/bin/python bash scripts/validate_main_no_gpu.sh"
  exit 0
fi

# vllm-omni submodule init + editable install
if [ ! -f "${REPO_ROOT}/vllm-omni/setup.py" ]; then
  echo "[setup_env] vllm-omni submodule not initialized; running git submodule update"
  git submodule update --init --recursive
fi

echo "[setup_env] pip install -e ./vllm-omni  (this is the slow step)"
pip install -e "${REPO_ROOT}/vllm-omni"

# Auxiliary deps for benchmark drivers / aggregators / plots
echo "[setup_env] installing aux deps"
pip install --no-cache-dir \
    httpx \
    aiohttp \
    pydantic \
    matplotlib \
    numpy \
    pandas \
    huggingface_hub \
    datasets

# Sanity
echo
echo "[setup_env] versions:"
python -c "import sys; print(f'  python: {sys.version.split()[0]}')"
python -c "import vllm_omni, vllm; print(f'  vllm: {vllm.__version__}'); print(f'  vllm_omni: ok')"
command -v vllm-omni

echo
echo "[setup_env] DONE. To use:"
echo "    source ${VENV}/bin/activate"
echo "    bash scripts/H200/start_server_2gpu_scheduler.sh"
