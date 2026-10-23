#!/usr/bin/env bash
# start_server_2gpu_scheduler.sh — Start vllm-omni on H200 with scheduler lead control enabled (wide 50/25).
#
# This is the scheduler-enabled companion to start_server_2gpu_baseline.sh.
# Used for ad-hoc smoke tests; the production sweep launchers
# (benchmarks/interactive/H200/scheduler_run_one_cell.sh and chain scripts) start
# the server themselves with the L_HI/L_LO they're sweeping over.
#
# Defaults: L_HI=50, L_LO=25 (the "wide" config from interactive_step2.md).
# Override via env vars OMNI_TALKER_LEAD_MAX / OMNI_TALKER_LEAD_LO.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_YAML="${LIVESERVE_DEPLOY_YAML:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_multi_replicas_2gpu.yaml}"
MODEL="${LIVESERVE_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
PYBIN="${LIVESERVE_PYBIN:-python3}"
VLLM_BIN="${LIVESERVE_VLLM_BIN:-$(command -v vllm-omni || true)}"
L_HI="${OMNI_TALKER_LEAD_MAX:-50}"
L_LO="${OMNI_TALKER_LEAD_LO:-25}"

if [ -z "${VLLM_BIN}" ]; then
  echo "[fatal] vllm-omni binary not found on PATH; activate .venv-liveserve first." >&2
  exit 2
fi

echo "[start_server] H200 scheduler lead control ON (L_hi=${L_HI} L_lo=${L_LO})"
echo "  REPO_ROOT   = ${REPO_ROOT}"
echo "  DEPLOY_YAML = ${DEPLOY_YAML}"
echo "  VLLM_BIN    = ${VLLM_BIN}"

# Prefer the LiveServe submodule over any editable install already present in
# the activated environment.
export PYTHONPATH="${REPO_ROOT}/vllm-omni${PYTHONPATH:+:${PYTHONPATH}}"

OMNI_TALKER_LEAD_MAX="${L_HI}" \
OMNI_TALKER_LEAD_LO="${L_LO}" \
VLLM_ATTENTION_BACKEND=FLASHINFER \
VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}" \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
exec "${PYBIN}" "${VLLM_BIN}" serve \
  "${MODEL}" \
  --omni \
  --port 8666 \
  --host 0.0.0.0 \
  --stage-init-timeout 1800 \
  --deploy-config "${DEPLOY_YAML}"
