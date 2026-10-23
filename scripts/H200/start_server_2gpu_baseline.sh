#!/usr/bin/env bash
# start_server_2gpu_baseline.sh — Start vllm-omni on H200 with scheduler lead control disabled.
#
# Used by Step 1 cells (baseline + naive barge-in). The chain scripts in
# benchmarks/interactive/H200/step1_*.sh assume a server is already running with
# lead control OFF; this script starts one in the foreground (Ctrl-C to stop).
#
# Honors env-var overrides: LIVESERVE_VLLM_BIN, LIVESERVE_PYBIN, LIVESERVE_DEPLOY_YAML,
# LIVESERVE_MODEL — same as benchmarks/interactive/H200/scheduler_run_one_cell.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_YAML="${LIVESERVE_DEPLOY_YAML:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_multi_replicas_2gpu.yaml}"
MODEL="${LIVESERVE_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
PYBIN="${LIVESERVE_PYBIN:-python3}"
VLLM_BIN="${LIVESERVE_VLLM_BIN:-$(command -v vllm-omni || true)}"

if [ -z "${VLLM_BIN}" ]; then
  echo "[fatal] vllm-omni binary not found on PATH; activate .venv-liveserve first." >&2
  exit 2
fi

echo "[start_server] H200 baseline (scheduler lead control OFF)"
echo "  REPO_ROOT   = ${REPO_ROOT}"
echo "  DEPLOY_YAML = ${DEPLOY_YAML}"
echo "  MODEL       = ${MODEL}"
echo "  VLLM_BIN    = ${VLLM_BIN}"

# Prefer the LiveServe submodule over any editable install already present in
# the activated environment.
export PYTHONPATH="${REPO_ROOT}/vllm-omni${PYTHONPATH:+:${PYTHONPATH}}"

# Explicitly UNSET the lead-control env vars so the scheduler runs in the baseline path.
unset OMNI_TALKER_LEAD_MAX OMNI_TALKER_LEAD_LO

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
