#!/usr/bin/env bash
# Start Ming-flash-omni 4GPU TP2/talker-DP2 with async chunk and lead governor.
#
# Defaults target the GPU4-7 validation lane:
#   CUDA_VISIBLE_DEVICES=4,5,6,7
#   stage0 thinker TP2 -> logical 0,1
#   stage1 talker DP2  -> logical 2,3
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_YAML="${LIVESERVE_DEPLOY_YAML:-${REPO_ROOT}/deploy/H200/ming_flash_omni_4gpu_tp2_talker_dp2_async.yaml}"
MODEL="${LIVESERVE_MODEL:-/workspace/models/Jonathan1909-Ming-flash-omni-2.0}"
VLLM_BIN="${LIVESERVE_VLLM_BIN:-$(command -v vllm-omni || true)}"
PORT="${LIVESERVE_PORT:-8098}"
CUDA_DEVS="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
HF_HOME="${HF_HOME:-/workspace/models}"
INIT_TIMEOUT="${LIVESERVE_INIT_TIMEOUT:-3000}"
STAGE_INIT_TIMEOUT="${LIVESERVE_STAGE_INIT_TIMEOUT:-1800}"
CHUNK_STEPS="${OMNI_MING_INCREMENTAL_CHUNK_STEPS:-4}"
L_HI="${OMNI_TALKER_LEAD_MAX:-12}"
L_LO="${OMNI_TALKER_LEAD_LO:-6}"
MING_CUDA_WHEEL_DIR="${MING_CUDA_WHEEL_DIR:-/workspace/xyz/omni_020/lib/python3.12/site-packages/nvidia/cu13}"
MING_NVRTC_LINK_DIR="${MING_NVRTC_LINK_DIR:-/workspace/xyz/exp_tmp_dir/ming_nvrtc_link/cu13}"

if [ -z "${VLLM_BIN}" ]; then
  echo "[fatal] vllm-omni binary not found on PATH; activate the LiveServe environment first." >&2
  exit 2
fi

while IFS= read -r name; do
  [ -n "${name}" ] && unset "${name}" || true
done < <(compgen -e OMNI_ || true)
while IFS= read -r name; do
  [ -n "${name}" ] && unset "${name}" || true
done < <(compgen -e SCHEDULER_POLICY_ || true)

export CUDA_VISIBLE_DEVICES="${CUDA_DEVS}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export HF_HOME
export MING_CUDA_WHEEL_DIR
export MING_NVRTC_LINK_DIR
if [ -d "${MING_CUDA_WHEEL_DIR}/include" ]; then
  export CPATH="${MING_CUDA_WHEEL_DIR}/include${CPATH:+:${CPATH}}"
  export CPLUS_INCLUDE_PATH="${MING_CUDA_WHEEL_DIR}/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
fi
if [ -d "${MING_CUDA_WHEEL_DIR}/lib" ]; then
  if [ ! -e "${MING_CUDA_WHEEL_DIR}/lib/libnvrtc.so" ] && [ -e "${MING_CUDA_WHEEL_DIR}/lib/libnvrtc.so.13" ]; then
    mkdir -p "${MING_NVRTC_LINK_DIR}"
    ln -sfn "${MING_CUDA_WHEEL_DIR}/lib/libnvrtc.so.13" "${MING_NVRTC_LINK_DIR}/libnvrtc.so"
  fi
  export LIBRARY_PATH="${MING_NVRTC_LINK_DIR}:${MING_CUDA_WHEEL_DIR}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${MING_NVRTC_LINK_DIR}:${MING_CUDA_WHEEL_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export OMNI_MING_INCREMENTAL=1
export OMNI_MING_INCREMENTAL_CHUNK_STEPS="${CHUNK_STEPS}"
export OMNI_TALKER_LEAD_MAX="${L_HI}"
export OMNI_TALKER_LEAD_LO="${L_LO}"
export OMNI_TALKER_STAGE_IDS=1
export OMNI_THINKER_STAGE_IDS=0

echo "[start_ming_4gpu_dp2_async]"
echo "  REPO_ROOT             = ${REPO_ROOT}"
echo "  DEPLOY_YAML           = ${DEPLOY_YAML}"
echo "  MODEL                 = ${MODEL}"
echo "  VLLM_BIN              = ${VLLM_BIN}"
echo "  PORT                  = ${PORT}"
echo "  CUDA_VISIBLE_DEVICES  = ${CUDA_VISIBLE_DEVICES}"
echo "  INIT_TIMEOUT          = ${INIT_TIMEOUT}"
echo "  STAGE_INIT_TIMEOUT    = ${STAGE_INIT_TIMEOUT}"
echo "  MING_CUDA_WHEEL_DIR   = ${MING_CUDA_WHEEL_DIR}"
echo "  MING_NVRTC_LINK_DIR   = ${MING_NVRTC_LINK_DIR}"
echo "  OMNI_TALKER_LEAD_MAX  = ${OMNI_TALKER_LEAD_MAX}"
echo "  OMNI_TALKER_LEAD_LO   = ${OMNI_TALKER_LEAD_LO}"

exec "${VLLM_BIN}" serve "${MODEL}" \
  --omni \
  --port "${PORT}" \
  --host 0.0.0.0 \
  --init-timeout "${INIT_TIMEOUT}" \
  --stage-init-timeout "${STAGE_INIT_TIMEOUT}" \
  --deploy-config "${DEPLOY_YAML}"
