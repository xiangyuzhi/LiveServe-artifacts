#!/usr/bin/env bash
# Start vllm-omni video streaming server on 3 GPUs with 2 thinker replicas and
# 1 downstream talker/code2wav replica. This is a diagnostic topology, not the
# default scheduler comparison lane.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIDEO_ENV_LIB="${REPO_ROOT}/scripts/H200/video_env_common.sh"
DEPLOY_YAML="${LIVESERVE_DEPLOY_YAML:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_2thinker_1talker_3gpu.yaml}"
MODEL_DEFAULT="/workspace/models/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"
MODEL="${LIVESERVE_MODEL:-${MODEL_DEFAULT}}"
PYBIN="${LIVESERVE_PYBIN:-/workspace/xyz/omni_020/bin/python}"
VLLM_BIN="${LIVESERVE_VLLM_BIN:-/workspace/xyz/omni_020/bin/vllm-omni}"
PORT="${LIVESERVE_PORT:-8777}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6}"

case ",${CUDA_DEVICES}," in
  *,0,*|*,1,*|*,2,*)
    echo "[fatal] refusing to use GPUs 0-2; set CUDA_VISIBLE_DEVICES within 4-7" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONPATH="${REPO_ROOT}/vllm-omni${PYTHONPATH:+:${PYTHONPATH}}"
source "${VIDEO_ENV_LIB}"
liveserve_clear_video_optimization_env

echo "[start_video] thinker2 diagnostic governor=OFF"
echo "  REPO_ROOT            = ${REPO_ROOT}"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  DEPLOY_YAML          = ${DEPLOY_YAML}"
echo "  AUDIO_OUTPUT         = ${STREAMINGBENCH_WITH_AUDIO_OUTPUT:-0}"
echo "  STAGE_INIT_WORKERS   = ${VLLM_OMNI_STAGE_INIT_MAX_WORKERS:-1}"
echo "  MODEL                = ${MODEL}"
echo "  PORT                 = ${PORT}"

VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}" \
VLLM_ATTENTION_BACKEND=FLASHINFER \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_STAGE_INIT_MAX_WORKERS="${VLLM_OMNI_STAGE_INIT_MAX_WORKERS:-1}" \
VLLM_VIDEO_ASYNC_CHUNK="${VLLM_VIDEO_ASYNC_CHUNK:-on}" \
VLLM_VIDEO_AUDIO_DELTA_MODE="${VLLM_VIDEO_AUDIO_DELTA_MODE:-fast}" \
exec "${PYBIN}" "${VLLM_BIN}" serve \
  "${MODEL}" \
  --omni \
  --port "${PORT}" \
  --host 0.0.0.0 \
  --stage-init-timeout 1800 \
  --deploy-config "${DEPLOY_YAML}"
