#!/usr/bin/env bash
# Start vllm-omni video streaming server on 2 GPUs with LiveServe scheduling enabled.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIDEO_ENV_LIB="${REPO_ROOT}/scripts/H200/video_env_common.sh"
TEXT_DEPLOY_YAML="${REPO_ROOT}/deploy/H200/qwen3_omni_moe_video_text_2gpu.yaml"
FULL_DEPLOY_YAML="${REPO_ROOT}/deploy/H200/qwen3_omni_moe_multi_replicas_2gpu.yaml"
AUDIO_OUTPUT_ENABLED=0
case "${STREAMINGBENCH_WITH_AUDIO_OUTPUT:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    AUDIO_OUTPUT_ENABLED=1
    DEFAULT_DEPLOY_YAML="${FULL_DEPLOY_YAML}"
    ;;
  *)
    DEFAULT_DEPLOY_YAML="${TEXT_DEPLOY_YAML}"
    ;;
esac
DEPLOY_YAML="${LIVESERVE_DEPLOY_YAML:-${DEFAULT_DEPLOY_YAML}}"
MODEL_DEFAULT="/workspace/models/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"
MODEL="${LIVESERVE_MODEL:-${MODEL_DEFAULT}}"
PYBIN="${LIVESERVE_PYBIN:-/workspace/xyz/omni_020/bin/python}"
VLLM_BIN="${LIVESERVE_VLLM_BIN:-/workspace/xyz/omni_020/bin/vllm-omni}"
PORT="${LIVESERVE_PORT:-8777}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
SCHEDULER_POLICY_LIB="${REPO_ROOT}/profiling/scheduler_analysis/scripts/scheduler_policy_common.sh"
serve_extra_args=()
if [ -n "${SERVER_MOE_BACKEND:-}" ]; then
  serve_extra_args+=(--moe-backend "${SERVER_MOE_BACKEND}")
fi

case ",${CUDA_DEVICES}," in
  *,0,*|*,1,*|*,2,*)
    case "${LIVESERVE_ALLOW_GPU_0_2:-0}" in
      1|true|TRUE|yes|YES|on|ON) ;;
      *)
        echo "[fatal] refusing to use GPUs 0-2; set LIVESERVE_ALLOW_GPU_0_2=1 to override" >&2
        exit 2
        ;;
    esac
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONPATH="${REPO_ROOT}/vllm-omni${PYTHONPATH:+:${PYTHONPATH}}"
source "${VIDEO_ENV_LIB}"
liveserve_clear_video_optimization_env
export OMNI_PREFIX_SHAPE_PAD="${OMNI_PREFIX_SHAPE_PAD:-1}"

if [ "${AUDIO_OUTPUT_ENABLED}" = "1" ]; then
  if [ ! -f "${SCHEDULER_POLICY_LIB}" ]; then
    echo "[start_video] fatal: missing scheduler policy helper ${SCHEDULER_POLICY_LIB}" >&2
    exit 2
  fi
  # shellcheck source=/dev/null
  source "${SCHEDULER_POLICY_LIB}"
  # Very short video answers can complete before a deferred talker emits audio.
  export SCHEDULER_POLICY_DEFAULT_TALKER_MIN_OUTPUT_TOKENS="${SCHEDULER_POLICY_DEFAULT_TALKER_MIN_OUTPUT_TOKENS:-48}"
  scheduler_policy_apply_and_export "${SCHEDULER_POLICY_PRESET:-scheduler}" "${SCHEDULER_POLICY_OVERLAYS:-talkerlead=12:6:75}"
  echo "[start_video] scheduler-only=ON talker=${OMNI_TALKER_LEAD_MAX}:${OMNI_TALKER_LEAD_LO}:hold${OMNI_TALKER_DEFER_HOLD_MAX}"
  echo "[start_video] talker_min_output_tokens_before_defer=${OMNI_TALKER_MIN_OUTPUT_TOKENS_BEFORE_DEFER}"
else
  unset OMNI_TALKER_LEAD_MAX OMNI_TALKER_LEAD_LO OMNI_TALKER_DEFER_HOLD_MAX
  echo "[start_video] scheduler-only=OFF text-only"
fi
echo "  REPO_ROOT            = ${REPO_ROOT}"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  DEPLOY_YAML          = ${DEPLOY_YAML}"
echo "  AUDIO_OUTPUT         = ${STREAMINGBENCH_WITH_AUDIO_OUTPUT:-0}"
echo "  PREFIX_SHAPE_PAD     = ${OMNI_PREFIX_SHAPE_PAD}"
echo "  STAGE_INIT_WORKERS   = ${VLLM_OMNI_STAGE_INIT_MAX_WORKERS:-1}"
echo "  INIT_TIMEOUT         = ${SERVER_INIT_TIMEOUT:-2400}"
echo "  STAGE_INIT_TIMEOUT   = ${SERVER_STAGE_INIT_TIMEOUT:-1800}"
echo "  MOE_BACKEND          = ${SERVER_MOE_BACKEND:-default}"
echo "  PORT                 = ${PORT}"

env_args=(
  VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
  VLLM_ATTENTION_BACKEND=FLASHINFER
  VLLM_WORKER_MULTIPROC_METHOD=spawn
  VLLM_OMNI_STAGE_INIT_MAX_WORKERS="${VLLM_OMNI_STAGE_INIT_MAX_WORKERS:-1}"
  VLLM_VIDEO_ASYNC_CHUNK="${VLLM_VIDEO_ASYNC_CHUNK:-on}"
  VLLM_VIDEO_AUDIO_DELTA_MODE="${VLLM_VIDEO_AUDIO_DELTA_MODE:-fast}"
)
exec env "${env_args[@]}" "${PYBIN}" "${VLLM_BIN}" serve \
  "${MODEL}" \
  --omni \
  --port "${PORT}" \
  --host 0.0.0.0 \
  --stage-init-timeout "${SERVER_STAGE_INIT_TIMEOUT:-1800}" \
  --init-timeout "${SERVER_INIT_TIMEOUT:-2400}" \
  --deploy-config "${DEPLOY_YAML}" \
  "${serve_extra_args[@]}"
