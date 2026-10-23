#!/usr/bin/env bash
# Start vllm-omni video streaming server on 2 GPUs with scheduler + KV preload + eviction.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
SCHEDULER_POLICY_LIB="${SCHEDULER_POLICY_LIB:-${REPO_ROOT}/profiling/scheduler_analysis/scripts/scheduler_policy_common.sh}"
EVICTION_POLICY_LIB="${EVICTION_POLICY_LIB:-${REPO_ROOT}/profiling/scheduler_analysis/scripts/eviction_policy_common.sh}"
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

if [ ! -f "${SCHEDULER_POLICY_LIB}" ]; then
  echo "[start_video_joint] fatal: missing scheduler policy helper ${SCHEDULER_POLICY_LIB}" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "${SCHEDULER_POLICY_LIB}"
scheduler_policy_apply_and_export \
  "${SCHEDULER_POLICY_PRESET:-scheduler}" \
  "${SCHEDULER_POLICY_OVERLAYS:-talkerlead=12:6:75}"
source "${EVICTION_POLICY_LIB}"

export OMNI_PREFIX_SHAPE_PAD="${OMNI_PREFIX_SHAPE_PAD:-1}"
export OMNI_PREFETCH_FAST_BYPASS="${OMNI_PREFETCH_FAST_BYPASS:-1}"
export OMNI_PREFETCH_ADMISSION="${OMNI_PREFETCH_ADMISSION:-1}"
export OMNI_PREFETCH_SKIP_OFFLOAD_ON_MISS="${OMNI_PREFETCH_SKIP_OFFLOAD_ON_MISS:-1}"
export OMNI_PREFETCH_HISTORY_HIT_ENABLE="${OMNI_PREFETCH_HISTORY_HIT_ENABLE:-1}"
export OMNI_PREFETCH_REGISTRY_TTL_SEC="${OMNI_PREFETCH_REGISTRY_TTL_SEC:-900}"
export OMNI_PREFETCH_STREAM_TIMEOUT_MS="${OMNI_PREFETCH_STREAM_TIMEOUT_MS:-500}"
export OMNI_PREFETCH_GPU_PIN="${OMNI_PREFETCH_GPU_PIN:-1}"
export OMNI_PREFETCH_GPU_PIN_TTL_MS="${OMNI_PREFETCH_GPU_PIN_TTL_MS:-600000}"
export OMNI_PREFETCH_GPU_PIN_MAX_BLOCKS="${OMNI_PREFETCH_GPU_PIN_MAX_BLOCKS:-8192}"
export OMNI_PREFETCH_PRIORITY="${OMNI_PREFETCH_PRIORITY:-100}"
export OMNI_SCHED_NATIVE_PREFETCH="${OMNI_SCHED_NATIVE_PREFETCH:-1}"
export OMNI_SCHED_NATIVE_PREFETCH_CANCEL="${OMNI_SCHED_NATIVE_PREFETCH_CANCEL:-1}"
export OMNI_SCHED_NATIVE_PREFETCH_MAX_LIVE="${OMNI_SCHED_NATIVE_PREFETCH_MAX_LIVE:-0}"

LIVE_EVICT_PROFILE="${LIVE_EVICT_PROFILE:-tail24}"
case "${LIVE_EVICT_PROFILE}" in
  g1|G1|liveserve_g1|allopt|latest)
    export OMNI_KV_EVICT_SHADOW="${OMNI_KV_EVICT_SHADOW:-0}"
    eviction_policy_apply_g1_env
    ;;
  tail24|legacy_tail24)
    export OMNI_KV_EVICT_SHADOW="${OMNI_KV_EVICT_SHADOW:-1}"
    export OMNI_KV_EVICT_ENFORCE_POLICY="${OMNI_KV_EVICT_ENFORCE_POLICY:-response_len_tail_aware}"
    export OMNI_KV_EVICT_ENFORCE_INDEX="${OMNI_KV_EVICT_ENFORCE_INDEX:-1}"
    export OMNI_KV_EVICT_ENFORCE_INDEX_FALLBACK_SCAN="${OMNI_KV_EVICT_ENFORCE_INDEX_FALLBACK_SCAN:-0}"
    export OMNI_KV_EVICT_ENFORCE_MIN_CACHED_VICTIMS_PER_ALLOC="${OMNI_KV_EVICT_ENFORCE_MIN_CACHED_VICTIMS_PER_ALLOC:-1}"
    export OMNI_KV_EVICT_ENFORCE_MAX_CACHED_VICTIMS_PER_ALLOC="${OMNI_KV_EVICT_ENFORCE_MAX_CACHED_VICTIMS_PER_ALLOC:-1}"
    export OMNI_KV_EVICT_INDEX_MAX_BLOCKS_PER_REQUEST="${OMNI_KV_EVICT_INDEX_MAX_BLOCKS_PER_REQUEST:-24}"
    ;;
  *)
    echo "[fatal] unknown LIVE_EVICT_PROFILE=${LIVE_EVICT_PROFILE}" >&2
    exit 2
    ;;
esac
export OMNI_KV_EVICT_ENFORCE_LOG="${OMNI_KV_EVICT_ENFORCE_LOG:-0}"
export OMNI_KV_LRU_STATS="${OMNI_KV_LRU_STATS:-1}"

echo "[start_video_joint] 2GPU scheduler+preload+${LIVE_EVICT_PROFILE}"
echo "  REPO_ROOT            = ${REPO_ROOT}"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "  DEPLOY_YAML          = ${DEPLOY_YAML}"
echo "  AUDIO_OUTPUT         = ${STREAMINGBENCH_WITH_AUDIO_OUTPUT:-0}"
echo "  SCHEDULER_POLICY     = ${SCHEDULER_POLICY_PRESET:-scheduler}"
echo "  SCHEDULER_OVERLAYS   = ${SCHEDULER_POLICY_OVERLAYS:-talkerlead=12:6:75}"
echo "  PRELOAD              = fast=${OMNI_PREFETCH_FAST_BYPASS} native=${OMNI_SCHED_NATIVE_PREFETCH} history_hit=${OMNI_PREFETCH_HISTORY_HIT_ENABLE}"
echo "  PRELOAD_ADMISSION    = admission=${OMNI_PREFETCH_ADMISSION} max_inflight=${OMNI_PREFETCH_SERVER_MAX_INFLIGHT:-1} stream_timeout_ms=${OMNI_PREFETCH_STREAM_TIMEOUT_MS}"
echo "  EVICTION             = ${OMNI_KV_EVICT_ENFORCE_POLICY} index_tail=${OMNI_KV_EVICT_INDEX_MAX_BLOCKS_PER_REQUEST} cap=${OMNI_KV_EVICT_ENFORCE_MAX_CACHED_VICTIMS_PER_ALLOC}"
echo "  STAGE_INIT_WORKERS   = ${VLLM_OMNI_STAGE_INIT_MAX_WORKERS:-1}"
echo "  INIT_TIMEOUT         = ${SERVER_INIT_TIMEOUT:-2400}"
echo "  STAGE_INIT_TIMEOUT   = ${SERVER_STAGE_INIT_TIMEOUT:-1800}"
echo "  MOE_BACKEND          = ${SERVER_MOE_BACKEND:-default}"
echo "  MODEL                = ${MODEL}"
echo "  PORT                 = ${PORT}"

exec env \
  VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}" \
  VLLM_ATTENTION_BACKEND=FLASHINFER \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_OMNI_STAGE_INIT_MAX_WORKERS="${VLLM_OMNI_STAGE_INIT_MAX_WORKERS:-1}" \
  VLLM_VIDEO_ASYNC_CHUNK="${VLLM_VIDEO_ASYNC_CHUNK:-on}" \
  VLLM_VIDEO_AUDIO_DELTA_MODE="${VLLM_VIDEO_AUDIO_DELTA_MODE:-fast}" \
  "${PYBIN}" "${VLLM_BIN}" serve \
    "${MODEL}" \
    --omni \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --stage-init-timeout "${SERVER_STAGE_INIT_TIMEOUT:-1800}" \
    --init-timeout "${SERVER_INIT_TIMEOUT:-2400}" \
    --deploy-config "${DEPLOY_YAML}" \
    "${serve_extra_args[@]}"
