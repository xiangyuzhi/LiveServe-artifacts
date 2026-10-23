#!/usr/bin/env bash
# StreamingBench barge-in probe tuned for scheduler lead control.
#
# Differences from run_long_dialogue_probe.sh:
#   - barge-in triggers are anchored to the first audio delta, not request submit time;
#   - smaller codec chunks are used to expose shorter downstream audio deltas
#     when the endpoint keeps streaming audio;
#   - runs 2GPU and 3GPU lanes separately so each can use its matching deploy yaml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

DEFAULT_METADATA="${REPO_ROOT}/results/video/long_audio_64v_2g3g_20260525_203022/full_streamingbench_metadata.json"
if [ ! -f "${DEFAULT_METADATA}" ]; then
  DEFAULT_METADATA="${ROOT}/example_questions.json"
fi

export RESULT_BASE="${RESULT_BASE:-${REPO_ROOT}/results/video/audio_anchor_chunk8_probe_${STAMP}}"
export REQUEST_ID_NAMESPACE="${REQUEST_ID_NAMESPACE:-${STAMP}_audio_anchor_chunk8}"
export STREAMINGBENCH_METADATA="${STREAMINGBENCH_METADATA:-${DEFAULT_METADATA}}"
export STREAMINGBENCH_DATA_ROOT="${STREAMINGBENCH_DATA_ROOT:-/workspace/xyz/test/video/StreamingBench/src/data}"

export VARIANTS="${VARIANTS:-baseline liveserve_scheduler}"
export MODES="${MODES:-bargein}"
export STREAMINGBENCH_WITH_AUDIO_OUTPUT="${STREAMINGBENCH_WITH_AUDIO_OUTPUT:-1}"
export STREAMINGBENCH_PROMPT_STYLE="${STREAMINGBENCH_PROMPT_STYLE:-detailed_answer}"
export STREAMINGBENCH_DIALOGUE_MODE="${STREAMINGBENCH_DIALOGUE_MODE:-multiturn}"
export STREAMINGBENCH_DIALOGUE_HISTORY_TURNS="${STREAMINGBENCH_DIALOGUE_HISTORY_TURNS:-3}"
export STREAMINGBENCH_DIALOGUE_HISTORY_ANSWER_CHARS="${STREAMINGBENCH_DIALOGUE_HISTORY_ANSWER_CHARS:-700}"
export STREAMINGBENCH_FOLLOWUP_TURNS_PER_QUESTION="${STREAMINGBENCH_FOLLOWUP_TURNS_PER_QUESTION:-1}"
export STREAMINGBENCH_BARGE_IN_TIME_ANCHOR="${STREAMINGBENCH_BARGE_IN_TIME_ANCHOR:-${STREAMINGBENCH_INTERRUPT_TIME_ANCHOR:-audio}}"

export FIXED_REPEAT="${FIXED_REPEAT:-1}"
export MAX_VIDEOS="${MAX_VIDEOS:-4}"
export MAX_QUESTIONS_PER_VIDEO="${MAX_QUESTIONS_PER_VIDEO:-2}"
export STREAMINGBENCH_MAX_TOKENS="${STREAMINGBENCH_MAX_TOKENS:-512}"
export STREAMINGBENCH_FPS="${STREAMINGBENCH_FPS:-2}"
export STREAMINGBENCH_FRAMES_PER_QUERY="${STREAMINGBENCH_FRAMES_PER_QUERY:-32}"
export STREAMINGBENCH_NUM_SAMPLE_FRAMES="${STREAMINGBENCH_NUM_SAMPLE_FRAMES:-16}"
export STREAMINGBENCH_MAX_BUFFER_FRAMES="${STREAMINGBENCH_MAX_BUFFER_FRAMES:-96}"

# With audio anchoring, 0.0 means "barge in immediately after the first audio
# delta is observed", which most closely simulates a user barge-in during
# playback when the wire protocol emits coarse chunks.
export BARGE_IN_AFTER_SECONDS="${BARGE_IN_AFTER_SECONDS:-${INTERRUPT_AFTER_SECONDS:-0.0}}"
export BARGE_IN_JITTER_SEC="${BARGE_IN_JITTER_SEC:-${INTERRUPT_JITTER_SEC:-0.0}}"
export BARGE_IN_PROB="${BARGE_IN_PROB:-${INTERRUPT_PROB:-1.0}}"
export TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
export POST_TEXT_AUDIO_GRACE_SEC="${POST_TEXT_AUDIO_GRACE_SEC:-30}"
export ENABLE_GPU_DMON="${ENABLE_GPU_DMON:-1}"
export GPU_DMON_INTERVAL_SEC="${GPU_DMON_INTERVAL_SEC:-1}"

echo "[audio-anchor] RESULT_BASE=${RESULT_BASE}"
echo "[audio-anchor] METADATA=${STREAMINGBENCH_METADATA}"
echo "[audio-anchor] GPU overrides: 2GPU=${LIVESERVE_GPU_DEVICES_2GPU:-4,5} 3GPU=${LIVESERVE_GPU_DEVICES_3GPU:-4,5,6}"
echo "[audio-anchor] prompt=${STREAMINGBENCH_PROMPT_STYLE} followups=${STREAMINGBENCH_FOLLOWUP_TURNS_PER_QUESTION}"
echo "[audio-anchor] barge_in_anchor=${STREAMINGBENCH_BARGE_IN_TIME_ANCHOR} delay=${BARGE_IN_AFTER_SECONDS}"

export GPU_COUNTS="2"
export CONCURRENCIES_2GPU="${CONCURRENCIES_2GPU:-4}"
export LIVESERVE_DEPLOY_YAML_2GPU="${LIVESERVE_DEPLOY_YAML_2GPU:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_multi_replicas_2gpu_chunk8.yaml}"
unset LIVESERVE_DEPLOY_YAML
bash "${ROOT}/run_paper_video_matrix.sh"

export GPU_COUNTS="3"
export CONCURRENCIES_3GPU="${CONCURRENCIES_3GPU:-6}"
export LIVESERVE_DEPLOY_YAML_3GPU="${LIVESERVE_DEPLOY_YAML_3GPU:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_multi_replicas_3gpu_chunk8.yaml}"
export LIVESERVE_DEPLOY_YAML_THINKER2_3GPU="${LIVESERVE_DEPLOY_YAML_THINKER2_3GPU:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_2thinker_1talker_3gpu_chunk8.yaml}"
unset LIVESERVE_DEPLOY_YAML
bash "${ROOT}/run_paper_video_matrix.sh"
