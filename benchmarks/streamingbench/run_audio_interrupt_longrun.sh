#!/usr/bin/env bash
# Paper-oriented StreamingBench video barge-in matrix.
#
# This assumes the video endpoint emits every audio tensor as a separate
# response.audio.delta.  The workload anchors barge-in to first output audio
# and waits a short playback window so the primary request accumulates
# cancellable downstream audio work before the follow-up request arrives.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

DEFAULT_METADATA="${REPO_ROOT}/results/video/long_audio_64v_2g3g_20260525_203022/full_streamingbench_metadata.json"
if [ ! -f "${DEFAULT_METADATA}" ]; then
  DEFAULT_METADATA="${ROOT}/example_questions.json"
fi

export RESULT_BASE="${RESULT_BASE:-${REPO_ROOT}/results/video/audio_bargein_longrun_${STAMP}}"
export REQUEST_ID_NAMESPACE="${REQUEST_ID_NAMESPACE:-${STAMP}_audio_bargein_longrun}"
export STREAMINGBENCH_METADATA="${STREAMINGBENCH_METADATA:-${DEFAULT_METADATA}}"
export STREAMINGBENCH_DATA_ROOT="${STREAMINGBENCH_DATA_ROOT:-/workspace/xyz/test/video/StreamingBench/src/data}"

export GPU_COUNTS="${GPU_COUNTS:-2 3}"
export VARIANTS="${VARIANTS:-baseline liveserve_scheduler thinker2}"
export MODES="${MODES:-bargein}"
export CONCURRENCIES_2GPU="${CONCURRENCIES_2GPU:-4}"
export CONCURRENCIES_3GPU="${CONCURRENCIES_3GPU:-6}"

export STREAMINGBENCH_WITH_AUDIO_OUTPUT="${STREAMINGBENCH_WITH_AUDIO_OUTPUT:-1}"
export STREAMINGBENCH_PROMPT_STYLE="${STREAMINGBENCH_PROMPT_STYLE:-detailed_answer}"
export STREAMINGBENCH_DIALOGUE_MODE="${STREAMINGBENCH_DIALOGUE_MODE:-multiturn}"
export STREAMINGBENCH_DIALOGUE_HISTORY_TURNS="${STREAMINGBENCH_DIALOGUE_HISTORY_TURNS:-3}"
export STREAMINGBENCH_DIALOGUE_HISTORY_ANSWER_CHARS="${STREAMINGBENCH_DIALOGUE_HISTORY_ANSWER_CHARS:-700}"
export STREAMINGBENCH_FOLLOWUP_TURNS_PER_QUESTION="${STREAMINGBENCH_FOLLOWUP_TURNS_PER_QUESTION:-1}"
export STREAMINGBENCH_BARGE_IN_TIME_ANCHOR="${STREAMINGBENCH_BARGE_IN_TIME_ANCHOR:-${STREAMINGBENCH_INTERRUPT_TIME_ANCHOR:-audio}}"

export FIXED_REPEAT="${FIXED_REPEAT:-1}"
export MAX_VIDEOS="${MAX_VIDEOS:-16}"
export MAX_QUESTIONS_PER_VIDEO="${MAX_QUESTIONS_PER_VIDEO:-3}"
export STREAMINGBENCH_MAX_TOKENS="${STREAMINGBENCH_MAX_TOKENS:-512}"
export STREAMINGBENCH_FPS="${STREAMINGBENCH_FPS:-2}"
export STREAMINGBENCH_FRAMES_PER_QUERY="${STREAMINGBENCH_FRAMES_PER_QUERY:-32}"
export STREAMINGBENCH_NUM_SAMPLE_FRAMES="${STREAMINGBENCH_NUM_SAMPLE_FRAMES:-16}"
export STREAMINGBENCH_MAX_BUFFER_FRAMES="${STREAMINGBENCH_MAX_BUFFER_FRAMES:-96}"

export BARGE_IN_AFTER_SECONDS="${BARGE_IN_AFTER_SECONDS:-${INTERRUPT_AFTER_SECONDS:-0.6}}"
export BARGE_IN_JITTER_SEC="${BARGE_IN_JITTER_SEC:-${INTERRUPT_JITTER_SEC:-0.0}}"
export BARGE_IN_PROB="${BARGE_IN_PROB:-${INTERRUPT_PROB:-1.0}}"
export TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"
export POST_TEXT_AUDIO_GRACE_SEC="${POST_TEXT_AUDIO_GRACE_SEC:-60}"
export ENABLE_GPU_DMON="${ENABLE_GPU_DMON:-1}"
export GPU_DMON_INTERVAL_SEC="${GPU_DMON_INTERVAL_SEC:-1}"

export LIVESERVE_DEPLOY_YAML_2GPU="${LIVESERVE_DEPLOY_YAML_2GPU:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_multi_replicas_2gpu_chunk8.yaml}"
export LIVESERVE_DEPLOY_YAML_3GPU="${LIVESERVE_DEPLOY_YAML_3GPU:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_multi_replicas_3gpu_chunk8.yaml}"
export LIVESERVE_DEPLOY_YAML_THINKER2_3GPU="${LIVESERVE_DEPLOY_YAML_THINKER2_3GPU:-${REPO_ROOT}/deploy/H200/qwen3_omni_moe_2thinker_1talker_3gpu_chunk8.yaml}"
unset LIVESERVE_DEPLOY_YAML

echo "[audio-longrun] RESULT_BASE=${RESULT_BASE}"
echo "[audio-longrun] METADATA=${STREAMINGBENCH_METADATA}"
echo "[audio-longrun] GPU_COUNTS=${GPU_COUNTS} VARIANTS=${VARIANTS} MODES=${MODES}"
echo "[audio-longrun] GPU overrides: 2GPU=${LIVESERVE_GPU_DEVICES_2GPU:-4,5} 3GPU=${LIVESERVE_GPU_DEVICES_3GPU:-4,5,6}"
echo "[audio-longrun] workload=${MAX_VIDEOS} videos x ${MAX_QUESTIONS_PER_VIDEO} questions followups=${STREAMINGBENCH_FOLLOWUP_TURNS_PER_QUESTION}"
echo "[audio-longrun] barge_in_anchor=${STREAMINGBENCH_BARGE_IN_TIME_ANCHOR} delay=${BARGE_IN_AFTER_SECONDS}"

bash "${ROOT}/run_paper_video_matrix.sh"
