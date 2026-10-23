#!/usr/bin/env bash
# Fast representative StreamingBench probe for long spoken answers + multi-turn dialogue.
#
# Defaults are intentionally small. The goal is to validate whether the workload
# reaches the audio/barge-in bottleneck before launching any paper-scale run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

DEFAULT_METADATA="${REPO_ROOT}/results/video/long_audio_64v_2g3g_20260525_203022/full_streamingbench_metadata.json"
if [ ! -f "${DEFAULT_METADATA}" ]; then
  DEFAULT_METADATA="${ROOT}/example_questions.json"
fi

export RESULT_BASE="${RESULT_BASE:-${REPO_ROOT}/results/video/long_dialogue_probe_${STAMP}}"
export STREAMINGBENCH_METADATA="${STREAMINGBENCH_METADATA:-${DEFAULT_METADATA}}"
export STREAMINGBENCH_DATA_ROOT="${STREAMINGBENCH_DATA_ROOT:-/workspace/xyz/test/video/StreamingBench/src/data}"

# Keep all GPU work on the current 4-7 profiling lane. Override GPU_COUNTS="2 3"
# when both cells are wanted.
export GPU_COUNTS="${GPU_COUNTS:-3}"
export CONCURRENCIES_2GPU="${CONCURRENCIES_2GPU:-4}"
export CONCURRENCIES_3GPU="${CONCURRENCIES_3GPU:-8}"
export VARIANTS="${VARIANTS:-baseline liveserve_scheduler}"
export MODES="${MODES:-bargein}"

export STREAMINGBENCH_WITH_AUDIO_OUTPUT="${STREAMINGBENCH_WITH_AUDIO_OUTPUT:-1}"
export STREAMINGBENCH_PROMPT_STYLE="${STREAMINGBENCH_PROMPT_STYLE:-detailed_answer}"
export STREAMINGBENCH_DIALOGUE_MODE="${STREAMINGBENCH_DIALOGUE_MODE:-multiturn}"
export STREAMINGBENCH_DIALOGUE_HISTORY_TURNS="${STREAMINGBENCH_DIALOGUE_HISTORY_TURNS:-3}"
export STREAMINGBENCH_DIALOGUE_HISTORY_ANSWER_CHARS="${STREAMINGBENCH_DIALOGUE_HISTORY_ANSWER_CHARS:-700}"
export STREAMINGBENCH_FOLLOWUP_TURNS_PER_QUESTION="${STREAMINGBENCH_FOLLOWUP_TURNS_PER_QUESTION:-1}"

export FIXED_REPEAT="${FIXED_REPEAT:-1}"
export MAX_VIDEOS="${MAX_VIDEOS:-6}"
export MAX_QUESTIONS_PER_VIDEO="${MAX_QUESTIONS_PER_VIDEO:-4}"
export STREAMINGBENCH_MAX_TOKENS="${STREAMINGBENCH_MAX_TOKENS:-384}"
export STREAMINGBENCH_FPS="${STREAMINGBENCH_FPS:-2}"
export STREAMINGBENCH_FRAMES_PER_QUERY="${STREAMINGBENCH_FRAMES_PER_QUERY:-32}"
export STREAMINGBENCH_NUM_SAMPLE_FRAMES="${STREAMINGBENCH_NUM_SAMPLE_FRAMES:-16}"
export STREAMINGBENCH_MAX_BUFFER_FRAMES="${STREAMINGBENCH_MAX_BUFFER_FRAMES:-96}"

# A later barge-in gives the baseline enough time to generate cancellable audio.
export BARGE_IN_AFTER_SECONDS="${BARGE_IN_AFTER_SECONDS:-${INTERRUPT_AFTER_SECONDS:-0.80}}"
export BARGE_IN_JITTER_SEC="${BARGE_IN_JITTER_SEC:-${INTERRUPT_JITTER_SEC:-0.10}}"
export BARGE_IN_PROB="${BARGE_IN_PROB:-${INTERRUPT_PROB:-1.0}}"
export TIMEOUT_SEC="${TIMEOUT_SEC:-600}"
export POST_TEXT_AUDIO_GRACE_SEC="${POST_TEXT_AUDIO_GRACE_SEC:-20}"

# Lightweight per-cell GPU utilization trace.
export ENABLE_GPU_DMON="${ENABLE_GPU_DMON:-1}"
export GPU_DMON_INTERVAL_SEC="${GPU_DMON_INTERVAL_SEC:-1}"

echo "[probe] RESULT_BASE=${RESULT_BASE}"
echo "[probe] METADATA=${STREAMINGBENCH_METADATA}"
echo "[probe] GPU_COUNTS=${GPU_COUNTS} (script refuses GPUs 0-2)"
echo "[probe] VARIANTS=${VARIANTS}"
echo "[probe] MODES=${MODES}"
echo "[probe] prompt=${STREAMINGBENCH_PROMPT_STYLE} dialogue=${STREAMINGBENCH_DIALOGUE_MODE}"

exec bash "${ROOT}/run_paper_video_matrix.sh"
