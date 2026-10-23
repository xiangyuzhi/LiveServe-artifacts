#!/usr/bin/env bash
# Download/prepare the full StreamingBench dataset and run high-load LiveServe
# cells on the current H200 profiling lane, GPUs 4-7 by default.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
PYBIN="${LIVESERVE_PYBIN:-/workspace/xyz/omni_020/bin/python}"
STREAMINGBENCH_ROOT="${STREAMINGBENCH_ROOT:-/workspace/xyz/test/video/StreamingBench}"
HF_LOCAL_DIR="${HF_LOCAL_DIR:-/workspace/xyz/test/video/hf_streamingbench_full}"
RESULT_BASE="${RESULT_BASE:-${REPO_ROOT}/results/video/full_streamingbench_longrun_$(date +%Y%m%d_%H%M%S)}"
METADATA_OUT="${METADATA_OUT:-${RESULT_BASE}/full_streamingbench_metadata.json}"

mkdir -p "${RESULT_BASE}" "${HF_LOCAL_DIR}"

log() {
  printf '[full] %s\n' "$*"
}

download_dataset() {
  log "downloading full StreamingBench dataset to ${HF_LOCAL_DIR}"
  hf download mjuicem/StreamingBench \
    --repo-type dataset \
    --local-dir "${HF_LOCAL_DIR}" \
    --include "*.zip" \
    --include "StreamingBench/*.csv" \
    --include "README.md"
}

zip_target_dir() {
  local name="$1"
  case "${name}" in
    Real-Time\ Visual\ Understanding_*.zip) printf '%s/data/real\n' "${STREAMINGBENCH_ROOT}" ;;
    Sequential\ Question\ Answering_*.zip) printf '%s/data/sqa\n' "${STREAMINGBENCH_ROOT}" ;;
    Proactive\ Output_*.zip) printf '%s/data/proactive\n' "${STREAMINGBENCH_ROOT}" ;;
    *)
      local stem
      stem="${name%.zip}"
      stem="${stem// /_}"
      printf '%s/data/omni/%s\n' "${STREAMINGBENCH_ROOT}" "${stem}"
      ;;
  esac
}

extract_dataset() {
  log "extracting dataset zips into ${STREAMINGBENCH_ROOT}/data"
  mkdir -p \
    "${STREAMINGBENCH_ROOT}/data/real" \
    "${STREAMINGBENCH_ROOT}/data/omni" \
    "${STREAMINGBENCH_ROOT}/data/sqa" \
    "${STREAMINGBENCH_ROOT}/data/proactive" \
    "${STREAMINGBENCH_ROOT}/.full_prepare_markers"

  while IFS= read -r -d '' zip_path; do
    local base marker marker_name target
    base="$(basename "${zip_path}")"
    target="$(zip_target_dir "${base}")"
    marker_name="${base}.${target#"${STREAMINGBENCH_ROOT}/data/"}"
    marker_name="${marker_name//\//_}"
    marker="${STREAMINGBENCH_ROOT}/.full_prepare_markers/${marker_name}.done"
    if [ -f "${marker}" ]; then
      log "already extracted ${base}"
      continue
    fi
    log "extracting ${base} -> ${target}"
    unzip -q -n "${zip_path}" -d "${target}"
    touch "${marker}"
  done < <(find "${HF_LOCAL_DIR}" -maxdepth 1 -name '*.zip' -print0 | sort -z)
}

move_videos() {
  log "moving videos into ${STREAMINGBENCH_ROOT}/src/data/videos"
  mkdir -p "${STREAMINGBENCH_ROOT}/src/data/videos"
  (
    cd "${STREAMINGBENCH_ROOT}/src/data"
    "${PYBIN}" move_video.py --src "../../data/real" --dest "./videos"
    "${PYBIN}" move_video.py --src "../../data/omni" --dest "./videos"
    "${PYBIN}" move_video.py --src "../../data/sqa" --dest "./videos"
    "${PYBIN}" move_video.py --src "../../data/proactive" --dest "./videos"
  )
}

prepare_metadata() {
  log "writing merged metadata ${METADATA_OUT}"
  "${PYBIN}" "${ROOT}/prepare_full_streamingbench.py" \
    --streamingbench-root "${STREAMINGBENCH_ROOT}" \
    --out "${METADATA_OUT}" \
    --attempt-link-missing \
    | tee "${RESULT_BASE}/prepare_metadata.log"
}

run_matrix() {
  log "starting full-dataset longrun at ${RESULT_BASE}"
  RESULT_BASE="${RESULT_BASE}" \
  STREAMINGBENCH_METADATA="${METADATA_OUT}" \
  STREAMINGBENCH_DATA_ROOT="${STREAMINGBENCH_ROOT}/src/data" \
  GPU_COUNTS="${GPU_COUNTS:-2 3}" \
  VARIANTS="${VARIANTS:-baseline liveserve_scheduler}" \
  MODES="${MODES:-baseline bargein}" \
  CONCURRENCIES_2GPU="${CONCURRENCIES_2GPU:-8}" \
  CONCURRENCIES_3GPU="${CONCURRENCIES_3GPU:-16}" \
  FIXED_REPEAT="${FIXED_REPEAT:-1}" \
  MAX_VIDEOS="${MAX_VIDEOS:-999999}" \
  MAX_QUESTIONS_PER_VIDEO="${MAX_QUESTIONS_PER_VIDEO:-5}" \
  STREAMINGBENCH_FPS="${STREAMINGBENCH_FPS:-2}" \
  STREAMINGBENCH_FRAMES_PER_QUERY="${STREAMINGBENCH_FRAMES_PER_QUERY:-32}" \
  STREAMINGBENCH_NUM_SAMPLE_FRAMES="${STREAMINGBENCH_NUM_SAMPLE_FRAMES:-16}" \
  STREAMINGBENCH_MAX_BUFFER_FRAMES="${STREAMINGBENCH_MAX_BUFFER_FRAMES:-96}" \
  STREAMINGBENCH_MAX_TOKENS="${STREAMINGBENCH_MAX_TOKENS:-48}" \
  BARGE_IN_AFTER_SECONDS="${BARGE_IN_AFTER_SECONDS:-${INTERRUPT_AFTER_SECONDS:-0.20}}" \
  BARGE_IN_JITTER_SEC="${BARGE_IN_JITTER_SEC:-${INTERRUPT_JITTER_SEC:-0.05}}" \
  TIMEOUT_SEC="${TIMEOUT_SEC:-480}" \
  POST_TEXT_AUDIO_GRACE_SEC="${POST_TEXT_AUDIO_GRACE_SEC:-10}" \
  POST_BOOT_SLEEP_SEC="${POST_BOOT_SLEEP_SEC:-10}" \
  bash "${ROOT}/run_paper_video_matrix.sh"
}

log "RESULT_BASE=${RESULT_BASE}"
log "STREAMINGBENCH_ROOT=${STREAMINGBENCH_ROOT}"
log "HF_LOCAL_DIR=${HF_LOCAL_DIR}"
download_dataset
extract_dataset
move_videos
prepare_metadata
run_matrix
