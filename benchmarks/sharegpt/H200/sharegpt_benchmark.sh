#!/usr/bin/env bash
#
# ShareGPT benchmark helper covering text/audio probes and batch loads.
#
# Usage:
#   bash sharegpt_benchmark.sh <feature_name> <mode>
#
# Modes:
#   probe_text     - one text-only sample through single_probe.py
#   probe_audio    - one text+audio sample, writing WAV output
#   bench_text     - ShareGPT text-only batch load, modalities=["text"]
#   bench_audio    - ShareGPT text+audio batch load through the full pipeline
#   all            - probe_text -> probe_audio -> bench_text -> bench_audio
#
# Examples:
#   bash sharegpt_benchmark.sh v1.0 probe_audio
#   bash sharegpt_benchmark.sh v1.0 bench_text
#   bash sharegpt_benchmark.sh v1.0 all
#
# Prerequisites:
#   1. vllm-omni is already serving on ${PORT}.
#      - bench_text/probe_text work best with a text-only stage config.
#      - bench_audio/probe_audio require the full three-stage config.
#   2. The dataset exists at ${DATASET_PATH}.
#
# Design notes:
#   - Only unset HTTP(S) proxy vars; keep all_proxy/no_proxy untouched.
#   - Use a local snapshot path for --model while keeping the served model name
#     as the repo id, avoiding Hugging Face repo-id resolution during the client run.
#

set -e

# ========== Basic configuration ==========
# Only unset HTTP(S) proxies; keep all_proxy/no_proxy available for host policy.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

FEATURE=${1:-"sharegpt_test"}
MODE=${2:-"probe_audio"}

# The benchmark client uses a local snapshot path for --model/get_tokenizer.
# The served model name remains the repo id so request body.model matches the server.
MODEL_PATH="${LIVESERVE_MODEL_PATH:-${LIVESERVE_MODEL:-/workspace/models/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695}}"
SERVED_MODEL_NAME="${LIVESERVE_SERVED_MODEL_NAME:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
PORT="${LIVESERVE_PORT:-8666}"
HOST="${LIVESERVE_HOST:-127.0.0.1}"
PYBIN="${LIVESERVE_PYBIN:-python3}"
VLLM_BENCH_BIN="${LIVESERVE_VLLM_BENCH_BIN:-$(command -v vllm || true)}"

# Resolve repo-relative defaults; override via env vars LIVESERVE_SHAREGPT_DATASET / LIVESERVE_DATA_ROOT.
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${WORK_DIR}/../../.." && pwd)"
DATASET_PATH="${LIVESERVE_SHAREGPT_DATASET:-${REPO_ROOT}/data/ShareGPT_Chinese_English_90k.json}"
LOG_DIR="${WORK_DIR}/logs/${FEATURE}"
AUDIO_DIR="${WORK_DIR}/output_audio"
PROBE_SCRIPT="${WORK_DIR}/single_probe.py"

mkdir -p "${LOG_DIR}" "${AUDIO_DIR}"

# ========== Preflight checks ==========
check_service() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${ts}] GET /health"
    if ! curl -sf "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
        echo "[ERROR] vllm-omni is not ready: http://${HOST}:${PORT}/health"
        echo "        Start the server before running this helper."
        exit 1
    fi
    echo "[${ts}] service health check passed"
}
check_dataset() {
    if [[ ! -f "${DATASET_PATH}" ]]; then
        echo "[ERROR] dataset does not exist: ${DATASET_PATH}"
        echo "        Run: python3 ${WORK_DIR}/download_dataset.py"
        exit 1
    fi
}

# Add timestamps around probes/benchmarks for server-log correlation.
banner_ts() {
    local tag="$1"
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "======== ${ts} :: ${tag} ========"
}

echo "============================================"
echo "Feature     : ${FEATURE}"
echo "Mode        : ${MODE}"
echo "Service     : http://${HOST}:${PORT}"
echo "Model path  : ${MODEL_PATH}"
echo "Served name : ${SERVED_MODEL_NAME}"
echo "Dataset     : ${DATASET_PATH}"
echo "Log dir     : ${LOG_DIR}"
echo "Audio dir   : ${AUDIO_DIR}"
echo "Python      : ${PYBIN}"
echo "vLLM bench  : ${VLLM_BENCH_BIN}"
echo "============================================"

if [[ -z "${VLLM_BENCH_BIN}" ]]; then
    echo "[ERROR] vllm bench binary not found; set LIVESERVE_VLLM_BENCH_BIN."
    exit 1
fi

# ========== Common vLLM bench serve arguments ==========
# Sampling params must be passed via --extra-body; CLI temperature/top-k flags
# are not merged into the omni request body here.
COMMON_ARGS=(
    --omni
    --port "${PORT}"
    --model "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --backend openai-chat-omni
    --endpoint /v1/chat/completions
    --dataset-name sharegpt
    --dataset-path "${DATASET_PATH}"
    --sharegpt-output-len 128
    --ready-check-timeout-sec 600
)

# ========== Mode implementations ==========

# Mode 1: one text-only sample.
run_probe_text() {
    check_service
    check_dataset
    local log="${LOG_DIR}/probe_text.log"
    banner_ts "probe_text START"
    echo ">>> [probe_text] single_probe.py (text-only) -> ${log}"
    "${PYBIN}" "${PROBE_SCRIPT}" \
        --url "http://${HOST}:${PORT}/v1/chat/completions" \
        --model "${SERVED_MODEL_NAME}" \
        --dataset "${DATASET_PATH}" \
        --index 0 \
        --max-tokens 128 \
        --max-prompt-chars 500 \
        2>&1 | tee "${log}"
    banner_ts "probe_text END"
}

# Mode 2: one text+audio sample.
run_probe_audio() {
    check_service
    local log="${LOG_DIR}/probe_audio.log"
    banner_ts "probe_audio START"
    echo ">>> [probe_audio] single_probe.py --with-audio -> ${log}"
    echo ">>> audio will be written under ${AUDIO_DIR}/"
    # A short Chinese prompt is a reliable full-pipeline smoke input.
    "${PYBIN}" "${PROBE_SCRIPT}" \
        --url "http://${HOST}:${PORT}/v1/chat/completions" \
        --model "${SERVED_MODEL_NAME}" \
        --audio-dir "${AUDIO_DIR}" \
        --with-audio \
        --custom-prompt "你好，请用一句话介绍一下你自己。" \
        --max-tokens 128 \
        2>&1 | tee "${log}"
    banner_ts "probe_audio END"
}

# Mode 3: text-only batch load at three concurrency points.
run_bench_text() {
    check_service
    check_dataset
    # modalities=["text"] keeps E2E on the thinker path.
    local extra='{"temperature":0.7,"top_p":0.8,"top_k":20,"modalities":["text"]}'
    local CONCS=(1 4 8)
    local PROMPTS=(10 40 80)
    for i in "${!CONCS[@]}"; do
        local c=${CONCS[$i]}
        local p=${PROMPTS[$i]}
        local log="${LOG_DIR}/bench_text_c${c}_p${p}.log"
        echo ">>> [bench_text] c=${c} p=${p} -> ${log}"
        "${VLLM_BENCH_BIN}" bench serve \
            "${COMMON_ARGS[@]}" \
            --num-prompts "${p}" \
            --max-concurrency "${c}" \
            --request-rate inf \
            --extra-body "${extra}" \
            --percentile-metrics ttft,tpot,itl,e2el \
            > "${log}" 2>&1
        echo "    Finished."
        grep -E "Successful|Failed|Total generated|Output token throughput|Mean TTFT|Mean TPOT|Mean E2EL" "${log}" || true
        echo
    done
}

# Mode 4: text+audio batch load through the full three-stage path.
run_bench_audio() {
    check_service
    check_dataset
    # Explicit modalities trigger thinker -> talker -> code2wav.
    # Sampling params must be carried through extra-body.
    local extra='{"temperature":0.7,"top_p":0.8,"top_k":20,"modalities":["text","audio"]}'
    local CONCS=(1 4 8)
    local PROMPTS=(5 20 40)   # Full audio pipeline is expensive; keep runs short.
    for i in "${!CONCS[@]}"; do
        local c=${CONCS[$i]}
        local p=${PROMPTS[$i]}
        local log="${LOG_DIR}/bench_audio_c${c}_p${p}.log"
        echo ">>> [bench_audio] c=${c} p=${p} -> ${log}"
        "${VLLM_BENCH_BIN}" bench serve \
            "${COMMON_ARGS[@]}" \
            --num-prompts "${p}" \
            --max-concurrency "${c}" \
            --request-rate inf \
            --extra-body "${extra}" \
            --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
            > "${log}" 2>&1
        echo "    Finished."
        grep -E "Successful|Failed|Total generated|Audio|Mean TTFT|Mean E2EL|audio_ttfp|audio_rtf" "${log}" || true
        echo
    done
}

# ========== Dispatch ==========
case "${MODE}" in
    probe_text)   run_probe_text ;;
    probe_audio)  run_probe_audio ;;
    bench_text)   run_bench_text ;;
    bench_audio)  run_bench_audio ;;
    all)
        run_probe_text
        run_probe_audio
        run_bench_text
        run_bench_audio
        ;;
    *)
        echo "[ERROR] Unknown mode: ${MODE}"
        echo "Options: probe_text | probe_audio | bench_text | bench_audio | all"
        exit 1
        ;;
esac

echo
echo "============================================"
echo "All tasks finished. Logs under: ${LOG_DIR}"
ls -la "${LOG_DIR}/" 2>/dev/null || true
echo "============================================"
