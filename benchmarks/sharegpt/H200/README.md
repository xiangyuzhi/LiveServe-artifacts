# ShareGPT Workload - H200

This folder contains the H200 ShareGPT clients used by the current LiveServe
evaluation.  ShareGPT covers single-turn text/audio sanity checks, synthetic
length-aligned pressure tests, and the paper arrival/barge-in workloads.

## Dataset

Prepare the local dataset from Hugging Face:

```bash
bash scripts/prepare_sharegpt.sh
```

The script downloads `shareAI/ShareGPT-Chinese-English-90k`, converts it to the
vLLM ShareGPT schema, and writes:

```text
data/ShareGPT_Chinese_English_90k.json
```

Useful overrides:

```bash
export LIVESERVE_DATA_ROOT=/path/to/data
export LIVESERVE_SHAREGPT_DATASET=/path/to/ShareGPT_Chinese_English_90k.json
```

## Current Scripts

| Script | Purpose |
|---|---|
| `sharegpt_driver.py` | Main open-loop/closed-loop ShareGPT client |
| `generate_burstgpt_arrivals.py` | Convert BurstGPT CSV windows to ShareGPT arrival manifests |
| `run_paper_workloads_gpu67.sh` | Paper ShareGPT audio Poisson/BurstGPT and barge-in runner |
| `summarize_paper_workloads.py` | Compact markdown/CSV summary for paper workload runs |
| `run_shareGPT-len1024.sh` | Length-aligned synthetic prompt/response wrapper |
| `download_dataset.py`, `verify_dataset.py` | Dataset preparation helpers |
| `single_probe.py`, `sharegpt_benchmark.sh` | Historical simple probes |
| `plot_controlled.py` and archived plotting helpers | Legacy diagnostics |

## Original ShareGPT Smoke

Start a text/audio LiveServe service first, then run:

```bash
LIVESERVE_PYBIN=${LIVESERVE_PYBIN:-/path/to/venv/bin/python}
LIVESERVE_MODEL=${LIVESERVE_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}

$LIVESERVE_PYBIN benchmarks/sharegpt/H200/sharegpt_driver.py \
  --dataset data/ShareGPT_Chinese_English_90k.json \
  --url http://127.0.0.1:8666/v1/chat/completions \
  --model "$LIVESERVE_MODEL" \
  --mode audio \
  --concurrency 2 \
  --num-requests 4 \
  --max-tokens 48 \
  --max-prompt-chars 500 \
  --result-dir benchmarks/sharegpt/H200/results/sharegpt_orig_smoke
```

## Length-Aligned Runs

The wrapper is named `run_shareGPT-len1024.sh`, but it accepts token mean
overrides for `len2048` and `len4096`.

```bash
PORT=8666 MODE=audio FORCE_OUTPUT_LEN=1 \
  bash benchmarks/sharegpt/H200/run_shareGPT-len1024.sh

RUN_ID=shareGPT-len2048 INPUT_MEAN=1024 OUTPUT_MEAN=2048 \
  PORT=8666 MODE=audio FORCE_OUTPUT_LEN=1 \
  bash benchmarks/sharegpt/H200/run_shareGPT-len1024.sh
```

Useful overrides: `MODE=text|audio`, `REQUEST_RATE`, `MAX_INFLIGHT`,
`NUM_REQUESTS`, `INPUT_MEAN`, `OUTPUT_MEAN`, and `FORCE_OUTPUT_LEN=1`.

## Paper Arrival And Barge-In Workloads

The paper-specific ShareGPT audio workloads are:

- arrival distribution: Poisson at `4` RPS vs a BurstGPT trace window with
  `32` requests and the same peak request rate;
- client barge-in: Bernoulli barge-in probability, anchored after audio TTFP,
  with either fixed-delay smoke timing or paper-aligned delay samples from a
  no-barge-in audio-duration distribution.

The integrated runner starts one 2-GPU text/audio server per variant and
defaults to GPUs 6-7.  On the shared machine, prefer two shorter phases under
one `RESULT_ROOT` so arrival and barge-in cases get separate server lifetimes
and separate server logs:

```bash
export RESULT_ROOT=benchmarks/sharegpt/H200/results/paper_workloads_$(date -u +%Y%m%d_%H%M%S)

TEXT_GPUS=6,7 \
TEXT_PORT=8896 \
SERVER_MOE_BACKEND=triton \
VARIANTS="baseline scheduler_l8 scheduler_l12 scheduler_l16" \
NUM_REQUESTS=32 \
CONCURRENCY=8 \
MAX_INFLIGHT=8 \
MAX_TOKENS=128 \
REQUEST_RATE=4.0 \
RUN_ARRIVAL=1 \
RUN_BARGE_IN=0 \
SERVER_DIR_SUFFIX=arrival \
CASE_MIN_OK=32 \
CASE_MAX_ERRORS=0 \
GPU_MIN_FREE_GIB=120 \
GPU_WAIT_FOR_FREE_SEC=1800 \
GPU_FREE_STABLE_SAMPLES=2 \
GPU_REQUIRE_NO_COMPUTE_APPS=1 \
SERVER_READY_TIMEOUT_SEC=300 \
bash benchmarks/sharegpt/H200/run_paper_workloads_gpu67.sh

TEXT_GPUS=6,7 \
TEXT_PORT=8896 \
SERVER_MOE_BACKEND=triton \
VARIANTS="baseline scheduler_l8_h25" \
NUM_REQUESTS=32 \
CONCURRENCY=8 \
MAX_INFLIGHT=8 \
MAX_TOKENS=128 \
RUN_ARRIVAL=0 \
RUN_BARGE_IN=1 \
SERVER_DIR_SUFFIX=bargein \
BARGE_IN_PROBS="0.0 0.3 0.7 1.0" \
BARGE_IN_AFTER_DISTRIBUTION=samples \
BARGE_IN_AFTER_SAMPLES=auto \
BARGE_IN_AFTER_SAMPLE_KEY=audio_duration_s \
BARGE_IN_AFTER_SAMPLE_TIME_SCALE=0.25 \
POSTHOC_PLAYBACK_BARGE_IN_WASTE=1 \
BARGE_IN_TIME_ANCHOR=ttfp \
CASE_WALL_TIMEOUT_SEC=600 \
STOP_ON_CASE_FAILURE=1 \
CASE_MIN_OK=32 \
CASE_MAX_ERRORS=0 \
GPU_MIN_FREE_GIB=120 \
GPU_WAIT_FOR_FREE_SEC=1800 \
GPU_FREE_STABLE_SAMPLES=2 \
GPU_REQUIRE_NO_COMPUTE_APPS=1 \
SERVER_READY_TIMEOUT_SEC=300 \
bash benchmarks/sharegpt/H200/run_paper_workloads_gpu67.sh

benchmarks/sharegpt/H200/summarize_paper_workloads.py \
  "$RESULT_ROOT" \
  --baseline-variant baseline \
  --min-ok 32 \
  --max-errors 0
```

Useful overrides:

- Slot check without server launch:

  ```bash
  TEXT_GPUS=6,7 GPU_MIN_FREE_GIB=120 GPU_REQUIRE_NO_COMPUTE_APPS=1 \
    GPU_PRECHECK_ONLY=1 GPU_WAIT_FOR_FREE_SEC=0 \
    bash benchmarks/sharegpt/H200/run_paper_workloads_gpu67.sh
  ```

- The runner default is the paper grid
  `BARGE_IN_PROBS="0.0 0.3 0.7 1.0"`.  Use `"0.0 0.5 0.75"` only for compact
  high-barge-in tuning points.
- `BARGE_IN_AFTER_DISTRIBUTION=samples` with
  `BARGE_IN_AFTER_SAMPLES=/path/to/no-bargein/per_request.jsonl` gives the
  paper-aligned barge-in cut time.  Set `BARGE_IN_AFTER_SAMPLES=auto` to let
  the runner first execute a baseline no-barge-in duration probe under
  `_duration_probe/` and reuse its `per_request.jsonl`.
- Keep `BARGE_IN_AFTER_SAMPLE_TIME_SCALE=1.0` for strict paper reproduction.
  Smaller values are server-visible stress/tuning settings and should be
  reported separately.
- Enable `POSTHOC_PLAYBACK_BARGE_IN_WASTE=1` for sampled-duration runs to
  record generated-but-unplayed audio when the sampled playback cut happens
  after the HTTP stream has already completed.
- Use `VARIANTS="baseline scheduler_l8 scheduler_l12 scheduler_l16"` for
  arrival scheduler lead tuning.  For barge-in performance claims, keep
  `baseline` as the current recommendation unless a new repeated scheduler win
  is established under a fixed duration-sample file.
- Add `SKIP_VALID_CASES=1` when resuming a partially completed `RESULT_ROOT`;
  variants whose requested cases already pass the formal summary gate are
  skipped before server launch.

Older barge-in and scheduler aliases remain accepted for historical automation,
but new commands should use `RUN_BARGE_IN`, `BARGE_IN_*`, and `scheduler_*`.

To generate only the arrival manifest:

```bash
benchmarks/sharegpt/H200/generate_burstgpt_arrivals.py \
  --output /tmp/burstgpt_sharegpt32.jsonl \
  --num-requests 32 \
  --log-type api \
  --window-mode peak \
  --window-sec 300 \
  --slice-mode densest \
  --target-peak-rps 4.0
```

Abort-path smoke only:

```bash
RESULT_ROOT=benchmarks/sharegpt/H200/results/bargein_smoke_$(date -u +%Y%m%d_%H%M%S) \
TEXT_GPUS=6,7 \
TEXT_PORT=8897 \
VARIANTS="baseline" \
NUM_REQUESTS=16 \
CONCURRENCY=4 \
MAX_INFLIGHT=4 \
MAX_TOKENS=64 \
RUN_ARRIVAL=0 \
RUN_BARGE_IN=1 \
BARGE_IN_PROBS="1.0" \
BARGE_IN_AFTER_DISTRIBUTION=samples \
BARGE_IN_AFTER_SAMPLES=auto \
BARGE_IN_AFTER_SAMPLE_TIME_SCALE=0.1 \
CASE_MIN_OK=16 \
CASE_MAX_ERRORS=0 \
bash benchmarks/sharegpt/H200/run_paper_workloads_gpu67.sh
```

## Notes

- Original ShareGPT is the cleanest workload for reproducing scheduler-only
  audio TTFP gains without retained-session KV reuse.
- Full-stack KV management should be evaluated on retained multi-turn workloads
  and video workloads where session history reuse is meaningful.
- Raw outputs under `benchmarks/sharegpt/H200/results/` are ignored by git.
