# StreamingBench Video Workload

This directory adapts StreamingBench to LiveServe's
`/v1/video/chat/stream` WebSocket endpoint.  It is the current video workload
for validating interaction-aware scheduling, retained-session KV preload, and
G1 eviction on Qwen3-Omni.

Repository-level conclusions live in:

- [`../../docs/README.md`](../../docs/README.md)
- [`../../docs/experiments/final_reproduction_matrix.md`](../../docs/experiments/final_reproduction_matrix.md)

## Data

Clone and prepare StreamingBench outside this repository:

```bash
git clone https://github.com/THUNLP-MT/StreamingBench.git /path/to/StreamingBench
```

Expected local shape:

```text
/path/to/StreamingBench/src/data/
+-- questions_real_stream.json
+-- questions_omni_stream.json
+-- videos/
+-- audio/                 # optional input audio
```

Validate the data root:

```bash
python benchmarks/streamingbench/prepare_full_streamingbench.py \
  --streamingbench-root /path/to/StreamingBench \
  --attempt-link-missing \
  --allow-missing
```

## Current Variants

Use these names with `run_paper_video_matrix.sh`:

| variant | purpose |
|---|---|
| `baseline_no_offload` | strict zero-optimization baseline: scheduler off, preload off, custom eviction off |
| `liveserve_scheduler` | scheduler-only attribution line |
| `liveserve_g1` | current full stack: scheduler `talkerlead=12:6:75`, retained-session KV preload, G1 `expected_gap_cost_aware` eviction with prefix/session group victims |

`liveserve_g1` is the current full-stack video recommendation.  Older variant
aliases remain accepted by scripts for historical artifacts.

## Current Matrix

The current H200 video matrix uses Qwen3-Omni, all 8 GPUs, multiturn video
sessions, one follow-up turn, and audio output:

```bash
cd <repo>
RESULT_BASE=results/video/qwen_video_g1_$(date -u +%Y%m%d_%H%M%S) \
SERVER_MOE_BACKEND=triton \
SERVER_INIT_TIMEOUT=2400 \
SERVER_STAGE_INIT_TIMEOUT=1800 \
VLLM_OMNI_STAGE_INIT_MAX_WORKERS=7 \
GPU_COUNTS=8 \
LIVESERVE_GPU_DEVICES_8GPU=0,1,2,3,4,5,6,7 \
VARIANTS="baseline_no_offload liveserve_g1" \
MODES=baseline \
CONCURRENCIES_8GPU="1 2 4" \
FIXED_REPEAT=2 \
MAX_VIDEOS=2 \
MAX_QUESTIONS_PER_VIDEO=2 \
STREAMINGBENCH_DIALOGUE_MODE=multiturn \
STREAMINGBENCH_FOLLOWUP_TURNS_PER_QUESTION=1 \
STREAMINGBENCH_WITH_AUDIO_OUTPUT=1 \
STREAMINGBENCH_PROMPT_STYLE=scripted_long_answer \
STREAMINGBENCH_METADATA=/path/to/StreamingBench/src/data/questions_real_stream.json \
STREAMINGBENCH_DATA_ROOT=/path/to/StreamingBench/src/data \
bash benchmarks/streamingbench/run_paper_video_matrix.sh
```

For strict-output follow-up runs, add:

```bash
STREAMINGBENCH_MAX_TOKENS=256 \
STREAMINGBENCH_MIN_TOKENS=255 \
STREAMINGBENCH_IGNORE_EOS=1 \
STREAMINGBENCH_DISABLE_THINKER_STOP_TOKEN_IDS=1 \
```

These are benchmark controls, not a guarantee of identical output length.  The
current Qwen video path can still terminate early, so report `text_chars_mean`
and `audio_duration_mean` with E2E/qps.

## Current Result Summary

Latest post-merge wiring smoke used one real StreamingBench video question and
audio output.  It is a deployment/regression check, not a performance claim:

| variant | ok/errors | text TTFT p50 | audio TTFP p50 | E2E p50 | continuity |
|---|---:|---:|---:|---:|---:|
| `baseline` | `1/0` | `0.548s` | `0.976s` | `1.636s` | `1.000` |
| `liveserve_scheduler` | `1/0` | `0.462s` | `0.909s` | `1.583s` | `1.000` |

Result root:
`qwen_streamingbench_video_audio_current_smoke_20260624_235245`.  The scheduler
server log confirms `TalkerLeadGovernor enabled` on the talker stage.  A
separate text-only video endpoint smoke
(`qwen_streamingbench_video_current_smoke_20260624_234505`) validates the video
request path but does not exercise the talker scheduler.

For audio-output scheduler runs, the H200 video scheduler launchers set
`SCHEDULER_POLICY_DEFAULT_TALKER_MIN_OUTPUT_TOKENS=48` by default.  This keeps
very short video answers from finishing before a deferred talker emits audio,
while still allowing callers to override the guard for tuning sweeps.  Guard
verification root:
`qwen_streamingbench_video_audio_scheduler_guard48_smoke_20260625_011404`;
`liveserve_scheduler` completed 4/4 requests with zero errors, 4/4 audio-output
requests evaluated, and continuity 1.0.

Current-tree baseline-vs-G1 smoke after the branch-integration cleanup:
`qwen_streamingbench_g1_baseline_currenttree_smoke_20260625_171348`.
It used real `questions_real_stream.json` metadata, GPU4-5, audio output,
`baseline` and `liveserve_g1`, one concurrency cell, and 16 deduplicated video
requests per variant.  This is still a smoke, not a replacement for the c4
performance artifacts.

| variant | ok/errors | audio TTFP p50 | audio TTFP p90 | E2E p50 | continuity | KV preload |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | `16/0` | `0.523s` | `0.684s` | `0.877s` | `1.000` | `0/0` |
| `liveserve_g1` | `16/0` | `0.527s` | `0.576s` | `1.161s` | `1.000` | `12/12` |

Fair-output c4 showed the strongest full-stack video case:

| metric | baseline | `liveserve_g1` |
|---|---:|---:|
| qps | `0.430` | `0.625` |
| audio TTFP p90 | `5.953s` | `3.283s` |
| continuity | `0.867` | `0.933` |
| continuity goodput | `0.349` | `0.547` |

Strict-output follow-up keeps the audio TTFP p90 win but removes the qps/goodput
headline:

| metric | baseline | `liveserve_g1` |
|---|---:|---:|
| qps | `0.639` | `0.500` |
| audio TTFP p90 | `5.712s` | `3.652s` |
| E2E p90 | `6.742s` | `5.888s` |
| continuity | `1.000` | `0.933` |
| continuity goodput | `0.599` | `0.438` |

`OMNI_PREFETCH_SERVER_MAX_INFLIGHT=2` recovered qps/goodput in strict-output c4
but worsened audio TTFP (`3.652s -> 4.390s`).  Keep the default preload
inflight setting for TTFP-first video tests; the environment variable keeps its
implementation name for compatibility.

The exact recorded artifact paths for these checks are encoded in
`scripts/validate_qwen_system_artifacts.py`.

## Driver Outputs

Each cell writes:

- `per_request.jsonl`: per-request latency, answer, audio timeline, continuity,
  and KV preload status.
- `summary.json`: aggregate latency, qps, accuracy, RTF, continuity, text
  length, audio duration, and KV metadata/preload stats.
- `meta.json`: argv and run metadata.

The summary script writes `paper_figures/paper_summary.csv` and plots under the
run root.  Raw outputs belong under ignored `results/video/` or an external
artifact root such as `$LIVESERVE_ARTIFACT_ROOT`.

## Driver KV Controls

- `--enable-kv-metadata`: attach session, prefix hash, token estimates, and
  expected next-use metadata.
- `--enable-kv-preload`: call `/v1/omni/prefetch_session` before live turns
  with committed history.
- `--preload-scope history_preload`: current paper setting.
- `--disable-thinker-stop-token-ids`: benchmark-only output control used with
  `--ignore-eos` and `--min-tokens`.

The driver still accepts older preload flag aliases for historical automation.
Keep current-turn preload and older audio-anchor probes as separate ablations;
they are not the current video headline path.
