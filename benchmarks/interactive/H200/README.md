# H200 Interactive Benchmarks

This directory keeps the active H200 interactive trace replay, KV eviction, and
profiling tools.  Raw run artifacts belong under an ignored external artifact
root such as `$LIVESERVE_ARTIFACT_ROOT` or `results/`;
only reusable runners, analysis helpers, and compact current-state notes should
live here.

For the repository-level Qwen system summary, start with:

- [`../../../docs/README.md`](../../../docs/README.md)
- [`../../../docs/experiments/final_reproduction_matrix.md`](../../../docs/experiments/final_reproduction_matrix.md)

## Active Runners

Core replay and eviction runners:

- `trace_replay_driver.py`: retained multi-turn trace replay client.
- `client_bargein.py`: audio/barge-in client path used by H200 scripts.
- `scheduler_run_one_cell.sh`: one scheduler/barge-in interactive cell.
- `step2_scheduler_chain.sh` and `step2_scheduler_chain_prob.sh`: scheduler
  sweeps over lead and barge-in probability.
- `run_bargein3gpu_matrix.sh`: three-GPU barge-in matrix runner.
- `run_original_baseline_vs_eviction_gpu45.sh`: canonical LRU/custom-eviction
  runner; GPU IDs are configurable despite the filename.
- `run_original_baseline_vs_eviction_gpu67.sh`: GPU6-7 wrapper.
- `run_original_baseline_vs_eviction_gpuall8.sh`: all-8-GPU wrapper.
- `run_eviction_fast_iter_gpu67.sh`: shorter screening runner.
- `run_eviction_fast_iter_gpuall8.sh`: all-8-GPU screening wrapper.

Current analysis helpers:

- `scan_interactive_eviction_windows.py`
- `analyze_interactive_eviction_workload.py`
- `simulate_interactive_kv_policies.py`
- `simulate_runtime_aligned_eviction.py`
- `analyze_eviction_regression_profile.py`
- `summarize_eviction_run.py`

Do not add new one-off wrapper scripts for each tuning idea.  Use the generic
runners with environment overrides and record only promoted or
decision-changing results in `CURRENT_WORKLOAD_EVICTION_PLAN.md`.

## Current Eviction Result

The current promoted text-only eviction configuration is G1:

```text
EVICT_POLICY=expected_gap_cost_aware
EVICT_INDEX=1
EVICT_INDEX_TAIL=64
EVICT_MAX_CACHED=16
EVICT_PARTIAL_OVER_CAP=1
EVICT_INDEX_PARTIAL_ON_NOT_ENOUGH_CANDIDATES=1
EVICT_MAX_RECENT_APPLIED_64=48
EVICT_MAX_RECENT_NO_CACHED_64=32
EVICT_SELECTED_PROTECT_NEAR_GAP_SEC=15
EVICT_SELECTED_PROTECT_NEAR_HISTORY_TOKENS=999999
EVICT_INDEX_GROUP_BY_PREFIX=1
EVICT_INDEX_GROUP_MAX_GROUPS_PER_ALLOC=4
EVICT_INDEX_GROUP_POP_MULTIPLIER=8
EVICT_INDEX_GROUP_REQUIRE_TAIL_CONTIGUOUS=1
EVICT_INDEX_GROUP_PARTIAL_ON_SHORT=1
EVICT_INDEX_GROUP_FILL_REMAINING=1
```

On the all-part text-only pressure window `[10200,10500)`, G1 improves over
LRU:

| metric | LRU | G1 | delta |
|---|---:|---:|---:|
| throughput | `15.2539 rps` | `17.3602 rps` | `+13.8%` |
| wall time | `768.59s` | `675.34s` | `-12.1%` |
| TTFP p90 | reference | G1 | `-11.0%` |
| E2E p90 | reference | G1 | `-13.9%` |
| KV hit ratio | `0.747` | `0.806` | `+7.9% relative` |

G1 is promoted for text/stage0 pressure and as the eviction component of the
full-stack `liveserve_g1` variant.  It is not promoted as a retained-audio
standalone win: current all-opt retained audio improves text TTFP but regresses
audio TTFP, E2E, and continuity.

## Reproduction

Full all-part G1 eviction-only run:

```bash
FAST_MODE=realmtdist_micro1200 \
VARIANTS="eviction_only" \
TRACE_PARTS="1 2 3 4 5 6 7" \
TIME_START=10200 \
TIME_LIMIT=10500 \
MAX_REQUESTS=0 \
CONCURRENCY_LIST=64 \
SPEEDUP=64 \
CELL_TIMEOUT_SEC=3600 \
REQUEST_TIMEOUT_SEC=900 \
RUN_WORKLOAD_ANALYSIS=0 \
ENABLE_KV_LRU_STATS=1 \
ENABLE_KV_LRU_REQUEST_STATS=1 \
ENABLE_KV_LRU_EVENT_TRACE=0 \
STREAM_STAGE_TIMINGS=1 \
WITH_AUDIO=0 \
EVICT_POLICY=expected_gap_cost_aware \
EVICT_INDEX_TAIL=64 \
EVICT_MAX_CACHED=16 \
EVICT_PARTIAL_OVER_CAP=1 \
EVICT_INDEX_GROUP_BY_PREFIX=1 \
EVICT_INDEX_GROUP_MAX_GROUPS_PER_ALLOC=4 \
EVICT_INDEX_GROUP_POP_MULTIPLIER=8 \
EVICT_INDEX_GROUP_REQUIRE_TAIL_CONTIGUOUS=1 \
EVICT_INDEX_GROUP_PARTIAL_ON_SHORT=1 \
EVICT_INDEX_GROUP_FILL_REMAINING=1 \
bash benchmarks/interactive/H200/run_eviction_fast_iter_gpuall8.sh
```

Useful artifacts:

```text
<artifact-root>/interactive_reqkv_fullG1_all8_win10200_10500_20260622_063948
<artifact-root>/interactive_reqkv_fullG1_all8_win10200_10500_20260622_063948/_analysis_reqkv_G1_vs_lru/regression_profile.md
```

## Cleanup Policy

- Keep promoted commands and compact metrics here.
- Put raw logs, JSONL probes, generated plots, and per-run summaries under
  ignored result directories.
- Delete or archive obsolete tuning reports once their conclusion is superseded
  by `CURRENT_WORKLOAD_EVICTION_PLAN.md` or the root documentation.
