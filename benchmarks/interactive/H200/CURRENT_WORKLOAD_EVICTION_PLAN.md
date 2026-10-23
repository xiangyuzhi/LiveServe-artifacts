# Current Interactive Eviction Plan

Updated: 2026-06-23

This is the compact source of truth for the current H200 interactive
eviction-only work.  Older tuning reports and failed canaries were removed from
the tracked tree; raw artifacts remain under an ignored external artifact root.

## Decision

Promote G1 as the current text-only eviction strategy:

- `expected_gap_cost_aware` scoring.
- Heap-index candidate selection with `EVICT_INDEX_TAIL=64`.
- Prefix/session group victim selection.
- Near-gap/history protection.
- Request-level KV stats from worker-side snapshots.

G1 is useful when the workload creates real retained-history KV pressure.  It
is not a standalone retained-audio promotion: the current full-stack retained
audio path improves text TTFP but regresses audio TTFP, E2E, and continuity.

## Workload

Trace source:

```text
<interactive-conversation checkout>/Interactive-conversation-workload/total_workload
```

Repo trace parts:

```text
trace/total_traces_part1.txt
...
trace/total_traces_part7.txt
```

Main pressure window:

| item | value |
|---|---:|
| trace window | `[10200,10500)` |
| requests | `11724` |
| users/sessions | `2334` |
| same-run returns | `9390` |
| same-run return ratio | `80.1%` |
| open sessions peak | `1800` |
| idle pending KV peak estimate | `49.953 GiB` |

This replaced earlier small windows that had too little same-run session reuse
to evaluate eviction.

## Result

Full all-part text-only run:

| variant | throughput | wall | hit ratio | recent misses | eviction blocks |
|---|---:|---:|---:|---:|---:|
| LRU | `15.2539 rps` | `768.59s` | `0.747` | `4501` | `174864` |
| G1 | `17.3602 rps` | `675.34s` | `0.806` | `3546` | `161152` |

G1 versus LRU:

| metric | delta |
|---|---:|
| throughput | `+13.8%` |
| wall time | `-12.1%` |
| TTFP mean / p50 / p90 | `-13.2% / -16.9% / -11.0%` |
| E2E p50 / p90 / p95 / p99 | `-8.7% / -13.9% / -13.9% / -15.0%` |
| E2E improved rate | `74.7%` |

The latest rejected variant was
`expected_gap_cost_response_tail_aware + G1 group/index`; it regressed
throughput, TTFP, E2E, hit ratio, and recent misses versus G1.

## Current Env Bundle

```text
EVICT_POLICY=expected_gap_cost_aware
EVICT_INDEX=1
EVICT_INDEX_TAIL=64
EVICT_INDEX_PARTIAL_ON_NOT_ENOUGH_CANDIDATES=1
EVICT_MIN_CACHED=1
EVICT_MAX_CACHED=16
EVICT_PARTIAL_OVER_CAP=1
EVICT_INDEX_MIN_PRESSURE_ATTEMPTS=0
EVICT_ENFORCE_MIN_PRESSURE_ATTEMPTS=0
EVICT_SELECTED_PROTECT_NEAR_GAP_SEC=15
EVICT_SELECTED_PROTECT_NEAR_HISTORY_TOKENS=999999
EVICT_MAX_RECENT_APPLIED_64=48
EVICT_MAX_RECENT_NO_CACHED_64=32
EVICT_INDEX_GROUP_BY_PREFIX=1
EVICT_INDEX_GROUP_MAX_GROUPS_PER_ALLOC=4
EVICT_INDEX_GROUP_POP_MULTIPLIER=8
EVICT_INDEX_GROUP_REQUIRE_TAIL_CONTIGUOUS=1
EVICT_INDEX_GROUP_PARTIAL_ON_SHORT=1
EVICT_INDEX_GROUP_FILL_REMAINING=1
```

The shared full-stack helper is:

```text
profiling/scheduler_analysis/scripts/eviction_policy_common.sh
```

Core implementation:

```text
vllm-omni/vllm_omni/core/sched/kv_eviction_shadow.py
vllm-omni/vllm_omni/patch.py
benchmarks/interactive/H200/analyze_eviction_regression_profile.py
```

## Artifacts

```text
<artifact-root>/interactive_reqkv_fullG1_all8_win10200_10500_20260622_063948
<artifact-root>/interactive_reqkv_fullG1_all8_win10200_10500_20260622_063948/_analysis_reqkv_G1_vs_lru/regression_profile.md
<artifact-root>/interactive_reqkv_G6respTail_G1base_full_all8_win10200_10500_20260622_135002
```

## Reproduction

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

## Next Work

1. Keep G1 as the text-only eviction baseline and do not reopen failed
   response-tail or simple guard variants without a new profile.
2. For retained audio, profile the prefix/offload/preload handoff before
   another eviction sweep.
3. For publishable E2E numbers, force or report output length; otherwise treat
   TTFP, stage0 first-output, throughput, and request-level KV stats as the
   promotion metrics.
