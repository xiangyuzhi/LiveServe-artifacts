#!/usr/bin/env bash
# Shared environment cleanup helpers for H200 video launchers.

liveserve_unset_prefix() {
  local prefix="$1"
  local name
  while IFS= read -r name; do
    unset "${name}" || true
  done < <(compgen -v "${prefix}")
}

liveserve_clear_bp_runtime_env() {
  liveserve_unset_prefix OMNI_TALKER_
  liveserve_unset_prefix OMNI_THINKER_
  liveserve_unset_prefix OMNI_BP_
}

liveserve_clear_prefetch_env() {
  liveserve_unset_prefix OMNI_PREFETCH_
  unset OMNI_SCHED_NATIVE_PREFETCH OMNI_SCHED_NATIVE_PREFETCH_CANCEL
  unset OMNI_SCHED_NATIVE_PREFETCH_MAX_LIVE
}

liveserve_clear_eviction_env() {
  liveserve_unset_prefix OMNI_KV_EVICT_
  unset OMNI_KV_LRU_STATS OMNI_KV_LRU_STATS_PATH
  unset OMNI_KV_LRU_STATS_LOOKUP_DEDUP_SEC OMNI_KV_LRU_STATS_LOOKUP_DEDUP_SIZE
}

liveserve_clear_video_optimization_env() {
  liveserve_clear_bp_runtime_env
  liveserve_clear_prefetch_env
  liveserve_clear_eviction_env
}
