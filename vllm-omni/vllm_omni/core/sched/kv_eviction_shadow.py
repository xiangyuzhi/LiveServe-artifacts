from __future__ import annotations

import heapq
import json
import math
import os
import time
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_POLICIES = (
    "oracle_next_use",
    "response_len_tail_aware",
    "prefetch_hit_tail_aware",
    "prefetch_hit_soft_tail_aware",
    "audio_late_then_kv",
    "hybrid_late_then_kv",
    "audio_late_tail_aware",
    "hybrid_late_tail_aware",
    "audio_reuse_potential",
    "hybrid_reuse_potential",
    "bidaw_answer_potential",
    "bidaw_hit_potential",
    "activity_size_aware",
    "expected_gap_bucket_response_tail_aware",
    "expected_gap_bucket_response_value_tail_aware",
    "near_gap_response_len_tail_aware",
    "expected_gap_bucket_response_near_cost_tail_aware",
    "expected_gap_bucket_response_cost_tail_aware",
    "expected_gap_bucket_near_cost_tail_aware",
    "expected_gap_bucket_near_gap_tail_aware",
    "expected_gap_bucket_cost_tail_aware",
    "expected_gap_bucket_tail_aware",
    "expected_gap_tail_aware",
    "expected_gap_cost_aware",
    "history_cost_aware",
    "expected_gap_size_cost_aware",
    "expected_gap_value_cost_aware",
    "expected_gap_response_penalty_cost_aware",
    "expected_gap_cost_response_tail_aware",
    "expected_gap_capacity_adaptive_tail_aware",
)
_DISABLED_ENFORCE_POLICIES = {"", "0", "false", "no", "off", "none", "lru", "actual_lru"}
_SUPPORTED_ENFORCE_POLICIES = {
    "response_len",
    "response_len_tail_aware",
    "reuse_protect_response_len",
    "prefetch_soft_response_len",
    "prefetch_hit_tail_aware",
    "prefetch_hit_soft_tail_aware",
    "hot_protect_response_len",
    "predicted_gap",
    "audio_late_then_kv",
    "hybrid_late_then_kv",
    "audio_late_tail_aware",
    "hybrid_late_tail_aware",
    "audio_reuse_potential",
    "hybrid_reuse_potential",
    "bidaw_answer_potential",
    "bidaw_hit_potential",
    "activity_size_aware",
    "expected_gap_bucket_response_tail_aware",
    "expected_gap_bucket_response_value_tail_aware",
    "expected_gap_bucket_response_loss_tail_aware",
    "near_gap_response_len_tail_aware",
    "expected_gap_bucket_response_near_cost_tail_aware",
    "expected_gap_bucket_response_cost_tail_aware",
    "expected_gap_bucket_near_cost_tail_aware",
    "expected_gap_bucket_near_gap_tail_aware",
    "expected_gap_bucket_cost_tail_aware",
    "expected_gap_bucket_tail_aware",
    "expected_gap_tail_aware",
    "expected_gap_cost_aware",
    "history_cost_aware",
    "expected_gap_size_cost_aware",
    "expected_gap_value_cost_aware",
    "expected_gap_response_penalty_cost_aware",
    "expected_gap_cost_response_tail_aware",
    "expected_gap_capacity_adaptive_tail_aware",
    "cost_aware",
}

_RESPONSE_GAP_SEC = {
    "<=16": 18.0,
    "17-32": 30.0,
    "33-64": 45.0,
    "65-128": 68.0,
    "129-256": 112.0,
    ">256": 191.0,
}

_QUERY_GAP_SEC = {
    "<=16": 39.0,
    "17-32": 39.0,
    "33-64": 44.0,
    "65-128": 57.0,
    "129-256": 68.0,
    ">256": 105.0,
}

_ACTIVITY_SIZE_WEIGHT_SEC = 18.0
_DEFAULT_AUDIO_SEC_PER_RESPONSE_TOKEN = 0.0
_DEFAULT_AUDIO_NEAR_SEC = 20.0
_DEFAULT_INDEX_POP_MULTIPLIER = 32
_DEFAULT_GAP_BUCKET_SEC = 30.0
_GAP_BUCKET_SEC_CACHE: float | None = None
_DEFAULT_ADAPTIVE_TINY_MAX_BLOCKS = 1000
_DEFAULT_ADAPTIVE_NANO_MAX_BLOCKS = 3000
_DEFAULT_ADAPTIVE_MICRO_MAX_BLOCKS = 6000
_DEFAULT_ADAPTIVE_TINY_POLICY = "expected_gap_bucket_response_tail_aware"
_DEFAULT_ADAPTIVE_NANO_POLICY = "expected_gap_cost_aware"
_DEFAULT_ADAPTIVE_MICRO_POLICY = "expected_gap_cost_aware"
_DEFAULT_ADAPTIVE_LARGE_POLICY = "expected_gap_bucket_response_tail_aware"
_ADAPTIVE_TIER_TINY = "TINY"
_ADAPTIVE_TIER_NANO = "NANO"
_ADAPTIVE_TIER_MICRO = "MICRO"
_ADAPTIVE_TIER_LARGE = "LARGE"


class _BlockShadowMeta(Mapping[str, Any]):
    __slots__ = ("_local", "_shared")

    def __init__(self, shared: dict[str, Any], local: dict[str, Any]) -> None:
        self._shared = shared
        self._local = local

    def __getitem__(self, key: str) -> Any:
        if key in self._local:
            return self._local[key]
        return self._shared[key]

    def __iter__(self):
        yield from self._local
        yield from self._shared

    def __len__(self) -> int:
        return len(self._local) + len(self._shared)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._local:
            return self._local[key]
        return self._shared.get(key, default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.3f", name, raw, default)
        return default


def _csv_env(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    return [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]


def _enforcement_log_reason_allowed(reason: str) -> bool:
    allowed = _csv_env("OMNI_KV_EVICT_ENFORCE_LOG_REASONS", ())
    if not allowed:
        return True
    allowed_set = {item.lower() for item in allowed}
    reason_l = str(reason).lower()
    if reason_l in allowed_set:
        return True
    if reason_l == "applied" and "success" in allowed_set:
        return True
    return reason_l != "applied" and "fallback" in allowed_set


def shadow_enabled() -> bool:
    return _env_bool("OMNI_KV_EVICT_SHADOW", False)


def enforcement_policy() -> str | None:
    raw = os.environ.get("OMNI_KV_EVICT_ENFORCE_POLICY", "")
    policy = str(raw).strip()
    if policy.lower() in _DISABLED_ENFORCE_POLICIES:
        return None
    if policy not in _SUPPORTED_ENFORCE_POLICIES:
        logger.warning(
            "[OmniKVEvict] unsupported OMNI_KV_EVICT_ENFORCE_POLICY=%r; "
            "falling back to LRU",
            policy,
        )
        return None
    return policy


def enforcement_enabled() -> bool:
    return enforcement_policy() is not None


def kv_eviction_metadata_enabled() -> bool:
    return shadow_enabled() or enforcement_enabled()


def record_allocation_shape_enabled() -> bool:
    return _env_bool("OMNI_KV_EVICT_RECORD_ALLOC_SHAPE", False)


def _adaptive_capacity_tier(block_pool: Any) -> tuple[str, int]:
    cached_tier = getattr(block_pool, "_omni_kv_evict_adaptive_tier", None)
    cached_capacity = getattr(
        block_pool,
        "_omni_kv_evict_adaptive_capacity_blocks",
        None,
    )
    if isinstance(cached_tier, str) and isinstance(cached_capacity, int):
        return cached_tier, cached_capacity

    capacity_blocks = _block_pool_capacity_blocks(block_pool)
    tiny_max = _env_int(
        "OMNI_KV_EVICT_ADAPTIVE_TINY_MAX_BLOCKS",
        _DEFAULT_ADAPTIVE_TINY_MAX_BLOCKS,
    )
    nano_max = _env_int(
        "OMNI_KV_EVICT_ADAPTIVE_NANO_MAX_BLOCKS",
        _DEFAULT_ADAPTIVE_NANO_MAX_BLOCKS,
    )
    micro_max = _env_int(
        "OMNI_KV_EVICT_ADAPTIVE_MICRO_MAX_BLOCKS",
        _DEFAULT_ADAPTIVE_MICRO_MAX_BLOCKS,
    )
    if capacity_blocks <= 0:
        tier = _ADAPTIVE_TIER_LARGE
    elif capacity_blocks <= tiny_max:
        tier = _ADAPTIVE_TIER_TINY
    elif capacity_blocks <= nano_max:
        tier = _ADAPTIVE_TIER_NANO
    elif capacity_blocks <= micro_max:
        tier = _ADAPTIVE_TIER_MICRO
    else:
        tier = _ADAPTIVE_TIER_LARGE
    try:
        setattr(block_pool, "_omni_kv_evict_adaptive_tier", tier)
        setattr(block_pool, "_omni_kv_evict_adaptive_capacity_blocks", capacity_blocks)
    except Exception:
        pass
    return tier, capacity_blocks


def _capacity_env_int(
    block_pool: Any | None,
    suffix: str,
    global_name: str,
    default: int,
) -> int:
    if block_pool is not None:
        tier, _ = _adaptive_capacity_tier(block_pool)
        tier_name = f"OMNI_KV_EVICT_ADAPTIVE_{tier}_{suffix}"
        if tier_name in os.environ:
            return _env_int(tier_name, default)
    return _env_int(global_name, default)


def _capacity_env_bool(
    block_pool: Any | None,
    suffix: str,
    global_name: str,
    default: bool,
) -> bool:
    if block_pool is not None:
        tier, _ = _adaptive_capacity_tier(block_pool)
        tier_name = f"OMNI_KV_EVICT_ADAPTIVE_{tier}_{suffix}"
        if tier_name in os.environ:
            return _env_bool(tier_name, default)
    return _env_bool(global_name, default)


def _capacity_env_float(
    block_pool: Any | None,
    suffix: str,
    global_name: str,
    default: float,
) -> float:
    if block_pool is not None:
        tier, _ = _adaptive_capacity_tier(block_pool)
        tier_name = f"OMNI_KV_EVICT_ADAPTIVE_{tier}_{suffix}"
        if tier_name in os.environ:
            return _env_float(tier_name, default)
    return _env_float(global_name, default)


def _enforcement_max_cached_victims_per_alloc(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "MAX_CACHED_VICTIMS_PER_ALLOC",
        "OMNI_KV_EVICT_ENFORCE_MAX_CACHED_VICTIMS_PER_ALLOC",
        0,
    )


def _enforcement_min_cached_victims_per_alloc(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "MIN_CACHED_VICTIMS_PER_ALLOC",
        "OMNI_KV_EVICT_ENFORCE_MIN_CACHED_VICTIMS_PER_ALLOC",
        1,
    )


def _enforcement_partial_over_cap(block_pool: Any | None = None) -> bool:
    return _capacity_env_bool(
        block_pool,
        "PARTIAL_OVER_CAP",
        "OMNI_KV_EVICT_ENFORCE_PARTIAL_OVER_CAP",
        False,
    )


def _enforcement_dynamic_replacement(block_pool: Any | None = None) -> bool:
    return _capacity_env_bool(
        block_pool,
        "DYNAMIC_REPLACEMENT",
        "OMNI_KV_EVICT_ENFORCE_DYNAMIC_REPLACEMENT",
        False,
    )


def _enforcement_activate_free_ratio(block_pool: Any | None = None) -> float:
    return _capacity_env_float(
        block_pool,
        "ACTIVATE_FREE_RATIO",
        "OMNI_KV_EVICT_ENFORCE_ACTIVATE_FREE_RATIO",
        0.0,
    )


def _enforcement_protect_actual_max_gap_sec(
    block_pool: Any | None = None,
) -> float:
    return _capacity_env_float(
        block_pool,
        "PROTECT_ACTUAL_MAX_GAP_SEC",
        "OMNI_KV_EVICT_ENFORCE_PROTECT_ACTUAL_MAX_GAP_SEC",
        0.0,
    )


def _enforcement_protect_actual_near_short_gap_sec(
    block_pool: Any | None = None,
) -> float:
    return _capacity_env_float(
        block_pool,
        "PROTECT_ACTUAL_NEAR_SHORT_GAP_SEC",
        "OMNI_KV_EVICT_ENFORCE_PROTECT_ACTUAL_NEAR_SHORT_GAP_SEC",
        0.0,
    )


def _enforcement_protect_actual_near_short_response_len(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "PROTECT_ACTUAL_NEAR_SHORT_RESPONSE_LEN",
        "OMNI_KV_EVICT_ENFORCE_PROTECT_ACTUAL_NEAR_SHORT_RESPONSE_LEN",
        0,
    )


def _enforcement_protect_actual_near_short_selected_terminal_ok(
    block_pool: Any | None = None,
) -> bool:
    return _capacity_env_bool(
        block_pool,
        "PROTECT_ACTUAL_NEAR_SHORT_SELECTED_TERMINAL_OK",
        "OMNI_KV_EVICT_ENFORCE_PROTECT_ACTUAL_NEAR_SHORT_SELECTED_TERMINAL_OK",
        True,
    )


def _enforcement_min_gap_margin_sec(block_pool: Any | None = None) -> float:
    return _capacity_env_float(
        block_pool,
        "MIN_GAP_MARGIN_SEC",
        "OMNI_KV_EVICT_ENFORCE_MIN_GAP_MARGIN_SEC",
        0.0,
    )


def _enforcement_gap_margin_terminal_ok(block_pool: Any | None = None) -> bool:
    return _capacity_env_bool(
        block_pool,
        "GAP_MARGIN_TERMINAL_OK",
        "OMNI_KV_EVICT_ENFORCE_GAP_MARGIN_TERMINAL_OK",
        True,
    )


def _enforcement_min_value_score_margin(block_pool: Any | None = None) -> float:
    return _capacity_env_float(
        block_pool,
        "MIN_VALUE_SCORE_MARGIN",
        "OMNI_KV_EVICT_ENFORCE_MIN_VALUE_SCORE_MARGIN",
        0.0,
    )


def _enforcement_current_cold_or_early_min_gap_margin_sec(
    block_pool: Any | None = None,
) -> float:
    return _capacity_env_float(
        block_pool,
        "CURRENT_COLD_OR_EARLY_MIN_GAP_MARGIN_SEC",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_COLD_OR_EARLY_MIN_GAP_MARGIN_SEC",
        0.0,
    )


def _enforcement_current_cold_prefix_max_tokens(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "CURRENT_COLD_PREFIX_MAX_TOKENS",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_COLD_PREFIX_MAX_TOKENS",
        -1,
    )


def _enforcement_current_early_trace_timestamp_max(
    block_pool: Any | None = None,
) -> float:
    return _capacity_env_float(
        block_pool,
        "CURRENT_EARLY_TRACE_TIMESTAMP_MAX",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_EARLY_TRACE_TIMESTAMP_MAX",
        0.0,
    )


def _enforcement_min_attempts_between_applied(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "MIN_ATTEMPTS_BETWEEN_APPLIED",
        "OMNI_KV_EVICT_ENFORCE_MIN_ATTEMPTS_BETWEEN_APPLIED",
        0,
    )


def _enforcement_max_recent_applied_64(block_pool: Any | None = None) -> int:
    return _capacity_env_int(
        block_pool,
        "MAX_RECENT_APPLIED_64",
        "OMNI_KV_EVICT_ENFORCE_MAX_RECENT_APPLIED_64",
        0,
    )


def _enforcement_max_recent_no_cached_64(block_pool: Any | None = None) -> int:
    return _capacity_env_int(
        block_pool,
        "MAX_RECENT_NO_CACHED_64",
        "OMNI_KV_EVICT_ENFORCE_MAX_RECENT_NO_CACHED_64",
        0,
    )


def _enforcement_low_value_actual_guard_recent_applied_64(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "LOW_VALUE_ACTUAL_GUARD_RECENT_APPLIED_64",
        "OMNI_KV_EVICT_ENFORCE_LOW_VALUE_ACTUAL_GUARD_RECENT_APPLIED_64",
        0,
    )


def _enforcement_low_value_actual_guard_gap_sec(
    block_pool: Any | None = None,
) -> float:
    return _capacity_env_float(
        block_pool,
        "LOW_VALUE_ACTUAL_GUARD_GAP_SEC",
        "OMNI_KV_EVICT_ENFORCE_LOW_VALUE_ACTUAL_GUARD_GAP_SEC",
        0.0,
    )


def _enforcement_low_value_actual_guard_response_len(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "LOW_VALUE_ACTUAL_GUARD_RESPONSE_LEN",
        "OMNI_KV_EVICT_ENFORCE_LOW_VALUE_ACTUAL_GUARD_RESPONSE_LEN",
        0,
    )


def _enforcement_low_value_actual_guard_selected_terminal_ok(
    block_pool: Any | None = None,
) -> bool:
    return _capacity_env_bool(
        block_pool,
        "LOW_VALUE_ACTUAL_GUARD_SELECTED_TERMINAL_OK",
        "OMNI_KV_EVICT_ENFORCE_LOW_VALUE_ACTUAL_GUARD_SELECTED_TERMINAL_OK",
        True,
    )


def _enforcement_max_lost_delta_tokens(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "MAX_LOST_DELTA_TOKENS",
        "OMNI_KV_EVICT_ENFORCE_MAX_LOST_DELTA_TOKENS",
        0,
    )


def _enforcement_lost_delta_min_gap_margin_sec(
    block_pool: Any | None = None,
) -> float:
    return _capacity_env_float(
        block_pool,
        "LOST_DELTA_MIN_GAP_MARGIN_SEC",
        "OMNI_KV_EVICT_ENFORCE_LOST_DELTA_MIN_GAP_MARGIN_SEC",
        0.0,
    )


def _enforcement_lost_delta_selected_terminal_ok(
    block_pool: Any | None = None,
) -> bool:
    return _capacity_env_bool(
        block_pool,
        "LOST_DELTA_SELECTED_TERMINAL_OK",
        "OMNI_KV_EVICT_ENFORCE_LOST_DELTA_SELECTED_TERMINAL_OK",
        True,
    )


def _enforcement_require_selected_terminal(
    block_pool: Any | None = None,
) -> bool:
    return _capacity_env_bool(
        block_pool,
        "REQUIRE_SELECTED_TERMINAL",
        "OMNI_KV_EVICT_ENFORCE_REQUIRE_SELECTED_TERMINAL",
        False,
    )


def _enforcement_selected_max_response_len(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "SELECTED_MAX_RESPONSE_LEN",
        "OMNI_KV_EVICT_ENFORCE_SELECTED_MAX_RESPONSE_LEN",
        0,
    )


def _enforcement_selected_max_history_tokens(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "SELECTED_MAX_HISTORY_TOKENS",
        "OMNI_KV_EVICT_ENFORCE_SELECTED_MAX_HISTORY_TOKENS",
        0,
    )


def _enforcement_selected_max_lost_tokens(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "SELECTED_MAX_LOST_TOKENS",
        "OMNI_KV_EVICT_ENFORCE_SELECTED_MAX_LOST_TOKENS",
        0,
    )


def _enforcement_selected_protect_near_gap_sec(
    block_pool: Any | None = None,
) -> float:
    return _capacity_env_float(
        block_pool,
        "SELECTED_PROTECT_NEAR_GAP_SEC",
        "OMNI_KV_EVICT_ENFORCE_SELECTED_PROTECT_NEAR_GAP_SEC",
        0.0,
    )


def _enforcement_selected_protect_near_history_tokens(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "SELECTED_PROTECT_NEAR_HISTORY_TOKENS",
        "OMNI_KV_EVICT_ENFORCE_SELECTED_PROTECT_NEAR_HISTORY_TOKENS",
        0,
    )


def _enforcement_selected_protect_near_terminal_ok(
    block_pool: Any | None = None,
) -> bool:
    return _capacity_env_bool(
        block_pool,
        "SELECTED_PROTECT_NEAR_TERMINAL_OK",
        "OMNI_KV_EVICT_ENFORCE_SELECTED_PROTECT_NEAR_TERMINAL_OK",
        True,
    )


def _enforcement_selected_min_next_audio_total_sec(
    block_pool: Any | None = None,
) -> float:
    return _capacity_env_float(
        block_pool,
        "SELECTED_MIN_NEXT_AUDIO_TOTAL_SEC",
        "OMNI_KV_EVICT_ENFORCE_SELECTED_MIN_NEXT_AUDIO_TOTAL_SEC",
        0.0,
    )


def _enforcement_selected_min_next_audio_reply_delay_sec(
    block_pool: Any | None = None,
) -> float:
    return _capacity_env_float(
        block_pool,
        "SELECTED_MIN_NEXT_AUDIO_REPLY_DELAY_SEC",
        "OMNI_KV_EVICT_ENFORCE_SELECTED_MIN_NEXT_AUDIO_REPLY_DELAY_SEC",
        0.0,
    )


def _enforcement_current_min_response_len(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "CURRENT_MIN_RESPONSE_LEN",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_MIN_RESPONSE_LEN",
        0,
    )


def _enforcement_current_max_response_len(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "CURRENT_MAX_RESPONSE_LEN",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_MAX_RESPONSE_LEN",
        0,
    )


def _enforcement_current_min_history_tokens(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "CURRENT_MIN_HISTORY_TOKENS",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_MIN_HISTORY_TOKENS",
        0,
    )


def _enforcement_current_max_history_tokens(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "CURRENT_MAX_HISTORY_TOKENS",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_MAX_HISTORY_TOKENS",
        0,
    )


def _enforcement_current_min_prefix_tokens(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "CURRENT_MIN_PREFIX_TOKENS",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_MIN_PREFIX_TOKENS",
        0,
    )


def _enforcement_current_max_prefix_tokens(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "CURRENT_MAX_PREFIX_TOKENS",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_MAX_PREFIX_TOKENS",
        0,
    )


def _enforcement_current_min_round_index(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "CURRENT_MIN_ROUND_INDEX",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_MIN_ROUND_INDEX",
        0,
    )


def _enforcement_current_max_round_index(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "CURRENT_MAX_ROUND_INDEX",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_MAX_ROUND_INDEX",
        0,
    )


def _enforcement_current_min_trace_timestamp(
    block_pool: Any | None = None,
) -> float:
    return _capacity_env_float(
        block_pool,
        "CURRENT_MIN_TRACE_TIMESTAMP",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_MIN_TRACE_TIMESTAMP",
        0.0,
    )


def _enforcement_current_max_trace_timestamp(
    block_pool: Any | None = None,
) -> float:
    return _capacity_env_float(
        block_pool,
        "CURRENT_MAX_TRACE_TIMESTAMP",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_MAX_TRACE_TIMESTAMP",
        0.0,
    )


def _enforcement_current_min_expected_gap_sec(
    block_pool: Any | None = None,
) -> float:
    return _capacity_env_float(
        block_pool,
        "CURRENT_MIN_EXPECTED_GAP_SEC",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_MIN_EXPECTED_GAP_SEC",
        0.0,
    )


def _enforcement_current_min_audio_total_sec(
    block_pool: Any | None = None,
) -> float:
    return _capacity_env_float(
        block_pool,
        "CURRENT_MIN_AUDIO_TOTAL_SEC",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_MIN_AUDIO_TOTAL_SEC",
        0.0,
    )


def _enforcement_current_min_audio_reply_delay_sec(
    block_pool: Any | None = None,
) -> float:
    return _capacity_env_float(
        block_pool,
        "CURRENT_MIN_AUDIO_REPLY_DELAY_SEC",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_MIN_AUDIO_REPLY_DELAY_SEC",
        0.0,
    )


def _enforcement_current_short_response_max_len(
    block_pool: Any | None = None,
) -> int:
    return _capacity_env_int(
        block_pool,
        "CURRENT_SHORT_RESPONSE_MAX_LEN",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_SHORT_RESPONSE_MAX_LEN",
        0,
    )


def _enforcement_current_short_expected_gap_max_sec(
    block_pool: Any | None = None,
) -> float:
    return _capacity_env_float(
        block_pool,
        "CURRENT_SHORT_EXPECTED_GAP_MAX_SEC",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_SHORT_EXPECTED_GAP_MAX_SEC",
        0.0,
    )


def _enforcement_current_require_scheduler_prefetch(
    block_pool: Any | None = None,
) -> bool:
    return _capacity_env_bool(
        block_pool,
        "CURRENT_REQUIRE_SCHEDULER_PREFETCH",
        "OMNI_KV_EVICT_ENFORCE_CURRENT_REQUIRE_SCHEDULER_PREFETCH",
        False,
    )


def _enforcement_max_candidate_free_ratio() -> float:
    return _env_float("OMNI_KV_EVICT_ENFORCE_MAX_CANDIDATE_FREE_RATIO", 0.95)


def _enforcement_log_candidate_detail_limit() -> int:
    return _env_int("OMNI_KV_EVICT_ENFORCE_LOG_CANDIDATE_DETAIL_LIMIT", 0)


def _enforcement_log_candidate_heap_pop_limit() -> int:
    return _env_int("OMNI_KV_EVICT_ENFORCE_LOG_CANDIDATE_HEAP_POP_LIMIT", 4096)


def _enforcement_index_enabled() -> bool:
    return _env_bool("OMNI_KV_EVICT_ENFORCE_INDEX", True)


def _index_respect_current_gate() -> bool:
    return _env_bool("OMNI_KV_EVICT_INDEX_RESPECT_CURRENT_GATE", False)


def _current_gate_early_bypass() -> bool:
    return _env_bool("OMNI_KV_EVICT_CURRENT_GATE_EARLY_BYPASS", False)


def _index_min_cached_victim_density_64() -> float:
    return _env_float("OMNI_KV_EVICT_INDEX_MIN_CACHED_VICTIM_DENSITY_64", 0.0)


def _index_min_pressure_attempts() -> int:
    return _env_int("OMNI_KV_EVICT_INDEX_MIN_PRESSURE_ATTEMPTS", 64)


def _enforcement_min_cached_victim_density_64() -> float:
    return _env_float(
        "OMNI_KV_EVICT_ENFORCE_MIN_CACHED_VICTIM_DENSITY_64",
        0.0,
    )


def _enforcement_min_pressure_attempts() -> int:
    return _env_int("OMNI_KV_EVICT_ENFORCE_MIN_PRESSURE_ATTEMPTS", 64)


def _index_activate_free_ratio() -> float:
    return _env_float("OMNI_KV_EVICT_INDEX_ACTIVATE_FREE_RATIO", 0.0)


def _index_activate_cached_head_scan_blocks() -> int:
    return _env_int("OMNI_KV_EVICT_INDEX_ACTIVATE_CACHED_HEAD_SCAN_BLOCKS", 0)


def _index_prime_scan_blocks() -> int:
    return _env_int("OMNI_KV_EVICT_INDEX_PRIME_SCAN_BLOCKS", 0)


def _index_max_blocks_per_request() -> int:
    return _env_int("OMNI_KV_EVICT_INDEX_MAX_BLOCKS_PER_REQUEST", 0)


def _index_group_by_prefix_enabled() -> bool:
    return _env_bool("OMNI_KV_EVICT_INDEX_GROUP_BY_PREFIX", False)


def _index_group_max_groups_per_alloc() -> int:
    return max(1, _env_int("OMNI_KV_EVICT_INDEX_GROUP_MAX_GROUPS_PER_ALLOC", 4))


def _index_group_pop_multiplier() -> int:
    return max(1, _env_int("OMNI_KV_EVICT_INDEX_GROUP_POP_MULTIPLIER", 4))


def _index_group_require_tail_contiguous() -> bool:
    return _env_bool("OMNI_KV_EVICT_INDEX_GROUP_REQUIRE_TAIL_CONTIGUOUS", True)


def _index_group_partial_on_short() -> bool:
    return _env_bool("OMNI_KV_EVICT_INDEX_GROUP_PARTIAL_ON_SHORT", False)


def _index_group_fill_remaining() -> bool:
    return _env_bool("OMNI_KV_EVICT_INDEX_GROUP_FILL_REMAINING", False)


def _index_adaptive_long_tail() -> int:
    return _env_int("OMNI_KV_EVICT_INDEX_ADAPTIVE_LONG_TAIL", 0)


def _index_adaptive_long_min_blocks() -> int:
    return _env_int("OMNI_KV_EVICT_INDEX_ADAPTIVE_LONG_MIN_BLOCKS", 0)


def _index_tail_limit_for_group(
    base_tail_limit: int,
    long_tail_limit: int,
    long_tail_min_blocks: int,
    num_blocks: int,
) -> int:
    if (
        base_tail_limit > 0
        and long_tail_limit > base_tail_limit
        and long_tail_min_blocks > 0
        and num_blocks >= long_tail_min_blocks
    ):
        return long_tail_limit
    return base_tail_limit


def _enforcement_index_partial_on_not_enough_candidates() -> bool:
    return _env_bool(
        "OMNI_KV_EVICT_ENFORCE_INDEX_PARTIAL_ON_NOT_ENOUGH_CANDIDATES",
        False,
    )


def _capture_min_cached_victim_density_64() -> float:
    return _env_float("OMNI_KV_EVICT_CAPTURE_MIN_CACHED_VICTIM_DENSITY_64", 0.0)


def _capture_min_pressure_attempts() -> int:
    return _env_int("OMNI_KV_EVICT_CAPTURE_MIN_PRESSURE_ATTEMPTS", 0)


def _capture_activate_free_ratio() -> float:
    return _env_float("OMNI_KV_EVICT_CAPTURE_ACTIVATE_FREE_RATIO", 0.0)


def _capture_activate_cached_head_scan_blocks() -> int:
    return _env_int("OMNI_KV_EVICT_CAPTURE_ACTIVATE_CACHED_HEAD_SCAN_BLOCKS", 0)


def _capture_periodic_interval() -> int:
    return _env_int("OMNI_KV_EVICT_CAPTURE_PERIODIC_INTERVAL", 0)


def _capture_respect_replacement_throttle() -> bool:
    return _env_bool("OMNI_KV_EVICT_CAPTURE_RESPECT_REPLACEMENT_THROTTLE", False)


def _capture_require_activation() -> bool:
    return _env_bool("OMNI_KV_EVICT_CAPTURE_REQUIRE_ACTIVATION", False)


def _capture_terminal_only() -> bool:
    return _env_bool("OMNI_KV_EVICT_CAPTURE_TERMINAL_ONLY", False)


def _capture_count_terminal_skips() -> bool:
    return _env_bool("OMNI_KV_EVICT_CAPTURE_COUNT_TERMINAL_SKIPS", True)


def _capture_pressure_warm_attempts() -> int:
    return _env_int("OMNI_KV_EVICT_CAPTURE_PRESSURE_WARM_ATTEMPTS", 0)


def _capture_pressure_warm_min_cached_victims() -> int:
    return _env_int("OMNI_KV_EVICT_CAPTURE_PRESSURE_WARM_MIN_CACHED_VICTIMS", 1)


def _capture_set_block_attr_enabled() -> bool:
    return _env_bool("OMNI_KV_EVICT_CAPTURE_SET_BLOCK_ATTR", True)


def _capture_shared_meta_enabled() -> bool:
    return _env_bool("OMNI_KV_EVICT_CAPTURE_SHARED_META", False)


def _request_kv_transfer_params(request: Any) -> dict[str, Any]:
    params = getattr(request, "kv_transfer_params", None)
    return params if isinstance(params, dict) else {}


def _request_is_terminal_turn(request: Any) -> bool:
    return bool(_request_kv_transfer_params(request).get("omni_is_terminal_turn"))


def _block_pool_capacity_blocks(block_pool: Any) -> int:
    blocks = getattr(block_pool, "blocks", None)
    if blocks is None:
        return 0
    try:
        return len(blocks)
    except TypeError:
        return 0


def _adaptive_policy_for_capacity(block_pool: Any, policy: str) -> str:
    if policy != "expected_gap_capacity_adaptive_tail_aware":
        return policy
    cached_policy = getattr(block_pool, "_omni_kv_evict_adaptive_policy", None)
    if isinstance(cached_policy, str) and cached_policy:
        return cached_policy
    tier, capacity_blocks = _adaptive_capacity_tier(block_pool)
    if tier == _ADAPTIVE_TIER_TINY:
        effective_policy = os.environ.get(
            "OMNI_KV_EVICT_ADAPTIVE_TINY_POLICY",
            _DEFAULT_ADAPTIVE_TINY_POLICY,
        )
    elif tier == _ADAPTIVE_TIER_NANO:
        effective_policy = os.environ.get(
            "OMNI_KV_EVICT_ADAPTIVE_NANO_POLICY",
            _DEFAULT_ADAPTIVE_NANO_POLICY,
        )
    elif tier == _ADAPTIVE_TIER_MICRO:
        effective_policy = os.environ.get(
            "OMNI_KV_EVICT_ADAPTIVE_MICRO_POLICY",
            _DEFAULT_ADAPTIVE_MICRO_POLICY,
        )
    else:
        effective_policy = os.environ.get(
            "OMNI_KV_EVICT_ADAPTIVE_LARGE_POLICY",
            _DEFAULT_ADAPTIVE_LARGE_POLICY,
        )
    try:
        setattr(block_pool, "_omni_kv_evict_adaptive_policy", effective_policy)
        setattr(block_pool, "_omni_kv_evict_adaptive_capacity_blocks", capacity_blocks)
    except Exception:
        pass
    return effective_policy


def _metadata_map(block_pool: Any) -> dict[int, dict[str, Any]]:
    mapping = getattr(block_pool, "_omni_kv_evict_block_meta", None)
    if not isinstance(mapping, dict):
        mapping = {}
        setattr(block_pool, "_omni_kv_evict_block_meta", mapping)
    return mapping


def _shadow_stats(block_pool: Any) -> dict[str, int]:
    stats = getattr(block_pool, "_omni_kv_evict_shadow_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "capture_attempts": 0,
            "captured_requests": 0,
            "captured_blocks": 0,
            "empty_block_requests": 0,
        }
        setattr(block_pool, "_omni_kv_evict_shadow_stats", stats)
    return stats


def _enforcement_log_enabled() -> bool:
    return _env_bool("OMNI_KV_EVICT_ENFORCE_LOG", False)


def _enforcement_log_sample_every() -> int:
    return max(1, _env_int("OMNI_KV_EVICT_ENFORCE_LOG_SAMPLE_EVERY", 1))


def _enforcement_log_always_applied() -> bool:
    return _env_bool("OMNI_KV_EVICT_ENFORCE_LOG_ALWAYS_APPLIED", True)


def _profile_timing_enabled(block_pool: Any | None = None) -> bool:
    if block_pool is None:
        return _env_bool("OMNI_KV_EVICT_PROFILE_TIMING", False)
    cached = getattr(block_pool, "_omni_kv_evict_profile_timing_enabled", None)
    if isinstance(cached, bool):
        return cached
    enabled = _env_bool("OMNI_KV_EVICT_PROFILE_TIMING", False)
    try:
        setattr(block_pool, "_omni_kv_evict_profile_timing_enabled", enabled)
    except Exception:
        pass
    return enabled


def _stats_add(stats: dict[str, int], key: str, delta: int = 1) -> None:
    stats[key] = int(stats.get(key, 0)) + int(delta)


def _record_profile_timing(
    stats: dict[str, int],
    prefix: str,
    elapsed_ns: int,
) -> None:
    elapsed_ns = max(0, int(elapsed_ns))
    _stats_add(stats, f"{prefix}_calls")
    _stats_add(stats, f"{prefix}_elapsed_ns_total", elapsed_ns)
    stats[f"{prefix}_elapsed_ns_max"] = max(
        int(stats.get(f"{prefix}_elapsed_ns_max", 0)),
        elapsed_ns,
    )


def _profile_phase_start(enabled: bool) -> int:
    return time.perf_counter_ns() if enabled else 0


def _record_enforcement_phase_profile(
    block_pool: Any,
    phase: str,
    started_ns: int,
    *,
    item_count: int = 0,
) -> None:
    if started_ns <= 0:
        return
    stats = _enforcement_stats(block_pool)
    prefix = f"enforce_phase_{phase}"
    if item_count > 0:
        _stats_add(stats, f"{prefix}_items_total", item_count)
    _record_profile_timing(
        stats,
        prefix,
        time.perf_counter_ns() - started_ns,
    )


def _record_capture_phase_profile(
    stats: dict[str, int],
    phase: str,
    started_ns: int,
    *,
    item_count: int = 0,
) -> None:
    if started_ns <= 0:
        return
    prefix = f"capture_phase_{phase}"
    if item_count > 0:
        _stats_add(stats, f"{prefix}_items_total", item_count)
    _record_profile_timing(
        stats,
        prefix,
        time.perf_counter_ns() - started_ns,
    )


def _record_allocation_shape(block_pool: Any, num_blocks: int) -> None:
    if not record_allocation_shape_enabled():
        return
    stats = _enforcement_stats(block_pool)
    requested = max(0, int(num_blocks))
    _stats_add(stats, "alloc_shape_calls")
    _stats_add(stats, "alloc_shape_requested_blocks_total", requested)
    stats["alloc_shape_requested_blocks_max"] = max(
        int(stats.get("alloc_shape_requested_blocks_max", 0)),
        requested,
    )
    for bound in (1, 2, 4, 8, 16, 32, 64):
        if requested <= bound:
            _stats_add(stats, f"alloc_shape_requested_le_{bound}")
            break
    else:
        _stats_add(stats, "alloc_shape_requested_gt_64")

    try:
        free_count = int(block_pool.get_num_free_blocks())
    except Exception:
        _stats_add(stats, "alloc_shape_free_count_errors")
        free_count = -1
    if free_count >= 0:
        _stats_add(stats, "alloc_shape_free_blocks_total", free_count)
        stats["alloc_shape_free_blocks_min"] = (
            free_count
            if "alloc_shape_free_blocks_min" not in stats
            else min(int(stats["alloc_shape_free_blocks_min"]), free_count)
        )
        stats["alloc_shape_free_blocks_max"] = max(
            int(stats.get("alloc_shape_free_blocks_max", 0)),
            free_count,
        )

    if requested <= 0:
        return
    try:
        head_blocks = _peek_free_blocks(block_pool, requested)
        cached_head = sum(1 for block in head_blocks if _is_cached_free_block(block))
    except Exception:
        _stats_add(stats, "alloc_shape_head_inspect_errors")
        return

    _stats_add(stats, "alloc_shape_head_blocks_total", len(head_blocks))
    _stats_add(stats, "alloc_shape_cached_head_blocks_total", cached_head)
    stats["alloc_shape_cached_head_blocks_max"] = max(
        int(stats.get("alloc_shape_cached_head_blocks_max", 0)),
        cached_head,
    )
    if cached_head > 0:
        _stats_add(stats, "alloc_shape_calls_with_cached_head")
    for bound in (1, 2, 4, 8, 16, 32, 64):
        if cached_head <= bound:
            _stats_add(stats, f"alloc_shape_cached_head_le_{bound}")
            break
    else:
        _stats_add(stats, "alloc_shape_cached_head_gt_64")


def _record_index_push_profile(
    block_pool: Any,
    started_ns: int,
    reason: str,
) -> None:
    if started_ns <= 0:
        return
    stats = _enforcement_stats(block_pool)
    _stats_add(stats, f"index_push_{reason}")
    _record_profile_timing(
        stats,
        "index_push",
        time.perf_counter_ns() - started_ns,
    )


def _record_index_prime_profile(
    block_pool: Any,
    started_ns: int,
    reason: str,
    *,
    scanned_blocks: int = 0,
    pushed_blocks: int = 0,
) -> None:
    if started_ns <= 0:
        return
    stats = _enforcement_stats(block_pool)
    _stats_add(stats, f"index_prime_{reason}")
    if scanned_blocks > 0:
        _stats_add(stats, "index_prime_scanned_blocks_total", scanned_blocks)
    if pushed_blocks > 0:
        _stats_add(stats, "index_prime_pushed_blocks_total", pushed_blocks)
    _record_profile_timing(
        stats,
        "index_prime",
        time.perf_counter_ns() - started_ns,
    )


def _record_capture_profile(
    stats: dict[str, int],
    started_ns: int,
    *,
    captured_blocks: int,
    considered_blocks: int,
) -> None:
    if started_ns <= 0:
        return
    if captured_blocks > 0:
        _stats_add(stats, "capture_profile_captured_blocks_total", captured_blocks)
    if considered_blocks > 0:
        _stats_add(
            stats,
            "capture_profile_considered_blocks_total",
            considered_blocks,
        )
    _record_profile_timing(
        stats,
        "capture",
        time.perf_counter_ns() - started_ns,
    )


def _record_capture_gate_skip(
    stats: dict[str, int],
    reason: str,
    profile_started_ns: int,
) -> None:
    stats["capture_gate_skipped_requests"] = (
        int(stats.get("capture_gate_skipped_requests", 0)) + 1
    )
    stats[f"capture_gate_skip_{reason}"] = (
        int(stats.get(f"capture_gate_skip_{reason}", 0)) + 1
    )
    _record_capture_profile(
        stats,
        profile_started_ns,
        captured_blocks=0,
        considered_blocks=0,
    )


def _capture_debug_metadata_enabled() -> bool:
    return _env_bool("OMNI_KV_EVICT_CAPTURE_DEBUG_METADATA", False)


def _index_free_ratio_active(block_pool: Any) -> bool:
    max_free_ratio = _index_activate_free_ratio()
    if max_free_ratio <= 0.0:
        return False
    capacity = _block_pool_capacity_blocks(block_pool)
    if capacity <= 0:
        return False
    try:
        free_count = int(block_pool.get_num_free_blocks())
    except Exception:
        return False
    free_ratio = max(0.0, min(1.0, free_count / max(1, capacity)))
    return free_ratio <= max_free_ratio


def _index_cached_head_active(block_pool: Any) -> bool:
    scan_blocks = _index_activate_cached_head_scan_blocks()
    return _cached_free_head_active(block_pool, scan_blocks)


def _cached_free_head_active(block_pool: Any, scan_blocks: int) -> bool:
    if scan_blocks <= 0:
        return False
    try:
        free_count = int(block_pool.get_num_free_blocks())
        blocks = _peek_free_blocks(block_pool, min(scan_blocks, max(0, free_count)))
    except Exception:
        return False
    return any(_is_cached_free_block(block) for block in blocks)


def _free_ratio_below(block_pool: Any, max_free_ratio: float) -> bool:
    if max_free_ratio <= 0.0:
        return False
    capacity = _block_pool_capacity_blocks(block_pool)
    if capacity <= 0:
        return False
    try:
        free_count = int(block_pool.get_num_free_blocks())
    except Exception:
        return False
    free_ratio = max(0.0, min(1.0, free_count / max(1, capacity)))
    return free_ratio <= max_free_ratio


def _index_metadata_active(block_pool: Any) -> bool:
    if _index_respect_current_gate():
        current_gate_reason, _ = _current_request_gate_reason(block_pool)
        if current_gate_reason is not None:
            return False
    min_density = _index_min_cached_victim_density_64()
    if min_density <= 0.0:
        return True
    if _index_free_ratio_active(block_pool) or _index_cached_head_active(block_pool):
        return True
    stats = getattr(block_pool, "_omni_kv_evict_enforce_stats", None)
    if not isinstance(stats, dict):
        return False
    attempts = int(stats.get("attempts", 0))
    min_attempts = max(1, _index_min_pressure_attempts())
    if attempts < min_attempts:
        return False
    denominator = min(64, max(1, attempts))
    recent_cached = int(stats.get("recent_cached_victim_attempts_64", 0))
    density = recent_cached / denominator
    return density >= min_density


def _capture_metadata_gate_reason(
    block_pool: Any,
    capture_attempt: int,
) -> str | None:
    if shadow_enabled():
        return None

    min_density = _capture_min_cached_victim_density_64()
    min_attempts = _capture_min_pressure_attempts()
    free_ratio = _capture_activate_free_ratio()
    head_scan = _capture_activate_cached_head_scan_blocks()
    periodic = _capture_periodic_interval()
    respect_throttle = _capture_respect_replacement_throttle()
    require_activation = _capture_require_activation()
    pressure_warm_attempts = _capture_pressure_warm_attempts()
    gate_configured = (
        min_density > 0.0
        or min_attempts > 0
        or free_ratio > 0.0
        or head_scan > 0
        or periodic > 0
        or respect_throttle
        or require_activation
        or pressure_warm_attempts > 0
    )
    if not gate_configured:
        return None

    activated = False
    if respect_throttle:
        throttle_reason, _ = _replacement_throttle_reason(block_pool)
        if throttle_reason is not None:
            return f"replacement_throttle_{throttle_reason}"
        activated = True

    if periodic > 0 and capture_attempt % max(1, periodic) == 0:
        activated = True
    if _free_ratio_below(block_pool, free_ratio):
        activated = True
    if _cached_free_head_active(block_pool, head_scan):
        activated = True

    stats = getattr(block_pool, "_omni_kv_evict_enforce_stats", None)
    attempts = int(stats.get("attempts", 0)) if isinstance(stats, dict) else 0
    consume_pressure_warm = False
    if pressure_warm_attempts > 0:
        warm_remaining = (
            int(stats.get("capture_pressure_warm_remaining", 0))
            if isinstance(stats, dict)
            else 0
        )
        if warm_remaining <= 0:
            return "pressure_warm_inactive"
        activated = True
        consume_pressure_warm = True
    if min_attempts > 0 and attempts < min_attempts:
        return "pressure_attempts_below_min"
    if min_density > 0.0:
        denominator = min(64, max(1, attempts))
        recent_cached = (
            int(stats.get("recent_cached_victim_attempts_64", 0))
            if isinstance(stats, dict)
            else 0
        )
        density = recent_cached / denominator
        if density < min_density:
            return "cached_victim_density_below_min"
        activated = True
    activation_configured = (
        free_ratio > 0.0
        or head_scan > 0
        or periodic > 0
        or pressure_warm_attempts > 0
    )
    if require_activation and activation_configured and not activated:
        return "activation_inactive"
    if consume_pressure_warm and isinstance(stats, dict):
        stats["capture_pressure_warm_remaining"] = max(
            0,
            int(stats.get("capture_pressure_warm_remaining", 0)) - 1,
        )
        stats["capture_pressure_warm_consumed"] = (
            int(stats.get("capture_pressure_warm_consumed", 0)) + 1
        )
    return None


def _candidate_index(block_pool: Any, policy: str) -> dict[str, Any]:
    state = getattr(block_pool, "_omni_kv_evict_candidate_index", None)
    if not isinstance(state, dict) or state.get("policy") != policy:
        state = {
            "policy": policy,
            "heap": [],
            "versions": {},
            "pushes": 0,
            "pops": 0,
            "stale_pops": 0,
            "invalid_pops": 0,
            "selected_size_cap_skips": 0,
            "selected_size_cap_pops": 0,
            "partial_not_enough_candidates": 0,
            "selected": 0,
            "fallbacks": 0,
        }
        setattr(block_pool, "_omni_kv_evict_candidate_index", state)
    return state


def _candidate_index_stats_snapshot(block_pool: Any) -> dict[str, int] | None:
    state = getattr(block_pool, "_omni_kv_evict_candidate_index", None)
    if not isinstance(state, dict):
        return None
    return {
        key: int(state.get(key, 0))
        for key in (
            "pushes",
            "pops",
            "stale_pops",
            "invalid_pops",
            "selected_size_cap_skips",
            "selected_size_cap_pops",
            "partial_not_enough_candidates",
            "selected",
            "fallbacks",
        )
    }


def _score_heap_key(score: tuple[Any, ...]) -> tuple[float, ...]:
    # heapq is a min-heap; negate numeric score components to pop the highest
    # policy score first.
    out: list[float] = []
    for value in score:
        try:
            out.append(-float(value))
        except (TypeError, ValueError):
            out.append(0.0)
    return tuple(out)


def _indexed_group_key(meta: Mapping[str, Any]) -> tuple[str, str, str] | None:
    session_id = meta.get("omni_session_id") or meta.get("omni_prefetch_session_id")
    prefix_hash = (
        meta.get("omni_committed_prefix_hash")
        or meta.get("omni_matched_prefix_hash")
        or meta.get("omni_prefetch_committed_prefix_hash")
    )
    if not session_id and not prefix_hash:
        return None
    kv_group_id = meta.get("kv_group_id")
    return (
        str(session_id or ""),
        str(prefix_hash or ""),
        str(kv_group_id if kv_group_id is not None else ""),
    )


def _select_grouped_index_candidates(
    entries: list[tuple[tuple[Any, ...], Any, Mapping[str, Any]]],
    replacement_count: int,
) -> tuple[list[Any], dict[str, Any]]:
    if replacement_count <= 0 or not entries:
        return [], {
            "group_candidates": 0,
            "group_selected": 0,
            "group_selected_blocks": 0,
        }

    require_tail_contiguous = _index_group_require_tail_contiguous()
    fill_remaining = _index_group_fill_remaining()
    max_groups = _index_group_max_groups_per_alloc()
    grouped: dict[tuple[str, str, str], list[tuple[tuple[Any, ...], Any, Mapping[str, Any]]]] = {}
    ungrouped = 0
    for entry in entries:
        key = _indexed_group_key(entry[2])
        if key is None:
            ungrouped += 1
            continue
        grouped.setdefault(key, []).append(entry)

    group_choices: list[
        tuple[
            tuple[Any, ...],
            int,
            tuple[str, str, str],
            list[tuple[tuple[Any, ...], Any, Mapping[str, Any]]],
        ]
    ] = []
    for key, group_entries in grouped.items():
        suffix_best: dict[int, tuple[tuple[Any, ...], Any, Mapping[str, Any]]] = {}
        for entry in group_entries:
            suffix = _as_int(entry[2].get("suffix_blocks_including"))
            if suffix is None or suffix <= 0:
                continue
            prev = suffix_best.get(suffix)
            if prev is None or entry[0] > prev[0]:
                suffix_best[suffix] = entry

        if require_tail_contiguous:
            selected_entries = []
            suffix = 1
            while suffix in suffix_best and len(selected_entries) < replacement_count:
                selected_entries.append(suffix_best[suffix])
                suffix += 1
        else:
            selected_entries = sorted(
                suffix_best.values(),
                key=lambda item: item[0],
                reverse=True,
            )[:replacement_count]
        if not selected_entries:
            continue

        best_score = max(entry[0] for entry in selected_entries)
        group_choices.append((best_score, len(selected_entries), key, selected_entries))

    group_choices.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected_blocks: list[Any] = []
    selected_block_ids: set[int] = set()
    selected_groups: list[tuple[str, str, str]] = []
    for _, _, key, group_entries in group_choices:
        if (
            len(selected_groups) >= max_groups
            or len(selected_blocks) >= replacement_count
        ):
            break
        remaining = replacement_count - len(selected_blocks)
        take_entries = group_entries[:remaining]
        group_added = 0
        for _, block, _ in take_entries:
            block_id = getattr(block, "block_id", None)
            if block_id is None:
                continue
            try:
                block_id_int = int(block_id)
            except (TypeError, ValueError):
                continue
            if block_id_int in selected_block_ids:
                continue
            selected_blocks.append(block)
            selected_block_ids.add(block_id_int)
            group_added += 1
        if group_added > 0:
            selected_groups.append(key)

    fill_blocks = 0
    if fill_remaining and len(selected_blocks) < replacement_count:
        for _, block, _ in sorted(entries, key=lambda item: item[0], reverse=True):
            if len(selected_blocks) >= replacement_count:
                break
            block_id = getattr(block, "block_id", None)
            if block_id is None:
                continue
            try:
                block_id_int = int(block_id)
            except (TypeError, ValueError):
                continue
            if block_id_int in selected_block_ids:
                continue
            selected_blocks.append(block)
            selected_block_ids.add(block_id_int)
            fill_blocks += 1

    return selected_blocks, {
        "group_candidates": len(grouped),
        "group_ungrouped_candidates": ungrouped,
        "group_selected": len(selected_groups),
        "group_selected_blocks": len(selected_blocks),
        "group_fill_remaining": fill_remaining,
        "group_fill_blocks": fill_blocks,
        "group_max_groups_per_alloc": max_groups,
        "group_require_tail_contiguous": require_tail_contiguous,
    }


def _next_candidate_version(state: dict[str, Any], block_id: int) -> int:
    versions = state.setdefault("versions", {})
    version = int(versions.get(block_id, 0)) + 1
    versions[block_id] = version
    return version


def _invalidate_indexed_block(block_pool: Any, block_id: int | None) -> None:
    if block_id is None:
        return
    state = getattr(block_pool, "_omni_kv_evict_candidate_index", None)
    if not isinstance(state, dict):
        return
    _next_candidate_version(state, int(block_id))


def _push_indexed_candidate(
    block_pool: Any,
    block: Any,
    policy: str | None = None,
) -> None:
    profile_started_ns = (
        time.perf_counter_ns() if _profile_timing_enabled(block_pool) else 0
    )
    if policy is None:
        policy = enforcement_policy()
    if policy is None:
        _record_index_push_profile(block_pool, profile_started_ns, "no_policy")
        return
    policy = _adaptive_policy_for_capacity(block_pool, policy)
    if not _index_metadata_active(block_pool):
        _record_index_push_profile(
            block_pool,
            profile_started_ns,
            "metadata_inactive",
        )
        return
    block_id = getattr(block, "block_id", None)
    if block_id is None:
        _record_index_push_profile(block_pool, profile_started_ns, "no_block_id")
        return
    meta = _candidate_metadata(block_pool, block)
    if not meta:
        _record_index_push_profile(block_pool, profile_started_ns, "no_metadata")
        return
    state = _candidate_index(block_pool, policy)
    if not _candidate_within_selected_size_caps(block_pool, meta):
        _next_candidate_version(state, int(block_id))
        state["selected_size_cap_skips"] = (
            int(state.get("selected_size_cap_skips", 0)) + 1
        )
        _record_index_push_profile(block_pool, profile_started_ns, "size_cap_skip")
        return
    hot_protect_sec = _env_float("OMNI_KV_EVICT_HOT_PROTECT_SEC", 10.0)
    score = _score_metadata(policy, meta, hot_protect_sec)
    version = _next_candidate_version(state, int(block_id))
    heapq.heappush(
        state["heap"],
        (_score_heap_key(score), version, int(block_id)),
    )
    state["pushes"] = int(state.get("pushes", 0)) + 1
    _record_index_push_profile(block_pool, profile_started_ns, "pushed")


def _prime_index_candidates(block_pool: Any, policy: str) -> None:
    profile_started_ns = (
        time.perf_counter_ns() if _profile_timing_enabled(block_pool) else 0
    )
    scan_blocks = _index_prime_scan_blocks()
    if scan_blocks <= 0 or not _index_metadata_active(block_pool):
        _record_index_prime_profile(
            block_pool,
            profile_started_ns,
            "inactive",
        )
        return
    state = _candidate_index(block_pool, policy)
    if bool(state.get("pressure_primed")):
        _record_index_prime_profile(
            block_pool,
            profile_started_ns,
            "already_primed",
        )
        return
    try:
        free_count = int(block_pool.get_num_free_blocks())
        blocks = _peek_free_blocks(block_pool, min(scan_blocks, max(0, free_count)))
    except Exception:
        _record_index_prime_profile(
            block_pool,
            profile_started_ns,
            "scan_error",
        )
        return
    before_pushes = int(state.get("pushes", 0))
    for block in blocks:
        if _is_cached_free_block(block) and _is_linked_free_block(block):
            _push_indexed_candidate(block_pool, block, policy)
    state["pressure_primed"] = 1
    state["prime_scanned"] = len(blocks)
    state["prime_pushes"] = max(0, int(state.get("pushes", 0)) - before_pushes)
    _record_index_prime_profile(
        block_pool,
        profile_started_ns,
        "scanned",
        scanned_blocks=len(blocks),
        pushed_blocks=int(state.get("prime_pushes", 0)),
    )


def _prefetch_soft_protect_response_penalty() -> float:
    return _env_float("OMNI_KV_EVICT_PREFETCH_SOFT_RESPONSE_PENALTY", 128.0)


def _enforcement_stats(block_pool: Any) -> dict[str, int]:
    stats = getattr(block_pool, "_omni_kv_evict_enforce_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "attempts": 0,
            "applied_allocations": 0,
            "fallback_allocations": 0,
            "selected_cached_victims": 0,
            "lru_cached_victims_seen": 0,
            "candidate_count_total": 0,
            "metadata_ratio_permille_total": 0,
            "elapsed_ns_total": 0,
            "elapsed_ns_max": 0,
        }
        setattr(block_pool, "_omni_kv_evict_enforce_stats", stats)
    return stats


def _record_enforcement_attempt(
    block_pool: Any,
    policy: str,
    num_blocks: int,
    reason: str,
    started_ns: int,
    *,
    cached_victim_count: int = 0,
    candidate_count: int = 0,
    metadata_ratio: float | None = None,
    candidate_free_ratio: float | None = None,
    selected_block_ids: list[Any] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    elapsed_ns = max(0, time.perf_counter_ns() - started_ns)
    stats = _enforcement_stats(block_pool)
    stats["attempts"] = stats.get("attempts", 0) + 1
    stats["elapsed_ns_total"] = stats.get("elapsed_ns_total", 0) + elapsed_ns
    stats["elapsed_ns_max"] = max(stats.get("elapsed_ns_max", 0), elapsed_ns)
    if cached_victim_count > 0:
        stats["lru_cached_victims_seen"] = stats.get("lru_cached_victims_seen", 0) + cached_victim_count
    if candidate_count > 0:
        stats["candidate_count_total"] = stats.get("candidate_count_total", 0) + candidate_count
    if metadata_ratio is not None:
        stats["metadata_ratio_permille_total"] = stats.get("metadata_ratio_permille_total", 0) + int(
            round(max(0.0, min(1.0, metadata_ratio)) * 1000.0)
        )

    if reason == "applied":
        stats["applied_allocations"] = stats.get("applied_allocations", 0) + 1
        stats["selected_cached_victims"] = stats.get("selected_cached_victims", 0) + cached_victim_count
    else:
        stats["fallback_allocations"] = stats.get("fallback_allocations", 0) + 1
        key = f"fallback_{reason}"
        stats[key] = stats.get(key, 0) + 1

    recent_mask = (1 << 64) - 1
    recent_denominator = min(64, max(1, int(stats.get("attempts", 1))))
    has_cached_victim = cached_victim_count > 0
    applied = reason == "applied"
    no_cached_victim = reason == "no_cached_victims"
    cached_mask = (
        (int(stats.get("recent_cached_victim_mask", 0)) << 1)
        | (1 if has_cached_victim else 0)
    ) & recent_mask
    applied_mask = (
        (int(stats.get("recent_applied_mask", 0)) << 1)
        | (1 if applied else 0)
    ) & recent_mask
    no_cached_mask = (
        (int(stats.get("recent_no_cached_victim_mask", 0)) << 1)
        | (1 if no_cached_victim else 0)
    ) & recent_mask
    stats["recent_cached_victim_mask"] = cached_mask
    stats["recent_applied_mask"] = applied_mask
    stats["recent_no_cached_victim_mask"] = no_cached_mask
    stats["recent_cached_victim_attempts_64"] = cached_mask.bit_count()
    stats["recent_applied_attempts_64"] = applied_mask.bit_count()
    stats["recent_no_cached_victim_attempts_64"] = no_cached_mask.bit_count()
    if has_cached_victim:
        stats["attempts_since_last_cached_victim"] = 0
        warm_attempts = _capture_pressure_warm_attempts()
        min_warm_cached = max(1, _capture_pressure_warm_min_cached_victims())
        if warm_attempts > 0 and cached_victim_count >= min_warm_cached:
            stats["capture_pressure_warm_remaining"] = max(
                int(stats.get("capture_pressure_warm_remaining", 0)),
                warm_attempts,
            )
            stats["capture_pressure_warm_signals"] = (
                int(stats.get("capture_pressure_warm_signals", 0)) + 1
            )
    else:
        stats["attempts_since_last_cached_victim"] = (
            int(stats.get("attempts_since_last_cached_victim", 0)) + 1
        )
    if applied:
        stats["attempts_since_last_applied"] = 0
    else:
        stats["attempts_since_last_applied"] = (
            int(stats.get("attempts_since_last_applied", 0)) + 1
        )
    if no_cached_victim:
        stats["no_cached_victim_burst"] = (
            int(stats.get("no_cached_victim_burst", 0)) + 1
        )
    else:
        stats["no_cached_victim_burst"] = 0
    cached_density = stats["recent_cached_victim_attempts_64"] / recent_denominator
    applied_density = stats["recent_applied_attempts_64"] / recent_denominator
    no_cached_density = stats["recent_no_cached_victim_attempts_64"] / recent_denominator
    if no_cached_density >= 0.75:
        pressure_phase = "mostly_no_cached"
    elif cached_density <= 0.10:
        pressure_phase = "low_cached_density"
    elif applied_density >= 0.25:
        pressure_phase = "replacement_active"
    else:
        pressure_phase = "mixed"

    if (
        not _enforcement_log_enabled()
        or not _enforcement_log_reason_allowed(reason)
    ):
        return
    sample_every = _enforcement_log_sample_every()
    attempt_index = int(stats.get("attempts", 0))
    if (
        sample_every > 1
        and attempt_index % sample_every != 0
        and not (reason == "applied" and _enforcement_log_always_applied())
    ):
        return
    event = {
        "event": "allocation_pressure_enforce",
        "ts": time.time(),
        "policy": policy,
        "applied": reason == "applied",
        "reason": reason,
        "num_requested_blocks": num_blocks,
        "cached_victim_count": cached_victim_count,
        "candidate_count": candidate_count,
        "candidate_free_ratio": candidate_free_ratio,
        "metadata_ratio": metadata_ratio,
        "selected_block_ids": selected_block_ids or [],
        "elapsed_ns": elapsed_ns,
        "attempts_since_last_cached_victim": stats["attempts_since_last_cached_victim"],
        "attempts_since_last_applied": stats["attempts_since_last_applied"],
        "no_cached_victim_burst": stats["no_cached_victim_burst"],
        "recent_cached_victim_attempts_64": stats["recent_cached_victim_attempts_64"],
        "recent_applied_attempts_64": stats["recent_applied_attempts_64"],
        "recent_no_cached_victim_attempts_64": (
            stats["recent_no_cached_victim_attempts_64"]
        ),
        "cached_victim_density_64": cached_density,
        "applied_density_64": applied_density,
        "no_cached_victim_density_64": no_cached_density,
        "pressure_phase": pressure_phase,
        "log_sample_every": sample_every,
        "log_sampled_attempt": attempt_index,
        "enforce_stats": dict(stats),
    }
    if _profile_timing_enabled(block_pool):
        event["capture_stats"] = dict(_shadow_stats(block_pool))
    current_meta = _current_request_meta(block_pool)
    if current_meta:
        event["current_request"] = current_meta
    adaptive_policy = getattr(block_pool, "_omni_kv_evict_adaptive_policy", None)
    adaptive_tier = getattr(block_pool, "_omni_kv_evict_adaptive_tier", None)
    adaptive_capacity = getattr(
        block_pool,
        "_omni_kv_evict_adaptive_capacity_blocks",
        None,
    )
    if isinstance(adaptive_policy, str) and adaptive_policy:
        event["adaptive_effective_policy"] = adaptive_policy
    if isinstance(adaptive_tier, str) and adaptive_tier:
        event["adaptive_tier"] = adaptive_tier.lower()
    if isinstance(adaptive_capacity, int):
        event["adaptive_capacity_blocks"] = adaptive_capacity
    index_stats = _candidate_index_stats_snapshot(block_pool)
    if index_stats is not None:
        event["index_stats"] = index_stats
    if extra_fields:
        event.update(extra_fields)
    _write_event(event)


def _replacement_throttle_reason(
    block_pool: Any,
) -> tuple[str | None, dict[str, Any] | None]:
    min_interval = _enforcement_min_attempts_between_applied(block_pool)
    max_recent_applied = _enforcement_max_recent_applied_64(block_pool)
    max_recent_no_cached = _enforcement_max_recent_no_cached_64(block_pool)
    if (
        min_interval <= 0
        and max_recent_applied <= 0
        and max_recent_no_cached <= 0
    ):
        return None, None

    stats = _enforcement_stats(block_pool)
    applied_allocations = int(stats.get("applied_allocations", 0))
    if applied_allocations <= 0:
        return None, None

    attempts_since_last = int(stats.get("attempts_since_last_applied", 0))
    recent_applied = int(stats.get("recent_applied_mask", 0)).bit_count()
    recent_no_cached = int(
        stats.get("recent_no_cached_victim_mask", 0),
    ).bit_count()
    attempts = int(stats.get("attempts", 0))
    recent_denominator = min(64, max(1, attempts))
    extra = {
        "throttle_min_attempts_between_applied": min_interval,
        "throttle_max_recent_applied_64": max_recent_applied,
        "throttle_max_recent_no_cached_64": max_recent_no_cached,
        "throttle_attempts_since_last_applied": attempts_since_last,
        "throttle_recent_applied_attempts_64": recent_applied,
        "throttle_recent_no_cached_attempts_64": recent_no_cached,
        "throttle_recent_denominator": recent_denominator,
    }
    if min_interval > 0 and attempts_since_last < min_interval:
        return "applied_interval_below_min", extra
    if (
        max_recent_applied > 0
        and attempts >= 64
        and recent_applied >= max_recent_applied
    ):
        return "recent_applied_over_cap", extra
    if (
        max_recent_no_cached > 0
        and attempts >= 64
        and recent_no_cached >= max_recent_no_cached
    ):
        return "recent_no_cached_over_cap", extra
    return None, None


def _replacement_pressure_gate_reason(
    block_pool: Any,
    cached_victim_count: int,
) -> tuple[str | None, dict[str, Any] | None]:
    min_density = _enforcement_min_cached_victim_density_64()
    if min_density <= 0.0:
        return None, None

    stats = _enforcement_stats(block_pool)
    previous_attempts = int(stats.get("attempts", 0))
    observed_attempts = previous_attempts + 1
    min_attempts = max(1, _enforcement_min_pressure_attempts())
    previous_recent_cached = int(stats.get("recent_cached_victim_attempts_64", 0))
    observed_recent_cached = previous_recent_cached + (
        1 if cached_victim_count > 0 else 0
    )
    denominator = min(64, max(1, observed_attempts))
    density = observed_recent_cached / denominator
    extra = {
        "pressure_gate_min_cached_victim_density_64": min_density,
        "pressure_gate_min_attempts": min_attempts,
        "pressure_gate_previous_attempts": previous_attempts,
        "pressure_gate_observed_attempts": observed_attempts,
        "pressure_gate_cached_victim_count": cached_victim_count,
        "pressure_gate_observed_cached_victim_attempts_64": (
            observed_recent_cached
        ),
        "pressure_gate_observed_density_64": density,
    }
    if observed_attempts < min_attempts:
        return "pressure_attempts_below_min", extra
    if density < min_density:
        return "cached_victim_density_below_min", extra
    return None, None


def _enforcement_free_ratio_gate_active(block_pool: Any) -> bool:
    max_free_ratio = _enforcement_activate_free_ratio(block_pool)
    if max_free_ratio <= 0.0:
        return True
    capacity = _block_pool_capacity_blocks(block_pool)
    if capacity <= 0:
        return True
    try:
        free_count = int(block_pool.get_num_free_blocks())
    except Exception:
        return True
    free_ratio = max(0.0, min(1.0, free_count / max(1, capacity)))
    return free_ratio <= max_free_ratio


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_CURRENT_REQUEST_META_KEYS = (
    "request_id",
    "omni_session_id",
    "omni_round_index",
    "omni_trace_timestamp",
    "omni_query_length",
    "omni_response_length",
    "omni_history_tokens_before",
    "omni_history_tokens_after",
    "omni_prefix_token_estimate",
    "omni_prev_gap_sec",
    "omni_expected_next_gap_sec",
    "omni_expected_next_use_ts",
    "omni_audio_total_sec",
    "omni_audio_remaining_at_finish_sec",
    "omni_audio_reply_delay_sec",
    "omni_next_audio_total_sec",
    "omni_next_audio_reply_delay_sec",
    "omni_is_terminal_turn",
    "omni_scheduler_prefetch",
    "omni_prefetch_request_id",
    "omni_prefetch_session_id",
    "omni_prefetch_committed_prefix_hash",
    "omni_prefetch_prefix_token_estimate",
    "omni_prefetch_prefetch_scope",
)


def _request_current_meta(request: Any) -> dict[str, Any]:
    params = getattr(request, "kv_transfer_params", None)
    if not isinstance(params, dict):
        params = {}
    meta: dict[str, Any] = {"request_id": getattr(request, "request_id", None)}
    for key in _CURRENT_REQUEST_META_KEYS:
        if key == "request_id":
            continue
        value = params.get(key)
        if value is not None:
            meta[key] = value
    if "omni_session_id" not in meta:
        session_id = params.get("omni_prefetch_session_id")
        if session_id is not None:
            meta["omni_session_id"] = session_id
    if "omni_prefix_token_estimate" not in meta:
        prefix_tokens = params.get("omni_prefetch_prefix_token_estimate")
        if prefix_tokens is not None:
            meta["omni_prefix_token_estimate"] = prefix_tokens
    return meta


def _set_current_request_meta(block_pool: Any, request: Any) -> None:
    try:
        setattr(
            block_pool,
            "_omni_kv_evict_current_request_meta",
            _request_current_meta(request),
        )
    except Exception:
        pass


def _clear_current_request_meta(block_pool: Any) -> None:
    try:
        setattr(block_pool, "_omni_kv_evict_current_request_meta", None)
    except Exception:
        pass


def _current_request_meta(block_pool: Any) -> dict[str, Any]:
    meta = getattr(block_pool, "_omni_kv_evict_current_request_meta", None)
    return meta if isinstance(meta, dict) else {}


def _current_request_gate_reason(
    block_pool: Any,
) -> tuple[str | None, dict[str, Any] | None]:
    min_response = _enforcement_current_min_response_len(block_pool)
    max_response = _enforcement_current_max_response_len(block_pool)
    min_history = _enforcement_current_min_history_tokens(block_pool)
    max_history = _enforcement_current_max_history_tokens(block_pool)
    min_prefix = _enforcement_current_min_prefix_tokens(block_pool)
    max_prefix = _enforcement_current_max_prefix_tokens(block_pool)
    min_round = _enforcement_current_min_round_index(block_pool)
    max_round = _enforcement_current_max_round_index(block_pool)
    min_trace_ts = _enforcement_current_min_trace_timestamp(block_pool)
    max_trace_ts = _enforcement_current_max_trace_timestamp(block_pool)
    min_expected_gap = _enforcement_current_min_expected_gap_sec(block_pool)
    min_audio_total = _enforcement_current_min_audio_total_sec(block_pool)
    min_audio_reply_delay = _enforcement_current_min_audio_reply_delay_sec(block_pool)
    short_response_max = _enforcement_current_short_response_max_len(block_pool)
    short_gap_max = _enforcement_current_short_expected_gap_max_sec(block_pool)
    require_scheduler_prefetch = _enforcement_current_require_scheduler_prefetch(
        block_pool
    )
    if (
        min_response <= 0
        and max_response <= 0
        and min_history <= 0
        and max_history <= 0
        and min_prefix <= 0
        and max_prefix <= 0
        and min_round <= 0
        and max_round <= 0
        and min_trace_ts <= 0.0
        and max_trace_ts <= 0.0
        and min_expected_gap <= 0.0
        and min_audio_total <= 0.0
        and min_audio_reply_delay <= 0.0
        and (short_response_max <= 0 or short_gap_max <= 0.0)
        and not require_scheduler_prefetch
    ):
        return None, None

    meta = _current_request_meta(block_pool)
    extra = {
        "current_gate_min_response_len": min_response,
        "current_gate_max_response_len": max_response,
        "current_gate_min_history_tokens": min_history,
        "current_gate_max_history_tokens": max_history,
        "current_gate_min_prefix_tokens": min_prefix,
        "current_gate_max_prefix_tokens": max_prefix,
        "current_gate_min_round_index": min_round,
        "current_gate_max_round_index": max_round,
        "current_gate_min_trace_timestamp": min_trace_ts,
        "current_gate_max_trace_timestamp": max_trace_ts,
        "current_gate_min_expected_gap_sec": min_expected_gap,
        "current_gate_min_audio_total_sec": min_audio_total,
        "current_gate_min_audio_reply_delay_sec": min_audio_reply_delay,
        "current_gate_short_response_max_len": short_response_max,
        "current_gate_short_expected_gap_max_sec": short_gap_max,
        "current_gate_require_scheduler_prefetch": require_scheduler_prefetch,
        "current_request": meta,
    }
    if not meta:
        return "current_request_metadata_missing", extra
    scheduler_prefetch = bool(meta.get("omni_scheduler_prefetch"))
    prefetch_request = bool(
        scheduler_prefetch
        or meta.get("omni_prefetch_request_id")
        or meta.get("omni_prefetch_prefetch_scope")
    )
    extra["current_request_scheduler_prefetch"] = scheduler_prefetch
    extra["current_request_prefetch"] = prefetch_request
    if require_scheduler_prefetch and not prefetch_request:
        return "current_request_not_scheduler_prefetch", extra

    response_len = _as_int(meta.get("omni_response_length"))
    history_tokens = _as_int(meta.get("omni_history_tokens_after"))
    prefix_tokens = _as_int(meta.get("omni_prefix_token_estimate"))
    round_index = _as_int(meta.get("omni_round_index"))
    trace_timestamp = _as_float(meta.get("omni_trace_timestamp"))
    expected_gap = _as_float(meta.get("omni_expected_next_gap_sec"))
    audio_total = _as_float(meta.get("omni_audio_total_sec"))
    audio_total_source = "direct" if audio_total is not None else None
    if audio_total is None and response_len is not None:
        audio_total = _estimated_audio_total_sec(meta, response_len)
        if audio_total is not None:
            audio_total_source = "response_length_proxy"
    audio_reply_delay = _as_float(meta.get("omni_audio_reply_delay_sec"))
    audio_reply_delay_source = "direct" if audio_reply_delay is not None else None
    if audio_reply_delay is None and min_audio_reply_delay > 0.0:
        audio_reply_delay = _audio_reply_delay_sec(meta)
        audio_reply_delay_source = "estimated"
    extra["current_request_audio_total_sec"] = audio_total
    extra["current_request_audio_total_source"] = audio_total_source
    extra["current_request_audio_reply_delay_sec"] = audio_reply_delay
    extra["current_request_audio_reply_delay_source"] = audio_reply_delay_source
    terminal = bool(meta.get("omni_is_terminal_turn"))

    def below(value: int | None, threshold: int) -> bool:
        return threshold > 0 and (value is None or value < threshold)

    def above(value: int | None, threshold: int) -> bool:
        return threshold > 0 and (value is None or value > threshold)

    def below_float(value: float | None, threshold: float) -> bool:
        return threshold > 0.0 and (value is None or value < threshold)

    if (
        short_response_max > 0
        and short_gap_max > 0.0
        and not terminal
        and response_len is not None
        and expected_gap is not None
        and response_len < short_response_max
        and expected_gap < short_gap_max
    ):
        return "current_short_response_near_gap_guard", extra
    if below(response_len, min_response):
        return "current_response_below_min", extra
    if above(response_len, max_response):
        return "current_response_above_max", extra
    if below(history_tokens, min_history):
        return "current_history_below_min", extra
    if above(history_tokens, max_history):
        return "current_history_above_max", extra
    if below(prefix_tokens, min_prefix):
        return "current_prefix_below_min", extra
    if above(prefix_tokens, max_prefix):
        return "current_prefix_above_max", extra
    if below(round_index, min_round):
        return "current_round_below_min", extra
    if above(round_index, max_round):
        return "current_round_above_max", extra
    if below_float(trace_timestamp, min_trace_ts):
        return "current_trace_timestamp_below_min", extra
    if max_trace_ts > 0.0 and (
        trace_timestamp is None or trace_timestamp > max_trace_ts
    ):
        return "current_trace_timestamp_above_max", extra
    if min_expected_gap > 0.0 and not terminal:
        if expected_gap is None or expected_gap < min_expected_gap:
            return "current_expected_gap_below_min", extra
    if below_float(audio_total, min_audio_total):
        return "current_audio_total_below_min", extra
    if below_float(audio_reply_delay, min_audio_reply_delay):
        return "current_audio_reply_delay_below_min", extra
    return None, None


def _block_hash_repr(block: Any) -> str | None:
    block_hash = getattr(block, "block_hash", None)
    if block_hash is None:
        return None
    try:
        return repr(block_hash)
    except Exception:
        return str(type(block_hash))


def _looks_like_block(value: Any) -> bool:
    return hasattr(value, "block_id") or hasattr(value, "block_hash")


def _normalize_block_groups(block_groups: Any) -> list[list[Any]]:
    if block_groups is None:
        return []
    if _looks_like_block(block_groups):
        return [[block_groups]]
    try:
        groups_raw = list(block_groups)
    except TypeError:
        return []
    if not groups_raw:
        return []
    if _looks_like_block(groups_raw[0]):
        return [groups_raw]
    groups: list[list[Any]] = []
    for group in groups_raw:
        if group is None:
            continue
        if _looks_like_block(group):
            groups.append([group])
            continue
        try:
            groups.append(list(group))
        except TypeError:
            continue
    return groups


def _coerce_block_sizes(
    block_size_tokens: int | Sequence[int] | None,
    num_groups: int,
) -> list[int | None]:
    if block_size_tokens is None:
        return [None] * num_groups
    if isinstance(block_size_tokens, int):
        return [max(1, block_size_tokens)] * num_groups
    try:
        sizes = list(block_size_tokens)
    except TypeError:
        return [None] * num_groups
    out: list[int | None] = []
    for idx in range(num_groups):
        raw = sizes[idx] if idx < len(sizes) else None
        try:
            size = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            size = None
        out.append(max(1, size) if size is not None else None)
    return out


def _response_bucket(response_length: int | None) -> str:
    if response_length is None:
        return "33-64"
    if response_length <= 16:
        return "<=16"
    if response_length <= 32:
        return "17-32"
    if response_length <= 64:
        return "33-64"
    if response_length <= 128:
        return "65-128"
    if response_length <= 256:
        return "129-256"
    return ">256"


def _query_bucket(query_length: int | None) -> str:
    if query_length is None:
        return "17-32"
    if query_length <= 16:
        return "<=16"
    if query_length <= 32:
        return "17-32"
    if query_length <= 64:
        return "33-64"
    if query_length <= 128:
        return "65-128"
    if query_length <= 256:
        return "129-256"
    return ">256"


def _predicted_gap_sec(meta: dict[str, Any]) -> float:
    if meta.get("omni_is_terminal_turn"):
        return float("inf")
    response_length = _as_int(meta.get("omni_response_length"))
    return _RESPONSE_GAP_SEC[_response_bucket(response_length)]


def _expected_or_predicted_gap_sec(meta: dict[str, Any]) -> float:
    if meta.get("omni_is_terminal_turn"):
        return float("inf")
    expected_gap = _as_float(meta.get("omni_expected_next_gap_sec"))
    if expected_gap is not None and expected_gap >= 0.0:
        return expected_gap
    return _predicted_gap_sec(meta)


def _gap_bucket_rank(gap_sec: float) -> float:
    if math.isinf(gap_sec):
        return gap_sec
    global _GAP_BUCKET_SEC_CACHE
    if _GAP_BUCKET_SEC_CACHE is None:
        _GAP_BUCKET_SEC_CACHE = _env_float(
            "OMNI_KV_EVICT_GAP_BUCKET_SEC",
            _DEFAULT_GAP_BUCKET_SEC,
        )
    bucket_sec = _GAP_BUCKET_SEC_CACHE
    if bucket_sec <= 0.0:
        return max(0.0, gap_sec)
    return math.floor(max(0.0, gap_sec) / bucket_sec)


def _activity_gap_sec(meta: dict[str, Any]) -> float:
    if meta.get("omni_is_terminal_turn"):
        return float("inf")
    response_length = _as_int(meta.get("omni_response_length"))
    query_length = _as_int(meta.get("omni_query_length"))
    response_gap = _RESPONSE_GAP_SEC[_response_bucket(response_length)]
    query_gap = _QUERY_GAP_SEC[_query_bucket(query_length)]
    prev_gap = _as_float(meta.get("omni_prev_gap_sec"))
    if prev_gap is None:
        return 0.75 * response_gap + 0.25 * query_gap
    return 0.60 * response_gap + 0.25 * query_gap + 0.15 * min(300.0, prev_gap)


def _prefetch_protected(meta: dict[str, Any]) -> bool:
    if meta.get("omni_scheduler_prefetch") or meta.get("omni_prefetch_gpu_pin"):
        return True
    if meta.get("omni_prefetch_hit_used"):
        return True
    lookup_state = str(meta.get("omni_prefetch_lookup_state") or "").lower()
    return lookup_state in {
        "prefetch_hit",
        "ready",
        "pending",
        "loading",
        "warm",
        "warm_fallback",
        "hit",
    }


def _narrow_prefetch_protected(meta: dict[str, Any]) -> bool:
    if meta.get("omni_prefetch_hit_used"):
        return True
    lookup_state = str(meta.get("omni_prefetch_lookup_state") or "").lower()
    return lookup_state in {"prefetch_hit", "ready", "hit"}


def _prefetch_protection_rank(meta: dict[str, Any]) -> int:
    return 0 if _prefetch_protected(meta) else 1


def _narrow_prefetch_protection_rank(meta: dict[str, Any]) -> int:
    return 0 if _narrow_prefetch_protected(meta) else 1


def _prefetch_soft_response_score(response_length: int, meta: dict[str, Any]) -> float:
    score = float(response_length)
    if _prefetch_protected(meta):
        score -= _prefetch_soft_protect_response_penalty()
    return max(0.0, score)


def _narrow_prefetch_soft_response_score(
    response_length: int,
    meta: dict[str, Any],
) -> float:
    score = float(response_length)
    if _narrow_prefetch_protected(meta):
        score -= _prefetch_soft_protect_response_penalty()
    return max(0.0, score)


def _score_lists_show_improvement(
    selected_scores: list[tuple[Any, ...]],
    actual_scores: list[tuple[Any, ...]],
) -> bool:
    """Return true only when policy victims are strictly worse than LRU victims."""
    if not selected_scores or len(selected_scores) != len(actual_scores):
        return False
    selected_sorted = sorted(selected_scores, reverse=True)
    actual_sorted = sorted(actual_scores, reverse=True)
    for selected, actual in zip(selected_sorted, actual_sorted):
        if selected > actual:
            return True
        if selected < actual:
            return False
    return False


def _numeric_percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(
        len(values) - 1,
        max(0, int(math.ceil(q * len(values))) - 1),
    )
    return values[idx]


def _summarize_numeric(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "p50": None, "p90": None, "max": None}
    return {
        "n": len(values),
        "p50": _numeric_percentile(values, 0.50),
        "p90": _numeric_percentile(values, 0.90),
        "max": max(values),
    }


def _json_safe_score_atom(value: Any) -> Any:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
    return value


def _score_tuple_details(
    scores: list[tuple[Any, ...]],
    limit: int = 8,
) -> dict[str, Any]:
    values = [
        [_json_safe_score_atom(item) for item in score]
        for score in scores[:limit]
    ]
    return {
        "count": len(scores),
        "values": values,
        "truncated": len(scores) > limit,
    }


def _victim_block_summary(
    block_pool: Any,
    blocks: list[Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": len(blocks),
        "metadata_present": 0,
        "terminal": 0,
    }
    response_lengths: list[float] = []
    history_tokens: list[float] = []
    next_audio_totals: list[float] = []
    next_audio_reply_delays: list[float] = []
    finite_gaps: list[float] = []
    gap_buckets: list[float] = []
    depth_ratios: list[float] = []
    lost_tokens_values: list[float] = []
    suffix_blocks_values: list[float] = []

    for block in blocks:
        meta = _candidate_metadata(block_pool, block)
        if not meta:
            continue
        summary["metadata_present"] += 1
        if bool(meta.get("omni_is_terminal_turn")):
            summary["terminal"] += 1
        response_length = _as_float(meta.get("omni_response_length"))
        if response_length is not None:
            response_lengths.append(response_length)
        history_after = _as_float(meta.get("omni_history_tokens_after"))
        if history_after is not None:
            history_tokens.append(history_after)
        next_audio_total = _as_float(meta.get("omni_next_audio_total_sec"))
        if next_audio_total is not None:
            next_audio_totals.append(next_audio_total)
        next_audio_reply_delay = _as_float(
            meta.get("omni_next_audio_reply_delay_sec")
        )
        if next_audio_reply_delay is not None:
            next_audio_reply_delays.append(next_audio_reply_delay)
        gap = _expected_or_predicted_gap_sec(meta)
        if not math.isinf(gap):
            finite_gaps.append(gap)
            gap_buckets.append(_gap_bucket_rank(gap))
        depth = _as_float(meta.get("block_depth_ratio"))
        if depth is not None:
            depth_ratios.append(depth)
        lost_tokens = _as_float(meta.get("estimated_lost_tokens_if_prefix_break"))
        if lost_tokens is not None:
            lost_tokens_values.append(lost_tokens)
        suffix_blocks = _as_float(meta.get("suffix_blocks_including"))
        if suffix_blocks is not None:
            suffix_blocks_values.append(suffix_blocks)

    summary.update(
        {
            "response_length": _summarize_numeric(response_lengths),
            "history_tokens_after": _summarize_numeric(history_tokens),
            "next_audio_total_sec": _summarize_numeric(next_audio_totals),
            "next_audio_reply_delay_sec": _summarize_numeric(
                next_audio_reply_delays
            ),
            "expected_gap_sec_finite": _summarize_numeric(finite_gaps),
            "gap_bucket_finite": _summarize_numeric(gap_buckets),
            "block_depth_ratio": _summarize_numeric(depth_ratios),
            "lost_tokens_if_prefix_break": _summarize_numeric(lost_tokens_values),
            "suffix_blocks_including": _summarize_numeric(suffix_blocks_values),
        }
    )
    return summary


def _victim_block_records(
    block_pool: Any,
    blocks: list[Any],
) -> dict[str, Any] | None:
    limit = _env_int("OMNI_KV_EVICT_LOG_VICTIM_DETAIL_LIMIT", 8)
    if limit <= 0:
        return None
    now_wall = time.time()
    return {
        "count": len(blocks),
        "limit": limit,
        "truncated": len(blocks) > limit,
        "records": [
            _candidate_record(block, now_wall, block_pool)
            for block in blocks[:limit]
        ],
    }


def _victim_summary_extra(
    block_pool: Any,
    actual_lru_blocks: list[Any],
    selected_blocks: list[Any] | None = None,
) -> dict[str, Any] | None:
    if not _enforcement_log_enabled():
        return None
    extra = {
        "actual_lru_victim_summary": _victim_block_summary(
            block_pool,
            actual_lru_blocks,
        ),
    }
    actual_records = _victim_block_records(block_pool, actual_lru_blocks)
    if actual_records is not None:
        extra["actual_lru_victim_records"] = actual_records
    if selected_blocks is not None:
        extra["selected_victim_summary"] = _victim_block_summary(
            block_pool,
            selected_blocks,
        )
        selected_records = _victim_block_records(block_pool, selected_blocks)
        if selected_records is not None:
            extra["selected_victim_records"] = selected_records
    return extra


def _json_safe_score_tuple(score: tuple[Any, ...]) -> list[Any]:
    return [_json_safe_score_atom(item) for item in score]


def _candidate_detail_enabled() -> bool:
    return (
        _enforcement_log_enabled()
        and _enforcement_log_candidate_detail_limit() > 0
    )


def _candidate_detail_record_from_block(
    block_pool: Any,
    policy: str,
    block: Any,
    hot_protect_sec: float,
    now_wall: float,
    source: str,
    score: tuple[Any, ...] | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    block_id = getattr(block, "block_id", None)
    if block_id is None:
        return None
    meta = _candidate_metadata(block_pool, block)
    if not meta:
        return None
    if score is None:
        score = _score_metadata(policy, meta, hot_protect_sec)
    record = _candidate_record(block, now_wall, block_pool)
    record["candidate_detail_source"] = source
    record["score_tuple"] = _json_safe_score_tuple(score)
    return score, record


def _candidate_detail_from_scored_blocks(
    block_pool: Any,
    policy: str,
    scored_blocks: Iterable[tuple[tuple[Any, ...], Any]],
    hot_protect_sec: float,
    source: str,
    total_candidates: int | None = None,
) -> dict[str, Any]:
    if not _candidate_detail_enabled():
        return {}
    limit = _enforcement_log_candidate_detail_limit()
    now_wall = time.time()
    seen: set[int] = set()
    entries: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for score, block in scored_blocks:
        block_id = getattr(block, "block_id", None)
        if block_id is None or int(block_id) in seen:
            continue
        seen.add(int(block_id))
        detail = _candidate_detail_record_from_block(
            block_pool,
            policy,
            block,
            hot_protect_sec,
            now_wall,
            source,
            score,
        )
        if detail is not None:
            entries.append(detail)
    entries.sort(key=lambda item: item[0], reverse=True)
    records = [record for _, record in entries[:limit]]
    count = len(entries) if total_candidates is None else total_candidates
    return {
        "candidate_detail": {
            "source": source,
            "count": count,
            "limit": limit,
            "logged": len(records),
            "truncated": count > len(records),
            "records": records,
        },
    }


def _candidate_detail_from_records(
    policy: str,
    candidates: list[dict[str, Any]],
    hot_protect_sec: float,
    source: str,
) -> dict[str, Any]:
    if not _candidate_detail_enabled():
        return {}
    limit = _enforcement_log_candidate_detail_limit()
    entries: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    seen: set[int] = set()
    for candidate in candidates:
        block_id = _as_int(candidate.get("block_id"))
        if block_id is None or block_id in seen:
            continue
        seen.add(block_id)
        score = _score(policy, candidate, hot_protect_sec)
        record = dict(candidate)
        record["candidate_detail_source"] = source
        record["score_tuple"] = _json_safe_score_tuple(score)
        entries.append((score, record))
    entries.sort(key=lambda item: item[0], reverse=True)
    records = [record for _, record in entries[:limit]]
    return {
        "candidate_detail": {
            "source": source,
            "count": len(entries),
            "limit": limit,
            "logged": len(records),
            "truncated": len(entries) > len(records),
            "records": records,
        },
    }


def _indexed_candidate_detail_extra(
    block_pool: Any,
    policy: str,
    state: dict[str, Any],
    selected_blocks: Iterable[Any],
    actual_cached_blocks: Iterable[Any],
    hot_protect_sec: float,
    source: str,
) -> dict[str, Any]:
    if not _candidate_detail_enabled():
        return {}

    limit = _enforcement_log_candidate_detail_limit()
    heap = state.get("heap") or []
    heap_copy = list(heap)
    heapq.heapify(heap_copy)
    versions = state.setdefault("versions", {})
    now_wall = time.time()
    actual_ids = {
        int(block_id)
        for block_id in (
            getattr(block, "block_id", None) for block in actual_cached_blocks
        )
        if block_id is not None
    }
    seen: set[int] = set()
    entries: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def add_block(block: Any, detail_source: str) -> bool:
        block_id = getattr(block, "block_id", None)
        if block_id is None:
            return False
        block_id = int(block_id)
        if block_id in seen or block_id in actual_ids:
            return False
        detail = _candidate_detail_record_from_block(
            block_pool,
            policy,
            block,
            hot_protect_sec,
            now_wall,
            detail_source,
        )
        if detail is None:
            return False
        seen.add(block_id)
        entries.append(detail)
        return True

    seeded_selected = 0
    for block in selected_blocks:
        if add_block(block, "selected"):
            seeded_selected += 1

    max_pops = _enforcement_log_candidate_heap_pop_limit()
    if max_pops <= 0:
        max_pops = len(heap_copy)
    heap_pops = 0
    stale = 0
    invalid = 0
    size_cap_skips = 0
    actual_excluded = 0
    while heap_copy and len(entries) < limit and heap_pops < max_pops:
        _, version, block_id = heapq.heappop(heap_copy)
        heap_pops += 1
        block_id = int(block_id)
        if int(versions.get(block_id, 0)) != int(version):
            stale += 1
            continue
        if block_id in actual_ids:
            actual_excluded += 1
            continue
        if block_id in seen or block_id < 0 or block_id >= len(block_pool.blocks):
            invalid += 1
            continue
        block = block_pool.blocks[block_id]
        if (
            not _is_cached_free_block(block)
            or not _is_linked_free_block(block)
        ):
            invalid += 1
            continue
        meta = _candidate_metadata(block_pool, block)
        if not meta:
            invalid += 1
            continue
        if not _candidate_within_selected_size_caps(block_pool, meta):
            size_cap_skips += 1
            continue
        add_block(block, "heap")

    entries.sort(key=lambda item: item[0], reverse=True)
    records = [record for _, record in entries[:limit]]
    return {
        "candidate_detail": {
            "source": source,
            "count": len(entries),
            "limit": limit,
            "logged": len(records),
            "truncated": bool(heap_copy) or len(entries) > len(records),
            "heap_size": len(heap),
            "heap_pops": heap_pops,
            "heap_pop_limit": max_pops,
            "seeded_selected": seeded_selected,
            "skipped_stale": stale,
            "skipped_invalid": invalid,
            "skipped_selected_size_cap": size_cap_skips,
            "skipped_actual_lru_head": actual_excluded,
            "records": records,
        },
    }


def _actual_victim_protect_eligible(
    block_pool: Any,
    block: Any,
    max_gap_sec: float,
) -> bool:
    if max_gap_sec <= 0.0:
        return True
    meta = _candidate_metadata(block_pool, block)
    if not meta:
        return False
    gap = _expected_or_predicted_gap_sec(meta)
    return not math.isinf(gap) and gap <= max_gap_sec


def _gap_margin_replacement_eligible(
    block_pool: Any,
    selected_block: Any,
    actual_block: Any,
    min_gap_margin_sec: float,
    terminal_ok: bool,
) -> bool:
    if min_gap_margin_sec <= 0.0:
        return True
    selected_meta = _candidate_metadata(block_pool, selected_block)
    actual_meta = _candidate_metadata(block_pool, actual_block)
    if not selected_meta or not actual_meta:
        return False
    actual_gap = _expected_or_predicted_gap_sec(actual_meta)
    if math.isinf(actual_gap):
        return False
    selected_gap = _expected_or_predicted_gap_sec(selected_meta)
    if math.isinf(selected_gap):
        return bool(terminal_ok)
    return selected_gap - actual_gap >= min_gap_margin_sec


def _current_cold_or_early_gap_margin_override(
    block_pool: Any,
) -> tuple[float | None, dict[str, Any]]:
    override_margin = _enforcement_current_cold_or_early_min_gap_margin_sec(
        block_pool
    )
    cold_prefix_max = _enforcement_current_cold_prefix_max_tokens(block_pool)
    early_trace_max = _enforcement_current_early_trace_timestamp_max(block_pool)
    extra: dict[str, Any] = {
        "current_cold_or_early_min_gap_margin_sec": override_margin,
        "current_cold_prefix_max_tokens": cold_prefix_max,
        "current_early_trace_timestamp_max": early_trace_max,
        "current_gap_margin_override_active": False,
    }
    if override_margin <= 0.0 or (
        cold_prefix_max < 0 and early_trace_max <= 0.0
    ):
        return None, extra

    meta = _current_request_meta(block_pool)
    prefix_tokens = _as_int(meta.get("omni_prefix_token_estimate"))
    trace_timestamp = _as_float(meta.get("omni_trace_timestamp"))
    cold_match = (
        cold_prefix_max >= 0
        and prefix_tokens is not None
        and prefix_tokens <= cold_prefix_max
    )
    early_match = (
        early_trace_max > 0.0
        and trace_timestamp is not None
        and trace_timestamp <= early_trace_max
    )
    extra.update(
        {
            "current_gap_margin_prefix_tokens": prefix_tokens,
            "current_gap_margin_trace_timestamp": trace_timestamp,
            "current_gap_margin_cold_match": cold_match,
            "current_gap_margin_early_match": early_match,
        }
    )
    if cold_match or early_match:
        extra["current_gap_margin_override_active"] = True
        return override_margin, extra
    return None, extra


def _effective_min_gap_margin_sec(
    block_pool: Any,
) -> tuple[float, dict[str, Any]]:
    base_margin = _enforcement_min_gap_margin_sec(block_pool)
    override_margin, extra = _current_cold_or_early_gap_margin_override(
        block_pool
    )
    if override_margin is not None:
        extra["base_min_gap_margin_sec"] = base_margin
        return override_margin, extra
    return base_margin, extra


def _selected_blocks_all_terminal(block_pool: Any, blocks: Sequence[Any]) -> bool:
    if not blocks:
        return False
    for block in blocks:
        meta = _candidate_metadata(block_pool, block)
        if not meta or not bool(meta.get("omni_is_terminal_turn")):
            return False
    return True


def _selected_near_reuse_guard_match(
    meta: dict[str, Any],
    gap_sec: float,
    history_tokens_threshold: int,
    terminal_ok: bool,
) -> tuple[bool, dict[str, Any]]:
    gap: float | None = None
    history_tokens = _as_int(meta.get("omni_history_tokens_after"))
    terminal = bool(meta.get("omni_is_terminal_turn"))
    details: dict[str, Any] = {
        "selected_protect_near_gap_sec": gap_sec,
        "selected_protect_near_history_tokens": history_tokens_threshold,
        "selected_protect_near_terminal_ok": terminal_ok,
        "selected_gap_sec": None,
        "selected_history_tokens": history_tokens,
        "selected_terminal": terminal,
        "selected_gap_is_near": False,
        "selected_history_is_short": False,
        "selected_terminal_bypass": False,
    }
    if terminal and terminal_ok:
        details["selected_terminal_bypass"] = True
        return False, details
    if gap_sec > 0.0:
        gap = _expected_or_predicted_gap_sec(meta)
        details["selected_gap_sec"] = _json_safe_score_atom(gap)
        details["selected_gap_is_near"] = (
            not math.isinf(gap) and gap < gap_sec
        )
    if history_tokens_threshold > 0:
        details["selected_history_is_short"] = (
            history_tokens is not None
            and history_tokens < history_tokens_threshold
        )
    return (
        bool(details["selected_gap_is_near"])
        and bool(details["selected_history_is_short"])
    ), details


def _selected_size_cap_reason(
    block_pool: Any,
    blocks: Sequence[Any],
) -> tuple[str | None, dict[str, Any]]:
    max_response = _enforcement_selected_max_response_len(block_pool)
    max_history = _enforcement_selected_max_history_tokens(block_pool)
    max_lost = _enforcement_selected_max_lost_tokens(block_pool)
    near_gap = _enforcement_selected_protect_near_gap_sec(block_pool)
    near_history = _enforcement_selected_protect_near_history_tokens(block_pool)
    near_terminal_ok = _enforcement_selected_protect_near_terminal_ok(block_pool)
    min_next_audio_total = _enforcement_selected_min_next_audio_total_sec(
        block_pool
    )
    min_next_audio_reply_delay = (
        _enforcement_selected_min_next_audio_reply_delay_sec(block_pool)
    )
    extra: dict[str, Any] = {
        "selected_max_response_len": max_response,
        "selected_max_history_tokens": max_history,
        "selected_max_lost_tokens": max_lost,
        "selected_protect_near_gap_sec": near_gap,
        "selected_protect_near_history_tokens": near_history,
        "selected_protect_near_terminal_ok": near_terminal_ok,
        "selected_min_next_audio_total_sec": min_next_audio_total,
        "selected_min_next_audio_reply_delay_sec": min_next_audio_reply_delay,
    }
    if (
        max_response <= 0
        and max_history <= 0
        and max_lost <= 0
        and (near_gap <= 0.0 or near_history <= 0)
        and min_next_audio_total <= 0.0
        and min_next_audio_reply_delay <= 0.0
    ):
        return None, extra

    violations: list[dict[str, Any]] = []
    near_reuse_violations = 0
    for index, block in enumerate(blocks):
        meta = _candidate_metadata(block_pool, block)
        block_details: dict[str, Any] = {
            "index": index,
            "block_id": getattr(block, "block_id", None),
            "metadata_present": bool(meta),
        }
        if not meta:
            violations.append(block_details)
            continue
        response_len = _as_int(meta.get("omni_response_length"))
        history_tokens = _as_int(meta.get("omni_history_tokens_after"))
        lost_tokens = _as_int(meta.get("estimated_lost_tokens_if_prefix_break"))
        if lost_tokens is None:
            lost_tokens = history_tokens
        terminal = bool(meta.get("omni_is_terminal_turn"))
        next_audio_total = _as_float(meta.get("omni_next_audio_total_sec"))
        next_audio_reply_delay = _as_float(
            meta.get("omni_next_audio_reply_delay_sec")
        )
        block_details.update(
            {
                "response_len": response_len,
                "history_tokens": history_tokens,
                "lost_tokens": lost_tokens,
                "terminal": terminal,
                "next_audio_total_sec": next_audio_total,
                "next_audio_reply_delay_sec": next_audio_reply_delay,
            }
        )
        over_response = (
            max_response > 0
            and (response_len is None or response_len > max_response)
        )
        over_history = (
            max_history > 0
            and (history_tokens is None or history_tokens > max_history)
        )
        over_lost = max_lost > 0 and (lost_tokens is None or lost_tokens > max_lost)
        under_next_audio_total = (
            min_next_audio_total > 0.0
            and not terminal
            and (
                next_audio_total is None
                or next_audio_total < min_next_audio_total
            )
        )
        under_next_audio_reply_delay = (
            min_next_audio_reply_delay > 0.0
            and not terminal
            and (
                next_audio_reply_delay is None
                or next_audio_reply_delay < min_next_audio_reply_delay
            )
        )
        selected_near_reuse = False
        if near_gap > 0.0 and near_history > 0:
            selected_near_reuse, near_details = _selected_near_reuse_guard_match(
                meta,
                near_gap,
                near_history,
                near_terminal_ok,
            )
            block_details.update(near_details)
        if (
            over_response
            or over_history
            or over_lost
            or selected_near_reuse
            or under_next_audio_total
            or under_next_audio_reply_delay
        ):
            if selected_near_reuse:
                near_reuse_violations += 1
            block_details.update(
                {
                    "over_response": over_response,
                    "over_history": over_history,
                    "over_lost": over_lost,
                    "selected_near_reuse": selected_near_reuse,
                    "under_next_audio_total": under_next_audio_total,
                    "under_next_audio_reply_delay": (
                        under_next_audio_reply_delay
                    ),
                }
            )
            violations.append(block_details)

    if violations:
        extra["selected_size_cap_violations"] = violations[:8]
        extra["selected_size_cap_violation_count"] = len(violations)
        extra["selected_near_reuse_violation_count"] = near_reuse_violations
        if near_reuse_violations > 0:
            return "selected_near_reuse_guard", extra
        return "selected_size_over_cap", extra
    return None, extra


def _candidate_within_selected_size_caps(
    block_pool: Any,
    meta: dict[str, Any],
) -> bool:
    max_response = _enforcement_selected_max_response_len(block_pool)
    max_history = _enforcement_selected_max_history_tokens(block_pool)
    max_lost = _enforcement_selected_max_lost_tokens(block_pool)
    near_gap = _enforcement_selected_protect_near_gap_sec(block_pool)
    near_history = _enforcement_selected_protect_near_history_tokens(block_pool)
    near_terminal_ok = _enforcement_selected_protect_near_terminal_ok(block_pool)
    min_next_audio_total = _enforcement_selected_min_next_audio_total_sec(
        block_pool
    )
    min_next_audio_reply_delay = (
        _enforcement_selected_min_next_audio_reply_delay_sec(block_pool)
    )
    if (
        max_response <= 0
        and max_history <= 0
        and max_lost <= 0
        and (near_gap <= 0.0 or near_history <= 0)
        and min_next_audio_total <= 0.0
        and min_next_audio_reply_delay <= 0.0
    ):
        return True

    response_len = _as_int(meta.get("omni_response_length"))
    history_tokens = _as_int(meta.get("omni_history_tokens_after"))
    lost_tokens = _as_int(meta.get("estimated_lost_tokens_if_prefix_break"))
    if lost_tokens is None:
        lost_tokens = history_tokens
    terminal = bool(meta.get("omni_is_terminal_turn"))
    next_audio_total = _as_float(meta.get("omni_next_audio_total_sec"))
    next_audio_reply_delay = _as_float(meta.get("omni_next_audio_reply_delay_sec"))
    if max_response > 0 and (response_len is None or response_len > max_response):
        return False
    if max_history > 0 and (history_tokens is None or history_tokens > max_history):
        return False
    if max_lost > 0 and (lost_tokens is None or lost_tokens > max_lost):
        return False
    if near_gap > 0.0 and near_history > 0:
        selected_near_reuse, _ = _selected_near_reuse_guard_match(
            meta,
            near_gap,
            near_history,
            near_terminal_ok,
        )
        if selected_near_reuse:
            return False
    if (
        min_next_audio_total > 0.0
        and not terminal
        and (next_audio_total is None or next_audio_total < min_next_audio_total)
    ):
        return False
    if (
        min_next_audio_reply_delay > 0.0
        and not terminal
        and (
            next_audio_reply_delay is None
            or next_audio_reply_delay < min_next_audio_reply_delay
        )
    ):
        return False
    return True


def _actual_near_short_guard_match(
    block_pool: Any,
    block: Any,
    gap_sec: float,
    response_len: int,
) -> tuple[bool, dict[str, Any]]:
    meta = _candidate_metadata(block_pool, block)
    details: dict[str, Any] = {
        "metadata_present": bool(meta),
        "actual_gap_sec": None,
        "actual_response_len": None,
        "actual_gap_near_threshold": gap_sec,
        "actual_response_short_threshold": response_len,
        "actual_gap_is_near": gap_sec <= 0.0,
        "actual_response_is_short": response_len <= 0,
    }
    if not meta:
        return False, details

    if gap_sec > 0.0:
        gap = _expected_or_predicted_gap_sec(meta)
        details["actual_gap_sec"] = _json_safe_score_atom(gap)
        details["actual_gap_is_near"] = (
            not math.isinf(gap) and gap <= gap_sec
        )
    if response_len > 0:
        response = _as_float(meta.get("omni_response_length"))
        details["actual_response_len"] = response
        details["actual_response_is_short"] = (
            response is not None and response <= response_len
        )
    return (
        bool(details["actual_gap_is_near"])
        and bool(details["actual_response_is_short"])
    ), details


def _actual_near_short_guard_reason(
    block_pool: Any,
    actual_blocks: Sequence[Any],
    selected_blocks: Sequence[Any],
) -> tuple[str | None, dict[str, Any] | None]:
    gap_sec = _enforcement_protect_actual_near_short_gap_sec(block_pool)
    response_len = _enforcement_protect_actual_near_short_response_len(
        block_pool,
    )
    if gap_sec <= 0.0 and response_len <= 0:
        return None, None

    selected_terminal_ok = (
        _enforcement_protect_actual_near_short_selected_terminal_ok(block_pool)
    )
    extra = {
        "actual_near_short_guard_gap_sec": gap_sec,
        "actual_near_short_guard_response_len": response_len,
        "actual_near_short_guard_selected_terminal_ok": selected_terminal_ok,
    }
    if selected_terminal_ok and _selected_blocks_all_terminal(
        block_pool,
        selected_blocks,
    ):
        extra["actual_near_short_guard_terminal_bypass"] = True
        return None, None

    match_details: list[dict[str, Any]] = []
    match_count = 0
    for block in actual_blocks:
        matched, details = _actual_near_short_guard_match(
            block_pool,
            block,
            gap_sec,
            response_len,
        )
        match_details.append(details)
        if matched:
            match_count += 1
    extra["actual_near_short_guard_match_count"] = match_count
    extra["actual_near_short_guard_block_details"] = match_details[:8]
    if actual_blocks and match_count == len(actual_blocks):
        return "actual_near_short_guard", extra
    return None, None


def _low_value_actual_block_guard_match(
    block_pool: Any,
    block: Any,
    gap_sec: float,
    response_len: int,
) -> tuple[bool, dict[str, Any]]:
    meta = _candidate_metadata(block_pool, block)
    details: dict[str, Any] = {
        "metadata_present": bool(meta),
        "actual_gap_sec": None,
        "actual_response_len": None,
        "actual_gap_over_threshold": False,
        "actual_response_over_threshold": False,
    }
    if not meta:
        return False, details
    matched = False
    if gap_sec > 0.0:
        gap = _expected_or_predicted_gap_sec(meta)
        if not math.isinf(gap):
            details["actual_gap_sec"] = gap
            if gap > gap_sec:
                details["actual_gap_over_threshold"] = True
                matched = True
    if response_len > 0:
        response = _as_float(meta.get("omni_response_length"))
        if response is not None:
            details["actual_response_len"] = response
            if response >= response_len:
                details["actual_response_over_threshold"] = True
                matched = True
    return matched, details


def _low_value_actual_density_guard_reason(
    block_pool: Any,
    actual_blocks: Sequence[Any],
    selected_blocks: Sequence[Any],
) -> tuple[str | None, dict[str, Any] | None]:
    recent_threshold = _enforcement_low_value_actual_guard_recent_applied_64(
        block_pool,
    )
    gap_sec = _enforcement_low_value_actual_guard_gap_sec(block_pool)
    response_len = _enforcement_low_value_actual_guard_response_len(block_pool)
    if recent_threshold <= 0 or (gap_sec <= 0.0 and response_len <= 0):
        return None, None

    stats = _enforcement_stats(block_pool)
    attempts = int(stats.get("attempts", 0))
    recent_applied = int(stats.get("recent_applied_mask", 0)).bit_count()
    recent_denominator = min(64, max(1, attempts))
    extra = {
        "low_value_actual_guard_recent_applied_64": recent_threshold,
        "low_value_actual_guard_gap_sec": gap_sec,
        "low_value_actual_guard_response_len": response_len,
        "low_value_actual_guard_selected_terminal_ok": (
            _enforcement_low_value_actual_guard_selected_terminal_ok(block_pool)
        ),
        "low_value_actual_guard_recent_applied_attempts_64": recent_applied,
        "low_value_actual_guard_recent_denominator": recent_denominator,
    }
    if attempts < 64 or recent_applied < recent_threshold:
        return None, None

    if (
        extra["low_value_actual_guard_selected_terminal_ok"]
        and _selected_blocks_all_terminal(block_pool, selected_blocks)
    ):
        extra["low_value_actual_guard_terminal_bypass"] = True
        return None, None

    match_details: list[dict[str, Any]] = []
    match_count = 0
    for block in actual_blocks:
        matched, details = _low_value_actual_block_guard_match(
            block_pool,
            block,
            gap_sec,
            response_len,
        )
        match_details.append(details)
        if matched:
            match_count += 1
    extra["low_value_actual_guard_match_count"] = match_count
    extra["low_value_actual_guard_block_details"] = match_details[:8]
    if actual_blocks and match_count == len(actual_blocks):
        return "low_value_actual_density_guard", extra
    return None, None


def _prefix_loss_delta_guard_reason(
    block_pool: Any,
    actual_blocks: Sequence[Any],
    selected_blocks: Sequence[Any],
) -> tuple[str | None, dict[str, Any] | None]:
    max_delta_tokens = _enforcement_max_lost_delta_tokens(block_pool)
    min_gap_margin_sec = _enforcement_lost_delta_min_gap_margin_sec(block_pool)
    if max_delta_tokens <= 0:
        return None, None

    selected_terminal_ok = _enforcement_lost_delta_selected_terminal_ok(
        block_pool,
    )
    details: list[dict[str, Any]] = []
    guard_count = 0
    for actual_block, selected_block in zip(actual_blocks, selected_blocks):
        actual_meta = _candidate_metadata(block_pool, actual_block)
        selected_meta = _candidate_metadata(block_pool, selected_block)
        actual_lost = _as_float(
            actual_meta.get("estimated_lost_tokens_if_prefix_break")
        ) if actual_meta else None
        selected_lost = _as_float(
            selected_meta.get("estimated_lost_tokens_if_prefix_break")
        ) if selected_meta else None
        selected_terminal = bool(
            selected_meta.get("omni_is_terminal_turn")
        ) if selected_meta else False
        actual_gap = (
            _expected_or_predicted_gap_sec(actual_meta)
            if actual_meta
            else None
        )
        selected_gap = (
            _expected_or_predicted_gap_sec(selected_meta)
            if selected_meta
            else None
        )
        lost_delta = (
            selected_lost - actual_lost
            if selected_lost is not None and actual_lost is not None
            else None
        )
        gap_margin = None
        if (
            actual_gap is not None
            and selected_gap is not None
            and not math.isinf(actual_gap)
            and not math.isinf(selected_gap)
        ):
            gap_margin = selected_gap - actual_gap

        guarded = False
        if lost_delta is not None and lost_delta >= max_delta_tokens:
            if not (selected_terminal and selected_terminal_ok):
                if min_gap_margin_sec <= 0.0:
                    guarded = True
                elif gap_margin is None or gap_margin < min_gap_margin_sec:
                    guarded = True
        if guarded:
            guard_count += 1
        details.append(
            {
                "actual_block_id": getattr(actual_block, "block_id", None),
                "selected_block_id": getattr(selected_block, "block_id", None),
                "actual_lost_tokens": actual_lost,
                "selected_lost_tokens": selected_lost,
                "lost_delta_tokens": lost_delta,
                "actual_gap_sec": _json_safe_score_atom(actual_gap),
                "selected_gap_sec": _json_safe_score_atom(selected_gap),
                "gap_margin_sec": _json_safe_score_atom(gap_margin),
                "selected_terminal": selected_terminal,
                "guarded": guarded,
            }
        )

    extra = {
        "max_lost_delta_tokens": max_delta_tokens,
        "lost_delta_min_gap_margin_sec": min_gap_margin_sec,
        "lost_delta_selected_terminal_ok": selected_terminal_ok,
        "lost_delta_guard_count": guard_count,
        "lost_delta_guard_details": details[:8],
    }
    if actual_blocks and guard_count == len(actual_blocks):
        return "prefix_loss_delta_guard", extra
    return None, extra if guard_count > 0 else None


def _gap_margin_pair_details(
    block_pool: Any,
    selected_blocks: Sequence[Any],
    actual_blocks: Sequence[Any],
    min_gap_margin_sec: float,
    terminal_ok: bool,
    limit: int = 8,
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    finite_margins: list[float] = []
    pass_count = 0
    for selected_block, actual_block in zip(selected_blocks, actual_blocks):
        selected_meta = _candidate_metadata(block_pool, selected_block)
        actual_meta = _candidate_metadata(block_pool, actual_block)
        selected_gap: float | None = None
        actual_gap: float | None = None
        selected_terminal = None
        actual_terminal = None
        passed = False
        margin_value: float | None = None
        if selected_meta:
            selected_terminal = bool(selected_meta.get("omni_is_terminal_turn"))
            selected_gap = _expected_or_predicted_gap_sec(selected_meta)
        if actual_meta:
            actual_terminal = bool(actual_meta.get("omni_is_terminal_turn"))
            actual_gap = _expected_or_predicted_gap_sec(actual_meta)
        if actual_gap is not None and not math.isinf(actual_gap):
            if selected_gap is not None and math.isinf(selected_gap):
                passed = bool(terminal_ok)
            elif selected_gap is not None:
                margin_value = selected_gap - actual_gap
                finite_margins.append(margin_value)
                passed = margin_value >= min_gap_margin_sec
        if passed:
            pass_count += 1
        pairs.append(
            {
                "selected_block_id": getattr(selected_block, "block_id", None),
                "actual_block_id": getattr(actual_block, "block_id", None),
                "selected_gap_sec": _json_safe_score_atom(selected_gap),
                "actual_gap_sec": _json_safe_score_atom(actual_gap),
                "gap_margin_sec": _json_safe_score_atom(margin_value),
                "selected_terminal": selected_terminal,
                "actual_terminal": actual_terminal,
                "passed": passed,
            }
        )
    return {
        "count": len(pairs),
        "pass_count": pass_count,
        "min_gap_margin_sec": min_gap_margin_sec,
        "terminal_ok": terminal_ok,
        "finite_gap_margin_sec": _summarize_numeric(finite_margins),
        "values": pairs[:limit],
        "truncated": len(pairs) > limit,
    }


def _history_size_score(history_tokens: int) -> float:
    return _ACTIVITY_SIZE_WEIGHT_SEC * math.log1p(
        max(0, history_tokens) / 1024.0
    )


def _response_recompute_penalty(response_length: int) -> float:
    weight = _env_float("OMNI_KV_EVICT_RESPONSE_PENALTY_WEIGHT", 4.0)
    return max(0.0, weight) * math.log1p(max(0, response_length))


def _size_eviction_bonus(
    gap: float,
    terminal: bool,
    history_tokens: int,
    response_length: int,
) -> float:
    if not terminal:
        min_gap = _env_float("OMNI_KV_EVICT_SIZE_BONUS_MIN_GAP_SEC", 45.0)
        if not math.isinf(gap) and gap < max(0.0, min_gap):
            return 0.0
    history_weight = _env_float("OMNI_KV_EVICT_SIZE_HISTORY_WEIGHT", 8.0)
    response_weight = _env_float("OMNI_KV_EVICT_SIZE_RESPONSE_WEIGHT", 2.0)
    history_term = math.log1p(max(0, history_tokens) / 64.0)
    response_term = math.log1p(max(0, response_length))
    return (
        max(0.0, history_weight) * history_term
        + max(0.0, response_weight) * response_term
    )


def _cost_loss_penalty(lost_tokens: int) -> float:
    weight = _env_float("OMNI_KV_EVICT_COST_LOSS_WEIGHT", 1.0)
    return max(0.0, weight) * math.log1p(max(0, lost_tokens))


def _near_reuse_history_penalty(
    gap: float,
    terminal: bool,
    history_tokens: int,
) -> float:
    weight = _env_float("OMNI_KV_EVICT_NEAR_REUSE_HISTORY_PENALTY_WEIGHT", 0.0)
    gap_sec = _env_float("OMNI_KV_EVICT_NEAR_REUSE_GAP_SEC", 0.0)
    history_target = _env_int("OMNI_KV_EVICT_NEAR_REUSE_HISTORY_TARGET_TOKENS", 0)
    if weight <= 0.0 or gap_sec <= 0.0 or history_target <= 0:
        return 0.0
    if terminal and _env_bool("OMNI_KV_EVICT_NEAR_REUSE_TERMINAL_OK", True):
        return 0.0
    if math.isinf(gap) or gap >= gap_sec:
        return 0.0
    history_deficit = max(0.0, history_target - max(0, history_tokens))
    if history_deficit <= 0.0:
        return 0.0
    history_ratio = history_deficit / max(1.0, float(history_target))
    gap_ratio = (gap_sec - max(0.0, gap)) / max(gap_sec, 1e-9)
    return max(0.0, weight) * history_ratio * gap_ratio


def _piecewise_risk_penalty(
    gap: float,
    terminal: bool,
    history_tokens: int,
) -> float:
    if terminal and _env_bool("OMNI_KV_EVICT_PIECEWISE_RISK_TERMINAL_OK", True):
        return 0.0
    if math.isinf(gap):
        return 0.0
    max_gap = _env_float("OMNI_KV_EVICT_PIECEWISE_RISK_MAX_GAP_SEC", 60.0)
    if max_gap > 0.0 and gap >= max_gap:
        return 0.0

    history_tokens = max(0, history_tokens)
    penalty = 0.0
    low_history = history_tokens < 128
    if history_tokens < 64:
        penalty += max(
            0.0,
            _env_float("OMNI_KV_EVICT_PIECEWISE_RISK_HIST_LT64_PENALTY", 0.0),
        )
    elif low_history:
        penalty += max(
            0.0,
            _env_float("OMNI_KV_EVICT_PIECEWISE_RISK_HIST_64_128_PENALTY", 0.0),
        )

    if low_history:
        if 15.0 <= gap < 30.0:
            penalty += max(
                0.0,
                _env_float("OMNI_KV_EVICT_PIECEWISE_RISK_GAP_15_30_PENALTY", 0.0),
            )
        elif 30.0 <= gap < 60.0:
            penalty += max(
                0.0,
                _env_float("OMNI_KV_EVICT_PIECEWISE_RISK_GAP_30_60_PENALTY", 0.0),
            )
    return penalty


def _value_cost_eviction_score(
    gap: float,
    terminal: bool,
    history_tokens: int,
    response_length: int,
    lost_tokens: int,
) -> tuple[float, float, float, float]:
    terminal_bonus = _env_float("OMNI_KV_EVICT_VALUE_TERMINAL_BONUS_SEC", 100.0)
    gap_cap = _env_float("OMNI_KV_EVICT_VALUE_GAP_CAP_SEC", 180.0)
    loss_weight = _env_float("OMNI_KV_EVICT_VALUE_LOSS_WEIGHT", 1.0)
    if terminal:
        gap_value = max(0.0, terminal_bonus)
    elif math.isinf(gap):
        gap_value = max(0.0, gap_cap) if gap_cap > 0.0 else 0.0
    else:
        gap_value = max(0.0, gap)
        if gap_cap > 0.0:
            gap_value = min(gap_value, gap_cap)
    size_bonus = _size_eviction_bonus(
        gap,
        terminal,
        history_tokens,
        response_length,
    )
    loss_penalty = max(0.0, loss_weight) * math.log1p(max(0, lost_tokens))
    return gap_value + size_bonus - loss_penalty, gap_value, size_bonus, loss_penalty


def _value_score_from_metadata(meta: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    gap = _expected_or_predicted_gap_sec(meta)
    response_length = _as_int(meta.get("omni_response_length")) or 0
    history_tokens = _as_int(meta.get("omni_history_tokens_after")) or 0
    terminal = bool(meta.get("omni_is_terminal_turn"))
    lost_tokens = _as_int(meta.get("estimated_lost_tokens_if_prefix_break"))
    if lost_tokens is None:
        lost_tokens = history_tokens
    score, gap_value, size_bonus, loss_penalty = _value_cost_eviction_score(
        gap,
        terminal,
        history_tokens,
        response_length,
        lost_tokens,
    )
    return score, {
        "score": _json_safe_score_atom(score),
        "gap_sec": _json_safe_score_atom(gap),
        "gap_value": _json_safe_score_atom(gap_value),
        "size_bonus": _json_safe_score_atom(size_bonus),
        "loss_penalty": _json_safe_score_atom(loss_penalty),
        "terminal": terminal,
        "history_tokens": history_tokens,
        "response_length": response_length,
        "lost_tokens": lost_tokens,
    }


def _value_score_margin_pair_details(
    block_pool: Any,
    selected_blocks: Sequence[Any],
    actual_blocks: Sequence[Any],
    min_margin: float,
    limit: int = 8,
) -> dict[str, Any]:
    selected_items: list[tuple[float, Any, dict[str, Any]]] = []
    actual_items: list[tuple[float, Any, dict[str, Any]]] = []
    missing_metadata = 0
    for block in selected_blocks:
        meta = _candidate_metadata(block_pool, block)
        if not meta:
            missing_metadata += 1
            continue
        score, detail = _value_score_from_metadata(meta)
        selected_items.append((score, block, detail))
    for block in actual_blocks:
        meta = _candidate_metadata(block_pool, block)
        if not meta:
            missing_metadata += 1
            continue
        score, detail = _value_score_from_metadata(meta)
        actual_items.append((score, block, detail))

    selected_items.sort(key=lambda item: item[0], reverse=True)
    actual_items.sort(key=lambda item: item[0])
    expected_count = min(len(selected_blocks), len(actual_blocks))
    pair_count = min(len(selected_items), len(actual_items))
    pass_count = 0
    margins: list[float] = []
    pairs: list[dict[str, Any]] = []
    for selected_item, actual_item in zip(
        selected_items[:pair_count],
        actual_items[:pair_count],
    ):
        selected_score, selected_block, selected_detail = selected_item
        actual_score, actual_block, actual_detail = actual_item
        margin = selected_score - actual_score
        margins.append(margin)
        passed = margin >= min_margin
        if passed:
            pass_count += 1
        pairs.append(
            {
                "selected_block_id": getattr(selected_block, "block_id", None),
                "actual_block_id": getattr(actual_block, "block_id", None),
                "selected": selected_detail,
                "actual": actual_detail,
                "value_score_margin": _json_safe_score_atom(margin),
                "min_value_score_margin": min_margin,
                "passed": passed,
            }
        )
    return {
        "expected_count": expected_count,
        "count": pair_count,
        "pass_count": pass_count,
        "min_value_score_margin": min_margin,
        "missing_metadata": missing_metadata,
        "value_score_margin": _summarize_numeric(margins),
        "values": pairs[:limit],
        "truncated": len(pairs) > limit,
    }


def _value_score_margin_guard_reason(
    block_pool: Any,
    selected_blocks: Sequence[Any],
    actual_blocks: Sequence[Any],
) -> tuple[str | None, dict[str, Any] | None]:
    min_margin = _enforcement_min_value_score_margin(block_pool)
    if min_margin <= 0.0:
        return None, None
    details = _value_score_margin_pair_details(
        block_pool,
        selected_blocks,
        actual_blocks,
        min_margin,
    )
    if (
        details["count"] <= 0
        or details["count"] < details["expected_count"]
        or details["pass_count"] < details["count"]
    ):
        return "value_score_margin_below_min", {
            "min_value_score_margin": min_margin,
            "value_score_margin_pair_details": details,
        }
    return None, {
        "min_value_score_margin": min_margin,
        "value_score_margin_pair_details": details,
    }


def _bucket_response_value_tail_tuple(meta: dict[str, Any]) -> tuple[Any, ...]:
    gap = _expected_or_predicted_gap_sec(meta)
    response_length = _as_int(meta.get("omni_response_length")) or 0
    history_tokens = _as_int(meta.get("omni_history_tokens_after")) or 0
    terminal = bool(meta.get("omni_is_terminal_turn"))
    depth_ratio = _as_float(meta.get("block_depth_ratio")) or 0.0
    suffix_blocks = _as_int(meta.get("suffix_blocks_including")) or 0
    lost_tokens = _as_int(meta.get("estimated_lost_tokens_if_prefix_break"))
    if lost_tokens is None:
        lost_tokens = history_tokens
    value_score, gap_value, size_bonus, loss_penalty = _value_cost_eviction_score(
        gap,
        terminal,
        history_tokens,
        response_length,
        lost_tokens,
    )
    return (
        _prefetch_protection_rank(meta),
        1 if terminal else 0,
        _gap_bucket_rank(gap),
        response_length,
        value_score,
        gap_value,
        size_bonus,
        -loss_penalty,
        depth_ratio,
        -lost_tokens,
        -suffix_blocks,
        gap,
        history_tokens,
    )


def _estimated_audio_total_sec(meta: dict[str, Any], response_length: int) -> float | None:
    direct = _as_float(meta.get("omni_audio_total_sec"))
    if direct is not None and direct >= 0.0:
        return direct
    sec_per_token = _env_float(
        "OMNI_KV_EVICT_AUDIO_SEC_PER_RESPONSE_TOKEN",
        _DEFAULT_AUDIO_SEC_PER_RESPONSE_TOKEN,
    )
    if sec_per_token <= 0.0:
        return None
    return max(0.0, response_length) * sec_per_token


def _audio_reply_delay_sec(meta: dict[str, Any]) -> float:
    if meta.get("omni_is_terminal_turn"):
        return float("inf")
    direct = _as_float(meta.get("omni_audio_reply_delay_sec"))
    if direct is not None and direct >= 0.0:
        return direct
    response_length = _as_int(meta.get("omni_response_length")) or 0
    total = _estimated_audio_total_sec(meta, response_length)
    if total is None:
        return _predicted_gap_sec(meta)
    remaining = (
        _as_float(meta.get("omni_audio_remaining_at_finish_sec"))
        or _as_float(meta.get("omni_audio_remaining_sec"))
        or 0.0
    )
    total_weight = _env_float("OMNI_KV_EVICT_AUDIO_TOTAL_WEIGHT", 1.0)
    return max(0.0, remaining) + max(0.0, total_weight) * total


def _near_time_bucket(value: float, near_sec: float | None = None) -> float:
    if math.isinf(value):
        return value
    if near_sec is None:
        near_sec = _env_float("OMNI_KV_EVICT_AUDIO_NEAR_SEC", _DEFAULT_AUDIO_NEAR_SEC)
    if near_sec <= 0.0:
        return value
    return math.floor(max(0.0, value) / near_sec)


def _hybrid_audio_gap_sec(meta: dict[str, Any], predicted_gap: float) -> float:
    audio_weight = _env_float("OMNI_KV_EVICT_HYBRID_AUDIO_WEIGHT", 1.0)
    response_weight = _env_float("OMNI_KV_EVICT_HYBRID_RESPONSE_WEIGHT", 1.0)
    audio_gap = _audio_reply_delay_sec(meta)
    if math.isinf(audio_gap) or math.isinf(predicted_gap):
        return float("inf")
    return (
        max(0.0, audio_weight) * audio_gap
        + max(0.0, response_weight) * predicted_gap
    )


def _reuse_pressure_score(gap_sec: float, history_tokens: int) -> float:
    if math.isinf(gap_sec):
        return float("inf")
    return max(0.0, gap_sec) * max(1.0, float(max(0, history_tokens)))


def _evict_potential_score(gap_sec: float, history_tokens: int) -> float:
    pressure = _reuse_pressure_score(gap_sec, history_tokens)
    if math.isinf(pressure):
        return float("inf")
    small = _env_float("OMNI_KV_EVICT_POTENTIAL_SMALL_TOKEN_SEC", 50_000.0)
    extreme = _env_float("OMNI_KV_EVICT_POTENTIAL_EXTREME_TOKEN_SEC", 200_000.0)
    if extreme <= small:
        return pressure
    if pressure <= small:
        return 0.0
    if pressure >= extreme:
        return 1.0
    return (pressure - small) / (extreme - small)


def _bidaw_block_size_tokens(block_size_tokens: int | Sequence[int] | None = None) -> int:
    if isinstance(block_size_tokens, int) and block_size_tokens > 0:
        return block_size_tokens
    if isinstance(block_size_tokens, Sequence) and not isinstance(
        block_size_tokens,
        (str, bytes),
    ):
        for item in block_size_tokens:
            value = _as_int(item)
            if value is not None and value > 0:
                return int(value)
    return max(1, _env_int("OMNI_KV_EVICT_BIDAW_BLOCK_SIZE_TOKENS", 16))


def _bidaw_blocks_per_sec_default() -> float:
    return max(0.0, _env_float("OMNI_KV_EVICT_BIDAW_BLOCKS_PER_SEC", 0.0))


def _bidaw_observe_blocks_per_sec(
    block_pool: Any,
    meta: dict[str, Any],
    block_size_tokens: int | Sequence[int] | None,
) -> float:
    """Track a lightweight demand-block rate for Bidaw-style reuse distance."""

    default_rate = _bidaw_blocks_per_sec_default()
    trace_ts = _as_float(meta.get("omni_trace_timestamp"))
    if trace_ts is None:
        return default_rate

    block_size = _bidaw_block_size_tokens(block_size_tokens)
    history_before = _as_int(meta.get("omni_history_tokens_before")) or 0
    demand_blocks = max(0, history_before // block_size)
    state = getattr(block_pool, "_omni_kv_evict_bidaw_rate_state", None)
    if not isinstance(state, dict):
        state = {
            "last_ts": trace_ts,
            "pending_blocks": demand_blocks,
            "ewma_blocks_per_sec": default_rate,
        }
        try:
            setattr(block_pool, "_omni_kv_evict_bidaw_rate_state", state)
        except Exception:
            pass
        return default_rate

    last_ts = _as_float(state.get("last_ts"))
    if last_ts is None:
        state["last_ts"] = trace_ts
        state["pending_blocks"] = demand_blocks
        return _as_float(state.get("ewma_blocks_per_sec")) or default_rate

    if trace_ts > last_ts:
        elapsed = max(1e-6, trace_ts - last_ts)
        pending_blocks = max(0.0, float(state.get("pending_blocks", 0.0)))
        sample_rate = pending_blocks / elapsed
        alpha = min(
            1.0,
            max(0.0, _env_float("OMNI_KV_EVICT_BIDAW_RATE_EWMA_ALPHA", 0.20)),
        )
        previous = _as_float(state.get("ewma_blocks_per_sec"))
        if previous is None or previous <= 0.0:
            ewma = sample_rate if sample_rate > 0.0 else default_rate
        else:
            ewma = alpha * sample_rate + (1.0 - alpha) * previous
        state["ewma_blocks_per_sec"] = max(0.0, ewma)
        state["last_ts"] = trace_ts
        state["pending_blocks"] = demand_blocks
    else:
        state["pending_blocks"] = (
            max(0.0, float(state.get("pending_blocks", 0.0))) + demand_blocks
        )
    return max(0.0, _as_float(state.get("ewma_blocks_per_sec")) or default_rate)


def _bidaw_attach_reuse_estimate(
    block_pool: Any,
    meta: dict[str, Any],
    block_size_tokens: int | Sequence[int] | None,
) -> None:
    blocks_per_sec = _bidaw_observe_blocks_per_sec(
        block_pool,
        meta,
        block_size_tokens,
    )
    capacity_blocks = _block_pool_capacity_blocks(block_pool)
    if capacity_blocks > 0:
        meta["omni_bidaw_capacity_blocks"] = capacity_blocks
    if blocks_per_sec > 0.0:
        meta["omni_bidaw_blocks_per_sec"] = blocks_per_sec
        gap = _expected_or_predicted_gap_sec(meta)
        if not math.isinf(gap):
            meta["omni_bidaw_reuse_lower_bound_blocks"] = (
                max(0.0, gap) * blocks_per_sec
            )
        predicted_gap = _predicted_gap_sec(meta)
        if not math.isinf(predicted_gap):
            meta["omni_bidaw_answer_reuse_lower_bound_blocks"] = (
                max(0.0, predicted_gap) * blocks_per_sec
            )


def _bidaw_reuse_lower_bound_blocks(
    meta: Mapping[str, Any],
    gap_sec: float,
    history_tokens: int,
    *,
    answer_only: bool,
) -> float:
    direct_key = (
        "omni_bidaw_answer_reuse_lower_bound_blocks"
        if answer_only
        else "omni_bidaw_reuse_lower_bound_blocks"
    )
    direct = _as_float(meta.get(direct_key))
    if direct is not None and direct >= 0.0:
        return direct
    if math.isinf(gap_sec):
        return float("inf")
    blocks_per_sec = _as_float(meta.get("omni_bidaw_blocks_per_sec"))
    if blocks_per_sec is None or blocks_per_sec <= 0.0:
        blocks_per_sec = _bidaw_blocks_per_sec_default()
    if blocks_per_sec > 0.0:
        return max(0.0, gap_sec) * blocks_per_sec
    block_size = max(1, _env_int("OMNI_KV_EVICT_BIDAW_BLOCK_SIZE_TOKENS", 16))
    return max(0.0, gap_sec) * max(1.0, history_tokens / block_size)


def _bidaw_capacity_blocks(meta: Mapping[str, Any]) -> int:
    capacity = _as_int(meta.get("omni_bidaw_capacity_blocks"))
    if capacity is not None and capacity > 0:
        return capacity
    return _env_int("OMNI_KV_EVICT_BIDAW_CAPACITY_BLOCKS", 0)


def _bidaw_eviction_potential_tuple(
    meta: Mapping[str, Any],
    gap_sec: float,
    terminal: bool,
    history_tokens: int,
    *,
    answer_only: bool,
) -> tuple[float, float, float, float]:
    if terminal:
        return 1.0, 0.0, float("inf"), float("inf")

    reuse_blocks = _bidaw_reuse_lower_bound_blocks(
        meta,
        gap_sec,
        history_tokens,
        answer_only=answer_only,
    )
    capacity = _bidaw_capacity_blocks(meta)
    if capacity <= 0:
        legacy = _evict_potential_score(gap_sec, history_tokens)
        return legacy, max(0.0, 1.0 - legacy), reuse_blocks, reuse_blocks

    small_ratio = max(
        0.0,
        _env_float("OMNI_KV_EVICT_BIDAW_SMALL_REUSE_RATIO", 1.0),
    )
    extreme_ratio = max(
        small_ratio + 1e-6,
        _env_float("OMNI_KV_EVICT_BIDAW_EXTREME_REUSE_RATIO", 3.0),
    )
    small = capacity * small_ratio
    extreme = capacity * extreme_ratio
    if reuse_blocks <= small:
        hit_potential = 1.0
    elif reuse_blocks >= extreme:
        hit_potential = 0.0
    else:
        hit_potential = 1.0 - ((reuse_blocks - small) / (extreme - small))
    evict_score = 1.0 - hit_potential
    reuse_bucket = (
        float("inf")
        if math.isinf(reuse_blocks)
        else math.floor(max(0.0, reuse_blocks) / max(1.0, capacity))
    )
    return evict_score, hit_potential, reuse_blocks, reuse_bucket


def _candidate_record(
    block: Any,
    now_wall: float,
    block_pool: Any | None = None,
) -> dict[str, Any]:
    raw_meta = getattr(block, "_omni_kv_evict_shadow_meta", None)
    meta = dict(raw_meta) if isinstance(raw_meta, Mapping) and raw_meta else {}
    if not meta and block_pool is not None:
        block_id_for_meta = getattr(block, "block_id", None)
        if block_id_for_meta is not None:
            mapped_meta = _metadata_map(block_pool).get(int(block_id_for_meta))
            meta = (
                dict(mapped_meta)
                if isinstance(mapped_meta, Mapping) and mapped_meta
                else {}
            )
    # Metadata is also stored centrally because KVCacheBlock uses normal Python
    # attributes in some builds; central lookup handles slot-based block types.
    record = {
        "block_id": getattr(block, "block_id", None),
        "block_hash_repr": _block_hash_repr(block),
        "has_hash": getattr(block, "block_hash", None) is not None,
        "ref_cnt": getattr(block, "ref_cnt", None),
        "is_null": bool(getattr(block, "is_null", False)),
        "freed_age_sec": None,
        "metadata_present": bool(meta),
    }
    record.update(meta)
    freed_wall = _as_float(record.get("freed_wall_time"))
    if freed_wall is not None:
        record["freed_age_sec"] = max(0.0, now_wall - freed_wall)
    return record


def _candidate_metadata(block_pool: Any, block: Any) -> Mapping[str, Any]:
    meta = getattr(block, "_omni_kv_evict_shadow_meta", None)
    if isinstance(meta, Mapping) and meta:
        return meta
    block_id = getattr(block, "block_id", None)
    if block_id is None:
        return {}
    mapped = _metadata_map(block_pool).get(int(block_id))
    return mapped if isinstance(mapped, Mapping) and mapped else {}


def _is_cached_free_block(block: Any) -> bool:
    return (
        getattr(block, "block_hash", None) is not None
        and getattr(block, "ref_cnt", None) == 0
        and not bool(getattr(block, "is_null", False))
    )


def _is_linked_free_block(block: Any) -> bool:
    return (
        getattr(block, "prev_free_block", None) is not None
        and getattr(block, "next_free_block", None) is not None
    )


def _peek_free_blocks(block_pool: Any, limit: int | None) -> list[Any]:
    queue = block_pool.free_block_queue
    if limit is not None and limit <= 0:
        return []
    head = getattr(queue, "fake_free_list_head", None)
    tail = getattr(queue, "fake_free_list_tail", None)
    if head is None or tail is None:
        blocks = list(queue.get_all_free_blocks())
        return blocks if limit is None else blocks[:limit]

    blocks: list[Any] = []
    curr = getattr(head, "next_free_block", None)
    while curr is not None and curr is not tail:
        blocks.append(curr)
        if limit is not None and len(blocks) >= limit:
            break
        curr = getattr(curr, "next_free_block", None)
    return blocks


def _free_queue_cached_profile(block_pool: Any) -> dict[str, Any] | None:
    scan_limit = _env_int("OMNI_KV_EVICT_PROFILE_NO_CACHED_SCAN_BLOCKS", 0)
    if scan_limit <= 0:
        return None
    try:
        free_count = block_pool.get_num_free_blocks()
        blocks = _peek_free_blocks(block_pool, min(scan_limit, free_count))
    except Exception:
        logger.exception("[OmniKVEvict] failed to profile free queue")
        return {"profile_error": "free_queue_scan_error"}

    cached_count = 0
    cached_with_metadata = 0
    first_cached_offset: int | None = None
    first_metadata_offset: int | None = None
    linked_cached_count = 0
    metadata = _metadata_map(block_pool)
    for offset, block in enumerate(blocks):
        if not _is_cached_free_block(block):
            continue
        cached_count += 1
        if first_cached_offset is None:
            first_cached_offset = offset
        if _is_linked_free_block(block):
            linked_cached_count += 1
        block_id = getattr(block, "block_id", None)
        if block_id is None:
            continue
        try:
            has_metadata = int(block_id) in metadata
        except (TypeError, ValueError):
            has_metadata = False
        if has_metadata:
            cached_with_metadata += 1
            if first_metadata_offset is None:
                first_metadata_offset = offset

    profile: dict[str, Any] = {
        "profile_scan_limit": scan_limit,
        "profile_scanned_blocks": len(blocks),
        "profile_free_blocks": free_count,
        "profile_cached_blocks": cached_count,
        "profile_cached_with_metadata": cached_with_metadata,
        "profile_linked_cached_blocks": linked_cached_count,
        "profile_first_cached_offset": first_cached_offset,
        "profile_first_metadata_offset": first_metadata_offset,
    }
    state = getattr(block_pool, "_omni_kv_evict_candidate_index", None)
    if isinstance(state, dict):
        heap = state.get("heap") or []
        versions = state.get("versions") or {}
        live_entries = 0
        stale_entries = 0
        invalid_entries = 0
        duplicate_live_entries = 0
        live_cached_blocks = 0
        live_linked_cached_blocks = 0
        live_cached_with_metadata = 0
        seen_live_block_ids: set[int] = set()
        for item in heap:
            try:
                _, version, block_id = item
                block_id = int(block_id)
            except (TypeError, ValueError):
                invalid_entries += 1
                continue
            if int(versions.get(block_id, 0)) != int(version):
                stale_entries += 1
                continue
            if block_id in seen_live_block_ids:
                duplicate_live_entries += 1
                continue
            seen_live_block_ids.add(block_id)
            live_entries += 1
            if block_id < 0 or block_id >= len(block_pool.blocks):
                invalid_entries += 1
                continue
            block = block_pool.blocks[block_id]
            if not _is_cached_free_block(block):
                continue
            live_cached_blocks += 1
            if _is_linked_free_block(block):
                live_linked_cached_blocks += 1
            if block_id in metadata:
                live_cached_with_metadata += 1
        profile.update(
            {
                "profile_index_present": True,
                "profile_index_policy": state.get("policy"),
                "profile_index_heap_size": len(heap),
                "profile_index_versions_size": len(versions),
                "profile_index_pushes": int(state.get("pushes", 0)),
                "profile_index_pops": int(state.get("pops", 0)),
                "profile_index_selected_size_cap_skips": int(
                    state.get("selected_size_cap_skips", 0)
                ),
                "profile_index_selected_size_cap_pops": int(
                    state.get("selected_size_cap_pops", 0)
                ),
                "profile_index_partial_not_enough_candidates": int(
                    state.get("partial_not_enough_candidates", 0)
                ),
                "profile_index_selected": int(state.get("selected", 0)),
                "profile_index_fallbacks": int(state.get("fallbacks", 0)),
                "profile_index_live_entries": live_entries,
                "profile_index_stale_entries": stale_entries,
                "profile_index_invalid_entries": invalid_entries,
                "profile_index_duplicate_live_entries": duplicate_live_entries,
                "profile_index_live_cached_blocks": live_cached_blocks,
                "profile_index_live_linked_cached_blocks": live_linked_cached_blocks,
                "profile_index_live_cached_with_metadata": live_cached_with_metadata,
            }
        )
    else:
        profile["profile_index_present"] = False
    return profile


def _score(policy: str, cand: dict[str, Any], hot_protect_sec: float) -> tuple[Any, ...]:
    if policy == "expected_gap_capacity_adaptive_tail_aware":
        policy = _DEFAULT_ADAPTIVE_LARGE_POLICY
    expected_gap = _as_float(cand.get("omni_expected_next_gap_sec"))
    response_length = _as_int(cand.get("omni_response_length")) or 0
    history_tokens = _as_int(cand.get("omni_history_tokens_after")) or 0
    predicted_gap = _predicted_gap_sec(cand)
    terminal = bool(cand.get("omni_is_terminal_turn"))
    depth_ratio = _as_float(cand.get("block_depth_ratio")) or 0.0
    suffix_blocks = _as_int(cand.get("suffix_blocks_including")) or 0
    lost_tokens = _as_int(cand.get("estimated_lost_tokens_if_prefix_break"))
    if lost_tokens is None:
        lost_tokens = history_tokens
    if policy == "oracle_next_use":
        gap = float("inf") if terminal or expected_gap is None else expected_gap
        return (gap, depth_ratio, -lost_tokens, -suffix_blocks, response_length)
    if policy == "response_len":
        return (float("inf") if terminal else response_length, history_tokens)
    if policy == "response_len_tail_aware":
        return (
            1 if terminal else 0,
            response_length,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
        )
    if policy == "prefetch_hit_tail_aware":
        return (
            _narrow_prefetch_protection_rank(cand),
            1 if terminal else 0,
            response_length,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
        )
    if policy == "prefetch_hit_soft_tail_aware":
        response_score = _narrow_prefetch_soft_response_score(
            response_length,
            cand,
        )
        return (
            1 if terminal else 0,
            response_score,
            response_length,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
        )
    if policy == "reuse_protect_response_len":
        protected = _prefetch_protected(cand)
        return (
            0 if protected else 1,
            1 if terminal else 0,
            response_length,
            predicted_gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
        )
    if policy == "prefetch_soft_response_len":
        response_score = _prefetch_soft_response_score(response_length, cand)
        return (
            float("inf") if terminal else response_score,
            response_length,
            history_tokens,
        )
    if policy == "predicted_gap":
        return (predicted_gap, depth_ratio, -lost_tokens, response_length)
    if policy == "hot_protect_response_len":
        protected = predicted_gap <= hot_protect_sec
        return (
            0 if protected else 1,
            predicted_gap,
            response_length,
            depth_ratio,
            -lost_tokens,
            history_tokens,
        )
    if policy == "audio_late_then_kv":
        audio_gap = _audio_reply_delay_sec(cand)
        return (
            _prefetch_protection_rank(cand),
            _near_time_bucket(audio_gap),
            history_tokens,
            audio_gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            response_length,
        )
    if policy == "hybrid_late_then_kv":
        hybrid_gap = _hybrid_audio_gap_sec(cand, predicted_gap)
        return (
            _prefetch_protection_rank(cand),
            _near_time_bucket(hybrid_gap),
            history_tokens,
            hybrid_gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            response_length,
        )
    if policy == "audio_late_tail_aware":
        audio_gap = _audio_reply_delay_sec(cand)
        return (
            _prefetch_protection_rank(cand),
            _near_time_bucket(audio_gap),
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
            audio_gap,
            response_length,
        )
    if policy == "hybrid_late_tail_aware":
        hybrid_gap = _hybrid_audio_gap_sec(cand, predicted_gap)
        return (
            _prefetch_protection_rank(cand),
            _near_time_bucket(hybrid_gap),
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
            hybrid_gap,
            response_length,
        )
    if policy == "audio_reuse_potential":
        audio_gap = _audio_reply_delay_sec(cand)
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            _evict_potential_score(audio_gap, history_tokens),
            _near_time_bucket(audio_gap),
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
            audio_gap,
            response_length,
        )
    if policy == "hybrid_reuse_potential":
        hybrid_gap = _hybrid_audio_gap_sec(cand, predicted_gap)
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            _evict_potential_score(hybrid_gap, history_tokens),
            _near_time_bucket(hybrid_gap),
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
            hybrid_gap,
            response_length,
        )
    if policy in {"bidaw_answer_potential", "bidaw_hit_potential"}:
        answer_only = policy == "bidaw_answer_potential"
        gap = predicted_gap if answer_only else _expected_or_predicted_gap_sec(cand)
        evict_score, hit_potential, reuse_blocks, reuse_bucket = (
            _bidaw_eviction_potential_tuple(
                cand,
                gap,
                terminal,
                history_tokens,
                answer_only=answer_only,
            )
        )
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            evict_score,
            reuse_bucket,
            -hit_potential,
            gap,
            reuse_blocks,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            response_length,
            history_tokens,
        )
    if policy == "activity_size_aware":
        protected = _prefetch_protected(cand)
        activity_gap = _activity_gap_sec(cand)
        size_score = _history_size_score(history_tokens)
        query_length = _as_int(cand.get("omni_query_length")) or 0
        return (
            0 if protected else 1,
            1 if terminal else 0,
            activity_gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            size_score,
            query_length,
            response_length,
            history_tokens,
        )
    if policy == "expected_gap_tail_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            response_length,
            history_tokens,
        )
    if policy == "expected_gap_bucket_response_tail_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            _gap_bucket_rank(gap),
            response_length,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            gap,
            history_tokens,
        )
    if policy == "expected_gap_bucket_response_value_tail_aware":
        return _bucket_response_value_tail_tuple(cand)
    if policy == "near_gap_response_len_tail_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        near = gap <= _env_float("OMNI_KV_EVICT_NEAR_GAP_SEC", 45.0) and not terminal
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            0 if near else 1,
            response_length,
            gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
        )
    if policy == "expected_gap_bucket_response_loss_tail_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            _gap_bucket_rank(gap),
            response_length,
            -lost_tokens,
            depth_ratio,
            -suffix_blocks,
            gap,
            history_tokens,
        )
    if policy == "expected_gap_bucket_response_near_cost_tail_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        bucket = _gap_bucket_rank(gap)
        near = bucket <= 1 and not terminal
        loss_penalty = math.log1p(max(0, lost_tokens))
        near_cost = gap - loss_penalty if near else 0.0
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            bucket,
            response_length,
            near_cost,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            gap,
            history_tokens,
        )
    if policy == "expected_gap_bucket_response_cost_tail_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        loss_penalty = math.log1p(max(0, lost_tokens))
        cost_score = float("inf") if math.isinf(gap) else gap - loss_penalty
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            _gap_bucket_rank(gap),
            response_length,
            cost_score,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            gap,
            history_tokens,
        )
    if policy == "expected_gap_bucket_near_cost_tail_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        bucket = _gap_bucket_rank(gap)
        near = bucket <= 1 and not terminal
        loss_penalty = math.log1p(max(0, lost_tokens))
        near_cost = gap - loss_penalty if near else 0.0
        response_score = 0 if near else response_length
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            bucket,
            near_cost,
            response_score,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            gap,
            response_length,
            history_tokens,
        )
    if policy == "expected_gap_bucket_near_gap_tail_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        bucket = _gap_bucket_rank(gap)
        near = bucket <= 1 and not terminal
        near_gap = gap if near else 0.0
        response_score = 0 if near else response_length
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            bucket,
            near_gap,
            response_score,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            gap,
            response_length,
            history_tokens,
        )
    if policy == "expected_gap_bucket_cost_tail_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        loss_penalty = math.log1p(max(0, lost_tokens))
        cost_score = float("inf") if math.isinf(gap) else gap - loss_penalty
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            _gap_bucket_rank(gap),
            cost_score,
            response_length,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            gap,
            history_tokens,
        )
    if policy == "expected_gap_bucket_tail_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            _gap_bucket_rank(gap),
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            response_length,
            gap,
            history_tokens,
        )
    if policy == "expected_gap_cost_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        loss_penalty = _cost_loss_penalty(lost_tokens)
        near_penalty = _near_reuse_history_penalty(
            gap,
            terminal,
            history_tokens,
        )
        risk_penalty = _piecewise_risk_penalty(
            gap,
            terminal,
            history_tokens,
        )
        score = (
            float("inf")
            if math.isinf(gap)
            else gap - loss_penalty - near_penalty - risk_penalty
        )
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            score,
            gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            response_length,
        )
    if policy == "history_cost_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        history_cost = math.log1p(max(1, history_tokens))
        score = float("inf") if math.isinf(gap) else gap - history_cost
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            score,
            gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            response_length,
        )
    if policy == "expected_gap_size_cost_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        loss_weight = _env_float("OMNI_KV_EVICT_SIZE_LOSS_WEIGHT", 1.0)
        loss_penalty = max(0.0, loss_weight) * math.log1p(max(0, lost_tokens))
        near_penalty = _near_reuse_history_penalty(
            gap,
            terminal,
            history_tokens,
        )
        risk_penalty = _piecewise_risk_penalty(
            gap,
            terminal,
            history_tokens,
        )
        size_bonus = _size_eviction_bonus(
            gap,
            terminal,
            history_tokens,
            response_length,
        )
        score = (
            size_bonus
            if terminal
            else gap + size_bonus - loss_penalty - near_penalty - risk_penalty
        )
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            score,
            gap,
            size_bonus,
            response_length,
            history_tokens,
            depth_ratio,
            -suffix_blocks,
            -lost_tokens,
        )
    if policy == "expected_gap_value_cost_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        score, gap_value, size_bonus, loss_penalty = _value_cost_eviction_score(
            gap,
            terminal,
            history_tokens,
            response_length,
            lost_tokens,
        )
        near_penalty = _near_reuse_history_penalty(
            gap,
            terminal,
            history_tokens,
        )
        risk_penalty = _piecewise_risk_penalty(
            gap,
            terminal,
            history_tokens,
        )
        if not math.isinf(score):
            score -= near_penalty + risk_penalty
        return (
            _prefetch_protection_rank(cand),
            score,
            gap_value,
            size_bonus,
            -loss_penalty,
            response_length,
            history_tokens,
            depth_ratio,
            -suffix_blocks,
            -lost_tokens,
            1 if terminal else 0,
        )
    if policy == "expected_gap_response_penalty_cost_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        loss_penalty = math.log1p(max(0, lost_tokens))
        response_penalty = _response_recompute_penalty(response_length)
        score = (
            float("inf")
            if math.isinf(gap)
            else gap - loss_penalty - response_penalty
        )
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            score,
            gap,
            -response_length,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
        )
    if policy == "expected_gap_cost_response_tail_aware":
        gap = _expected_or_predicted_gap_sec(cand)
        loss_penalty = math.log1p(max(0, lost_tokens))
        score = float("inf") if math.isinf(gap) else gap - loss_penalty
        return (
            _prefetch_protection_rank(cand),
            1 if terminal else 0,
            score,
            response_length,
            gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
        )
    if policy == "cost_aware":
        gap = predicted_gap
        if terminal:
            gap = float("inf")
        score = gap + math.log1p(max(0, history_tokens))
        return (
            _prefetch_protection_rank(cand),
            score,
            depth_ratio,
            -lost_tokens,
            response_length,
        )
    return (0,)


def _score_metadata(
    policy: str,
    meta: dict[str, Any],
    hot_protect_sec: float,
) -> tuple[Any, ...]:
    if policy == "expected_gap_capacity_adaptive_tail_aware":
        policy = _DEFAULT_ADAPTIVE_LARGE_POLICY
    expected_gap = _as_float(meta.get("omni_expected_next_gap_sec"))
    response_length = _as_int(meta.get("omni_response_length")) or 0
    history_tokens = _as_int(meta.get("omni_history_tokens_after")) or 0
    predicted_gap = _predicted_gap_sec(meta)
    terminal = bool(meta.get("omni_is_terminal_turn"))
    depth_ratio = _as_float(meta.get("block_depth_ratio")) or 0.0
    suffix_blocks = _as_int(meta.get("suffix_blocks_including")) or 0
    lost_tokens = _as_int(meta.get("estimated_lost_tokens_if_prefix_break"))
    if lost_tokens is None:
        lost_tokens = history_tokens
    if policy == "oracle_next_use":
        gap = float("inf") if terminal or expected_gap is None else expected_gap
        return (gap, depth_ratio, -lost_tokens, -suffix_blocks, response_length)
    if policy == "response_len":
        return (float("inf") if terminal else response_length, history_tokens)
    if policy == "response_len_tail_aware":
        return (
            1 if terminal else 0,
            response_length,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
        )
    if policy == "prefetch_hit_tail_aware":
        return (
            _narrow_prefetch_protection_rank(meta),
            1 if terminal else 0,
            response_length,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
        )
    if policy == "prefetch_hit_soft_tail_aware":
        response_score = _narrow_prefetch_soft_response_score(
            response_length,
            meta,
        )
        return (
            1 if terminal else 0,
            response_score,
            response_length,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
        )
    if policy == "reuse_protect_response_len":
        protected = _prefetch_protected(meta)
        return (
            0 if protected else 1,
            1 if terminal else 0,
            response_length,
            predicted_gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
        )
    if policy == "prefetch_soft_response_len":
        response_score = _prefetch_soft_response_score(response_length, meta)
        return (
            float("inf") if terminal else response_score,
            response_length,
            history_tokens,
        )
    if policy == "predicted_gap":
        return (predicted_gap, depth_ratio, -lost_tokens, response_length)
    if policy == "hot_protect_response_len":
        protected = predicted_gap <= hot_protect_sec
        return (
            0 if protected else 1,
            predicted_gap,
            response_length,
            depth_ratio,
            -lost_tokens,
            history_tokens,
        )
    if policy == "audio_late_then_kv":
        audio_gap = _audio_reply_delay_sec(meta)
        return (
            _prefetch_protection_rank(meta),
            _near_time_bucket(audio_gap),
            history_tokens,
            audio_gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            response_length,
        )
    if policy == "hybrid_late_then_kv":
        hybrid_gap = _hybrid_audio_gap_sec(meta, predicted_gap)
        return (
            _prefetch_protection_rank(meta),
            _near_time_bucket(hybrid_gap),
            history_tokens,
            hybrid_gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            response_length,
        )
    if policy == "audio_late_tail_aware":
        audio_gap = _audio_reply_delay_sec(meta)
        return (
            _prefetch_protection_rank(meta),
            _near_time_bucket(audio_gap),
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
            audio_gap,
            response_length,
        )
    if policy == "hybrid_late_tail_aware":
        hybrid_gap = _hybrid_audio_gap_sec(meta, predicted_gap)
        return (
            _prefetch_protection_rank(meta),
            _near_time_bucket(hybrid_gap),
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
            hybrid_gap,
            response_length,
        )
    if policy == "audio_reuse_potential":
        audio_gap = _audio_reply_delay_sec(meta)
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            _evict_potential_score(audio_gap, history_tokens),
            _near_time_bucket(audio_gap),
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
            audio_gap,
            response_length,
        )
    if policy == "hybrid_reuse_potential":
        hybrid_gap = _hybrid_audio_gap_sec(meta, predicted_gap)
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            _evict_potential_score(hybrid_gap, history_tokens),
            _near_time_bucket(hybrid_gap),
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
            hybrid_gap,
            response_length,
        )
    if policy in {"bidaw_answer_potential", "bidaw_hit_potential"}:
        answer_only = policy == "bidaw_answer_potential"
        gap = predicted_gap if answer_only else _expected_or_predicted_gap_sec(meta)
        evict_score, hit_potential, reuse_blocks, reuse_bucket = (
            _bidaw_eviction_potential_tuple(
                meta,
                gap,
                terminal,
                history_tokens,
                answer_only=answer_only,
            )
        )
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            evict_score,
            reuse_bucket,
            -hit_potential,
            gap,
            reuse_blocks,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            response_length,
            history_tokens,
        )
    if policy == "activity_size_aware":
        protected = _prefetch_protected(meta)
        activity_gap = _activity_gap_sec(meta)
        size_score = _history_size_score(history_tokens)
        query_length = _as_int(meta.get("omni_query_length")) or 0
        return (
            0 if protected else 1,
            1 if terminal else 0,
            activity_gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            size_score,
            query_length,
            response_length,
            history_tokens,
        )
    if policy == "expected_gap_tail_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            response_length,
            history_tokens,
        )
    if policy == "expected_gap_bucket_response_tail_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            _gap_bucket_rank(gap),
            response_length,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            gap,
            history_tokens,
        )
    if policy == "expected_gap_bucket_response_value_tail_aware":
        return _bucket_response_value_tail_tuple(meta)
    if policy == "near_gap_response_len_tail_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        near = gap <= _env_float("OMNI_KV_EVICT_NEAR_GAP_SEC", 45.0) and not terminal
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            0 if near else 1,
            response_length,
            gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
        )
    if policy == "expected_gap_bucket_response_loss_tail_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            _gap_bucket_rank(gap),
            response_length,
            -lost_tokens,
            depth_ratio,
            -suffix_blocks,
            gap,
            history_tokens,
        )
    if policy == "expected_gap_bucket_response_near_cost_tail_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        bucket = _gap_bucket_rank(gap)
        near = bucket <= 1 and not terminal
        loss_penalty = math.log1p(max(0, lost_tokens))
        near_cost = gap - loss_penalty if near else 0.0
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            bucket,
            response_length,
            near_cost,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            gap,
            history_tokens,
        )
    if policy == "expected_gap_bucket_response_cost_tail_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        loss_penalty = math.log1p(max(0, lost_tokens))
        cost_score = float("inf") if math.isinf(gap) else gap - loss_penalty
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            _gap_bucket_rank(gap),
            response_length,
            cost_score,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            gap,
            history_tokens,
        )
    if policy == "expected_gap_bucket_near_cost_tail_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        bucket = _gap_bucket_rank(gap)
        near = bucket <= 1 and not terminal
        loss_penalty = math.log1p(max(0, lost_tokens))
        near_cost = gap - loss_penalty if near else 0.0
        response_score = 0 if near else response_length
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            bucket,
            near_cost,
            response_score,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            gap,
            response_length,
            history_tokens,
        )
    if policy == "expected_gap_bucket_near_gap_tail_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        bucket = _gap_bucket_rank(gap)
        near = bucket <= 1 and not terminal
        near_gap = gap if near else 0.0
        response_score = 0 if near else response_length
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            bucket,
            near_gap,
            response_score,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            gap,
            response_length,
            history_tokens,
        )
    if policy == "expected_gap_bucket_cost_tail_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        loss_penalty = math.log1p(max(0, lost_tokens))
        cost_score = float("inf") if math.isinf(gap) else gap - loss_penalty
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            _gap_bucket_rank(gap),
            cost_score,
            response_length,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            gap,
            history_tokens,
        )
    if policy == "expected_gap_bucket_tail_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            _gap_bucket_rank(gap),
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            response_length,
            gap,
            history_tokens,
        )
    if policy == "expected_gap_cost_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        loss_penalty = _cost_loss_penalty(lost_tokens)
        near_penalty = _near_reuse_history_penalty(
            gap,
            terminal,
            history_tokens,
        )
        risk_penalty = _piecewise_risk_penalty(
            gap,
            terminal,
            history_tokens,
        )
        score = (
            float("inf")
            if math.isinf(gap)
            else gap - loss_penalty - near_penalty - risk_penalty
        )
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            score,
            gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            response_length,
        )
    if policy == "history_cost_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        history_cost = math.log1p(max(1, history_tokens))
        score = float("inf") if math.isinf(gap) else gap - history_cost
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            score,
            gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            response_length,
        )
    if policy == "expected_gap_size_cost_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        loss_weight = _env_float("OMNI_KV_EVICT_SIZE_LOSS_WEIGHT", 1.0)
        loss_penalty = max(0.0, loss_weight) * math.log1p(max(0, lost_tokens))
        near_penalty = _near_reuse_history_penalty(
            gap,
            terminal,
            history_tokens,
        )
        risk_penalty = _piecewise_risk_penalty(
            gap,
            terminal,
            history_tokens,
        )
        size_bonus = _size_eviction_bonus(
            gap,
            terminal,
            history_tokens,
            response_length,
        )
        score = (
            size_bonus
            if terminal
            else gap + size_bonus - loss_penalty - near_penalty - risk_penalty
        )
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            score,
            gap,
            size_bonus,
            response_length,
            history_tokens,
            depth_ratio,
            -suffix_blocks,
            -lost_tokens,
        )
    if policy == "expected_gap_value_cost_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        score, gap_value, size_bonus, loss_penalty = _value_cost_eviction_score(
            gap,
            terminal,
            history_tokens,
            response_length,
            lost_tokens,
        )
        near_penalty = _near_reuse_history_penalty(
            gap,
            terminal,
            history_tokens,
        )
        risk_penalty = _piecewise_risk_penalty(
            gap,
            terminal,
            history_tokens,
        )
        if not math.isinf(score):
            score -= near_penalty + risk_penalty
        return (
            _prefetch_protection_rank(meta),
            score,
            gap_value,
            size_bonus,
            -loss_penalty,
            response_length,
            history_tokens,
            depth_ratio,
            -suffix_blocks,
            -lost_tokens,
            1 if terminal else 0,
        )
    if policy == "expected_gap_response_penalty_cost_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        loss_penalty = math.log1p(max(0, lost_tokens))
        response_penalty = _response_recompute_penalty(response_length)
        score = (
            float("inf")
            if math.isinf(gap)
            else gap - loss_penalty - response_penalty
        )
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            score,
            gap,
            -response_length,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
        )
    if policy == "expected_gap_cost_response_tail_aware":
        gap = _expected_or_predicted_gap_sec(meta)
        loss_penalty = math.log1p(max(0, lost_tokens))
        score = float("inf") if math.isinf(gap) else gap - loss_penalty
        return (
            _prefetch_protection_rank(meta),
            1 if terminal else 0,
            score,
            response_length,
            gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            history_tokens,
        )
    if policy == "cost_aware":
        gap = predicted_gap
        if terminal:
            gap = float("inf")
        score = gap + math.log1p(max(0, history_tokens))
        return (
            _prefetch_protection_rank(meta),
            score,
            depth_ratio,
            -lost_tokens,
            response_length,
        )
    return (0,)


def _choose_shadow_victims(
    policy: str,
    candidates: list[dict[str, Any]],
    count: int,
    hot_protect_sec: float,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if count >= len(candidates):
        return list(candidates)
    return heapq.nlargest(
        count,
        candidates,
        key=lambda cand: _score(policy, cand, hot_protect_sec),
    )


def _choose_policy_victim_records(
    policy: str,
    candidates: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    hot_protect_sec = _env_float("OMNI_KV_EVICT_HOT_PROTECT_SEC", 10.0)
    return _choose_shadow_victims(policy, candidates, count, hot_protect_sec)


def _write_event(event: dict[str, Any]) -> None:
    log_path = os.environ.get("OMNI_KV_EVICT_LOG_PATH")
    if not log_path:
        log_path = "/tmp/omni_kv_eviction_shadow.jsonl"
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
    except Exception:
        logger.exception("[OmniKVEvictShadow] failed to write %s", log_path)


def capture_request_blocks_for_shadow(
    block_pool: Any,
    request: Any,
    block_groups: Iterable[Any],
    block_size_tokens: int | Sequence[int] | None = None,
) -> None:
    if not kv_eviction_metadata_enabled():
        return
    profile_enabled = _profile_timing_enabled(block_pool)
    profile_started_ns = _profile_phase_start(profile_enabled)
    phase_started_ns = _profile_phase_start(profile_enabled)
    stats = _shadow_stats(block_pool)
    stats["capture_attempts"] += 1
    params = _request_kv_transfer_params(request)
    _record_capture_phase_profile(stats, "request_params", phase_started_ns)
    if _capture_terminal_only() and not bool(params.get("omni_is_terminal_turn")):
        if _capture_count_terminal_skips():
            _record_capture_gate_skip(stats, "nonterminal_request", profile_started_ns)
        return
    gate_reason = _capture_metadata_gate_reason(
        block_pool,
        int(stats.get("capture_attempts", 0)),
    )
    if gate_reason is not None:
        _record_capture_gate_skip(stats, gate_reason, profile_started_ns)
        return
    stats["capture_gate_active_requests"] = (
        int(stats.get("capture_gate_active_requests", 0)) + 1
    )
    phase_started_ns = _profile_phase_start(profile_enabled)
    now_wall = time.time()
    session_id = params.get("omni_session_id") or params.get("omni_prefetch_session_id")
    committed_hash = (
        params.get("omni_committed_prefix_hash")
        or params.get("omni_prefetch_committed_prefix_hash")
    )
    matched_hash = (
        params.get("omni_matched_prefix_hash")
        or params.get("omni_prefetch_matched_prefix_hash")
    )
    prefix_tokens = (
        params.get("omni_prefix_token_estimate")
        or params.get("omni_prefetch_prefix_token_estimate")
    )
    meta = {
        "request_id": getattr(request, "request_id", None),
        "omni_session_id": session_id,
        "omni_committed_prefix_hash": committed_hash,
        "omni_matched_prefix_hash": matched_hash,
        "omni_prefix_token_estimate": prefix_tokens,
        "omni_round_index": params.get("omni_round_index"),
        "omni_trace_timestamp": params.get("omni_trace_timestamp"),
        "omni_query_length": params.get("omni_query_length"),
        "omni_response_length": params.get("omni_response_length"),
        "omni_history_tokens_before": params.get("omni_history_tokens_before"),
        "omni_history_tokens_after": params.get("omni_history_tokens_after"),
        "omni_prev_gap_sec": params.get("omni_prev_gap_sec"),
        "omni_expected_next_gap_sec": params.get("omni_expected_next_gap_sec"),
        "omni_expected_next_use_ts": params.get("omni_expected_next_use_ts"),
        "omni_audio_total_sec": params.get("omni_audio_total_sec"),
        "omni_audio_remaining_at_finish_sec": params.get(
            "omni_audio_remaining_at_finish_sec"
        ),
        "omni_audio_reply_delay_sec": params.get("omni_audio_reply_delay_sec"),
        "omni_next_audio_total_sec": params.get("omni_next_audio_total_sec"),
        "omni_next_audio_reply_delay_sec": params.get(
            "omni_next_audio_reply_delay_sec"
        ),
        "omni_is_terminal_turn": bool(params.get("omni_is_terminal_turn")),
        "omni_scheduler_prefetch": bool(params.get("omni_scheduler_prefetch")),
        "omni_prefetch_cancellable": bool(params.get("omni_prefetch_cancellable")),
        "omni_prefetch_gpu_pin": bool(params.get("omni_prefetch_gpu_pin")),
        "omni_prefetch_hit_used": bool(params.get("omni_prefetch_hit_used")),
        "omni_prefetch_lookup_state": params.get("omni_prefetch_lookup_state"),
        "omni_prefetch_lead_ms": params.get("omni_prefetch_lead_ms"),
        "omni_prefetch_prefetch_scope": params.get("omni_prefetch_prefetch_scope"),
        "freed_wall_time": now_wall,
    }
    _bidaw_attach_reuse_estimate(block_pool, meta, block_size_tokens)
    mapping = _metadata_map(block_pool)
    captured = 0
    considered = 0
    groups = _normalize_block_groups(block_groups)
    _record_capture_phase_profile(stats, "request_meta", phase_started_ns)
    if not groups or not any(groups):
        stats["empty_block_requests"] += 1
        _record_capture_profile(
            stats,
            profile_started_ns,
            captured_blocks=captured,
            considered_blocks=considered,
        )
        return
    index_tail_limit = 0
    index_long_tail_limit = 0
    index_long_tail_min_blocks = 0
    if not shadow_enabled() and _enforcement_index_enabled():
        index_tail_limit = _index_max_blocks_per_request()
        if index_tail_limit > 0:
            index_long_tail_limit = _index_adaptive_long_tail()
            index_long_tail_min_blocks = _index_adaptive_long_min_blocks()
    debug_block_metadata = shadow_enabled() or _capture_debug_metadata_enabled()
    set_block_attr = _capture_set_block_attr_enabled()
    use_shared_meta = _capture_shared_meta_enabled()
    block_sizes = _coerce_block_sizes(block_size_tokens, len(groups))
    phase_started_ns = _profile_phase_start(profile_enabled)
    for kv_group_id, group in enumerate(groups):
        num_blocks = len(group)
        block_size = block_sizes[kv_group_id] if kv_group_id < len(block_sizes) else None
        group_tail_limit = _index_tail_limit_for_group(
            index_tail_limit,
            index_long_tail_limit,
            index_long_tail_min_blocks,
            num_blocks,
        )
        start_index = (
            max(0, num_blocks - group_tail_limit)
            if group_tail_limit > 0
            else 0
        )
        for block_index in range(start_index, num_blocks):
            block = group[block_index]
            considered += 1
            if bool(getattr(block, "is_null", False)):
                continue
            block_id = getattr(block, "block_id", None)
            if block_id is None:
                continue
            suffix_blocks = max(0, num_blocks - block_index)
            local_meta = {
                "kv_group_id": kv_group_id,
                "block_depth_ratio": (
                    0.0 if num_blocks <= 1 else block_index / (num_blocks - 1)
                ),
                "suffix_blocks_including": suffix_blocks,
                "estimated_lost_tokens_if_prefix_break": (
                    suffix_blocks * block_size if block_size is not None else None
                ),
            }
            if debug_block_metadata:
                token_start = (
                    block_index * block_size if block_size is not None else None
                )
                token_end = (
                    (block_index + 1) * block_size
                    if block_size is not None
                    else None
                )
                local_meta.update(
                    {
                        "block_index": block_index,
                        "num_blocks": num_blocks,
                        "block_size_tokens": block_size,
                        "token_start": token_start,
                        "token_end": token_end,
                        "is_head_block": block_index == 0,
                        "is_tail_block": block_index == num_blocks - 1,
                        "block_hash_repr": _block_hash_repr(block),
                    }
                )
            block_meta = (
                _BlockShadowMeta(meta, local_meta)
                if use_shared_meta
                else {**meta, **local_meta}
            )
            mapping[int(block_id)] = block_meta
            if set_block_attr:
                try:
                    setattr(block, "_omni_kv_evict_shadow_meta", block_meta)
                except Exception:
                    pass
            captured += 1
    _record_capture_phase_profile(
        stats,
        "block_loop",
        phase_started_ns,
        item_count=considered,
    )
    if captured:
        stats["captured_requests"] += 1
        stats["captured_blocks"] += captured
    _record_capture_profile(
        stats,
        profile_started_ns,
        captured_blocks=captured,
        considered_blocks=considered,
    )


def _hydrate_candidate_metadata(block_pool: Any, block: Any) -> None:
    block_id = getattr(block, "block_id", None)
    if block_id is None:
        return
    mapping = _metadata_map(block_pool)
    meta = mapping.get(int(block_id))
    if meta is None:
        return
    try:
        setattr(block, "_omni_kv_evict_shadow_meta", meta)
    except Exception:
        pass


def _build_shadow_event(block_pool: Any, num_blocks: int) -> dict[str, Any] | None:
    if not shadow_enabled():
        return None
    if not bool(getattr(block_pool, "enable_caching", False)):
        return None
    free_count = block_pool.get_num_free_blocks()
    if free_count <= 0:
        return None
    try:
        actual_blocks = _peek_free_blocks(block_pool, num_blocks)
    except Exception:
        logger.exception("[OmniKVEvictShadow] failed to inspect free queue")
        return None
    if not actual_blocks:
        return None
    max_scan = _env_int("OMNI_KV_EVICT_MAX_SCAN_BLOCKS", 4096)
    scan_limit = None if max_scan <= 0 else min(max_scan, free_count)
    try:
        scanned = _peek_free_blocks(block_pool, scan_limit)
    except Exception:
        logger.exception("[OmniKVEvictShadow] failed to scan free queue")
        return None
    now_wall = time.time()
    for block in scanned:
        _hydrate_candidate_metadata(block_pool, block)

    actual_cached = [
        _candidate_record(block, now_wall, block_pool)
        for block in actual_blocks
        if _is_cached_free_block(block)
    ]
    if not actual_cached:
        return None
    candidates = [
        _candidate_record(block, now_wall, block_pool)
        for block in scanned
        if _is_cached_free_block(block)
    ]
    if not candidates:
        return None
    policies = _csv_env("OMNI_KV_EVICT_SHADOW_POLICIES", _DEFAULT_POLICIES)
    hot_protect_sec = _env_float("OMNI_KV_EVICT_HOT_PROTECT_SEC", 10.0)
    shadow = {}
    for policy in policies:
        effective_policy = _adaptive_policy_for_capacity(block_pool, policy)
        shadow[policy] = _choose_shadow_victims(
            effective_policy,
            candidates,
            len(actual_cached),
            hot_protect_sec,
        )
    return {
        "event": "allocation_pressure_shadow",
        "schema_version": 2,
        "wall_time": now_wall,
        "num_requested_blocks": num_blocks,
        "num_free_blocks": free_count,
        "num_scanned_blocks": len(scanned),
        "num_candidate_blocks": len(candidates),
        "hot_protect_sec": hot_protect_sec,
        "capture_stats": dict(_shadow_stats(block_pool)),
        "actual_lru_victims": actual_cached,
        "shadow_victims": shadow,
    }


def _allocate_specific_free_blocks(block_pool: Any, blocks: list[Any]) -> list[Any]:
    queue = block_pool.free_block_queue
    for block in blocks:
        queue.remove(block)

    if block_pool.enable_caching:
        for block in blocks:
            block_pool._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if block_pool.metrics_collector:
                block_pool.metrics_collector.on_block_allocated(block)
    else:
        for block in blocks:
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if block_pool.metrics_collector:
                block_pool.metrics_collector.on_block_allocated(block)
    return blocks


def _build_indexed_enforced_allocation(
    block_pool: Any,
    num_blocks: int,
    policy: str,
    started_ns: int,
) -> tuple[list[Any], dict[str, Any]] | None:
    profile_enabled = _profile_timing_enabled(block_pool)
    policy = _adaptive_policy_for_capacity(block_pool, policy)
    if not _enforcement_free_ratio_gate_active(block_pool):
        return None
    phase_started_ns = _profile_phase_start(profile_enabled)
    try:
        actual_blocks = _peek_free_blocks(block_pool, num_blocks)
    except Exception:
        logger.exception("[OmniKVEvict] failed to inspect free queue head for indexed enforcement")
        _record_enforcement_attempt(block_pool, policy, num_blocks, "indexed_head_inspect_error", started_ns)
        return None
    _record_enforcement_phase_profile(
        block_pool,
        "indexed_head_inspect",
        phase_started_ns,
        item_count=len(actual_blocks),
    )
    if len(actual_blocks) != num_blocks:
        _record_enforcement_attempt(block_pool, policy, num_blocks, "indexed_not_enough_free_blocks", started_ns)
        return None

    phase_started_ns = _profile_phase_start(profile_enabled)
    actual_cached_blocks = [
        block for block in actual_blocks if _is_cached_free_block(block)
    ]
    cached_victim_count = len(actual_cached_blocks)
    _record_enforcement_phase_profile(
        block_pool,
        "indexed_head_filter",
        phase_started_ns,
        item_count=len(actual_cached_blocks),
    )
    phase_started_ns = _profile_phase_start(profile_enabled)
    if cached_victim_count == 0:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "no_cached_victims",
            started_ns,
            extra_fields=_free_queue_cached_profile(block_pool),
        )
        return None
    min_cached_victims = _enforcement_min_cached_victims_per_alloc(block_pool)
    if cached_victim_count < min_cached_victims:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "cached_victim_count_below_min",
            started_ns,
            cached_victim_count=cached_victim_count,
        )
        return None
    max_cached_victims = _enforcement_max_cached_victims_per_alloc(block_pool)
    partial_over_cap = _enforcement_partial_over_cap(block_pool)
    replacement_count = cached_victim_count
    if max_cached_victims > 0 and cached_victim_count > max_cached_victims:
        if partial_over_cap:
            replacement_count = max_cached_victims
        else:
            _record_enforcement_attempt(
                block_pool,
                policy,
                num_blocks,
                "cached_victim_count_over_cap",
                started_ns,
                cached_victim_count=cached_victim_count,
                extra_fields=_victim_summary_extra(block_pool, actual_cached_blocks),
            )
            return None
    if replacement_count <= 0:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "replacement_count_zero",
            started_ns,
            cached_victim_count=cached_victim_count,
        )
        return None

    pressure_reason, pressure_extra = _replacement_pressure_gate_reason(
        block_pool,
        cached_victim_count,
    )
    if pressure_reason is not None:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            pressure_reason,
            started_ns,
            cached_victim_count=cached_victim_count,
            extra_fields={
                **(pressure_extra or {}),
                "replacement_count": replacement_count,
                "initial_replacement_count": replacement_count,
                "partial_over_cap": partial_over_cap
                and replacement_count < cached_victim_count,
            },
        )
        return None

    current_gate_reason, current_gate_extra = _current_request_gate_reason(block_pool)
    if current_gate_reason is not None:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            current_gate_reason,
            started_ns,
            cached_victim_count=cached_victim_count,
            extra_fields={
                **(current_gate_extra or {}),
                "replacement_count": replacement_count,
                "initial_replacement_count": replacement_count,
                "partial_over_cap": partial_over_cap
                and replacement_count < cached_victim_count,
            },
        )
        return None

    throttle_reason, throttle_extra = _replacement_throttle_reason(block_pool)
    if throttle_reason is not None:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            throttle_reason,
            started_ns,
            cached_victim_count=cached_victim_count,
            extra_fields={
                **(throttle_extra or {}),
                "replacement_count": replacement_count,
                "initial_replacement_count": replacement_count,
                "partial_over_cap": partial_over_cap
                and replacement_count < cached_victim_count,
            },
        )
        return None

    _record_enforcement_phase_profile(
        block_pool,
        "indexed_precheck",
        phase_started_ns,
        item_count=len(actual_cached_blocks),
    )
    phase_started_ns = _profile_phase_start(profile_enabled)
    _prime_index_candidates(block_pool, policy)
    _record_enforcement_phase_profile(
        block_pool,
        "indexed_prime",
        phase_started_ns,
    )
    state = _candidate_index(block_pool, policy)
    heap = state.get("heap")
    if not heap:
        state["fallbacks"] = int(state.get("fallbacks", 0)) + 1
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "indexed_empty",
            started_ns,
            cached_victim_count=cached_victim_count,
        )
        return None

    versions = state.setdefault("versions", {})
    pop_multiplier = _env_int(
        "OMNI_KV_EVICT_ENFORCE_INDEX_POP_MULTIPLIER",
        _DEFAULT_INDEX_POP_MULTIPLIER,
    )
    max_attempts = max(64, replacement_count * max(1, pop_multiplier))

    selected_blocks: list[Any] = []
    selected_ids: set[int] = set()
    group_by_prefix = _index_group_by_prefix_enabled()
    group_selection_extra: dict[str, Any] = {}
    group_nonselected_blocks: list[Any] = []
    actual_cached_ids = {
        int(block_id)
        for block_id in (
            getattr(block, "block_id", None) for block in actual_cached_blocks
        )
        if block_id is not None
    }
    attempts = 0
    hot_protect_sec = _env_float("OMNI_KV_EVICT_HOT_PROTECT_SEC", 10.0)
    phase_started_ns = _profile_phase_start(profile_enabled)
    popped_group_entries: list[tuple[tuple[Any, ...], Any, Mapping[str, Any]]] = []
    attempt_limit = max_attempts
    if group_by_prefix:
        attempt_limit = min(
            max_attempts,
            max(64, replacement_count * _index_group_pop_multiplier()),
        )
    while heap and attempts < attempt_limit and (
        group_by_prefix or len(selected_blocks) < replacement_count
    ):
        _, version, block_id = heapq.heappop(heap)
        attempts += 1
        state["pops"] = int(state.get("pops", 0)) + 1
        if int(versions.get(block_id, 0)) != int(version):
            state["stale_pops"] = int(state.get("stale_pops", 0)) + 1
            continue
        if (
            block_id in selected_ids
            or block_id in actual_cached_ids
            or block_id < 0
            or block_id >= len(block_pool.blocks)
        ):
            state["invalid_pops"] = int(state.get("invalid_pops", 0)) + 1
            continue
        block = block_pool.blocks[block_id]
        if (
            not _is_cached_free_block(block)
            or not _is_linked_free_block(block)
        ):
            state["invalid_pops"] = int(state.get("invalid_pops", 0)) + 1
            _invalidate_indexed_block(block_pool, block_id)
            continue
        meta = _candidate_metadata(block_pool, block)
        if not meta:
            state["invalid_pops"] = int(state.get("invalid_pops", 0)) + 1
            _invalidate_indexed_block(block_pool, block_id)
            continue
        if not _candidate_within_selected_size_caps(block_pool, meta):
            state["selected_size_cap_pops"] = (
                int(state.get("selected_size_cap_pops", 0)) + 1
            )
            _invalidate_indexed_block(block_pool, block_id)
            continue
        if group_by_prefix:
            popped_group_entries.append(
                (
                    _score_metadata(policy, meta, hot_protect_sec),
                    block,
                    meta,
                )
            )
            continue
        selected_ids.add(block_id)
        selected_blocks.append(block)
    if group_by_prefix:
        selected_blocks, group_selection_extra = _select_grouped_index_candidates(
            popped_group_entries,
            replacement_count,
        )
        selected_ids = {
            int(block_id)
            for block_id in (
                getattr(block, "block_id", None) for block in selected_blocks
            )
            if _as_int(block_id) is not None
        }
        group_nonselected_blocks = [
            block
            for _, block, _ in popped_group_entries
            if (block_id := _as_int(getattr(block, "block_id", None))) is not None
            and block_id not in selected_ids
        ]
    _record_enforcement_phase_profile(
        block_pool,
        "indexed_pop_select",
        phase_started_ns,
        item_count=attempts,
    )

    def restore_blocks(blocks: Iterable[Any]) -> None:
        for block in blocks:
            if _is_cached_free_block(block) and _is_linked_free_block(block):
                _push_indexed_candidate(block_pool, block, policy)

    def restore_selected() -> None:
        restore_blocks(selected_blocks)

    if group_nonselected_blocks:
        restore_blocks(group_nonselected_blocks)

    initial_replacement_count = replacement_count
    partial_not_enough_candidates = False
    group_partial_on_short = (
        group_by_prefix and _index_group_partial_on_short()
    )
    if len(selected_blocks) != replacement_count:
        allow_partial_not_enough = (
            partial_over_cap
            and _enforcement_index_partial_on_not_enough_candidates()
        ) or group_partial_on_short
        if allow_partial_not_enough and selected_blocks:
            partial_not_enough_candidates = True
            state["partial_not_enough_candidates"] = (
                int(state.get("partial_not_enough_candidates", 0)) + 1
            )
            replacement_count = len(selected_blocks)
        else:
            state["fallbacks"] = int(state.get("fallbacks", 0)) + 1
            restore_selected()
            _record_enforcement_attempt(
                block_pool,
                policy,
                num_blocks,
                "indexed_not_enough_candidates",
                started_ns,
                cached_victim_count=cached_victim_count,
                candidate_count=len(selected_blocks),
                extra_fields={
                    **(_victim_summary_extra(
                        block_pool,
                        actual_cached_blocks,
                        selected_blocks,
                    ) or {}),
                    **_indexed_candidate_detail_extra(
                        block_pool,
                        policy,
                        state,
                        selected_blocks,
                        actual_cached_blocks,
                        _env_float("OMNI_KV_EVICT_HOT_PROTECT_SEC", 10.0),
                        source="indexed_not_enough_candidates",
                    ),
                    **group_selection_extra,
                    "replacement_count": len(selected_blocks),
                    "initial_replacement_count": initial_replacement_count,
                    "partial_not_enough_candidates": False,
                    "index_partial_on_not_enough_candidates": (
                        _enforcement_index_partial_on_not_enough_candidates()
                    ),
                    "group_partial_on_short": group_partial_on_short,
                },
            )
            return None

    phase_started_ns = _profile_phase_start(profile_enabled)
    selected_scored_blocks = [
        (
            _score_metadata(
                policy,
                _candidate_metadata(block_pool, block),
                hot_protect_sec,
            ),
            block,
        )
        for block in selected_blocks
    ]
    scored_actual_blocks = [
        (
            _score_metadata(
                policy,
                _candidate_metadata(block_pool, block),
                hot_protect_sec,
            ),
            block,
        )
        for block in actual_cached_blocks
    ]
    scored_actual_blocks.sort(key=lambda item: item[0])
    _record_enforcement_phase_profile(
        block_pool,
        "indexed_score",
        phase_started_ns,
        item_count=len(selected_blocks) + len(actual_cached_blocks),
    )
    protect_actual_max_gap_sec = _enforcement_protect_actual_max_gap_sec(block_pool)
    if protect_actual_max_gap_sec > 0.0:
        protect_count = 0
        for _, block in scored_actual_blocks[:replacement_count]:
            if not _actual_victim_protect_eligible(
                block_pool,
                block,
                protect_actual_max_gap_sec,
            ):
                break
            protect_count += 1
        if protect_count <= 0:
            state["fallbacks"] = int(state.get("fallbacks", 0)) + 1
            restore_selected()
            _record_enforcement_attempt(
                block_pool,
                policy,
                num_blocks,
                "actual_victim_gap_above_max",
                started_ns,
                cached_victim_count=cached_victim_count,
                candidate_count=len(selected_blocks),
                extra_fields={
                    **(_victim_summary_extra(
                        block_pool,
                        [item[1] for item in scored_actual_blocks[:replacement_count]],
                        selected_blocks,
                    ) or {}),
                    "protect_actual_max_gap_sec": protect_actual_max_gap_sec,
                },
            )
            return None
        if protect_count < replacement_count:
            selected_scored_blocks.sort(key=lambda item: item[0], reverse=True)
            dropped_blocks = [
                item[1] for item in selected_scored_blocks[protect_count:]
            ]
            restore_blocks(dropped_blocks)
            selected_scored_blocks = selected_scored_blocks[:protect_count]
            selected_blocks = [item[1] for item in selected_scored_blocks]
            replacement_count = protect_count
    min_gap_margin_sec, gap_margin_extra = _effective_min_gap_margin_sec(
        block_pool
    )
    gap_margin_terminal_ok = _enforcement_gap_margin_terminal_ok(block_pool)
    if min_gap_margin_sec > 0.0:
        selected_scored_blocks.sort(key=lambda item: item[0], reverse=True)
        margin_count = 0
        for selected_item, actual_item in zip(
            selected_scored_blocks,
            scored_actual_blocks,
        ):
            if _gap_margin_replacement_eligible(
                block_pool,
                selected_item[1],
                actual_item[1],
                min_gap_margin_sec,
                gap_margin_terminal_ok,
            ):
                margin_count += 1
                continue
            break
        if margin_count <= 0:
            state["fallbacks"] = int(state.get("fallbacks", 0)) + 1
            restore_selected()
            _record_enforcement_attempt(
                block_pool,
                policy,
                num_blocks,
                "gap_margin_below_min",
                started_ns,
                cached_victim_count=cached_victim_count,
                candidate_count=len(selected_blocks),
                extra_fields={
                    **(_victim_summary_extra(
                        block_pool,
                        [
                            item[1]
                            for item in scored_actual_blocks[:initial_replacement_count]
                        ],
                        selected_blocks,
                    ) or {}),
                    "min_gap_margin_sec": min_gap_margin_sec,
                    "gap_margin_terminal_ok": gap_margin_terminal_ok,
                    **gap_margin_extra,
                    "gap_margin_pair_details": _gap_margin_pair_details(
                        block_pool,
                        [item[1] for item in selected_scored_blocks],
                        [
                            item[1]
                            for item in scored_actual_blocks[:initial_replacement_count]
                        ],
                        min_gap_margin_sec,
                        gap_margin_terminal_ok,
                    ),
                    "selected_score_tuples": _score_tuple_details(
                        [item[0] for item in selected_scored_blocks]
                    ),
                    "actual_score_tuples": _score_tuple_details(
                        [
                            item[0]
                            for item in scored_actual_blocks[:initial_replacement_count]
                        ]
                    ),
                },
            )
            return None
        dropped_blocks = [
            item[1] for item in selected_scored_blocks[margin_count:]
        ]
        restore_blocks(dropped_blocks)
        selected_scored_blocks = selected_scored_blocks[:margin_count]
        selected_blocks = [item[1] for item in selected_scored_blocks]
        replacement_count = margin_count
    dynamic_replacement = _enforcement_dynamic_replacement(block_pool)
    if dynamic_replacement:
        selected_scored_blocks.sort(key=lambda item: item[0], reverse=True)
        dynamic_count = 0
        for selected_item, actual_item in zip(
            selected_scored_blocks,
            scored_actual_blocks,
        ):
            if selected_item[0] > actual_item[0]:
                dynamic_count += 1
                continue
            break
        if dynamic_count <= 0:
            state["fallbacks"] = int(state.get("fallbacks", 0)) + 1
            restore_selected()
            _record_enforcement_attempt(
                block_pool,
                policy,
                num_blocks,
                "no_policy_score_improvement",
                started_ns,
                cached_victim_count=cached_victim_count,
                candidate_count=len(selected_blocks),
                extra_fields={
                    **(_victim_summary_extra(
                        block_pool,
                        [
                            item[1]
                            for item in scored_actual_blocks[:initial_replacement_count]
                        ],
                        selected_blocks,
                    ) or {}),
                    **_indexed_candidate_detail_extra(
                        block_pool,
                        policy,
                        state,
                        selected_blocks,
                        [
                            item[1]
                            for item in scored_actual_blocks[:initial_replacement_count]
                        ],
                        hot_protect_sec,
                        source="indexed_dynamic_no_policy_score_improvement",
                    ),
                    "selected_score_tuples": _score_tuple_details(
                        [item[0] for item in selected_scored_blocks]
                    ),
                    "actual_score_tuples": _score_tuple_details(
                        [
                            item[0]
                            for item in scored_actual_blocks[:initial_replacement_count]
                        ]
                    ),
                },
            )
            return None
        dropped_blocks = [
            item[1] for item in selected_scored_blocks[dynamic_count:]
        ]
        restore_blocks(dropped_blocks)
        selected_scored_blocks = selected_scored_blocks[:dynamic_count]
        selected_blocks = [item[1] for item in selected_scored_blocks]
        replacement_count = dynamic_count
    selected_scores = [item[0] for item in selected_scored_blocks]
    actual_blocks_to_preserve = [
        item[1] for item in scored_actual_blocks[:replacement_count]
    ]
    actual_cached_blocks_to_allocate = [
        item[1] for item in scored_actual_blocks[replacement_count:]
    ]
    actual_scores = [item[0] for item in scored_actual_blocks[:replacement_count]]
    require_selected_terminal = _enforcement_require_selected_terminal(block_pool)
    if require_selected_terminal and not _selected_blocks_all_terminal(
        block_pool,
        selected_blocks,
    ):
        state["fallbacks"] = int(state.get("fallbacks", 0)) + 1
        restore_selected()
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "selected_nonterminal",
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(selected_blocks),
            extra_fields={
                **(_victim_summary_extra(
                    block_pool,
                    actual_blocks_to_preserve,
                    selected_blocks,
                ) or {}),
                "selected_score_tuples": _score_tuple_details(selected_scores),
                "actual_score_tuples": _score_tuple_details(actual_scores),
                "require_selected_terminal": require_selected_terminal,
            },
        )
        return None
    selected_size_cap_reason, selected_size_cap_extra = _selected_size_cap_reason(
        block_pool,
        selected_blocks,
    )
    if selected_size_cap_reason is not None:
        state["fallbacks"] = int(state.get("fallbacks", 0)) + 1
        restore_selected()
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            selected_size_cap_reason,
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(selected_blocks),
            extra_fields={
                **(_victim_summary_extra(
                    block_pool,
                    actual_blocks_to_preserve,
                    selected_blocks,
                ) or {}),
                "selected_score_tuples": _score_tuple_details(selected_scores),
                "actual_score_tuples": _score_tuple_details(actual_scores),
                **selected_size_cap_extra,
            },
        )
        return None
    if (
        not dynamic_replacement
        and not _score_lists_show_improvement(selected_scores, actual_scores)
    ):
        state["fallbacks"] = int(state.get("fallbacks", 0)) + 1
        restore_selected()
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "no_policy_score_improvement",
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(selected_blocks),
            extra_fields={
                **(_victim_summary_extra(
                    block_pool,
                    actual_blocks_to_preserve,
                    selected_blocks,
                ) or {}),
                **_indexed_candidate_detail_extra(
                    block_pool,
                    policy,
                    state,
                    selected_blocks,
                    actual_blocks_to_preserve,
                    hot_protect_sec,
                    source="indexed_no_policy_score_improvement",
                ),
                "selected_score_tuples": _score_tuple_details(selected_scores),
                "actual_score_tuples": _score_tuple_details(actual_scores),
            },
        )
        return None

    guard_reason, guard_extra = _value_score_margin_guard_reason(
        block_pool,
        selected_blocks,
        actual_blocks_to_preserve,
    )
    if guard_reason is not None:
        state["fallbacks"] = int(state.get("fallbacks", 0)) + 1
        restore_selected()
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            guard_reason,
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(selected_blocks),
            extra_fields={
                **(_victim_summary_extra(
                    block_pool,
                    actual_blocks_to_preserve,
                    selected_blocks,
                ) or {}),
                "selected_score_tuples": _score_tuple_details(selected_scores),
                "actual_score_tuples": _score_tuple_details(actual_scores),
                **(guard_extra or {}),
            },
        )
        return None

    guard_reason, guard_extra = _actual_near_short_guard_reason(
        block_pool,
        actual_blocks_to_preserve,
        selected_blocks,
    )
    if guard_reason is not None:
        state["fallbacks"] = int(state.get("fallbacks", 0)) + 1
        restore_selected()
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            guard_reason,
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(selected_blocks),
            extra_fields={
                **(_victim_summary_extra(
                    block_pool,
                    actual_blocks_to_preserve,
                    selected_blocks,
                ) or {}),
                "selected_score_tuples": _score_tuple_details(selected_scores),
                "actual_score_tuples": _score_tuple_details(actual_scores),
                **(guard_extra or {}),
            },
        )
        return None

    guard_reason, guard_extra = _low_value_actual_density_guard_reason(
        block_pool,
        actual_blocks_to_preserve,
        selected_blocks,
    )
    if guard_reason is not None:
        state["fallbacks"] = int(state.get("fallbacks", 0)) + 1
        restore_selected()
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            guard_reason,
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(selected_blocks),
            extra_fields={
                **(_victim_summary_extra(
                    block_pool,
                    actual_blocks_to_preserve,
                    selected_blocks,
                ) or {}),
                "selected_score_tuples": _score_tuple_details(selected_scores),
                "actual_score_tuples": _score_tuple_details(actual_scores),
                **(guard_extra or {}),
            },
        )
        return None

    guard_reason, guard_extra = _prefix_loss_delta_guard_reason(
        block_pool,
        actual_blocks_to_preserve,
        selected_blocks,
    )
    if guard_reason is not None:
        state["fallbacks"] = int(state.get("fallbacks", 0)) + 1
        restore_selected()
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            guard_reason,
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(selected_blocks),
            extra_fields={
                **(_victim_summary_extra(
                    block_pool,
                    actual_blocks_to_preserve,
                    selected_blocks,
                ) or {}),
                "selected_score_tuples": _score_tuple_details(selected_scores),
                "actual_score_tuples": _score_tuple_details(actual_scores),
                **(guard_extra or {}),
            },
        )
        return None

    min_meta_ratio = _env_float("OMNI_KV_EVICT_ENFORCE_MIN_METADATA_RATIO", 0.75)
    metadata_ratio = 1.0
    if metadata_ratio < min_meta_ratio:
        state["fallbacks"] = int(state.get("fallbacks", 0)) + 1
        restore_selected()
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "metadata_ratio_below_min",
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(selected_blocks),
            metadata_ratio=metadata_ratio,
        )
        return None

    noncached_head_blocks = [
        block for block in actual_blocks if not _is_cached_free_block(block)
    ]
    allocation = (
        selected_blocks
        + actual_cached_blocks_to_allocate
        + noncached_head_blocks
    )
    if len(allocation) != num_blocks:
        state["fallbacks"] = int(state.get("fallbacks", 0)) + 1
        restore_selected()
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "allocation_size_mismatch",
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(selected_blocks),
            metadata_ratio=metadata_ratio,
        )
        return None

    phase_started_ns = _profile_phase_start(profile_enabled)
    allocated = _allocate_specific_free_blocks(block_pool, allocation)
    _record_enforcement_phase_profile(
        block_pool,
        "indexed_allocate",
        phase_started_ns,
        item_count=len(allocation),
    )
    state["selected"] = int(state.get("selected", 0)) + len(selected_blocks)
    phase_started_ns = _profile_phase_start(profile_enabled)
    detail_needed = _enforcement_log_enabled() or shadow_enabled()
    selected_block_ids = (
        [getattr(block, "block_id", None) for block in selected_blocks]
        if detail_needed
        else None
    )
    selected_records = (
        [
            _candidate_record(block, time.time(), block_pool)
            for block in selected_blocks
        ]
        if detail_needed
        else []
    )
    applied_extra_fields = None
    if _enforcement_log_enabled():
        applied_extra_fields = {
            **(_victim_summary_extra(
                block_pool,
                actual_blocks_to_preserve,
                selected_blocks,
            ) or {}),
            **_indexed_candidate_detail_extra(
                block_pool,
                policy,
                state,
                selected_blocks,
                actual_blocks_to_preserve,
                hot_protect_sec,
                source="indexed_applied",
            ),
            "selected_score_tuples": _score_tuple_details(selected_scores),
            "actual_score_tuples": _score_tuple_details(actual_scores),
            "partial_over_cap": partial_over_cap
            and replacement_count < cached_victim_count,
            "partial_not_enough_candidates": partial_not_enough_candidates,
            "index_partial_on_not_enough_candidates": (
                _enforcement_index_partial_on_not_enough_candidates()
            ),
            "group_partial_on_short": group_partial_on_short,
            "replacement_count": replacement_count,
            "initial_replacement_count": initial_replacement_count,
            "dynamic_replacement": dynamic_replacement,
            "require_selected_terminal": require_selected_terminal,
            **selected_size_cap_extra,
            "protect_actual_max_gap_sec": protect_actual_max_gap_sec,
            "min_gap_margin_sec": min_gap_margin_sec,
            "gap_margin_terminal_ok": gap_margin_terminal_ok,
            "gap_margin_pair_details": (
                _gap_margin_pair_details(
                    block_pool,
                    selected_blocks,
                    actual_blocks_to_preserve,
                    min_gap_margin_sec,
                    gap_margin_terminal_ok,
                )
                if min_gap_margin_sec > 0.0
                else None
            ),
            "actual_lru_cached_victim_count": cached_victim_count,
            "actual_cached_blocks_allocated": len(actual_cached_blocks_to_allocate),
            "group_by_prefix": group_by_prefix,
            **group_selection_extra,
        }
    _record_enforcement_attempt(
        block_pool,
        policy,
        num_blocks,
        "applied",
        started_ns,
        cached_victim_count=cached_victim_count,
        candidate_count=len(selected_blocks),
        metadata_ratio=metadata_ratio,
        selected_block_ids=selected_block_ids,
        extra_fields=applied_extra_fields,
    )
    _record_enforcement_phase_profile(
        block_pool,
        "indexed_record",
        phase_started_ns,
    )
    info = {
        "policy": policy,
        "indexed": True,
        "metadata_ratio": metadata_ratio,
        "min_cached_victims_per_alloc": min_cached_victims,
        "max_cached_victims_per_alloc": max_cached_victims,
        "partial_over_cap": partial_over_cap
        and replacement_count < cached_victim_count,
        "partial_not_enough_candidates": partial_not_enough_candidates,
        "index_partial_on_not_enough_candidates": (
            _enforcement_index_partial_on_not_enough_candidates()
        ),
        "group_partial_on_short": group_partial_on_short,
        "replacement_count": replacement_count,
        "initial_replacement_count": initial_replacement_count,
        "dynamic_replacement": dynamic_replacement,
        "require_selected_terminal": require_selected_terminal,
        **selected_size_cap_extra,
        "protect_actual_max_gap_sec": protect_actual_max_gap_sec,
        "min_gap_margin_sec": min_gap_margin_sec,
        "gap_margin_terminal_ok": gap_margin_terminal_ok,
        "actual_cached_blocks_allocated": len(actual_cached_blocks_to_allocate),
        "candidate_free_ratio": 0.0,
        "max_candidate_free_ratio": _enforcement_max_candidate_free_ratio(),
        "actual_lru_cached_victim_count": cached_victim_count,
        "candidate_count": len(heap),
        "group_by_prefix": group_by_prefix,
        **group_selection_extra,
        "index_pop_attempts": attempts,
        "index_stats": {
            key: state.get(key, 0)
            for key in (
                "pushes",
                "pops",
                "stale_pops",
                "invalid_pops",
                "selected_size_cap_skips",
                "selected_size_cap_pops",
                "partial_not_enough_candidates",
                "selected",
                "fallbacks",
            )
        },
        "selected_victims": selected_records,
        "selected_block_ids": selected_block_ids or [],
    }
    return allocated, info


def _build_enforced_allocation(
    block_pool: Any,
    num_blocks: int,
    policy: str,
) -> tuple[list[Any], dict[str, Any]] | None:
    started_ns = time.perf_counter_ns()
    profile_enabled = _profile_timing_enabled(block_pool)
    policy = _adaptive_policy_for_capacity(block_pool, policy)
    if not bool(getattr(block_pool, "enable_caching", False)):
        _record_enforcement_attempt(block_pool, policy, num_blocks, "caching_disabled", started_ns)
        return None
    free_count = block_pool.get_num_free_blocks()
    if num_blocks > free_count:
        _record_enforcement_attempt(block_pool, policy, num_blocks, "not_enough_free_blocks", started_ns)
        raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
    if _current_gate_early_bypass():
        current_gate_reason, _ = _current_request_gate_reason(block_pool)
        if current_gate_reason is not None:
            return None
    if not shadow_enabled() and _enforcement_index_enabled():
        indexed = _build_indexed_enforced_allocation(
            block_pool,
            num_blocks,
            policy,
            started_ns,
        )
        if indexed is not None:
            return indexed
        if not _env_bool("OMNI_KV_EVICT_ENFORCE_INDEX_FALLBACK_SCAN", False):
            return None
    phase_started_ns = _profile_phase_start(profile_enabled)
    try:
        actual_blocks = _peek_free_blocks(block_pool, num_blocks)
    except Exception:
        logger.exception("[OmniKVEvict] failed to inspect free queue head for enforcement")
        _record_enforcement_attempt(block_pool, policy, num_blocks, "free_queue_head_inspect_error", started_ns)
        return None
    _record_enforcement_phase_profile(
        block_pool,
        "scan_head_inspect",
        phase_started_ns,
        item_count=len(actual_blocks),
    )
    if not actual_blocks:
        _record_enforcement_attempt(block_pool, policy, num_blocks, "no_free_blocks", started_ns)
        return None

    phase_started_ns = _profile_phase_start(profile_enabled)
    actual_cached_blocks = [
        block for block in actual_blocks if _is_cached_free_block(block)
    ]
    cached_victim_count = len(actual_cached_blocks)
    _record_enforcement_phase_profile(
        block_pool,
        "scan_head_filter",
        phase_started_ns,
        item_count=len(actual_cached_blocks),
    )
    phase_started_ns = _profile_phase_start(profile_enabled)
    if cached_victim_count == 0:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "no_cached_victims",
            started_ns,
            extra_fields=_free_queue_cached_profile(block_pool),
        )
        return None
    min_cached_victims = _enforcement_min_cached_victims_per_alloc(block_pool)
    if cached_victim_count < min_cached_victims:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "cached_victim_count_below_min",
            started_ns,
            cached_victim_count=cached_victim_count,
        )
        return None
    max_cached_victims = _enforcement_max_cached_victims_per_alloc(block_pool)
    if max_cached_victims > 0 and cached_victim_count > max_cached_victims:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "cached_victim_count_over_cap",
            started_ns,
            cached_victim_count=cached_victim_count,
            extra_fields=_victim_summary_extra(block_pool, actual_cached_blocks),
        )
        return None

    pressure_reason, pressure_extra = _replacement_pressure_gate_reason(
        block_pool,
        cached_victim_count,
    )
    if pressure_reason is not None:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            pressure_reason,
            started_ns,
            cached_victim_count=cached_victim_count,
            extra_fields=pressure_extra,
        )
        return None

    current_gate_reason, current_gate_extra = _current_request_gate_reason(block_pool)
    if current_gate_reason is not None:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            current_gate_reason,
            started_ns,
            cached_victim_count=cached_victim_count,
            extra_fields=current_gate_extra,
        )
        return None

    throttle_reason, throttle_extra = _replacement_throttle_reason(block_pool)
    if throttle_reason is not None:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            throttle_reason,
            started_ns,
            cached_victim_count=cached_victim_count,
            extra_fields=throttle_extra,
        )
        return None

    _record_enforcement_phase_profile(
        block_pool,
        "scan_precheck",
        phase_started_ns,
        item_count=len(actual_cached_blocks),
    )
    max_scan = _env_int("OMNI_KV_EVICT_MAX_SCAN_BLOCKS", 4096)
    scan_limit = None if max_scan <= 0 else min(max_scan, free_count)
    phase_started_ns = _profile_phase_start(profile_enabled)
    try:
        scanned = _peek_free_blocks(block_pool, scan_limit)
    except Exception:
        logger.exception("[OmniKVEvict] failed to scan free queue for enforcement")
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "free_queue_scan_error",
            started_ns,
            cached_victim_count=cached_victim_count,
        )
        return None
    _record_enforcement_phase_profile(
        block_pool,
        "scan_peek",
        phase_started_ns,
        item_count=len(scanned),
    )
    if not shadow_enabled():
        hot_protect_sec = _env_float("OMNI_KV_EVICT_HOT_PROTECT_SEC", 10.0)
        candidate_entries: list[tuple[tuple[Any, ...], int, Any, bool]] = []
        phase_started_ns = _profile_phase_start(profile_enabled)
        for block in scanned:
            if not _is_cached_free_block(block):
                continue
            block_id = getattr(block, "block_id", None)
            if block_id is None:
                continue
            meta = _candidate_metadata(block_pool, block)
            candidate_entries.append(
                (
                    _score_metadata(policy, meta, hot_protect_sec),
                    int(block_id),
                    block,
                    bool(meta),
                )
            )
        _record_enforcement_phase_profile(
            block_pool,
            "scan_candidate_build",
            phase_started_ns,
            item_count=len(candidate_entries),
        )
        if len(candidate_entries) < cached_victim_count:
            _record_enforcement_attempt(
                block_pool,
                policy,
                num_blocks,
                "not_enough_candidates",
                started_ns,
                cached_victim_count=cached_victim_count,
                candidate_count=len(candidate_entries),
            )
            return None

        max_candidate_free_ratio = _enforcement_max_candidate_free_ratio()
        candidate_free_ratio = len(candidate_entries) / max(1, free_count)
        if (
            max_candidate_free_ratio > 0.0
            and candidate_free_ratio > max_candidate_free_ratio
        ):
            _record_enforcement_attempt(
                block_pool,
                policy,
                num_blocks,
                "candidate_free_ratio_over_cap",
                started_ns,
                cached_victim_count=cached_victim_count,
                candidate_count=len(candidate_entries),
                candidate_free_ratio=candidate_free_ratio,
            )
            return None

        phase_started_ns = _profile_phase_start(profile_enabled)
        selected_entries = heapq.nlargest(
            cached_victim_count,
            candidate_entries,
            key=lambda item: item[0],
        )
        actual_scores = [
            _score_metadata(
                policy,
                _candidate_metadata(block_pool, block),
                hot_protect_sec,
            )
            for block in actual_cached_blocks
        ]
        selected_scores = [item[0] for item in selected_entries]
        _record_enforcement_phase_profile(
            block_pool,
            "scan_select_score",
            phase_started_ns,
            item_count=len(candidate_entries) + len(actual_cached_blocks),
        )
        if not _score_lists_show_improvement(selected_scores, actual_scores):
            _record_enforcement_attempt(
                block_pool,
                policy,
                num_blocks,
                "no_policy_score_improvement",
                started_ns,
                cached_victim_count=cached_victim_count,
                candidate_count=len(candidate_entries),
                candidate_free_ratio=candidate_free_ratio,
                extra_fields={
                    **(_victim_summary_extra(
                        block_pool,
                        actual_cached_blocks,
                        [item[2] for item in selected_entries],
                    ) or {}),
                    **_candidate_detail_from_scored_blocks(
                        block_pool,
                        policy,
                        [(item[0], item[2]) for item in candidate_entries],
                        hot_protect_sec,
                        source="scan_no_policy_score_improvement",
                        total_candidates=len(candidate_entries),
                    ),
                },
            )
            return None
        min_meta_ratio = _env_float("OMNI_KV_EVICT_ENFORCE_MIN_METADATA_RATIO", 0.75)
        metadata_ratio = sum(1 for item in selected_entries if item[3]) / max(
            1, len(selected_entries)
        )
        if metadata_ratio < min_meta_ratio:
            _record_enforcement_attempt(
                block_pool,
                policy,
                num_blocks,
                "metadata_ratio_below_min",
                started_ns,
                cached_victim_count=cached_victim_count,
                candidate_count=len(candidate_entries),
                metadata_ratio=metadata_ratio,
                candidate_free_ratio=candidate_free_ratio,
            )
            return None

        selected_blocks = [item[2] for item in selected_entries]
        guard_reason, guard_extra = _value_score_margin_guard_reason(
            block_pool,
            selected_blocks,
            actual_cached_blocks,
        )
        if guard_reason is not None:
            _record_enforcement_attempt(
                block_pool,
                policy,
                num_blocks,
                guard_reason,
                started_ns,
                cached_victim_count=cached_victim_count,
                candidate_count=len(candidate_entries),
                metadata_ratio=metadata_ratio,
                candidate_free_ratio=candidate_free_ratio,
                extra_fields={
                    **(_victim_summary_extra(
                        block_pool,
                        actual_cached_blocks,
                        selected_blocks,
                    ) or {}),
                    "selected_score_tuples": _score_tuple_details(selected_scores),
                    "actual_score_tuples": _score_tuple_details(actual_scores),
                    **(guard_extra or {}),
                },
            )
            return None

        guard_reason, guard_extra = _actual_near_short_guard_reason(
            block_pool,
            actual_cached_blocks,
            selected_blocks,
        )
        if guard_reason is not None:
            _record_enforcement_attempt(
                block_pool,
                policy,
                num_blocks,
                guard_reason,
                started_ns,
                cached_victim_count=cached_victim_count,
                candidate_count=len(candidate_entries),
                metadata_ratio=metadata_ratio,
                candidate_free_ratio=candidate_free_ratio,
                extra_fields={
                    **(_victim_summary_extra(
                        block_pool,
                        actual_cached_blocks,
                        selected_blocks,
                    ) or {}),
                    "selected_score_tuples": _score_tuple_details(selected_scores),
                    "actual_score_tuples": _score_tuple_details(actual_scores),
                    **(guard_extra or {}),
                },
            )
            return None

        guard_reason, guard_extra = _low_value_actual_density_guard_reason(
            block_pool,
            actual_cached_blocks,
            selected_blocks,
        )
        if guard_reason is not None:
            _record_enforcement_attempt(
                block_pool,
                policy,
                num_blocks,
                guard_reason,
                started_ns,
                cached_victim_count=cached_victim_count,
                candidate_count=len(candidate_entries),
                metadata_ratio=metadata_ratio,
                candidate_free_ratio=candidate_free_ratio,
                extra_fields={
                    **(_victim_summary_extra(
                        block_pool,
                        actual_cached_blocks,
                        selected_blocks,
                    ) or {}),
                    "selected_score_tuples": _score_tuple_details(selected_scores),
                    "actual_score_tuples": _score_tuple_details(actual_scores),
                    **(guard_extra or {}),
                },
            )
            return None
        guard_reason, guard_extra = _prefix_loss_delta_guard_reason(
            block_pool,
            actual_cached_blocks,
            selected_blocks,
        )
        if guard_reason is not None:
            _record_enforcement_attempt(
                block_pool,
                policy,
                num_blocks,
                guard_reason,
                started_ns,
                cached_victim_count=cached_victim_count,
                candidate_count=len(candidate_entries),
                metadata_ratio=metadata_ratio,
                candidate_free_ratio=candidate_free_ratio,
                extra_fields={
                    **(_victim_summary_extra(
                        block_pool,
                        actual_cached_blocks,
                        selected_blocks,
                    ) or {}),
                    "selected_score_tuples": _score_tuple_details(selected_scores),
                    "actual_score_tuples": _score_tuple_details(actual_scores),
                    **(guard_extra or {}),
                },
            )
            return None
        noncached_head_blocks = [
            block for block in actual_blocks if not _is_cached_free_block(block)
        ]
        allocation = selected_blocks + noncached_head_blocks
        if len(allocation) != num_blocks:
            _record_enforcement_attempt(
                block_pool,
                policy,
                num_blocks,
                "allocation_size_mismatch",
                started_ns,
                cached_victim_count=cached_victim_count,
                candidate_count=len(candidate_entries),
                metadata_ratio=metadata_ratio,
                candidate_free_ratio=candidate_free_ratio,
            )
            return None

        phase_started_ns = _profile_phase_start(profile_enabled)
        allocated = _allocate_specific_free_blocks(block_pool, allocation)
        _record_enforcement_phase_profile(
            block_pool,
            "scan_allocate",
            phase_started_ns,
            item_count=len(allocation),
        )
        selected_block_ids = [getattr(block, "block_id", None) for block in selected_blocks]
        phase_started_ns = _profile_phase_start(profile_enabled)
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "applied",
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(candidate_entries),
            metadata_ratio=metadata_ratio,
            candidate_free_ratio=candidate_free_ratio,
            selected_block_ids=selected_block_ids,
            extra_fields={
                **(_victim_summary_extra(
                    block_pool,
                    actual_cached_blocks,
                    selected_blocks,
                ) or {}),
                **_candidate_detail_from_scored_blocks(
                    block_pool,
                    policy,
                    [(item[0], item[2]) for item in candidate_entries],
                    hot_protect_sec,
                    source="scan_applied",
                    total_candidates=len(candidate_entries),
                ),
            },
        )
        _record_enforcement_phase_profile(
            block_pool,
            "scan_record",
            phase_started_ns,
        )
        info = {
            "policy": policy,
            "metadata_ratio": metadata_ratio,
            "max_cached_victims_per_alloc": max_cached_victims,
            "candidate_free_ratio": candidate_free_ratio,
            "max_candidate_free_ratio": max_candidate_free_ratio,
            "actual_lru_cached_victim_count": cached_victim_count,
            "candidate_count": len(candidate_entries),
            "selected_block_ids": selected_block_ids,
        }
        return allocated, info

    now_wall = time.time()
    id_to_block: dict[int, Any] = {}
    candidates: list[dict[str, Any]] = []
    for block in scanned:
        if not _is_cached_free_block(block):
            continue
        _hydrate_candidate_metadata(block_pool, block)
        record = _candidate_record(block, now_wall, block_pool)
        block_id = _as_int(record.get("block_id"))
        if block_id is None:
            continue
        id_to_block[block_id] = block
        candidates.append(record)
    if len(candidates) < cached_victim_count:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "not_enough_candidates",
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(candidates),
        )
        return None

    max_candidate_free_ratio = _enforcement_max_candidate_free_ratio()
    candidate_free_ratio = len(candidates) / max(1, free_count)
    if (
        max_candidate_free_ratio > 0.0
        and candidate_free_ratio > max_candidate_free_ratio
    ):
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "candidate_free_ratio_over_cap",
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(candidates),
            candidate_free_ratio=candidate_free_ratio,
        )
        return None

    selected_records = _choose_policy_victim_records(
        policy,
        candidates,
        cached_victim_count,
    )
    actual_records = [
        _candidate_record(block, now_wall, block_pool)
        for block in actual_cached_blocks
    ]
    hot_protect_sec = _env_float("OMNI_KV_EVICT_HOT_PROTECT_SEC", 10.0)
    selected_scores = [
        _score(policy, record, hot_protect_sec)
        for record in selected_records
    ]
    actual_scores = [
        _score(policy, record, hot_protect_sec)
        for record in actual_records
    ]
    if not _score_lists_show_improvement(selected_scores, actual_scores):
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "no_policy_score_improvement",
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(candidates),
            candidate_free_ratio=candidate_free_ratio,
            extra_fields={
                **(_victim_summary_extra(
                    block_pool,
                    actual_cached_blocks,
                    [
                        id_to_block.get(_as_int(record.get("block_id")))
                        for record in selected_records
                        if id_to_block.get(_as_int(record.get("block_id"))) is not None
                    ],
                ) or {}),
                **_candidate_detail_from_records(
                    policy,
                    candidates,
                    hot_protect_sec,
                    source="shadow_scan_no_policy_score_improvement",
                ),
            },
        )
        return None
    selected_blocks: list[Any] = []
    selected_ids: set[int] = set()
    for record in selected_records:
        block_id = _as_int(record.get("block_id"))
        if block_id is None or block_id in selected_ids:
            continue
        block = id_to_block.get(block_id)
        if block is None:
            continue
        selected_ids.add(block_id)
        selected_blocks.append(block)
    if len(selected_blocks) != cached_victim_count:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "selected_block_count_mismatch",
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(candidates),
            candidate_free_ratio=candidate_free_ratio,
        )
        return None

    min_meta_ratio = _env_float("OMNI_KV_EVICT_ENFORCE_MIN_METADATA_RATIO", 0.75)
    metadata_ratio = sum(
        1 for record in selected_records if record.get("metadata_present")
    ) / max(1, len(selected_records))
    if metadata_ratio < min_meta_ratio:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "metadata_ratio_below_min",
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(candidates),
            metadata_ratio=metadata_ratio,
            candidate_free_ratio=candidate_free_ratio,
        )
        return None

    guard_reason, guard_extra = _value_score_margin_guard_reason(
        block_pool,
        selected_blocks,
        actual_cached_blocks,
    )
    if guard_reason is not None:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            guard_reason,
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(candidates),
            metadata_ratio=metadata_ratio,
            candidate_free_ratio=candidate_free_ratio,
            extra_fields={
                **(_victim_summary_extra(
                    block_pool,
                    actual_cached_blocks,
                    selected_blocks,
                ) or {}),
                "selected_score_tuples": _score_tuple_details(selected_scores),
                "actual_score_tuples": _score_tuple_details(actual_scores),
                **(guard_extra or {}),
            },
        )
        return None

    guard_reason, guard_extra = _actual_near_short_guard_reason(
        block_pool,
        actual_cached_blocks,
        selected_blocks,
    )
    if guard_reason is not None:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            guard_reason,
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(candidates),
            metadata_ratio=metadata_ratio,
            candidate_free_ratio=candidate_free_ratio,
            extra_fields={
                **(_victim_summary_extra(
                    block_pool,
                    actual_cached_blocks,
                    selected_blocks,
                ) or {}),
                "selected_score_tuples": _score_tuple_details(selected_scores),
                "actual_score_tuples": _score_tuple_details(actual_scores),
                **(guard_extra or {}),
            },
        )
        return None

    guard_reason, guard_extra = _low_value_actual_density_guard_reason(
        block_pool,
        actual_cached_blocks,
        selected_blocks,
    )
    if guard_reason is not None:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            guard_reason,
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(candidates),
            metadata_ratio=metadata_ratio,
            candidate_free_ratio=candidate_free_ratio,
            extra_fields={
                **(_victim_summary_extra(
                    block_pool,
                    actual_cached_blocks,
                    selected_blocks,
                ) or {}),
                "selected_score_tuples": _score_tuple_details(selected_scores),
                "actual_score_tuples": _score_tuple_details(actual_scores),
                **(guard_extra or {}),
            },
        )
        return None

    guard_reason, guard_extra = _prefix_loss_delta_guard_reason(
        block_pool,
        actual_cached_blocks,
        selected_blocks,
    )
    if guard_reason is not None:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            guard_reason,
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(candidates),
            metadata_ratio=metadata_ratio,
            candidate_free_ratio=candidate_free_ratio,
            extra_fields={
                **(_victim_summary_extra(
                    block_pool,
                    actual_cached_blocks,
                    selected_blocks,
                ) or {}),
                "selected_score_tuples": _score_tuple_details(selected_scores),
                "actual_score_tuples": _score_tuple_details(actual_scores),
                **(guard_extra or {}),
            },
        )
        return None

    # Preserve the number of cached victims that baseline LRU would have evicted.
    # Non-cached head blocks are kept in the allocation set; cached LRU head
    # victims can be replaced by better cached candidates from the scanned queue.
    noncached_head_blocks = [
        block for block in actual_blocks if not _is_cached_free_block(block)
    ]
    allocation = selected_blocks + noncached_head_blocks
    if len(allocation) != num_blocks:
        _record_enforcement_attempt(
            block_pool,
            policy,
            num_blocks,
            "allocation_size_mismatch",
            started_ns,
            cached_victim_count=cached_victim_count,
            candidate_count=len(candidates),
            metadata_ratio=metadata_ratio,
            candidate_free_ratio=candidate_free_ratio,
        )
        return None

    allocated = _allocate_specific_free_blocks(block_pool, allocation)
    selected_block_ids = [getattr(block, "block_id", None) for block in selected_blocks]
    _record_enforcement_attempt(
        block_pool,
        policy,
        num_blocks,
        "applied",
        started_ns,
        cached_victim_count=cached_victim_count,
        candidate_count=len(candidates),
        metadata_ratio=metadata_ratio,
        candidate_free_ratio=candidate_free_ratio,
        selected_block_ids=selected_block_ids,
        extra_fields={
            **(_victim_summary_extra(
                block_pool,
                actual_cached_blocks,
                selected_blocks,
            ) or {}),
            **_candidate_detail_from_records(
                policy,
                candidates,
                hot_protect_sec,
                source="shadow_scan_applied",
            ),
        },
    )
    info = {
        "policy": policy,
        "metadata_ratio": metadata_ratio,
        "max_cached_victims_per_alloc": max_cached_victims,
        "candidate_free_ratio": candidate_free_ratio,
        "max_candidate_free_ratio": max_candidate_free_ratio,
        "actual_lru_cached_victim_count": cached_victim_count,
        "candidate_count": len(candidates),
        "selected_victims": selected_records,
        "selected_block_ids": selected_block_ids,
    }
    return allocated, info


def install_block_pool_shadow_patch() -> None:
    if not (kv_eviction_metadata_enabled() or record_allocation_shape_enabled()):
        return
    try:
        from vllm.v1.core.block_pool import BlockPool
    except Exception:  # pragma: no cover - vLLM import can vary by build.
        logger.exception("[OmniKVEvictShadow] failed to import BlockPool")
        return
    get_new_patched = getattr(
        BlockPool.get_new_blocks,
        "_omni_kv_evict_shadow_patch",
        False,
    )
    free_blocks_patched = getattr(
        BlockPool.free_blocks,
        "_omni_kv_evict_index_patch",
        False,
    )
    if get_new_patched and free_blocks_patched:
        return
    original_get_new_blocks = BlockPool.get_new_blocks

    def patched_get_new_blocks(self, num_blocks: int):
        _record_allocation_shape(self, num_blocks)
        event = _build_shadow_event(self, num_blocks)
        enforce_info: dict[str, Any] | None = None
        policy = enforcement_policy()
        enforced = None
        if policy is not None:
            enforced = _build_enforced_allocation(self, num_blocks, policy)
        if enforced is None:
            blocks = original_get_new_blocks(self, num_blocks)
        else:
            blocks, enforce_info = enforced
        if event is not None:
            event["allocated_block_ids"] = [
                getattr(block, "block_id", None) for block in blocks
            ]
            if policy is not None:
                max_cached_victims = _enforcement_max_cached_victims_per_alloc(self)
                actual_cached_count = len(event.get("actual_lru_victims") or [])
                event["enforce_policy_requested"] = policy
                event["enforce_applied"] = enforce_info is not None
                event["enforce_actual_lru_cached_victim_count"] = actual_cached_count
                event["enforce_max_cached_victims_per_alloc"] = max_cached_victims
                max_candidate_free_ratio = _enforcement_max_candidate_free_ratio()
                candidate_count = _as_float(event.get("num_candidate_blocks")) or 0.0
                free_count = _as_float(event.get("num_free_blocks")) or 0.0
                candidate_free_ratio = candidate_count / max(1.0, free_count)
                event["enforce_candidate_free_ratio"] = candidate_free_ratio
                event["enforce_max_candidate_free_ratio"] = max_candidate_free_ratio
                if enforce_info is None:
                    if (
                        max_cached_victims > 0
                        and actual_cached_count > max_cached_victims
                    ):
                        skip_reason = "cached_victim_count_over_cap"
                    elif (
                        max_candidate_free_ratio > 0.0
                        and candidate_free_ratio > max_candidate_free_ratio
                    ):
                        skip_reason = "candidate_free_ratio_over_cap"
                    else:
                        skip_reason = "fallback_lru"
                    event["enforce_skip_reason"] = skip_reason
            if enforce_info is not None:
                event["enforced_policy"] = enforce_info["policy"]
                event["enforced_metadata_ratio"] = enforce_info["metadata_ratio"]
                event["enforced_max_cached_victims_per_alloc"] = (
                    enforce_info["max_cached_victims_per_alloc"]
                )
                event["enforced_actual_lru_cached_victim_count"] = (
                    enforce_info["actual_lru_cached_victim_count"]
                )
                event["enforced_candidate_free_ratio"] = (
                    enforce_info["candidate_free_ratio"]
                )
                event["enforced_max_candidate_free_ratio"] = (
                    enforce_info["max_candidate_free_ratio"]
                )
                event["enforced_candidate_count"] = enforce_info["candidate_count"]
                event["enforced_block_ids"] = enforce_info["selected_block_ids"]
                event["enforced_victims"] = enforce_info["selected_victims"]
            event["enforce_stats"] = dict(_enforcement_stats(self))
            _write_event(event)
        mapping = getattr(self, "_omni_kv_evict_block_meta", None)
        if isinstance(mapping, dict):
            for block in blocks:
                block_id = getattr(block, "block_id", None)
                if block_id is not None:
                    mapping.pop(int(block_id), None)
                    _invalidate_indexed_block(self, int(block_id))
        return blocks

    if not get_new_patched:
        patched_get_new_blocks._omni_kv_evict_shadow_patch = True
        BlockPool.get_new_blocks = patched_get_new_blocks

    original_free_blocks = BlockPool.free_blocks

    def patched_free_blocks(self, ordered_blocks: Iterable[Any]) -> None:
        blocks_list = list(ordered_blocks)
        result = original_free_blocks(self, blocks_list)
        if _enforcement_index_enabled() and _index_metadata_active(self):
            policy = enforcement_policy()
            if policy is not None:
                for block in blocks_list:
                    if _is_cached_free_block(block) and _is_linked_free_block(block):
                        _push_indexed_candidate(self, block, policy)
        return result

    if not free_blocks_patched:
        patched_free_blocks._omni_kv_evict_index_patch = True
        BlockPool.free_blocks = patched_free_blocks


def install_kv_cache_manager_shadow_patch() -> None:
    if not kv_eviction_metadata_enabled():
        return
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
    except Exception:  # pragma: no cover - vLLM import can vary by build.
        logger.exception("[OmniKVEvictShadow] failed to import KVCacheManager")
        return
    if getattr(KVCacheManager.allocate_slots, "_omni_kv_evict_shadow_patch", False):
        return
    original_allocate_slots = KVCacheManager.allocate_slots

    def patched_allocate_slots(self, request: Any, *args: Any, **kwargs: Any):
        block_pool = getattr(self, "block_pool", None)
        if block_pool is not None:
            _set_current_request_meta(block_pool, request)
        try:
            allocated = original_allocate_slots(self, request, *args, **kwargs)
        finally:
            if block_pool is not None:
                _clear_current_request_meta(block_pool)
        if allocated is None:
            return allocated
        try:
            if (
                block_pool is not None
                and _capture_terminal_only()
                and not _request_is_terminal_turn(request)
            ):
                if _capture_count_terminal_skips():
                    stats = _shadow_stats(block_pool)
                    stats["capture_attempts"] += 1
                    profile_started_ns = (
                        time.perf_counter_ns()
                        if _profile_timing_enabled(block_pool)
                        else 0
                    )
                    _record_capture_gate_skip(
                        stats,
                        "nonterminal_request",
                        profile_started_ns,
                    )
                return allocated
            kv_blocks = self.get_blocks(request.request_id)
            block_groups = getattr(kv_blocks, "blocks", ()) or ()
            block_size = getattr(block_pool, "hash_block_size", None)
            capture_request_blocks_for_shadow(
                self.block_pool,
                request,
                block_groups,
                block_size,
            )
        except Exception:
            logger.exception(
                "[OmniKVEvictShadow] failed to capture cache blocks for %s",
                getattr(request, "request_id", None),
            )
        return allocated

    patched_allocate_slots._omni_kv_evict_shadow_patch = True
    KVCacheManager.allocate_slots = patched_allocate_slots


def install_current_request_meta_patch() -> None:
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
    except Exception:  # pragma: no cover - vLLM import can vary by build.
        logger.exception("[OmniKVEvictShadow] failed to import KVCacheManager")
        return

    if not getattr(
        KVCacheManager.get_computed_blocks,
        "_omni_kv_evict_current_meta_patch",
        False,
    ):
        original_get_computed_blocks = KVCacheManager.get_computed_blocks

        def patched_get_computed_blocks(self, request: Any, *args: Any, **kwargs: Any):
            block_pool = getattr(self, "block_pool", None)
            if block_pool is not None:
                _set_current_request_meta(block_pool, request)
            try:
                return original_get_computed_blocks(self, request, *args, **kwargs)
            finally:
                if block_pool is not None:
                    _clear_current_request_meta(block_pool)

        patched_get_computed_blocks._omni_kv_evict_current_meta_patch = True
        KVCacheManager.get_computed_blocks = patched_get_computed_blocks

    if not (
        getattr(KVCacheManager.allocate_slots, "_omni_kv_evict_shadow_patch", False)
        or getattr(
            KVCacheManager.allocate_slots,
            "_omni_kv_evict_current_meta_patch",
            False,
        )
    ):
        original_allocate_slots = KVCacheManager.allocate_slots

        def patched_allocate_slots(self, request: Any, *args: Any, **kwargs: Any):
            block_pool = getattr(self, "block_pool", None)
            if block_pool is not None:
                _set_current_request_meta(block_pool, request)
            try:
                return original_allocate_slots(self, request, *args, **kwargs)
            finally:
                if block_pool is not None:
                    _clear_current_request_meta(block_pool)

        patched_allocate_slots._omni_kv_evict_current_meta_patch = True
        KVCacheManager.allocate_slots = patched_allocate_slots
