#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import simulate_interactive_kv_policies as simpol
from simulate_compact_prewarm_kv import stable_user_hash


TRACE_DIR = Path(__file__).parent / "trace"


@dataclass(frozen=True)
class RuntimeSpec:
    label: str
    policy: str
    capture_terminal_only: bool
    require_selected_terminal: bool
    max_cached: int
    partial_over_cap: bool
    index_tail: int
    max_recent_applied_64: int = 0
    max_recent_no_cached_64: int = 0
    min_attempts_between_applied: int = 0


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


class RuntimeReplicaSim:
    def __init__(
        self,
        *,
        spec: RuntimeSpec,
        capacity_blocks: int,
        block_size_tokens: int,
        allocation_mode: str,
        runtime_alloc_scale: float,
        runtime_shape_sizes: list[int],
        runtime_shape_counts: list[int],
        runtime_shape_shuffle_seed: int,
        runtime_shape_strategy: str,
        runtime_mean_blocks_per_call: float,
        free_eligible_delay_sec: float,
        free_eligible_lru_scan_limit: int,
        runtime_cached_head_mode: str,
        runtime_cached_head_call_ratio: float,
    ) -> None:
        self.spec = spec
        self.policy = simpol.resolve_capacity_adaptive_policy(
            spec.policy,
            capacity_blocks,
        )
        self.capacity_blocks = int(capacity_blocks)
        self.block_size_tokens = int(block_size_tokens)
        self.allocation_mode = allocation_mode
        self.runtime_alloc_scale = float(runtime_alloc_scale)
        if len(runtime_shape_sizes) != len(runtime_shape_counts):
            raise ValueError("runtime shape sizes/counts length mismatch")
        self.runtime_shape_sequence: list[int] = []
        for size, count in zip(runtime_shape_sizes, runtime_shape_counts):
            if size <= 0 or count < 0:
                raise ValueError("runtime shape sizes must be >0 and counts >=0")
            self.runtime_shape_sequence.extend([int(size)] * int(count))
        if not self.runtime_shape_sequence:
            self.runtime_shape_sequence = [1]
        random.Random(runtime_shape_shuffle_seed).shuffle(self.runtime_shape_sequence)
        self.runtime_shape_pos = 0
        self.runtime_shape_strategy = runtime_shape_strategy
        self.runtime_mean_blocks_per_call = max(1.0, float(runtime_mean_blocks_per_call))
        self.free_eligible_delay_sec = max(0.0, float(free_eligible_delay_sec))
        self.free_eligible_lru_scan_limit = max(1, int(free_eligible_lru_scan_limit))
        self.runtime_cached_head_mode = runtime_cached_head_mode
        self.runtime_cached_head_call_ratio = min(
            1.0,
            max(0.0, float(runtime_cached_head_call_ratio)),
        )
        self.runtime_alloc_residual = 0.0
        self.runtime_call_residual = 0.0
        self.runtime_cached_head_residual = 0.0
        self.anon_counter = 0

        self.cache: dict[tuple[str, int], simpol.CacheBlock] = {}
        self.busy_until: dict[tuple[str, int], float] = {}
        self.lru_heap: list[tuple[int, int, tuple[str, int]]] = []
        self.lru_versions: defaultdict[tuple[str, int], int] = defaultdict(int)
        self.index_heap: list[tuple[tuple[Any, ...], int, tuple[str, int]]] = []
        self.index_versions: defaultdict[tuple[str, int], int] = defaultdict(int)
        self.history_tokens_by_user: dict[str, int] = defaultdict(int)
        self.ghost_evicted: set[tuple[str, int]] = set()
        self.touch_order = 0

        self.counters: defaultdict[str, int] = defaultdict(int)
        self.recent_applied_mask = 0
        self.recent_no_cached_mask = 0
        self.attempts_since_last_applied = 0

        self.hit_blocks_per_request: list[int] = []
        self.demand_blocks_per_request: list[int] = []
        self.miss_blocks_per_request: list[int] = []

    def _next_runtime_shape_size(self) -> int:
        size = self.runtime_shape_sequence[
            self.runtime_shape_pos % len(self.runtime_shape_sequence)
        ]
        self.runtime_shape_pos += 1
        return size

    def _split_runtime_allocation(self, target_blocks: int) -> list[int]:
        remaining = int(target_blocks)
        if remaining <= 0:
            return []
        if self.runtime_shape_strategy == "mean_call":
            raw_calls = remaining / self.runtime_mean_blocks_per_call + self.runtime_call_residual
            num_calls = int(math.floor(raw_calls))
            self.runtime_call_residual = raw_calls - num_calls
            if num_calls <= 0:
                num_calls = 1
            num_calls = min(num_calls, remaining)
            base = remaining // num_calls
            extra = remaining % num_calls
            return [base + (1 if idx < extra else 0) for idx in range(num_calls)]
        if self.runtime_shape_strategy == "observed_global":
            chunks: list[int] = []
            while remaining > 0:
                size = self._next_runtime_shape_size()
                chunks.append(size)
                remaining -= size
            return chunks
        if self.runtime_shape_strategy != "observed_sequence":
            raise ValueError(f"unknown runtime_shape_strategy={self.runtime_shape_strategy!r}")
        chunks: list[int] = []
        while remaining > 0:
            size = min(self._next_runtime_shape_size(), remaining)
            chunks.append(size)
            remaining -= size
        return chunks

    def _next_anon_key(self) -> tuple[str, int]:
        self.anon_counter += 1
        return ("__runtime_alloc_filler__", self.anon_counter)

    def _mark_busy(self, key: tuple[str, int], now_ts: int) -> None:
        if self.free_eligible_delay_sec <= 0.0:
            return
        self.busy_until[key] = max(
            self.busy_until.get(key, 0.0),
            float(now_ts) + self.free_eligible_delay_sec,
        )

    def _is_free_eligible(self, key: tuple[str, int], now_ts: int) -> bool:
        return self.busy_until.get(key, 0.0) <= float(now_ts)

    def _cached_head_available(self) -> bool:
        if self.runtime_cached_head_mode == "capacity":
            return True
        if self.runtime_cached_head_mode != "calibrated":
            raise ValueError(
                f"unknown runtime_cached_head_mode={self.runtime_cached_head_mode!r}"
            )
        self.runtime_cached_head_residual += self.runtime_cached_head_call_ratio
        if self.runtime_cached_head_residual >= 1.0:
            self.runtime_cached_head_residual -= 1.0
            self.counters["cached_head_model_available_calls"] += 1
            return True
        self.counters["cached_head_model_no_cached_calls"] += 1
        return False

    @staticmethod
    def _score_heap_key(score: tuple[Any, ...]) -> tuple[Any, ...]:
        out: list[Any] = []
        for value in score:
            if isinstance(value, (int, float)):
                out.append(-float(value))
            else:
                out.append(value)
        return tuple(out)

    @staticmethod
    def _score_lists_show_improvement(
        selected_scores: list[tuple[Any, ...]],
        actual_scores: list[tuple[Any, ...]],
    ) -> bool:
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

    def _push_lru(self, key: tuple[str, int]) -> None:
        block = self.cache.get(key)
        if block is None:
            return
        self.lru_versions[key] += 1
        heapq.heappush(
            self.lru_heap,
            (int(block.last_touch_order), self.lru_versions[key], key),
        )

    def _invalidate_index(self, key: tuple[str, int]) -> None:
        self.index_versions[key] += 1

    def _push_index(self, key: tuple[str, int], now_ts: int) -> None:
        block = self.cache.get(key)
        if block is None:
            return
        self.index_versions[key] += 1
        score = simpol.score_block(self.policy, block, now_ts)
        heapq.heappush(
            self.index_heap,
            (self._score_heap_key(score), self.index_versions[key], key),
        )
        self.counters["index_pushes"] += 1

    def _pop_lru_valid(
        self,
        *,
        now_ts: int,
        exclude: set[tuple[str, int]] | None = None,
    ) -> tuple[tuple[str, int], simpol.CacheBlock] | None:
        exclude = exclude or set()
        held: list[tuple[tuple[str, int], simpol.CacheBlock]] = []
        inspected = 0
        while self.lru_heap:
            inspected += 1
            if inspected > self.free_eligible_lru_scan_limit:
                self.counters["lru_scan_limit_stops"] += 1
                break
            _, version, key = heapq.heappop(self.lru_heap)
            block = self.cache.get(key)
            if block is None or self.lru_versions[key] != version:
                continue
            if key in exclude:
                held.append((key, block))
                continue
            if not self._is_free_eligible(key, now_ts):
                self.counters["lru_busy_skips"] += 1
                held.append((key, block))
                continue
            for held_key, _ in held:
                self._push_lru(held_key)
            return key, block
        for held_key, _ in held:
            self._push_lru(held_key)
        return None

    def _take_lru_victims(
        self,
        count: int,
        *,
        now_ts: int,
        exclude: set[tuple[str, int]] | None = None,
    ) -> list[tuple[tuple[str, int], simpol.CacheBlock]]:
        victims: list[tuple[tuple[str, int], simpol.CacheBlock]] = []
        exclude = exclude or set()
        held: list[tuple[tuple[str, int], simpol.CacheBlock]] = []
        while len(victims) < count:
            item = self._pop_lru_valid(now_ts=now_ts, exclude=exclude)
            if item is None:
                break
            key, block = item
            if key in exclude:
                held.append(item)
                continue
            victims.append(item)
            exclude.add(key)
        for key, _ in held:
            self._push_lru(key)
        return victims

    def _restore_lru_victims(
        self,
        victims: list[tuple[tuple[str, int], simpol.CacheBlock]],
    ) -> None:
        for key, _ in victims:
            if key in self.cache:
                self._push_lru(key)

    def _evict_victim_list(
        self,
        victims: list[tuple[tuple[str, int], simpol.CacheBlock]],
        *,
        now_ts: int,
    ) -> None:
        for key, _ in victims:
            self._remove_block(key, now_ts=now_ts, count_eviction=True)

    def _remove_block(
        self,
        key: tuple[str, int],
        *,
        now_ts: int,
        count_eviction: bool,
    ) -> None:
        block = self.cache.pop(key, None)
        if block is None:
            return
        self._invalidate_index(key)
        self.ghost_evicted.add(key)
        if count_eviction:
            self.counters["evictions"] += 1
            if block.next_use_ts is not None and block.next_use_ts > now_ts:
                self.counters["useful_evictions"] += 1

    def _recent_count(self, mask: int) -> int:
        return int(mask & ((1 << 64) - 1)).bit_count()

    def _record_attempt(
        self,
        reason: str,
        *,
        cached_victim_count: int = 0,
        candidate_count: int = 0,
        selected_count: int = 0,
    ) -> None:
        self.counters["attempts"] += 1
        if cached_victim_count:
            self.counters["lru_cached_victims_seen"] += cached_victim_count
        if candidate_count:
            self.counters["candidate_count_total"] += candidate_count
        if selected_count:
            self.counters["selected_replacement_victims"] += selected_count
        applied = reason == "applied"
        no_cached = reason == "no_cached_victims"
        if applied:
            self.counters["applied_allocations"] += 1
            if cached_victim_count:
                self.counters["selected_cached_victims"] += cached_victim_count
            self.attempts_since_last_applied = 0
        else:
            self.counters["fallback_allocations"] += 1
            self.counters[f"fallback_{reason}"] += 1
            self.attempts_since_last_applied += 1
        self.recent_applied_mask = (
            ((self.recent_applied_mask << 1) | (1 if applied else 0))
            & ((1 << 64) - 1)
        )
        self.recent_no_cached_mask = (
            ((self.recent_no_cached_mask << 1) | (1 if no_cached else 0))
            & ((1 << 64) - 1)
        )

    def _throttle_reason(self) -> str | None:
        if self.counters["applied_allocations"] <= 0:
            return None
        if (
            self.spec.min_attempts_between_applied > 0
            and self.attempts_since_last_applied
            < self.spec.min_attempts_between_applied
        ):
            return "applied_interval_below_min"
        attempts = self.counters["attempts"]
        if (
            self.spec.max_recent_applied_64 > 0
            and attempts >= 64
            and self._recent_count(self.recent_applied_mask)
            >= self.spec.max_recent_applied_64
        ):
            return "recent_applied_over_cap"
        if (
            self.spec.max_recent_no_cached_64 > 0
            and attempts >= 64
            and self._recent_count(self.recent_no_cached_mask)
            >= self.spec.max_recent_no_cached_64
        ):
            return "recent_no_cached_over_cap"
        return None

    def _block_terminal(self, block: simpol.CacheBlock) -> bool:
        return block.next_use_ts is None

    def _score(self, block: simpol.CacheBlock, now_ts: int) -> tuple[Any, ...]:
        return simpol.score_block(self.policy, block, now_ts)

    def _select_index_candidates(
        self,
        count: int,
        *,
        actual_keys: set[tuple[str, int]],
        now_ts: int,
    ) -> tuple[list[tuple[tuple[str, int], simpol.CacheBlock]], int]:
        selected: list[tuple[tuple[str, int], simpol.CacheBlock]] = []
        seen: set[tuple[str, int]] = set()
        held_busy: list[tuple[str, int]] = []
        pop_limit = max(64, count * 32)
        pops = 0
        while self.index_heap and len(selected) < count and pops < pop_limit:
            _, version, key = heapq.heappop(self.index_heap)
            pops += 1
            self.counters["index_pops"] += 1
            block = self.cache.get(key)
            if block is None or self.index_versions[key] != version:
                self.counters["index_invalid_pops"] += 1
                continue
            if not self._is_free_eligible(key, now_ts):
                self.counters["index_busy_pops"] += 1
                held_busy.append(key)
                continue
            if key in actual_keys or key in seen:
                self.counters["index_invalid_pops"] += 1
                continue
            if self.spec.require_selected_terminal and not self._block_terminal(block):
                self.counters["index_nonterminal_pops"] += 1
                continue
            selected.append((key, block))
            seen.add(key)
        for key in held_busy:
            if key in self.cache:
                self._push_index(key, now_ts)
        return selected, pops

    def _enforce_allocation(self, alloc_blocks: int, now_ts: int) -> None:
        if alloc_blocks <= 0:
            return
        free_uncached = max(0, self.capacity_blocks - len(self.cache))
        cached_victim_count = max(0, alloc_blocks - free_uncached)
        if cached_victim_count <= 0 or not self._cached_head_available():
            self._record_attempt("no_cached_victims")
            return

        actual_victims = self._take_lru_victims(
            cached_victim_count,
            now_ts=now_ts,
        )
        if not actual_victims:
            self._record_attempt("indexed_empty", cached_victim_count=cached_victim_count)
            return
        if len(actual_victims) < cached_victim_count:
            cached_victim_count = len(actual_victims)

        replacement_count = cached_victim_count
        if self.spec.max_cached > 0 and cached_victim_count > self.spec.max_cached:
            if self.spec.partial_over_cap:
                replacement_count = self.spec.max_cached
            else:
                self._evict_victim_list(actual_victims, now_ts=now_ts)
                self._record_attempt(
                    "cached_victim_count_over_cap",
                    cached_victim_count=cached_victim_count,
                )
                return
        if replacement_count <= 0:
            self._evict_victim_list(actual_victims, now_ts=now_ts)
            self._record_attempt(
                "replacement_count_zero",
                cached_victim_count=cached_victim_count,
            )
            return

        throttle = self._throttle_reason()
        if throttle is not None:
            self._evict_victim_list(actual_victims, now_ts=now_ts)
            self._record_attempt(throttle, cached_victim_count=cached_victim_count)
            return

        actual_keys = {key for key, _ in actual_victims}
        selected, _ = self._select_index_candidates(
            replacement_count,
            actual_keys=actual_keys,
            now_ts=now_ts,
        )
        if len(selected) != replacement_count:
            for key, _ in selected:
                self._push_index(key, now_ts)
            self._evict_victim_list(actual_victims, now_ts=now_ts)
            self._record_attempt(
                "indexed_not_enough_candidates" if selected else "indexed_empty",
                cached_victim_count=cached_victim_count,
                candidate_count=len(selected),
            )
            return

        selected_scores = [self._score(block, now_ts) for _, block in selected]
        scored_actual = [
            (self._score(block, now_ts), key, block)
            for key, block in actual_victims
        ]
        scored_actual.sort(key=lambda item: item[0])
        actual_replace = [
            (key, block)
            for _, key, block in scored_actual[:replacement_count]
        ]
        actual_keep = [
            (key, block)
            for _, key, block in scored_actual[replacement_count:]
        ]
        actual_scores = [
            score for score, _, _ in scored_actual[:replacement_count]
        ]
        if not self._score_lists_show_improvement(selected_scores, actual_scores):
            for key, _ in selected:
                self._push_index(key, now_ts)
            self._evict_victim_list(actual_victims, now_ts=now_ts)
            self._record_attempt(
                "no_policy_score_improvement",
                cached_victim_count=cached_victim_count,
                candidate_count=len(selected),
            )
            return

        for key, _ in selected:
            self._remove_block(key, now_ts=now_ts, count_eviction=True)
        for key, _ in actual_keep:
            self._remove_block(key, now_ts=now_ts, count_eviction=True)
        self._restore_lru_victims(actual_replace)
        self._record_attempt(
            "applied",
            cached_victim_count=cached_victim_count,
            candidate_count=len(selected),
            selected_count=len(selected),
        )

    def _fallback_evict_until_fit(self, now_ts: int) -> None:
        while len(self.cache) > self.capacity_blocks:
            item = self._pop_lru_valid(now_ts=now_ts)
            if item is None:
                break
            key, _ = item
            self._remove_block(key, now_ts=now_ts, count_eviction=True)

    def _capture_request(self, entry: simpol.Entry, total_blocks_after: int) -> None:
        self.counters["capture_attempts"] += 1
        terminal = entry.next_gap_sec is None
        if self.spec.capture_terminal_only and not terminal:
            if self.spec.capture_terminal_only:
                self.counters["capture_skipped_nonterminal"] += 1
            return
        self.counters["capture_gate_active_requests"] += 1
        if total_blocks_after <= 0:
            return
        start_index = (
            max(1, total_blocks_after - self.spec.index_tail + 1)
            if self.spec.index_tail > 0
            else 1
        )
        captured = 0
        for block_index in range(start_index, total_blocks_after + 1):
            key = (entry.user_id, block_index)
            if key not in self.cache:
                continue
            self._push_index(key, entry.timestamp)
            captured += 1
        if captured:
            self.counters["captured_requests"] += 1
            self.counters["captured_blocks"] += captured

    def _upsert_real_block(
        self,
        entry: simpol.Entry,
        *,
        block_index: int,
        total_blocks_after: int,
        history_after: int,
    ) -> None:
        key = (entry.user_id, block_index)
        existing = self.cache.get(key)
        if existing is None:
            existing = simpol.CacheBlock(
                user_id=entry.user_id,
                block_index=block_index,
                total_blocks=total_blocks_after,
                created_ts=entry.timestamp,
                last_touch_order=self.touch_order,
                next_use_ts=entry.next_use_ts,
                next_gap_sec=entry.next_gap_sec,
                response_length=entry.response_length,
                history_tokens_after=history_after,
            )
            self.cache[key] = existing
        else:
            existing.total_blocks = total_blocks_after
            existing.last_touch_order = self.touch_order
            existing.next_use_ts = entry.next_use_ts
            existing.next_gap_sec = entry.next_gap_sec
            existing.response_length = entry.response_length
            existing.history_tokens_after = history_after
        self._push_lru(key)
        self._mark_busy(key, entry.timestamp)

    def _insert_filler_block(self, entry: simpol.Entry) -> None:
        key = self._next_anon_key()
        self.cache[key] = simpol.CacheBlock(
            user_id=key[0],
            block_index=key[1],
            total_blocks=1,
            created_ts=entry.timestamp,
            last_touch_order=self.touch_order,
            next_use_ts=None,
            next_gap_sec=None,
            response_length=0,
            history_tokens_after=0,
        )
        self._push_lru(key)
        self._mark_busy(key, entry.timestamp)

    def _apply_runtime_full_prompt_allocation(
        self,
        entry: simpol.Entry,
        *,
        total_blocks_after: int,
        history_after: int,
    ) -> None:
        missing_keys = [
            (entry.user_id, block_index)
            for block_index in range(1, total_blocks_after + 1)
            if (entry.user_id, block_index) not in self.cache
        ]
        raw_target = (
            float(total_blocks_after) * self.runtime_alloc_scale
            + self.runtime_alloc_residual
        )
        target_blocks = int(math.floor(raw_target))
        self.runtime_alloc_residual = raw_target - target_blocks
        target_blocks = max(target_blocks, len(missing_keys))
        if total_blocks_after > 0 and target_blocks <= 0:
            target_blocks = 1

        chunks = self._split_runtime_allocation(target_blocks)
        requested_blocks = sum(chunks)
        filler_count = max(0, requested_blocks - len(missing_keys))
        allocation_targets: list[tuple[str, int] | None] = list(missing_keys)
        allocation_targets.extend([None] * filler_count)

        self.counters["runtime_shape_requested_blocks"] += requested_blocks
        self.counters["runtime_shape_missing_real_blocks"] += len(missing_keys)
        self.counters["runtime_shape_filler_blocks"] += filler_count

        offset = 0
        for chunk_size in chunks:
            chunk_targets = allocation_targets[offset : offset + chunk_size]
            offset += chunk_size
            self.counters["runtime_shape_calls"] += 1
            self.counters["runtime_shape_call_blocks"] += chunk_size
            self._enforce_allocation(chunk_size, entry.timestamp)
            for key in chunk_targets:
                if key is None:
                    self._insert_filler_block(entry)
                else:
                    block_index = key[1]
                    self._upsert_real_block(
                        entry,
                        block_index=block_index,
                        total_blocks_after=total_blocks_after,
                        history_after=history_after,
                    )

    def apply_entry(self, entry: simpol.Entry) -> None:
        self.touch_order += 1
        if entry.round_index == 0:
            keys = [key for key in self.cache if key[0] == entry.user_id]
            for key in keys:
                self._remove_block(key, now_ts=entry.timestamp, count_eviction=False)
            self.history_tokens_by_user[entry.user_id] = 0

        history_before = int(self.history_tokens_by_user.get(entry.user_id, 0))
        demand_blocks = history_before // self.block_size_tokens
        contiguous_hits = 0
        recent_miss_blocks = 0
        for block_index in range(1, demand_blocks + 1):
            key = (entry.user_id, block_index)
            block = self.cache.get(key)
            if block is None:
                if key in self.ghost_evicted:
                    recent_miss_blocks += 1
                break
            block.last_touch_order = self.touch_order
            self._push_lru(key)
            self._mark_busy(key, entry.timestamp)
            contiguous_hits += 1

        miss_blocks = max(0, demand_blocks - contiguous_hits)
        if miss_blocks and recent_miss_blocks == 0:
            first_miss = contiguous_hits + 1
            for block_index in range(first_miss, demand_blocks + 1):
                if (entry.user_id, block_index) in self.ghost_evicted:
                    recent_miss_blocks += 1

        self.counters["requests"] += 1
        self.counters["demand_blocks"] += demand_blocks
        self.counters["hit_blocks"] += contiguous_hits
        self.counters["miss_blocks"] += miss_blocks
        self.counters["recent_miss_blocks"] += recent_miss_blocks
        if demand_blocks > 0:
            self.counters["requests_with_demand"] += 1
            if contiguous_hits == demand_blocks:
                self.counters["request_full_hits"] += 1
        if contiguous_hits > 0:
            self.counters["request_any_hits"] += 1
        self.hit_blocks_per_request.append(contiguous_hits)
        self.demand_blocks_per_request.append(demand_blocks)
        self.miss_blocks_per_request.append(miss_blocks)

        history_after = (
            history_before
            + int(entry.query_length)
            + int(entry.response_length)
        )
        total_blocks_after = history_after // self.block_size_tokens
        if self.allocation_mode == "full_prompt":
            alloc_blocks = total_blocks_after
        elif self.allocation_mode == "suffix_after_miss":
            alloc_blocks = max(0, total_blocks_after - contiguous_hits)
        elif self.allocation_mode == "missing_keys":
            alloc_blocks = sum(
                1
                for block_index in range(1, total_blocks_after + 1)
                if (entry.user_id, block_index) not in self.cache
            )
        elif self.allocation_mode == "runtime_full_prompt":
            alloc_blocks = 0
        else:
            raise ValueError(f"unknown allocation_mode={self.allocation_mode!r}")

        if self.allocation_mode == "runtime_full_prompt":
            self._apply_runtime_full_prompt_allocation(
                entry,
                total_blocks_after=total_blocks_after,
                history_after=history_after,
            )
        else:
            self._enforce_allocation(alloc_blocks, entry.timestamp)

        for block_index in range(1, total_blocks_after + 1):
            self._upsert_real_block(
                entry,
                block_index=block_index,
                total_blocks_after=total_blocks_after,
                history_after=history_after,
            )
        self.history_tokens_by_user[entry.user_id] = history_after

        self._fallback_evict_until_fit(entry.timestamp)
        self.counters["occupancy_peak_blocks"] = max(
            self.counters["occupancy_peak_blocks"],
            len(self.cache),
        )
        self._capture_request(entry, total_blocks_after)

    def result(self) -> dict[str, Any]:
        c = self.counters
        demand = int(c.get("demand_blocks", 0))
        hit = int(c.get("hit_blocks", 0))
        requests_with_demand = int(c.get("requests_with_demand", 0))
        out: dict[str, Any] = dict(c)
        out.update(
            {
                "hit_ratio": hit / demand if demand else 0.0,
                "request_full_hit_rate": (
                    int(c.get("request_full_hits", 0)) / requests_with_demand
                    if requests_with_demand
                    else 0.0
                ),
                "request_any_hit_rate": (
                    int(c.get("request_any_hits", 0)) / int(c.get("requests", 1))
                    if c.get("requests", 0)
                    else 0.0
                ),
                "final_occupancy_blocks": len(self.cache),
                "index_heap_entries": len(self.index_heap),
                "recent_applied_attempts_64": self._recent_count(
                    self.recent_applied_mask,
                ),
                "recent_no_cached_attempts_64": self._recent_count(
                    self.recent_no_cached_mask,
                ),
                "hit_blocks_per_request": describe(self.hit_blocks_per_request),
                "demand_blocks_per_request": describe(self.demand_blocks_per_request),
                "miss_blocks_per_request": describe(self.miss_blocks_per_request),
            },
        )
        return out


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def describe(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    xs = [float(v) for v in values]
    return {
        "n": len(xs),
        "p50": percentile(xs, 0.50),
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
        "max": max(xs),
        "mean": statistics.mean(xs),
    }


def add_dicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summed: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, int):
                summed[key] += value
    demand = int(summed.get("demand_blocks", 0))
    hit = int(summed.get("hit_blocks", 0))
    requests_with_demand = int(summed.get("requests_with_demand", 0))
    requests = int(summed.get("requests", 0))
    out: dict[str, Any] = dict(summed)
    out["hit_ratio"] = hit / demand if demand else 0.0
    out["request_full_hit_rate"] = (
        int(summed.get("request_full_hits", 0)) / requests_with_demand
        if requests_with_demand
        else 0.0
    )
    out["request_any_hit_rate"] = (
        int(summed.get("request_any_hits", 0)) / requests if requests else 0.0
    )
    return out


def pct_reduction(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return (baseline - candidate) / baseline * 100.0


def make_specs(args: argparse.Namespace) -> list[RuntimeSpec]:
    if args.spec:
        specs = []
        for raw in args.spec:
            parts = dict(item.split("=", 1) for item in raw.split(",") if "=" in item)
            specs.append(
                RuntimeSpec(
                    label=parts.get("label", f"spec{len(specs)}"),
                    policy=parts.get("policy", args.policy),
                    capture_terminal_only=parts.get("capture_terminal_only", "1") == "1",
                    require_selected_terminal=parts.get("require_selected_terminal", "1") == "1",
                    max_cached=int(parts.get("max_cached", args.max_cached)),
                    partial_over_cap=parts.get("partial_over_cap", "1") == "1",
                    index_tail=int(parts.get("index_tail", args.index_tail)),
                    max_recent_applied_64=int(parts.get("max_recent_applied_64", args.max_recent_applied_64)),
                    max_recent_no_cached_64=int(parts.get("max_recent_no_cached_64", args.max_recent_no_cached_64)),
                    min_attempts_between_applied=int(parts.get("min_attempts_between_applied", args.min_attempts_between_applied)),
                )
            )
        return specs

    specs = [
        RuntimeSpec(
            label="current_terminal_m8_tail128",
            policy=args.policy,
            capture_terminal_only=True,
            require_selected_terminal=True,
            max_cached=8,
            partial_over_cap=True,
            index_tail=128,
            max_recent_applied_64=args.max_recent_applied_64,
            max_recent_no_cached_64=args.max_recent_no_cached_64,
            min_attempts_between_applied=args.min_attempts_between_applied,
        )
    ]
    for max_cached in (16, 32):
        specs.append(
            RuntimeSpec(
                label=f"terminal_m{max_cached}_tail128",
                policy=args.policy,
                capture_terminal_only=True,
                require_selected_terminal=True,
                max_cached=max_cached,
                partial_over_cap=True,
                index_tail=128,
                max_recent_applied_64=args.max_recent_applied_64,
                max_recent_no_cached_64=args.max_recent_no_cached_64,
                min_attempts_between_applied=args.min_attempts_between_applied,
            )
        )
    for max_cached in (8, 16, 32):
        specs.append(
            RuntimeSpec(
                label=f"capture_all_req_terminal_m{max_cached}_tail128",
                policy=args.policy,
                capture_terminal_only=False,
                require_selected_terminal=True,
                max_cached=max_cached,
                partial_over_cap=True,
                index_tail=128,
                max_recent_applied_64=args.max_recent_applied_64,
                max_recent_no_cached_64=args.max_recent_no_cached_64,
                min_attempts_between_applied=args.min_attempts_between_applied,
            )
        )
        specs.append(
            RuntimeSpec(
                label=f"capture_all_any_m{max_cached}_tail128",
                policy=args.policy,
                capture_terminal_only=False,
                require_selected_terminal=False,
                max_cached=max_cached,
                partial_over_cap=True,
                index_tail=128,
                max_recent_applied_64=args.max_recent_applied_64,
                max_recent_no_cached_64=args.max_recent_no_cached_64,
                min_attempts_between_applied=args.min_attempts_between_applied,
            )
        )
    specs.extend(
        [
            RuntimeSpec(
                label="capture_all_any_m16_tail256",
                policy=args.policy,
                capture_terminal_only=False,
                require_selected_terminal=False,
                max_cached=16,
                partial_over_cap=True,
                index_tail=256,
                max_recent_applied_64=args.max_recent_applied_64,
                max_recent_no_cached_64=args.max_recent_no_cached_64,
                min_attempts_between_applied=args.min_attempts_between_applied,
            ),
            RuntimeSpec(
                label="capture_all_any_uncapped_tail256",
                policy=args.policy,
                capture_terminal_only=False,
                require_selected_terminal=False,
                max_cached=0,
                partial_over_cap=False,
                index_tail=256,
                max_recent_applied_64=0,
                max_recent_no_cached_64=0,
                min_attempts_between_applied=0,
            ),
        ]
    )
    return specs


def load_entries(args: argparse.Namespace) -> list[simpol.Entry]:
    entries = simpol.select_window(
        simpol.load_entries(args.trace_dir, args.parts),
        args.time_start,
        args.time_limit,
    )
    if args.next_use_scope == "selected":
        simpol.reannotate_next_use_within_selected(entries)
    return entries


def run_spec(
    entries: list[simpol.Entry],
    spec: RuntimeSpec,
    *,
    replicas: int,
    capacity_blocks_per_replica: int,
    block_size_tokens: int,
    assignment: str,
    allocation_mode: str,
    runtime_alloc_scale: float,
    runtime_shape_sizes: list[int],
    runtime_shape_counts: list[int],
    runtime_shape_shuffle_seed: int,
    runtime_shape_strategy: str,
    runtime_mean_blocks_per_call: float,
    free_eligible_delay_sec: float,
    free_eligible_lru_scan_limit: int,
    runtime_cached_head_mode: str,
    runtime_cached_head_call_ratio: float,
) -> dict[str, Any]:
    sims = [
        RuntimeReplicaSim(
            spec=spec,
            capacity_blocks=capacity_blocks_per_replica,
            block_size_tokens=block_size_tokens,
            allocation_mode=allocation_mode,
            runtime_alloc_scale=runtime_alloc_scale,
            runtime_shape_sizes=runtime_shape_sizes,
            runtime_shape_counts=runtime_shape_counts,
            runtime_shape_shuffle_seed=runtime_shape_shuffle_seed,
            runtime_shape_strategy=runtime_shape_strategy,
            runtime_mean_blocks_per_call=runtime_mean_blocks_per_call,
            free_eligible_delay_sec=free_eligible_delay_sec,
            free_eligible_lru_scan_limit=free_eligible_lru_scan_limit,
            runtime_cached_head_mode=runtime_cached_head_mode,
            runtime_cached_head_call_ratio=runtime_cached_head_call_ratio,
        )
        for _ in range(replicas)
    ]
    per_replica_requests = [0 for _ in range(replicas)]
    for idx, entry in enumerate(entries):
        if assignment == "user_hash":
            rep = stable_user_hash(entry.user_id) % replicas
        elif assignment == "round_robin":
            rep = idx % replicas
        else:
            raise ValueError(f"unknown assignment={assignment!r}")
        per_replica_requests[rep] += 1
        sims[rep].apply_entry(entry)
    per_replica = [sim.result() for sim in sims]
    total = add_dicts(per_replica)
    total.update(
        {
            "label": spec.label,
            "policy": spec.policy,
            "effective_policy": sims[0].policy if sims else spec.policy,
            "capture_terminal_only": spec.capture_terminal_only,
            "require_selected_terminal": spec.require_selected_terminal,
            "max_cached": spec.max_cached,
            "partial_over_cap": spec.partial_over_cap,
            "index_tail": spec.index_tail,
            "max_recent_applied_64": spec.max_recent_applied_64,
            "max_recent_no_cached_64": spec.max_recent_no_cached_64,
            "min_attempts_between_applied": spec.min_attempts_between_applied,
            "replicas": replicas,
            "assignment": assignment,
            "allocation_mode": allocation_mode,
            "runtime_alloc_scale": runtime_alloc_scale,
            "runtime_shape_sizes": runtime_shape_sizes,
            "runtime_shape_counts": runtime_shape_counts,
            "runtime_shape_shuffle_seed": runtime_shape_shuffle_seed,
            "runtime_shape_strategy": runtime_shape_strategy,
            "runtime_mean_blocks_per_call": runtime_mean_blocks_per_call,
            "free_eligible_delay_sec": free_eligible_delay_sec,
            "free_eligible_lru_scan_limit": free_eligible_lru_scan_limit,
            "runtime_cached_head_mode": runtime_cached_head_mode,
            "runtime_cached_head_call_ratio": runtime_cached_head_call_ratio,
            "capacity_blocks_per_replica": capacity_blocks_per_replica,
            "capacity_tokens_total": capacity_blocks_per_replica
            * replicas
            * block_size_tokens,
            "per_replica_requests": per_replica_requests,
            "per_replica": per_replica,
        }
    )
    return total


def lru_spec() -> RuntimeSpec:
    return RuntimeSpec(
        label="lru",
        policy="lru",
        capture_terminal_only=False,
        require_selected_terminal=False,
        max_cached=0,
        partial_over_cap=False,
        index_tail=0,
    )


def add_deltas(rows: list[dict[str, Any]]) -> None:
    baseline = next(row for row in rows if row["label"] == "lru")
    for row in rows:
        for key in (
            "miss_blocks",
            "recent_miss_blocks",
            "evictions",
            "useful_evictions",
        ):
            row[f"{key}_reduction_pct"] = pct_reduction(
                float(row.get(key, 0)),
                float(baseline.get(key, 0)),
            )
        row["hit_ratio_gain_pp"] = (
            float(row.get("hit_ratio", 0.0))
            - float(baseline.get("hit_ratio", 0.0))
        ) * 100.0
        row["full_hit_gain_pp"] = (
            float(row.get("request_full_hit_rate", 0.0))
            - float(baseline.get("request_full_hit_rate", 0.0))
        ) * 100.0


def add_counter_scaled_fields(rows: list[dict[str, Any]], scale: float) -> None:
    scale = float(scale)
    for row in rows:
        for key in ("evictions", "useful_evictions", "recent_miss_blocks"):
            row[f"{key}_counter_scaled"] = float(row.get(key, 0)) * scale


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# Runtime-Aligned Eviction Simulation",
        "",
        "This is an offline model for policy triage, not GPU performance evidence.",
        "",
        f"- window: [{payload['config']['time_start']}, {payload['config']['time_limit']})",
        f"- entries: {payload['config']['entries']}",
        f"- replicas: {payload['config']['replicas']}",
        f"- capacity_blocks_per_replica: {payload['config']['capacity_blocks_per_replica']}",
        f"- allocation_mode: `{payload['config']['allocation_mode']}`",
        f"- assignment: `{payload['config']['assignment']}`",
        f"- next_use_scope: `{payload['config']['next_use_scope']}`",
        f"- counter_scale: `{payload['config']['counter_scale']}`",
        f"- runtime_alloc_scale: `{payload['config']['runtime_alloc_scale']}`",
        f"- runtime_shape_sizes: `{payload['config']['runtime_shape_sizes']}`",
        f"- runtime_shape_counts: `{payload['config']['runtime_shape_counts']}`",
        f"- runtime_shape_shuffle_seed: `{payload['config']['runtime_shape_shuffle_seed']}`",
        f"- runtime_shape_strategy: `{payload['config']['runtime_shape_strategy']}`",
        f"- runtime_mean_blocks_per_call: `{payload['config']['runtime_mean_blocks_per_call']}`",
        f"- lost_token_block_size_tokens: `{payload['config']['lost_token_block_size_tokens']}`",
        f"- free_eligible_delay_sec: `{payload['config']['free_eligible_delay_sec']}`",
        f"- free_eligible_lru_scan_limit: `{payload['config']['free_eligible_lru_scan_limit']}`",
        f"- runtime_cached_head_mode: `{payload['config']['runtime_cached_head_mode']}`",
        f"- runtime_cached_head_call_ratio: `{payload['config']['runtime_cached_head_call_ratio']}`",
        "",
        "| rank | label | hit | miss red | recent red | evict red | useful red | evict scaled | recent scaled | attempts/applied | fallbacks | capture req/blocks |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(
        rows,
        key=lambda row: (
            -9999.0
            if row["label"] == "lru"
            else float(row.get("useful_evictions_reduction_pct") or -9999.0),
            float(row.get("evictions_reduction_pct") or -9999.0),
            float(row.get("hit_ratio_gain_pp") or -9999.0),
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | `{row['label']}` | {float(row.get('hit_ratio', 0.0)):.4f} | "
            f"{fmt(row.get('miss_blocks_reduction_pct'), 2)} | "
            f"{fmt(row.get('recent_miss_blocks_reduction_pct'), 2)} | "
            f"{fmt(row.get('evictions_reduction_pct'), 2)} | "
            f"{fmt(row.get('useful_evictions_reduction_pct'), 2)} | "
            f"{fmt(row.get('evictions_counter_scaled'), 1)} | "
            f"{fmt(row.get('recent_miss_blocks_counter_scaled'), 1)} | "
            f"{row.get('attempts', 0)}/{row.get('applied_allocations', 0)} | "
            f"{row.get('fallback_allocations', 0)} | "
            f"{row.get('captured_requests', 0)}/{row.get('captured_blocks', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Fallback Details",
            "",
            "| label | indexed_empty | not_enough | no_policy | no_cached | recent_applied_cap | recent_no_cached_cap | candidate_total | index_pushes | invalid_pops |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["label"] == "lru":
            continue
        lines.append(
            f"| `{row['label']}` | "
            f"{row.get('fallback_indexed_empty', 0)} | "
            f"{row.get('fallback_indexed_not_enough_candidates', 0)} | "
            f"{row.get('fallback_no_policy_score_improvement', 0)} | "
            f"{row.get('fallback_no_cached_victims', 0)} | "
            f"{row.get('fallback_recent_applied_over_cap', 0)} | "
            f"{row.get('fallback_recent_no_cached_over_cap', 0)} | "
            f"{row.get('candidate_count_total', 0)} | "
            f"{row.get('index_pushes', 0)} | "
            f"{row.get('index_invalid_pops', 0)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--parts", type=int, nargs="+", default=[1])
    parser.add_argument("--time-start", type=int, default=6600)
    parser.add_argument("--time-limit", type=int, default=7200)
    parser.add_argument("--next-use-scope", choices=["full_trace", "selected"], default="full_trace")
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--assignment", choices=["user_hash", "round_robin"], default="user_hash")
    parser.add_argument("--capacity-blocks-per-replica", type=int, default=10744)
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument(
        "--lost-token-block-size-tokens",
        type=int,
        default=0,
        help=(
            "Scale suffix blocks into estimated lost tokens for cost-aware scores. "
            "Default 0 uses --block-size-tokens to match online metadata."
        ),
    )
    parser.add_argument(
        "--allocation-mode",
        choices=[
            "full_prompt",
            "suffix_after_miss",
            "missing_keys",
            "runtime_full_prompt",
        ],
        default="full_prompt",
    )
    parser.add_argument("--runtime-alloc-scale", type=float, default=1.0884610711642755)
    parser.add_argument(
        "--runtime-shape-sizes",
        default="1,2,3,6,12,24,48,70",
        help="Representative allocation sizes for runtime_full_prompt mode.",
    )
    parser.add_argument(
        "--runtime-shape-counts",
        default="30844,599,1266,2643,2405,3074,5302,1205",
        help="Relative counts for runtime_full_prompt allocation sizes.",
    )
    parser.add_argument("--runtime-shape-shuffle-seed", type=int, default=1)
    parser.add_argument(
        "--runtime-shape-strategy",
        choices=["observed_sequence", "observed_global", "mean_call"],
        default="mean_call",
    )
    parser.add_argument(
        "--runtime-mean-blocks-per-call",
        type=float,
        default=10.408762516371626,
    )
    parser.add_argument("--free-eligible-delay-sec", type=float, default=0.0)
    parser.add_argument("--free-eligible-lru-scan-limit", type=int, default=4096)
    parser.add_argument(
        "--runtime-cached-head-mode",
        choices=["capacity", "calibrated"],
        default="capacity",
    )
    parser.add_argument(
        "--runtime-cached-head-call-ratio",
        type=float,
        default=34505.0 / 47338.0,
    )
    parser.add_argument("--policy", default="expected_gap_cost_aware")
    parser.add_argument("--max-cached", type=int, default=8)
    parser.add_argument("--index-tail", type=int, default=128)
    parser.add_argument("--max-recent-applied-64", type=int, default=60)
    parser.add_argument("--max-recent-no-cached-64", type=int, default=32)
    parser.add_argument("--min-attempts-between-applied", type=int, default=0)
    parser.add_argument("--counter-scale", type=float, default=1.0)
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help=(
            "Optional comma-separated spec fields, for example "
            "label=x,capture_terminal_only=0,require_selected_terminal=0,max_cached=16,index_tail=128"
        ),
    )
    parser.add_argument("--gap-bucket-sec", type=float, default=30.0)
    parser.add_argument("--size-bonus-min-gap-sec", type=float, default=45.0)
    parser.add_argument("--size-history-weight", type=float, default=8.0)
    parser.add_argument("--size-response-weight", type=float, default=2.0)
    parser.add_argument("--size-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--bidaw-reuse-blocks-per-sec",
        type=float,
        default=0.0,
        help=(
            "Reuse-distance rate per replica for bidaw_* policies. 0 estimates "
            "it from the selected window and replica count."
        ),
    )
    parser.add_argument("--bidaw-small-reuse-ratio", type=float, default=1.0)
    parser.add_argument("--bidaw-extreme-reuse-ratio", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    simpol.GAP_BUCKET_SEC = float(args.gap_bucket_sec)
    simpol.SIZE_BONUS_MIN_GAP_SEC = float(args.size_bonus_min_gap_sec)
    simpol.SIZE_HISTORY_WEIGHT = float(args.size_history_weight)
    simpol.SIZE_RESPONSE_WEIGHT = float(args.size_response_weight)
    simpol.SIZE_LOSS_WEIGHT = float(args.size_loss_weight)
    lost_token_block_size_tokens = (
        int(args.lost_token_block_size_tokens)
        if int(args.lost_token_block_size_tokens) > 0
        else int(args.block_size_tokens)
    )
    simpol.LOST_TOKEN_BLOCK_SIZE_TOKENS = max(1, lost_token_block_size_tokens)

    entries = load_entries(args)
    bidaw_reuse_blocks_per_sec = float(args.bidaw_reuse_blocks_per_sec)
    if bidaw_reuse_blocks_per_sec <= 0.0:
        bidaw_reuse_blocks_per_sec = simpol.estimate_demand_blocks_per_sec(
            entries,
            block_size_tokens=args.block_size_tokens,
            replicas=args.replicas,
        )
    simpol.BIDAW_REUSE_BLOCKS_PER_SEC = bidaw_reuse_blocks_per_sec
    simpol.BIDAW_SMALL_REUSE_RATIO = float(args.bidaw_small_reuse_ratio)
    simpol.BIDAW_EXTREME_REUSE_RATIO = float(args.bidaw_extreme_reuse_ratio)
    specs = [lru_spec()] + make_specs(args)
    runtime_shape_sizes = parse_int_list(args.runtime_shape_sizes)
    runtime_shape_counts = parse_int_list(args.runtime_shape_counts)
    rows = [
        run_spec(
            entries,
            spec,
            replicas=args.replicas,
            capacity_blocks_per_replica=args.capacity_blocks_per_replica,
            block_size_tokens=args.block_size_tokens,
            assignment=args.assignment,
            allocation_mode=args.allocation_mode,
            runtime_alloc_scale=args.runtime_alloc_scale,
            runtime_shape_sizes=runtime_shape_sizes,
            runtime_shape_counts=runtime_shape_counts,
            runtime_shape_shuffle_seed=args.runtime_shape_shuffle_seed,
            runtime_shape_strategy=args.runtime_shape_strategy,
            runtime_mean_blocks_per_call=args.runtime_mean_blocks_per_call,
            free_eligible_delay_sec=args.free_eligible_delay_sec,
            free_eligible_lru_scan_limit=args.free_eligible_lru_scan_limit,
            runtime_cached_head_mode=args.runtime_cached_head_mode,
            runtime_cached_head_call_ratio=args.runtime_cached_head_call_ratio,
        )
        for spec in specs
    ]
    add_deltas(rows)
    add_counter_scaled_fields(rows, args.counter_scale)
    rows.sort(
        key=lambda row: (
            row["label"] == "lru",
            float(row.get("useful_evictions_reduction_pct") or -9999.0),
            float(row.get("evictions_reduction_pct") or -9999.0),
            float(row.get("hit_ratio_gain_pp") or -9999.0),
        ),
        reverse=True,
    )

    config = {
        "trace_dir": str(args.trace_dir),
        "parts": args.parts,
        "time_start": args.time_start,
        "time_limit": args.time_limit,
        "entries": len(entries),
        "next_use_scope": args.next_use_scope,
        "replicas": args.replicas,
        "assignment": args.assignment,
        "capacity_blocks_per_replica": args.capacity_blocks_per_replica,
        "block_size_tokens": args.block_size_tokens,
        "lost_token_block_size_tokens": simpol.LOST_TOKEN_BLOCK_SIZE_TOKENS,
        "allocation_mode": args.allocation_mode,
        "runtime_alloc_scale": args.runtime_alloc_scale,
        "runtime_shape_sizes": runtime_shape_sizes,
        "runtime_shape_counts": runtime_shape_counts,
        "runtime_shape_shuffle_seed": args.runtime_shape_shuffle_seed,
        "runtime_shape_strategy": args.runtime_shape_strategy,
        "runtime_mean_blocks_per_call": args.runtime_mean_blocks_per_call,
        "policy": args.policy,
        "free_eligible_delay_sec": args.free_eligible_delay_sec,
        "free_eligible_lru_scan_limit": args.free_eligible_lru_scan_limit,
        "runtime_cached_head_mode": args.runtime_cached_head_mode,
        "runtime_cached_head_call_ratio": args.runtime_cached_head_call_ratio,
        "bidaw_reuse_blocks_per_sec": bidaw_reuse_blocks_per_sec,
        "bidaw_small_reuse_ratio": args.bidaw_small_reuse_ratio,
        "bidaw_extreme_reuse_ratio": args.bidaw_extreme_reuse_ratio,
        "counter_scale": args.counter_scale,
    }
    payload = {"config": config, "rows": rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runtime_aligned_simulation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "runtime_aligned_simulation.md", payload)
    fieldnames = [
        "label",
        "policy",
        "allocation_mode",
        "capture_terminal_only",
        "require_selected_terminal",
        "max_cached",
        "index_tail",
        "requests",
        "demand_blocks",
        "hit_blocks",
        "miss_blocks",
        "hit_ratio",
        "request_full_hit_rate",
        "recent_miss_blocks",
        "evictions",
        "useful_evictions",
        "evictions_counter_scaled",
        "useful_evictions_counter_scaled",
        "recent_miss_blocks_counter_scaled",
        "miss_blocks_reduction_pct",
        "recent_miss_blocks_reduction_pct",
        "evictions_reduction_pct",
        "useful_evictions_reduction_pct",
        "hit_ratio_gain_pp",
        "attempts",
        "applied_allocations",
        "fallback_allocations",
        "runtime_shape_calls",
        "runtime_shape_requested_blocks",
        "runtime_shape_missing_real_blocks",
        "runtime_shape_filler_blocks",
        "fallback_no_cached_victims",
        "fallback_indexed_empty",
        "fallback_indexed_not_enough_candidates",
        "fallback_no_policy_score_improvement",
        "fallback_recent_applied_over_cap",
        "fallback_recent_no_cached_over_cap",
        "lru_cached_victims_seen",
        "candidate_count_total",
        "selected_replacement_victims",
        "capture_attempts",
        "capture_gate_active_requests",
        "captured_requests",
        "captured_blocks",
        "index_pushes",
        "index_pops",
        "index_invalid_pops",
        "index_nonterminal_pops",
    ]
    with (args.output_dir / "runtime_aligned_simulation.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(args.output_dir / "runtime_aligned_simulation.json")
    print(args.output_dir / "runtime_aligned_simulation.md")
    print(args.output_dir / "runtime_aligned_simulation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
