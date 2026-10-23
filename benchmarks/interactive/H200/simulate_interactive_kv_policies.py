#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRACE_DIR = Path(__file__).parent / "trace"
GAP_BUCKET_SEC = 30.0
VALUE_TERMINAL_BONUS_SEC = 100.0
VALUE_GAP_CAP_SEC = 180.0
VALUE_LOSS_WEIGHT = 1.0
SIZE_BONUS_MIN_GAP_SEC = 45.0
SIZE_HISTORY_WEIGHT = 8.0
SIZE_RESPONSE_WEIGHT = 2.0
SIZE_LOSS_WEIGHT = 1.0
LOST_TOKEN_BLOCK_SIZE_TOKENS = 1
BIDAW_REUSE_BLOCKS_PER_SEC = 0.0
BIDAW_CAPACITY_BLOCKS = 0
BIDAW_SMALL_REUSE_RATIO = 1.0
BIDAW_EXTREME_REUSE_RATIO = 3.0
ADAPTIVE_TINY_MAX_BLOCKS = 1000
ADAPTIVE_NANO_MAX_BLOCKS = 3000
ADAPTIVE_MICRO_MAX_BLOCKS = 6000
ADAPTIVE_TINY_POLICY = "expected_gap_bucket_response_tail_aware"
ADAPTIVE_NANO_POLICY = "expected_gap_cost_aware"
ADAPTIVE_MICRO_POLICY = "expected_gap_cost_aware"
ADAPTIVE_LARGE_POLICY = "expected_gap_bucket_response_tail_aware"


@dataclass
class Entry:
    user_id: str
    timestamp: int
    query_length: int
    response_length: int
    round_index: int
    next_gap_sec: int | None = None
    next_use_ts: int | None = None


@dataclass
class CacheBlock:
    user_id: str
    block_index: int
    total_blocks: int
    created_ts: int
    last_touch_order: int
    next_use_ts: int | None
    next_gap_sec: int | None
    response_length: int
    history_tokens_after: int


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[idx]


def describe(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "p50": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None}
    xs = [float(v) for v in values]
    return {
        "n": len(xs),
        "p50": percentile(xs, 0.50),
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
        "max": max(xs),
        "mean": statistics.mean(xs),
    }


def load_entries(trace_dir: Path, parts: list[int]) -> list[Entry]:
    entries: list[Entry] = []
    for part in parts:
        path = trace_dir / f"total_traces_part{part}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                fields = line.split()
                if len(fields) < 5:
                    continue
                entries.append(
                    Entry(
                        user_id=fields[0],
                        timestamp=int(fields[1]),
                        query_length=int(fields[2]),
                        response_length=int(fields[3]),
                        round_index=int(fields[4]),
                    )
                )
    entries.sort(key=lambda e: (e.timestamp, e.user_id, e.round_index))
    next_by_user: dict[str, Entry] = {}
    for entry in reversed(entries):
        nxt = next_by_user.get(entry.user_id)
        if nxt is not None:
            entry.next_gap_sec = nxt.timestamp - entry.timestamp
            entry.next_use_ts = nxt.timestamp
        next_by_user[entry.user_id] = entry
    return entries


def response_bucket_gap(response_length: int) -> float:
    if response_length <= 16:
        return 18.0
    if response_length <= 32:
        return 30.0
    if response_length <= 64:
        return 45.0
    if response_length <= 128:
        return 68.0
    if response_length <= 256:
        return 112.0
    return 191.0


def gap_bucket(gap_sec: float) -> float:
    if math.isinf(gap_sec):
        return gap_sec
    if GAP_BUCKET_SEC <= 0.0:
        return max(0.0, gap_sec)
    return math.floor(max(0.0, gap_sec) / GAP_BUCKET_SEC)


def resolve_capacity_adaptive_policy(policy: str, capacity_blocks: int) -> str:
    if policy != "expected_gap_capacity_adaptive_tail_aware":
        return policy
    if capacity_blocks <= ADAPTIVE_TINY_MAX_BLOCKS:
        return ADAPTIVE_TINY_POLICY
    if capacity_blocks <= ADAPTIVE_NANO_MAX_BLOCKS:
        return ADAPTIVE_NANO_POLICY
    if capacity_blocks <= ADAPTIVE_MICRO_MAX_BLOCKS:
        return ADAPTIVE_MICRO_POLICY
    return ADAPTIVE_LARGE_POLICY


def bidaw_eviction_tuple(
    *,
    gap_sec: float,
    terminal: bool,
    history_tokens: int,
    capacity_blocks: int,
) -> tuple[float, float, float, float]:
    if terminal:
        return 1.0, 0.0, float("inf"), float("inf")
    if math.isinf(gap_sec):
        return 1.0, 0.0, float("inf"), float("inf")

    if BIDAW_REUSE_BLOCKS_PER_SEC > 0.0:
        reuse_blocks = max(0.0, gap_sec) * BIDAW_REUSE_BLOCKS_PER_SEC
    else:
        reuse_blocks = (
            max(0.0, gap_sec)
            * max(1.0, history_tokens / max(1, LOST_TOKEN_BLOCK_SIZE_TOKENS))
        )
    capacity = int(capacity_blocks or BIDAW_CAPACITY_BLOCKS)
    if capacity <= 0:
        capacity = 1
    small_ratio = max(0.0, BIDAW_SMALL_REUSE_RATIO)
    extreme_ratio = max(small_ratio + 1e-6, BIDAW_EXTREME_REUSE_RATIO)
    small = capacity * small_ratio
    extreme = capacity * extreme_ratio
    if reuse_blocks <= small:
        hit_potential = 1.0
    elif reuse_blocks >= extreme:
        hit_potential = 0.0
    else:
        hit_potential = 1.0 - ((reuse_blocks - small) / (extreme - small))
    evict_score = 1.0 - hit_potential
    reuse_bucket = math.floor(max(0.0, reuse_blocks) / max(1.0, capacity))
    return evict_score, hit_potential, reuse_blocks, reuse_bucket


def select_window(entries: list[Entry], start: int, limit: int) -> list[Entry]:
    return [e for e in entries if start <= e.timestamp < limit]


def reannotate_next_use_within_selected(entries: list[Entry]) -> None:
    """Make next-use metadata refer only to the selected replay sample.

    The trace replay driver annotates next-use from the full trace before
    applying time windows or diagnostic request caps. That matches an online
    system that knows the logical future, but it can overstate measured reuse
    for capped probes whose later turns are not actually replayed. This helper
    lets the simulator evaluate the capped sample's own hit-rate surface.
    """
    next_by_user: dict[str, Entry] = {}
    for entry in reversed(entries):
        nxt = next_by_user.get(entry.user_id)
        if nxt is None:
            entry.next_gap_sec = None
            entry.next_use_ts = None
        else:
            entry.next_gap_sec = nxt.timestamp - entry.timestamp
            entry.next_use_ts = nxt.timestamp
        next_by_user[entry.user_id] = entry


def estimate_demand_blocks_per_sec(
    entries: list[Entry],
    *,
    block_size_tokens: int,
    replicas: int = 1,
) -> float:
    if not entries:
        return 0.0
    history_tokens_by_user: dict[str, int] = defaultdict(int)
    total_demand_blocks = 0
    for entry in entries:
        if entry.round_index == 0:
            history_tokens_by_user[entry.user_id] = 0
        history_before = int(history_tokens_by_user.get(entry.user_id, 0))
        total_demand_blocks += history_before // max(1, block_size_tokens)
        history_tokens_by_user[entry.user_id] = (
            history_before + int(entry.query_length) + int(entry.response_length)
        )
    duration = max(1.0, float(max(e.timestamp for e in entries) - min(e.timestamp for e in entries) + 1))
    return total_demand_blocks / duration / max(1, int(replicas))


def cap_real_multiturn_peak(
    entries: list[Entry],
    max_requests: int,
    turns_per_session: int,
) -> list[Entry]:
    if max_requests <= 0 or len(entries) <= max_requests:
        return entries
    turns_per_session = max(2, turns_per_session)
    session_cap = max(1, max_requests // turns_per_session)
    per_user: dict[str, list[int]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        per_user[entry.user_id].append(idx)

    prefixes: list[tuple[str, list[int]]] = []
    for uid, idxs in per_user.items():
        idxs.sort(key=lambda i: entries[i].timestamp)
        if not idxs or entries[idxs[0]].round_index != 0:
            continue
        prefix: list[int] = []
        expected_round = 0
        for idx in idxs:
            round_index = entries[idx].round_index
            if round_index == expected_round:
                prefix.append(idx)
                expected_round += 1
                if len(prefix) >= turns_per_session:
                    break
            elif round_index > expected_round:
                break
        if len(prefix) >= turns_per_session:
            prefixes.append((uid, prefix))

    events: list[tuple[float, int, int]] = []
    for pos, (_, prefix) in enumerate(prefixes):
        events.append((float(entries[prefix[0]].timestamp), 1, pos))
        events.append((float(entries[prefix[-1]].timestamp) + 1e-6, -1, pos))

    active: set[int] = set()
    best_active: set[int] = set()
    for _, delta, pos in sorted(events, key=lambda item: (item[0], -item[1])):
        if delta > 0:
            active.add(pos)
        else:
            active.discard(pos)
        if len(active) > len(best_active):
            best_active = set(active)

    def sort_key(pos: int) -> tuple[int, int, str]:
        uid, prefix = prefixes[pos]
        return entries[prefix[0]].timestamp, entries[prefix[-1]].timestamp, uid

    chosen = sorted(best_active, key=sort_key)[:session_cap]
    chosen_set = set(chosen)
    if len(chosen) < session_cap:
        for pos in sorted(range(len(prefixes)), key=sort_key):
            if pos in chosen_set:
                continue
            chosen.append(pos)
            chosen_set.add(pos)
            if len(chosen) >= session_cap:
                break
    selected: set[int] = set()
    for pos in chosen:
        selected.update(prefixes[pos][1])
    return [entries[i] for i in sorted(selected)]


def cap_real_multiturn_full_sessions_peak(
    entries: list[Entry],
    max_requests: int,
    min_turns: int,
    max_turns: int = 0,
) -> list[Entry]:
    if max_requests <= 0 or len(entries) <= max_requests:
        return entries
    min_turns = max(2, min_turns)
    max_turns = max(0, max_turns)
    if max_turns > 0:
        max_turns = max(min_turns, max_turns)
    per_user: dict[str, list[int]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        per_user[entry.user_id].append(idx)

    prefixes: list[tuple[str, list[int]]] = []
    for uid, idxs in per_user.items():
        idxs.sort(key=lambda i: entries[i].timestamp)
        if not idxs or entries[idxs[0]].round_index != 0:
            continue
        prefix: list[int] = []
        expected_round = 0
        for idx in idxs:
            round_index = entries[idx].round_index
            if round_index == expected_round:
                prefix.append(idx)
                expected_round += 1
                if max_turns > 0 and len(prefix) >= max_turns:
                    break
            elif round_index > expected_round:
                break
        if len(prefix) >= min_turns:
            prefixes.append((uid, prefix))

    events: list[tuple[float, int, int]] = []
    for pos, (_, prefix) in enumerate(prefixes):
        events.append((float(entries[prefix[0]].timestamp), 1, pos))
        events.append((float(entries[prefix[-1]].timestamp) + 1e-6, -1, pos))

    active: set[int] = set()
    best_active: set[int] = set()
    for _, delta, pos in sorted(events, key=lambda item: (item[0], -item[1])):
        if delta > 0:
            active.add(pos)
        else:
            active.discard(pos)
        if len(active) > len(best_active):
            best_active = set(active)

    def sort_key(pos: int) -> tuple[int, int, str]:
        uid, prefix = prefixes[pos]
        return entries[prefix[0]].timestamp, entries[prefix[-1]].timestamp, uid

    ordered = sorted(best_active, key=sort_key)
    ordered_set = set(ordered)
    ordered.extend(
        pos
        for pos in sorted(range(len(prefixes)), key=sort_key)
        if pos not in ordered_set
    )

    selected: set[int] = set()
    for pos in ordered:
        prefix = prefixes[pos][1]
        if len(selected) + len(prefix) <= max_requests:
            selected.update(prefix)
        elif not selected:
            selected.update(prefix[:max_requests])
            break
    return [entries[i] for i in sorted(selected)]


def cap_real_multiturn_full_sessions(
    entries: list[Entry],
    max_requests: int,
    min_turns: int,
    max_turns: int = 0,
) -> list[Entry]:
    if max_requests <= 0 or len(entries) <= max_requests:
        return entries
    min_turns = max(2, min_turns)
    max_turns = max(0, max_turns)
    if max_turns > 0:
        max_turns = max(min_turns, max_turns)
    per_user: dict[str, list[int]] = defaultdict(list)
    first_pos: dict[str, int] = {}
    for idx, entry in enumerate(entries):
        per_user[entry.user_id].append(idx)
        first_pos.setdefault(entry.user_id, idx)

    prefixes: list[tuple[str, list[int]]] = []
    for uid in sorted(per_user, key=lambda u: first_pos[u]):
        idxs = sorted(per_user[uid], key=lambda i: entries[i].timestamp)
        if not idxs or entries[idxs[0]].round_index != 0:
            continue
        prefix: list[int] = []
        expected_round = 0
        for idx in idxs:
            round_index = entries[idx].round_index
            if round_index == expected_round:
                prefix.append(idx)
                expected_round += 1
                if max_turns > 0 and len(prefix) >= max_turns:
                    break
            elif round_index > expected_round:
                break
        if len(prefix) >= min_turns:
            prefixes.append((uid, prefix))

    selected: set[int] = set()
    for _, prefix in sorted(
        prefixes,
        key=lambda item: (
            entries[item[1][0]].timestamp,
            entries[item[1][-1]].timestamp,
            item[0],
        ),
    ):
        if len(selected) + len(prefix) <= max_requests:
            selected.update(prefix)
        elif not selected:
            selected.update(prefix[:max_requests])
            break
    return [entries[i] for i in sorted(selected)]


def cap_session_balanced(
    entries: list[Entry],
    max_requests: int,
    min_turns: int,
) -> list[Entry]:
    if max_requests <= 0 or len(entries) <= max_requests:
        return entries
    min_turns = max(2, min_turns)
    per_user: dict[str, list[int]] = defaultdict(list)
    first_pos: dict[str, int] = {}
    for idx, entry in enumerate(entries):
        per_user[entry.user_id].append(idx)
        first_pos.setdefault(entry.user_id, idx)
    eligible_users = [
        uid for uid, idxs in per_user.items()
        if len(idxs) >= min_turns
    ]
    eligible_users.sort(key=lambda uid: first_pos[uid])
    selected: set[int] = set()
    for uid in eligible_users:
        needed = per_user[uid][:min_turns]
        if len(selected) + len(needed) > max_requests:
            break
        selected.update(needed)
    if len(selected) < max_requests:
        for idx in range(len(entries)):
            selected.add(idx)
            if len(selected) >= max_requests:
                break
    return [entries[i] for i in sorted(selected)]


def score_block(policy: str, block: CacheBlock, now_ts: int) -> tuple[Any, ...]:
    terminal = block.next_use_ts is None
    true_gap = float("inf") if terminal else max(0.0, float(block.next_use_ts - now_ts))
    expected_gap = (
        float("inf")
        if terminal
        else (
            float(block.next_gap_sec)
            if block.next_gap_sec is not None
            else response_bucket_gap(block.response_length)
        )
    )
    total_blocks = max(1, block.total_blocks)
    depth_ratio = block.block_index / total_blocks
    suffix_blocks = max(1, total_blocks - block.block_index + 1)
    lost_tokens = suffix_blocks * max(1, int(LOST_TOKEN_BLOCK_SIZE_TOKENS))
    history_cost = math.log1p(max(1, block.history_tokens_after))
    prefetch_protection_rank = 1

    def size_bonus() -> float:
        if (
            not terminal
            and not math.isinf(expected_gap)
            and expected_gap < max(0.0, SIZE_BONUS_MIN_GAP_SEC)
        ):
            return 0.0
        return (
            max(0.0, SIZE_HISTORY_WEIGHT)
            * math.log1p(max(0, block.history_tokens_after) / 64.0)
            + max(0.0, SIZE_RESPONSE_WEIGHT)
            * math.log1p(max(0, block.response_length))
        )

    def value_cost_score() -> tuple[float, float, float, float]:
        if terminal:
            gap_value = max(0.0, VALUE_TERMINAL_BONUS_SEC)
        elif math.isinf(expected_gap):
            gap_value = max(0.0, VALUE_GAP_CAP_SEC)
        else:
            gap_value = max(0.0, expected_gap)
            if VALUE_GAP_CAP_SEC > 0.0:
                gap_value = min(gap_value, VALUE_GAP_CAP_SEC)
        bonus = size_bonus()
        loss_penalty = max(0.0, VALUE_LOSS_WEIGHT) * math.log1p(max(0, lost_tokens))
        return gap_value + bonus - loss_penalty, gap_value, bonus, loss_penalty

    if policy == "lru":
        return (-block.last_touch_order,)
    if policy == "oracle_next_use":
        return (1 if terminal else 0, true_gap, depth_ratio, -lost_tokens)
    if policy == "response_len_tail_aware":
        return (1 if terminal else 0, block.response_length, depth_ratio, -lost_tokens)
    if policy == "near_gap_response_len_tail_aware":
        near = expected_gap <= 45.0 and not terminal
        return (
            1 if terminal else 0,
            0 if near else 1,
            block.response_length,
            expected_gap,
            depth_ratio,
            -lost_tokens,
        )
    if policy == "expected_gap_tail_aware":
        return (1 if terminal else 0, expected_gap, depth_ratio, -lost_tokens)
    if policy == "expected_gap_bucket_response_tail_aware":
        return (
            1 if terminal else 0,
            gap_bucket(expected_gap),
            block.response_length,
            depth_ratio,
            -lost_tokens,
            expected_gap,
        )
    if policy == "expected_gap_bucket_response_value_tail_aware":
        value_score, gap_value, bonus, loss_penalty = value_cost_score()
        return (
            1 if terminal else 0,
            gap_bucket(expected_gap),
            block.response_length,
            value_score,
            gap_value,
            bonus,
            -loss_penalty,
            depth_ratio,
            -lost_tokens,
            expected_gap,
            block.history_tokens_after,
        )
    if policy == "expected_gap_bucket_response_loss_tail_aware":
        return (
            1 if terminal else 0,
            gap_bucket(expected_gap),
            block.response_length,
            -lost_tokens,
            depth_ratio,
            expected_gap,
        )
    if policy == "expected_gap_bucket_response_near_cost_tail_aware":
        bucket = gap_bucket(expected_gap)
        near = bucket <= 1 and not terminal
        near_cost = expected_gap - math.log1p(lost_tokens) if near else 0.0
        return (
            1 if terminal else 0,
            bucket,
            block.response_length,
            near_cost,
            depth_ratio,
            -lost_tokens,
            expected_gap,
        )
    if policy == "expected_gap_bucket_response_cost_tail_aware":
        cost_score = float("inf") if terminal else expected_gap - math.log1p(lost_tokens)
        return (
            1 if terminal else 0,
            gap_bucket(expected_gap),
            block.response_length,
            cost_score,
            depth_ratio,
            -lost_tokens,
            expected_gap,
        )
    if policy == "expected_gap_bucket_near_cost_tail_aware":
        bucket = gap_bucket(expected_gap)
        near = bucket <= 1 and not terminal
        near_cost = (
            expected_gap - math.log1p(lost_tokens)
            if near else 0.0
        )
        response_score = 0 if near else block.response_length
        return (
            1 if terminal else 0,
            bucket,
            near_cost,
            response_score,
            depth_ratio,
            -lost_tokens,
            expected_gap,
            block.response_length,
        )
    if policy == "expected_gap_bucket_near_gap_tail_aware":
        bucket = gap_bucket(expected_gap)
        near = bucket <= 1 and not terminal
        near_gap = expected_gap if near else 0.0
        response_score = 0 if near else block.response_length
        return (
            1 if terminal else 0,
            bucket,
            near_gap,
            response_score,
            depth_ratio,
            -lost_tokens,
            expected_gap,
            block.response_length,
        )
    if policy == "expected_gap_bucket_cost_tail_aware":
        cost_score = float("inf") if terminal else expected_gap - math.log1p(lost_tokens)
        return (
            1 if terminal else 0,
            gap_bucket(expected_gap),
            cost_score,
            block.response_length,
            depth_ratio,
            -lost_tokens,
            expected_gap,
        )
    if policy == "expected_gap_bucket_tail_aware":
        return (
            1 if terminal else 0,
            gap_bucket(expected_gap),
            depth_ratio,
            -lost_tokens,
            block.response_length,
            expected_gap,
        )
    if policy == "expected_gap_cost_aware":
        score = float("inf") if terminal else expected_gap - math.log1p(lost_tokens)
        return (
            prefetch_protection_rank,
            1 if terminal else 0,
            score,
            expected_gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            block.response_length,
        )
    if policy in {"bidaw_answer_potential", "bidaw_hit_potential"}:
        gap = response_bucket_gap(block.response_length)
        if policy == "bidaw_hit_potential":
            gap = expected_gap
        evict_score, hit_potential, reuse_blocks, reuse_bucket = bidaw_eviction_tuple(
            gap_sec=gap,
            terminal=terminal,
            history_tokens=block.history_tokens_after,
            capacity_blocks=BIDAW_CAPACITY_BLOCKS,
        )
        return (
            prefetch_protection_rank,
            1 if terminal else 0,
            evict_score,
            reuse_bucket,
            -hit_potential,
            gap,
            reuse_blocks,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            block.response_length,
            block.history_tokens_after,
        )
    if policy == "expected_gap_size_cost_aware":
        loss_penalty = max(0.0, SIZE_LOSS_WEIGHT) * math.log1p(lost_tokens)
        bonus = size_bonus()
        score = bonus if terminal else expected_gap + bonus - loss_penalty
        return (
            prefetch_protection_rank,
            1 if terminal else 0,
            score,
            expected_gap,
            bonus,
            block.response_length,
            block.history_tokens_after,
            depth_ratio,
            -suffix_blocks,
            -lost_tokens,
        )
    if policy == "expected_gap_value_cost_aware":
        value_score, gap_value, bonus, loss_penalty = value_cost_score()
        return (
            value_score,
            gap_value,
            bonus,
            -loss_penalty,
            block.response_length,
            block.history_tokens_after,
            depth_ratio,
            -suffix_blocks,
            -lost_tokens,
            1 if terminal else 0,
        )
    if policy == "expected_gap_cost_response_tail_aware":
        score = float("inf") if terminal else expected_gap - math.log1p(lost_tokens)
        return (
            1 if terminal else 0,
            score,
            block.response_length,
            expected_gap,
            depth_ratio,
            -lost_tokens,
        )
    if policy == "history_cost_aware":
        score = float("inf") if terminal else expected_gap - history_cost
        return (
            prefetch_protection_rank,
            1 if terminal else 0,
            score,
            expected_gap,
            depth_ratio,
            -lost_tokens,
            -suffix_blocks,
            block.response_length,
        )
    if policy == "protect_near_gap":
        near = expected_gap <= 45.0 and not terminal
        return (0 if near else 1, 1 if terminal else 0, expected_gap, depth_ratio, -lost_tokens)
    raise ValueError(f"unknown policy: {policy}")


def simulate_policy(
    entries: list[Entry],
    *,
    policy: str,
    capacity_blocks: int,
    block_size_tokens: int,
) -> dict[str, Any]:
    global BIDAW_CAPACITY_BLOCKS
    BIDAW_CAPACITY_BLOCKS = int(capacity_blocks)
    effective_policy = resolve_capacity_adaptive_policy(policy, capacity_blocks)
    cache: dict[tuple[str, int], CacheBlock] = {}
    heap: list[tuple[tuple[Any, ...], int, tuple[str, int]]] = []
    versions: defaultdict[tuple[str, int], int] = defaultdict(int)
    history_tokens_by_user: dict[str, int] = defaultdict(int)
    touch_order = 0
    demand_blocks_total = 0
    hit_blocks_total = 0
    request_full_hits = 0
    request_partial_hits = 0
    useful_evictions = 0
    total_evictions = 0
    occupancy_peak = 0
    per_request_hit_blocks: list[int] = []
    per_request_demand_blocks: list[int] = []

    def heap_key(score: tuple[Any, ...]) -> tuple[Any, ...]:
        out: list[Any] = []
        for value in score:
            if isinstance(value, (int, float)):
                out.append(-float(value))
            else:
                out.append(value)
        return tuple(out)

    def push_block(key: tuple[str, int], block: CacheBlock, now_ts: int) -> None:
        versions[key] += 1
        heapq.heappush(
            heap,
            (heap_key(score_block(effective_policy, block, now_ts)), versions[key], key),
        )

    def evict_until_fits(now_ts: int) -> None:
        nonlocal total_evictions, useful_evictions
        while len(cache) > capacity_blocks:
            victim_key: tuple[str, int] | None = None
            victim: CacheBlock | None = None
            while heap:
                _, version, key = heapq.heappop(heap)
                block = cache.get(key)
                if block is None or versions[key] != version:
                    continue
                victim_key = key
                victim = block
                break
            if victim_key is None or victim is None:
                break
            if victim.next_use_ts is not None and victim.next_use_ts > now_ts:
                useful_evictions += 1
            total_evictions += 1
            del cache[victim_key]

    for entry in entries:
        touch_order += 1
        if entry.round_index == 0:
            for key in [k for k in cache if k[0] == entry.user_id]:
                del cache[key]
            history_tokens_by_user[entry.user_id] = 0

        history_before = int(history_tokens_by_user.get(entry.user_id, 0))
        demand_blocks = history_before // block_size_tokens
        contiguous_hits = 0
        for block_index in range(1, demand_blocks + 1):
            block = cache.get((entry.user_id, block_index))
            if block is None:
                break
            block.last_touch_order = touch_order
            push_block((entry.user_id, block_index), block, entry.timestamp)
            contiguous_hits += 1

        demand_blocks_total += demand_blocks
        hit_blocks_total += contiguous_hits
        per_request_hit_blocks.append(contiguous_hits)
        per_request_demand_blocks.append(demand_blocks)
        if demand_blocks > 0 and contiguous_hits == demand_blocks:
            request_full_hits += 1
        if contiguous_hits > 0:
            request_partial_hits += 1

        history_after = history_before + int(entry.query_length) + int(entry.response_length)
        total_blocks_after = history_after // block_size_tokens
        for block_index in range(1, total_blocks_after + 1):
            key = (entry.user_id, block_index)
            existing = cache.get(key)
            if existing is None:
                cache[key] = CacheBlock(
                    user_id=entry.user_id,
                    block_index=block_index,
                    total_blocks=total_blocks_after,
                    created_ts=entry.timestamp,
                    last_touch_order=touch_order,
                    next_use_ts=entry.next_use_ts,
                    next_gap_sec=entry.next_gap_sec,
                    response_length=entry.response_length,
                    history_tokens_after=history_after,
                )
                push_block(key, cache[key], entry.timestamp)
            else:
                existing.total_blocks = total_blocks_after
                existing.last_touch_order = touch_order
                existing.next_use_ts = entry.next_use_ts
                existing.next_gap_sec = entry.next_gap_sec
                existing.response_length = entry.response_length
                existing.history_tokens_after = history_after
                push_block(key, existing, entry.timestamp)
        history_tokens_by_user[entry.user_id] = history_after
        evict_until_fits(entry.timestamp)
        occupancy_peak = max(occupancy_peak, len(cache))

    hit_ratio = hit_blocks_total / demand_blocks_total if demand_blocks_total else 0.0
    requests_with_demand = sum(1 for x in per_request_demand_blocks if x > 0)
    return {
        "policy": policy,
        "effective_policy": effective_policy,
        "capacity_blocks": capacity_blocks,
        "capacity_tokens": capacity_blocks * block_size_tokens,
        "requests": len(entries),
        "requests_with_history_blocks": requests_with_demand,
        "demand_blocks": demand_blocks_total,
        "hit_blocks": hit_blocks_total,
        "miss_blocks": max(0, demand_blocks_total - hit_blocks_total),
        "hit_ratio": hit_ratio,
        "request_full_hit_rate": (
            request_full_hits / requests_with_demand if requests_with_demand else 0.0
        ),
        "request_any_hit_rate": request_partial_hits / len(entries) if entries else 0.0,
        "evictions": total_evictions,
        "useful_evictions": useful_evictions,
        "occupancy_peak_blocks": occupancy_peak,
        "hit_blocks_per_request": describe(per_request_hit_blocks),
        "demand_blocks_per_request": describe(per_request_demand_blocks),
    }


def write_markdown(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Interactive KV Policy Simulation",
        "",
        f"- window: [{config['time_start']}, {config['time_limit']})",
        f"- entries_after_cap: {config['entries_after_cap']}",
        f"- block_size_tokens: {config['block_size_tokens']}",
        "",
        "| capacity_tokens | policy | effective_policy | hit_ratio | full_hit_rate | evictions | useful_evictions | demand_blocks | hit_blocks |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['capacity_tokens']} | {row['policy']} | "
            f"{row.get('effective_policy', row['policy'])} | "
            f"{row['hit_ratio']:.4f} | {row['request_full_hit_rate']:.4f} | "
            f"{row['evictions']} | {row['useful_evictions']} | "
            f"{row['demand_blocks']} | {row['hit_blocks']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global GAP_BUCKET_SEC
    global VALUE_TERMINAL_BONUS_SEC
    global VALUE_GAP_CAP_SEC
    global VALUE_LOSS_WEIGHT
    global SIZE_BONUS_MIN_GAP_SEC
    global SIZE_HISTORY_WEIGHT
    global SIZE_RESPONSE_WEIGHT
    global SIZE_LOSS_WEIGHT
    global BIDAW_REUSE_BLOCKS_PER_SEC
    global BIDAW_SMALL_REUSE_RATIO
    global BIDAW_EXTREME_REUSE_RATIO
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--parts", type=int, nargs="+", default=[1])
    parser.add_argument("--time-start", type=int, default=6780)
    parser.add_argument("--time-limit", type=int, default=6840)
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument(
        "--capacity-tokens",
        type=int,
        nargs="+",
        default=[687568],
        help="One or more simulated cache capacities in tokens.",
    )
    parser.add_argument(
        "--policies",
        default=(
            "lru,oracle_next_use,response_len_tail_aware,"
            "near_gap_response_len_tail_aware,"
            "expected_gap_bucket_response_tail_aware,"
            "expected_gap_bucket_response_value_tail_aware,"
            "expected_gap_bucket_response_near_cost_tail_aware,"
            "expected_gap_bucket_response_cost_tail_aware,"
            "expected_gap_bucket_near_cost_tail_aware,"
            "expected_gap_bucket_near_gap_tail_aware,"
            "expected_gap_bucket_cost_tail_aware,"
            "expected_gap_bucket_tail_aware,"
            "expected_gap_tail_aware,expected_gap_cost_aware,"
            "bidaw_answer_potential,bidaw_hit_potential,"
            "expected_gap_size_cost_aware,expected_gap_value_cost_aware,"
            "expected_gap_cost_response_tail_aware,"
            "expected_gap_capacity_adaptive_tail_aware,"
            "history_cost_aware,protect_near_gap"
        ),
    )
    parser.add_argument(
        "--gap-bucket-sec",
        type=float,
        default=30.0,
        help="Gap bucket width for expected_gap_bucket_* policies.",
    )
    parser.add_argument(
        "--value-terminal-bonus-sec",
        type=float,
        default=VALUE_TERMINAL_BONUS_SEC,
        help="Terminal bonus for expected_gap_value_cost_aware.",
    )
    parser.add_argument(
        "--value-gap-cap-sec",
        type=float,
        default=VALUE_GAP_CAP_SEC,
        help="Finite-gap cap for expected_gap_value_cost_aware.",
    )
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=VALUE_LOSS_WEIGHT,
        help="Loss penalty weight for expected_gap_value_cost_aware.",
    )
    parser.add_argument(
        "--size-bonus-min-gap-sec",
        type=float,
        default=SIZE_BONUS_MIN_GAP_SEC,
        help="Only apply size bonus to nonterminal blocks at or beyond this gap.",
    )
    parser.add_argument(
        "--size-history-weight",
        type=float,
        default=SIZE_HISTORY_WEIGHT,
        help="History-token weight for size/value-cost policies.",
    )
    parser.add_argument(
        "--size-response-weight",
        type=float,
        default=SIZE_RESPONSE_WEIGHT,
        help="Response-length weight for size/value-cost policies.",
    )
    parser.add_argument(
        "--size-loss-weight",
        type=float,
        default=SIZE_LOSS_WEIGHT,
        help="Loss penalty weight for expected_gap_size_cost_aware.",
    )
    parser.add_argument(
        "--bidaw-reuse-blocks-per-sec",
        type=float,
        default=0.0,
        help=(
            "Reuse-distance rate for bidaw_* policies. 0 estimates it from "
            "the selected window."
        ),
    )
    parser.add_argument(
        "--bidaw-small-reuse-ratio",
        type=float,
        default=BIDAW_SMALL_REUSE_RATIO,
        help="Reuse distance <= ratio*capacity is treated as high hit potential.",
    )
    parser.add_argument(
        "--bidaw-extreme-reuse-ratio",
        type=float,
        default=BIDAW_EXTREME_REUSE_RATIO,
        help="Reuse distance >= ratio*capacity is treated as zero hit potential.",
    )
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument(
        "--max-requests-policy",
        choices=[
            "first",
            "real_multiturn_peak",
            "real_multiturn_full_sessions",
            "real_multiturn_full_sessions_peak",
            "session_balanced",
        ],
        default="first",
    )
    parser.add_argument(
        "--next-use-scope",
        choices=["full_trace", "selected"],
        default="full_trace",
        help=(
            "full_trace matches trace_replay_driver metadata; selected "
            "recomputes next-use after window/cap for capped probe analysis."
        ),
    )
    parser.add_argument("--session-balanced-min-turns", type=int, default=3)
    parser.add_argument(
        "--session-balanced-max-turns",
        type=int,
        default=0,
        help=(
            "Optional per-session turn cap for real_multiturn_full_sessions "
            "policies. 0 keeps the previous full-session behavior."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    GAP_BUCKET_SEC = float(args.gap_bucket_sec)
    VALUE_TERMINAL_BONUS_SEC = float(args.value_terminal_bonus_sec)
    VALUE_GAP_CAP_SEC = float(args.value_gap_cap_sec)
    VALUE_LOSS_WEIGHT = float(args.value_loss_weight)
    SIZE_BONUS_MIN_GAP_SEC = float(args.size_bonus_min_gap_sec)
    SIZE_HISTORY_WEIGHT = float(args.size_history_weight)
    SIZE_RESPONSE_WEIGHT = float(args.size_response_weight)
    SIZE_LOSS_WEIGHT = float(args.size_loss_weight)
    BIDAW_SMALL_REUSE_RATIO = float(args.bidaw_small_reuse_ratio)
    BIDAW_EXTREME_REUSE_RATIO = float(args.bidaw_extreme_reuse_ratio)

    all_entries = load_entries(args.trace_dir, args.parts)
    entries = select_window(all_entries, args.time_start, args.time_limit)
    entries_before_cap = len(entries)
    if args.max_requests > 0 and len(entries) > args.max_requests:
        if args.max_requests_policy == "real_multiturn_peak":
            entries = cap_real_multiturn_peak(
                entries,
                args.max_requests,
                args.session_balanced_min_turns,
            )
        elif args.max_requests_policy == "real_multiturn_full_sessions":
            entries = cap_real_multiturn_full_sessions(
                entries,
                args.max_requests,
                args.session_balanced_min_turns,
                args.session_balanced_max_turns,
            )
        elif args.max_requests_policy == "real_multiturn_full_sessions_peak":
            entries = cap_real_multiturn_full_sessions_peak(
                entries,
                args.max_requests,
                args.session_balanced_min_turns,
                args.session_balanced_max_turns,
            )
        elif args.max_requests_policy == "session_balanced":
            entries = cap_session_balanced(
                entries,
                args.max_requests,
                args.session_balanced_min_turns,
            )
        else:
            entries = entries[:args.max_requests]
    if args.next_use_scope == "selected":
        reannotate_next_use_within_selected(entries)
    BIDAW_REUSE_BLOCKS_PER_SEC = float(args.bidaw_reuse_blocks_per_sec)
    if BIDAW_REUSE_BLOCKS_PER_SEC <= 0.0:
        BIDAW_REUSE_BLOCKS_PER_SEC = estimate_demand_blocks_per_sec(
            entries,
            block_size_tokens=args.block_size_tokens,
        )

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    rows: list[dict[str, Any]] = []
    for capacity_tokens in args.capacity_tokens:
        capacity_blocks = max(1, capacity_tokens // args.block_size_tokens)
        for policy in policies:
            rows.append(
                simulate_policy(
                    entries,
                    policy=policy,
                    capacity_blocks=capacity_blocks,
                    block_size_tokens=args.block_size_tokens,
                )
            )
    rows.sort(key=lambda row: (row["capacity_tokens"], -row["hit_ratio"], row["policy"]))

    config = {
        "trace_dir": str(args.trace_dir),
        "parts": args.parts,
        "time_start": args.time_start,
        "time_limit": args.time_limit,
        "entries_before_cap": entries_before_cap,
        "entries_after_cap": len(entries),
        "max_requests": args.max_requests,
        "max_requests_policy": args.max_requests_policy,
        "next_use_scope": args.next_use_scope,
        "session_balanced_min_turns": args.session_balanced_min_turns,
        "session_balanced_max_turns": args.session_balanced_max_turns,
        "block_size_tokens": args.block_size_tokens,
        "gap_bucket_sec": args.gap_bucket_sec,
        "value_terminal_bonus_sec": args.value_terminal_bonus_sec,
        "value_gap_cap_sec": args.value_gap_cap_sec,
        "value_loss_weight": args.value_loss_weight,
        "size_bonus_min_gap_sec": args.size_bonus_min_gap_sec,
        "size_history_weight": args.size_history_weight,
        "size_response_weight": args.size_response_weight,
        "size_loss_weight": args.size_loss_weight,
        "bidaw_reuse_blocks_per_sec": BIDAW_REUSE_BLOCKS_PER_SEC,
        "bidaw_small_reuse_ratio": BIDAW_SMALL_REUSE_RATIO,
        "bidaw_extreme_reuse_ratio": BIDAW_EXTREME_REUSE_RATIO,
        "capacity_tokens": args.capacity_tokens,
        "policies": policies,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "simulation.json").write_text(
        json.dumps({"config": config, "results": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (args.output_dir / "simulation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "capacity_tokens",
                "policy",
                "requests",
                "requests_with_history_blocks",
                "demand_blocks",
                "hit_blocks",
                "miss_blocks",
                "hit_ratio",
                "request_full_hit_rate",
                "request_any_hit_rate",
                "evictions",
                "useful_evictions",
                "occupancy_peak_blocks",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    write_markdown(args.output_dir / "simulation.md", config, rows)
    print(args.output_dir / "simulation.json")
    print(args.output_dir / "simulation.md")
    print(args.output_dir / "simulation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
