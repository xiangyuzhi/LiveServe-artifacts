#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import simulate_interactive_kv_policies as simpol


TRACE_DIR = Path(__file__).parent / "trace"


@dataclass
class Event:
    kind: str
    user_id: str
    timestamp: int
    round_index: int
    history_before_tokens: int
    query_length: int
    response_length: int
    trace_entry_index: int
    next_use_ts: int | None
    next_gap_sec: int | None
    rank: int


class CacheSim:
    def __init__(self, *, policy: str, capacity_blocks: int, block_size_tokens: int):
        self.policy = simpol.resolve_capacity_adaptive_policy(policy, capacity_blocks)
        self.capacity_blocks = int(capacity_blocks)
        self.block_size_tokens = int(block_size_tokens)
        self.cache: dict[tuple[str, int], simpol.CacheBlock] = {}
        self.heap: list[tuple[tuple[Any, ...], int, tuple[str, int]]] = []
        self.versions: defaultdict[tuple[str, int], int] = defaultdict(int)
        self.touch_order = 0
        self.evictions_by_phase: defaultdict[str, int] = defaultdict(int)
        self.useful_evictions_by_phase: defaultdict[str, int] = defaultdict(int)
        self.occupancy_peak = 0

    @staticmethod
    def _heap_key(score: tuple[Any, ...]) -> tuple[Any, ...]:
        out: list[Any] = []
        for value in score:
            if isinstance(value, (int, float)):
                out.append(-float(value))
            else:
                out.append(value)
        return tuple(out)

    def _push(self, key: tuple[str, int], block: simpol.CacheBlock, now_ts: int) -> None:
        self.versions[key] += 1
        heapq.heappush(
            self.heap,
            (
                self._heap_key(simpol.score_block(self.policy, block, now_ts)),
                self.versions[key],
                key,
            ),
        )

    def _evict_until_fits(self, now_ts: int, phase: str) -> None:
        while len(self.cache) > self.capacity_blocks:
            victim_key: tuple[str, int] | None = None
            victim: simpol.CacheBlock | None = None
            while self.heap:
                _, version, key = heapq.heappop(self.heap)
                block = self.cache.get(key)
                if block is None or self.versions[key] != version:
                    continue
                victim_key = key
                victim = block
                break
            if victim_key is None or victim is None:
                break
            if victim.next_use_ts is not None and victim.next_use_ts > now_ts:
                self.useful_evictions_by_phase[phase] += 1
            self.evictions_by_phase[phase] += 1
            del self.cache[victim_key]

    def apply_event(self, event: Event) -> dict[str, int]:
        self.touch_order += 1
        phase = event.kind
        history_before = max(0, int(event.history_before_tokens))
        demand_blocks = 0 if phase == "prewarm" else history_before // self.block_size_tokens
        contiguous_hits = 0
        for block_index in range(1, demand_blocks + 1):
            block = self.cache.get((event.user_id, block_index))
            if block is None:
                break
            block.last_touch_order = self.touch_order
            self._push((event.user_id, block_index), block, event.timestamp)
            contiguous_hits += 1

        history_after = history_before
        if phase == "live":
            history_after += max(0, int(event.query_length)) + max(0, int(event.response_length))
        total_blocks_after = history_after // self.block_size_tokens
        for block_index in range(1, total_blocks_after + 1):
            key = (event.user_id, block_index)
            existing = self.cache.get(key)
            if existing is None:
                block = simpol.CacheBlock(
                    user_id=event.user_id,
                    block_index=block_index,
                    total_blocks=total_blocks_after,
                    created_ts=event.timestamp,
                    last_touch_order=self.touch_order,
                    next_use_ts=event.next_use_ts,
                    next_gap_sec=event.next_gap_sec,
                    response_length=event.response_length,
                    history_tokens_after=history_after,
                )
                self.cache[key] = block
                self._push(key, block, event.timestamp)
            else:
                existing.total_blocks = total_blocks_after
                existing.last_touch_order = self.touch_order
                existing.next_use_ts = event.next_use_ts
                existing.next_gap_sec = event.next_gap_sec
                existing.response_length = event.response_length
                existing.history_tokens_after = history_after
                self._push(key, existing, event.timestamp)

        self._evict_until_fits(event.timestamp, phase)
        self.occupancy_peak = max(self.occupancy_peak, len(self.cache))
        return {
            "demand_blocks": demand_blocks,
            "hit_blocks": contiguous_hits,
            "miss_blocks": max(0, demand_blocks - contiguous_hits),
            "full_hit": int(demand_blocks > 0 and contiguous_hits == demand_blocks),
            "has_demand": int(demand_blocks > 0),
        }


def stable_user_hash(uid: str) -> int:
    digest = hashlib.blake2b(str(uid).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


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
        return {"n": 0, "p50": None, "p90": None, "p95": None, "max": None, "mean": None}
    xs = [float(v) for v in values]
    return {
        "n": len(xs),
        "p50": percentile(xs, 0.50),
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
        "max": max(xs),
        "mean": statistics.mean(xs),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_trace_next_map(trace_dir: Path, parts: list[int]) -> dict[tuple[str, int, int], tuple[int | None, int | None]]:
    entries = simpol.load_entries(trace_dir, parts)
    return {
        (entry.user_id, entry.timestamp, entry.round_index): (
            entry.next_use_ts,
            entry.next_gap_sec,
        )
        for entry in entries
    }


def build_events(
    *,
    cell_dir: Path,
    trace_dir: Path,
    parts: list[int],
    prewarm_next_use_mode: str,
) -> tuple[list[Event], dict[str, Any]]:
    prefetch_rows = read_jsonl(cell_dir / "prefetch.jsonl")
    request_rows = read_jsonl(cell_dir / "per_request.jsonl")
    request_rows.sort(key=lambda r: int(r.get("trace_entry_index", 10**12)))

    first_live_by_user: dict[str, dict[str, Any]] = {}
    for row in request_rows:
        first_live_by_user.setdefault(str(row["user_id"]), row)

    next_map = load_trace_next_map(trace_dir, parts)
    events: list[Event] = []

    prewarm_rows = [
        row for row in prefetch_rows
        if row.get("record_kind") == "prewarm_history"
    ]
    prewarm_rows.sort(key=lambda r: int(r.get("prewarm_rank") or 0))
    for rank, row in enumerate(prewarm_rows, start=1):
        uid = str(row["user_id"])
        first_live = first_live_by_user.get(uid)
        next_use_ts: int | None = None
        next_gap_sec: int | None = None
        if prewarm_next_use_mode == "first_live" and first_live is not None:
            next_use_ts = int(first_live["timestamp"])
            next_gap_sec = max(0, int(first_live["timestamp"]) - int(row["timestamp"]))
        events.append(
            Event(
                kind="prewarm",
                user_id=uid,
                timestamp=int(row.get("timestamp") or 0),
                round_index=int(row.get("round_index") or 0),
                history_before_tokens=int(row.get("prewarm_history_tokens") or row.get("prefix_token_estimate") or 0),
                query_length=0,
                response_length=0,
                trace_entry_index=-1,
                next_use_ts=next_use_ts,
                next_gap_sec=next_gap_sec,
                rank=rank,
            )
        )

    for rank, row in enumerate(request_rows, start=1):
        uid = str(row["user_id"])
        ts = int(row["timestamp"])
        round_index = int(row["round_index"])
        next_use_ts, next_gap_sec = next_map.get((uid, ts, round_index), (None, None))
        history_before = row.get("kv_prefix_token_estimate")
        if history_before is None:
            history_before = max(0, int(row.get("input_tokens_total") or 0) - int(row.get("query_length") or 0))
        events.append(
            Event(
                kind="live",
                user_id=uid,
                timestamp=ts,
                round_index=round_index,
                history_before_tokens=int(history_before or 0),
                query_length=int(row.get("query_length") or 0),
                response_length=int(row.get("response_length") or 0),
                trace_entry_index=int(row.get("trace_entry_index") or -1),
                next_use_ts=next_use_ts,
                next_gap_sec=next_gap_sec,
                rank=rank,
            )
        )

    meta = {
        "prefetch_rows": len(prefetch_rows),
        "prewarm_rows": len(prewarm_rows),
        "request_rows": len(request_rows),
        "first_live_users": len(first_live_by_user),
        "prewarm_tokens": sum(e.history_before_tokens for e in events if e.kind == "prewarm"),
    }
    return events, meta


def replica_for_event(
    event: Event,
    *,
    assignment: str,
    replicas: int,
    absolute_index: int,
    live_index: int,
) -> int:
    if replicas <= 1:
        return 0
    if assignment == "user_hash":
        return stable_user_hash(event.user_id) % replicas
    if assignment == "round_robin":
        return absolute_index % replicas
    if assignment == "round_robin_reset_live":
        if event.kind == "live":
            return live_index % replicas
        return absolute_index % replicas
    raise ValueError(f"unknown assignment={assignment!r}")


def run_simulation(
    events: list[Event],
    *,
    policy: str,
    layout: str,
    replicas: int,
    capacity_blocks: int,
    block_size_tokens: int,
    assignment: str,
) -> dict[str, Any]:
    if layout == "global":
        sims = [CacheSim(policy=policy, capacity_blocks=capacity_blocks * replicas, block_size_tokens=block_size_tokens)]
    elif layout == "per_replica":
        sims = [CacheSim(policy=policy, capacity_blocks=capacity_blocks, block_size_tokens=block_size_tokens) for _ in range(replicas)]
    else:
        raise ValueError(f"unknown layout={layout!r}")

    live_index = -1
    demand_blocks = 0
    hit_blocks = 0
    full_hits = 0
    live_with_demand = 0
    live_miss_blocks_by_request: list[int] = []
    live_demand_blocks_by_request: list[int] = []
    live_hit_blocks_by_request: list[int] = []

    for absolute_index, event in enumerate(events):
        if event.kind == "live":
            live_index += 1
        rep = 0
        if layout == "per_replica":
            rep = replica_for_event(
                event,
                assignment=assignment,
                replicas=replicas,
                absolute_index=absolute_index,
                live_index=max(0, live_index),
            )
        out = sims[rep].apply_event(event)
        if event.kind == "live":
            demand_blocks += out["demand_blocks"]
            hit_blocks += out["hit_blocks"]
            full_hits += out["full_hit"]
            live_with_demand += out["has_demand"]
            live_miss_blocks_by_request.append(out["miss_blocks"])
            live_demand_blocks_by_request.append(out["demand_blocks"])
            live_hit_blocks_by_request.append(out["hit_blocks"])

    evictions_by_phase: defaultdict[str, int] = defaultdict(int)
    useful_by_phase: defaultdict[str, int] = defaultdict(int)
    occupancy_peaks = []
    final_occupancy = []
    for s in sims:
        for phase, value in s.evictions_by_phase.items():
            evictions_by_phase[phase] += value
        for phase, value in s.useful_evictions_by_phase.items():
            useful_by_phase[phase] += value
        occupancy_peaks.append(s.occupancy_peak)
        final_occupancy.append(len(s.cache))

    miss_blocks = max(0, demand_blocks - hit_blocks)
    return {
        "policy": policy,
        "effective_policy": sims[0].policy,
        "layout": layout,
        "assignment": assignment if layout == "per_replica" else "global",
        "replicas": replicas if layout == "per_replica" else 1,
        "capacity_blocks_per_replica": capacity_blocks,
        "capacity_blocks_total": capacity_blocks * (replicas if layout == "per_replica" else replicas),
        "capacity_tokens_total": capacity_blocks * block_size_tokens * (replicas if layout == "per_replica" else replicas),
        "live_demand_blocks": demand_blocks,
        "live_hit_blocks": hit_blocks,
        "live_miss_blocks": miss_blocks,
        "live_hit_ratio": hit_blocks / demand_blocks if demand_blocks else 0.0,
        "live_full_hit_rate": full_hits / live_with_demand if live_with_demand else 0.0,
        "live_requests_with_demand": live_with_demand,
        "prewarm_evictions": int(evictions_by_phase.get("prewarm", 0)),
        "live_evictions": int(evictions_by_phase.get("live", 0)),
        "prewarm_useful_evictions": int(useful_by_phase.get("prewarm", 0)),
        "live_useful_evictions": int(useful_by_phase.get("live", 0)),
        "occupancy_peak_blocks": sum(occupancy_peaks),
        "occupancy_peak_blocks_by_replica": occupancy_peaks,
        "final_occupancy_blocks_by_replica": final_occupancy,
        "live_miss_blocks_per_request": describe(live_miss_blocks_by_request),
        "live_demand_blocks_per_request": describe(live_demand_blocks_by_request),
        "live_hit_blocks_per_request": describe(live_hit_blocks_by_request),
    }


def load_online_summary(cell_dir: Path) -> dict[str, Any]:
    path = cell_dir / "summary.json"
    if not path.exists():
        return {}
    summary = json.loads(path.read_text(encoding="utf-8"))
    prewarm = (summary.get("replay") or {}).get("prewarm") or {}
    kv = summary.get("kv_lru") or {}
    return {
        "measurement_included": (summary.get("measurement") or {}).get("included_results"),
        "prewarm_gpu_evictions": (prewarm.get("kv_lru_delta") or {}).get("gpu_lru_eviction_blocks_total"),
        "prewarm_gpu_lookup_miss": (prewarm.get("kv_lru_delta") or {}).get("gpu_prefix_block_lookup_miss"),
        "live_gpu_evictions": kv.get("gpu_evictions"),
        "live_gpu_lookup_misses": kv.get("gpu_lookup_misses"),
        "live_recent_eviction_misses": kv.get("recent_eviction_misses"),
        "audio_ttfp_p90": ((summary.get("latency") or {}).get("audio_ttfp_sec") or {}).get("p90"),
        "e2e_p90": ((summary.get("latency") or {}).get("e2e_sec") or {}).get("p90"),
        "goodput_rps": (((summary.get("audio") or {}).get("streaming_continuity") or {}).get("goodput_rps")),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Compact Prewarm KV Simulation",
        "",
        f"- cell_dir: `{payload['cell_dir']}`",
        f"- prewarm_next_use_mode: `{payload['prewarm_next_use_mode']}`",
        f"- prewarm_rows: {payload['event_meta']['prewarm_rows']}",
        f"- request_rows: {payload['event_meta']['request_rows']}",
        f"- prewarm_tokens: {payload['event_meta']['prewarm_tokens']}",
        "",
        "| policy | layout | assignment | live_hit_ratio | live_miss_blocks | prewarm_evictions | live_evictions | occupancy_peak_blocks |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['policy']} | {row['layout']} | {row['assignment']} | "
            f"{row['live_hit_ratio']:.4f} | {row['live_miss_blocks']} | "
            f"{row['prewarm_evictions']} | {row['live_evictions']} | "
            f"{row['occupancy_peak_blocks']} |"
        )
    online = payload.get("online_summary") or {}
    if online:
        lines.extend(["", "## Online Summary", "", "```json", json.dumps(online, indent=2, sort_keys=True), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--parts", type=int, nargs="+", default=[1])
    parser.add_argument("--policies", default="lru,expected_gap_cost_aware,expected_gap_size_cost_aware")
    parser.add_argument("--layouts", default="global,per_replica")
    parser.add_argument("--assignments", default="user_hash,round_robin,round_robin_reset_live")
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--capacity-blocks-per-replica", type=int, default=5000)
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument(
        "--prewarm-next-use-mode",
        choices=["none", "first_live"],
        default="none",
        help="Whether prewarm-created blocks carry no reuse metadata or point to the first live turn.",
    )
    parser.add_argument("--gap-bucket-sec", type=float, default=30.0)
    parser.add_argument("--size-bonus-min-gap-sec", type=float, default=45.0)
    parser.add_argument("--size-history-weight", type=float, default=8.0)
    parser.add_argument("--size-response-weight", type=float, default=2.0)
    parser.add_argument("--size-loss-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    simpol.GAP_BUCKET_SEC = float(args.gap_bucket_sec)
    simpol.SIZE_BONUS_MIN_GAP_SEC = float(args.size_bonus_min_gap_sec)
    simpol.SIZE_HISTORY_WEIGHT = float(args.size_history_weight)
    simpol.SIZE_RESPONSE_WEIGHT = float(args.size_response_weight)
    simpol.SIZE_LOSS_WEIGHT = float(args.size_loss_weight)

    events, event_meta = build_events(
        cell_dir=args.cell_dir,
        trace_dir=args.trace_dir,
        parts=args.parts,
        prewarm_next_use_mode=args.prewarm_next_use_mode,
    )
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    layouts = [x.strip() for x in args.layouts.split(",") if x.strip()]
    assignments = [x.strip() for x in args.assignments.split(",") if x.strip()]

    rows: list[dict[str, Any]] = []
    for policy in policies:
        for layout in layouts:
            if layout == "global":
                rows.append(
                    run_simulation(
                        events,
                        policy=policy,
                        layout=layout,
                        replicas=args.replicas,
                        capacity_blocks=args.capacity_blocks_per_replica,
                        block_size_tokens=args.block_size_tokens,
                        assignment="global",
                    )
                )
                continue
            for assignment in assignments:
                rows.append(
                    run_simulation(
                        events,
                        policy=policy,
                        layout=layout,
                        replicas=args.replicas,
                        capacity_blocks=args.capacity_blocks_per_replica,
                        block_size_tokens=args.block_size_tokens,
                        assignment=assignment,
                    )
                )

    payload = {
        "cell_dir": str(args.cell_dir),
        "event_meta": event_meta,
        "prewarm_next_use_mode": args.prewarm_next_use_mode,
        "config": jsonable_config(args),
        "online_summary": load_online_summary(args.cell_dir),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "compact_prewarm_simulation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "compact_prewarm_simulation.md", payload)
    with (args.output_dir / "compact_prewarm_simulation.csv").open("w", encoding="utf-8") as f:
        cols = [
            "policy",
            "layout",
            "assignment",
            "live_hit_ratio",
            "live_miss_blocks",
            "prewarm_evictions",
            "live_evictions",
            "prewarm_useful_evictions",
            "live_useful_evictions",
            "occupancy_peak_blocks",
        ]
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
