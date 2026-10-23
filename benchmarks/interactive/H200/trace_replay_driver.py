#!/usr/bin/env python3
"""
Trace Replay Driver for Interactive Conversation Workload.

Replays multi-round conversation traces from the Interactive-conversation-workload
dataset against a vllm-omni server via OpenAI-compatible SSE streaming API.

Key features:
  - Trace-driven arrival: requests are issued at their original timestamps
  - Multi-turn state: per-user conversation history is maintained
  - Speedup factor: compress wall-clock time for faster experiments
  - Concurrency control: semaphore-based max_inflight limit
  - Metrics collection: TTFP, e2e latency, throughput, per-request JSONL output

Usage:
  # Smoke test (first 300s of part1, speedup=1x)
  python3 trace_replay_driver.py --mode smoke --url http://127.0.0.1:8666/v1/chat/completions

  # Mini run (first 1800s, speedup=5x)
  python3 trace_replay_driver.py --mode mini --speedup 5

  # Full part1 (speedup=10x)
  python3 trace_replay_driver.py --mode full_part1 --speedup 10 --max-inflight 12

  # Custom: specific time window and parts
  python3 trace_replay_driver.py --parts 1 2 --time-limit 600 --speedup 2
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from client_bargein import (
    BargeInPolicy,
    StreamResult,
    build_audio_payload,
    build_text_payload,
    close_response,
    consume_sse_audio,
    get_audio_continuity_threshold_s,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_URL = "http://127.0.0.1:8666/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Trace data location:
#   Default: <this script's dir>/trace/total_traces_part{N}.txt
#   Override via env LIVESERVE_TRACE_DIR (e.g. point to an external mount with parts 2-7).
TRACE_DIR = Path(
    os.environ.get("LIVESERVE_TRACE_DIR")
    or (Path(__file__).parent / "trace")
)

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group. "
    "You are helpful, harmless, and honest."
)

# Dummy vocabulary for generating synthetic prompts with controlled token count.
# We use short common words so that most tokenizers map them to single tokens.
_DUMMY_VOCAB = [
    "the", "a", "is", "of", "to", "in", "and", "for", "on", "it",
    "that", "this", "with", "as", "was", "are", "be", "at", "by", "an",
    "from", "or", "but", "not", "have", "had", "has", "do", "did", "will",
    "can", "may", "if", "so", "no", "up", "out", "all", "one", "two",
    "new", "old", "big", "get", "go", "see", "say", "use", "way", "day",
]

# Mode presets
MODE_PRESETS = {
    "smoke": {"time_limit": 300, "parts": [1]},
    "mini": {"time_limit": 1800, "parts": [1]},
    "full_part1": {"time_limit": None, "parts": [1]},
    "full_all": {"time_limit": None, "parts": [1, 2, 3, 4, 5, 6, 7]},
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TraceEntry:
    """A single line from the trace file."""
    user_id: str
    timestamp: int        # seconds from trace start
    query_length: int     # token count for user query
    response_length: int  # token count for assistant response
    round_index: int      # 0-based conversation round
    prev_gap_sec: int | None = None
    next_gap_sec: int | None = None
    expected_next_use_ts: int | None = None


AudioMetadataMap = dict[tuple[str, int], dict[str, float]]


@dataclass
class RequestResult:
    """Metrics for a single completed request.

    Step-1 extension (interactive_workload.md §12): when running in audio
    mode with interrupt policies, we additionally record audio_ttfp_sec,
    interrupt outcome and the three-pointer talker waste accounting
    (g_cut / r_cut / p_cut, see client_bargein.py).
    """

    request_id: str
    user_id: str
    round_index: int
    timestamp: int
    query_length: int
    response_length: int
    input_tokens_total: int  # cumulative input tokens (with history)
    trace_entry_index: int = -1

    # Timing
    issued_at: float = 0.0       # wall-clock time when request was sent
    ttfp_sec: float | None = None  # time to first text token (TTFT)
    audio_ttfp_sec: float | None = None  # time to first audio chunk
    e2e_sec: float | None = None   # total request duration
    connect_sec: float | None = None

    # Response
    status: int = 0
    output_tokens: int = 0
    text_content: str = ""
    audio_chunks: int = 0
    audio_duration_sec: float | None = None
    audio_first_recv_sec: float | None = None
    audio_last_recv_sec: float | None = None
    audio_first_duration_sec: float | None = None
    audio_stream_span_sec: float | None = None
    audio_duration_delta_sec: float | None = None
    audio_stream_rtf: float | None = None
    audio_post_ttfp_rtf: float | None = None
    audio_underrun_sec: float | None = None
    audio_underrun_event_count: int | None = None
    audio_continuity_ok: bool | None = None
    audio_continuity_threshold_sec: float | None = None
    audio_continuity_evaluated: bool = False
    audio_continuity_excluded_reason: str = ""
    audio_timeline: list[dict[str, float | int]] = dataclasses.field(
        default_factory=list)
    error: str | None = None

    # Derived
    tokens_per_sec: float | None = None

    # Audio mode + interrupt outcome (Step-1)
    interrupted: bool = False
    interrupt_reason: str = ""
    interrupt_stage: str = ""  # "text" / "audio" / "none"
    interrupt_triggered_at_sec: float | None = None
    audio_chunks_at_interrupt: int = 0
    threshold_sec: float | None = None  # sampled per-request threshold (fixed_time)
    skipped_by_prob: bool = False  # True if this request rolled out of interrupt_prob
    talker_frames_generated: int | None = None
    wasted_talker_frames_total: int | None = None
    wasted_talker_frames_inaudible: int | None = None
    wasted_talker_frames_pipeline: int | None = None
    waste_ratio_total: float | None = None
    waste_ratio_inaudible: float | None = None
    waste_method: str = ""
    kv_prefetch_hit_used: bool | None = None
    kv_prefetch_lookup_state: str | None = None
    kv_prefetch_lookup_key_kind: str | None = None
    kv_prefetch_warm_scope: str | None = None
    kv_prefetch_gpu_pin_requested: bool | None = None
    kv_skip_offload: bool | None = None
    kv_live_abort_aborted: int | None = None
    kv_lru_request_stats: dict[str, int | float | str] = dataclasses.field(
        default_factory=dict
    )
    server_stage0_to_first_final_output_ms: float | None = None
    server_final_stage_submit_to_first_output_ms: float | None = None
    server_first_final_output_stage_id: int | None = None
    server_queue_wait_ms: float | None = None
    server_preprocess_ms: float | None = None
    server_ar2diffusion_ms: float | None = None
    server_final_stage_input_process_ms: float | None = None
    server_ar2code2wav_ms: float | None = None
    server_stage0_gen_ms: float | None = None
    server_stage1_gen_ms: float | None = None
    server_stage2_gen_ms: float | None = None
    server_stage0_submit_to_first_output_ms: float | None = None
    server_stage1_submit_to_first_output_ms: float | None = None
    server_stage2_submit_to_first_output_ms: float | None = None
    server_stage0_submit_to_finished_ms: float | None = None
    server_stage1_submit_to_finished_ms: float | None = None
    server_stage2_submit_to_finished_ms: float | None = None
    server_stage0_to_stage1_process_inputs_ms: float | None = None
    server_stage1_to_stage2_process_inputs_ms: float | None = None
    server_stage_durations: dict[str, float] = dataclasses.field(
        default_factory=dict)
    kv_committed_prefix_hash: str | None = None
    kv_history_prefix_hash: str | None = None
    kv_prefix_token_estimate: int | None = None
    kv_history_prefix_token_estimate: int | None = None
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
    usage_total_tokens: int | None = None
    usage_cached_prompt_tokens: int | None = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        # Remove large text content from JSONL output
        if len(d.get("text_content", "")) > 200:
            d["text_content"] = d["text_content"][:200] + "..."
        if not d.get("audio_timeline"):
            d.pop("audio_timeline", None)
        if not d.get("server_stage_durations"):
            d.pop("server_stage_durations", None)
        if not d.get("kv_lru_request_stats"):
            d.pop("kv_lru_request_stats", None)
        return d


def load_audio_metadata(path: Path | None) -> AudioMetadataMap:
    """Load optional oracle audio metadata keyed by (user_id, round_index)."""
    if path is None:
        return {}
    mapping: AudioMetadataMap = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_id = row.get("user_id")
            round_index = row.get("round_index")
            if user_id is None or not isinstance(round_index, int):
                continue
            item: dict[str, float] = {}
            duration = row.get("audio_duration_sec")
            if isinstance(duration, (int, float)) and duration > 0:
                item["audio_total_sec"] = float(duration)
            reply_delay = row.get("audio_ttfp_sec")
            if isinstance(reply_delay, (int, float)) and reply_delay > 0:
                item["audio_reply_delay_sec"] = float(reply_delay)
            if item:
                mapping[(str(user_id), int(round_index))] = item
    return mapping


def _request_audio_metadata(
    audio_metadata: AudioMetadataMap,
    user_id: str,
    round_index: int,
) -> dict[str, float] | None:
    current = audio_metadata.get((str(user_id), int(round_index)))
    next_turn = audio_metadata.get((str(user_id), int(round_index) + 1))
    if not current and not next_turn:
        return None
    merged: dict[str, float] = dict(current or {})
    if next_turn:
        total = next_turn.get("audio_total_sec")
        if isinstance(total, (int, float)) and total > 0:
            merged["next_audio_total_sec"] = float(total)
        reply_delay = next_turn.get("audio_reply_delay_sec")
        if isinstance(reply_delay, (int, float)) and reply_delay > 0:
            merged["next_audio_reply_delay_sec"] = float(reply_delay)
    return merged


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

def annotate_trace_reuse_metadata(entries: list[TraceEntry]) -> None:
    """Attach same-session gap metadata for interaction-aware KV eviction."""
    prev_by_user: dict[str, TraceEntry] = {}
    for entry in entries:
        prev = prev_by_user.get(entry.user_id)
        entry.prev_gap_sec = None if prev is None else entry.timestamp - prev.timestamp
        prev_by_user[entry.user_id] = entry

    next_by_user: dict[str, TraceEntry] = {}
    for entry in reversed(entries):
        nxt = next_by_user.get(entry.user_id)
        if nxt is None:
            entry.next_gap_sec = None
            entry.expected_next_use_ts = None
        else:
            entry.next_gap_sec = nxt.timestamp - entry.timestamp
            entry.expected_next_use_ts = nxt.timestamp
        next_by_user[entry.user_id] = entry


def load_trace_entries(parts: list[int],
                       time_limit: int | None = None,
                       time_start: int | None = None) -> list[TraceEntry]:
    """Load and merge trace entries from specified part files.

    time_start, time_limit form a closed-open window [time_start, time_limit).
    If time_start is given, entries with timestamp < time_start are skipped
    BUT we still scan their user_id/round_index so the caller can decide
    whether to keep history. By default we drop them entirely.
    """
    all_entries: list[TraceEntry] = []
    for part_num in parts:
        fpath = TRACE_DIR / f"total_traces_part{part_num}.txt"
        if not fpath.exists():
            print(f"[WARN] Trace file not found: {fpath}", file=sys.stderr)
            continue
        with open(fpath, "r") as f:
            header = f.readline()  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts_split = line.split()
                if len(parts_split) < 5:
                    continue
                entry = TraceEntry(
                    user_id=parts_split[0],
                    timestamp=int(parts_split[1]),
                    query_length=int(parts_split[2]),
                    response_length=int(parts_split[3]),
                    round_index=int(parts_split[4]),
                )
                all_entries.append(entry)

    # Sort before annotating so next-gap metadata can see turns just outside
    # the replay window; otherwise near-window-tail turns look terminal.
    all_entries.sort(key=lambda e: e.timestamp)
    annotate_trace_reuse_metadata(all_entries)
    entries = [
        entry for entry in all_entries
        if (time_start is None or entry.timestamp >= time_start)
        and (time_limit is None or entry.timestamp < time_limit)
    ]
    return entries


def sample_truncated_normal(
    rng: random.Random,
    mean: float,
    std: float,
    min_value: int,
    max_value: int | None = None,
) -> int:
    if std <= 0:
        value = int(round(mean))
    else:
        value = int(round(mean))
        for _ in range(100):
            sampled = int(round(rng.gauss(mean, std)))
            if sampled >= min_value and (max_value is None or sampled <= max_value):
                value = sampled
                break
        else:
            value = int(round(rng.gauss(mean, std)))
    value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def stable_length_rng(seed: int, entry: TraceEntry, field: str) -> random.Random:
    key = "|".join(
        (
            str(seed),
            field,
            entry.user_id,
            str(entry.timestamp),
            str(entry.round_index),
        )
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big", signed=False))


def describe_ints(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "p50": None, "p90": None, "max": None, "mean": None}
    xs = sorted(values)
    return {
        "n": len(xs),
        "min": xs[0],
        "p50": xs[len(xs) // 2],
        "p90": xs[int(len(xs) * 0.9)],
        "max": xs[-1],
        "mean": statistics.mean(xs),
    }


def apply_length_profile(
    entries: list[TraceEntry],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.length_profile == "trace":
        return {
            "profile": "trace",
            "query_length": describe_ints([e.query_length for e in entries]),
            "response_length": describe_ints([e.response_length for e in entries]),
        }

    if args.length_profile == "trace-scaled":
        max_input = args.max_input_tokens if args.max_input_tokens > 0 else None
        max_output = args.max_output_tokens if args.max_output_tokens > 0 else None
        for entry in entries:
            entry.query_length = max(
                args.min_input_tokens,
                int(round(entry.query_length * args.input_tokens_scale)),
            )
            entry.response_length = max(
                args.min_output_tokens,
                int(round(entry.response_length * args.output_tokens_scale)),
            )
            if max_input is not None:
                entry.query_length = min(entry.query_length, max_input)
            if max_output is not None:
                entry.response_length = min(entry.response_length, max_output)

        return {
            "profile": args.length_profile,
            "input_tokens_scale": args.input_tokens_scale,
            "output_tokens_scale": args.output_tokens_scale,
            "query_length": describe_ints([e.query_length for e in entries]),
            "response_length": describe_ints([e.response_length for e in entries]),
        }

    if args.length_profile != "interactive-len1024":
        raise ValueError(f"unknown length profile: {args.length_profile}")

    input_mean = args.input_tokens_mean or 512
    output_mean = args.output_tokens_mean or 1024
    max_input = args.max_input_tokens if args.max_input_tokens > 0 else None
    max_output = args.max_output_tokens if args.max_output_tokens > 0 else None

    for entry in entries:
        entry.query_length = sample_truncated_normal(
            stable_length_rng(args.seed, entry, "query"),
            float(input_mean),
            float(args.input_tokens_std),
            args.min_input_tokens,
            max_input,
        )
        entry.response_length = sample_truncated_normal(
            stable_length_rng(args.seed, entry, "response"),
            float(output_mean),
            float(args.output_tokens_std),
            args.min_output_tokens,
            max_output,
        )

    return {
        "profile": args.length_profile,
        "input_tokens_mean": input_mean,
        "input_tokens_std": args.input_tokens_std,
        "output_tokens_mean": output_mean,
        "output_tokens_std": args.output_tokens_std,
        "query_length": describe_ints([e.query_length for e in entries]),
        "response_length": describe_ints([e.response_length for e in entries]),
    }


def cap_trace_entries(
    entries: list[TraceEntry],
    max_requests: int,
    policy: str = "first",
    session_min_turns: int = 2,
    session_max_turns: int = 0,
) -> list[TraceEntry]:
    """Apply a request cap without destroying repeated-session structure.

    The original first-N cap is useful for smoke tests, but for KV offload it can
    turn a multi-turn trace into a set of one-shot logical-history prompts.  The
    session-balanced policy keeps at least `session_min_turns` entries for each
    selected user, then fills any remaining slots with background traffic.
    """
    real_multiturn_policies = {
        "real_multiturn",
        "real_multiturn_peak",
        "real_multiturn_full_sessions",
        "real_multiturn_full_sessions_peak",
    }
    if policy not in real_multiturn_policies and (
        max_requests <= 0 or len(entries) <= max_requests
    ):
        return entries
    if policy == "first":
        return entries[:max_requests]

    if policy in real_multiturn_policies:
        turns_per_session = max(2, int(session_min_turns))
        max_turns_per_session = max(0, int(session_max_turns))
        if max_turns_per_session > 0:
            max_turns_per_session = max(turns_per_session, max_turns_per_session)
        request_cap = max_requests if max_requests > 0 else len(entries)
        per_user: dict[str, list[int]] = {}
        first_pos: dict[str, int] = {}
        for idx, entry in enumerate(entries):
            per_user.setdefault(entry.user_id, []).append(idx)
            first_pos.setdefault(entry.user_id, idx)

        prefixes: list[tuple[str, list[int]]] = []
        for uid in sorted(per_user, key=lambda u: first_pos[u]):
            prefix: list[int] = []
            expected_round = 0
            user_indices = sorted(per_user[uid], key=lambda i: entries[i].timestamp)
            if not user_indices or entries[user_indices[0]].round_index != 0:
                continue
            for idx in user_indices:
                round_index = entries[idx].round_index
                if round_index == expected_round:
                    prefix.append(idx)
                    expected_round += 1
                    if (
                        max_turns_per_session > 0
                        and len(prefix) >= max_turns_per_session
                    ):
                        break
                    if (
                        policy
                        not in {
                            "real_multiturn_full_sessions",
                            "real_multiturn_full_sessions_peak",
                        }
                        and len(prefix) >= turns_per_session
                    ):
                        break
                elif round_index > expected_round:
                        break
            if len(prefix) < turns_per_session:
                continue
            prefixes.append((uid, prefix))

        if policy == "real_multiturn_full_sessions":
            selected: set[int] = set()
            for _, prefix in sorted(
                prefixes,
                key=lambda item: (
                    entries[item[1][0]].timestamp,
                    entries[item[1][-1]].timestamp,
                    item[0],
                ),
            ):
                if len(selected) + len(prefix) <= request_cap:
                    selected.update(prefix)
                elif not selected:
                    selected.update(prefix[:request_cap])
                    break
            return [entries[idx] for idx in sorted(selected)]

        if policy == "real_multiturn_full_sessions_peak":
            events: list[tuple[float, int, int]] = []
            for pos, (_, prefix) in enumerate(prefixes):
                start = float(entries[prefix[0]].timestamp)
                end = float(entries[prefix[-1]].timestamp)
                events.append((start, 1, pos))
                events.append((end + 1e-6, -1, pos))
            active: set[int] = set()
            best_active: set[int] = set()
            for _, delta, pos in sorted(events, key=lambda item: (item[0], -item[1])):
                if delta > 0:
                    active.add(pos)
                else:
                    active.discard(pos)
                if len(active) > len(best_active):
                    best_active = set(active)

            def prefix_sort_key(pos: int) -> tuple[int, int, str]:
                uid, prefix = prefixes[pos]
                return (
                    entries[prefix[0]].timestamp,
                    entries[prefix[-1]].timestamp,
                    uid,
                )

            ordered_positions = sorted(best_active, key=prefix_sort_key)
            ordered_set = set(ordered_positions)
            ordered_positions.extend(
                pos
                for pos in sorted(range(len(prefixes)), key=prefix_sort_key)
                if pos not in ordered_set
            )

            selected: set[int] = set()
            for pos in ordered_positions:
                prefix = prefixes[pos][1]
                if len(selected) + len(prefix) <= request_cap:
                    selected.update(prefix)
                elif not selected:
                    selected.update(prefix[:request_cap])
                    break
            return [entries[idx] for idx in sorted(selected)]

        if policy == "real_multiturn_peak":
            session_cap = max(1, request_cap // turns_per_session)
            events: list[tuple[float, int, int]] = []
            for pos, (_, prefix) in enumerate(prefixes):
                start = float(entries[prefix[0]].timestamp)
                end = float(entries[prefix[-1]].timestamp)
                events.append((start, 1, pos))
                events.append((end + 1e-6, -1, pos))
            active: set[int] = set()
            best_active: set[int] = set()
            for _, delta, pos in sorted(events, key=lambda item: (item[0], -item[1])):
                if delta > 0:
                    active.add(pos)
                else:
                    active.discard(pos)
                if len(active) > len(best_active):
                    best_active = set(active)

            def prefix_sort_key(pos: int) -> tuple[int, int, str]:
                uid, prefix = prefixes[pos]
                return (
                    entries[prefix[0]].timestamp,
                    entries[prefix[-1]].timestamp,
                    uid,
                )

            chosen_positions = sorted(best_active, key=prefix_sort_key)[:session_cap]
            chosen_set = set(chosen_positions)
            if len(chosen_positions) < session_cap:
                for pos in sorted(range(len(prefixes)), key=prefix_sort_key):
                    if pos in chosen_set:
                        continue
                    chosen_positions.append(pos)
                    chosen_set.add(pos)
                    if len(chosen_positions) >= session_cap:
                        break
            selected: set[int] = set()
            for pos in chosen_positions:
                selected.update(prefixes[pos][1])
            return [entries[idx] for idx in sorted(selected)]

        selected: set[int] = set()
        for _, prefix in prefixes:
            if len(selected) + len(prefix) > request_cap:
                break
            selected.update(prefix)

        return [entries[idx] for idx in sorted(selected)]

    if policy != "session_balanced":
        raise ValueError(f"unknown max_requests policy: {policy}")

    if session_min_turns < 2:
        session_min_turns = 2

    per_user: dict[str, list[int]] = {}
    first_pos: dict[str, int] = {}
    for idx, entry in enumerate(entries):
        per_user.setdefault(entry.user_id, []).append(idx)
        first_pos.setdefault(entry.user_id, idx)

    eligible_users = [
        uid for uid, idxs in per_user.items()
        if len(idxs) >= session_min_turns
    ]
    eligible_users.sort(key=lambda uid: first_pos[uid])

    selected: set[int] = set()
    for uid in eligible_users:
        needed = per_user[uid][:session_min_turns]
        if len(selected) + len(needed) > max_requests:
            break
        selected.update(needed)

    if len(selected) < max_requests:
        for idx in range(len(entries)):
            selected.add(idx)
            if len(selected) >= max_requests:
                break

    return [entries[idx] for idx in sorted(selected)]


def summarize_trace_shape(entries: list[TraceEntry]) -> dict[str, Any]:
    per_user: dict[str, list[TraceEntry]] = {}
    for entry in entries:
        per_user.setdefault(entry.user_id, []).append(entry)
    same_run_returns = sum(max(0, len(v) - 1) for v in per_user.values())
    same_run_users = sum(1 for v in per_user.values() if len(v) > 1)
    same_run_gaps = []
    for user_entries in per_user.values():
        user_entries.sort(key=lambda e: e.timestamp)
        for prev, cur in zip(user_entries, user_entries[1:]):
            same_run_gaps.append(cur.timestamp - prev.timestamp)
    session_lengths = [len(v) for v in per_user.values()]
    first_round_indices = []
    max_round_indices = []
    for user_entries in per_user.values():
        user_entries.sort(key=lambda e: e.timestamp)
        first_round_indices.append(user_entries[0].round_index)
        max_round_indices.append(max(e.round_index for e in user_entries))
    return {
        "same_run_users": same_run_users,
        "same_run_returns": same_run_returns,
        "same_run_return_ratio": (
            same_run_returns / len(entries) if entries else 0.0
        ),
        "same_run_gap_p50_sec": (
            statistics.median(same_run_gaps) if same_run_gaps else None
        ),
        "same_run_gap_max_sec": max(same_run_gaps) if same_run_gaps else None,
        "round0_start_users": sum(1 for x in first_round_indices if x == 0),
        "all_sessions_start_at_round0": (
            bool(first_round_indices)
            and all(x == 0 for x in first_round_indices)
        ),
        "session_turns": describe_ints(session_lengths),
        "max_round_index": describe_ints(max_round_indices),
    }


def summarize_session_lifecycle(entries: list[TraceEntry]) -> dict[str, Any]:
    """Trace-side pressure from logical multi-turn sessions.

    `concurrency` measures requests currently in flight.  This summary measures
    the other pressure source: sessions that have started, have future turns,
    and would benefit from retaining committed history KV while idle.
    """

    per_user: dict[str, list[TraceEntry]] = {}
    for entry in entries:
        per_user.setdefault(entry.user_id, []).append(entry)
    for user_entries in per_user.values():
        user_entries.sort(key=lambda e: e.timestamp)

    open_events: list[tuple[float, int]] = []
    waiting_events: list[tuple[float, int]] = []
    inter_turn_gaps: list[int] = []
    for user_entries in per_user.values():
        if not user_entries:
            continue
        open_events.append((float(user_entries[0].timestamp), 1))
        open_events.append((float(user_entries[-1].timestamp) + 1e-6, -1))
        for prev, cur in zip(user_entries, user_entries[1:]):
            gap = cur.timestamp - prev.timestamp
            inter_turn_gaps.append(gap)
            waiting_events.append((float(prev.timestamp) + 1e-6, 1))
            waiting_events.append((float(cur.timestamp), -1))

    def peak_from_events(events: list[tuple[float, int]]) -> int:
        cur = 0
        peak = 0
        # End events before start events at the same timestamp.
        for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
            cur += delta
            peak = max(peak, cur)
        return peak

    # Peak retained committed history after trace events.  This is an idealized
    # "if all session history KV remains useful/retained" pressure estimate.
    max_turns_by_user = {
        uid: max(e.round_index for e in user_entries)
        for uid, user_entries in per_user.items()
        if user_entries
    }
    history_tokens_by_user: dict[str, int] = {uid: 0 for uid in per_user}
    events_by_time: dict[int, list[TraceEntry]] = {}
    for entry in entries:
        events_by_time.setdefault(entry.timestamp, []).append(entry)

    retained_history_tokens_peak = 0
    pending_retained_history_tokens_peak = 0
    idle_pending_retained_history_tokens_peak = 0
    open_session_count_peak = peak_from_events(open_events)
    waiting_session_count_peak = peak_from_events(waiting_events)

    for ts in sorted(events_by_time):
        current_users = {entry.user_id for entry in events_by_time[ts]}
        for entry in sorted(events_by_time[ts], key=lambda e: (e.user_id, e.round_index)):
            if entry.round_index == 0:
                history_tokens_by_user[entry.user_id] = 0
            history_tokens_by_user[entry.user_id] = (
                int(history_tokens_by_user.get(entry.user_id, 0))
                + int(entry.query_length)
                + int(entry.response_length)
            )

        retained_now = sum(history_tokens_by_user.values())
        pending_now = 0
        idle_pending_now = 0
        for uid, tokens in history_tokens_by_user.items():
            if tokens <= 0:
                continue
            # Pending means this session has at least one future selected turn.
            last_seen_round = None
            for entry in reversed(per_user.get(uid, [])):
                if entry.timestamp <= ts:
                    last_seen_round = entry.round_index
                    break
            if last_seen_round is None:
                continue
            if last_seen_round < max_turns_by_user.get(uid, last_seen_round):
                pending_now += tokens
                if uid not in current_users:
                    idle_pending_now += tokens

        retained_history_tokens_peak = max(retained_history_tokens_peak, retained_now)
        pending_retained_history_tokens_peak = max(
            pending_retained_history_tokens_peak,
            pending_now,
        )
        idle_pending_retained_history_tokens_peak = max(
            idle_pending_retained_history_tokens_peak,
            idle_pending_now,
        )

    bytes_per_token = int(os.environ.get("OMNI_SESSION_KV_BYTES_PER_TOKEN", "98304"))

    def kv_gib(tokens: int) -> float:
        return round(float(tokens) * float(bytes_per_token) / (1024.0**3), 3)

    return {
        "session_number": len(per_user),
        "logical_open_sessions_peak": open_session_count_peak,
        "logical_waiting_sessions_peak": waiting_session_count_peak,
        "inter_turn_gap_sec": describe_ints(inter_turn_gaps),
        "retained_history_tokens_peak": retained_history_tokens_peak,
        "pending_retained_history_tokens_peak": pending_retained_history_tokens_peak,
        "idle_pending_retained_history_tokens_peak": idle_pending_retained_history_tokens_peak,
        "retained_history_kv_gib_peak_est": kv_gib(retained_history_tokens_peak),
        "pending_retained_history_kv_gib_peak_est": kv_gib(
            pending_retained_history_tokens_peak
        ),
        "idle_pending_retained_history_kv_gib_peak_est": kv_gib(
            idle_pending_retained_history_tokens_peak
        ),
        "bytes_per_token_est": bytes_per_token,
    }


# ---------------------------------------------------------------------------
# Dummy text generation
# ---------------------------------------------------------------------------

def generate_dummy_text(token_count: int, rng: random.Random) -> str:
    """Generate dummy text that approximates `token_count` tokens.

    Most common English words are single tokens for most tokenizers.
    We add slight variation to avoid trivial prefix caching on content.
    """
    if token_count <= 0:
        token_count = 1
    words = [rng.choice(_DUMMY_VOCAB) for _ in range(token_count)]
    # Add a unique suffix to prevent exact-match caching across requests
    words[-1] = f"req{rng.randint(0, 99999)}"
    return " ".join(words)


# ---------------------------------------------------------------------------
# Conversation state manager
# ---------------------------------------------------------------------------

class ConversationManager:
    """Maintains per-user multi-round conversation history.

    cold_mode=True: simulate worst-case for prefix caching by
      (1) forcing every request to be round=0 (no history reuse),
      (2) injecting a per-request random nonce into the system prompt
          so even the system-prompt prefix can't be cached across reqs,
      (3) not recording any history.
    Used as the cold-cache control for KV-cache hit-rate experiments.
    """

    def __init__(
        self,
        rng: random.Random,
        cold_mode: bool = False,
        system_prompt_suffix: str = "",
        seed: int = 42,
    ):
        self._histories: dict[str, list[dict]] = {}
        self._history_token_lengths: dict[str, list[int]] = {}
        self._cumulative_tokens: dict[str, int] = {}
        self._user_texts: dict[tuple[str, int, int], str] = {}
        self._rng = rng
        self._seed = seed
        self.cold_mode = cold_mode
        self._system_prompt = (
            f"{SYSTEM_PROMPT} {system_prompt_suffix}".strip()
            if system_prompt_suffix
            else SYSTEM_PROMPT
        )

    @staticmethod
    def _entry_key(entry: TraceEntry) -> tuple[str, int, int]:
        return (entry.user_id, entry.timestamp, entry.round_index)

    def user_text_for(self, entry: TraceEntry) -> str:
        """Return a stable synthetic user turn shared by prefetch and live."""

        key = self._entry_key(entry)
        text = self._user_texts.get(key)
        if text is None:
            text = generate_dummy_text(entry.query_length, self._rng)
            self._user_texts[key] = text
        return text

    def assistant_trace_text_for(self, entry: TraceEntry) -> str:
        """Return a deterministic trace-length assistant turn for history."""

        rng = stable_length_rng(self._seed, entry, "history_response")
        return generate_dummy_text(entry.response_length, rng)

    def prepare_messages(self, entry: TraceEntry) -> tuple[list[dict], int]:
        """Build messages list for this trace entry.

        Returns (messages, total_input_tokens).
        """
        uid = entry.user_id

        if self.cold_mode:
            # Worst-case for prefix caching: unique system prompt per req,
            # no history, treat every request as round 0.
            nonce = f"{self._rng.getrandbits(64):016x}"
            sys_text = f"{self._system_prompt} [session={nonce}]"
            user_text = generate_dummy_text(entry.query_length, self._rng)
            messages = [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": user_text},
            ]
            return messages, entry.query_length

        # New conversation
        if entry.round_index == 0:
            self._histories[uid] = []
            self._history_token_lengths[uid] = []
            self._cumulative_tokens[uid] = 0

        # Generate user message
        user_text = self.user_text_for(entry)
        user_msg = {"role": "user", "content": user_text}

        # Build full messages with history
        messages = [
            {"role": "system", "content": self._system_prompt},
        ] + self._histories.get(uid, []) + [user_msg]

        # Track cumulative tokens
        cum_tokens = self._cumulative_tokens.get(uid, 0) + entry.query_length
        self._cumulative_tokens[uid] = cum_tokens

        return messages, cum_tokens

    def record_response(
        self,
        entry: TraceEntry,
        user_text: str,
        response_text: str,
        *,
        history_record_mode: str = "generated",
    ):
        """Record the user message and assistant response in history.

        No-op in cold mode.
        """
        if self.cold_mode:
            return
        if history_record_mode == "none":
            return
        if history_record_mode == "trace_dummy":
            response_text = self.assistant_trace_text_for(entry)
        elif history_record_mode != "generated":
            raise ValueError(f"unsupported history_record_mode={history_record_mode!r}")
        uid = entry.user_id
        if uid not in self._histories:
            self._histories[uid] = []
            self._history_token_lengths[uid] = []

        self._histories[uid].append({"role": "user", "content": user_text})
        self._histories[uid].append({"role": "assistant", "content": response_text})
        self._history_token_lengths[uid].extend([entry.query_length, entry.response_length])

        # Update cumulative tokens with response
        self._cumulative_tokens[uid] = (
            self._cumulative_tokens.get(uid, 0) + entry.response_length
        )

    def _trim_history(self, uid: str, max_turns: int) -> None:
        if max_turns <= 0:
            return
        max_items = max_turns * 2
        history = self._histories.get(uid) or []
        lengths = self._history_token_lengths.get(uid) or []
        if len(history) <= max_items:
            return
        self._histories[uid] = history[-max_items:]
        self._history_token_lengths[uid] = lengths[-max_items:]
        self._cumulative_tokens[uid] = sum(self._history_token_lengths[uid])

    def seed_histories(
        self,
        entries: list[TraceEntry],
        target_users: set[str],
        *,
        max_turns_per_user: int = 4,
    ) -> int:
        """Seed committed histories from trace entries before the replay window."""
        if self.cold_mode:
            return 0
        seeded = 0
        for entry in entries:
            uid = entry.user_id
            if uid not in target_users:
                continue
            if entry.round_index == 0 or uid not in self._histories:
                self._histories[uid] = []
                self._history_token_lengths[uid] = []
                self._cumulative_tokens[uid] = 0
            user_text = generate_dummy_text(entry.query_length, self._rng)
            assistant_text = generate_dummy_text(entry.response_length, self._rng)
            self._histories[uid].append({"role": "user", "content": user_text})
            self._histories[uid].append({"role": "assistant", "content": assistant_text})
            self._history_token_lengths[uid].extend(
                [entry.query_length, entry.response_length]
            )
            self._cumulative_tokens[uid] = sum(self._history_token_lengths[uid])
            self._trim_history(uid, max_turns_per_user)
            seeded += 1
        return seeded

    def history_messages(self, uid: str) -> list[dict]:
        """Return the committed conversation prefix for history-only prefetch.

        This intentionally excludes the current user turn. It matches the P3e
        conservative policy: preload only already-committed session KV during a
        VAD/end-of-speech lead window.
        """
        if self.cold_mode:
            return []
        history = self._histories.get(uid) or []
        if not history:
            return []
        return [{"role": "system", "content": self._system_prompt}] + list(history)

    def prefetch_messages(
        self,
        entry: TraceEntry,
        *,
        include_current_user: bool = False,
    ) -> tuple[list[dict], list[dict]]:
        """Return (payload messages, prefix-key messages).

        The conservative path warms only committed history. The experimental
        prefill-while-speaking path warms the exact live prompt and therefore
        must key on that same full prompt to avoid mismatching KV length.
        """

        history = self.history_messages(entry.user_id)
        if not include_current_user or not history:
            return history, history
        user_msg = {"role": "user", "content": self.user_text_for(entry)}
        payload = history + [user_msg]
        return payload, payload

    def history_token_count(self, uid: str) -> int:
        """Return trace-token estimate for committed history, excluding system."""
        if self.cold_mode:
            return 0
        return int(self._cumulative_tokens.get(uid, 0) or 0)

    @property
    def active_users(self) -> int:
        return len(self._histories)

    def get_user_rounds(self, uid: str) -> int:
        return len(self._histories.get(uid, [])) // 2


# ---------------------------------------------------------------------------
# Single request execution (audio-mode aware, with optional interrupt)
# ---------------------------------------------------------------------------

def _to_omni_messages(messages: list[dict]) -> list[dict]:
    """Convert plain-string-content messages into Qwen3-Omni's structured
    content format expected by build_audio_payload."""
    out = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            out.append({"role": m["role"],
                        "content": [{"type": "text", "text": c}]})
        else:
            out.append(m)
    return out


def _prefetch_url_from_chat_url(chat_url: str) -> str:
    """Map `/v1/chat/completions` to `/v1/omni/prefetch_session`."""
    parts = urlsplit(chat_url)
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    path = path.rstrip("/") + "/omni/prefetch_session"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def _metrics_url_from_chat_url(chat_url: str) -> str:
    parts = urlsplit(chat_url)
    return urlunsplit((parts.scheme, parts.netloc, "/metrics", "", ""))


_PROM_METRIC_RE = re.compile(
    r"^(?P<name>[^\s{]+)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)"
)
_PROM_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')


_KV_PROM_TOKEN_METRIC_KEYS = {
    "vllm:prefix_cache_queries": "local_prefix_cache_queries_tokens",
    "vllm:prefix_cache_hits": "local_prefix_cache_hits_tokens",
    "vllm:external_prefix_cache_queries": "external_prefix_cache_queries_tokens",
    "vllm:external_prefix_cache_hits": "external_prefix_cache_hits_tokens",
}


def _prom_counter_base_name(name: str) -> str:
    if name.endswith("_total"):
        return name[: -len("_total")]
    return name


async def _fetch_kv_prometheus_metrics(
    session: aiohttp.ClientSession,
    chat_url: str,
) -> dict[str, float]:
    """Fetch colocated vLLM KV counters from `/metrics`.

    The returned keys are already summed across Prometheus labels so replay
    summary generation can compute start/end deltas without knowing engine IDs.
    """
    metrics_url = _metrics_url_from_chat_url(chat_url)
    out: dict[str, float] = {}
    try:
        resp = await session.get(
            metrics_url,
            timeout=aiohttp.ClientTimeout(total=5),
        )
        try:
            if resp.status != 200:
                return out
            body = await resp.text()
        finally:
            resp.release()
    except Exception:
        return out

    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_METRIC_RE.match(line)
        if match is None:
            continue
        name = match.group("name")
        if name.endswith("_created"):
            continue
        base_name = _prom_counter_base_name(name)
        labels = dict(_PROM_LABEL_RE.findall(match.group("labels") or ""))
        try:
            value = float(match.group("value"))
        except ValueError:
            continue

        token_key = _KV_PROM_TOKEN_METRIC_KEYS.get(base_name)
        if token_key is not None:
            out[token_key] = out.get(token_key, 0.0) + value
            continue

        if "kv_offload_total_bytes" not in base_name and "kv_offload_total_time" not in base_name:
            continue
        transfer_type = labels.get("transfer_type", "unknown")
        suffix = "bytes" if "kv_offload_total_bytes" in base_name else "time_sec"
        key = f"{transfer_type}_{suffix}"
        out[key] = out.get(key, 0.0) + value
    return out


def _metric_delta(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    keys = set(before) | set(after)
    return {
        key: max(0.0, float(after.get(key, 0.0)) - float(before.get(key, 0.0)))
        for key in sorted(keys)
    }


def _metric_snapshot_delta(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, dict[str, float]]:
    return {
        "start": before,
        "end": after,
        "delta": _metric_delta(before, after),
    }


def _filter_kv_offload_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        key: value
        for key, value in metrics.items()
        if key.endswith("_bytes") or key.endswith("_time_sec")
    }


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _usage_cached_prompt_tokens(usage: dict[str, Any]) -> int | None:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = _as_int(details.get("cached_tokens"))
        if cached is not None:
            return max(0, cached)
    cached = _as_int(usage.get("cached_tokens"))
    if cached is not None:
        return max(0, cached)
    return None


def _force_first_stage_output_tokens(payload: dict[str, Any], max_tokens: int) -> None:
    """Force the thinker/text stage to run to the trace response length."""

    forced_tokens = max(1, int(max_tokens))
    params = payload.get("sampling_params_list")
    if isinstance(params, list) and params:
        first_stage = params[0]
        if isinstance(first_stage, dict):
            first_stage["max_tokens"] = forced_tokens
            first_stage["min_tokens"] = forced_tokens
            first_stage["ignore_eos"] = True
            first_stage.pop("stop_token_ids", None)
        return

    payload["max_tokens"] = forced_tokens
    payload["min_tokens"] = forced_tokens
    payload["ignore_eos"] = True


def _kv_lru_stats_path() -> Path | None:
    raw = os.environ.get("OMNI_KV_LRU_STATS_PATH")
    return Path(raw) if raw else None


def _read_kv_lru_stats_snapshot(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    max_counters: dict[str, float] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or row.get("kind") != "gpu_lru_stats":
                    continue
                counters = row.get("counters")
                if not isinstance(counters, dict):
                    counters = row
                for key, value in counters.items():
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, (int, float)):
                        key_s = str(key)
                        max_counters[key_s] = max(
                            max_counters.get(key_s, float(value)),
                            float(value),
                        )
    except OSError:
        return {}
    return max_counters


def _committed_prefix_hash(messages: list[dict]) -> str:
    """Stable hash for the committed history prefix used by prefetch/live match."""

    blob = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


async def prefetch_history_only(
    session: aiohttp.ClientSession,
    *,
    prefetch_url: str,
    model: str,
    entry: TraceEntry,
    messages: list[dict],
    committed_hash_messages: list[dict] | None = None,
    alias_hash_messages: list[list[dict]] | None = None,
    record_kind: str = "prefetch_history",
    prefetch_scope: str = "history_preload",
    lead_ms: float,
    prefix_token_estimate: int | None,
    max_tokens: int,
    timeout_sec: float,
    expected_next_gap_sec: float | None = None,
) -> dict[str, Any]:
    """Issue a service-side warm prefetch for committed history or full prompt."""
    rid = f"kvpref-{entry.user_id}-r{entry.round_index}-{uuid.uuid4().hex[:6]}"
    key_messages = committed_hash_messages if committed_hash_messages is not None else messages
    committed_hash = _committed_prefix_hash(key_messages)
    alias_hashes = [
        _committed_prefix_hash(alias_messages)
        for alias_messages in (alias_hash_messages or [])
        if alias_messages
    ]
    include_current_user = bool(prefetch_scope in {"current_turn_prefill", "full_prompt"})
    try:
        next_gap_sec = max(0.0, float(expected_next_gap_sec))
    except (TypeError, ValueError):
        next_gap_sec = 0.0
    prefetch_eviction_metadata: dict[str, Any] = {
        "omni_round_index": entry.round_index,
        "omni_trace_timestamp": entry.timestamp,
        "omni_query_length": entry.query_length,
        "omni_response_length": entry.response_length,
        "omni_expected_next_gap_sec": next_gap_sec,
        "omni_expected_next_use_ts": entry.timestamp,
        "omni_is_terminal_turn": False,
    }
    if prefix_token_estimate is not None:
        prefix_tokens = max(0, int(prefix_token_estimate))
        prefetch_eviction_metadata["omni_history_tokens_before"] = prefix_tokens
        prefetch_eviction_metadata["omni_history_tokens_after"] = prefix_tokens
    if entry.prev_gap_sec is not None:
        prefetch_eviction_metadata["omni_prev_gap_sec"] = entry.prev_gap_sec
    payload = {
        "model": model,
        "messages": _to_omni_messages(messages),
        "request_id": rid,
        "session_id": entry.user_id,
        "committed_prefix_hash": committed_hash,
        "alias_prefix_hashes": alias_hashes,
        "prefix_token_estimate": prefix_token_estimate,
        "max_tokens": max(1, int(max_tokens)),
        "lead_ms": float(lead_ms),
        "stream_options": {"include_usage": True},
        "metadata": {
            "prefetch_include_current_user": include_current_user,
            "prefetch_scope": prefetch_scope,
            **prefetch_eviction_metadata,
        },
    }
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "record_kind": record_kind,
        "prefetch_scope": prefetch_scope,
        "request_id": rid,
        "user_id": entry.user_id,
        "round_index": entry.round_index,
        "timestamp": entry.timestamp,
        "message_count": len(messages),
        "payload_message_count": len(messages),
        "committed_message_count": len(key_messages),
        "committed_prefix_hash": committed_hash,
        "alias_prefix_hashes": alias_hashes,
        "prefix_token_estimate": prefix_token_estimate,
        "eviction_metadata_attached": bool(prefetch_eviction_metadata),
        "eviction_metadata": prefetch_eviction_metadata,
        "issued_at": t0,
        "ok": False,
    }
    try:
        resp = await session.post(
            prefetch_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_sec),
        )
        try:
            body = await resp.text()
            out["http_status"] = resp.status
            if resp.status != 200:
                out["error"] = body[:500]
            else:
                data = json.loads(body)
                out.update(
                    {
                        "ok": data.get("status") == "SUCCESS",
                        "status": data.get("status"),
                        "strategy": data.get("strategy"),
                        "skip_reason": data.get("skip_reason"),
                        "admission": data.get("admission"),
                        "controller": data.get("controller"),
                        "registry": data.get("registry"),
                        "server_wall_s": data.get("wall_s"),
                        "first_chunk_s": data.get("first_chunk_s"),
                        "text_len": data.get("text_len"),
                        "chunks": data.get("chunks"),
                        "usage": data.get("usage"),
                    }
                )
        finally:
            resp.release()
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    finished_at = time.perf_counter()
    out["finished_at"] = finished_at
    out["wall_s"] = finished_at - t0
    return out


async def execute_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    entry: TraceEntry,
    messages: list[dict],
    total_input_tokens: int,
    *,
    with_audio: bool,
    policy: BargeInPolicy,
    rng: random.Random,
    request_timeout_sec: float = 300.0,
    record_audio_timeline: bool = False,
    audio_continuity_threshold_sec: float | None = None,
    audio_continuity_chunk_mode: str = "fragment",
    continuity_include_interrupted: bool = False,
    kv_prefix_include_current_user: bool = False,
    kv_history_fallback: bool = False,
    force_trace_output_tokens: bool = False,
    audio_metadata: dict[str, float] | None = None,
) -> RequestResult:
    """Send one request and collect metrics.

    `with_audio=False` reproduces Phase 3/4 behaviour (text-only). When
    `with_audio=True` we use the same payload + SSE handling as the barge-in
    replay client and additionally honour the per-request cut threshold sampled
    from `policy`.
    """
    rid = f"itrace-{entry.user_id}-r{entry.round_index}-{uuid.uuid4().hex[:6]}"

    result = RequestResult(
        request_id=rid,
        user_id=entry.user_id,
        round_index=entry.round_index,
        timestamp=entry.timestamp,
        query_length=entry.query_length,
        response_length=entry.response_length,
        input_tokens_total=total_input_tokens,
    )

    # Sample per-request interrupt threshold (no-op if policy=='none')
    thresh = policy.resolve_for_request(rng)
    if thresh.get("skipped_by_prob"):
        result.skipped_by_prob = True
    if thresh.get("enabled") and thresh.get("kind") == "time":
        result.threshold_sec = float(thresh.get("threshold_sec") or 0.0)

    # Build payload
    if with_audio:
        payload = build_audio_payload(
            model, _to_omni_messages(messages), rid,
            max_tokens=max(entry.response_length, 1))
    else:
        payload = build_text_payload(
            model, messages, rid,
            max_tokens=max(entry.response_length, 1))
    if force_trace_output_tokens:
        _force_first_stage_output_tokens(payload, max(entry.response_length, 1))

    history_prefix_messages = messages[:-1] if len(messages) > 1 else []
    history_prefix_token_estimate = max(0, total_input_tokens - entry.query_length)
    if kv_prefix_include_current_user:
        prefix_messages = messages
        prefix_token_estimate = total_input_tokens
    else:
        prefix_messages = history_prefix_messages
        prefix_token_estimate = history_prefix_token_estimate
    prefix_hash = _committed_prefix_hash(prefix_messages) if prefix_messages else None
    if prefix_hash is not None:
        payload["omni_session_id"] = entry.user_id
        payload["committed_prefix_hash"] = prefix_hash
        payload["omni_prefix_token_estimate"] = prefix_token_estimate
        result.kv_committed_prefix_hash = prefix_hash
        result.kv_prefix_token_estimate = prefix_token_estimate
    if kv_history_fallback and history_prefix_messages:
        history_hash = _committed_prefix_hash(history_prefix_messages)
        payload["omni_history_prefix_hash"] = history_hash
        payload["omni_history_prefix_token_estimate"] = history_prefix_token_estimate
        result.kv_history_prefix_hash = history_hash
        result.kv_history_prefix_token_estimate = history_prefix_token_estimate

    payload["omni_round_index"] = entry.round_index
    payload["omni_trace_timestamp"] = entry.timestamp
    payload["omni_query_length"] = entry.query_length
    payload["omni_response_length"] = entry.response_length
    payload["omni_history_tokens_before"] = history_prefix_token_estimate
    payload["omni_history_tokens_after"] = (
        history_prefix_token_estimate + entry.query_length + entry.response_length
    )
    if entry.prev_gap_sec is not None:
        payload["omni_prev_gap_sec"] = entry.prev_gap_sec
    if entry.next_gap_sec is not None:
        payload["omni_expected_next_gap_sec"] = entry.next_gap_sec
        payload["omni_expected_next_use_ts"] = entry.expected_next_use_ts
    payload["omni_is_terminal_turn"] = entry.next_gap_sec is None
    if audio_metadata:
        audio_total = audio_metadata.get("audio_total_sec")
        if isinstance(audio_total, (int, float)) and audio_total > 0:
            payload["omni_audio_total_sec"] = float(audio_total)
            payload["omni_audio_metadata_source"] = "oracle_jsonl"
        audio_reply_delay = audio_metadata.get("audio_reply_delay_sec")
        if isinstance(audio_reply_delay, (int, float)) and audio_reply_delay > 0:
            payload["omni_audio_reply_delay_sec"] = float(audio_reply_delay)
        next_audio_total = audio_metadata.get("next_audio_total_sec")
        if isinstance(next_audio_total, (int, float)) and next_audio_total > 0:
            payload["omni_next_audio_total_sec"] = float(next_audio_total)
        next_audio_reply_delay = audio_metadata.get("next_audio_reply_delay_sec")
        if (
            isinstance(next_audio_reply_delay, (int, float))
            and next_audio_reply_delay > 0
        ):
            payload["omni_next_audio_reply_delay_sec"] = float(
                next_audio_reply_delay
            )

    t0 = time.perf_counter()
    result.issued_at = t0
    sr: StreamResult | None = None

    try:
        resp = await session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=request_timeout_sec),
        )
        try:
            result.status = resp.status
            result.connect_sec = time.perf_counter() - t0

            if resp.status != 200:
                result.error = (await resp.text())[:500]
                return result

            sr = await consume_sse_audio(
                resp, t0, thresh, audio_mode=with_audio,
                record_audio_timeline=record_audio_timeline,
                audio_continuity_threshold_sec=audio_continuity_threshold_sec,
                audio_continuity_chunk_mode=audio_continuity_chunk_mode,
                continuity_include_interrupted=continuity_include_interrupted)
        finally:
            # If we triggered an interrupt mid-stream we MUST close the
            # underlying connection to make the server abort the request.
            if sr is not None and sr.interrupted:
                await close_response(resp)
            else:
                # Normal release path so the connection can be pooled.
                try:
                    resp.release()
                except Exception:
                    pass

    except asyncio.CancelledError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        result.error = f"{type(e).__name__}: {e}"
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
    finally:
        result.e2e_sec = time.perf_counter() - t0
        if sr is not None:
            # Copy stream-level results back into the request-level record
            result.ttfp_sec = sr.text_ttft_sec
            result.audio_ttfp_sec = sr.audio_ttfp_sec
            result.text_content = sr.text_content
            result.output_tokens = sr.text_chars  # text-mode output token count proxy
            result.audio_chunks = sr.audio_chunks
            result.audio_duration_sec = sr.audio_duration_sec
            result.audio_first_recv_sec = sr.audio_first_recv_sec
            result.audio_last_recv_sec = sr.audio_last_recv_sec
            result.audio_first_duration_sec = sr.audio_first_duration_sec
            result.audio_stream_span_sec = sr.audio_stream_span_sec
            result.audio_duration_delta_sec = sr.audio_duration_delta_sec
            result.audio_stream_rtf = sr.audio_stream_rtf
            result.audio_post_ttfp_rtf = sr.audio_post_ttfp_rtf
            result.audio_underrun_sec = sr.audio_underrun_sec
            result.audio_underrun_event_count = sr.audio_underrun_event_count
            result.audio_continuity_ok = sr.audio_continuity_ok
            result.audio_continuity_threshold_sec = (
                sr.audio_continuity_threshold_sec
            )
            result.audio_continuity_evaluated = sr.audio_continuity_evaluated
            result.audio_continuity_excluded_reason = (
                sr.audio_continuity_excluded_reason
            )
            result.audio_timeline = sr.audio_timeline
            result.interrupted = sr.interrupted
            result.interrupt_reason = sr.interrupt_reason
            result.interrupt_stage = sr.interrupt_stage
            result.interrupt_triggered_at_sec = sr.interrupt_triggered_at_sec
            result.audio_chunks_at_interrupt = sr.audio_chunks_at_interrupt
            result.talker_frames_generated = sr.talker_frames_generated
            result.wasted_talker_frames_total = sr.wasted_talker_frames_total
            result.wasted_talker_frames_inaudible = sr.wasted_talker_frames_inaudible
            result.wasted_talker_frames_pipeline = sr.wasted_talker_frames_pipeline
            result.waste_ratio_total = sr.waste_ratio_total
            result.waste_ratio_inaudible = sr.waste_ratio_inaudible
            result.waste_method = sr.waste_method
            result.kv_prefetch_hit_used = sr.kv_prefetch_hit_used
            result.kv_prefetch_lookup_state = sr.kv_prefetch_lookup_state
            result.kv_prefetch_lookup_key_kind = sr.kv_prefetch_lookup_key_kind
            result.kv_prefetch_warm_scope = sr.kv_prefetch_warm_scope
            result.kv_prefetch_gpu_pin_requested = (
                sr.kv_prefetch_gpu_pin_requested
            )
            result.kv_skip_offload = sr.kv_skip_offload
            result.kv_live_abort_aborted = sr.kv_live_abort_aborted
            result.kv_lru_request_stats = dict(sr.kv_lru_request_stats)
            result.server_stage0_to_first_final_output_ms = (
                sr.server_stage0_to_first_final_output_ms
            )
            result.server_final_stage_submit_to_first_output_ms = (
                sr.server_final_stage_submit_to_first_output_ms
            )
            result.server_first_final_output_stage_id = (
                sr.server_first_final_output_stage_id
            )
            result.server_queue_wait_ms = sr.server_queue_wait_ms
            result.server_preprocess_ms = sr.server_preprocess_ms
            result.server_ar2diffusion_ms = sr.server_ar2diffusion_ms
            result.server_final_stage_input_process_ms = (
                sr.server_final_stage_input_process_ms
            )
            result.server_ar2code2wav_ms = sr.server_ar2code2wav_ms
            result.server_stage0_gen_ms = sr.server_stage0_gen_ms
            result.server_stage1_gen_ms = sr.server_stage1_gen_ms
            result.server_stage2_gen_ms = sr.server_stage2_gen_ms
            result.server_stage0_submit_to_first_output_ms = (
                sr.server_stage0_submit_to_first_output_ms
            )
            result.server_stage1_submit_to_first_output_ms = (
                sr.server_stage1_submit_to_first_output_ms
            )
            result.server_stage2_submit_to_first_output_ms = (
                sr.server_stage2_submit_to_first_output_ms
            )
            result.server_stage0_submit_to_finished_ms = (
                sr.server_stage0_submit_to_finished_ms
            )
            result.server_stage1_submit_to_finished_ms = (
                sr.server_stage1_submit_to_finished_ms
            )
            result.server_stage2_submit_to_finished_ms = (
                sr.server_stage2_submit_to_finished_ms
            )
            result.server_stage0_to_stage1_process_inputs_ms = (
                sr.server_stage0_to_stage1_process_inputs_ms
            )
            result.server_stage1_to_stage2_process_inputs_ms = (
                sr.server_stage1_to_stage2_process_inputs_ms
            )
            result.server_stage_durations = dict(sr.server_stage_durations)
            if sr.usage and isinstance(sr.usage, dict):
                prompt = _as_int(sr.usage.get("prompt_tokens"))
                completion = _as_int(sr.usage.get("completion_tokens"))
                total = _as_int(sr.usage.get("total_tokens"))
                cached = _usage_cached_prompt_tokens(sr.usage)
                if prompt is not None:
                    result.usage_prompt_tokens = prompt
                if completion is not None:
                    result.usage_completion_tokens = completion
                    result.output_tokens = completion
                if total is not None:
                    result.usage_total_tokens = total
                if cached is not None:
                    result.usage_cached_prompt_tokens = cached
        if result.output_tokens and result.e2e_sec and result.e2e_sec > 0:
            result.tokens_per_sec = result.output_tokens / result.e2e_sec

    return result


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------

async def replay_trace(
    entries: list[TraceEntry],
    url: str,
    model: str,
    max_inflight: int,
    speedup: float,
    result_dir: Path,
    progress_interval: int = 50,
    cold_mode: bool = False,
    *,
    with_audio: bool = False,
    policy: BargeInPolicy | None = None,
    seed: int = 42,
    record_audio_timeline: bool = False,
    audio_continuity_threshold_sec: float | None = None,
    audio_continuity_chunk_mode: str = "fragment",
    continuity_include_interrupted: bool = False,
    request_timeout_sec: float = 300.0,
    warm_history_entries: list[TraceEntry] | None = None,
    warm_history_max_turns_per_user: int = 4,
    prewarm_history_before_replay: bool = False,
    prewarm_max_users: int = 0,
    prewarm_min_history_tokens: int = 1,
    prewarm_max_history_tokens: int = 0,
    prewarm_max_inflight: int = 0,
    prewarm_timeout_sec: float | None = None,
    prewarm_sort: str = "history_tokens_desc",
    prefetch_mode: str = "none",
    prefetch_url: str | None = None,
    prefetch_lead_ms: float = 150.0,
    prefetch_max_tokens: int = 1,
    prefetch_timeout_sec: float = 30.0,
    prefetch_max_inflight: int = 8,
    prefetch_allow_prestart: bool = False,
    prefetch_include_current_user: bool = False,
    prefetch_current_turn: bool = False,
    current_prefill_lead_ms: float | None = None,
    current_prefill_max_tokens: int | None = None,
    current_prefill_timeout_sec: float | None = None,
    prefetch_min_remaining_lead_ms: float = 0.0,
    prefetch_history_min_remaining_lead_ms: float | None = None,
    current_prefill_min_remaining_lead_ms: float | None = None,
    prefetch_clipped_start_limit: int = -1,
    prefetch_same_time_request_first: bool = False,
    system_prompt_suffix: str = "",
    enforce_session_order: bool = False,
    history_record_mode: str = "generated",
    force_trace_output_tokens: bool = False,
    max_measurement_results: int = 0,
    max_pending_request_tasks: int = 0,
    metrics_start_index: int = 0,
    metrics_start_trace_time: int | None = None,
    metrics_end_trace_time: int | None = None,
    audio_start_trace_time: int | None = None,
    audio_end_trace_time: int | None = None,
    audio_metadata: AudioMetadataMap | None = None,
) -> tuple[list[RequestResult], dict]:
    """Replay trace entries with trace-driven timing.

    Returns (results, replay_stats) where replay_stats includes inflight_samples,
    inflight_max, etc.
    """

    conv_mgr = ConversationManager(
        rng=random.Random(seed),
        cold_mode=cold_mode,
        system_prompt_suffix=system_prompt_suffix,
        seed=seed,
    )
    # Independent RNG for interrupt sampling so seeding stays deterministic
    # regardless of how many turns each user has.
    interrupt_rng = random.Random(seed + 1)
    if policy is None:
        policy = BargeInPolicy(policy="none")
    audio_metadata = audio_metadata or {}
    seeded_history_entries = conv_mgr.seed_histories(
        warm_history_entries or [],
        {e.user_id for e in entries},
        max_turns_per_user=warm_history_max_turns_per_user,
    )
    entry_order_by_id = {id(entry): idx for idx, entry in enumerate(entries)}
    semaphore = asyncio.Semaphore(max_inflight)
    results: list[RequestResult] = []
    prefetch_results: list[dict[str, Any]] = []
    results_lock = asyncio.Lock()
    prefetch_lock = asyncio.Lock()
    prefetch_sem = asyncio.Semaphore(max(1, prefetch_max_inflight))
    effective_prefetch_clipped_start_limit = int(prefetch_clipped_start_limit)
    prefetch_clipped_start_kept = 0
    prefetch_clipped_start_skipped = 0

    # Output file
    result_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = result_dir / "per_request.jsonl"
    jsonl_f = open(jsonl_path, "w")
    prefetch_jsonl_path = result_dir / "prefetch.jsonl"
    prefetch_jsonl_f = (
        open(prefetch_jsonl_path, "w")
        if prefetch_mode != "none" or prewarm_history_before_replay
        else None
    )

    total = len(entries)
    completed = 0
    errors = 0
    measurement_completed = 0
    dispatched_events = 0
    stopped_by_measurement_cap = False
    stop_dispatch = asyncio.Event()
    if max_pending_request_tasks > 0:
        effective_max_pending_request_tasks = max(1, int(max_pending_request_tasks))
    elif max_measurement_results > 0:
        effective_max_pending_request_tasks = max(1, int(max_inflight))
    else:
        effective_max_pending_request_tasks = 0
    pending_request_tasks: set[asyncio.Task] = set()
    pending_request_tasks_max = 0
    prewarm_stats: dict[str, Any] = {
        "enabled": bool(prewarm_history_before_replay),
        "attempts": 0,
        "ok": 0,
        "errors": 0,
        "candidates": 0,
        "selected": 0,
        "history_tokens": 0,
        "wall_s": 0.0,
    }
    start_wall = time.perf_counter()
    trace_t0 = entries[0].timestamp if entries else 0
    replay_wall_trace_t0 = trace_t0

    # Inflight tracking
    inflight = 0
    inflight_max = 0
    inflight_samples: list[int] = []
    last_progress_print = [start_wall]  # mutable holder
    session_locks: dict[str, asyncio.Lock] = {}

    # Shared aiohttp session with connection pool (large-scale runs)
    timeout = aiohttp.ClientTimeout(total=300, connect=30)
    connector = aiohttp.TCPConnector(
        limit=max_inflight * 2,
        limit_per_host=max_inflight * 2,
        keepalive_timeout=60,
    )
    shared_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    effective_prefetch_url = prefetch_url or _prefetch_url_from_chat_url(url)
    effective_current_prefill_lead_ms = (
        prefetch_lead_ms if current_prefill_lead_ms is None else current_prefill_lead_ms
    )
    effective_current_prefill_max_tokens = (
        prefetch_max_tokens
        if current_prefill_max_tokens is None
        else current_prefill_max_tokens
    )
    effective_current_prefill_timeout_sec = (
        prefetch_timeout_sec
        if current_prefill_timeout_sec is None
        else current_prefill_timeout_sec
    )
    effective_min_remaining_lead_ms = max(0.0, float(prefetch_min_remaining_lead_ms))
    effective_history_min_remaining_lead_ms = (
        effective_min_remaining_lead_ms
        if prefetch_history_min_remaining_lead_ms is None
        else max(0.0, float(prefetch_history_min_remaining_lead_ms))
    )
    effective_current_min_remaining_lead_ms = (
        effective_min_remaining_lead_ms
        if current_prefill_min_remaining_lead_ms is None
        else max(0.0, float(current_prefill_min_remaining_lead_ms))
    )
    kv_lru_path = _kv_lru_stats_path()
    kv_prometheus_metrics_before_prewarm: dict[str, float] = {}
    kv_lru_metrics_before_prewarm: dict[str, float] = {}
    kv_prometheus_metrics_start: dict[str, float] = {}
    kv_lru_metrics_start: dict[str, float] = {}
    prewarm_kv_prometheus_metrics: dict[str, dict[str, float]] = {}
    prewarm_kv_lru_metrics: dict[str, dict[str, float]] = {}

    def _entry_uses_audio(entry: TraceEntry) -> bool:
        if not with_audio:
            return False
        if audio_start_trace_time is not None and entry.timestamp < audio_start_trace_time:
            return False
        if audio_end_trace_time is not None and entry.timestamp >= audio_end_trace_time:
            return False
        return True

    def _result_in_measurement_window(result: RequestResult) -> bool:
        if metrics_start_index and int(result.trace_entry_index) < int(metrics_start_index):
            return False
        if metrics_start_trace_time is not None and result.timestamp < metrics_start_trace_time:
            return False
        if metrics_end_trace_time is not None and result.timestamp >= metrics_end_trace_time:
            return False
        return True

    def _expected_wall_time(trace_time: float) -> float:
        return start_wall + (float(trace_time) - float(replay_wall_trace_t0)) / float(speedup)

    def _min_remaining_lead_for_scope(prefetch_scope: str) -> float:
        if prefetch_scope in {"current_turn_prefill", "full_prompt"}:
            return effective_current_min_remaining_lead_ms
        return effective_history_min_remaining_lead_ms

    def _append_prefetch_row(row: dict[str, Any]) -> None:
        prefetch_results.append(row)
        if prefetch_jsonl_f is not None:
            prefetch_jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            prefetch_jsonl_f.flush()

    def _session_lock(uid: str) -> asyncio.Lock:
        lock = session_locks.get(uid)
        if lock is None:
            lock = asyncio.Lock()
            session_locks[uid] = lock
        return lock

    async def _run_prefetch(
        entry: TraceEntry,
        *,
        prefetch_scope: str,
        scheduled_trace_time: float,
        intended_trace_time: float,
        target_trace_time: float,
    ):
        if prefetch_mode == "none":
            return
        if prefetch_scope == "current_turn_prefill":
            messages, committed_key_messages = conv_mgr.prefetch_messages(
                entry,
                include_current_user=True,
            )
            history_messages = conv_mgr.history_messages(entry.user_id)
            alias_hash_messages = [history_messages] if history_messages else []
            record_kind = "prefetch_current_turn"
            max_tokens = effective_current_prefill_max_tokens
            timeout_sec = effective_current_prefill_timeout_sec
            lead_ms = effective_current_prefill_lead_ms
            prefix_estimate = conv_mgr.history_token_count(entry.user_id) + entry.query_length
        else:
            messages, committed_key_messages = conv_mgr.prefetch_messages(
                entry,
                include_current_user=(
                    prefetch_include_current_user and not prefetch_current_turn
                ),
            )
            alias_hash_messages = []
            record_kind = "prefetch_history"
            max_tokens = prefetch_max_tokens
            timeout_sec = prefetch_timeout_sec
            lead_ms = prefetch_lead_ms
            prefix_estimate = conv_mgr.history_token_count(entry.user_id)
            if prefetch_include_current_user and not prefetch_current_turn:
                prefix_estimate += entry.query_length
                prefetch_scope = "full_prompt"
                record_kind = "prefetch_current_turn"
        if not committed_key_messages:
            row = {
                "record_kind": record_kind,
                "prefetch_scope": prefetch_scope,
                "user_id": entry.user_id,
                "round_index": entry.round_index,
                "timestamp": entry.timestamp,
                "scheduled_trace_time": scheduled_trace_time,
                "intended_trace_time": intended_trace_time,
                "target_trace_time": target_trace_time,
                "effective_lead_trace_sec": target_trace_time - scheduled_trace_time,
                "clipped_to_trace_start": scheduled_trace_time > intended_trace_time,
                "include_current_user": prefetch_scope in {"current_turn_prefill", "full_prompt"},
                "ok": False,
                "skipped": "no_committed_history",
            }
        else:
            async with prefetch_sem:
                expected_live_wall = _expected_wall_time(target_trace_time)
                issue_wall = time.perf_counter()
                remaining_lead_ms = max(0.0, (expected_live_wall - issue_wall) * 1000.0)
                min_remaining_lead_ms = _min_remaining_lead_for_scope(prefetch_scope)
                if (
                    min_remaining_lead_ms > 0.0
                    and remaining_lead_ms < min_remaining_lead_ms
                ):
                    row = {
                        "record_kind": record_kind,
                        "prefetch_scope": prefetch_scope,
                        "user_id": entry.user_id,
                        "round_index": entry.round_index,
                        "timestamp": entry.timestamp,
                        "scheduled_trace_time": scheduled_trace_time,
                        "intended_trace_time": intended_trace_time,
                        "target_trace_time": target_trace_time,
                        "effective_lead_trace_sec": target_trace_time - scheduled_trace_time,
                        "clipped_to_trace_start": scheduled_trace_time > intended_trace_time,
                        "include_current_user": prefetch_scope in {"current_turn_prefill", "full_prompt"},
                        "configured_lead_ms": lead_ms,
                        "remaining_lead_ms": remaining_lead_ms,
                        "min_remaining_lead_ms": min_remaining_lead_ms,
                        "issued_at": issue_wall,
                        "finished_at": issue_wall,
                        "wall_s": 0.0,
                        "ok": False,
                        "status": "SKIPPED",
                        "skip_reason": "driver_prefetch_late",
                    }
                else:
                    row = await prefetch_history_only(
                        shared_session,
                        prefetch_url=effective_prefetch_url,
                        model=model,
                        entry=entry,
                        messages=messages,
                        committed_hash_messages=committed_key_messages,
                        alias_hash_messages=alias_hash_messages,
                        record_kind=record_kind,
                        prefetch_scope=prefetch_scope,
                        lead_ms=remaining_lead_ms,
                        prefix_token_estimate=prefix_estimate,
                        max_tokens=max_tokens,
                        timeout_sec=timeout_sec,
                        expected_next_gap_sec=max(
                            0.0,
                            float(target_trace_time) - float(scheduled_trace_time),
                        ),
                    )
                    row["configured_lead_ms"] = lead_ms
                    row["remaining_lead_ms"] = remaining_lead_ms
                row.update(
                    {
                        "scheduled_trace_time": scheduled_trace_time,
                        "intended_trace_time": intended_trace_time,
                        "target_trace_time": target_trace_time,
                        "effective_lead_trace_sec": target_trace_time - scheduled_trace_time,
                        "clipped_to_trace_start": scheduled_trace_time > intended_trace_time,
                        "include_current_user": prefetch_scope in {"current_turn_prefill", "full_prompt"},
                    }
                )
        async with prefetch_lock:
            _append_prefetch_row(row)

    async def _run_entry_request(entry: TraceEntry):
        nonlocal completed, errors, inflight, inflight_max, measurement_completed

        async with semaphore:
            inflight += 1
            inflight_max = max(inflight_max, inflight)
            inflight_samples.append(inflight)

            try:
                messages, total_input_tokens = conv_mgr.prepare_messages(entry)

                result = await execute_request(
                    shared_session, url, model, entry, messages, total_input_tokens,
                    with_audio=_entry_uses_audio(entry), policy=policy, rng=interrupt_rng,
                    request_timeout_sec=request_timeout_sec,
                    record_audio_timeline=record_audio_timeline,
                    audio_continuity_threshold_sec=audio_continuity_threshold_sec,
                    audio_continuity_chunk_mode=audio_continuity_chunk_mode,
                    continuity_include_interrupted=continuity_include_interrupted,
                    kv_prefix_include_current_user=(
                        prefetch_mode != "none"
                        and (prefetch_include_current_user or prefetch_current_turn)
                    ),
                    kv_history_fallback=(
                        prefetch_mode != "none" and prefetch_current_turn
                    ),
                    force_trace_output_tokens=force_trace_output_tokens,
                    audio_metadata=_request_audio_metadata(
                        audio_metadata,
                        entry.user_id,
                        entry.round_index,
                    ),
                )
                result.trace_entry_index = entry_order_by_id.get(id(entry), -1)

                # Record response in conversation history.
                # §12.2 Q8: write *partial* assistant text on interrupt
                # (selected option A — matches real user experience).
                should_record_history = (
                    result.error is None
                    and history_record_mode != "none"
                    and (
                        bool(result.text_content)
                        or history_record_mode == "trace_dummy"
                    )
                )
                if should_record_history:
                    user_text = messages[-1]['content']
                    if isinstance(user_text, list):
                        # _to_omni_messages may have wrapped content for audio mode
                        user_text = next(
                            (p.get("text", "") for p in user_text
                             if p.get("type") == "text"),
                            "")
                    conv_mgr.record_response(
                        entry,
                        user_text,
                        result.text_content or "",
                        history_record_mode=history_record_mode,
                    )

                async with results_lock:
                    results.append(result)
                    jsonl_f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                    jsonl_f.flush()
                    completed += 1
                    if max_measurement_results > 0 and _result_in_measurement_window(result):
                        measurement_completed += 1
                        if measurement_completed >= max_measurement_results:
                            stop_dispatch.set()
                    if result.error:
                        errors += 1

                # Progress: by count OR every 30 wall-clock seconds
                now_wall = time.perf_counter()
                should_print = (
                    completed % progress_interval == 0
                    or (now_wall - last_progress_print[0]) > 30.0
                )
                if should_print:
                    last_progress_print[0] = now_wall
                    elapsed = now_wall - start_wall
                    rps = completed / elapsed if elapsed > 0 else 0
                    extra = (
                        f" measurement={measurement_completed}/{max_measurement_results}"
                        if max_measurement_results > 0
                        else ""
                    )
                    print(
                        f"  [{completed}/{total}] "
                        f"elapsed={elapsed:.1f}s rps={rps:.2f} "
                        f"inflight={inflight} errors={errors} "
                        f"users={conv_mgr.active_users}{extra}",
                        flush=True,
                    )
            finally:
                inflight -= 1

    async def _run_entry(entry: TraceEntry):
        if not enforce_session_order:
            await _run_entry_request(entry)
            return
        async with _session_lock(entry.user_id):
            await _run_entry_request(entry)
    # Schedule prefetch and request events according to their timestamps.
    tasks: list[asyncio.Task] = []
    events: list[tuple[float, int, str, TraceEntry, float]] = []
    prefetch_trace_lead = max(0.0, prefetch_lead_ms / 1000.0 * speedup)
    current_prefill_trace_lead = max(
        0.0,
        effective_current_prefill_lead_ms / 1000.0 * speedup,
    )

    def _should_skip_clipped_start_history_prefetch(
        *,
        scheduled_trace_time: float,
        intended_trace_time: float,
    ) -> bool:
        nonlocal prefetch_clipped_start_kept, prefetch_clipped_start_skipped
        if effective_prefetch_clipped_start_limit < 0:
            return False
        clipped_to_trace_start = (
            not prefetch_allow_prestart
            and scheduled_trace_time == float(trace_t0)
            and scheduled_trace_time > intended_trace_time
        )
        if not clipped_to_trace_start:
            return False
        if prefetch_clipped_start_kept < effective_prefetch_clipped_start_limit:
            prefetch_clipped_start_kept += 1
            return False
        prefetch_clipped_start_skipped += 1
        return True

    def _record_clipped_start_prefetch_skip(
        entry: TraceEntry,
        *,
        prefetch_scope: str,
        scheduled_trace_time: float,
        intended_trace_time: float,
        target_trace_time: float,
    ) -> None:
        _append_prefetch_row(
            {
                "record_kind": "prefetch_history",
                "prefetch_scope": prefetch_scope,
                "user_id": entry.user_id,
                "round_index": entry.round_index,
                "timestamp": entry.timestamp,
                "scheduled_trace_time": scheduled_trace_time,
                "intended_trace_time": intended_trace_time,
                "target_trace_time": target_trace_time,
                "effective_lead_trace_sec": target_trace_time - scheduled_trace_time,
                "clipped_to_trace_start": True,
                "include_current_user": False,
                "configured_lead_ms": prefetch_lead_ms,
                "issued_at": None,
                "finished_at": None,
                "wall_s": 0.0,
                "ok": False,
                "status": "SKIPPED",
                "skip_reason": "driver_clipped_start_limit",
                "clipped_start_limit": effective_prefetch_clipped_start_limit,
            }
        )

    async def _run_prewarm_history_before_replay() -> dict[str, Any]:
        if not prewarm_history_before_replay:
            return dict(prewarm_stats)

        first_entry_by_user: dict[str, TraceEntry] = {}
        for entry in entries:
            first_entry_by_user.setdefault(entry.user_id, entry)

        candidates: list[tuple[str, TraceEntry, list[dict], int]] = []
        min_tokens = max(0, int(prewarm_min_history_tokens))
        max_tokens = max(0, int(prewarm_max_history_tokens))
        for uid, entry in first_entry_by_user.items():
            messages = conv_mgr.history_messages(uid)
            if not messages:
                continue
            token_estimate = int(conv_mgr.history_token_count(uid) or 0)
            if token_estimate < min_tokens:
                continue
            if max_tokens > 0 and token_estimate > max_tokens:
                continue
            candidates.append((uid, entry, messages, token_estimate))

        if prewarm_sort == "trace_order":
            candidates.sort(key=lambda item: (item[1].timestamp, item[0]))
        elif prewarm_sort == "history_tokens_asc":
            candidates.sort(key=lambda item: (item[3], item[1].timestamp, item[0]))
        else:
            candidates.sort(key=lambda item: (-item[3], item[1].timestamp, item[0]))

        if prewarm_max_users > 0:
            candidates = candidates[: int(prewarm_max_users)]

        timeout_sec = float(
            prewarm_timeout_sec
            if prewarm_timeout_sec is not None
            else prefetch_timeout_sec
        )
        stats: dict[str, Any] = {
            "enabled": True,
            "candidates": len(first_entry_by_user),
            "selected": len(candidates),
            "attempts": 0,
            "ok": 0,
            "errors": 0,
            "history_tokens": sum(item[3] for item in candidates),
            "max_users": int(prewarm_max_users),
            "min_history_tokens": min_tokens,
            "max_history_tokens": max_tokens,
            "sort": prewarm_sort,
            "max_inflight": int(prewarm_max_inflight or prefetch_max_inflight),
            "timeout_sec": timeout_sec,
            "wall_s": 0.0,
        }
        if not candidates:
            print("[PREWARM] No history prefixes selected before replay", flush=True)
            return stats

        sem = asyncio.Semaphore(max(1, int(stats["max_inflight"])))
        t0 = time.perf_counter()
        print(
            "[PREWARM] Starting history prewarm: "
            f"selected={stats['selected']} "
            f"history_tokens={stats['history_tokens']} "
            f"max_inflight={stats['max_inflight']}",
            flush=True,
        )

        completed_prewarm = 0

        async def _prewarm_one(
            rank: int,
            uid: str,
            entry: TraceEntry,
            messages: list[dict],
            token_estimate: int,
        ) -> dict[str, Any]:
            nonlocal completed_prewarm
            async with sem:
                row = await prefetch_history_only(
                    shared_session,
                    prefetch_url=effective_prefetch_url,
                    model=model,
                    entry=entry,
                    messages=messages,
                    committed_hash_messages=messages,
                    alias_hash_messages=[],
                    record_kind="prewarm_history",
                    prefetch_scope="history_prewarm",
                    lead_ms=0.0,
                    prefix_token_estimate=token_estimate,
                    max_tokens=prefetch_max_tokens,
                    timeout_sec=timeout_sec,
                    expected_next_gap_sec=max(
                        0.0,
                        float(entry.timestamp) - float(trace_t0),
                    ),
                )
                row.update(
                    {
                        "prewarm_rank": rank,
                        "prewarm_before_replay": True,
                        "prewarm_history_tokens": token_estimate,
                    }
                )
                async with prefetch_lock:
                    _append_prefetch_row(row)
                completed_prewarm += 1
                if completed_prewarm % 100 == 0 or completed_prewarm == stats["selected"]:
                    print(
                        f"[PREWARM] {completed_prewarm}/{stats['selected']} done",
                        flush=True,
                    )
                return row

        rows = await asyncio.gather(
            *(
                _prewarm_one(rank, uid, entry, messages, token_estimate)
                for rank, (uid, entry, messages, token_estimate)
                in enumerate(candidates, 1)
            ),
            return_exceptions=True,
        )
        ok = 0
        errors = 0
        attempts = 0
        for row in rows:
            if isinstance(row, Exception):
                errors += 1
                continue
            attempts += 1
            if row.get("ok"):
                ok += 1
            else:
                errors += 1
        stats.update(
            {
                "attempts": attempts,
                "ok": ok,
                "errors": errors,
                "wall_s": time.perf_counter() - t0,
            }
        )
        print(
            "[PREWARM] Done: "
            f"ok={ok}/{attempts} errors={errors} "
            f"wall={stats['wall_s']:.1f}s",
            flush=True,
        )
        return stats

    async def _wait_for_request_task_slot() -> bool:
        nonlocal pending_request_tasks_max
        if effective_max_pending_request_tasks <= 0:
            return True
        while len(pending_request_tasks) >= effective_max_pending_request_tasks:
            pending_request_tasks_max = max(
                pending_request_tasks_max,
                len(pending_request_tasks),
            )
            if stop_dispatch.is_set():
                return False
            done, _ = await asyncio.wait(
                set(pending_request_tasks),
                timeout=0.1,
                return_when=asyncio.FIRST_COMPLETED,
            )
            pending_request_tasks.difference_update(done)
        return not stop_dispatch.is_set()

    for order, entry in enumerate(entries):
        if prefetch_mode != "none" and entry.round_index > 0:
            intended_prefetch_time = float(entry.timestamp) - prefetch_trace_lead
            scheduled_prefetch_time = (
                intended_prefetch_time
                if prefetch_allow_prestart
                else max(float(trace_t0), intended_prefetch_time)
            )
            scope = (
                "full_prompt"
                if prefetch_include_current_user and not prefetch_current_turn
                else "history_preload"
            )
            if (
                scope == "history_preload"
                and _should_skip_clipped_start_history_prefetch(
                    scheduled_trace_time=scheduled_prefetch_time,
                    intended_trace_time=intended_prefetch_time,
                )
            ):
                _record_clipped_start_prefetch_skip(
                    entry,
                    prefetch_scope=scope,
                    scheduled_trace_time=scheduled_prefetch_time,
                    intended_trace_time=intended_prefetch_time,
                    target_trace_time=float(entry.timestamp),
                )
            else:
                events.append(
                    (
                        scheduled_prefetch_time,
                        order,
                        scope,
                        entry,
                        intended_prefetch_time,
                    )
                )
            if prefetch_current_turn:
                intended_current_time = (
                    float(entry.timestamp) - current_prefill_trace_lead
                )
                scheduled_current_time = (
                    intended_current_time
                    if prefetch_allow_prestart
                    else max(float(trace_t0), intended_current_time)
                )
                events.append(
                    (
                        scheduled_current_time,
                        order,
                        "current_turn_prefill",
                        entry,
                        intended_current_time,
                    )
                )
        events.append((float(entry.timestamp), order, "request", entry, float(entry.timestamp)))
    if prefetch_same_time_request_first:
        event_priority = {
            "request": 0,
            "current_turn_prefill": 1,
            "history_preload": 2,
            "full_prompt": 2,
        }
    else:
        event_priority = {
            "history_preload": 0,
            "full_prompt": 0,
            "current_turn_prefill": 1,
            "request": 2,
        }
    events.sort(key=lambda item: (item[0], event_priority.get(item[2], 1), item[1]))
    prev_event_time = events[0][0] if events else float(trace_t0)
    replay_wall_trace_t0 = prev_event_time

    kv_prometheus_metrics_before_prewarm = await _fetch_kv_prometheus_metrics(
        shared_session, url)
    kv_lru_metrics_before_prewarm = _read_kv_lru_stats_snapshot(kv_lru_path)
    prewarm_stats = await _run_prewarm_history_before_replay()
    kv_prometheus_metrics_start = await _fetch_kv_prometheus_metrics(
        shared_session, url)
    kv_lru_metrics_start = _read_kv_lru_stats_snapshot(kv_lru_path)
    prewarm_kv_prometheus_metrics = _metric_snapshot_delta(
        kv_prometheus_metrics_before_prewarm,
        kv_prometheus_metrics_start,
    )
    prewarm_kv_lru_metrics = _metric_snapshot_delta(
        kv_lru_metrics_before_prewarm,
        kv_lru_metrics_start,
    )
    if prewarm_stats.get("enabled"):
        prewarm_stats["kv_prometheus_delta"] = prewarm_kv_prometheus_metrics.get(
            "delta",
            {},
        )
        prewarm_stats["kv_lru_delta"] = prewarm_kv_lru_metrics.get("delta", {})

    print(f"[REPLAY] Starting: {total} entries, speedup={speedup}x, "
          f"max_inflight={max_inflight}")
    if effective_max_pending_request_tasks > 0:
        print(
            "[REPLAY] Pending request task cap: "
            f"{effective_max_pending_request_tasks}",
            flush=True,
        )
    print(f"[REPLAY] Trace time range: {trace_t0}s - {entries[-1].timestamp}s "
          f"({(entries[-1].timestamp - trace_t0)/60:.1f} min)")
    print(f"[REPLAY] Estimated wall time: "
          f"{(entries[-1].timestamp - trace_t0) / speedup / 60:.1f} min")
    print()

    for event_time, _, event_kind, entry, intended_event_time in events:
        if stop_dispatch.is_set():
            stopped_by_measurement_cap = True
            break
        # Calculate delay based on timestamp difference
        delay_trace_sec = event_time - prev_event_time
        delay_wall_sec = delay_trace_sec / speedup

        if delay_wall_sec > 0:
            if max_measurement_results > 0:
                try:
                    await asyncio.wait_for(stop_dispatch.wait(), timeout=delay_wall_sec)
                    stopped_by_measurement_cap = True
                    break
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(delay_wall_sec)

        prev_event_time = event_time

        if stop_dispatch.is_set():
            stopped_by_measurement_cap = True
            break

        if event_kind == "request":
            if not await _wait_for_request_task_slot():
                stopped_by_measurement_cap = True
                break

        # Launch request/prefetch as a task.
        task = asyncio.create_task(
            _run_prefetch(
                entry,
                prefetch_scope=event_kind,
                scheduled_trace_time=event_time,
                intended_trace_time=intended_event_time,
                target_trace_time=float(entry.timestamp),
            )
            if event_kind != "request"
            else _run_entry(entry)
        )
        tasks.append(task)
        if event_kind == "request" and effective_max_pending_request_tasks > 0:
            pending_request_tasks.add(task)
            pending_request_tasks_max = max(
                pending_request_tasks_max,
                len(pending_request_tasks),
            )
            task.add_done_callback(pending_request_tasks.discard)
        dispatched_events += 1

    # Wait for all tasks to complete
    if stopped_by_measurement_cap:
        print(
            f"\n[REPLAY] Stopped dispatch after measurement cap: "
            f"{measurement_completed}/{max_measurement_results} measurement results, "
            f"{dispatched_events}/{len(events)} events dispatched. Waiting for completion...",
            flush=True,
        )
    else:
        print(f"\n[REPLAY] All {total} entries dispatched. Waiting for completion...",
              flush=True)
    await asyncio.gather(*tasks, return_exceptions=True)

    kv_prometheus_metrics_end = await _fetch_kv_prometheus_metrics(
        shared_session, url)
    kv_lru_metrics_end = _read_kv_lru_stats_snapshot(kv_lru_path)
    await shared_session.close()
    jsonl_f.close()
    if prefetch_jsonl_f is not None:
        prefetch_jsonl_f.close()

    # Final stats
    elapsed = time.perf_counter() - start_wall
    print(f"\n[REPLAY] Done: {completed}/{total} completed, {errors} errors, "
          f"{elapsed:.1f}s wall time")

    kv_offload_metrics_start = _filter_kv_offload_metrics(
        kv_prometheus_metrics_start)
    kv_offload_metrics_end = _filter_kv_offload_metrics(
        kv_prometheus_metrics_end)

    replay_stats = {
        "inflight_max": inflight_max,
        "inflight_samples": inflight_samples,
        "completed": completed,
        "errors": errors,
        "wall_time_sec": elapsed,
        "dispatched_events": dispatched_events,
        "total_events": len(events),
        "stopped_by_measurement_cap": stopped_by_measurement_cap,
        "max_measurement_results": max_measurement_results,
        "measurement_completed": measurement_completed,
        "max_pending_request_tasks": max_pending_request_tasks,
        "effective_max_pending_request_tasks": effective_max_pending_request_tasks,
        "pending_request_tasks_max": pending_request_tasks_max,
        "seeded_history_entries": seeded_history_entries,
        "warm_history_max_turns_per_user": warm_history_max_turns_per_user,
        "history_record_mode": history_record_mode,
        "prefetch_mode": prefetch_mode,
        "prefetch_url": (
            effective_prefetch_url
            if prefetch_mode != "none" or prewarm_history_before_replay
            else None
        ),
        "prewarm_history_before_replay": prewarm_history_before_replay,
        "prewarm": prewarm_stats,
        "prewarm_kv_prometheus_metrics": prewarm_kv_prometheus_metrics,
        "prewarm_kv_lru_metrics": prewarm_kv_lru_metrics,
        "prefetch_allow_prestart": prefetch_allow_prestart,
        "prefetch_include_current_user": prefetch_include_current_user,
        "prefetch_current_turn": prefetch_current_turn,
        "prefetch_same_time_request_first": prefetch_same_time_request_first,
        "enforce_session_order": enforce_session_order,
        "prefetch_clipped_start_limit": effective_prefetch_clipped_start_limit,
        "force_trace_output_tokens": force_trace_output_tokens,
        "audio_start_trace_time": audio_start_trace_time,
        "audio_end_trace_time": audio_end_trace_time,
        "prefetch_clipped_start_kept": prefetch_clipped_start_kept,
        "prefetch_clipped_start_skipped": prefetch_clipped_start_skipped,
        "current_prefill_lead_ms": effective_current_prefill_lead_ms,
        "current_prefill_max_tokens": effective_current_prefill_max_tokens,
        "prefetch_min_remaining_lead_ms": effective_min_remaining_lead_ms,
        "prefetch_history_min_remaining_lead_ms": (
            effective_history_min_remaining_lead_ms
        ),
        "current_prefill_min_remaining_lead_ms": (
            effective_current_min_remaining_lead_ms
        ),
        "prefetch_attempts": len(prefetch_results),
        "prefetch_ok": sum(1 for r in prefetch_results if r.get("ok")),
        "prefetch_by_kind": {
            kind: {
                "attempts": sum(
                    1 for r in prefetch_results if r.get("record_kind") == kind
                ),
                "ok": sum(
                    1 for r in prefetch_results
                    if r.get("record_kind") == kind and r.get("ok")
                ),
            }
            for kind in sorted(
                {
                    str(r.get("record_kind"))
                    for r in prefetch_results
                    if r.get("record_kind")
                }
            )
        },
        "prefetch_skipped_no_history": sum(
            1 for r in prefetch_results if r.get("skipped") == "no_committed_history"
        ),
        "prefetch_skipped_admission": sum(
            1 for r in prefetch_results if r.get("status") == "SKIPPED"
        ),
        "prefetch_skip_reasons": {
            reason: sum(1 for r in prefetch_results if r.get("skip_reason") == reason)
            for reason in sorted(
                {
                    str(r.get("skip_reason"))
                    for r in prefetch_results
                    if r.get("skip_reason")
                }
            )
        },
        "prefetch_wall_s": [
            float(r["wall_s"]) for r in prefetch_results
            if isinstance(r.get("wall_s"), (int, float))
        ],
        "prefetch_prefix_token_estimates": [
            float(r["prefix_token_estimate"]) for r in prefetch_results
            if isinstance(r.get("prefix_token_estimate"), (int, float))
        ],
        "prefetch_deadline_estimate_sources": {
            source: sum(
                1
                for r in prefetch_results
                if (((r.get("admission") or {}).get("deadline") or {}).get(
                    "estimate_source"
                ) == source)
            )
            for source in sorted(
                {
                    str(((r.get("admission") or {}).get("deadline") or {}).get(
                        "estimate_source"
                    ))
                    for r in prefetch_results
                    if ((r.get("admission") or {}).get("deadline") or {}).get(
                        "estimate_source"
                    )
                }
            )
        },
        "prefetch_deadline_prefix_buckets": {
            bucket: sum(
                1
                for r in prefetch_results
                if (((r.get("admission") or {}).get("deadline") or {}).get(
                    "prefix_bucket"
                ) == bucket)
            )
            for bucket in sorted(
                {
                    str(((r.get("admission") or {}).get("deadline") or {}).get(
                        "prefix_bucket"
                    ))
                    for r in prefetch_results
                    if ((r.get("admission") or {}).get("deadline") or {}).get(
                        "prefix_bucket"
                    )
                }
            )
        },
        "kv_prometheus_metrics": _metric_snapshot_delta(
            kv_prometheus_metrics_start,
            kv_prometheus_metrics_end,
        ),
        "kv_offload_metrics": _metric_snapshot_delta(
            kv_offload_metrics_start,
            kv_offload_metrics_end,
        ),
        "kv_lru_metrics": {
            "enabled": os.environ.get("OMNI_KV_LRU_STATS") in {"1", "true", "True"},
            "stats_path": str(kv_lru_path) if kv_lru_path is not None else None,
            **_metric_snapshot_delta(kv_lru_metrics_start, kv_lru_metrics_end),
        },
    }
    return results, replay_stats


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def _ratio(num: float, den: float) -> float | None:
    return (num / den) if den > 0 else None


def _round_count(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    return 0


def _build_kv_hit_ratio_summary(
    ok_results: list[RequestResult],
    replay_stats: dict | None,
) -> dict[str, Any]:
    prom = ((replay_stats or {}).get("kv_prometheus_metrics") or {})
    prom_delta = prom.get("delta") if isinstance(prom, dict) else {}
    if not isinstance(prom_delta, dict):
        prom_delta = {}

    local_queries = _round_count(
        prom_delta.get("local_prefix_cache_queries_tokens"))
    local_hits = _round_count(prom_delta.get("local_prefix_cache_hits_tokens"))
    external_queries = _round_count(
        prom_delta.get("external_prefix_cache_queries_tokens"))
    external_hits = _round_count(
        prom_delta.get("external_prefix_cache_hits_tokens"))
    local_misses = max(0, local_queries - local_hits)
    external_misses = max(0, external_queries - external_hits)
    overall_queries = local_queries + external_queries
    overall_hits = local_hits + external_hits

    usage_results = [
        r for r in ok_results if r.usage_prompt_tokens is not None
    ]
    cached_observed = any(
        r.usage_cached_prompt_tokens is not None for r in usage_results
    )
    usage_prompt_tokens = sum(
        int(r.usage_prompt_tokens or 0) for r in usage_results
    )
    if cached_observed:
        usage_cached_tokens = sum(
            int(r.usage_cached_prompt_tokens or 0) for r in usage_results
        )
        usage_hit_requests = sum(
            1 for r in usage_results if int(r.usage_cached_prompt_tokens or 0) > 0
        )
        request_cached_ratio = _ratio(usage_cached_tokens, usage_prompt_tokens)
        request_hit_rate = _ratio(usage_hit_requests, len(usage_results))
    else:
        usage_cached_tokens = None
        usage_hit_requests = None
        request_cached_ratio = None
        request_hit_rate = None

    lookup_results = [
        r for r in ok_results if r.kv_prefetch_lookup_state is not None
    ]
    primary_warm = sum(
        1 for r in lookup_results if r.kv_prefetch_lookup_state == "warm"
    )
    fallback_warm = sum(
        1 for r in lookup_results
        if r.kv_prefetch_lookup_state == "warm_fallback"
    )
    warm_hits = primary_warm + fallback_warm

    return {
        "source": {
            "prometheus_metrics": bool(prom_delta),
            "request_usage": bool(usage_results),
            "request_cached_tokens_observed": cached_observed,
            "prefetch_lookup_state": bool(lookup_results),
        },
        "local_tokens": {
            "queries": local_queries,
            "hits": local_hits,
            "misses": local_misses,
            "hit_ratio": _ratio(local_hits, local_queries),
        },
        "external_tokens": {
            "queries": external_queries,
            "hits": external_hits,
            "misses": external_misses,
            "hit_ratio": _ratio(external_hits, external_queries),
        },
        "overall_tokens": {
            "queries": overall_queries,
            "hits": overall_hits,
            "misses": max(0, overall_queries - overall_hits),
            "hit_ratio": _ratio(overall_hits, overall_queries),
        },
        "local_prefix_hit_ratio_tokens": _ratio(local_hits, local_queries),
        "external_prefix_hit_ratio_tokens": _ratio(
            external_hits,
            external_queries,
        ),
        "overall_kv_hit_ratio_tokens": _ratio(overall_hits, overall_queries),
        "requests": {
            "usage_rows": len(usage_results),
            "prompt_tokens": usage_prompt_tokens if usage_results else None,
            "cached_prompt_tokens": usage_cached_tokens,
            "hit_requests": usage_hit_requests,
            "request_cached_ratio": request_cached_ratio,
            "request_hit_rate": request_hit_rate,
        },
        "prefetch_proxy": {
            "lookup_rows": len(lookup_results),
            "warm_hits": warm_hits,
            "primary_warm_hits": primary_warm,
            "fallback_warm_hits": fallback_warm,
            "warm_hit_rate": _ratio(warm_hits, len(lookup_results)),
            "primary_warm_hit_rate": _ratio(primary_warm, len(lookup_results)),
            "fallback_warm_hit_rate": _ratio(fallback_warm, len(lookup_results)),
        },
        "prometheus": prom,
    }


def _build_kv_lru_summary(
    ok_results: list[RequestResult],
    replay_stats: dict | None,
) -> dict[str, Any]:
    lru = ((replay_stats or {}).get("kv_lru_metrics") or {})
    lru_delta = lru.get("delta") if isinstance(lru, dict) else {}
    if not isinstance(lru_delta, dict):
        lru_delta = {}

    gpu_evictions = _round_count(lru_delta.get("gpu_lru_eviction_blocks_total"))
    gpu_lookup_misses = _round_count(
        lru_delta.get("gpu_prefix_block_lookup_miss"))
    recent_misses = _round_count(
        lru_delta.get("gpu_prefix_block_lookup_recent_eviction_miss"))
    cpu_evictions = _round_count(
        lru_delta.get("cpu_offload_lru_eviction_blocks_total"))

    prompt_tokens = sum(
        int(r.usage_prompt_tokens or 0)
        for r in ok_results
        if r.usage_prompt_tokens is not None
    )
    if prompt_tokens <= 0:
        prompt_tokens = sum(int(r.input_tokens_total or 0) for r in ok_results)

    return {
        "enabled": bool(lru.get("enabled")) if isinstance(lru, dict) else False,
        "stats_path": lru.get("stats_path") if isinstance(lru, dict) else None,
        "source": "jsonl" if lru_delta else None,
        "gpu_evictions": gpu_evictions,
        "gpu_lookup_misses": gpu_lookup_misses,
        "recent_eviction_misses": recent_misses,
        "recent_eviction_miss_ratio_blocks": _ratio(
            recent_misses,
            gpu_lookup_misses,
        ),
        "gpu_lru_evictions_per_request": _ratio(gpu_evictions, len(ok_results)),
        "gpu_lru_evictions_per_1k_prompt_tokens": (
            _ratio(gpu_evictions * 1000.0, prompt_tokens)
        ),
        "cpu_lru_evictions": cpu_evictions,
        "raw": lru,
    }


def _select_metric_results(
    results: list[RequestResult],
    args: argparse.Namespace,
    full_wall_time: float,
) -> tuple[list[RequestResult], float, dict[str, Any]]:
    """Select the request subset used for latency/throughput aggregates."""

    start_index = max(0, int(getattr(args, "metrics_start_index", 0) or 0))
    start_trace_time = getattr(args, "metrics_start_trace_time", None)
    if start_trace_time is not None:
        start_trace_time = int(start_trace_time)
    end_trace_time = getattr(args, "metrics_end_trace_time", None)
    if end_trace_time is not None:
        end_trace_time = int(end_trace_time)

    selected: list[RequestResult] = []
    for result in results:
        if start_index > 0 and result.trace_entry_index < start_index:
            continue
        if start_trace_time is not None and result.timestamp < start_trace_time:
            continue
        if end_trace_time is not None and result.timestamp >= end_trace_time:
            continue
        selected.append(result)

    enabled = (
        start_index > 0
        or start_trace_time is not None
        or end_trace_time is not None
    )
    measurement_wall_time = full_wall_time
    if enabled:
        starts = [
            float(result.issued_at)
            for result in selected
            if isinstance(result.issued_at, (int, float)) and result.issued_at > 0
        ]
        ends = [
            float(result.issued_at) + float(result.e2e_sec)
            for result in selected
            if isinstance(result.issued_at, (int, float))
            and result.issued_at > 0
            and isinstance(result.e2e_sec, (int, float))
            and result.e2e_sec >= 0
        ]
        if starts and ends and max(ends) > min(starts):
            measurement_wall_time = max(ends) - min(starts)
        else:
            measurement_wall_time = 0.0

    timestamps = [int(result.timestamp) for result in selected]
    indices = [
        int(result.trace_entry_index)
        for result in selected
        if result.trace_entry_index >= 0
    ]
    measurement = {
        "enabled": enabled,
        "metrics_start_index": start_index,
        "metrics_start_trace_time": start_trace_time,
        "metrics_end_trace_time": end_trace_time,
        "input_results": len(results),
        "included_results": len(selected),
        "excluded_results": max(0, len(results) - len(selected)),
        "full_wall_time_sec": round(full_wall_time, 2),
        "measurement_wall_time_sec": round(measurement_wall_time, 2),
        "included_trace_time_min": min(timestamps) if timestamps else None,
        "included_trace_time_max": max(timestamps) if timestamps else None,
        "included_trace_entry_index_min": min(indices) if indices else None,
        "included_trace_entry_index_max": max(indices) if indices else None,
        "global_counter_scope": "whole_replay",
        "note": (
            "Latency/throughput aggregates use the included request subset. "
            "KV Prometheus/LRU counters are sampled once at replay start/end "
            "and therefore remain whole-replay counters."
        ),
    }
    return selected, measurement_wall_time, measurement


def generate_summary(results: list[RequestResult], args: argparse.Namespace,
                     result_dir: Path, wall_time: float,
                     replay_stats: dict | None = None):
    """Generate summary.json with aggregate statistics."""
    full_results = results
    full_ok_results = [r for r in full_results if r.error is None]
    metric_results, metric_wall_time, measurement = _select_metric_results(
        full_results,
        args,
        wall_time,
    )
    results = metric_results
    wall_time = metric_wall_time
    ok_results = [r for r in metric_results if r.error is None]
    err_results = [r for r in metric_results if r.error is not None]

    ttfps = [r.ttfp_sec for r in ok_results if r.ttfp_sec is not None]
    e2es = [r.e2e_sec for r in ok_results if r.e2e_sec is not None]
    tps_list = [r.tokens_per_sec for r in ok_results if r.tokens_per_sec is not None]
    input_tokens = [r.input_tokens_total for r in ok_results]
    round_indices = [r.round_index for r in ok_results]

    def percentiles(arr: list[float]) -> dict:
        if not arr:
            return {}
        arr_s = sorted(arr)
        n = len(arr_s)
        return {
            "n": n,
            "min": arr_s[0],
            "p50": arr_s[n // 2],
            "p90": arr_s[int(n * 0.9)],
            "p95": arr_s[int(n * 0.95)],
            "p99": arr_s[int(n * 0.99)],
            "max": arr_s[-1],
            "mean": statistics.mean(arr_s),
        }

    # §12 Step-1 audio + interrupt aggregates
    audio_ttfps = [r.audio_ttfp_sec for r in ok_results
                   if r.audio_ttfp_sec is not None]
    audio_durations = [r.audio_duration_sec for r in ok_results
                       if r.audio_duration_sec is not None and r.audio_duration_sec > 0]
    audio_post_ttfp_rtfs = [
        r.audio_post_ttfp_rtf for r in ok_results
        if r.audio_post_ttfp_rtf is not None
        and r.audio_post_ttfp_rtf >= 0
    ]
    audio_stream_rtfs = [
        r.audio_stream_rtf for r in ok_results
        if r.audio_stream_rtf is not None
        and r.audio_stream_rtf >= 0
    ]
    continuity_results = [
        r for r in ok_results if r.audio_continuity_evaluated
    ]
    audio_underruns = [
        r.audio_underrun_sec for r in continuity_results
        if r.audio_underrun_sec is not None
    ]
    audio_underrun_event_counts = [
        float(r.audio_underrun_event_count)
        for r in continuity_results
        if r.audio_underrun_event_count is not None
    ]
    audio_continuity_ok_count = sum(
        1 for r in continuity_results if r.audio_continuity_ok is True
    )
    audio_continuity_excluded_reasons: dict[str, int] = {}
    for r in ok_results:
        if r.audio_continuity_evaluated:
            continue
        reason = r.audio_continuity_excluded_reason or "not_evaluated"
        audio_continuity_excluded_reasons[reason] = (
            audio_continuity_excluded_reasons.get(reason, 0) + 1
        )
    interrupted_results = [r for r in ok_results if r.interrupted]
    not_interrupted_ok = [r for r in ok_results if not r.interrupted]
    waste_total = [r.wasted_talker_frames_total for r in interrupted_results
                   if r.wasted_talker_frames_total is not None]
    waste_inaudible = [r.wasted_talker_frames_inaudible
                       for r in interrupted_results
                       if r.wasted_talker_frames_inaudible is not None]
    interrupt_stages: dict[str, int] = {}
    for r in interrupted_results:
        s = r.interrupt_stage or "unknown"
        interrupt_stages[s] = interrupt_stages.get(s, 0) + 1

    counter_ok_results = (
        full_ok_results if measurement.get("enabled") else ok_results
    )
    kv_hit_ratio = _build_kv_hit_ratio_summary(counter_ok_results, replay_stats)
    kv_lru = _build_kv_lru_summary(counter_ok_results, replay_stats)

    # useful_qps: completed-without-interrupt + (interrupted but at least
    # something was heard, weighted by p_cut/r_cut). For Step-1 baseline
    # we report a simple variant: completed_ok / wall_time.
    useful_qps = (
        round(len(not_interrupted_ok) / wall_time, 4)
        if wall_time > 0 else 0
    )

    def continuity_breakdown(key_fn) -> dict[str, dict[str, float | int | None]]:
        groups: dict[str, dict[str, float | int | None]] = {}
        for r in ok_results:
            key = key_fn(r)
            bucket = groups.setdefault(
                key,
                {
                    "n_total_ok": 0,
                    "n_evaluated": 0,
                    "n_ok": 0,
                    "ok_rate": None,
                },
            )
            bucket["n_total_ok"] = int(bucket["n_total_ok"] or 0) + 1
            if r.audio_continuity_evaluated:
                bucket["n_evaluated"] = int(bucket["n_evaluated"] or 0) + 1
                if r.audio_continuity_ok is True:
                    bucket["n_ok"] = int(bucket["n_ok"] or 0) + 1
        for bucket in groups.values():
            n_eval = int(bucket["n_evaluated"] or 0)
            bucket["ok_rate"] = (
                float(bucket["n_ok"] or 0) / n_eval if n_eval > 0 else None
            )
        return groups

    audio_streaming_continuity = {
        "threshold_sec": getattr(args, "audio_continuity_threshold_sec", None),
        "chunk_mode": getattr(args, "audio_continuity_chunk_mode", "fragment"),
        "include_interrupted": getattr(
            args, "continuity_include_interrupted", False),
        "n_total_ok": len(ok_results),
        "n_evaluated": len(continuity_results),
        "n_ok": audio_continuity_ok_count,
        "ok_rate": (
            audio_continuity_ok_count / len(continuity_results)
            if continuity_results else None
        ),
        "audio_underrun_sec": percentiles(
            [float(x) for x in audio_underruns]),
        "audio_underrun_event_count": percentiles(
            audio_underrun_event_counts),
        "goodput_rps": (
            round(audio_continuity_ok_count / wall_time, 4)
            if wall_time > 0 else 0
        ),
        "excluded_reasons": audio_continuity_excluded_reasons,
        "by_prefetch_lookup_state": continuity_breakdown(
            lambda r: r.kv_prefetch_lookup_state or "none"),
        "by_prefetch_warm_scope": continuity_breakdown(
            lambda r: r.kv_prefetch_warm_scope or "none"),
        "by_interrupt_status": continuity_breakdown(
            lambda r: "interrupted" if r.interrupted else "completed"),
    }

    summary = {
        "args": {
            "mode": args.mode,
            "parts": args.parts,
            "time_start": args.time_start,
            "time_limit": args.time_limit,
            "speedup": args.speedup,
            "max_inflight": args.max_inflight,
            "max_requests": getattr(args, "max_requests", 0),
            "max_requests_policy": getattr(args, "max_requests_policy", "first"),
            "metrics_start_index": getattr(args, "metrics_start_index", 0),
            "metrics_start_trace_time": getattr(
                args,
                "metrics_start_trace_time",
                None,
            ),
            "metrics_end_trace_time": getattr(
                args,
                "metrics_end_trace_time",
                None,
            ),
            "max_measurement_results": getattr(
                args,
                "max_measurement_results",
                0,
            ),
            "max_pending_request_tasks": getattr(
                args,
                "max_pending_request_tasks",
                0,
            ),
            "session_balanced_min_turns": getattr(
                args,
                "session_balanced_min_turns",
                2,
            ),
            "session_balanced_max_turns": getattr(
                args,
                "session_balanced_max_turns",
                0,
            ),
            "enforce_session_order": getattr(args, "enforce_session_order", False),
            "warm_history_before_start": getattr(args, "warm_history_before_start", False),
            "warm_history_max_turns_per_user": getattr(
                args, "warm_history_max_turns_per_user", 4
            ),
            "prewarm_history_before_replay": getattr(
                args,
                "prewarm_history_before_replay",
                False,
            ),
            "prewarm_max_users": getattr(args, "prewarm_max_users", 0),
            "prewarm_min_history_tokens": getattr(
                args,
                "prewarm_min_history_tokens",
                1,
            ),
            "prewarm_max_history_tokens": getattr(
                args,
                "prewarm_max_history_tokens",
                0,
            ),
            "prewarm_max_inflight": getattr(args, "prewarm_max_inflight", 0),
            "prewarm_timeout_sec": getattr(args, "prewarm_timeout_sec", None),
            "prewarm_sort": getattr(args, "prewarm_sort", "history_tokens_desc"),
            "history_record_mode": getattr(args, "history_record_mode", "generated"),
            "url": args.url,
            "model": args.model,
            "with_audio": getattr(args, "with_audio", False),
            "audio_start_trace_time": getattr(
                args,
                "audio_start_trace_time",
                None,
            ),
            "audio_end_trace_time": getattr(
                args,
                "audio_end_trace_time",
                None,
            ),
            "policy": getattr(args, "policy", "none"),
            "interrupt_prob": getattr(args, "interrupt_prob", 1.0),
            "interrupt_after_seconds": getattr(args, "interrupt_after_seconds", 2.0),
            "interrupt_jitter_sec": getattr(args, "interrupt_jitter_sec", 0.5),
            "interrupt_time_anchor": getattr(args, "interrupt_time_anchor", "ttfp"),
            "record_audio_timeline": getattr(args, "record_audio_timeline", False),
            "audio_continuity_threshold_sec": getattr(
                args,
                "audio_continuity_threshold_sec",
                None,
            ),
            "audio_continuity_chunk_mode": getattr(
                args,
                "audio_continuity_chunk_mode",
                "fragment",
            ),
            "continuity_include_interrupted": getattr(
                args,
                "continuity_include_interrupted",
                False,
            ),
            "request_timeout_sec": getattr(args, "request_timeout_sec", 300.0),
            "force_trace_output_tokens": getattr(
                args,
                "force_trace_output_tokens",
                False,
            ),
            "prefetch_mode": getattr(args, "prefetch_mode", "none"),
            "prefetch_lead_ms": getattr(args, "prefetch_lead_ms", 150.0),
            "prefetch_max_tokens": getattr(args, "prefetch_max_tokens", 1),
            "prefetch_max_inflight": getattr(args, "prefetch_max_inflight", 8),
            "prefetch_allow_prestart": getattr(args, "prefetch_allow_prestart", False),
            "prefetch_include_current_user": getattr(
                args,
                "prefetch_include_current_user",
                False,
            ),
            "prefetch_current_turn": getattr(args, "prefetch_current_turn", False),
            "prefetch_same_time_request_first": getattr(
                args,
                "prefetch_same_time_request_first",
                False,
            ),
            "prefetch_clipped_start_limit": getattr(
                args,
                "prefetch_clipped_start_limit",
                -1,
            ),
            "current_prefill_lead_ms": getattr(args, "current_prefill_lead_ms", None),
            "current_prefill_max_tokens": getattr(
                args,
                "current_prefill_max_tokens",
                None,
            ),
            "prefetch_min_remaining_lead_ms": getattr(
                args,
                "prefetch_min_remaining_lead_ms",
                0.0,
            ),
            "prefetch_history_min_remaining_lead_ms": getattr(
                args,
                "prefetch_history_min_remaining_lead_ms",
                None,
            ),
            "current_prefill_min_remaining_lead_ms": getattr(
                args,
                "current_prefill_min_remaining_lead_ms",
                None,
            ),
            "system_prompt_suffix": getattr(args, "system_prompt_suffix", ""),
        },
        "measurement": measurement,
        "replay": {
            "completed": (replay_stats or {}).get("completed", len(full_results)),
            "errors": (replay_stats or {}).get(
                "errors",
                sum(1 for r in full_results if r.error is not None),
            ),
            "dispatched_events": (replay_stats or {}).get(
                "dispatched_events",
                len(full_results),
            ),
            "total_events": (replay_stats or {}).get(
                "total_events",
                len(full_results),
            ),
            "stopped_by_measurement_cap": (replay_stats or {}).get(
                "stopped_by_measurement_cap",
                False,
            ),
            "max_measurement_results": (replay_stats or {}).get(
                "max_measurement_results",
                getattr(args, "max_measurement_results", 0),
            ),
            "max_pending_request_tasks": (replay_stats or {}).get(
                "max_pending_request_tasks",
                getattr(args, "max_pending_request_tasks", 0),
            ),
            "effective_max_pending_request_tasks": (replay_stats or {}).get(
                "effective_max_pending_request_tasks",
            ),
            "pending_request_tasks_max": (replay_stats or {}).get(
                "pending_request_tasks_max",
            ),
            "measurement_completed": (replay_stats or {}).get(
                "measurement_completed",
                measurement.get("included_results"),
            ),
            "prewarm": (replay_stats or {}).get("prewarm", {}),
            "audio_start_trace_time": (replay_stats or {}).get(
                "audio_start_trace_time",
                getattr(args, "audio_start_trace_time", None),
            ),
            "audio_end_trace_time": (replay_stats or {}).get(
                "audio_end_trace_time",
                getattr(args, "audio_end_trace_time", None),
            ),
        },
        "trace_shape": getattr(args, "trace_shape", {}),
        "session_lifecycle": getattr(args, "session_lifecycle", {}),
        "totals": {
            "entries_dispatched": len(results),
            "completed_ok": len(ok_results),
            "errors": len(err_results),
            "interrupted": len(interrupted_results),
            "completed_without_interrupt": len(not_interrupted_ok),
            "wall_time_sec": round(wall_time, 2),
            "throughput_rps": round(len(ok_results) / wall_time, 4) if wall_time > 0 else 0,
            "useful_qps": useful_qps,
        },
        "latency": {
            "ttfp_sec": percentiles(ttfps),
            "audio_ttfp_sec": percentiles(audio_ttfps),
            "e2e_sec": percentiles(e2es),
            "tokens_per_sec": percentiles(tps_list),
        },
        "interrupt": {
            "stages": interrupt_stages,
            "wasted_talker_frames_total": percentiles(
                [float(x) for x in waste_total]),
            "wasted_talker_frames_inaudible": percentiles(
                [float(x) for x in waste_inaudible]),
        },
        "audio": {
            "audio_duration_sec": percentiles(audio_durations),
            "audio_post_ttfp_rtf": percentiles(audio_post_ttfp_rtfs),
            "audio_stream_rtf": percentiles(audio_stream_rtfs),
            "n_with_audio": len(audio_ttfps),
            "n_with_stream_rtf": len(audio_stream_rtfs),
            "streaming_continuity": audio_streaming_continuity,
        },
        "context": {
            "input_tokens_total": percentiles([float(x) for x in input_tokens]),
            "round_index": percentiles([float(x) for x in round_indices]),
            "length_profile": getattr(args, "length_profile", "trace"),
            "length_profile_stats": getattr(args, "length_profile_stats", {}),
            "seeded_history_entries": (replay_stats or {}).get("seeded_history_entries", 0),
            "warm_history_max_turns_per_user": (replay_stats or {}).get(
                "warm_history_max_turns_per_user", 0
            ),
            "prewarm": (replay_stats or {}).get("prewarm", {}),
            "history_record_mode": (replay_stats or {}).get(
                "history_record_mode", "generated"
            ),
        },
        "concurrency": {
            "max_inflight_observed": (replay_stats or {}).get("inflight_max", 0),
            "inflight_p50": (
                sorted((replay_stats or {}).get("inflight_samples", []))[
                    len((replay_stats or {}).get("inflight_samples", [])) // 2
                ] if (replay_stats or {}).get("inflight_samples") else 0
            ),
            "inflight_p90": (
                sorted((replay_stats or {}).get("inflight_samples", []))[
                    int(len((replay_stats or {}).get("inflight_samples", [])) * 0.9)
                ] if (replay_stats or {}).get("inflight_samples") else 0
            ),
            "limit": args.max_inflight,
        },
        "prefetch": {
            "mode": (replay_stats or {}).get("prefetch_mode", "none"),
            "url": (replay_stats or {}).get("prefetch_url"),
            "allow_prestart": (replay_stats or {}).get("prefetch_allow_prestart", False),
            "include_current_user": (replay_stats or {}).get(
                "prefetch_include_current_user",
                False,
            ),
            "current_turn": (replay_stats or {}).get(
                "prefetch_current_turn",
                False,
            ),
            "same_time_request_first": (replay_stats or {}).get(
                "prefetch_same_time_request_first",
                False,
            ),
            "clipped_start_limit": (replay_stats or {}).get(
                "prefetch_clipped_start_limit",
                -1,
            ),
            "clipped_start_kept": (replay_stats or {}).get(
                "prefetch_clipped_start_kept",
                0,
            ),
            "clipped_start_skipped": (replay_stats or {}).get(
                "prefetch_clipped_start_skipped",
                0,
            ),
            "current_prefill_lead_ms": (replay_stats or {}).get(
                "current_prefill_lead_ms"
            ),
            "current_prefill_max_tokens": (replay_stats or {}).get(
                "current_prefill_max_tokens"
            ),
            "min_remaining_lead_ms": (replay_stats or {}).get(
                "prefetch_min_remaining_lead_ms",
                0.0,
            ),
            "history_min_remaining_lead_ms": (replay_stats or {}).get(
                "prefetch_history_min_remaining_lead_ms",
                0.0,
            ),
            "current_min_remaining_lead_ms": (replay_stats or {}).get(
                "current_prefill_min_remaining_lead_ms",
                0.0,
            ),
            "attempts": (replay_stats or {}).get("prefetch_attempts", 0),
            "ok": (replay_stats or {}).get("prefetch_ok", 0),
            "by_kind": (replay_stats or {}).get("prefetch_by_kind", {}),
            "skipped_no_history": (replay_stats or {}).get(
                "prefetch_skipped_no_history", 0
            ),
            "skipped_admission": (replay_stats or {}).get(
                "prefetch_skipped_admission", 0
            ),
            "skip_reasons": (replay_stats or {}).get(
                "prefetch_skip_reasons", {}
            ),
            "wall_s": percentiles(
                [float(x) for x in (replay_stats or {}).get("prefetch_wall_s", [])]
            ),
            "prefix_token_estimate": percentiles(
                [
                    float(x)
                    for x in (replay_stats or {}).get(
                        "prefetch_prefix_token_estimates", []
                    )
                ]
            ),
            "deadline_estimate_sources": (replay_stats or {}).get(
                "prefetch_deadline_estimate_sources", {}
            ),
            "deadline_prefix_buckets": (replay_stats or {}).get(
                "prefetch_deadline_prefix_buckets", {}
            ),
            "hit_used": sum(1 for r in ok_results if r.kv_prefetch_hit_used),
            "live_skip_offload": sum(1 for r in ok_results if r.kv_skip_offload),
            "live_lookup_states": {
                state: sum(1 for r in ok_results if r.kv_prefetch_lookup_state == state)
                for state in sorted(
                    {
                        str(r.kv_prefetch_lookup_state)
                        for r in ok_results
                        if r.kv_prefetch_lookup_state
                    }
                )
            },
            "live_lookup_key_kinds": {
                kind: sum(
                    1 for r in ok_results if r.kv_prefetch_lookup_key_kind == kind
                )
                for kind in sorted(
                    {
                        str(r.kv_prefetch_lookup_key_kind)
                        for r in ok_results
                        if r.kv_prefetch_lookup_key_kind
                    }
                )
            },
            "live_warm_scopes": {
                scope: sum(1 for r in ok_results if r.kv_prefetch_warm_scope == scope)
                for scope in sorted(
                    {
                        str(r.kv_prefetch_warm_scope)
                        for r in ok_results
                        if r.kv_prefetch_warm_scope
                    }
                )
            },
            "live_gpu_pin_requested": sum(
                1 for r in ok_results if r.kv_prefetch_gpu_pin_requested
            ),
            "live_abort_aborted": sum(
                int(r.kv_live_abort_aborted or 0) for r in ok_results
            ),
        },
        "kv_offload": (replay_stats or {}).get("kv_offload_metrics", {}),
        "kv_hit_ratio": kv_hit_ratio,
        "kv_lru": kv_lru,
    }

    # ---- sanity check ----------------------------------------------------
    # If the client is the bottleneck (not enough concurrency, slow event
    # loop, etc.) then  wall * c / n_reqs  should be roughly equal to the
    # average per-request E2E. Big deviation (>30%) means either client is
    # under-driving the server (ratio>>1) or the server is queueing beyond
    # max-inflight (ratio<<1, only possible with HTTP/2 multiplexing).
    n_done = len(ok_results)
    avg_e2e = (sum(e2es) / len(e2es)) if e2es else 0.0
    expected_e2e = (wall_time * args.max_inflight / n_done) if n_done > 0 else 0.0
    sanity_ratio = (expected_e2e / avg_e2e) if avg_e2e > 0 else 0.0
    sanity_ok = bool(0.7 <= sanity_ratio <= 1.3) if avg_e2e > 0 else False
    summary["sanity_check"] = {
        "avg_e2e_sec": round(avg_e2e, 4),
        "expected_e2e_sec": round(expected_e2e, 4),
        "ratio_expected_over_actual": round(sanity_ratio, 4),
        "within_tolerance": sanity_ok,
        "note": (
            "ratio ~1.0 = healthy server-bound; ratio>>1 = client under-driving; "
            "ratio<<1 = server queueing past max-inflight"
        ),
    }

    summary_path = result_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[SUMMARY] Written to {summary_path}")
    print(f"  Total: {len(ok_results)} OK / {len(err_results)} errors")
    print(f"  Throughput: {summary['totals']['throughput_rps']:.3f} req/s")
    if ttfps:
        print(f"  TTFP: p50={statistics.median(ttfps)*1000:.1f}ms "
              f"p90={sorted(ttfps)[int(len(ttfps)*0.9)]*1000:.1f}ms")
    if e2es:
        print(f"  E2E:  p50={statistics.median(e2es)*1000:.1f}ms "
              f"p90={sorted(e2es)[int(len(e2es)*0.9)]*1000:.1f}ms")
    if audio_post_ttfp_rtfs:
        print(
            "  Audio post-TTFP RTF: "
            f"p50={statistics.median(audio_post_ttfp_rtfs):.2f} "
            f"p90={sorted(audio_post_ttfp_rtfs)[int(len(audio_post_ttfp_rtfs)*0.9)]:.2f}"
        )
    if audio_stream_rtfs:
        print(
            "  Audio stream RTF: "
            f"p50={statistics.median(audio_stream_rtfs):.2f} "
            f"p90={sorted(audio_stream_rtfs)[int(len(audio_stream_rtfs)*0.9)]:.2f}"
        )
    sc_audio = summary["audio"].get("streaming_continuity") or {}
    if sc_audio.get("n_evaluated"):
        underrun = sc_audio.get("audio_underrun_sec") or {}
        ok_rate = sc_audio.get("ok_rate")
        p90 = underrun.get("p90")
        p99 = underrun.get("p99")
        ok_text = f"{ok_rate * 100:.1f}%" if isinstance(ok_rate, (int, float)) else "-"
        p90_text = f"{p90:.3f}s" if isinstance(p90, (int, float)) else "-"
        p99_text = f"{p99:.3f}s" if isinstance(p99, (int, float)) else "-"
        print(
            "  Audio continuity: "
            f"ok_rate={ok_text} underrun_p90={p90_text} "
            f"p99={p99_text} n={sc_audio.get('n_evaluated')}"
        )
    if input_tokens:
        print(f"  Input tokens: mean={statistics.mean(input_tokens):.0f} "
              f"max={max(input_tokens)}")
    pf = summary.get("prefetch", {})
    if pf and pf.get("mode") != "none":
        wall_pf = pf.get("wall_s") or {}
        p50 = wall_pf.get("p50")
        p90 = wall_pf.get("p90")
        p50_s = f"{p50 * 1000:.1f}ms" if isinstance(p50, (int, float)) else "-"
        p90_s = f"{p90 * 1000:.1f}ms" if isinstance(p90, (int, float)) else "-"
        print(
            f"  Prefetch: ok={pf.get('ok')}/{pf.get('attempts')} "
            f"skipped_no_history={pf.get('skipped_no_history')} "
            f"wall_p50={p50_s} wall_p90={p90_s}"
        )
        if pf.get("by_kind"):
            print(f"  Prefetch by kind: {pf.get('by_kind')}")
    kvh = summary.get("kv_hit_ratio", {})
    if kvh:
        def _fmt_ratio(value: Any) -> str:
            return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else "-"

        print(
            "  KV hit ratio: "
            f"local={_fmt_ratio(kvh.get('local_prefix_hit_ratio_tokens'))} "
            f"external={_fmt_ratio(kvh.get('external_prefix_hit_ratio_tokens'))} "
            f"overall={_fmt_ratio(kvh.get('overall_kv_hit_ratio_tokens'))}"
        )
    lru = summary.get("kv_lru", {})
    if lru and (lru.get("source") or lru.get("enabled")):
        print(
            "  KV LRU: "
            f"gpu_evictions={lru.get('gpu_evictions', 0)} "
            f"recent_eviction_misses={lru.get('recent_eviction_misses', 0)}"
        )
    sc = summary.get("sanity_check", {})
    if sc:
        flag = "OK" if sc.get("within_tolerance") else "WARN"
        print(f"  [sanity {flag}] avg_e2e={sc['avg_e2e_sec']*1000:.0f}ms "
              f"expected={sc['expected_e2e_sec']*1000:.0f}ms "
              f"ratio={sc['ratio_expected_over_actual']:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Replay interactive conversation trace against vllm-omni server"
    )
    ap.add_argument("--url", default=DEFAULT_URL,
                    help="Server endpoint URL")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Model name for API requests")
    ap.add_argument("--mode", choices=list(MODE_PRESETS.keys()),
                    default="smoke",
                    help="Preset mode (smoke/mini/full_part1/full_all)")
    ap.add_argument("--parts", type=int, nargs="+", default=None,
                    help="Which trace parts to load (1-7). Overrides mode preset.")
    ap.add_argument("--time-limit", type=int, default=None,
                    help="Max trace timestamp in seconds. Overrides mode preset.")
    ap.add_argument("--time-start", type=int, default=None,
                    help="Min trace timestamp in seconds (sample a window).")
    ap.add_argument("--cold-cache", action="store_true",
                    help="Phase-4 control: kill prefix caching by per-request "
                         "system-prompt nonce + force round=0 + drop history.")
    ap.add_argument("--speedup", type=float, default=1.0,
                    help="Time compression factor (e.g., 10 = 10x faster)")
    ap.add_argument("--max-inflight", type=int, default=8,
                    help="Maximum concurrent requests")
    ap.add_argument("--length-profile",
                    choices=["trace", "interactive-len1024", "trace-scaled"],
                    default="trace",
                    help="Synthetic length override. interactive-len1024 "
                         "preserves trace timestamps and sessions while "
                         "sampling query/response lengths from normal "
                         "distributions with default means 512/1024 tokens. "
                         "trace-scaled preserves the trace's long/short "
                         "ordering and multiplies query/response lengths by "
                         "the scale factors below.")
    ap.add_argument("--input-tokens-mean", type=int, default=None,
                    help="Mean query length for synthetic length profiles.")
    ap.add_argument("--input-tokens-std", type=int, default=128,
                    help="Stddev for synthetic query lengths.")
    ap.add_argument("--output-tokens-mean", type=int, default=None,
                    help="Mean response length for synthetic length profiles.")
    ap.add_argument("--output-tokens-std", type=int, default=256,
                    help="Stddev for synthetic response lengths.")
    ap.add_argument("--input-tokens-scale", type=float, default=1.0,
                    help="Query length multiplier for --length-profile=trace-scaled.")
    ap.add_argument("--output-tokens-scale", type=float, default=1.0,
                    help="Response length multiplier for --length-profile=trace-scaled.")
    ap.add_argument("--min-input-tokens", type=int, default=1)
    ap.add_argument("--max-input-tokens", type=int, default=0,
                    help="0 means no upper clamp.")
    ap.add_argument("--min-output-tokens", type=int, default=1)
    ap.add_argument("--max-output-tokens", type=int, default=0,
                    help="0 means no upper clamp.")
    ap.add_argument("--force-trace-output-tokens", action="store_true",
                    help="Diagnostic mode: force the first generation stage "
                         "to emit exactly the trace response length by setting "
                         "min_tokens=max_tokens, ignore_eos=true, and removing "
                         "first-stage stop tokens. This reduces paired-run "
                         "output-length drift in pressure workload validation.")
    ap.add_argument("--max-requests", type=int, default=0,
                    help="Optional cap after trace loading. Intended for "
                         "smoke tests; 0 means use all entries in the window.")
    ap.add_argument("--max-requests-policy",
                    choices=[
                        "first",
                        "session_balanced",
                        "real_multiturn",
                        "real_multiturn_peak",
                        "real_multiturn_full_sessions",
                        "real_multiturn_full_sessions_peak",
                    ],
                    default="first",
                    help="How to cap loaded entries. 'session_balanced' keeps "
                         "multiple turns from selected users so offload/reuse "
                         "experiments contain same-run session returns. "
                         "'real_multiturn' keeps only session prefixes that "
                         "start at round_index=0 and contain the requested "
                         "number of consecutive turns. 'real_multiturn_peak' "
                         "selects such prefixes from the highest-overlap "
                         "logical session window. "
                         "'real_multiturn_full_sessions' selects whole "
                         "consecutive sessions in trace order to preserve "
                         "real turn-count variation under a request budget. "
                         "'real_multiturn_full_sessions_peak' selects whole "
                         "consecutive sessions from that peak-overlap region "
                         "to preserve real turn-count variation under a "
                         "request budget.")
    ap.add_argument("--metrics-start-index", type=int, default=0,
                    help="Only include loaded trace entries with this 0-based "
                         "entry index or later in summary latency/throughput "
                         "aggregates. The replay still dispatches earlier "
                         "entries as warm/cache-fill traffic. 0 disables this "
                         "filter.")
    ap.add_argument("--metrics-start-trace-time", type=int, default=None,
                    help="Only include requests whose trace timestamp is at "
                         "least this value in summary latency/throughput "
                         "aggregates. The replay still dispatches earlier "
                         "entries as warm/cache-fill traffic.")
    ap.add_argument("--metrics-end-trace-time", type=int, default=None,
                    help="Only include requests whose trace timestamp is below "
                         "this value in summary latency/throughput aggregates. "
                         "The replay still dispatches later entries.")
    ap.add_argument("--max-measurement-results", type=int, default=0,
                    help="Stop dispatching new events after this many completed "
                         "requests pass the metrics-start/metrics-end filters. "
                         "Already-dispatched requests are still allowed to "
                         "finish. 0 disables early stop.")
    ap.add_argument("--max-pending-request-tasks", type=int, default=0,
                    help="Bound request tasks that have been created but not "
                         "completed. 0 keeps historical unbounded behavior, "
                         "except when --max-measurement-results is set, where "
                         "the default cap is --max-inflight. This prevents "
                         "measurement early-stop from being bypassed by a large "
                         "client-side queue ahead of the concurrency semaphore.")
    ap.add_argument("--audio-start-trace-time", type=int, default=None,
                    help="When --with-audio is set, issue only requests at or "
                         "after this trace timestamp in audio mode. Earlier "
                         "requests remain text-only warm/cache-fill traffic.")
    ap.add_argument("--audio-end-trace-time", type=int, default=None,
                    help="When --with-audio is set, issue only requests below "
                         "this trace timestamp in audio mode. Later requests "
                         "remain text-only.")
    ap.add_argument("--session-balanced-min-turns", type=int, default=2,
                    help="Minimum in-window turns to keep per selected session "
                         "when --max-requests-policy=session_balanced; for "
                         "real_multiturn this is the fixed turns-per-session "
                         "prefix length.")
    ap.add_argument("--session-balanced-max-turns", type=int, default=0,
                    help="Optional per-session turn cap for real_multiturn "
                         "selection policies. 0 preserves full selected "
                         "sessions. Use a positive value to keep original-like "
                         "multi-turn pressure while avoiding extreme single-user "
                         "tails in iteration lanes.")
    ap.add_argument("--enforce-session-order", action="store_true",
                    help="Serialize requests from the same user so later turns "
                         "are built only after the previous turn's response has "
                         "been recorded. Use this for true multi-turn KV reuse "
                         "experiments.")
    ap.add_argument("--warm-history-before-start", action="store_true",
                    help="Seed committed per-user history from trace entries "
                         "before --time-start. This preserves multi-turn "
                         "context for windowed KV-cache experiments.")
    ap.add_argument("--warm-history-max-turns-per-user", type=int, default=4,
                    help="When warm-history seeding is enabled, retain at most "
                         "this many recent user/assistant turns per target user. "
                         "Use 0 for unlimited.")
    ap.add_argument("--prewarm-history-before-replay", action="store_true",
                    help="After seeding warm history, call /v1/omni/prefetch_session "
                         "for selected committed history prefixes before dispatching "
                         "live replay requests. This creates compact KV pressure "
                         "without replaying the entire warmup prefix.")
    ap.add_argument("--prewarm-max-users", type=int, default=0,
                    help="Maximum number of seeded user histories to prewarm. "
                         "0 means all eligible target users.")
    ap.add_argument("--prewarm-min-history-tokens", type=int, default=1,
                    help="Minimum seeded history token estimate required for prewarm.")
    ap.add_argument("--prewarm-max-history-tokens", type=int, default=0,
                    help="Maximum seeded history token estimate allowed for prewarm. "
                         "0 disables the upper bound.")
    ap.add_argument("--prewarm-max-inflight", type=int, default=0,
                    help="Maximum concurrent prewarm requests. 0 reuses "
                         "--prefetch-max-inflight.")
    ap.add_argument("--prewarm-timeout-sec", type=float, default=None,
                    help="HTTP timeout for each prewarm request. Defaults to "
                         "--prefetch-timeout-sec.")
    ap.add_argument("--prewarm-sort",
                    choices=["history_tokens_desc", "history_tokens_asc", "trace_order"],
                    default="history_tokens_desc",
                    help="Order used when --prewarm-max-users caps eligible histories.")
    ap.add_argument("--history-record-mode",
                    choices=["generated", "trace_dummy", "none"],
                    default="generated",
                    help="How to append successful in-window assistant turns "
                         "to conversation history. 'generated' preserves the "
                         "actual model text. 'trace_dummy' records deterministic "
                         "trace-length dummy assistant turns to remove cross-run "
                         "generation drift from later prompts. 'none' disables "
                         "in-window history growth.")
    ap.add_argument("--result-dir", type=str, default=None,
                    help="Output directory for results")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for dummy text generation")
    ap.add_argument("--progress-interval", type=int, default=50,
                    help="Print progress every N completed requests")
    # ---- Audio mode + barge-in policy -----------------------------------
    ap.add_argument("--with-audio", action="store_true",
                    help="Issue requests in modalities=[text,audio] mode "
                         "(Qwen3-Omni). Required for governor / VAD / waste "
                         "experiments.")
    ap.add_argument("--policy", choices=["none", "fixed_time"],
                    default="none",
                    help="Barge-in policy. 'fixed_time' samples a per-"
                         "request threshold and aborts when reached.")
    ap.add_argument("--barge-in-prob", "--interrupt-prob",
                    dest="interrupt_prob", type=float, default=1.0,
                    help="Per-request probability (Bernoulli) of injecting "
                         "a barge-in. Only used when --policy != 'none'.")
    ap.add_argument("--barge-in-after-seconds", "--interrupt-after-seconds",
                    dest="interrupt_after_seconds", type=float, default=2.0,
                    help="fixed_time mean threshold seconds.")
    ap.add_argument("--barge-in-jitter-sec", "--interrupt-jitter-sec",
                    dest="interrupt_jitter_sec", type=float, default=0.5,
                    help="fixed_time uniform jitter range (\xb1jitter).")
    ap.add_argument("--barge-in-time-anchor", "--interrupt-time-anchor",
                    dest="interrupt_time_anchor", choices=["t0", "ttfp"],
                    default="ttfp",
                    help="fixed_time time origin: 't0'=request issued, "
                         "'ttfp'=first audio chunk received.")
    ap.add_argument("--record-audio-timeline", action="store_true",
                    help="Write per-audio-chunk receive time and decoded WAV "
                         "duration to per_request.jsonl. Core audio metrics "
                         "are computed even when this is disabled.")
    ap.add_argument("--audio-continuity-threshold-sec", type=float,
                    default=get_audio_continuity_threshold_s(),
                    help="Max allowed audio underrun seconds for continuity "
                         "OK. Defaults to LIVESERVE_AUDIO_CONTINUITY_THRESHOLD_S, "
                         "then VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S, "
                         "then 0.1.")
    ap.add_argument("--audio-continuity-chunk-mode",
                    choices=["fragment", "cumulative", "auto"],
                    default="fragment",
                    help="Interpret decoded WAV chunk durations as independent "
                         "fragments, cumulative totals, or auto-detect.")
    ap.add_argument("--continuity-include-interrupted", action="store_true",
                    help="Include interrupted requests in the main audio "
                         "continuity denominator. By default they are excluded.")
    ap.add_argument("--audio-metadata-path", type=Path, default=None,
                    help="Optional per_request.jsonl used as oracle audio "
                         "metadata. Matching (user_id, round_index) rows "
                         "provide omni_audio_total_sec to the server. Use only "
                         "for upper-bound experiments.")
    ap.add_argument("--request-timeout-sec", type=float, default=300.0,
                    help="End-to-end HTTP timeout for each chat request. "
                         "Timeouts are recorded as per-request errors so one "
                         "stalled stream cannot block a full matrix run.")
    # ---- KV-cache preload path ------------------------------------------
    ap.add_argument("--preload-mode", "--prefetch-mode",
                    dest="prefetch_mode", choices=["none", "omni"],
                    default="none",
                    help="History-only service-side preload. 'omni' calls "
                         "/v1/omni/prefetch_session before returning-session "
                         "requests.")
    ap.add_argument("--preload-url", "--prefetch-url",
                    dest="prefetch_url", default=None,
                    help="Preload endpoint URL. Defaults to the chat URL's "
                         "/v1/omni/prefetch_session sibling.")
    ap.add_argument("--preload-lead-ms", "--prefetch-lead-ms",
                    dest="prefetch_lead_ms", type=float, default=150.0,
                    help="Simulated VAD lead time before request arrival. The "
                         "driver schedules history-only preload this many "
                         "wall-clock ms earlier.")
    ap.add_argument("--preload-max-tokens", "--prefetch-max-tokens",
                    dest="prefetch_max_tokens", type=int, default=1,
                    help="Warm preload max_tokens.")
    ap.add_argument("--preload-timeout-sec", "--prefetch-timeout-sec",
                    dest="prefetch_timeout_sec", type=float, default=30.0,
                    help="HTTP timeout for each preload request.")
    ap.add_argument("--preload-max-inflight", "--prefetch-max-inflight",
                    dest="prefetch_max_inflight", type=int, default=8,
                    help="Maximum concurrent preload requests.")
    ap.add_argument("--preload-allow-prestart", "--prefetch-allow-prestart",
                    dest="prefetch_allow_prestart", action="store_true",
                    help="Allow preload events before --time-start. This gives "
                         "the first in-window returning-session request its "
                         "simulated VAD lead instead of clipping the preload "
                         "to the same timestamp as the live request.")
    ap.add_argument("--preload-include-current-user", "--prefetch-include-current-user",
                    dest="prefetch_include_current_user", action="store_true",
                    help="Include the current user turn in the preload payload "
                         "and use the full-prompt hash as the live matching key. "
                         "This is the legacy one-stage approximation of "
                         "prefill-while-speaking.")
    ap.add_argument("--preload-current-turn", "--prefetch-current-turn",
                    dest="prefetch_current_turn", action="store_true",
                    help="Run a second preload event for the current user turn. "
                         "The live request uses the full-prompt key first and "
                         "falls back to the history key if current-turn prefill "
                         "was skipped or cancelled.")
    ap.add_argument("--current-prefill-lead-ms", type=float, default=None,
                    help="Simulated VAD lead for the current-turn prefill event. "
                         "Defaults to --prefetch-lead-ms.")
    ap.add_argument("--current-prefill-max-tokens", type=int, default=None,
                    help="Warm request max_tokens for current-turn prefill. "
                         "Defaults to --prefetch-max-tokens.")
    ap.add_argument("--current-prefill-timeout-sec", type=float, default=None,
                    help="HTTP timeout for current-turn prefill. Defaults to "
                         "--prefetch-timeout-sec.")
    ap.add_argument("--preload-min-remaining-lead-ms", "--prefetch-min-remaining-lead-ms",
                    dest="prefetch_min_remaining_lead_ms", type=float, default=0.0,
                    help="Skip a preload if client-side queueing leaves less "
                         "than this many wall-clock ms before the live request. "
                         "Default 0 disables the late-drop guard.")
    ap.add_argument("--preload-history-min-remaining-lead-ms", "--prefetch-history-min-remaining-lead-ms",
                    dest="prefetch_history_min_remaining_lead_ms", type=float, default=None,
                    help="History-preload-specific late-drop guard. Defaults "
                         "to --preload-min-remaining-lead-ms when omitted.")
    ap.add_argument("--current-prefill-min-remaining-lead-ms", type=float, default=None,
                    help="Current-turn-prefill-specific late-drop guard. "
                         "Defaults to --preload-min-remaining-lead-ms when "
                         "omitted.")
    ap.add_argument("--preload-clipped-start-limit", "--prefetch-clipped-start-limit",
                    dest="prefetch_clipped_start_limit", type=int, default=-1,
                    help="Maximum number of history preload events clipped "
                         "to --time-start that may be issued. Negative keeps "
                         "the legacy unlimited behavior; skipped rows are "
                         "recorded with driver_clipped_start_limit.")
    ap.add_argument("--preload-same-time-request-first", "--prefetch-same-time-request-first",
                    dest="prefetch_same_time_request_first", action="store_true",
                    help="When a preload and live request share the same "
                         "trace timestamp, dispatch the live request first.")
    ap.add_argument("--system-prompt-suffix", default="",
                    help="Optional suffix appended to the system prompt. The "
                         "large-scale KV runner uses this to avoid cross-cell "
                         "prefix-cache contamination.")
    return ap


async def amain():
    parser = build_parser()
    args = parser.parse_args()

    # Apply mode presets (CLI overrides take priority)
    preset = MODE_PRESETS.get(args.mode, {})
    if args.parts is None:
        args.parts = preset.get("parts", [1])
    if args.time_limit is None:
        args.time_limit = preset.get("time_limit")

    # Result directory
    if args.result_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.result_dir = str(
            Path(__file__).parent / "results" / f"trace_{args.mode}_{ts}"
        )
    result_dir = Path(args.result_dir)

    # Load trace
    print(f"[LOAD] Loading trace parts={args.parts}, "
          f"time_start={args.time_start}s, time_limit={args.time_limit}s")
    entries = load_trace_entries(args.parts, args.time_limit, args.time_start)
    if not entries:
        print("[ERROR] No trace entries loaded!", file=sys.stderr)
        sys.exit(1)
    if args.max_requests_policy == "real_multiturn":
        before = len(entries)
        request_cap = args.max_requests if args.max_requests > 0 else len(entries)
        print(
            f"[LOAD] Selecting real multiturn session prefixes: cap={request_cap} "
            f"turns_per_session={args.session_balanced_min_turns}"
        )
        entries = cap_trace_entries(
            entries,
            request_cap,
            policy=args.max_requests_policy,
            session_min_turns=args.session_balanced_min_turns,
            session_max_turns=args.session_balanced_max_turns,
        )
        print(f"[LOAD] Real multiturn selected entries: {before} -> {len(entries)}")
        if not entries:
            print("[ERROR] real_multiturn selection produced no entries!", file=sys.stderr)
            sys.exit(1)
    elif args.max_requests and args.max_requests > 0 and len(entries) > args.max_requests:
        print(
            f"[LOAD] Capping entries: {len(entries)} -> {args.max_requests} "
            f"(policy={args.max_requests_policy})"
        )
        entries = cap_trace_entries(
            entries,
            args.max_requests,
            policy=args.max_requests_policy,
            session_min_turns=args.session_balanced_min_turns,
            session_max_turns=args.session_balanced_max_turns,
        )
        if not entries:
            print(
                "[ERROR] max request selection produced no entries: "
                f"policy={args.max_requests_policy} cap={args.max_requests} "
                f"session_min_turns={args.session_balanced_min_turns}",
                file=sys.stderr,
            )
            sys.exit(1)
    args.length_profile_stats = apply_length_profile(entries, args)
    trace_shape = summarize_trace_shape(entries)
    args.trace_shape = trace_shape
    args.session_lifecycle = summarize_session_lifecycle(entries)
    print(f"[LOAD] Loaded {len(entries)} entries from {len(set(e.user_id for e in entries))} users")
    print(f"[LOAD] Time range: {entries[0].timestamp}s - {entries[-1].timestamp}s")
    if args.length_profile != "trace":
        length_stats = args.length_profile_stats
        print(
            "[LOAD] Length profile: "
            f"{args.length_profile} "
            f"query_mean={length_stats['query_length']['mean']:.1f} "
            f"response_mean={length_stats['response_length']['mean']:.1f}"
        )
    print(
        "[LOAD] Same-run returns: "
        f"{trace_shape['same_run_returns']} "
        f"(users={trace_shape['same_run_users']}, "
        f"ratio={trace_shape['same_run_return_ratio']:.3f}, "
        f"gap_p50={trace_shape['same_run_gap_p50_sec']}, "
        f"gap_max={trace_shape['same_run_gap_max_sec']})"
    )
    lifecycle = args.session_lifecycle
    print(
        "[LOAD] Session lifecycle: "
        f"sessions={lifecycle['session_number']} "
        f"open_peak={lifecycle['logical_open_sessions_peak']} "
        f"waiting_peak={lifecycle['logical_waiting_sessions_peak']} "
        f"idle_pending_kv_peak_est="
        f"{lifecycle['idle_pending_retained_history_kv_gib_peak_est']:.3f}GiB"
    )
    print()
    audio_metadata = load_audio_metadata(args.audio_metadata_path)
    if args.audio_metadata_path is not None:
        print(
            "[LOAD] Audio metadata: "
            f"{len(audio_metadata)} rows from {args.audio_metadata_path}"
        )

    # Build interrupt policy from CLI args
    policy = BargeInPolicy(
        policy=args.policy,
        after_seconds=args.interrupt_after_seconds,
        jitter_sec=args.interrupt_jitter_sec,
        time_anchor=args.interrupt_time_anchor,
        interrupt_prob=args.interrupt_prob,
    )

    warm_entries: list[TraceEntry] = []
    if args.warm_history_before_start and args.time_start is not None:
        warm_entries = load_trace_entries(
            args.parts,
            time_limit=max(0, args.time_start - 1),
            time_start=None,
        )
        target_users = {e.user_id for e in entries}
        warm_entries = [e for e in warm_entries if e.user_id in target_users]
        print(
            f"[LOAD] Warm history entries before start: {len(warm_entries)} "
            f"for {len(target_users)} target users "
            f"(max_turns={args.warm_history_max_turns_per_user})"
        )

    # Run replay
    t_start = time.perf_counter()
    results, replay_stats = await replay_trace(
        entries=entries,
        url=args.url,
        model=args.model,
        max_inflight=args.max_inflight,
        speedup=args.speedup,
        result_dir=result_dir,
        progress_interval=args.progress_interval,
        cold_mode=args.cold_cache,
        with_audio=args.with_audio,
        policy=policy,
        seed=args.seed,
        record_audio_timeline=args.record_audio_timeline,
        audio_continuity_threshold_sec=args.audio_continuity_threshold_sec,
        audio_continuity_chunk_mode=args.audio_continuity_chunk_mode,
        continuity_include_interrupted=args.continuity_include_interrupted,
        request_timeout_sec=args.request_timeout_sec,
        warm_history_entries=warm_entries,
        warm_history_max_turns_per_user=args.warm_history_max_turns_per_user,
        prewarm_history_before_replay=args.prewarm_history_before_replay,
        prewarm_max_users=args.prewarm_max_users,
        prewarm_min_history_tokens=args.prewarm_min_history_tokens,
        prewarm_max_history_tokens=args.prewarm_max_history_tokens,
        prewarm_max_inflight=args.prewarm_max_inflight,
        prewarm_timeout_sec=args.prewarm_timeout_sec,
        prewarm_sort=args.prewarm_sort,
        prefetch_mode=args.prefetch_mode,
        prefetch_url=args.prefetch_url,
        prefetch_lead_ms=args.prefetch_lead_ms,
        prefetch_max_tokens=args.prefetch_max_tokens,
        prefetch_timeout_sec=args.prefetch_timeout_sec,
        prefetch_max_inflight=args.prefetch_max_inflight,
        prefetch_allow_prestart=args.prefetch_allow_prestart,
        prefetch_include_current_user=args.prefetch_include_current_user,
        prefetch_current_turn=args.prefetch_current_turn,
        current_prefill_lead_ms=args.current_prefill_lead_ms,
        current_prefill_max_tokens=args.current_prefill_max_tokens,
        current_prefill_timeout_sec=args.current_prefill_timeout_sec,
        prefetch_min_remaining_lead_ms=args.prefetch_min_remaining_lead_ms,
        prefetch_history_min_remaining_lead_ms=(
            args.prefetch_history_min_remaining_lead_ms
        ),
        current_prefill_min_remaining_lead_ms=(
            args.current_prefill_min_remaining_lead_ms
        ),
        prefetch_clipped_start_limit=args.prefetch_clipped_start_limit,
        prefetch_same_time_request_first=args.prefetch_same_time_request_first,
        system_prompt_suffix=args.system_prompt_suffix,
        enforce_session_order=args.enforce_session_order,
        history_record_mode=args.history_record_mode,
        force_trace_output_tokens=args.force_trace_output_tokens,
        max_measurement_results=args.max_measurement_results,
        max_pending_request_tasks=args.max_pending_request_tasks,
        metrics_start_index=args.metrics_start_index,
        metrics_start_trace_time=args.metrics_start_trace_time,
        metrics_end_trace_time=args.metrics_end_trace_time,
        audio_start_trace_time=args.audio_start_trace_time,
        audio_end_trace_time=args.audio_end_trace_time,
        audio_metadata=audio_metadata,
    )
    wall_time = time.perf_counter() - t_start

    # Generate summary
    generate_summary(results, args, result_dir, wall_time, replay_stats)

    # Save args
    meta_path = result_dir / "meta.json"
    args_meta = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with open(meta_path, "w") as f:
        json.dump({
            "args": args_meta,
            "trace_entries": len(entries),
            "unique_users": len(set(e.user_id for e in entries)),
            "trace_time_range_sec": entries[-1].timestamp - entries[0].timestamp,
            "wall_time_sec": round(wall_time, 2),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, f, indent=2)

    print(f"\n[DONE] Results in: {result_dir}")


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
