"""Audio streaming continuity helpers for Omni benchmarks.

The metric models a player that starts when the first audio chunk arrives and
plays at real time.  The reported underrun is the largest extra prebuffering
time that would have been needed to avoid a gap between streamed chunks.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Literal


DEFAULT_AUDIO_CONTINUITY_THRESHOLD_S = 0.1
AudioChunkDurationMode = Literal["fragment", "cumulative", "auto"]


@dataclass(frozen=True)
class ContinuityStats:
    """Per-request audio continuity result."""

    audio_underrun_s: float = 0.0
    audio_underrun_event_count: int = 0
    audio_continuity_ok: bool = True
    audio_continuity_threshold_s: float = DEFAULT_AUDIO_CONTINUITY_THRESHOLD_S

    @property
    def max_underrun_s(self) -> float:
        return self.audio_underrun_s

    @property
    def underrun_event_count(self) -> int:
        return self.audio_underrun_event_count

    @property
    def is_continuous(self) -> bool:
        return self.audio_continuity_ok


def get_audio_continuity_threshold_s(
    default: float = DEFAULT_AUDIO_CONTINUITY_THRESHOLD_S,
    *,
    env: dict[str, str] | None = None,
) -> float:
    """Read the continuity threshold from LiveServe/vLLM-Omni env vars."""

    source = os.environ if env is None else env
    for name in (
        "LIVESERVE_AUDIO_CONTINUITY_THRESHOLD_S",
        "VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S",
    ):
        raw = source.get(name)
        if raw is None or raw == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0.0:
            return value
    return default


def _clean_nonnegative(values: Iterable[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            f = float(value)
        except (TypeError, ValueError):
            f = 0.0
        out.append(max(0.0, f) if math.isfinite(f) else 0.0)
    return out


def _looks_cumulative(durations_s: list[float]) -> bool:
    if len(durations_s) <= 1:
        return False
    eps = 1e-9
    if any(curr <= prev + eps for prev, curr in zip(durations_s, durations_s[1:])):
        return False
    total_as_fragments = sum(durations_s)
    if total_as_fragments <= 0.0:
        return False
    return durations_s[-1] < total_as_fragments * 0.9


def normalize_audio_chunk_durations(
    chunk_durations_s: Iterable[float],
    *,
    mode: AudioChunkDurationMode = "fragment",
) -> list[float]:
    """Convert fragment/cumulative WAV durations into per-chunk durations."""

    durations = _clean_nonnegative(chunk_durations_s)
    if mode == "fragment":
        return durations
    if mode == "auto":
        mode = "cumulative" if _looks_cumulative(durations) else "fragment"
    if mode != "cumulative":
        raise ValueError(f"unknown audio chunk duration mode: {mode!r}")

    chunks: list[float] = []
    prev = 0.0
    for total in durations:
        chunks.append(max(0.0, total - prev))
        prev = max(prev, total)
    return chunks


def compute_continuity_stats_from_durations(
    chunk_arrival_times_s: list[float],
    chunk_durations_s: list[float],
    threshold_s: float = DEFAULT_AUDIO_CONTINUITY_THRESHOLD_S,
    *,
    chunk_duration_mode: AudioChunkDurationMode = "fragment",
) -> ContinuityStats:
    """Compute audio underrun from chunk arrival times and WAV durations."""

    try:
        threshold_s = float(threshold_s)
    except (TypeError, ValueError):
        threshold_s = DEFAULT_AUDIO_CONTINUITY_THRESHOLD_S
    if not math.isfinite(threshold_s) or threshold_s < 0.0:
        threshold_s = DEFAULT_AUDIO_CONTINUITY_THRESHOLD_S

    if not chunk_arrival_times_s:
        return ContinuityStats(audio_continuity_threshold_s=threshold_s)
    if len(chunk_arrival_times_s) != len(chunk_durations_s):
        return ContinuityStats(audio_continuity_threshold_s=threshold_s)

    arrivals = _clean_nonnegative(chunk_arrival_times_s)
    durations = normalize_audio_chunk_durations(
        chunk_durations_s, mode=chunk_duration_mode)
    if len(arrivals) != len(durations):
        return ContinuityStats(audio_continuity_threshold_s=threshold_s)

    t0 = arrivals[0]
    received_before_s = 0.0
    max_underrun_s = 0.0
    event_count = 0

    for idx, (arrival_s, duration_s) in enumerate(zip(arrivals, durations)):
        if idx > 0:
            played_s = max(0.0, arrival_s - t0)
            deficit_s = played_s - received_before_s
            if deficit_s > 0.0:
                max_underrun_s = max(max_underrun_s, deficit_s)
                event_count += 1
        received_before_s += duration_s

    return ContinuityStats(
        audio_underrun_s=max_underrun_s,
        audio_underrun_event_count=event_count,
        audio_continuity_ok=max_underrun_s <= threshold_s,
        audio_continuity_threshold_s=threshold_s,
    )


def compute_continuity_stats_from_pcm_bytes(
    chunk_arrival_times_s: list[float],
    chunk_bytes: list[int],
    sample_rate: int,
    sample_width: int = 2,
    channels: int = 1,
    threshold_s: float = DEFAULT_AUDIO_CONTINUITY_THRESHOLD_S,
) -> ContinuityStats:
    """Compute audio underrun for raw PCM chunks."""

    try:
        bytes_per_second = int(sample_rate) * int(sample_width) * int(channels)
    except (TypeError, ValueError):
        bytes_per_second = 0
    if bytes_per_second <= 0:
        return ContinuityStats(audio_continuity_threshold_s=threshold_s)
    durations_s = [max(0, int(n)) / bytes_per_second for n in chunk_bytes]
    return compute_continuity_stats_from_durations(
        chunk_arrival_times_s,
        durations_s,
        threshold_s=threshold_s,
        chunk_duration_mode="fragment",
    )
