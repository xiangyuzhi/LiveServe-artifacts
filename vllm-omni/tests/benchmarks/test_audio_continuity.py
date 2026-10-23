import importlib.util
import math
import sys
from pathlib import Path

import pytest

_HELPER_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm_omni"
    / "benchmarks"
    / "audio_continuity.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_test_audio_continuity", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_audio_continuity = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _audio_continuity
_SPEC.loader.exec_module(_audio_continuity)

DEFAULT_AUDIO_CONTINUITY_THRESHOLD_S = (
    _audio_continuity.DEFAULT_AUDIO_CONTINUITY_THRESHOLD_S
)
compute_continuity_stats_from_durations = (
    _audio_continuity.compute_continuity_stats_from_durations
)
compute_continuity_stats_from_pcm_bytes = (
    _audio_continuity.compute_continuity_stats_from_pcm_bytes
)
get_audio_continuity_threshold_s = (
    _audio_continuity.get_audio_continuity_threshold_s
)
normalize_audio_chunk_durations = (
    _audio_continuity.normalize_audio_chunk_durations
)


def test_empty_timeline_is_continuous():
    stats = compute_continuity_stats_from_durations([], [])

    assert stats.audio_underrun_s == 0.0
    assert stats.audio_underrun_event_count == 0
    assert stats.audio_continuity_ok


def test_single_chunk_has_no_underrun():
    stats = compute_continuity_stats_from_durations([1.5], [0.4])

    assert stats.audio_underrun_s == 0.0
    assert stats.audio_underrun_event_count == 0
    assert stats.audio_continuity_ok


def test_chunks_arriving_faster_than_realtime_have_no_underrun():
    stats = compute_continuity_stats_from_durations(
        [0.0, 0.15, 0.30],
        [0.2, 0.2, 0.2],
    )

    assert stats.audio_underrun_s == 0.0
    assert stats.audio_underrun_event_count == 0
    assert stats.audio_continuity_ok


def test_gap_above_threshold_marks_not_ok():
    stats = compute_continuity_stats_from_durations(
        [0.0, 0.35],
        [0.2, 0.2],
        threshold_s=0.1,
    )

    assert stats.audio_underrun_s == pytest.approx(0.15)
    assert stats.audio_underrun_event_count == 1
    assert not stats.audio_continuity_ok


def test_multiple_gaps_report_max_cumulative_deficit():
    stats = compute_continuity_stats_from_durations(
        [0.0, 0.25, 1.0, 1.3],
        [0.2, 0.2, 0.2, 0.2],
        threshold_s=0.1,
    )

    assert stats.audio_underrun_s == pytest.approx(0.7)
    assert stats.audio_underrun_event_count == 3
    assert not stats.audio_continuity_ok


def test_threshold_boundary_is_ok():
    stats = compute_continuity_stats_from_durations(
        [0.0, 0.3],
        [0.2, 0.2],
        threshold_s=0.1,
    )

    assert stats.audio_underrun_s == pytest.approx(0.1)
    assert stats.audio_continuity_ok


def test_length_mismatch_does_not_crash():
    stats = compute_continuity_stats_from_durations(
        [0.0, 0.3],
        [0.2],
        threshold_s=0.1,
    )

    assert stats.audio_underrun_s == 0.0
    assert stats.audio_underrun_event_count == 0
    assert stats.audio_continuity_ok


def test_duration_and_pcm_byte_inputs_match():
    arrivals = [0.0, 0.35, 0.6]
    durations = [0.2, 0.2, 0.2]
    sample_rate = 24000
    sample_width = 2
    channels = 1
    chunk_bytes = [
        int(duration * sample_rate * sample_width * channels)
        for duration in durations
    ]

    from_duration = compute_continuity_stats_from_durations(
        arrivals, durations, threshold_s=0.1)
    from_pcm = compute_continuity_stats_from_pcm_bytes(
        arrivals,
        chunk_bytes,
        sample_rate=sample_rate,
        sample_width=sample_width,
        channels=channels,
        threshold_s=0.1,
    )

    assert from_pcm.audio_underrun_s == pytest.approx(
        from_duration.audio_underrun_s)
    assert from_pcm.audio_underrun_event_count == (
        from_duration.audio_underrun_event_count
    )
    assert from_pcm.audio_continuity_ok == from_duration.audio_continuity_ok


def test_cumulative_wav_durations_are_converted_to_chunk_durations():
    assert normalize_audio_chunk_durations(
        [0.3, 0.6, 0.9],
        mode="cumulative",
    ) == pytest.approx([0.3, 0.3, 0.3])

    stats = compute_continuity_stats_from_durations(
        [0.0, 0.4, 1.0],
        [0.3, 0.6, 0.9],
        threshold_s=0.2,
        chunk_duration_mode="cumulative",
    )

    assert stats.audio_underrun_s == pytest.approx(0.4)
    assert stats.audio_underrun_event_count == 2
    assert not stats.audio_continuity_ok


def test_auto_mode_detects_strictly_increasing_cumulative_durations():
    stats = compute_continuity_stats_from_durations(
        [0.0, 0.4, 1.0],
        [0.3, 0.6, 0.9],
        threshold_s=0.2,
        chunk_duration_mode="auto",
    )

    assert stats.audio_underrun_s == pytest.approx(0.4)


def test_env_threshold_precedence_and_fallback():
    assert get_audio_continuity_threshold_s(
        env={
            "LIVESERVE_AUDIO_CONTINUITY_THRESHOLD_S": "0.05",
            "VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S": "0.2",
        }
    ) == 0.05
    assert get_audio_continuity_threshold_s(
        env={"VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S": "0.2"}
    ) == 0.2
    assert get_audio_continuity_threshold_s(
        env={"LIVESERVE_AUDIO_CONTINUITY_THRESHOLD_S": "not-a-number"}
    ) == DEFAULT_AUDIO_CONTINUITY_THRESHOLD_S
    assert math.isclose(
        get_audio_continuity_threshold_s(default=0.15, env={}),
        0.15,
    )
