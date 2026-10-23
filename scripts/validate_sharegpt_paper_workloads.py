#!/usr/bin/env python3
"""No-GPU validation for ShareGPT paper workload helpers."""

from __future__ import annotations

import csv
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "benchmarks/sharegpt/H200/generate_burstgpt_arrivals.py"
SUMMARIZER = REPO_ROOT / "benchmarks/sharegpt/H200/summarize_paper_workloads.py"
DRIVER = REPO_ROOT / "benchmarks/sharegpt/H200/sharegpt_driver.py"


class Audit:
    def __init__(self) -> None:
        self.passes: list[str] = []
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if condition:
            self.passes.append(message)
        else:
            self.failures.append(message)


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def load_sharegpt_driver():
    spec = importlib.util.spec_from_file_location("_sharegpt_driver_validation", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {DRIVER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_burstgpt_csv(path: Path) -> None:
    rows = [
        [100, "ChatGPT", 10, 20, 30, "API log"],
        [100, "ChatGPT", 11, 21, 32, "API log"],
        [101, "GPT-4", 12, 22, 34, "API log"],
        [102, "ChatGPT", 13, 23, 36, "API log"],
        [900, "ChatGPT", 100, 200, 300, "Conversation log"],
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp",
            "Model",
            "Request tokens",
            "Response tokens",
            "Total tokens",
            "Log Type",
        ])
        writer.writerows(rows)


def validate_generator(tmp: Path, audit: Audit) -> None:
    csv_path = tmp / "BurstGPT_tiny.csv"
    manifest = tmp / "burst32.jsonl"
    peak_manifest = tmp / "burst_peak.jsonl"
    write_burstgpt_csv(csv_path)

    proc = run_cmd([
        sys.executable,
        str(GENERATOR),
        "--burstgpt-csv",
        str(csv_path),
        "--output",
        str(manifest),
        "--num-requests",
        "4",
        "--log-type",
        "api",
        "--window-mode",
        "peak",
        "--window-sec",
        "10",
        "--slice-mode",
        "first",
        "--target-rps",
        "2.0",
        "--seed",
        "7",
    ])
    audit.check(proc.returncode == 0, "BurstGPT generator CLI exits successfully")
    if proc.returncode != 0:
        audit.failures.append(proc.stderr[-1000:])
        return

    lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    meta = json.loads((tmp / "burst32.jsonl.meta.json").read_text(encoding="utf-8"))
    audit.check(len(lines) == 4, "BurstGPT generator writes requested row count")
    audit.check(lines[0]["t_sec"] == 0.0, "BurstGPT manifest is rebased to t=0")
    audit.check(all(lines[i]["t_sec"] <= lines[i + 1]["t_sec"] for i in range(3)),
                "BurstGPT manifest offsets are monotonic")
    audit.check(abs(float(meta["scaled_avg_rps"]) - 2.0) < 1e-6,
                "BurstGPT target average RPS scaling is recorded")
    audit.check(meta["window_start"] == 100, "BurstGPT peak window selection is deterministic")

    peak_proc = run_cmd([
        sys.executable,
        str(GENERATOR),
        "--burstgpt-csv",
        str(csv_path),
        "--output",
        str(peak_manifest),
        "--num-requests",
        "4",
        "--log-type",
        "api",
        "--window-mode",
        "peak",
        "--window-sec",
        "10",
        "--slice-mode",
        "first",
        "--target-peak-rps",
        "1.0",
        "--peak-bin-sec",
        "1.0",
        "--seed",
        "7",
    ])
    audit.check(peak_proc.returncode == 0, "BurstGPT generator target-peak CLI exits successfully")
    if peak_proc.returncode == 0:
        peak_rows = [
            json.loads(line)
            for line in peak_manifest.read_text(encoding="utf-8").splitlines()
        ]
        peak_meta = json.loads((tmp / "burst_peak.jsonl.meta.json").read_text(encoding="utf-8"))
        peak_bins: dict[int, int] = {}
        for row in peak_rows:
            bucket = int(float(row["t_sec"]) // 1.0)
            peak_bins[bucket] = peak_bins.get(bucket, 0) + 1
        audit.check(peak_meta["scale_source"] == "target_peak_rps",
                    "BurstGPT target-peak scaling path is recorded")
        audit.check(abs(float(peak_meta["scaled_peak_rps"]) - 1.0) < 1e-6,
                    "BurstGPT target-peak scaled peak is recorded")
        audit.check(max(peak_bins.values()) == 1,
                    "BurstGPT target-peak scaling preserves requested peak bin rate")
    else:
        audit.failures.append(peak_proc.stderr[-1000:])


def summary_payload(
    *,
    ok: int,
    interrupted: int,
    posthoc_interrupted: int = 0,
    rps: float,
    useful: float,
    ttfp_ms: float,
    e2e_ms: float,
    continuity: float | None,
    waste_frames_p90: float | None,
    posthoc_waste_frames_p90: float | None = None,
) -> dict:
    return {
        "totals": {
            "completed_ok": ok,
            "errors": 0,
            "interrupted": interrupted,
            "posthoc_playback_interrupted": posthoc_interrupted,
            "throughput_rps": rps,
            "useful_qps": useful,
        },
        "latency": {
            "audio_ttfp_ms": {"p90": ttfp_ms},
            "e2e_ms": {"p90": e2e_ms},
        },
        "audio": {
            "audio_stream_rtf": {"p90": 0.5},
            "streaming_continuity": {"ok_rate": continuity},
        },
        "interrupt": {
            "waste_ratio_tokens": 0.0,
            "wasted_talker_frames_total": {"p90": waste_frames_p90},
            "wasted_talker_frames_inaudible": {"p90": waste_frames_p90},
            "posthoc_wasted_talker_frames_total": {"p90": posthoc_waste_frames_p90},
            "posthoc_audio_unplayed_sec": {"p90": 3.0},
            "posthoc_waste_ratio_audio": {"p90": 0.5},
        },
    }


def validate_summarizer(tmp: Path, audit: Audit) -> None:
    root = tmp / "results"
    base_dir = root / "baseline" / "case_a"
    live_dir = root / "scheduler_l8" / "case_a"
    invalid_dir = root / "baseline" / "case_failed"
    base_dir.mkdir(parents=True)
    live_dir.mkdir(parents=True)
    invalid_dir.mkdir(parents=True)
    (base_dir / "summary.json").write_text(
        json.dumps(summary_payload(
            ok=10,
            interrupted=5,
            posthoc_interrupted=2,
            rps=0.5,
            useful=0.25,
            ttfp_ms=2000.0,
            e2e_ms=5000.0,
            continuity=0.7,
            waste_frames_p90=80.0,
            posthoc_waste_frames_p90=40.0,
        )),
        encoding="utf-8",
    )
    (live_dir / "summary.json").write_text(
        json.dumps(summary_payload(
            ok=10,
            interrupted=5,
            posthoc_interrupted=3,
            rps=1.0,
            useful=0.5,
            ttfp_ms=1000.0,
            e2e_ms=4000.0,
            continuity=0.9,
            waste_frames_p90=30.0,
            posthoc_waste_frames_p90=20.0,
        )),
        encoding="utf-8",
    )
    (invalid_dir / "summary.json").write_text(
        json.dumps(summary_payload(
            ok=0,
            interrupted=0,
            posthoc_interrupted=0,
            rps=0.0,
            useful=0.0,
            ttfp_ms=0.0,
            e2e_ms=0.0,
            continuity=None,
            waste_frames_p90=None,
            posthoc_waste_frames_p90=None,
        ) | {"totals": {
            "completed_ok": 0,
            "errors": 32,
            "interrupted": 0,
            "posthoc_playback_interrupted": 0,
            "throughput_rps": 0.0,
            "useful_qps": 0.0,
        }}),
        encoding="utf-8",
    )

    proc = run_cmd([
        sys.executable,
        str(SUMMARIZER),
        str(root),
        "--baseline-variant",
        "baseline",
        "--format",
        "csv",
    ])
    audit.check(proc.returncode == 0, "ShareGPT paper summarizer CLI exits successfully")
    if proc.returncode != 0:
        audit.failures.append(proc.stderr[-1000:])
        return

    rows = list(csv.DictReader(proc.stdout.splitlines()))
    live_rows = [row for row in rows if row["variant"] == "scheduler_l8"]
    invalid_rows = [row for row in rows if row["case"] == "case_failed"]
    audit.check(len(rows) == 3 and len(live_rows) == 1,
                "ShareGPT paper summarizer loads baseline and live summaries")
    audit.check(len(invalid_rows) == 1 and invalid_rows[0].get("valid") == "False",
                "ShareGPT paper summarizer marks failed cases invalid")
    if not live_rows:
        return
    row = live_rows[0]
    audit.check(row.get("valid") == "True",
                "ShareGPT paper summarizer reports case validity")
    audit.check(abs(float(row["audio_ttfp_p90_delta_pct"]) - 50.0) < 1e-6,
                "ShareGPT paper summarizer reports latency improvement")
    audit.check(abs(float(row["throughput_rps_delta_pct"]) - 100.0) < 1e-6,
                "ShareGPT paper summarizer reports throughput improvement")
    audit.check(abs(float(row["continuity_delta"]) - 0.2) < 1e-6,
                "ShareGPT paper summarizer reports continuity delta")
    audit.check(abs(float(row["interrupt_rate"]) - 0.5) < 1e-6,
                "ShareGPT paper summarizer reports barge-in rate")
    audit.check(abs(float(row["waste_talker_frames_total_p90"]) - 30.0) < 1e-6,
                "ShareGPT paper summarizer reports talker-frame waste")
    audit.check(int(row["posthoc_playback_interrupted"]) == 3,
                "ShareGPT paper summarizer reports posthoc playback cuts")
    audit.check(abs(float(row["posthoc_waste_talker_frames_total_p90"]) - 20.0) < 1e-6,
                "ShareGPT paper summarizer reports posthoc playback waste")


def validate_barge_in_samples(tmp: Path, audit: Audit) -> None:
    driver = load_sharegpt_driver()
    samples_path = tmp / "barge_in_samples.jsonl"
    samples_path.write_text(
        "\n".join([
            json.dumps({"audio_duration_s": 2.0}),
            json.dumps({"metrics": {"duration": 4.0}}),
            "6.0",
        ]) + "\n",
        encoding="utf-8",
    )
    default_samples = driver.load_interrupt_after_samples(
        samples_path,
        "audio_duration_s",
        0.5,
    )
    nested_samples = driver.load_interrupt_after_samples(
        samples_path,
        "metrics.duration",
        1.0,
    )
    audit.check(default_samples == (1.0, 3.0),
                "ShareGPT barge-in sample loader reads default/plain durations")
    audit.check(nested_samples == (2.0, 4.0, 6.0),
                "ShareGPT barge-in sample loader reads nested/fallback/plain durations")
    policy = driver.BargeInPolicy(
        policy="fixed_time",
        interrupt_prob=1.0,
        after_distribution="samples",
        after_samples=(1.25,),
    )
    threshold = policy.resolve_for_request(random.Random(1))
    audit.check(threshold.get("sampled_after_sec") == 1.25,
                "ShareGPT barge-in policy supports sampled cut times")
    posthoc = driver.compute_posthoc_playback_waste(
        {
            "enabled": True,
            "kind": "time",
            "threshold_sec": 2.0,
            "time_anchor": "ttfp",
        },
        audio_duration_s=5.0,
        audio_ttfp_s=0.5,
        audio_chunks=10,
        talker_frames_generated=100,
    )
    audit.check(
        posthoc is not None
        and posthoc["posthoc_playback_interrupt"] is True
        and abs(posthoc["posthoc_audio_unplayed_sec"] - 3.0) < 1e-6
        and posthoc["posthoc_wasted_talker_frames_total"] == 60,
        "ShareGPT barge-in posthoc playback waste is computed",
    )


def validate_open_c_bound_summary(audit: Audit) -> None:
    driver = load_sharegpt_driver()
    args = Namespace(
        mode="audio",
        concurrency=2,
        num_requests=4,
        max_tokens=32,
        max_prompt_chars=500,
        length_profile="sharegpt",
        prompt_source="dataset",
        input_tokens_mean=None,
        input_tokens_std=128,
        output_tokens_mean=None,
        output_tokens_std=256,
        hetero_long_count=4,
        hetero_short_input_tokens=96,
        hetero_long_input_tokens=512,
        hetero_short_output_tokens=64,
        hetero_long_output_tokens=448,
        arrival_process="open_c_bound",
        request_rate=0.0,
        arrival_manifest=None,
        arrival_time_scale=1.0,
        staged_short_delay_sec=1.0,
        warmup_sec=1.0,
        run_duration_sec=2.0,
        cooldown_sec=1.0,
        max_inflight=2,
        force_output_len=False,
        seed=0,
        url="http://127.0.0.1:1/v1/chat/completions",
        model="test-model",
        dataset=Path("dummy.json"),
        tag="unit",
        policy="fixed_time",
        interrupt_prob=0.0,
        interrupt_after_seconds=8.0,
        interrupt_jitter_sec=0.0,
        interrupt_after_distribution="fixed",
        interrupt_after_min=0.3,
        interrupt_after_max=5.0,
        interrupt_after_mean=1.5,
        interrupt_after_samples=None,
        interrupt_after_samples_values=(),
        posthoc_playback_interrupt_waste=False,
        interrupt_time_anchor="t0",
        audio_continuity_threshold_sec=0.1,
        audio_continuity_chunk_mode="fragment",
        continuity_include_interrupted=False,
    )
    rows = []
    for idx, submit_offset, counted in (
        (0, 0.25, False),
        (1, 1.25, True),
        (2, 2.25, True),
        (3, 3.25, False),
    ):
        rows.append({
            "idx": idx,
            "ok": True,
            "error": None,
            "interrupted": False,
            "elapsed_ms": 100.0,
            "ttft_ms": 10.0,
            "audio_ttfp_ms": 20.0,
            "audio_duration_s": 1.0,
            "audio_continuity_evaluated": False,
            "audio_continuity_excluded_reason": "unit",
            "client_submit_offset_s": submit_offset,
            "counted_in_measurement": counted,
            "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            "generated_completion_tokens_total": 4,
            "wasted_completion_tokens": 0,
            "prompt_target_tokens": None,
            "output_target_tokens": None,
            "length_class": "default",
        })
    summary = driver.summarize(rows, 2.0, args, actual_wall_s=4.0)
    audit.check(
        summary["measurement"]["rows_total"] == 4
        and summary["measurement"]["rows_counted"] == 2
        and summary["measurement"]["rows_excluded"] == 2
        and summary["measurement"]["actual_client_wall_time_s"] == 4.0
        and summary["totals"]["completed_ok"] == 2
        and abs(float(summary["totals"]["throughput_rps"]) - 1.0) < 1e-9,
        "ShareGPT open-c-bound summary counts measured window only",
    )


def main() -> int:
    audit = Audit()
    with tempfile.TemporaryDirectory(prefix="liveserve_sharegpt_paper_") as tmp_name:
        tmp = Path(tmp_name)
        validate_generator(tmp, audit)
        validate_summarizer(tmp, audit)
        validate_barge_in_samples(tmp, audit)
        validate_open_c_bound_summary(audit)

    for message in audit.passes:
        print(f"PASS {message}")
    for message in audit.failures:
        print(f"FAIL {message}", file=sys.stderr)
    print(f"summary: {len(audit.passes)} passed, {len(audit.failures)} failed")
    return 1 if audit.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
