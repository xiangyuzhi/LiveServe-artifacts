#!/usr/bin/env python3
"""Validate the latest reduced Qwen and Ming GPU smoke artifacts.

This is an offline audit. It does not start servers or use GPUs; it checks that
the documented reduced smoke roots are present, structurally complete, and
carry the expected successful marker/summary/env evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path("/workspace/xyz/exp_tmp_dir")
DEFAULT_QWEN_ROOT = "qwen_allworkloads_gpu47_currenttree_final_20260626_163306"
DEFAULT_MING_ROOT = "ming_4gpu_dp2_goal_cont_20260626_181650"

QWEN_VARIANTS = ("baseline", "scheduler", "liveserve_g1")
QWEN_WORKLOADS = (
    "sharegpt_orig",
    "sharegpt_len1024",
    "sharegpt_len2048",
    "sharegpt_len4096",
    "interactive_orig",
    "interactive_len1024",
    "interactive_len2048",
    "interactive_len4096",
    "streamingbench_small",
    "mix_v010_small",
)
QWEN_SERVICE_COUNTS = {
    ("baseline", "text"): 8,
    ("baseline", "video"): 2,
    ("scheduler", "text"): 8,
    ("scheduler", "video"): 2,
    ("liveserve_g1", "text"): 8,
    ("liveserve_g1", "video"): 2,
}
QWEN_VIDEO_WORKLOADS = {"streamingbench_small", "mix_v010_small"}

MING_VARIANTS = ("baseline", "ming_async_scheduler", "ming_liveserve_g1", "ming_liveserve_ttfp_best")
MING_FULL_STACK_VARIANTS = ("ming_liveserve_g1", "ming_liveserve_ttfp_best")
MING_WORKLOADS = ("sharegpt", "interactive", "mix")
MING_CONCURRENCY = 2
MING_NUM_REQUESTS = 4
MING_MAX_TOKENS = 64
SERIOUS_LOG_RE = re.compile(
    r"Traceback|OutOfMemory|OOM|RuntimeError|Exception|(^|[^A-Za-z])ERROR([^A-Za-z]|$)"
)


class Audit:
    def __init__(self) -> None:
        self.passes: list[str] = []
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if condition:
            self.passes.append(message)
        else:
            self.failures.append(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def has_tokens(text: str, tokens: tuple[str, ...]) -> bool:
    return all(token in text for token in tokens)


def text_has_no_serious_log_error(text: str) -> bool:
    return not SERIOUS_LOG_RE.search(text)


def qwen_server_log_has_no_unexpected_errors(text: str) -> bool:
    for line in text.splitlines():
        if not SERIOUS_LOG_RE.search(line):
            continue
        # Qwen audio decoder warmup can log a warning traceback when Triton
        # SnakeBeta falls back to eager for bf16; the run can still complete.
        if "snake_activation.py:101" in line and "WARNING" in line:
            continue
        # Current vLLM-omni shutdown reports some stage workers as "died
        # unexpectedly" even when the exit code is clean and all markers passed.
        if "subprocess died unexpectedly" in line and "exit code 0" in line:
            continue
        return False
    return True


def ming_server_log_has_no_unexpected_errors(text: str) -> bool:
    shutdown_started = False
    for line in text.splitlines():
        if "Shutdown initiated" in line or "Shutdown complete" in line:
            shutdown_started = True
        if not SERIOUS_LOG_RE.search(line):
            continue
        # Ming TP shutdown can close the c10d TCPStore before the NCCL
        # heartbeat monitor exits. This appears after a clean request run and
        # shutdown and should not invalidate the artifact.
        if shutdown_started and "Exception raised from recvBytes" in line:
            continue
        return False
    return True


def file_has_lines(path: Path) -> bool:
    return path.is_file() and any(line.strip() for line in path.read_text(encoding="utf-8").splitlines())


def qwen_summary_ok(data: dict[str, Any]) -> bool:
    totals = data.get("totals") or {}
    if not isinstance(totals, dict):
        return False
    top_errors = data.get("errors")
    if isinstance(top_errors, list) and top_errors:
        return False
    if int(totals.get("errors") or 0) != 0:
        return False
    ok_fields = (
        totals.get("completed_ok"),
        totals.get("ok"),
        totals.get("ok_like"),
        totals.get("requests") if int(totals.get("errors") or 0) == 0 else None,
    )
    return any(float(value or 0) > 0 for value in ok_fields)


def validate_qwen(root: Path, audit: Audit) -> None:
    audit.check(root.is_dir(), f"Qwen reduced smoke root exists: {root.name}")
    if not root.is_dir():
        return

    rows: list[dict[str, Any]] = []
    marker_files = sorted(root.glob("**/markers.jsonl"))
    for path in marker_files:
        rows.extend(load_jsonl(path))
    audit.check(len(marker_files) == 6, f"Qwen marker files present: {len(marker_files)}")
    audit.check(len(rows) == 30, f"Qwen marker rows present: {len(rows)}")
    audit.check(
        Counter(row.get("rc") for row in rows) == Counter({0: 30}),
        "Qwen all workload markers exited rc=0",
    )
    audit.check(
        {str(row.get("variant")) for row in rows} == set(QWEN_VARIANTS),
        "Qwen marker variants match baseline/scheduler/liveserve_g1",
    )
    audit.check(
        tuple(sorted({str(row.get("workload")) for row in rows})) == tuple(sorted(QWEN_WORKLOADS)),
        "Qwen marker workloads cover all reduced workload lanes",
    )
    audit.check(
        Counter((row.get("variant"), row.get("service")) for row in rows) == Counter(QWEN_SERVICE_COUNTS),
        "Qwen text/video service marker distribution is complete",
    )
    audit.check(
        len({(row.get("variant"), row.get("workload")) for row in rows}) == len(QWEN_VARIANTS) * len(QWEN_WORKLOADS),
        "Qwen marker variant/workload pairs are unique and complete",
    )
    audit.check(
        all(
            (row.get("workload") in QWEN_VIDEO_WORKLOADS and row.get("service") == "video")
            or (row.get("workload") not in QWEN_VIDEO_WORKLOADS and row.get("service") == "text")
            for row in rows
        ),
        "Qwen marker workload service lanes match text/video expectations",
    )

    missing_results: list[str] = []
    outside_results: list[str] = []
    bad_summaries: list[str] = []
    for row in rows:
        result_dir = Path(str(row.get("result_dir") or ""))
        summary = result_dir / "summary.json"
        try:
            result_dir.relative_to(root)
        except ValueError:
            outside_results.append(str(result_dir))
        if not result_dir.is_dir() or not summary.is_file():
            missing_results.append(str(result_dir))
            continue
        if not qwen_summary_ok(load_json(summary)):
            bad_summaries.append(str(summary))
    audit.check(
        not missing_results,
        "Qwen marker result directories and summaries exist"
        + ("" if not missing_results else ": " + ", ".join(missing_results[:5])),
    )
    audit.check(
        not outside_results,
        "Qwen marker result directories stay under the validated root"
        + ("" if not outside_results else ": " + ", ".join(outside_results[:5])),
    )
    audit.check(
        not bad_summaries,
        "Qwen summaries report successful zero-error requests"
        + ("" if not bad_summaries else ": " + ", ".join(bad_summaries[:5])),
    )

    analysis = root / "_analysis" / "kv_pressure_by_workload.json"
    audit.check(analysis.is_file(), "Qwen KV pressure analysis JSON exists")
    if analysis.is_file():
        windows = load_json(analysis).get("windows") or []
        audit.check(len(windows) == 30, f"Qwen KV pressure windows present: {len(windows)}")
        audit.check(
            len({(item.get("variant"), item.get("workload")) for item in windows}) == len(QWEN_VARIANTS) * len(QWEN_WORKLOADS),
            "Qwen KV pressure windows cover all variant/workload pairs",
        )

    qwen_envs = {
        "baseline": ("CUDA_VISIBLE_DEVICES=4,5,6,7",),
        "scheduler": ("CUDA_VISIBLE_DEVICES=4,5,6,7", "OMNI_TALKER_LEAD_MAX=50", "OMNI_TALKER_LEAD_LO=25"),
        "liveserve_g1": (
            "CUDA_VISIBLE_DEVICES=4,5,6,7",
            "OMNI_PREFETCH_ADMISSION=1",
            "OMNI_SCHED_NATIVE_PREFETCH=1",
            "OMNI_KV_EVICT_ENFORCE_INDEX=1",
            "OMNI_KV_EVICT_ENFORCE_POLICY=expected_gap_cost_aware",
            "OMNI_KV_EVICT_INDEX_GROUP_BY_PREFIX=1",
            "OMNI_TALKER_LEAD_MAX=12",
            "OMNI_TALKER_LEAD_LO=6",
            "OMNI_TALKER_DEFER_HOLD_MAX=75",
        ),
    }
    for variant, tokens in qwen_envs.items():
        for service in ("text", "video"):
            server_root = root / variant / "_servers"
            env_path = server_root / f"{service}_runtime_env.txt"
            server_log_path = server_root / f"{service}_server.log"
            workload_log_path = server_root / ("text_workloads.log" if service == "text" else "video_mix_workloads.log")
            audit.check(env_path.is_file(), f"Qwen {variant} {service} runtime env snapshot exists")
            audit.check(server_log_path.is_file(), f"Qwen {variant} {service} server log exists")
            audit.check(workload_log_path.is_file(), f"Qwen {variant} {service} workload log exists")
            if env_path.is_file():
                audit.check(
                    has_tokens(env_path.read_text(encoding="utf-8"), tokens),
                    f"Qwen {variant} {service} runtime env confirms expected knobs",
                )
            if server_log_path.is_file():
                server_log = server_log_path.read_text(encoding="utf-8", errors="replace")
                audit.check(
                    qwen_server_log_has_no_unexpected_errors(server_log),
                    f"Qwen {variant} {service} server log has no unexpected runtime errors",
                )
                if variant == "scheduler":
                    audit.check(
                        has_tokens(server_log, ("TalkerLeadGovernor enabled", "L_hi=50 L_lo=25 hold_max=100")),
                        f"Qwen scheduler {service} server log confirms 50:25:100 talker governor",
                    )
                elif variant == "liveserve_g1":
                    audit.check(
                        has_tokens(server_log, ("Created connector: SharedMemoryConnector", "TalkerLeadGovernor enabled", "L_hi=12 L_lo=6 hold_max=75")),
                        f"Qwen liveserve_g1 {service} server log confirms connector and 12:6:75 talker governor",
                    )
            if workload_log_path.is_file():
                workload_log = workload_log_path.read_text(encoding="utf-8", errors="replace")
                audit.check(
                    "0 errors" in workload_log or '"ok"' in workload_log,
                    f"Qwen {variant} {service} workload log records successful requests",
                )

    for service in ("text", "video"):
        probe_root = root / "liveserve_g1" / "_probe" / service
        audit.check(file_has_lines(probe_root / "kv_lru_stats.jsonl"), f"Qwen liveserve_g1 {service} KV/LRU stats probe is non-empty")
        session_probe_files = sorted(probe_root.glob("session_kv.*.jsonl"))
        audit.check(
            len(session_probe_files) >= 3,
            f"Qwen liveserve_g1 {service} session KV probe files are present: {len(session_probe_files)}",
        )
        audit.check(
            all(file_has_lines(path) for path in session_probe_files),
            f"Qwen liveserve_g1 {service} session KV probe files are non-empty",
        )


def ming_summary_path(root: Path, variant: str, workload: str) -> Path:
    if workload == "sharegpt":
        return root / variant / f"c{MING_CONCURRENCY}_pint0p0" / "summary.json"
    return root / variant / workload / f"c{MING_CONCURRENCY}_pint0p0" / "summary.json"


def validate_ming(root: Path, audit: Audit) -> None:
    audit.check(root.is_dir(), f"Ming reduced smoke root exists: {root.name}")
    if not root.is_dir():
        return

    missing: list[str] = []
    bad: list[str] = []
    summary_files = sorted(root.glob("**/summary.json"))
    expected_summary_count = len(MING_VARIANTS) * len(MING_WORKLOADS)
    audit.check(
        len(summary_files) == expected_summary_count,
        f"Ming reduced smoke summary count is exact: {len(summary_files)}",
    )
    for variant in MING_VARIANTS:
        for workload in MING_WORKLOADS:
            path = ming_summary_path(root, variant, workload)
            if not path.is_file():
                missing.append(str(path))
                continue
            data = load_json(path)
            counts = data.get("counts") or {}
            totals = data.get("totals") or {}
            if (
                data.get("workload") != workload
                or int(data.get("num_requests") or 0) != MING_NUM_REQUESTS
                or int(data.get("concurrency") or data.get("max_inflight") or 0) != MING_CONCURRENCY
                or int(data.get("max_tokens") or 0) != MING_MAX_TOKENS
                or int(counts.get("total") or 0) != MING_NUM_REQUESTS
                or int(counts.get("completed_ok") or totals.get("completed_ok") or 0) != MING_NUM_REQUESTS
                or int(counts.get("failed") or totals.get("failed") or 0) != 0
                or bool(data.get("errors"))
                or data.get("url") != "http://127.0.0.1:8098/v1/chat/completions"
                or not (data.get("ming_adapter") or {}).get("stop_after_first_audio")
            ):
                bad.append(str(path))
    audit.check(not missing, "Ming summaries exist for all variants/workloads" + ("" if not missing else ": " + ", ".join(missing[:5])))
    audit.check(not bad, "Ming summaries report c=2 n=4 completed_ok with failed=0 on port 8098" + ("" if not bad else ": " + ", ".join(bad[:5])))

    envs = {
        "baseline": ("CUDA_VISIBLE_DEVICES=4,5,6,7",),
        "ming_async_scheduler": (
            "CUDA_VISIBLE_DEVICES=4,5,6,7",
            "OMNI_MING_INCREMENTAL=1",
            "OMNI_MING_INCREMENTAL_CHUNK_STEPS=4",
            "OMNI_TALKER_LEAD_MAX=12",
            "OMNI_TALKER_LEAD_LO=6",
            "OMNI_TALKER_DEFER_HOLD_MAX=75",
        ),
        "ming_liveserve_g1": (
            "CUDA_VISIBLE_DEVICES=4,5,6,7",
            "OMNI_MING_INCREMENTAL=1",
            "OMNI_MING_INCREMENTAL_CHUNK_STEPS=4",
            "OMNI_PREFETCH_ADMISSION=1",
            "OMNI_SCHED_NATIVE_PREFETCH=1",
            "OMNI_KV_EVICT_ENFORCE_INDEX=1",
            "OMNI_KV_EVICT_ENFORCE_POLICY=expected_gap_cost_aware",
            "OMNI_KV_EVICT_INDEX_GROUP_BY_PREFIX=1",
            "OMNI_TALKER_LEAD_MAX=12",
            "OMNI_TALKER_LEAD_LO=6",
            "OMNI_TALKER_DEFER_HOLD_MAX=75",
            "SCHEDULER_POLICY_OVERLAYS=talkerlead=12:6:75",
        ),
        "ming_liveserve_ttfp_best": (
            "CUDA_VISIBLE_DEVICES=4,5,6,7",
            "OMNI_MING_INCREMENTAL=1",
            "OMNI_MING_INCREMENTAL_CHUNK_STEPS=1",
            "OMNI_PREFETCH_ADMISSION=1",
            "OMNI_SCHED_NATIVE_PREFETCH=1",
            "OMNI_KV_EVICT_ENFORCE_INDEX=1",
            "OMNI_KV_EVICT_ENFORCE_POLICY=expected_gap_cost_aware",
            "OMNI_KV_EVICT_INDEX_GROUP_BY_PREFIX=1",
            "OMNI_TALKER_LEAD_MAX=0",
            "OMNI_TALKER_LEAD_LO=0",
            "OMNI_TALKER_DEFER_HOLD_MAX=1",
            "SCHEDULER_POLICY_OVERLAYS=talkerlead=0:0:1",
        ),
    }
    for variant, tokens in envs.items():
        env_path = root / variant / "_server" / "env_snapshot.txt"
        config_path = root / variant / "_server" / "config.json"
        log_path = root / variant / "_server" / "server.log"
        audit.check(env_path.is_file(), f"Ming {variant} env snapshot exists")
        audit.check(config_path.is_file(), f"Ming {variant} config snapshot exists")
        audit.check(log_path.is_file(), f"Ming {variant} server log exists")
        if env_path.is_file():
            audit.check(
                has_tokens(env_path.read_text(encoding="utf-8"), tokens),
                f"Ming {variant} env snapshot confirms expected runtime knobs",
            )
        if config_path.is_file():
            config = load_json(config_path)
            expected_deploy_suffix = (
                "ming_flash_omni_4gpu_tp2_talker_dp2.yaml"
                if variant == "baseline"
                else "ming_flash_omni_4gpu_tp2_talker_dp2_async.yaml"
            )
            audit.check(
                config.get("variant") == variant
                and str(config.get("text_gpus")) == "4,5,6,7"
                and int(config.get("port", 0)) == 8098
                and str(config.get("deploy_yaml", "")).endswith(expected_deploy_suffix)
                and str(config.get("workloads")) == "sharegpt interactive mix"
                and str(config.get("c_levels")) == str(MING_CONCURRENCY)
                and int(config.get("num_requests", 0)) == MING_NUM_REQUESTS
                and int(config.get("max_tokens", 0)) == MING_MAX_TOKENS
                and int(config.get("stop_after_first_audio", 0)) == 1,
                f"Ming {variant} config records expected 4GPU/8098 c=2 n=4 validation lane",
            )
        if log_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            audit.check(
                ming_server_log_has_no_unexpected_errors(log_text),
                f"Ming {variant} server log has no serious runtime errors",
            )
            if variant == "baseline":
                audit.check(
                    "Loaded OmniTransferConfig with 0 connector configurations" in log_text,
                    "Ming baseline server log confirms no async connector",
                )
            elif variant == "ming_async_scheduler":
                audit.check(
                    has_tokens(
                        log_text,
                        (
                            "Loaded OmniTransferConfig with 1 connector configurations",
                            "Created connector: SharedMemoryConnector",
                            "Generation TalkerLeadGovernor enabled",
                            "L_hi=12 L_lo=6 hold_max=75",
                        ),
                    ),
                    "Ming async scheduler log confirms connector and 12:6:75 talker governor",
                )
            elif variant == "ming_liveserve_g1":
                audit.check(
                    has_tokens(
                        log_text,
                        (
                            "Loaded OmniTransferConfig with 1 connector configurations",
                            "Created connector: SharedMemoryConnector",
                            "[session_kv_probe] apply_patch DONE",
                            "Generation TalkerLeadGovernor enabled",
                            "L_hi=12 L_lo=6 hold_max=75",
                        ),
                    ),
                    "Ming liveserve_g1 log confirms connector, probe hooks, and 12:6:75 talker governor",
                )
            elif variant == "ming_liveserve_ttfp_best":
                audit.check(
                    has_tokens(
                        log_text,
                        (
                            "Loaded OmniTransferConfig with 1 connector configurations",
                            "Created connector: SharedMemoryConnector",
                            "[session_kv_probe] apply_patch DONE",
                        ),
                    ),
                    "Ming full-stack log confirms connector and session KV probe hooks",
                )

    for variant in MING_FULL_STACK_VARIANTS:
        probe_root = root / variant / "_probe"
        audit.check(file_has_lines(probe_root / "kv_lru_stats.jsonl"), f"Ming {variant} KV/LRU stats probe is non-empty")
        session_probe_files = sorted(probe_root.glob("session_kv.*.jsonl"))
        audit.check(
            len(session_probe_files) >= 3,
            f"Ming {variant} session KV probe files are present: {len(session_probe_files)}",
        )
        audit.check(
            all(file_has_lines(path) for path in session_probe_files),
            f"Ming {variant} session KV probe files are non-empty",
        )


def validate(artifact_root: Path, qwen_root_name: str, ming_root_name: str) -> Audit:
    audit = Audit()
    validate_qwen(artifact_root / qwen_root_name, audit)
    validate_ming(artifact_root / ming_root_name, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--qwen-root-name", default=DEFAULT_QWEN_ROOT)
    parser.add_argument("--ming-root-name", default=DEFAULT_MING_ROOT)
    args = parser.parse_args()

    audit = validate(args.artifact_root, args.qwen_root_name, args.ming_root_name)
    for message in audit.passes:
        print(f"PASS {message}")
    for message in audit.failures:
        print(f"FAIL {message}", file=sys.stderr)
    print(f"summary: {len(audit.passes)} passed, {len(audit.failures)} failed")
    return 1 if audit.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
