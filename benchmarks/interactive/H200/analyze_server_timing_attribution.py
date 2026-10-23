#!/usr/bin/env python3
"""Analyze server-side timing fields persisted by trace_replay_driver.

The script is intentionally offline-only. It reads one or more per_request.jsonl
files, summarizes timing field coverage, and attributes client audio TTFP/E2E
to server first-final-output time plus client connect/receive residuals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TIMING_FIELDS_MS = (
    "server_stage0_to_first_final_output_ms",
    "server_final_stage_submit_to_first_output_ms",
    "server_queue_wait_ms",
    "server_preprocess_ms",
    "server_final_stage_input_process_ms",
    "server_ar2code2wav_ms",
    "server_ar2diffusion_ms",
    "server_stage0_gen_ms",
    "server_stage1_gen_ms",
    "server_stage2_gen_ms",
    "server_stage0_submit_to_first_output_ms",
    "server_stage1_submit_to_first_output_ms",
    "server_stage2_submit_to_first_output_ms",
    "server_stage0_submit_to_finished_ms",
    "server_stage1_submit_to_finished_ms",
    "server_stage2_submit_to_finished_ms",
    "server_stage0_to_stage1_process_inputs_ms",
    "server_stage1_to_stage2_process_inputs_ms",
)

SERVER_STAGE_DURATION_KEYS = {
    "server_stage0_to_first_final_output_ms": "stage0_submit_to_first_final_output_ms",
    "server_final_stage_submit_to_first_output_ms": "final_stage_submit_to_first_output_ms",
    "server_queue_wait_ms": "queue_wait_ms",
    "server_preprocess_ms": "preprocess_ms",
    "server_final_stage_input_process_ms": "final_stage_input_process_ms",
    "server_ar2code2wav_ms": "ar2code2wav_ms",
    "server_ar2diffusion_ms": "ar2diffusion_ms",
    "server_stage0_gen_ms": "stage_0_gen_ms",
    "server_stage1_gen_ms": "stage_1_gen_ms",
    "server_stage2_gen_ms": "stage_2_gen_ms",
    "server_stage0_submit_to_first_output_ms": "stage0_submit_to_first_output_ms",
    "server_stage1_submit_to_first_output_ms": "stage1_submit_to_first_output_ms",
    "server_stage2_submit_to_first_output_ms": "stage2_submit_to_first_output_ms",
    "server_stage0_submit_to_finished_ms": "stage0_submit_to_finished_ms",
    "server_stage1_submit_to_finished_ms": "stage1_submit_to_finished_ms",
    "server_stage2_submit_to_finished_ms": "stage2_submit_to_finished_ms",
    "server_stage0_to_stage1_process_inputs_ms": "stage0_to_stage1_process_inputs_ms",
    "server_stage1_to_stage2_process_inputs_ms": "stage1_to_stage2_process_inputs_ms",
}


@dataclass
class RunInput:
    label: str
    path: Path


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * p))))
    return xs[idx]


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    xs = sorted(values)
    return {
        "n": len(xs),
        "min": xs[0],
        "p50": percentile(xs, 0.50),
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
        "max": xs[-1],
        "mean": sum(xs) / len(xs),
    }


def pearson(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [
        (x, y)
        for x, y in zip(xs, ys)
        if isinstance(x, (int, float)) and isinstance(y, (int, float))
    ]
    if len(pairs) < 3:
        return None
    vx = [float(x) for x, _ in pairs]
    vy = [float(y) for _, y in pairs]
    mx = sum(vx) / len(vx)
    my = sum(vy) / len(vy)
    ssx = sum((x - mx) ** 2 for x in vx)
    ssy = sum((y - my) ** 2 for y in vy)
    if ssx <= 0.0 or ssy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(vx, vy)) / math.sqrt(ssx * ssy)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def find_per_request_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("per_request.jsonl"))


def default_label(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            rel = path.relative_to(root)
            return str(rel.parent)
        except ValueError:
            pass
    if path.name == "per_request.jsonl" and len(path.parents) >= 2:
        return f"{path.parents[1].name}/{path.parent.name}"
    return path.stem


def discover_inputs(raw_inputs: list[str]) -> list[RunInput]:
    discovered: list[RunInput] = []
    for raw in raw_inputs:
        label: str | None = None
        path_raw = raw
        if "::" in raw:
            label, path_raw = raw.split("::", 1)
        path = Path(path_raw).expanduser().resolve()
        files = find_per_request_files(path)
        if not files:
            raise FileNotFoundError(f"no per_request.jsonl found under {path}")
        if path.is_file():
            discovered.append(RunInput(label or default_label(path), path))
        else:
            for file_path in files:
                item_label = label
                if len(files) > 1:
                    suffix = default_label(file_path, root=path)
                    item_label = f"{label}/{suffix}" if label else suffix
                discovered.append(RunInput(item_label or default_label(file_path), file_path))
    return discovered


def value_ms(row: dict[str, Any], field: str) -> float | None:
    if field.endswith("_sec"):
        value = as_float(row.get(field))
        return value * 1000.0 if value is not None else None
    value = as_float(row.get(field))
    if value is not None:
        return value
    stage_durations = row.get("server_stage_durations")
    if isinstance(stage_durations, dict):
        source_key = SERVER_STAGE_DURATION_KEYS.get(field)
        if source_key is not None:
            return as_float(stage_durations.get(source_key))
    return None


def summarize_run(run: RunInput, top_k: int = 10) -> dict[str, Any]:
    rows = load_rows(run.path)
    ok_rows = [row for row in rows if row.get("error") is None]
    err_rows = [row for row in rows if row.get("error") is not None]

    derived_rows: list[dict[str, Any]] = []
    for row in ok_rows:
        server_stage0 = value_ms(row, "server_stage0_to_first_final_output_ms")
        server_final = value_ms(row, "server_final_stage_submit_to_first_output_ms")
        audio_ttfp = value_ms(row, "audio_ttfp_sec")
        e2e = value_ms(row, "e2e_sec")
        connect = value_ms(row, "connect_sec") or 0.0
        item = dict(row)
        if server_stage0 is not None and audio_ttfp is not None:
            item["audio_ttfp_minus_server_stage0_ms"] = audio_ttfp - server_stage0
            item["audio_ttfp_minus_server_stage0_minus_connect_ms"] = (
                audio_ttfp - server_stage0 - connect
            )
        if server_final is not None and audio_ttfp is not None:
            item["audio_ttfp_minus_server_final_submit_ms"] = audio_ttfp - server_final
        if server_stage0 is not None and e2e is not None:
            item["e2e_minus_server_stage0_ms"] = e2e - server_stage0
            item["e2e_minus_server_stage0_minus_connect_ms"] = e2e - server_stage0 - connect
        item["audio_ttfp_ms"] = audio_ttfp
        item["e2e_ms"] = e2e
        item["connect_ms"] = connect if row.get("connect_sec") is not None else None
        derived_rows.append(item)

    metric_names = list(TIMING_FIELDS_MS) + [
        "audio_ttfp_ms",
        "e2e_ms",
        "connect_ms",
        "audio_ttfp_minus_server_stage0_ms",
        "audio_ttfp_minus_server_stage0_minus_connect_ms",
        "audio_ttfp_minus_server_final_submit_ms",
        "e2e_minus_server_stage0_ms",
        "e2e_minus_server_stage0_minus_connect_ms",
    ]

    metrics: dict[str, dict[str, float | int | None]] = {}
    coverage: dict[str, dict[str, int | float]] = {}
    for name in metric_names:
        if name in TIMING_FIELDS_MS:
            vals = [value_ms(row, name) for row in derived_rows]
        else:
            vals = [as_float(row.get(name)) for row in derived_rows]
        vals = [v for v in vals if v is not None]
        metrics[name] = describe(vals)
        coverage[name] = {
            "non_null": len(vals),
            "ok_rows": len(ok_rows),
            "ratio": (len(vals) / len(ok_rows)) if ok_rows else 0.0,
        }

    server_stage0 = [as_float(row.get("server_stage0_to_first_final_output_ms")) for row in derived_rows]
    residual = [as_float(row.get("audio_ttfp_minus_server_stage0_ms")) for row in derived_rows]
    residual_adj = [
        as_float(row.get("audio_ttfp_minus_server_stage0_minus_connect_ms"))
        for row in derived_rows
    ]
    correlations = {
        "audio_ttfp_ms_vs_server_stage0_ms": pearson(
            [as_float(row.get("audio_ttfp_ms")) for row in derived_rows],
            server_stage0,
        ),
        "raw_residual_vs_output_tokens": pearson(
            residual,
            [as_float(row.get("output_tokens")) for row in derived_rows],
        ),
        "raw_residual_vs_talker_frames_generated": pearson(
            residual,
            [as_float(row.get("talker_frames_generated")) for row in derived_rows],
        ),
        "raw_residual_vs_audio_duration_sec": pearson(
            residual,
            [as_float(row.get("audio_duration_sec")) for row in derived_rows],
        ),
        "connect_adjusted_residual_vs_audio_duration_sec": pearson(
            residual_adj,
            [as_float(row.get("audio_duration_sec")) for row in derived_rows],
        ),
    }

    tail_rows = sorted(
        derived_rows,
        key=lambda row: as_float(row.get("audio_ttfp_minus_server_stage0_minus_connect_ms"))
        if as_float(row.get("audio_ttfp_minus_server_stage0_minus_connect_ms")) is not None
        else float("-inf"),
        reverse=True,
    )[:top_k]

    def compact_tail(row: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "request_id",
            "user_id",
            "round_index",
            "timestamp",
            "trace_entry_index",
            "audio_ttfp_ms",
            "server_stage0_to_first_final_output_ms",
            "audio_ttfp_minus_server_stage0_ms",
            "connect_ms",
            "audio_ttfp_minus_server_stage0_minus_connect_ms",
            "e2e_ms",
            "e2e_minus_server_stage0_minus_connect_ms",
            "server_queue_wait_ms",
            "server_preprocess_ms",
            "output_tokens",
            "talker_frames_generated",
            "audio_duration_sec",
            "audio_chunks",
        ]
        return {key: row.get(key) for key in keys}

    return {
        "label": run.label,
        "path": str(run.path),
        "rows": len(rows),
        "ok_rows": len(ok_rows),
        "error_rows": len(err_rows),
        "coverage": coverage,
        "metrics": metrics,
        "correlations": correlations,
        "tail_rows_by_connect_adjusted_residual": [compact_tail(row) for row in tail_rows],
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Server Timing Attribution")
    lines.append("")
    lines.append(f"Runs: {len(report['runs'])}")
    lines.append("")

    for run in report["runs"]:
        lines.append(f"## {run['label']}")
        lines.append("")
        lines.append(f"Path: `{run['path']}`")
        lines.append("")
        lines.append(
            f"Rows: total={run['rows']} ok={run['ok_rows']} errors={run['error_rows']}"
        )
        lines.append("")
        lines.append("### Field Coverage")
        lines.append("")
        lines.append("| field | non_null/ok | ratio |")
        lines.append("|---|---:|---:|")
        for field in TIMING_FIELDS_MS:
            cov = run["coverage"].get(field, {})
            lines.append(
                f"| `{field}` | {cov.get('non_null', 0)}/{cov.get('ok_rows', 0)} | "
                f"{fmt(cov.get('ratio'), 3)} |"
            )
        lines.append("")
        lines.append("### Key Distributions")
        lines.append("")
        lines.append("| metric | n | p50 | p90 | p95 | p99 | max | mean |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for metric in [
            "server_stage0_to_first_final_output_ms",
            "server_final_stage_submit_to_first_output_ms",
            "audio_ttfp_ms",
            "e2e_ms",
            "connect_ms",
            "audio_ttfp_minus_server_stage0_ms",
            "audio_ttfp_minus_server_stage0_minus_connect_ms",
            "e2e_minus_server_stage0_minus_connect_ms",
            "server_queue_wait_ms",
            "server_preprocess_ms",
        ]:
            desc = run["metrics"].get(metric, {})
            lines.append(
                f"| `{metric}` | {desc.get('n', 0)} | {fmt(desc.get('p50'))} | "
                f"{fmt(desc.get('p90'))} | {fmt(desc.get('p95'))} | "
                f"{fmt(desc.get('p99'))} | {fmt(desc.get('max'))} | "
                f"{fmt(desc.get('mean'))} |"
            )
        lines.append("")
        lines.append("### Correlations")
        lines.append("")
        lines.append("| pair | pearson |")
        lines.append("|---|---:|")
        for key, value in run["correlations"].items():
            lines.append(f"| `{key}` | {fmt(value, 4)} |")
        lines.append("")
        lines.append("### Largest Connect-Adjusted Residuals")
        lines.append("")
        lines.append(
            "| request_id | raw_res_ms | connect_ms | adjusted_res_ms | "
            "audio_ttfp_ms | server_stage0_ms | e2e_ms | output_tokens | "
            "talker_frames | audio_sec | chunks |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in run["tail_rows_by_connect_adjusted_residual"]:
            lines.append(
                f"| `{row.get('request_id')}` | "
                f"{fmt(row.get('audio_ttfp_minus_server_stage0_ms'))} | "
                f"{fmt(row.get('connect_ms'))} | "
                f"{fmt(row.get('audio_ttfp_minus_server_stage0_minus_connect_ms'))} | "
                f"{fmt(row.get('audio_ttfp_ms'))} | "
                f"{fmt(row.get('server_stage0_to_first_final_output_ms'))} | "
                f"{fmt(row.get('e2e_ms'))} | "
                f"{fmt(row.get('output_tokens'), 0)} | "
                f"{fmt(row.get('talker_frames_generated'), 0)} | "
                f"{fmt(row.get('audio_duration_sec'))} | "
                f"{fmt(row.get('audio_chunks'), 0)} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "label",
        "rows",
        "ok_rows",
        "error_rows",
        "metric",
        "n",
        "p50",
        "p90",
        "p95",
        "p99",
        "max",
        "mean",
        "coverage_non_null",
        "coverage_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in report["runs"]:
            for metric, desc in run["metrics"].items():
                cov = run["coverage"].get(metric, {})
                writer.writerow(
                    {
                        "label": run["label"],
                        "rows": run["rows"],
                        "ok_rows": run["ok_rows"],
                        "error_rows": run["error_rows"],
                        "metric": metric,
                        "n": desc.get("n"),
                        "p50": desc.get("p50"),
                        "p90": desc.get("p90"),
                        "p95": desc.get("p95"),
                        "p99": desc.get("p99"),
                        "max": desc.get("max"),
                        "mean": desc.get("mean"),
                        "coverage_non_null": cov.get("non_null"),
                        "coverage_ratio": cov.get("ratio"),
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "per_request.jsonl or directory containing per_request.jsonl. "
            "Use LABEL::PATH to override the label. Repeatable."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [summarize_run(run, top_k=max(1, args.top_k)) for run in discover_inputs(args.input)]
    report = {"runs": runs}

    json_path = args.output_dir / "server_timing_attribution.json"
    md_path = args.output_dir / "server_timing_attribution.md"
    csv_path = args.output_dir / "server_timing_attribution_summary.csv"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, md_path)
    write_summary_csv(report, csv_path)

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
