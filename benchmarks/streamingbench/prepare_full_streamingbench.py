#!/usr/bin/env python3
"""Prepare LiveServe metadata from the full StreamingBench checkout.

The official StreamingBench repo already ships JSON metadata whose video paths
point at ``src/data/videos``.  This helper merges the streaming splits that are
useful for LiveServe, validates that the expected videos exist after dataset
extraction/preprocessing, and writes one JSON file accepted by
``streamingbench_driver.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_SPLITS = (
    "questions_real_stream.json",
    "questions_omni_stream.json",
    "questions_sqa_stream.json",
    "questions_proactive_50.json",
)
SCRIPT_DIR = Path(__file__).resolve().parent


def _flatten(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, list):
            out.extend(x for x in item if isinstance(x, dict))
    return out


def _load_split(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return _flatten(json.load(f))


def _resolve_video(data_dir: Path, video_path: str) -> Path:
    path = Path(video_path)
    if path.is_absolute():
        return path
    return (data_dir / path).resolve()


def _maybe_link_video(data_dir: Path, group: dict[str, Any], search_roots: list[Path]) -> bool:
    """Try to recover a missing video by matching its expected basename."""
    video_path = group.get("video_path")
    if not isinstance(video_path, str) or not video_path:
        return False
    expected = _resolve_video(data_dir, video_path)
    if expected.exists():
        return True

    stem = expected.stem
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        candidates.extend(root.rglob(f"{stem}.mp4"))
        if "_" in stem:
            sample = stem.rsplit("_", 1)[0]
            candidates.extend(root.rglob(f"{sample}.mp4"))
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        return False

    expected.parent.mkdir(parents=True, exist_ok=True)
    source = sorted(candidates, key=lambda p: len(str(p)))[0]
    try:
        expected.symlink_to(source.resolve())
    except FileExistsError:
        return expected.exists()
    except OSError:
        shutil.copy2(source, expected)
    return expected.exists()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--streamingbench-root",
        default="/workspace/xyz/test/video/StreamingBench",
        help="Official StreamingBench checkout root.",
    )
    parser.add_argument(
        "--out",
        default=str(SCRIPT_DIR / "full_streamingbench_metadata.json"),
        help="Merged JSON metadata path for streamingbench_driver.py.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="JSON split files under StreamingBench/src/data to merge.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write metadata even if some videos are still missing.",
    )
    parser.add_argument(
        "--attempt-link-missing",
        action="store_true",
        help="Search extracted data folders and symlink/copy matching videos into src/data/videos.",
    )
    args = parser.parse_args()

    root = Path(args.streamingbench_root).resolve()
    data_dir = root / "src" / "data"
    extracted_roots = [root / "data", data_dir / "videos"]
    merged: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}

    for split in args.splits:
        split_path = data_dir / split
        groups = _load_split(split_path)
        counts[split] = {
            "videos": len(groups),
            "questions": sum(len(g.get("questions") or []) for g in groups),
        }
        merged.extend(groups)

    missing: list[str] = []
    for group in merged:
        video_path = group.get("video_path")
        if not isinstance(video_path, str) or not video_path:
            missing.append("<empty video_path>")
            continue
        resolved = _resolve_video(data_dir, video_path)
        if resolved.exists():
            continue
        if args.attempt_link_missing and _maybe_link_video(data_dir, group, extracted_roots):
            continue
        missing.append(video_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    summary = {
        "streamingbench_root": str(root),
        "out": str(out_path),
        "splits": counts,
        "total_videos": len(merged),
        "total_questions": sum(len(g.get("questions") or []) for g in merged),
        "missing_videos": len(missing),
        "missing_examples": missing[:20],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if missing and not args.allow_missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
