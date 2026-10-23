#!/usr/bin/env python3
"""Validate local Markdown links in LiveServe-owned documentation.

The vllm-omni submodule carries upstream docs with their own link structure, so
this checker intentionally scopes to the LiveServe project docs and runner
documentation.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCOPES = (
    "README.md",
    "CONTRIBUTING.md",
    "docs",
    "benchmarks",
    "profiling",
    "scripts",
    "deploy",
)

ROOT_CODE_PATH_PREFIXES = (
    "README.md",
    "benchmarks/",
    "deploy/",
    "docs/",
    "profiling/",
    "scripts/",
    "vllm-omni/",
)

INLINE_LINK_RE = re.compile(r"!??\[[^\]\n]*(?:\][^\]\n]*)*\]\(([^\)\n]+)\)")
REFERENCE_LINK_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)")
URI_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
CODE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_@])"
    r"(?:\./)?[A-Za-z0-9_./-]+"
    r"\.(?:sh|py|md|yaml|yml|json|csv)"
    r"(?![A-Za-z0-9_])"
)


def iter_markdown_files(root: Path, scopes: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for scope in scopes:
        path = root / scope
        if path.is_file() and path.suffix == ".md":
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("*.md"))
    return sorted(set(files))


def should_skip_file(root: Path, path: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    return (
        "/results/" in rel
        or rel.startswith("docs/reference/")
        or "/__pycache__/" in rel
    )


def extract_targets(text: str) -> list[str]:
    targets = [match.group(1).strip() for match in INLINE_LINK_RE.finditer(text)]
    targets.extend(match.group(1).strip() for match in REFERENCE_LINK_RE.finditer(text))
    return targets


def extract_code_paths(text: str) -> list[str]:
    return [match.group(0).strip("`'\"") for match in CODE_PATH_RE.finditer(text)]


def should_skip_target(target: str) -> bool:
    if not target or target.startswith("#"):
        return True
    if URI_RE.match(target):
        return True
    no_fragment = target.split("#", 1)[0]
    if not no_fragment:
        return True
    return any(char in no_fragment for char in "*?$") or no_fragment.startswith("<")


def normalize_target(target: str) -> str:
    target = target.split()[0].strip("<>")
    return urllib.parse.unquote(target.split("#", 1)[0])


def should_skip_code_path(target: str) -> bool:
    if not target or URI_RE.match(target):
        return True
    if target.startswith(("/workspace/", "/tmp/", "/path/")):
        return True
    if target.startswith(("test/", "./test/")):
        return True
    if any(char in target for char in "*?$<>{}|"):
        return True
    normalized = normalize_target(target).lstrip("./")
    return (
        "/results/" in normalized
        or normalized.startswith("results/")
        or "/__pycache__/" in normalized
    )


def is_root_code_path(target: str) -> bool:
    normalized = normalize_target(target).lstrip("./")
    return normalized.startswith(ROOT_CODE_PATH_PREFIXES)


def resolve_doc_path(root: Path, doc_path: Path, target: str) -> Path | None:
    normalized = normalize_target(target)
    candidates = [
        (doc_path.parent / normalized).resolve(),
        (root / normalized.lstrip("./")).resolve(),
    ]
    root_resolved = root.resolve()
    for candidate in candidates:
        try:
            candidate.relative_to(root_resolved)
        except ValueError:
            continue
        if candidate.exists():
            return candidate
    return None


def validate(root: Path, scopes: tuple[str, ...]) -> tuple[int, int, list[str]]:
    missing: list[str] = []
    checked_docs = 0
    checked_code_paths = 0
    root_resolved = root.resolve()
    for path in iter_markdown_files(root, scopes):
        if should_skip_file(root, path):
            continue
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        for target in extract_targets(text):
            if should_skip_target(target):
                continue
            normalized = normalize_target(target)
            dest = (path.parent / normalized).resolve()
            try:
                dest.relative_to(root_resolved)
            except ValueError:
                missing.append(f"{rel}: {target} points outside the repository")
                continue
            if not dest.exists():
                missing.append(f"{rel}: {target} does not exist")
        for target in extract_code_paths(text):
            if should_skip_code_path(target):
                continue
            resolved = resolve_doc_path(root, path, target)
            if resolved is None and is_root_code_path(target):
                missing.append(f"{rel}: code path {target} does not exist")
            elif resolved is not None:
                checked_code_paths += 1
        checked_docs += 1
    return checked_docs, checked_code_paths, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument(
        "--scope",
        action="append",
        default=None,
        help="Repo-relative file or directory to scan. Repeatable.",
    )
    args = parser.parse_args()

    scopes = tuple(args.scope) if args.scope else DEFAULT_SCOPES
    checked, checked_code_paths, missing = validate(args.repo_root, scopes)
    if missing:
        for item in missing:
            print(f"FAIL {item}", file=sys.stderr)
        print(
            f"summary: checked_docs={checked} "
            f"checked_code_paths={checked_code_paths} missing={len(missing)}",
            file=sys.stderr,
        )
        return 1
    print(
        f"PASS Markdown local links/code paths: "
        f"checked_docs={checked}, checked_code_paths={checked_code_paths}, missing=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
