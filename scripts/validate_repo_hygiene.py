#!/usr/bin/env python3
"""No-GPU repository hygiene checks for LiveServe-owned files.

This guard is intentionally conservative: it catches tracked generated
artifacts, personal machine paths, stale warning text, and missing ignore rules.
It allows documented workspace dataset/artifact paths such as
``/workspace/xyz/test`` and ``/workspace/xyz/exp_tmp_dir`` because the current
benchmark documentation uses those as local reproduction roots. Public
Markdown still uses portable placeholders for virtualenv and model locations.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]


GENERATED_FILE_RE = re.compile(
    r"(^|/)(results|result|paper_figures|figures|logs?|tmp|__pycache__|\.pytest_cache)(/|$)"
    r"|"
    r"\.(jsonl|log|png|pdf|wav|mp4|pt|npy|npz|csv)$"
)

ALLOWED_GENERATED_RE = re.compile(
    r"(^|/)scripts/validate_.*\.py$"
)

FORBIDDEN_TEXT_PATTERNS = [
    (re.compile("/root/" + "siamzhi"), "old personal root path"),
    (re.compile("/root/" + "live-proj"), "old personal root project path"),
    (re.compile("/" + "home" + r"/[^/\s]+"), "personal home path"),
    (re.compile("/" + "root" + r"/\.cache/huggingface"), "personal root model cache path"),
    (re.compile(r"/workspace/xyz/" + "LiveServe" + r"(?=[/\s`'\"),]|$)"), "local checkout path"),
    (re.compile(r"/workspace/xyz/test/" + "LiveServe" + r"(?=[/\s`'\"),]|$)"), "local smoke checkout path"),
    (re.compile(r"/workspace/xyz/" + "current\\.md"), "local handoff current.md reference"),
    (re.compile("LiveServe_" + r"(agent|main|full)"), "local worktree name reference"),
    (re.compile("latbd_dumper_autoload" + " warning"), "stale latbd warning text"),
    (re.compile("M18_SESSION_BALANCED" + "_RESULTS"), "deleted M18 result artifact reference"),
    (re.compile(r"benchmarks/(interactive|sharegpt)/" + "H20/"), "stale H20 benchmark path"),
    (re.compile(r"deploy/" + "H20/"), "stale H20 deploy path"),
    (re.compile(r"H20 " + "agent"), "stale internal H20-worker note"),
    (re.compile(r"paper " + "authors"), "internal paper-author note"),
    (re.compile(r"2x H20" + r"-96G"), "stale H20 capacity label"),
    (re.compile(r"client_" + r"interrupt\.py"), "stale interrupt helper path"),
    (
        re.compile(r"benchmarks/sharegpt/H200/" + "interrupt/"),
        "stale ShareGPT interrupt path",
    ),
    (
        re.compile(r"profiling/kv_offload_prefetch/sharegpt_" + "interrupt/"),
        "stale ShareGPT interrupt profiling path",
    ),
    (
        re.compile(r"step1_" + "interrupt" + r"_chain"),
        "stale Step-1 interrupt entrypoint",
    ),
    (
        re.compile(r"\.get_cmap\("),
        "deprecated Matplotlib get_cmap API",
    ),
    (
        re.compile(r"Ming TP2/" + "3GPU smoke"),
        "stale Ming 3GPU default-smoke heading",
    ),
    (
        re.compile(r"Configured " + "interrupt probability"),
        "stale public interrupt probability label",
    ),
    (
        re.compile(r"ShareGPT " + "interrupt frontier"),
        "stale public ShareGPT interrupt plot title",
    ),
]

FORBIDDEN_MARKDOWN_PATTERNS = [
    (re.compile(r"/workspace/xyz/" + "omni_020"), "local virtualenv path"),
    (re.compile(r"/workspace/models"), "local model cache path"),
]

REQUIRED_GITIGNORE_LINES = [
    "results/",
    "**/results/",
    "benchmarks/**/analysis/figures/",
    "profiling/**/figures/*.png",
    "profiling/**/figures/*.pdf",
    "profiling/kv_offload_prefetch/figures/*.json",
    "*.jsonl",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
]

REQUIRED_LICENSE_MARKERS = {
    "LICENSE": "MIT License",
    "README.md": "LiveServe code is MIT licensed",
    "vllm-omni/LICENSE": "Apache License",
}

PUBLIC_EXECUTABLE_SHELL_RE = re.compile(
    r"^(scripts/(prepare_sharegpt|setup_env|validate_.*)\.sh|"
    r"scripts/H200/(start_|reproduce_).+\.sh|"
    r"benchmarks/.+/(run_|scheduler_|step[0-9]_).+\.sh|"
    r"profiling/.+/(run_|scripts/(run_|install_)).+\.sh)$"
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


def git_lines(root: Path, *args: str) -> list[str]:
    out = subprocess.check_output(["git", *args], cwd=root, text=True)
    return [line for line in out.splitlines() if line]


def project_files(root: Path) -> list[str]:
    """Return tracked plus untracked, non-ignored files visible to this worktree."""
    paths = git_lines(root, "ls-files", "--cached", "--others", "--exclude-standard")
    return sorted(set(paths))


def is_text_file(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if b"\0" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def validate(root: Path) -> Audit:
    audit = Audit()
    files = project_files(root)

    generated = [
        path
        for path in files
        if GENERATED_FILE_RE.search(path) and not ALLOWED_GENERATED_RE.search(path)
    ]
    audit.check(
        not generated,
        "no tracked/untracked raw/generated artifacts"
        + ("" if not generated else ": " + ", ".join(generated[:20])),
    )

    text_hits: list[str] = []
    for rel in files:
        path = root / rel
        if not path.is_file() or not is_text_file(path):
            continue
        text = path.read_text(encoding="utf-8")
        for pattern, label in FORBIDDEN_TEXT_PATTERNS:
            if pattern.search(text):
                text_hits.append(f"{rel}: {label}")
    audit.check(
        not text_hits,
        "no forbidden personal/stale paths in tracked/untracked text"
        + ("" if not text_hits else ": " + "; ".join(text_hits[:20])),
    )

    markdown_hits: list[str] = []
    for rel in files:
        if not rel.endswith(".md"):
            continue
        path = root / rel
        if not path.is_file() or not is_text_file(path):
            continue
        text = path.read_text(encoding="utf-8")
        for pattern, label in FORBIDDEN_MARKDOWN_PATTERNS:
            if pattern.search(text):
                markdown_hits.append(f"{rel}: {label}")
    audit.check(
        not markdown_hits,
        "public Markdown avoids machine-local venv/model paths"
        + ("" if not markdown_hits else ": " + "; ".join(markdown_hits[:20])),
    )

    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    missing_ignore = [line for line in REQUIRED_GITIGNORE_LINES if line not in gitignore]
    audit.check(
        not missing_ignore,
        "generated-artifact ignore rules are present"
        + ("" if not missing_ignore else ": missing " + ", ".join(missing_ignore)),
    )

    research_refs = [path for path in files if path.startswith("research/")]
    audit.check(
        not research_refs,
        "no top-level research branch snapshots tracked/untracked"
        + ("" if not research_refs else ": " + ", ".join(research_refs[:20])),
    )

    docs_index = root / "docs" / "README.md"
    root_readme = root / "README.md"
    audit.check(docs_index.is_file(), "docs/README.md exists")
    audit.check(root_readme.is_file(), "README.md exists")
    if docs_index.is_file() and root_readme.is_file():
        docs_text = docs_index.read_text(encoding="utf-8")
        root_text = root_readme.read_text(encoding="utf-8")
        scripts_readme = (root / "scripts" / "README.md").read_text(encoding="utf-8") if (root / "scripts" / "README.md").is_file() else ""
        audit.check(
            "validate_main_no_gpu.sh" in docs_text
            and "validate_main_no_gpu.sh" in root_text,
            "root/docs validation entries use the project-level no-GPU gate",
        )
        audit.check(
            all(
                token in scripts_readme
                for token in (
                    "validate_qwen_system_artifacts.py",
                    "validate_qwen_runner_wiring.py",
                    "validate_sharegpt_paper_workloads.py",
                    "validate_repo_hygiene.py",
                    "validate_markdown_links.py",
                    "validate_final_reproduction_matrix.py",
                    "validate_reduced_smoke_artifacts.py",
                    "validate_ming_runner_wiring.py",
                )
            ),
            "scripts README documents the no-GPU validator entrypoints",
        )
        gitmodules = (root / ".gitmodules").read_text(encoding="utf-8") if (root / ".gitmodules").is_file() else ""
        gitlink = git_lines(root, "ls-files", "--stage", "vllm-omni")
        audit.check(
            "path = vllm-omni" in gitmodules
            and "url = https://github.com/xiangyuzhi/vllm-omni-liveserve.git" in gitmodules
            and not re.search(r"^\s*branch\s*=", gitmodules, re.MULTILINE)
            and len(gitlink) == 1 and gitlink[0].startswith("160000 ")
            and "pinned" in docs_text and "pinned" in root_text,
            "runtime uses a pinned Git submodule without a floating branch default",
        )

    shell_without_shebang: list[str] = []
    non_executable_entrypoints: list[str] = []
    for rel in files:
        if not rel.endswith(".sh"):
            continue
        path = root / rel
        if not path.is_file():
            continue
        try:
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
        except (IndexError, UnicodeDecodeError):
            first_line = ""
        if not first_line.startswith("#!"):
            shell_without_shebang.append(rel)
        if PUBLIC_EXECUTABLE_SHELL_RE.search(rel) and not (path.stat().st_mode & 0o111):
            non_executable_entrypoints.append(rel)
    audit.check(
        not shell_without_shebang,
        "shell scripts have shebangs"
        + (
            ""
            if not shell_without_shebang
            else ": " + ", ".join(shell_without_shebang[:20])
        ),
    )
    audit.check(
        not non_executable_entrypoints,
        "public shell entrypoints are executable"
        + (
            ""
            if not non_executable_entrypoints
            else ": " + ", ".join(non_executable_entrypoints[:20])
        ),
    )

    missing_license_markers: list[str] = []
    for rel, marker in REQUIRED_LICENSE_MARKERS.items():
        path = root / rel
        if not path.is_file():
            missing_license_markers.append(f"{rel}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        if marker not in text:
            missing_license_markers.append(f"{rel}: missing {marker!r}")
    audit.check(
        not missing_license_markers,
        "license and submodule license markers are present"
        + (
            ""
            if not missing_license_markers
            else ": " + "; ".join(missing_license_markers)
        ),
    )

    return audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    args = ap.parse_args()

    audit = validate(args.repo_root)
    for message in audit.passes:
        print(f"PASS {message}")
    for message in audit.failures:
        print(f"FAIL {message}", file=sys.stderr)
    print(f"summary: {len(audit.passes)} passed, {len(audit.failures)} failed")
    return 1 if audit.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
