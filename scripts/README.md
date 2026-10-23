# Scripts in this source distribution

This repository contains the runtime as ordinary files under `vllm-omni/`.
The original top-level `docs/` and `profiling/` research directories are not
included. Some retained scripts still reference those directories or external
experiment artifacts; their presence does not mean their full workflows can
run from this distribution alone.

## Supported CPU regression entry point

From the repository root, with Python 3.11+ and Bash available:

```bash
python3 scripts/validate_cpu_tests.py
```

No package installation, Git metadata, model weights, CUDA, or GPU is needed.
CI runs the same command on Python 3.11 and 3.12. At publication, it runs
**121 tests**: 10 launcher/environment tests and 111 targeted runtime tests.
The runner fails if any declared test file is missing or a suite fails.

The runtime suites cover interaction tracing, prefetch routing and completion,
cancellation, first-chunk scheduling protection, prefetch pinning, lazy CUDA
profiling imports, KV policy forwarding, and codec trace output. They exercise
CPU-compatible paths and isolated runtime methods; they do not load a model or
establish GPU performance or end-to-end serving correctness.

The launcher tests use temporary fixtures to verify path resolution, argument
validation, and early failure without starting a service. In particular, they
do not establish that all inputs of a retained benchmark are present here.
The original 95 profiling-tool tests were removed with `profiling/`; they are
outside this command's scope and are not counted as passing or skipped tests.

## Other retained scripts

| Files | Scope and requirements |
|---|---|
| `env_defaults.sh` | Shared interpreter and cache-path resolution used by the CPU tests and launchers. |
| `setup_env.sh` | Environment setup utility. Its printed reference to the original full validation gate is historical; use the CPU command above for this distribution. Follow the repository README for runtime installation. |
| `H200/`, `prepare_sharegpt.sh`, `diff_summary.py` | Service, dataset, and result utilities retained with their original behavior. Inspect each script's inputs and dependencies before use; these workflows are not validated by the CPU CI. |
| `validate_main_no_gpu.sh`, `validate_qwen_main_no_gpu.sh`, `validate_ming_no_gpu.sh` | Original full-repository gates. They depend on removed research directories and/or the original submodule layout, so they are not supported validation entry points for this distribution. |
| Other `validate_*.py` / `validate_*.sh` files | Historical wiring, documentation, dataset, and artifact audits. Their required inputs may be absent. They remain reference tools, not part of the supported CPU gate. |

Use an original complete research checkout with its corresponding experiment
inputs to run the full reproduction gates. The passing CPU tests here do not
substitute for those checks or for a GPU experiment.
