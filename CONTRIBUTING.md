# Contributing to LiveServe artifacts

This repository contains an expanded runtime under `vllm-omni/`. Edit and
review those files as part of this repository; there is no submodule update
or separate runtime push step. Preserve both licenses and existing copyright
notices.

## Validate focused changes

From the repository root, with Python 3.11 or 3.12 and Bash:

```bash
python scripts/validate_cpu_tests.py
```

The selected 121 CPU regressions use only the standard library and do not
start a model service. They cover 10 repository tests and 111 runtime tests;
they are not the full upstream suite or a GPU validation result. Run the
relevant additional integration tests for model, CUDA, or serving changes,
and state clearly when they were not run. Do not claim that CI ran merely
because a local check passed.

Keep changes focused. Preserve model input/output contracts, feature defaults,
exception behavior, and cancellation/resource-ownership boundaries. Add
behavioral regressions for affected paths, including failures and disabled
features. Keep CUDA imports out of the selected CPU suites. Avoid unrelated
formatting changes and make new paths and service endpoints configurable.

## Report experiments accurately

Record source hashes, environment/package versions, GPU layout, exact inputs,
output budgets, arrivals, feature settings, and error/cancellation counts.
Separate scheduled and submitted first-audio latency, and report output
lengths and the total cost of prefetch. Keep negative results. API completion
is not proof of cached data residency, playback, GPU drain, or a statistically
established speedup.

Historical measurements used earlier source snapshots. Packaging or
refactoring the source does not revalidate those measurements. The omitted
research directories and raw evidence are not prerequisites for this
repository's selected CPU checks, but may be needed by historical wrappers.

## Before sharing a change

- Run the selected CPU suite from a clean checkout without private artifacts.
- Check that newly documented files and commands exist in this trimmed tree.
- Report any required dependencies, hardware assumptions, and untested paths.
- Keep model weights, datasets, raw audio/traces, caches, results, and
  credentials out of Git. Preserve small provenance records when useful.
- Keep cleanup bounded to resources owned by the current run.

GPU installation requires a separately prepared vLLM 0.20.0 and compatible
PyTorch/CUDA environment; see the [README](README.md). Source publication does
not imply that every legacy launcher or performance experiment is supported.
