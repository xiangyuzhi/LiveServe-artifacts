#!/usr/bin/env bash
# Run the no-GPU validation gate for the Ming integration branch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

source "${ROOT}/scripts/env_defaults.sh"
PYBIN="$(liveserve_resolve_pybin "${ROOT}")"
liveserve_require_command "${PYBIN}" "python"
liveserve_export_pycache_prefix

run() {
  printf '\n[validate-ming-no-gpu] +'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

run "${PYBIN}" scripts/validate_ming_runner_wiring.py

shell_scripts=(
  scripts/env_defaults.sh
  profiling/scheduler_analysis/scripts/gpu_preflight_common.sh
  profiling/scheduler_analysis/scripts/run_ming_cbound_smoke.sh
  profiling/scheduler_analysis/scripts/run_ming_4gpu_tuning_sweep.sh
  scripts/H200/start_ming_4gpu_dp2_baseline.sh
  scripts/H200/start_ming_4gpu_dp2_async.sh
)
for script in "${shell_scripts[@]}"; do
  run bash -n "${script}"
done

python_files=(
  scripts/validate_ming_runner_wiring.py
  benchmarks/ming/H200/ming_stream_benchmark.py
  benchmarks/ming/H200/ming_quality_probe.py
  benchmarks/ming/H200/ming_cbound_driver.py
  profiling/scheduler_analysis/scripts/summarize_ming_4gpu_tuning.py
)
run "${PYBIN}" -m py_compile "${python_files[@]}"

run git diff --check

echo
echo "[validate-ming-no-gpu] PASS"
