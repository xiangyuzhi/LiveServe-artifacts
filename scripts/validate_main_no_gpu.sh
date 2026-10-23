#!/usr/bin/env bash
# Run project-level no-GPU validation for the current main worktree.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

source "${ROOT}/scripts/env_defaults.sh"
PYBIN="$(liveserve_resolve_pybin "${ROOT}")"
liveserve_require_command "${PYBIN}" "python"
liveserve_export_pycache_prefix

case "${LIVESERVE_VALIDATE_EXTERNAL_ARTIFACTS:-0}" in
  0) export SKIP_QWEN_ARTIFACT_VALIDATION="${SKIP_QWEN_ARTIFACT_VALIDATION:-1}"
     export SKIP_REDUCED_SMOKE_ARTIFACT_VALIDATION="${SKIP_REDUCED_SMOKE_ARTIFACT_VALIDATION:-1}" ;;
  1) ;;
  *) echo "LIVESERVE_VALIDATE_EXTERNAL_ARTIFACTS must be 0 or 1" >&2; exit 2 ;;
esac

run() {
  printf '\n[validate-main-no-gpu] +'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

run bash scripts/validate_qwen_main_no_gpu.sh
run bash scripts/validate_ming_no_gpu.sh
run "${PYBIN}" scripts/validate_cpu_tests.py
if [ "${SKIP_REDUCED_SMOKE_ARTIFACT_VALIDATION:-0}" != "1" ]; then
  run "${PYBIN}" scripts/validate_reduced_smoke_artifacts.py
else
  echo "[validate-main-no-gpu] skipping reduced smoke artifact validation by SKIP_REDUCED_SMOKE_ARTIFACT_VALIDATION=1"
fi

echo
echo "[validate-main-no-gpu] PASS"
