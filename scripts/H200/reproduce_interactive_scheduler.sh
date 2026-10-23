#!/usr/bin/env bash
# reproduce_interactive_scheduler.sh — H200 scheduler/barge-in cells 3+4.
#
# Cells:
#   3. scheduler_wide_p10  — scheduler wide (L_hi=50 L_lo=25), c=8, p_bi=1.0
#   4. scheduler_wide_p05  — scheduler wide, c=8, p_bi=0.5  (the headline result)
#
# This script DOES NOT need an existing server — scheduler_run_one_cell.sh kills+restarts
# the server itself with the right scheduler lead-control env vars per cell.
#
# Result dirs:
#   results/H200/step2/scheduler_wide_p10/...
#   results/H200/step2/scheduler_wide_p05/...
#
# Wall budget: ~25–30 min on 2x H200-96G (each cell ~12 min).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
H200_DIR="${REPO_ROOT}/benchmarks/interactive/H200"
RUNNER="${H200_DIR}/scheduler_run_one_cell.sh"

SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "[reproduce_interactive_scheduler] unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [ "${SMOKE}" -eq 1 ]; then
  echo "[reproduce] SMOKE mode not yet wired through scheduler_run_one_cell.sh."
  echo "             Run a single cell manually:"
  echo "             bash benchmarks/interactive/H200/scheduler_run_one_cell.sh 50 25 scheduler_smoke 0.5"
  exit 0
fi

# Cell 3: scheduler wide, prob=1.0
bash "${RUNNER}" 50 25 scheduler_wide_p10 1.0

# Cell 4: scheduler wide, prob=0.5  (headline)
bash "${RUNNER}" 50 25 scheduler_wide_p05 0.5

echo
echo "[reproduce] DONE. Cells written to results/H200/step2/{scheduler_wide_p10,scheduler_wide_p05}/"
echo "Validate against frozen reference:"
echo "  python scripts/diff_summary.py --runs results/H200 --reference results/reference --tolerance 0.15"
