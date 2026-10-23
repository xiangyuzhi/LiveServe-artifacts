#!/usr/bin/env bash
# reproduce_sharegpt_baseline.sh — H200 ShareGPT baseline probe (governor OFF).
#
# Wraps benchmarks/sharegpt/H200/sharegpt_benchmark.sh with sane H200 defaults.
# Server must already be running with governor OFF
# (use scripts/H200/start_server_2gpu_baseline.sh).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHAREGPT_DIR="${REPO_ROOT}/benchmarks/sharegpt/H200"

if ! curl -s -o /dev/null -w '%{http_code}' -m 2 http://localhost:8666/health | grep -q 200; then
  echo "[fatal] server not healthy on :8666 — start it first via scripts/H200/start_server_2gpu_baseline.sh" >&2
  exit 3
fi

cd "${SHAREGPT_DIR}"
exec bash sharegpt_benchmark.sh "$@"
