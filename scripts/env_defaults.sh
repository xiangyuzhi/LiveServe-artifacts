#!/usr/bin/env bash
# Shared environment defaults for validation and launcher scripts.
#
# This file is intended to be sourced.  It keeps local-machine defaults in one
# place, preferring explicit overrides and repo-local virtual environments.

liveserve_command_available() {
  local cmd="$1"
  if [[ -x "${cmd}" ]]; then
    return 0
  fi
  command -v "${cmd}" >/dev/null 2>&1
}

liveserve_resolve_pybin() {
  local repo_root="${1:-}"
  if [[ -n "${LIVESERVE_PYBIN:-}" ]]; then
    printf '%s\n' "${LIVESERVE_PYBIN}"
    return 0
  fi

  if [[ -n "${repo_root}" && -x "${repo_root}/.venv-liveserve/bin/python" ]]; then
    printf '%s\n' "${repo_root}/.venv-liveserve/bin/python"
    return 0
  fi

  if [[ -n "${repo_root}" && -x "${repo_root}/.venv-liveserve-dev/bin/python" ]]; then
    printf '%s\n' "${repo_root}/.venv-liveserve-dev/bin/python"
    return 0
  fi

  printf '%s\n' "${PYBIN_FALLBACK:-python3}"
}

liveserve_require_command() {
  local cmd="$1"
  local label="${2:-command}"
  if ! liveserve_command_available "${cmd}"; then
    echo "[env-defaults] ${label} not found or not executable: ${cmd}" >&2
    return 127
  fi
}

liveserve_export_pycache_prefix() {
  export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${TMPDIR:-/tmp}/liveserve-pycache-${UID}}"
}
