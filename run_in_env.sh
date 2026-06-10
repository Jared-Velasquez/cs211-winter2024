#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${SCRIPT_DIR}/.conda-baseline"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
CONDA_PYTHON="${ENV_PREFIX}/bin/python"

if [[ -x "${VENV_PYTHON}" ]]; then
  PYTHON_BIN="${VENV_PYTHON}"
elif [[ -x "${CONDA_PYTHON}" ]]; then
  PYTHON_BIN="${CONDA_PYTHON}"
else
  echo "No repo-local environment was found."
  echo "Run ${SCRIPT_DIR}/setup_env.sh first."
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/.cache/matplotlib" "${SCRIPT_DIR}/.cache/fontconfig"
export MPLCONFIGDIR="${SCRIPT_DIR}/.cache/matplotlib"
export XDG_CACHE_HOME="${SCRIPT_DIR}/.cache"

if [[ $# -eq 0 ]]; then
  echo "Usage: ${SCRIPT_DIR}/run_in_env.sh python <script> [args...]"
  exit 1
fi

case "$1" in
  python)
    shift
    exec "${PYTHON_BIN}" "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
