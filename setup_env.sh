#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${SCRIPT_DIR}/.conda-baseline"
ENV_FILE="${SCRIPT_DIR}/environment.yml"
VENV_DIR="${SCRIPT_DIR}/.venv"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

usage() {
  cat <<EOF
Usage: ${0##*/} [--auto|--venv|--conda]

  --auto   Prefer a repo-local .venv if a compatible Python is available.
           Fall back to conda otherwise. (default)
  --venv   Force creation of a repo-local .venv.
  --conda  Force creation of a repo-local .conda-baseline environment.
EOF
}

MODE="auto"
if [[ $# -gt 1 ]]; then
  usage
  exit 1
fi

if [[ $# -eq 1 ]]; then
  case "$1" in
    --auto)
      MODE="auto"
      ;;
    --venv)
      MODE="venv"
      ;;
    --conda)
      MODE="conda"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
fi

find_python_for_venv() {
  local candidates=(python3.10 python3.11 python3.9)
  local candidate
  for candidate in "${candidates[@]}"; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      printf '%s\n' "$(command -v "${candidate}")"
      return 0
    fi
  done
  return 1
}

create_venv() {
  local python_bin="$1"
  echo "Creating or updating baseline virtualenv at ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
  "${python_bin}" -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${VENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS_FILE}"
  echo
  echo "Environment ready."
  echo "Run Python inside it with:"
  echo "  ${SCRIPT_DIR}/run_in_env.sh python run_baseline.py --config configs/task_a_dlc.json --frame-limit 10"
}

find_conda() {
  if command -v mamba >/dev/null 2>&1; then
    command -v mamba
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  if [[ -x "${SCRIPT_DIR}/.miniforge3/bin/mamba" ]]; then
    printf '%s\n' "${SCRIPT_DIR}/.miniforge3/bin/mamba"
    return
  fi
  if [[ -x "${SCRIPT_DIR}/.miniforge3/bin/conda" ]]; then
    printf '%s\n' "${SCRIPT_DIR}/.miniforge3/bin/conda"
    return
  fi
  return 1
}

if [[ "${MODE}" == "auto" || "${MODE}" == "venv" ]]; then
  PYTHON_BIN="$(find_python_for_venv || true)"
  if [[ -n "${PYTHON_BIN}" ]]; then
    create_venv "${PYTHON_BIN}"
    exit 0
  fi

  if [[ "${MODE}" == "venv" ]]; then
    echo "No compatible Python interpreter was found for a repo-local .venv."
    echo "Install Python 3.9, 3.10, or 3.11 and rerun this script, or use --conda."
    exit 1
  fi
fi

CONDA_BIN="$(find_conda || true)"
if [[ -z "${CONDA_BIN}" ]]; then
  echo "No compatible Python interpreter was found for .venv setup, and no conda or mamba installation was found."
  echo "Install Python 3.9, 3.10, or 3.11 for the default .venv path, or make conda/mamba available."
  echo "If you want a repo-local conda install, place Miniforge at ${SCRIPT_DIR}/.miniforge3."
  exit 1
fi

echo "Creating or updating baseline conda environment at ${ENV_PREFIX}"
rm -rf "${ENV_PREFIX}"
"${CONDA_BIN}" env create -p "${ENV_PREFIX}" -f "${ENV_FILE}"

echo
echo "Environment ready."
echo "Run Python inside it with:"
echo "  ${SCRIPT_DIR}/run_in_env.sh python run_baseline.py --config configs/task_a_dlc.json --frame-limit 10"
