#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${SCRIPT_DIR}/.conda-baseline"
ENV_FILE="${SCRIPT_DIR}/environment.yml"

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

CONDA_BIN="$(find_conda || true)"
if [[ -z "${CONDA_BIN}" ]]; then
  echo "No conda or mamba installation was found."
  echo "Install Miniforge or make conda/mamba available on PATH, then rerun this script."
  echo "If you want a repo-local install, place Miniforge at ${SCRIPT_DIR}/.miniforge3."
  exit 1
fi

echo "Creating or updating baseline environment at ${ENV_PREFIX}"
if [[ -d "${ENV_PREFIX}" ]]; then
  rm -rf "${ENV_PREFIX}"
fi
"${CONDA_BIN}" env create -p "${ENV_PREFIX}" -f "${ENV_FILE}"

echo
echo "Environment ready."
echo "Run Python inside it with:"
echo "  ${SCRIPT_DIR}/run_in_env.sh python run_baseline.py --config configs/task_a_dlc.json --frame-limit 10"
