#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="/home/amax/miniconda3/bin/conda"
CONDA_ENV="/home/amax/miniconda3/envs/r_isaac_sim"

if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "error: conda executable not found at ${CONDA_BIN}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
exec "${CONDA_BIN}" run --no-capture-output -p "${CONDA_ENV}" python "$@"
