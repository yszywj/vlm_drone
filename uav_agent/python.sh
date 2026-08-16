#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_PATH="${UAV_AGENT_CONDA_ENV:-/home/amax/miniconda3/envs/r_isaac_sim}"

if [[ ! -d "${CONDA_ENV_PATH}" ]]; then
    echo "error: Conda environment not found at ${CONDA_ENV_PATH}" >&2
    echo "set UAV_AGENT_CONDA_ENV to the r_isaac_sim environment prefix" >&2
    exit 1
fi

CONDA_BIN="${UAV_AGENT_CONDA_BIN:-}"
if [[ -z "${CONDA_BIN}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BIN="$(command -v conda)"
    elif [[ "${CONDA_ENV_PATH}" == */envs/* ]]; then
        CONDA_BASE="${CONDA_ENV_PATH%%/envs/*}"
        if [[ -x "${CONDA_BASE}/bin/conda" ]]; then
            CONDA_BIN="${CONDA_BASE}/bin/conda"
        fi
    fi
fi

if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
    echo "error: Conda executable not found" >&2
    echo "set UAV_AGENT_CONDA_BIN to the conda executable" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
# Keep the historical top-level imports (configs, env, skills, ...) working,
# while also supporting ``python -m uav_agent.<module>`` from documentation.
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/.."
export PYTHONDONTWRITEBYTECODE=1
exec "${CONDA_BIN}" run --no-capture-output -p "${CONDA_ENV_PATH}" python "$@"
