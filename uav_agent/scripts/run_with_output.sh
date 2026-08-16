#!/usr/bin/env bash
# Generic unbuffered launcher for an experiment directory created by RunManager.
set -uo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
    echo "error: RUN_DIR must point to an experiment run directory" >&2
    exit 2
fi
if [[ "$#" -eq 0 ]]; then
    echo "usage: RUN_DIR=/path/to/run $0 <python arguments...>" >&2
    exit 2
fi

mkdir -p -- "${RUN_DIR}/logs"
TERMINAL_LOG="${RUN_DIR}/logs/terminal.log"

if [[ "${VLM_DRONE_RESUME:-0}" == "1" ]]; then
    printf '%s\n' '================ RUN RESUMED ================' \
        | tee -a "${TERMINAL_LOG}"
fi

python -u "$@" 2>&1 | tee -a "${TERMINAL_LOG}"
EXIT_CODE=${PIPESTATUS[0]}
printf '%s\n' "${EXIT_CODE}" > "${RUN_DIR}/exit_code.txt"

exit "${EXIT_CODE}"
