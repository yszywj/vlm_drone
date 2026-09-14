#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
instruction="$(cat "${PROJECT_ROOT}/resources/fleet_open_4x4/mission.txt")"
cd "${PROJECT_ROOT}"

# Additional arguments can select --no-headless --debug-visualization,
# override the instruction, or use --fleet-planner scripted with its local planner.
exec "${PROJECT_ROOT}/python.sh" scripts/run_fleet_mission.py \
  --config configs/multi_uav_open_4x4.yaml \
  --target-perception-mode oracle \
  --perception-runtime-profile oracle_evaluation \
  --acknowledge-privileged-oracle \
  --fleet-planner llm \
  --local-planner dynamic_llm \
  --interpreter-max-tokens 6144 \
  --fleet-max-tokens 4096 \
  --planning-contract v3 \
  --runtime-program linear \
  --max-sim-time 300 \
  --instruction "${instruction}" \
  "$@"
