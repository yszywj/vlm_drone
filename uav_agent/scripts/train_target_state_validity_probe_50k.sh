#!/usr/bin/env bash
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run && "$1" != --fit-only ) ]]; then
  echo "usage: bash scripts/train_target_state_validity_probe_50k.sh [--dry-run|--fit-only]" >&2
  exit 2
fi
AGENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AGENT_ROOT"
PROBE_CONFIG=configs/target_state/validity_probe_geometry_v2_50k.yaml
if [[ "${1:-}" == --dry-run ]]; then
  ./python.sh scripts/target_state_validity_probe.py prepare --config "$PROBE_CONFIG" --dry-run
  exit 0
fi
if [[ "${1:-}" != --fit-only ]]; then
  ./python.sh scripts/target_state_validity_probe.py prepare --config "$PROBE_CONFIG"
fi
./python.sh scripts/target_state_validity_probe.py train --config "$PROBE_CONFIG"
