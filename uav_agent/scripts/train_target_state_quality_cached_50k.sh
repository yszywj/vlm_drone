#!/usr/bin/env bash
# Read the existing cache; no YOLO, simulator, GPU or PC transfer is required.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
  echo "usage: bash scripts/train_target_state_quality_cached_50k.sh [--dry-run]" >&2
  exit 2
fi
QUALITY_AGENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QUALITY_AGENT_ROOT"
./python.sh scripts/fit_target_state_quality_cached.py \
  --config configs/target_state/quality_cached_50k.yaml "$@"
