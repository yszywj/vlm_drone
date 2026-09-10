#!/usr/bin/env bash
# CPU head fitting only: never request shards or rerun the feature producer.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
  echo "usage: bash scripts/train_target_state_validity_cached_50k.sh [--dry-run]" >&2
  exit 2
fi
AGENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AGENT_ROOT"
./python.sh scripts/fit_target_state_validity_cached.py \
  --config configs/target_state/validity_cached_regularized_50k.yaml "$@"
