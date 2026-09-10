#!/usr/bin/env bash
# Two fixed weight comparisons on the existing CPU cache; no PC transfer.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
  echo "usage: bash scripts/train_target_state_quality_mild_50k.sh [--dry-run]" >&2
  exit 2
fi
QUALITY_MILD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QUALITY_MILD_ROOT"
QUALITY_MILD_CONFIGS=(
  configs/target_state/quality_cached_power025_50k.yaml
  configs/target_state/quality_cached_power050_50k.yaml
)

# Verify both caches/config contracts before starting either experiment.
# Failures stop immediately; never fall back to extraction or skip a bad run.
for quality_config in "${QUALITY_MILD_CONFIGS[@]}"; do
  ./python.sh scripts/fit_target_state_quality_cached.py --config "$quality_config" --dry-run
done
if [[ $# -eq 1 ]]; then
  exit 0
fi

for quality_config in "${QUALITY_MILD_CONFIGS[@]}"; do
  ./python.sh scripts/fit_target_state_quality_cached.py --config "$quality_config"
done
echo "Both quality-weight comparisons completed; inspect BOTH report.json files. No model was deployed."
