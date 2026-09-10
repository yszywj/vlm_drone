#!/usr/bin/env bash
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
  echo "usage: bash scripts/audit_target_state_quality_test_50k.sh [--dry-run]" >&2
  exit 2
fi
QUALITY_TEST_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QUALITY_TEST_ROOT"
./python.sh scripts/audit_target_state_quality_test.py \
  --config configs/target_state/audit_quality_joint_test_50k.yaml "$@"
