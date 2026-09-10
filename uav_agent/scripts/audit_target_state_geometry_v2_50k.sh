#!/usr/bin/env bash
# Read-only model replay; writes only a separate, resumable audit/evidence run.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
  echo "usage: bash scripts/audit_target_state_geometry_v2_50k.sh [--dry-run]" >&2
  exit 2
fi
AGENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AGENT_ROOT"
./python.sh scripts/audit_target_state_sharded.py \
  --model-manifest /home/amax/ry/vlm_drones/outputs/trained_models/target_state_geometry_v2_50k/stageb_geometry_v2_50k_v1/model_manifest.json \
  --shard-index /home/amax/ry/vlm_drones/datasets/stage_a_indexes/target_state_extreme_v1_50k_shard_index.json \
  --pc-trans-root /home/amax/ry/pc_trans \
  --pc-trans-config /home/amax/ry/pc_trans/config/config.json \
  --bridge-root /home/amax/ry/vlm_drones/datasets/_bridge \
  --run-id-prefix audit_geometry_v2_50k_v1 \
  --output-dir /home/amax/ry/vlm_drones/outputs/diagnostics/audit_geometry_v2_50k_v1 \
  --case-episode episode_000516 \
  --case-assets-max-gib 2 \
  --case-error-m 1 \
  --device cuda:0 \
  --num-workers 4 \
  --wait-timeout 86400 \
  "$@"
