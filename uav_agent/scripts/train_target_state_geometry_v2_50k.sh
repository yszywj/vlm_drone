#!/usr/bin/env bash
# New model-only Stage A initialization, then Stage B; safe same-run resume.
# Does not collect data, alter source shards, or deploy either checkpoint.
set -euo pipefail
if [[ $# -ne 0 ]]; then
  echo "usage: bash scripts/train_target_state_geometry_v2_50k.sh" >&2
  exit 2
fi
AGENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AGENT_ROOT"
MODEL_ROOT=/home/amax/ry/vlm_drones/outputs/trained_models/target_state_geometry_v2_50k

for stage in stagea stageb; do
  run_id="${stage}_geometry_v2_50k_v1"
  latest="${MODEL_ROOT}/${run_id}/latest.pt"
  resume_arguments=()
  if [[ -f "$latest" ]]; then
    resume_arguments=(--resume-checkpoint "$latest")
  fi
  ./python.sh scripts/train_target_state_sharded.py \
    --config "configs/target_state/train_geometry_v2_${stage}_50k.yaml" \
    --shard-index /home/amax/ry/vlm_drones/datasets/stage_a_indexes/target_state_extreme_v1_50k_shard_index.json \
    --pc-trans-root /home/amax/ry/pc_trans \
    --pc-trans-config /home/amax/ry/pc_trans/config/config.json \
    --bridge-root /home/amax/ry/vlm_drones/datasets/_bridge \
    --run-id-prefix "$run_id" \
    --wait-timeout 86400 \
    "${resume_arguments[@]}"
done
echo "Geometry V2 training/evaluation finished. Inspect Stage B model_manifest.json; no model was deployed."
