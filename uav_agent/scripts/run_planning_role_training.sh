#!/usr/bin/env bash
# One tmux window per independently trained planning adapter.
set -euo pipefail

role="${1:?role required}"
gpu="${2:?GPU index required}"
run_id="${3:?run ID required}"
launch_dir="${4:?launch directory required}"
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
training_python="${project_root}/../.venvs/planning_lora/bin/python"
case "$role" in mission_interpreter|fleet_planner|spatial_mission) ;; *) exit 2 ;; esac
[[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
mkdir -p -- "$launch_dir"
trap 'training_exit=$?; printf "%s\n" "$training_exit" > "$launch_dir/$role.exit_code"' EXIT

export CUDA_VISIBLE_DEVICES="$gpu"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd -- "$project_root"
"$training_python" training/lora/train_planning_role_lora.py \
    --config "configs/lora/${role}_train.json" \
    --role "$role" --run-id "$run_id" 2>&1 | tee "$launch_dir/$role.log"
