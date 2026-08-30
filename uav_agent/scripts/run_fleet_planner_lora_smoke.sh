#!/usr/bin/env bash
set -euo pipefail

# Real, opt-in Qwen3-VL smoke training. Run this only inside the dedicated
# qwen_lora environment; it deliberately never invokes python.sh/Isaac Sim.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${QWEN_LORA_PYTHON:-python}"
VISIBLE_GPU="${QWEN_LORA_CUDA_VISIBLE_DEVICES:-0}"
TIMESTAMP="$(date -u +%Y%m%d-%H%M%SZ)"
RUN_ID="${QWEN_LORA_SMOKE_RUN_ID:-smoke_${TIMESTAMP}_$$}"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]]; then
  echo "invalid QWEN_LORA_SMOKE_RUN_ID: ${RUN_ID}" >&2
  exit 2
fi
EXAMPLE_CONFIG="${QWEN_LORA_SMOKE_BASE_CONFIG:-${PROJECT_ROOT}/configs/lora/fleet_planner_lora_train.example.json}"
OUTPUT_ROOT="${QWEN_LORA_SMOKE_OUTPUT_ROOT:-/home/amax/ry/vlm_drones/outputs/lora/fleet_planner_smoke}"
ADAPTER_ROOT="${QWEN_LORA_SMOKE_ADAPTER_ROOT:-/home/amax/ry/vlm_drones/models/adapters/fleet_planner_smoke}"
CONFIG_DIR="${OUTPUT_ROOT}/configs"
SMOKE_CONFIG="${CONFIG_DIR}/${RUN_ID}.json"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
ADAPTER_DIR="${ADAPTER_ROOT}/${RUN_ID}"
STAGED_LOG="${OUTPUT_ROOT}/.${RUN_ID}.terminal.log"

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [[ -e "${SMOKE_CONFIG}" || -e "${RUN_DIR}" || -e "${ADAPTER_DIR}" || -e "${STAGED_LOG}" ]]; then
  echo "smoke run already exists; choose a new QWEN_LORA_SMOKE_RUN_ID: ${RUN_ID}" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${EXAMPLE_CONFIG}" "${SMOKE_CONFIG}" "${OUTPUT_ROOT}" "${ADAPTER_ROOT}" "${PROJECT_ROOT}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

source, destination, output_root, adapter_root, project_root = map(Path, sys.argv[1:])
payload = json.loads(source.read_text(encoding="utf-8"))
output_root = output_root.expanduser().resolve()
adapter_root = adapter_root.expanduser().resolve()
destination = destination.expanduser().resolve()
if destination.parent != output_root / "configs":
    raise SystemExit("smoke config path escaped the reviewed output root")
base_model = Path(payload["base_model_path"]).expanduser().resolve()
dataset_value = Path(payload["dataset_dir"]).expanduser()
dataset_root = (
    (project_root.resolve() / dataset_value).resolve()
    if dataset_value.parts and dataset_value.parts[0] == "datasets"
    else dataset_value.resolve()
)

def overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents

if overlaps(output_root, adapter_root):
    raise SystemExit("smoke output and adapter roots must be separate, non-nested paths")
for name, candidate in (("output", output_root), ("adapter", adapter_root)):
    if overlaps(candidate, base_model) or overlaps(candidate, dataset_root):
        raise SystemExit(f"smoke {name} root overlaps protected model/dataset data")
payload.update(
    {
        "status": "active",
        "output_dir": str(output_root),
        "adapter_output_dir": str(adapter_root),
        "max_steps": 1,
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "eval_steps": 1,
        "save_steps": 1,
        "save_total_limit": 1,
        "dataloader_num_workers": 0,
        "max_train_samples": 2,
        "max_validation_samples": 1,
        "notes": "Real one-step smoke only; not an effectiveness result or deployable release.",
    }
)
if destination.parent.exists() and (
    destination.parent.is_symlink() or not destination.parent.is_dir()
):
    raise SystemExit(f"smoke config directory is unsafe: {destination.parent}")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(
    json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "[qwen-lora-smoke] run_id=${RUN_ID}"
echo "[qwen-lora-smoke] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[qwen-lora-smoke] config=${SMOKE_CONFIG}"

set +e
"${PYTHON_BIN}" "${PROJECT_ROOT}/training/lora/train_fleet_planner_lora.py" \
  --config "${SMOKE_CONFIG}" \
  --run-id "${RUN_ID}" 2>&1 | tee "${STAGED_LOG}"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

mkdir -p "${RUN_DIR}"
mv "${STAGED_LOG}" "${RUN_DIR}/terminal.log"
if [[ ${TRAIN_STATUS} -ne 0 ]]; then
  echo "[qwen-lora-smoke] failed; terminal log preserved at ${RUN_DIR}/terminal.log" >&2
  exit "${TRAIN_STATUS}"
fi

for required in \
  "${RUN_DIR}/config.json" \
  "${RUN_DIR}/run_manifest.json" \
  "${RUN_DIR}/terminal.log" \
  "${ADAPTER_DIR}/adapter_config.json" \
  "${ADAPTER_DIR}/adapter_model.safetensors" \
  "${ADAPTER_DIR}/adapter_manifest.json"; do
  if [[ ! -s "${required}" ]]; then
    echo "[qwen-lora-smoke] required artifact missing or empty: ${required}" >&2
    exit 3
  fi
done

"${PYTHON_BIN}" - "${RUN_DIR}/run_manifest.json" "${ADAPTER_DIR}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

manifest_path, adapter_dir = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {
    "training_started": True,
    "weights_created": True,
    "train_count": 2,
    "validation_count": 1,
    "max_steps": 1,
    "global_step": 1,
}
bad = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
if bad:
    raise SystemExit(f"smoke manifest invariant failed: {bad}")
if Path(manifest.get("final_adapter_path", "")).resolve() != adapter_dir.resolve():
    raise SystemExit("smoke manifest final_adapter_path mismatch")
print(json.dumps({"ok": True, "run_manifest": str(manifest_path), "adapter": str(adapter_dir)}, sort_keys=True))
PY

echo "[qwen-lora-smoke] success"
