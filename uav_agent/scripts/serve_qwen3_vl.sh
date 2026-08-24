#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${REPO_ROOT}/models/initial_model/Qwen3-VL-4B-Instruct}"
QWEN_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-Qwen3-VL-4B-Instruct}"
QWEN_HOST="${QWEN_HOST:-127.0.0.1}"
QWEN_PORT="${QWEN_PORT:-8000}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-4096}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.90}"
QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-1}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
QWEN_ADAPTER_CONFIG="${QWEN_ADAPTER_CONFIG:-${SCRIPT_DIR}/../configs/adapters.json}"

fail() {
  local exit_code="$1"
  shift
  printf 'Error: %s\n' "$*" >&2
  exit "${exit_code}"
}

[[ -d "${QWEN_MODEL_PATH}" ]] || \
  fail 2 "model directory does not exist: ${QWEN_MODEL_PATH}"

required_model_files=(
  "config.json"
  "model.safetensors.index.json"
  "model-00001-of-00002.safetensors"
  "model-00002-of-00002.safetensors"
)

for filename in "${required_model_files[@]}"; do
  [[ -f "${QWEN_MODEL_PATH}/${filename}" ]] || \
    fail 2 "required model file does not exist: ${QWEN_MODEL_PATH}/${filename}"
done

if [[ "${VLLM_BIN}" == */* ]]; then
  [[ -x "${VLLM_BIN}" ]] || \
    fail 127 "vLLM executable is unavailable or not executable: ${VLLM_BIN}"
elif ! command -v "${VLLM_BIN}" >/dev/null 2>&1; then
  fail 127 "vLLM command not found: ${VLLM_BIN}. Install vLLM in a separate compatible environment or set VLLM_BIN."
fi

[[ -f "${QWEN_ADAPTER_CONFIG}" ]] || \
  fail 2 "adapter config does not exist: ${QWEN_ADAPTER_CONFIG}"
if [[ "${PYTHON_BIN}" == */* ]]; then
  [[ -x "${PYTHON_BIN}" ]] || fail 127 "Python executable is unavailable: ${PYTHON_BIN}"
elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  fail 127 "Python command not found: ${PYTHON_BIN}"
fi

lora_args_file="$(mktemp)"
trap 'rm -f -- "${lora_args_file}"' EXIT
if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/build_vllm_lora_args.py" \
  --config "${QWEN_ADAPTER_CONFIG}" \
  --expected-base-model-name "${QWEN_SERVED_MODEL_NAME}" \
  --format lines >"${lora_args_file}"; then
  fail 2 "invalid Adapter/LoRA configuration"
fi
mapfile -t QWEN_LORA_ARGS <"${lora_args_file}"
rm -f -- "${lora_args_file}"
trap - EXIT

printf '%s\n' \
  '[Qwen3-VL server]' \
  "Model path: ${QWEN_MODEL_PATH}" \
  "Served model name: ${QWEN_SERVED_MODEL_NAME}" \
  "Bind address: ${QWEN_HOST}:${QWEN_PORT}" \
  "CUDA visible devices: ${QWEN_CUDA_VISIBLE_DEVICES}" \
  "Maximum model length: ${QWEN_MAX_MODEL_LEN}" \
  "GPU memory utilization: ${QWEN_GPU_MEMORY_UTILIZATION}"
if ((${#QWEN_LORA_ARGS[@]} > 0)); then
  printf 'Static active LoRA adapters (including mission interpreter when active): enabled (%s vLLM arguments)\n' "${#QWEN_LORA_ARGS[@]}"
else
  printf 'Static active LoRA adapters: none; serving base model only\n'
fi

export CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES}"
exec "${VLLM_BIN}" serve "${QWEN_MODEL_PATH}" \
  --served-model-name "${QWEN_SERVED_MODEL_NAME}" \
  --host "${QWEN_HOST}" \
  --port "${QWEN_PORT}" \
  --dtype float16 \
  --max-model-len "${QWEN_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${QWEN_GPU_MEMORY_UTILIZATION}" \
  "${QWEN_LORA_ARGS[@]}"
