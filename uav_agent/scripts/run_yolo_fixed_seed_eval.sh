#!/usr/bin/env bash
set -euo pipefail

# One isolated detector/tracker worker is shared by all fixed-seed Isaac runs.
# Environment variables intentionally match run_single_uav_yolo_e2e.sh.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${UAV_AGENT_SINGLE_YOLO_CONFIG:-${PROJECT_ROOT}/configs/yolo/runtime_yolo26.yaml}"
MODEL="${UAV_AGENT_YOLO_MODEL:-/home/amax/ry/vlm_drones/outputs/trained_models/yolo/yolo26s_cube_v1_baseline_v2/weights/best.pt}"
EXPECTED_SHA256="895de7caa8af200c12f343c72e3a726ffae65e4d96d2092decaf96ef4558de07"
YOLO_ENV="${UAV_AGENT_YOLO_CONDA_ENV:-yolo_perception}"
CONDA_EXE="${UAV_AGENT_CONDA_EXE:-/home/amax/miniconda3/bin/conda}"
WORKER_URL="http://127.0.0.1:8011"
LOG_ROOT="${UAV_AGENT_YOLO_LOG_ROOT:-${PROJECT_ROOT}/logs/yolo_service}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORKER_LOG="${LOG_ROOT}/fixed_seed_${RUN_STAMP}.log"
WORKER_PID=""
OWNS_WORKER=0

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "${OWNS_WORKER}" -eq 1 && -n "${WORKER_PID}" ]] && kill -0 -- "-${WORKER_PID}" 2>/dev/null; then
    kill -TERM -- "-${WORKER_PID}" 2>/dev/null || true
    for _ in $(seq 1 50); do
      if ! kill -0 -- "-${WORKER_PID}" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if kill -0 -- "-${WORKER_PID}" 2>/dev/null; then
      kill -KILL -- "-${WORKER_PID}" 2>/dev/null || true
    fi
    wait "${WORKER_PID}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cd "${PROJECT_ROOT}"
  exec ./python.sh scripts/run_yolo_fixed_seed_eval.py "$@"
fi

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "error: conda executable is unavailable: ${CONDA_EXE}" >&2
  exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "error: setsid is required for bounded worker cleanup" >&2
  exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "error: trained YOLO checkpoint is unavailable: ${MODEL}" >&2
  exit 2
fi
ACTUAL_SHA256="$(sha256sum "${MODEL}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  echo "error: checkpoint SHA256 mismatch: expected=${EXPECTED_SHA256} actual=${ACTUAL_SHA256}" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

if curl --fail --silent --show-error --connect-timeout 1 --max-time 2 \
  "${WORKER_URL}/health" 2>/dev/null \
  | grep -Eq '"ready"[[:space:]]*:[[:space:]]*true'; then
  echo "[fixed-seed-yolo] reusing healthy worker at ${WORKER_URL}; model identity will be preflighted"
else
  setsid "${CONDA_EXE}" run --no-capture-output -n "${YOLO_ENV}" \
    python scripts/serve_yolo.py \
    --config configs/yolo/service_yolo26.yaml \
    --host 127.0.0.1 \
    --port 8011 \
    --model-family yolo \
    --model "${MODEL}" \
    --device "${UAV_AGENT_YOLO_DEVICE:-0}" \
    --tracker configs/yolo/botsort_uav.yaml \
    >"${WORKER_LOG}" 2>&1 &
  WORKER_PID=$!
  OWNS_WORKER=1
  echo "[fixed-seed-yolo] worker_pid=${WORKER_PID} worker_log=${WORKER_LOG} model_sha256=${ACTUAL_SHA256}"
  for _ in $(seq 1 120); do
    if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
      echo "error: YOLO worker exited before becoming ready; see ${WORKER_LOG}" >&2
      exit 2
    fi
    if curl --fail --silent --show-error --connect-timeout 1 --max-time 2 \
      "${WORKER_URL}/health" \
      | grep -Eq '"ready"[[:space:]]*:[[:space:]]*true'; then
      break
    fi
    sleep 1
  done
fi

if ! curl --fail --silent --show-error --connect-timeout 1 --max-time 2 \
  "${WORKER_URL}/health" \
  | grep -Eq '"ready"[[:space:]]*:[[:space:]]*true'; then
  echo "error: YOLO worker did not report ready within 120 seconds; see ${WORKER_LOG}" >&2
  exit 2
fi

# This endpoint validates service readiness, family, class map, tracker, and
# the exact model digest before the first expensive Isaac startup.
./python.sh scripts/check_fleet_yolo_services.py \
  --config "${CONFIG}" \
  --uav-id uav_1

./python.sh scripts/run_yolo_fixed_seed_eval.py \
  --config "${CONFIG}" \
  "$@"
