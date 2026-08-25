#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${UAV_AGENT_SINGLE_YOLO_CONFIG:-${PROJECT_ROOT}/configs/yolo/runtime_yolo26.yaml}"
MODEL="${UAV_AGENT_YOLO_MODEL:-/home/amax/ry/vlm_drones/outputs/trained_models/yolo/yolo26s_cube_v1_baseline_v2/weights/best.pt}"
EXPECTED_SHA256="895de7caa8af200c12f343c72e3a726ffae65e4d96d2092decaf96ef4558de07"
YOLO_ENV="${UAV_AGENT_YOLO_CONDA_ENV:-yolo_perception}"
CONDA_EXE="${UAV_AGENT_CONDA_EXE:-/home/amax/miniconda3/bin/conda}"
WORKER_URL="http://127.0.0.1:8011"
LOG_ROOT="${UAV_AGENT_YOLO_LOG_ROOT:-${PROJECT_ROOT}/logs/yolo_service}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORKER_LOG="${LOG_ROOT}/single_uav_${RUN_STAMP}.log"
WORKER_PID=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${WORKER_PID}" ]] && kill -0 -- "-${WORKER_PID}" 2>/dev/null; then
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

echo "[single-yolo-e2e] worker_pid=${WORKER_PID} worker_log=${WORKER_LOG} model_sha256=${ACTUAL_SHA256}"
for _ in $(seq 1 120); do
  if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
    echo "error: YOLO worker exited before becoming ready; see ${WORKER_LOG}" >&2
    exit 2
  fi
  if curl --fail --silent --connect-timeout 1 --max-time 2 \
    "${WORKER_URL}/health" \
    | grep -Eq '"ready"[[:space:]]*:[[:space:]]*true'; then
    break
  fi
  sleep 1
done

if ! curl --fail --silent --show-error --connect-timeout 1 --max-time 2 \
  "${WORKER_URL}/health" \
  | grep -Eq '"ready"[[:space:]]*:[[:space:]]*true'; then
  echo "error: YOLO worker did not report ready within 120 seconds; see ${WORKER_LOG}" >&2
  exit 2
fi

./python.sh scripts/check_fleet_yolo_services.py \
  --config "${CONFIG}" \
  --uav-id uav_1

./python.sh scripts/run_fleet_mission.py \
  --config "${CONFIG}" \
  --mission-interpreter scripted \
  --fleet-planner scripted \
  --local-planner dynamic_scripted \
  --planning-contract v3 \
  --runtime-program linear \
  --target-perception-mode yolo \
  --perception-runtime-profile production \
  --headless \
  --max-sim-time "${UAV_AGENT_MAX_SIM_TIME_S:-300}" \
  --instruction "${UAV_AGENT_MISSION_INSTRUCTION:-uav_1起飞到十米，前往世界坐标10,0附近20米范围搜索红色立方体目标target，找到后保持约六米距离跟踪二十秒，完成后返回起点降落}"
