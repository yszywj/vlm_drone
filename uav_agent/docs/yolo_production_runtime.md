# YOLO 生产运行手册

本文只描述 `production + ultralytics_service` 路径。YOLO worker 不可用或模型身份不匹配时，任务会在 Isaac Sim 导入前失败；该路径不会回退到 Oracle，也不会把 backend 改成 `disabled`。计数器和候选转换日志的完整语义见 [Production YOLO runtime evidence](perception_runtime_evidence.md)。

## 固定模型身份

本仓库当前生产配置固定到以下训练结果：

| 项目 | 值 |
| --- | --- |
| checkpoint | `/home/amax/ry/vlm_drones/outputs/trained_models/yolo/yolo26s_cube_v1_baseline_v2/weights/best.pt` |
| checkpoint SHA256 | `895de7caa8af200c12f343c72e3a726ffae65e4d96d2092decaf96ef4558de07` |
| 模型类别 | `{0: cube}` |
| model family | `yolo` |
| 训练 manifest | `/home/amax/ry/vlm_drones/outputs/trained_models/yolo/yolo26s_cube_v1_baseline_v2/model_manifest.json` |
| 数据 manifest | `/home/amax/ry/vlm_drones/datasets/perception/cube_v1_fixed_20260825/manifest.jsonl` |

`model_manifest.json` 记录 base/best checkpoint 路径和 SHA、数据集 manifest SHA、训练参数、依赖版本和验证指标。路径本身不是身份：部署检查以 checkpoint 内容的 SHA256、worker 返回的 `model_family` 和严格的类别映射为准。

`configs/yolo/runtime_yolo26.yaml` 与 `configs/multi_uav_cube_yolo.yaml` 都声明同一个 `expected_model_sha256` 和 `expected_model_names`。更换模型时必须同时审查训练 manifest、计算新 SHA，并显式更新配置；禁止只替换磁盘文件。

## 启动单个 worker

YOLO/BoT-SORT 应在隔离的 `yolo_perception` 环境中运行，Isaac runtime 则留在 `r_isaac_sim`。worker 只允许绑定 loopback，并保持 `workers=1`，因为 BoT-SORT stream 状态由单进程持有。

```bash
cd /home/amax/ry/vlm_drones/uav_agent

export YOLO_MODEL=/home/amax/ry/vlm_drones/outputs/trained_models/yolo/yolo26s_cube_v1_baseline_v2/weights/best.pt

CUDA_VISIBLE_DEVICES=1 \
conda run --no-capture-output -n yolo_perception \
python scripts/serve_yolo.py \
  --config configs/yolo/service_yolo26.yaml \
  --host 127.0.0.1 \
  --port 8011 \
  --model-family yolo \
  --model "$YOLO_MODEL" \
  --device 0 \
  --tracker configs/yolo/botsort_uav.yaml
```

`CUDA_VISIBLE_DEVICES=1` 后，进程内这张卡仍用 `--device 0`。启动日志必须打印解析后的模型路径、模型 SHA、类别、Torch/Ultralytics 和 CUDA 信息。

另一个终端执行只读检查：

```bash
curl --fail http://127.0.0.1:8011/health
curl --fail http://127.0.0.1:8011/v1/model-info

cd /home/amax/ry/vlm_drones/uav_agent
./python.sh scripts/check_fleet_yolo_services.py \
  --config configs/yolo/runtime_yolo26.yaml \
  --uav-id uav_1
```

检查内容包括 ready、URL、model family、严格的 `{0: cube}` 和 checkpoint SHA。任何不一致都会 fail-closed。

## 单机端到端命令

正式便捷入口会校验本地 checkpoint SHA、启动 worker、等待 `/health`、执行 preflight、调用现有 `run_fleet_mission.py`，并通过 `trap` 清理 worker：

```bash
cd /home/amax/ry/vlm_drones/uav_agent
CUDA_VISIBLE_DEVICES=0 \
./scripts/run_single_uav_yolo_e2e.sh
```

脚本默认任务为搜索红色 cube、保持约 6 m 距离跟踪 20 s、返航并降落。可使用以下环境变量覆盖资源而不修改脚本：

```bash
UAV_AGENT_YOLO_DEVICE=1 \
UAV_AGENT_MAX_SIM_TIME_S=300 \
UAV_AGENT_MISSION_INSTRUCTION='uav_1起飞到十米，前往世界坐标10,0附近20米范围搜索红色立方体目标target，找到后保持约六米距离跟踪二十秒，完成后返回起点降落' \
./scripts/run_single_uav_yolo_e2e.sh
```

worker 独立日志位于 `logs/yolo_service/single_uav_<UTC timestamp>.log`，任务结果位于 Fleet 入口打印的 `result_dir`。合格启动记录应包含：

```text
target_perception_mode: yolo
runtime_profile: production
backend_by_uav: ultralytics_service
privileged: false
```

还必须显示本次 YOLO SHA。若看到 `backend_by_uav: disabled`、Oracle acknowledgement 或 privileged source，应把本次运行判为失败。

### 当前真实冒烟状态

真实训练 checkpoint 已由 worker 加载并完成一次 YOLO + Isaac 单机闭环。证据目录为：

```text
/home/amax/ry/vlm_drones/uav_agent/logs/fleet_missions/runs/fleet_mission/20260825-185911_fleet_mission_seed0_nogit
```

[summary.json](../logs/fleet_missions/runs/fleet_mission/20260825-185911_fleet_mission_seed0_nogit/summary.json) 记录 `strict_success=true`、`status=SUCCEEDED`、`exit_code=0`、`production_vision_result=true` 和 `privileged_perception=false`，并保留 preflight 验证的 worker URL、SHA 和 `{0: cube}`。实际 Skill 生命周期见 [skill_executions.csv](../logs/fleet_missions/runs/fleet_mission/20260825-185911_fleet_mission_seed0_nogit/metrics/skill_executions.csv)：

```text
TAKEOFF_COMPLETE
SEARCH -> TARGET_FOUND
TRACK -> TRACK_COMPLETE (20.000001 s)
GOTO home -> GOAL_REACHED
LAND -> LAND_COMPLETE
```

本次 run 的 `return_success=true`、`landing_success=true`、`valid_track_duration_s=20.000001043081284`；首次 detection 为 1.367 s、lock 为 12.667 s。lock 前两个错误候选被显式拒绝，第三个 red candidate 累积稳定颜色和三维证据；这不是 Oracle shortcut。关键感知计数如下：

| 指标 | 实际值 |
| --- | ---: |
| `camera_frames_received` | 354 |
| `yolo_requests_submitted` / `yolo_results_received` | 342 / 341 |
| `detections_total` / `tracked_detections_total` | 270 / 270 |
| `candidate_created` / `candidate_confirmed` / `candidate_rejected` | 3 / 1 / 2 |
| `attribute_confirmed` / `attribute_ambiguous` | 1 / 9 |
| `depth_resolution_attempts` / successes / failures | 236 / 235 / 1 |
| `measurement_created` / `measurement_rejected` | 235 / 1 |
| `kalman_updates_accepted` / `kalman_updates_rejected` | 175 / 0 |
| `position_world_outputs` / `predicted_only_outputs` | 261 / 0 |
| `search_target_found` | 1 |
| `track_visible_updates` / `track_predicted_updates` | 200 / 0 |
| `yolo_timeouts` / response errors / stream busy | 0 / 0 / 0 |
| `target_lost_count` | 0 |

候选生命周期证据保存在 [target_perception_transitions.jsonl](../logs/fleet_missions/runs/fleet_mission/20260825-185911_fleet_mission_seed0_nogit/agents/uav_1/target_perception_transitions.jsonl)。确认行同时包含 `tracker_id=track_10`、red match、`measurement_created`、`measurement_source=isaac_depth_foreground_cluster_median`、有限 `position_world_m`、`confirmed=true` 和逻辑 `target_id=target`，且不包含 target truth、prim path 或 motion seed。四张有界调试图展示 detection、candidate、reject 和 confirmation；未保存视频或原始帧流。

早期相机 warm-up 后的 0.5 s timeout/同 stream 409 已修复。本次成功 run 还验证了持久化后的明确退出路径：旧实现只在任务和结果均成功后的 CPython/Isaac GC teardown 出现 exit 139，现于 summary、CSV 和日志落盘后调用 `os._exit(0)`，证据文件中的 `exit_code.txt` 为 0。

### 五个固定 seed

固定 seed harness、逐 episode timeout 和聚合器位于 `scripts/run_yolo_fixed_seed_eval.sh` / `experiments/yolo_fixed_seed_eval.py`。最终 `temporal_ray_depth` 批次证据在：

```text
/home/amax/ry/vlm_drones/uav_agent/logs/yolo_fixed_seed_eval/task9_temporal_5seeds
```

seed 101、211、307、401、503 均加载同一 YOLO SHA 和已晋级 Stage B v2 temporal checkpoint，完成 detection、deterministic 颜色确认、三维定位、SEARCH、20 s TRACK、返航和 LAND；五次进程 exit code 均为 0。聚合 `strict_success_rate=1.0`，六个阶段成功率均为 1.0，YOLO response rate 为 0.9961，三维 resolution success rate 为 0.99915；报告明确 `oracle_metrics_used=false`，并保留了模型 SHA。该结果只是这五个固定场景的实际观测，不代表总体成功率保证。

早期 `/logs/yolo_fixed_seed_eval/task9_5seeds` 批次使用 `geometry.mode=isaac_depth`：原始结果为 4/5，seed 503 的外部 SIGTERM 被分类为 `LAUNCH`，之后该 seed 在 `20260825-182012_fleet_mission_seed0_nogit` 独立恢复成功。旧 summary 未被覆写，继续作为失败分类与 deterministic baseline 的审计证据；不能与上述最终 temporal 5/5 混为同一批次。训练 artifact、promotion gate 和单次详细 temporal 指标见 [时序射线深度估计器](temporal_ray_depth_estimator.md)。

## 多 UAV worker URL 规则

每架并行 UAV 必须使用不同的 loopback URL、worker 进程和 tracker stream。`configs/multi_uav_cube_yolo.yaml` 的固定映射是：

```yaml
target_perception:
  backend: ultralytics_service
  yolo_service:
    url: http://127.0.0.1:8011
    per_uav_urls:
      uav_a: http://127.0.0.1:8011
      uav_b: http://127.0.0.1:8012
```

第二个 worker 的命令只改变 GPU 映射和端口：

```bash
CUDA_VISIBLE_DEVICES=2 \
conda run --no-capture-output -n yolo_perception \
python scripts/serve_yolo.py \
  --config configs/yolo/service_yolo26.yaml \
  --host 127.0.0.1 --port 8012 \
  --model-family yolo --model "$YOLO_MODEL" --device 0 \
  --tracker configs/yolo/botsort_uav.yaml
```

所有活动 UAV 都必须在 `per_uav_urls` 中恰好出现一次，URL 不得重复。启动任务前运行：

```bash
./python.sh scripts/check_fleet_yolo_services.py \
  --config configs/multi_uav_cube_yolo.yaml
```

多机只共享不可变 checkpoint；不得共享 `FrameStore`、`CandidateBank`、BoT-SORT stream 或三维 Kalman 状态。每个 UAV 的运行统计在 `perception_by_uav.<uav_id>` 下独立汇总。

## 生产调用链和确认规则

生产帧按以下路径流动：

```text
CameraSample
  -> YoloTargetPerceptionRuntime.observe
  -> CoordinatedVisionPerceptionBackend.observe
  -> TargetPerceptionCoordinator
  -> YoloServiceClient /v1/track
  -> CandidateBank
  -> deterministic RGB-D attribute / optional typed Qwen review
  -> TargetMeasurement
  -> constant-velocity Kalman TargetStateEstimator
  -> TargetEstimate
  -> MissionAgent Observation
  -> SEARCH / TRACK / REACQUIRE
```

`SEARCH`、`TRACK` 和 `REACQUIRE` 只读取 `Observation.target_estimate`，不访问 YOLO client。普通 YOLO 只检测 `cube`；red/blue 由同帧 RGB-D 的多帧确定性属性证据确认。Qwen 关闭时，清晰且受支持的颜色仍可确认；模糊、遮挡或 unsupported 属性必须保持 pending，不能降级成“自动确认”。生产配置应使用：

```yaml
confirmation:
  mode: class_track_attribute_or_qwen
```

## 常见错误与定位

| 症状 | 含义和处理 |
| --- | --- |
| worker 未启动、connection refused | 运行只读 preflight；检查 8011/8012、worker 日志、conda 环境和 GPU。任务应在 Isaac 导入前退出，不能改用 Oracle。 |
| SHA 不匹配 | 错误应同时打印 URL、期望/实际 SHA 和实际类别。核对 `sha256sum best.pt` 与训练 manifest；不要绕过检查。 |
| 类别不是严格 `{0: cube}` | 加载了 COCO/base 权重或错误训练产物。换回固定 `best.pt`，不能用 class alias 模糊匹配。 |
| confirmation mode 错误 | 将配置改为 `class_track_attribute_or_qwen`；旧的 `class_track_or_qwen` 不满足颜色候选确认契约。 |
| 颜色一直 pending | 检查目标颜色是否在 `supported_values`、HSV 饱和度/亮度、bbox 内有效前景深度比例、最少 3 次观测和 0.4 s 持续时间。模糊时 pending 是安全行为；需要 Qwen 时必须显式启用 gate。 |
| 深度采到地面 | 使用 `depth_anchor: foreground_cluster_median`；查看有界 debug image 中的 bbox、前景 cluster、采样 pixel 和 raw depth。检查相机姿态、同步 depth、bbox 是否包含太多地面。 |
| Kalman 拒绝 measurement | 查 `kalman_updates_rejected`、`measurement_rejected` 和 `world-position innovation rejected`。常见原因是时间戳不递增、covariance 非法、位置跳变超过 10 m 或 Mahalanobis gate 超限。不要提高 gate 来掩盖错误深度。 |
| target lost | 比较 `track_visible_updates`、`track_predicted_updates`、`predicted_only_outputs` 和 `target_lost_count`。2 s 内可用 bounded prediction；超龄后进入 REACQUIRE/既有安全策略。 |
| 首帧超时后 HTTP 409 busy | 当前请求尚在 worker 内执行，新请求占用同一 tracker stream。查看 worker 日志和 timeout 指标；先修正 warm-up/超时与 stream 退休时序，不能把 409 当成“无检测”。 |

所有指标只用于观察和验收，不得反向改变飞行控制行为。精确定义和候选日志字段见 [运行证据说明](perception_runtime_evidence.md)。
