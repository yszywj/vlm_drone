# 感知信息边界

本项目把 Agent Runtime 和 Evaluator/Training 的能力分开，而不只是隐藏几个字段。生产 YOLO runtime 构造函数不接收 environment、target object、evaluator frame getter 或 Oracle factory；生产 bridge 只能获得传感器 sample、封闭的 `TargetQuerySpec` 和本 UAV 的视觉状态。

## 运行 profile

| 用途 | runtime profile | backend | privileged acknowledgement |
| --- | --- | --- | --- |
| 生产目标视觉 | `production` | `ultralytics_service` | 必须为 false |
| 无目标感知任务 | `production` | `disabled` | 必须为 false |
| 理想上界、回归、标签/evaluator | `oracle_evaluation` | `oracle_evaluation` | 必须显式为 true |

生产 profile 不允许 Oracle acknowledgement。Oracle profile 未 acknowledgement 也会失败。YOLO 不可用、无检测或几何失败时，不允许静默回退 Oracle。

## 允许进入生产目标查询的字段

Assignment 只能通过 closed dataclass `TargetQuerySpec` 传入以下五个顶层字段：

```text
target_alias
detector_class_id
detector_class_name
hard_attributes
soft_description
```

- `target_alias` 只用于 Assignment 路由，确认前不会作为 detector identity 泄露真实对象 ID。
- `detector_class_id/name` 必须与 worker model-info 和配置的严格类别映射一致。
- `hard_attributes` 只能是语义字符串，例如 `color=red`、`shape=cube`。
- `soft_description` 只用于语义确认；Qwen 不能由此返回飞行使用的三维坐标、速度或控制量。

schema 是 fail-closed 的；新增 dataclass 字段而未同步白名单会直接报错。

## 生产链允许的数据

| 边界位置 | 允许字段/能力 |
| --- | --- |
| YOLO 请求 | 本 UAV 当前 RGB frame、stream/frame/timestamp routing、允许的 detector class/prompt |
| CandidateBank | detector bbox、confidence、tracker ID、candidate ID、同 candidate 的有限历史 |
| 确认器 | 同步 RGB-D ROI、支持的颜色/语义属性、typed 且经过 routing/新鲜度门的低频 Qwen review |
| 几何 resolver | candidate bbox/history、同步 metric depth、相机内参/位姿、UAV 自身运动、有限 FrameStore |
| `TargetMeasurement` | timestamp、candidate/tracker、sample pixel、raw/corrected depth、camera/world position、3×3 covariance、quality、visual source |
| `TargetEstimate` | target alias（确认后）、visible/confirmed/predicted-only、视觉估计位置/速度、confidence、age、allowlisted source |
| Skill | 只读 `Observation.target_estimate`；SEARCH/TRACK/REACQUIRE 不访问 detector、environment target 或 evaluator |

生产 `TargetEstimate.source` 只接受视觉链 allowlist：`ultralytics_service`、`yolo26_botsort`、`yoloe26_botsort`、`rgbd_depth_geometry`、`temporal_ray_depth`、`kalman_prediction`。`TargetMeasurement.source` 会在有界 candidate transition 中单独记录：deterministic baseline 使用显式 `isaac_depth_<sampling strategy>`，时序成功使用 `temporal_ray_depth`，时序回退使用 `rgbd_depth_geometry_fallback`，无 measurement 时为 `null`。这些 source 都是标量 provenance；transition schema 仍禁止 RGB/depth payload。measurement 通过 Kalman/协调器后，最终 estimate 继续遵守视觉 provenance allowlist。

## 禁止字段与能力

以下信息不得进入 Planner prompt、生产 query、YOLO 请求、CandidateBank、resolver、Kalman 输入、`TargetEstimate` 或 Skill 控制：

- `oracle_target_*`、`ground_truth`、`sim_truth`；
- 目标真实 `position/pose/velocity/speed` 或 future trajectory；
- target `initial_region`、motion region/seed；
- simulator prim path、object/instance/segmentation ID；
- environment target object、evaluator frame getter、Oracle factory；
- 世界坐标/search region 伪装成目标语义属性；
- training label 类型或 `datasets.target_state` 对生产 runtime 的导入。

目标搜索区域属于 `SEARCH Goal` 和航路规划，不属于目标身份查询。生产估计得到的 `position_world_m` 是允许输出的视觉结果，但真实 `target_position_world_m` 绝不允许作为输入；两者不能因为名字相近而混淆。

生产 observation 如果携带 `oracle_evaluation`、`ground_truth`、`sim_truth` 等 source，会在 Agent 使用前立即抛出 boundary error。未知 provenance 也按拒绝处理。

## Oracle 在训练和评测中的位置

Oracle 可以用于：

- 理想能力上界和回归测试；
- 离线数据标注；
- 专家策略/训练 label 生成；
- evaluator-only 的位置和速度 RMSE side channel。

时序训练数据把 `sensor_input`、真实 `detector_prediction` 和 privileged `training_label` 分开。Oracle GT 只存在于 label；生产输入 tensor 由前两部分构造。数据采集需要双 acknowledgement，生产启动则必须 `privileged: false`。

Evaluator 可按 timestamp 对齐视觉 estimate 与 GT 并写出 `position_rmse_m` / `velocity_rmse_mps`，但 `evaluate()` 不向 Agent 返回 target state。候选日志同样只允许 bounded visual metadata，不记录 simulator truth、raw RGB/depth 或视频。完整日志字段与指标语义见 [Production YOLO runtime evidence](perception_runtime_evidence.md)。

## 启动审计

生产启动记录必须至少包含：

```json
{
  "runtime_profile": "production",
  "target_perception_mode": "yolo",
  "backend_by_uav": {"uav_1": "ultralytics_service"},
  "privileged": false,
  "allowed_target_query_fields": [
    "target_alias",
    "detector_class_id",
    "detector_class_name",
    "hard_attributes",
    "soft_description"
  ],
  "yolo_model_sha256": "895de7caa8af200c12f343c72e3a726ffae65e4d96d2092decaf96ef4558de07",
  "temporal_model_sha256": "2a8acd63347a568a4e5588cfc335979709958f7859617405841062af679e023c"
}
```

审计记录只写 schema/后端/模型身份，不写任务 query 内容、目标位置或 motion 配置。
单机生产配置已启用通过 promotion 的 Stage B v2 artifact，因此这里记录其真实
checkpoint SHA；使用 `isaac_depth` baseline 时该字段为 null。

## 边界回归测试

纯 Python 检查可独立运行：

```bash
cd /home/amax/ry/vlm_drones/uav_agent

./python.sh -m pytest -q \
  tests/perception/test_target_query_spec.py \
  tests/perception/test_production_information_boundary.py \
  tests/runtime/test_yolo_no_oracle_fallback.py \
  tests/fleet/test_fleet_request_builder.py \
  tests/fleet/test_fleet_world_belief.py
```

这些测试验证 closed query schema、构造函数能力边界、privileged source 拒绝、无 Oracle fallback，以及 Fleet prompt/world belief 不携带真值。它们不能替代真实 YOLO + Isaac 端到端验收。

## 常见边界错误

| 错误 | 正确处理 |
| --- | --- |
| production 启用了 Oracle acknowledgement | 删除 privileged 开关；若确实做上界评测，则同时显式切换到 `oracle_evaluation` profile/backend。 |
| YOLO worker 失联或 SHA 不符 | 在 Isaac 导入前失败并修复 worker/model；不得切到 Oracle 或 disabled 继续目标任务。 |
| confirmation mode 仍是 `class_track_or_qwen` | 生产 cube 配置改用 `class_track_attribute_or_qwen`，保留时序属性证据。 |
| 颜色 pending | 不得注入场景 appearance/GT 来“帮助”确认；改善 RGB-D 质量或显式启用受门控 Qwen。 |
| 深度采到地面、Kalman 拒绝、target lost | 只使用视觉 debug/metrics 定位 foreground sampling、measurement/covariance 和 prediction age；不得读取 target truth 纠正控制。 |

生产边界的核心原则是 capability separation：即便调用方知道 simulator truth，也没有把它传给生产 runtime 的参数或对象引用。
