# 时序射线深度估计器

`TemporalRayDepthResolver` 是 YOLO candidate 到 `TargetMeasurement` 的可选生产几何后端。它学习对确定性 RGB-D 射线的像素/深度残差和测量协方差，不直接生成 `TargetEstimate`，也不读取目标真值。

## 当前可用状态

代码、训练接口、artifact 预检查和 deterministic fallback 已接线。正式
`target_state_v1` 数据集上的 Stage B v2 已通过 validation/test promotion gate，
`configs/yolo/runtime_yolo26.yaml` 现使用 `geometry.mode: temporal_ray_depth`。
生产 artifact 是：

```text
/home/amax/ry/vlm_drones/outputs/trained_models/target_state/temporal_ray_depth_yolo_deployment_v2/best.pt
SHA256: 2a8acd63347a568a4e5588cfc335979709958f7859617405841062af679e023c
```

`isaac_depth` 仍保留为确定性 baseline 和显式 fallback；未通过 promotion 的
checkpoint 仍会在 Isaac import 前被拒绝。

## `isaac_depth` 与 `temporal_ray_depth`

| 项目 | `isaac_depth` | `temporal_ray_depth` |
| --- | --- | --- |
| 输入 | 当前同步 RGB-D frame、candidate bbox、相机内参/位姿 | 同一 candidate 最近 7 帧（`history_size=6` 加当前帧）的 RGB-D ROI、YOLO confidence/tracker continuity、相机相对位姿、UAV 自身线/角速度与时间差 |
| 深度锚点 | `foreground_cluster_median` | 先计算相同 baseline，再预测 `delta_u/v`、depth residual、有效性和三轴 log variance |
| 输出 | typed `TargetMeasurement` 和传播后的 covariance | typed `TargetMeasurement`，source=`temporal_ray_depth`，使用网络 covariance |
| 状态估计 | 后续统一进入 constant-velocity Kalman | 后续统一进入同一个 Kalman；网络不会替换 Kalman/`TargetEstimate` |
| 失效策略 | measurement unavailable/rejected | 可配置回退到相同 deterministic RGB-D；source=`rgbd_depth_geometry_fallback`，记录原因和计数器 |
| 真值访问 | 无 | 无 |

两种模式都使用相机 optical `x-right/y-down/z-forward`，然后转换到 camera FLU 和 world FLU `x-forward/y-left/z-up`。前景采样会 inset bbox 并排除底部区域，以降低地面深度进入测量的概率。

## 数据集的信息分层

时序数据记录严格拆成三个顶层部分：

```text
sensor_input        # 可部署输入：同步 RGB/depth 路径、相机内参/位姿、UAV 自身状态
detector_prediction # 真实部署 YOLO bbox/confidence/miss/tracker/candidate
training_label      # privileged：目标 GT 位置/速度/中心像素/可见性/遮挡/颜色
```

Oracle 只允许在离线采集时生成 `training_label`，并且必须同时给出 `--oracle-label-generation` 和 `--acknowledge-privileged-oracle`。生产 feature tensor 只由 `sensor_input` 和 `detector_prediction` 构造；label 只进入 loss/evaluator。`perception/` 生产 runtime 不导入 `datasets.target_state` 或 label 类型。

当前 collector 默认直接启动 Isaac，并在任何 `isaacsim` import 之前检查
`127.0.0.1:8011` 的 `/health`、`/v1/model-info`、精确类别 `{0: cube}` 和
模型 SHA。每个 Camera/World barrier 只向 deployed worker 发送一次同步 RGB；
worker 返回 bbox/confidence/tracker 后，privileged GT 才在 collector 内用于写
独立 label。多目标记录共享同一份 RGB-D 文件。无目标或 detector false
positive 使用 `training_label: null`，不会用 `(0,0,0)` 伪造目标位置。

先用 `scripts/serve_yolo.py`（或项目的单机 YOLO 启动脚本）在 8011 启动本次
训练得到的 `best.pt`，再运行：

```bash
cd /home/amax/ry/vlm_drones/uav_agent

./python.sh scripts/collect_target_state_dataset.py \
  --mode isaac \
  --config configs/default.yaml \
  --collection-config configs/yolo/collect_cube.yaml \
  --output /home/amax/ry/vlm_drones/datasets/target_state_v1 \
  --yolo-url http://127.0.0.1:8011 \
  --yolo-model-sha256 895de7caa8af200c12f343c72e3a726ffae65e4d96d2092decaf96ef4558de07 \
  --scene-seed 42 \
  --max-episodes 100 \
  --frames-per-episode 20 \
  --sample-hz 5 \
  --history-size 6 \
  --max-history-age-s 2.0 \
  --oracle-label-generation \
  --acknowledge-privileged-oracle \
  --headless

./python.sh scripts/check_target_state_dataset.py \
  --dataset /home/amax/ry/vlm_drones/datasets/target_state_v1
```

采集场景按 episode 确定性轮换普通 positive、三目标交叉/部分遮挡和 no-target；
实际漏检、bbox 抖动、误检和 tracker ID 切换来自 worker 输出，collector 不合成。
manifest 会报告实际 physical capture、多目标、无目标、crossing、tracker switch、
遮挡和 YOLO miss 统计。`--mode external --source-dataset ...` 仅用于复制取证
spool，其 manifest 固定标为 `external_capture_spool_unverified`，不能用于
`yolo_deployment` stage 冒充真实 detector 数据。

checker 会从 `dataset_manifest.json` 读取 `history_size` 和 `max_history_age_s`（也可用同名 CLI 参数显式覆盖），从 `frames.jsonl` 真实重建序列并校验时间排序、逐帧 `Δt`、missing/target-present mask、tracker ID 切换，以及 UAV/Assignment/candidate 隔离；输出的 `sequence_count` 不是固定占位值。它还会验证 RGB/depth 可解码且同步、可见目标中心位于分辨率内、数据 SHA/统计与实际文件一致，以及 `real_yolo_deployment_output` 必须带精确 worker preflight receipt。dataset split 按 episode 隔离，避免同一轨迹帧跨 train/validation/test 泄露。

早期实现曾用真实 worker + Isaac 完成一份有界小样验收，产物位于
`/home/amax/ry/vlm_drones/datasets/target_state_v1_smoke_20260825_task_e483_fixed`。
checker 实际结果为 136 records、64 次物理采样、24 条合法序列、8 个 episode、
0 error / 0 warning，dataset SHA256 为
`27eed2e8218dd60c7614e956238a3a8d0687c3ec70ac4cb4c6dd138e96539f43`。
manifest 记录 16 个 no-target frame、4 次 crossing、2 次 tracker switch、
20% YOLO miss rate，并带精确部署 receipt。交叉时同一 sensor-only candidate
若短时关联到不同 GT instance，混合实例的滑窗会被丢弃，不会把两条真值轨迹
拼成一个训练样本。该产物只作为集成 smoke 保留，不作为生产晋级依据。

正式数据集位于 `/home/amax/ry/vlm_drones/datasets/target_state_v1`，已完成
100 episode、2,000 次物理采集、3,995 条 frame/target records 和 1,997 条时序
sequence；train/validation/test sequence 数为 1,379/316/302，split 按 episode
隔离。数据集包含 660 次 no-target capture、1,100 次 multi-target capture、
53 次 crossing、13 次 tracker ID switch，实际 YOLO miss rate 为 0.4861；
checker 返回 0 error / 0 warning，dataset SHA256 为
`74ef77b93fe0dc23b91a4476c125943eb615667a64d3a6c4d6f3c96feddb021b`。

## 两阶段训练

Stage A 使用 Oracle/较干净 bbox 学习基础射线几何和残差：

```bash
./python.sh scripts/train_target_state.py \
  --config configs/target_state/train_oracle_clean.yaml \
  --dataset-root /home/amax/ry/vlm_drones/datasets/target_state_v1 \
  --device cuda:0
```

Stage B 必须使用真实 YOLO bbox、confidence、漏检和 tracker 状态，并从 Stage A 的 `best.pt` 初始化：

```bash
./python.sh scripts/train_target_state.py \
  --config configs/target_state/train_yolo_deployment.yaml \
  --dataset-root /home/amax/ry/vlm_drones/datasets/target_state_v1 \
  --initial-checkpoint /absolute/path/to/stage_a/best.pt \
  --device cuda:0
```

早期小样训练的 Stage B 因 validation/test 覆盖不足而
`promotion.passed=false`，该失败 artifact 仍保留用于审计。随后在正式数据集上
完成生产训练：

- Stage A：30 epochs，`best.pt` SHA256
  `7d80400186954f8a4110d338123440a05c8e85820daef02542b13bdc48e19c76`；
- Stage B v2：50 epochs，精确从上述 Stage A checkpoint 初始化，`best.pt`
  SHA256 `2a8acd63347a568a4e5588cfc335979709958f7859617405841062af679e023c`；
- 两个 run 都产生了 `best.pt`、`latest.pt`、`metrics.csv`、TensorBoard event 和
  `model_manifest.json`，数据集 SHA 为上述正式数据集 SHA。

Stage B v2 的 validation/test median 位置误差为 0.2519/0.2700 m，优于
deterministic baseline 的 0.5819/0.5877 m；p95 为 24.0122/26.7858 m，未超过
baseline 的 1.05 倍；measurement failure 与 no-target false-positive 均未恶化，
遮挡和 bbox jitter 子集均改善，covariance/error Spearman 为 0.7813/0.6913，
非法输出均为 0。manifest 的 validation/test receipt 和总
`promotion.passed` 均为 true。

评测器会无条件拒绝任一 network head 的 NaN/Inf；只有模型声明 measurement 有效
时，才要求派生的 corrected depth/world position 有限且在范围内。模型主动拒绝的
缺深度样本记入 measurement failure，而不会被重复误记为数值非法；这不改变任何
median、p95、failure、遮挡、jitter 或 covariance 门槛。

训练包含 depth Huber、3D position Huber、多帧 reprojection Huber、Gaussian NLL 和 measurement-validity BCE，并对漏检、遮挡和无效深度使用 mask。每次 run 只保存 `best.pt`、`latest.pt`、CSV、TensorBoard scalars、manifest 和少量结果图，不保存训练视频或全部预测图。

## 模型 manifest 和晋级

每个训练 run 生成 `model_manifest.json`，生产加载器严格检查：

- `model_type=temporal_ray_depth_residual` 和 `schema_version=1`；
- checkpoint 绝对路径及内容 SHA256；
- dataset SHA 和训练 commit 标识；
- 精确的 `input_fields`、`output_fields` 和 `model_config`；
- `history_size`、`max_history_age_s`、ROI/preprocessing；
- camera/coordinate conventions；
- validation/test metrics 和 promotion 结果。

生产预检查还会拒绝 manifest 路径与实际 checkpoint 不一致、Stage A/Stage B 协议未满足、未使用已验证真实 YOLO 数据集、validation/test promotion receipt 未通过，以及没有真实 no-target 样本或 no-target false-positive rate 高于 deterministic baseline 的 artifact。checkpoint 自身的 `model_type`/schema 也必须与 manifest 一致。

默认输入是 RGB + `depth/maximum_depth_m` 的 128×128 ROI、25 维 geometry 和 missing mask。tracker continuity 是 geometry 第 6 维（与上一非 missing tracker ID 比较），不是一个伪造的独立 `tracker_change_mask` 模型输入。25D 中的 UAV 线速度明确采用 world frame，角速度采用 body frame；生产值由与 CameraSample 同时刻的 Agent `Observation` 提供，而不是用相机位姿差分冒充。当前 canonical `UAVState` 是 yaw-only 姿态，因此 bridge 对首帧角速度置零、后续用 wrap-safe UAV yaw 差分计算 body-z yaw rate，body x/y 为该 yaw-only 控制抽象的零值。默认输出字段是：

```text
delta_u_px
delta_v_px
depth_residual_m
position_log_variance_xyz
measurement_valid_logit
```

只有 stage B 同时在 validation 和 test 上通过以下 gate 才能晋级：median 3D error 优于 deterministic baseline、p95 未显著恶化、failure rate 不高于 baseline、no-target false-positive rate 不高于 baseline、遮挡和 bbox jitter 子集有改善、uncertainty 与实际误差相关、无 NaN/Inf/越界输出。validation/test 都必须实际含有 no-target 样本，不能用空子集绕过该 gate。

## 生产配置与预检查

当前单机生产配置使用的 geometry 块等价于：

```yaml
target_perception:
  geometry:
    mode: temporal_ray_depth
    depth_anchor: foreground_cluster_median
    depth_patch_radius_px: 4
    min_depth_m: 0.2
    max_depth_m: 200.0
    max_measurement_age_s: 0.5
    temporal_ray_depth:
      checkpoint_path: /home/amax/ry/vlm_drones/outputs/trained_models/target_state/temporal_ray_depth_yolo_deployment_v2/best.pt
      expected_sha256: 2a8acd63347a568a4e5588cfc335979709958f7859617405841062af679e023c
      manifest_path: /home/amax/ry/vlm_drones/outputs/trained_models/target_state/temporal_ray_depth_yolo_deployment_v2/model_manifest.json
      history_size: 6
      max_history_age_s: 2.0
      roi_size_px: 128
      use_rgb: true
      use_depth: true
      deterministic_fallback: true
      device: cpu
```

若省略 `manifest_path`，加载器只会查找 checkpoint 同目录的 `model_manifest.json`；`device` 省略时为 `cpu`。`run_fleet_mission.py` 与旧单机入口 `run_dynamic_visual_mission.py` 都会在首次 Isaac import 前验证 checkpoint 存在、配置 SHA、manifest 内路径/SHA、checkpoint payload schema、精确 input/output/input-semantics/preprocessing、history/ROI/conventions、Stage-B promotion receipt，并在所选 device 上执行一次 shape/finite dry-run。两条入口和 Coordinator 共用同一个 resolver factory，temporal 模式不会静默退化为以 `isaac_depth` 作为主 resolver。`history_size` 必须在 4～8；RGB/depth 至少启用一个；时序模式的 deterministic baseline 必须保持 `foreground_cluster_median` 和 seed patch radius 4，防止训练/生产预处理漂移。

resolver 从已有有界 `FrameStore` 原子取得同一 frame generation 的 RGB、depth、相机几何和 UAV self-motion，不创建第二套无限缓存。使用前必须绑定非空 Assignment；每次 reset 会清空前一/当前 UAV 的 store 历史。bbox/tracker 检测历史按 candidate 构造，跨 candidate 不复用检测，跨 UAV、未 reset、字段未对齐或过期输入都会被拒绝。网络失败、输出无效或在当前新鲜 frame 下历史不足时：

- `deterministic_fallback: true`：返回经过验证的 deterministic measurement，source=`rgbd_depth_geometry_fallback`，并累计 fallback 原因；
- `deterministic_fallback: false`：显式 unavailable，绝不使用随机输出；
- 两种情况都绝不读取 Oracle 或目标真实位置。

过期 candidate 不属于允许 fallback 的“网络不可用”，即使 deterministic depth 仍在内存中也会直接拒绝。运行指标包括 `temporal_resolution_attempts/successes`、`temporal_fallback_total`、`temporal_unavailable_total`、`temporal_invalid_output_total`、有界的 fallback/unavailable reason counts 和各自 last reason。候选生命周期 JSONL 另写严格 `measurement_source`：时序成功为 `temporal_ray_depth`，时序回退为 `rgbd_depth_geometry_fallback`，无 measurement 时为 `null`；不写 RGB/depth 数组或真值。

## Kalman、SEARCH 与丢失目标

网络只产生 `TargetMeasurement`；其三轴 covariance 经过正值裁剪后交给 Kalman。Kalman 接受后才形成视觉 `TargetEstimate`，短时遮挡可形成 source=`kalman_prediction` 的 predicted-only estimate。

`SEARCH` 首次成功必须是 visible + confirmed + measured，predicted-only 不可触发 `TARGET_FOUND`。`TRACK` 可在配置的短窗口使用 prediction；超过 `max_prediction_age_s` 后必须进入 target-lost，再按计划执行 REACQUIRE 或安全失败。

排障时重点查看：

- 深度采到地面：检查 foreground cluster debug overlay、bbox 底部排除、相机同步和 raw/corrected depth；
- Kalman 拒绝：检查 covariance、时间戳、10 m jump gate、Mahalanobis innovation，不要用放宽 gate 掩盖错误 measurement；
- target lost：检查 temporal fallback reasons、`measurement_rejected`、`predicted_only_outputs`、`track_predicted_updates` 和 `target_lost_count`。

这些计数器的精确定义见 [Production YOLO runtime evidence](perception_runtime_evidence.md)。

## 真实生产 E2E 证据

已使用 `scripts/run_single_uav_yolo_e2e.sh` 完成真实 YOLO worker + Isaac Sim +
temporal resolver 任务，run 位于：

```text
logs/fleet_missions/runs/fleet_mission/20260825-194528_fleet_mission_seed0_nogit
```

运行结果为 `strict_success=true`：SEARCH 找到目标、TRACK 有效跟踪
20.000001 s、返回起点并 LAND。时序统计为 282 次 resolution attempt、267 次
`temporal_ray_depth` success、15 次有原因的 deterministic fallback、0 次非法输出；
fallback 中 14 次是历史不足、1 次是模型有效性拒绝。Kalman 接受 201 次测量，
SEARCH `TARGET_FOUND=1`，TRACK visible update=200。该 run 使用 production profile、
`ultralytics_service`、`privileged=false`，并在 Isaac import 前核验 YOLO 与 temporal
checkpoint 的两个 SHA。

随后用同一最终配置完成了固定 seed 101、211、307、401、503 的五次真实
YOLO + Isaac 评估，聚合文件为：

```text
logs/yolo_fixed_seed_eval/task9_temporal_5seeds/summary.json
```

五次均为 `strict_success=true` 且进程 exit code 0；detection、deterministic
颜色确认、3D measurement、SEARCH、20 s TRACK、return/LAND 的阶段成功率均为
1.0。累计 1297 次 detector request、1292 次成功 response、1177 次深度解析尝试、
1176 次成功 measurement，聚合报告保留 YOLO SHA 且
`oracle_metrics_used=false`。这是固定五场景的运行证据，不替代更大规模泛化评测。
