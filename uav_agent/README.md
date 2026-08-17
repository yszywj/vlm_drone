# UAV Agent

基于 VLM Agent 的层次化无人机目标搜索与跟踪项目。Stage 0 已具备第一版 Isaac Sim 场景、运动学 UAV、固定 RGB Camera、有界移动 Target，以及 TAKEOFF / GOTO / SEARCH / TRACK / REACQUIRE / LAND 的完整 Oracle 任务流水线。Stage 1 已把 `MissionAgent` 接入该流水线，支持确定性 Scripted Planner 与文本 Qwen Planner 两种高层规划入口；当前视觉搜索、跟踪和重捕获仍由 Oracle 真值驱动，并不代表已经实现真实图片识别。

## 快速开始

仓库提供 `environment.yml` 作为 Isaac 运行环境的可复现基线。新克隆首次安装（Isaac wheels 体积较大）：

```bash
cd /path/to/vlm_drones/uav_agent
conda env create -f environment.yml
conda activate r_isaac_sim
export UAV_AGENT_CONDA_ENV="${CONDA_PREFIX}"
./python.sh scripts/run_demo.py --config configs/default.yaml --validate-only
```

`python.sh` 默认仍使用这台服务器的 `/home/amax/miniconda3/envs/r_isaac_sim`，但不再写死 Conda 可执行文件。其他机器应通过 `UAV_AGENT_CONDA_ENV=/absolute/env/prefix` 指定环境；当 `conda` 不在 `PATH` 中时，脚本会尝试由 `<prefix>/../..` 定位 Conda，也可显式设置 `UAV_AGENT_CONDA_BIN=/absolute/path/to/conda`。脚本始终用 `conda run -p` 运行，并把项目目录及其父目录加入 `PYTHONPATH`，因此历史顶层导入与 `python -m uav_agent...` 两种入口均可使用。

```bash
cd /home/amax/ry/vlm_drones/uav_agent
./python.sh scripts/run_demo.py --config configs/default.yaml
```

运行完整 Stage-0 Oracle 任务（默认 RANDOM_WALK Target，跟踪 30 s 后降落）：

```bash
./python.sh scripts/run_oracle_pipeline.py --config configs/default.yaml \
  --start-altitude 0 --takeoff-altitude 10 --track-duration 30
```

运行经 `MissionAgent` 调度的 Stage 1A Scripted Planner + Oracle 任务：

```bash
CUDA_VISIBLE_DEVICES=0 \
./python.sh scripts/run_llm_oracle_pipeline.py \
  --config configs/default.yaml \
  --planner scripted \
  --instruction "前往 search_area 搜寻移动目标，找到后跟踪十秒，然后返回 home 降落" \
  --takeoff-altitude 10 \
  --track-duration 10 \
  --start-altitude 0 \
  --max-sim-time 300 \
  --headless
```

常用命令：

```bash
# 仅校验配置，不导入或启动 Isaac Sim
./python.sh scripts/run_demo.py --config configs/default.yaml --validate-only

# GUI 中查看 Ground、UAV、CameraHousing、Target 和障碍物
./python.sh scripts/run_demo.py --config configs/default.yaml --steps 1200 --no-headless

# Phase 2 验收：连续飞向 (10, 5, 8)，约 134 个 60 Hz step 后到达附近
./python.sh scripts/run_demo.py --config configs/default.yaml \
  --uav-goal 10 5 8 --steps 180

# 保存最终机载 RGB 图像
./python.sh scripts/run_demo.py --config configs/default.yaml \
  --steps 180 --save-rgb logs/final_rgb.png

# Camera/Target 调试：同一 Camera tick 的 privileged truth 与 frustum 投影
./python.sh scripts/run_demo.py --config configs/default.yaml \
  --steps 300 --debug-ground-truth --save-rgb logs/debug_rgb.png
```

除 `SimulationApp` 本身外，所有 `isaacsim.core`、`omni`、`carb`、`pxr` 相关导入都发生在应用创建之后。`scripts/run_demo.py`、`scripts/run_oracle_pipeline.py` 和 `scripts/run_llm_oracle_pipeline.py` 都采用 standalone 启动顺序，并保证环境和 `SimulationApp` 最终关闭。

## 环境版本、GPU 与模型权重

`environment.yml` 固定 Python 3.11、Isaac Sim 5.1.0.0、NumPy 1.26.0、PyYAML 6.0.2 和 Pillow 11.3.0。当前服务器 `r_isaac_sim` 环境实际读取到的版本为 Python 3.11.15 和 Isaac Sim 5.1.0.0，与该基线一致。Isaac Sim 依赖 NVIDIA 的 Python package index，安装及首次启动仍受 NVIDIA 许可条款约束。

Conda 文件不安装或固定主机 NVIDIA driver，也不能替代 [Isaac Sim 5.1 官方系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)。运行前应在主机上用 `nvidia-smi` 确认 GPU 和 driver 正常，再依官方 5.1 兼容矩阵核对 driver/CUDA 支持；`CUDA_VISIBLE_DEVICES` 只选择运行 GPU，不会安装 driver 或 CUDA。本次受控 shell 无法通过 `nvidia-smi` 读取主机 driver，因此仓库不声称已验证某个具体 driver 版本。

Qwen 服务端必须使用另一个 Conda 环境和独立进程；不要把 vLLM 或 Transformers 安装进 `r_isaac_sim`。Qwen3-VL 上游当前给出的兼容下限是 `transformers>=4.57.0` 与 `vllm>=0.11.0`（离线多模态工具另建议 `qwen-vl-utils==0.0.14`）；这些是服务环境的最低版本，不是本项目已经验收的精确 lock。当前 Isaac 环境中 vLLM/Transformers 均未安装，本地模型 `config.json` 的 `transformers_version=4.57.0.dev0` 也只是模型元数据。首次真实服务验收后，应在独立服务环境中导出实际 lock，而不是把它并入 Isaac 环境。

模型权重不进入 Git。在独立的 Qwen 服务环境中安装 Hugging Face Hub CLI 并下载，再把得到的本地路径交给 `QWEN_MODEL_PATH`：

```bash
cd /path/to/vlm_drones
hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir models/initial_model/Qwen3-VL-4B-Instruct
export QWEN_MODEL_PATH="$PWD/models/initial_model/Qwen3-VL-4B-Instruct"
```

根目录 `/models/` 按重量级权重处理并由 `.gitignore` 忽略；`uav_agent/models/` 是 OpenAI-compatible 客户端源码，必须跟踪。公共的 `configs/__init__.py`、`configs/schema.py`、`configs/loader.py` 和 `configs/default.yaml` 同样必须跟踪；只有 `configs/local.yaml` 和 `configs/private/` 被忽略。

## Stage 1A / 1B：MissionAgent + Oracle

两种模式共用相同的可信编译、安全检查、目标生命周期和底层 Skill 调度链：

```text
Instruction
    ↓
MissionPlanner
    ├─ legacy: MissionIntent                 # 固定模板 baseline
    └─ dynamic: SkillPlanDraft               # 有界线性 Skill 选择
    ↓
PlanValidator                  # 可信 WorldContext → 通用 TaskPlan
    ↓
MissionAgent                   # 每个新 Camera sample 最多 tick 一次
    ↓
SkillManager
    ↓
Oracle-backed Ideal Skills     # SEARCH / TRACK / REACQUIRE 使用 evaluator truth
```

旧的 `scripted` / `llm` 模式保持不变：Planner 输出严格五字段 `MissionIntent`，可信编译器生成固定六步 baseline。新增的 `dynamic_scripted` / `dynamic_llm` 显式启用受约束动态规划：Planner 输出 2～10 步的 `SkillPlanDraft`，可以按指令省略 SEARCH/TRACK、有限重复 GOTO/TRACK，并为 TRACK 选择是否启用有界 REACQUIRE。两条路径都只在 `MissionAgent.start()` 规划一次；仿真 tick 不调用模型，也没有运行时 LLM replanning。

动态模式不会把飞控交给 Qwen。模型只看到场景总边界、默认高度/时长、Skill Catalog、调用上限、具名区域/降落区/导航点及文字描述，并只能填写高层参数；它看不到这些名称背后的具体坐标、搜索中心/半径、目标 spawn/真值、图像、速度向量、实际 max speed 或普通 Skill timeout。`PlanValidator` 才负责把名称解析为可信世界坐标、补充 motion policy/timeout/速度上限、检查引用与计划状态，并拒绝缺失 TAKEOFF/LAND、越界高度或非法顺序，而不是静默补步骤。

WorldContext 的 `search_area` 只从配置中的 `target.initial_region`、`search.radius_m` 和任务起飞高度构造，`home` 只取配置的 UAV 初始 XY 与地面高度。它不包含本次随机 Target spawn 坐标、Target 速度、`EvaluatorFrame` 或 Oracle Observation。`--debug-ground-truth` 仅允许把真值打印给人工调试/evaluator，不会把它加入 instruction、Planner prompt 或 GOTO 目标。

脚本保持 Isaac standalone 导入顺序：先解析参数、加载纯 Python 配置并创建 Planner 配置，随后创建 `SimulationApp`，最后才导入 Isaac-backed 环境模块。默认 `--headless` 适合服务器验收；需要 GUI 时使用 `--no-headless`。只有 `environment.step()` 产生新 Camera sample 后才构造 Oracle Observation 并推进 Agent。

`--max-sim-time` 是外部任务时限。到期后脚本调用 `MissionAgent.cancel()`，继续推进仿真以完成 fail-safe LAND，而不是立刻把进程结束在空中；降落另有有限 shutdown guard，超时会明确返回失败。`KeyboardInterrupt` 也会尽力触发相同的取消和安全降落流程。成功验收同时检查 Agent/Task 状态、最终 `LAND_COMPLETE`、home XY 误差和地面高度误差。

Stage 1B 必须让 Qwen 服务和 Isaac Sim 在独立进程运行，推荐双 GPU 分工：GPU 0 运行 Isaac Sim，GPU 1 运行 Qwen3-VL。先按下一节启动并检查 Qwen 服务，再运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
QWEN_API_BASE=http://127.0.0.1:8000/v1 \
QWEN_API_KEY=EMPTY \
QWEN_MODEL=Qwen3-VL-4B-Instruct \
./python.sh scripts/run_llm_oracle_pipeline.py \
  --config configs/default.yaml \
  --planner llm \
  --instruction "起飞到十米，前往 search_area 搜寻移动目标，找到以后持续跟踪三十秒，最后返回 home 降落" \
  --headless
```

将上例的 `--planner` 改为 `dynamic_llm` 即可由 Qwen 选择动态 Skill 序列；离线验证动态执行器可使用 `dynamic_scripted`。纯 Planner 演示会分别打印模型选择的 `SkillPlanDraft`、可信编译后的 `TaskPlan` 和 compiler notes：

```bash
./python.sh scripts/run_planner_demo.py \
  --planner dynamic_llm \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-4B-Instruct \
  --api-key EMPTY \
  --instruction "起飞到十米，前往 search_area 搜寻移动目标，找到后跟踪十秒，返回 home 降落"
```

当前能力边界必须明确：Qwen3-VL 在此阶段只接收文本，不接收 Camera 图像；SEARCH / TRACK / REACQUIRE 仍消费带 `oracle_` 前缀的真值字段；LOCK 只是 `TargetManager` 的逻辑生命周期状态。真实图片 detector、视觉 tracker、VLM 语义验证和 ReID 尚未实现，`DetectorTrackerPerception`、`VLMVerifier` 与 `ReIDVerifier` 调用时仍明确抛出 `NotImplementedError`，绝不返回占位命中。因此 Stage 1A/1B 都是 Oracle 集成里程碑，不能称为真实视觉搜索闭环。

感知运行时默认使用 `PerceptionRuntimeProfile.PRODUCTION`。`GuardedPerceptionBackend` 会拒绝声明为 `PRIVILEGED_ORACLE` 的 backend，也会二次检查任何伪装成视觉 backend 却输出 `oracle_target_*` 的 Observation；`MissionAgent` 在 Safety 和 Skill 之前还有同样的 production gate。Oracle 只能在明确选择 `ORACLE_EVALUATION` 并设置 `acknowledge_privileged_oracle=True` 后运行，两个 Oracle demo 会在控制台打印醒目标记。该 profile 仅用于上界、回归测试、数据标注和专家轨迹，不是部署配置，也不能与真实视觉配置静默互换。

真实视觉候选确认的纯 Python 边界已定义，但 backend 尚未实现：

```text
Detector proposal
    ↓
TargetManager.SEARCHING → CANDIDATE
    ↓
stable short-track evidence
    ↓
VLM semantic match
    ↓
ReID + temporal identity consistency
    ↓
TargetManager.LOCKED → TRACKING
```

`CandidateConfirmationCoordinator` 只接受带有限 timestamp、confidence、一致 candidate id、合法时间顺序和可实现轨迹时长的类型化 evidence；任一明确否定会清除 SEARCH 候选并回到 `SEARCHING`，REACQUIRE 候选被否定时则恢复原 target id 与 last-seen 状态并回到 `REACQUIRING`。证据不足时保持 `CANDIDATE`，短轨迹、语义、ReID 和时序一致性全部通过才写入非 Oracle 的 `confirmed_vision` lock。裸 `TargetManager.lock()` / `mark_reacquired()` 会拒绝直接调用，避免绕过 coordinator。生产模式下，`MissionAgent` 的 SEARCH/REACQUIRE→TRACK 转换要求这个 lock 已存在且 target id 匹配；它不会再把 Skill 成功结果直接伪造成 `confidence=1.0` 的 Oracle lock。当前 Oracle profile 只保留名称明确的 evaluator shortcut，以便已有 Ideal Skills 回归。

## 本地 Qwen OpenAI-compatible 服务

`uav_agent/models/` 是纯 Python 客户端包，与仓库根目录存放权重的 `models/` 不同。客户端只使用标准库 `urllib`，默认访问 `http://127.0.0.1:8000/v1` 的 `GET /models` 与 `POST /chat/completions`；Stage 1B 仅通过 `LLMPlanner` 发送文本消息，不发送 Camera、Oracle 或完整环境对象。可通过 `QWEN_API_BASE / QWEN_API_KEY / QWEN_MODEL / QWEN_REQUEST_TIMEOUT_S` 设置默认值，显式构造参数优先。

服务脚本不会安装 vLLM，也不会修改 `r_isaac_sim`。请先进入服务器上已有的兼容 vLLM 环境，或把 `VLLM_BIN` 指向该环境中的可执行文件，再运行：

```bash
cd /home/amax/ry/vlm_drones/uav_agent
QWEN_CUDA_VISIBLE_DEVICES=1 \
QWEN_MODEL_PATH=/home/amax/ry/vlm_drones/models/initial_model/Qwen3-VL-4B-Instruct \
./scripts/serve_qwen3_vl.sh
```

默认只绑定 `127.0.0.1:8000`。模型路径、served name、host、port、最大上下文、GPU memory utilization、CUDA device 和 vLLM binary 都可用清单中对应的 `QWEN_*` / `VLLM_BIN` 环境变量覆盖。另一个终端执行：

```bash
./python.sh scripts/check_qwen_server.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-4B-Instruct
```

检查器先调用 models endpoint，再发送一个最小文本 completion；普通失败只显示连接、HTTP 或协议错误类型，`--debug` 才附带已脱敏 traceback。

## 场景与稳定 prim 路径

场景完全由本地 primitive 创建，不依赖在线 USD 资产：

- 白色平坦 Ground、DomeLight 和斜向 DistantLight；
- 三个带碰撞的彩色固定 Cube 障碍物；
- 矩形机身、十字机臂、电机块和 Nose 拼成的 UAV 简模，不使用四旋翼 USD；
- 固定在 UAV 根节点上的 RGB Camera，以及 GUI 中可见的 CameraHousing；
- 红色 Cube Target，后续可替换成 Person USD。

```text
/World
├── Ground
├── Lights
├── Obstacles
├── UAV                         # 运动学 pose root Xform
│   ├── Body / ArmX / ArmY
│   ├── Motor* / Nose
│   ├── CameraHousing           # 可渲染调试外壳
│   └── Camera                  # 实际 RGB sensor
└── Target                      # 移动目标 pose root Xform
    └── Body                    # 第一版 VisualCuboid
```

Camera prim 本身不是可渲染几何体；overview 中看到的青色物体是 CameraHousing。Camera 和 Housing 都是 `/World/UAV` 的 child，因此 UAV 平移或改变 yaw 时会同步移动。替换 Person USD 时保留 `/World/Target`，只替换其 `Body` 视觉子树。

## World frame、yaw 与单位

- 右手坐标系、Z-up；所有位置和尺寸统一为 meter，速度为 m/s，时间为 s。
- `+X` 是 `yaw=0` 时 UAV Nose 指向的前方，`+Y` 是左方，`+Z` 是上方。
- 正 yaw 按右手定则绕 `+Z`，从上方看由 `+X` 朝 `+Y` 逆时针旋转。
- 控制器内部 yaw 使用 radian；YAML 的 yaw rate、camera pitch 和 target initial heading 使用 degree。
- 所有姿态四元数采用 scalar-first `[w, x, y, z]`。
- `simulation.stage_units_in_meters` 必须为 `1.0`，保证所有 `_m` 字段可直接对应 stage unit。
- scene 的 x/y 范围以世界原点为中心，z 范围为 `[0, size_z]`；Target region 是闭区间 AABB。

## Phase 2：Kinematic UAV

`env/uav_controller.py` 定义 simulator-independent 的 `UAVState` 与 `UAVController` Protocol；`env/kinematic_uav.py` 只是其中一个不导入 Isaac 的实现，并兼容 re-export `UAVState`。PX4、Pegasus、MAVSDK、ROS 2 或真实飞控适配器可实现同一 world-frame contract，不需要继承 `KinematicUAV`。当前 ideal 实现不模拟电机、thrust、roll/pitch dynamics 或 aerodynamic forces，也暂不处理碰撞与避障，因此可能穿过障碍物。

`set_pose()` 只用于初始化、reset 和 debug。正常导航必须先下达命令，再逐 step 积分：

```python
environment.move_uav_toward([10.0, 5.0, 8.0])
while not environment.goal_reached():
    environment.step()

distance_m = environment.distance_to_goal()
heading_error_rad = environment.heading_error()
```

每个 simulation step 都执行 `position(t+1) = position(t) + velocity * dt`。三维速度向量范数不超过 `uav.max_speed_mps`，yaw 每步变化不超过 `uav.max_yaw_rate_deg_s * dt`，并在接近目标时缩短最后一步防止 overshoot。可用接口还包括 `set_uav_velocity()`、`rotate_uav_yaw()` 和 `stop_uav()`；底层完整接口位于 `KinematicUAV`。

## Phase 3：RGB Camera

`env/camera_sensor.py` 管理一个固定安装、朝前下方倾斜的 RGB Camera：

```python
while not environment.step():
    pass  # step() 返回 True 表示新的 Camera sample 已到达
observation = environment.get_agent_observation()
rgb = observation.rgb                       # [height, width, 3]
saved = environment.save_rgb("logs/frame") # 自动补 .png
camera_position = observation.camera_position_m
camera_orientation = observation.camera_orientation_wxyz

# debug/evaluation：Target 真值和投影都缓存于同一 Camera tick
evaluator_frame = environment.get_evaluator_frame()
projection = evaluator_frame.target_projection
uv = projection.pixels_uv[0]
in_frustum = bool(projection.visible[0])
```

Camera 配置包含 `[width, height]` resolution、采样 frequency、水平 FOV、可选 focal length 和相对 UAV body 的 pitch。负 pitch 表示向下。`focal_length_m: null` 时由水平 FOV 推导；设置正值时 focal length 覆盖 FOV 推导结果。默认 Camera 10 Hz、physics/rendering 60 Hz；首帧通常需要多个 physics step。`world_to_image().visible` 只表示点位于 clip range 和 image frustum 内，不判断是否被障碍物遮挡。

环境固定采用以下 step 顺序：更新 UAV/Target 运动学状态并写入 Xform，随后 `World.step(render=True)`。只有新 RGB 到达时，才把 RGB、UAV state、Camera pose 和 timestamp 一次性缓存为 `AgentObservation`；非采样 tick 不会把旧图像和当前 pose 混在一起。`get_camera_pose()` 仍可用于读取当前调试 pose，而同步观测应读取 `AgentObservation.camera_position_m`。CLI 使用 `--save-rgb` 或 `--debug-ground-truth` 时，可能在 `--steps` 之后继续推进最多一个 Camera 采样周期，以获得完整同帧数据；最终会打印实际 step 数。

## Phase 4：移动 Target

`env/moving_target.py` 同样是纯数学模块，支持：

- `STATIC`：保持位置；
- `LINEAR`：以固定速度直线运动，在边界反射；
- `RANDOM_WALK`：每隔配置时间随机改变 XY 速度方向，并在边界反射。

Target root 只在 `target.motion.region` 闭区间内运动，速度受 `target.max_speed_mps` 限制。随机序列使用配置 seed，可重复复现实验；默认测试覆盖连续五分钟的 60 Hz 随机运动且不越界。第一版不做 obstacle avoidance/collision response，Target 可能穿过障碍物；region 约束的是 root，不是视觉几何边缘。底层提供 `get_pose()`、`get_velocity()`、`reset()` 和 `step()`。

环境内部知道 Target 真值以便仿真和评估，但不会把移动目标控制器交给 Planner/VLM。默认 Agent 入口：

```python
agent_view = environment.get_agent_view()
observation = agent_view.observe()
# 仅含 rgb、uav_state、uav_velocity_mps、camera pose、camera_timestamp_s
```

早期低层控制接口可通过 `AgentView` 读取非真值观测；当前高层 `MissionPlanner` 只接收纯 Python 的 `PlannerRequest + PlannerWorldContext`，Skill 则只接收下面定义的 `SkillContext`。这些接口都不得接收完整 environment。`read_poses()`、`target_position`、`target_orientation`、`get_evaluator_frame()` 和 `world_to_image()` 属于 simulator/evaluator 的 privileged API；CLI 也只有显式传入 `--debug-ground-truth` 才输出 Target 真值。

## Phase 5：统一 Skill API 与 Motion Policy

`skills/` 已提供 Qwen/VLM 工具调用所需的纯 Python 合同：

- `SkillStatus` 只描述 `IDLE / RUNNING / SUCCEEDED / FAILED / CANCELED` 生命周期；
- `SkillResultCode` 描述 `TARGET_FOUND`、`GOAL_REACHED`、`TIMEOUT` 等业务结果；
- TAKEOFF、GOTO、SEARCH、TRACK、REACQUIRE、LAND 各自使用独立 Goal dataclass；
- `SkillFeedback` 和 `SkillResult` 提供稳定的结构化返回，不用通用 Goal dict；
- `SkillManager` 同时只推进一个注册 Skill；手动模式不做隐式切换，显式 `start_task()` 后才启用 Stage-0 自动任务状态机；
- 生命周期完成后必须显式 `reset_active()`，才能启动下一项。

Phase 5 最初建立的 TAKEOFF、GOTO、SEARCH、TRACK、REACQUIRE 与 LAND 六个类型合同现在均已有 ideal-kinematic 实现。`HOVER / ORBIT / FOLLOW_PATH / RETURN_HOME` 尚未加入。

MotionPolicy 支持：

- `COURSE_ALIGNED`：yaw 跟随水平运动方向；
- `KEEP_CURRENT`：保存 Skill `start()` 时的 yaw；
- `FIXED`：保持给定 world yaw，缺少 `yaw_value` 会得到 `FAILED + INVALID_GOAL`；
- `FACE_POINT`：持续朝向 world point，缺少 `look_at_point` 同样得到 `INVALID_GOAL`。

`max_speed` 单位为 m/s，`max_yaw_rate` 和 `yaw_value` 使用 rad/s 与 rad。MotionPolicy 固定采用“先下发 world-frame translation，再独立下发 yaw target”，因此 UAV 可以侧飞，不是汽车运动学。
`FACE_POINT` 的 world point 保存在运动控制器中，每个 kinematic step 都会按 UAV 新位置重算 yaw；不是只在命令下发瞬间计算一次。

终态 ResultCode 与状态保持一致：完成类代码（`TAKEOFF_COMPLETE / LAND_COMPLETE / GOAL_REACHED / TARGET_FOUND / TRACK_COMPLETE`）对应 `SUCCEEDED`；`TARGET_LOST / SEARCH_EXHAUSTED / TIMEOUT / INVALID_* / INTERNAL_ERROR` 对应 `FAILED`；`CANCELED` 只对应 `CANCELED`。手动模式下终态后必须显式 `reset_active()`；任务模式由 Manager 在保存 Result 后执行 reset，再启动后继 Skill。

```python
context = environment.make_skill_context(simulation_clock, perception=perception)
manager = SkillManager(context)

# 具体行为阶段完成 register(...) 和 start(...) 后，runtime 每当 step()
# 返回新 Camera sample 时构造统一输入
observation = environment.get_skill_observation()
status = manager.tick(observation)
```

普通运行时只能在 `environment.step()` 返回新 Camera sample 时调用一次 `manager.tick()`；新 Skill 可以在该 tick 的转换阶段启动，但不会重复消费触发上一 Skill 终态的旧图像。

`SkillResult.to_dict()` 会把枚举转换为 Qwen 工具层使用的 `"SUCCEEDED"`、`"GOAL_REACHED"` 等字符串；各 Skill 写入的 `data` 仍应保持 JSON-compatible。

普通 `Observation` 的所有 `oracle_target_*` 均为 `None`。只有 evaluator/test 显式调用 `get_skill_observation(include_oracle=True)` 才会注入 Target 真值；其中当前 `oracle_target_visible` 表示几何 frustum 内，不包含遮挡判断。`SkillContext` 只含结构化 `UAVController`、`CameraSensor`、perception 和 simulation clock，不含具体 `KinematicUAV` 类型、scene、target 或全局 Manager。

## Phase 6A：Ideal Kinematic TAKEOFF

`TakeoffSkill` 已实现为可直接注册的具体 Skill：

```python
manager.register(SkillName.TAKEOFF, TakeoffSkill())
manager.start(
    SkillName.TAKEOFF,
    TakeoffGoal(
        target_altitude=10.0,
        tolerance=0.2,
        climb_speed=1.0,
        yaw_mode=YawMode.KEEP_CURRENT,
        timeout=20.0,
    ),
)
```

`target_altitude` 是 world-frame 绝对 Z，单位 m；`climb_speed` 为 m/s；`yaw_value` 为 rad；`timeout` 为 simulation seconds。默认 `KEEP_CURRENT` 保存 `start()` 时的 yaw；只有选择 `FIXED` 并提供 `yaw_value` 时才渐进旋转机头。TAKEOFF 当前只接受这两种 yaw mode，`COURSE_ALIGNED / FACE_POINT` 会返回 `INVALID_GOAL`。

TAKEOFF 不调用 `set_pose()`。每次 tick 只向 KinematicUAV 写入垂直目标、爬升速度和 yaw policy，真正的位置更新仍发生在后续 simulation step。目标感知积分会按每个 physics step 限速，并在接近目标时缩短最后一步，避免跨过 TAKEOFF tolerance。到达后返回 `SUCCEEDED + TAKEOFF_COMPLETE`；仿真时钟超限返回 `FAILED + TIMEOUT`；完成、超时和 cancel 都会停止 UAV。

## Phase 6B：Ideal Kinematic GOTO

`GotoSkill` 接受 world-frame 三维位置，并可直接注册到统一 Manager：

```python
manager.register(SkillName.GOTO, GotoSkill())
manager.start(
    SkillName.GOTO,
    GotoGoal(
        position=(20.0, 30.0, 10.0),
        tolerance=1.0,
        motion_policy=MotionPolicy(
            max_speed=3.0,
            yaw_mode=YawMode.FACE_POINT,
            look_at_point=(30.0, 40.0, 0.0),
        ),
        timeout=60.0,
    ),
)
```

`position`、`tolerance` 和 `look_at_point` 使用 m，`max_speed` 使用 m/s，`yaw_value` 使用 rad，`max_yaw_rate` 使用 rad/s，`timeout` 使用 simulation seconds。默认 MotionPolicy 为 `COURSE_ALIGNED`；也支持保存起始 yaw 的 `KEEP_CURRENT`、世界系固定 yaw 的 `FIXED`，以及每个 kinematic step 都重新朝向指定世界点的 `FACE_POINT`。

GOTO 不调用 `set_pose()`，也不等待机头先转完。每次 tick 同时写入 world-frame translation target 与独立 yaw policy，随后两者在同一个 simulation step 中受各自速度上限连续更新。位置进入 tolerance 即返回 `SUCCEEDED + GOAL_REACHED`，不额外等待 yaw 收敛；超时返回 `FAILED + TIMEOUT`，完成、超时和 cancel 都会停止剩余平移与转向命令。

## Phase 7：Ideal SEARCH

第一版 SEARCH 只实现一种确定性策略：围绕区域中心生成固定六边形 waypoints，每个 waypoint 完成一次连续 FULL_360 yaw scan。它不调用 Qwen3-VL、不训练策略，也不实例化或调用 GOTO Skill：

```python
search_skill = SearchSkill(
    transit_yaw_mode=config.search.transit_yaw_mode,
)
manager.register(SkillName.SEARCH, search_skill)
manager.start(
    SkillName.SEARCH,
    SearchGoal(
        center=(0.0, 0.0, 0.5),
        radius=25.0,
        target_description="person wearing a red jacket",
        search_altitude=10.0,
        transit_speed=1.5,
        scan_yaw_rate=0.5,
        timeout=60.0,
    ),
)
```

waypoint 的 XY 只由 `center + radius` 生成，Z 固定为 `search_altitude`。默认 transit yaw mode 来自 `search.transit_yaw_mode: FACE_POINT`，使 UAV 沿区域外围移动时持续朝向中心；配置也允许 `COURSE_ALIGNED` 和 `KEEP_CURRENT`。扫描阶段停止平移，以受 UAV 硬件上限约束的正 yaw rate 连续转动；累计角由已知的受限 yaw rate 与相邻帧 simulation time 积分得到，并用 wrapped Camera-frame UAV yaw 校验实际执行，因此可以正确跨越 ±π。扫描不能使用 pose 跳转，也不能把 `start_yaw + 2π` 当作普通 yaw target。

Ideal SEARCH 必须接收 `environment.get_skill_observation(include_oracle=True)`。Oracle 仅提供与 RGB 同帧的“目标是否在 Camera frustum 内”；只有 `oracle_target_visible=True` 后，Skill 才读取带 `oracle_` 前缀的 id/pose 并复制进 Result，绝不使用真值位置生成航点或导航。成功返回 `SUCCEEDED + TARGET_FOUND`，数据包含同帧的 `target_id / found_timestamp / uav_pose / camera_pose / oracle_target_pose`；六点全部扫描后返回 `FAILED + SEARCH_EXHAUSTED`，超时返回 `FAILED + TIMEOUT`。Skill clock 与 Observation timestamp 必须共享同一 simulation-time epoch；默认 10 Hz Camera、90° FOV 和 0.5 rad/s scan 在相邻图像间具有充分视场重叠，提高 scan rate 或降低 Camera frequency 时也应保持这一采样覆盖关系。

六个 waypoint 各扫描 360° 时，仅默认 `0.5 rad/s` 扫描就约需 75.4 s，尚未包含 waypoint 间移动。因此 `SearchGoal.timeout=60.0` 是单次请求默认值，不保证能够走完整条搜索轨迹；需要验收 `SEARCH_EXHAUSTED` 时应显式提供足够长的 timeout。默认 YAML 保留更长的 120 s，并可按场景半径继续调大。

## Phase 8：Ideal TRACK

TRACK 面向已经确认的单个目标，第一版使用 Oracle target pose 生成理想跟随位置，不调用 PID、MPC、RL 或 Qwen3-VL：

```python
manager.register(SkillName.TRACK, TrackSkill())
manager.start(
    SkillName.TRACK,
    TrackGoal(
        target_id="target",
        desired_distance=6.0,
        desired_altitude=8.0,
        max_speed=2.0,
        max_target_lost_time=2.0,
        timeout=None,
        track_duration=30.0,
    ),
)
```

`desired_distance` 是 Target 与 UAV 的 XY 水平距离，`desired_altitude` 是 world-frame 绝对 Z；两者单位均为 m。首次收到目标真值时，TRACK 固定 Target→UAV 的水平相对方位，随后让该跟随位置随 Target 平移。每个 tick 同时下发 world-frame xyz 目标和动态 `FACE_POINT(current_target_position)` yaw，二者分别受 `max_speed` 和 UAV 硬件 yaw rate 限制。因此 UAV 可以侧飞、斜飞或后退，不需要先转完机头，也不会调用 `set_pose()` 瞬移。

运行时必须使用与 RGB 同帧的 `environment.get_skill_observation(include_oracle=True)`。TRACK 只把 `oracle_target_visible` 当作 Camera frustum 判定；Oracle pose 可以在短暂不可见期间继续支持 ideal control，但只有新鲜且可见的目标帧才能刷新 `last_seen_time / position / velocity`。last-seen age 以图像采集 timestamp 计算，迟到且已越过丢失 deadline 的可见帧不能复活 TRACK。不可见时间严格超过 `max_target_lost_time` 后返回 `FAILED + TARGET_LOST`，Result 保留最后一次真正可见的真值，不会被隐藏期间的 Oracle pose 覆盖。可选 `timeout` 到期返回 `FAILED + TIMEOUT`；`track_duration` 到期返回 `SUCCEEDED + TRACK_COMPLETE`，而 `None` 保留无限跟踪行为。当多个 deadline 在一次低频采样中同时越过时，按绝对仿真时间最早发生者决定 ResultCode；同刻采用 `TRACK_COMPLETE > TIMEOUT > TARGET_LOST`。终态或 cancel 都会清除剩余平移和 yaw 命令。

`target_distance` 与 `distance_error` 同样采用水平距离；`target_relative_bearing` 使用 rad，`last_seen_age` 和 `tracking_duration` 使用 simulation seconds。Camera pitch 仍由固定安装配置决定，TRACK 当前只控制 UAV xyz 与 yaw。

## Phase 9A：Ideal REACQUIRE

REACQUIRE 接收 TRACK 的 `TARGET_LOST` Result 中保存的最后可见状态，但不会自行实例化或启动 TRACK：

```python
manager.register(SkillName.REACQUIRE, ReacquireSkill())
manager.start(
    SkillName.REACQUIRE,
    ReacquireGoal(
        target_id="target",
        last_seen_position=(4.0, 2.0, 0.5),
        last_seen_velocity=(0.4, -0.1, 0.0),
        last_seen_time=12.5,
        search_radius=10.0,
        timeout=30.0,
    ),
)
```

Skill 启动时用 `last_seen_position + last_seen_velocity * (start_time - last_seen_time)` 做一次常速度预测，并冻结该预测中心。`search_radius` 是预测中心周围的 XY 水平到达半径；由于 Goal 没有搜索高度，接近阶段保持 REACQUIRE 启动时的 UAV 高度，并用 `FACE_POINT(predicted_position)` 持续观察预测区域。当前位置进入半径后停止平移，以受 UAV yaw-rate 上限约束的 `0.5 rad/s` 正向连续扫描；每完成 360° 后继续下一圈，直到重新发现目标或超时。

Ideal detection 仍只消费与 RGB 同帧的 `oracle_target_visible`。只有可见目标的 id 与 Goal `target_id` 相同才返回 `SUCCEEDED + TARGET_FOUND`；当前 Oracle pose 仅在成功帧写入 Result，绝不参与预测或导航。有效 deadline 帧优先于 timeout，正常失败只有 `FAILED + TIMEOUT`。成功后的 TRACK 切换必须由外部 `SkillManager`/Planner 在 reset 后发起，REACQUIRE 内部没有 Skill-to-Skill 调用。

## Phase 9B：Ideal Kinematic LAND

LAND 在 `start()` 时锁定当前 world-frame XY，并以目标感知的 Kinematic motion 连续下降：

```python
manager.register(SkillName.LAND, LandSkill())
manager.start(
    SkillName.LAND,
    LandGoal(
        ground_altitude=0.0,
        tolerance=0.1,
        descent_speed=0.5,
        yaw_mode=YawMode.KEEP_CURRENT,
        timeout=30.0,
    ),
)
```

`ground_altitude`、`tolerance` 使用 m，`descent_speed` 使用 m/s，`timeout` 使用 simulation seconds。默认 `KEEP_CURRENT` 保存 LAND 启动时的 yaw；也可选择 `FIXED` 并提供 world yaw `yaw_value`。每个 tick 的目标始终是 `(landing_x, landing_y, ground_altitude)`，因此正常下降时 XY 不动，发生小偏移时也会向锁定点修正；平移和 yaw 同时受限更新，不会先旋转、不会调用 `set_pose()`，最后一步也不会越过地面。

高度进入 tolerance 后返回 `SUCCEEDED + LAND_COMPLETE`；超时返回 `FAILED + TIMEOUT`。Result 中的 `is_airborne` 按本次 Goal 的 `z > ground_altitude + tolerance` 判断，不假定所有场景地面恒为零。完成、超时和 cancel 均清除下降及未完成 yaw 命令。本阶段不模拟 landing gear、ground effect、motor disarm 或 PX4 land mode。

## Phase 10：SkillManager 与完整 Oracle Pipeline

任务模式在统一 Manager 上显式开启。`TaskPlan.from_dicts()` 接受手写编排映射，但启动每个 Skill 前都会把参数转换为对应 Goal dataclass；Skill 本身仍不接收通用 dict。`TaskPlan` 现在是带稳定 step id 的通用线性容器，不再硬编码五步/六步顺序；顺序、2～10 步长度、次数、世界语义和安全约束由 `PlanValidator` 与 `SafetySupervisor` 双重检查。顶层 REACQUIRE 仍禁止，只能作为 TRACK 的 `RecoveryPolicy`：

```python
from skills.manager import (
    SkillManager,
    TaskPlan,
    create_default_skill_registry,
)

plan = TaskPlan.from_dicts([
    {"id": "takeoff_1", "skill": "TAKEOFF", "target_altitude": 10.0},
    {"id": "goto_search", "skill": "GOTO", "position": [20.0, 30.0, 10.0]},
    {
        "id": "search_1",
        "skill": "SEARCH",
        "center": [20.0, 30.0, 0.0],
        "radius": 15.0,
        "target_description": "moving target",
        "search_altitude": 10.0,
    },
    {
        "id": "track_1",
        "skill": "TRACK",
        "target_id": "$search_1.target_id",
        "track_duration": 30.0,
        "recovery": {
            "skill": "REACQUIRE",
            "max_attempts": 2,
            "search_radius_m": 10.0,
            "timeout_s": 30.0,
        },
    },
    {"id": "goto_home", "skill": "GOTO", "position": [0.0, 0.0, 10.0]},
    {"id": "land_1", "skill": "LAND"},
])

registry = create_default_skill_registry(
    transit_yaw_mode=config.search.transit_yaw_mode,
)
manager = SkillManager(context, registry=registry)
manager.start_task(plan)

while manager.task_status.name == "RUNNING":
    if environment.step():
        frame = environment.get_evaluator_frame()
        observation = oracle_perception.observe(frame)
        manager.tick(observation)
```

`TaskStatus` 独立于 `SkillStatus`。Manager 按 step id 保存 SEARCH 输出，并在启动 TRACK 时解析结构化 `StepOutputRef`；旧 `$SEARCH.result.target_id` 仍只作为 baseline 兼容形式。动态计划的 TRACK 返回 `TARGET_LOST` 时，Manager 只执行 Compiler 已经附着的 `RecoveryPolicy`：非空且预算未耗尽才启动内部 REACQUIRE，成功后回到同一 TRACK；为 `None` 时直接进入任务失败和 emergency LAND。恢复不是顶层计划步骤，也不会形成无限循环。任何 Skill 失败、引用解析失败、取消或意外提前结束都保持安全降落语义；每个外部 observation 最多 tick 一个 Skill。

### Legacy MissionIntent 与动态 SkillPlanDraft

`planner.schemas` 定义不可变的 `MissionIntent`、具名 `SearchRegionSpec` / `LandingZoneSpec` 和只读 `PlannerWorldContext`。高层意图只包含目标描述、区域名、跟踪时长、降落区名和可选起飞高度，不允许携带 Target/Oracle 坐标，也不包含低层 timeout 或控制参数。`MissionIntent.from_dict()` 对未知字段、缺失字段、bool、NaN 和 Inf 严格报错。

`ScriptedPlanner` / `LLMPlanner` 继续返回 `MissionIntent` 并走固定模板。`ScriptedDynamicPlanner` / `DynamicLLMPlanner` 返回严格 `SkillPlanDraft`：对象拒绝未知字段、重复 step id、bool 数字、NaN/Inf、未知 Skill/参数、前向或非 SEARCH 引用；LLM 首次输出不合法时最多修复一次，temperature 固定为 0。动态 `source` 为 `dynamic_scripted` 或 `dynamic_llm`。

`SkillPlanDraft` 是模型唯一允许输出的动态协议；其中只保留具名地点、高层语义参数和前序 SEARCH 输出引用，不含解析后的坐标、速度或普通 Skill timeout。例如导航任务可以省略 SEARCH/TRACK：

```json
{
  "schema_version": 1,
  "steps": [
    {"id": "takeoff_1", "skill": "TAKEOFF", "args": {"altitude_m": 8.0}},
    {"id": "goto_search", "skill": "GOTO", "args": {"destination": "search_area"}},
    {"id": "goto_home", "skill": "GOTO", "args": {"destination": "home"}},
    {"id": "land_1", "skill": "LAND", "args": {"zone": "home"}}
  ]
}
```

需要跟踪时，`TRACK.args.target_ref` 只能写成 `$<先前SEARCH步骤id>.target_id`。`REACQUIRE` 不占顶层步骤，只能附着在 TRACK 的 `recovery` 中，并由 `max_attempts` 和全局恢复预算共同限制。TRACK 还可用 `on_target_lost` 表达 `REACQUIRE` 或 `FAIL`；未写该字段时继承可信 `PlannerPolicy`，默认是 `REACQUIRE`。显式 `FAIL` 禁止同时携带 `recovery`，编译结果不含恢复策略；它表示目标丢失后任务失败并原地紧急降落，不表示条件式返航。当前协议没有实现 `RETURN_HOME`、无目标继续执行或运行时询问 LLM。

Skill Catalog 是模型可见的功能标签白名单，当前只注册已实现的 TAKEOFF/GOTO/SEARCH/TRACK/REACQUIRE/LAND。REACQUIRE 标记为 recovery-only。默认可信限制为最多 10 个顶层步骤、5 次 GOTO、1 次 SEARCH、2 次 TRACK、每个 TRACK 最多 2 次恢复、总计最多 4 次恢复，TRACK duration 为 1～600 s；可在 `configs/default.yaml` 的 `planner` 段收紧。`PlannerPolicy` 另行保存默认目标丢失动作及可信的恢复次数、半径和 timeout。模型可以选择允许的动作或给出有界覆盖，但不能修改这些边界。v1 仍是单目标、有限线性计划，不支持分支图、多目标或运行时 LLM 重规划。

### Structured output、符号检查与可信编译

`DynamicLLMPlanner` 的首次生成和唯一一次 repair 都使用同一份 `SkillPlanDraft` JSON Schema structured output。请求通过 OpenAI-compatible `response_format.type=json_schema` 发送；若服务端不支持并返回错误，客户端会明确失败，不会静默降级为自由文本。Schema 用 `oneOf` 区分五种顶层 Skill，只暴露 world context 中的具名区域、降落区和导航点枚举，不包含这些名称背后的坐标，也不包含 Oracle 数据、速度或底层控制参数。受约束生成之后仍执行严格 JSON、duplicate key、有限数值、dataclass 和 Catalog 校验，不能把 JSON Schema 当作唯一信任边界。

跨步骤规则集中由共享 `SymbolicPlanChecker` 检查，包括 TAKEOFF/LAND 位置与次数、LAND 前匹配的 GOTO、调用预算、TRACK 引用、顶层 REACQUIRE 禁止项，以及 `FAIL` 与 recovery 的冲突。`DynamicLLMPlanner` 用稳定的 `PlanIssueCode` 生成结构化 repair 请求，`PlanValidator` 在可信编译前复用同一个 Checker；后者仍独占具名地点到坐标、速度/timeout、安全参数、默认 recovery 和正常 LAND 几何的编译权。`SafetySupervisor` 随后独立检查编译后的 `TaskPlan`，属于不同的运行时安全边界。

恢复策略在 dynamic 与 legacy 路径上的边界不同：dynamic Compiler 按 `on_target_lost`、显式 recovery、`PlannerPolicy` 的优先级生成最终 `TaskStep.recovery`，Manager 不再猜测默认策略；旧 `MissionIntent` 模板和历史 placeholder 继续保留原有恢复 fallback，以免破坏兼容入口。显式 `REACQUIRE` 可以使用可信默认值或有界覆盖，省略动作则继承 policy；旧手写 draft 仅提供 recovery 时按 REACQUIRE 兼容，`max_attempts=0` 仅作为弃用的禁用写法接受。Compiler notes 会说明恢复来自可信默认、显式有界启用或显式关闭。

正常计划 LAND 与 emergency LAND 具有不同语义。Compiler 从可信 `LandingZoneSpec` 为正常 LAND 附加 `expected_position_xy` 和 `zone_tolerance_m`；这些字段不由 Qwen 输出。`LandSkill` 启动和下降期间都检查实际 XY 是否仍在容差内，通过后锁定当前 XY 垂直下降，而不会把 LAND 当作第二个 GOTO。前置 Skill 失败或取消时，Manager 创建 `expected_position_xy=None` 的 emergency LAND，在当前位置下降，不受 home XY 检查约束；其成功只完成 fail-safe termination，最终任务状态仍为原先的 `FAILED` 或 `CANCELED`。若正常计划 LAND 的区域检查失败，只允许再尝试一次 emergency LAND；紧急降落自身失败后直接结束，禁止递归 LAND 重试。

Planner 可通过 `plan_with_diagnostics()` 返回输出及不含原始模型文本的诊断。`run_planner_demo.py` 的文本和 `--json-output` 模式均报告 `model_calls`、`repair_used`、`repair_succeeded`、`initial_output_valid`、`final_output_valid`、`initial_error_code`、`initial_error_message` 与 `structured_output_enabled`。首轮成功的 LLM 调用数为 1，repair 路径为 2；scripted 模式为 0 且不启用 structured output。错误码区分 `INVALID_JSON`、`SCHEMA_INVALID`、`CATALOG_CONTRACT_VIOLATION` 和稳定的符号问题码，失败输出也只保留脱敏诊断，不默认保存或打印完整首轮响应。

主流程失败不会立即让程序把 UAV 留在空中。TAKEOFF、GOTO、SEARCH、TRACK 或 REACQUIRE 失败时，Manager 先把 `pending_task_result` 设为 `FAILED`，随后执行 emergency LAND；LAND_COMPLETE 后才提交最终 Task `FAILED`。TRACK_COMPLETE 同样先设置待定 `SUCCEEDED`，正常计划 LAND 完成后才提交 Task `SUCCEEDED`。正常 LAND 失败可降级为一次 emergency LAND，emergency LAND 再失败则直接 Task `FAILED`。`transition_log` 为每次切换保存 simulation timestamp、旧 Skill/status、ResultCode、新 Skill 和 reason。

`cancel_task()` 采用相同安全策略：非 LAND Skill 会被取消并切换到 LAND；若已经在下降，则不会中断 LAND。此时已有 `FAILED` 保持最高优先级，否则把待提交结果改为 `CANCELED`。

`perception.OraclePerception` 是 evaluator-only 的薄适配器：它只复制同一 Camera tick 已缓存的 RGB、UAV/Camera pose、Target truth 和 frustum flag，不持有 environment/scene，也不自行导航或重新计算可见性。该对象只用于 Stage-0 测试与理想 pipeline，不能注入 Planner/VLM 的普通观测路径。

Oracle demo 使用 `target.motion.seed` 在 `target.initial_region` 内可重复地随机生成初始位置，随后按默认 `RANDOM_WALK` 连续运动；单元/集成测试可切换为 STATIC 以稳定验证完整状态机。

完整 standalone 入口：

```bash
./python.sh scripts/run_oracle_pipeline.py --config configs/default.yaml \
  --start-altitude 0 --takeoff-altitude 10 --track-duration 30
```

## Planner 数据集 v1：Gold 任务链与离线评测

Planner v1 是一个纯文本、Planner-only 的中文数据集。它只训练和评测下面这一段映射：

```text
自然语言任务指令
    ↓
MissionIntent（五个严格字段）
```

数据中不包含 Camera 图片、视频、Observation dump、目标出生坐标或速度、Oracle/evaluator 真值、Skill sequence、航点、姿态、电机动作和控制轨迹。生成、验证和离线评测均为纯 Python，不启动 Isaac Sim；本仓库也没有因此新增 Qwen SFT、LoRA、RL 或其他实际训练入口。开放 VLN 数据集的任务定义、坐标系、动作空间和标签 schema 与当前 `MissionIntent` 不同，因此不能直接把开放 VLN 样本当作 Planner v1 的监督标签；使用前必须经过独立的语义映射、人工审核和本项目 validator，而不能简单拼接进 JSONL。

标签采用 Gold-first 流程：先从封闭 target ontology、公开的 world context 和可信默认值构造不可变 `GoldPlannerSpec`，再由 Gold 确定性生成 assistant JSON，最后渲染自然语言 instruction 和与运行时逐字节一致的 prompt。Gold 在调用任何模型前就已经确定，绝不从 Planner prediction 反推或回填。`IntentJudge` 分开报告严格输出 `exact_match` 和解析默认值后的五字段 `semantic_match`；执行自己的错误计划不能获得 instruction-grounded success。

在 `uav_agent/` 目录生成 pilot：

```bash
cd /path/to/vlm_drones/uav_agent
./python.sh scripts/generate_planner_dataset.py \
  --config resources/planner_v1/dataset_config.yaml \
  --output-root ../datasets/planner_v1 \
  --seed 42 \
  --profile pilot
```

已存在输出时生成器默认拒绝覆盖；只有确认替换整个正式数据集时才加入 `--overwrite`。若目录内存在尚未审核的 `_candidates/`，覆盖也会被拒绝，须先移动或完成审核，避免丢失候选。相同资源、配置和 seed 会产生相同的 JSONL 内容与 checksums。pilot 的精确数量为 train 1,000、validation 200、test_iid 200、test_compositional 200、test_language 200、test_robustness 100，共 1,900 条；它们全部标记为 `generation_source=template`、`review_status=unreviewed`，不会把自动模板伪装成人工数据。

`full` profile 定义了 8,000 / 1,000 / 1,000 / 1,000 / 1,000 / 500，共 12,500 条的候选规模，但它不是“自动生成即正式发布”的捷径。正式 full 数据必须先在人工审核流程中达到 test_language 至少 50% 人工编写或逐条审核、test_robustness 100% 人工审核，并保留真实 provenance；当前仓库不冒充或自动执行这项人工工作。因此未审核的 `--profile full` 正式写盘会按预期失败，validator 也拒绝把未达比例的数据当作 full。当前可直接复现并通过完整验收的是下述 1,900 条 pilot。

默认 `--paraphraser none`，不会加载 Qwen、调用网络或付费 API。可选外部 paraphraser 只能改写 instruction，不能生成或修改 Gold/assistant label。调用方先在仓库外得到只含 `sample_id` 与 `candidate_instruction` 的 JSONL，再用下列显式 candidate-only 命令做封闭词表检查和暂存；CLI 本身绝不会调用外部服务：

```bash
./python.sh scripts/generate_planner_dataset.py \
  --config resources/planner_v1/dataset_config.yaml \
  --output-root ../datasets/planner_v1 \
  --seed 42 \
  --profile pilot \
  --paraphraser external \
  --candidate-only \
  --candidate-input /path/to/external-candidates.jsonl
```

原始改写只允许先进入 `datasets/planner_v1/_candidates/`，通过语义和泄漏检查及人工审核后才可进入正式数据。`_candidates/` 被 Git 忽略，正式 split、manifest、统计和 checksums 不被忽略。缺少 `--candidate-only` 或 candidate input 时 external 模式会明确拒绝，且绝不会改变正式 split。

完整验证只需一条命令，任何 JSON/schema、Gold、prompt、ontology、跨 split 泄漏或 checksum 错误都会返回非零退出码：

```bash
./python.sh scripts/validate_planner_dataset.py \
  --dataset-root ../datasets/planner_v1
```

生成后的仓库根目录结构为：

```text
datasets/planner_v1/
├── dataset_manifest.json
├── train.jsonl
├── validation.jsonl
├── test_iid.jsonl
├── test_compositional.jsonl
├── test_language.jsonl
├── test_robustness.jsonl
├── statistics.json
├── checksums.sha256
└── _candidates/                 # 可选、未审核且被忽略
```

Evaluator 提供四种模式：`scripted`、`llm` 保留 MissionIntent baseline；`dynamic_scripted`、`dynamic_llm` 评测受约束 SkillPlanDraft。先用 scripted evaluator 自检 evaluator 和 Gold judge；它不调用模型，所有有效 split 的 exact/semantic match 都应为 100%：

```bash
./python.sh scripts/evaluate_planner_dataset.py \
  --dataset-root ../datasets/planner_v1 \
  --split test_iid \
  --planner scripted \
  --output-root ../outputs/planner_eval

for split in train validation test_iid test_compositional test_language test_robustness; do
  ./python.sh scripts/evaluate_planner_dataset.py \
    --dataset-root ../datasets/planner_v1 \
    --split "$split" \
    --planner scripted \
    --output-root ../outputs/planner_eval
done
```

`dynamic_scripted` 从同一份 Planner v1 Gold 确定性构造标准 dynamic draft，用来验证动态 evaluator 自身，不需要另建或改写数据集；`dynamic_llm` 则调用生产 `DynamicLLMPlanner`。动态判定会 canonicalize step id，并把 draft 投影成 instruction-grounded 语义，因此语义相同但 step id 不同不会误判，可信默认 REACQUIRE 是否由模型显式写出也不影响旧 Gold 的 semantic match。额外的无任务依据 SEARCH、TRACK 或绕行 GOTO 会单独降低 `minimal_plan_match`。

```bash
# 纯 Python 的确定性动态 evaluator
./python.sh scripts/evaluate_planner_dataset.py \
  --dataset-root ../datasets/planner_v1 \
  --split test_iid \
  --planner dynamic_scripted \
  --output-root ../outputs/planner_eval

# 已在独立进程启动 Qwen 服务后再运行；本命令不会启动 Isaac Sim
./python.sh scripts/evaluate_planner_dataset.py \
  --dataset-root ../datasets/planner_v1 \
  --split test_iid \
  --planner dynamic_llm \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-4B-Instruct \
  --api-key EMPTY \
  --output-root ../outputs/planner_eval \
  --limit 20
```

动态 evaluator 对首轮和最终输出分别记录 `schema_valid`、`catalog_valid`、`symbolic_valid`、`compile_success`，并报告 `semantic_match` 与 `minimal_plan_match`。汇总还包含初次/最终 schema、symbolic 和 compile 成功率、repair request/success、平均模型调用数与延迟、默认 recovery 注入率和显式 FAIL 比例；`errors.csv` 使用稳定错误码，`field_metrics.csv` 增加 Skill sequence 与 lost-target policy 统计。动态 LLM 模式同时验证最多两次 chat、所有调用 `temperature=0` 且 structured output 已启用；旧 evaluator 的字段准确率继续保留。

连接现有 OpenAI-compatible Qwen 文本服务时，evaluator 复用生产 `LLMPlanner` 的严格 JSON parser、固定 `temperature=0` 和最多一次 repair；它不发送图像或 Gold，也不用 `PlanValidator` 代替 Gold judge：

```bash
QWEN_API_BASE=http://127.0.0.1:8000/v1 \
QWEN_API_KEY=EMPTY \
QWEN_MODEL=Qwen3-VL-4B-Instruct \
./python.sh scripts/evaluate_planner_dataset.py \
  --dataset-root ../datasets/planner_v1 \
  --split test_iid \
  --planner llm \
  --output-root ../outputs/planner_eval \
  --limit 200
```

可用 `--start-index N --limit M` 评测一个窗口。中断后把首次输出的 run 目录传给 `--resume`，已完成 sample ID 不会重复请求模型：

```bash
QWEN_API_BASE=http://127.0.0.1:8000/v1 \
QWEN_API_KEY=EMPTY \
QWEN_MODEL=Qwen3-VL-4B-Instruct \
./python.sh scripts/evaluate_planner_dataset.py \
  --dataset-root ../datasets/planner_v1 \
  --split test_iid \
  --planner llm \
  --resume /path/to/existing/planner-eval-run
```

每次运行输出到 `outputs/planner_eval/<run_id>/`，只保存 Planner 文本结果：

```text
summary.json
predictions.jsonl
errors.csv
field_metrics.csv
terminal.log
```

`summary.json` 按所选模式聚合上述 legacy 或 dynamic 指标；单条模型、解析、符号检查或编译失败会记录后继续下一条，不会中止整个 split。`predictions.jsonl` 的 dynamic 记录包含初次/最终合法性、`initial_error_code`、模型调用数和 repair 结果，但不保存 API key、隐藏世界真值、完整环境对象、图像或默认的完整首轮模型输出。

## 统一配置

`configs/default.yaml` 管理：scene 大小、UAV 初始位置/最大速度/最大 yaw rate、Camera resolution/frequency/FOV/focal length/pitch、Target 初始区域/最大速度/运动模式与边界，以及 Search radius、timeout 和 transit yaw mode。加载器在启动昂贵的 Isaac Sim 之前完成类型、有限值、单位、空间边界、统一 physics/render tick 和 Camera frequency 校验；当前运动学环境要求 physics/render dt 相等。

episode reset 必须调用 `environment.reset(target_seed=...)`，它会一起重置 World、UAV、Target 和 Camera observation cache；业务代码不要直接调用 `environment.world.reset()`，否则会绕开运动控制器状态。

## 目录职责

```text
uav_agent/
├── configs/       # 公共 schema、统一 YAML 配置和纯 Python 校验器
├── env/           # scene、kinematic UAV、RGB Camera、moving Target、World wrapper
├── models/        # 纯 Python 模型合同与 OpenAI-compatible HTTP 客户端
├── agents/        # VLM/LLM Agent
├── experiments/   # 轻量运行目录、CSV/TensorBoard、checkpoint、评测与图表
├── skills/        # 统一 Skill API、MotionPolicy、Manager 与六类 Goal 合同
├── perception/    # 视觉感知；Stage-0 含 evaluator-only OraclePerception
├── planner/       # 任务分解与层次规划
├── tasks/         # 独立 Gold Spec、封闭目标 ontology 与 Intent Judge
├── planner_data/  # Planner v1 渲染、生成、分割、验证、泄漏检查与离线评测
├── resources/     # Planner v1 ontology、公开 world、中文词表和模板配置
├── runtime/       # MissionIntent 校验与确定性 TaskPlan 编译边界
├── prompts/       # Prompt 模板
├── scripts/       # scene demo、Oracle 入口与纯 Python Planner 数据 CLI
├── tests/         # 快速纯测试及一个显式 opt-in Isaac 集成测试
├── logs/          # 运行日志（默认忽略产物）
├── python.sh      # 默认 r_isaac_sim，支持通过环境变量覆盖 prefix
├── environment.yml # Python 3.11 / Isaac Sim 5.1 环境基线
└── README.md
```

仓库根目录的 `outputs/` 是默认实验结果根目录，不属于源码树并由 `.gitignore` 忽略。

## 轻量级训练与评测结果输出

`experiments` 包提供一套与 Isaac、Torch 和模型实现解耦的公共输出层。输出根目录按以下优先级解析：显式 `--output-root` 或 `RunManager.create(output_root=...)`，其次 `VLM_DRONE_OUTPUT_ROOT`，最后是仓库根目录的 `outputs/`。开始运行前会验证目录可写并检查剩余空间；默认至少需要 20 GiB。

每个 run 只创建以下目录，不创建图片帧、视频、轨迹、Observation dump 或周期 checkpoint：

```text
outputs/runs/<experiment_name>/<run_id>/
├── manifest.yaml
├── resolved_config.yaml
├── command.sh
├── exit_code.txt
├── logs/terminal.log
├── metrics/
│   ├── train_metrics.csv
│   ├── eval_metrics.csv
│   ├── episode_metrics.csv
│   ├── failure_cases.csv
│   └── final_metrics.csv
├── tensorboard/events.out.tfevents.*
├── checkpoints/
│   ├── best/
│   └── latest/
└── figures/
    ├── train_success_rate.png
    ├── eval_success_rate.png
    ├── final_success_rate.png
    ├── stage_success_rate.png
    ├── failure_breakdown.png
    └── training_curve.png
```

`MetricLogger` 的 CSV 是固定 schema；不存在的测量值写空单元格，不伪造为零。episode 使用 `(run_id, phase, episode_id)` 防重复，恢复运行会继续追加并要求 train/eval global step 严格递增。`compute_mission_success_strict()` 是训练、验证和测试唯一的完整任务成功定义；错误锁定、碰撞、越界、安全中止、超时或未成功降落都会判失败。验证和测试使用互斥固定种子，最终测试只接受 `best` checkpoint，并报告 Wilson 95% 置信区间。

TensorBoard writer 只公开 `add_scalar()`，没有 image、video、histogram、graph、embedding 或 text API；其 TFEvent writer 不要求运行环境安装 TensorBoard，安装 `environment.yml` 中的 TensorBoard 后即可读取。`CheckpointManager` 只管理 `best` 与 `latest` 两个原子替换槽位。`latest` 可包含恢复所需 optimizer、scheduler、RNG 和 normalizer state；`best` 会移除训练状态。默认 adapter-only 模式拒绝完整 Qwen 基础模型和其 shard，基础模型只在 manifest 中记录名称及路径。

训练入口的推荐接线顺序是：创建 `RunManager`，在 update 粒度调用 `MetricLogger.log_train()`，固定间隔通过 `Evaluator` 验证并调用 `CheckpointManager.maybe_save_best()`，周期覆盖 `latest`，最后加载 `best` 进行测试、写 `final_metrics.csv`，再由 `ExperimentPlotter` 从 CSV 生成 PNG。当前仓库尚无实际 RL/SFT 训练入口，因此没有擅自把该模块接入仿真 tick；现有 Isaac demo 和 MissionAgent 流程保持不变。

外部训练进程可使用统一 tee 启动器，保留实时 terminal、合并 stdout/stderr、真实 Python 退出码以及恢复分隔线：

```bash
RUN_DIR=/path/to/existing/run \
  uav_agent/scripts/run_with_output.sh -m your_training_module --config configs/local.yaml

VLM_DRONE_RESUME=1 RUN_DIR=/path/to/existing/run \
  uav_agent/scripts/run_with_output.sh -m your_training_module --resume
```

默认输出配置位于 `configs/default.yaml` 的 `experiment`、`logging`、`tensorboard`、`checkpoint`、`evaluation`、`artifacts`、`figures` 和 `storage` 段。运行期可周期调用 storage guard；低于 10 GiB 或 run 达到 5 GiB 时，统一协调层会尝试覆盖保存 `latest`、flush CSV/TensorBoard、更新 manifest 并要求训练循环安全结束。它不会删除其他 run 或其他用户文件。

真实训练入口应显式启用中断保护；构造 `ExperimentRuntime` 本身不会修改进程级 signal handler。下面的 context 会在 Ctrl+C（SIGINT）或 SIGTERM 时按 `latest → flush CSV/TensorBoard → 更新 manifest` 的顺序做 best-effort 持久化，并恢复原 handler：

```python
from experiments import InterruptState

with runtime.interrupt_handlers(
    lambda: InterruptState(
        global_step=global_step,
        update=update,
        payload=adapter_and_resume_state,
    )
):
    training_loop()
```

未捕获的中断分别以 130（SIGINT）或 143（SIGTERM）退出，和 manifest/`exit_code.txt` 中记录的进程段状态保持一致。该机制只允许在主线程显式安装；signal handler 本身不执行 checkpoint 或文件 I/O。

纯 Python 冒烟测试会生成 20 个训练、10 个验证和 20 个测试 episode，并自动核验精确目录、五个 CSV、真实 scalar TFEvent、best/latest、六张图、总体大小和禁用 artifact 不存在：

```bash
cd /path/to/vlm_drones
./uav_agent/python.sh -m uav_agent.experiments.smoke_test \
  --output-root /tmp/vlm_drone_output_smoke
```

## 测试约定

一次修改会话结束时再集中运行测试，避免重复启动 Kit。纯配置和运动学测试不导入 Isaac Sim：

```bash
./python.sh -m unittest discover -s tests -v
```

真实 Isaac smoke test 只在模块全部完成后运行一次。如果首次启动提示 NVIDIA Omniverse EULA，需由账户使用者本人按服务器规范接受；项目不会自动代替用户设置接受标志。

Phase 10 的完整 Oracle 集成默认由 unittest 跳过，避免普通回归启动 Kit。只在一次修改会话全部结束时显式运行：

```bash
UAV_AGENT_RUN_ISAAC_TESTS=1 \
  ./python.sh -m unittest tests.test_full_skill_pipeline_oracle -v
```

该测试只创建一个 `SimulationApp`，使用 KinematicUAV、RGB Camera、MovingTarget、OraclePerception 和 SkillManager；不导入或调用 Qwen、LLM、VLM。
