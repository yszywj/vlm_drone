# UAV Agent

基于 VLM Agent 的层次化无人机目标搜索与跟踪项目。Stage 0 已具备第一版 Isaac Sim 场景、运动学 UAV、固定 RGB Camera、有界移动 Target，以及 TAKEOFF / GOTO / SEARCH / TRACK / REACQUIRE / LAND 的完整 Oracle 任务流水线。Stage 1 已把 `MissionAgent` 接入该流水线，支持确定性 Scripted Planner 与文本 Qwen Planner 两种高层规划入口；当前视觉搜索、跟踪和重捕获仍由 Oracle 真值驱动，并不代表已经实现真实图片识别。

## 快速开始

项目级 `python.sh` 固定使用服务器 Conda 环境 `r_isaac_sim`，不依赖当前 shell 激活环境，也不会修改其他用户的环境。

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

## Stage 1A / 1B：MissionAgent + Oracle

两种模式共用相同的可信编译、安全检查、目标生命周期和底层 Skill 调度链：

```text
Instruction
    ↓
MissionPlanner                 # ScriptedPlanner 或文本 LLMPlanner
    ↓
MissionIntent                  # 只含语义任务意图
    ↓
PlanValidator                  # 可信 WorldContext → 六步 TaskPlan
    ↓
MissionAgent                   # 每个新 Camera sample 最多 tick 一次
    ↓
SkillManager
    ↓
Oracle-backed Ideal Skills     # SEARCH / TRACK / REACQUIRE 使用 evaluator truth
```

Stage 1A 使用 `ScriptedPlanner`，不依赖模型服务，适合作为 Isaac 集成基线。Stage 1B 使用 `LLMPlanner` 调用 OpenAI-compatible Qwen 服务；Qwen 只在 `MissionAgent.start()` 的规划阶段读取文本指令和去真值的具名区域描述，输出严格 `MissionIntent`。坐标、速度、航点和六步 `TAKEOFF → GOTO → SEARCH → TRACK → GOTO → LAND` 计划均由可信的 `PlanValidator` 生成，仿真 tick 不调用模型。

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

当前能力边界必须明确：Qwen3-VL 在此阶段只接收文本，不接收 Camera 图像；SEARCH / TRACK / REACQUIRE 仍消费带 `oracle_` 前缀的真值字段；LOCK 只是 `TargetManager` 的逻辑生命周期状态。真实图片 detector、视觉 tracker 和 ReID 尚未实现，`perception/detector_tracker.py` 与 `perception/vlm_verifier.py` 仍是占位接口。因此 Stage 1A/1B 都是 Oracle 集成里程碑，不能称为真实视觉搜索闭环。

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

`env/kinematic_uav.py` 是不导入 Isaac 的纯运动学模块，状态为 `UAVState(x, y, z, yaw)`。它不模拟电机、thrust、roll/pitch dynamics 或 aerodynamic forces，也暂不处理碰撞与避障，因此可能穿过障碍物。

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

普通 `Observation` 的所有 `oracle_target_*` 均为 `None`。只有 evaluator/test 显式调用 `get_skill_observation(include_oracle=True)` 才会注入 Target 真值；其中当前 `oracle_target_visible` 表示几何 frustum 内，不包含遮挡判断。SkillContext 只含 KinematicUAV、CameraSensor、perception 和 simulation clock，不含 scene、target 或全局 Manager。

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

任务模式在统一 Manager 上显式开启。`TaskPlan.from_dicts()` 接受手写编排映射，但启动每个 Skill 前都会把参数转换为对应 Goal dataclass；Skill 本身仍不接收通用 dict。TaskPlan 只接受旧五步 TAKEOFF → GOTO → SEARCH → TRACK → LAND，或带返航点的新六步 TAKEOFF → GOTO → SEARCH → TRACK → GOTO → LAND；REACQUIRE 只由恢复规则动态插入：

```python
from skills.manager import (
    SkillManager,
    TaskPlan,
    create_default_skill_registry,
)

plan = TaskPlan.from_dicts([
    {"skill": "TAKEOFF", "target_altitude": 10.0},
    {"skill": "GOTO", "position": [20.0, 30.0, 10.0]},
    {
        "skill": "SEARCH",
        "center": [20.0, 30.0, 0.0],
        "radius": 15.0,
        "target_description": "moving target",
        "search_altitude": 10.0,
    },
    {
        "skill": "TRACK",
        "target_id": "$SEARCH.result.target_id",
        "track_duration": 30.0,
    },
    {"skill": "GOTO", "position": [0.0, 0.0, 10.0]},
    {"skill": "LAND"},
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

`TaskStatus` 独立于 `SkillStatus`。正常路径用 SEARCH Result 的 `target_id` 构造 TrackGoal；TRACK 返回 `TARGET_LOST` 时，Manager 校验并传递 `last_seen_position / velocity / time` 给 ReacquireGoal；REACQUIRE 找回后复用先前 TrackGoal 参数重新进入 TRACK，并从 `track_duration` 中扣除恢复前已经执行的跟踪时间，因此完成条件按多段 TRACK 累计。恢复后的 TRACK 完成时，六步计划仍会继续第二个 GOTO；该 GOTO 失败或任务取消时则跳过返航点，直接执行 fail-safe LAND。Skill 之间不会直接互相调用。

### MissionIntent 与确定性编译

`planner.schemas` 定义不可变的 `MissionIntent`、具名 `SearchRegionSpec` / `LandingZoneSpec` 和只读 `PlannerWorldContext`。高层意图只包含目标描述、区域名、跟踪时长、降落区名和可选起飞高度，不允许携带 Target/Oracle 坐标，也不包含低层 timeout 或控制参数。`MissionIntent.from_dict()` 对未知字段、缺失字段、bool、NaN 和 Inf 严格报错。

`ScriptedPlanner` 是不读取环境、Camera 或 Oracle 的确定性基线，只返回构造时保存的高层意图副本。`LLMPlanner` 让本地 Qwen 文本模型只生成相同的严格 `MissionIntent`，首次输出无效时最多请求一次修复。`runtime.PlanValidator` 在可信世界上下文中解析区域名，检查场景边界、正飞行高度、1～600 s 跟踪时长和描述长度，再确定性生成六步 TaskPlan。当前上下文没有单独的 TAKEOFF timeout，因此编译器明确复用 `goto_timeout_s`。`source` 只允许 `scripted` 或 `llm`；两类 Planner 均由 `MissionAgent.start()` 仅调用一次。

主流程失败不会立即让程序把 UAV 留在空中。TAKEOFF、GOTO、SEARCH、TRACK 或 REACQUIRE 失败时，Manager 先把 `pending_task_result` 设为 `FAILED`，随后执行 LAND；LAND_COMPLETE 后才提交最终 Task `FAILED`。TRACK_COMPLETE 同样先设置待定 `SUCCEEDED`，完成 LAND 后才提交 Task `SUCCEEDED`；LAND 自身失败则直接 Task `FAILED`。`transition_log` 为每次切换保存 simulation timestamp、旧 Skill/status、ResultCode、新 Skill 和 reason。

`cancel_task()` 采用相同安全策略：非 LAND Skill 会被取消并切换到 LAND；若已经在下降，则不会中断 LAND。此时已有 `FAILED` 保持最高优先级，否则把待提交结果改为 `CANCELED`。

`perception.OraclePerception` 是 evaluator-only 的薄适配器：它只复制同一 Camera tick 已缓存的 RGB、UAV/Camera pose、Target truth 和 frustum flag，不持有 environment/scene，也不自行导航或重新计算可见性。该对象只用于 Stage-0 测试与理想 pipeline，不能注入 Planner/VLM 的普通观测路径。

Oracle demo 使用 `target.motion.seed` 在 `target.initial_region` 内可重复地随机生成初始位置，随后按默认 `RANDOM_WALK` 连续运动；单元/集成测试可切换为 STATIC 以稳定验证完整状态机。

完整 standalone 入口：

```bash
./python.sh scripts/run_oracle_pipeline.py --config configs/default.yaml \
  --start-altitude 0 --takeoff-altitude 10 --track-duration 30
```

## 统一配置

`configs/default.yaml` 管理：scene 大小、UAV 初始位置/最大速度/最大 yaw rate、Camera resolution/frequency/FOV/focal length/pitch、Target 初始区域/最大速度/运动模式与边界，以及 Search radius、timeout 和 transit yaw mode。加载器在启动昂贵的 Isaac Sim 之前完成类型、有限值、单位、空间边界、统一 physics/render tick 和 Camera frequency 校验；当前运动学环境要求 physics/render dt 相等。

episode reset 必须调用 `environment.reset(target_seed=...)`，它会一起重置 World、UAV、Target 和 Camera observation cache；业务代码不要直接调用 `environment.world.reset()`，否则会绕开运动控制器状态。

## 目录职责

```text
uav_agent/
├── configs/       # 统一 YAML 配置和纯 Python 校验器
├── env/           # scene、kinematic UAV、RGB Camera、moving Target、World wrapper
├── models/        # 纯 Python 模型合同与 OpenAI-compatible HTTP 客户端
├── agents/        # VLM/LLM Agent
├── skills/        # 统一 Skill API、MotionPolicy、Manager 与六类 Goal 合同
├── perception/    # 视觉感知；Stage-0 含 evaluator-only OraclePerception
├── planner/       # 任务分解与层次规划
├── runtime/       # MissionIntent 校验与确定性 TaskPlan 编译边界
├── prompts/       # Prompt 模板
├── scripts/       # scene demo 与完整 Oracle standalone 入口
├── tests/         # 快速纯测试及一个显式 opt-in Isaac 集成测试
├── logs/          # 运行日志（默认忽略产物）
├── python.sh      # 固定使用 r_isaac_sim
└── README.md
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
