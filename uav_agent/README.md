# UAV Agent

基于 VLM Agent 的层次化无人机目标搜索与跟踪项目。仓库同时保留受保护的 Oracle 理想感知评测路径，以及独立进程运行的 YOLO26/YOLOE + BoT-SORT 生产感知路径；两者都只通过中立的 `TargetEstimate` 接入 TAKEOFF / GOTO / SEARCH / TRACK / REACQUIRE / LAND Skill，不能互相静默回退。`MissionAgent` 支持确定性 Scripted Planner 与文本 Qwen Planner 两种高层规划入口。

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

## 双目标感知模式：Oracle 与 YOLO

Fleet 目标任务现在使用显式的双模式协议：

生产部署、时序几何和无真值泄露的审计说明分别见：

- [YOLO 生产运行手册](docs/yolo_production_runtime.md)
- [时序射线深度估计器](docs/temporal_ray_depth_estimator.md)
- [感知信息边界](docs/perception_information_boundary.md)
- [生产调用链指标与候选日志语义](docs/perception_runtime_evidence.md)

当前单机生产配置使用已晋级的 `temporal_ray_depth` Stage B v2 checkpoint（SHA256 `2a8acd63347a568a4e5588cfc335979709958f7859617405841062af679e023c`），并保留 `isaac_depth` 作为确定性 baseline/fallback。真实 YOLO + Isaac temporal smoke 已在 [20260825-194528 run summary](logs/fleet_missions/runs/fleet_mission/20260825-194528_fleet_mission_seed0_nogit/summary.json) 中完成：`strict_success=true`，267 次时序测量成功、15 次显式 RGB-D fallback、0 次时序非法输出；SEARCH 确认 red cube 后完成 20 s TRACK、返回起点和 LAND，且 `privileged_perception=false`、exit code 0。确认边沿证据见同一 run 的 [target_perception_transitions.jsonl](logs/fleet_missions/runs/fleet_mission/20260825-194528_fleet_mission_seed0_nogit/agents/uav_1/target_perception_transitions.jsonl)。最终 temporal 配置还完成了 seed 101/211/307/401/503 的真实五次评估，[聚合结果](logs/yolo_fixed_seed_eval/task9_temporal_5seeds/summary.json) 为 5/5 `strict_success=true`，检测、颜色确认、3D、SEARCH、20 s TRACK 和 LAND 的阶段成功率均为 1.0。此前 [20260825-185911 baseline](logs/fleet_missions/runs/fleet_mission/20260825-185911_fleet_mission_seed0_nogit/summary.json) 与原始 `isaac_depth` 固定-seed 批次仍作为独立回归证据保留，详见 [YOLO 生产运行手册](docs/yolo_production_runtime.md)。

- `--target-perception-mode oracle`：Isaac GT 只经 assignment-scoped `OraclePerception` 形成 `TargetEstimate`。它仍受 Camera 可见性约束，只用于架构调试、理想能力上界、回归、专家轨迹和标签生成；不是生产视觉，结果会标记为 privileged upper bound。必须同时给出 `--acknowledge-privileged-oracle`，Qwen Vision 只允许 `shadow`。
- `--target-perception-mode yolo`：每架 UAV 使用独立 loopback worker，执行单类 `cube` YOLO、BoT-SORT、同步 RGB-D 颜色确认、已晋级的时序射线深度（或显式 deterministic RGB-D fallback）和 Kalman 估计；结果标记为 production vision。Gate 模式还必须给出 `--acknowledge-vision-gate`。

“不启用 YOLO”对于含 SEARCH / INSPECT / TRACK / REACQUIRE 的任务明确表示选择 Oracle，并不表示 `disabled`。`disabled` 只保留给起飞、纯导航、悬停、返航和降落等完全无目标任务。YOLO 不可用时绝不自动切换 Oracle。

两种模式使用同一场景、UAV、Camera、目标运动、Fleet、搜索和 Planner 参数；`configs/multi_uav_oracle.yaml` 与 `configs/multi_uav_cube_yolo.yaml` 只在目标感知配置及实验名上不同。两种模式也共用相同的 MissionAgent、Skill、任务完成判定和通用结果指标。

双机 Oracle 上界任务不连接 8011/8012，也不创建 detector、tracker、CandidateBank 或颜色验证器：

```bash
./python.sh scripts/run_fleet_mission.py \
  --config configs/multi_uav_oracle.yaml \
  --target-perception-mode oracle \
  --mission-interpreter llm \
  --fleet-planner llm \
  --local-planner dynamic_llm \
  --planning-contract v3 \
  --runtime-program linear \
  --adapter-config configs/adapters.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-4B-Instruct \
  --api-key EMPTY \
  --acknowledge-privileged-oracle \
  --enable-qwen-vision \
  --vision-review-mode shadow \
  --headless \
  --max-sim-time 300 \
  --instruction "无人机A前往世界坐标20,30附近15米范围搜索并跟踪目标i20秒；无人机B前往世界坐标-25,10附近12米范围搜索并跟踪目标j20秒；完成后分别返回各自起点降落"
```

双机 YOLO 任务要求操作者预先训练一个 `names: {0: cube}` 的本地权重，并在 8011、8012 各启动一个独立 worker。启动 Isaac 前可单独执行只读检查；该检查只访问每个 worker 的 `/health` 和 `/v1/model-info`，并验证 ready、`model_family=yolo`、唯一类别 `0=cube` 及记录模型 SHA：

```bash
./python.sh scripts/check_fleet_yolo_services.py \
  --config configs/multi_uav_cube_yolo.yaml
```

服务检查通过后，生产闭环命令为：

```bash
./python.sh scripts/run_fleet_mission.py \
  --config configs/multi_uav_cube_yolo.yaml \
  --target-perception-mode yolo \
  --mission-interpreter llm \
  --fleet-planner llm \
  --local-planner dynamic_llm \
  --planning-contract v3 \
  --runtime-program linear \
  --adapter-config configs/adapters.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-4B-Instruct \
  --api-key EMPTY \
  --enable-qwen-vision \
  --vision-review-mode gate \
  --acknowledge-vision-gate \
  --headless \
  --max-sim-time 300 \
  --instruction "无人机A前往世界坐标20,30附近15米范围搜索并跟踪目标i20秒；无人机B前往世界坐标-25,10附近12米范围搜索并跟踪目标j20秒；完成后分别返回各自起点降落"
```

若不需要 Qwen 属性兜底，可移除三个 Qwen Vision gate 参数；多帧 RGB-D 颜色明确时，运行时本来就不会发起 Qwen 候选审查。`target_i`、`target_j` 始终只是 Assignment 路由 ID：普通 YOLO 只请求同一个 `cube` class，red/blue 由独立、同帧 RGB-D 时序证据确认。Qwen 仅可在长期模糊、不支持属性或新 tracker 重获时低频介入，不能提供三维位置、速度或控制量。

### 运行边界与三个独立环境

目标感知后端与感知运行 profile 是两个独立、严格校验的选择。允许的组合如下：

| 用途 | `perception_runtime_profile` | `target_perception.backend` | Oracle acknowledgement |
| --- | --- | --- | --- |
| 生产视觉 | `production` | `ultralytics_service` | 必须为 false |
| 无目标感知任务 | `production` | `disabled` | 必须为 false |
| 理想上界、回归或 evaluator | `oracle_evaluation` | `oracle_evaluation` | 必须显式为 true |

`production + oracle_evaluation`、未 acknowledgement 的 Oracle、带 acknowledgement 的 production 都会在启动时失败。YOLO 模型缺失、服务不可用、超时、返回空检测或目标类别不支持时也不会切换到 Oracle；任务只能继续搜索、显式失败或进入既有安全取消/降落逻辑。

三个重量级运行时必须隔离：

- `r_isaac_sim` 来自 `environment.yml`，使用 Python 3.11 和 Isaac Sim 5.1.0.0，负责仿真、相机、Agent 和控制；不要在其中安装训练版 Torch/Ultralytics。
- `yolo_perception` 来自 `environment-yolo.yml`，真实 `yolo26s.pt` 冒烟使用的直接 pin 是 Python 3.11、PyTorch 2.7.0 + CUDA 12.6、Torchvision 0.22.0、Ultralytics 8.4.0、LAP 0.5.13、ONNX 1.22.0、FastAPI 0.115.7、Uvicorn 0.29.0、OpenCV-headless 4.11.0、Pillow 11.3.0 和 NumPy 1.26.0，负责 YOLO 推理、BoT-SORT、训练与 ONNX 导出。YOLO26 使用 8.4.0 引入的扩展 SPPF/C3k2 结构；权重中自报的旧版本字段不能作为运行时兼容 pin。`requirements/yolo.lock` 是精确的直接依赖 pin，不冒充带 hash 的完整传递 lock；需要逐字节复现时，应在目标 GPU 主机验收后归档 `python -m pip freeze --all` 和对应 wheelhouse。
- Qwen/vLLM 使用第三个独立环境和 `127.0.0.1:8000`；YOLO 服务默认使用 `127.0.0.1:8011`。只有选择 `dynamic_llm` 或启用 Qwen 语义审查时才需要 Qwen 服务。

Conda 文件不会安装 NVIDIA driver。CUDA 12.6 版 Torch 要求主机 driver 与该 CUDA runtime 兼容；先用 `nvidia-smi` 检查实际 driver/GPU，再按服务器的兼容矩阵选择 wheel。`CUDA_VISIBLE_DEVICES=1` 后，该进程内唯一可见的卡编号仍是 `--device 0`。单卡机器可改用自己的映射，或在明确接受性能差异时使用 `--device cpu`。

创建隔离环境：

```bash
cd /home/amax/ry/vlm_drones/uav_agent
conda env create -f environment-yolo.yml
conda activate yolo_perception
python --version
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
```

模型必须先由操作者下载到仓库外或仓库根目录被忽略的 `models/` 下，并在服务启动时用绝对本地路径显式传入。服务会在加载前要求文件存在，启动时打印 Python、Torch、Ultralytics、CUDA/GPU、模型路径与 SHA256；它不会根据模型名联网下载，也不接受逐帧请求改模型。不要把权重放进源码包 `uav_agent/models/`。

```bash
export YOLO26_MODEL=/home/amax/ry/vlm_drones/models/initial_model/yolo26s.pt
test -f "$YOLO26_MODEL"
sha256sum "$YOLO26_MODEL"
```

### 命令组 1：Oracle 理想感知与旧路径回归

Oracle 只适用于 evaluator 上界、回归测试、训练标签或专家轨迹。下面两个开关缺一不可；该命令不需要启动 YOLO 或 Qwen：

```bash
cd /home/amax/ry/vlm_drones/uav_agent
CUDA_VISIBLE_DEVICES=0 \
./python.sh scripts/run_dynamic_visual_mission.py \
  --config configs/default.yaml \
  --planner dynamic_scripted \
  --perception-runtime-profile oracle_evaluation \
  --target-perception-backend oracle_evaluation \
  --acknowledge-privileged-oracle \
  --instruction "起飞到十米，前往 search_area 搜寻一个移动目标，找到以后跟踪十秒，最后返回 home 降落" \
  --headless
```

原有两个入口也继续保留，且不依赖 YOLO 服务：

```bash
./python.sh scripts/run_oracle_pipeline.py \
  --config configs/default.yaml \
  --start-altitude 0 --takeoff-altitude 10 --track-duration 10

./python.sh scripts/run_llm_oracle_pipeline.py \
  --config configs/default.yaml \
  --planner scripted \
  --instruction "前往 search_area 搜寻移动目标，找到后跟踪十秒，然后返回 home 降落" \
  --takeoff-altitude 10 --track-duration 10 --headless
```

`--debug-ground-truth` 只把 evaluator 信息用于人工调试和只写 RMSE side-channel，不会加入 Planner prompt、YOLO 请求、`TargetEstimate` 的生产来源或控制输入。评测器按相机时间戳匹配异步估计与真值，只更新 `position_rmse_m` / `velocity_rmse_mps`，并且 `evaluate()` 固定返回 `None`。`oracle_evaluation` 上界模式也会显式启用同一评测通道；正常生产命令不要带这个选项。

### 命令组 2：普通 YOLO26 + BoT-SORT 生产路径

普通 YOLO 是闭集检测器，只能从加载后 `model.names` 中精确选择类别。本项目的双机 Cube 模式要求模型只报告 `cube`；`target_i` / `target_j` 只是路由 ID，red / blue 由独立 RGB-D 属性证据确认，颜色不会进入 YOLO `TargetQuery`。通用 COCO `yolo26s.pt` 不包含可用的 `cube` 类，不能直接用于该闭环，也不能通过模糊别名放宽成任意类别。

终端 A 启动单 worker、loopback-only 服务。模型路径只在这里出现；Qwen 占用 8000 时不会冲突：

```bash
cd /home/amax/ry/vlm_drones/uav_agent
export YOLO26_MODEL=/home/amax/ry/vlm_drones/models/initial_model/yolo26s.pt

CUDA_VISIBLE_DEVICES=1 \
conda run -n yolo_perception \
python scripts/serve_yolo.py \
  --config configs/yolo/service_yolo26.yaml \
  --host 127.0.0.1 \
  --port 8011 \
  --model-family yolo \
  --model "$YOLO26_MODEL" \
  --device 0 \
  --tracker configs/yolo/botsort_uav.yaml
```

终端 B 检查服务。真实 smoke 必须由操作者提供一张本地、已知含 `person` 的图片；脚本会 reset stream、连续发送两帧并核验响应与 track ID。`--allow-no-detections` 只用于验证协议，不等价于模型检测验收。

```bash
curl --fail http://127.0.0.1:8011/health
curl --fail http://127.0.0.1:8011/v1/model-info

conda run -n yolo_perception \
python scripts/smoke_yolo.py \
  --base-url http://127.0.0.1:8011 \
  --image /absolute/path/to/person.jpg \
  --class-id 0
```

终端 C 运行生产任务。便捷脚本默认补上 `production + ultralytics_service`，这里仍显式写出以便审计；绝不能添加 Oracle acknowledgement：

```bash
cd /home/amax/ry/vlm_drones/uav_agent
CUDA_VISIBLE_DEVICES=0 \
./python.sh scripts/run_yolo_pipeline.py \
  --config configs/yolo/runtime_yolo26.yaml \
  --planner dynamic_scripted \
  --perception-runtime-profile production \
  --target-perception-backend ultralytics_service \
  --yolo-service-url http://127.0.0.1:8011 \
  --yolo-request-timeout-s 0.5 \
  --yolo-max-result-age-s 0.5 \
  --instruction "起飞到十米，前往 search_area 搜寻一个人，确认以后跟踪十秒，最后返回 home 降落" \
  --headless
```

如果要由 Qwen 决定 Skill 顺序，把 `dynamic_scripted` 改为 `dynamic_llm`，并通过 `--base-url http://127.0.0.1:8000/v1 --model Qwen3-VL-4B-Instruct --api-key EMPTY` 指向已独立启动的 Qwen 服务。YOLO 仍只负责候选与跟踪，Qwen 不能输出飞行使用的三维坐标。

### 命令组 3：YOLOE 开放词汇路径

YOLOE 与普通 YOLO 共用相同 HTTP 协议、候选确认、深度几何、状态估计和 Skill 接口，但必须使用兼容 YOLOE 的本地权重及 `open_vocabulary` runtime 配置。不要把普通 `yolo26s.pt` 传给 `--model-family yoloe`，也不要宣称普通 YOLO 支持 `set_classes()`。

```bash
cd /home/amax/ry/vlm_drones/uav_agent
export YOLOE_MODEL=/absolute/local/path/to/yoloe-model.pt
test -f "$YOLOE_MODEL"

CUDA_VISIBLE_DEVICES=1 \
conda run -n yolo_perception \
python scripts/serve_yolo.py \
  --config configs/yolo/service_yoloe.yaml \
  --host 127.0.0.1 \
  --port 8011 \
  --model-family yoloe \
  --model "$YOLOE_MODEL" \
  --device 0 \
  --tracker configs/yolo/botsort_uav.yaml
```

另一个终端先检查模型信息，再运行 Agent：

```bash
curl --fail http://127.0.0.1:8011/health
curl --fail http://127.0.0.1:8011/v1/model-info

CUDA_VISIBLE_DEVICES=0 \
./python.sh scripts/run_yolo_pipeline.py \
  --config configs/yolo/runtime_yoloe.yaml \
  --planner dynamic_scripted \
  --perception-runtime-profile production \
  --target-perception-backend ultralytics_service \
  --yolo-service-url http://127.0.0.1:8011 \
  --instruction "起飞到十米，前往 search_area 搜寻一个人，确认以后跟踪十秒，最后返回 home 降落" \
  --headless
```

这个基础命令用无额外属性的类别提示验收开放词汇协议。YOLOE 的 `text_prompts` 只在目标提示改变时重新编码；带颜色、关系、排除条件或特定 identity 的候选还必须取得 typed Qwen semantic evidence，否则会安全地停留在 `CANDIDATE`。运行脚本会在显式启用 `--enable-qwen-vision --vision-review-mode gate --acknowledge-vision-gate` 时，将通过 routing、新鲜度和时序门的 typed review 接入 target coordinator；SEARCH 中只有同一 detector track 达到配置的最小观测数、持续时间且 bbox 无异常跳变后才会低频调用 Qwen，Qwen 不承担逐帧检测。正匹配和负匹配都需要同一 candidate ID 的重复时间证据，单次输出不能锁定或拒绝候选。仅仅把 Planner 改成 `dynamic_llm` 不会自动授权候选锁定。静态 ONNX/TensorRT 导出会固化动态提示，必须使用 `export_yolo.py --freeze-yoloe-prompts` 明确承认这一点，且不能把导出物描述成仍支持动态 `set_classes()`。

### 命令组 4：Isaac 标注、数据检查、训练、验证与导出

采集器与任务执行脚本完全分离。Cube v1 数据将 red / blue / green / yellow / gray 等所有立方体统一标为 class 0 `cube`，颜色只写入标量 metadata；`target_i` / `target_j` 不进入标签。一张图允许 0～3 个 cube，所有可见 cube 都必须标注，同时保留球、圆柱、长方体和彩色背景块等 hard negatives。它使用同一原子帧的同步 RGB-D、相机投影和仿真真值生成标签，并按真实 rendered bounds 计算 bbox。最终按 scene/episode/trajectory 分组后的 `train`、`val`、`test` 必须覆盖正负样本、red/blue cube、部分遮挡和多 cube 图像。两个显式 privileged 开关仅授权标签生成，不能流入 `MissionAgent`：

```bash
cd /home/amax/ry/vlm_drones/uav_agent
./python.sh scripts/collect_yolo_dataset.py \
  --config configs/default.yaml \
  --collection-config configs/yolo/collect_cube.yaml \
  --output /home/amax/ry/vlm_drones/datasets/perception/cube_v1 \
  --scene-seed 42 \
  --max-samples 2000 \
  --max-episodes 100 \
  --frames-per-episode 20 \
  --sample-hz 2 \
  --class-id 0 \
  --class-name cube \
  --oracle-label-generation \
  --acknowledge-privileged-oracle \
  --headless
```

训练侧只在 `yolo_perception` 环境中运行，不导入 Isaac Sim。数据检查要求 `train`、`val`、`test` 三个 split 都存在且数据集至少含一个正标注，并只读检查图像、标签、归一化框、重复样本、跨 split 哈希泄漏和类别分布，再执行 dry-run：

```bash
export DATA_YAML=/home/amax/ry/vlm_drones/datasets/perception/cube_v1/data.yaml
export YOLO26_MODEL=/home/amax/ry/vlm_drones/models/initial_model/yolo26s.pt

conda run -n yolo_perception \
python scripts/check_yolo_dataset.py --data "$DATA_YAML" --task detect --protocol cube-v1

conda run -n yolo_perception \
python scripts/train_yolo.py \
  --config configs/yolo/train_yolo26s_cube.yaml \
  --model "$YOLO26_MODEL" \
  --data "$DATA_YAML" \
  --device 0 \
  --dry-run
```

正式微调、验证及通过验证门控后的 ONNX 导出：

```bash
conda run -n yolo_perception \
python scripts/train_yolo.py \
  --config configs/yolo/train_yolo26s_cube.yaml \
  --model "$YOLO26_MODEL" \
  --data "$DATA_YAML" \
  --device 0 \
  --epochs 100 \
  --imgsz 960 \
  --batch 16 \
  --run-name yolo26s_cube_v1

export TRAINED_MODEL=/home/amax/ry/vlm_drones/outputs/perception/yolo/yolo26s_cube_v1/weights/best.pt
export VALIDATION_REPORT=/home/amax/ry/vlm_drones/outputs/perception/yolo/yolo26s_cube_v1/validation_report.json

conda run -n yolo_perception \
python scripts/validate_yolo.py \
  --model "$TRAINED_MODEL" \
  --data "$DATA_YAML" \
  --device 0 \
  --output "$VALIDATION_REPORT"

conda run -n yolo_perception \
python scripts/export_yolo.py \
  --model "$TRAINED_MODEL" \
  --validation-report "$VALIDATION_REPORT" \
  --format onnx \
  --device 0
```

ONNX 导出固定使用环境中的 `onnx==1.22.0` 并关闭 graph simplify，缺依赖会在 Ultralytics 运行前失败，绝不触发进程内自动安装。验证报告只有在常规指标和完整、有限的 small-object precision/recall/mAP50/mAP50-95 都可用时才会标记通过；Ultralytics 版本或模型不能提供 small 指标时必须明确失败，不能伪造聚合值或绕过 export gate。TensorRT 是明确的可选能力，不纳入可移植环境：使用 `--format tensorrt --device 0` 前须由操作者按当前 NVIDIA driver/CUDA 安装兼容 TensorRT Python runtime；CPU、不可见 GPU 或缺少 `tensorrt` 都会 fail-fast，脚本不会盲装平台包。

恢复训练必须显式传入具体 `last.pt`：`--resume /absolute/path/to/last.pt`，脚本不会猜“最近一次”运行。训练权重、数据集和输出分别放在仓库根目录被忽略的 `/models/`、`/datasets/`、`/outputs/`；不要提交或复制到源码包。验证通过后，停止旧服务并用新的 `best.pt` 作为 `serve_yolo.py --model` 重启服务。

每次生产任务结束会在终端输出有界 `TargetPerceptionMetrics`，包括请求/成功/超时/过期/丢帧、延迟、检测与候选确认、track 切换/碎片、可见率/丢失/重获、深度失败与测量年龄。`position_rmse_m` 和 `velocity_rmse_mps` 只允许 evaluator 侧在显式真值评测中填写，真值不得回流到控制；普通生产运行应保持为空。默认不保存连续图片或视频。只有显式设置 `debug_images.enabled: true` 时，生产 YOLO 路径才会在 mission run 目录的 `debug_images/target_perception/` 中保存代表帧；全局数量严格受 `max_images_per_run` 限制，并且 `first_detection`、`first_candidate`、`confirmation_success`、`candidate_rejected`、`target_lost`、`reacquire_success` 每类最多一张。每张图标注 bbox、class/class ID、confidence、track ID、candidate ID、confirmed、三维世界位置和 measurement age；`run_manifest.json` 只汇总实际图片的 count/bytes。该写图器不在 Oracle evaluation 路径创建，并会额外拒绝任何 Oracle source。

当前限制：服务第一版每进程只允许一个活跃 `mission_id:uav_id` stream，并在任务开始/结束时 reset BoT-SORT；双机配置固定使用 8011 / 8012 两个独立 worker。通用闭集 `yolo26s.pt` 没有 `cube`，生产成功验收必须由操作者自行训练并提供单类 Cube 权重，不能用 Oracle 掩盖这一限制。YOLOE 路径继续保留并要求操作者自行提供兼容权重。新 tracker ID 的重获不会仅凭类别继承身份；它需要确定性属性证据或通过 gate 的新鲜 typed identity review。Isaac 深度提供当前三维估计上界；真实相机、PX4/Pegasus/MAVSDK/ROS 2 的同步 RGB-D 与外参适配仍属于后续硬件集成。

## Stage 1A / 1B：MissionAgent + 可替换目标感知

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
SEARCH / TRACK / REACQUIRE     # 只读取中立 TargetEstimate
    ↑
Guarded target backend         # Oracle evaluator 或 YOLO/YOLOE production
```

旧的 `scripted` / `llm` 模式保持不变：Planner 输出严格五字段 `MissionIntent`，可信编译器生成固定六步 baseline。新增的 `dynamic_scripted` / `dynamic_llm` 显式启用受约束动态规划：Planner 输出 2～10 步的 `SkillPlanDraft`，可以按指令省略 SEARCH/TRACK、有限重复 GOTO/TRACK，并为 TRACK 选择是否启用有界 REACQUIRE。这个 Stage 1 基线只在 `MissionAgent.start()` 做初始规划；后文显式启用的视觉 gate、计划修订和障碍路线修订是在此基础上的独立运行时能力。

动态模式不会把飞控交给 Qwen。模型只看到场景总边界、默认高度/时长、Skill Catalog、调用上限、具名区域/降落区/导航点及文字描述，并只能填写高层参数；它看不到这些名称背后的具体坐标、搜索中心/半径、目标 spawn/真值、图像、速度向量、实际 max speed 或普通 Skill timeout。`PlanValidator` 才负责把名称解析为可信世界坐标、补充 motion policy/timeout/速度上限、检查引用与计划状态，并拒绝缺失 TAKEOFF/LAND、越界高度或非法顺序，而不是静默补步骤。

WorldContext 的 `search_area` 只从配置中的 `target.initial_region`、`search.radius_m` 和任务起飞高度构造，`home` 只取配置的 UAV 初始 XY 与地面高度。它不包含本次随机 Target spawn 坐标、Target 速度、`EvaluatorFrame` 或 Oracle Observation。`--debug-ground-truth` 仅允许把真值打印给人工调试/evaluator，不会把它加入 instruction、Planner prompt 或 GOTO 目标。

脚本保持 Isaac standalone 导入顺序：先解析参数、加载纯 Python 配置并创建 Planner 配置，随后创建 `SimulationApp`，最后才导入 Isaac-backed 环境模块。默认 `--headless` 适合服务器验收；需要 GUI 时使用 `--no-headless`。只有 `environment.step()` 产生新的同步 Camera sample 后才更新所选目标 backend、构造标准化 `Observation` 并推进 Agent。

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

当前能力边界必须明确：文本 Planner 仍只接收文字和受限 world context；`QwenVLMVerifier` 在显式启用后通过异步 worker 接收最多三张新鲜 Camera RGB 帧，并只返回严格的语义审查 JSON。它不是 detector、tracker 或 ReID，也不能输出可信世界坐标、速度或控制命令。SEARCH / TRACK / REACQUIRE 只消费 `TargetEstimate`；Oracle evaluator 与 YOLO/YOLOE production 分别生成该中立类型，skill 不读取 `oracle_target_*`、不调用 HTTP 或神经网络。

感知运行时默认使用 `PerceptionRuntimeProfile.PRODUCTION`。`GuardedPerceptionBackend` 会拒绝声明为 `PRIVILEGED_ORACLE` 的 backend，也会二次检查任何伪装成视觉 backend 却输出 `oracle_target_*` 的 Observation；`MissionAgent` 在 Safety 和 Skill 之前还有同样的 production gate。Oracle 只能在明确选择 `ORACLE_EVALUATION` 并设置 `acknowledge_privileged_oracle=True` 后运行，两个 Oracle demo 会在控制台打印醒目标记。该 profile 仅用于上界、回归测试、数据标注和专家轨迹，不是部署配置，也不能与真实视觉配置静默互换。

真实视觉 backend 已把 YOLO/YOLOE proposal、BoT-SORT 短轨迹、语义/身份 evidence、深度几何和有界状态估计接入既有确认边界：

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

`CandidateConfirmationCoordinator` 只接受带有限 timestamp、confidence、一致 candidate id、合法时间顺序和可实现轨迹时长的类型化 evidence；任一明确否定会清除 SEARCH 候选并回到 `SEARCHING`，REACQUIRE 候选被否定时则恢复原 target id 与 last-seen 状态并回到 `REACQUIRING`。证据不足时保持 `CANDIDATE`，短轨迹、语义、ReID 和时序一致性全部通过才写入非 Oracle 的 `confirmed_vision` lock。裸 `TargetManager.lock()` / `mark_reacquired()` 会拒绝直接调用，避免绕过 coordinator。`MissionAgent` 的 SEARCH/REACQUIRE→TRACK 转换在两种模式下都只验证 provider 已提交的 `LOCKED` 状态和 target id；它不会按模式直接锁定，也不会把 Skill 成功结果伪造成 `confidence=1.0` 的 Oracle lock。Oracle 的特权 shortcut 完全封装在明确标记且已确认授权的 evaluator provider 内。

## 本地 Qwen OpenAI-compatible 服务

`uav_agent/models/` 是纯 Python 客户端包，与仓库根目录存放权重的 `models/` 不同。客户端只使用标准库 `urllib`，默认访问 `http://127.0.0.1:8000/v1` 的 `GET /models` 与 `POST /chat/completions`。`LLMPlanner` 只发送文字；显式启用的视觉 smoke/动态审查入口才发送经尺寸和 JPEG 质量硬限制编码的 Camera RGB，且请求中永远没有 Oracle 字段或完整环境对象。可通过 `QWEN_API_BASE / QWEN_API_KEY / QWEN_MODEL / QWEN_REQUEST_TIMEOUT_S` 设置默认值，显式构造参数优先。

服务脚本不会安装 vLLM，也不会修改 `r_isaac_sim`。请先进入服务器上已有的兼容 vLLM 环境，或把 `VLLM_BIN` 指向该环境中的可执行文件，再运行：

```bash
cd /home/amax/ry/vlm_drones/uav_agent
QWEN_CUDA_VISIBLE_DEVICES=1 \
QWEN_MODEL_PATH=/home/amax/ry/vlm_drones/models/initial_model/Qwen3-VL-4B-Instruct \
./scripts/serve_qwen3_vl.sh
```

默认只绑定 `127.0.0.1:8000`。模型路径、served name、host、port、最大上下文、GPU memory utilization、CUDA device 和 vLLM binary 都可用清单中对应的 `QWEN_*` / `VLLM_BIN` 环境变量覆盖。另一个终端执行：

`QWEN_ADAPTER_CONFIG` 默认指向 `configs/adapters.json`。启动脚本只静态加载
其中 `status=active`、路径为真实目录且声明 rank 的 Adapter；没有 active
Adapter 时不会追加任何 LoRA 参数。`--max-lora-rank` 取 active Adapter 的实际
最大 rank，不使用猜测常量，也不开启运行时 Adapter 更新 API。

```bash
./python.sh scripts/check_qwen_server.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-4B-Instruct
```

检查器先调用 models endpoint，要求基础模型和所有 active Adapter 的 served
model 都存在；placeholder 不要求出现在服务端。随后发送一个最小文本
completion；普通失败只显示连接、HTTP 或协议错误类型，`--debug` 才附带已脱敏
traceback，API key 不会写入输出。

## 动态视觉审查、路由与受控恢复

三个感知角色必须分开理解：Qwen3-VL 负责低频语义审查；`OraclePerception` 只负责 `ORACLE_EVALUATION` 中的理想几何/实例上界；YOLO/YOLOE、BoT-SORT 与身份一致性 evidence 负责生产候选和图像运动。Qwen 的 bbox 始终是归一化图像坐标，不能被当作飞行坐标；生产三维位置来自同步深度与相机几何。Oracle pose/velocity 不会进入 Qwen prompt 或 YOLO 服务，日志也分别标记 semantic 与 geometry source，绝不把 Oracle 几何归因给 Qwen。

视觉审查默认关闭，显式启用后有两种模式：

- `shadow` 会真正异步调用 Qwen、打印并写入稀疏审查记录，但不能锁定/切换目标、修改 Skill、请求重规划或触发 HOVER。GOTO/SEARCH/INSPECT/TRACK 的普通周期审查不会让仿真 tick 等待 HTTP。
- `gate` 是实验控制模式，除 `--enable-qwen-vision` 外还必须给出 `--acknowledge-vision-gate`。单次视觉结果不会锁定/切换目标，也不会直接插入 `INSPECT` 或修改计划；候选驱动的控制建议至少需要同一可信 candidate ID 下的多次时间一致结果。由可信运行时产生的 typed `PATH_BLOCKED` 事件使用独立授权策略，不把 Qwen 自报建议当成权限。在没有真实 ReID 时系统不会伪造 ReID evidence。阻塞事件等待模型时由可信运行时插入 `HOVER`，模型只能建议有限动作，不能直接调用 controller。

`HOVER` 是可恢复的监督性暂停，不是硬安全机制：它每个 tick 都继续发送有界位置保持 setpoint，并有有限等待超时和可信 fallback。`qwen_visual_review.hover_position_tolerance_m`、`hover_max_correction_speed_mps`、`blocking_hover_timeout_s` 与 `blocking_timeout_fallback` 都由受限配置加载；模型无权设置这些参数。越界、非法 Observation、时间倒退等硬安全判定优先于任何模型请求，不等待 Qwen，直接走现有 `CANCEL_AND_LAND` 或 `ABORT`。

计划修订采用独立的第二阶段 Planner，视觉 review 本身不能夹带飞行计划。普通语义修订只能原子替换当前被中断步骤或未执行后缀，已完成前缀和 target immutable identity 不可改；routing ID、基础/新版本、步数与 revision 预算、cooldown、Symbolic checker、`PlanValidator` 和 Safety preflight 任一检查失败都保留原计划。默认最多三次修订；LAND 已开始后不接受普通语义修订。V3 的障碍专用修订使用独立多模态 schema，只允许在可信 `UAV_HOLD_FLU` 锚点中提出有限航点，并始终通过 RouteRegistry、可配置 Critic 和再次 Safety preflight；模型仍不能输出速度或控制量。

所有 Planner v2 step、Observation、review、event、invocation、execution report 和 transition 都带 `uav_id`，并与 `mission_id`、`plan_version` 一起逐边界精确比对。当前仍是“一架 UAV 一个 `MissionAgent`”，不是并发机群调度；属于 `uav_2` 的帧、review 或 revision 会被绑定到 `uav_1` 的 Agent 立即拒绝。

动态飞行 demo 的 `production + ultralytics_service` 路径要求独立 YOLO 服务已健康、runtime 配置与服务模型 family 一致；否则 fail closed，不会使用 Oracle。保留的 Oracle 路径要求三个显式值：`--perception-runtime-profile oracle_evaluation`、`--target-perception-backend oracle_evaluation` 和 `--acknowledge-privileged-oracle`。下面的 evaluator shadow 命令读取每个新鲜 Camera sample，将 RGB 送到后台 Qwen worker；主仿真 tick 不同步等待 HTTP，也不保存连续图片或视频：

```bash
./python.sh scripts/run_dynamic_visual_mission.py \
  --planner dynamic_llm \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-4B-Instruct \
  --api-key EMPTY \
  --uav-id uav_1 \
  --enable-qwen-vision \
  --vision-review-mode shadow \
  --perception-runtime-profile oracle_evaluation \
  --target-perception-backend oracle_evaluation \
  --acknowledge-privileged-oracle \
  --instruction "起飞到十米，前往 search_area 搜寻红色立方体，找到后跟踪十秒，返回 home 降落" \
  --headless
```

要实验 gate mode，把 `shadow` 改成 `gate` 并额外添加 `--acknowledge-vision-gate`。仅用于恢复路径测试的三个选项是 `--inject-path-blocked-at-s`、`--inject-progress-stall-at-s` 与 `--inject-identity-conflict-at-s`；它们产生的事件和 CSV 都明确写 `source=test_injection`，不伪装成 detector 或 Qwen 发现。启动时会打印 UAV/mission routing、Planner、视觉模式、感知 profile 与 Oracle acknowledgement。每次运行只在 `logs/dynamic_visual_missions/<mission_id>/` 写有界 `qwen_reviews.jsonl`、`mission_events.csv`、`skill_transitions.csv` 和原子更新的 `run_manifest.json`；manifest 汇总模型与配置、review 接受/过期/超时计数、修订次数、监督性 HOVER 次数/时长、终态和调试图片占用。默认不写 base64 prompt、原图或视频。

### Spatial Contract V3 与障碍路线实验

V2 保持默认兼容合同；只有显式传入 `--planning-contract v3` 才允许 Qwen 构造点、相对点和区域。V3 坐标均带 frame：`WORLD_ENU`（东、北、上）、`HOME_ENU`（home 原点，轴仍为 ENU）、`UAV_START_FLU`（任务开始机头前、左、上）、`UAV_HOLD_FLU`（障碍 HOLD 时机头前、左、上）以及仅供摄像机观测的 `CAMERA_FLU`。未绑定 pose 或未经过视觉 grounding 的相对对象会明确失败，不由编译器猜测“左边”的参考系。

SEARCH V3 支持 `CIRCLE / RECTANGLE / SECTOR / POLYGON / CORRIDOR / RELATIONAL`，以及 PERIMETER、割草、螺旋、扇区扫描、走廊跟随、随机覆盖和模型宏观观察点等策略。`ADAPTIVE_NEXT_BEST_VIEW` 仅在显式传入 `--enable-qwen-next-best-view` 时启用：每到达一个宏观观察点后，后台 Qwen worker 接收最新 RGB、覆盖率和可信区域边界，异步选择一个新的区域内 `WORLD_ENU` 观察点或返回 `EXHAUSTED`；它不输出 60 Hz 控制量，也不重写整份任务。连续 SEARCH 是同一个 Target 生命周期内的有限回退链，总步骤和总 SEARCH 时间仍受可信预算约束。

障碍物只来自 `configs/default.yaml` 的共享 registry。`ideal_camera` 是带 frustum、像素面积、遮挡、距离和 active-corridor 约束的 privileged 上界感知，不会把相机外的全局障碍表发给 Qwen。低层报告或 Qwen 疑似报告任一方都可立即触发 HOVER；只有 registry-backed camera geometry 才能启动坐标路线生成。模式差异如下：

- `open_sim` 只做结构检查，原样执行模型航点，并由运行时 swept-volume CollisionMonitor 记录碰撞后取消/降落；
- `critic_sim` 返回几何反例，模型最多自行修订三次，Critic 不生成替代航点；
- `strict` 拒绝不安全路线，预算耗尽或发布失败时可信取消/降落。

实验开关的默认值仍是 `v2 / linear / strict / disabled`。MissionProgram 的不可变图类型、线性适配器、纯 Python executor、事件边和 suffix-only `ProgramPatch` 已接入 `SkillManager` 调度；静态 `PATH_BLOCKED` 边或 Qwen patch 都先建立 supervisory HOVER，再原子发布新图版本，并拒绝修改已完成节点或前缀边。当前 CLI 只开放不含障碍 TaskPlan 重规划的 graph 运行；graph 障碍模式仍会在启动 Isaac/Qwen 前明确 fail-closed，因为图片与 grounded 障碍几何尚未接到 `ProgramPatchCoordinator`。默认不保存 RGB、原始帧或视频，模型原始结构化提案、Critic 结果和最终执行计划则写入有界稀疏日志。

在启动 Isaac demo 前，可先用一张用户指定图片验证真实 Qwen 多模态协议。该 smoke 只输出严格目标存在判断和归一化 bbox，不接触 Oracle 或 controller：

```bash
./python.sh scripts/run_qwen_vision_smoke.py \
  --image /absolute/path/to/frame.png \
  --uav-id uav_1 \
  --target-description "red cube" \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-4B-Instruct \
  --api-key EMPTY
```

## 场景与稳定 prim 路径

场景完全由本地 primitive 创建，不依赖在线 USD 资产：

- 白色平坦 Ground、DomeLight 和斜向 DistantLight；
- 由共享配置 registry 创建的带碰撞彩色固定 Cube 障碍物；
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

环境固定采用以下 barrier 顺序：先按稳定 ID 顺序更新全部 Target/UAV 运动学状态并写入 Xform，再执行唯一一次 `World.step(render=True)`，随后读取 World simulation time、全体 pose 和 Camera batch。Isaac 的多个 render product 底层统一按 renderer cadence 采集，并在所有 RGB/depth annotator 绑定后共同重置采样相位；Fleet 层再原子下采样到配置的公共 Camera frequency（默认 10 Hz）。发布前会同时严格比较每架 Camera 的原始 timestamp 和 ReferenceTime 帧 ID；跨 Camera skew、部分更新和超过一个 rendering dt 的 World/Camera 滞后都会 fail-closed，renderer timestamp 不会被改写。Isaac `NEW_FRAME` 回调正常比 `World.current_time` 落后最多一个 rendering dt，因此 Camera 时间是同步 observation 的规范时间。warm-up 或同步故障超过 2 个仿真秒会明确报错，避免 Agent 永久得不到 tick。非 Camera 采样 tick 仍发布 Fleet pose 供全局安全检查，但不会发布混时 `AgentObservation`。`get_camera_pose()` 仍可用于读取当前调试 pose，而同步观测应读取 `AgentObservation.camera_position_m`。CLI 使用 `--save-rgb` 或 `--debug-ground-truth` 时，可能在 `--steps` 之后继续推进最多一个 Camera 采样周期，以获得完整同帧数据；最终会打印实际 step 数。

## Phase 4：移动 Target

`env/moving_target.py` 同样是纯数学模块，支持：

- `STATIC`：保持位置；
- `LINEAR`：以固定速度直线运动，在边界反射；
- `RANDOM_WALK`：每隔配置时间随机改变 XY 速度方向，并在边界反射。

Target root 只在 `target.motion.region` 闭区间内运动，速度受 `target.max_speed_mps` 限制。随机序列使用配置 seed，可重复复现实验；默认测试覆盖连续五分钟的 60 Hz 随机运动且不越界。运动更新对共享 registry 中的 collidable AABB 加上 Target half extent 后做连续碰撞预测并反射速度，因此不会穿过固定障碍；region 约束的仍是 root。底层提供 `get_pose()`、`get_velocity()`、`reset()` 和 `step()`。

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

Phase 5 最初建立的 TAKEOFF、GOTO、SEARCH、TRACK、REACQUIRE 与 LAND 六个类型合同现在均已有 ideal-kinematic 实现；另外已有受信运行时使用的监督性 `HOVER`、只消费候选 ID 的理想 `INSPECT`，以及只通过 `RouteRegistry.route_ref` 消费已接受多航点路线的 `FOLLOW_ROUTE`。`ORBIT / FOLLOW_PATH / RETURN_HOME` 仍未作为独立 Skill 加入。

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

普通 `Observation` 的所有 legacy `oracle_target_*` 均为 `None`；production guard 也会拒绝任何夹带这些字段或 `source=oracle_evaluation` 的 estimate。只有 evaluator/test 显式调用 `get_skill_observation(include_oracle=True)` 才会取得 Target 真值，再由受保护的 Oracle backend 转为中立 `TargetEstimate`；SEARCH/TRACK/REACQUIRE 本身始终只读中立字段。`SkillContext` 只含结构化 `UAVController`、`CameraSensor`、perception 和 simulation clock，不含具体 `KinematicUAV` 类型、scene、target 或全局 Manager。

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

SEARCH 只读取与 RGB 同帧的中立 `Observation.target_estimate`。只有新鲜、可见且已经候选确认的 estimate 才返回 `SUCCEEDED + TARGET_FOUND`；未确认的 candidate 保持 SEARCH 运行，预测值也不能冒充一次新发现。Oracle evaluator 会把同帧真值适配为 `source=oracle_evaluation, confirmed=true` 的 estimate，以保留理想上界的快速成功；生产 estimate 则必须经过 detector、短轨迹与确认链。搜索航点始终只由区域几何生成，任何 estimate 都不用于偷取目标位置规划航点。六点全部扫描后返回 `FAILED + SEARCH_EXHAUSTED`，超时返回 `FAILED + TIMEOUT`。Skill clock 与 Observation timestamp 必须共享同一 simulation-time epoch；默认 10 Hz Camera、90° FOV 和 0.5 rad/s scan 在相邻图像间具有充分视场重叠，提高 scan rate 或降低 Camera frequency 时也应保持这一采样覆盖关系。

六个 waypoint 各扫描 360° 时，仅默认 `0.5 rad/s` 扫描就约需 75.4 s，尚未包含 waypoint 间移动。因此 `SearchGoal.timeout=60.0` 是单次请求默认值，不保证能够走完整条搜索轨迹；需要验收 `SEARCH_EXHAUSTED` 时应显式提供足够长的 timeout。默认 YAML 保留更长的 120 s，并可按场景半径继续调大。

## Phase 8：Ideal TRACK

TRACK 面向已经确认的单个目标，使用 `TargetEstimate` 的可信三维位置、可选速度与有界短时预测生成跟随位置，不调用 detector、HTTP、PID、MPC、RL 或 Qwen3-VL：

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

`desired_distance` 是 Target 与 UAV 的 XY 水平距离，`desired_altitude` 是 world-frame 绝对 Z；两者单位均为 m。首次收到确认的三维 estimate 时，TRACK 固定 Target→UAV 的水平相对方位，随后让该跟随位置随目标估计平移。每个 tick 同时下发 world-frame xyz 目标和动态 `FACE_POINT(current_target_position)` yaw，二者分别受 `max_speed` 和 UAV 硬件 yaw rate 限制。因此 UAV 可以侧飞、斜飞或后退，不需要先转完机头，也不会调用 `set_pose()` 瞬移。

TRACK 要求 estimate 已确认、target ID 匹配且包含可用三维位置；可见测量或状态估计器给出的有界 `predicted_only` 位置都可用于短时连续控制，但只有新鲜可见帧刷新 `last_seen_time / position / velocity`。last-seen age 以图像采集 timestamp 计算，迟到且已越过丢失 deadline 的帧不能复活 TRACK。不可见时间严格超过 `max_target_lost_time` 后返回 `FAILED + TARGET_LOST`，Result 保留最后一次真正可见的估计，不会被无限预测覆盖。可选 `timeout` 到期返回 `FAILED + TIMEOUT`；`track_duration` 到期返回 `SUCCEEDED + TRACK_COMPLETE`，而 `None` 保留无限跟踪行为。当多个 deadline 在一次低频采样中同时越过时，按绝对仿真时间最早发生者决定 ResultCode；同刻采用 `TRACK_COMPLETE > TIMEOUT > TARGET_LOST`。终态或 cancel 都会清除剩余平移和 yaw 命令。

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

REACQUIRE 只消费与 RGB 同帧、可见、已确认且 target ID 与 Goal 一致的 `TargetEstimate`；未确认候选、错误 identity 或纯预测 estimate 都继续搜索。Oracle evaluator 可直接生成符合该条件的理想 estimate，生产模式则必须完成候选确认和身份一致性检查。estimate 的位置只在成功帧写入 Result，不改变冻结的预测搜索中心。有效 deadline 帧优先于 timeout，正常失败只有 `FAILED + TIMEOUT`。成功后的 TRACK 切换必须由外部 `SkillManager`/Planner 在 reset 后发起，REACQUIRE 内部没有 Skill-to-Skill 调用。

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

`ScriptedPlanner` / `LLMPlanner` 继续返回 `MissionIntent` 并走固定模板。`ScriptedDynamicPlanner` / `DynamicLLMPlanner` 返回严格的 routed `SkillPlanDraftV2`：对象拒绝未知字段、重复 step id、bool 数字、NaN/Inf、未知 Skill/参数、前向或非 SEARCH 引用；LLM 首次输出不合法时最多修复一次，temperature 固定为 0。动态 `source` 为 `dynamic_scripted` 或 `dynamic_llm`。

历史 schema v1 只保留给旧数据集、Gold 评测和显式兼容读取。它不会在缺少路由信息时被新 Qwen 入口自动执行；可信运行时必须调用 `migrate_plan_v1_to_v2(old_plan, mission_id=..., uav_id=..., plan_version=...)`，明确提供三个路由值后才能形成 v2。新的 `DynamicLLMPlanner` 在模型调用前就拒绝没有 routing IDs 的 `PlannerRequest`，首次生成和 repair 均只发送 schema v2；不存在隐式补 `mission_legacy` 或默认 UAV ID 的 Qwen 生成路径。

`SkillPlanDraftV2` 是新动态模型唯一允许输出的协议；其中只保留 routing IDs、结构化目标语义、具名地点、高层参数和前序 SEARCH 输出引用，不含解析后的坐标、速度或普通 Skill timeout。例如导航任务可以省略 SEARCH/TRACK：

```json
{
  "schema_version": 2,
  "mission_id": "mission_001",
  "uav_id": "uav_1",
  "plan_version": 1,
  "target_spec": {
    "original_description": "unspecified mission target",
    "category": "unspecified",
    "hard_attributes": [],
    "soft_attributes": [],
    "negative_constraints": [],
    "relation_constraints": [],
    "query_ladder": [],
    "inspection_questions": [],
    "immutable_identity_summary": "unspecified mission target",
    "mutable_appearance_notes": []
  },
  "steps": [
    {"id": "takeoff_1", "uav_id": "uav_1", "skill": "TAKEOFF", "args": {"altitude_m": 8.0}},
    {"id": "goto_search", "uav_id": "uav_1", "skill": "GOTO", "args": {"destination": "search_area"}},
    {"id": "goto_home", "uav_id": "uav_1", "skill": "GOTO", "args": {"destination": "home"}},
    {"id": "land_1", "uav_id": "uav_1", "skill": "LAND", "args": {"zone": "home"}}
  ]
}
```

需要跟踪时，`TRACK.args.target_ref` 只能写成 `$<先前SEARCH步骤id>.target_id`。`REACQUIRE` 不占顶层步骤，只能附着在 TRACK 的 `recovery` 中，并由 `max_attempts` 和全局恢复预算共同限制。TRACK 还可用 `on_target_lost` 表达 `REACQUIRE` 或 `FAIL`；未写该字段时继承可信 `PlannerPolicy`，默认是 `REACQUIRE`。显式 `FAIL` 禁止同时携带 `recovery`，编译结果不含恢复策略；它表示目标丢失后任务失败并原地紧急降落，不表示条件式返航。当前协议没有实现 `RETURN_HOME`、无目标继续执行或运行时询问 LLM。

完整 Skill Catalog 注册 TAKEOFF/GOTO/HOVER/SEARCH/INSPECT/TRACK/REACQUIRE/LAND；REACQUIRE 标记为 recovery-only。初始 Planner 没有可信 CandidateBank ID，因此它的模型可见投影只开放 TAKEOFF/GOTO/HOVER/SEARCH/TRACK/LAND，并明确隐藏 INSPECT。只有受信 revision 触发携带一个已验证 candidate ID 时，revision Catalog/Schema 才开放 INSPECT，且 `candidate_id` 被绑定为该 ID 的 `const`，模型不能自行编造。默认可信限制为最多 10 个顶层步骤、5 次 GOTO、1 次 SEARCH、2 次 TRACK、每个 TRACK 最多 2 次恢复、总计最多 4 次恢复，TRACK duration 为 1～600 s；可在 `configs/default.yaml` 的 `planner` 段收紧。`PlannerPolicy` 另行保存默认目标丢失动作及可信的恢复次数、半径和 timeout。模型可以选择允许的动作或给出有界覆盖，但不能修改这些边界。历史 v1 仍只是单目标有限线性计划；v2 的运行时 revision 也只能原子替换受限后缀，不是自由分支图或多目标 Planner。

### Structured output、符号检查与可信编译

`DynamicLLMPlanner` 的首次生成和唯一一次 repair 都使用同一份 `SkillPlanDraftV2` JSON Schema structured output。请求通过 OpenAI-compatible `response_format.type=json_schema` 发送；若服务端不支持并返回错误，客户端会明确失败，不会静默降级为自由文本。初始 Schema 用 `oneOf` 区分六种允许的顶层 Skill；revision Schema 只在有可信候选时增加受 const 约束的 INSPECT 变体。两者都只暴露 world context 中的具名区域、降落区和导航点枚举，不包含这些名称背后的坐标，也不包含 Oracle 数据、速度或底层控制参数。受约束生成之后仍执行严格 JSON、duplicate key、有限数值、dataclass 和 Catalog 校验，不能把 JSON Schema 当作唯一信任边界。

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

原有 Planner v1 数据及解析接口保持不变。若要检查带 `mission_id`、`uav_id`、`plan_version`、逐 step `uav_id` 和结构化 `TargetSpec` 的动态 schema v2，可从同一份可信 Gold 生成独立的 pilot 预览目录：

```bash
./python.sh scripts/generate_planner_dataset.py \
  --config resources/planner_v1/dataset_config.yaml \
  --output-root /tmp/planner_v2_preview \
  --seed 42 \
  --profile pilot \
  --schema-version 2 \
  --uav-id uav_1
```

该路径不调用 Qwen、不修改 `datasets/planner_v1/`，也不伪装成已人工审核的正式 v2 数据集；当前只开放 pilot 预览，`full` v2 发布流程会明确拒绝。

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

`configs/default.yaml` 管理：scene 大小、UAV 初始位置/最大速度/最大 yaw rate、Camera resolution/frequency/FOV/focal length/pitch、Target 初始区域/最大速度/运动模式与边界、Search，以及默认关闭的 `target_perception`。`configs/yolo/runtime_yolo26.yaml` 与 `runtime_yoloe.yaml` 是完整、可审计的生产配置，不是隐式 overlay；模型权重路径只属于独立服务配置/CLI。加载器在启动昂贵的 Isaac Sim 之前完成未知键、类型、有限值、单位、空间边界、profile/backend 组合、detector family/proposal mode、loopback URL、统一 physics/render tick 和 Camera frequency 校验；当前运动学环境要求 physics/render dt 相等。

episode reset 必须调用 `environment.reset(target_seed=...)`，它会一起重置 World、UAV、Target 和 Camera observation cache；业务代码不要直接调用 `environment.world.reset()`，否则会绕开运动控制器状态。

## 目录职责

```text
uav_agent/
├── configs/       # 公共 schema、统一 YAML 配置和纯 Python 校验器
├── env/           # scene、UAV、同步 RGB-D CameraSample、moving Target、World wrapper
├── models/        # 纯 Python 模型合同与 OpenAI-compatible HTTP 客户端
├── agents/        # VLM/LLM Agent
├── experiments/   # 轻量运行目录、CSV/TensorBoard、checkpoint、评测与图表
├── skills/        # 统一 Skill API、MotionPolicy、Manager 与分型 Goal 合同
├── perception/    # Oracle/YOLO backend、候选确认、深度几何与 TargetEstimate
├── yolo_service/  # 隔离的 FastAPI、严格协议、Ultralytics 与 BoT-SORT
├── training/yolo/ # 数据检查、Isaac 标签采集接口、训练、验证与模型 registry
├── planner/       # 任务分解与层次规划
├── tasks/         # 独立 Gold Spec、封闭目标 ontology 与 Intent Judge
├── planner_data/  # Planner v1 渲染、生成、分割、验证、泄漏检查与离线评测
├── resources/     # Planner v1 ontology、公开 world、中文词表和模板配置
├── runtime/       # MissionIntent 校验与确定性 TaskPlan 编译边界
├── prompts/       # Prompt 模板
├── scripts/       # mission、Oracle/YOLO 服务、数据采集、训练与评测 CLI
├── tests/         # 快速纯测试及一个显式 opt-in Isaac 集成测试
├── logs/          # 运行日志（默认忽略产物）
├── python.sh      # 默认 r_isaac_sim，支持通过环境变量覆盖 prefix
├── environment.yml      # Python 3.11 / Isaac Sim 5.1 Agent 环境基线
├── environment-yolo.yml # 独立 YOLO 推理/训练环境
├── requirements/yolo.*  # 独立 YOLO 输入依赖与精确 lock
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

## 多无人机 Fleet Planner 与运行时

多机入口保持两级规划边界：Fleet Planner 只输出 UAV—目标—区域
assignment 与协调策略；每个 assignment 再独立进入现有 Spatial Contract
V3，并由各自的 `MissionAgent`、`SkillManager`、`TargetManager` 和
`SafetySupervisor` 执行。共享层只维护 target claim、无图像的
`FleetWorldBelief`、空域冲突和模型请求优先级，不合并本地计划版本，也不向
任一局部 Planner 暴露其他目标的图像或 Oracle 真值。

Fleet Runtime v1 只执行 `PARALLEL` assignment；`SEQUENTIAL` 仍保留为未来
契约枚举，但 request、JSON Schema、Planner 输出和 runtime 任一边界遇到它都会
在 Isaac 启动前明确拒绝。每个 required target 必须恰好由一个 assignment 覆盖，
或以 `target_alias: reason` 形式明确列入 `unassigned_requirements`；后一种计划完成
已有 assignment 后最多只能得到 `PARTIAL_SUCCESS`，不会误报 `SUCCEEDED`。
模型可自由生成的 Fleet `RegionSpec` 只接受 `WORLD_ENU` 或 `HOME_ENU`；这一限制
不仅写在 structured-output JSON Schema 中，正式 parser/validator 也会独立复核，
因此不遵守 Schema 的客户端不能用 `CAMERA_FLU`、`UAV_START_FLU` 等 frame 绕过
可信空间边界。

纯 Python Scripted 演示不会导入 Isaac Sim：

```bash
./python.sh scripts/run_fleet_planner_demo.py \
  --config configs/multi_uav_demo.yaml \
  --fleet-planner scripted \
  --local-planner dynamic_scripted \
  --planning-contract v3 \
  --adapter-config configs/adapters.json \
  --instruction "无人机A前往世界坐标二十、三十附近十五米范围搜索并跟踪目标i二十秒；无人机B前往世界坐标负二十五、十附近十二米范围搜索并跟踪目标j二十秒；完成后分别返回各自起点降落"
```

连接本地 Qwen 时，将两个 Planner 参数换成 `llm`/`dynamic_llm`，并传入
`--base-url http://127.0.0.1:8000/v1 --model Qwen3-VL-4B-Instruct
--api-key EMPTY`。所有 Planner 调用、严格 schema 校验、本地 V3 编译、感知
preflight 和 Safety preflight 都在 `SimulationApp` 之前完成；失败不会启动
Isaac。

两机两目标 Oracle 评估入口必须同时给出 profile 与特权确认：

```bash
./python.sh scripts/run_fleet_mission.py \
  --config configs/multi_uav_demo.yaml \
  --fleet-planner scripted \
  --local-planner dynamic_scripted \
  --planning-contract v3 \
  --runtime-program linear \
  --adapter-config configs/adapters.json \
  --perception-runtime-profile oracle_evaluation \
  --acknowledge-privileged-oracle \
  --headless \
  --max-sim-time 300 \
  --instruction "无人机A前往世界坐标二十、三十附近十五米范围搜索并跟踪目标i二十秒；无人机B前往世界坐标负二十五、十附近十二米范围搜索并跟踪目标j二十秒；完成后分别返回各自起点降落"
```

GUI 模式使用 `--no-headless --debug-visualization`，并按服务器设置 `DISPLAY`
和 `XAUTHORITY`。Oracle 实例严格绑定 `(uav_id, assigned_target_id)`；production
模式不会接受特权确认。启用 `--enable-qwen-vision` 时，纯 Python preflight 会为
每个 assignment 构造隔离的 `RUNTIME_VISUAL_REVIEW` 角色客户端；运行时客户端只会
通过一个共享 `GlobalModelRequestBroker` dispatcher 暴露给协调器，不能绕过全局
inflight、per-UAV 限流和替换策略。可信 scheduler 将事件/阻塞审查标为 P3，将普通
周期审查标为 P4；prompt 内容不能自行提权。

Fleet 日志位于 `logs/fleet_missions/<fleet_mission_id>/`，包含 manifest、Fleet
plan、assignment CSV、模型选择/调用 CSV、空域冲突、每机本地计划与真实
transition/review/revision JSONL，以及最终 summary。日志层拒绝 Camera RGB、
base64 图像、API key 和 Oracle 图像旁路。

终态会重写 `assignments.csv`，因此其中保存的是最终 assignment 状态；完整事件只
保存在 `fleet_events.jsonl`，`summary.json` 仅保留有界终态快照和
`event_count`，不会再次复制整条事件流。终态 JSON/assignment CSV 使用临时文件
原子替换；相同 mission ID 的目录已存在时会 fail-fast，绝不把新 JSONL 混入旧 run。
异常、中断和 cleanup 失败会保留原始退出语义，凭据形状的错误文本在写盘和 stderr
前统一脱敏。

`model_calls.csv` 对真实 LLM 请求逐次记录 assignment/UAV 路由、实际
prompt/completion token、延迟、finish reason 和错误码；一次 repair 会形成另一条
真实记录。Scripted Planner 没有模型请求，对应角色只写明确的
`finish_reason=not_called` 占位行，不再把 Adapter selection 伪装成一次零耗时调用。
Broker 中只有显式标记为 `replaceable` 的普通视觉请求可被 P0/P1 抢占；抢占立即写
`PREEMPTED_BY_HIGHER_PRIORITY` 的 STALE 记录，迟到的模型结果也只能返回该 STALE
记录，不能重新成为有效控制输入。Fleet 视觉 dispatcher 已负责 Broker 的
submit/acquire/complete，并将 COMPLETED、FAILED、pending STALE 和抢占后的迟到
STALE 各写一次 `model_calls.csv`。底层客户端不再重复记账。未来若增加另一类并发
runtime consumer（例如真正的 Fleet replan），必须把它注册到同一个 acquire owner，
不能与 dispatcher 竞争读取同一 Broker 队列。

多机 home 的可信 landing tolerance 会在纯 Python 预规划中按最小间距推导；
无法同时保留控制容差与安全余量的 home 布局会在 Isaac import 前失败。空域层仍
记录进入警戒带的 `AIRSPACE_CONFLICT`，但只有预测/当前突破硬最小间距、活动路线
交叉或共用着陆区才立即 HOLD，避免两个彼此安全但处于警戒半径内的独立 home
形成永久着陆死锁。`predicted_collision_time_s` 表示首次进入硬最小间距球的时间，
不是更晚的最近接近时刻。全局 cancel 已让 `MissionAgent` 进入可信 fail-safe LAND
后，Runtime 不再因同一空域 HOLD 跳过其 tick；它保留原冲突记录并额外写一次
`AIRSPACE_HOLD_OVERRIDDEN_FOR_FAILSAFE_LAND`，明确表示紧急有界下降优先，避免
持久冲突把 LAND 永久冻结。

Fleet 调试绘制只写 viewport debug-draw：不同 UAV 使用确定性颜色和独立轨迹，
同时绘制 assignment 搜索区域、目标 claim 连线、最小安全距离圆与空域冲突线；
配套 overlay snapshot 携带 assignment/semantic alias、Fleet plan version 和每机
local plan version。其 API 不接收 Camera RGB，因此这些调试线不会进入机载图像。

### Adapter、数据与 LoRA 占位脚手架

`configs/adapters.json` 注册 `fleet_planner`、`spatial_mission`、
`runtime_visual` 和 `runtime_replanner` 四个角色槽位。当前全部为
`placeholder`，因此明确回退到基础 Qwen；只有 `status=active`、真实路径存在、
包含 regular non-symlink `adapter_config.json` 和至少一个 `.safetensors`、base
lineage 一致且声明真实 rank 的 Adapter 才会加入 vLLM 静态启动参数。
仓库不会创建随机、零权重或其他伪造 Adapter，也不包含 `safetensors`。

数据生成、校验和 Gold 离线评测均为纯 Python：

```bash
./python.sh scripts/generate_fleet_planner_dataset.py \
  --output datasets/fleet_planner_v1 --seed 42 --overwrite
./python.sh scripts/validate_fleet_planner_dataset.py \
  --dataset-root datasets/fleet_planner_v1
./python.sh scripts/evaluate_fleet_planner.py \
  --gold datasets/fleet_planner_v1/test_iid.jsonl
```

普通样本的 input/output 直接使用正式 `FleetMissionRequest` 与
`FleetMissionPlan` 契约；`test_reassignment` 中，规划前已不可用 UAV 仍使用普通
`FleetMissionPlan`，只有执行中 assignment 失败的重分配样本使用
`FleetPlanPatch`，不会把两种生命周期伪装成同一种 patch。七个 split 各含两条
样本并完整覆盖十四类场景，包括显式/别名分配、五类 RegionSpec、能力与可用性、
目标冲突、任务/UAV 数量不等和失败重分配。manifest 的 schema、seed、生产契约、
split count 与每个 JSONL 的 SHA-256 都是 validator 的强制输入；缺场景、内容篡改
或 manifest 漂移都会返回非零。别名样本的真实
`original_instruction` 覆盖“第一架无人机”“左边那架无人机”
“速度较快的无人机”和“带高分辨率相机的无人机”，且不通过
`requested_uav_id` 提前泄露别名答案。

离线字段指标按稳定的 `target_alias` 语义做一一匹配，不使用模型可自由生成的
`assignment_id` 对齐；合法更换 assignment ID 不影响 UAV、目标、区域或时长准确率。
重复 target alias 只有在合法共享目标且 UAV 路由可唯一对应时才会匹配，其他重复或
歧义输出一律保守计为未匹配，避免评测器事后挑选得分更高的 duplicate claim。
prediction 中出现 Gold split 之外的 sample ID 会被直接拒绝；missing-task coverage
只按唯一、可信 assignment 或精确 `target_alias: reason` 计算，重复 assignment、
重复/无关 note 与无效 UAV 都不能改善指标，所有 rate 保持在 `[0, 1]`。

能力样本把 `required_payload:<capability>` 放在正式 `TargetSpec.hard_attributes`
中；生产 Fleet validator 会将完整 TargetSpec 绑定回请求，模型不能删除或改写该
要求。当前生产契约尚未定义通用能力表达式求解器，因此 UAV payload 是否满足该
命名空间由 Fleet dataset v1 validator 额外校验；这是一条显式的数据层边界，不应
误称为通用 runtime capability policy。

LoRA 训练脚手架仅验证配置与数据，不加载 Qwen、不训练、不创建权重：

```bash
./python.sh training/lora/train_fleet_planner_lora.py \
  --config configs/lora/fleet_planner_lora.json
```

真实 module 检查与未来训练必须在独立 `qwen_lora` 环境完成，不能修改
`r_isaac_sim` 的 Transformers/vLLM 依赖；具体边界见
`training/lora/README.md`。

## 测试约定

一次修改会话结束时再集中运行测试，避免重复启动 Kit。以下 Fleet 清单回归全部为纯 Python，不导入 Isaac Sim：

```bash
./python.sh -m pytest -q \
  tests/fleet \
  tests/test_dynamic_mission_agent.py \
  tests/test_dynamic_llm_planner.py \
  tests/test_spatial_types_v3.py \
  tests/test_mission_program.py
```

真实 Isaac smoke test 只在模块全部完成后运行一次。如果首次启动提示 NVIDIA Omniverse EULA，需由账户使用者本人按服务器规范接受；项目不会自动代替用户设置接受标志。

Phase 10 的完整 Oracle 集成默认由 unittest 跳过，避免普通回归启动 Kit。只在一次修改会话全部结束时显式运行：

```bash
UAV_AGENT_RUN_ISAAC_TESTS=1 \
  ./python.sh -m unittest tests.test_full_skill_pipeline_oracle -v
```

该测试只创建一个 `SimulationApp`，使用 KinematicUAV、RGB Camera、MovingTarget、OraclePerception 和 SkillManager；不导入或调用 Qwen、LLM、VLM。
