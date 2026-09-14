# 规划 LoRA 正式接入与定向数据

2026-09-13 接入、2026-09-14 完成验证：首轮三个 rank-16 LoRA 已接入正式角色路由。
权重版本仍为 `roles_r16_20260912T133403Z`；本次接入不改变模型权重。
新增数据和 Interpreter 重训配置已准备。2026-09-14 已在 GPU 2 启动下一轮训练：
`tmux attach -t planning_lora_v2`，run ID 为 `interpreter_v2_r16_20260913T161534Z`。
训练状态见 `../outputs/lora/planning_roles/mission_interpreter_hardening/interpreter_v2_r16_20260913T161534Z/status.json`，
日志见 `../outputs/lora/launches/interpreter_v2_r16_20260913T161534Z/mission_interpreter.log`。
正式 GPU 0 服务继续使用首轮权重；新权重完成后还需评测再决定是否切换。

## 正式配置与服务

`configs/adapters.json` 的 `mission_interpreter`、`fleet_planner`、`spatial_mission`
均为 active，并配置：

```json
"generation": {"json_schema_property_order": "alphabetical"}
```

`ModelClientFactory` 在对应 active 角色调用时，使用不可变副本排序 JSON Schema 的 properties。
required、enum、oneOf 的顺序、字段约束及模型提示均保持一致。未声明策略的 adapter、基模和
placeholder fallback 保留原字段顺序。角色调用日志保存策略及保序的 `generation_options_json`。

训练标签按字母序序列化，原推理 Schema 的顺序不同，会导致目标顺序反向、重复目标和漏分配。
上一轮相同权重的对齐补测为全链 95/100、10 机 18/20；这不是新一轮训练成绩。

当前本机服务：

```text
tmux: qwen_planning
API: http://127.0.0.1:8000/v1
GPU: 0
BF16 / context 16384 / max sequences 32
```

本地模型客户端对 localhost、127/8 和 ::1 直接连接，远程地址继续采用原代理配置。
正式 Interpreter 和规划入口的默认输出预算已提高到 6144；Fleet 为 2048，单机为 1024。
`serve_qwen3_vl.sh` 默认采用 BF16、16384 上下文；仍支持相应环境变量覆盖。

进入当前服务窗口：

```bash
tmux attach -t qwen_planning
```

已通过正式 CLI 的 4 机对照调用（仅生成全局分配方案，不启动飞行）：

```bash
cd /home/amax/ry/vlm_drones/uav_agent
/home/amax/miniconda3/envs/r_isaac_sim/bin/python scripts/plan_fleet.py \
  --config configs/benchmarks/fleet_scaling/open_4x4.yaml \
  --instruction-file ../outputs/planning_integration/integration_20260913T153948Z/formal_control_4x4_instruction.txt
```

### 本次实际验证及已知失败

使用正式 AdapterRegistry/ModelClientFactory，固定 6144/2048/1024 输出预算、零修复次数，
原有 2/4/6/8/10 机扩容回归首轮 5/5 通过。40 次调用均命中对应 LoRA，无服务错误或截断；
30 份单机原始输出均包含返回自己起点和降落，未依赖编译器补齐。
这是五个固定搜索跟踪案例的生成、语义和编译验证，不是飞行成功率。
正式 `plan_fleet.py` 对照调用返回 `proposal_ready`；该工具只生成全局方案，
返回的 `executable` 仍为 false，不代表已验证单机计划或原文意图完整性。

另测现有 `configs/multi_uav_open_4x4.yaml` + `resources/fleet_open_4x4/mission.txt`，
结果为 `incomplete_assignment`（退出码 2）：Interpreter 识别了四个返航降落目标，
但在 assignment_constraints 和 track→home 顺序边中遗漏它们；Fleet 随后漏分配四个目标。
服务保留语义校验并拒绝执行。相同系统提示、模型、字段顺序和预算下，回归文本通过、
该文本失败，说明现有模型仍有表达泛化问题，不能把 5/5 推广为任意任务均可靠。
旧 Fleet 训练输入中的返航目标均同时出现在分配约束和顺序边中，模型对不完整上游表达
的鲁棒性仍需单独加强。失败原始输出保存在 `formal_tool_plan_4x4.json`，未改写为成功，
也未把该原题加入新训练集。本次准备的训练配置仍仅针对 Interpreter。

接入验证阶段自动化回归：585 项通过。新数据的完整重放和训练入口预检查另行通过；
这些是训练启动前的检查结果，本轮训练状态见文首。

当前服务停止后，可使用已保存的启动脚本重新启动：

```bash
bash /home/amax/ry/vlm_drones/outputs/planning_integration/integration_20260913T153948Z/serve_planning.sh
```

该脚本使用本地 qwen_vllm 环境，显式指定 GPU 0、端口 8000、BF16 和离线模型文件。
应在端口和 GPU 可用时使用。端口 8001、8011 的既有服务未被替换。

## 新数据

目录：`/home/amax/ry/vlm_drones/datasets/planning_roles_v2_intent`。

|任务来源|train|validation|test|
|---|---:|---:|---:|
|保留的 v1 任务|800|100|100|
|新定向任务|400|100|100|
|合计|1200|200|200|

共 1600 个任务、12800 条角色样本。Interpreter/Fleet 各 1200/200/200 条，Spatial 为
7200/1200/1200 条。所有角色都保留完整投影，下一轮训练仅需更新 Interpreter，另两个 adapter 沿用。

新任务覆盖五类重点及 2/4/6/8/10 机：

- 混合 WAIT：单个 WAIT 不建立自环顺序边。
- 零坐标轴：区分 X/Y、零值和正负号。
- 混合归属：打乱条款顺序，完整保留 SEARCH、TRACK、NAVIGATE 与对应 UAV。
- 时长对照：区分 20 秒和 120 秒。
- 分钟换算：在 WAIT 和 TRACK 合法时长范围内覆盖分钟、秒转换。

新任务与旧全部 1000 个任务之间、以及新任务自身之间，两种语义哈希均去重。
旧任务的 ID、蓝图和 split 原样保留，失败测试原题没有加入 train。
新三分区使用独立模板池及坐标小数取值，仍属于合成任务族，不能视为开放任务泛化证明。
新 test 的 ID 单独记在 `manifest.curriculum.new_held_out_test_task_ids`；不应用其成绩挑 checkpoint。

所有标签通过生产解析、编译、独立蓝图语义和新任务原文数值审计；完整序列按本地 tokenizer
重新计数，不截断。全部样本均适合 8192 的训练上限，具体长度见 manifest 的 training_length_audit。
这些验证不包含新权重生成成绩、飞行仿真或实际跟踪执行。

旧数据文件和 manifest 保持原样。因为 Interpreter 默认预算与日志代码发生变化，当前源码直接
检查旧 v1 会报告版本不一致。原 manifest 所要求的 113 份源码已在改动前逐哈希核对并保存于
本次产物的 `v1_source_snapshot/`；v2 用当前代码重新渲染全部新旧任务，并记录双方来源。

重放完整新数据检查：

```bash
/home/amax/miniconda3/envs/qwen_vllm/bin/python scripts/generate_planning_roles_curriculum.py \
  --validate-only ../datasets/planning_roles_v2_intent
```

## 下一轮 Interpreter 训练

配置：`configs/lora/mission_interpreter_hardening_train.json`。
保留 r16、alpha32、学习率 1e-4、3 epochs、梯度累积 16、BF16、完整 assistant 标签监督。
这是从冻结基模训练一个新的独立 LoRA；现有入口不支持从旧 adapter 续训。
计划 225 个优化步，按首轮相近长度估计约 65 分钟；实际时间和最长样本显存以运行检查为准。

已经检查训练数据入口和全部 train/validation token 长度，现已按文首 run ID 启动训练。
若之后复现实验，应使用另一个新 run ID 和届时空闲的 GPU。示例：

```bash
CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
../.venvs/planning_lora/bin/python training/lora/train_planning_role_lora.py \
  --config configs/lora/mission_interpreter_hardening_train.json \
  --role mission_interpreter --run-id NEW_UNIQUE_RUN_ID
```

新 adapter 完成后应先比较旧回归、新 validation 与独立新 test，再切换正式 Interpreter。
评测脚本已改为读取实际 AdapterRegistry，通过生产 factory 调用 LoRA，并动态读取测试集数量；
不再依赖旧的 100 任务/800 样本常量。

本次日志和实际生产入口检查保存在：
`/home/amax/ry/vlm_drones/outputs/planning_integration/integration_20260913T153948Z`。
