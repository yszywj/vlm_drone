# 无障碍四机四目标规划测试

另有固定任务模板的 [2→4→6→8→10 机递增测试](fleet_scaling_to_10.md)，用于单独观察数量增加的影响。

场景配置：[`configs/multi_uav_open_4x4.yaml`](../configs/multi_uav_open_4x4.yaml)。
场地为 100 × 100 × 30 米，`scene.obstacles: []`；四架无人机各有独立相机，
四个彩色立方体以 0.5 m/s 在各自中心附近的小范围内随机运动。起点相距至少 60 米，
机间最小安全距离仍为 5 米。

| UAV | 起点 / home (m) | 基准目标 | 搜索中心 (m) | 半径 (m) | 跟踪 (s) |
|---|---|---|---|---|---|
| A / uav_a | (-30, -30, 0) | 红色 target_i | (-20, -20, 0) | 8 | 15 |
| B / uav_b | (30, -30, 0) | 蓝色 target_j | (20, -20, 0) | 10 | 20 |
| C / uav_c | (30, 30, 0) | 绿色 target_k | (20, 20, 0) | 8 | 25 |
| D / uav_d | (-30, 30, 0) | 黄色 target_l | (-20, 20, 0) | 10 | 30 |

坐标使用 WORLD_ENU。表中的 z=0 是搜索区域中心，飞行高度由技能规划和验证器确定。
基准自然语言任务保存在 [`resources/fleet_open_4x4/mission.txt`](../resources/fleet_open_4x4/mission.txt)，
四机并行执行，完成后各自返航降落。

## 测试范围

Oracle 感知用于提供确定的目标观测，从而减少检测、颜色确认、深度估计对规划评估的干扰。
任务解释器、Fleet Planner 和每机 Spatial V3 Planner 仍使用真实 Qwen 模型。
这三个规划阶段的输入是文字和结构化信息，因此此测试衡量任务理解、分配及技能计划，
不直接衡量 Qwen 的图像理解能力。

当前运行时每机同时只支持一个目标；多目标排队、跨机先后依赖或同步执行不属于本套评分范围。
自主分工接受任意完整的一一映射。初始 Fleet 请求不提供各机起点坐标，故不以最短路径或最近分配评分。
交叉分配主要用于纯规划压力测试；无障碍物仍可能存在机间空域冲突。

## 五组独立语义检查

任务与人工预期值位于 [`configs/benchmarks/fleet_open_4x4.json`](../configs/benchmarks/fleet_open_4x4.json)。

| case | 变化 |
|---|---|
| explicit_4x4 | 完整显式分配，四组搜索区域和跟踪时长 |
| shuffled_aliases | 保持相同任务，打乱叙述顺序，混用登记名称与 ID |
| unit_conversion | 跟踪时长为半分钟、四分之三分钟、一分钟、一分半钟 |
| cross_assignment | C→i、D→j、A→k、B→l，避免按列表顺序默认配对 |
| open_assignment | 模型自行给四机分配四个目标，每机恰好一个 |

Gold 只供评估器核对，不送入模型。检查覆盖目标遗漏/重复、机与目标绑定、圆形区域、
跟踪时长、返航降落及编译后的技能计划。原始模型响应、调用 token/延迟、各阶段诊断和
修复记录分别保存；首轮通过与修复后通过分开计算。服务错误和输出长度耗尽单独标记。
五例各一次只是初步探测，不足以给出模型普遍能力上限；重复运行和等价改写用于进一步验证。

## 纯规划测试（不启动 Isaac）

以下命令从 `uav_agent` 目录执行。

```bash
./python.sh scripts/evaluate_fleet_planning_stress.py \
  --suite configs/benchmarks/fleet_open_4x4.json --validate-only

./python.sh scripts/evaluate_fleet_planning_stress.py \
  --suite configs/benchmarks/fleet_open_4x4.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-4B-Instruct \
  --repeat 1
```

可用 `--case explicit_4x4` 只测试一例；`--repeat 3` 重复评估。
默认输出目录为仓库根下 `outputs/fleet_planning_stress/`，每次创建独立结果目录。
模型身份按 `configs/adapters.json` 路由；其中 placeholder adapter 会明确回落到基模。

默认沿用现有阶段输出预算：解释器 3072、Fleet 2048、局部计划 768 token。
如出现 `finish_reason=length`，应以更高预算单独重测，例如：

```bash
./python.sh scripts/evaluate_fleet_planning_stress.py \
  --suite configs/benchmarks/fleet_open_4x4.json \
  --base-url http://127.0.0.1:8000/v1 \
  --interpreter-max-tokens 6144 --fleet-max-tokens 4096 --local-max-tokens 1536
```

服务上下文必须容纳输入加输出。仓库启动脚本默认上下文为 4096；多机测试可在启动自己的
服务时设置 `QWEN_MAX_MODEL_LEN=16384`，并在比较结果时固定、记录相同服务配置。
不要把上下文不足或输出截断直接归结为模型推理失败。

## 运行四机仿真

先使用确定性基线检查场景和技能执行：

```bash
bash scripts/run_open_fleet_scene.sh \
  --fleet-planner scripted --local-planner dynamic_scripted --headless
```

使用 Qwen 完成解释、分配和局部规划后执行：

```bash
bash scripts/run_open_fleet_scene.sh \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-4B-Instruct --headless
```

此场景启动器将解释器/Fleet 输出预算设为 6144/4096，避免四机任务在原有较小预算下截断。
通用 `run_fleet_mission.py` 的默认值仍为 3072/2048；现在可显式使用
`--interpreter-max-tokens` 和 `--fleet-max-tokens` 调整，后者也用于运行时 Fleet 重分配。
使用启动器的扩大预算时，模型服务上下文应至少为本文示例的 16384。

在已配置 DISPLAY/XAUTHORITY 的桌面会话中，将 `--headless` 换成
`--no-headless --debug-visualization` 可查看四机、目标、搜索区域和轨迹。
可追加 `--instruction "你的完整四机任务"` 更换指令。
启动器默认上限为 300 仿真秒，并显式选择 Oracle 评估模式；这是本场景的感知实验设置。
闭环结果使用生产 Fleet 日志，纯规划 Gold 得分与实际飞行成功率应分别查看。

## 本机首次实测（2026-09-11）

Scripted + Oracle 场景验收为四机全部成功、全部返航降落、0 碰撞、0 越界，
最终 `strict_success=true`、退出码 0。新增场景暴露的多相机资源销毁顺序问题已修复；
Fleet 回归测试 445 项通过。

Qwen3-VL-4B-Instruct 基模在五组四机任务中，默认预算和扩大预算的完整语义通过数均为 0/5。
扩大预算消除了输出截断，剩余问题包括重复任务 ID、证据文本不符合契约、遗漏分配约束、
漏分配返航降落目标。双机对照也未通过，不能由本次结果认定四机是数量上限。
逐次模型输出、评分、实验条件和场景布局见本机
[实验报告](../../outputs/fleet_open_4x4/report.md)。
