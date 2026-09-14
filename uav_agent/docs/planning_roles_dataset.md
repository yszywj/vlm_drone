# 三角色规划数据集

2026-09-12 生成的首批数据位于 `/home/amax/ry/vlm_drones/datasets/planning_roles_v1`。
它包含 **1000 个独立任务底稿、8000 条角色训练/评估样本**，无模型生成答案、无权重训练。

| 角色目录 | 标签契约 | 训练 | 验证 | 测试 |
| --- | --- | ---: | ---: | ---: |
| `mission_interpreter` | `FleetTaskSpecV1` | 800 | 100 | 100 |
| `fleet_planner` | `FleetMissionPlanV2` | 800 | 100 | 100 |
| `spatial_mission` | `SkillPlanDraftV3` | 4800 | 600 | 600 |

五类任务各 200 个：指定世界坐标导航、起飞原地悬停、搜索后返航、搜索后跟踪再返航、多机混合分工。
每类均包含 2 / 4 / 6 / 8 / 10 机，每个类型与规模组合 40 个底稿，按 32 / 4 / 4 划分。
场景上下文仅提供无障碍空间边界和各机 home；导航/悬停任务没有语义目标，搜索系列最多涉及 10 个目标。
混合任务独立组合各机任务类型，目标数随实际搜索任务变化。

`tasks.jsonl` 保存中文原文与独立底稿；三个角色目录分别包含 `train.jsonl`、`validation.jsonl`、`test.jsonl`。
每条角色样本的 `messages` 包含实际生产规划器构造的 system/user 提示及完整 assistant JSON 标签。
其余字段是审计元数据，不应输入模型。训练仅对 assistant 答案计算监督损失。
`manifest.json` 记录划分、来源与数据哈希、目标数分布、长度和检查结果；数据目录下的 README 提供使用说明。

## 标签与检查

先采样底稿，再生成中文及正确结构化答案；生产模型客户端被离线标签回放客户端替代。
由真实 Interpreter/Fleet 解析器、V3 局部规划器及 FleetAssignmentCompiler 验证答案。
独立检查器另从底稿核对逐机归属、目标、坐标、搜索半径、精确时长、动作先后及本机降落位置，
不以同一标签生成器的输出作为核验期望值。

所有 WAIT-only 任务仅抽取用户要求的等待语义。当前局部执行契约仍要求 TAKEOFF、HOVER、
本机 home GOTO、LAND；后两步标为 `runtime_contract_closure`，不虚构 TaskSpec 返航目标，
也不计为编译器事后补齐。其他任务由原文明确要求返航降落。

按整个任务底稿划分，所有角色投影跟随同一 split。另对解释器可见的指令语义去重，
防止只改变隐藏 home/default altitude 或中文表述就跨 split 重复相同任务。
五档既有压测任务没有导入此数据集，继续作为额外回归检查。

## 长度与接入

使用本地 Qwen3-VL-4B-Instruct tokenizer 的实际 chat template 计数；完整序列及分开编码的
prompt/completion 均检查，超限直接拒绝，绝不截断 JSON。

| 角色 | 最长完整序列 tokens | 最长答案 tokens | 超过当前默认输出预算的样本数 |
| --- | ---: | ---: | ---: |
| Interpreter | 6109 | 4516 | 146 / 1000（默认 3072） |
| Fleet | 6092 | 615 | 0 / 1000 |
| Spatial | 3643 | 372 | 0 / 6000 |

答案计数包括 chat template 的结束标记。生成器检查上限为 16384；本批完整样本均低于 8192。
8192 可作为本批数据长度配置的候选，实际训练批量及显存仍需实测。推理时需同步考虑服务上下文与输出预算，
尤其 Interpreter 默认 3072 无法容纳上述 146 条完整 gold；不能将截断导致的失败直接归为规划能力不足。

旧 `training/lora/dataset.py` 面向 Fleet V1；现已增加 `roles_dataset.py` 与
`train_planning_role_lora.py` 接入三种消息契约，collator 只监督完整 assistant 并拒绝超长样本。
首轮训练已启动，见 [训练运行说明](planning_role_training.md)；`adapters.json` 尚未启用新权重。

## 重现

在 `uav_agent` 目录运行，输出必须是一个不存在的新目录；已有数据不会被覆盖。

```bash
/home/amax/miniconda3/envs/qwen_vllm/bin/python scripts/generate_planning_roles_dataset.py \
  --output ../datasets/planning_roles_v2 --count 1000 --seed 42

/home/amax/miniconda3/envs/qwen_vllm/bin/python scripts/generate_planning_roles_dataset.py \
  --validate-only ../datasets/planning_roles_v1

/home/amax/miniconda3/envs/r_isaac_sim/bin/python -m pytest tests/planning_data -q
```

复验会重新读取磁盘样本，核对哈希与划分、回放所有答案、重算逐条及聚合 token 统计。
源码或 tokenizer 变化会要求重新生成并审阅数据。

本批是多任务正样本初版，只训练明确分工，每机一个 assignment，每个 assignment 至多一个语义目标。
尚无自由分配优化、失联/故障重规划、澄清、已锁定目标直接追踪、跨机同步或多目标串行调度样本。
训练与测试共享类型、模板池及机数，不能据此宣称未见模板或十机数量泛化。
标签通过检查也不等同于 Qwen 的规划成功率、仿真任务成功率或机间避碰验证。
