# 多无人机本机异步修复：基线记录

记录日期：2026-09-15。参考提交：`b2aec6472a541f0531523043faed01a231d99586`。
开发开始时该提交为实际 HEAD，工作区干净；没有执行 checkout、reset、commit 或 push。
已检查仓库及本次修改目录的开发约定，没有发现适用的 `AGENTS.md`。
所有新增文件、隔离快照、测试缓存和临时目录均在 `/home/amax/ry/vlm_drones` 内。

## 基线的真实调用链

```text
scripts/run_fleet_mission.py: prepare_fleet_mission
  → FleetPlanningService / mission interpreter
  → FleetTaskSpecV1
  → LLMFleetPlannerV2 → FleetMissionPlanV2（分配 goal_id）
  → FleetAssignmentCompiler.compile_assignment_v2
  → DynamicLLMPlanner → SkillPlanDraftV3（本机详细技能）
  → GoalSatisfactionChecker + PlanValidator + SafetySupervisor
  → RoutedPreplannedSpatialPlanner → MissionAgent → SkillManager

运行时原有换机路径：
FleetMissionRuntime.tick
  → _service_pending_reassignments
  → 同步调用 runtime replan handler
  → Fleet V2 模型调用 + 本机 Spatial V3 模型调用、编译、准备
  → FleetReplanPublication → _publish_fleet_replan
```

初始化规划在启动飞行循环前完成，允许同步调用模型。原运行时换机 handler 内的模型调用
则位于 `tick()` 调用栈中，慢请求会影响健康无人机继续推进。

原生产 handler 处理失败 assignment 的目标子集，只接纳一个替代 assignment；它主要从
未占用的可用无人机中选择备用机。它并不是每次都重新生成全队详细计划。
原 `source_v2.goal_ids` 只是该 assignment 的目标清单，并未通过可信执行结果扣除已完成目标。
原 `_subset_task_spec_for_replan()` 只留下两端均位于子集内的顺序边，会遗漏跨子集条件。

## 契约与已接通能力

| 能力 | 基线事实 |
|---|---|
| Fleet V2 | `FleetMissionPlanV2` 表示目标分配；它不是单机 `SkillPlanDraftV2` |
| 单机空间规划 | Fleet 的详细计划为 `SkillPlanDraftV3`，编译为线性 `TaskPlan` |
| 旧异步计划修订 | `PlanRevisionCoordinator` / `planner.revision` 已有路由、异步调用、前缀保护和接纳机制，输出契约主要为单机 V2；不能将 V3 改标签后直接复用 |
| 图执行 | 已有单独图程序和补丁机制，但不代表 Fleet 的所有 V3 路径都可执行任意任务图 |
| 失锁 | 原有 TRACK/REACQUIRE 及安全策略存在；原累计跟踪时间不等于连续有效视觉观测时间 |
| 模型资源 | 已有 Broker、dispatcher、角色注册和独立 client；视觉请求受其优先级权限约束 |
| 共同悬停参考 | 单机 dynamic 障碍修订入口已在稳定 HOLD 之后采样位姿，共用该轮 resolver；不能由此推断 Fleet 已全部接入 |
| LoRA | 使用已有角色路由；本次没有新增训练，`runtime_replanner` 占位及回退策略不应称为已训练修复能力 |

## 测试基线及环境差异

测试解释器为 `/home/amax/miniconda3/envs/r_isaac_sim/bin/python`，pytest 8.4.1。
默认 `/home/amax/miniconda3/bin/python` 缺少 pytest，因此首次默认解释器检查未运行任何测试。
辅助解释器检查还遇到 PIL 缺失；这些环境错误不计作代码基线失败。
测试统一设置 `PYTHONDONTWRITEBYTECODE=1`，并显式指定项目内的 `cache_dir` 和 `--basetemp`。

| 运行 | 已实际观察到的结果 | 解释 |
|---|---|---|
| 修改前的目标、任务、空间 resolver、旧修订测试 | **52 passed** | 本机契约子任务的直接基线 |
| 从仓库根目录运行主代理选择的基线集合 | **551 passed，4 failed** | 相对配置路径与测试预期工作目录不一致，不能归因于新增实现 |
| 隔离 HEAD 快照，从其 `uav_agent` 目录运行同一 555 项集合 | 补齐模型软链接后 **554 passed，1 failed** | 剩余失败依赖被 Git 忽略的数据；不是本次新代码失败 |
| 补齐数据软链接后的隔离基线 | **555 passed in 4.87s** | 主整合者已完成并确认；无代码基线失败 |

隔离源码快照位于 `outputs/dev_local_repair/baseline_snapshot`。
快照中的模型和数据软链接只读复用本机已有资源，没有修改权重或数据内容。
不能拿不同工作目录或缺资源的运行冒充代码回归。

555 项隔离基线选择集合为 `tests/fleet`、`tests/test_plan_revision.py`、
`tests/test_plan_revision_coordinator.py`、`tests/test_reacquire.py`、`tests/test_async_worker.py`。
工作目录为 `outputs/dev_local_repair/baseline_snapshot/uav_agent`，使用上述测试解释器，
`PYTHONPATH=.`，缓存与临时目录显式放在项目的 `outputs/dev_local_repair` 下。

本机契约子任务实际运行的命令：

```bash
cd /home/amax/ry/vlm_drones
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=uav_agent \
  /home/amax/miniconda3/envs/r_isaac_sim/bin/python -m pytest -q \
  -o cache_dir=outputs/dev_local_repair/contracts_cache \
  --basetemp=outputs/dev_local_repair/contracts_tmp \
  uav_agent/tests/fleet/test_goal_checker.py \
  uav_agent/tests/fleet/test_task_spec.py \
  uav_agent/tests/test_spatial_resolver.py \
  uav_agent/tests/test_plan_revision.py
```

## 本次改造与基线的分界

启用 `fleet_recovery` 后，新增 `FleetRecoveryController` 持有 recovery episode，
Broker 管理的文本任务执行模型调用及纯编译；运行所有者在串行边界完成资源准备、
活状态复核和发布。`scripts.run_fleet_mission._build_fleet_recovery_controller`
由生产 `main` 装配，初始 Agent 与换机后的 Agent 均开启本机修复入口。

关闭新增开关时保留原有入口及 scripted/Graph 行为。原同步换机 handler 仍是兼容路径，
因此“运行中模型计算移出主循环”的结论限于已启用的新恢复路径。

本文不报告真实 Qwen、YOLO、Isaac 或飞行实验成功；本轮已核实的执行证据来自 Python 测试。
仓库既有历史实验记录不能替代本次恢复功能的真实运行验收。

## 最终核验补记

隔离原 HEAD 基线为 **555 passed**；全 Python 集合实际为 **2859 passed, 1 skipped,
2 deselected in 114.36s**。最后补充范围/期限日志后，全部相关集合为 **741 passed in 5.67s**。
全量唯一跳过项为显式启用的 Isaac 集成；两个未选择项为真实 YOLO 和 Qwen LoRA 实验。
没有未解决的已运行测试失败，没有安装项目外依赖。详细命令、19 类场景映射、
新增问题的修复及首轮限制见 [使用与验收](fleet_local_repair.md)。

端到端验证修复了一个原备用机接口问题：编译/发布已推进局部版本，但 MissionAgent.start
原先固定从 1 开始，下一帧被 Fleet 判定版本回退。现由 Runtime 将已发布版本传入
新 Agent，默认初始版本仍为 1；原机修复不调用 start。
