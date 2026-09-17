# 剩余任务契约、空间参考有效性与跨机依赖统一判定

2026-09-17 增量。在既有 `assess_remaining_goals()` / `RemainingGoalAssessment` /
`RepairAnchor` / `ExternalDependencySnapshot` 之上，统一三个机制的单一实现，
供"本机修复"和"备用机接管"共同消费。本文是机制说明；操作命令见
[fleet_local_repair.md](fleet_local_repair.md)。

## 修改文件及职责

| 文件 | 职责 |
| --- | --- |
| `uav_agent/fleet/local_repair.py` | 全部纯契约：`RemainingTaskContract`、`SpatialReferenceValidity`、`CrossUAVDependencyCheck` 及其唯一构建/判定函数；`check_external_dependencies` 降级为兼容包装 |
| `uav_agent/fleet/recovery_controller.py` | 三个机制的消费与最终 compare-and-commit；本机修复与换机接管都改为调用同一构建函数 |
| `uav_agent/configs/schema.py` | `FleetRecoveryConfig.valid_pose_tolerance_m`（VALID/REPROJECTABLE 分界，默认 0.05m，必须 ≤ `max_hold_drift_m`） |
| `uav_agent/tests/fleet/test_remaining_task_contract.py` | 三个机制的纯函数测试（新增） |
| `uav_agent/tests/fleet/test_recovery_controller.py` | 重投影提交、无关版本变化、共享通道争用三个控制器级测试（新增） |

## 数据结构与调用链

```
SkillManager 终态证据 (SkillExecutionEvidence)
  └─ assess_remaining_goals()                # 唯一剩余量计算（不变）
       └─ build_remaining_task_contract()    # 唯一契约构建
            ├─ LocalRepairContextV3.remaining_task_contract   # 本机修复
            ├─ FleetRecoveryController._prepare_reassignment  # 换机接管
            └─ FleetRecoveryController._goal_state            # 舰队级目标状态
```

`RemainingTaskContract`（每 goal 一条 `GoalTaskContract`）：

- `goal_id / goal_type / target_binding`（目标身份=target_alias）
- `original_condition`（原始完成条件：空间约束、duration、distance、strength）
- `completion_basis`（valid_execution / continuous）
- `confirmed_amount + evidence_refs`（已确认完成量及其终态 invocation）
- `remaining_amount / remaining_goal`（剩余义务；时长目标的剩余秒数）
- `restart_policy`：COMPLETED / CONTINUE / RESTART / CANNOT_RESUME
- `transferability`：SAME_UAV_ONLY / TRANSFERABLE / REQUIRES_SHARED_EVIDENCE
- 证据不足条目显式 `status=INSUFFICIENT` 并携带 reasons，整体 fail-closed

`evaluate_spatial_reference(anchor, ...)` 由代码判定并返回
`VALID / REPROJECTABLE / INVALID`（含可信错误码与 reasons）；
`reproject_world_route()` 仅把当前位置接回既有 WORLD 折线，锚点之后逐点保持不变。

`check_cross_uav_dependencies(...) -> CrossUAVDependencyCheck` 返回
`LOCAL_OK / COORDINATION_REQUIRED(affected_uav_ids, dependency_ids, reasons) / INVALID`，
输入为期望/当前依赖快照、版本表与可信空域给出的航迹冲突对。受影响集合是
可靠超集，不声称数学最小。

## 两类消费者如何共享契约

- 本机修复：`LocalRepairContextV3.remaining_task_contract` 在上下文内派生，
  `validate_local_repair` 与 `_check_protected_semantics` 只读它。
- 换机接管：`_prepare_reassignment` 调用同一个 `build_remaining_task_contract`
  （`consumer="HANDOFF"` 仅用于日志），按 `transferability` 过滤可移交目标，
  把 `pending_goals/pending_goal_ids` 原样交给 replan boundary。
- 两侧都不再各自实现剩余量计算；`consumer` 参数不改变任何数值。

## 可信代码与 Qwen 的职责边界

可信代码完成：证据归类与剩余量、restart/transfer 判定、时间/时间域/版本/
漂移/障碍/跨机航迹冲突判定、重投影接入、依赖判定与最终 compare-and-commit。

Qwen 仅获得：`trusted_repair_context`（含只读 `remaining_task_contract` 视图、
冻结 anchor、授权后缀原文），并被要求输出固定 envelope 的 Spatial V3 后缀。
线路模式 `additionalProperties:false` 且 envelope 为常量，模型无法通过输出改写
契约、锚点、目标身份或他机计划；`validate_local_repair` 再独立复核
版本/空间/依赖/期限。重投影路径完全不调用 Qwen。

## 测试

```bash
cd /home/amax/ry/vlm_drones/uav_agent
./python.sh -m pytest tests/fleet/test_remaining_task_contract.py -q   # 22 passed
./python.sh -m pytest tests/fleet/test_local_repair.py \
    tests/fleet/test_recovery_controller.py -q                         # 81 passed
./python.sh -m pytest tests/fleet -q                                   # 665 passed
```

任务清单场景对应：30s 累计 20s 剩余 10s；continuous 失锁不继承；已完成目标
不重放；证据不足 fail-closed；小幅位姿变化重投影且不重新请求模型；
版本/时间/新障碍/航迹冲突判 INVALID；外部前置未完成返回 COORDINATION_REQUIRED；
共享通道仅一个提交者；无关无人机版本变化不废弃修复；模型改写契约/锚点/目标/
他机被拒。

## 当前仍未支持

- COORDINATION_REQUIRED 只输出受影响集合与原因，不自动小组联合重规划。
- 无共享目标注册表接入：TRACK/INSPECT 的移交一律按 SAME_UAV_ONLY 拒绝
  （REQUIRES_SHARED_EVIDENCE 预留给未来的可信共享证据源）。
- 重投影只重接"当前位置→既有世界折线"的接入段，不在代码内绕开新障碍；
  遇到新障碍仍判 INVALID 并走新一轮候选。
- WAIT(HOVER) 部分时长没有可信部分账本，中途故障即 INSUFFICIENT。
- 不支持 Graph 计划、多目标别名混排、一次移交多替换机的 publication。
