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

- 2026-09-17 时的边界；2026-09-18 起配置开启后已支持有界 2~3 UAV 联合修复（见下节），仍不做全 Fleet 自动重规划。
- 无共享目标注册表接入：TRACK/INSPECT 的移交一律按 SAME_UAV_ONLY 拒绝
  （REQUIRES_SHARED_EVIDENCE 预留给未来的可信共享证据源）。
- 重投影只重接"当前位置→既有世界折线"的接入段，不在代码内绕开新障碍；
  遇到新障碍仍判 INVALID 并走新一轮候选。
- WAIT(HOVER) 部分时长没有可信部分账本，中途故障即 INSUFFICIENT。
- 不支持 Graph 计划、多目标别名混排、一次移交多替换机的 publication。

## 2026-09-18 增量：Contract Registry 与有界小组联合修复

### Contract Registry（`fleet/contract_registry.py`）

- `SkillContractEvaluator`：每个 Skill 的终态证据语义（成功码、目标身份要求、
  时长账本）；`TrackSkillContractEvaluator` 原样承载 TRACK 有效时长/连续时长账本。
- `GoalContractEvaluator` 基类提供
  `evaluate_evidence / compute_progress / compute_remaining / restart_policy /
  transferability / required_resources / validate_completion`；
  NAVIGATE/SEARCH/TRACK/INSPECT/WAIT/RETURN_HOME/LAND/RHAL 各一个 evaluator。
- `GoalContractRegistry` + `DEFAULT_GOAL_CONTRACT_REGISTRY`：`assess_remaining_goals`
  与 `build_remaining_task_contract` 改为查表；未注册（如 REPORT）显式
  `UNREGISTERED_GOAL_CONTRACT:<type>` fail-closed。对于已有执行能力的新任务语义，
  只需新增/注册 Contract Evaluator；若新增全新的物理 Skill，仍需实现 Skill、输入
  schema、执行结果证据和必要编译支持，但无需修改 FleetRecoveryController、
  RemainingTaskContract、空间/依赖/联合恢复主框架。`FleetRecoveryController`
  已无任何 goal_type 分支（测试做源码级断言）。
- 链路不变：Execution Evidence → Contract Registry → RemainingTaskContract
  → Recovery/Repair；TRACK 连续/有效时长与 handoff 约束语义逐字保留。

### 有界相关小组联合修复（`recovery_controller._tick_joint` 等）

- 触发：单机候选提交 guard 报 `SHARED_SPACE_CONFLICT/COORDINATION_REQUIRED`
  且受影响集合（含故障机）在 `[2, max_joint_repair_scope_uavs]`（2..3）内、
  每个 peer 健康（RUNNING、GOTO 转移边界、无在途 episode）→ 同一 episode 进入
  `JOINT_*`；超限或 peer 不合适 → 原有安全退出，绝不全队重规划。scope 为
  可靠超集，不声称最小。
- 请求：`JointRepairRequest`（scope 内每机冻结 `LocalRepairContextV3`——含各自
  RemainingTaskContract/前缀/anchor/依赖；组外 UAV 只读路线）。响应 schema
  `additionalProperties:false` 且键恰为 editable UAV；`parse_joint_repair_drafts`
  严格比对键集合（`JOINT_SCOPE_MUTATED`）。
- 验证：逐机 `validate_local_repair` 全流水线 + `_check_joint_live`
  （故障机全量 `_check_local_live`；peer 版本/步骤/锚点/依赖重验；组内组外
  统一空域与障碍检查 `ROUTE_CONFLICT`）。
- 提交（四阶段，软件状态与执行释放彻底分离）：`PREPARE → FINAL GUARD →
  SOFTWARE COMMIT（全 scope 统一）→ EXECUTION RELEASE（逐机）`。
  - PREPARE 对 scope 内全部 UAV 完成所有可失败准备（TaskPlan/compiled_mission/
    Skill 与 Goal 解析/前缀保护/版本/当前步骤/契约/空间与依赖检查，含保留步骤
    Goal 预解析），期间不修改任何计划、版本、路线、assignment 记录或执行状态。
  - 提交锁内先对全 scope 重跑 final guard，再对全 scope 统一验证 prepared
    绑定（uav/版本/步骤/执行代次/事件 + **重算当前 candidate 摘要**与 PREPARE
    摘要比对，不一致即 `ROUTING_MISMATCH`）；任一失败则全 scope 零变更。
  - SOFTWARE COMMIT：`apply_prepared_state` 只切换软件状态（Manager TaskPlan/
    计划索引/账本、Agent 计划状态与版本、AssignmentRecord.local_plan_version/
    路线/进度/编译），不取消/启动任何 Skill；正常路径下全 scope 一次性完成，
    不存在"UAV2 已切换、UAV3 仍旧状态"的中间态。
  - EXECUTION RELEASE：软件状态全部一致后才逐机 `release_prepared_execution`
    （取消旧 Skill=真实取消证据，启动已发布计划的保留步骤）。Skill 启动失败走
    既有可信 `skill_start_failed` 失败路径（安全降落），已发布软件版本不回滚，
    其他 UAV 的已提交计划不撤销。
  - 软件状态原子性 ≠ 物理动作可回滚：peer 旧 Skill 的取消/新 Skill 的启动是
    物理执行，一旦发生不回滚。`_rollback_joint_peers`（原后缀+版本递增并同步
    记录/编译）仅保留为"全 scope 绑定验证通过后、两次内存状态交换之间出现真
    正意外异常"这一理论窗口的异常兜底，正常 prepare/guard/software commit
    路径绝不调用，也不是 all-or-none 的实现机制。
- 配置：`joint_repair_enabled`（默认关）、`max_joint_repair_scope_uavs`（2..3）、
  `max_joint_peer_drift_m`。

### 本轮边界

- 联合修复一次一搏：候选被拒即安全退出，不在组内自动重试。
- peer 只接受 GOTO 转移边界的协调；SEARCH/TRACK 进行中的无人机不参与。
- 突变阶段（Skill 启动失败）的 peer 补偿以“原后缀+版本递增”恢复；该路径
  未在测试中强制触发（验证失败路径已覆盖）。
- 未支持：REPORT、多别名混排、Graph 计划、>3 机小组、INSPECT 初始计划
  （仅可信运行时修订后可见）。
