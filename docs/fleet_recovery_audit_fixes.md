# 多无人机异步恢复：审查修复记录

日期：2026-09-16。所有代码和文档修改位于本仓库；未进行 Git 提交、推送或历史修改。
本轮继续原任务的首轮范围：线性 Spatial V3 本机修复、异步单任务接管、可信证据及最终接纳。
自动 Group Coordination 属于原任务明确划出的后续独立阶段，本轮没有用空接口宣称实现。

## 修复内容与代码位置

| 审查问题 | 实际修改 | 主要文件 |
|---|---|---|
| 换机后 HOME_ENU / UAV_START_FLU 相对目标被备用机起点重新解释 | 在 owner 快照阶段用原机可信 home/start 转为 WORLD；后续修复及再次接管保留该任务版本；终止动作仍使用执行机自身 home | `fleet/handoff.py`、`scripts/run_fleet_mission.py`、`fleet/recovery_controller.py` |
| 运行期区域元数据与实际编译坐标不一致 | 发布时从 runtime 自己保存的原起始上下文独立计算 WORLD 区域，校验候选并更新请求元数据 | `fleet/runtime.py` |
| 命名地点被错误解释为地面 GOTO | 命名导航目标使用原机编译计划中唯一可信的实际导航位置；缺少或存在歧义时拒绝接管 | `fleet/handoff.py` |
| TRACK 按经过时间完成或扣减剩余时长 | 新增 `track_progress.v1` 执行证据；分别记录 `elapsed_s`、`valid_execution_s`、`continuous_execution_s`；重复、预测和失锁观测不增加有效时长 | `skills/track.py`、`skills/manager.py`、`fleet/local_repair.py` |
| 连续跟踪无法贯通任务到执行 | `completion_basis` 支持 `valid_execution`（默认）与 `continuous`，贯通 TaskSpec、模型输出 schema、技能目录、覆盖检查及编译；连续要求不能靠多个短 TRACK 相加 | `fleet/task_spec*.py`、`planner/schemas.py`、`planner/json_schema_v3.py`、`planner/goal_checker.py`、`runtime/plan_validator.py`、相关 prompts |
| TRACK 成功后的返航 GOTO 不能本机修复 | 区分暂存的目标成功与任务终止；进入合法修复时清除暂存成功，保留 TRACK 执行证据 | `skills/manager.py` |
| 合法本机 assignment 缺少源文本引用时被当成外部冲突 | 由 owner 生成真实分配证明；源文本引用单独保存；标明 LOCAL/EXTERNAL 和约束强度；MUST 冲突、未知或重复归属仍拒绝 | `fleet/local_repair.py` |
| 绕行候选通过验证却不能提交 | V3 与 Manager 都要求保留当前步骤 ID/skill，只允许在其前面插入新 ID 的 GOTO；由已验证 V3 路径显式开启该权限 | `fleet/local_repair.py`、`agents/mission_agent.py`、`skills/manager.py` |
| 并发接管中一个版本变化令另一个永久失败 | 过期候选丢弃，以新 request ID 重新取得 owner 快照；仍受同一个 episode 的次数、冷却和墙钟期限限制 | `fleet/recovery_controller.py`、`scripts/run_fleet_mission.py`、`configs/schema.py` |
| 一旦有完成目标就拒绝全部接管 | 对证据充分、剩余目标可独立执行且包含导航目标的情形，扣除已完成目标并保留完成证明、跨子集前置约束；不转移原机感知输出 | `fleet/recovery_controller.py`、`scripts/run_fleet_mission.py` |
| 空间参考版本只有读取端 | Fleet 环境 reset 会推进参考版本；提供 owner 的 `invalidate_recovery_reference()` 重定位入口；生产观察显式携带 frame、pose 时间和时间域 | `env/fleet_uav_search_env.py`、`env/observation_types.py`、`skills/types.py` |
| 验证期间超过旧 Plan Revision 截止时间仍可能提交 | 验证结束重新读取时钟；旧单机生产入口请求期限改用 monotonic，观察时间保持独立 | `agents/plan_revision_coordinator.py`、`scripts/run_dynamic_visual_mission.py` |

上述相对路径均以 `uav_agent/` 为根。

## 调用链与配置

初始化仍为任务解释 → Fleet 分配 → 每机局部计划 → 编译/安全检查 → 执行。

开启恢复后，支持的 GOTO/SEARCH 异常进入：确定性恢复 → 原机监督 HOVER →
异步候选生成 → 前缀/目标/依赖验证 → 重新读取执行状态、墙钟、空间参考及共享空间 → 提交。
本机失败后才异步生成一个失败 assignment 的接管候选；无合法备用机或证据不足时有界安全退出。

本轮复用已有 Broker/worker，不新增同步运行期模型调用。新恢复路径的本机模型、
接管 Fleet Planner 和接管 Local Planner 在 worker 计算；owner 执行最终接纳。
未启用恢复时的旧 Fleet handler 仍保持同步行为；不能声称全仓库所有模式均异步。

使用方式见 [fleet_local_repair.md](fleet_local_repair.md)；示例配置为
[`multi_uav_local_repair_demo.yaml`](../uav_agent/configs/multi_uav_local_repair_demo.yaml)。
`enabled` 默认仍为 `false`；示例显式开启。`max_reassign_attempts` 默认及示例为 2，允许 1–8。
只有仍拥有源任务的 `STALE_VERSION` 冲突会触发上述刷新重试；不放宽空间或任务检查。

连续跟踪目标示意：

```json
{"goal_type":"TRACK_TARGET","duration_s":10,"completion_basis":"continuous"}
```

这是字段片段，完整目标仍需原 schema 的身份等必填字段。对应 TRACK 步骤必须保留该模式。
`valid_execution` 可累计有效区间，确定性 REACQUIRE 只扣除可信有效时间；
`continuous` 在失锁后重新计满要求时长。旧 `tracking_duration` 保留作经过时间遥测，不能用于完成判断。

## 验证范围

新增回归涵盖：五机中实际插入绕行 GOTO 的原机提交；TRACK 后返航失败恢复；
连续/累计时长与失锁、重复测量、REACQUIRE；原机相对目标换机后的 WORLD 位置；
接管后的再次修复；已完成导航及跨子集前置证据保持；版本冲突刷新与预算耗尽；
环境参考失效、观察时间域变化和验证期间超时。

主要测试文件：

- `tests/test_recovery_audit_regressions.py`
- `tests/fleet/test_recovery_controller.py`
- `tests/fleet/test_handoff_spatial_binding.py`
- `tests/fleet/test_runtime_replan_handler.py`
- `tests/fleet/test_local_repair.py`
- `tests/fleet/test_fleet_env.py`
- `tests/test_track.py`、`tests/test_plan_revision_coordinator.py`

基线正确命令下 6 个相关测试文件为 **95 passed**。默认 Python 缺少 pytest，
从仓库根运行且未设置 PYTHONPATH 时不能导入项目模块；使用既有 r_isaac_sim 解释器、
工作目录 `uav_agent` 后解决，未安装或改动项目外依赖。

最终全量命令（所有测试缓存、临时目录和日志均显式放在项目内）：

```bash
cd /home/amax/ry/vlm_drones/uav_agent
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/home/amax/ry/vlm_drones/outputs/dev_local_repair/root_tmp \
/home/amax/miniconda3/envs/r_isaac_sim/bin/python -m pytest -q --tb=short \
  -o cache_dir=../outputs/dev_local_repair/pytest_cache \
  --basetemp=../outputs/dev_local_repair/audit_final \
  -m 'not yolo_smoke and not qwen_lora_integration' tests \
  | tee ../outputs/dev_local_repair/audit_final_pytest.log
```

日志：[audit_final_pytest.log](../outputs/dev_local_repair/audit_final_pytest.log)。

最终结果：**2909 passed, 1 skipped, 2 deselected in 116.66s**，退出码 0。
1 项跳过为需要显式开启的 Isaac 集成测试；2 项排除为真实 YOLO 权重与 Qwen LoRA 集成测试。
这些未运行项目不能计入真实模型或飞行验证。最终源码检查 `git diff --check` 无问题。
接管发布边界的独立回归另为 **15 passed**，与全量结果重叠，不作累加。

集成中发现并修复了冷却时间跨恢复阶段误用、完成目标的证据引用混入其他步骤、
命名地点导航高度丢失等问题。旧测试中依赖经过时间扣减的输入更新为明确的有效执行证据；
技能参数白名单测试增加新的完成模式。新增测试构造问题也已修正，未删除安全断言或放松校验。

## 当前明确边界

- 可以识别需要协调的情况并安全升级，但尚不支持自动相关小组重规划；也没有全机队联合提交或数学最小影响集合求解。
- 并非所有异常均可本机修复：紧急安全、启动/终态故障、缺少原 V3 语义及部分执行的 TRACK/HOVER 仍按可信边界退出或升级。
- 部分完成接管仅支持可独立执行的剩余导航/终止要求；跨机传递 SEARCH 锁、TRACK 输出及不确定的部分时长仍不支持。仅剩原机返航/降落动作不能交给备用机冒充完成。
- 连续时间是依据可信离散观测及技能状态判定，不是对两次采样之间物理连续性的数学证明。
- 环境托管 reset 自动推进参考版本；外部定位系统发生参考重置必须调用失效入口，未实现外部定位重置的自动侦测器。
- 旧 PlanRevisionCoordinator 本轮修复了验证后期限检查，并未统一成 Fleet V3 的全部空间接纳契约。
- 未运行真实 Qwen/YOLO、Isaac、实机飞行或性能/长期压力实验；不声称硬实时、物理回滚或专利新颖性已获证明。
