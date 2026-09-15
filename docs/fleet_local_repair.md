# 多无人机本机异步修复：使用与验收

本功能默认关闭。启用后优先让发生受支持异常的原无人机进入稳定监督悬停，
异步生成 Spatial V3 授权后缀；可信检查和提交前活状态复核通过后继续原任务。
原机无法恢复时，才进入有界备用机接管或安全退出。

基线与测试环境见 [fleet_local_repair_baseline.md](fleet_local_repair_baseline.md)。
本说明中的命令是按实际 CLI/配置编写的操作方法；本轮没有运行真实模型、Isaac 或飞行实验。

## 开关和配置

新增完整示例：[multi_uav_local_repair_demo.yaml](../uav_agent/configs/multi_uav_local_repair_demo.yaml)。
它复制原双机 demo，只增加显式 `fleet_recovery` 配置，不修改训练、权重或原配置文件。
目标感知保持 `disabled`，适合先检查不依赖目标识别的导航任务装配。
它既不是五机实测记录，也不自动添加备用无人机。

```yaml
fleet_recovery:
  enabled: true
  mode: LOCAL_THEN_REASSIGN
  request_timeout_s: 30.0
  episode_timeout_s: 90.0
  retry_cooldown_s: 1.0
  max_local_attempts: 2
  max_reassign_attempts: 1
  max_concurrent_requests: 2
  max_pose_time_error_s: 0.05
  max_anchor_age_s: 45.0
  max_hold_drift_m: 2.0
  max_suffix_steps: 10
  shutdown_timeout_s: 0.1
```

`mode` 支持 `LOCAL_ONLY` 和 `LOCAL_THEN_REASSIGN`。
前者在本机修复预算用尽后安全退出；后者再尝试合法备用机。
不支持强占健康无人机。没有备用机不妨碍先尝试原机修复，也不应导致无限等待。

`request_timeout_s`、`episode_timeout_s`、冷却和关闭等待采用单调墙钟秒；
它们不能与仿真时间相减。位姿/观察对齐误差使用观察时间域，漂移使用米。
后缀步数还受整个原计划 `planner.max_plan_steps` 和技能调用预算限制，配置为 10
不意味着已经完成的前缀之外总能再增加 10 步。
请求并发上限同时受现有 `model_broker` 全局和每机限额约束。

必须使用 `--fleet-planner llm`、相容的任务解释器、Spatial V3 和
`--runtime-program linear`。开启恢复却选择 scripted Fleet 或 Graph 会在飞行前报错；
关闭恢复时原路径保持兼容。旧 YAML 没有 `fleet_recovery` 时等价于 `enabled: false`。

## 导航任务运行示例

先按现有部署准备 Isaac 环境、`python.sh`、模型服务及
`configs/adapters.json` 指向的模型资源。下例不请求目标搜索，因此不需要将禁用的感知配置
伪装成生产检测器；该命令不会自动注入故障。

```bash
cd /home/amax/ry/vlm_drones/uav_agent
./python.sh scripts/run_fleet_mission.py \
  --config configs/multi_uav_local_repair_demo.yaml \
  --mission-interpreter llm \
  --fleet-planner llm \
  --local-planner dynamic_llm \
  --planning-contract v3 \
  --runtime-program linear \
  --adapter-config configs/adapters.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-4B-Instruct \
  --api-key EMPTY \
  --headless \
  --max-sim-time 300 \
  --output-root ../outputs/dev_local_repair/manual_runs \
  --instruction "无人机A前往世界坐标(-20,20,10)，随后返回自己的起点降落；无人机B前往世界坐标(20,-20,10)，随后返回自己的起点降落。"
```

`EMPTY` 仅适用于现有无需认证的本地服务；需要认证时按部署方式提供凭据，不写入文档、
仓库或日志。模型注册及回退保持原策略，本功能不训练或晋级新 adapter。

## 生产目标搜索任务

应从现有 [multi_uav_cube_yolo.yaml](../uav_agent/configs/multi_uav_cube_yolo.yaml)
创建项目内副本，仅添加上述恢复块。不能把 demo 的 `backend: disabled` 当作真实目标确认。
现有生产流程要求与配置匹配的单类 `cube` 权重、每机独立 YOLO worker、健康检查及颜色/身份
确认资源；通用 COCO 权重不能代替 Cube 权重。
具体资源准备见 [YOLO 生产运行手册](../uav_agent/docs/yolo_production_runtime.md)。

以下命令创建配置副本，不改原生产配置或权重路径：

```bash
cd /home/amax/ry/vlm_drones/uav_agent
PYTHONDONTWRITEBYTECODE=1 ./python.sh - <<'PY'
from pathlib import Path
import yaml
source = Path("configs/multi_uav_cube_yolo.yaml")
destination = Path("configs/multi_uav_cube_yolo_local_repair.yaml")
if destination.exists():
    raise SystemExit("配置已存在，请检查后使用，避免覆盖已有设置")
config = yaml.safe_load(source.read_text())
demo = yaml.safe_load(Path("configs/multi_uav_local_repair_demo.yaml").read_text())
config["fleet_recovery"] = demo["fleet_recovery"]
destination.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
PY
./python.sh scripts/check_fleet_yolo_services.py \
  --config configs/multi_uav_cube_yolo_local_repair.yaml
```

服务检查通过后运行：

```bash
./python.sh scripts/run_fleet_mission.py \
  --config configs/multi_uav_cube_yolo_local_repair.yaml \
  --target-perception-mode yolo \
  --perception-runtime-profile production \
  --mission-interpreter llm --fleet-planner llm --local-planner dynamic_llm \
  --planning-contract v3 --runtime-program linear \
  --adapter-config configs/adapters.json \
  --base-url http://127.0.0.1:8000/v1 --model Qwen3-VL-4B-Instruct --api-key EMPTY \
  --enable-qwen-vision --vision-review-mode gate --acknowledge-vision-gate \
  --headless --max-sim-time 300 \
  --output-root ../outputs/dev_local_repair/production_runs \
  --instruction "无人机A前往世界坐标20,30附近15米范围搜索并跟踪目标i20秒；无人机B前往世界坐标-25,10附近12米范围搜索并跟踪目标j20秒；完成后分别返回各自起点降落。"
```

这段命令保留既有视觉 gate 的显式授权条件；本机文本修复没有借用视觉请求权限。
不要给旧视觉结果重新标记采集时间，也不要用 Oracle 状态补齐生产缺失证据。

## 新运行链与责任边界

```text
SkillManager 可信可恢复事件
  → MissionAgent 保留活动任务并建立稳定监督 HOVER
  → FleetRecoveryController 建立有期限、有预算的 episode
  → Broker 排队
  → 真正取得槽位时由运行所有者复核状态、捕获不可变上下文
  → worker 调用模型，严格解析 V3 后缀，纯编译和语义检查
  → owner 取得候选，检查最新世界/依赖/当前接入
  → 高开销验证之后再次检查活状态、版本、取消和墙钟期限
  → 串行提交，更新原 Agent/Manager、局部版本、Fleet 路线及进度
```

本机成功不调用 `FLEET_REPLAN`。换机升级通过拆开的
`snapshot_request → compute_candidate → prepare_candidate → committed`
边界执行，模型计算位于 worker，环境与 Agent 发布属于运行所有者。
初始化仍在飞行前同步完成。关闭新增恢复时保留原同步兼容 handler。

取消或超时撤销候选的执行资格，不代表底层 HTTP/GPU 已停止；未结束的调用仍占用实际名额。
安全取消不等待模型返回。软件状态发布一致性不意味着物理降落动作可以回滚。

## 支持范围与保守升级

- 首轮支持有原 V3 语义来源的线性 GOTO/SEARCH 修复，用有预算的 GOTO 路点表达绕行。
- 已完成前缀及其输出保持不变；模型不能重放 TAKEOFF、已确认目标或改写目标身份。
- 原后缀 TRACK/HOVER/LAND 条件保持不变；SEARCH 保留目标描述、区域和高度，允许受限策略调整。
- 已确认完成的目标由可信终端 Skill 结果判定。失败反馈中的经过时间不计作有效完成时长。
- 当前 TRACK/HOVER 已部分执行而不能可靠续接时，明确返回证据不足，安全升级或退出。
  换机路径对部分完成和跨机输出转移也采取保守限制，不能宣称任意剩余任务都能自动接管。
- Graph、任意任务图修复、没有原 V3 语义来源的当前步骤、INSPECT/REPORT 等未支持映射，
  不进入本机线性修订器。既有确定性 REACQUIRE 和紧急安全策略仍先行。
- 跨子集 A→B 必须保留只读外部依赖、端点、版本及证据；条件未知或变化时不删除约束强行执行。
  默认保持健康无人机计划不变。影响集合是保守集合，不是数学最小集合。

**可以识别需要协调的情况并安全升级，但尚不支持自动相关小组重规划。**

## 时间、共同参考与空间接纳

修复上下文分别记录任务路由、request/episode、执行代次、原计划版本、允许替换位置、
完成前缀、输出摘要、目标及依赖，另有独立 `frame_id` 和 `anchor_id`。
观察时间与单调墙钟提交/截止时间属于不同时间域。

稳定悬停确认后，在本轮重规划开始时采样位置和航向；模型相对坐标、可信空间解析及路线检查
使用同一 HOLD 锚，原 home/start 参考另行保留。模型等待期间转向或漂移不会重解释旧候选。
历史前缀需要重建编译状态时使用原已编译 WORLD 几何，不将旧 HOLD 坐标绑定到新锚。

候选 SEARCH 入区点被可信编译器固定到已接纳的 WORLD 坐标，避免执行时重新选择另一条入区线。
提交仍须检查当前位置到路线的接入段、所有当前可碰撞障碍和相关共享空间；
参考版本变化、时间错配、漂移超限、依赖变化或不安全接入都会使候选失效。
当前依赖版本摘要包括 Fleet 全局版本、本机 assignment 版本、地图/参考版本，
以及显式外部依赖涉及的其他 assignment 版本。未列入依赖集合的健康机局部版本变化
不会单独使请求失效，但发布时仍重新检查其当前共享空间/路线，不能省去该检查。
世界路线只表示确定 GOTO/SEARCH/LAND 几何；未来 TRACK 目标运动仍由既有实时控制和安全检查处理，
不应称静态候选已证明任意未来轨迹安全。

## 日志与排障

`RECOVERY_EPISODE_OPENED`、`RECOVERY_QUEUED`、`RECOVERY_MODEL_SUBMITTED`、
`RECOVERY_MODEL_COMPLETED`、`RECOVERY_VALIDATION_COMPLETED`、`RECOVERY_COMMITTED`
用于区分阶段。检查记录中的 request、episode、执行代次、本机版本、anchor、墙钟时间和拒绝原因；
模型返回或编译成功都不代表已经提交。

常见拒绝包括 `EVIDENCE_INSUFFICIENT`、`COORDINATION_REQUIRED`、
`EXTERNAL_DEPENDENCY_MISSING`、`ROUTING_MISMATCH`、`DEADLINE_EXPIRED`、
`UNSAFE_ENTRY_OR_ROUTE`、`REFERENCE_CHANGED`。应修复对应的可信输入或使用支持场景，
不要放宽校验绕过拒绝。

## 修改文件与职责

路径以仓库根目录为基准。初始化仍使用原有解释器、Fleet 分配器、V3 编译器与安全检查。

| 文件 | 职责 |
|---|---|
| uav_agent/fleet/recovery_controller.py（新增） | 恢复所有者、本机优先、episode、可信证据、参考绑定、接纳与异步升级 |
| uav_agent/fleet/local_repair.py（新增） | 不可变 V3 请求、严格后缀输出、前缀/目标保护、剩余目标、外部依赖和共同 WORLD 路线 |
| uav_agent/agents/mission_agent.py、skills/manager.py | 确定性恢复后的可信事件、监督 HOVER、保留执行状态的后缀提交；按 Fleet 已发布版本启动新 Agent |
| uav_agent/fleet/model_request_broker.py、model_request_dispatcher.py | 共享配额、提交前刷新、异步文本任务、按请求消费、撤销与实际占用分离、候选视觉 facade |
| uav_agent/models/async_worker.py | 保留其他消费者结果、有界关闭 |
| uav_agent/models/runtime_deadline.py（新增）、model_client_factory.py | 可注入墙钟总期限、独立角色客户端、恢复调用的 HTTP 超时及重试限制 |
| uav_agent/scripts/run_fleet_mission.py | 生产装配、早期检查、拆分换机 handler、冻结计算输入、候选清理与发布后元数据 |
| uav_agent/fleet/runtime.py、target_registry.py | 主循环轮询、最终比较与发布、隔离认领准备、版本一致及旧机降落推进 |
| uav_agent/env/fleet_uav_search_env.py、perception/factory.py | 环境 assignment 分阶段发布、无目标真值的可信几何、候选感知不提前获得帧权限 |
| uav_agent/fleet/llm_planner_v2.py | 单 assignment schema/prompt、只读外部依赖 |
| uav_agent/configs/schema.py、loader.py、multi_uav_local_repair_demo.yaml | 默认关闭、严格配置校验和使用示例 |
| 新增 tests/fleet/test_local_repair.py、test_recovery_controller.py、test_recovery_publication.py、test_recovery_configuration.py、test_target_registry_staging.py；tests/test_local_repair_agent.py | 契约、真实 Fleet/Agent 的假环境集成、并发、失效发布、配置、资源隔离和确定性恢复优先 |
| 原 Broker/dispatcher、async_worker、runtime_replan_handler 测试 | 补充资源与结果隔离；调整旧“撤销立即释放运行槽”的断言，没有删除测试或取消校验 |
| docs/fleet_local_repair_baseline.md、本文 | 真实基线、操作方式、验收证据与边界 |

表中省略重复目录的相邻文件与前一文件属于同一目录；测试路径均位于 uav_agent 下。

## T1—T9 验收矩阵

“通过”表示本机 Python 自动化测试通过。模型客户端、飞行环境和部分规划输出使用可控替身；
FleetMissionRuntime、生产控制器装配 helper、MissionAgent、SkillManager、V3 契约、
编译器与接纳检查使用实际代码。这不等于真实模型、Isaac 或飞行效果验证。

| 任务 | 完成内容与验证 |
|---|---|
| T1 | 确定性恢复优先、本机安全等待、故障去重/冷却/总期限；五机中仅三号生成补丁，其他四机继续执行，通过 |
| T2 | V3 后缀生成、解析、编译、原机继续；不重放 TAKEOFF；接管新 Agent 再次本机修复，通过 |
| T3 | 本机和备用机模型在 Broker worker；事件屏障期间环境、安全检查和取消推进；乱序/共享配额，通过 |
| T4 | HOLD/home/start、独立 frame/anchor、时间对齐、地图版本、最新障碍和当前接入；航向变化不重解释，通过 |
| T5 | 高开销检查后复核；版本/步骤/取消/超时拒绝；隔离准备、短提交边界、旧机降落推进，通过 |
| T6 | 可信结果扣除完成目标、证据不足拒绝、不伪造时长、保留跨子集依赖和版本，通过 |
| T7 | 本机失败后生产拆分 handler 异步接管；无备用机退出；争用无双重授权，新 Agent 可修复，通过 |
| T8 | 默认关闭、旧配置兼容、Graph/scripted 开启时早失败、回退不变、阶段/范围/版本/期限日志，通过 |
| T9 | 下列 19 类场景均有实际运行通过的测试；真实环境与小组联合规划的边界见后文 |

测试文件简称：C = test_recovery_controller.py，P = test_recovery_publication.py，
V = test_local_repair.py，A = tests/test_local_repair_agent.py，
D = test_model_request_dispatcher.py，F = test_recovery_configuration.py。
除 A 外均位于 uav_agent/tests/fleet/。

| T9 场景 | 通过的证据入口 |
|---|---|
| 1 五机仅三号修复 | C：test_five_uavs_only_third_repairs_while_healthy_safety_and_cancel_keep_progressing[False]；四架健康机 tick 增长，计划内容/版本和启动次数不变 |
| 2 慢模型下健康机/安全取消推进 | 上述测试的两个参数；Event 阻塞模型，环境和真实 Safety 继续，取消先于模型返回完成 |
| 3 重复故障限流与预算 | C：test_preparation_errors_consume_attempt_budget_and_do_not_create_request_storm、墙钟总期限；正常成功后重复 tick 不再调用 |
| 4 两机乱序返回隔离 | C：test_two_independent_uavs_complete_out_of_order_without_crossing_results_or_versions；D 的乱序和视觉/文本共享配额 |
| 5 排队期间状态改变 | C：test_queued_request_is_rejected_before_model_start_when_execution_changes；D 的真实槽位可用时 owner 刷新快照 |
| 6 生产装配 V3 闭环 | C 五机测试使用生产 helper、真实 Runtime/Agent/Manager/V3；备用机测试使用生产拆分 handler 和真实编译器 |
| 7 取消/降落后的迟到返回 | C 五机取消分支、A 取消 guard；结果不能恢复已取消任务 |
| 8 验证期间超时/步骤/版本变化 | C：test_final_guard_rechecks_wall_deadline_after_safety_preflight、test_change_during_agent_preflight_has_no_commit_or_old_fallback；P 两次 guard |
| 9 重复结果、旧 episode | A 重复提交和 reset 代次；C：test_expired_old_episode_cannot_land_a_newer_local_execution；D 按 request 消费 |
| 10 航向/漂移/参考重置/时间错配 | C 航向保持、漂移退出、参考变化、新障碍；V 锚时间错配、旧 HOLD 前缀及固定 SEARCH entry |
| 11 前缀/输出/目标/时长被改 | V 的路由、完成前缀、引用、目标、TRACK 时长/距离、终止 LAND 严格拒绝 |
| 12 A→B 和共享通道变化 | V 跨边保留、缺失拒绝和依赖摘要变化；C 外部前驱活动、共享资源未知/参考变化的协调退出 |
| 13 可修但无备用机 | C：test_no_standby_still_attempts_local_then_exits_if_unrepairable[True]；不调用 FLEET_REPLAN |
| 14 不可修且无备用机 | 同一测试 [False]；本机预算用尽后 owner 准备阶段发现无合法备用机，安全退出 |
| 15 两请求争用一个备用机 | P：test_two_ready_requests_cannot_publish_same_standby_uav；旧版本和重新基准化后仍占用均拒绝 |
| 16 准备/发布抛异常 | P 的 Agent 构造、环境准备、目标绑定、两次 guard、版本变化；F 被拒视觉 facade 不污染备用机绑定 |
| 17 已完成/证据不足目标 | V 成功证据、缺失证据、先失败后成功、部分 TRACK；不把 elapsed 当有效时长；部分目标换机明确拒绝 |
| 18 失效底层调用仍在运行 | D 撤销后实际名额仍占用、后续请求不得越额、有界 close；F 过期流水线不发起下一次 HTTP |
| 19 Graph/scripted/关闭开关 | F 默认关闭和早失败、A Graph 拒绝；原 Fleet、Graph、旧修订器及全 Python 回归通过 |

## 实际测试命令与结果

工作目录必须是 uav_agent。以下命令均实际运行，测试临时文件位于项目内。

全 Python 测试集合：

~~~bash
cd /home/amax/ry/vlm_drones/uav_agent
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/home/amax/ry/vlm_drones/outputs/dev_local_repair/root_tmp \
/home/amax/miniconda3/envs/r_isaac_sim/bin/python -m pytest -q --tb=short \
  -o cache_dir=../outputs/dev_local_repair/pytest_cache \
  --basetemp=../outputs/dev_local_repair/full_tmp \
  -m 'not yolo_smoke and not qwen_lora_integration' tests
~~~

实际结果：**2859 passed, 1 skipped, 2 deselected in 114.36s**。

- 1 项 skipped：既有 FullOracleSkillPipelineIsaacTest 要求显式设置
  UAV_AGENT_RUN_ISAAC_TESTS=1，本轮没有启用。
- 2 项 deselected：显式排除加载真实权重的 yolo_smoke 和 qwen_lora_integration，
  没有执行真实 LoRA 优化器实验。
- 没有为了得到通过结果删除测试、放宽授权或跳过契约校验。

全量测试之后仅补充了恢复日志的范围及 episode 截止字段，随后对最终版本运行全部相关集合：

~~~bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/home/amax/ry/vlm_drones/outputs/dev_local_repair/root_tmp \
/home/amax/miniconda3/envs/r_isaac_sim/bin/python -m pytest -q --tb=short \
  -o cache_dir=../outputs/dev_local_repair/pytest_cache \
  --basetemp=../outputs/dev_local_repair/final_tmp \
  tests/fleet tests/test_plan_revision.py tests/test_plan_revision_coordinator.py \
  tests/test_reacquire.py tests/test_async_worker.py tests/test_local_repair_agent.py \
  tests/test_mission_agent.py tests/test_graph_skill_runtime.py tests/test_config.py \
  tests/test_spatial_resolver.py tests/test_plan_validator_v3.py
~~~

最终相关集合：**741 passed in 5.67s**。
补充阶段的 V3 契约组合为 94 passed，最终控制器集成文件为 19 passed in 0.82s，
配置/期限/候选绑定文件为 14 passed in 0.39s。这些数字存在重叠，不作累加。

### 基线失败、新增问题与未运行项目

- 隔离原 HEAD 的正确环境基线为 **555 passed**。早期 4 个失败来自工作目录，
  隔离快照另有模型/数据软链接缺失；补齐只读引用后消失，见基线记录。
- 默认 Python 缺少 pytest，采用已有 r_isaac_sim 解释器；没有安装或修改项目外依赖。
- 集成中发现并修正的实际问题包括：备用机启动局部版本回退、对终态 Agent 再次取消、
  候选准备污染正式绑定的风险、悬停失稳后 episode 过早结束、
  旧步骤请求对新步骤触发 fallback。对应最终测试通过。
- 新测试初写时也出现 Safety 方法名、多机配置构造、终态 tick、测试规划器装配错误，
  已按真实接口修正；没有改弱验收断言来掩盖问题。
- 未运行真实 Qwen/YOLO 推理、Isaac、实机飞行、性能基准或长时压力实验。
  替身测试不能证明真实模型成功率、航迹质量或飞行安全性能。

## 首轮限制与后续工作

首轮支持本机 GOTO/SEARCH 授权后缀闭环及备用机升级。
TAKEOFF/LAND 紧急失败、部分 TRACK/HOVER、缺少原 V3 语义或可信证据均保守处置。
已完成目标可以通过证据识别，但首轮换机不转移已有输出及部分时间义务；
此类 handoff 明确拒绝并安全退出，不让备用机重做已确认目标。

max_reassign_attempts 首轮只支持 **1**，与原单次接管边界一致；更大值配置早失败。
该次接管内部保留现有有限结构/语义修复次数，各调用共用同一墙钟期限。
本机的 max_local_attempts、冷却与 episode 总期限可以配置。

备用机按原 Fleet 规则递增全局版本，启动时显式传入已发布的局部版本；
本机修复只递增该 assignment 的局部版本，保留任务前缀与输出。

候选准备和校验仍有 Python 计算开销，移出 HTTP 不构成硬实时保证。
无法强制终止的底层调用撤销后继续占用名额，可能使后续恢复排队或超时。
关闭等待有界；超出预算仍在执行的调用保留占用，完成后可由 owner 再次 pump/close
确认或随整个进程退出，不能假报资源已释放。

软件状态分阶段发布不意味着物理动作可以回滚。旧机 cancel/LAND 一旦开始，即使候选随后失效，
仍由保留的 Agent 完成退出，不用旧快照覆盖新的正式状态。

**可以识别需要协调的情况并安全升级，但尚不支持自动相关小组重规划。**
影响集合是保守集合，没有实现数学最小集合求解。
后续应分别开展共享通道/前置关系的小组联合规划，以及真实服务、Isaac、飞行环境的故障注入验收。
本轮没有新增网络、RL、aLoRA、训练或权重修改，也没有据此证明专利创新性。
