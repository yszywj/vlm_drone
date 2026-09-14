# 多无人机规划 LoRA 设计

独立服务、任务编排与工具调用接口见 [规划服务说明](planning_service.md)。

状态：首批三角色数据已生成，首轮 LoRA 训练已启动，尚未部署新权重；见 [训练运行说明](planning_role_training.md)。数据位于仓库根目录的 `datasets/planning_roles_v1`，含 1000 个任务底稿、8000 条角色样本；具体范围和长度检查见 [数据集说明](planning_roles_dataset.md)。配套的 `configs/lora/planning_roles.design.json` 是设计记录，**不是训练器或适配器路由器的配置**；训练使用独立的 `*_train.json`。实际运行仍由 `configs/adapters.json` 决定使用哪个模型。

## 选什么神经网络

继续使用当前 **Qwen3-VL-4B-Instruct 的语言 decoder Transformer**，在语言注意力与前馈网络的线性投影上训练标准 PEFT LoRA。视觉编码器、视觉 merger、词嵌入、输出头及其他基座参数保持冻结。第一轮数据只包含文字任务和结构化状态，适合先隔离并改善规划能力。

LoRA 本身是权重更新方式。对基座的线性层 `W`，推理时的计算为：

```text
y = W x + (alpha / r) B (A x)
A: [r, input_dim]，B: [output_dim, r]
```

训练时仅更新 `A`、`B`；这两个矩阵之间没有新增激活函数。带 dropout 的实现会在 LoRA 分支输入上应用训练期 dropout。这里不需要另建 CNN、RNN 或一个独立 MLP 来替代 Qwen 的规划器。语言 Transformer 中原有的 MLP 仍然存在，我们只在它的投影层加 LoRA。[PEFT 官方 LoRA 说明](https://huggingface.co/docs/peft/en/package_reference/lora)

建议起始实验使用 `r=16, alpha=32, dropout=0.05, bias="none"`，与仓库现有训练示例一致。这是待验证的实验初值，不是已经证明的最优值。在数据、提示和评测固定后，再做 `r=32, alpha=64` 的容量对照，保持 `alpha/r=2`。先使用现有普通 LoRA 实现；如果显存不足，再单独评估量化训练及其对结果的影响。

本地文件核验于 2026-09-12 完成，未加载 GPU 权重或启动训练：

| 项目 | 本地结果 |
| --- | --- |
| checkpoint 架构 | `Qwen3VLForConditionalGeneration` |
| 语言 decoder | 36 层，`hidden_size=2560`，`intermediate_size=9728` |
| 语言注意力目标 | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| 语言 MLP 目标 | `gate_proj`, `up_proj`, `down_proj` |
| checkpoint 索引匹配 | 36 × 7 = 252 个语言投影权重 |
| 运行时模块核验 | 本次未执行；真实训练前必须针对 `named_modules()` 再核验 |

上述事实来自本地 `models/initial_model/Qwen3-VL-4B-Instruct/config.json` 和 `model.safetensors.index.json`。索引只能证明权重命名，不能替代真实模块检查。`training/lora/modeling.py` 已具备语言模块模式展开、冻结基座、PEFT 注入及可训练参数隔离检查。不要把全限定路径简化为跨整个多模态模型的 `all-linear`。官方 Qwen 微调项目也提供 LoRA 及视觉、语言模块的训练开关，但本项目应继续使用自身的精确模块检查。[Qwen 官方微调说明](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md)

## 三个独立角色

程序编排器明确选择阶段。规划模块与视觉、交互模块是并列能力；规划工具可以暴露为 `plan_fleet(...)` 一类接口，但调用结果必须通过校验后才能进入执行层。视觉输出通过程序维护的结构化观测提供给规划器。

三个 LoRA 分别从同一冻结基模开始训练、独立保存，按请求选择，初期不叠加权重。它们是逻辑模块，可以共享同一基模服务；“独立”不要求复制三份完整基模。

| 适配器角色 / 调用类型 | 当前生产契约 | 训练重点 |
| --- | --- | --- |
| `mission_interpreter` / `MISSION_INTERPRETATION` | 原始任务与可信别名 → `FleetTaskSpecV1` | 完整抽取目标、正确单位与坐标、唯一 ID、逐机约束、逐字证据、正确终止目标 |
| `fleet_planner` / `FLEET_PLAN`, `FLEET_REPLAN` | `FleetMissionRequestV2` → `FleetMissionPlanV2` | 搜索、跟踪和终止目标全部分配；忠实处理 MUST/PREFER/OPEN 与可信可用性证据；说明未分配目标 |
| `spatial_mission` / `AGENT_SPATIAL_PLAN` | `AgentPlannerRequestV2` → `SkillPlanDraftV3` | 搜索区域与目标一致、跟踪时长、执行顺序、返回本机起点并降落；通过编译及原始任务覆盖检查 |

`FLEET_REPLAN` 使用同一 Fleet V2 输出契约，也需要独立的失败、失联、可用性变化样本。不要直接沿用旧 `FleetPlanPatch` 的训练标签。视觉角色 `runtime_visual` 不纳入本轮三个规划适配器。

## 开训前必须补齐的内容

`training/lora/dataset.py` 面向旧 `FleetMissionRequest → FleetMissionPlan / FleetPlanPatch` 训练契约；新增的 `roles_dataset.py` 承担上表三阶段数据入口。训练前检查生产提示、解析器、数据完整性、监督边界和 token 长度。

三角色数据生成与逐条检查现由 `scripts/generate_planning_roles_dataset.py` 完成；`training/lora/train_planning_role_lora.py` 已接入三角色数据，collator 已改为拒绝截断并解决 Qwen BPE 的监督边界。旧 Fleet V1 数据入口继续保留。首批仅包含可执行正样本与明确的逐机 MUST 分工，尚未覆盖下述完整设计中的自由分配、澄清及故障重规划。

训练样本由独立任务规格生成正确标签，并逐条验证。不能把模型失败输出、程序自动补齐后的结果当作“模型本来就答对了”；用于修复训练时，应保留原始错误与诊断，再配上完整的正确答案。监督目标使用完整结构化答案，沿用 assistant-only loss，不要求输出隐藏推理过程。

第一批数据扩展为多任务类型，覆盖 2–10 架无人机，分别采样无人机数、语义目标数和任务数。
导航、悬停等任务可以没有语义目标；不再要求每机必须搜索一个目标。
变化别名、目标绑定、坐标、搜索半径、时长、单位、句序和表达方式，并包含需澄清、无法全部分配的合法情况。
当前每机仍只允许一个活跃 assignment；单个 assignment 最多涉及一个语义目标，
可以包含与该目标无冲突的导航或终止要求。跨目标串行调度等超出现有运行契约的任务，
应扩展代码后再加入可执行训练集。同一底层任务的正常、改写、局部投影及修复样本放在同一个划分中。

## 扩展的任务生成范围

先用程序构造独立任务底稿与正确标签，再渲染中文；不会直接让模型自由编造任务和答案后自判正确。
下面的“当前支持”表示契约和代码路径可承载，不代表当前 Qwen 已经规划成功或已经采集了训练样本。

| 任务类型 | 中文示例 | 纳入方式 |
| --- | --- | --- |
| 指定坐标导航 | A 前往世界坐标 `(10, 5, 10)` 米，随后返回自己的起点降落 | 首批基础类型；`NAVIGATE` → `GOTO`，坐标和高度必须在飞行范围内 |
| 悬停等待 | A 起飞后悬停 10 秒，再返航降落 | 首批基础类型；`WAIT` 语义目标与 `HOVER` 执行步骤，不附加虚构的搜索任务 |
| 搜索目标 | A 在指定圆形区域搜索目标 i，找到后返航降落 | 首批基础类型；不因为模板存在就擅自增加跟踪要求 |
| 搜索后跟踪 | A 找到目标 i 后跟踪 20 秒，然后返航降落 | 首批基础类型；目标引用来自先前搜索确认 |
| 多机混合分工 | A 前往指定坐标，B 搜索并跟踪目标 i，C 起飞悬停；分别完成后返航 | 首批重点组合，保持各机任务并行，任务数与目标数可不同 |
| 导航后停留、多点访问 | A 到 P1 悬停 5 秒，再到 P2，然后返航 | 正确标签需同时验证动作顺序和停留位置；不能只用当前逐目标覆盖检查认证 |
| 命名地点导航 | A 前往“观察点甲”后返航 | 须先接入宿主可信的名称→坐标表；默认 Fleet 上下文只提供本机 home，不能编造地点坐标 |
| 已确认目标直接跟踪 | 已锁定目标 i，A 继续跟踪 20 秒后返航 | 编译器支持宿主传入 `trusted_target_id`；当前主入口尚未接入该状态，完成接线后再纳入端到端样本 |

训练中应按任务类型平衡抽样，并显式包含无需 `SEARCH`、无需 `TRACK` 的任务。
返航、降落与等待要求从任务底稿及宿主明确的执行策略派生，不能无条件给所有语义目标追加相同结尾。
执行策略要求的补全与用户原文要求分别记录，避免模型把运行时默认策略误当成用户说过的内容。

组合泛化需要单独测试：例如训练包含单点导航和目标跟踪，测试未见过的“不同无人机分别导航、跟踪和等待”。
先增加每机子任务复杂度，再独立增加无人机数量，记录两种因素的影响。
既有五档全搜索测试继续保留，同时增加导航、等待及混合任务的独立评估集。

当前 `GoalSatisfactionChecker` 按目标匹配动作和累计时长，不接收完整 `ordering_constraints`。
生成“到 A 后在那里停留，再去 B”等组合标签时，需要额外检查顺序与位置，
不能将 `coverage.complete` 单独作为这类样本正确的依据。
`INSPECT_TARGET` / `REPORT` 虽出现在语义枚举中，初始执行链还有限制，暂不作为正常可执行正样本。
任意地点降落、跨机同步、同机连续处理多个不同目标，也需要相应执行接口支持。

## 长样本处理

**完整答案不能截尾。**生成数据时发现旧 `AssistantOnlyDataCollator._encode_feature()` 会使用 `full_ids[:model_max_length]` 截断 JSON；本次训练接入已将其改为超长报错，且在加载模型前预编码全部训练与验证数据：

1. 用本地 tokenizer 和实际 chat template，分别统计三角色 `prompt + 完整 answer` 的 token 长度及答案 UTF-8 字节数。
2. 对超长样本拒绝进入训练并输出样本 ID，或提高预算并重新核验；不能静默截断末尾返航降落。
3. 本轮配置长度上限 8192，动态 padding；三个角色均已通过各自最长训练样本的真实 forward/backward/AdamW 检查。现有旧示例的 4096 不适用本批十机样本。
4. 同时检查服务上下文、各阶段输出预算、解析器字节上限。最近压测服务上下文为 16384，解释器/Fleet 响应上限分别为 32768 字节；这些上限不等同于 checkpoint 宣告的最大位置数。

当前任务契约最多 32 个 mission goals、16 个 termination goals。十机每机搜索和跟踪可表示为 20 个 mission goals，再加 10 个 `RETURN_HOME_AND_LAND` 终止目标。不要为这个基准将返航和降落拆成 20 个终止目标后，误认为失败只来自模型规划能力。

## 如何验证提升

保留 `fleet_open_scaling_2_to_10.json` 的五档任务作为回归集，并建立更大的独立测试集。对基模、r16 和 r32 使用相同代码、提示、解码设置、输出预算及修复预算；每档覆盖多种任务，而非只重复同一个确定性提示。

每个角色先以正确的上游输入独立评估，再评估完整链路。至少记录：首轮严格通过率、修复后严格通过率、原始目标覆盖率、目标分配遗漏或重复、完整单机计划数、程序补全比例、延迟、token 数、截断和服务错误。最终计划检查必须对照原始任务，不能只对照可能已经丢失约束的解释结果。

做两种独立实验：训练覆盖 2–10 机并测试未见过的同规模任务；另建只训练 2–8 机、保留 10 机的数量泛化实验。不能把两者混为一个“十机能力提升”结论。接口验证、模块检查、长度检查、小规模训练与离线评测通过后，再导出真实 adapter manifest 并配置路由。此前 `adapters.json` 中的 placeholder 应继续如实显示基模回退。
