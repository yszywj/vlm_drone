# 三角色 LoRA 训练

首轮训练于 2026-09-12 在 tmux 会话 `planning_lora` 启动。
运行 ID：`roles_r16_20260912T133403Z`。

| tmux 窗口 | GPU | 训练/验证样本 | epochs | 计划优化步数 |
| --- | ---: | ---: | ---: | ---: |
| mission_interpreter | 0 | 800 / 100 | 3 | 150 |
| fleet_planner | 2 | 800 / 100 | 3 | 150 |
| spatial_mission | 3 | 4800 / 600 | 3 | 900 |

进入会话：

```bash
tmux attach -t planning_lora
```

用 `Ctrl-b n` 切换窗口，`Ctrl-b d` 离开会话后训练继续运行。
各窗口在进程结束后保留，便于查看完成信息或异常。

## 参数和位置

三个角色分别从同一本地 Qwen3-VL-4B-Instruct 初始权重训练，LoRA 不叠加。
配置位于 `configs/lora/{mission_interpreter,fleet_planner,spatial_mission}_train.json`。
初始参数为 `r=16, alpha=32, dropout=0.05`，语言 attention/MLP 共 252 个 Linear，
可训练参数 33,030,144；冻结基座和视觉模块。使用 BF16、梯度检查点、动态 padding，
每卡 batch 1、梯度累积 16、学习率 1e-4、cosine 调度及 3% warmup。
序列上限 8192；只监督完整 assistant 答案，超长直接拒绝。

日志目录：

```text
/home/amax/ry/vlm_drones/outputs/lora/planning_roles/launches/roles_r16_20260912T133403Z/
```

每个角色日志名为 `<role>.log`，正常或异常退出后写 `<role>.exit_code`。
对应训练目录为：

```text
/home/amax/ry/vlm_drones/outputs/lora/planning_roles/<role>/roles_r16_20260912T133403Z/
```

其中 `status.json` 持续记录阶段、步数、最近 loss/梯度范数和检查点；`preflight.json` 保存数据及
完整序列检查；`longest_sample_probe.json` 保存最长训练样本的完整优化步骤检查；
`training_sources.json` 保存实际训练代码摘要、Python 和依赖版本；`tensorboard/` 保存训练曲线。
第 1、5 步额外保存早期检查点，之后按照配置保存，最多保留两个。
只有训练和最终验证结束后才生成 `run_manifest.json` 和最终 adapter：

```text
/home/amax/ry/vlm_drones/models/adapters/<role>/roles_r16_20260912T133403Z/
```

2026-09-13 已完成任务级生成评估，并将这三个权重接入 `configs/adapters.json`。
三个规划角色使用 `generation.json_schema_property_order=alphabetical`，使约束解码字段顺序与训练标签一致。
正式服务及后续数据说明见 [规划 LoRA 接入与定向数据](planning_lora_integration.md)。

## 训练入口

项目内 `.venvs/planning_lora` 使用 Python 3.12，读取现有 `qwen_vllm` 环境中的 Torch/Transformers，
新增训练依赖只安装在项目 venv。启动时使用的主要版本：Torch 2.13.0+cu130、Transformers 5.15.0、
PEFT 0.20.0、Accelerate 1.15.0、TensorBoard 2.21.0、Hugging Face Datasets 5.0.1。
训练入口按已安装 distribution 显式导入 Hugging Face datasets，避免与项目的同名包冲突。
不需要改动项目 target_state 数据代码。

在 `uav_agent` 目录仅检查某个角色的数据与完整 token 长度：

```bash
../.venvs/planning_lora/bin/python training/lora/train_planning_role_lora.py \
  --config configs/lora/mission_interpreter_hardening_train.json \
  --role mission_interpreter --validate-only
```

原首轮配置保留作历史记录，数据为 v1；其源码快照也已保存。当前下一轮应使用
`mission_interpreter_hardening_train.json` 和 v2 数据，旧数据的严格来源检查仍保留。
启动首轮配置形式的独立运行（需相应历史源码和空闲 GPU）：

```bash
bash scripts/run_planning_role_training.sh mission_interpreter 0 NEW_RUN_ID \
  /home/amax/ry/vlm_drones/outputs/lora/planning_roles/launches/NEW_RUN_ID
```

新入口先校验全部数据文件与划分，再预编码 train/validation；test 不参与训练。
最长训练样本先执行 forward/backward/AdamW 检查，随后恢复初始 LoRA 权重并重设随机种子，
正式 Trainer 不继承该检查的参数更新或 optimizer 状态。
当前新入口拒绝覆盖已有运行目录，也尚未暴露断点续训选项。

首次启动 `roles_r16_20260912T132930Z` 在 Trainer 数据加载时遇到 datasets 同名冲突，已保留失败记录；
上面的运行在修复后重新启动。最长样本检查和启动训练仅证明训练链路可用，规划能力提升仍需独立任务评估。
