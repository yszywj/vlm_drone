# 无障碍多机规划：2、4、6、8、10 机递增测试

生产规划已抽为 [独立服务与工具接口](planning_service.md)；
后续训练方案见 [三阶段 LoRA 设计](planning_lora_design.md)。

本套测试沿用生产任务解释器、Fleet V2 分配器和每机 Spatial V3 规划器，
实际调用本地 Qwen 模型，逐档增加无人机和目标数量。
任务与独立预期值保存在
[`fleet_open_scaling_2_to_10.json`](../configs/benchmarks/fleet_open_scaling_2_to_10.json)。

## 场景和固定任务

五档完整场景配置位于 [`configs/benchmarks/fleet_scaling/`](../configs/benchmarks/fleet_scaling/)，
分别为 `open_2x2.yaml`、`open_4x4.yaml`、`open_6x6.yaml`、`open_8x8.yaml`、`open_10x10.yaml`。
场地均为 100 × 100 × 30 米，障碍物列表为空。每次扩容保留已有无人机和目标的位置，
增加新的无人机和目标；每机仍只负责一个目标。

每架无人机并行搜索指定目标，搜索范围为指定世界坐标中心、半径 6 米的圆形区域，
发现后跟踪 20 秒，最后返回自己的起点并降落。目标在中心附近以 0.5 m/s 小范围随机运动。
所有档位使用相同指令模板、搜索半径、跟踪时间和输出预算，只增加任务组数。
本套显式指定机与目标的对应关系，用于检验完整性和步骤规划；不评估自主最优分工。

## 运行

从 `uav_agent` 目录执行：

```bash
./python.sh scripts/evaluate_fleet_planning_stress.py \
  --suite configs/benchmarks/fleet_open_scaling_2_to_10.json --validate-only

./python.sh scripts/evaluate_fleet_planning_stress.py \
  --suite configs/benchmarks/fleet_open_scaling_2_to_10.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-4B-Instruct --repeat 1 \
  --interpreter-max-tokens 8192 --fleet-max-tokens 4096 --local-max-tokens 1536 \
  --timeout-s 180 --output-root ../outputs/fleet_scaling_to_10
```

实际测试使用服务上下文 16384、temperature=0、无 LoRA 的基模。
解释器最多修复 1 次，Fleet 和每机局部规划分别最多修复 2 次；全部修复均保留记录。
评分不作为模型修复提示，修复仅使用生产验证器的反馈。
可增加 `--case scale_10x10` 单测十机。评估失败的退出码为 2，
不会阻止同批后续规模继续运行。

## 如何读结果

- 全链严格通过：任务解释保留原意、完整分配全部任务及返航降落、所有单机计划通过。
- 最终完整单机计划：额外对照原始任务，检查编译结果中的目标、区域、跟踪时间、执行顺序和各自返航降落。
- 程序补全：生产编译器可能添加 `trusted_return_home` / `trusted_land`，
  这种可执行结果不能算作模型自主生成了完整步骤。

`status=completed` 只表示评估流程已运行完，成功需看 `first_pass` / `final_pass`。
即使单机编译通过，也可能存在上游遗漏或程序补全，因此各项结果需分别报告。
原始响应、token、延迟、验证错误、修复和最终编译结果均保存在每例 `result.json`。

本次只测试文字及结构化规划，不启动 Isaac，不评估图像理解或实际飞行成功率。
每档一次属于初步探测，不能据此确定模型普遍能力上限或稳定成功率。
本机实测结果及逐机核对记录见 [实验报告](../../outputs/fleet_scaling_to_10/report.md)。
