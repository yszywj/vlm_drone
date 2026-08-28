# Target State Episode-Sharded Training v1

本文档描述如何在 PC 永久保留完整 Target State 数据，同时让服务器仅缓存少量
shard，完成可恢复的 Stage A 连续训练。这里的 shard 是按 episode 构建的不可变
`shard_*.tar`，不是 PyTorch 的流式 tar DataLoader。

## 架构与职责边界

```text
PC permanent Target State dataset
  → vlm_drone shard builder（按 episode 构建 immutable shard_*.tar）
  → 直接 rsync 一个很小的 shard_index.json
  → pc_trans request + PC 每 epoch 逐 shard push
  → server ready/<run_id> → active/<run_id>
  → vlm_drone 校验 tar、原子 materialize、构造 Dataset/DataLoader
  → 连续 model/optimizer 训练、checkpoint、resume
  → checkpoint 安全提交后，vlm_drone cleanup，pc_trans consume
```

两个仓库的边界不能混合：

- `pc_trans` 只负责 request、rsync 生命周期、ready/active、wait、状态、consume、
  recover、quota 和 pause。它不理解 `frames.jsonl`、episode、Target State、Dataset
  或训练阶段，也不是 streaming DataLoader。
- `vlm_drone/uav_agent` 负责 shard 格式、episode-aware 分片、tar 校验与解包、
  dataset checker、Dataset/DataLoader、model、optimizer、global epoch、checkpoint、
  validation/test 和 resume。
- 服务器通过稳定的 `python -m pc_trans.cli` 边界使用独立的 `pc_trans` 仓库，
  不导入其 Python package，也不复制它的生命周期实现。

不要让 PyTorch 随机读取 PC 文件、`.rsync-partial` 或尚未原子发布的 materialized
目录；不要使用 SSHFS/FUSE、远程 Dataset、tar streaming、rsync daemon、新网络
服务或服务器到 PC 的反向连接。PC 上的原始数据集和构建完成的 shard 始终是永久
副本。

换成最直接的边界表述：

```text
pc_trans is NOT a streaming DataLoader.
pc_trans does NOT understand Target State dataset format.
```

## 1. 在 PC 构建真实 15k 数据集的 shard

以下命令在安装了本项目数据处理依赖的 PC/WSL 环境执行，不需要 Isaac Sim 或
GPU。输入数据集不会被修改。输出 tar 完成后应视为 immutable：

```bash
cd <PC上的vlm_drone/uav_agent>

python scripts/build_target_state_shards.py \
  --dataset-root \
  /mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_15k \
  --output-dir \
  /mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_15k_shards \
  --target-shard-size-mib 512 \
  --history-size 6 \
  --max-history-age-s 2.0 \
  --split-seed 42
```

构建器按 episode 分组后分片，同一 episode 不跨 shard，因此 `history_size=6` 的
序列不会在 shard 边界丢失历史。超过目标大小的单个 episode 独占一个 shard，
不会为了满足 512 MiB 而拆开。相同输入、seed 和参数应得到确定性的 shard plan。

构建成功后核对终端汇总至少包含：

```text
parent dataset SHA
train shard count
validation shard count
test shard count
episode count
frame count
sequence count
total tar bytes
shard_index path
```

同时保留以下两个 PC 路径，不要让训练上传命令删除其中任何文件：

```text
/mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_15k/
/mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_15k_shards/
```

## 2. 先直接传输 shard index

`shard_index.json` 是很小的控制文件。它不匹配 `shard_*.tar`，不要通过
`pc_trans` 传输。先在 PC 执行一次普通 rsync：

```bash
rsync -avh \
  -e ssh \
  /mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_15k_shards/shard_index.json \
  vlm-data:/home/amax/ry/vlm_drones/datasets/stage_a_indexes/target_state_extreme_v1_15k_shard_index.json
```

在服务器确认文件存在且可读：

```bash
ls -lh \
  /home/amax/ry/vlm_drones/datasets/stage_a_indexes/target_state_extreme_v1_15k_shard_index.json
```

`pc_trans` 仍然只管理 index 所引用的 `shard_*.tar` 数据文件。

## 3. 在服务器启动 Stage A sharded training

在服务器执行：

```bash
cd /home/amax/ry/vlm_drones/uav_agent

./python.sh scripts/train_target_state_sharded.py \
  --config configs/target_state/train_oracle_clean.yaml \
  --shard-index /home/amax/ry/vlm_drones/datasets/stage_a_indexes/target_state_extreme_v1_15k_shard_index.json \
  --pc-trans-root /home/amax/ry/pc_trans \
  --pc-trans-config /home/amax/ry/pc_trans/config/config.json \
  --bridge-root /home/amax/ry/vlm_drones/datasets/_bridge \
  --run-id-prefix stagea_extreme_15k \
  --output-dir /home/amax/ry/vlm_drones/outputs/trained_models/target_state_extreme_15k \
  --run-name temporal_ray_depth_oracle_extreme_15k \
  --device cuda:0
```

`epochs` 默认来自 `train_oracle_clean.yaml`，其含义是 global epochs。只有进行过
显式估算或实验时才使用 `--epochs N` 覆盖；不要在代码或运维脚本中把 15k 数据集
写死为 4 epochs。

启动后，服务器会为 global epoch 1 创建 request、打印实际 RUN_ID
`stagea_extreme_15k.e0001`，然后只等待当前 shard，而不是等待整个 epoch 的文件
全部上传。先看到服务器打印 RUN_ID，再执行对应 PC 命令。

## 4. PC 每个 epoch 逐 shard 推送

以下命令在 PC 上的 `pc_trans` checkout 中执行。训练上传方向禁止
`--remove-source-files`，PC shard 是永久源数据：

```bash
cd <PC上的pc_trans>

SSH_TARGET=vlm-data \
LOCAL_SHARDS=/mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_15k_shards \
./pc_scripts/push_prefetch_once.sh stagea_extreme_15k.e0001
```

完成 epoch 1 的全部 train shards 和完整 validation 后，服务器创建并打印下一
request。随后在 PC 执行：

```bash
SSH_TARGET=vlm-data \
LOCAL_SHARDS=/mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_15k_shards \
./pc_scripts/push_prefetch_once.sh stagea_extreme_15k.e0002
```

后续严格使用服务器打印的 `stagea_extreme_15k.eNNNN`。每个 global epoch 必须是
新的 run ID，绝不能复用 epoch 1；否则旧 consumed record 会与新一轮同名 shard
冲突。如果服务器为最终 test 打印 `stagea_extreme_15k.finaltest`，同样在 PC 执行：

```bash
SSH_TARGET=vlm-data \
LOCAL_SHARDS=/mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_15k_shards \
./pc_scripts/push_prefetch_once.sh stagea_extreme_15k.finaltest
```

`push_prefetch_once.sh` 可以在断线或进程中断后用同一个 RUN_ID 重复执行。它在每
个 rsync 前查询 `shard-state` 和 pause 状态：已 ready/active 的 shard 跳过，已经
成功消费的 shard不会重新上传；状态查询失败则 fail closed、等待并重试。不要用
一个 `rsync --files-from=<整个 epoch 列表>` 绕过逐 shard 背压。

## 5. 每 epoch 的 producer-consumer 生命周期

一个 epoch 的 request 可以列出有序的 train shards 和本 epoch validation shards，
但服务器不能执行 `wait-ready` 等它们全部到齐。正常并行过程是：

```text
PC                                  Server
push train shard 1            →    wait-shard 1 → activate → materialize
push train shard 2 (prefetch)       train shard 1
push train shard 3 (prefetch)       checkpoint shard 1 → consume → train shard 2
...                                 ...
                                      ↓
                                    full validation across all val shards
```

每个服务器 shard 的完整生命周期是：

```text
wait-shard
→ shard-state
→ activate
→ verify archive SHA
→ atomic materialize
→ check_dataset
→ construct Dataset/DataLoader
→ train or evaluate the entire shard
→ stop/join all DataLoader workers
→ atomically commit and reread authoritative checkpoint
→ cleanup materialized directory
→ pc_trans consume --delete active tar
→ next shard
```

因此第一片到达即可训练，GPU 处理 shard N 时 PC 可上传 N+1。服务器正常只需容纳
少量 ready shards、一个 active shard、一个 materialized shard，以及 checkpoint、
输出和安全余量；不需要保存完整 15k dataset。

## 6. Global epoch、validation 与 test

global epoch 的循环顺序只能是 `epoch → shard → batch`：

```text
Global Epoch 1
  → 按该 epoch 的 deterministic order 将每个 train shard 各训练一次
  → 合并所有 validation shard 的原始 accumulator
  → 对完整 validation set 生成一次指标并判断 best.pt

Global Epoch 2
  → 重新得到 deterministic shard order
  → 将每个 train shard各训练一次
  → 再进行一次完整 validation
```

禁止在 shard 1 内先跑完全部 epochs 再进入 shard 2，也禁止在只评估一个
validation shard 后更新 best。model、optimizer 和 `global_step` 在 shard/epoch 间
连续。每个 shard 的 batch RNG 使用稳定派生值，例如基于 base seed、global epoch
和 shard filename 的 SHA256，而不是依赖进程此前调用了多少次 `random()`；checkpoint
记录 `shard_rng_protocol = per_shard_stable_seed_v1`。因此相同 base seed、global
epoch 和 shard 必须产生相同 batch order，当前 shard 从安全边界重训也可复现。
最终 test 覆盖完整 test split，并使用完整 validation 选择出的 Stage A `best.pt`。

## 7. Resume 与删除安全

`latest.pt` 是同一次 sharded run **唯一**允许传给 `--resume-checkpoint` 的恢复
authority；`training_state.json`、日志和 `best.pt` 都不能代替它。`best.pt` 只用于
完整 validation 后保留最佳权重，以及作为后续 Stage B 的
`--initial-checkpoint`，绝不能用作 Stage A shard-boundary resume。
参数还必须精确指向 `<output-dir>/<run-name>/latest.pt`；旧副本、重命名副本或
其他路径都会被拒绝，避免把 canonical 训练进度回滚。

`--resume-checkpoint latest.pt` 必须恢复 model、optimizer、global epoch/step、phase、
shard index 和 validation/test accumulator。checkpoint 还保留本次阶段初始化所用的
`initial_checkpoint_path` 与其 SHA256；resume 会验证这份初始化 lineage，不允许把
另一个 initial checkpoint 静默混入同一次 run。新阶段初始化和同一次 run 恢复是
两套不同语义。

进程崩溃后，在服务器使用首次启动时完全相同的 index、config、run prefix 和输出
设置，并显式添加已提交的 `latest.pt`：

```bash
cd /home/amax/ry/vlm_drones/uav_agent

./python.sh scripts/train_target_state_sharded.py \
  --config configs/target_state/train_oracle_clean.yaml \
  --shard-index /home/amax/ry/vlm_drones/datasets/stage_a_indexes/target_state_extreme_v1_15k_shard_index.json \
  --pc-trans-root /home/amax/ry/pc_trans \
  --pc-trans-config /home/amax/ry/pc_trans/config/config.json \
  --bridge-root /home/amax/ry/vlm_drones/datasets/_bridge \
  --run-id-prefix stagea_extreme_15k \
  --output-dir /home/amax/ry/vlm_drones/outputs/trained_models/target_state_extreme_15k \
  --run-name temporal_ray_depth_oracle_extreme_15k \
  --device cuda:0 \
  --resume-checkpoint /home/amax/ry/vlm_drones/outputs/trained_models/target_state_extreme_15k/temporal_ray_depth_oracle_extreme_15k/latest.pt
```

恢复程序依据 checkpoint 中的 epoch/phase/RUN_ID 与 `pc_trans shard-state` 对账，
而不是根据目录中碰巧存在的文件猜测进度。

### consume 中断时先恢复 pc_trans 事务

如果崩溃发生在 `consume --delete` 内部，或 active 目录出现未完成的私有
`.consume-backup.*`，不要直接启动 trainer，也不要手工移动/删除这些文件。先停止
同一 run 的旧 trainer，根据 `terminal.log` 或 `latest.pt` 确认受影响的 RUN_ID，
然后在服务器执行：

```bash
cd /home/amax/ry/pc_trans

python -m pc_trans.cli \
  --config /home/amax/ry/pc_trans/config/config.json \
  recover-active \
  --run-id stagea_extreme_15k.e0001
```

将示例 RUN_ID 换成 checkpoint/日志记录的实际值。`recover-active` 是 run-scoped：
它会把尚未发布删除记录的 active shard 恢复到该 run 的 ready 目录，并协调中断的
consume backup；若 `deleted=true` 已经可靠记录，则完成相应清理。它不会自动运行。
命令失败时保持 fail closed，不要用 `rm` 绕过。命令成功后，再用上一节的
`--resume-checkpoint .../latest.pt` 启动 trainer，让它重新查询 `shard-state` 并与
authoritative checkpoint 对账。

典型恢复判定如下：

- checkpoint 表示 shard 已完成、active tar 仍在：崩溃发生在 checkpoint commit
  后、consume 前。重新读取并验证 checkpoint，不重训，清理 materialized 后
  consume。
- checkpoint 表示 shard 未完成、active tar 仍在：从上一个安全 checkpoint 重新
  materialize，并从头重训当前 shard；不做 mid-shard resume。
- checkpoint 表示 shard 已完成，且 consumed record 为 `deleted=true`：正常进入
  下一个 shard。
- consumed 已为 true，但 authoritative checkpoint 表示 shard 未完成：违反提交
  顺序，必须 fail closed，不能猜测继续。

任何 shard 失败时，`terminal.log` 至少给出以下四个可直接用于恢复的字段：

```text
FAILED_SHARD=<failed shard identifier>
RUN_ID=<pc_trans run id>
SHARD=<shard_*.tar filename>
LATEST_SAFE_CHECKPOINT=<absolute latest.pt path>
```

只使用 `LATEST_SAFE_CHECKPOINT` 指向的 `latest.pt` 恢复；不要改用当时存在的
`best.pt`。

服务器删除任意 training/evaluation shard 前，以下条件必须全部成立：

1. 当前 shard train/eval 成功。
2. 对应 progress 已写入 authoritative checkpoint。
3. checkpoint 已完成临时文件、flush、fsync、原子 replace 和父目录 fsync。
4. committed checkpoint 已重新读取并验证。
5. checkpoint 的 `last_completed_shard` 与当前 shard 一致。
6. DataLoader 及其 worker 已全部退出。
7. materialized 数据已不再被访问并完成安全清理。
8. 最后才允许调用 `consume --delete`。

任何一步失败，active tar 必须保留。tar SHA、dataset checker、OOM、NaN、CUDA、
DataLoader 或 checkpoint 失败都不得 consume。临时 materialized 目录可以清理，
因为 active tar 和 PC 永久 shard 仍在。中断的 rsync 可在 PC 重跑同一 RUN_ID；
PC 原始数据和 shard 永远不由服务器训练流程删除。

### Run directory 独占锁

每个 `<output-dir>/<run-name>` run directory 由非阻塞的进程锁
`.sharded_training.lock` 独占。新训练和 resume 都必须先取得该锁；同一 run 已有
trainer 时，第二个进程会立即失败，不能并发执行 checkpoint、cleanup 或 consume。
恢复前先确认旧进程确实退出。锁由操作系统在进程退出时释放，残留的锁文件本身
不表示锁仍被占用，也不应手工删除来规避一个仍存活的持锁进程。

## 8. Stage A 输出与 Stage B

Stage A sharded checkpoint 除 model/optimizer 外还记录 parent dataset SHA、shard
index SHA、global progress、per-epoch run IDs 和 `training_protocol =
episode_sharded_v1`。最终 `best.pt` 顶层继续满足现有 Stage B 初始化契约：

```text
model_type = temporal_ray_depth_residual
schema_version = 1
training_stage = oracle_clean
model_state_dict = <Stage A best weights>
```

Stage B 的 YOLO deployment fine-tuning 应把这个文件作为新阶段初始化，而不是
同一 run 的 resume checkpoint。若 Stage B 也使用 sharded trainer，其 checkpoint
继续保留该 Stage A 文件的绝对 path 和 SHA256；下方非 sharded Stage B 则在最终
model manifest 的 `initial_checkpoint` 中记录同样的 path/SHA256，使初始化来源可
审计。例如非 sharded Stage B：

```bash
cd /home/amax/ry/vlm_drones/uav_agent

./python.sh scripts/train_target_state.py \
  --config configs/target_state/train_yolo_deployment.yaml \
  --dataset-root /path/to/stage_b_dataset \
  --initial-checkpoint /home/amax/ry/vlm_drones/outputs/trained_models/target_state_extreme_15k/temporal_ray_depth_oracle_extreme_15k/best.pt \
  --output-dir /home/amax/ry/vlm_drones/outputs/trained_models/target_state_stage_b \
  --device cuda:0
```

实际 `best.pt` 位于服务器启动命令最终打印的 run directory；使用前以该输出为准，
不要仅依赖上例的推测路径。

## 9. 估算 15k 数据集的起始 epoch 数

不要在代码中自动决定 epoch 数。先取得旧基线训练集和新 shard index 的 train
sequence count。旧完整数据集可在 PC 项目环境中用 dry-run 统计：

```bash
cd <PC上的vlm_drone/uav_agent>

python scripts/train_target_state.py \
  --config configs/target_state/train_oracle_clean.yaml \
  --dataset-root /path/to/old_target_state_dataset \
  --dry-run
```

记录输出 `splits.train.sequence_count` 为 `old_sequence_count`。新数据直接从刚生成
的 index 汇总 train entries：

```bash
python -c 'import json; from pathlib import Path; p=Path("/mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_15k_shards/shard_index.json"); d=json.loads(p.read_text(encoding="utf-8")); print(sum(int(s["sequence_count"]) for s in d["shards"] if s["split"] == "train"))'
```

将结果记为 `new_sequence_count`，用旧实验的 `old_epochs` 估算起始值：

```text
new_epochs ≈ old_epochs × old_sequence_count / new_sequence_count
```

该值只用于选择第一轮实验的 CLI `--epochs N`，不是自动策略，也不能写回代码
默认值。最终 epoch 数由 validation 曲线、收敛情况和后续真实实验确定。
