# Target State collection spool workflow

Collection archives are a transport format, not Stage A training shards.  The
supported lifecycle is:

```text
Isaac + deployed YOLO
  -> complete-episode collection tar files
  -> pc_trans collection_spool/ready
  -> PC permanent archive storage
  -> one finalized parent dataset
  -> build_target_state_shards.py
  -> Stage A training shards
```

The collector never opens an SSH connection and never writes a PC mount.  It
only invokes the local `pc_trans seal` CLI after a complete archive has been
written and fsynced.

## Server preparation

Initialize the bridge and keep the quota guard running:

```bash
cd /home/amax/ry/pc_trans
python -m pc_trans.cli --config config/config.json init
./scripts/start_guard_tmux.sh
```

Start the deployed YOLO service separately and verify that
`http://127.0.0.1:8011` reports the model SHA expected by the collector.

## 200-capture acceptance run

Ten complete episodes at twenty captures each produce 200 physical captures.
The deliberately small 64 MiB soft target should exercise more than one
collection shard (an episode is allowed to take a shard above the target).

```bash
cd /home/amax/ry/vlm_drones/uav_agent
./python.sh scripts/collect_target_state_dataset.py \
  --mode isaac \
  --storage-mode collection-spool \
  --config configs/default.yaml \
  --collection-config configs/yolo/collect_cube.yaml \
  --pc-trans-root /home/amax/ry/pc_trans \
  --pc-trans-config /home/amax/ry/pc_trans/config/config.json \
  --bridge-root /home/amax/ry/vlm_drones/datasets/_bridge \
  --collection-session-dir /home/amax/ry/vlm_drones/outputs/collection_sessions \
  --collection-shard-size-mib 64 \
  --scene-seed 42 \
  --max-episodes 10 \
  --frames-per-episode 20 \
  --max-frames 200 \
  --sample-hz 5 \
  --gpu-device 0 \
  --oracle-label-generation \
  --acknowledge-privileged-oracle \
  --headless
```

The command prints the exact session directory and collection index.  To
resume after a collector or server restart, repeat the identity and collection
arguments and add:

```bash
--resume-session /home/amax/ry/vlm_drones/outputs/collection_sessions/<collection_id>
```

Resume starts at `session.json.next_episode_index`, which advances only after
an entire shard has been sealed.  A partial episode/workspace is never treated
as progress.

## Continuous PC/WSL pull

Run this on the PC/WSL side.  The destination is permanent storage:

```bash
cd /path/to/pc_trans
SSH_TARGET=vlm-data \
REMOTE_BRIDGE_ROOT=/home/amax/ry/vlm_drones/datasets/_bridge \
LOCAL_COLLECTION_SHARDS=/mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_collection_shards \
POLL_SECONDS=5 \
./pc_scripts/pull_collection_follow.sh
```

The one-shot script uses `--remove-source-files` only after a successful file
transfer and never uses `--delete`.  A network failure therefore leaves the
sealed server archive available for a later retry.  The server quota guard
creates `control/pause_collection.flag`; the collector finishes and seals its
current episode, then waits before starting another episode.

After collection completes, copy the small completed index to the PC (archive
payloads continue to arrive only through the follow script):

```bash
rsync -av \
  vlm-data:/home/amax/ry/vlm_drones/outputs/collection_sessions/<collection_id>/collection_index.json \
  /mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_collection_shards/
```

## Finalize on the PC

The finalizer verifies every archive checksum, safely extracts it, rejects
duplicate episodes/frames/assets, rebuilds the full manifest and dataset hash,
and requires `check_dataset().ok`.  Source tar files remain untouched.

```bash
cd /path/to/vlm_drones/uav_agent
python3 scripts/finalize_target_state_collection.py \
  --collection-index /mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_collection_shards/collection_index.json \
  --shard-dir /mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_collection_shards \
  --output-dir /mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_50k
```

Only after this succeeds should the ordinary training-shard builder consume
the parent dataset:

```bash
python3 scripts/build_target_state_shards.py \
  --dataset-root /mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_50k \
  --output-dir /mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_50k_shards
```

## 50,000-capture production run

After the 200-capture server/PC acceptance and interruption recovery tests have
all passed, use 2,500 complete episodes and the normal 512 MiB soft target:

```bash
cd /home/amax/ry/vlm_drones/uav_agent
./python.sh scripts/collect_target_state_dataset.py \
  --mode isaac \
  --storage-mode collection-spool \
  --config configs/default.yaml \
  --collection-config configs/yolo/collect_cube.yaml \
  --pc-trans-root /home/amax/ry/pc_trans \
  --pc-trans-config /home/amax/ry/pc_trans/config/config.json \
  --bridge-root /home/amax/ry/vlm_drones/datasets/_bridge \
  --collection-session-dir /home/amax/ry/vlm_drones/outputs/collection_sessions \
  --collection-shard-size-mib 512 \
  --scene-seed 42 \
  --max-episodes 2500 \
  --frames-per-episode 20 \
  --max-frames 50000 \
  --sample-hz 5 \
  --gpu-device 0 \
  --oracle-label-generation \
  --acknowledge-privileged-oracle \
  --headless
```
