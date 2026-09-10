# Geometry V2 correction and verification run

## What changed

* New captures store the actual projected target centre, including off-image
  centres, and explicit `center_in_image` metadata. The checker verifies explicit
  projections against 3D labels. Legacy serialization/hashes remain unchanged.
* `supervision_protocol: projected_center_v2` recomputes centres in the loader
  AFTER raw shard validation. It changes labels, not Stage B input tensors.
  Visibility, target presence and centre-in-image are distinct. A centre outside
  the production output domain is not a valid current-centre measurement; the
  target/observation/history are retained. Out-of-image history centres do not
  supervise reprojection loss. Raw records and sequence counts are preserved.
* Legacy large-depth-residual associations with otherwise valid sensor input
  are uncertain, not proven negatives. Their geometric and validity supervision
  is masked, rather than rewriting labels or deleting windows. This uses the
  documented broad old-data depth prior (1 m / 10% + surface tolerance), NOT
  recovered object geometry. This prior is offline supervision only.
* `reference_guard_protocol: rgbd_consistency_v1` is a shared sensor-only guard:
  depth groups, RGB chromaticity, and bounded-window surface motion with camera
  pose compensation. It rejects multiple supported surfaces or an unsupported
  centre seed; it does not blindly select the nearest/farthest/largest object.
  It conservatively abstains for same-colour occluders too. Motion/color jumps
  are checked only within a candidate's supplied window, without global truth
  or persistent target state. Missing/rejected observations reset the hint.
* The old foreground depth sampler and network architecture are unchanged.
  A guarded model cannot bypass ambiguity rejection through RGB-D fallback.
  Checkpoint/manifest guard protocols must agree. Old artifacts default to no
  new guard; no existing production config or checkpoint is changed.
* V2 supervision, guard protocol and semantic source hashes enter the sharded
  resume contract. Never resume an old training run under these semantics.
  The sharded manifest now includes the explicit runtime input/preprocessing
  contract as well as the config/checkpoint identity.

## Limits and verification

The guard is a conservative heuristic, not segmentation or an identity oracle.
A stable dominant distractor can still escape it. Unknown associations require
future review; the old dataset was NOT completely/autonomously relabelled.
Rejected detections and out-of-domain targets remain in full evaluation
denominators. Model and deterministic baseline use the same sensor guard; do not
compare their guarded failure rate directly to an old unguarded baseline as if
the evaluation protocol were unchanged. No promotion threshold was relaxed.

The known training exports contain 60 physical captures, 180 records and 51
windows. Every asset hash matched. V2 retains all windows; no positive current
measurement contradicts the ideal geometric gate. In episode_001852, all eight
windows are sensor-rejected. In episode_001708, five detected references are
sensor-rejected and four have unknown association supervision. These are selected
hard training examples, not an independent accuracy or recall estimate.

CPU regression report:
`/home/amax/ry/vlm_drones/outputs/diagnostics/geometry_v2_local_check.json`

```bash
cd /home/amax/ry/vlm_drones/uav_agent
./python.sh scripts/check_target_state_geometry_v2.py \
  --review-root /home/amax/ry/vlm_drones/outputs/diagnostics/association_train_review_v1 \
  --output /home/amax/ry/vlm_drones/outputs/diagnostics/geometry_v2_local_check.json
```

## Training plan

A short VERIFICATION run, not a production-accuracy claim:

1. Stage A: two epochs, model-only warm start from the original 50k Stage A best
   checkpoint; new labels/guard, fresh optimizer and new output directory.
2. Stage B: five epochs from the NEW Stage A best checkpoint, followed by final
   test evaluation. Stage B must never accidentally initialize from the old
   Stage B model or resume its optimizer.

Existing PC shard/index data suffices. Source archives and their index SHA are
unchanged; no recollection/repacking/upload of a new 50k dataset is required.
Streaming still transfers each epoch's requested data. Total planned archive
traffic is 394,874,695,680 bytes (~395 GB), including both final tests; it is NOT
one 62.5 GB upload. Keep PC/WSL awake. Server cache limits remain in effect,
with additional extraction/checkpoint space required.

### Server

Enter `tmux new -As geometry_v2_50k`, then:

```bash
cd /home/amax/ry/vlm_drones/uav_agent
bash scripts/train_target_state_geometry_v2_50k.sh
```

The launcher runs A then B, stopping on failure. It automatically uses each
NEW run's canonical `latest.pt` if present. Stage A already completed on a retry
is resumed/finalized without re-requesting consumed training shards.

### PC / WSL

Enter `tmux new -As geometry_v2_send`, then:

```bash
cd ~/pc_trans
(
  set -euo pipefail
  export SSH_TARGET=vlm-data
  export LOCAL_SHARDS=/mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_50k_shards
  export RSYNC_TIMEOUT=300

  send_run() {
    local run_id="$1" attempts=0
    until ./pc_scripts/push_prefetch_once.sh "$run_id"; do
      attempts=$((attempts + 1))
      if [ "$attempts" -ge 10 ]; then
        echo "连续发送失败10次；检查网络后重跑，已有进度保留。" >&2
        return 1
      fi
      sleep 10
    done
  }

  for stage in stagea stageb; do
    if [ "$stage" = stagea ]; then epochs=2; else epochs=5; fi
    for ((epoch=1; epoch<=epochs; epoch++)); do
      run_id=$(printf '%s_geometry_v2_50k_v1.e%04d' "$stage" "$epoch")
      send_run "$run_id"
    done
    send_run "${stage}_geometry_v2_50k_v1.finaltest"
  done
  echo "PC发送完成；训练结果以服务器Stage B model_manifest.json为准。"
)
```

Either side can rerun its identical command after interruption; never start
duplicate senders/trainers, change code/protocol/epoch counts mid-run, or remove
receipts/checkpoints. A missing future request is an expected wait. If changing
epoch counts intentionally, synchronize BOTH PC loop and server config first.
PC completion alone does not mean model evaluation/promotion completed.

Final artifacts are under:
`/home/amax/ry/vlm_drones/outputs/trained_models/target_state_geometry_v2_50k/`

Inspect `stageb_geometry_v2_50k_v1/model_manifest.json`, matched model/baseline
acceptance, failure/false-positive counts, uncertainty calibration and hard-case
replays. A passed offline gate still requires independent-scene runtime/Kalman
and closed-loop validation. No production deployment is performed by this launcher.
