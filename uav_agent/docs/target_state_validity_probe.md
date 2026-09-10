# Geometry V2: frozen-feature validity head experiment

This is a constrained experiment, not an automatic dataset repair or deployment.
It does NOT claim that a validity classifier solves task-target identity. A
stable distractor may still need upstream class/attribute/association evidence.

## What runs

1. Verify the original Stage B best.pt, manifest, shard index and completed
   TRAIN association-review provenance. Require Geometry V2 protocols.
2. Transfer/extract TRAIN (84 shards) and validation (19 shards) once. Run the
   original model in eval mode, with all parameters frozen. Capture the 128-D
   input to validity_head; assert it reproduces the original logits. Store all
   windows, predictions, review metadata and small CPU feature tensors.
3. Train only a SEPARATE Linear(128,1), 129 parameters, on cached TRAIN features.
   Encoder, GRU/dropout, BatchNorm buffers, pixel/depth and covariance heads are
   unchanged. The raw geometry predictions cannot drift during this experiment.
4. Select an epoch using validation, at the unchanged **0.5** threshold. Demand
   fewer rejected supervision-positive measurements, with no increase over the
   original validation counts of no-target false positives, out-of-domain
   accepted targets or accepted position errors >1 m. The full original metric
   denominators remain in the report; this is not a replacement promotion gate.
   If no admissible improvement exists, retain the original head (best_epoch=0,
   candidate_improved=false). Do not manufacture a successful candidate.

Only TRAIN contributes gradients/class weights/hard-example weights. Validation
does not contribute gradients or repaired labels. No TEST shards or test error
cases are used for extraction, loss, sample weighting or epoch selection. This
validation set is a development set, not a new independent generalization test.

## Supervision and review

Training uses existing V2 measurement-valid labels. Raw archives stay unchanged.
Supervise a head only when the sensor input and frozen corrected geometry allow
a measurement; changing the head cannot rescue missing input/invalid geometry.
Unknown V2 supervision is masked. Windows flagged by the existing TRAIN review,
collector-unresolved windows, multiple labelled instance IDs or null/positive
transitions are conservatively masked from head fitting, not turned into
negatives or removed from evaluation. Such transitions can be legitimate
occlusion/reappearance: this is an ambiguity policy, not proof of mislabelling.

Other qualified negative labels are inherited from the existing dataset, not
certified by this experiment. TRAIN positive/negative classes are balanced;
original-head mistakes receive fixed 3x weight. Empty qualified positive or
negative sets stop training. No threshold sweep or test-derived label changes.

The prepare phase writes train_review.json and validation_review.json, including
the original validation supervision-positive rejections (58 in the current run).
It exports full RGB-D for the existing case categories (accepted false positives,
>=1m errors and model-only rejections). Not every masked ambiguous window gets
an image copy; all get review metadata. RGB-D case copies share a 2 GiB cap,
with budget omissions explicitly marked in per-shard receipts.

## Server

Enter `tmux new -As validity_probe`, then:

```bash
cd /home/amax/ry/vlm_drones/uav_agent
bash scripts/train_target_state_validity_probe_50k.sh
```

Preflight only: append `--dry-run` (no requests, training or output writes).

## PC / WSL

Enter `tmux new -As validity_probe_send`, then run:

```bash
cd ~/pc_trans
(
  set -euo pipefail
  export SSH_TARGET=vlm-data
  export LOCAL_SHARDS=/mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_50k_shards
  export RSYNC_TIMEOUT=300
  for split in train validation; do
    run_id="audit_validity_probe_50k_v1.${split}"
    attempts=0
    until ./pc_scripts/push_prefetch_once.sh "$run_id"; do
      attempts=$((attempts + 1))
      if [ "$attempts" -ge 10 ]; then
        echo "连续发送失败10次；检查网络后重跑，已有进度保留。" >&2
        exit 1
      fi
      sleep 10
    done
  done
  echo "PC发送完成；等待服务器Validity head experiment complete。"
)
```

103 shards, 53,982,556,160 bytes (~54.0 GB), 41,425 windows. No YOLO service or
Isaac Sim needed. Keep pc_trans_guard running and PC/WSL awake. Cache backpressure
remains active; extraction/report space is additional to the 2 GiB case limit.
PC source archives and previous model/audit files are never deleted. Only this
probe's verified server extraction/archive is consumed after receipt commit,
reread and case asset verification. Same commands resume preparation after a
disconnect. Do not change code/protocol or run duplicate processes mid-run.

Head fitting runs on CPU and needs no further transmission. If interrupted, the
small deterministic fit restarts from the original head, not a partial optimizer.
Once cache_manifest.json exists and is complete, rerun fitting alone with:

```bash
bash scripts/train_target_state_validity_probe_50k.sh --fit-only
```

Changing ONLY fit hyperparameters requires a NEW experiment_name in the YAML;
cached features can be reused with --fit-only, no PC command. A changed feature
contract/model/source requires a new preparation prefix and directory instead.

## Outputs

Root: `/home/amax/ry/vlm_drones/outputs/diagnostics/validity_probe_geometry_v2_50k_v1/`

* cache_manifest.json: complete split/receipt hashes, supervised counts, review
  counts and original model/baseline diagnostics. Receipts contain all features,
  predictions, masked windows and case evidence, not just successful examples.
* train_review.json / validation_review.json: suspect windows and head mistakes.
* head_balanced_v1/report.json: every epoch, baseline/candidate validation,
  candidate_improved, best_epoch, frozen-geometry checks and promotion_passed=false.
* head_balanced_v1/best_validity_head.pt: **head-only research artifact** tied to
  the original model SHA. It is not a complete temporal-model checkpoint and
  must NOT replace production best.pt or model_manifest.json.

Even an admissible head improvement requires review of remaining target-identity
errors, uncertainty reliability and fresh independent scenes/runtime/Kalman
validation. Current known TEST examples must not be recycled as training data
and then reported as an untouched independent test.
