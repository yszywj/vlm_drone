# RGB-D association review V1 (offline only)

This change reviews labels; it does not retrain, promote, overwrite checkpoints,
rewrite the old dataset, or use oracle information in production perception.

## New captures

`training/target_state/isaac_capture.py` now shares one association implementation:
class/IoU proposals need detection-interior depth support within the object's
projected optical-z extent. Competing supported assignments are left unresolved,
not selected by greedy IoU. Fully occluded projected objects can also make an
overlapping detection unresolved. Sensor-side candidate/tracker IDs are unchanged.

`association_review_required: true` is optional OFFLINE record metadata, outside
`sensor_input` and `detector_prediction`. The records and original observations
are retained. Windows containing an unresolved reference OR history record are
excluded from supervision; rows are not removed before window construction, so
the builder cannot bridge a gap. An unresolved null label is not a negative
training example, even if its legacy frame-ID suffix says `false_positive`.
Legacy records without the optional field retain identical canonical serialization.
Old readers reject the added field rather than silently training on it.

Defaults: at least 3 valid pixels, 20% valid interior coverage, 50% support among
valid pixels. Surface tolerance is max(0.15 m, 2% midpoint depth). These are fixed
screening/association rules, not optimized against validation/test model errors.
Depth consistency is necessary evidence, not proof of physical object identity;
same-depth distractors still require better evidence such as mapped instance masks.

## Existing 50k captures

`scripts/audit_target_state_associations.py` scans train, validation and test with
the same policy. Old records lack object extents and an instance-ID mask mapping;
the scanner uses a deliberately broad centre-depth prior: +/-max(1 m, 10% depth),
plus the surface tolerance above. It also records when the existing foreground
sampler's selected ray lies outside that interval (0.2–200 m sampler limits,
matching this 50k model's manifest). A correct box under partial occlusion may
trigger this second flag: it is NOT automatically a wrong label.

Projected-centre-outside-box and null-to-positive candidate transitions are
diagnostics only. Neither is an automatic rejection. A finding marks all affected
temporal windows, including those with a clean reference and suspect history.
Findings are a versioned review manifest, not a ready-to-train repaired dataset.
No detections are rerun and neither YOLO nor Isaac/GPU training is needed.

Local regression on exported episode_000516: 20 physical captures, 64 records,
40 windows; 11 records need review, affecting 12 windows (7 suspect references).
The known track_3/red-target mismatch in frames 13–19 is flagged. The other four
are track_1 frames 1–4, with dominant target-depth ROI support but a foreground
sampled ray; these require occlusion/sampler review, not automatic relabelling.
This test episode cannot establish TRAIN corruption prevalence.

## Server

Run inside `tmux new -As association_audit`, then:

```bash
cd /home/amax/ry/vlm_drones/uav_agent
./python.sh scripts/audit_target_state_associations.py \
  --shard-index /home/amax/ry/vlm_drones/datasets/stage_a_indexes/target_state_extreme_v1_50k_shard_index.json \
  --pc-trans-root /home/amax/ry/pc_trans \
  --run-id-prefix audit_association_50k_v1 \
  --output-dir /home/amax/ry/vlm_drones/outputs/diagnostics/audit_association_50k_v1 \
  --wait-timeout 86400
```

Add `--dry-run` for a non-writing preflight. Expected: 120 shards, 62,480,957,440
archive bytes, 100,783 frame/target records, 48,213 temporal windows. Physical
capture count is computed from actual records during the scan, not confused with
those logical record/window counts.

## PC / WSL

Run inside `tmux new -As association_send`, then the existing PC script suffices:

```bash
cd ~/pc_trans
(
  set -euo pipefail
  export SSH_TARGET=vlm-data
  export LOCAL_SHARDS=/mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_50k_shards
  export RSYNC_TIMEOUT=300
  for split in train validation test; do
    run_id="audit_association_50k_v1.${split}"
    attempts=0
    until ./pc_scripts/push_prefetch_once.sh "$run_id"; do
      attempts=$((attempts + 1))
      if [ "$attempts" -ge 10 ]; then
        echo "连续发送失败10次；检查网络后重跑此命令，已有进度保留。"
        exit 1
      fi
      sleep 10
    done
  done
  echo "三个 split 已发送完成；审查是否完成请看服务器 report.json。"
)
```

Both blocks can be rerun after interruption with the SAME prefix/output. Do not
run duplicate senders or reviewers. Do not edit audit code/policy or remove its
receipts mid-run. Changed contracts fail closed and require a NEW prefix AND
output directory; they must not reuse consumed state.

SHA/index/materialization validation is reused from the existing shard pipeline.
Each per-shard JSON receipt has an identity/content checksum and is atomically
saved, fsynced, and reread before this audit's server extraction/archive is deleted.
Resume handles a crash between receipt commit and cache consumption. The PC source
archives and previous review exports/training outputs are never deleted. The
configured 30 GiB server transfer cache limit remains in effect; extraction and
reports require extra headroom, so it is not an exact total-disk upper bound.

Successful completion is `report.json` with `complete: true`, plus
`train_review_manifest.json`, `validation_review_manifest.json`,
`test_review_manifest.json`, and durable per-shard `receipts/`. A finished PC loop
means upload completion, not server review completion. Review the split-specific
findings before choosing repairs/recollection or training again; do not filter
only test outliers and report the resulting score as untouched held-out accuracy.
