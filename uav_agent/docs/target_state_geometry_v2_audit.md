# Geometry V2 diagnostic replay (not more training)

This run evaluates the SHA-verified Stage B best checkpoint (epoch 4) on the
original validation/test splits. YOLO detections are already recorded in those
archives: no YOLO service or Isaac Sim process is needed. It does not update
weights, supervision, source archives, production config, or the 0.5 gate.

New evidence includes:

* Exclusive failure counts: missed detection, invalid/out-of-range raw depth,
  RGB-D consistency guard, bad input anchor/image, corrected pixel outside the
  image, corrected depth outside the range, invalid ray/world, validity head.
  Independent overlapping flags are also retained; ordered attribution is not
  proof that changing the first flag alone would recover a measurement.
* Model/baseline comparison on **the identical both-accepted samples**, plus
  model-only rejections and baseline-only rejections. Reported main denominators
  stay unchanged. Offline-only target-domain/unknown-supervision subgroups help
  distinguish impossible centre measurements from potential mistaken rejection.
* Every model false positive, accepted error >= 1 m, model-only rejection, and
  sequence from the known hard TEST episode_000516 is indexed in cases.json.
  Its full temporal RGB-D window is copied unchanged while its shard is present.
  All selected cases retain raw records and predictions, even if asset storage
  is exhausted. The two splits share a **2 GiB** raw-asset budget (not a limit on
  reports, extraction, receipts, transfer cache or filesystem overhead). Cases
  skipped due to this budget are explicitly marked, not claimed replayable.
  Selection is per-shard with false positives first, not a global top-K ranking.
  Earlier TRAIN case episodes are deliberately not mixed into test statistics.
* Metric replay is compared to the original model manifest with small numeric
  tolerance; any difference appears in report.json. This audit is not runtime
  Kalman or closed-loop verification and does not certify promotion.

## Server

Enter `tmux new -As geometry_v2_audit`, then run:

```bash
cd /home/amax/ry/vlm_drones/uav_agent
bash scripts/audit_target_state_geometry_v2_50k.sh
```

Optional preflight: append `--dry-run`. It validates hashes, protocols, source
index and requested episode without issuing transfer requests or writing files.

## PC / WSL

Enter `tmux new -As geometry_v2_audit_send`, then run this single block:

```bash
cd ~/pc_trans
(
  set -euo pipefail
  export SSH_TARGET=vlm-data
  export LOCAL_SHARDS=/mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_50k_shards
  export RSYNC_TIMEOUT=300

  for split in validation test; do
    run_id="audit_geometry_v2_50k_v1.${split}"
    attempts=0
    until ./pc_scripts/push_prefetch_once.sh "$run_id"; do
      attempts=$((attempts + 1))
      if [ "$attempts" -ge 10 ]; then
        echo "连续发送失败10次；检查网络后重跑本命令，已有进度保留。" >&2
        exit 1
      fi
      sleep 10
    done
  done
  echo "PC发送完成；诊断是否完成以服务器report.json为准。"
)
```

This sends 19 validation + 17 test shards once, not the training split or seven
epochs again. A missing future test request is normal while validation runs.
Keep WSL/PC awake; tmux does not prevent Windows sleep or WSL shutdown. Existing
transfer backpressure still applies. The server has additional extraction space
requirements beyond archive cache and evidence budgets.

Both sides may rerun their identical commands after interruption, without
deleting receipts or changing the prefix, options, model or source code mid-run.
Server deletion is limited to this audit's verified extraction/archive, AFTER
durable per-shard predictions/metrics and promised case assets are reread and
verified. PC originals and training artifacts are never deleted. Corrupted or
missing promised case evidence stops the audit rather than claiming completion.

## Outputs

Root: `/home/amax/ry/vlm_drones/outputs/diagnostics/audit_geometry_v2_50k_v1/`

* `report.json`: complete=true, original-metric replay comparison, paired errors,
  failure/false-positive counts and case export coverage.
* `{validation,test}_samples.json`: all predictions/flags, including rejects.
* `{validation,test}_cases.json`: selected temporal windows, unchanged source
  records, predicted points/pixels, relative RGB-D paths and SHA256 digests.
* `case_assets/`: unchanged RGB images and depth arrays (bounded total bytes).
* `receipts/`: durable per-shard metric tensors and evidence for safe resume.

No threshold calibration on the test set. Diagnose first; tune any future
changes on validation, then use fresh held-out scenes for final confirmation.
