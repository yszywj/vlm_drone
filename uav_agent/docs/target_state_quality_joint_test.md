# Pinned quality candidate: joint historical TEST audit

This evaluates a FIXED candidate against the original Geometry V2 model. It
does not train, select another epoch/head/threshold using TEST, or deploy a model.
The existing feature producer, fitters, audits, original model and all prior
reports remain unchanged. Do not rerun the quality training launcher for this.

## Candidate and preflight

Selected before TEST: `quality_cached_power050_v2/quality_linear`, epoch **19**,
`best_quality_bundle.pt` (NOT last, MLP32 or the original fallback artifact).

Candidate SHA256:

`584c8902d513005b3064cdaf1673d9ba01c0e212decd2f6109243eeafc1eea9d`

Original full Geometry V2 Stage B checkpoint SHA256:

`3c74793784f958f6cf46aec3f7b661132923cbfbabb85cf22b754df99e64119d`

Before any PC request, verify the pinned SHA/epoch, full model/index/protocol
provenance, source hashes and the completed selection report. The existing
103 TRAIN/validation feature receipts and promised case assets are verified
locally; no TRAIN/validation archives are requested. Replay the selected
validation scores AND acceptance-change records (normalizing JSON tuple/list
representation), and confirm the saved normalization is exactly TRAIN-derived.
There are no optimizers, refits or threshold sweeps in this audit.

Preflight failure stops before requests. In normal mode, unavailable CUDA also
stops before requests; the script never silently changes evaluation devices.

## What is evaluated

Only TEST: **17 shards, 6,788 windows, 8,498,401,280 archive bytes (~8.50 GB)**.
They are requested once under `audit_quality_joint_50k_v1.test`, not per epoch.
The base backbone runs once per batch, in eval/inference mode with frozen
parameters and BatchNorm. Its 128-D validity input is captured and checked against
the original logit. Original and candidate reuse IDENTICAL geometry/covariance.
The quality heads execute on CPU, with their saved normalization.

Candidate acceptance remains:

`sensor/geometry eligible AND validity >= 0.5 AND quality >= 0.5`

Only sensor-derived features and the explicit runtime quality-input allowlist
are passed to the heads. Ground-truth positions, IDs, occlusion and offline
measurement labels are evaluation evidence only; none may admit/reject a window
or remove it from the full split denominator. All TEST features are saved in a
NEW evaluation-only receipt directory, never merged into the TRAIN cache.

Both paths use the original full evaluator for failure, position and uncertainty
metrics. The candidate adapter uses `min(validity_logit, quality_logit)` solely
to represent the exact AND gate; this is not a calibrated joint probability.
Its evaluator mean_loss is diagnostic, not a new training objective. Original
metrics are compared with the base model's saved TEST metrics; any mismatch must
be investigated before interpreting differences.

Paired outputs include acceptance changes, identical-both-accepted errors,
no-target false positives and >1m/>5m case-set additions/removals. Equal error
counts alone do not conceal replacement by different bad cases. RGB-D evidence
is the union of original/candidate false positives, >=1m accepted outliers,
model-only rejections, acceptance changes and known episodes 000516/001821.
The byte-preserving case copier has a shared **2 GiB** budget; metadata is kept
for budget-omitted cases with explicit omission flags. This is NOT a bound on
total filesystem use: archives, extraction, reports and receipts are additional.

## Server

Enter:

```bash
tmux new -As quality_test
```

Then run:

```bash
cd /home/amax/ry/vlm_drones/uav_agent
bash scripts/audit_target_state_quality_test_50k.sh
```

Optional full read-only precheck:

```bash
bash scripts/audit_target_state_quality_test_50k.sh --dry-run
```

The precheck performs no TEST evaluation, PC request or output writes. Normal
evaluation needs GPU `cuda:0` for backbone inference and PC TEST shards. YOLO
and Isaac Sim services are NOT required: detections/RGB-D are already recorded.
The script uses the project's `python.sh` environment, not the active shell env.

## PC / WSL

Keep `pc_trans_guard` running on the server. Keep Windows/WSL awake: tmux does
not prevent Windows sleep or WSL shutdown. Enter a WSL terminal:

```bash
tmux new -As quality_test_send
```

Inside it, run this ONE block after starting the server command:

```bash
cd ~/pc_trans
(
  set -euo pipefail
  export SSH_TARGET=vlm-data
  export LOCAL_SHARDS=/mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_50k_shards
  export RSYNC_TIMEOUT=300
  attempts=0
  until ./pc_scripts/push_prefetch_once.sh audit_quality_joint_50k_v1.test; do
    attempts=$((attempts + 1))
    if [ "$attempts" -ge 10 ]; then
      echo "连续发送失败10次；检查网络后重跑相同命令，已有进度保留。" >&2
      exit 1
    fi
    sleep 10
  done
  echo "PC发送完成；联合评估是否完成以服务器report.json为准。"
)
```

This uses the EXISTING PC sender; no new script needs to be installed on the PC.
Missing request initially is normal while server preflight runs; if the server
errors out, fix that error rather than waiting indefinitely on the PC.

## Resume and deletion boundaries

Rerun the SAME commands after interruption. Do not edit code/config/pins or run
duplicate evaluators/senders mid-run. A dedicated prefix has one persistent
owner, and both owner/output directories are locked and contract-checked.

Per shard: verify/materialize archive -> infer both paths -> copy bounded cases
-> atomically persist paired rows/features/metric accumulators -> atomically
persist receipt SHA256 -> reread and replay receipt, verify every promised asset
-> clean ONLY that verified extraction -> consume/delete ONLY this run's server
archive. PC originals, old caches, models and past audit evidence are not removed.

An uncommitted receipt/checksum pair can be recomputed while the server archive
remains available. A committed receipt is never silently regenerated to hide
corruption. Missing/corrupt evidence after consume stops with an explicit error.
Completed receipts resume without backbone inference or re-upload. The existing
pc_trans disk/cache backpressure remains active; no quota is bypassed or changed.

## Outputs and next decision

Root:

`/home/amax/ry/vlm_drones/outputs/diagnostics/audit_quality_joint_50k_v1/`

* `report.json`: complete, pinned contract, original metric replay, full metrics
  for both paths, paired comparison and artifact/receipt digests.
* `test_original_samples.json`, `test_candidate_samples.json`: ALL windows.
* `test_changes.json`: each newly accepted/rejected window and original status.
* `test_cases.json`, `case_assets/`: union cases, unchanged temporal RGB-D and hashes.
* `receipts/`: TEST-only features/predictions/accumulators with SHA256 sidecars.

Check `original_training_metric_replay.matched` before the candidate comparison.
`comparison.fixed_test_safety_counts_nonregressing` and
`comparison.fixed_test_primary_improved` are diagnostic outcomes for the pinned
candidate, NOT model-selection or promotion decisions. `promotion_passed` remains
false regardless of test counts, and no deployment file is modified.

This historical TEST was previously inspected. Even if the fixed candidate
improves here, it is NOT new independent generalization evidence. Do not tune on
its failures and continue calling it untouched TEST. Fresh held-out scenes,
target association, uncertainty and runtime/Kalman/closed-loop verification are
still required before deployment. Do not replace production best.pt with the
head-only quality bundle.
