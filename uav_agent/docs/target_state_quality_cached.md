# Frozen-geometry conditional quality experiment

This is an offline development experiment, NOT a deployed estimator or a
target-identity fix. It reuses the completed Geometry V2 feature cache without
PC uploads, raw archive extraction, YOLO, Isaac Sim, or a GPU. The original
producer, cached validity fitter, labels, source archives and model are unchanged.

## Why this experiment

The previous three validity-head fits completed but none satisfied validation
safety constraints. The mild one-epoch proposal recovered nine positive windows,
including one with 1.218 m error. In the original validation output, five of the
58 rejected supervision-positive windows already had frozen-prediction error
above 1 m. Observation validity and prediction accuracy are different targets.

Do not change a valid-target label to "no target" just because this model is
inaccurate. This experiment learns a SEPARATE conditional quality label, while
preserving the original measurement-valid supervision and full evaluation counts.

## Fixed experiment protocol

1. Strictly verify the existing TRAIN/validation cache using the unchanged
   reader: model/manifest/index/source hashes, all receipts and promised RGB-D
   assets, original probability replay, split identities and supervision masks.
   There is no cache migration, hash bypass, test extraction or PC request path.
2. From the original validity head, reproduce ONE TRAIN epoch of mild weighting
   (power .25, hard weight 1, lr 1e-4, original-logit anchor .1, seed 42). Freeze
   that proposal for the rest of the experiment. No previous saved "best" head
   is mistaken for this proposal: the previous best artifacts contain the
   original head. The proposal by itself is NOT an approved replacement.
3. Train a separate quality classifier only on original supervised, unambiguous,
   sensor/geometry-eligible TRAIN measurement positives. Its target is
   `frozen_model_error_m <= 1.0`. No-target, out-of-domain and unknown/reviewed
   observations are excluded from this conditional quality loss, not relabelled
   as geometric negatives. Validity training keeps their original policy.
4. Default independent comparisons: linear versus Linear(142,32)-SiLU-Linear(32,1),
   20 epochs each, CPU, lr .001, batch 256. Both start afresh. Quality class weights
   are derived from TRAIN only. The 13,368 accurate versus 39 inaccurate examples
   receive equal total class weight (individual risk weight about 342.77x).
   This is an explicit cost-sensitive objective, NOT calibrated probabilities.
5. Offline combined gate:

   `sensor/geometry eligible AND proposal_validity >= .5 AND quality >= .5`

   The quality score never bypasses a sensor/geometry/validity rejection. It
   cannot rescue an original rejection by itself: the separate mild validity
   proposal provides the possible recoveries. Geometry, uncertainty parameters,
   encoder, GRU and BatchNorm buffers never change.
6. Select using validation only, with the UNCHANGED main target (fewer rejected
   supervision-positive measurements) and no increases over original counts of
   no-target false positives, accepted out-of-domain targets, or >1 m accepted
   errors. Additional accurate-positive counts are diagnostic, not a replacement
   denominator or a relaxed promotion criterion. Original baseline is NOT replaced
   by the weaker one-epoch proposal when checking safety.

## Inputs and truth boundary

Quality inputs are 128 cached sensor-derived backbone features plus these 14
explicitly allowed, runtime-available values:

* log1p raw/corrected depth; signed log1p depth residual;
* normalized predicted pixel corrections and sensor anchor coordinates;
* four normalized detector bounding-box coordinates;
* three log predicted position variances.

No target coordinates, target velocity, target depth/pixel labels, target/object
IDs, episode/candidate/tracker IDs, occlusion ground truth, regions, motion seeds,
prim paths, offline error or evaluator response enter the input tensor. Runtime
eligibility uses sensor and model-geometry flags only. Offline error is TRAIN
supervision, never an input or runtime gate. Non-finite eligible inputs fail
closed rather than silently using a truth-derived replacement.

Mean/std normalization uses the same qualified TRAIN subset only (std floor
1e-4, normalized values clipped to [-10,10]); buffers are saved with the quality
head. Validation never provides normalization, class weights or gradients.

Important limitations: these TRAIN errors are in-sample for the already-trained
frozen backbone. The 39 risk windows represent only 19 training episodes, not 39
independent environments. This is not out-of-fold risk calibration. Validation
has been used repeatedly for development, not untouched generalization evidence.
Neither a high quality score nor a passed development comparison certifies task
identity, uncertainty calibration, Kalman safety, or real-world operation. New
held-out scenes and runtime/closed-loop checks are still required before deployment.

## Server command (PC: no command required)

Enter a persistent terminal:

```bash
tmux new -As quality_cached
```

Inside it:

```bash
cd /home/amax/ry/vlm_drones/uav_agent
bash scripts/train_target_state_quality_cached_50k.sh
```

Optional read-only full precheck (does not train or write experiment outputs):

```bash
bash scripts/train_target_state_quality_cached_50k.sh --dry-run
```

Do not run the old PC sender or the old validity suite for this experiment.
Configuration: `configs/target_state/quality_cached_50k.yaml`.

If interrupted, rerun the SAME server command. Completed subexperiments are
hash-verified and skipped. An unfinished small fit starts deterministically from
scratch; this is not partial optimizer resume. Do not modify code/config or run
duplicate instances mid-experiment. Intentional changes require a NEW
`experiment_name`, but the unchanged source feature cache can still be reused.

## Outputs and interpretation

Root:

`/home/amax/ry/vlm_drones/outputs/diagnostics/validity_probe_geometry_v2_50k_v1/quality_cached_v1/`

* `report.json`: complete/candidate_improved/selected_experiment, comparisons,
  class statistics and `promotion_passed=false`.
* `{quality_linear,quality_mlp32}/report.json`: every epoch, original baseline,
  validity-only proposal, selected joint result and artifact SHA256 values.
* `best_quality_bundle.pt`: original-model-bound HEADS ONLY; deployable=false.
  If no admissible improvement exists, epoch=0 means original validity weights,
  quality_enabled=false and quality_state_dict=null. The fallback truly preserves
  the original pipeline; it does NOT apply an unapproved quality gate.
* `last_quality_bundle.pt`: last-epoch heads, diagnostic only, even when a better
  earlier epoch was selected. This permits exact score/case replay after training.
* `validation_changes.json`: selected/last acceptance changes versus the original,
  including predictions and offline evidence fields for investigation. These
  evidence labels are NOT model inputs. RGB-D is not copied again; available
  cases remain in the verified cache/audit exports. Not every new case necessarily
  has a copied full RGB-D window.

Artifacts are atomically written, hashed and reread before completion. Changed
contracts, corrupt evidence and unowned/symlink output directories stop execution.
Neither bundle is a complete temporal checkpoint: NEVER replace production
`best.pt` or `model_manifest.json` with one. No automatic promotion occurs.
