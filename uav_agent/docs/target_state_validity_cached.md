# Cached validity V2: mild class weighting and original-output constraint

The completed head_balanced_v1 experiment is preserved. Its 600 qualified
negatives received about 22.3x the per-example weight of 13,407 positives at
equal hard-example status, and its selected head remained the original model.
This new small development experiment changes weighting, NOT the 0.5 gate or
the geometric/uncertainty network. It does not certify a target-identity fix.

## Cache compatibility and safety

The producer scripts/target_state_validity_probe.py stays byte-for-byte intact.
Its cache fingerprint includes the producer file, so editing the producer just
to change a head loss would invalidate that cache. A separate fitter reads the
existing V1 contract, validates the original model/index/review and all source
hashes, checks all 103 receipt hashes and their promised RGB-D evidence, then
checks that cached features reproduce the original probabilities/acceptance.
It also verifies split identities and cached supervision masks. There is no
hash bypass, rewritten cache manifest, original-data deletion, automatic
re-extraction or PC request path. Missing/mismatched cache stops the run.

Only CPU feature tensors and a fresh Linear(128,1) are used for fitting. No
encoder, GRU, BatchNorm, delta-pixel, depth or covariance parameter changes.
TRAIN alone supplies gradients, label counts, hard weights and anchor targets.
Unknown/ambiguous TRAIN supervision stays masked exactly as in the V1 cache.
Validation selects epochs/configurations; TEST does not participate. Validation
has now been used for repeated development, not independent generalization.

## Configurable objective

Negative/positive base sample-weight ratio is `(N_positive/N_negative)^power`:

* power=0: equal weight per example (no class balancing).
* power=0.25: about 2.17x per negative on this cache, not the old 22.3x.
* power=1: original full class balancing, available but not used by this suite.

Weights are subsequently multiplied for original-head training mistakes by
hard_example_weight and normalized to mean one. Class counts and resulting
negative loss-weight fraction are reported so the effective objective is clear.

`loss = weighted_BCE + anchor_logit_weight * mean((new_logit-old_logit)^2)`

The frozen original logits are targets only on qualified TRAIN samples. The
anchor is a SOFT output-preservation penalty, not a hard guarantee on score
changes. Per-epoch reports include BCE, anchor MSE and mean/max score shifts.

Default suite (10 epochs each, lr=1e-4, weight_decay=0, hard weight=1):

| Experiment | balance power | anchor weight |
|---|---:|---:|
| natural | 0 | 0 |
| natural_anchor | 0 | 0.1 |
| mild_anchor | 0.25 | 0.1 |

Every run starts from the ORIGINAL head, not the previous experiment. Selection
still requires fewer supervision-positive rejections with no increase over the
original validation counts of no-target false positives, accepted out-of-domain
centres or accepted >1m errors. No constraint is relaxed to manufacture progress.
If nothing improves, best_epoch=0 retains the original head and the suite reports
candidate_improved=false. Even an improvement is an offline head-only artifact,
not a deployable best.pt or a passed production promotion gate.

## Server commands (PC: no command required)

Enter `tmux new -As validity_cached`, then:

```bash
cd /home/amax/ry/vlm_drones/uav_agent
bash scripts/train_target_state_validity_cached_50k.sh
```

Optional read-only check first:

```bash
bash scripts/train_target_state_validity_cached_50k.sh --dry-run
```

This checks the FULL existing cache without training, uploads or output writes.
No YOLO, Isaac Sim or GPU service is required. No 54GB retransmission: all three
experiments share the same verified ~62MB receipt cache. Original case assets
remain in place for integrity verification and future inspection.

After interruption rerun the same command. Completed experiments are hash-
checked and skipped; an unfinished small deterministic fit restarts from the
original head. Do not modify code/config mid-run or run duplicate instances.

Config: configs/target_state/validity_cached_regularized_50k.yaml. For intentionally
different settings, choose a NEW suite_name; the cache can still be reused. Head
fitter code/config identity is recorded independently from the producer contract.

Outputs under the existing cache root:

`/home/amax/ry/vlm_drones/outputs/diagnostics/validity_probe_geometry_v2_50k_v1/head_regularized_v2/`

* report.json: complete, selected_experiment (null if no improvement), comparison
  and promotion_passed=false.
* natural/, natural_anchor/, mild_anchor/: separate per-epoch reports, contracts
  and best_validity_head.pt. Each artifact declares deployable=false.

The previous cache_manifest.json, head_balanced_v1 outputs, original complete
temporal best.pt and model_manifest.json are not overwritten. Do not point a
production model loader at these head-only artifacts. Fresh independent scenes
and runtime/association validation are still required.
