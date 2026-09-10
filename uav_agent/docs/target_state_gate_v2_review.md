# Measurement gate V2 and the 50k Stage B review

The network architecture, best.pt, probability threshold (0.5), and production
configuration are unchanged. No model is promoted by this review.

`reference_rgbd_gate_v2` requires a current reference detection, finite in-range
raw depth, in-image anchor and corrected pixel, in-range corrected depth, and
validity probability >= 0.5. The shared sensor-only predicate is used by the
production resolver, training evaluator, and full sharded auditor. No target
presence/visibility/truth label participates in acceptance.

Explicit learned rejection or invalid corrected geometry cannot be overridden
by deterministic fallback. Fallback for unavailable temporal history remains
available when configured, with its own source and statistics. These are not
Kalman prediction or closed-loop task-success metrics.

Position median/P95, occlusion/jitter error, and covariance/error correlation
now use accepted visible measurements. Failure rate still uses **all** visible
targets, including misses. Accepted and evaluated counts are both recorded.
For matched baseline comparisons, use the auditor's `both_accepted` block.
No promotion threshold was loosened. The evaluation state schema and sharded
training contract changed, preventing old/new evaluation states from mixing.
Do not resume old training/audit run IDs across this change. Existing model
weights can still be loaded for a new, separately named offline evaluation.

## Saved-output reanalysis

Run `scripts/review_target_state_audit.py --help`. It needs only existing audit
JSON and the pinned model manifest, not PC data or GPU. It writes to a separate
directory. Older rows lack image dimensions: the report explicitly marks this
as a **partial** reference/depth guard replay, not a full V2 runtime evaluation.
Threshold sweeps and a candidate global covariance scale are fitted to the
validation split only. Neither is applied to model/configuration files.
An empirical global coverage adjustment is not an extreme-outlier guarantee.

Current 50k review output:
`/home/amax/ry/vlm_drones/outputs/diagnostics/audit_stageb_50k_v1/review_gate_v2/review.json`

## Review episode 000516

The only required archive is `shard_stagea_test_000004.tar` (525,107,200 bytes).
The PC original stays intact; the inspector also retains the server archive.
The inspector verifies archive SHA/index before exporting the episode's
unchanged RGB/depth/mask assets and offline labels. It does not relabel data.

Server request (already prepared for this review):

```bash
cd /home/amax/ry/vlm_drones/uav_agent
./python.sh scripts/inspect_target_state_episode.py \
  --episode-id episode_000516 \
  --shard-index /home/amax/ry/vlm_drones/datasets/stage_a_indexes/target_state_extreme_v1_50k_shard_index.json \
  --output-dir /home/amax/ry/vlm_drones/outputs/diagnostics/episode_000516_review \
  --request-only
```

PC/WSL upload:

```bash
cd ~/pc_trans
SSH_TARGET=vlm-data \
LOCAL_SHARDS=/mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_extreme_v1_50k_shards \
RSYNC_TIMEOUT=300 \
./pc_scripts/push_prefetch_once.sh audit_review_episode_000516_v1
```

Server verification/export after upload (or run while waiting for PC):

```bash
cd /home/amax/ry/vlm_drones/uav_agent
./python.sh scripts/inspect_target_state_episode.py \
  --episode-id episode_000516 \
  --shard-index /home/amax/ry/vlm_drones/datasets/stage_a_indexes/target_state_extreme_v1_50k_shard_index.json \
  --output-dir /home/amax/ry/vlm_drones/outputs/diagnostics/episode_000516_review
```

Inspect `episode_review.json` and its `assets/`, especially frames 13–16 and
the preceding false-positive track. Distinguish foreground/occluder depth,
detector error, and offline truth association before changing data or training.
After root-cause correction, perform a full V2 replay with a NEW audit prefix
and output directory; do not overwrite the original audit. Test cases already
inspected become regression cases, so reserve independent scenes for final
acceptance rather than repeatedly tuning on this test split.
