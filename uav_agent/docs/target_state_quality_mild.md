# Fixed mild-weight quality comparison

The completed `quality_cached_v1` used 39 risk windows versus 13,368 accurate
windows with equal total class weight: about 342.77x per risk window. It reduced
large-error acceptance but rejected too many accurate measurements; neither
head was selected. All old configs, source hashes, checkpoints and reports stay
unchanged. This comparison uses the SAME quality fitter and verified cache.

Only `quality_fit.class_balance_power` changes (plus separate output names):

| Power | Per-risk relative weight | Risk share of total loss weight |
|---|---:|---:|
| 0.25 | about 4.30x | about 1.24% |
| 0.5 | about 18.51x | about 5.12% |

Both retain the original linear and MLP32 heads: **four independent quality
fits, 20 epochs each**, sequentially on CPU. Each starts fresh; the 0.5 run does
not continue the 0.25 run. The one-epoch mild validity proposal is reproduced
unchanged within each configuration. No geometry weights, labels, normalization
policy, seed, learning rate, epoch budget, threshold or safety condition changes.
Do not confuse the QUALITY power with the unchanged VALIDITY power of 0.25.

The two configurations are checked before any fitting. A failed check or fit
stops the launcher. Cache/source/model/receipt/asset verification remains strict;
there is no upload, extraction, deletion, hash bypass or test-split usage.

## Server (PC: no command)

Enter `tmux new -As quality_mild`, then run:

```bash
cd /home/amax/ry/vlm_drones/uav_agent
bash scripts/train_target_state_quality_mild_50k.sh
```

Optional check only (both configurations; no training/output writes):

```bash
bash scripts/train_target_state_quality_mild_50k.sh --dry-run
```

No GPU, YOLO or Isaac Sim service is needed. Do not start PC rsync. If interrupted,
rerun the SAME command: completed subexperiments are verified/skipped; unfinished
ones restart deterministically. Do not change code/config or run duplicates
mid-run. No prior experiment directories need to be deleted or renamed.

## Results

Under `/home/amax/ry/vlm_drones/outputs/diagnostics/validity_probe_geometry_v2_50k_v1/`:

* `quality_cached_power025_v2/report.json`
* `quality_cached_power050_v2/report.json`

Inspect BOTH reports, plus each `quality_linear/` and `quality_mlp32/` history.
There is no combined cross-power promotion report or automatic deployment.
`admissible=true` is only the safety-count check; a candidate must ALSO improve
original positive-measurement rejection count. No improved candidate means the
original validity head with quality disabled. Best/last bundles are head-only
offline artifacts, NOT replacements for production `best.pt`.

This is one predeclared two-power comparison, not an open-ended search on the
same validation set. If neither satisfies the original criteria, stop repeating
hyperparameter trials here and collect/review independent difficult scenes.
The 39 risk windows still span only 19 TRAIN episodes. Even a selected candidate
needs fresh held-out scene, association, uncertainty and runtime/Kalman testing.
