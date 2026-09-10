# V3 instance evidence and multi-surface pilot

This is an **opt-in, at-most-200 physical capture pilot**, not a new production
model, not bulk collection and not another hyperparameter search on historical
TEST. The previous 50k dataset, Geometry V2 checkpoint, source contracts and
runtime routing remain unchanged. All implementation is in new files outside
the old feature producer's source glob.

## Interfaces

* `target_state_v3/isaac_instances.py`: attach the installed Isaac Camera's
  uncolorized `instance_id_segmentation` annotator. Check the same renderer ID,
  timestamp, RGB and optical-Z depth as the published CameraSample. Validate
  mapped visible surfaces against the current rendered object depth extent.
  Missing/malformed/stale evidence stops collection; no synthetic mask fallback.
  A missing renderer ID 0 is accepted only if EVERY zero pixel has raw optical-Z
  depth **positive infinity** and the frame also contains a mapped positive,
  finite-depth surface. NaN, negative infinity, zero, finite/clipped depth and
  all-zero/uninitialized frames do not qualify. Nonzero missing IDs still fail.
  ID 1 is not assumed to mean UNLABELLED: renderer IDs may identify real prims.
  Original `id_to_prim` is unchanged (no fabricated `0: BACKGROUND` entry).
  The normalized zero entry has no prim/object ID; `background_evidence` records
  the rule and counts. Raw depth is retained in the oracle NPZ alongside the
  uint32 mask and replayed at append/seal/acceptance, never used as neural input.
* `association.py`: preserve uint32 instance IDs (>65535 included), full per-frame
  ID-to-prim mapping and object catalog. Exact path-component matching avoids
  cube_0/cube_01 confusion. Only unambiguous supported cube matches get positive
  labels. Known non-cubes/background and unresolved mappings are DISTINCT.
  Duplicate detections for one instance remain unresolved. Candidate IDs come
  only from the detector linker; missed objects never get truth-derived tracks.
* `measurement.py`: sensor-only RGB-D/calibration/self-pose/tracker observations.
  Form multiple depth-gap clusters without using the centre ray as a seed; choose
  an actual pixel belonging to a cluster. A unique dominant surface can initialize
  history; competing surfaces require a unique motion/appearance-supported match
  to a previously accepted surface. Reject ambiguous/discontinuous observations.
* `data.py`: opt-in sensor-only **Stage B** loader. Preserve raw unresolved rows;
  the temporal builder excludes whole windows containing them without bridging
  gaps. Oracle mappings are verification/labels, not model features. Existing ROI
  and geometry features are reused, while reference sampling and supervision
  are explicitly replaced with V3 semantics. The dataset exposes the exact
  `artifact_preprocessing` contract to pin in a future V3 training artifact.
* `runtime.py`: FrameStore -> same measurement function -> optional residual
  correction -> backprojection -> TargetMeasurement (compatible with Kalman).
  Default residual mode requires the exact V3 policy and explicit production
  approval. No V3 trained/approved artifact exists yet. Surface-only mode must be
  explicitly requested and is diagnostic, not object-centre localization.

The runtime module never imports the offline association/catalog module. A
surface can be geometrically consistent yet belong to a distractor; neither
temporal consistency nor depth proves class, colour, or mission target identity.
Attribute/target confirmation remains a separate required production step.
No assignment ID, object ID, prim path, target position/velocity, region, motion
seed or evaluator data is an input to the measurement selector.

The policy values are predeclared engineering defaults, NOT fitted to the old
TEST failures. The prototype may lose recall. Thin occluders, two same-depth
objects, lighting changes and missing history still need real pilot review.
Do not swap samplers underneath an old best.pt, use oracle checks as runtime
gates, or relabel/filter the historical TEST and call it independent validation.

## Server: first 200 physical captures

YOLO must stay running (unlike the completed offline tests). The new entrypoint
verifies loopback `/health`, `/v1/model-info`, class 0=cube and the pinned SHA before
importing Isaac. The checked model SHA is:

`895de7caa8af200c12f343c72e3a726ffae65e4d96d2092decaf96ef4558de07`

The ID-zero fix changes the source contract. Keep the failed `v3_pilot_200_v1`
directory and PC snapshot; use **v3_pilot_200_v2** for the corrected pilot.
Do not edit the old session/hash manifest to force a resume.

Check `/health` before starting. `active_stream_id` must be null or
`v3s20260910:uav_1`. A different stream must be released by its owner only after
confirming that its consumer is stopped. The collector does NOT take over other
streams automatically; do not restart/reset a shared YOLO service blindly.

Enter `tmux new -As collect_v3_pilot`, then:

```bash
cd /home/amax/ry/vlm_drones/uav_agent
(
  set -euo pipefail
  ./python.sh scripts/collect_target_state_v3_pilot.py \
    --output /home/amax/ry/vlm_drones/outputs/collection_sessions/v3_pilot_200_v2 \
    --run-id v3_pilot_200_v2 \
    --captures 200 \
    --scene-seed 20260910 \
    --request-timeout-s 30 \
    --gpu-device 3 \
    --oracle-label-generation \
    --acknowledge-privileged-oracle

  python3 - <<'PY'
import json
from pathlib import Path
root = Path("/home/amax/ry/vlm_drones/outputs/collection_sessions/v3_pilot_200_v2")
state = json.loads((root / "session.json").read_text())
if state.get("complete") is not True or state.get("last_error"):
    raise SystemExit("Collection incomplete: " + str(state.get("last_error")))
PY

  ./python.sh scripts/check_target_state_v3_pilot.py \
    --session /home/amax/ry/vlm_drones/outputs/collection_sessions/v3_pilot_200_v2
)
```

`--preflight-only` on the collection command checks YOLO/configuration without
creating files, allocating an Isaac GPU, or proving live segmentation works.
The actual pilot is needed to verify the installed renderer/scene combination.
The explicit journal check is intentional: Isaac's fast shutdown can terminate
the process before Python propagates a collection exception. A shell exit code
alone does not establish success. GPU 3 avoids the concurrent GPU 0/2 workload
observed during diagnosis; check `nvidia-smi` before running if workloads change.

This is 10 complete episodes x20 synchronized RGB-D/instance acquisitions at
5 Hz, NOT 200 target records or temporal windows. Positive, crossing/partial
occlusion and negative scene plans cycle with a new scene seed. The CLI refuses
more than 200 captures; do not bypass the cap before acceptance review.

## PC / WSL: parallel receive, no source deletion

Enter `tmux new -As receive_v3_pilot`. Copy the two small receiver files:

```bash
mkdir -p ~/target_state_v3_tools
for script_path in scripts/pull_target_state_v3_pilot.py target_state_v3/verify_tar.py; do
  rsync -rvh --no-perms --no-owner --no-group \
    "vlm-data:/home/amax/ry/vlm_drones/uav_agent/$script_path" \
    ~/target_state_v3_tools/
done

python3 ~/target_state_v3_tools/pull_target_state_v3_pilot.py \
  --ssh-target vlm-data \
  --server-session /home/amax/ry/vlm_drones/outputs/collection_sessions/v3_pilot_200_v2 \
  --output /mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_v3_pilot_200_v2
```

Only Python's standard library, ssh and rsync are needed on PC. Complete episode
archives are copied while the next episode collects. Archive SHA256 and every
internal asset SHA256 are checked without extraction; links, traversal, duplicate
members and oversized archives are refused. `.rsync-partial` enables retry.
There is no `--delete`, `--remove-source-files`, remote deletion or overwrite of
a different verified archive. Do NOT also run the old pull_collection_follow
against this PC folder: V3 pilot envelopes are not V1 collection shards.

Network errors retry up to 10 times. Rerun the same command after fixing the
connection. Keep Windows/WSL awake; tmux does not prevent system sleep.

## Resume, disk and acceptance

The server command resumes the SAME run ID/output after checking its contract.
It reuses complete verified episodes; incomplete work is moved to `recovery/`
for inspection, not counted as progress. Completed episodes and archives remain
on the server. A code/config/seed/YOLO mismatch requires a new run ID/output.
Do not launch duplicate collectors/receivers or edit code mid-collection.

The default pilot receiver reads retained server `archives/`, not the global
pc_trans ready queue. Existing pc_trans free-space/pause thresholds are respected
before an episode. Optional `--publish` additionally seals a copy through the
existing pc_trans CLI, but is NOT used in the commands above. The recommended
pilot is intentionally not the bulk rolling-deletion pipeline.

At 640x480, float32 depth alone is about 246 MB for 200 captures. Lossless RGB,
compressed uint32 instances, compressed raw-depth evidence and metadata add
variable overhead. Retained episode
directories plus uncompressed tar copies roughly double that; budget **1-2 GB**
on the server for this pilot, more if many failures are preserved. Recovery files
are not an exact bounded cache; no unattended bulk loop should use this entrypoint.

Server outputs in the run directory:

* `session.json`: complete and summary.physical_captures=200; per-episode checksums.
* `s20260910_episode_*/`: lossless RGB-D and separate `oracle/` instance assets;
  captures contain V3 envelopes, not legacy frames.jsonl.
* `archives/`: one immutable tar per complete episode.
* `acceptance_report.json`: CPU schema/integrity and training/runtime sampling
  checks. `structural_checks_passed=true` is NOT accuracy or production approval.

PC completion is `pc_receipt.json` with `complete=true`, `physical_captures=200`
and `archive_and_internal_assets_verified=true`. PC receive completion does not
prove the server's subsequent CPU acceptance check passed.

If `supervised_windows=0`, the report explicitly sets preprocessing parity to
`null` with `not_run_no_supervised_windows`: structural checks alone do not
prove any training/runtime window was compared.

## ID-zero fix validation (2026-09-10)

The independent `outputs/collection_sessions/v3_smoke_20_v2` run completed 20
physical captures and one verified tar. Separate live probes passed six frames
across positive, partial-occlusion and negative scenes. The previous missing-zero
exception no longer occurs; raw zero-pixel depths were confirmed to be +inf.
Old 50k training/feature source hashes were unchanged.

The smoke run is **not training-ready**: all 20 YOLO detections were unresolved,
with zero supervised windows. In its first detection, the ROI had 270 cube
pixels and 34 `/World/Ground/geom` pixels. Ground is outside the current object
catalog and counts as unknown (11.18%), above the existing 5% unknown gate.
Later frames also contain real distractor overlap. No thresholds were relaxed
and no unknowns were relabeled as negatives to make the pilot pass.

The 200-frame command remains a diagnostic evidence pilot for reviewing this
association issue and scene diversity, not approval to expand to 50k or train.
The retained masks, original per-frame mapping, raw depth and RGB-D enable a
subsequent explicitly versioned label-policy review without losing raw evidence.

## Offline association and temporal-window diagnosis

The server still holds the verified episode directories and archives. No PC
upload, YOLO service, Isaac process, GPU or training job is needed for this step.
Use the V3 entrypoint below, not the legacy shard/cache association audit.

```bash
cd /home/amax/ry/vlm_drones/uav_agent
./python.sh scripts/diagnose_target_state_v3_pilot.py \
  --session /home/amax/ry/vlm_drones/outputs/collection_sessions/v3_pilot_200_v2 \
  --output /home/amax/ry/vlm_drones/outputs/diagnostics/v3_pilot_200_v2_diagnosis_v1
```

The command verifies source hashes, archives and raw evidence, replays the
unchanged association policy, attributes unknown ROI pixels to exact prims,
and lists candidate continuity and each proposed window's first failing gate.
It cross-checks the window ledger against both `build_sequences` and the actual
CPU V3 loader. Unknown rows are never dropped to bridge temporal gaps, and no
ground pixels are relabeled. Its output is diagnostic evidence, not a dataset.

`summary.json` contains aggregate counts; `diagnosis.json` also contains all
detection/candidate/window details. Both are in the separate output directory.
Existing output directories are refused: choose a new suffix for a rerun.
Collection files, their labels and manifests, old training sources and all
model checkpoints remain untouched. The new script is outside producer source
hash globs, so adding the diagnostic does not invalidate collection resumes.

Next inspect matched/unresolved/non-cube counts, missing detections, visibility,
ID mapping images and selected surfaces. Both `ready_for_bulk_collection` and
`production_approved` remain false. No automatic expansion, formal training CLI,
model deployment, Kalman/flight control test or promotion is performed. A V3
training/streaming integration should follow real pilot acceptance; the old
finalizer/train command must not be pointed at these archives.

## Versioned exact-ground reassociation sidecar

After reviewing the pilot's unknown-pixel attribution, this separate offline
tool can reinterpret **only** `/World/Ground/geom` as a known non-target scene
surface. It verifies the original producer hashes (including `env/scene.py`),
the original same-frame ID-to-prim mapping and positive finite raw depth at
every ground pixel. This is NOT the ID-zero/+infinity no-hit background rule.
Other unknown prims, duplicate detections and all association thresholds remain
unchanged. Renderer IDs are resolved per frame, never assumed to be 1 or 13.

Labels come only from the original same-frame matched/missed object records.
Candidate/tracker IDs, RGB-D, camera calibration and self-pose are untouched.
Original no-candidate oracle rows remain as evidence and are skipped by the
same temporal builder; their old `missed_cube` count is not a newly computed
detector-miss count. Unresolved rows cannot be removed to bridge a window.

The new `target_state_v3_derived/` package and CLI files are outside the original
producer hash globs. No old dataset/model/manifest is rewritten. The output is
a metadata-only **sidecar**, not a self-contained copy of RGB-D and not a format
automatically accepted by an existing training CLI. Keep the server originals.

Server, CPU only (no YOLO/Isaac/GPU or PC upload):

```bash
cd /home/amax/ry/vlm_drones/uav_agent
(
  set -euo pipefail
  ./python.sh scripts/reassociate_target_state_v3_pilot.py \
    --session /home/amax/ry/vlm_drones/outputs/collection_sessions/v3_pilot_200_v2 \
    --output /home/amax/ry/vlm_drones/outputs/derived_annotations/v3_pilot_200_v2_ground_v1

  ./python.sh scripts/check_target_state_v3_reassociation.py \
    --derived /home/amax/ry/vlm_drones/outputs/derived_annotations/v3_pilot_200_v2_ground_v1
)
```

Creation refuses any existing output directory. For a fresh derivation choose
a new version suffix in BOTH commands; to recheck an existing version run only
the second command. The checker replays all new associations/labels from the
original evidence, checks sidecar/code/config hashes, builds the actual 7-frame
V3 training windows and compares their sensor-only measurement results. It
writes `acceptance_report.json` in the sidecar directory, never in the source.

`complete=true` and `overlay_replay_verified=true` mean the sidecar passed this
structural/replay check. `ready_for_training=false`, `production_approved=false`
and `needs_manual_visual_review=true` are intentional: accepted measurement
windows do not establish label accuracy, generalization or deployment safety.
Inspect recovered positive windows and remaining unresolved cases before
designing a training run or expanding collection.

Optional PC/WSL metadata backup AFTER the server check succeeds:

```bash
mkdir -p /mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_v3_pilot_200_v2_ground_v1
rsync -rvh --no-perms --no-owner --no-group --partial --checksum \
  -e ssh \
  vlm-data:/home/amax/ry/vlm_drones/outputs/derived_annotations/v3_pilot_200_v2_ground_v1/ \
  /mnt/c/Users/ry/datasets/trans/yolo_datasets/target_state_v3_pilot_200_v2_ground_v1/
```

This backs up annotations/reports only; retain the original verified 10 PC tar
archives too. No source deletion and no PC training/prefetch sender is involved.

Validation on 2026-09-10: 91 related CPU tests passed. A temporary sidecar for
the real `v3_pilot_200_v2` (session SHA256
`265eb7c75c46d7daab38fe5fb52bcfc07d952c5538c6fc5d57f134f15cb7296f`)
passed full replay of all 200 captures/10 episodes. Detected-row counts changed
from 22 matched / 86 unresolved / 3 known non-cube to 73 / 35 / 3. Eligible
7-frame windows changed from 41 (all negative) to 63 (22 positive-reference,
41 negative-reference), with 20 positive measurement windows. The other two
positive-reference windows have projected target centres outside the image;
the existing supervision gate correctly retains their rejection. All 55 pinned
producer file hashes and the source session SHA stayed unchanged. This is
annotation/loader validation, not a model accuracy result.

## V3 positive-only geometry overfit smoke test

The independent `target_state_v3_smoke/` package does not change the producer
or ground-overlay source globs. It accepts a fully replay-verified sidecar only,
retains episode splits with seed 42, and caches the eligible positive windows.
For `v3_pilot_200_v2_ground_v1` these are 18 TRAIN and 2 VALIDATION windows; TEST
has no eligible windows. The 41 all-missing negative windows and two off-image
centre windows are reported but do not participate in this geometry-only test.
No test model evaluation or validation-based selection/early stopping occurs.

This is **not a continuation of the old Stage B weights**. It uses the existing
CNN/geometry/GRU architecture with a randomly initialized backbone and exactly
zero initial pixel/depth residuals. Only the two residual heads and shared
encoders/GRU are optimized. The validity head is frozen at reject-by-default;
the variance head is frozen and uncalibrated. No deployment or full training
approval is implied, so `--acknowledge-diagnostic-only` is required explicitly.

The objective supervises the same reference timestamp's pixel centre, optical-Z
depth and backprojected world centre. It does not project a moving reference
target into old images as if stationary. Labels and oracle mappings never enter
the network forward inputs (`roi_rgbd`, `geometry`, `missing_mask`). The cached
zero-residual geometry is checked against the shared sensor-only V3 selector.
All preselected positives stay in the evaluation denominator; invalid rays fail
the test rather than disappearing from the loss/metrics.

Server: optionally enter `tmux new -As v3_geometry_smoke`, then run:

```bash
cd /home/amax/ry/vlm_drones/uav_agent
./python.sh scripts/train_target_state_v3_smoke.py \
  --derived /home/amax/ry/vlm_drones/outputs/derived_annotations/v3_pilot_200_v2_ground_v1 \
  --output /home/amax/ry/vlm_drones/outputs/diagnostics/v3_pilot_200_v2_geometry_smoke_v1 \
  --steps 300 \
  --batch-size 6 \
  --learning-rate 0.001 \
  --device cpu \
  --threads 4 \
  --acknowledge-diagnostic-only
```

CPU is intentional for this small bounded test; no YOLO, Isaac, PC prefetch
sender or GPU allocation is required. Original RGB-D and archives remain on
the server, are not recopied per step and are never deleted. Maximum limits are
200 physical captures, 64 positive windows per split and 1,000 optimizer steps.
Existing output directories are rejected; an interrupted/finished run needs a
new version suffix, not editing source hashes or overwriting old diagnostics.

The initial evidence replay/cache phase prints each episode. Optimization prints
metrics every 25 steps and writes `progress.json`. Final outputs are:

* `diagnostic_manifest.json`: exact sources, episode/window inventory, split,
  model/loss options, fixed-step and non-deployment declarations.
* `smoke_report.json`: zero-residual baseline, TRAIN fit, descriptive held-out
  VALIDATION metrics, gradient/reload checks and `smoke_passed`.
* `diagnostic_checkpoint.pt`: deliberately incompatible diagnostic model type;
  not `best.pt`, not a runtime artifact, not a resumable formal Stage B model.
* `failure.json`: only when a run fails after creating its own output directory.

The predeclared pass criteria are: TRAIN mean centre error and reference loss
each at most half the zero-residual baseline, nonzero gradients through all five
trainable component groups, and an identical saved-checkpoint replay. This
only tests wiring/fit; 18 highly overlapping windows cannot establish useful
generalization. `complete=true` means the requested steps/evaluation finished;
if `smoke_passed=false` the CLI exits 2 and the report remains available. Never
lower the criteria or mix in validation/test examples just to make it pass.

After completion, inspect on the server:

```bash
python3 -m json.tool /home/amax/ry/vlm_drones/outputs/diagnostics/v3_pilot_200_v2_geometry_smoke_v1/smoke_report.json
```

PC/WSL needs no upload. Optionally back up the small diagnostic directory:

```bash
mkdir -p /mnt/c/Users/ry/datasets/trans/yolo_datasets/v3_pilot_200_v2_geometry_smoke_v1
rsync -rvh --no-perms --no-owner --no-group --partial --checksum -e ssh \
  vlm-data:/home/amax/ry/vlm_drones/outputs/diagnostics/v3_pilot_200_v2_geometry_smoke_v1/ \
  /mnt/c/Users/ry/datasets/trans/yolo_datasets/v3_pilot_200_v2_geometry_smoke_v1/
```

Next review TRAIN improvement versus the unchanged surface baseline, then the
two held-out positives (descriptive only). If wiring succeeds, design fresh
independent episodes with longer tracks, partial occlusion and persistent false
detections before formal geometry/validity/covariance training. This script does
not automatically collect additional data or change any production routing.

Implementation validation (2026-09-10): 107 related CPU tests passed. A separate
temporary 30-step real-pilot run completed in about 68 seconds including evidence
replay/cache creation. TRAIN mean centre distance decreased from 0.43866 m to
0.13572 m over 18 windows; all five required trainable component groups received
nonzero gradients and checkpoint reload matched. The two held-out positives
were evaluated only descriptively (0.59044 m baseline / 0.19757 m final). This
short implementation test used no TEST model evaluation and establishes neither
generalization nor an approved production artifact. The user command above
creates its own independent 300-step run in a new persistent output directory.
