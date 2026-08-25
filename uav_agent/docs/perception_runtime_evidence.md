# Production YOLO runtime evidence

Production target perception is routed per UAV. Every UAV owns a distinct
`FrameStore`, `CandidateBank`, YOLO tracker stream and `TargetStateEstimator`.
Fleet summaries publish these counters under `perception_by_uav.<uav_id>`;
they are observational only and never feed flight-control decisions.

The production call chain is:

```text
CameraSample
  -> YoloTargetPerceptionRuntime.observe
  -> CoordinatedVisionPerceptionBackend.observe
  -> TargetPerceptionCoordinator
  -> YoloServiceClient /v1/track
  -> CandidateBank + temporal attribute evidence
  -> TargetMeasurement -> TargetStateEstimator
  -> TargetEstimate -> MissionAgent Observation
  -> SEARCH / TRACK / REACQUIRE
```

The three target Skills consume only `Observation.target_estimate`; they do
not import or call the YOLO client and therefore remain backend-neutral.

## Call-chain counters

| Counter | Meaning |
| --- | --- |
| `camera_frames_received` | Fresh synchronized `CameraSample` values accepted by the coordinator. |
| `yolo_requests_submitted` | Requests actually submitted to `/v1/track`; superseded pending frames are not included. |
| `yolo_results_received` | Valid route-matched worker responses received, including responses later rejected as stale. |
| `detections_total` | All detections in accepted worker responses. |
| `tracked_detections_total` | Detections carrying the worker's tracker ID. The current service contract requires this for every detection. |
| `candidate_created` | Candidate evidence epochs registered with `TargetManager`. |
| `candidate_confirmed` | Candidate epochs accepted by track, semantic and identity confirmation. |
| `candidate_rejected` | Candidate epochs explicitly rejected by confirmation. |
| `attribute_confirmed` | Deterministic/Qwen semantic results that matched the requested attributes. |
| `attribute_ambiguous` | Attribute checks that remained pending or ambiguous. |
| `depth_resolution_attempts` | Candidate-to-3D resolver calls, including stale-measurement rejection. |
| `depth_resolution_successes` | Resolver calls producing a validated `TargetMeasurement`. |
| `depth_resolution_failures` | Resolver failures plus the retained legacy count for estimator rejection. |
| `measurement_created` | Validated RGB-D `TargetMeasurement` values created. |
| `measurement_rejected` | Measurements rejected by geometry/age validation or by the estimator gate. |
| `kalman_updates_accepted` | Target measurements accepted by the Kalman innovation gate. |
| `kalman_updates_rejected` | Target measurements rejected by the Kalman timestamp, covariance or innovation gate. |
| `position_world_outputs` | Non-null visual or predicted world positions exposed as `TargetEstimate`. |
| `predicted_only_outputs` | Exposed `TargetEstimate` values using bounded Kalman prediction only. |
| `search_target_found` | Candidate-to-lock confirmations that allow SEARCH to return `TARGET_FOUND`. |
| `track_visible_updates` | Visible confirmed updates exposed while `TargetManager` is tracking. |
| `track_predicted_updates` | Prediction-only updates exposed while `TargetManager` is tracking. |

The older `yolo_requests`, `yolo_successful_responses`, `candidates_total`,
`candidates_confirmed` and `candidates_rejected` counters remain for report
compatibility. Assignment-scoped segments are summed only within the same UAV.

## Candidate transition log

`[PerceptionCandidate]` is emitted only on candidate creation, rejection and
confirmation—not on every frame. Each JSON record is restricted to timestamp,
UAV/Assignment routing, tracker/candidate IDs, normalized bbox, detector
confidence, attribute/color result, geometry state, estimated world position,
confirmation state, logical target alias and estimate source.
Terminal emission is additionally capped at 256 transition records per
assignment, so a pathological stream of new tracker IDs cannot create an
unbounded log.

The record never contains simulator target truth, prim paths, instance IDs,
motion seeds, raw RGB/depth frames or video. Raw frames and videos remain
disabled by default; debug images remain separately bounded by the existing
`debug_images.max_images_per_run` configuration.

## Production preflight

`scripts/run_fleet_mission.py` contacts `/health` and `/v1/model-info` before
its first `isaacsim` import. It requires the configured model family, exact
class mapping and checkpoint SHA256. `scripts/run_single_uav_yolo_e2e.sh`
starts the isolated worker, waits for readiness, runs this preflight and then
uses the existing Fleet mission runtime. Stopping the worker therefore aborts
before Isaac startup; there is no Oracle or disabled fallback.
