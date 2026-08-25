#!/usr/bin/env python3
"""Collect synchronized Isaac RGB-D plus deployed-YOLO target-state labels.

The default ``isaac`` mode contacts and verifies the real loopback YOLO worker
*before* importing Isaac, then runs one detector request at each Camera/World
barrier.  ``external`` mode remains available only to copy forensic spools and
marks their detector provenance unverified; it cannot masquerade as deployed
YOLO training data.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from math import cos, sin
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from datasets.target_state.dataset import read_frame_records  # noqa: E402
from datasets.target_state.schema import UavFrameInput  # noqa: E402
from perception.yolo_client import YoloServiceClient  # noqa: E402
from configs.loader import ConfigError, load_config  # noqa: E402
from scripts.collect_yolo_dataset import (  # noqa: E402
    _SimpleSceneCollectionAdapter,
    _randomization_bounds,
)
from training.target_state.collector import (  # noqa: E402
    TargetStateCollectionError,
    TargetStateDatasetWriter,
    require_privileged_collection_acknowledgements,
)
from training.target_state.isaac_capture import (  # noqa: E402
    TargetStateFrameAssembler,
    preflight_deployed_yolo,
)
from training.yolo.collection_scene import load_cube_collection_protocol  # noqa: E402
from training.yolo.isaac_collector import EpisodeRandomizer  # noqa: E402
from yolo_service.protocol import (  # noqa: E402
    ResetStreamRequest,
    TargetQuery,
    TrackRequest,
)


EXPECTED_ENV = Path(
    os.environ.get("UAV_AGENT_CONDA_ENV", "/home/amax/miniconda3/envs/r_isaac_sim")
).expanduser()
DEFAULT_MODEL_SHA256 = (
    "895de7caa8af200c12f343c72e3a726ffae65e4d96d2092decaf96ef4558de07"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("isaac", "external"), default="isaac")
    parser.add_argument("--source-dataset", type=Path)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--collection-config",
        type=Path,
        default=_PACKAGE_ROOT / "configs" / "yolo" / "collect_cube.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yolo-model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--yolo-url", default="http://127.0.0.1:8011")
    parser.add_argument("--request-timeout-s", type=float, default=5.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--scene-seed", type=int, default=42)
    parser.add_argument("--history-size", type=int, default=6)
    parser.add_argument("--max-history-age-s", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--frames-per-episode", type=int, default=20)
    parser.add_argument("--sample-hz", type=float, default=5.0)
    parser.add_argument("--min-bbox-area-px", type=float, default=16.0)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--oracle-label-generation", action="store_true")
    parser.add_argument("--acknowledge-privileged-oracle", action="store_true")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser


def _load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"depth_m"}:
                raise ValueError(f"NPZ depth must contain only depth_m: {path}")
            return archive["depth_m"]
    with Image.open(path) as image:
        return np.asarray(image)


def _materialize_external(args: argparse.Namespace) -> int:
    """Copy a forensic spool without granting deployed-detector provenance."""

    if args.source_dataset is None:
        raise ValueError("--source-dataset is required when --mode external")
    source = args.source_dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output:
        raise ValueError("source-dataset and output must be different directories")
    records = read_frame_records(source / "frames.jsonl")
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("max-frames must be positive")
        records = records[: args.max_frames]
    writer = TargetStateDatasetWriter(
        output,
        yolo_model_sha256=args.yolo_model_sha256,
        split_seed=args.split_seed,
        history_size=args.history_size,
        max_history_age_s=args.max_history_age_s,
    )
    try:
        for record in records:
            with Image.open(source / record.sensor_input.rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            depth = _load_depth(source / record.sensor_input.depth_path)
            mask = None
            if record.sensor_input.instance_mask_path is not None:
                with Image.open(source / record.sensor_input.instance_mask_path) as image:
                    mask = np.asarray(image)
            writer.append(record, rgb=rgb, depth_m=depth, instance_mask=mask)
        manifest, report = writer.finalize()
    except Exception:
        writer.abort()
        raise
    print(
        json.dumps(
            {
                "ok": True,
                "mode": "external",
                "note": (
                    "external spool copied with unverified detector provenance; "
                    "it is not eligible for yolo_deployment training"
                ),
                "dataset_manifest": str(manifest),
                "report": report.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


def _uav_input(environment: object) -> UavFrameInput:
    observation = environment.get_agent_observation()
    state = observation.uav_state
    velocity = np.asarray(observation.uav_velocity_mps, dtype=np.float64)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise RuntimeError("Isaac UAV velocity snapshot is invalid")
    yaw = float(state.yaw)
    return UavFrameInput(
        position_world_m=(float(state.x), float(state.y), float(state.z)),
        orientation_world_wxyz=(cos(yaw / 2.0), 0.0, 0.0, sin(yaw / 2.0)),
        linear_velocity_world_mps=tuple(float(value) for value in velocity),
        # This collection adapter commands constant yaw during each episode.
        angular_velocity_body_radps=(0.0, 0.0, 0.0),
    )


def _collect_isaac(args: argparse.Namespace) -> int:
    if args.source_dataset is not None:
        raise ValueError("--source-dataset is only valid with --mode external")
    for name in ("max_episodes", "frames_per_episode"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.scene_seed < 0 or args.gpu_device < 0:
        raise ValueError("scene-seed and gpu-device must be non-negative")
    if args.sample_hz <= 0.0 or args.min_bbox_area_px <= 0.0:
        raise ValueError("sample-hz and min-bbox-area-px must be positive")
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        raise ValueError(f"configuration error: {exc}") from exc
    protocol = load_cube_collection_protocol(args.collection_config)
    client = YoloServiceClient(
        base_url=args.yolo_url,
        request_timeout_s=args.request_timeout_s,
    )
    # Critical ordering: this completes before the first isaacsim import.
    receipt = preflight_deployed_yolo(
        client,
        expected_model_sha256=args.yolo_model_sha256,
    )
    print(
        "Target-state YOLO preflight passed: "
        f"url={receipt.worker_url}, family={receipt.model_family}, "
        f"names={dict(receipt.model_names)}, sha256={receipt.model_sha256}"
    )
    if args.preflight_only:
        return 0
    if Path(sys.prefix).resolve() != EXPECTED_ENV.resolve():
        raise RuntimeError(
            "run Isaac collection with ./python.sh or set UAV_AGENT_CONDA_ENV"
        )

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(args.headless),
            "active_gpu": args.gpu_device,
            "physics_gpu": args.gpu_device,
            "anti_aliasing": 0,
            "multi_gpu": False,
            "fast_shutdown": True,
            "extra_args": [
                "--/rtx/post/aa/op=0",
                "--/rtx-defaults/post/aa/op=0",
                "--/rtx-transient/post/aa/limitedOps=false",
                "--/app/hydra/renderSettings/useUsdAttributes=false",
                "--/app/hydra/renderSettings/useFabricAttributes=false",
            ],
        }
    )
    environment = None
    writer: TargetStateDatasetWriter | None = None
    try:
        from env.simple_uav_search_env import SimpleUavSearchEnv

        environment = SimpleUavSearchEnv(config)
        environment.setup()
        adapter = _SimpleSceneCollectionAdapter(
            environment,
            simulation_app,
            config,
            protocol=protocol,
            crossing_trajectories=True,
        )
        writer = TargetStateDatasetWriter(
            args.output,
            verified_yolo_deployment=receipt,
            split_seed=args.split_seed,
            history_size=args.history_size,
            max_history_age_s=args.max_history_age_s,
        )
        assembler = TargetStateFrameAssembler(
            uav_id="uav_1",
            minimum_bbox_area_px=args.min_bbox_area_px,
            maximum_candidate_gap_s=args.max_history_age_s,
        )
        randomizer = EpisodeRandomizer(
            _randomization_bounds(config),
            scene_seed=args.scene_seed,
        )
        maximum_captures = (
            args.max_episodes * args.frames_per_episode
            if args.max_frames is None
            else args.max_frames
        )
        mission_id = f"targetstate_{args.scene_seed}"
        stream_id = f"{mission_id}:uav_1"
        captures = records_written = 0
        for episode_index in range(args.max_episodes):
            if captures >= maximum_captures:
                break
            plan = randomizer.plan(episode_index)
            # Deterministic coverage: multi-target crossing, no-target, and
            # ordinary positives are present without relying on lucky draws.
            sample_kind = ("positive", "partial_occlusion", "negative")[
                episode_index % 3
            ]
            plan = replace(
                plan,
                sample_kind=sample_kind,
                # At the default 5 Hz x 20 frames, this lower bound makes the
                # two crossing-profile cubes pass one another inside the
                # recorded episode instead of merely approaching off-camera.
                target_speed_mps=(
                    max(
                        plan.target_speed_mps,
                        min(0.75, float(config.target.max_speed_mps)),
                    )
                    if sample_kind == "partial_occlusion"
                    else plan.target_speed_mps
                ),
            )
            adapter.begin_episode(plan)
            assembler.reset_episode()
            client.reset_stream(
                ResetStreamRequest(
                    schema_version=1,
                    request_id=f"reset_{episode_index:06d}",
                    mission_id=mission_id,
                    uav_id="uav_1",
                    stream_id=stream_id,
                )
            )
            assignment_id = f"assignment_{episode_index:06d}"
            for frame_index in range(args.frames_per_episode):
                if captures >= maximum_captures:
                    break
                adapter.advance_to_next_sample(1.0 / args.sample_hz)
                capture_id = (
                    f"episode_{episode_index:06d}_frame_{frame_index:06d}"
                )
                # The adapter rechecks Camera/World synchronization before
                # exposing privileged geometry.  World state remains frozen
                # while the loopback detector processes this exact RGB frame.
                truth = adapter.capture_oracle_frame(capture_id)
                sample = truth.camera_sample
                response = client.track(
                    TrackRequest(
                        schema_version=1,
                        request_id=f"request_{episode_index:06d}_{frame_index:06d}",
                        mission_id=mission_id,
                        uav_id="uav_1",
                        stream_id=stream_id,
                        frame_id=capture_id,
                        timestamp_s=sample.timestamp_s,
                        target_query=TargetQuery(class_ids=(0,), text_prompts=()),
                    ),
                    sample.rgb,
                )
                records = assembler.assemble(
                    capture_id=capture_id,
                    episode_id=plan.key.episode_id,
                    assignment_id=assignment_id,
                    truth=truth,
                    uav_input=_uav_input(environment),
                    response=response,
                )
                if sample.depth_to_image_plane_m is None:
                    raise RuntimeError("target-state collection requires synchronized depth")
                for record in records:
                    writer.append(
                        record,
                        rgb=sample.rgb,
                        depth_m=sample.depth_to_image_plane_m,
                        asset_id=capture_id,
                    )
                    records_written += 1
                captures += 1
        manifest, report = writer.finalize()
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": "isaac",
                    "physical_capture_count": captures,
                    "record_count": records_written,
                    "dataset_manifest": str(manifest),
                    "detector_deployment": receipt.to_manifest_dict(),
                    "report": report.to_dict(),
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 0
    except Exception:
        if writer is not None:
            writer.abort()
        raise
    finally:
        try:
            if environment is not None:
                environment.close()
        finally:
            simulation_app.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        require_privileged_collection_acknowledgements(
            oracle_label_generation=args.oracle_label_generation,
            acknowledge_privileged_oracle=args.acknowledge_privileged_oracle,
        )
        return (
            _materialize_external(args)
            if args.mode == "external"
            else _collect_isaac(args)
        )
    except KeyboardInterrupt:
        print(json.dumps({"ok": False, "error": "collection interrupted"}), file=sys.stderr)
        return 130
    except (OSError, TypeError, ValueError, RuntimeError, TargetStateCollectionError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        if os.environ.get("UAV_AGENT_COLLECTION_TRACEBACK") == "1":
            traceback.print_exc(file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
