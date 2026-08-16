#!/usr/bin/env python3
"""Run the minimal UAV search scene as an Isaac Sim standalone application."""

from __future__ import annotations

import argparse
import os
import sys
from math import ceil, isfinite
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = Path(
    os.environ.get(
        "UAV_AGENT_CONDA_ENV",
        "/home/amax/miniconda3/envs/r_isaac_sim",
    )
).expanduser()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import ConfigError, load_config  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to the unified YAML config")
    parser.add_argument("--steps", type=_positive_int, help="override simulation.demo_steps")
    parser.add_argument(
        "--uav-goal",
        nargs=3,
        type=_finite_float,
        metavar=("X", "Y", "Z"),
        help="command a continuous kinematic flight toward a world-frame goal in meters",
    )
    parser.add_argument(
        "--save-rgb",
        metavar="PATH",
        help="save the final onboard RGB frame (PNG is used when PATH has no suffix)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the config without importing or launching Isaac Sim",
    )
    parser.add_argument(
        "--debug-ground-truth",
        action="store_true",
        help="print synchronized privileged Target truth and in-frustum projection",
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true", help="force headless mode")
    display.add_argument("--no-headless", dest="headless", action="store_false", help="force GUI mode")
    parser.set_defaults(headless=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if Path(sys.prefix).resolve() != EXPECTED_ENV.resolve():
        print(
            "error: this project must run in the environment selected by "
            "UAV_AGENT_CONDA_ENV; use ./python.sh scripts/run_demo.py "
            "--config configs/default.yaml",
            file=sys.stderr,
        )
        return 2

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    config_path = Path(args.config).expanduser().resolve()
    print(f"Configuration valid: {config_path}")
    if args.validate_only:
        return 0

    # Standalone ordering is deliberate: SimulationApp must exist before any
    # isaacsim.core, omni, carb, or pxr dependent module is imported.
    from isaacsim import SimulationApp

    headless = config.simulation.headless if args.headless is None else args.headless
    simulation_app = SimulationApp({"headless": headless})
    environment = None
    try:
        from env.simple_uav_search_env import SimpleUavSearchEnv

        environment = SimpleUavSearchEnv(config)
        environment.setup()
        if not headless:
            environment.configure_overview_viewport()
        poses = environment.read_poses()
        print(f"uav_position={poses.uav_position.tolist()}")
        print(f"uav_orientation_wxyz={poses.uav_orientation.tolist()}")
        if args.debug_ground_truth:
            print(f"target_position={poses.target_position.tolist()}")
            print(f"target_orientation_wxyz={poses.target_orientation.tolist()}")
            print(f"target_motion_mode={config.target.motion.mode}")
        if args.uav_goal is not None:
            environment.move_uav_toward(args.uav_goal)
            print(f"uav_goal_m={args.uav_goal}")
        steps = args.steps if args.steps is not None else config.simulation.demo_steps
        print(
            "Isaac Sim scene ready: "
            f"headless={headless}, steps={steps}, camera={config.camera.resolution_wh_px}"
        )
        completed_steps = 0
        new_camera_frame = False
        while simulation_app.is_running() and completed_steps < steps:
            new_camera_frame = environment.step()
            completed_steps += 1

        # Saving or evaluator projection must use a Camera sample whose RGB,
        # UAV pose, Camera pose and Target truth all share one timestamp.
        if args.save_rgb is not None or args.debug_ground_truth:
            maximum_alignment_steps = ceil(
                (1.0 / config.camera.frequency_hz) / config.simulation.physics_dt_s
            ) + 2
            alignment_steps = 0
            while (
                simulation_app.is_running()
                and not new_camera_frame
                and alignment_steps < maximum_alignment_steps
            ):
                new_camera_frame = environment.step()
                alignment_steps += 1
                completed_steps += 1
            if not new_camera_frame:
                raise RuntimeError("Camera did not produce a synchronized RGB frame in time")

        final_poses = environment.read_poses()
        print(f"final_uav_position={final_poses.uav_position.tolist()}")
        if args.debug_ground_truth:
            evaluator_frame = environment.get_evaluator_frame()
            projection = evaluator_frame.target_projection
            print(f"final_target_position={evaluator_frame.target_position_m.tolist()}")
            print(
                "target_projection="
                f"uv={projection.pixels_uv[0].tolist()}, "
                f"depth_m={float(projection.depth_m[0]):.3f}, "
                f"in_frustum={bool(projection.visible[0])}, "
                f"camera_timestamp_s={evaluator_frame.observation.camera_timestamp_s:.3f}"
            )
        if args.uav_goal is not None:
            print(
                f"uav_goal_distance_m={environment.distance_to_goal():.3f}, "
                f"goal_reached={environment.goal_reached()}"
            )
        if args.save_rgb is not None:
            saved_path = environment.save_rgb(args.save_rgb)
            print(f"saved_rgb={saved_path}")
        print(f"Demo finished after {completed_steps} simulation steps")
        return 0
    except KeyboardInterrupt:
        print("Demo interrupted by user")
        return 130
    finally:
        try:
            if environment is not None:
                environment.close()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
