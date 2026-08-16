"""Opt-in Isaac integration test for the complete Stage-0 Oracle pipeline.

The normal unit-test suite deliberately skips this module's test.  Run the
single expensive integration session at the end of a change set with:

    UAV_AGENT_RUN_ISAAC_TESTS=1 \
      ./python.sh -m unittest tests.test_full_skill_pipeline_oracle -v
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.loader import load_config  # noqa: E402


RUN_ISAAC_TESTS = os.environ.get("UAV_AGENT_RUN_ISAAC_TESTS") == "1"


class _WorldClock:
    def __init__(self, environment: object) -> None:
        self._environment = environment

    def now(self) -> float:
        world = getattr(self._environment, "world", None)
        if world is None:
            raise RuntimeError("environment World is not available")
        return float(world.current_time)


def _integration_config() -> object:
    """Return a small deterministic scene without writing a temporary YAML."""

    base = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    initial_region = replace(
        base.target.initial_region,
        min_xyz_m=(6.0, 0.0, 0.5),
        max_xyz_m=(6.0, 0.0, 0.5),
    )
    motion = replace(
        base.target.motion,
        mode="STATIC",
        speed_mps=0.0,
        region=replace(
            base.target.motion.region,
            min_xyz_m=(-10.0, -10.0, 0.5),
            max_xyz_m=(10.0, 10.0, 0.5),
        ),
    )
    return replace(
        base,
        simulation=replace(base.simulation, headless=True),
        uav=replace(
            base.uav,
            initial_position_xyz_m=(0.0, 0.0, 0.0),
            max_speed_mps=5.0,
            max_yaw_rate_deg_s=180.0,
        ),
        camera=replace(base.camera, resolution_wh_px=(160, 120)),
        target=replace(base.target, initial_region=initial_region, motion=motion),
        search=replace(
            base.search,
            radius_m=2.0,
            timeout_s=8.0,
            transit_yaw_mode="FACE_POINT",
        ),
    )


def _plan_dicts() -> list[dict[str, object]]:
    return [
        {
            "skill": "TAKEOFF",
            "target_altitude": 4.0,
            "tolerance": 0.15,
            "climb_speed": 2.0,
            "timeout": 5.0,
        },
        {
            "skill": "GOTO",
            "position": [0.0, 0.0, 4.0],
            "tolerance": 0.2,
            "timeout": 2.0,
        },
        {
            "skill": "SEARCH",
            "center": [6.0, 0.0, 0.5],
            "radius": 2.0,
            "target_description": "red moving target",
            "search_altitude": 4.0,
            "transit_speed": 3.0,
            "scan_yaw_rate": 1.0,
            "timeout": 8.0,
        },
        {
            "skill": "TRACK",
            "target_id": "$SEARCH.result.target_id",
            "desired_distance": 6.0,
            "desired_altitude": 4.0,
            "max_speed": 3.0,
            "max_target_lost_time": 0.25,
            "track_duration": 0.6,
        },
        {
            "skill": "LAND",
            "ground_altitude": 0.0,
            "tolerance": 0.1,
            "descent_speed": 2.0,
            "timeout": 5.0,
        },
    ]


def _name(value: object) -> str:
    if value is None:
        return "NONE"
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    enum_name = getattr(value, "name", None)
    if isinstance(enum_name, str):
        return enum_name
    return str(value)


def _transition_mapping(entry: object) -> Mapping[str, object]:
    if isinstance(entry, Mapping):
        return entry
    to_dict = getattr(entry, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    if is_dataclass(entry) and not isinstance(entry, type):
        return asdict(entry)
    raise AssertionError(f"unsupported transition entry: {entry!r}")


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    iterator = iter(actual)
    return all(any(item == wanted for item in iterator) for wanted in expected)


@unittest.skipUnless(
    RUN_ISAAC_TESTS,
    "set UAV_AGENT_RUN_ISAAC_TESTS=1 for the one final Isaac integration run",
)
class FullOracleSkillPipelineIsaacTest(unittest.TestCase):
    """Own exactly one SimulationApp and execute exactly one integration test."""

    simulation_app = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # No env.scene/simple_uav_search_env import may precede this line.
        from isaacsim import SimulationApp

        cls.simulation_app = SimulationApp({"headless": True})

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            if cls.simulation_app is not None:
                cls.simulation_app.close()
        finally:
            cls.simulation_app = None
            super().tearDownClass()

    def test_complete_pipeline_with_camera_fov_recovery(self) -> None:
        # Isaac-backed imports are deliberately local and occur after app setup.
        from env.simple_uav_search_env import SimpleUavSearchEnv
        from perception.oracle import OraclePerception
        from skills.manager import (
            SkillManager,
            TaskPlan,
            TaskStatus,
            create_default_skill_registry,
        )

        config = _integration_config()
        environment = SimpleUavSearchEnv(config)
        try:
            environment.setup()
            oracle = OraclePerception(target_id="target")
            clock = _WorldClock(environment)
            context = environment.make_skill_context(clock, perception=oracle)
            registry = create_default_skill_registry(
                transit_yaw_mode=config.search.transit_yaw_mode
            )
            manager = SkillManager(context, registry=registry)
            manager.start_task(TaskPlan.from_dicts(_plan_dicts()))

            deadline_s = clock.now() + 20.0
            seen_active: list[str] = []
            saw_rgb = False
            saw_visible_projection = False
            fault_injected = False
            target_restored = False

            while (
                self.simulation_app.is_running()
                and manager.task_status is TaskStatus.RUNNING
                and clock.now() <= deadline_s
            ):
                if not environment.step():
                    continue

                evaluator = environment.get_evaluator_frame()
                observation = oracle.observe(evaluator)
                self.assertEqual(
                    observation.camera_rgb.shape,
                    (
                        config.camera.resolution_wh_px[1],
                        config.camera.resolution_wh_px[0],
                        3,
                    ),
                )
                self.assertEqual(
                    observation.oracle_target_visible,
                    bool(evaluator.target_projection.visible[0]),
                )
                self.assertLessEqual(
                    abs(clock.now() - observation.timestamp),
                    1.0 / config.camera.frequency_hz
                    + config.simulation.physics_dt_s
                    + 1e-9,
                )
                saw_rgb = True
                saw_visible_projection |= bool(observation.oracle_target_visible)

                manager.tick(observation)
                active_name = _name(manager.active_name)
                if not seen_active or seen_active[-1] != active_name:
                    seen_active.append(active_name)

                # Exercise the recovery branch without faking the Oracle bit:
                # move the real STATIC MovingTarget behind the Camera, wait for
                # TRACK's FOV-based lost deadline, then restore the real prim at
                # the last-seen point after REACQUIRE begins its local scan.
                if active_name == "TRACK" and not fault_injected:
                    feedback = manager.get_feedback().data
                    if feedback.get("target_visible") is True:
                        environment.set_target_pose((-6.0, 0.0, 0.5))
                        fault_injected = True
                elif active_name == "REACQUIRE" and fault_injected and not target_restored:
                    feedback = manager.get_feedback().data
                    if feedback.get("phase") == "SCANNING":
                        environment.set_target_pose((6.0, 0.0, 0.5))
                        target_restored = True

            self.assertLessEqual(clock.now(), deadline_s, "pipeline exceeded simulation deadline")
            self.assertIs(manager.task_status, TaskStatus.SUCCEEDED)
            self.assertTrue(saw_rgb)
            self.assertTrue(saw_visible_projection)
            self.assertTrue(fault_injected)
            self.assertTrue(target_restored)

            transitions = [
                _transition_mapping(entry) for entry in manager.transition_log
            ]
            new_skills = [_name(entry.get("new_skill")) for entry in transitions]
            self.assertTrue(
                _is_subsequence(
                    [
                        "TAKEOFF",
                        "GOTO",
                        "SEARCH",
                        "TRACK",
                        "REACQUIRE",
                        "TRACK",
                        "LAND",
                    ],
                    new_skills,
                ),
                new_skills,
            )
            result_codes = [_name(entry.get("result_code")) for entry in transitions]
            self.assertIn("TARGET_LOST", result_codes)
            self.assertIn("TARGET_FOUND", result_codes)
            self.assertIn("TRACK_COMPLETE", result_codes)
            self.assertIn("LAND_COMPLETE", result_codes)

            final_uav = environment.uav_controller.get_pose()
            self.assertLessEqual(abs(final_uav.z), 0.1 + 1e-6)
            final_target = environment.get_evaluator_frame().target_state
            bounds = config.target.motion.region
            self.assertTrue(
                all(
                    low - 1e-9 <= value <= high + 1e-9
                    for value, low, high in zip(
                        (final_target.x, final_target.y, final_target.z),
                        bounds.min_xyz_m,
                        bounds.max_xyz_m,
                    )
                )
            )
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()
