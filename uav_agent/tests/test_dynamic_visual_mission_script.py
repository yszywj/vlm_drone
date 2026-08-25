from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from env.uav_controller import UAVState
from skills.types import Observation
from scripts.run_dynamic_visual_mission import (
    LaunchConfigurationError,
    TestInjectionSpec,
    _VisualRuntime,
    _best_effort_production_failure_land,
    _build_initial_spatial_resolver,
    _create_logging_runtime,
    _preflight_dynamic_target_geometry,
    _route_debug_records,
    _run_until_terminal,
    build_argument_parser,
    make_test_injection_event,
    parse_args,
    resolve_experiment_launch_args,
    startup_fields,
    build_test_injection_specs,
    validate_launch_args,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_dynamic_visual_mission.py"
)


def _args(*extra: str) -> argparse.Namespace:
    return parse_args(
        [
            "--instruction",
            "search for a red cube",
            "--perception-runtime-profile",
            "oracle_evaluation",
            "--acknowledge-privileged-oracle",
            *extra,
        ]
    )


class DynamicVisualMissionScriptTest(unittest.TestCase):
    def test_temporal_preflight_dispatch_precedes_first_isaac_import(self) -> None:
        from unittest.mock import patch

        config = object()
        receipt = {"dry_run": "passed"}
        with patch(
            "perception.factory.preflight_temporal_ray_depth",
            return_value=receipt,
        ) as preflight:
            self.assertIsNone(
                _preflight_dynamic_target_geometry(
                    config,
                    backend_name="oracle_evaluation",
                )
            )
            preflight.assert_not_called()
            self.assertIs(
                _preflight_dynamic_target_geometry(
                    config,
                    backend_name="ultralytics_service",
                ),
                receipt,
            )
            preflight.assert_called_once_with(config)

        source = SCRIPT.read_text(encoding="utf-8")
        self.assertLess(
            source.index("temporal_model_metadata = _preflight_dynamic_target_geometry("),
            source.index("from isaacsim import SimulationApp"),
        )

    def test_unexpected_production_failure_ticks_oracle_free_cancel_and_land(self) -> None:
        running = SimpleNamespace(
            status=SimpleNamespace(value="RUNNING"),
            active_skill="TRACK",
        )
        landing = SimpleNamespace(
            status=SimpleNamespace(value="RUNNING"),
            active_skill="LAND",
        )
        terminal = SimpleNamespace(
            status=SimpleNamespace(value="CANCELED"),
            active_skill=None,
        )

        class Agent:
            def __init__(self) -> None:
                self.current = running
                self.cancel_count = 0
                self.tick_count = 0

            def snapshot(self):
                return self.current

            def cancel(self):
                self.cancel_count += 1
                self.current = landing
                return landing

            def tick(self, observation):
                self.tick_count += 1
                self.asserted_observation = observation
                self.current = terminal
                return terminal

        class Environment:
            def __init__(self) -> None:
                self.uav_controller = SimpleNamespace(stop=lambda: None)
                self.include_oracle_values = []

            def step(self):
                return True

            def get_skill_observation(self, *, include_oracle):
                self.include_oracle_values.append(include_oracle)
                return "base_observation"

        agent = Agent()
        environment = Environment()
        perception = SimpleNamespace(
            attach_target_estimate=lambda base, estimate: (base, estimate)
        )
        result = _best_effort_production_failure_land(
            simulation_app=SimpleNamespace(is_running=lambda: True),
            environment=environment,
            perception=perception,
            agent=agent,
            clock=SimpleNamespace(now=lambda: 1.0),
            shutdown_guard_s=5.0,
        )

        self.assertIs(result.snapshot, terminal)
        self.assertIsNone(result.guard_error)
        self.assertEqual(agent.cancel_count, 1)
        self.assertEqual(agent.tick_count, 1)
        self.assertEqual(environment.include_oracle_values, [False])
        self.assertEqual(agent.asserted_observation, ("base_observation", None))

    def test_target_query_failure_requests_cancel_and_land_once(self) -> None:
        from perception.target_perception_coordinator import TargetQueryUnsupported

        running = SimpleNamespace(
            status=SimpleNamespace(value="RUNNING"),
            active_skill="SEARCH",
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
        )
        landing = SimpleNamespace(
            status=SimpleNamespace(value="RUNNING"),
            active_skill="LAND",
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
        )
        terminal = SimpleNamespace(
            status=SimpleNamespace(value="CANCELED"),
            active_skill=None,
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
        )

        class Agent:
            def __init__(self) -> None:
                self.current = running
                self.cancel_count = 0

            def snapshot(self):
                return self.current

            def cancel(self):
                self.cancel_count += 1
                self.current = landing
                return landing

            def tick(self, observation):
                del observation
                self.current = terminal
                return terminal

        class Coordinator:
            def __init__(self) -> None:
                self.submit_count = 0
                self.submission = None

            def submit_frame(self, **kwargs):
                self.submission = kwargs
                self.submit_count += 1
                raise TargetQueryUnsupported("unsupported category 'red cube'")

            def poll(self, **kwargs):  # pragma: no cover - fail-fast assertion
                raise AssertionError(kwargs)

        observation = Observation(
            uav_id="uav_1",
            timestamp=1.0,
            uav_pose=UAVState(0.0, 0.0, 1.0, 0.25),
            uav_velocity=np.asarray((1.0, 2.0, 3.0)),
            camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        )
        agent = Agent()
        coordinator = Coordinator()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = _run_until_terminal(
                simulation_app=SimpleNamespace(is_running=lambda: True),
                debug_visualizer=None,
                environment=SimpleNamespace(
                    step=lambda: True,
                    get_skill_observation=lambda **kwargs: observation,
                    get_camera_sample=lambda: object(),
                ),
                perception=SimpleNamespace(
                    attach_target_estimate=lambda base, estimate: base,
                ),
                agent=agent,
                manager=SimpleNamespace(
                    task_plan=None,
                    active_invocation=None,
                    active_planned_step_id="search",
                    transition_log=(),
                ),
                clock=SimpleNamespace(now=lambda: 1.0),
                task_start_s=0.0,
                max_sim_time_s=10.0,
                shutdown_guard_s=5.0,
                debug_ground_truth=False,
                injections=(),
                visual_runtime=None,
                target_perception_coordinator=coordinator,
                target_manager=SimpleNamespace(target_spec=object()),
                production_target_perception=True,
            )

        self.assertIs(result.snapshot, terminal)
        self.assertEqual(agent.cancel_count, 1)
        self.assertEqual(coordinator.submit_count, 1)
        self.assertEqual(
            coordinator.submission["uav_linear_velocity_world_mps"],
            (1.0, 2.0, 3.0),
        )
        self.assertEqual(
            coordinator.submission["uav_angular_velocity_body_radps"],
            (0.0, 0.0, 0.0),
        )
        self.assertIn("action=CANCEL_AND_LAND", stderr.getvalue())

    def test_route_debug_records_include_rejected_proposals_before_registry(self) -> None:
        rejected = SimpleNamespace(outcome="REVISE")
        accepted = SimpleNamespace(state="ACCEPTED")
        manager = SimpleNamespace(
            route_registry=SimpleNamespace(records=(accepted,))
        )
        visual_runtime = SimpleNamespace(
            obstacle_revision_coordinator=SimpleNamespace(records=(rejected,))
        )

        self.assertEqual(
            _route_debug_records(manager, visual_runtime),
            (rejected, accepted),
        )

    def test_main_loop_samples_route_collision_monitor_before_agent_tick(self) -> None:
        running = SimpleNamespace(
            status=SimpleNamespace(value="RUNNING"),
            active_skill="FOLLOW_ROUTE",
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=2,
        )
        terminal = SimpleNamespace(
            status=SimpleNamespace(value="CANCELED"),
            active_skill="LAND",
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=2,
        )

        class Agent:
            def __init__(self) -> None:
                self.current = running
                self.tick_count = 0

            def snapshot(self) -> object:
                return self.current

            def tick(self, observation: object) -> object:
                del observation
                self.tick_count += 1
                self.current = terminal
                return terminal

        class Manager:
            active_invocation = None
            active_planned_step_id = "follow_route"
            task_plan = None
            transition_log = ()

            def __init__(self) -> None:
                self.cancel_count = 0

            def cancel_task(self) -> None:
                self.cancel_count += 1

        manager = Manager()

        class Monitor:
            def __init__(self) -> None:
                self.samples = []

            def observe(self, position: object, *, timestamp_s: float) -> object:
                self.samples.append((position, timestamp_s))
                manager.cancel_task()
                return SimpleNamespace(route_id="route_1")

        observation = SimpleNamespace(
            timestamp=1.0,
            uav_pose=SimpleNamespace(x=4.0, y=5.0, z=6.0),
            uav_velocity=(0.0, 0.0, 0.0),
        )
        monitor = Monitor()
        agent = Agent()
        result = _run_until_terminal(
            simulation_app=SimpleNamespace(is_running=lambda: True),
            debug_visualizer=None,
            environment=SimpleNamespace(
                step=lambda: True,
                get_evaluator_frame=lambda: object(),
            ),
            perception=SimpleNamespace(observe=lambda frame: observation),
            agent=agent,
            manager=manager,
            clock=SimpleNamespace(now=lambda: 1.0),
            task_start_s=0.0,
            max_sim_time_s=10.0,
            shutdown_guard_s=10.0,
            debug_ground_truth=False,
            injections=(),
            visual_runtime=None,
            route_collision_monitor=monitor,
        )

        self.assertEqual(monitor.samples, [((4.0, 5.0, 6.0), 1.0)])
        self.assertEqual(manager.cancel_count, 1)
        self.assertEqual(agent.tick_count, 1)
        self.assertIs(result.snapshot, terminal)

    def test_initial_spatial_resolver_uses_trusted_controller_yaw(self) -> None:
        from math import pi

        from planner.schemas import LandingZoneSpec, PlannerWorldContext
        from planner.spatial import CoordinateFrame

        context = PlannerWorldContext(
            scene_min_xyz_m=(-50.0, -50.0, 0.0),
            scene_max_xyz_m=(50.0, 50.0, 30.0),
            initial_uav_xyz_m=(1.0, 2.0, 0.0),
            search_regions={},
            landing_zones={
                "home": LandingZoneSpec("home", (1.0, 2.0), 0.0)
            },
            default_takeoff_altitude_m=10.0,
            default_track_duration_s=10.0,
            search_timeout_s=60.0,
        )
        resolver = _build_initial_spatial_resolver(
            context,
            SimpleNamespace(x=1.0, y=2.0, z=0.0, yaw=pi / 2.0),
        )

        self.assertAlmostEqual(
            resolver.resolve_point(CoordinateFrame.UAV_START_FLU, (1, 0, 0))[0],
            1.0,
        )
        self.assertAlmostEqual(
            resolver.resolve_point(CoordinateFrame.UAV_START_FLU, (1, 0, 0))[1],
            3.0,
        )

    def test_help_is_pure_python_and_lists_experimental_switches(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for flag in (
            "--uav-id",
            "--enable-qwen-vision",
            "--enable-qwen-next-best-view",
            "--vision-review-mode",
            "--acknowledge-vision-gate",
            "--perception-runtime-profile",
            "--acknowledge-privileged-oracle",
            "--inject-path-blocked-at-s",
            "--inject-progress-stall-at-s",
            "--inject-identity-conflict-at-s",
            "--runtime-program",
            "--experiment-mode",
            "--unseen-spatial-instruction",
        ):
            self.assertIn(flag, completed.stdout)

    def test_qwen_next_best_view_is_explicit_v3_only_capability(self) -> None:
        disabled = _args()
        self.assertFalse(disabled.enable_qwen_next_best_view)

        enabled = _args(
            "--planner",
            "dynamic_llm",
            "--planning-contract",
            "v3",
            "--enable-qwen-next-best-view",
        )
        mode, injections = validate_launch_args(enabled)
        self.assertEqual(mode, "shadow")
        self.assertEqual(injections, ())
        self.assertEqual(enabled.experiment_mode, "unspecified")

        with self.assertRaisesRegex(LaunchConfigurationError, "requires"):
            validate_launch_args(
                _args("--enable-qwen-next-best-view")
            )
        with self.assertRaisesRegex(LaunchConfigurationError, "non-Qwen"):
            validate_launch_args(
                _args(
                    "--experiment-mode",
                    "scripted_baseline",
                    "--enable-qwen-next-best-view",
                )
            )

    def test_five_experiment_modes_map_exactly_and_conflicts_fail_closed(self) -> None:
        expected = {
            "scripted_baseline": ("dynamic_scripted", "v2", "strict", "none"),
            "classical_baseline": (
                "dynamic_scripted",
                "v2",
                "strict",
                "classical",
            ),
            "qwen_open_sim": ("dynamic_llm", "v3", "open_sim", "qwen"),
            "qwen_critic_sim": (
                "dynamic_llm",
                "v3",
                "critic_sim",
                "qwen",
            ),
            "qwen_strict": ("dynamic_llm", "v3", "strict", "qwen"),
        }
        for mode, values in expected.items():
            with self.subTest(mode=mode):
                args = _args("--experiment-mode", mode)
                self.assertEqual(
                    (
                        args.planner,
                        args.planning_contract,
                        args.route_validation_mode,
                        args.route_planner_backend,
                    ),
                    values,
                )
        with self.assertRaisesRegex(LaunchConfigurationError, "conflicts"):
            _args(
                "--experiment-mode",
                "qwen_open_sim",
                "--route-validation-mode",
                "strict",
            )

    def test_legacy_defaults_are_not_mislabeled_as_v3_benchmark(self) -> None:
        args = _args()
        self.assertEqual(args.experiment_mode, "unspecified")
        self.assertEqual(args.inferred_experiment_mode, "qwen_strict")
        self.assertEqual(args.planning_contract, "v2")

        direct = build_argument_parser().parse_args(
            [
                "--instruction",
                "search",
                "--experiment-mode",
                "qwen_open_sim",
            ]
        )
        resolve_experiment_launch_args(direct)
        self.assertEqual(direct.route_validation_mode, "open_sim")
        self.assertEqual(direct.planning_contract, "v3")

        legacy_review = _args(
            "--planner",
            "dynamic_scripted",
            "--enable-qwen-vision",
        )
        self.assertEqual(legacy_review.experiment_mode, "unspecified")
        validate_launch_args(legacy_review)

    def test_classical_mode_is_isolated_from_qwen_visual_review(self) -> None:
        mode, injections = validate_launch_args(
            _args("--experiment-mode", "classical_baseline")
        )
        self.assertEqual(mode, "shadow")
        self.assertEqual(injections, ())
        with self.assertRaisesRegex(LaunchConfigurationError, "forbids"):
            validate_launch_args(
                _args(
                    "--experiment-mode",
                    "classical_baseline",
                    "--enable-qwen-vision",
                )
            )

    def test_logging_runtime_does_not_start_or_require_qwen(self) -> None:
        args = _args(
            "--experiment-mode",
            "scripted_baseline",
            "--unseen-spatial-instruction",
        )
        with tempfile.TemporaryDirectory() as temporary:
            args.output_dir = temporary
            runtime = _create_logging_runtime(args=args)
            self.assertIsNone(runtime.worker)
            self.assertIsNone(runtime.coordinator)
            runtime.begin_logging(
                "mission_logging_only",
                manifest_context={
                    "mission_id": "mission_logging_only",
                    "experiment_mode": args.experiment_mode,
                    "route_planner_backend": args.route_planner_backend,
                    "unseen_spatial_instruction": True,
                },
            )
            runtime.set_terminal_manifest(
                agent_status="SUCCEEDED",
                task_status="SUCCEEDED",
                plan_version=1,
                guard_error=None,
            )
            runtime.close(timeout_s=0.0)
            manifest = json.loads(
                (
                    Path(temporary)
                    / "mission_logging_only"
                    / "run_manifest.json"
                ).read_text(encoding="utf-8")
            )
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["experiment_mode"], "scripted_baseline")
        self.assertEqual(manifest["route_planner_backend"], "none")
        self.assertTrue(manifest["mission_success"])
        self.assertTrue(manifest["unseen_spatial_instruction"])
        self.assertIsNone(manifest["shadow_strict_route_valid"])

    def test_logging_runtime_records_non_visual_qwen_model_provenance(self) -> None:
        qwen_args = _args(
            "--experiment-mode",
            "qwen_strict",
            "--model",
            "Qwen3-VL-4B-Instruct",
        )
        scripted_args = _args("--experiment-mode", "scripted_baseline")
        with tempfile.TemporaryDirectory() as temporary:
            qwen_args.output_dir = temporary
            scripted_args.output_dir = temporary
            self.assertEqual(
                _create_logging_runtime(args=qwen_args).model_name,
                "Qwen3-VL-4B-Instruct",
            )
            self.assertEqual(
                _create_logging_runtime(args=scripted_args).model_name,
                "none",
            )

    def test_graph_runtime_is_available_but_unsafe_revision_paths_fail_closed(self) -> None:
        mode, injections = validate_launch_args(
            _args("--runtime-program", "graph")
        )
        self.assertEqual(mode, "shadow")
        self.assertEqual(injections, ())
        with self.assertRaisesRegex(LaunchConfigurationError, "ProgramPatch"):
            validate_launch_args(
                _args(
                    "--runtime-program",
                    "graph",
                    "--enable-qwen-vision",
                    "--vision-review-mode",
                    "gate",
                    "--acknowledge-vision-gate",
                )
            )
        obstacle = _args(
            "--runtime-program",
            "graph",
            "--obstacle-perception",
            "ideal_camera",
        )
        obstacle.effective_obstacle_perception_mode = "ideal_camera"
        with self.assertRaisesRegex(LaunchConfigurationError, "obstacle TaskPlan"):
            validate_launch_args(obstacle)

    def test_import_does_not_import_isaac_or_start_qwen(self) -> None:
        project_root = str(Path(__file__).resolve().parents[1])
        code = (
            "import sys; "
            f"sys.path.insert(0, {project_root!r}); "
            "import scripts.run_dynamic_visual_mission; "
            "assert 'isaacsim' not in sys.modules; "
            "assert not any(name == 'omni' or name.startswith('omni.') "
            "for name in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_gate_requires_qwen_vision_and_separate_acknowledgement(self) -> None:
        with self.assertRaisesRegex(LaunchConfigurationError, "enable-qwen"):
            validate_launch_args(
                _args("--vision-review-mode", "gate"),
            )
        with self.assertRaisesRegex(LaunchConfigurationError, "acknowledge-vision"):
            validate_launch_args(
                _args("--enable-qwen-vision", "--vision-review-mode", "gate"),
            )
        mode, injections = validate_launch_args(
            _args(
                "--enable-qwen-vision",
                "--vision-review-mode",
                "gate",
                "--acknowledge-vision-gate",
            )
        )
        self.assertEqual(mode, "gate")
        self.assertEqual(injections, ())

    def test_shadow_never_requires_gate_acknowledgement(self) -> None:
        mode, _ = validate_launch_args(
            _args("--enable-qwen-vision", "--vision-review-mode", "shadow")
        )
        self.assertEqual(mode, "shadow")
        with self.assertRaisesRegex(LaunchConfigurationError, "only valid"):
            validate_launch_args(
                _args(
                    "--enable-qwen-vision",
                    "--vision-review-mode",
                    "shadow",
                    "--acknowledge-vision-gate",
                )
            )

    def test_oracle_requires_two_part_profile_and_acknowledgement(self) -> None:
        without_ack = parse_args(
            [
                "--instruction",
                "search",
                "--perception-runtime-profile",
                "oracle_evaluation",
            ]
        )
        with self.assertRaisesRegex(LaunchConfigurationError, "acknowledge"):
            validate_launch_args(without_ack)

        production_ack = parse_args(
            ["--instruction", "search", "--acknowledge-privileged-oracle"]
        )
        with self.assertRaisesRegex(LaunchConfigurationError, "invalid"):
            validate_launch_args(production_ack)

        production = parse_args(["--instruction", "search"])
        with self.assertRaisesRegex(LaunchConfigurationError, "detector/tracker"):
            validate_launch_args(production)

    def test_three_injection_flags_are_bounded_and_source_is_explicit(self) -> None:
        args = _args(
            "--enable-qwen-vision",
            "--inject-path-blocked-at-s",
            "1.0",
            "--inject-progress-stall-at-s",
            "2.0",
            "--inject-identity-conflict-at-s",
            "3.0",
        )
        _, specs = validate_launch_args(args)
        self.assertEqual(
            tuple(spec.event_type for spec in specs),
            (
                "PATH_BLOCKED",
                "SKILL_PROGRESS_STALLED",
                "TARGET_IDENTITY_UNCERTAIN",
            ),
        )
        event = make_test_injection_event(
            specs[0],
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
            timestamp_s=4.0,
        )
        serialized = event.to_dict()
        self.assertEqual(serialized["payload"]["source"], "test_injection")
        self.assertEqual(serialized["uav_id"], "uav_1")
        self.assertNotIn("oracle", str(serialized).lower())

    def test_sparse_event_log_preserves_test_injection_source(self) -> None:
        from runtime.events import MissionEventBus

        class Worker:
            def close(self, *, timeout_s: float) -> None:
                del timeout_s

        class Coordinator:
            records: tuple[object, ...] = ()

        event = make_test_injection_event(
            TestInjectionSpec("PATH_BLOCKED", 1.0, "--inject-path-blocked-at-s"),
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
            timestamp_s=1.0,
        )
        bus = MissionEventBus()
        bus.publish(event)
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _VisualRuntime(
                coordinator=Coordinator(),
                worker=Worker(),
                event_bus=bus,
                output_parent=Path(temporary),
                model_name="fake",
            )
            runtime.begin_logging("mission_1")
            with redirect_stdout(io.StringIO()):
                runtime.emit_new_events(skill="GOTO", step_id="goto_1")
            runtime.close(timeout_s=0.0)
            with (
                Path(temporary)
                / "mission_1"
                / "mission_events.csv"
            ).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reason"], "source=test_injection")

    def test_next_best_view_proposal_is_persisted_without_image_data(self) -> None:
        from planner.qwen_next_best_view import NextBestViewProposalRecord
        from runtime.events import MissionEventBus

        record = NextBestViewProposalRecord(
            request_id="request_nbv_1",
            review_id="review_nbv_1",
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
            frame_id="frame_nbv_1",
            observation_timestamp_s=2.0,
            proposal_index=0,
            proposal={
                "decision": "NEXT_VIEW",
                "coordinate_frame": "WORLD_ENU",
                "viewpoint_xyz_m": [2.0, 0.0, 5.0],
            },
            decision="NEXT_VIEW",
            viewpoint_xyz_m=(2.0, 0.0, 5.0),
            error_code=None,
            token_usage={"total_tokens": 10},
        )
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _VisualRuntime(
                coordinator=None,
                worker=None,
                event_bus=MissionEventBus(),
                output_parent=Path(temporary),
                model_name="fake-qwen",
                next_best_view_provider=SimpleNamespace(records=(record,)),
            )
            runtime.begin_logging("mission_1")
            with redirect_stdout(io.StringIO()):
                runtime.emit_new_next_best_view_proposals()
                runtime.emit_new_next_best_view_proposals()
            runtime.close(timeout_s=0.0)
            rendered = (
                Path(temporary) / "mission_1" / "model_proposals.jsonl"
            ).read_text(encoding="utf-8")
            manifest = json.loads(
                (
                    Path(temporary) / "mission_1" / "run_manifest.json"
                ).read_text(encoding="utf-8")
            )
        self.assertEqual(len(rendered.splitlines()), 1)
        self.assertNotIn("data:image/", rendered)
        self.assertNotIn("base64,", rendered)
        self.assertEqual(manifest["next_best_view_model_calls"], 1)

    def test_injection_without_visual_runtime_is_rejected(self) -> None:
        with self.assertRaisesRegex(LaunchConfigurationError, "enable-qwen"):
            validate_launch_args(
                _args("--inject-path-blocked-at-s", "0")
            )

    def test_test_injection_spec_rejects_negative_and_nonfinite_times(self) -> None:
        for value in (-1.0, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    TestInjectionSpec("PATH_BLOCKED", value, "--flag")

    def test_startup_fields_include_route_and_never_include_credentials(self) -> None:
        values = dict(
            startup_fields(
                uav_id="uav_7",
                mission_id="mission_7",
                planner="dynamic_llm",
                review_enabled=True,
                review_mode="shadow",
                perception_profile="oracle_evaluation",
                oracle_acknowledged=True,
            )
        )
        self.assertEqual(values["uav_id"], "uav_7")
        self.assertEqual(values["mission_id"], "mission_7")
        self.assertEqual(values["qwen_visual_review"], "enabled:shadow")
        self.assertTrue(values["oracle_acknowledged"])
        self.assertNotIn("api", " ".join(values).lower())

    def test_parser_rejects_invalid_uav_id_and_negative_trigger(self) -> None:
        parser = build_argument_parser()
        for argv in (
            ["--instruction", "x", "--uav-id", "../uav"],
            ["--instruction", "x", "--inject-path-blocked-at-s", "-1"],
        ):
            with (
                self.subTest(argv=argv),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    parser.parse_args(argv)

    def test_specs_helper_has_no_implicit_events(self) -> None:
        self.assertEqual(build_test_injection_specs(_args()), ())

    def test_run_manifest_records_revision_hover_and_terminal_route(self) -> None:
        from skills.types import SkillName, SkillStatus

        class Worker:
            def close(self, *, timeout_s: float) -> None:
                del timeout_s

        class Coordinator:
            records: tuple[object, ...] = ()

        revision = SimpleNamespace(
            request_id="request_1",
            outcome="ACCEPTED",
            timestamp_s=3.0,
            mission_id="mission_1",
            uav_id="uav_1",
            old_plan_version=1,
            new_plan_version=2,
            old_step_id="goto_1",
            new_step_id="goto_2",
        )
        revision_coordinator = SimpleNamespace(records=(revision,))
        transitions = (
            SimpleNamespace(
                timestamp=1.0,
                mission_id="mission_1",
                uav_id="uav_1",
                plan_version=1,
                old_step_id="goto_1",
                new_step_id="goto_1",
                old_skill=SkillName.GOTO,
                new_skill=SkillName.HOVER,
                old_status=SkillStatus.CANCELED,
                result_code=None,
                reason="supervisory_hover_started",
            ),
            SimpleNamespace(
                timestamp=2.5,
                mission_id="mission_1",
                uav_id="uav_1",
                plan_version=2,
                old_step_id="goto_1",
                new_step_id="goto_2",
                old_skill=SkillName.HOVER,
                new_skill=SkillName.GOTO,
                old_status=SkillStatus.SUCCEEDED,
                result_code=None,
                reason="interrupted_step_and_suffix_replaced",
            ),
        )
        manager = SimpleNamespace(transition_log=transitions)

        with tempfile.TemporaryDirectory() as temporary:
            runtime = _VisualRuntime(
                coordinator=Coordinator(),
                worker=Worker(),
                event_bus=SimpleNamespace(recent=lambda: ()),
                output_parent=Path(temporary),
                model_name="fake-qwen",
                revision_coordinator=revision_coordinator,
            )
            runtime.begin_logging(
                "mission_1",
                manifest_context={
                    "mission_id": "mission_1",
                    "uav_id": "uav_1",
                    "model": "fake-qwen",
                    "git_commit": "0" * 40,
                    "configuration": {"frame_store": {"max_frames": 24}},
                },
            )
            with redirect_stdout(io.StringIO()):
                runtime.emit_new_transitions(manager)
                runtime.emit_new_revisions(skill="GOTO", step_id="goto_2")
                runtime.record_route_collision()
            runtime.set_terminal_manifest(
                agent_status="SUCCEEDED",
                task_status="SUCCEEDED",
                plan_version=2,
                guard_error=None,
            )
            runtime.close(timeout_s=0.0)
            manifest = json.loads(
                (
                    Path(temporary)
                    / "mission_1"
                    / "run_manifest.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["model"], "fake-qwen")
        self.assertEqual(manifest["plan_revisions"], 1)
        self.assertEqual(manifest["collision_count"], 1)
        self.assertEqual(manifest["supervisory_hover"]["count"], 1)
        self.assertEqual(manifest["supervisory_hover"]["total_time_s"], 1.5)
        self.assertEqual(manifest["agent_status"], "SUCCEEDED")
        self.assertEqual(manifest["final_plan_version"], 2)


if __name__ == "__main__":
    unittest.main()
