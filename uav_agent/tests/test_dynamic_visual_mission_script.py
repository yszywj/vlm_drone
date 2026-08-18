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

from scripts.run_dynamic_visual_mission import (
    LaunchConfigurationError,
    TestInjectionSpec,
    _VisualRuntime,
    build_argument_parser,
    make_test_injection_event,
    parse_args,
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
            "--vision-review-mode",
            "--acknowledge-vision-gate",
            "--perception-runtime-profile",
            "--acknowledge-privileged-oracle",
            "--inject-path-blocked-at-s",
            "--inject-progress-stall-at-s",
            "--inject-identity-conflict-at-s",
        ):
            self.assertIn(flag, completed.stdout)

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
        self.assertEqual(manifest["supervisory_hover"]["count"], 1)
        self.assertEqual(manifest["supervisory_hover"]["total_time_s"], 1.5)
        self.assertEqual(manifest["agent_status"], "SUCCEEDED")
        self.assertEqual(manifest["final_plan_version"], 2)


if __name__ == "__main__":
    unittest.main()
