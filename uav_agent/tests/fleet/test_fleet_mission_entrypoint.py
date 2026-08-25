from __future__ import annotations

import builtins
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from fleet.preplanned_planner import RoutedPreplannedFleetPlanner
from fleet.logging import FleetMissionLogger
from fleet.model_request_broker import GlobalModelRequestBroker
from fleet.runtime import FleetRuntimeSnapshot, FleetStatus
from fleet.types import FleetStartPolicy
from scripts import run_fleet_mission


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def prepared_oracle():
    args = run_fleet_mission.parse_args(
        [
            "--config",
            str(ROOT / "configs/multi_uav_demo.yaml"),
            "--perception-runtime-profile",
            "oracle_evaluation",
            "--acknowledge-privileged-oracle",
        ]
    )
    return run_fleet_mission.prepare_fleet_mission(args)


def test_parser_exposes_complete_fleet_runtime_surface() -> None:
    args = run_fleet_mission.parse_args(
        [
            "--config",
            str(ROOT / "configs/multi_uav_demo.yaml"),
            "--instruction",
            run_fleet_mission.DEFAULT_INSTRUCTION,
            "--fleet-planner",
            "llm",
            "--local-planner",
            "dynamic_llm",
            "--planning-contract",
            "v3",
            "--runtime-program",
            "graph",
            "--adapter-config",
            str(ROOT / "configs/adapters.json"),
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--model",
            "Qwen3-VL-4B-Instruct",
            "--api-key",
            "EMPTY",
            "--enable-qwen-vision",
            "--vision-review-mode",
            "gate",
            "--perception-runtime-profile",
            "oracle_evaluation",
            "--acknowledge-privileged-oracle",
            "--debug-visualization",
            "--no-headless",
            "--max-sim-time",
            "300",
            "--output-root",
            str(ROOT / "logs/fleet_missions"),
        ]
    )
    assert args.fleet_planner == "llm"
    assert args.local_planner == "dynamic_llm"
    assert args.planning_contract == "v3"
    assert args.runtime_program == "graph"
    assert args.enable_qwen_vision is True
    assert args.vision_review_mode == "gate"
    assert args.headless is False
    assert args.max_sim_time == 300.0


def test_fleet_simulation_app_uses_graceful_shutdown() -> None:
    assert run_fleet_mission._simulation_app_launch_config(headless=True) == {
        "headless": True,
        "fast_shutdown": False,
        "multi_gpu": False,
    }
    with pytest.raises(TypeError, match="headless"):
        run_fleet_mission._simulation_app_launch_config(  # type: ignore[arg-type]
            headless=1
        )


def test_scripted_run_retains_preflight_verified_yolo_identity(
    prepared_oracle,
) -> None:
    prepared = replace(prepared_oracle, preparation_context=None)
    metadata = {
        "uav_a": {
            "url": "http://127.0.0.1:8011",
            "model_family": "yolo",
            "model_names": {"0": "cube"},
            "model_sha256": "ab" * 32,
            "ready": True,
        }
    }
    run_fleet_mission._retain_yolo_service_metadata(prepared, metadata)
    metadata["uav_a"]["ready"] = False

    assert prepared.preparation_context is not None
    assert prepared.preparation_context["yolo_service_metadata"] == {
        "uav_a": {
            "url": "http://127.0.0.1:8011",
            "model_family": "yolo",
                "model_names": {"0": "cube"},
            "model_sha256": "ab" * 32,
            "ready": True,
        }
    }


def test_pure_preparation_completes_without_isaac_import(monkeypatch) -> None:
    attempted: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "isaacsim" or name.startswith("isaacsim."):
            attempted.append(name)
            raise AssertionError("prepare_fleet_mission attempted an Isaac import")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    args = run_fleet_mission.parse_args(
        [
            "--config",
            str(ROOT / "configs/multi_uav_demo.yaml"),
            "--perception-runtime-profile",
            "oracle_evaluation",
            "--acknowledge-privileged-oracle",
        ]
    )
    prepared = run_fleet_mission.prepare_fleet_mission(args)
    assert sorted(prepared.compilations) == ["uav_a", "uav_b"]
    assert all(
        item.compiled_mission is not None
        for item in prepared.compilations.values()
    )
    assert attempted == []
    assert not any(name == "isaacsim" for name in sys.modules)


def test_sequential_request_is_rejected_before_first_isaac_import(
    prepared_oracle,
    monkeypatch,
) -> None:
    attempted: list[str] = []
    original_import = builtins.__import__
    sequential_request = replace(
        prepared_oracle.request,
        target_requests=(
            replace(
                prepared_oracle.request.target_requests[0],
                start_policy=FleetStartPolicy.SEQUENTIAL,
            ),
            *prepared_oracle.request.target_requests[1:],
        ),
    )

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "isaacsim" or name.startswith("isaacsim."):
            attempted.append(name)
            raise AssertionError("SEQUENTIAL request crossed the Isaac boundary")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        run_fleet_mission,
        "build_fleet_mission_request",
        lambda *args, **kwargs: sequential_request,
    )
    args = run_fleet_mission.parse_args(
        [
            "--config",
            str(ROOT / "configs/multi_uav_demo.yaml"),
            "--perception-runtime-profile",
            "oracle_evaluation",
            "--acknowledge-privileged-oracle",
        ]
    )

    with pytest.raises(Exception, match="does not support SEQUENTIAL"):
        run_fleet_mission.prepare_fleet_mission(args)

    assert attempted == []


def test_preparation_extracts_oracle_free_assignment_routes(prepared_oracle) -> None:
    assert prepared_oracle.planned_routes["uav_a"] == (
        (-3.0, 0.0, 0.0),
        (-3.0, 0.0, 10.0),
        (20.0, 30.0, 10.0),
        (-3.0, 0.0, 10.0),
        (-3.0, 0.0, 0.0),
    )
    assert prepared_oracle.planned_routes["uav_b"] == (
        (3.0, 0.0, 0.0),
        (3.0, 0.0, 10.0),
        (-25.0, 10.0, 10.0),
        (3.0, 0.0, 10.0),
        (3.0, 0.0, 0.0),
    )
    serialized = repr(dict(prepared_oracle.planned_routes))
    assert "target_states" not in serialized
    assert "oracle" not in serialized.casefold()


def test_oracle_requires_two_part_opt_in() -> None:
    args = run_fleet_mission.parse_args(
        [
            "--config",
            str(ROOT / "configs/multi_uav_demo.yaml"),
            "--perception-runtime-profile",
            "oracle_evaluation",
        ]
    )
    with pytest.raises(
        run_fleet_mission.FleetLaunchConfigurationError,
        match="acknowledge-privileged-oracle",
    ):
        run_fleet_mission.prepare_fleet_mission(args)


def test_qwen_vision_preparation_declares_brokered_runtime_adapter() -> None:
    args = run_fleet_mission.parse_args(
        [
            "--config",
            str(ROOT / "configs/multi_uav_demo.yaml"),
            "--enable-qwen-vision",
            "--perception-runtime-profile",
            "oracle_evaluation",
            "--acknowledge-privileged-oracle",
        ]
    )
    prepared = run_fleet_mission.prepare_fleet_mission(args)

    selection = next(
        item
        for item in prepared.adapter_selections
        if item["call_role"] == "RUNTIME_VISUAL_REVIEW"
    )
    assert selection["requested_adapter"] == "runtime_visual"
    assert selection["adapter_status"] == "placeholder"
    assert selection["effective_model"] == "Qwen3-VL-4B-Instruct"
    assert selection["fallback_used"] is True
    assert selection["used"] is True
    assert selection["brokered"] is True


def test_runtime_visual_workers_share_exactly_one_global_broker() -> None:
    args = run_fleet_mission.parse_args(
        [
            "--config",
            str(ROOT / "configs/multi_uav_demo.yaml"),
            "--enable-qwen-vision",
            "--perception-runtime-profile",
            "oracle_evaluation",
            "--acknowledge-privileged-oracle",
        ]
    )
    prepared = run_fleet_mission.prepare_fleet_mission(args)
    broker = GlobalModelRequestBroker(
        max_inflight_global=prepared.config.model_broker.max_inflight_global,
        max_inflight_per_uav=(
            prepared.config.model_broker.max_inflight_per_uav
        ),
        max_pending_per_uav=prepared.config.model_broker.max_pending_per_uav,
        starvation_timeout_s=prepared.config.model_broker.starvation_timeout_s,
    )

    class _Logger:
        def __init__(self) -> None:
            self.rows: list[dict[str, object]] = []

        def log_model_call(self, value: dict[str, object]) -> None:
            self.rows.append(dict(value))

    logger = _Logger()
    dispatcher, facades = run_fleet_mission._build_brokered_visual_workers(
        prepared,
        broker,
        logger,
    )
    try:
        assert dispatcher.broker is broker
        assert set(dispatcher.workers) == {"uav_a", "uav_b"}
        assert set(facades) == {"uav_a", "uav_b"}
        expected_assignments = {
            assignment.uav_id: assignment.assignment_id
            for assignment in prepared.plan.assignments
        }
        assert facades["uav_a"].assignment_id == expected_assignments["uav_a"]
        assert facades["uav_b"].assignment_id == expected_assignments["uav_b"]
        assert facades["uav_a"]._dispatcher is dispatcher
        assert facades["uav_b"]._dispatcher is dispatcher
        assert logger.rows == []
        assert "isaacsim" not in sys.modules
    finally:
        dispatcher.close(timeout_s=1.0)


def test_production_disabled_perception_fails_before_isaac() -> None:
    args = run_fleet_mission.parse_args(
        ["--config", str(ROOT / "configs/multi_uav_demo.yaml")]
    )
    with pytest.raises(Exception, match="disabled cannot execute target Skills"):
        run_fleet_mission.prepare_fleet_mission(args)


def test_preplanned_fleet_replay_accepts_only_exact_request(prepared_oracle) -> None:
    planner = RoutedPreplannedFleetPlanner(
        prepared_oracle.request,
        prepared_oracle.plan,
        source=prepared_oracle.fleet_planner_source,
    )
    replayed = planner.plan(prepared_oracle.request)
    assert replayed == prepared_oracle.plan
    assert replayed is not prepared_oracle.plan

    changed = replace(
        prepared_oracle.request,
        original_instruction=prepared_oracle.request.original_instruction + "。",
    )
    with pytest.raises(ValueError, match="differs from the preplanned"):
        planner.plan(changed)


def test_prepared_logging_writes_manifest_models_and_local_plans(
    prepared_oracle,
    tmp_path: Path,
) -> None:
    args = run_fleet_mission.parse_args(
        [
            "--config",
            str(ROOT / "configs/multi_uav_demo.yaml"),
            "--api-key",
            "DO_NOT_PERSIST",
            "--perception-runtime-profile",
            "oracle_evaluation",
            "--acknowledge-privileged-oracle",
            "--output-root",
            str(tmp_path),
        ]
    )
    logger = FleetMissionLogger(
        tmp_path,
        prepared_oracle.request.fleet_mission_id,
        uav_ids=tuple(prepared_oracle.compilations),
    )
    run_fleet_mission._write_prepared_logs(logger, prepared_oracle, args)
    run_dir = tmp_path / prepared_oracle.request.fleet_mission_id

    manifest = (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    assert "DO_NOT_PERSIST" not in manifest
    assert (run_dir / "fleet_plan.json").is_file()
    with (run_dir / "assignments.csv").open(encoding="utf-8", newline="") as stream:
        assignment_rows = list(csv.DictReader(stream))
    assert {row["status"] for row in assignment_rows} == {"PENDING"}
    with (run_dir / "model_calls.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["call_role"] for row in rows} >= {
        "FLEET_PLAN",
        "AGENT_SPATIAL_PLAN",
    }
    assert all(row["requested_adapter"] for row in rows)
    assert all(row["effective_model"] for row in rows)
    assert {row["finish_reason"] for row in rows} == {"not_called"}
    assert {row["call_id"] for row in rows} == {
        "not_called_fleet_plan",
        "not_called_agent_spatial_plan",
    }
    for uav_id in prepared_oracle.compilations:
        local_plan = run_dir / "agents" / uav_id / "local_plan_v1.json"
        assert local_plan.is_file()
        assert "spatial_plan_draft_v3" in local_plan.read_text(encoding="utf-8")


def test_planning_error_does_not_cross_isaac_boundary(monkeypatch) -> None:
    crossed = False

    def forbidden_runtime(*args, **kwargs):
        nonlocal crossed
        crossed = True
        raise AssertionError("Isaac runtime must not be called")

    monkeypatch.setattr(
        run_fleet_mission,
        "run_prepared_fleet_mission",
        forbidden_runtime,
    )
    code = run_fleet_mission.main(
        [
            "--config",
            str(ROOT / "configs/multi_uav_demo.yaml"),
            "--instruction",
            "无法映射的任务",
            "--perception-runtime-profile",
            "oracle_evaluation",
            "--acknowledge-privileged-oracle",
        ]
    )
    assert code == 2
    assert crossed is False


def test_isaac_import_failure_still_writes_failed_terminal_artifacts(
    prepared_oracle,
    monkeypatch,
    tmp_path: Path,
) -> None:
    original_import = builtins.__import__

    def reject_isaac(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "isaacsim":
            raise RuntimeError("synthetic Isaac import failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_isaac)
    args = run_fleet_mission.parse_args(
        [
            "--config",
            str(ROOT / "configs/multi_uav_demo.yaml"),
            "--perception-runtime-profile",
            "oracle_evaluation",
            "--acknowledge-privileged-oracle",
            "--output-root",
            str(tmp_path),
        ]
    )

    with pytest.raises(RuntimeError, match="synthetic Isaac import failure"):
        run_fleet_mission.run_prepared_fleet_mission(prepared_oracle, args)

    run_dir = tmp_path / prepared_oracle.request.fleet_mission_id
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED"
    assert summary["exit_code"] == 1
    assert "synthetic Isaac import failure" in summary["last_error"]
    assert "events" not in summary
    assert (run_dir / "fleet_plan.json").is_file()
    with (run_dir / "assignments.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["status"] for row in rows} == {"FAILED"}


def test_primary_failure_is_not_masked_by_cleanup_and_errors_are_redacted(
    prepared_oracle,
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    class PrimaryError(RuntimeError):
        pass

    original_import = builtins.__import__

    def reject_isaac(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "isaacsim":
            raise PrimaryError(
                "primary Authorization: Bearer SUPER_SECRET"
            )
        return original_import(name, globals, locals, fromlist, level)

    def fail_cleanup(*args, **kwargs):
        raise RuntimeError("cleanup password=ALSO_SECRET")

    monkeypatch.setattr(builtins, "__import__", reject_isaac)
    monkeypatch.setattr(
        run_fleet_mission,
        "_close_execution_resources",
        fail_cleanup,
    )
    args = run_fleet_mission.parse_args(
        [
            "--config",
            str(ROOT / "configs/multi_uav_demo.yaml"),
            "--perception-runtime-profile",
            "oracle_evaluation",
            "--acknowledge-privileged-oracle",
            "--output-root",
            str(tmp_path),
        ]
    )

    with pytest.raises(PrimaryError, match="SUPER_SECRET"):
        run_fleet_mission.run_prepared_fleet_mission(prepared_oracle, args)

    run_dir = tmp_path / prepared_oracle.request.fleet_mission_id
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    persisted = json.dumps(summary, ensure_ascii=False)
    assert summary["status"] == "FAILED"
    assert summary["exit_code"] == 1
    assert "SUPER_SECRET" not in persisted
    assert "ALSO_SECRET" not in persisted
    assert "[REDACTED]" in persisted
    assert "ALSO_SECRET" not in capsys.readouterr().err


def test_keyboard_interrupt_remains_130_when_cleanup_fails(
    prepared_oracle,
    monkeypatch,
    tmp_path: Path,
) -> None:
    original_import = builtins.__import__

    def interrupt_isaac(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "isaacsim":
            raise KeyboardInterrupt
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", interrupt_isaac)
    monkeypatch.setattr(
        run_fleet_mission,
        "_close_execution_resources",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("cleanup failed")
        ),
    )
    args = run_fleet_mission.parse_args(
        [
            "--config",
            str(ROOT / "configs/multi_uav_demo.yaml"),
            "--perception-runtime-profile",
            "oracle_evaluation",
            "--acknowledge-privileged-oracle",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert run_fleet_mission.run_prepared_fleet_mission(prepared_oracle, args) == 130
    summary = json.loads(
        (
            tmp_path
            / prepared_oracle.request.fleet_mission_id
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["status"] == "CANCELED"
    assert summary["exit_code"] == 130
    assert summary["interrupted"] is True
    assert "cleanup failed" in summary["last_error"]


def test_terminal_payload_overlays_empty_runtime_on_prepared_assignments(
    prepared_oracle,
) -> None:
    class EmptySummary:
        @staticmethod
        def summary_snapshot() -> dict[str, object]:
            return {}

    class EarlyFailureRuntime:
        targets = EmptySummary()
        model_broker = EmptySummary()

        @staticmethod
        def snapshot() -> FleetRuntimeSnapshot:
            return FleetRuntimeSnapshot(
                status=FleetStatus.IDLE,
                fleet_mission_id=None,
                fleet_plan_version=None,
                agent_plan_versions={},
                agent_statuses={},
                assignments={},
                last_airspace_decision=None,
                events=(),
            )

    rows, summary = run_fleet_mission._terminal_log_payload(
        prepared_oracle,
        EarlyFailureRuntime(),
        exit_code=1,
        terminal_error="RuntimeError: --api-key TOP_SECRET",
        interrupted=False,
    )

    expected_ids = {
        assignment.assignment_id for assignment in prepared_oracle.plan.assignments
    }
    assert {row["assignment_id"] for row in rows} == expected_ids
    assert {row["status"] for row in rows} == {"FAILED"}
    assert set(summary["assignments"]) == expected_ids
    assert set(summary["agent_statuses"].values()) == {"NOT_STARTED"}
    assert summary["fleet_mission_id"] == prepared_oracle.request.fleet_mission_id
    assert summary["fleet_plan_version"] == prepared_oracle.plan.fleet_plan_version
    assert "TOP_SECRET" not in json.dumps(summary, ensure_ascii=False)


def test_final_runtime_log_drain_is_incremental_and_idempotent() -> None:
    class Logger:
        def __init__(self) -> None:
            self.visual: list[tuple[str, object]] = []
            self.transitions: list[tuple[str, object]] = []

        def log_visual_review(self, uav_id: str, record: object) -> None:
            self.visual.append((uav_id, record))

        def log_agent_transition(self, uav_id: str, record: object) -> None:
            self.transitions.append((uav_id, record))

    class Coordinator:
        records = ("old_review", "final_review")

    class Manager:
        transition_log = ("old_transition", "final_land")

    logger = Logger()
    visual_cursors = {"uav_a": 1}
    transition_cursors = {"uav_a": 1}
    arguments = (
        logger,
        {"uav_a": Coordinator()},
        visual_cursors,
        {"uav_a": Manager()},
        transition_cursors,
    )

    run_fleet_mission._drain_runtime_logs(*arguments)
    run_fleet_mission._drain_runtime_logs(*arguments)

    assert logger.visual == [("uav_a", "final_review")]
    assert logger.transitions == [("uav_a", "final_land")]
    assert visual_cursors == {"uav_a": 2}
    assert transition_cursors == {"uav_a": 2}


def test_terminal_summary_write_failure_cannot_return_success(
    prepared_oracle,
) -> None:
    class SummaryComponent:
        @staticmethod
        def summary_snapshot() -> dict[str, object]:
            return {}

    assignment_rows = {
        assignment.assignment_id: {
            "assignment_id": assignment.assignment_id,
            "uav_id": assignment.uav_id,
            "target_alias": assignment.target_alias,
            "priority": assignment.priority,
            "status": "SUCCEEDED",
            "local_plan_version": 1,
        }
        for assignment in prepared_oracle.plan.assignments
    }

    class Runtime:
        targets = SummaryComponent()
        model_broker = SummaryComponent()

        @staticmethod
        def snapshot() -> FleetRuntimeSnapshot:
            return FleetRuntimeSnapshot(
                status=FleetStatus.SUCCEEDED,
                fleet_mission_id=prepared_oracle.request.fleet_mission_id,
                fleet_plan_version=prepared_oracle.plan.fleet_plan_version,
                agent_plan_versions={uav_id: 1 for uav_id in prepared_oracle.compilations},
                agent_statuses={uav_id: "SUCCEEDED" for uav_id in prepared_oracle.compilations},
                assignments=assignment_rows,
                last_airspace_decision=None,
                events=(),
            )

    class FailingLogger:
        assignments_written = False

        def write_assignments(self, rows: object) -> None:
            self.assignments_written = True

        @staticmethod
        def write_summary(summary: object) -> None:
            raise OSError("disk full")

    logger = FailingLogger()
    exit_code = run_fleet_mission._finalize_fleet_execution(
        prepared_oracle,
        logger,
        runtime=Runtime(),
        simulation_app=None,
        environment=None,
        clock=None,
        debug_draw=None,
        workers=(),
        visual_coordinators={},
        visual_log_cursors={},
        managers={},
        transition_log_cursors={},
        exit_code=0,
        terminal_error=None,
        interrupted=False,
        print_terminal_summary=False,
        primary_error_present=False,
    )

    assert logger.assignments_written is True
    assert exit_code == 1


def test_best_effort_shutdown_ticks_fail_safe_land_to_terminal_state() -> None:
    class Clock:
        timestamp_s = 10.0

        def now(self) -> float:
            return self.timestamp_s

    class Runtime:
        status = FleetStatus.RUNNING
        cancel_requested = False
        cancel_calls = 0
        tick_calls = 0

        def cancel(self) -> None:
            self.cancel_calls += 1
            self.cancel_requested = True

        def tick(self) -> None:
            self.tick_calls += 1
            clock.timestamp_s += 0.5
            if self.tick_calls == 3:
                self.status = FleetStatus.CANCELED

    class App:
        @staticmethod
        def is_running() -> bool:
            return True

    clock = Clock()
    runtime = Runtime()

    run_fleet_mission._best_effort_cancel_and_land(
        runtime,
        App(),
        clock,
        guard_s=5.0,
    )

    assert runtime.cancel_calls == 1
    assert runtime.tick_calls == 3
    assert runtime.status is FleetStatus.CANCELED


def test_simulation_app_closes_even_when_runtime_cleanup_fails() -> None:
    calls: list[str] = []

    class Runtime:
        @staticmethod
        def close() -> None:
            calls.append("runtime")
            raise RuntimeError("runtime close failed")

    class Environment:
        @staticmethod
        def close() -> None:
            calls.append("environment")

    class App:
        @staticmethod
        def close() -> None:
            calls.append("app")

    with pytest.raises(RuntimeError, match="runtime close failed"):
        run_fleet_mission._close_execution_resources(
            Runtime(),
            Environment(),
            App(),
        )

    assert calls == ["runtime", "environment", "app"]


def test_gui_debug_visualizer_injects_fleet_status_overlay(
    prepared_oracle,
    monkeypatch,
) -> None:
    import visualization

    events: list[object] = []

    class Overlay:
        def close(self) -> None:
            events.append("overlay.close")

    class Draw:
        def __init__(self, *, status_overlay: object) -> None:
            self.status_overlay = status_overlay
            events.append(("draw", status_overlay))

        def set_plan(self, plan, *, agent_plan_versions) -> None:
            events.append(("plan", plan, dict(agent_plan_versions)))

        def close(self) -> None:
            self.status_overlay.close()

    overlay = Overlay()
    monkeypatch.setattr(visualization, "FleetStatusOverlay", lambda: overlay)
    monkeypatch.setattr(visualization, "FleetDebugDraw", Draw)
    visualizer = run_fleet_mission._create_fleet_debug_visualizer(
        prepared_oracle.plan,
        {"uav_a": 7, "uav_b": 3},
        headless=False,
    )

    assert visualizer.status_overlay is overlay
    assert events[0] == ("draw", overlay)
    assert events[1][0:2] == ("plan", prepared_oracle.plan)
    assert events[1][2] == {"uav_a": 7, "uav_b": 3}
    visualizer.close()
    assert events[-1] == "overlay.close"


def test_headless_debug_visualizer_never_constructs_ui(prepared_oracle) -> None:
    overlay_calls = 0

    def forbidden_overlay():
        nonlocal overlay_calls
        overlay_calls += 1
        raise AssertionError("headless visualization must not construct omni.ui")

    class Draw:
        def __init__(self, *, status_overlay: object) -> None:
            self.status_overlay = status_overlay

        def set_plan(self, plan, *, agent_plan_versions) -> None:
            self.plan = plan
            self.agent_plan_versions = dict(agent_plan_versions)

        def close(self) -> None:
            pass

    visualizer = run_fleet_mission._create_fleet_debug_visualizer(
        prepared_oracle.plan,
        {"uav_a": 7, "uav_b": 3},
        headless=True,
        draw_factory=Draw,
        overlay_factory=forbidden_overlay,
    )

    assert overlay_calls == 0
    assert visualizer.status_overlay is None
    assert visualizer.plan == prepared_oracle.plan


def test_debug_visualizer_closes_overlay_when_draw_construction_fails(
    prepared_oracle,
) -> None:
    class Overlay:
        closed = False

        def close(self) -> None:
            self.closed = True

    overlay = Overlay()

    def fail_draw(*, status_overlay: object):
        assert status_overlay is overlay
        raise RuntimeError("draw failed")

    with pytest.raises(RuntimeError, match="draw failed"):
        run_fleet_mission._create_fleet_debug_visualizer(
            prepared_oracle.plan,
            {"uav_a": 7, "uav_b": 3},
            headless=False,
            draw_factory=fail_draw,
            overlay_factory=lambda: overlay,
        )
    assert overlay.closed


def test_main_preserves_runtime_exit_code_and_maps_runtime_errors(monkeypatch) -> None:
    prepared = object()
    monkeypatch.setattr(
        run_fleet_mission,
        "prepare_fleet_mission",
        lambda args: prepared,
    )
    monkeypatch.setattr(
        run_fleet_mission,
        "run_prepared_fleet_mission",
        lambda value, args: 130,
    )
    assert run_fleet_mission.main([]) == 130

    def fail_runtime(value, args):
        raise RuntimeError("runtime failed")

    monkeypatch.setattr(
        run_fleet_mission,
        "run_prepared_fleet_mission",
        fail_runtime,
    )
    assert run_fleet_mission.main([]) == 1


def test_terminal_summary_is_bounded_and_does_not_duplicate_events() -> None:
    snapshot = FleetRuntimeSnapshot(
        status=FleetStatus.SUCCEEDED,
        fleet_mission_id="fleet_mission_test",
        fleet_plan_version=1,
        agent_plan_versions={"uav_a": 2},
        agent_statuses={"uav_a": "SUCCEEDED"},
        assignments={
            "assignment_a": {
                "assignment_id": "assignment_a",
                "uav_id": "uav_a",
                "status": "SUCCEEDED",
            }
        },
        last_airspace_decision=None,
        events=(
            {"event_type": "FLEET_STARTED"},
            {"event_type": "FLEET_SUCCEEDED"},
        ),
    )

    summary = snapshot.to_summary_dict()

    assert "events" not in summary
    assert summary["event_count"] == 2
    assert summary["status"] == "SUCCEEDED"
