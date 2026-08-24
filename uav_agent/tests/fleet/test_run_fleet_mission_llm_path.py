from __future__ import annotations

import builtins
import json
from pathlib import Path
import sys
from typing import Sequence

import pytest

from models.adapter_registry import ModelCallRole
from models.base import ChatMessage, GenerationOptions, ModelResponse
from scripts import run_fleet_mission


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/multi_uav_demo.yaml"
INSTRUCTION = (
    "无人机A在世界坐标二十、三十附近搜索目标i并跟踪十秒后返航降落；"
    "无人机B在世界坐标负二十五、十附近搜索目标j并跟踪十秒后返航降落"
)


def _task_spec_payload(source_text: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_text": source_text,
        "goals": [
            {
                "goal_id": "goal_search_i",
                "goal_type": "SEARCH_TARGET",
                "target_alias": "target_i",
                "spatial_constraint": {
                    "shape": "CIRCLE",
                    "frame": "WORLD_ENU",
                    "center_xyz_m": [20.0, 30.0, 0.0],
                    "radius_m": 15.0,
                },
                "duration_s": None,
                "distance_m": None,
                "strength": "MUST",
                "evidence_refs": [],
            },
            {
                "goal_id": "goal_track_i",
                "goal_type": "TRACK_TARGET",
                "target_alias": "target_i",
                "spatial_constraint": None,
                "duration_s": 10.0,
                "distance_m": None,
                "strength": "MUST",
                "evidence_refs": [],
            },
            {
                "goal_id": "goal_search_j",
                "goal_type": "SEARCH_TARGET",
                "target_alias": "target_j",
                "spatial_constraint": {
                    "shape": "CIRCLE",
                    "frame": "WORLD_ENU",
                    "center_xyz_m": [-25.0, 10.0, 0.0],
                    "radius_m": 12.0,
                },
                "duration_s": None,
                "distance_m": None,
                "strength": "MUST",
                "evidence_refs": [],
            },
            {
                "goal_id": "goal_track_j",
                "goal_type": "TRACK_TARGET",
                "target_alias": "target_j",
                "spatial_constraint": None,
                "duration_s": 10.0,
                "distance_m": None,
                "strength": "MUST",
                "evidence_refs": [],
            },
        ],
        "assignment_constraints": [
            {
                "constraint_id": "constraint_uav_a",
                "uav_id": "uav_a",
                "goal_ids": ["goal_search_i", "goal_track_i", "goal_land_a"],
                "strength": "MUST",
                "evidence_refs": [],
            },
            {
                "constraint_id": "constraint_uav_b",
                "uav_id": "uav_b",
                "goal_ids": ["goal_search_j", "goal_track_j", "goal_land_b"],
                "strength": "MUST",
                "evidence_refs": [],
            },
        ],
        "ordering_constraints": [
            {
                "constraint_id": "order_search_track_i",
                "before_goal_id": "goal_search_i",
                "after_goal_id": "goal_track_i",
                "strength": "MUST",
                "evidence_refs": [],
            },
            {
                "constraint_id": "order_search_track_j",
                "before_goal_id": "goal_search_j",
                "after_goal_id": "goal_track_j",
                "strength": "MUST",
                "evidence_refs": [],
            },
        ],
        "termination_goals": [
            {
                "goal_id": "goal_land_a",
                "goal_type": "RETURN_HOME_AND_LAND",
                "uav_id": "uav_a",
                "duration_s": None,
                "strength": "MUST",
                "evidence_refs": [],
            },
            {
                "goal_id": "goal_land_b",
                "goal_type": "RETURN_HOME_AND_LAND",
                "uav_id": "uav_b",
                "duration_s": None,
                "strength": "MUST",
                "evidence_refs": [],
            },
        ],
        "ambiguities": [],
        "source_evidence": [],
    }


def _fleet_plan_payload(request: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "fleet_mission_id": request["fleet_mission_id"],
        "fleet_plan_version": request["fleet_plan_version"],
        "assignments": [
            {
                "assignment_id": "assignment_target_i",
                "uav_id": "uav_a",
                "goal_ids": ["goal_search_i", "goal_track_i", "goal_land_a"],
                "priority": 100,
                "start_policy": "PARALLEL",
                "deviations": [],
            },
            {
                "assignment_id": "assignment_target_j",
                "uav_id": "uav_b",
                "goal_ids": ["goal_search_j", "goal_track_j", "goal_land_b"],
                "priority": 90,
                "start_policy": "PARALLEL",
                "deviations": [],
            },
        ],
        "coordination_policy": request["coordination_policy"],
        "assumptions": [],
        "unassigned_goal_ids": [],
    }


def _local_plan_payload(messages: Sequence[ChatMessage]) -> dict[str, object]:
    outer = json.loads(messages[1].content)
    routing = outer["trusted_routing"]
    focused = json.loads(outer["user_instruction"])
    goals = focused["assigned_goals"]
    search_goal = next(item for item in goals if item["goal_type"] == "SEARCH_TARGET")
    track_goal = next(item for item in goals if item["goal_type"] == "TRACK_TARGET")
    target_spec = next(iter(focused["trusted_target_specs"].values()))
    uav_id = routing["uav_id"]
    home = focused["own_home"]
    return {
        "schema_version": 3,
        "mission_id": routing["mission_id"],
        "uav_id": uav_id,
        "plan_version": routing["plan_version"],
        "target_spec": target_spec,
        "assumptions": [],
        "steps": [
            {
                "id": "takeoff_1",
                "uav_id": uav_id,
                "skill": "TAKEOFF",
                "args": {"altitude_m": 10.0},
            },
            {
                "id": "search_1",
                "uav_id": uav_id,
                "skill": "SEARCH",
                "args": {
                    "region": search_goal["spatial_constraint"],
                    "strategy": {"kind": "SPIRAL_OUT", "spacing_m": 4.0},
                    "entry_policy": "START_IN_PLACE_IF_INSIDE",
                    "target_description": target_spec["original_description"],
                    "search_altitude_m": 10.0,
                    "timeout_s": 60.0,
                },
            },
            {
                "id": "track_1",
                "uav_id": uav_id,
                "skill": "TRACK",
                "args": {
                    "target_ref": "$search_1.target_id",
                    "duration_s": track_goal["duration_s"],
                },
            },
            {
                "id": "goto_home",
                "uav_id": uav_id,
                "skill": "GOTO",
                "args": {"target": {"kind": "NAMED_LOCATION", "name": home}},
            },
            {
                "id": "land_1",
                "uav_id": uav_id,
                "skill": "LAND",
                "args": {"zone": home},
            },
        ],
    }


class _RoleClient:
    def __init__(
        self,
        role: ModelCallRole,
        trace: list[ModelCallRole],
        response_names: list[str],
        *,
        failing_role: ModelCallRole | None,
    ) -> None:
        self._role = role
        self._trace = trace
        self._response_names = response_names
        self._failing_role = failing_role

    def healthcheck(self) -> None:
        return None

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions | None = None,
    ) -> ModelResponse:
        self._trace.append(self._role)
        assert options is not None
        assert options.temperature == 0.0
        assert options.response_format is not None
        self._response_names.append(options.response_format.name)
        if self._role is self._failing_role:
            return ModelResponse("not-json", "fake-qwen", "stop", {})
        if self._role is ModelCallRole.MISSION_INTERPRETATION:
            request = json.loads(messages[1].content)
            payload = _task_spec_payload(request["source_text"])
        elif self._role is ModelCallRole.FLEET_PLAN:
            request = json.loads(messages[1].content)["trusted_request"]
            payload = _fleet_plan_payload(request)
        elif self._role is ModelCallRole.AGENT_SPATIAL_PLAN:
            payload = _local_plan_payload(messages)
        else:  # pragma: no cover - the test launch does not enable vision
            raise AssertionError(f"unexpected model role: {self._role.value}")
        return ModelResponse(
            json.dumps(payload, ensure_ascii=False, allow_nan=False),
            "fake-qwen",
            "stop",
            {},
        )


def _install_fake_factory(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failing_role: ModelCallRole | None = None,
) -> tuple[list[ModelCallRole], list[str]]:
    trace: list[ModelCallRole] = []
    response_names: list[str] = []

    class _FakeFactory:
        def __init__(self, registry, **kwargs) -> None:
            self._registry = registry
            self._selection_logger = kwargs.get("selection_logger")

        def for_role(self, role, **routing):
            del routing
            normalized = role if isinstance(role, ModelCallRole) else ModelCallRole(role)
            if self._selection_logger is not None:
                self._selection_logger(self._registry.resolve(normalized).to_dict())
            return _RoleClient(
                normalized,
                trace,
                response_names,
                failing_role=failing_role,
            )

    monkeypatch.setattr(run_fleet_mission, "ModelClientFactory", _FakeFactory)
    return trace, response_names


def _guard_isaac_import(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    attempted: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "isaacsim" or name.startswith("isaacsim."):
            attempted.append(name)
            raise AssertionError("pure planning crossed the Isaac boundary")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return attempted


def _llm_argv(*extra: str) -> list[str]:
    return [
        "--config",
        str(CONFIG),
        "--instruction",
        INSTRUCTION,
        "--fleet-planner",
        "llm",
        *extra,
    ]


def test_llm_defaults_run_interpreter_then_fleet_then_each_local_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace, response_names = _install_fake_factory(monkeypatch)
    attempted_isaac = _guard_isaac_import(monkeypatch)
    monkeypatch.setattr(
        run_fleet_mission,
        "parse_explicit_assignment_instruction",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("LLM path called the fixed grammar parser")
        ),
    )
    args = run_fleet_mission.parse_args(
        _llm_argv(
            "--perception-runtime-profile",
            "oracle_evaluation",
            "--acknowledge-privileged-oracle",
        )
    )

    prepared = run_fleet_mission.prepare_fleet_mission(args)

    assert args.mission_interpreter is None
    assert args.local_planner is None
    assert prepared.mission_interpreter_source == "qwen_task_spec_v1"
    assert prepared.fleet_planner_source == "fleet_llm_v2"
    assert prepared.local_planner_source == "dynamic_llm"
    assert prepared.task_spec is not None
    assert prepared.fleet_plan_v2 is not None
    assert [item.uav_id for item in prepared.fleet_plan_v2.assignments] == [
        "uav_a",
        "uav_b",
    ]
    assert sorted(prepared.compilations) == ["uav_a", "uav_b"]
    assert all(item.executable for item in prepared.compilations.values())
    assert all(item.semantically_valid for item in prepared.compilations.values())
    assert trace == [
        ModelCallRole.MISSION_INTERPRETATION,
        ModelCallRole.FLEET_PLAN,
        ModelCallRole.AGENT_SPATIAL_PLAN,
        ModelCallRole.AGENT_SPATIAL_PLAN,
    ]
    assert response_names == [
        "fleet_task_spec_v1",
        "fleet_mission_plan_v2",
        "skill_plan_draft_v3",
        "skill_plan_draft_v3",
    ]
    assert attempted_isaac == []
    assert not any(name == "isaacsim" for name in sys.modules)


def test_initial_local_structural_repair_is_focused_and_bounded_to_three_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[ModelCallRole] = []
    local_messages: list[Sequence[ChatMessage]] = []

    class _RepairClient:
        def __init__(self, role: ModelCallRole) -> None:
            self._role = role

        def healthcheck(self) -> None:
            return None

        def chat(self, messages, *, options=None):
            assert options is not None and options.temperature == 0.0
            trace.append(self._role)
            if self._role is ModelCallRole.MISSION_INTERPRETATION:
                request = json.loads(messages[1].content)
                payload = _task_spec_payload(request["source_text"])
            elif self._role is ModelCallRole.FLEET_PLAN:
                request = json.loads(messages[1].content)["trusted_request"]
                payload = _fleet_plan_payload(request)
            elif self._role is ModelCallRole.AGENT_SPATIAL_PLAN:
                local_messages.append(messages)
                # The first assignment consumes exactly initial + two focused
                # Fleet-owned retries; the second assignment succeeds once.
                if len(local_messages) <= 2:
                    return ModelResponse("not-json", "fake-qwen", "stop", {})
                payload = _local_plan_payload(messages)
            else:  # pragma: no cover - vision is disabled in this test
                raise AssertionError(f"unexpected role: {self._role.value}")
            return ModelResponse(
                json.dumps(payload, ensure_ascii=False, allow_nan=False),
                "fake-qwen",
                "stop",
                {},
            )

    class _RepairFactory:
        def __init__(self, registry, **kwargs) -> None:
            self._registry = registry
            self._selection_logger = kwargs.get("selection_logger")

        def for_role(self, role, **routing):
            del routing
            normalized = role if isinstance(role, ModelCallRole) else ModelCallRole(role)
            if self._selection_logger is not None:
                self._selection_logger(self._registry.resolve(normalized).to_dict())
            return _RepairClient(normalized)

    monkeypatch.setattr(run_fleet_mission, "ModelClientFactory", _RepairFactory)
    attempted_isaac = _guard_isaac_import(monkeypatch)
    args = run_fleet_mission.parse_args(
        _llm_argv(
            "--perception-runtime-profile",
            "oracle_evaluation",
            "--acknowledge-privileged-oracle",
        )
    )

    prepared = run_fleet_mission.prepare_fleet_mission(args)

    assert sorted(prepared.compilations) == ["uav_a", "uav_b"]
    assert len(local_messages) == 4
    outer = [json.loads(messages[1].content) for messages in local_messages]
    uav_a_attempts = [
        item for item in outer if item["trusted_routing"]["uav_id"] == "uav_a"
    ]
    assert len(uav_a_attempts) == 3
    assert [item["trusted_routing"]["plan_version"] for item in uav_a_attempts] == [
        1,
        2,
        3,
    ]
    focused = [json.loads(item["user_instruction"]) for item in uav_a_attempts]
    assert "proposal_repair_findings" not in focused[0]
    assert [item["proposal_repair_findings"][0]["code"] for item in focused[1:]] == [
        "INVALID_JSON",
        "INVALID_JSON",
    ]
    assert all("semantic_repair_findings" not in item for item in focused)
    retry_text = json.dumps(focused[1:], ensure_ascii=False)
    assert "not-json" not in retry_text
    assert "raw_output" not in retry_text
    assert "previous_prompt" not in retry_text
    assert attempted_isaac == []


def test_scripted_baseline_still_calls_fixed_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = run_fleet_mission.parse_explicit_assignment_instruction
    calls: list[str] = []

    def recording_parser(instruction, config, **kwargs):
        calls.append(instruction)
        return original(instruction, config, **kwargs)

    monkeypatch.setattr(
        run_fleet_mission,
        "parse_explicit_assignment_instruction",
        recording_parser,
    )
    attempted_isaac = _guard_isaac_import(monkeypatch)
    args = run_fleet_mission.parse_args(
        [
            "--config",
            str(CONFIG),
            "--fleet-planner",
            "scripted",
            "--perception-runtime-profile",
            "oracle_evaluation",
            "--acknowledge-privileged-oracle",
        ]
    )

    prepared = run_fleet_mission.prepare_fleet_mission(args)

    assert calls == [run_fleet_mission.DEFAULT_INSTRUCTION]
    assert prepared.mission_interpreter_source == "scripted_fixed_parser"
    assert prepared.local_planner_source == "dynamic_scripted"
    assert attempted_isaac == []


@pytest.mark.parametrize(
    ("failing_role", "expected_stage", "expected_calls"),
    [
        (ModelCallRole.MISSION_INTERPRETATION, "MISSION_INTERPRETATION", 2),
        (ModelCallRole.FLEET_PLAN, "FLEET_PLANNING", 3),
    ],
)
def test_llm_preparation_failure_still_writes_bounded_run_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_role: ModelCallRole,
    expected_stage: str,
    expected_calls: int,
) -> None:
    trace, _ = _install_fake_factory(monkeypatch, failing_role=failing_role)
    attempted_isaac = _guard_isaac_import(monkeypatch)
    monkeypatch.setattr(
        run_fleet_mission,
        "parse_explicit_assignment_instruction",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("LLM failure fell back to the fixed grammar parser")
        ),
    )

    exit_code = run_fleet_mission.main(
        _llm_argv(
            "--output-root",
            str(tmp_path),
            "--no-summary-figures",
        )
    )

    assert exit_code == 2
    run_root = tmp_path / "runs" / "fleet_mission"
    run_dirs = tuple(item for item in run_root.iterdir() if item.is_dir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "logs/terminal.log").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "exit_code.txt").read_text(encoding="utf-8").strip() == "2"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAILED_PREPARATION"
    assert summary["stage"] == expected_stage
    assert summary["exit_code"] == 2
    assert summary["isaac_started"] is False
    terminal = (run_dir / "logs/terminal.log").read_text(encoding="utf-8")
    assert "result_dir=" in terminal
    assert "fleet launch configuration error" in terminal
    assert trace.count(failing_role) == expected_calls
    if failing_role is ModelCallRole.FLEET_PLAN:
        assert trace[0] is ModelCallRole.MISSION_INTERPRETATION
    assert attempted_isaac == []


@pytest.mark.parametrize(
    "argv",
    [
        _llm_argv("--mission-interpreter", "scripted"),
        [
            "--config",
            str(CONFIG),
            "--fleet-planner",
            "scripted",
            "--mission-interpreter",
            "llm",
        ],
        _llm_argv("--local-planner", "dynamic_scripted"),
    ],
)
def test_explicit_incompatible_planning_modes_fail_fast(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_isaac = _guard_isaac_import(monkeypatch)
    fixed_parser_calls: list[str] = []
    monkeypatch.setattr(
        run_fleet_mission,
        "parse_explicit_assignment_instruction",
        lambda instruction, *args, **kwargs: fixed_parser_calls.append(instruction),
    )
    args = run_fleet_mission.parse_args(argv)

    with pytest.raises(run_fleet_mission.FleetLaunchConfigurationError):
        run_fleet_mission.prepare_fleet_mission(args)

    assert fixed_parser_calls == []
    assert attempted_isaac == []
