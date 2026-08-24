from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from fleet.task_spec import AssignmentConstraint, FleetTaskSpecV1, MissionGoal
from fleet.types import FleetStartPolicy
from fleet.types_v2 import FleetAssignmentV2, FleetMissionPlanV2
from models.adapter_registry import ModelCallRole
from planner.spatial import CircleRegion
from scripts import run_fleet_planner_demo


_ROOT = Path(__file__).resolve().parents[2]
_INSTRUCTION = (
    "无人机A前往世界坐标二十、三十附近十五米范围搜索并跟踪目标i二十秒；"
    "无人机B前往世界坐标负二十五、十附近十二米范围搜索并跟踪目标j二十秒；"
    "完成后分别返回各自起点降落"
)


def test_scripted_demo_prints_all_required_sections_without_isaac(capsys) -> None:
    before = {name for name in sys.modules if name.startswith(("isaacsim", "omni"))}
    code = run_fleet_planner_demo.main(
        [
            "--config",
            str(_ROOT / "configs/multi_uav_demo.yaml"),
            "--fleet-planner",
            "scripted",
            "--local-planner",
            "dynamic_scripted",
            "--planning-contract",
            "v3",
            "--adapter-config",
            str(_ROOT / "configs/adapters.json"),
            "--instruction",
            _INSTRUCTION,
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    for heading in (
        "FleetMissionPlan",
        "Assignment summary",
        "Per-agent local plan",
        "Adapter selection",
        "Fleet Planner diagnostics",
        "Per-agent planner diagnostics",
        "FleetTaskSpec",
        "Mission Interpreter diagnostics",
        "Fleet Plan semantic findings",
    ):
        assert f"=== {heading} ===" in output
    assert '"effective_model": "Qwen3-VL-4B-Instruct"' in output
    assert '"fallback_used": true' in output
    after = {name for name in sys.modules if name.startswith(("isaacsim", "omni"))}
    assert after == before


def test_scripted_demo_preserves_both_routing_relationships() -> None:
    arguments = run_fleet_planner_demo._parser().parse_args(
        [
            "--config",
            str(_ROOT / "configs/multi_uav_demo.yaml"),
            "--adapter-config",
            str(_ROOT / "configs/adapters.json"),
            "--instruction",
            _INSTRUCTION,
        ]
    )
    result = run_fleet_planner_demo.run(arguments)
    summaries = result["assignment_summary"]
    assert [(row["uav_id"], row["target_alias"]) for row in summaries] == [
        ("uav_a", "target_i"),
        ("uav_b", "target_j"),
    ]
    assert summaries[0]["search_region"]["center_xyz_m"] == [20.0, 30.0, 0.0]
    assert summaries[1]["search_region"]["center_xyz_m"] == [-25.0, 10.0, 0.0]


def test_planner_modes_resolve_matching_interpreter_and_local_defaults() -> None:
    scripted = run_fleet_planner_demo._parser().parse_args([])
    assert scripted.mission_interpreter is None
    assert scripted.local_planner is None
    assert run_fleet_planner_demo._effective_interpreter(scripted) == "scripted"
    assert (
        run_fleet_planner_demo._effective_local_planner(scripted)
        == "dynamic_scripted"
    )

    llm = run_fleet_planner_demo._parser().parse_args(["--fleet-planner", "llm"])
    assert run_fleet_planner_demo._effective_interpreter(llm) == "llm"
    assert run_fleet_planner_demo._effective_local_planner(llm) == "dynamic_llm"


def test_cross_contract_modes_fail_instead_of_falling_back() -> None:
    args = run_fleet_planner_demo._parser().parse_args(
        [
            "--fleet-planner",
            "llm",
            "--mission-interpreter",
            "scripted",
        ]
    )
    with pytest.raises(ValueError, match="must match"):
        run_fleet_planner_demo._effective_interpreter(args)

    args = run_fleet_planner_demo._parser().parse_args(
        [
            "--fleet-planner",
            "llm",
            "--local-planner",
            "dynamic_scripted",
        ]
    )
    with pytest.raises(ValueError, match="cross-contract fallback is disabled"):
        run_fleet_planner_demo._effective_local_planner(args)


def test_scripted_path_still_uses_fixed_parser(monkeypatch) -> None:
    calls: list[str] = []
    original = run_fleet_planner_demo.parse_explicit_assignment_instruction

    def recording_parser(instruction, config):
        calls.append(instruction)
        return original(instruction, config)

    monkeypatch.setattr(
        run_fleet_planner_demo,
        "parse_explicit_assignment_instruction",
        recording_parser,
    )
    args = run_fleet_planner_demo._parser().parse_args(
        [
            "--config",
            str(_ROOT / "configs/multi_uav_demo.yaml"),
            "--adapter-config",
            str(_ROOT / "configs/adapters.json"),
            "--instruction",
            _INSTRUCTION,
        ]
    )

    result = run_fleet_planner_demo.run(args)

    assert calls == [_INSTRUCTION]
    assert result["fleet_task_spec"] is None
    assert result["mission_interpreter_diagnostics"]["model_calls"] == 0


def test_llm_v2_never_calls_fixed_parser_and_uses_roles_in_order(
    monkeypatch,
) -> None:
    task_spec = FleetTaskSpecV1(
        source_text="请找一下目标i，具体派谁由系统决定",
        goals=(
            MissionGoal(
                "goal_search_i",
                "SEARCH_TARGET",
                "target_i",
                CircleRegion("WORLD_ENU", (20.0, 30.0, 0.0), 15.0),
                None,
                None,
                "MUST",
            ),
        ),
        assignment_constraints=(
            AssignmentConstraint(
                "constraint_prefer_a",
                "uav_a",
                ("goal_search_i",),
                "PREFER",
            ),
        ),
    )

    class FakeFactory:
        latest = None

        def __init__(self, *args, **kwargs):
            self.roles = []
            FakeFactory.latest = self

        def for_role(self, role, **routing):
            self.roles.append((role, dict(routing)))
            return SimpleNamespace(role=role)

    diagnostics = SimpleNamespace(
        to_dict=lambda: {
            "model_calls": 1,
            "repair_used": False,
            "repair_succeeded": False,
            "initial_output_valid": True,
            "final_output_valid": True,
            "initial_error_code": None,
            "initial_error_message": None,
            "structured_output_enabled": True,
        }
    )

    class FakeInterpreter:
        source = "mission_interpreter_llm"

        def __init__(self, client, **kwargs):
            assert client.role is ModelCallRole.MISSION_INTERPRETATION
            self.last_diagnostics = diagnostics
            self.model_proposals = ({"accepted": True},)

        def interpret(self, instruction):
            assert instruction == task_spec.source_text
            return task_spec

    class FakeFleetPlanner:
        source = "fleet_llm_v2"

        def __init__(self, client):
            assert client.role is ModelCallRole.FLEET_PLAN
            self.last_diagnostics = diagnostics
            self.model_proposals = ({"accepted": True},)

        def plan(self, request):
            return FleetMissionPlanV2(
                request.fleet_mission_id,
                request.fleet_plan_version,
                (
                    FleetAssignmentV2(
                        "assignment_search_i",
                        "uav_b",
                        ("goal_search_i",),
                        100,
                        FleetStartPolicy.PARALLEL,
                    ),
                ),
                request.coordination_policy,
            )

    class FakeLocalPlanner:
        source = "dynamic_llm"

        def __init__(self, client, **kwargs):
            assert client.role is ModelCallRole.AGENT_SPATIAL_PLAN
            self.last_diagnostics = diagnostics

        def plan(self, request):  # pragma: no cover - fake compiler owns this seam
            raise AssertionError("unexpected direct local plan call")

    class Payload:
        def __init__(self, value):
            self.value = value

        def to_dict(self):
            return self.value

    class FakeCompiler:
        def __init__(self, local_planners, *, validator=None):
            self.local_planners = local_planners

        def compile_v2(
            self,
            request,
            plan,
            contexts,
            *,
            target_catalog,
        ):
            assert set(self.local_planners) == {"uav_b"}
            assert set(contexts) == {"uav_b"}
            assert "target_i" in target_catalog
            return {
                "uav_b": SimpleNamespace(
                    agent_request=Payload({"goals": ["goal_search_i"]}),
                    planner_output=Payload({"steps": [{"skill": "SEARCH"}]}),
                    compiled_mission=SimpleNamespace(
                        task_plan=Payload({"steps": [{"skill": "SEARCH"}]})
                    ),
                    goal_coverage=Payload({"complete": True}),
                    validation_report=Payload({"hard_blocked": False}),
                )
            }

    def forbidden_fixed_parser(*args, **kwargs):
        raise AssertionError("LLM v2 path called the fixed parser")

    import planner.dynamic_llm_planner as dynamic_llm_planner

    monkeypatch.setattr(run_fleet_planner_demo, "ModelClientFactory", FakeFactory)
    monkeypatch.setattr(
        run_fleet_planner_demo, "LLMFleetTaskInterpreter", FakeInterpreter
    )
    monkeypatch.setattr(run_fleet_planner_demo, "LLMFleetPlannerV2", FakeFleetPlanner)
    monkeypatch.setattr(run_fleet_planner_demo, "FleetAssignmentCompiler", FakeCompiler)
    monkeypatch.setattr(dynamic_llm_planner, "DynamicLLMPlanner", FakeLocalPlanner)
    monkeypatch.setattr(
        run_fleet_planner_demo,
        "parse_explicit_assignment_instruction",
        forbidden_fixed_parser,
    )
    args = run_fleet_planner_demo._parser().parse_args(
        [
            "--config",
            str(_ROOT / "configs/multi_uav_demo.yaml"),
            "--adapter-config",
            str(_ROOT / "configs/adapters.json"),
            "--fleet-planner",
            "llm",
            "--instruction",
            task_spec.source_text,
        ]
    )

    result = run_fleet_planner_demo.run(args)

    assert [item[0] for item in FakeFactory.latest.roles] == [
        ModelCallRole.MISSION_INTERPRETATION,
        ModelCallRole.FLEET_PLAN,
        ModelCallRole.AGENT_SPATIAL_PLAN,
    ]
    mission_ids = [
        item[1]["fleet_mission_id"] for item in FakeFactory.latest.roles
    ]
    assert len(set(mission_ids)) == 1
    assert FakeFactory.latest.roles[2][1] == {
        "fleet_mission_id": mission_ids[0],
        "assignment_id": "assignment_search_i",
        "uav_id": "uav_b",
    }
    assert result["fleet_task_spec"] == task_spec.to_dict()
    assert result["assignment_summary"][0]["uav_id"] == "uav_b"
    assert result["assignment_summary"][0]["goal_ids"] == ["goal_search_i"]
    assert {
        item["code"] for item in result["fleet_plan_semantic_findings"]
    } == {"UNEXPLAINED_ASSIGNMENT_DEVIATION"}
    assert result["per_agent_local_plan"]["uav_b"]["goal_coverage"] == {
        "complete": True
    }
