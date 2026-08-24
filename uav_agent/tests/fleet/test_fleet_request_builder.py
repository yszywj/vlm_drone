from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from configs.loader import load_config
from fleet.compiler import FleetAssignmentCompiler
from fleet.local_spatial_planner import ScriptedAssignmentSpatialPlanner
from fleet.request_builder import (
    FleetRequestBuildError,
    build_agent_world_contexts,
    build_fleet_mission_request,
    derive_safe_home_landing_tolerances,
    parse_explicit_assignment_instruction,
)
from fleet.scripted_planner import ScriptedFleetPlanner
from runtime.plan_validator import PlanValidator


_ROOT = Path(__file__).resolve().parents[2]
_INSTRUCTION = (
    "无人机A前往世界坐标二十、三十附近十五米范围搜索并跟踪目标i二十秒；"
    "无人机B前往世界坐标负二十五、十附近十二米范围搜索并跟踪目标j二十秒；"
    "完成后分别返回各自起点降落"
)


def test_documented_explicit_demo_builds_expected_assignments() -> None:
    config = load_config(_ROOT / "configs/multi_uav_demo.yaml")
    directives = parse_explicit_assignment_instruction(_INSTRUCTION, config)
    request = build_fleet_mission_request(config, _INSTRUCTION, directives=directives)
    plan = ScriptedFleetPlanner().plan(request)

    assert [item.uav_id for item in plan.assignments] == ["uav_a", "uav_b"]
    assert [item.target_alias for item in plan.assignments] == ["target_i", "target_j"]
    assert plan.assignments[0].search_region.to_dict() == {
        "shape": "CIRCLE",
        "frame": "WORLD_ENU",
        "center_xyz_m": [20.0, 30.0, 0.0],
        "radius_m": 15.0,
    }
    assert plan.assignments[1].search_region.to_dict()["center_xyz_m"] == [-25.0, 10.0, 0.0]
    assert plan.assignments[1].search_region.to_dict()["radius_m"] == 12.0


def test_request_and_context_do_not_expose_target_ground_truth() -> None:
    config = load_config(_ROOT / "configs/multi_uav_demo.yaml")
    request = build_fleet_mission_request(
        config,
        _INSTRUCTION,
        directives=parse_explicit_assignment_instruction(_INSTRUCTION, config),
    )
    plan = ScriptedFleetPlanner().plan(request)
    serialized = json.dumps(request.to_dict(), ensure_ascii=False)
    for forbidden in (
        "initial_region",
        "oracle_target",
        "motion",
        "target_velocity",
    ):
        assert forbidden not in serialized
    contexts = build_agent_world_contexts(config, plan)
    assert contexts["uav_a"].initial_uav_xyz_m == (-3.0, 0.0, 0.0)
    assert contexts["uav_b"].initial_uav_xyz_m == (3.0, 0.0, 0.0)
    assert contexts["uav_a"].search_regions == {}


def test_each_assignment_compiles_to_independent_spatial_v3_plan() -> None:
    config = load_config(_ROOT / "configs/multi_uav_demo.yaml")
    request = build_fleet_mission_request(
        config,
        _INSTRUCTION,
        directives=parse_explicit_assignment_instruction(_INSTRUCTION, config),
    )
    plan = ScriptedFleetPlanner().plan(request)
    compiled = FleetAssignmentCompiler(
        ScriptedAssignmentSpatialPlanner(),
        validator=PlanValidator(),
    ).compile(request, plan, build_agent_world_contexts(config, plan))

    assert set(compiled) == {"uav_a", "uav_b"}
    for uav_id, result in compiled.items():
        assert result.agent_request.uav_id == uav_id
        assert result.compiled_mission is not None
        assert result.compiled_mission.task_plan.uav_id == uav_id
        assert [step.skill.value for step in result.compiled_mission.task_plan.steps] == [
            "TAKEOFF",
            "SEARCH",
            "TRACK",
            "GOTO",
            "LAND",
        ]


def test_demo_home_tolerances_preserve_worst_case_fleet_separation() -> None:
    config = load_config(_ROOT / "configs/multi_uav_demo.yaml")
    request = build_fleet_mission_request(
        config,
        _INSTRUCTION,
        directives=parse_explicit_assignment_instruction(_INSTRUCTION, config),
    )
    plan = ScriptedFleetPlanner().plan(request)

    tolerances = derive_safe_home_landing_tolerances(config, plan)
    contexts = build_agent_world_contexts(config, plan)

    # Homes are 6 m apart and minimum separation is 5 m.  A 5% safety
    # margin leaves 0.75 m total tolerance budget, split symmetrically.
    assert tolerances == {
        "uav_a": pytest.approx(0.375),
        "uav_b": pytest.approx(0.375),
    }
    worst_case_distance = 6.0 - tolerances["uav_a"] - tolerances["uav_b"]
    assert worst_case_distance == pytest.approx(5.25)
    assert worst_case_distance > config.fleet.minimum_uav_separation_m
    for uav_id, context in contexts.items():
        zone = next(iter(context.landing_zones.values()))
        assert zone.horizontal_tolerance_m == pytest.approx(
            tolerances[uav_id]
        )


def test_compiled_return_and_land_use_derived_safe_home_tolerance() -> None:
    config = load_config(_ROOT / "configs/multi_uav_demo.yaml")
    request = build_fleet_mission_request(
        config,
        _INSTRUCTION,
        directives=parse_explicit_assignment_instruction(_INSTRUCTION, config),
    )
    plan = ScriptedFleetPlanner().plan(request)
    contexts = build_agent_world_contexts(config, plan)
    compiled = FleetAssignmentCompiler(
        ScriptedAssignmentSpatialPlanner(),
        validator=PlanValidator(),
    ).compile(request, plan, contexts)

    for result in compiled.values():
        assert result.compiled_mission is not None
        steps = result.compiled_mission.task_plan.steps
        goto_home = next(step for step in steps if step.skill.value == "GOTO")
        land = next(step for step in steps if step.skill.value == "LAND")
        assert goto_home.params["tolerance"] == pytest.approx(0.375)
        assert land.params["zone_tolerance_m"] == pytest.approx(0.375)


def test_home_geometry_fails_before_runtime_when_no_safe_tolerance_exists() -> None:
    config = load_config(_ROOT / "configs/multi_uav_demo.yaml")
    unsafe = replace(
        config,
        uav=None,
        camera=None,
        target=None,
        uavs=(
            replace(
                config.uavs[0],
                initial_position_xyz_m=(-2.65, 0.0, 0.0),
            ),
            replace(
                config.uavs[1],
                initial_position_xyz_m=(2.65, 0.0, 0.0),
            ),
        ),
    )
    request = build_fleet_mission_request(
        unsafe,
        _INSTRUCTION,
        directives=parse_explicit_assignment_instruction(_INSTRUCTION, unsafe),
    )
    plan = ScriptedFleetPlanner().plan(request)

    with pytest.raises(FleetRequestBuildError, match="homes are too close"):
        build_agent_world_contexts(unsafe, plan)


def test_active_uavs_cannot_share_a_landing_zone_id() -> None:
    config = load_config(_ROOT / "configs/multi_uav_demo.yaml")
    duplicate_zone = replace(
        config,
        uav=None,
        camera=None,
        target=None,
        uavs=(
            config.uavs[0],
            replace(config.uavs[1], home_name="home_a"),
        ),
    )
    request = build_fleet_mission_request(
        duplicate_zone,
        _INSTRUCTION,
        directives=parse_explicit_assignment_instruction(
            _INSTRUCTION,
            duplicate_zone,
        ),
    )
    plan = ScriptedFleetPlanner().plan(request)

    with pytest.raises(FleetRequestBuildError, match="distinct landing-zone IDs"):
        build_agent_world_contexts(duplicate_zone, plan)
