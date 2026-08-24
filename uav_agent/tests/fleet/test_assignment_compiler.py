from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fleet.compiler import AssignmentCompilerError, FleetAssignmentCompiler
from fleet.scripted_planner import ScriptedFleetPlanner
from fleet.types import (
    AssignmentCompilation,
    FleetMissionRequest,
    FleetTargetRequest,
    FleetUavCapability,
)
from planner.schemas import (
    LandingZoneSpec,
    PlannerRequest,
    PlannerWorldContext,
    SearchRegionSpec,
)
from planner.schemas_v3 import SkillPlanDraftV3
from planner.spatial import CircleRegion, CoordinateFrame
from runtime.plan_validator import PlanValidator
from target.types import TargetSpec


class _LocalSpatialPlanner:
    source = "dynamic_scripted"

    def __init__(self, mutation: str | None = None) -> None:
        self.mutation = mutation
        self.requests: list[PlannerRequest] = []

    def plan(self, request: PlannerRequest) -> object:
        self.requests.append(request)
        payload = json.loads(request.instruction)
        plan: dict[str, object] = {
            "schema_version": 3,
            "mission_id": request.mission_id,
            "uav_id": request.uav_id,
            "plan_version": request.plan_version,
            "assumptions": [],
            "target_spec": payload["target_spec"],
            "steps": [
                {
                    "id": "takeoff_1",
                    "uav_id": request.uav_id,
                    "skill": "TAKEOFF",
                    "args": {"altitude_m": 10},
                },
                {
                    "id": "search_1",
                    "uav_id": request.uav_id,
                    "skill": "SEARCH",
                    "args": {
                        "region": payload["search_region"],
                        "strategy": {"kind": "LAWNMOWER", "spacing_m": 4},
                        "entry_policy": "START_IN_PLACE_IF_INSIDE",
                        "target_description": payload["target_spec"][
                            "original_description"
                        ],
                        "search_altitude_m": 10,
                        "timeout_s": 60,
                    },
                },
                {
                    "id": "track_1",
                    "uav_id": request.uav_id,
                    "skill": "TRACK",
                    "args": {
                        "target_ref": "$search_1.target_id",
                        "duration_s": payload["track_duration_s"],
                    },
                },
                {
                    "id": "goto_home",
                    "uav_id": request.uav_id,
                    "skill": "GOTO",
                    "args": {
                        "target": {
                            "kind": "NAMED_LOCATION",
                            "name": payload["return_home"],
                        }
                    },
                },
                {
                    "id": "land_1",
                    "uav_id": request.uav_id,
                    "skill": "LAND",
                    "args": {"zone": payload["return_home"]},
                },
            ],
        }
        if self.mutation == "routing":
            plan["mission_id"] = "mission_tampered"
        elif self.mutation == "target":
            plan["target_spec"] = TargetSpec("different target").to_dict()
        elif self.mutation == "region":
            plan["steps"][1]["args"]["region"] = CircleRegion(  # type: ignore[index]
                CoordinateFrame.WORLD_ENU, (80, 80, 0), 5
            ).to_dict()
        elif self.mutation == "duration":
            plan["steps"][2]["args"]["duration_s"] = 999  # type: ignore[index]
        elif self.mutation == "legacy":
            return {"steps": plan["steps"]}
        return SkillPlanDraftV3.from_dict(plan)


def _uav(uav_id: str) -> FleetUavCapability:
    return FleetUavCapability(
        uav_id,
        uav_id.upper(),
        True,
        f"home_{uav_id[-1]}",
        5,
        30,
    )


def _fleet_request() -> FleetMissionRequest:
    return FleetMissionRequest(
        "fleet_mission_compile",
        2,
        "target_i is confidential to uav_a; target_j is confidential to uav_b",
        (_uav("uav_a"), _uav("uav_b")),
        (
            FleetTargetRequest(
                "target_i",
                TargetSpec("red target i", immutable_identity_summary="target i"),
                "uav_a",
                CircleRegion(CoordinateFrame.WORLD_ENU, (10, 20, 0), 8),
                20,
            ),
            FleetTargetRequest(
                "target_j",
                TargetSpec("blue target j", immutable_identity_summary="target j"),
                "uav_b",
                CircleRegion(CoordinateFrame.WORLD_ENU, (30, 40, 0), 9),
                15,
            ),
        ),
    )


def _world_context() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-100, -100, 0),
        scene_max_xyz_m=(100, 100, 40),
        initial_uav_xyz_m=(0, 0, 0),
        search_regions={
            "target_i_private_region": SearchRegionSpec(
                "target_i_private_region", (10, 20, 0), 8, (0, 0, 10)
            ),
            "target_j_private_region": SearchRegionSpec(
                "target_j_private_region", (30, 40, 0), 9, (0, 0, 10)
            ),
        },
        landing_zones={
            "home_a": LandingZoneSpec("home_a", (-3, 0), 0),
            "home_b": LandingZoneSpec("home_b", (3, 0), 0),
        },
        default_takeoff_altitude_m=10,
        default_track_duration_s=30,
        search_timeout_s=75,
    )


def test_compiler_builds_one_isolated_routed_request_per_assignment() -> None:
    request = _fleet_request()
    plan = ScriptedFleetPlanner().plan(request)
    planner_a = _LocalSpatialPlanner()
    planner_b = _LocalSpatialPlanner()
    compiler = FleetAssignmentCompiler(
        {"uav_a": planner_a, "uav_b": planner_b}
    )
    compiled = compiler.compile(
        request,
        plan,
        {"uav_a": _world_context(), "uav_b": _world_context()},
        local_plan_versions={"uav_a": 3, "uav_b": 8},
    )
    assert set(compiled) == {"uav_a", "uav_b"}
    assert all(isinstance(value, AssignmentCompilation) for value in compiled.values())
    assert compiled["uav_a"].planner_output.plan_version == 3
    assert compiled["uav_b"].planner_output.plan_version == 8
    assert compiled["uav_a"].compiled_mission is None
    assert compiled["uav_a"].agent_request.fleet_safety_summary[0].uav_id == "uav_b"
    assert compiled["uav_a"].agent_request.fleet_safety_summary[0].assignment_id == (
        compiled["uav_b"].agent_request.assignment_id
    )

    for local, own_alias, other_alias, own_home in (
        (planner_a, "target_i", "target_j", "home_a"),
        (planner_b, "target_j", "target_i", "home_b"),
    ):
        assert len(local.requests) == 1
        local_request = local.requests[0]
        focused = json.loads(local_request.instruction)
        assert focused["target_alias"] == own_alias
        assert local_request.trusted_target_spec == compiled[
            local_request.uav_id
        ].agent_request.target_spec
        assert local_request.require_empty_spatial_assumptions is True
        assert focused["required_spatial_assumptions"] == []
        assert any(
            "Set top-level assumptions to []" in requirement
            for requirement in focused["requirements"]
        )
        assert any(
            "top-level output field target_spec" in requirement
            for requirement in focused["requirements"]
        )
        own_payload = dict(focused)
        safety_summary = own_payload.pop("fleet_safety_summary")
        assert other_alias not in json.dumps(own_payload, ensure_ascii=False)
        assert local_request.world_context.search_regions == {}
        assert set(local_request.world_context.landing_zones) == {own_home}
        assert "Oracle" not in local_request.instruction
        assert len(safety_summary) == 1
        assert safety_summary[0]["uav_id"] != local_request.uav_id
        assert set(safety_summary[0]) == {
            "uav_id",
            "assignment_id",
            "status",
            "plan_version",
            "current_region",
            "altitude_layer",
        }


def test_compiler_can_reuse_existing_plan_validator_for_each_local_plan() -> None:
    request = _fleet_request()
    plan = ScriptedFleetPlanner().plan(request)
    compiler = FleetAssignmentCompiler(
        _LocalSpatialPlanner(), validator=PlanValidator()
    )
    result = compiler.compile_assignment(
        request, plan, plan.assignments[0], _world_context()
    )
    assert result.compiled_mission is not None
    assert result.compiled_mission.source == "dynamic_scripted"
    assert result.compiled_mission.planner_output == result.planner_output


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("routing", "routing/version"),
        ("target", "target_spec"),
        ("region", "RegionSpec"),
        ("duration", "track duration"),
        ("legacy", "SkillPlanDraftV3"),
    ),
)
def test_compiler_fails_closed_when_local_planner_changes_assignment(
    mutation: str, message: str
) -> None:
    request = _fleet_request()
    plan = ScriptedFleetPlanner().plan(request)
    compiler = FleetAssignmentCompiler(_LocalSpatialPlanner(mutation))
    with pytest.raises(AssignmentCompilerError, match=message):
        compiler.compile_assignment(
            request, plan, plan.assignments[0], _world_context()
        )


def test_compiler_rejects_non_member_and_does_not_flatten_sequential_work() -> None:
    request = _fleet_request()
    plan = ScriptedFleetPlanner().plan(request)
    foreign = deepcopy(plan.assignments[0].to_dict())
    foreign["assignment_id"] = "assignment_foreign"
    with pytest.raises(AssignmentCompilerError, match="exact member"):
        FleetAssignmentCompiler(_LocalSpatialPlanner()).build_agent_request(
            request,
            plan,
            type(plan.assignments[0]).from_dict(foreign),
        )
