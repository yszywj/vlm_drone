"""Pure-Python replay coverage for precomputed V2 local plans.

The Fleet runtime still owns a validated V1 execution envelope while the
precomputed local input is the exact, Goal-focused V2 request.  These tests
make sure that boundary does not quietly re-introduce the historical fixed
SEARCH/TRACK Skill skeleton.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from agents.mission_agent import AgentStatus, MissionAgent
from env.kinematic_uav import KinematicUAV, UAVState
from fleet.local_spatial_planner import RoutedPreplannedSpatialPlanner
from fleet.runtime import FleetMissionRuntime, FleetRuntimeError, FleetStatus
from fleet.scripted_planner import ScriptedFleetPlanner
from fleet.types import (
    FleetCoordinationPolicy,
    FleetMissionRequest,
    FleetTargetRequest,
    FleetUavCapability,
)
from planner.schemas import LandingZoneSpec, PlannerRequest, PlannerWorldContext
from planner.schemas_v3 import SkillPlanDraftV3
from planner.spatial import CircleRegion, CoordinateFrame
from runtime.plan_validator import PlanValidator
from runtime.safety_supervisor import SafetySupervisor
from skills.manager import SkillManager, create_default_skill_registry
from skills.types import SkillContext, SkillName
from target.target_manager import TargetManager
from target.types import TargetSpec


class _Clock:
    def __init__(self) -> None:
        self.time_s = 0.0

    def now(self) -> float:
        return self.time_s


class _Camera:
    def get_rgb(self) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros(3, dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )


class _Environment:
    def __init__(self) -> None:
        self.started = 0

    def start(self, plan: object) -> None:
        del plan
        self.started += 1


def _context() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
        initial_uav_xyz_m=(0.0, 0.0, 0.0),
        search_regions={},
        landing_zones={"home": LandingZoneSpec("home", (0.0, 0.0), 0.0)},
        default_takeoff_altitude_m=10.0,
        default_track_duration_s=10.0,
        search_timeout_s=60.0,
        goto_timeout_s=120.0,
        land_timeout_s=60.0,
    )


def _request() -> FleetMissionRequest:
    return FleetMissionRequest(
        fleet_mission_id="fleet_mission_v2_replay",
        fleet_plan_version=1,
        original_instruction="飞到观察点悬停，然后返航降落",
        uav_inventory=(
            FleetUavCapability(
                "uav_1", "UAV 1", True, "home", 5.0, 30.0
            ),
        ),
        # FleetMissionRuntime currently validates a V1 execution envelope.
        # The precomputed local plan below is deliberately Goal-based and is
        # not constrained to this envelope's historical Skill skeleton.
        target_requests=(
            FleetTargetRequest(
                "execution_envelope_target",
                TargetSpec("bounded execution envelope target"),
                requested_uav_id="uav_1",
                search_region=CircleRegion(
                    CoordinateFrame.WORLD_ENU,
                    (5.0, 5.0, 0.0),
                    4.0,
                ),
                track_duration_s=5.0,
            ),
        ),
        coordination_policy=FleetCoordinationPolicy(),
    )


def _v2_focused_instruction() -> str:
    return json.dumps(
        {
            "schema_version": 2,
            "task": "Create one independent Spatial SkillPlanDraftV3.",
            "assignment_id": "assignment_uav_1_goal_visit",
            "uav_id": "uav_1",
            "assigned_goals": [
                {
                    "goal_id": "goal_visit",
                    "goal_type": "NAVIGATE",
                    "spatial_constraint": {
                        "kind": "POINT",
                        "frame": "WORLD_ENU",
                        "xyz_m": [8.0, 2.0, 10.0],
                    },
                },
                {
                    "goal_id": "goal_wait",
                    "goal_type": "WAIT",
                    "duration_s": 2.0,
                },
                {"goal_id": "goal_land", "goal_type": "LAND"},
            ],
            "trusted_target_specs": {},
            "own_home": "home",
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _v2_draft_without_search_or_track() -> SkillPlanDraftV3:
    return SkillPlanDraftV3.from_dict(
        {
            "schema_version": 3,
            "mission_id": "agent_assignment_uav_1_goal_visit",
            "uav_id": "uav_1",
            "plan_version": 1,
            "assumptions": [],
            "steps": [
                {
                    "id": "takeoff_1",
                    "uav_id": "uav_1",
                    "skill": "TAKEOFF",
                    "args": {"altitude_m": 10.0},
                },
                {
                    "id": "goto_observation_point",
                    "uav_id": "uav_1",
                    "skill": "GOTO",
                    "args": {
                        "target": {
                            "kind": "POINT",
                            "frame": "WORLD_ENU",
                            "xyz_m": [8.0, 2.0, 10.0],
                        }
                    },
                },
                {
                    "id": "wait_1",
                    "uav_id": "uav_1",
                    "skill": "HOVER",
                    "args": {"duration_s": 2.0},
                },
                {
                    "id": "return_home",
                    "uav_id": "uav_1",
                    "skill": "GOTO",
                    "args": {
                        "target": {"kind": "NAMED_LOCATION", "name": "home"}
                    },
                },
                {
                    "id": "land_1",
                    "uav_id": "uav_1",
                    "skill": "LAND",
                    "args": {"zone": "home"},
                },
            ],
        }
    )


def _focused_wait_without_return_instruction() -> str:
    payload = json.loads(_v2_focused_instruction())
    payload["assigned_goals"] = [
        {
            "goal_id": "goal_wait",
            "goal_type": "WAIT",
            "duration_s": 2.0,
        }
    ]
    payload["trusted_runtime_safety_completion"] = True
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _wait_draft_without_return_or_land() -> SkillPlanDraftV3:
    data = _v2_draft_without_search_or_track().to_dict()
    data["steps"] = data["steps"][:3]
    return SkillPlanDraftV3.from_dict(data)


def _mission_agent(
    planner: RoutedPreplannedSpatialPlanner,
) -> tuple[MissionAgent, SkillManager]:
    context = _context()
    clock = _Clock()
    manager = SkillManager(
        SkillContext(
            uav=KinematicUAV(
                UAVState(0.0, 0.0, 0.0, 0.0),
                max_speed_mps=5.0,
                max_yaw_rate_rad_s=2.0,
            ),
            camera=_Camera(),
            perception=None,
            clock=clock,
            uav_id="uav_1",
        ),
        registry=create_default_skill_registry(),
    )
    agent = MissionAgent(
        planner=planner,
        validator=PlanValidator(),
        safety=SafetySupervisor(
            context.scene_min_xyz_m,
            context.scene_max_xyz_m,
            max_mission_time_s=300.0,
            max_safe_altitude_m=25.0,
        ),
        skill_manager=manager,
        target_manager=TargetManager(),
        clock=clock,
    )
    return agent, manager


def test_exact_precomputed_v2_request_starts_arbitrary_valid_skill_plan() -> None:
    focused = _v2_focused_instruction()
    draft = _v2_draft_without_search_or_track()
    planner = RoutedPreplannedSpatialPlanner(
        draft,
        source="dynamic_llm",
        expected_instruction=focused,
    )
    agent, manager = _mission_agent(planner)
    environment = _Environment()
    request = _request()
    runtime = FleetMissionRuntime(
        environment,
        ScriptedFleetPlanner(),
        {"uav_1": agent},
        inventory=request.uav_inventory,
        target_requests=request.target_requests,
        coordination_policy=request.coordination_policy,
        precomputed_start_inputs={"uav_1": (focused, _context())},
    )

    runtime.start(request.original_instruction, request=request)

    assert environment.started == 1
    assert runtime.status is FleetStatus.RUNNING
    assert agent.snapshot().status is AgentStatus.RUNNING
    assert manager.task_plan is not None
    assert [
        step.skill for step in manager.task_plan.steps
    ] == [
        SkillName.TAKEOFF,
        SkillName.GOTO,
        SkillName.HOVER,
        SkillName.GOTO,
        SkillName.LAND,
    ]
    assert all(
        step.skill not in {SkillName.SEARCH, SkillName.TRACK}
        for step in manager.task_plan.steps
    )


def test_preplanned_replay_preserves_trusted_runtime_safety_completion() -> None:
    focused = _focused_wait_without_return_instruction()
    planner = RoutedPreplannedSpatialPlanner(
        _wait_draft_without_return_or_land(),
        source="dynamic_llm",
        expected_instruction=focused,
    )
    assert planner.allow_trusted_safety_completion is True
    agent, manager = _mission_agent(planner)

    compiled = agent.start(focused, _context())

    assert tuple(step.skill for step in compiled.planner_output.steps) == (
        "TAKEOFF",
        "GOTO",
        "HOVER",
    )
    assert [step.skill for step in manager.task_plan.steps] == [
        SkillName.TAKEOFF,
        SkillName.GOTO,
        SkillName.HOVER,
        SkillName.GOTO,
        SkillName.LAND,
    ]
    assert manager.task_plan.steps[-2].step_id.startswith("trusted_return_home")
    assert manager.task_plan.steps[-1].step_id.startswith("trusted_land")


def test_v2_preplanned_replay_rejects_any_instruction_tampering() -> None:
    focused = _v2_focused_instruction()
    planner = RoutedPreplannedSpatialPlanner(
        _v2_draft_without_search_or_track(),
        source="dynamic_llm",
        expected_instruction=focused,
    )
    tampered_payload = json.loads(focused)
    tampered_payload["assigned_goals"][0]["goal_id"] = "goal_intruder"
    tampered = json.dumps(
        tampered_payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    request = PlannerRequest(
        instruction=tampered,
        world_context=_context(),
        mission_id="mission_runtime_replay",
        uav_id="uav_1",
        plan_version=1,
    )

    with pytest.raises(
        ValueError,
        match="differs from the prevalidated V2 request",
    ):
        planner.plan(request)


@pytest.mark.parametrize(
    "value",
    [
        ["focused", None],
        ("focused",),
        ("", None),
        (object(), None),
    ],
)
def test_precomputed_start_input_rejects_invalid_tuple(value: object) -> None:
    request = _request()
    planner = RoutedPreplannedSpatialPlanner(
        _v2_draft_without_search_or_track(),
        source="dynamic_scripted",
        expected_instruction=_v2_focused_instruction(),
    )
    with pytest.raises(TypeError, match="precomputed_start_inputs values"):
        FleetMissionRuntime(
            _Environment(),
            ScriptedFleetPlanner(),
            {"uav_1": _mission_agent(planner)[0]},
            inventory=request.uav_inventory,
            target_requests=request.target_requests,
            precomputed_start_inputs={"uav_1": value},  # type: ignore[dict-item]
        )


def test_precomputed_start_input_rejects_unknown_uav() -> None:
    request = _request()
    planner = RoutedPreplannedSpatialPlanner(
        _v2_draft_without_search_or_track(),
        source="dynamic_scripted",
        expected_instruction=_v2_focused_instruction(),
    )
    with pytest.raises(FleetRuntimeError, match="contains an unknown UAV"):
        FleetMissionRuntime(
            _Environment(),
            ScriptedFleetPlanner(),
            {"uav_1": _mission_agent(planner)[0]},
            inventory=request.uav_inventory,
            target_requests=request.target_requests,
            precomputed_start_inputs={"uav_unknown": ("focused", None)},
        )


def test_v1_local_replay_contract_remains_compatible() -> None:
    region = {
        "shape": "CIRCLE",
        "frame": "WORLD_ENU",
        "center_xyz_m": [5.0, 5.0, 0.0],
        "radius_m": 4.0,
    }
    draft = SkillPlanDraftV3.from_dict(
        {
            "schema_version": 3,
            "mission_id": "mission_v1_replay",
            "uav_id": "uav_1",
            "plan_version": 1,
            "assumptions": [],
            "steps": [
                {
                    "id": "takeoff_1",
                    "uav_id": "uav_1",
                    "skill": "TAKEOFF",
                    "args": {"altitude_m": 10.0},
                },
                {
                    "id": "search_1",
                    "uav_id": "uav_1",
                    "skill": "SEARCH",
                    "args": {
                        "region": region,
                        "strategy": {"kind": "SPIRAL_OUT", "spacing_m": 2.0},
                        "entry_policy": "START_IN_PLACE_IF_INSIDE",
                        "target_description": "moving target",
                        "search_altitude_m": 10.0,
                        "timeout_s": 30.0,
                    },
                },
                {
                    "id": "track_1",
                    "uav_id": "uav_1",
                    "skill": "TRACK",
                    "args": {
                        "target_ref": "$search_1.target_id",
                        "duration_s": 5.0,
                    },
                },
                {
                    "id": "return_home",
                    "uav_id": "uav_1",
                    "skill": "GOTO",
                    "args": {
                        "target": {"kind": "NAMED_LOCATION", "name": "home"}
                    },
                },
                {
                    "id": "land_1",
                    "uav_id": "uav_1",
                    "skill": "LAND",
                    "args": {"zone": "home"},
                },
            ],
        }
    )
    planner = RoutedPreplannedSpatialPlanner(
        draft,
        source="dynamic_scripted",
    )
    focused = json.dumps(
        {
            "target_spec": None,
            "search_region": region,
            "track_duration_s": 5.0,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    replayed = planner.plan(
        PlannerRequest(
            instruction=focused,
            world_context=_context(),
            mission_id="mission_v1_runtime",
            uav_id="uav_1",
            plan_version=3,
        )
    )

    assert replayed.mission_id == "mission_v1_runtime"
    assert replayed.plan_version == 3
    assert [step.skill for step in replayed.steps].count("SEARCH") == 1
    assert [step.skill for step in replayed.steps].count("TRACK") == 1
