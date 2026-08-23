"""Independent Spatial Contract V3 compilation and safety tests."""

from __future__ import annotations

import math
import unittest

from planner.schemas import (
    LandingZoneSpec,
    PlannerWorldContext,
    SearchRegionSpec,
)
from planner.schemas_v3 import SkillPlanDraftV3
from planner.spatial_resolver import FramePose, SpatialResolver
from runtime.plan_validator import PlanValidationError, PlanValidator
from runtime.safety_supervisor import SafetyAction, SafetySupervisor
from skills.motion_types import MotionPolicy
from skills.plan import TaskPlan, TaskStep
from skills.search import SearchGoalV3
from skills.search_strategy import SearchRuntimeCapabilities
from skills.types import SkillName


def _context() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
        initial_uav_xyz_m=(0.0, 0.0, 0.0),
        search_regions={
            "legacy_area": SearchRegionSpec(
                "legacy_area", (15, 10, 0), 5, (8, 10, 10)
            )
        },
        landing_zones={"home": LandingZoneSpec("home", (0, 0), 0)},
        default_takeoff_altitude_m=10,
        default_track_duration_s=20,
        search_timeout_s=75,
        goto_timeout_s=120,
        land_timeout_s=60,
    )


def _takeoff() -> dict[str, object]:
    return {"id": "takeoff_1", "uav_id": "uav_1", "skill": "TAKEOFF", "args": {"altitude_m": 10}}


def _land() -> dict[str, object]:
    return {"id": "land_1", "uav_id": "uav_1", "skill": "LAND", "args": {"zone": "home"}}


def _goto_named(step_id: str = "goto_home", name: str = "home") -> dict[str, object]:
    return {"id": step_id, "uav_id": "uav_1", "skill": "GOTO", "args": {"target": {"kind": "NAMED_LOCATION", "name": name}}}


def _search(step_id: str, *, center: tuple[float, float] = (10, 10), frame: str = "WORLD_ENU") -> dict[str, object]:
    return {
        "id": step_id,
        "uav_id": "uav_1",
        "skill": "SEARCH",
        "args": {
            "region": {
                "shape": "RECTANGLE",
                "frame": frame,
                "center_xyz_m": [center[0], center[1], 0],
                "width_m": 10,
                "height_m": 8,
            },
            "strategy": {"kind": "LAWNMOWER", "spacing_m": 3},
            "entry_policy": "START_IN_PLACE_IF_INSIDE",
            "target_description": "moving person",
            "search_altitude_m": 10,
            "timeout_s": 30,
        },
    }


def _draft(steps: list[dict[str, object]]) -> SkillPlanDraftV3:
    return SkillPlanDraftV3.from_dict(
        {
            "schema_version": 3,
            "mission_id": "mission_v3",
            "uav_id": "uav_1",
            "plan_version": 1,
            "assumptions": [],
            "steps": steps,
        }
    )


def _compile(
    steps: list[dict[str, object]],
    *,
    resolver: SpatialResolver | None = None,
    search_runtime_capabilities: SearchRuntimeCapabilities | None = None,
):
    return PlanValidator(
        spatial_resolver=resolver,
        search_runtime_capabilities=search_runtime_capabilities,
    ).validate_and_compile(
        _draft(steps),
        _context(),
        source="dynamic_scripted",
        mission_id="mission_v3",
        uav_id="uav_1",
        plan_version=1,
    )


class PlanValidatorV3Tests(unittest.TestCase):
    def test_point_target_compiles_to_single_goto(self) -> None:
        compiled = _compile(
            [
                _takeoff(),
                {
                    "id": "goto_point",
                    "uav_id": "uav_1",
                    "skill": "GOTO",
                    "args": {
                        "target": {
                            "kind": "POINT",
                            "frame": "WORLD_ENU",
                            "xyz_m": [0, 0, 10],
                        }
                    },
                },
                _land(),
            ]
        )
        goto = compiled.task_plan.steps[1]
        self.assertIs(goto.skill, SkillName.GOTO)
        self.assertEqual(goto.params["position"], (0.0, 0.0, 10.0))
        self.assertEqual(len([s for s in compiled.task_plan.steps if s.skill is SkillName.GOTO]), 1)

    def test_named_goto_hover_land_is_valid(self) -> None:
        compiled = _compile(
            [
                _takeoff(),
                _goto_named(),
                {"id": "hover_1", "uav_id": "uav_1", "skill": "HOVER", "args": {"duration_s": 5}},
                _land(),
            ]
        )
        self.assertEqual(
            [step.skill for step in compiled.task_plan.steps],
            [SkillName.TAKEOFF, SkillName.GOTO, SkillName.HOVER, SkillName.LAND],
        )
        self.assertIs(
            SafetySupervisor((-50, -50, 0), (50, 50, 30)).preflight(compiled).action,
            SafetyAction.CONTINUE,
        )

    def test_relational_target_requires_and_uses_trusted_resolver(self) -> None:
        steps = [
            _takeoff(),
            {
                "id": "goto_left",
                "uav_id": "uav_1",
                "skill": "GOTO",
                "args": {
                    "target": {
                        "kind": "RELATIONAL_POINT",
                        "relation": "LEFT_OF",
                        "reference_id": "tower",
                        "distance_m": 5,
                        "frame": "UAV_START_FLU",
                    },
                    "altitude_m": 10,
                },
            },
            _goto_named(),
            _land(),
        ]
        with self.assertRaisesRegex(PlanValidationError, "SpatialResolver"):
            _compile(steps)
        resolver = SpatialResolver(
            home_pose=FramePose((0, 0, 0)),
            uav_start_pose=FramePose((0, 0, 0), math.pi / 2),
            landmarks={"tower": (20, 20, 0)},
        )
        compiled = _compile(steps, resolver=resolver)
        self.assertAlmostEqual(compiled.task_plan.steps[1].params["position"][0], 15)
        self.assertAlmostEqual(compiled.task_plan.steps[1].params["position"][1], 20)

    def test_non_world_point_and_region_use_injected_frame_snapshot(self) -> None:
        resolver = SpatialResolver(
            home_pose=FramePose((5, 5, 0)),
            uav_start_pose=FramePose((10, 0, 0), math.pi / 2),
        )
        compiled = _compile(
            [
                _takeoff(),
                {
                    "id": "goto_relative",
                    "uav_id": "uav_1",
                    "skill": "GOTO",
                    "args": {"target": {"kind": "POINT", "frame": "UAV_START_FLU", "xyz_m": [2, 0, 10]}},
                },
                _search("search_home", center=(5, 5), frame="HOME_ENU"),
                _goto_named(),
                _land(),
            ],
            resolver=resolver,
        )
        self.assertAlmostEqual(compiled.task_plan.steps[1].params["position"][0], 10)
        self.assertAlmostEqual(compiled.task_plan.steps[1].params["position"][1], 2)
        search_params = compiled.task_plan.steps[2].params
        self.assertEqual(search_params["region"].center_xyz_m[:2], (10.0, 10.0))

    def test_route_target_is_rejected_and_requires_follow_route(self) -> None:
        with self.assertRaisesRegex(PlanValidationError, "FOLLOW_ROUTE"):
            _compile(
                [
                    _takeoff(),
                    {
                        "id": "goto_route",
                        "uav_id": "uav_1",
                        "skill": "GOTO",
                        "args": {"target": {"kind": "ROUTE", "frame": "WORLD_ENU", "waypoints_xyz_m": [[1, 0, 10], [2, 0, 10]]}},
                    },
                    _goto_named(),
                    _land(),
                ]
            )

    def test_multiple_searches_need_no_preceding_goto(self) -> None:
        compiled = _compile(
            [
                _takeoff(),
                _search("search_1", center=(12, 10)),
                _search("search_2", center=(-12, -10)),
                _goto_named(),
                _land(),
            ]
        )
        searches = [step for step in compiled.task_plan.steps if step.skill is SkillName.SEARCH]
        self.assertEqual(len(searches), 2)
        self.assertTrue(all("region" in step.params for step in searches))
        self.assertTrue(all(isinstance(SearchGoalV3(**dict(step.params)), SearchGoalV3) for step in searches))
        self.assertIs(
            SafetySupervisor((-50, -50, 0), (50, 50, 30)).preflight(compiled).action,
            SafetyAction.CONTINUE,
        )

    def test_adaptive_search_requires_matching_compile_and_safety_capability(self) -> None:
        adaptive = _search("search_adaptive", center=(0, 0))
        adaptive["args"]["strategy"] = {  # type: ignore[index]
            "kind": "ADAPTIVE_NEXT_BEST_VIEW",
            "max_viewpoints": 4,
        }
        steps = [_takeoff(), adaptive, _goto_named(), _land()]
        with self.assertRaisesRegex(PlanValidationError, "next-best-view provider"):
            _compile(steps)

        capabilities = SearchRuntimeCapabilities(adaptive_next_best_view=True)
        compiled = _compile(
            steps,
            search_runtime_capabilities=capabilities,
        )
        self.assertIs(
            SafetySupervisor((-50, -50, 0), (50, 50, 30)).preflight(
                compiled
            ).action,
            SafetyAction.ABORT,
        )
        self.assertIs(
            SafetySupervisor(
                (-50, -50, 0),
                (50, 50, 30),
                search_runtime_capabilities=capabilities,
            ).preflight(compiled).action,
            SafetyAction.CONTINUE,
        )

    def test_track_reference_can_select_either_prior_search(self) -> None:
        track = {
            "id": "track_1",
            "uav_id": "uav_1",
            "skill": "TRACK",
            "args": {"target_ref": "$search_1.target_id", "duration_s": 10},
        }
        compiled = _compile(
            [_takeoff(), _search("search_1"), _search("search_2", center=(-10, -10)), track, _goto_named(), _land()]
        )
        self.assertEqual(compiled.task_plan.steps[3].params["target_id"].step_id, "search_1")

    def test_land_after_position_uncertain_search_is_rejected(self) -> None:
        with self.assertRaisesRegex(PlanValidationError, "not guaranteed"):
            _compile([_takeoff(), _search("search_1", center=(0, 0)), _land()])

    def test_point_altitude_disagreement_and_routing_mismatch_fail_closed(self) -> None:
        steps = [
            _takeoff(),
            {
                "id": "goto_point",
                "uav_id": "uav_1",
                "skill": "GOTO",
                "args": {"target": {"kind": "POINT", "frame": "WORLD_ENU", "xyz_m": [0, 0, 10]}, "altitude_m": 12},
            },
            _land(),
        ]
        with self.assertRaisesRegex(PlanValidationError, "disagree"):
            _compile(steps)
        with self.assertRaisesRegex(PlanValidationError, "routing IDs"):
            PlanValidator().validate_and_compile(
                _draft([_takeoff(), _goto_named(), _land()]),
                _context(),
                source="dynamic_scripted",
                mission_id="mission_other",
                uav_id="uav_1",
                plan_version=1,
            )

    def test_safety_recognizes_follow_route_reference_and_resolved_geometry(self) -> None:
        base = _compile([_takeoff(), _goto_named(), _land()]).task_plan
        raw_route = TaskStep(
            "follow_1",
            SkillName.FOLLOW_ROUTE,
            {"route_ref": "route_1", "tolerance_m": 0.5, "timeout_s": 30},
        )
        plan = TaskPlan(
            (base.steps[0], raw_route, base.steps[1], base.steps[2]),
            mission_id=base.mission_id,
            uav_id=base.uav_id,
            plan_version=base.plan_version,
        )
        supervisor = SafetySupervisor((-50, -50, 0), (50, 50, 30))
        self.assertIs(supervisor.preflight(plan).action, SafetyAction.CONTINUE)

        resolved = TaskStep(
            "follow_2",
            SkillName.FOLLOW_ROUTE,
            {
                "route_id": "route_2",
                "waypoints": ((1, 0, 10), (100, 0, 10)),
                "motion_policy": MotionPolicy(max_speed=2),
            },
        )
        bad = TaskPlan(
            (base.steps[0], resolved, base.steps[1], base.steps[2]),
            mission_id=base.mission_id,
            uav_id=base.uav_id,
            plan_version=base.plan_version,
        )
        decision = supervisor.preflight(bad)
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("waypoints[1]", decision.reason)


if __name__ == "__main__":
    unittest.main()
