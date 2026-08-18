"""Pure-Python tests for deterministic mission and observation safety checks."""

from __future__ import annotations

import unittest

import numpy as np

from env.kinematic_uav import UAVState
from planner.schemas import (
    LandingZoneSpec,
    MissionIntent,
    SkillPlanDraft,
    PlannerWorldContext,
    SearchRegionSpec,
)
from runtime.plan_validator import PlannerLimits, PlanValidator
from runtime.safety_supervisor import SafetyAction, SafetySupervisor
from skills.manager import TaskPlan, TaskStep
from skills.plan import RecoveryPolicy, StepOutputRef
from skills.types import Observation, SkillName


class SafetySupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        context = PlannerWorldContext(
            scene_min_xyz_m=(-50.0, -50.0, 0.0),
            scene_max_xyz_m=(50.0, 50.0, 30.0),
            initial_uav_xyz_m=(0.0, 0.0, 0.0),
            search_regions={
                "search_area": SearchRegionSpec(
                    name="search_area",
                    center_xyz_m=(20.0, 20.0, 0.0),
                    radius_m=10.0,
                    approach_xyz_m=(20.0, 5.0, 10.0),
                )
            },
            landing_zones={
                "home": LandingZoneSpec(
                    name="home",
                    position_xy_m=(0.0, 0.0),
                    ground_altitude_m=0.0,
                )
            },
            default_takeoff_altitude_m=10.0,
            default_track_duration_s=30.0,
            search_timeout_s=75.0,
        )
        self.context = context
        intent = MissionIntent(
            target_description="moving target",
            search_region="search_area",
            track_duration_s=30.0,
            landing_zone="home",
        )
        self.compiled = PlanValidator().validate_and_compile(
            intent,
            context,
            source="scripted",
        )
        self.supervisor = SafetySupervisor(
            scene_min_xyz_m=(-50.0, -50.0, 0.0),
            scene_max_xyz_m=(50.0, 50.0, 30.0),
            max_mission_time_s=120.0,
            position_margin_m=0.1,
            max_safe_altitude_m=25.0,
        )

    def test_legal_preflight(self) -> None:
        decision = self.supervisor.preflight(self.compiled)
        self.assertIs(decision.action, SafetyAction.CONTINUE)
        self.assertTrue(decision.reason)

    def test_preflight_accepts_bare_task_plan(self) -> None:
        decision = self.supervisor.preflight(self.compiled.task_plan)
        self.assertIs(decision.action, SafetyAction.CONTINUE)

    def test_landing_zone_geometry_is_validated_when_present(self) -> None:
        land = self.compiled.task_plan.steps[-1]
        land.params["expected_position_xy"] = (0.0, 0.0)
        land.params["zone_tolerance_m"] = 0.75
        self.assertIs(
            self.supervisor.preflight(self.compiled).action,
            SafetyAction.CONTINUE,
        )

        land.params["expected_position_xy"] = (51.0, 0.0)
        decision = self.supervisor.preflight(self.compiled)
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("expected_position_xy", decision.reason)

        land.params["expected_position_xy"] = None
        land.params["zone_tolerance_m"] = 0.0
        decision = self.supervisor.preflight(self.compiled)
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("zone_tolerance_m", decision.reason)

    def test_dynamic_plan_receives_independent_preflight_validation(self) -> None:
        draft = SkillPlanDraft.from_dict(
            {
                "schema_version": 1,
                "steps": [
                    {"id": "takeoff_1", "skill": "TAKEOFF", "args": {"altitude_m": 10}},
                    {"id": "goto_search", "skill": "GOTO", "args": {"destination": "search_area"}},
                    {
                        "id": "search_1",
                        "skill": "SEARCH",
                        "args": {"region": "search_area", "target_description": "moving target"},
                    },
                    {
                        "id": "track_1",
                        "skill": "TRACK",
                        "args": {"target_ref": "$search_1.target_id", "duration_s": 10},
                        "recovery": {
                            "skill": "REACQUIRE",
                            "max_attempts": 2,
                            "search_radius_m": 10,
                            "timeout_s": 30,
                        },
                    },
                    {"id": "goto_home", "skill": "GOTO", "args": {"destination": "home"}},
                    {"id": "land_1", "skill": "LAND", "args": {"zone": "home"}},
                ],
            }
        )
        compiled = PlanValidator().validate_and_compile(
            draft,
            self.context,
            source="dynamic_scripted",
        )
        decision = self.supervisor.preflight(compiled)
        self.assertIs(decision.action, SafetyAction.CONTINUE)

    def test_dynamic_structure_guards_are_defense_in_depth(self) -> None:
        steps = self.compiled.task_plan.steps
        cases = (
            (steps[1:], "TAKEOFF"),
            (steps[:-1], "LAND"),
            (
                (
                    steps[0],
                    TaskStep(
                        "track_early",
                        SkillName.TRACK,
                        {
                            "target_id": StepOutputRef("search_later"),
                            "track_duration": 10.0,
                        },
                    ),
                    TaskStep(
                        "search_later",
                        SkillName.SEARCH,
                        {
                            "center": (20.0, 20.0, 0.0),
                            "radius": 10.0,
                            "target_description": "target",
                            "search_altitude": 10.0,
                            "timeout": 30.0,
                        },
                    ),
                    steps[-1],
                ),
                "after SEARCH",
            ),
        )
        for damaged_steps, reason in cases:
            with self.subTest(reason=reason):
                decision = self.supervisor.preflight(_unsafe_plan(tuple(damaged_steps)))
                self.assertIs(decision.action, SafetyAction.ABORT)
                self.assertIn(reason, decision.reason)

        duplicate = list(steps)
        object.__setattr__(duplicate[1], "step_id", duplicate[0].step_id)
        try:
            decision = self.supervisor.preflight(_unsafe_plan(tuple(duplicate)))
            self.assertIs(decision.action, SafetyAction.ABORT)
            self.assertIn("unique", decision.reason)
        finally:
            object.__setattr__(duplicate[1], "step_id", "step_02")

    def test_step_output_ref_must_point_backward_to_search(self) -> None:
        steps = list(self.compiled.task_plan.steps)
        steps[3] = TaskStep(
            "track_bad_ref",
            SkillName.TRACK,
            {
                **steps[3].params,
                "target_id": StepOutputRef(steps[1].step_id),
            },
        )
        decision = self.supervisor.preflight(_unsafe_plan(tuple(steps)))
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("SEARCH", decision.reason)

        direct_target = list(self.compiled.task_plan.steps)
        direct_target[3] = TaskStep(
            "track_direct_truth",
            SkillName.TRACK,
            {**direct_target[3].params, "target_id": "oracle_target_0"},
        )
        decision = self.supervisor.preflight(_unsafe_plan(tuple(direct_target)))
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("StepOutputRef", decision.reason)

    def test_recovery_budget_is_rechecked_by_safety(self) -> None:
        limits = PlannerLimits(max_total_reacquire_attempts=3)
        supervisor = SafetySupervisor(
            (-50.0, -50.0, 0.0),
            (50.0, 50.0, 30.0),
            planner_limits=limits,
        )
        recovery = RecoveryPolicy(SkillName.REACQUIRE, 2, 10.0, 30.0)
        base = list(self.compiled.task_plan.steps)
        first_track = TaskStep(
            "track_one",
            SkillName.TRACK,
            base[3].params,
            recovery,
        )
        second_track = TaskStep(
            "track_two",
            SkillName.TRACK,
            base[3].params,
            recovery,
        )
        dynamic_steps = (
            base[0],
            base[1],
            base[2],
            first_track,
            second_track,
            base[4],
            base[5],
        )
        decision = supervisor.preflight(_unsafe_plan(dynamic_steps))
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("total recovery", decision.reason)

    def test_unknown_compiled_parameter_aborts(self) -> None:
        plan = _copy_plan(self.compiled.task_plan)
        plan.steps[1].params["motor_value"] = 0.5
        decision = self.supervisor.preflight(plan)
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("unknown compiled params", decision.reason)

    def test_track_timeout_none_is_valid(self) -> None:
        plan = _copy_plan(self.compiled.task_plan)
        plan.steps[3].params["timeout"] = None
        decision = self.supervisor.preflight(plan)
        self.assertIs(decision.action, SafetyAction.CONTINUE)

    def test_damaged_compiled_mission_aborts_instead_of_raising(self) -> None:
        damaged = object.__new__(type(self.compiled))
        decision = self.supervisor.preflight(damaged)
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("structurally invalid", decision.reason)

    def test_task_plan_missing_steps_aborts_instead_of_raising(self) -> None:
        damaged = object.__new__(TaskPlan)
        decision = self.supervisor.preflight(damaged)
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("structurally invalid", decision.reason)

    def test_task_step_missing_fields_aborts_instead_of_raising(self) -> None:
        damaged_step = object.__new__(TaskStep)
        damaged_plan = _unsafe_plan((damaged_step,))
        decision = self.supervisor.preflight(damaged_plan)
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("step 0", decision.reason)

    def test_plan_missing_land_aborts(self) -> None:
        plan = _unsafe_plan(self.compiled.task_plan.steps[:-1])
        decision = self.supervisor.preflight(plan)
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("LAND", decision.reason)

    def test_explicit_reacquire_aborts(self) -> None:
        steps = list(self.compiled.task_plan.steps)
        # The public TaskStep constructor already rejects this.  Corrupt a
        # value deliberately to verify SafetySupervisor's independent guard.
        damaged_step = object.__new__(TaskStep)
        object.__setattr__(damaged_step, "step_id", "bad_reacquire")
        object.__setattr__(damaged_step, "skill", SkillName.REACQUIRE)
        object.__setattr__(
            damaged_step,
            "params",
            {
                "target_id": "target_0",
                "last_seen_position": (0.0, 0.0, 0.0),
                "last_seen_velocity": (0.0, 0.0, 0.0),
                "last_seen_time": 1.0,
            },
        )
        object.__setattr__(damaged_step, "recovery", None)
        steps[2] = damaged_step
        decision = self.supervisor.preflight(_unsafe_plan(tuple(steps)))
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("REACQUIRE", decision.reason)

    def test_out_of_bounds_goal_aborts(self) -> None:
        plan = _copy_plan(self.compiled.task_plan)
        plan.steps[1].params["position"] = [51.0, 0.0, 10.0]
        decision = self.supervisor.preflight(plan)
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("outside", decision.reason)

    def test_search_radius_leaving_scene_aborts(self) -> None:
        plan = _copy_plan(self.compiled.task_plan)
        plan.steps[2].params["center"] = [45.0, 0.0, 0.0]
        plan.steps[2].params["radius"] = 10.0
        decision = self.supervisor.preflight(plan)
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("radius", decision.reason)

    def test_nonfinite_plan_value_aborts(self) -> None:
        plan = _copy_plan(self.compiled.task_plan)
        plan.steps[1].params["timeout"] = float("inf")
        decision = self.supervisor.preflight(plan)
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("finite", decision.reason)

    def test_invalid_takeoff_and_track_values_abort(self) -> None:
        takeoff = _copy_plan(self.compiled.task_plan)
        takeoff.steps[0].params["target_altitude"] = 26.0
        self.assertIs(
            self.supervisor.preflight(takeoff).action,
            SafetyAction.ABORT,
        )

        track = _copy_plan(self.compiled.task_plan)
        track.steps[3].params["track_duration"] = 600.1
        self.assertIs(
            self.supervisor.preflight(track).action,
            SafetyAction.ABORT,
        )

        distance = _copy_plan(self.compiled.task_plan)
        distance.steps[3].params["desired_distance"] = 1e308
        self.assertIs(
            self.supervisor.preflight(distance).action,
            SafetyAction.ABORT,
        )

        yaw = _copy_plan(self.compiled.task_plan)
        yaw.steps[0].params["yaw_mode"] = "KEEP_CURRENT"
        yaw.steps[0].params["yaw_value"] = 0.5
        self.assertIs(
            self.supervisor.preflight(yaw).action,
            SafetyAction.ABORT,
        )

    def test_invalid_optional_skill_parameter_aborts(self) -> None:
        plan = _copy_plan(self.compiled.task_plan)
        plan.steps[1].params["tolerance"] = -1.0
        self.assertIs(
            self.supervisor.preflight(plan).action,
            SafetyAction.ABORT,
        )

    def test_invalid_motion_policy_aborts(self) -> None:
        plan = _copy_plan(self.compiled.task_plan)
        plan.steps[1].params["motion_policy"] = {
            "yaw_mode": "FIXED",
            "yaw_value": None,
        }
        self.assertIs(
            self.supervisor.preflight(plan).action,
            SafetyAction.ABORT,
        )

    def test_face_point_outside_scene_aborts(self) -> None:
        plan = _copy_plan(self.compiled.task_plan)
        plan.steps[1].params["motion_policy"] = {
            "yaw_mode": "FACE_POINT",
            "look_at_point": (51.0, 0.0, 0.0),
        }
        decision = self.supervisor.preflight(plan)
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("look_at_point", decision.reason)
        self.assertIn("outside", decision.reason)

    def test_cyclic_plan_parameter_aborts_instead_of_recursing(self) -> None:
        plan = _copy_plan(self.compiled.task_plan)
        cyclic: dict[str, object] = {}
        cyclic["look_at_point"] = cyclic
        plan.steps[1].params["motion_policy"] = cyclic
        decision = self.supervisor.preflight(plan)
        self.assertIs(decision.action, SafetyAction.ABORT)
        self.assertIn("cyclic", decision.reason)

    def test_normal_observation_continues(self) -> None:
        decision = self.supervisor.evaluate(
            _observation(timestamp=1.0),
            mission_elapsed_s=1.0,
        )
        self.assertIs(decision.action, SafetyAction.CONTINUE)

    def test_uav_outside_scene_requests_landing(self) -> None:
        decision = self.supervisor.evaluate(
            _observation(timestamp=1.0, pose=UAVState(51.0, 0.0, 10.0, 0.0)),
            mission_elapsed_s=1.0,
        )
        self.assertIs(decision.action, SafetyAction.CANCEL_AND_LAND)

    def test_position_margin_is_runtime_tolerance(self) -> None:
        decision = self.supervisor.evaluate(
            _observation(timestamp=1.0, pose=UAVState(50.05, 0.0, 10.0, 0.0)),
            mission_elapsed_s=1.0,
        )
        self.assertIs(decision.action, SafetyAction.CONTINUE)

    def test_uav_above_safe_altitude_requests_landing(self) -> None:
        decision = self.supervisor.evaluate(
            _observation(timestamp=1.0, pose=UAVState(0.0, 0.0, 25.1, 0.0)),
            mission_elapsed_s=1.0,
        )
        self.assertIs(decision.action, SafetyAction.CANCEL_AND_LAND)

    def test_nan_pose_aborts(self) -> None:
        decision = self.supervisor.evaluate(
            _observation(
                timestamp=1.0,
                pose=UAVState(float("nan"), 0.0, 10.0, 0.0),
            ),
            mission_elapsed_s=1.0,
        )
        self.assertIs(decision.action, SafetyAction.ABORT)

    def test_nan_velocity_aborts(self) -> None:
        decision = self.supervisor.evaluate(
            _observation(
                timestamp=1.0,
                velocity=np.array([0.0, float("nan"), 0.0]),
            ),
            mission_elapsed_s=1.0,
        )
        self.assertIs(decision.action, SafetyAction.ABORT)

    def test_timestamp_rollback_aborts_without_rewinding_history(self) -> None:
        self.assertIs(
            self.supervisor.evaluate(
                _observation(timestamp=2.0),
                mission_elapsed_s=2.0,
            ).action,
            SafetyAction.CONTINUE,
        )
        self.assertIs(
            self.supervisor.evaluate(
                _observation(timestamp=1.0),
                mission_elapsed_s=3.0,
            ).action,
            SafetyAction.ABORT,
        )
        self.assertIs(
            self.supervisor.evaluate(
                _observation(timestamp=1.5),
                mission_elapsed_s=4.0,
            ).action,
            SafetyAction.ABORT,
        )
        self.assertIs(
            self.supervisor.evaluate(
                _observation(timestamp=3.0),
                mission_elapsed_s=5.0,
            ).action,
            SafetyAction.CONTINUE,
        )

    def test_elapsed_time_rollback_aborts_without_rewinding_history(self) -> None:
        self.assertIs(
            self.supervisor.evaluate(
                _observation(timestamp=1.0),
                mission_elapsed_s=10.0,
            ).action,
            SafetyAction.CONTINUE,
        )
        self.assertIs(
            self.supervisor.evaluate(
                _observation(timestamp=2.0),
                mission_elapsed_s=9.0,
            ).action,
            SafetyAction.ABORT,
        )
        self.assertIs(
            self.supervisor.evaluate(
                _observation(timestamp=3.0),
                mission_elapsed_s=9.5,
            ).action,
            SafetyAction.ABORT,
        )
        self.assertIs(
            self.supervisor.evaluate(
                _observation(timestamp=2.0),
                mission_elapsed_s=11.0,
            ).action,
            SafetyAction.CONTINUE,
        )

    def test_mission_timeout_requests_landing(self) -> None:
        decision = self.supervisor.evaluate(
            _observation(timestamp=121.0),
            mission_elapsed_s=120.01,
        )
        self.assertIs(decision.action, SafetyAction.CANCEL_AND_LAND)

    def test_equal_time_limit_and_timestamp_are_allowed(self) -> None:
        self.assertIs(
            self.supervisor.evaluate(
                _observation(timestamp=1.0),
                mission_elapsed_s=120.0,
            ).action,
            SafetyAction.CONTINUE,
        )
        self.assertIs(
            self.supervisor.evaluate(
                _observation(timestamp=1.0),
                mission_elapsed_s=120.0,
            ).action,
            SafetyAction.CONTINUE,
        )

    def test_invalid_elapsed_time_aborts(self) -> None:
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                decision = self.supervisor.evaluate(
                    _observation(timestamp=1.0),
                    mission_elapsed_s=value,
                )
                self.assertIs(decision.action, SafetyAction.ABORT)

    def test_constructor_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            SafetySupervisor((0.0, 0.0, 0.0), (0.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            SafetySupervisor(
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                position_margin_m=-0.1,
            )


def _observation(
    *,
    timestamp: float,
    pose: UAVState = UAVState(0.0, 0.0, 10.0, 0.0),
    velocity: np.ndarray | None = None,
) -> Observation:
    return Observation(
        uav_id="uav_1",
        timestamp=timestamp,
        uav_pose=pose,
        uav_velocity=(
            np.zeros(3, dtype=np.float64) if velocity is None else velocity
        ),
        camera_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
    )


def _copy_plan(plan: TaskPlan) -> TaskPlan:
    return TaskPlan.from_dicts(plan.to_dicts())


def _unsafe_plan(steps: tuple[TaskStep, ...]) -> TaskPlan:
    plan = object.__new__(TaskPlan)
    object.__setattr__(plan, "steps", steps)
    return plan


if __name__ == "__main__":
    unittest.main()
