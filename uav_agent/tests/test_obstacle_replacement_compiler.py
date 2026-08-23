"""Pure-Python tests for trusted obstacle suffix compilation."""

from __future__ import annotations

import unittest

from planner.obstacle_replacement_compiler import (
    ObstacleReplacementCompilationError,
    ObstacleReplacementCompiler,
    TrustedFollowRouteDefaults,
    compile_obstacle_replacement,
)
from planner.obstacle_revision import (
    ObstacleReplacementStep,
    ObstacleRouteRevisionDraft,
)
from planner.route_types import (
    AvoidanceStrategy,
    AvoidanceStrategyType,
    RouteDraft,
    RouteWaypoint,
)
from planner.spatial import CoordinateFrame
from planner.spatial_resolver import FramePose, SpatialResolver
from runtime.safety_supervisor import SafetyAction, SafetySupervisor
from skills.motion_types import MotionPolicy, YawMode
from skills.plan import RecoveryPolicy, StepOutputRef, TaskPlan, TaskStep
from skills.types import SkillName


def _original_plan() -> TaskPlan:
    return TaskPlan(
        (
            TaskStep(
                "takeoff_1",
                SkillName.TAKEOFF,
                {"target_altitude": 10.0, "timeout": 120.0},
            ),
            TaskStep(
                "goto_search",
                SkillName.GOTO,
                {
                    "position": (20.0, 12.0, 10.0),
                    "timeout": 120.0,
                    "motion_policy": MotionPolicy(max_speed=2.0),
                },
            ),
            TaskStep(
                "search_1",
                SkillName.SEARCH,
                {
                    "center": (20.0, 30.0, 0.0),
                    "radius": 15.0,
                    "target_description": "moving target",
                    "search_altitude": 10.0,
                    "timeout": 75.0,
                },
            ),
            TaskStep(
                "track_1",
                SkillName.TRACK,
                {
                    "target_id": StepOutputRef("search_1"),
                    "desired_distance": 6.0,
                    "desired_altitude": 10.0,
                    "track_duration": 15.0,
                    "timeout": 20.0,
                },
                RecoveryPolicy(SkillName.REACQUIRE, 2, 10.0, 30.0),
            ),
            TaskStep(
                "goto_home",
                SkillName.GOTO,
                {
                    "position": (0.0, 0.0, 10.0),
                    "tolerance": 0.75,
                    "timeout": 120.0,
                    "motion_policy": MotionPolicy(max_speed=2.0),
                },
            ),
            TaskStep(
                "land_1",
                SkillName.LAND,
                {
                    "ground_altitude": 0.0,
                    "expected_position_xy": (0.0, 0.0),
                    "zone_tolerance_m": 0.75,
                    "timeout": 60.0,
                },
            ),
        ),
        mission_id="mission_test",
        uav_id="uav_1",
        plan_version=1,
    )


def _resolver() -> SpatialResolver:
    return SpatialResolver(
        home_pose=FramePose((0.0, 0.0, 0.0)),
        uav_start_pose=FramePose((0.0, 0.0, 0.0)),
        uav_hold_pose=FramePose((5.0, 5.0, 10.0)),
        named_locations={"home": (0.0, 0.0, 0.0)},
    )


def _step(
    step_id: str,
    skill: SkillName | str,
    args: dict[str, object] | None = None,
    *,
    uav_id: str = "uav_1",
) -> ObstacleReplacementStep:
    name = skill.value if isinstance(skill, SkillName) else skill
    return ObstacleReplacementStep(
        step_id,
        uav_id,
        name,
        {} if args is None else args,
    )


def _draft(
    replace_from_step_id: str,
    continuations: tuple[ObstacleReplacementStep, ...],
    *,
    first_id: str = "follow_detour",
    first_args: dict[str, object] | None = None,
    mission_id: str = "mission_test",
    uav_id: str = "uav_1",
    base_plan_version: int = 1,
) -> ObstacleRouteRevisionDraft:
    route = RouteDraft(
        "route_1",
        CoordinateFrame.UAV_HOLD_FLU,
        (
            RouteWaypoint("wp_1", (2.0, 3.0, 0.0)),
            RouteWaypoint("wp_2", (8.0, 3.0, 0.0)),
            RouteWaypoint("wp_3", (10.0, 0.0, 0.0)),
        ),
    )
    return ObstacleRouteRevisionDraft(
        mission_id=mission_id,
        uav_id=uav_id,
        base_plan_version=base_plan_version,
        new_plan_version=base_plan_version + 1,
        replace_from_step_id=replace_from_step_id,
        avoidance_strategy=AvoidanceStrategy(
            AvoidanceStrategyType.BYPASS_LEFT,
            "original_goto_target",
            ("LEFT_CLEARANCE_VISIBLE",),
        ),
        route_draft=route,
        replacement_steps=(
            _step(
                first_id,
                SkillName.FOLLOW_ROUTE,
                {"route_ref": "route_1"} if first_args is None else first_args,
                uav_id=uav_id,
            ),
            *continuations,
        ),
    )


def _goto_chain_draft(**kwargs: object) -> ObstacleRouteRevisionDraft:
    uav_id = str(kwargs.get("uav_id", "uav_1"))
    return _draft(
        str(kwargs.pop("replace_from_step_id", "goto_search")),
        (
            _step("resume_search", SkillName.SEARCH, uav_id=uav_id),
            _step("track_target", SkillName.TRACK, uav_id=uav_id),
            _step("return_home", SkillName.GOTO, uav_id=uav_id),
            _step("land_home", SkillName.LAND, uav_id=uav_id),
        ),
        **kwargs,  # type: ignore[arg-type]
    )


class ObstacleReplacementCompilerTest(unittest.TestCase):
    def test_goto_obstacle_reconnects_search_track_return_land(self) -> None:
        original = _original_plan()
        result = compile_obstacle_replacement(
            _goto_chain_draft(),
            original,
            1,
        )

        self.assertEqual(result.mission_id, original.mission_id)
        self.assertEqual(result.uav_id, original.uav_id)
        self.assertEqual(result.plan_version, 2)
        self.assertIs(result.steps[0], original.steps[0])
        self.assertEqual(
            [step.step_id for step in result.steps],
            [
                "takeoff_1",
                "follow_detour",
                "resume_search",
                "track_target",
                "return_home",
                "land_home",
            ],
        )

        follow = result.steps[1]
        self.assertEqual(
            set(follow.params),
            {"route_ref", "tolerance_m", "timeout_s", "motion_policy"},
        )
        self.assertEqual(follow.params["route_ref"], "route_1")
        self.assertEqual(follow.params["tolerance_m"], 0.75)
        self.assertEqual(follow.params["timeout_s"], 120.0)
        policy = follow.params["motion_policy"]
        self.assertIsInstance(policy, MotionPolicy)
        self.assertEqual(policy.max_speed, 2.0)
        self.assertEqual(policy.max_yaw_rate, 1.0)
        self.assertIs(policy.yaw_mode, YawMode.COURSE_ALIGNED)

        self.assertEqual(result.steps[2].params, original.steps[2].params)
        self.assertIsNot(result.steps[2].params, original.steps[2].params)
        self.assertEqual(
            result.steps[3].params["target_id"],
            StepOutputRef("resume_search"),
        )
        self.assertEqual(result.steps[3].recovery, original.steps[3].recovery)
        self.assertEqual(result.steps[4].params, original.steps[4].params)
        self.assertEqual(result.steps[5].params, original.steps[5].params)
        self.assertEqual(
            original.steps[3].params["target_id"],
            StepOutputRef("search_1"),
        )
        safety = SafetySupervisor(
            scene_min_xyz_m=(-50.0, -50.0, 0.0),
            scene_max_xyz_m=(50.0, 50.0, 30.0),
            max_safe_altitude_m=30.0,
        ).preflight(result)
        self.assertIs(safety.action, SafetyAction.CONTINUE)

    def test_search_and_track_interruptions_resume_the_current_skill(self) -> None:
        original = _original_plan()
        cases = (
            (
                2,
                _draft(
                    "search_1",
                    (
                        _step("resume_search", SkillName.SEARCH),
                        _step("resume_track", SkillName.TRACK),
                        _step("return_home", SkillName.GOTO),
                        _step("land_home", SkillName.LAND),
                    ),
                ),
                "resume_search",
                StepOutputRef("resume_search"),
            ),
            (
                3,
                _draft(
                    "track_1",
                    (
                        _step("resume_track", SkillName.TRACK),
                        _step("return_home", SkillName.GOTO),
                        _step("land_home", SkillName.LAND),
                    ),
                ),
                "resume_track",
                StepOutputRef("search_1"),
            ),
        )
        for index, draft, resumed_id, expected_ref in cases:
            with self.subTest(interrupted_index=index):
                result = compile_obstacle_replacement(draft, original, index)
                self.assertEqual(result.steps[index + 1].step_id, resumed_id)
                track = next(
                    step
                    for step in result.steps[index:]
                    if step.skill is SkillName.TRACK
                )
                self.assertEqual(track.params["target_id"], expected_ref)
                self.assertEqual(result.steps[-1].skill, SkillName.LAND)

    def test_search_interruption_explicitly_restarts_trusted_search(self) -> None:
        original = _original_plan()
        draft = _draft(
            "search_1",
            (
                _step(
                    "restart_search",
                    SkillName.SEARCH,
                    {"target_continuation": "RESTART_SEARCH"},
                ),
                _step("track_after_search", SkillName.TRACK),
                _step("return_home", SkillName.GOTO),
                _step("land_home", SkillName.LAND),
            ),
        )

        result = compile_obstacle_replacement(draft, original, 2)

        self.assertEqual(result.plan_version, original.plan_version + 1)
        self.assertEqual(
            [step.skill for step in result.steps[2:]],
            [
                SkillName.FOLLOW_ROUTE,
                SkillName.SEARCH,
                SkillName.TRACK,
                SkillName.GOTO,
                SkillName.LAND,
            ],
        )
        restarted = result.steps[3]
        self.assertEqual(restarted.params, original.steps[2].params)
        self.assertNotIn("target_continuation", restarted.params)
        self.assertEqual(
            result.steps[4].params["target_id"],
            StepOutputRef("restart_search"),
        )

    def test_track_interruption_can_continue_track_or_select_reacquire_policy(self) -> None:
        original = _original_plan()
        for action in ("CONTINUE_TRACK", "REACQUIRE"):
            with self.subTest(action=action):
                draft = _draft(
                    "track_1",
                    (
                        _step(
                            "resume_track",
                            SkillName.TRACK,
                            {"target_continuation": action},
                        ),
                        _step("return_home", SkillName.GOTO),
                        _step("land_home", SkillName.LAND),
                    ),
                )

                result = compile_obstacle_replacement(draft, original, 3)

                resumed = result.steps[4]
                self.assertEqual(result.plan_version, 2)
                self.assertIs(resumed.skill, SkillName.TRACK)
                self.assertEqual(
                    resumed.params["target_id"],
                    StepOutputRef("search_1"),
                )
                self.assertNotIn("target_continuation", resumed.params)
                self.assertEqual(resumed.recovery, original.steps[3].recovery)
                self.assertNotIn(
                    SkillName.REACQUIRE,
                    [step.skill for step in result.steps],
                )

    def test_track_interruption_can_restart_its_trusted_source_search(self) -> None:
        original = _original_plan()
        draft = _draft(
            "track_1",
            (
                _step(
                    "search_again",
                    SkillName.SEARCH,
                    {"target_continuation": "RESTART_SEARCH"},
                ),
                _step("track_again", SkillName.TRACK),
                _step("return_home", SkillName.GOTO),
                _step("land_home", SkillName.LAND),
            ),
        )

        result = compile_obstacle_replacement(draft, original, 3)

        self.assertEqual(result.plan_version, 2)
        self.assertEqual(
            [step.skill for step in result.steps[3:]],
            [
                SkillName.FOLLOW_ROUTE,
                SkillName.SEARCH,
                SkillName.TRACK,
                SkillName.GOTO,
                SkillName.LAND,
            ],
        )
        self.assertEqual(result.steps[4].params, original.steps[2].params)
        self.assertEqual(
            result.steps[5].params["target_id"],
            StepOutputRef("search_again"),
        )

    def test_reacquire_remains_internal_and_requires_trusted_track_recovery(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "replacement skill is not supported",
        ):
            _step("bad_top_level", SkillName.REACQUIRE)

        original = _original_plan()
        no_recovery_steps = list(original.steps)
        track = no_recovery_steps[3]
        no_recovery_steps[3] = TaskStep(
            track.step_id,
            track.skill,
            track.params,
        )
        no_recovery_plan = TaskPlan(
            tuple(no_recovery_steps),
            mission_id=original.mission_id,
            uav_id=original.uav_id,
            plan_version=original.plan_version,
        )
        draft = _draft(
            "track_1",
            (
                _step(
                    "recover_track",
                    SkillName.TRACK,
                    {"target_continuation": "REACQUIRE"},
                ),
                _step("return_home", SkillName.GOTO),
                _step("land_home", SkillName.LAND),
            ),
        )

        with self.assertRaisesRegex(
            ObstacleReplacementCompilationError,
            "trusted bounded RecoveryPolicy",
        ):
            compile_obstacle_replacement(draft, no_recovery_plan, 3)

        delayed_action = _draft(
            "track_1",
            (
                _step("wait_first", SkillName.HOVER, {"duration_s": 1.0}),
                _step(
                    "late_recovery",
                    SkillName.TRACK,
                    {"target_continuation": "REACQUIRE"},
                ),
                _step("return_home", SkillName.GOTO),
                _step("land_home", SkillName.LAND),
            ),
        )
        with self.assertRaisesRegex(
            ObstacleReplacementCompilationError,
            "first step after FOLLOW_ROUTE",
        ):
            compile_obstacle_replacement(delayed_action, original, 3)

    def test_terminal_goto_may_be_replaced_by_the_accepted_route(self) -> None:
        original = _original_plan()
        draft = _draft(
            "goto_home",
            (_step("land_home", SkillName.LAND),),
        )

        result = ObstacleReplacementCompiler().compile(draft, original, 4)

        self.assertEqual(result.steps[-2].skill, SkillName.FOLLOW_ROUTE)
        self.assertEqual(result.steps[-1].skill, SkillName.LAND)
        self.assertEqual(result.steps[-1].params, original.steps[-1].params)

    def test_custom_follow_route_defaults_remain_bounded_and_trusted(self) -> None:
        defaults = TrustedFollowRouteDefaults(
            tolerance_m=0.5,
            timeout_s=90.0,
            max_speed_mps=1.25,
            max_yaw_rate_rad_s=0.75,
        )
        result = ObstacleReplacementCompiler(defaults).compile(
            _goto_chain_draft(),
            _original_plan(),
            1,
        )

        self.assertEqual(result.steps[1].params["tolerance_m"], 0.5)
        self.assertEqual(result.steps[1].params["timeout_s"], 90.0)
        policy = result.steps[1].params["motion_policy"]
        self.assertEqual(policy.max_speed, 1.25)
        with self.assertRaisesRegex(ValueError, "timeout_s"):
            TrustedFollowRouteDefaults(timeout_s=901.0)

    def test_rejects_routing_version_replace_id_and_prefix_id_changes(self) -> None:
        original = _original_plan()
        cases = (
            ("mission", _goto_chain_draft(mission_id="mission_other"), 1),
            ("base_plan_version", _goto_chain_draft(base_plan_version=2), 1),
            (
                "replace_from_step_id",
                _goto_chain_draft(replace_from_step_id="search_1"),
                1,
            ),
            ("duplicates", _goto_chain_draft(first_id="takeoff_1"), 1),
        )
        for error, draft, index in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(
                    ObstacleReplacementCompilationError,
                    error,
                ):
                    compile_obstacle_replacement(draft, original, index)

        other_uav = _goto_chain_draft(uav_id="uav_2")
        with self.assertRaisesRegex(
            ObstacleReplacementCompilationError,
            "uav_id",
        ):
            compile_obstacle_replacement(other_uav, original, 1)

    def test_rejects_unterminated_or_nonterminal_land_replacements(self) -> None:
        original = _original_plan()
        cases = (
            _draft(
                "goto_search",
                (_step("resume_search", SkillName.SEARCH),),
            ),
            _draft(
                "goto_search",
                (
                    _step("land_home", SkillName.LAND),
                    _step("hover_after_land", SkillName.HOVER),
                ),
            ),
        )
        for draft in cases:
            with self.subTest(steps=[step.skill for step in draft.replacement_steps]):
                with self.assertRaisesRegex(
                    ObstacleReplacementCompilationError,
                    "terminate",
                ):
                    compile_obstacle_replacement(draft, original, 1)

    def test_rejects_model_follow_route_overrides_and_unresolved_explicit_geometry(self) -> None:
        original = _original_plan()
        first_override = _goto_chain_draft(
            first_args={"route_ref": "route_1", "timeout_s": 5.0}
        )
        explicit_goto = _draft(
            "goto_search",
            (
                _step("resume_search", SkillName.SEARCH),
                _step("resume_track", SkillName.TRACK),
                _step(
                    "return_home",
                    SkillName.GOTO,
                    {"target": {"kind": "NAMED_LOCATION", "name": "home"}},
                ),
                _step("land_home", SkillName.LAND),
            ),
        )

        with self.assertRaisesRegex(
            ObstacleReplacementCompilationError,
            "only route_ref",
        ):
            compile_obstacle_replacement(first_override, original, 1)
        with self.assertRaisesRegex(
            ObstacleReplacementCompilationError,
            "SpatialResolver",
        ):
            compile_obstacle_replacement(explicit_goto, original, 1)

    def test_explicit_trusted_return_hover_and_land_are_compiled_without_model_controls(self) -> None:
        draft = _draft(
            "goto_search",
            (
                _step("resume_search", SkillName.SEARCH),
                _step("resume_track", SkillName.TRACK),
                _step(
                    "return_home",
                    SkillName.GOTO,
                    {"target": {"kind": "NAMED_LOCATION", "name": "home"}},
                ),
                _step("observe_before_land", SkillName.HOVER, {"duration_s": 2.5}),
                _step("land_home", SkillName.LAND, {"zone": "home"}),
            ),
        )

        compiled = compile_obstacle_replacement(
            draft,
            _original_plan(),
            1,
            spatial_resolver=_resolver(),
        )

        self.assertEqual(
            [step.skill for step in compiled.steps[-3:]],
            [SkillName.GOTO, SkillName.HOVER, SkillName.LAND],
        )
        self.assertEqual(compiled.steps[-3].params, _original_plan().steps[-2].params)
        self.assertEqual(compiled.steps[-1].params, _original_plan().steps[-1].params)
        hover = compiled.steps[-2]
        self.assertEqual(hover.params["duration_s"], 2.5)
        self.assertEqual(hover.params["reason_code"], "PLANNED_HOVER")
        self.assertNotIn("velocity", hover.params)

    def test_reordering_before_terminal_approach_is_allowed_but_work_after_it_is_rejected(self) -> None:
        unsafe = _draft(
            "goto_search",
            (
                _step(
                    "return_home_too_early",
                    SkillName.GOTO,
                    {"target": {"kind": "NAMED_LOCATION", "name": "home"}},
                ),
                _step("late_search", SkillName.SEARCH),
                _step("late_track", SkillName.TRACK),
                _step("land_home", SkillName.LAND, {"zone": "home"}),
            ),
        )

        with self.assertRaisesRegex(
            ObstacleReplacementCompilationError,
            "after the trusted terminal approach",
        ):
            compile_obstacle_replacement(
                unsafe,
                _original_plan(),
                1,
                spatial_resolver=_resolver(),
            )

    def test_model_may_omit_optional_tracking_while_retaining_safe_termination(self) -> None:
        draft = _draft(
            "goto_search",
            (
                _step("search_again", SkillName.SEARCH),
                _step(
                    "return_home",
                    SkillName.GOTO,
                    {"target": {"kind": "NAMED_LOCATION", "name": "home"}},
                ),
                _step("land_home", SkillName.LAND, {"zone": "home"}),
            ),
        )

        compiled = compile_obstacle_replacement(
            draft,
            _original_plan(),
            1,
            spatial_resolver=_resolver(),
        )

        self.assertNotIn(SkillName.TRACK, [step.skill for step in compiled.steps])
        self.assertEqual(
            [step.skill for step in compiled.steps[-3:]],
            [SkillName.SEARCH, SkillName.GOTO, SkillName.LAND],
        )

    def test_rejects_track_when_its_source_search_was_removed(self) -> None:
        draft = _draft(
            "goto_search",
            (
                _step("track_without_search", SkillName.TRACK),
                _step("return_home", SkillName.GOTO),
                _step("land_home", SkillName.LAND),
            ),
        )

        with self.assertRaisesRegex(
            ObstacleReplacementCompilationError,
            "references removed SEARCH",
        ):
            compile_obstacle_replacement(draft, _original_plan(), 1)

    def test_rejects_invalid_interrupted_index(self) -> None:
        draft = _goto_chain_draft()
        original = _original_plan()
        with self.assertRaises(TypeError):
            compile_obstacle_replacement(draft, original, True)
        with self.assertRaisesRegex(
            ObstacleReplacementCompilationError,
            "outside",
        ):
            compile_obstacle_replacement(draft, original, len(original.steps))


if __name__ == "__main__":
    unittest.main()
