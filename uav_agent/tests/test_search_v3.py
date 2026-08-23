"""Runtime compatibility tests for SearchGoalV3."""

from __future__ import annotations

import unittest

from env.moving_target import TargetState
from planner.spatial import CircleRegion, CoordinateFrame, RelationalRegion, SpatialRelation
from skills.search import SearchGoal, SearchGoalV3, SearchSkill
from skills.search_strategy import (
    NextBestViewPollResult,
    NextBestViewRequest,
    SearchEntryPolicy,
    SearchStrategySpec,
    SearchStrategyType,
)
from skills.types import SkillResultCode, SkillStatus
from tests.test_search import (
    ManualClock,
    make_context,
    make_observation,
    make_uav,
    run_with_oracle,
)


def goal_v3(**overrides: object) -> SearchGoalV3:
    values: dict[str, object] = {
        "region": CircleRegion(CoordinateFrame.WORLD_ENU, (0, 0, 0), 4),
        "strategy": SearchStrategySpec(SearchStrategyType.PERIMETER_V1),
        "entry_policy": SearchEntryPolicy.START_IN_PLACE_IF_INSIDE,
        "target_description": "moving target",
        "search_altitude_m": 5,
        "timeout_s": 60,
        "transit_speed_mps": 2,
        "scan_yaw_rate_rad_s": 2,
    }
    values.update(overrides)
    return SearchGoalV3(**values)  # type: ignore[arg-type]


class SearchV3Tests(unittest.TestCase):
    def test_v2_goal_and_six_point_builder_remain_supported(self) -> None:
        self.assertIsInstance(
            SearchGoal((0, 0, 0), 5, "target", 8), SearchGoal
        )
        self.assertIn(SearchGoal, SearchSkill.goal_type)
        self.assertIn(SearchGoalV3, SearchSkill.goal_type)

    def test_inside_entry_starts_at_current_xy_not_west_boundary(self) -> None:
        uav = make_uav((1, 1, 5))
        skill = SearchSkill()
        skill.start(goal_v3(), make_context(uav, ManualClock()))
        self.assertIs(skill.status, SkillStatus.RUNNING)
        self.assertEqual(tuple(skill._waypoints[0]), (1.0, 1.0, 5.0))

    def test_v3_target_found_outputs_coverage_and_actual_points(self) -> None:
        uav = make_uav((0, 0, 5))
        clock = ManualClock()
        skill = SearchSkill()
        skill.start(goal_v3(), make_context(uav, clock))
        observation = make_observation(
            uav,
            clock,
            target_pose=TargetState(0, 0, 0, 0),
            visible=True,
        )
        self.assertIs(skill.tick(observation), SkillStatus.SUCCEEDED)
        result = skill.get_result()
        self.assertIs(result.code, SkillResultCode.TARGET_FOUND)
        self.assertIn("coverage_ratio", result.data)
        self.assertIn("visited_viewpoints", result.data)
        self.assertIn("elapsed_time", result.data)
        self.assertIsNone(result.data["search_exhausted_reason"])

    def test_unresolved_relational_region_fails_closed(self) -> None:
        uav = make_uav((0, 0, 5))
        skill = SearchSkill()
        skill.start(
            goal_v3(
                region=RelationalRegion(
                    SpatialRelation.LEFT_OF, "red_building", 10, (20, 10)
                )
            ),
            make_context(uav, ManualClock()),
        )
        self.assertIs(skill.status, SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INVALID_STATE)

    def test_invalid_v3_goal_is_reported_as_invalid_goal(self) -> None:
        uav = make_uav((0, 0, 5))
        skill = SearchSkill()
        skill.start(goal_v3(target_description="  "), make_context(uav, ManualClock()))
        self.assertIs(skill.status, SkillStatus.FAILED)
        self.assertIs(skill.get_result().code, SkillResultCode.INVALID_GOAL)

    def test_adaptive_provider_receives_fresh_frames_and_adds_macro_views(self) -> None:
        class Provider:
            def __init__(self) -> None:
                self.requests: list[NextBestViewRequest] = []

            def next_best_view(
                self,
                request: NextBestViewRequest,
            ) -> tuple[float, float, float] | None:
                self.requests.append(request)
                return (2.0, 0.0, 5.0) if len(self.requests) == 1 else None

        provider = Provider()
        uav = make_uav((0, 0, 5), max_speed=5, max_yaw_rate=5)
        clock = ManualClock()
        skill = SearchSkill(next_best_view_provider=provider)
        skill.start(
            goal_v3(
                strategy=SearchStrategySpec(
                    SearchStrategyType.ADAPTIVE_NEXT_BEST_VIEW,
                    max_viewpoints=3,
                ),
                timeout_s=30,
            ),
            make_context(uav, clock),
        )

        status, _ = run_with_oracle(
            skill,
            uav,
            clock,
            force_visible=False,
            dt_s=0.05,
            max_steps=2000,
        )

        self.assertIs(status, SkillStatus.FAILED)
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(
            provider.requests[0].visited_viewpoints_xyz_m,
            ((0.0, 0.0, 5.0),),
        )
        request = provider.requests[0]
        self.assertEqual(request.camera_rgb.shape, (4, 4, 3))
        self.assertEqual(request.search_altitude_m, 5.0)
        self.assertFalse(request.camera_rgb.flags.writeable)
        self.assertFalse(hasattr(request, "observation"))
        result = skill.get_result()
        self.assertIs(result.code, SkillResultCode.SEARCH_EXHAUSTED)
        self.assertEqual(result.data["search_exhausted_reason"], "ADAPTIVE_PROVIDER_EXHAUSTED")
        self.assertEqual(
            result.data["visited_viewpoints"],
            ((0.0, 0.0, 5.0), (2.0, 0.0, 5.0)),
        )

    def test_async_adaptive_provider_is_polled_without_duplicate_submission(self) -> None:
        class Provider:
            def __init__(self) -> None:
                self.requests: list[NextBestViewRequest] = []
                self.poll_count = 0

            def submit_next_best_view(self, request: NextBestViewRequest) -> None:
                self.requests.append(request)
                self.poll_count = 0

            def poll_next_best_view(self) -> NextBestViewPollResult:
                self.poll_count += 1
                if self.poll_count < 3:
                    return NextBestViewPollResult(completed=False)
                if len(self.requests) == 1:
                    return NextBestViewPollResult(True, (2, 0, 5))
                return NextBestViewPollResult(True, None)

        provider = Provider()
        uav = make_uav((0, 0, 5), max_speed=5, max_yaw_rate=5)
        clock = ManualClock()
        skill = SearchSkill(next_best_view_provider=provider)
        skill.start(
            goal_v3(
                strategy=SearchStrategySpec(
                    SearchStrategyType.ADAPTIVE_NEXT_BEST_VIEW,
                    max_viewpoints=3,
                ),
                timeout_s=30,
            ),
            make_context(uav, clock),
        )

        status, _ = run_with_oracle(
            skill,
            uav,
            clock,
            force_visible=False,
            dt_s=0.05,
            max_steps=2000,
        )

        self.assertIs(status, SkillStatus.FAILED)
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(
            skill.get_result().data["search_exhausted_reason"],
            "ADAPTIVE_PROVIDER_EXHAUSTED",
        )


if __name__ == "__main__":
    unittest.main()
