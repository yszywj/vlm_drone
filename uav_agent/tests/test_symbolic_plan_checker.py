from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from planner.policy import PlannerLimits, PlannerPolicy, TargetLostAction
from planner.prompt_builder import build_dynamic_skill_planner_messages
from planner.schemas import (
    LandingZoneSpec,
    NavigationPointSpec,
    PlannerWorldContext,
    RecoveryDraft,
    SearchRegionSpec,
    SkillPlanDraft,
)
from planner.symbolic_checker import PlanIssueCode, SymbolicPlanChecker
from planner.skill_catalog import build_default_skill_catalog


def _world(*, initial_xy: tuple[float, float] = (0.0, 0.0)) -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-50, -50, 0),
        scene_max_xyz_m=(50, 50, 30),
        initial_uav_xyz_m=(initial_xy[0], initial_xy[1], 0),
        search_regions={
            "search_area": SearchRegionSpec(
                "search_area",
                center_xyz_m=(20, 20, 0),
                radius_m=10,
                approach_xyz_m=(20, 20, 10),
            )
        },
        landing_zones={
            "home": LandingZoneSpec("home", position_xy_m=(0, 0))
        },
        navigation_points={
            "checkpoint": NavigationPointSpec(
                "checkpoint",
                position_xyz_m=(10, 0, 10),
            )
        },
        default_takeoff_altitude_m=10,
        default_track_duration_s=10,
        search_timeout_s=60,
    )


def _raw_plan() -> dict[str, object]:
    return {
        "schema_version": 1,
        "steps": [
            {"id": "takeoff", "skill": "TAKEOFF", "args": {}},
            {
                "id": "goto_search",
                "skill": "GOTO",
                "args": {"destination": "search_area"},
            },
            {
                "id": "search",
                "skill": "SEARCH",
                "args": {
                    "region": "search_area",
                    "target_description": "moving target",
                },
            },
            {
                "id": "track",
                "skill": "TRACK",
                "args": {
                    "target_ref": "$search.target_id",
                    "duration_s": 10,
                },
            },
            {
                "id": "goto_home",
                "skill": "GOTO",
                "args": {"destination": "home"},
            },
            {"id": "land", "skill": "LAND", "args": {"zone": "home"}},
        ],
    }


def _draft(raw: dict[str, object] | None = None) -> SkillPlanDraft:
    return SkillPlanDraft.from_dict(_raw_plan() if raw is None else raw)


def _codes(result: object) -> list[PlanIssueCode]:
    return [issue.code for issue in result.issues]  # type: ignore[attr-defined]


class SymbolicPlanCheckerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.checker = SymbolicPlanChecker()
        self.world = _world()
        self.limits = PlannerLimits()
        self.policy = PlannerPolicy()

    def check(self, draft: SkillPlanDraft, **overrides: object):
        return self.checker.check(
            draft,
            world_context=overrides.get("world_context", self.world),
            limits=overrides.get("limits", self.limits),
            policy=overrides.get("policy", self.policy),
        )

    def test_valid_plan_has_no_issues(self) -> None:
        result = self.check(_draft())
        self.assertTrue(result.valid)
        self.assertEqual(result.issues, ())

    def test_plan_length_and_terminal_shape_have_stable_codes(self) -> None:
        result = self.check(_draft(), limits=PlannerLimits(max_plan_steps=5))
        self.assertIn(PlanIssueCode.PLAN_TOO_LONG, _codes(result))

        reordered = _raw_plan()
        reordered["steps"][0], reordered["steps"][1] = (  # type: ignore[index]
            reordered["steps"][1],  # type: ignore[index]
            reordered["steps"][0],  # type: ignore[index]
        )
        result = self.check(_draft(reordered))
        self.assertIn(PlanIssueCode.TAKEOFF_NOT_FIRST, _codes(result))

        nonfinal = _raw_plan()
        nonfinal["steps"][4], nonfinal["steps"][5] = (  # type: ignore[index]
            nonfinal["steps"][5],  # type: ignore[index]
            nonfinal["steps"][4],  # type: ignore[index]
        )
        result = self.check(_draft(nonfinal))
        self.assertIn(PlanIssueCode.LAND_NOT_FINAL, _codes(result))

        duplicate_terminal = _raw_plan()
        duplicate_terminal["steps"].insert(1, {  # type: ignore[union-attr]
            "id": "takeoff_2",
            "skill": "TAKEOFF",
            "args": {},
        })
        duplicate_terminal["steps"].insert(-1, {  # type: ignore[union-attr]
            "id": "land_2",
            "skill": "LAND",
            "args": {"zone": "home"},
        })
        result = self.check(_draft(duplicate_terminal))
        self.assertIn(PlanIssueCode.TAKEOFF_COUNT_INVALID, _codes(result))
        self.assertIn(PlanIssueCode.LAND_COUNT_INVALID, _codes(result))

    def test_land_requires_matching_preceding_goto(self) -> None:
        missing = _raw_plan()
        missing["steps"].pop(4)  # type: ignore[union-attr]
        result = self.check(_draft(missing))
        self.assertIn(PlanIssueCode.LAND_GOTO_MISSING, _codes(result))

        mismatch = _raw_plan()
        mismatch["steps"][4]["args"]["destination"] = "checkpoint"  # type: ignore[index]
        result = self.check(_draft(mismatch))
        self.assertIn(PlanIssueCode.LAND_ZONE_MISMATCH, _codes(result))

    def test_takeoff_land_exception_requires_actual_home_location(self) -> None:
        raw = {
            "schema_version": 1,
            "steps": [
                {"id": "takeoff", "skill": "TAKEOFF", "args": {}},
                {"id": "land", "skill": "LAND", "args": {"zone": "home"}},
            ],
        }
        self.assertTrue(self.check(_draft(raw)).valid)
        away = self.check(_draft(raw), world_context=_world(initial_xy=(5, 0)))
        self.assertIn(PlanIssueCode.LAND_GOTO_MISSING, _codes(away))

    def test_track_forward_and_non_search_references_are_distinct(self) -> None:
        future = _raw_plan()
        track = future["steps"].pop(3)  # type: ignore[union-attr]
        track["args"]["target_ref"] = "$search_later.target_id"  # type: ignore[index]
        future["steps"].insert(2, track)  # type: ignore[union-attr]
        future["steps"].insert(3, {  # type: ignore[union-attr]
            "id": "search_later",
            "skill": "SEARCH",
            "args": {
                "region": "search_area",
                "target_description": "moving target",
            },
        })
        # Remove the original SEARCH to keep the v1 call budget valid.
        future["steps"] = [  # type: ignore[index]
            step for step in future["steps"] if step["id"] != "search"  # type: ignore[index]
        ]
        result = self.check(_draft(future))
        self.assertIn(PlanIssueCode.TARGET_REF_FORWARD, _codes(result))

        wrong = _raw_plan()
        wrong["steps"][3]["args"]["target_ref"] = "$goto_search.target_id"  # type: ignore[index]
        result = self.check(_draft(wrong))
        self.assertIn(PlanIssueCode.TARGET_REF_NOT_SEARCH, _codes(result))

    def test_track_without_search_and_invalid_reference(self) -> None:
        raw = _raw_plan()
        raw["steps"].pop(2)  # type: ignore[union-attr]
        raw["steps"][2]["args"]["target_ref"] = "target_0"  # type: ignore[index]
        result = self.check(_draft(raw))
        self.assertIn(PlanIssueCode.TARGET_REF_INVALID, _codes(result))
        self.assertIn(PlanIssueCode.TRACK_WITHOUT_SEARCH, _codes(result))

    def test_skill_call_limits_and_duplicate_ids_are_centralized(self) -> None:
        result = self.check(_draft(), limits=PlannerLimits(max_goto_calls=1))
        self.assertIn(PlanIssueCode.GOTO_LIMIT_EXCEEDED, _codes(result))

        duplicate = _raw_plan()
        duplicate["steps"][1]["id"] = "takeoff"  # type: ignore[index]
        result = self.check(_draft(duplicate))
        self.assertIn(PlanIssueCode.STEP_ID_DUPLICATE, _codes(result))

    def test_non_track_recovery_and_top_level_reacquire_defense(self) -> None:
        draft = _draft()
        object.__setattr__(
            draft.steps[1],
            "recovery",
            RecoveryDraft("REACQUIRE", 1),
        )
        object.__setattr__(draft.steps[2], "skill", "REACQUIRE")
        result = self.check(draft)
        self.assertIn(PlanIssueCode.RECOVERY_ON_NON_TRACK, _codes(result))
        self.assertIn(
            PlanIssueCode.TOP_LEVEL_REACQUIRE_FORBIDDEN,
            _codes(result),
        )

    def test_fail_conflicts_with_recovery(self) -> None:
        raw = _raw_plan()
        raw["steps"][3]["args"]["on_target_lost"] = "FAIL"  # type: ignore[index]
        raw["steps"][3]["recovery"] = {  # type: ignore[index]
            "skill": "REACQUIRE",
            "max_attempts": 1,
        }
        result = self.check(_draft(raw))
        issue = next(
            issue
            for issue in result.issues
            if issue.code is PlanIssueCode.RECOVERY_CONFLICTS_WITH_FAIL
        )
        self.assertEqual(issue.step_id, "track")
        self.assertTrue(issue.repairable)

    def test_recovery_per_track_and_total_budgets(self) -> None:
        raw = _raw_plan()
        raw["steps"][3]["recovery"] = {  # type: ignore[index]
            "skill": "REACQUIRE",
            "max_attempts": 2,
        }
        narrow_limits = PlannerLimits(max_reacquire_attempts_per_track=1)
        narrow_policy = PlannerPolicy(
            default_reacquire_max_attempts=1
        ).validate_against(narrow_limits)
        result = self.check(
            _draft(raw),
            limits=narrow_limits,
            policy=narrow_policy,
        )
        self.assertIn(PlanIssueCode.RECOVERY_BUDGET_EXCEEDED, _codes(result))

        second = copy.deepcopy(raw["steps"][3])  # type: ignore[index]
        second["id"] = "track_2"
        raw["steps"].insert(4, second)  # type: ignore[union-attr]
        total_limits = PlannerLimits(max_total_reacquire_attempts=3)
        total_policy = PlannerPolicy(
            default_reacquire_max_attempts=1
        ).validate_against(total_limits)
        result = self.check(
            _draft(raw),
            limits=total_limits,
            policy=total_policy,
        )
        self.assertIn(PlanIssueCode.RECOVERY_BUDGET_EXCEEDED, _codes(result))

    def test_total_budget_counts_explicit_and_policy_injected_recovery(self) -> None:
        raw = _raw_plan()
        raw["steps"][3]["recovery"] = {  # type: ignore[index]
            "skill": "REACQUIRE",
            "max_attempts": 2,
        }
        second = copy.deepcopy(raw["steps"][3])  # type: ignore[index]
        second["id"] = "track_2"
        second.pop("recovery")
        raw["steps"].insert(4, second)  # type: ignore[union-attr]
        limits = PlannerLimits(max_total_reacquire_attempts=2)
        policy = PlannerPolicy(
            default_reacquire_max_attempts=1
        ).validate_against(limits)

        result = self.check(_draft(raw), limits=limits, policy=policy)

        self.assertIn(PlanIssueCode.RECOVERY_BUDGET_EXCEEDED, _codes(result))

    def test_unknown_named_locations_share_one_stable_code(self) -> None:
        for index, key, value in (
            (1, "destination", "invented"),
            (2, "region", "invented"),
            (5, "zone", "invented"),
        ):
            raw = _raw_plan()
            raw["steps"][index]["args"][key] = value  # type: ignore[index]
            with self.subTest(index=index):
                result = self.check(_draft(raw))
                issue = next(
                    issue
                    for issue in result.issues
                    if issue.code is PlanIssueCode.UNKNOWN_NAMED_LOCATION
                )
                self.assertTrue(issue.repairable)

    def test_policy_accepts_fail_and_rejects_untrusted_values(self) -> None:
        fail = PlannerPolicy(default_on_target_lost=TargetLostAction.FAIL)
        self.assertEqual(fail.default_on_target_lost, TargetLostAction.FAIL)
        for kwargs in (
            {"default_on_target_lost": "RETURN_HOME"},
            {"default_reacquire_max_attempts": True},
            {"default_reacquire_max_attempts": 0},
            {"default_reacquire_max_attempts": 3},
            {"default_reacquire_search_radius_m": 2.9},
            {"default_reacquire_search_radius_m": True},
            {"default_reacquire_timeout_s": 61},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PlannerPolicy(**kwargs)

        with self.assertRaisesRegex(ValueError, "per-TRACK"):
            PlannerPolicy().validate_against(
                PlannerLimits(max_reacquire_attempts_per_track=1)
            )

    def test_prompt_projects_only_trusted_lost_target_actions(self) -> None:
        prompt = (
            Path(__file__).resolve().parents[1]
            / "prompts"
            / "dynamic_skill_planner_system.txt"
        ).read_text(encoding="utf-8")
        messages = build_dynamic_skill_planner_messages(
            "起飞后搜索目标，丢失即失败",
            self.world,
            build_default_skill_catalog(),
            self.limits,
            prompt,
            PlannerPolicy(default_on_target_lost="FAIL"),
        )
        payload = json.loads(str(messages[1].content))
        self.assertEqual(
            payload["trusted_planner_policy"],
            {
                "default_on_target_lost": "FAIL",
                "allowed_on_target_lost": ["REACQUIRE", "FAIL"],
            },
        )
        serialized = json.dumps(payload["trusted_planner_policy"])
        self.assertNotIn("search_radius", serialized)
        self.assertNotIn("timeout", serialized)

        with self.assertRaises(ValueError):
            build_dynamic_skill_planner_messages(
                "search",
                self.world,
                build_default_skill_catalog(),
                self.limits,
                prompt,
                {"default_on_target_lost": "RETURN_HOME"},
            )


if __name__ == "__main__":
    unittest.main()
