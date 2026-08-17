from __future__ import annotations

from dataclasses import FrozenInstanceError
import copy
import math
import unittest

from planner.schemas import (
    PlanStepDraft,
    RecoveryDraft,
    SkillPlanDraft,
)


def _valid_plan() -> dict[str, object]:
    return {
        "schema_version": 1,
        "steps": [
            {
                "id": "takeoff_1",
                "skill": "TAKEOFF",
                "args": {"altitude_m": 10.0, "yaw_mode": "KEEP_CURRENT"},
            },
            {
                "id": "goto_search",
                "skill": "GOTO",
                "args": {
                    "destination": "search_area",
                    "altitude_m": 10.0,
                    "yaw_mode": "COURSE_ALIGNED",
                },
            },
            {
                "id": "search_1",
                "skill": "SEARCH",
                "args": {
                    "region": "search_area",
                    "target_description": "moving target",
                    "altitude_m": 10.0,
                },
            },
            {
                "id": "track_1",
                "skill": "TRACK",
                "args": {
                    "target_ref": "$search_1.target_id",
                    "duration_s": 10.0,
                    "desired_altitude_m": 10.0,
                    "desired_distance_m": 6.0,
                },
                "recovery": {
                    "skill": "REACQUIRE",
                    "max_attempts": 2,
                    "search_radius_m": 10.0,
                    "timeout_s": 30.0,
                },
            },
            {
                "id": "goto_home",
                "skill": "GOTO",
                "args": {"destination": "home"},
            },
            {
                "id": "land_1",
                "skill": "LAND",
                "args": {"zone": "home"},
            },
        ],
    }


class DynamicPlanSchemasTest(unittest.TestCase):
    def test_valid_plan_round_trips_and_is_defensive(self) -> None:
        raw = _valid_plan()
        draft = SkillPlanDraft.from_dict(raw)
        raw["steps"][0]["args"]["altitude_m"] = 99  # type: ignore[index]

        self.assertEqual(draft.steps[0].args["altitude_m"], 10.0)
        self.assertEqual(SkillPlanDraft.from_dict(draft.to_dict()), draft)
        copied = draft.to_dict()
        copied["steps"][0]["args"]["altitude_m"] = 77  # type: ignore[index]
        self.assertEqual(draft.steps[0].args["altitude_m"], 10.0)
        with self.assertRaises(TypeError):
            draft.steps[0].args["altitude_m"] = 12  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            draft.schema_version = 2  # type: ignore[misc]

    def test_unknown_top_level_step_and_args_fields_are_rejected(self) -> None:
        variants = []
        top = _valid_plan()
        top["oracle"] = True
        variants.append(top)
        step = _valid_plan()
        step["steps"][0]["note"] = "extra"  # type: ignore[index]
        variants.append(step)
        args = _valid_plan()
        args["steps"][1]["args"]["position"] = [1, 2, 3]  # type: ignore[index]
        variants.append(args)
        target = _valid_plan()
        target["steps"][3]["args"]["target_id"] = "real_target"  # type: ignore[index]
        variants.append(target)

        for raw in variants:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                SkillPlanDraft.from_dict(raw)

    def test_required_fields_are_exact(self) -> None:
        for field in ("schema_version", "steps"):
            raw = _valid_plan()
            raw.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                SkillPlanDraft.from_dict(raw)
        for field in ("id", "skill", "args"):
            raw = _valid_plan()
            raw["steps"][0].pop(field)  # type: ignore[index]
            with self.subTest(field=field), self.assertRaises(ValueError):
                SkillPlanDraft.from_dict(raw)

    def test_schema_version_length_ids_and_skill_names_are_strict(self) -> None:
        for version in (True, 1.0, 0, 2):
            raw = _valid_plan()
            raw["schema_version"] = version
            with self.subTest(version=version), self.assertRaises((TypeError, ValueError)):
                SkillPlanDraft.from_dict(raw)

        for step_count in (1, 11):
            raw = _valid_plan()
            raw["steps"] = copy.deepcopy(raw["steps"][:step_count])  # type: ignore[index]
            if step_count == 11:
                raw["steps"] = [
                    {
                        "id": f"goto_{index}",
                        "skill": "GOTO",
                        "args": {"destination": "home"},
                    }
                    for index in range(11)
                ]
            with self.subTest(step_count=step_count), self.assertRaises(ValueError):
                SkillPlanDraft.from_dict(raw)

        duplicate = _valid_plan()
        duplicate["steps"][1]["id"] = "takeoff_1"  # type: ignore[index]
        # Cross-step uniqueness is diagnosed by SymbolicPlanChecker.
        self.assertEqual(
            SkillPlanDraft.from_dict(duplicate).steps[1].id,
            "takeoff_1",
        )

        for invalid_id in ("1_takeoff", "Takeoff", "a-b", "a" * 33):
            raw = _valid_plan()
            raw["steps"][0]["id"] = invalid_id  # type: ignore[index]
            with self.subTest(invalid_id=invalid_id), self.assertRaises(ValueError):
                SkillPlanDraft.from_dict(raw)

        unknown = _valid_plan()
        unknown["steps"][1]["skill"] = "ORBIT"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "unknown"):
            SkillPlanDraft.from_dict(unknown)

    def test_nonfinite_and_bool_numbers_are_rejected(self) -> None:
        for value in (True, math.nan, math.inf, -math.inf):
            raw = _valid_plan()
            raw["steps"][0]["args"]["altitude_m"] = value  # type: ignore[index]
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                SkillPlanDraft.from_dict(raw)

        for value in (True, 1.0):
            raw = _valid_plan()
            raw["steps"][3]["recovery"]["max_attempts"] = value  # type: ignore[index]
            with self.subTest(value=value), self.assertRaises(TypeError):
                SkillPlanDraft.from_dict(raw)

    def test_top_level_reacquire_and_non_track_recovery_are_rejected(self) -> None:
        top_level = _valid_plan()
        top_level["steps"][1] = {
            "id": "reacquire_1",
            "skill": "REACQUIRE",
            "args": {},
        }  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "recovery-only"):
            SkillPlanDraft.from_dict(top_level)

        wrong_owner = _valid_plan()
        wrong_owner["steps"][1]["recovery"] = copy.deepcopy(  # type: ignore[index]
            wrong_owner["steps"][3]["recovery"]  # type: ignore[index]
        )
        with self.assertRaisesRegex(ValueError, "only allowed on TRACK"):
            SkillPlanDraft.from_dict(wrong_owner)

    def test_recovery_fields_and_ranges_are_strict(self) -> None:
        base = {
            "skill": "REACQUIRE",
            "max_attempts": 2,
            "search_radius_m": 10,
            "timeout_s": 30,
        }
        self.assertEqual(RecoveryDraft.from_dict(base).to_dict()["timeout_s"], 30.0)
        defaults = RecoveryDraft.from_dict(
            {"skill": "REACQUIRE", "max_attempts": 1}
        )
        self.assertIsNone(defaults.search_radius_m)
        self.assertIsNone(defaults.timeout_s)
        self.assertEqual(
            defaults.to_dict(),
            {"skill": "REACQUIRE", "max_attempts": 1},
        )
        # Compatibility only: the trusted compiler normalizes the legacy
        # max_attempts=0 convention to disabled recovery and emits a note.
        self.assertEqual(
            RecoveryDraft.from_dict(
                {"skill": "REACQUIRE", "max_attempts": 0}
            ).max_attempts,
            0,
        )

        for field, value in (
            ("skill", "SEARCH"),
            ("max_attempts", 3),
            ("search_radius_m", 2.9),
            ("search_radius_m", 20.1),
            ("timeout_s", 4.9),
            ("timeout_s", 60.1),
        ):
            raw = dict(base)
            raw[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                RecoveryDraft.from_dict(raw)
        extra = dict(base, extra=True)
        with self.assertRaisesRegex(ValueError, "unknown"):
            RecoveryDraft.from_dict(extra)

        for skill, args in (
            ("TAKEOFF", {}),
            (
                "TRACK",
                {"target_ref": "$search_1.target_id", "duration_s": 10},
            ),
        ):
            with self.subTest(skill=skill), self.assertRaises(TypeError):
                PlanStepDraft.from_dict(
                    {
                        "id": "step_1",
                        "skill": skill,
                        "args": args,
                        "recovery": None,
                    }
                )

    def test_fixed_yaw_is_conditional(self) -> None:
        with self.assertRaisesRegex(ValueError, "required for FIXED"):
            PlanStepDraft.from_dict(
                {"id": "goto_1", "skill": "GOTO", "args": {"destination": "home", "yaw_mode": "FIXED"}}
            )
        with self.assertRaisesRegex(ValueError, "only allowed"):
            PlanStepDraft.from_dict(
                {"id": "goto_1", "skill": "GOTO", "args": {"destination": "home", "yaw_mode": "COURSE_ALIGNED", "yaw_deg": 0}}
            )
        step = PlanStepDraft.from_dict(
            {"id": "goto_1", "skill": "GOTO", "args": {"destination": "home", "yaw_mode": "FIXED", "yaw_deg": 0}}
        )
        self.assertEqual(step.args["yaw_deg"], 0.0)

        for yaw_deg in (-360.1, 360.1, 1e308):
            with self.subTest(yaw_deg=yaw_deg), self.assertRaises(ValueError):
                PlanStepDraft.from_dict(
                    {
                        "id": "goto_1",
                        "skill": "GOTO",
                        "args": {
                            "destination": "home",
                            "yaw_mode": "FIXED",
                            "yaw_deg": yaw_deg,
                        },
                    }
                )

    def test_target_description_rejects_hidden_or_low_level_content(self) -> None:
        for description in (
            "oracle_target_pose=(20,30,0)",
            "目标坐标是 20,30,0",
            "moving target; PID kp=9",
            "目标图像位于 frame 4",
            "x=1 y=2 z=3",
            "target_20,30,0",
        ):
            raw = _valid_plan()
            raw["steps"][2]["args"]["target_description"] = description  # type: ignore[index]
            with self.subTest(description=description), self.assertRaises(ValueError):
                SkillPlanDraft.from_dict(raw)

    def test_target_description_allows_bare_physical_object_terms(self) -> None:
        for description in (
            "moving security camera",
            "red picture frame",
            "electric motor on a cart",
            "携带相机的移动目标",
            "装有电机的红色设备",
        ):
            raw = _valid_plan()
            raw["steps"][2]["args"]["target_description"] = description  # type: ignore[index]
            with self.subTest(description=description):
                draft = SkillPlanDraft.from_dict(raw)
                self.assertEqual(
                    draft.steps[2].args["target_description"],
                    description,
                )

    def test_target_description_rejects_structured_media_or_motor_control(self) -> None:
        for description in (
            "camera RGB frame data",
            "motor command=0.75",
            "actuator value 4",
            "目标图像数据位于缓存中",
            "电机命令设为 0.75",
        ):
            raw = _valid_plan()
            raw["steps"][2]["args"]["target_description"] = description  # type: ignore[index]
            with self.subTest(description=description), self.assertRaises(ValueError):
                SkillPlanDraft.from_dict(raw)

    def test_target_reference_must_point_backward_to_search(self) -> None:
        future = _valid_plan()
        future["steps"][3]["args"]["target_ref"] = "$future_search.target_id"  # type: ignore[index]
        self.assertEqual(
            SkillPlanDraft.from_dict(future).steps[3].args["target_ref"],
            "$future_search.target_id",
        )

        wrong_skill = _valid_plan()
        wrong_skill["steps"][3]["args"]["target_ref"] = "$goto_search.target_id"  # type: ignore[index]
        self.assertEqual(
            SkillPlanDraft.from_dict(wrong_skill).steps[3].args["target_ref"],
            "$goto_search.target_id",
        )

    def test_track_lost_action_is_strict_and_optional(self) -> None:
        for action in ("REACQUIRE", "FAIL"):
            raw = _valid_plan()
            raw["steps"][3]["args"]["on_target_lost"] = action  # type: ignore[index]
            draft = SkillPlanDraft.from_dict(raw)
            self.assertEqual(draft.steps[3].args["on_target_lost"], action)

        omitted = SkillPlanDraft.from_dict(_valid_plan())
        self.assertNotIn("on_target_lost", omitted.steps[3].args)
        for invalid in ("RETURN_HOME", "reacquire", True, 1):
            raw = _valid_plan()
            raw["steps"][3]["args"]["on_target_lost"] = invalid  # type: ignore[index]
            with self.subTest(invalid=invalid), self.assertRaises(
                (TypeError, ValueError)
            ):
                SkillPlanDraft.from_dict(raw)

if __name__ == "__main__":
    unittest.main()
