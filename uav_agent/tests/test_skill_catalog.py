from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import unittest

from planner.skill_catalog import (
    SkillArgumentSpec,
    SkillCatalog,
    SkillContract,
    build_default_skill_catalog,
)


class SkillCatalogTest(unittest.TestCase):
    def test_default_catalog_has_stable_complete_order(self) -> None:
        catalog = build_default_skill_catalog()

        self.assertEqual(
            [contract.name for contract in catalog],
            ["TAKEOFF", "GOTO", "SEARCH", "TRACK", "REACQUIRE", "LAND"],
        )
        self.assertEqual(
            catalog.to_prompt_dict(),
            build_default_skill_catalog().to_prompt_dict(),
        )
        json.dumps(catalog.to_prompt_dict(), allow_nan=False)

    def test_only_reacquire_is_recovery_only(self) -> None:
        catalog = build_default_skill_catalog()
        reacquire = catalog.get("reacquire")

        self.assertFalse(reacquire.top_level_allowed)
        self.assertTrue(reacquire.recovery_only)
        self.assertEqual(
            [argument.name for argument in reacquire.arguments],
            ["max_attempts", "search_radius_m", "timeout_s"],
        )
        self.assertEqual(
            [argument.required for argument in reacquire.arguments],
            [True, False, False],
        )
        for name in ("TAKEOFF", "GOTO", "SEARCH", "TRACK", "LAND"):
            self.assertTrue(catalog.get(name).top_level_allowed)
            self.assertFalse(catalog.get(name).recovery_only)

    def test_contract_argument_allow_lists_match_dynamic_protocol(self) -> None:
        catalog = build_default_skill_catalog()
        expected = {
            "TAKEOFF": {"altitude_m", "yaw_mode", "yaw_deg"},
            "GOTO": {"destination", "altitude_m", "yaw_mode", "yaw_deg"},
            "SEARCH": {"region", "target_description", "altitude_m"},
            "TRACK": {
                "target_ref",
                "duration_s",
                "desired_altitude_m",
                "desired_distance_m",
                "on_target_lost",
            },
            "REACQUIRE": {"max_attempts", "search_radius_m", "timeout_s"},
            "LAND": {"zone", "yaw_mode", "yaw_deg"},
        }
        for skill, arguments in expected.items():
            with self.subTest(skill=skill):
                self.assertEqual(
                    {item.name for item in catalog.get(skill).arguments},
                    arguments,
                )

        goto_yaw = next(
            arg for arg in catalog.get("GOTO").arguments if arg.name == "yaw_mode"
        )
        self.assertEqual(
            goto_yaw.allowed_values,
            ("KEEP_CURRENT", "COURSE_ALIGNED", "FACE_POINT", "FIXED"),
        )
        lost_action = next(
            arg
            for arg in catalog.get("TRACK").arguments
            if arg.name == "on_target_lost"
        )
        self.assertFalse(lost_action.required)
        self.assertEqual(lost_action.allowed_values, ("REACQUIRE", "FAIL"))
        attempts = next(
            arg
            for arg in catalog.get("REACQUIRE").arguments
            if arg.name == "max_attempts"
        )
        self.assertEqual((attempts.minimum, attempts.maximum), (1.0, 2.0))

    def test_catalog_does_not_expose_runtime_goal_fields_or_world_data(self) -> None:
        serialized = json.dumps(
            build_default_skill_catalog().to_prompt_dict(),
            ensure_ascii=False,
            sort_keys=True,
        ).casefold()
        for forbidden in (
            "target_altitude",
            "center_xyz",
            "approach_xyz",
            "position_xyz",
            "target_position",
            "target_velocity",
            "oracle_target",
            "max_speed",
            "climb_speed",
            "descent_speed",
            "velocity vector",
            "pid",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, serialized)

    def test_catalog_is_immutable_and_prompt_output_is_defensive(self) -> None:
        catalog = build_default_skill_catalog()
        with self.assertRaises(FrozenInstanceError):
            catalog.skills = ()  # type: ignore[misc]

        prompt_data = catalog.to_prompt_dict()
        prompt_data["skills"][0]["name"] = "ALTERED"  # type: ignore[index]
        self.assertEqual(catalog.get("TAKEOFF").name, "TAKEOFF")

    def test_invalid_contracts_fail_closed(self) -> None:
        argument = SkillArgumentSpec("x", "semantic x", "number")
        with self.assertRaises(ValueError):
            SkillContract("BAD", "bad", True, True, (argument,))
        with self.assertRaises(ValueError):
            SkillCatalog(
                (
                    SkillContract("A", "a", True, False, (argument,)),
                    SkillContract("A", "b", True, False, (argument,)),
                )
            )

        with self.assertRaisesRegex(ValueError, "unsupported v1"):
            SkillCatalog(
                (SkillContract("LOCK", "semantic lock", True, False, ()),)
            )
        with self.assertRaisesRegex(ValueError, "low-level arguments"):
            SkillCatalog(
                (
                    SkillContract(
                        "GOTO",
                        "semantic navigation",
                        True,
                        False,
                        (SkillArgumentSpec("position", "raw position", "string"),),
                    ),
                )
            )
        with self.assertRaisesRegex(ValueError, "coordinate"):
            SkillContract(
                "GOTO",
                "navigate to [1,2,3]",
                True,
                False,
                (),
            )


if __name__ == "__main__":
    unittest.main()
