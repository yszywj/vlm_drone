from __future__ import annotations

import unittest

from planner.schemas import MissionIntent
from tasks.intent_canonicalizer import (
    IntentCanonicalizationError,
    canonicalize_gold,
    canonicalize_prediction,
    finite_numbers_match,
)
from tasks.schemas import GoldPlannerSpec, PlannerWorldCase
from tasks.target_ontology import TargetOntology


CONCEPT_ID = "person_upper_red_backpack_black"
CANONICAL = "穿红色上衣并背黑色背包的人"


def world() -> PlannerWorldCase:
    return PlannerWorldCase(
        context_id="world_01",
        search_regions={
            "east_area": "东侧",
            "west_area": "西侧",
            "north_area": "北侧",
            "south_area": "南侧",
        },
        landing_zones={
            "home": "起点",
            "north_pad": "北平台",
            "south_pad": "南平台",
        },
        default_takeoff_altitude_m=10,
        default_track_duration_s=30,
        scene_min_xyz_m=(-50, -50, 0),
        scene_max_xyz_m=(50, 50, 30),
    )


def gold(**updates: object) -> GoldPlannerSpec:
    values: dict[str, object] = {
        "spec_id": "spec",
        "target_concept_id": CONCEPT_ID,
        "target_description": CANONICAL,
        "search_region": "east_area",
        "track_duration_s": 30,
        "landing_zone": "home",
        "takeoff_altitude_m": None,
        "explicit_fields": {
            "target_description",
            "search_region",
            "landing_zone",
        },
    }
    values.update(updates)
    return GoldPlannerSpec(**values)


class IntentCanonicalizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ontology = TargetOntology.load_default()
        cls.world = world()

    def test_gold_omitted_duration_and_altitude_use_world_defaults(self) -> None:
        effective = canonicalize_gold(gold(), self.world, self.ontology)
        self.assertEqual(effective.track_duration_s, 30)
        self.assertEqual(effective.takeoff_altitude_m, 10)
        self.assertEqual(effective.target_concept_id, CONCEPT_ID)

    def test_explicit_gold_values_are_retained(self) -> None:
        effective = canonicalize_gold(
            gold(
                track_duration_s=45,
                takeoff_altitude_m=12,
                explicit_fields={
                    "target_description",
                    "search_region",
                    "track_duration_s",
                    "landing_zone",
                    "takeoff_altitude_m",
                },
            ),
            self.world,
            self.ontology,
        )
        self.assertEqual(effective.track_duration_s, 45)
        self.assertEqual(effective.takeoff_altitude_m, 12)

    def test_omitted_duration_must_equal_the_trusted_world_default(self) -> None:
        with self.assertRaisesRegex(
            IntentCanonicalizationError,
            "omitted Gold track_duration_s",
        ):
            canonicalize_gold(
                gold(track_duration_s=20),
                self.world,
                self.ontology,
            )

    def test_prediction_none_altitude_uses_same_world_default(self) -> None:
        effective = canonicalize_prediction(
            MissionIntent(CANONICAL, "east_area", 30, "home", None),
            self.world,
            self.ontology,
        )
        self.assertEqual(effective.takeoff_altitude_m, 10)

    def test_registered_alias_maps_but_unknown_text_is_not_guessed(self) -> None:
        alias = canonicalize_prediction(
            MissionIntent("红衣黑包的人", "east_area", 30, "home", None),
            self.world,
            self.ontology,
        )
        unknown = canonicalize_prediction(
            MissionIntent("很像红衣黑包的目标", "east_area", 30, "home", None),
            self.world,
            self.ontology,
        )
        self.assertEqual(alias.target_concept_id, CONCEPT_ID)
        self.assertEqual(alias.target_description, CANONICAL)
        self.assertIsNone(unknown.target_concept_id)
        self.assertEqual(unknown.target_description, "很像红衣黑包的目标")

    def test_unknown_gold_region_or_landing_zone_is_rejected(self) -> None:
        with self.assertRaisesRegex(IntentCanonicalizationError, "search_region"):
            canonicalize_gold(
                gold(search_region="missing"), self.world, self.ontology
            )
        with self.assertRaisesRegex(IntentCanonicalizationError, "landing_zone"):
            canonicalize_gold(
                gold(landing_zone="missing"), self.world, self.ontology
            )

    def test_numeric_comparison_is_finite_and_tolerant(self) -> None:
        self.assertTrue(finite_numbers_match(30, 30.0000005))
        self.assertFalse(finite_numbers_match(30, 30.01))
        self.assertFalse(finite_numbers_match(float("nan"), 30))
        self.assertFalse(finite_numbers_match(float("inf"), 30))


if __name__ == "__main__":
    unittest.main()
