from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest

from planner.schemas import MissionIntent
from tasks.schemas import GoldPlannerSpec, PlannerWorldCase, TargetConcept
from tasks.target_ontology import (
    TargetOntology,
    TargetOntologyError,
    render_canonical_target_description,
)


CONCEPT_ID = "person_upper_red_backpack_black"
CANONICAL = "穿红色上衣并背黑色背包的人"


def make_gold(**updates: object) -> GoldPlannerSpec:
    values: dict[str, object] = {
        "spec_id": "spec_000001",
        "target_concept_id": CONCEPT_ID,
        "target_description": CANONICAL,
        "search_region": "east_area",
        "track_duration_s": 30.0,
        "landing_zone": "home",
        "takeoff_altitude_m": None,
        "explicit_fields": frozenset(
            {
                "target_description",
                "search_region",
                "track_duration_s",
                "landing_zone",
            }
        ),
    }
    values.update(updates)
    return GoldPlannerSpec(**values)


class TargetOntologyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ontology = TargetOntology.load_default()

    def test_default_closed_set_is_sampled_and_valid(self) -> None:
        self.assertGreaterEqual(len(self.ontology.concepts), 15)
        self.assertLessEqual(len(self.ontology.concepts), 30)
        self.assertEqual(len(self.ontology.concepts), 20)
        self.assertEqual(
            self.ontology.require_concept(CONCEPT_ID).canonical_description,
            CANONICAL,
        )

    def test_canonical_render_is_independent_of_attribute_input_order(self) -> None:
        first = render_canonical_target_description(
            "person",
            {"upper_clothing_color": "red", "backpack_color": "black"},
        )
        second = render_canonical_target_description(
            "person",
            {"backpack_color": "black", "upper_clothing_color": "red"},
        )
        self.assertEqual(first, CANONICAL)
        self.assertEqual(second, first)

    def test_only_explicit_deterministic_aliases_resolve(self) -> None:
        self.assertEqual(
            self.ontology.resolve_concept_id("红衣黑包的人"),
            CONCEPT_ID,
        )
        self.assertEqual(
            self.ontology.resolve_concept_id(f"  {CANONICAL}  "),
            CONCEPT_ID,
        )
        self.assertIsNone(self.ontology.resolve_concept_id("看起来像红衣黑包的人"))

    def test_unknown_category_attribute_and_value_are_rejected(self) -> None:
        with self.assertRaisesRegex(TargetOntologyError, "unknown target category"):
            self.ontology.render_description("vehicle", {"color": "red"})
        with self.assertRaisesRegex(TargetOntologyError, "unknown attributes"):
            self.ontology.render_description("person", {"shoe_color": "red"})
        with self.assertRaisesRegex(TargetOntologyError, "unknown value"):
            self.ontology.render_description(
                "person", {"upper_clothing_color": "purple"}
            )
        with self.assertRaisesRegex(TargetOntologyError, "unknown target category"):
            self.ontology.validate_concept(
                TargetConcept(
                    concept_id="vehicle_red",
                    category="vehicle",
                    attributes={"color": "red"},
                    canonical_description="红色车辆",
                )
            )

    def test_duplicate_concept_id_description_attributes_and_alias_are_rejected(self) -> None:
        base = {
            "schema_version": "test",
            "categories": {
                "person": {
                    "upper_clothing_color": ["red", "blue"],
                    "lower_clothing_color": ["black"],
                    "backpack_color": ["black"],
                }
            },
        }
        concept = {
            "concept_id": "a",
            "category": "person",
            "attributes": {"upper_clothing_color": "red"},
            "aliases": ["红衣人"],
        }
        with self.assertRaisesRegex(TargetOntologyError, "duplicate concept_id"):
            TargetOntology.from_mapping({**base, "concepts": [concept, concept]})

        duplicate_semantics = {
            **concept,
            "concept_id": "b",
            "aliases": ["另一说法"],
        }
        with self.assertRaisesRegex(TargetOntologyError, "duplicate canonical"):
            TargetOntology.from_mapping(
                {**base, "concepts": [concept, duplicate_semantics]}
            )

        other = {
            "concept_id": "b",
            "category": "person",
            "attributes": {"upper_clothing_color": "blue"},
            "aliases": ["红衣人"],
        }
        with self.assertRaisesRegex(TargetOntologyError, "maps to both"):
            TargetOntology.from_mapping({**base, "concepts": [concept, other]})

    def test_supplied_noncanonical_description_is_rejected(self) -> None:
        raw = {
            "schema_version": "test",
            "categories": {
                "person": {
                    "upper_clothing_color": ["red"],
                    "lower_clothing_color": ["black"],
                    "backpack_color": ["black"],
                }
            },
            "concepts": [
                {
                    "concept_id": "a",
                    "category": "person",
                    "attributes": {"upper_clothing_color": "red"},
                    "canonical_description": "红衣服的人",
                }
            ],
        }
        with self.assertRaisesRegex(TargetOntologyError, "non-canonical"):
            TargetOntology.from_mapping(raw)

    def test_malformed_yaml_is_rejected_without_path_disclosure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("categories: [", encoding="utf-8")
            with self.assertRaisesRegex(
                TargetOntologyError, "could not read target ontology"
            ):
                TargetOntology.from_file(path)

    def test_conflicting_duplicate_yaml_attribute_is_rejected(self) -> None:
        text = """
schema_version: test
categories:
  person:
    upper_clothing_color: [red, blue]
    lower_clothing_color: [black]
    backpack_color: [black]
concepts:
  - concept_id: duplicate_attribute
    category: person
    attributes:
      upper_clothing_color: red
      upper_clothing_color: blue
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicates.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(
                TargetOntologyError, "could not read target ontology"
            ):
                TargetOntology.from_file(path)


class GoldMissionSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ontology = TargetOntology.load_default()

    def test_valid_gold_is_immutable_json_compatible_and_gold_to_intent_only(self) -> None:
        gold = make_gold()
        self.ontology.validate_gold_spec(gold)
        with self.assertRaises(FrozenInstanceError):
            gold.search_region = "west_area"  # type: ignore[misc]
        encoded = json.loads(json.dumps(gold.to_dict(), ensure_ascii=False))
        self.assertEqual(encoded["target_description"], CANONICAL)
        expected = gold.to_expected_intent()
        self.assertIsInstance(expected, MissionIntent)
        self.assertEqual(expected.target_description, CANONICAL)
        self.assertFalse(hasattr(GoldPlannerSpec, "from_mission_intent"))
        self.assertFalse(hasattr(GoldPlannerSpec, "from_intent"))

    def test_empty_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "spec_id"):
            make_gold(spec_id=" ")
        with self.assertRaisesRegex(ValueError, "target_concept_id"):
            make_gold(target_concept_id="")

    def test_invalid_duration_and_altitude_are_rejected(self) -> None:
        for duration in (0, -1, float("nan"), float("inf")):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                make_gold(track_duration_s=duration)
        for altitude in (0, -1, float("nan"), float("inf")):
            with self.subTest(altitude=altitude), self.assertRaises(ValueError):
                make_gold(
                    takeoff_altitude_m=altitude,
                    explicit_fields={
                        "target_description",
                        "search_region",
                        "track_duration_s",
                        "landing_zone",
                        "takeoff_altitude_m",
                    },
                )

    def test_omitted_altitude_must_be_none_and_explicit_altitude_must_exist(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be None"):
            make_gold(takeoff_altitude_m=10.0)
        with self.assertRaisesRegex(ValueError, "cannot be None"):
            make_gold(
                explicit_fields={
                    "target_description",
                    "search_region",
                    "track_duration_s",
                    "landing_zone",
                    "takeoff_altitude_m",
                }
            )

    def test_unknown_concept_and_noncanonical_description_are_rejected(self) -> None:
        with self.assertRaisesRegex(TargetOntologyError, "unknown target concept"):
            self.ontology.validate_gold_spec(make_gold(target_concept_id="unknown"))
        with self.assertRaisesRegex(TargetOntologyError, "canonical description"):
            self.ontology.validate_gold_spec(
                make_gold(target_description="红衣黑包的人")
            )

    def test_target_attribute_mapping_is_a_defensive_readonly_copy(self) -> None:
        source = {"upper_clothing_color": "red", "backpack_color": "black"}
        concept = TargetConcept(
            concept_id="test",
            category="person",
            attributes=source,
            canonical_description=CANONICAL,
        )
        source["upper_clothing_color"] = "blue"
        self.assertEqual(concept.attributes["upper_clothing_color"], "red")
        with self.assertRaises(TypeError):
            concept.attributes["upper_clothing_color"] = "blue"  # type: ignore[index]

    def test_gold_from_dict_is_strict_and_not_model_derived(self) -> None:
        raw = make_gold().to_dict()
        self.assertEqual(GoldPlannerSpec.from_dict(raw), make_gold())
        raw["oracle_target_pose"] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            GoldPlannerSpec.from_dict(raw)


class PlannerWorldCaseTest(unittest.TestCase):
    def test_world_case_is_public_semantics_only_and_defensive(self) -> None:
        regions = {
            "east_area": "场地东侧",
            "west_area": "场地西侧",
            "north_area": "场地北侧",
            "south_area": "场地南侧",
        }
        zones = {"home": "起点", "north_pad": "北侧平台", "south_pad": "南侧平台"}
        world = PlannerWorldCase(
            context_id="world_01",
            search_regions=regions,
            landing_zones=zones,
            default_takeoff_altitude_m=10,
            default_track_duration_s=30,
            scene_min_xyz_m=(-50, -50, 0),
            scene_max_xyz_m=(50, 50, 30),
        )
        regions["east_area"] = "tampered"
        zones["home"] = "tampered"
        self.assertEqual(world.search_regions["east_area"], "场地东侧")
        self.assertEqual(world.landing_zones["home"], "起点")
        with self.assertRaises(TypeError):
            world.search_regions["x"] = "x"  # type: ignore[index]
        serialized = json.dumps(world.to_dict(), ensure_ascii=False, allow_nan=False)
        for forbidden in (
            "spawn",
            "oracle",
            "target_id",
            "target_velocity",
            "evaluator",
        ):
            self.assertNotIn(forbidden, serialized.lower())

    def test_world_requires_multiple_regions_and_landing_zones(self) -> None:
        common = dict(
            context_id="world",
            default_takeoff_altitude_m=10,
            default_track_duration_s=30,
            scene_min_xyz_m=(0, 0, 0),
            scene_max_xyz_m=(10, 10, 10),
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            PlannerWorldCase(
                search_regions={"only": "one"},
                landing_zones={"a": "a", "b": "b"},
                **common,
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            PlannerWorldCase(
                search_regions={"a": "a", "b": "b"},
                landing_zones={"only": "one"},
                **common,
            )


if __name__ == "__main__":
    unittest.main()
