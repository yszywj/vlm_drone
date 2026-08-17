from __future__ import annotations

import unittest

from planner_data.schemas import (
    DatasetManifest,
    PLANNER_DATASET_SCHEMA_VERSION,
    PLANNER_DATASET_SPLITS,
    PlannerDataSchemaError,
    compute_semantic_spec_family,
)
from tasks.schemas import GoldPlannerSpec


def gold_spec() -> GoldPlannerSpec:
    return GoldPlannerSpec(
        spec_id="gold_1",
        target_concept_id="person_red_black",
        target_description="穿红色上衣和黑色裤子的人",
        search_region="search_area",
        track_duration_s=30.0,
        landing_zone="home",
        takeoff_altitude_m=10.0,
        explicit_fields=frozenset(
            {
                "target_description",
                "search_region",
                "track_duration_s",
                "landing_zone",
                "takeoff_altitude_m",
            }
        ),
    )


def manifest(**overrides: object) -> DatasetManifest:
    values: dict[str, object] = {
        "schema_version": PLANNER_DATASET_SCHEMA_VERSION,
        "dataset_name": "planner_v1",
        "profile": "pilot",
        "seed": 42,
        "split_counts": {name: 0 for name in PLANNER_DATASET_SPLITS},
        "resource_sha256": {"resource.yaml": "a" * 64},
        "split_sha256": {name: "b" * 64 for name in PLANNER_DATASET_SPLITS},
        "statistics_sha256": "c" * 64,
        "generated_at_utc": "deterministic-from-seed-42",
    }
    values.update(overrides)
    return DatasetManifest(**values)


class PlannerDataSchemaTests(unittest.TestCase):
    def test_semantic_family_has_exact_public_canonical_payload(self) -> None:
        self.assertEqual(
            compute_semantic_spec_family(gold_spec(), "world_a"),
            "semantic_2082cccecb658354ebc0",
        )

    def test_duration_explicitness_changes_semantic_family(self) -> None:
        explicit = gold_spec()
        implicit = GoldPlannerSpec(
            spec_id="gold_2",
            target_concept_id=explicit.target_concept_id,
            target_description=explicit.target_description,
            search_region=explicit.search_region,
            track_duration_s=explicit.track_duration_s,
            landing_zone=explicit.landing_zone,
            takeoff_altitude_m=10.0,
            explicit_fields=explicit.explicit_fields - {"track_duration_s"},
        )
        self.assertNotEqual(
            compute_semantic_spec_family(explicit, "world_a"),
            compute_semantic_spec_family(implicit, "world_a"),
        )

    def test_manifest_accepts_only_planner_v1_name(self) -> None:
        with self.assertRaisesRegex(PlannerDataSchemaError, "dataset_name"):
            manifest(dataset_name="other")

    def test_manifest_accepts_only_public_profiles(self) -> None:
        with self.assertRaisesRegex(PlannerDataSchemaError, "profile"):
            manifest(profile="debug")
        self.assertEqual(manifest(profile="full").profile, "full")


if __name__ == "__main__":
    unittest.main()
