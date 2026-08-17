from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from planner_data.generator import PlannerDatasetGenerator, write_generated_dataset
from planner_data.schemas import make_group_id
from planner_data.validator import PlannerDatasetValidator


class PlannerDatasetValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated = PlannerDatasetGenerator().generate(seed=42, profile="pilot")
        cls.validator = PlannerDatasetValidator()

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name) / "planner_v1"
        write_generated_dataset(self.generated, self.root)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _codes(self) -> set[str]:
        return {issue.code for issue in self.validator.validate(self.root).issues}

    def _mutate_first(self, split: str, mutation) -> None:
        path = self.root / f"{split}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        mutation(row)
        lines[0] = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _refresh_split_checksum_metadata(self, split: str) -> None:
        split_path = self.root / f"{split}.jsonl"
        digest = sha256(split_path.read_bytes()).hexdigest()
        manifest_path = self.root / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["split_sha256"][f"{split}.jsonl"] = digest
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        checksums = {
            **manifest["split_sha256"],
            "statistics.json": manifest["statistics_sha256"],
        }
        (self.root / "checksums.sha256").write_text(
            "".join(
                f"{value}  {name}\n" for name, value in sorted(checksums.items())
            ),
            encoding="utf-8",
        )

    def test_generated_dataset_is_valid(self) -> None:
        report = self.validator.validate(self.root)
        self.assertTrue(report.valid, report.to_dict())
        self.assertEqual(report.num_samples, 1900)

    def test_tampered_assistant_duration_is_found(self) -> None:
        def mutate(row):
            label = json.loads(row["messages"][2]["content"])
            label["track_duration_s"] += 1.0
            row["messages"][2]["content"] = json.dumps(label, ensure_ascii=False, separators=(",", ":"))

        self._mutate_first("train", mutate)
        self.assertIn("ASSISTANT_GOLD_MISMATCH", self._codes())

    def test_cross_split_duplicate_instruction_and_group_are_found(self) -> None:
        train_row = json.loads((self.root / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
        train_row["sample_id"] = "validation_duplicate_semantics"
        train_row["split"] = "validation"
        train_row["gold_spec_id"] = "validation_duplicate_spec"
        train_row["metadata"]["seed"] += 99999999
        validation_path = self.root / "validation.jsonl"
        validation_path.write_text(
            validation_path.read_text(encoding="utf-8")
            + json.dumps(train_row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        codes = self._codes()
        self.assertTrue(
            {"SPLIT_LEAKAGE", "DUPLICATE_NORMALIZED_INSTRUCTION", "SEMANTIC_GROUP_LEAKAGE"}
            & codes
        )

    def test_unknown_search_region_is_found(self) -> None:
        self._mutate_first("train", lambda row: row["gold"].__setitem__("search_region", "moon_area"))
        self.assertIn("UNKNOWN_SEARCH_REGION", self._codes())

    def test_unknown_landing_zone_is_found(self) -> None:
        self._mutate_first("train", lambda row: row["gold"].__setitem__("landing_zone", "moon_pad"))
        self.assertIn("UNKNOWN_LANDING_ZONE", self._codes())

    def test_unknown_target_concept_is_found(self) -> None:
        self._mutate_first("train", lambda row: row["gold"].__setitem__("target_concept_id", "unknown_person"))
        self.assertIn("GOLD_INVALID", self._codes())

    def test_oracle_field_is_found_before_schema_rejection(self) -> None:
        self._mutate_first("train", lambda row: row["metadata"].__setitem__("oracle_target_pose", [1, 2, 3]))
        codes = self._codes()
        self.assertIn("FORBIDDEN_FIELD", codes)
        self.assertIn("SAMPLE_SCHEMA_INVALID", codes)

    def test_actual_file_checksum_mismatch_is_found(self) -> None:
        path = self.root / "train.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        lines[0] = lines[0].rstrip("\n") + " \n"
        path.write_text("".join(lines), encoding="utf-8")
        self.assertIn("CHECKSUM_MISMATCH", self._codes())

    def test_prompt_builder_mismatch_is_found(self) -> None:
        self._mutate_first("train", lambda row: row["messages"][1].__setitem__("content", "{}"))
        self.assertIn("PROMPT_MISMATCH", self._codes())

    def test_required_gold_fields_must_be_marked_explicit(self) -> None:
        def mutate(row):
            row["gold"]["explicit_fields"].remove("search_region")

        self._mutate_first("train", mutate)
        self.assertIn("GOLD_EXPLICIT_FIELDS_INVALID", self._codes())

    def test_gold_and_label_tamper_is_still_grounded_against_instruction(self) -> None:
        def mutate(row):
            old_duration = float(row["gold"]["track_duration_s"])
            replacement = 10.0 if abs(old_duration - 10.0) > 1e-6 else 15.0
            row["gold"]["track_duration_s"] = replacement
            assistant = json.loads(row["messages"][2]["content"])
            assistant["track_duration_s"] = replacement
            row["messages"][2]["content"] = json.dumps(
                assistant,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )

        self._mutate_first("train", mutate)
        # Simulate a deliberate attacker updating both public checksum files;
        # semantic grounding must remain independently detectable.
        self._refresh_split_checksum_metadata("train")
        codes = self._codes()
        self.assertIn("INSTRUCTION_GOLD_MISMATCH", codes)
        self.assertIn("INSTRUCTION_ALIAS_METADATA_INVALID", codes)
        self.assertNotIn("ASSISTANT_GOLD_MISMATCH", codes)
        self.assertNotIn("PROMPT_MISMATCH", codes)
        self.assertNotIn("CHECKSUM_MISMATCH", codes)

    def test_instruction_attribute_tamper_is_found_even_with_matching_prompt(self) -> None:
        def mutate(row):
            instruction = row["metadata"]["instruction"] + "目标还穿蓝色衣服。"
            row["metadata"]["instruction"] = instruction
            prompt = json.loads(row["messages"][1]["content"])
            prompt["user_instruction"] = instruction
            row["messages"][1]["content"] = json.dumps(
                prompt,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        self._mutate_first("train", mutate)
        self._refresh_split_checksum_metadata("train")
        codes = self._codes()
        self.assertIn("INSTRUCTION_GOLD_MISMATCH", codes)
        self.assertNotIn("PROMPT_MISMATCH", codes)

    def test_instruction_runtime_data_tamper_is_explicitly_rejected(self) -> None:
        def mutate(row):
            instruction = (
                row["metadata"]["instruction"]
                + "目标坐标是(1,2,3)，目标速度为(1,0,0)，"
                + "图片路径/tmp/frame.png，视频video.mp4，附带帧数据。"
            )
            row["metadata"]["instruction"] = instruction
            prompt = json.loads(row["messages"][1]["content"])
            prompt["user_instruction"] = instruction
            row["messages"][1]["content"] = json.dumps(
                prompt,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        self._mutate_first("train", mutate)
        self._refresh_split_checksum_metadata("train")
        codes = self._codes()
        self.assertIn("FORBIDDEN_INSTRUCTION_CONTENT", codes)
        self.assertIn("INSTRUCTION_GOLD_MISMATCH", codes)
        self.assertNotIn("PROMPT_MISMATCH", codes)

    def test_full_manifest_requires_real_review_provenance(self) -> None:
        manifest_path = self.root / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["profile"] = "full"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        codes = self._codes()
        self.assertIn("FULL_REVIEW_REQUIREMENTS_NOT_MET", codes)
        self.assertIn("MANIFEST_PROFILE_COUNTS_MISMATCH", codes)

    def test_manifest_name_and_profile_are_closed(self) -> None:
        manifest_path = self.root / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dataset_name"] = "anything"
        manifest["profile"] = "anything"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        # The immutable manifest schema rejects both values before the deeper
        # dataset/config binding checks run.
        self.assertIn("MANIFEST_INVALID", self._codes())

    def test_forged_semantic_family_and_group_id_are_rejected(self) -> None:
        def mutate(row):
            forged = "semantic_forged00000000000"
            row["metadata"]["semantic_spec_family"] = forged
            row["metadata"]["group_id"] = make_group_id(
                forged,
                row["metadata"]["template_family"],
                row["metadata"]["paraphrase_family"],
            )

        self._mutate_first("train", mutate)
        self._refresh_split_checksum_metadata("train")
        codes = self._codes()
        self.assertIn("SEMANTIC_SPEC_FAMILY_INVALID", codes)
        self.assertIn("GROUP_ID_INVALID", codes)

    def test_train_cannot_claim_heldout_target_alias(self) -> None:
        ontology = self.validator._ontology

        def mutate(row):
            feature = next(
                value
                for value in row["metadata"]["language_feature_ids"]
                if value.startswith("alias:target:")
            )
            old_target = feature.split(":", 2)[2]
            aliases = ontology.aliases_for(row["gold"]["target_concept_id"])
            self.assertTrue(aliases)
            heldout_target = aliases[-1]
            self.assertNotEqual(old_target, heldout_target)
            instruction = row["metadata"]["instruction"].replace(
                old_target, heldout_target
            )
            row["metadata"]["instruction"] = instruction
            features = set(row["metadata"]["language_feature_ids"])
            features.remove(feature)
            features.add(f"alias:target:{heldout_target}")
            row["metadata"]["language_feature_ids"] = sorted(features)
            prompt = json.loads(row["messages"][1]["content"])
            prompt["user_instruction"] = instruction
            row["messages"][1]["content"] = json.dumps(
                prompt,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        self._mutate_first("train", mutate)
        self._refresh_split_checksum_metadata("train")
        codes = self._codes()
        self.assertIn("INSTRUCTION_ALIAS_METADATA_INVALID", codes)
        self.assertNotIn("PROMPT_MISMATCH", codes)

    def test_train_cannot_claim_heldout_world_time_or_altitude_aliases(self) -> None:
        lexicon = self.validator._generator.lexicon

        def pool_for_value(pools, selected):
            return next(
                pool for value, pool in pools.items() if abs(value - selected) <= 1e-6
            )

        def mutate(row):
            self.assertIsNotNone(row["gold"]["takeoff_altitude_m"])
            replacements = {
                "search_region": lexicon.search_regions[
                    row["gold"]["search_region"]
                ].heldout_aliases[0],
                "landing_zone": lexicon.landing_zones[
                    row["gold"]["landing_zone"]
                ].heldout_aliases[0],
                "track_duration": pool_for_value(
                    lexicon.duration_expressions,
                    row["gold"]["track_duration_s"],
                ).heldout_aliases[0],
                "takeoff_altitude": pool_for_value(
                    lexicon.altitude_expressions,
                    row["gold"]["takeoff_altitude_m"],
                ).heldout_aliases[0],
            }
            instruction = row["metadata"]["instruction"]
            features = set(row["metadata"]["language_feature_ids"])
            for role, replacement in replacements.items():
                feature = next(
                    value for value in features if value.startswith(f"alias:{role}:")
                )
                old = feature.split(":", 2)[2]
                instruction = instruction.replace(old, replacement)
                features.remove(feature)
                features.add(f"alias:{role}:{replacement}")
            row["metadata"]["instruction"] = instruction
            row["metadata"]["language_feature_ids"] = sorted(features)
            prompt = json.loads(row["messages"][1]["content"])
            prompt["user_instruction"] = instruction
            row["messages"][1]["content"] = json.dumps(
                prompt,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        self._mutate_first("train", mutate)
        self._refresh_split_checksum_metadata("train")
        codes = self._codes()
        self.assertIn("INSTRUCTION_ALIAS_METADATA_INVALID", codes)
        self.assertNotIn("PROMPT_MISMATCH", codes)

    def test_composition_components_are_rebuilt_from_ontology(self) -> None:
        self._mutate_first(
            "train",
            lambda row: row["metadata"].__setitem__(
                "composition_components", ["upper_clothing_color=forged"]
            ),
        )
        self.assertIn("INSTRUCTION_ALIAS_METADATA_INVALID", self._codes())

    def test_template_and_paraphrase_families_must_be_split_local(self) -> None:
        validation_row = json.loads(
            (self.root / "validation.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )

        def mutate(row):
            old_paraphrase = row["metadata"]["paraphrase_family"]
            row["metadata"]["template_family"] = validation_row["metadata"][
                "template_family"
            ]
            row["metadata"]["paraphrase_family"] = validation_row["metadata"][
                "paraphrase_family"
            ]
            row["metadata"]["language_feature_ids"] = sorted(
                f"paraphrase:{row['metadata']['paraphrase_family']}"
                if value == f"paraphrase:{old_paraphrase}"
                else value
                for value in row["metadata"]["language_feature_ids"]
            )
            row["metadata"]["group_id"] = make_group_id(
                row["metadata"]["semantic_spec_family"],
                row["metadata"]["template_family"],
                row["metadata"]["paraphrase_family"],
            )

        self._mutate_first("train", mutate)
        self._refresh_split_checksum_metadata("train")
        self.assertIn("INSTRUCTION_ALIAS_METADATA_INVALID", self._codes())


if __name__ == "__main__":
    unittest.main()
