from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from planner.prompt_builder import build_mission_planner_messages
from planner_data.renderers import (
    DATASET_SPLITS,
    InstructionRenderer,
    RendererConfigError,
    load_language_lexicon,
    load_template_catalog,
    load_world_cases,
    world_case_to_runtime_context,
)
from tasks.schemas import GoldPlannerSpec
from tasks.target_ontology import TargetOntology


class InstructionRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ontology = TargetOntology.load_default()
        cls.lexicon = load_language_lexicon()
        cls.catalog = load_template_catalog()
        cls.worlds = load_world_cases()
        cls.world = cls.worlds["world_01"]
        cls.renderer = InstructionRenderer(
            cls.ontology,
            cls.lexicon,
            cls.catalog,
        )
        cls.concept = cls.ontology.require_concept(
            "person_upper_red_backpack_black"
        )

    def _gold(
        self,
        *,
        spec_id: str = "spec_renderer",
        duration: float = 30.0,
        duration_explicit: bool = True,
        altitude: float | None = None,
    ) -> GoldPlannerSpec:
        explicit = {
            "target_description",
            "search_region",
            "landing_zone",
        }
        if duration_explicit:
            explicit.add("track_duration_s")
        if altitude is not None:
            explicit.add("takeoff_altitude_m")
        return GoldPlannerSpec(
            spec_id=spec_id,
            target_concept_id=self.concept.concept_id,
            target_description=self.concept.canonical_description,
            search_region="east_area",
            track_duration_s=duration,
            landing_zone="home",
            takeoff_altitude_m=altitude,
            explicit_fields=frozenset(explicit),
        )

    def test_resources_define_multiple_public_choices(self) -> None:
        self.assertGreaterEqual(len(self.worlds), 1)
        for world in self.worlds.values():
            self.assertGreaterEqual(len(world.search_regions), 4)
            self.assertGreaterEqual(len(world.landing_zones), 3)
            serialized = repr(world.to_dict()).casefold()
            for forbidden in (
                "oracle_target",
                "target_spawn",
                "target_velocity",
                "evaluator_frame",
                "camera_rgb",
            ):
                self.assertNotIn(forbidden, serialized)

    def test_lexicon_covers_every_world_name(self) -> None:
        for world in self.worlds.values():
            self.assertTrue(
                set(world.search_regions).issubset(self.lexicon.search_regions)
            )
            self.assertTrue(
                set(world.landing_zones).issubset(self.lexicon.landing_zones)
            )

    def test_dataset_config_has_exact_pilot_and_full_sizes(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "resources"
            / "planner_v1"
            / "dataset_config.yaml"
        )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["schema_version"], "planner_dataset_config_v1")
        pilot = config["profiles"]["pilot"]["split_counts"]
        full = config["profiles"]["full"]["split_counts"]
        self.assertEqual(
            pilot,
            {
                "train": 1000,
                "validation": 200,
                "test_iid": 200,
                "test_compositional": 200,
                "test_language": 200,
                "test_robustness": 100,
            },
        )
        self.assertEqual(
            full,
            {
                "train": 8000,
                "validation": 1000,
                "test_iid": 1000,
                "test_compositional": 1000,
                "test_language": 1000,
                "test_robustness": 500,
            },
        )
        generation = config["generation"]
        self.assertTrue(
            set(map(float, generation["track_durations_s"])).issubset(
                self.lexicon.duration_expressions
            )
        )
        self.assertTrue(
            set(map(float, generation["takeoff_altitudes_m"])).issubset(
                self.lexicon.altitude_expressions
            )
        )
        self.assertTrue(
            set(generation["compositional_holdout_concepts"]).issubset(
                self.ontology.concepts
            )
        )

    def test_train_and_heldout_aliases_are_disjoint_and_unambiguous(self) -> None:
        for named_pools in (
            self.lexicon.search_regions,
            self.lexicon.landing_zones,
        ):
            all_aliases: dict[str, str] = {}
            for name, pool in named_pools.items():
                self.assertTrue(set(pool.train_aliases).isdisjoint(pool.heldout_aliases))
                for alias in (*pool.train_aliases, *pool.heldout_aliases):
                    self.assertNotIn(alias, all_aliases)
                    all_aliases[alias] = name

    def test_template_catalog_covers_every_split_and_explicit_mode(self) -> None:
        for split in DATASET_SPLITS:
            for duration_mode in ("explicit", "default"):
                for altitude_mode in ("explicit", "default"):
                    self.assertTrue(
                        self.catalog.for_split(
                            split,
                            duration_mode=duration_mode,
                            altitude_mode=altitude_mode,
                        )
                    )

    def test_ordinary_template_families_and_skeletons_are_split_isolated(self) -> None:
        ordinary = ("train", "validation", "test_iid", "test_compositional")
        families: dict[str, str] = {}
        skeletons: dict[str, str] = {}
        for split in ordinary:
            templates = [
                template
                for template in self.catalog.templates
                if split in template.splits
            ]
            for template in templates:
                self.assertEqual(template.splits, (split,))
                self.assertNotIn(template.template_family, families)
                self.assertNotIn(template.text, skeletons)
                families[template.template_family] = split
                skeletons[template.text] = split

    def test_same_seed_and_variant_are_reproducible(self) -> None:
        gold = self._gold()
        first = self.renderer.render(gold, self.world, split="train", seed=42)
        second = self.renderer.render(gold, self.world, split="train", seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_seed_and_variant_produce_language_variety(self) -> None:
        gold = self._gold()
        instructions = {
            self.renderer.render(
                gold,
                self.world,
                split="train",
                seed=seed,
                variant_index=variant,
            ).instruction
            for seed in range(5)
            for variant in range(20)
        }
        self.assertGreater(len(instructions), 20)

    def test_rendered_instruction_contains_every_explicit_semantic_alias(self) -> None:
        gold = self._gold(duration=30.0)
        rendered = self.renderer.render(
            gold,
            self.world,
            split="train",
            seed=101,
        )
        for role in (
            "target",
            "search_region",
            "track_duration",
            "landing_zone",
        ):
            self.assertIn(rendered.aliases[role], rendered.instruction)
        self.assertNotIn("takeoff_altitude", rendered.aliases)
        self.assertNotIn("robustness_injection", rendered.aliases)

    def test_default_duration_expression_is_used_only_for_world_default(self) -> None:
        gold = self._gold(duration_explicit=False)
        rendered = self.renderer.render(
            gold,
            self.world,
            split="train",
            seed=7,
        )
        self.assertIn(
            rendered.aliases["track_duration"],
            self.lexicon.default_duration_expressions.train_aliases,
        )
        with self.assertRaisesRegex(RendererConfigError, "world default"):
            self.renderer.render(
                self._gold(duration=20.0, duration_explicit=False),
                self.world,
                split="train",
                seed=7,
            )

    def test_explicit_altitude_is_expressed_without_changing_gold(self) -> None:
        gold = self._gold(altitude=10.0)
        before = gold.to_dict()
        rendered = self.renderer.render(
            gold,
            self.world,
            split="validation",
            seed=19,
        )
        self.assertIn("takeoff_altitude", rendered.aliases)
        self.assertIn(
            rendered.aliases["takeoff_altitude"],
            rendered.instruction,
        )
        self.assertEqual(before, gold.to_dict())

    def test_test_language_uses_only_heldout_region_and_landing_aliases(self) -> None:
        for variant in range(40):
            rendered = self.renderer.render(
                self._gold(spec_id=f"lang_{variant}"),
                self.world,
                split="test_language",
                seed=42,
                variant_index=variant,
            )
            self.assertIn(
                rendered.aliases["search_region"],
                self.lexicon.search_regions["east_area"].heldout_aliases,
            )
            self.assertIn(
                rendered.aliases["landing_zone"],
                self.lexicon.landing_zones["home"].heldout_aliases,
            )
            self.assertIn(
                rendered.aliases["track_duration"],
                self.lexicon.duration_expressions[30.0].heldout_aliases,
            )

    def test_final_target_alias_is_strictly_held_out_for_language_test(self) -> None:
        ordinary_splits = (
            "train",
            "validation",
            "test_iid",
            "test_compositional",
        )
        for index, concept in enumerate(self.ontology.concepts.values()):
            aliases = self.ontology.aliases_for(concept.concept_id)
            if not aliases:
                continue
            gold = GoldPlannerSpec(
                spec_id=f"alias_holdout_{index}",
                target_concept_id=concept.concept_id,
                target_description=concept.canonical_description,
                search_region="east_area",
                track_duration_s=30.0,
                landing_zone="home",
                takeoff_altitude_m=None,
                explicit_fields=frozenset(
                    {
                        "target_description",
                        "search_region",
                        "track_duration_s",
                        "landing_zone",
                    }
                ),
            )
            heldout = aliases[-1]
            language = self.renderer.render(
                gold,
                self.world,
                split="test_language",
                seed=42,
            )
            self.assertEqual(language.aliases["target"], heldout)
            for split in ordinary_splits:
                for variant in range(8):
                    ordinary = self.renderer.render(
                        gold,
                        self.world,
                        split=split,
                        seed=42,
                        variant_index=variant,
                    )
                    self.assertNotEqual(ordinary.aliases["target"], heldout)

    def test_prompt_injection_is_isolated_to_robustness_split(self) -> None:
        gold = self._gold()
        for split in DATASET_SPLITS:
            rendered = self.renderer.render(gold, self.world, split=split, seed=5)
            if split == "test_robustness":
                self.assertIn("robustness_injection", rendered.aliases)
                self.assertIsNotNone(rendered.robustness_category)
                self.assertIn(
                    rendered.aliases["robustness_injection"],
                    rendered.instruction,
                )
            else:
                self.assertNotIn("robustness_injection", rendered.aliases)
                self.assertFalse(
                    any(
                        injection in rendered.instruction
                        for injection in (
                            item.text for item in self.lexicon.robustness_injections
                        )
                    )
                )

    def test_robustness_categories_are_structured_and_complete(self) -> None:
        required = {
            "prompt_injection",
            "extra_field",
            "format_interference",
            "irrelevant_text",
            "long_instruction",
            "repeated_requirement",
        }
        self.assertTrue(
            required.issubset(
                {item.category for item in self.lexicon.robustness_injections}
            )
        )
        rendered_categories = {
            self.renderer.render(
                self._gold(spec_id=f"robust_{index}"),
                self.world,
                split="test_robustness",
                seed=index,
            ).robustness_category
            for index in range(200)
        }
        self.assertTrue(required.issubset(rendered_categories))

    def test_all_planner_resource_loaders_reject_duplicate_yaml_keys(self) -> None:
        resource_root = Path(__file__).resolve().parents[2] / "resources" / "planner_v1"
        cases = (
            ("language_lexicon_zh.yaml", "\nlanguage: zh-CN\n", load_language_lexicon),
            ("templates_zh.yaml", "\nlanguage: zh-CN\n", load_template_catalog),
            ("world_contexts.yaml", "\nworlds: []\n", load_world_cases),
        )
        with tempfile.TemporaryDirectory() as directory:
            for filename, duplicate, loader in cases:
                with self.subTest(filename=filename):
                    source = resource_root / filename
                    target = Path(directory) / filename
                    target.write_text(
                        source.read_text(encoding="utf-8") + duplicate,
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(RendererConfigError, "duplicate key"):
                        loader(target)

    def test_lexicon_rejects_cross_domain_and_numeric_alias_ambiguity(self) -> None:
        resource_root = Path(__file__).resolve().parents[2] / "resources" / "planner_v1"
        raw = yaml.safe_load(
            (resource_root / "language_lexicon_zh.yaml").read_text(encoding="utf-8")
        )
        mutations = (
            lambda value: value["landing_zones"]["north_pad"]["train_aliases"].__setitem__(
                0, value["search_regions"]["east_area"]["train_aliases"][0]
            ),
            lambda value: value["search_regions"]["east_area"]["train_aliases"].__setitem__(
                0, "north_pad"
            ),
            lambda value: value["duration_expressions"]["15"]["train_aliases"].__setitem__(
                0, value["duration_expressions"]["10"]["train_aliases"][0]
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, mutate in enumerate(mutations):
                with self.subTest(case=index):
                    # Round-trip produces an independent nested structure.
                    candidate = yaml.safe_load(
                        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
                    )
                    mutate(candidate)
                    path = Path(directory) / f"ambiguous_{index}.yaml"
                    path.write_text(
                        yaml.safe_dump(candidate, allow_unicode=True, sort_keys=False),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(RendererConfigError, "alias|ambiguous"):
                        load_language_lexicon(path)

    def test_common_prompt_builder_sees_only_public_world_projection(self) -> None:
        context = world_case_to_runtime_context(self.world)
        messages = build_mission_planner_messages(
            "请执行搜索任务",
            context,
            "Only output JSON.",
        )
        joined = "\n".join(message.content for message in messages)
        self.assertIn("east_area", joined)
        self.assertIn(self.world.search_regions["east_area"], joined)
        for forbidden in (
            "oracle_target",
            "target_spawn",
            "evaluator_frame",
            "gold_spec",
            self.concept.concept_id,
        ):
            self.assertNotIn(forbidden, joined.casefold())

    def test_loaded_mappings_are_defensive(self) -> None:
        with self.assertRaises(TypeError):
            self.world.search_regions["new_area"] = "bad"  # type: ignore[index]
        with self.assertRaises(TypeError):
            self.lexicon.search_regions["new_area"] = object()  # type: ignore[index]

    def test_world_resource_has_no_oracle_or_target_truth_fields(self) -> None:
        path = Path(__file__).resolve().parents[2] / "resources" / "planner_v1" / "world_contexts.yaml"
        text = path.read_text(encoding="utf-8")
        # Comments are ignored; parsed keys and values are checked above.  This
        # source-level check covers common machine-readable forbidden tokens.
        content_lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
        content = "\n".join(content_lines).casefold()
        for forbidden in (
            "oracle_target",
            "target_spawn",
            "target_velocity",
            "evaluator_frame",
            "camera_rgb",
        ):
            self.assertNotIn(forbidden, content)


if __name__ == "__main__":
    unittest.main()
