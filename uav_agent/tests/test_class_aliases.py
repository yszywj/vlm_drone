from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from perception.class_aliases import (
    ClassAliasConfigError,
    ClassAliasMapper,
    UNSUPPORTED_TARGET_CATEGORY,
    UnsupportedTargetCategory,
    compile_target_query,
)
from target import TargetSpec


ALIASES = """\
person:
  aliases:
    - person
    - pedestrian
    - 人
car:
  aliases:
    - car
    - 汽车
"""


def mapper_from_text(value: str = ALIASES) -> ClassAliasMapper:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "aliases.yaml"
        path.write_text(value, encoding="utf-8")
        return ClassAliasMapper.from_yaml(path)


class ClassAliasMapperTest(unittest.TestCase):
    def test_exact_alias_resolves_against_runtime_model_names(self) -> None:
        mapper = mapper_from_text()
        result = mapper.resolve("人", {0: "person", 2: "car"})

        self.assertEqual(result.class_id, 0)
        self.assertEqual(result.class_name, "person")
        self.assertEqual(result.canonical_name, "person")
        self.assertEqual(result.matched_alias, "人")

    def test_lookup_is_exact_not_substring(self) -> None:
        mapper = mapper_from_text()
        with self.assertRaises(UnsupportedTargetCategory) as captured:
            mapper.resolve("red person", {0: "person"})
        self.assertEqual(captured.exception.code, UNSUPPORTED_TARGET_CATEGORY)

    def test_alias_target_must_exist_in_loaded_model_names(self) -> None:
        mapper = mapper_from_text()
        with self.assertRaisesRegex(
            UnsupportedTargetCategory,
            "absent from loaded model.names",
        ):
            mapper.resolve("person", {0: "car"})

    def test_yaml_is_strict_and_rejects_ambiguous_or_unknown_fields(self) -> None:
        invalid = (
            "person:\n  aliases: [person]\n  fuzzy: true\n",
            "person:\n  aliases: [person, pedestrian]\ncar:\n  aliases: [pedestrian]\n",
            "person:\n  aliases: [person]\nperson:\n  aliases: [human]\n",
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ClassAliasConfigError):
                    mapper_from_text(value)

    def test_ordinary_yolo_uses_only_closed_set_class_ids(self) -> None:
        mapper = mapper_from_text()
        query = compile_target_query(
            TargetSpec(
                "person",
                category="person",
                hard_attributes=("wearing red",),
            ),
            "yolo",
            {0: "person", 1: "car"},
            mapper,
        )
        self.assertEqual(query.class_ids, (0,))
        self.assertEqual(query.text_prompts, ())

    def test_red_cube_is_explicitly_unsupported_by_unmodified_closed_set(self) -> None:
        mapper = mapper_from_text()
        with self.assertRaises(UnsupportedTargetCategory) as captured:
            compile_target_query(
                TargetSpec("red cube", category="red_cube"),
                "yolo",
                {0: "person", 1: "car"},
                mapper,
            )
        self.assertIn(UNSUPPORTED_TARGET_CATEGORY, str(captured.exception))

    def test_yoloe_compiles_open_vocabulary_prompts_without_alias_fallback(self) -> None:
        mapper = mapper_from_text()
        query = compile_target_query(
            TargetSpec(
                "red cube",
                category="red_cube",
                hard_attributes=("red",),
                query_ladder=("red cube", "cube"),
            ),
            "yoloe",
            {0: "placeholder"},
            mapper,
        )
        self.assertEqual(query.class_ids, ())
        self.assertIn("red cube", query.text_prompts)
        self.assertIn("red_cube", query.text_prompts)

    def test_repository_alias_file_loads(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "yolo" / "class_aliases.yaml"
        mapper = ClassAliasMapper.from_yaml(path)
        self.assertIn("person", mapper.canonical_names)


if __name__ == "__main__":
    unittest.main()
