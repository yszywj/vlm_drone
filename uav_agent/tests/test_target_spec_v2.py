from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import unittest

from perception.prompt_types import DeterministicPromptCompiler, TargetPromptAdapter
from target import TargetManager, TargetSpec


class TargetSpecV2Test(unittest.TestCase):
    def test_legacy_constructor_and_keyword_remain_compatible(self) -> None:
        positional = TargetSpec(" moving target ")
        keyword = TargetSpec(description="moving target")

        self.assertEqual(positional.description, "moving target")
        self.assertEqual(positional, keyword)
        self.assertEqual(positional.original_description, "moving target")
        self.assertEqual(positional.immutable_identity_summary, "moving target")

    def test_full_spec_is_strict_and_json_compatible(self) -> None:
        spec = TargetSpec(
            "person in a red coat",
            category="person",
            hard_attributes=("red coat", "black backpack"),
            soft_attributes=("adult",),
            negative_constraints=("not blue coat",),
            relation_constraints=("near the gate",),
            query_ladder=("red-coated person", "person with backpack"),
            inspection_questions=("Is the backpack black?",),
            immutable_identity_summary="red coat and black backpack",
            mutable_appearance_notes=("coat partly occluded",),
        )

        self.assertEqual(json.loads(json.dumps(spec.to_dict())), spec.to_dict())
        with self.assertRaises(FrozenInstanceError):
            spec.original_description = "different target"  # type: ignore[misc]
        with self.assertRaises(ValueError):
            TargetSpec("target", hard_attributes=("same", "same"))
        with self.assertRaises(ValueError):
            TargetSpec("target", category=" ")

    def test_exact_schema_v2_parser_rejects_missing_and_unknown_fields(self) -> None:
        encoded = TargetSpec(
            "red vehicle",
            category="vehicle",
            hard_attributes=("roof rack",),
            immutable_identity_summary="red vehicle with roof rack",
        ).to_dict()

        self.assertEqual(TargetSpec.from_dict(encoded).to_dict(), encoded)
        missing = dict(encoded)
        missing.pop("query_ladder")
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            TargetSpec.from_dict(missing)
        unknown = dict(encoded)
        unknown["oracle_target_position"] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            TargetSpec.from_dict(unknown)

    def test_appearance_update_cannot_change_identity(self) -> None:
        original = TargetSpec(
            "red vehicle",
            category="vehicle",
            immutable_identity_summary="red vehicle with roof rack",
        )
        updated = original.append_appearance_note("mud on left side")

        self.assertEqual(original.mutable_appearance_notes, ())
        self.assertEqual(updated.mutable_appearance_notes, ("mud on left side",))
        self.assertEqual(
            updated.immutable_identity_summary,
            original.immutable_identity_summary,
        )
        self.assertEqual(updated.original_description, original.original_description)

    def test_prompt_bundle_is_deterministic_and_uses_no_hidden_state(self) -> None:
        spec = TargetSpec(
            "red vehicle",
            category="vehicle",
            hard_attributes=("roof rack",),
            soft_attributes=("dusty",),
            negative_constraints=("not a blue vehicle",),
            query_ladder=("red car",),
            immutable_identity_summary="red vehicle with roof rack",
            mutable_appearance_notes=("currently shadowed",),
        )
        compiler = DeterministicPromptCompiler()

        first = compiler.compile(spec, reference_handles=("frame_1",))
        second = compiler.compile(spec, reference_handles=("frame_1",))

        self.assertIsInstance(compiler, TargetPromptAdapter)
        self.assertEqual(first, second)
        self.assertEqual(first.reference_handles, ("frame_1",))
        self.assertNotIn("currently shadowed", first.positive_phrases)
        self.assertEqual(
            first.immutable_target_identity,
            "red vehicle with roof rack",
        )

    def test_target_manager_only_updates_mutable_appearance(self) -> None:
        manager = TargetManager()
        spec = TargetSpec(
            "red vehicle",
            immutable_identity_summary="red vehicle with roof rack",
        )
        manager.start_search(spec, timestamp_s=1.0)

        updated = manager.update_mutable_appearance_notes(("partly occluded",))

        self.assertEqual(updated.mutable_appearance_notes, ("partly occluded",))
        self.assertEqual(manager.target_spec, updated)
        self.assertEqual(manager.snapshot().description, "red vehicle")
        self.assertIsNotNone(manager.target_spec)
        self.assertEqual(
            manager.target_spec.immutable_identity_summary,  # type: ignore[union-attr]
            "red vehicle with roof rack",
        )

        fresh = TargetManager()
        with self.assertRaisesRegex(RuntimeError, "active target lifecycle"):
            fresh.update_mutable_appearance_notes(("not allowed",))


if __name__ == "__main__":
    unittest.main()
