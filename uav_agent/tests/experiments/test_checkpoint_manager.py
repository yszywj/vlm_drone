from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest

from experiments.checkpoint_manager import (
    CheckpointFormatError,
    CheckpointManager,
    CheckpointSecurityError,
)


def metrics(
    success: float,
    false_lock: float = 0.1,
    collision: float = 0.1,
    safety_abort: float = 0.1,
    mission_time: float = 60.0,
) -> dict[str, float]:
    return {
        "mission_success_rate": success,
        "false_lock_rate": false_lock,
        "collision_rate": collision,
        "safety_abort_rate": safety_abort,
        "mean_mission_time_s": mission_time,
    }


class CheckpointManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "checkpoints"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_latest_is_overwritten_and_retains_resume_state(self) -> None:
        manager = CheckpointManager(self.root, latest_interval_steps=10)
        rng_state = random.Random(42).getstate()
        manager.save_latest(
            global_step=10,
            update=2,
            payload={"adapter": b"first", "optimizer_state": {"lr": 0.1}},
        )
        manager.save_latest(
            global_step=20,
            update=4,
            payload={
                "adapter": b"second",
                "optimizer_state": {"lr": 0.05},
                "rng_state": rng_state,
            },
        )

        loaded = manager.load("latest")
        self.assertEqual(loaded["adapter"], b"second")
        self.assertEqual(loaded["global_step"], 20)
        self.assertEqual(loaded["update"], 4)
        self.assertEqual(loaded["rng_state"], rng_state)
        self.assertTrue(manager.latest_meta["has_optimizer_state"])
        self.assertEqual(
            sorted(path.name for path in self.root.iterdir()), ["latest"]
        )

    def test_maybe_latest_obeys_interval_and_force(self) -> None:
        manager = CheckpointManager(self.root, latest_interval_steps=10)
        self.assertIsNone(manager.maybe_save_latest(global_step=9))
        self.assertEqual(
            manager.maybe_save_latest(global_step=10), manager.latest_path
        )
        self.assertIsNone(manager.maybe_save_latest(global_step=19))
        self.assertEqual(
            manager.maybe_save_latest(global_step=19, force=True),
            manager.latest_path,
        )
        self.assertEqual(manager.latest_meta["global_step"], 19)

    def test_best_uses_all_documented_tie_breakers_in_order(self) -> None:
        manager = CheckpointManager(self.root)
        candidates = [
            (1, metrics(0.7, 0.2, 0.2, 0.2, 90.0)),
            (2, metrics(0.8, 0.9, 0.9, 0.9, 999.0)),  # success wins
            (3, metrics(0.8, 0.8, 0.9, 0.9, 999.0)),  # false lock
            (4, metrics(0.8, 0.8, 0.8, 0.9, 999.0)),  # collision
            (5, metrics(0.8, 0.8, 0.8, 0.8, 999.0)),  # safety abort
            (6, metrics(0.8, 0.8, 0.8, 0.8, 90.0)),  # mission time
        ]
        for step, candidate_metrics in candidates:
            self.assertEqual(
                manager.maybe_save_best(
                    global_step=step,
                    metrics=candidate_metrics,
                    payload={"adapter": f"step-{step}"},
                ),
                manager.best_path,
            )
        self.assertEqual(manager.load("best")["adapter"], "step-6")

        self.assertIsNone(
            manager.maybe_save_best(
                global_step=7,
                metrics=metrics(0.8, 0.8, 0.8, 0.8, 91.0),
                payload={"adapter": "worse"},
            )
        )
        self.assertEqual(manager.best_meta["global_step"], 6)

    def test_best_drops_training_state_while_latest_keeps_it(self) -> None:
        manager = CheckpointManager(self.root)
        payload = {
            "adapter": "weights",
            "optimizer_state": {"momentum": 1},
            "scheduler_state": {"epoch": 2},
            "rng_state": (1, 2, 3),
        }
        manager.save_latest(global_step=5, payload=payload)
        manager.save_best(global_step=5, metrics=metrics(0.5), payload=payload)

        latest = manager.load("latest")
        best = manager.load("best")
        self.assertIn("optimizer_state", latest)
        self.assertIn("scheduler_state", latest)
        self.assertIn("rng_state", latest)
        self.assertNotIn("optimizer_state", best)
        self.assertNotIn("scheduler_state", best)
        self.assertNotIn("rng_state", best)
        self.assertFalse(manager.best_meta["has_optimizer_state"])

    def test_atomic_callback_failure_keeps_previous_checkpoint(self) -> None:
        calls = 0

        def writer(path: Path, payload: object) -> None:
            nonlocal calls
            calls += 1
            (path / "adapter.txt").write_text(str(payload["adapter"]), encoding="utf-8")
            if calls == 2:
                raise RuntimeError("simulated serialization failure")

        def loader(path: Path) -> dict[str, str]:
            return {
                "adapter": (path / "adapter.txt").read_text(encoding="utf-8")
            }

        manager = CheckpointManager(
            self.root, save_callback=writer, load_callback=loader
        )
        manager.save_latest(global_step=1, payload={"adapter": "old"})
        with self.assertRaisesRegex(RuntimeError, "serialization failure"):
            manager.save_latest(global_step=2, payload={"adapter": "new"})

        self.assertEqual(manager.load("latest")["adapter"], "old")
        self.assertEqual(manager.latest_meta["global_step"], 1)
        self.assertFalse(any(".tmp-" in path.name for path in self.root.iterdir()))

    def test_callback_receives_filtered_read_only_best_payload(self) -> None:
        observed: dict[str, object] = {}

        def writer(path: Path, payload: object) -> None:
            observed.update(payload)
            with self.assertRaises(TypeError):
                payload["mutate"] = True
            (path / "adapter_model.safetensors").write_bytes(b"adapter")

        manager = CheckpointManager(self.root, save_callback=writer)
        manager.save_best(
            global_step=1,
            metrics=metrics(0.9),
            payload={"adapter": "ok", "optimizer_state": "not-for-best"},
        )
        self.assertEqual(observed["adapter"], "ok")
        self.assertNotIn("optimizer_state", observed)
        self.assertTrue(
            (manager.best_path / "adapter_model.safetensors").is_file()
        )

    def test_rejects_full_base_model_configuration_and_payload(self) -> None:
        with self.assertRaises(CheckpointSecurityError):
            CheckpointManager(self.root, save_full_base_model=True)

        manager = CheckpointManager(self.root)
        with self.assertRaises(CheckpointSecurityError):
            manager.save_latest(
                global_step=1,
                payload={"base_model_state_dict": {"qwen": "weights"}},
            )
        with self.assertRaises(CheckpointSecurityError):
            manager.save_latest(
                global_step=1,
                payload={"model-00001-of-00002.safetensors": b"weights"},
            )
        with self.assertRaises(CheckpointSecurityError):
            manager.save_latest(
                global_step=1,
                payload={"nested": {"model_state_dict": {"qwen": "weights"}}},
            )
        with self.assertRaises(CheckpointSecurityError):
            manager.save_latest(
                global_step=1,
                payload={"nested": ["model-00002-of-00002.safetensors"]},
            )
        with self.assertRaises(CheckpointSecurityError):
            manager.save_latest(global_step=1, payload={"weights": b"renamed-full-model"})

    def test_best_recursively_drops_training_state(self) -> None:
        manager = CheckpointManager(self.root)
        manager.save_best(
            global_step=1,
            metrics=metrics(0.5),
            payload={
                "adapter": "ok",
                "trainer": {
                    "optimizer_state": {"momentum": 1},
                    "normalizer_state": {"mean": 2},
                    "public": "kept",
                },
            },
        )
        loaded = manager.load("best")
        self.assertEqual(loaded["trainer"], {"public": "kept"})

    def test_non_qwen_inference_weights_require_explicit_non_adapter_mode(self) -> None:
        manager = CheckpointManager(self.root, save_adapter_only=False)
        manager.save_best(
            global_step=1,
            metrics=metrics(0.5),
            payload={"model_state_dict": {"policy": b"small-inference-weights"}},
        )
        self.assertIn("model_state_dict", manager.load("best"))

    def test_constructor_recovers_hidden_old_slot_after_interrupted_commit(self) -> None:
        backup = self.root / ".latest.old-recovery"
        backup.mkdir(parents=True)
        (backup / "checkpoint_meta.json").write_text(
            '{"global_step": 3, "update": 3}', encoding="utf-8"
        )
        import pickle

        with (backup / "checkpoint_state.pkl").open("wb") as stream:
            pickle.dump({"global_step": 3, "update": 3}, stream)
        manager = CheckpointManager(self.root)
        self.assertEqual(manager.latest_meta["global_step"], 3)
        self.assertFalse(backup.exists())

    def test_constructor_removes_only_its_stale_staging_directory(self) -> None:
        stale = self.root / ".best.tmp-interrupted"
        unrelated = self.root / "another_users_checkpoint"
        stale.mkdir(parents=True)
        unrelated.mkdir()
        (stale / "partial.bin").write_bytes(b"partial")
        (unrelated / "keep.bin").write_bytes(b"keep")
        CheckpointManager(self.root)
        self.assertFalse(stale.exists())
        self.assertEqual((unrelated / "keep.bin").read_bytes(), b"keep")

    def test_rejects_qwen_shards_written_by_callback(self) -> None:
        def writer(path: Path, payload: object) -> None:
            del payload
            (path / "model-00001-of-00002.safetensors").write_bytes(b"qwen")

        manager = CheckpointManager(self.root, save_callback=writer)
        with self.assertRaises(CheckpointSecurityError):
            manager.save_latest(global_step=1)
        self.assertFalse(manager.latest_path.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_meta_and_load_reject_arbitrary_paths(self) -> None:
        manager = CheckpointManager(self.root)
        with self.assertRaises(ValueError):
            manager.meta(self.root / "step_000001")
        with self.assertRaises(ValueError):
            manager.load("periodic")
        with self.assertRaises(FileNotFoundError):
            manager.load("best")

    def test_invalid_metrics_and_mismatched_resume_counters_are_rejected(self) -> None:
        manager = CheckpointManager(self.root)
        with self.assertRaises(CheckpointFormatError):
            manager.maybe_save_best(global_step=1, metrics={})
        with self.assertRaises(CheckpointFormatError):
            manager.maybe_save_best(
                global_step=1, metrics=metrics(float("nan"))
            )
        with self.assertRaises(CheckpointFormatError):
            manager.save_latest(global_step=2, payload={"global_step": 1})
        with self.assertRaises(CheckpointFormatError):
            manager.save_latest(global_step=2, update=3, payload={"update": 2})

        manager.save_latest(global_step=3)
        with self.assertRaisesRegex(CheckpointFormatError, "move backwards"):
            manager.save_latest(global_step=2)

    def test_generic_maybe_save_supports_only_bounded_slots(self) -> None:
        manager = CheckpointManager(self.root, latest_interval_steps=2)
        self.assertEqual(
            manager.maybe_save("latest", global_step=2), manager.latest_path
        )
        self.assertEqual(
            manager.maybe_save(
                "best", global_step=2, metrics=metrics(0.4), payload={"x": 1}
            ),
            manager.best_path,
        )
        with self.assertRaises(ValueError):
            manager.maybe_save("periodic", global_step=2)
        self.assertEqual(
            sorted(path.name for path in self.root.iterdir()), ["best", "latest"]
        )

    def test_documented_latest_call_accepts_positional_global_step(self) -> None:
        manager = CheckpointManager(self.root, latest_interval_steps=2)
        self.assertEqual(manager.maybe_save_latest(2), manager.latest_path)

    def test_checkpoint_metadata_is_json_and_contains_resume_counters(self) -> None:
        manager = CheckpointManager(self.root)
        manager.save_latest(global_step=12, update=3, payload={"adapter": "x"})
        meta_path = manager.latest_path / "checkpoint_meta.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["kind"], "latest")
        self.assertEqual(metadata["global_step"], 12)
        self.assertEqual(metadata["update"], 3)


if __name__ == "__main__":
    unittest.main()
