"""Offline CPU integration tests for episode-sharded target-state training."""

from __future__ import annotations

import copy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import torch

from datasets.target_state.dataset import (
    build_manifest,
    compute_dataset_sha256,
    read_frame_records,
)
from datasets.target_state.sequence import build_sequences
from tests.training.target_state.test_shards import _episode_for, _write_parent_dataset
from training.target_state.config import TargetStateTrainingConfig, TrainingStage
from training.target_state.shards import build_target_state_shards
import training.target_state.sharded_trainer as sharded_trainer
from training.target_state.sharded_trainer import (
    PCTransCLI,
    RESUME_PROTOCOL,
    TRAINING_PROTOCOL,
    ShardedTrainingError,
    ShardedTrainingHooks,
    ShardedTrainingOptions,
    deterministic_batch_seed,
    deterministic_shard_order,
    train_target_state_sharded,
)
from training.target_state.trainer import sha256_file, validate_initial_checkpoint


def _append_second_validation_episode(root: Path) -> None:
    """Extend the shared fixture so integration really merges two val shards."""

    records = list(read_frame_records(root / "frames.jsonl"))
    source_episode = next(
        item.episode_id
        for item in records
        if item.episode_id.startswith("episode_validation_")
    )
    destination_episode = _episode_for("validation", 1)
    for record in tuple(item for item in records if item.episode_id == source_episode):
        suffix = f"validation_extra_{record.frame_id}"
        sensor = record.sensor_input
        copied_paths: dict[str, str | None] = {}
        for field, relative in (
            ("rgb_path", sensor.rgb_path),
            ("depth_path", sensor.depth_path),
            ("instance_mask_path", sensor.instance_mask_path),
        ):
            if relative is None:
                copied_paths[field] = None
                continue
            original = root / relative
            destination = original.with_name(f"{suffix}{original.suffix}")
            shutil.copy2(original, destination)
            copied_paths[field] = destination.relative_to(root).as_posix()
        new_sensor = replace(sensor, **copied_paths)
        records.append(
            replace(
                record,
                frame_id=suffix,
                episode_id=destination_episode,
                assignment_id=f"assignment_{destination_episode}",
                sensor_input=new_sensor,
                detector_prediction=replace(
                    record.detector_prediction,
                    candidate_id=f"candidate_{destination_episode}",
                    tracker_id=f"tracker_{destination_episode}",
                ),
            )
        )
    (root / "frames.jsonl").write_text(
        "".join(
            json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in records
        ),
        encoding="utf-8",
    )
    sequences = build_sequences(records, history_size=4, max_history_age_s=2.0)
    dataset_sha = compute_dataset_sha256(root, records)
    previous = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest = build_manifest(
        records,
        sequences,
        dataset_sha256=dataset_sha,
        split_seed=42,
        generation_commit_sha=str(previous["generation_commit_sha"]),
    )
    for field in (
        "detector_prediction_source",
        "candidate_id_source",
        "detector_truth_association",
        "detector_deployment",
        "yolo_model_sha256",
        "oracle_usage",
        "history_size",
        "max_history_age_s",
    ):
        manifest[field] = previous[field]
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class _FakeLifecycle:
    """Local ready/active cache with the same observable lifecycle as pc_trans."""

    def __init__(self, source: Path, bridge: Path) -> None:
        self.source = source
        self.bridge = bridge
        self.events: list[tuple[object, ...]] = []
        self.requests: dict[str, tuple[str, ...]] = {}
        self.consumed: set[tuple[str, str]] = set()

    def _ready(self, run_id: str, filename: str) -> Path:
        return self.bridge / "train_cache" / "ready" / run_id / filename

    def _active(self, run_id: str, filename: str) -> Path:
        return self.bridge / "train_cache" / "active" / run_id / filename

    def request(self, run_id: str, filenames: tuple[str, ...]) -> None:
        names = tuple(filenames)
        self.requests[run_id] = names
        (self.bridge / "train_cache" / "ready" / run_id).mkdir(
            parents=True, exist_ok=True
        )
        (self.bridge / "train_cache" / "active" / run_id).mkdir(
            parents=True, exist_ok=True
        )
        for filename in names:
            source = self.source / filename
            if not source.is_file():
                raise AssertionError(f"request names an unknown source shard: {filename}")
            ready = self._ready(run_id, filename)
            active = self._active(run_id, filename)
            if (
                (run_id, filename) not in self.consumed
                and not ready.exists()
                and not active.exists()
            ):
                shutil.copy2(source, ready)
        self.events.append(("request", run_id, names))

    def wait_shard(self, run_id: str, filename: str, timeout_s: float) -> None:
        if filename not in self.requests.get(run_id, ()):
            raise AssertionError("wait-shard was called before request")
        if not self._ready(run_id, filename).is_file():
            raise AssertionError("requested shard is not ready")
        self.events.append(("wait", run_id, filename, timeout_s))

    def shard_state(self, run_id: str, filename: str) -> dict[str, object]:
        ready = self._ready(run_id, filename).is_file()
        active = self._active(run_id, filename).is_file()
        consumed = (run_id, filename) in self.consumed
        state: dict[str, object] = {
            "requested": filename in self.requests.get(run_id, ()),
            "ready": ready,
            "active": active,
            "consumed_record": consumed,
            "deleted": True if consumed else None,
        }
        if active:
            state["active_path"] = str(self._active(run_id, filename))
        self.events.append(("state", run_id, filename, ready, active, consumed))
        return state

    def activate(self, run_id: str, filename: str) -> Path:
        source = self._ready(run_id, filename)
        target = self._active(run_id, filename)
        if not source.is_file() or target.exists():
            raise AssertionError("activate requires exactly one ready shard")
        source.replace(target)
        self.events.append(("activate", run_id, filename))
        return target

    def consume(self, run_id: str, filename: str, *, delete: bool) -> None:
        if not delete:
            raise AssertionError("trainer must evict completed active shards")
        path = self._active(run_id, filename)
        if not path.is_file():
            raise AssertionError("consume requires an active shard")
        path.unlink()
        self.consumed.add((run_id, filename))
        self.events.append(("consume", run_id, filename))


class _CheckpointCrash(RuntimeError):
    pass


class ShardedTargetStateTrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._shared_temporary = tempfile.TemporaryDirectory()
        cls.shared_root = Path(cls._shared_temporary.name)
        cls.parent_dataset = cls.shared_root / "parent"
        cls.parent_dataset.mkdir()
        _write_parent_dataset(cls.parent_dataset)
        _append_second_validation_episode(cls.parent_dataset)
        cls.built = build_target_state_shards(
            cls.parent_dataset,
            cls.shared_root / "shards",
            target_shard_size_bytes=1,
            history_size=4,
            max_history_age_s=2.0,
            split_seed=42,
        )
        cls.index = cls.built.shard_index
        if len(cls.index.shards_for_split("train")) < 2:
            raise AssertionError("fixture must produce at least two train shards")
        if len(cls.index.shards_for_split("validation")) < 2:
            raise AssertionError("fixture must produce at least two validation shards")
        if len(cls.index.shards_for_split("test")) < 1:
            raise AssertionError("fixture must produce a test shard")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._shared_temporary.cleanup()

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.case_root = Path(self.temporary.name)

    def _config(self, run_name: str) -> TargetStateTrainingConfig:
        return TargetStateTrainingConfig(
            dataset_root=self.parent_dataset,
            output_dir=self.case_root / "output",
            stage=TrainingStage.ORACLE_CLEAN,
            history_size=4,
            max_history_age_s=2.0,
            roi_size_px=32,
            roi_feature_dim=8,
            geometry_feature_dim=8,
            hidden_dim=8,
            gru_layers=1,
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            num_workers=0,
            seed=42,
            device="cpu",
            run_name=run_name,
            save_figures=0,
            promotion_min_covariance_correlation=-1.0,
            require_dataset_manifest=True,
        )

    def _options(
        self,
        bridge: Path,
        run_id_prefix: str,
        *,
        resume_checkpoint: Path | None = None,
    ) -> ShardedTrainingOptions:
        return ShardedTrainingOptions(
            shard_index_path=self.index.source_path,
            pc_trans_root=self.case_root / "unused_pc_trans",
            pc_trans_config=self.case_root / "unused_pc_trans.json",
            bridge_root=bridge,
            run_id_prefix=run_id_prefix,
            resume_checkpoint=resume_checkpoint,
            wait_timeout_s=0.0,
        )

    @staticmethod
    def _load(path: Path) -> dict[str, object]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise AssertionError(f"checkpoint is not a dict: {path}")
        return payload

    @staticmethod
    def _optimizer_steps(checkpoint: dict[str, object]) -> set[int]:
        optimizer = checkpoint["optimizer_state_dict"]
        if not isinstance(optimizer, dict):
            raise AssertionError("optimizer state is not a dict")
        state = optimizer["state"]
        if not isinstance(state, dict):
            raise AssertionError("optimizer parameter state is not a dict")
        return {
            int(float(value["step"]))
            for value in state.values()
            if isinstance(value, dict) and "step" in value
        }

    def _pc_trans_client(self) -> PCTransCLI:
        project_root = self.case_root / "pc_trans_project"
        package = project_root / "pc_trans"
        package.mkdir(parents=True)
        (package / "cli.py").write_text("# test stub\n", encoding="utf-8")
        config_path = self.case_root / "pc_trans_config.json"
        config_path.write_text("{}\n", encoding="utf-8")
        return PCTransCLI(project_root=project_root, config_path=config_path)

    @staticmethod
    def _state_payload(**updates: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "run_id": "run_1",
            "shard": "shard_stagea_train_000001.tar",
            "requested": True,
            "ready": False,
            "active": False,
            "consumed_record": False,
            "deleted": None,
        }
        payload.update(updates)
        return payload

    def test_pc_trans_shard_state_validates_identity_types_and_valid_states(
        self,
    ) -> None:
        client = self._pc_trans_client()
        filename = "shard_stagea_train_000001.tar"
        valid_states = (
            ({}, "missing"),
            ({"ready": True}, "ready"),
            ({"active": True}, "active"),
            ({"consumed_record": True, "deleted": True}, "consumed"),
            (
                {"active": True, "consumed_record": True, "deleted": False},
                "consumed",
            ),
        )
        for updates, expected_state in valid_states:
            with self.subTest(updates=updates):
                payload = self._state_payload(**updates)
                completed = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=json.dumps(payload), stderr=""
                )
                with mock.patch.object(PCTransCLI, "_run", return_value=completed):
                    result = client.shard_state("run_1", filename)
                self.assertEqual(result["run_id"], "run_1")
                self.assertEqual(result["shard"], filename)
                self.assertEqual(result["state"], expected_state)

    def test_pc_trans_shard_state_rejects_noncanonical_or_contradictory_json(
        self,
    ) -> None:
        client = self._pc_trans_client()
        filename = "shard_stagea_train_000001.tar"
        invalid_payloads = {
            "missing exact field": {
                key: value
                for key, value in self._state_payload().items()
                if key != "deleted"
            },
            "unexpected field": self._state_payload(state="ready"),
            "run identity mismatch": self._state_payload(run_id="run_2"),
            "shard identity mismatch": self._state_payload(shard="shard_other.tar"),
            "requested false": self._state_payload(requested=False),
            "integer is not bool": self._state_payload(ready=1),
            "deleted has wrong type": self._state_payload(deleted="false"),
            "ready and active": self._state_payload(ready=True, active=True),
            "deleted without record": self._state_payload(deleted=False),
            "record without deleted": self._state_payload(consumed_record=True),
            "deleted record remains ready": self._state_payload(
                ready=True, consumed_record=True, deleted=True
            ),
            "deleted record remains active": self._state_payload(
                active=True, consumed_record=True, deleted=True
            ),
            "retained record lacks active": self._state_payload(
                consumed_record=True, deleted=False
            ),
            "retained record is ready": self._state_payload(
                ready=True, consumed_record=True, deleted=False
            ),
        }
        for description, payload in invalid_payloads.items():
            with self.subTest(description=description):
                completed = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=json.dumps(payload), stderr=""
                )
                with mock.patch.object(PCTransCLI, "_run", return_value=completed):
                    with self.assertRaises(ShardedTrainingError):
                        client.shard_state("run_1", filename)

        duplicate = (
            '{"run_id":"run_1","shard":"shard_stagea_train_000001.tar",'
            '"requested":true,"ready":false,"ready":true,"active":false,'
            '"consumed_record":false,"deleted":null}'
        )
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=duplicate, stderr=""
        )
        with mock.patch.object(PCTransCLI, "_run", return_value=completed):
            with self.assertRaisesRegex(ShardedTrainingError, "duplicate"):
                client.shard_state("run_1", filename)

    def test_shard_order_and_batch_seed_are_process_independent(self) -> None:
        train = self.index.shards_for_split("train")
        first = deterministic_shard_order(train, base_seed=42, global_epoch=1)
        repeated = deterministic_shard_order(train, base_seed=42, global_epoch=1)
        reversed_input = deterministic_shard_order(
            tuple(reversed(train)), base_seed=42, global_epoch=1
        )
        self.assertEqual(first, repeated)
        self.assertEqual(first, reversed_input)
        self.assertEqual(set(first), {entry.filename for entry in train})

        filename = first[0]
        expected = int.from_bytes(
            sha256(f"42:batches:1:{filename}".encode("utf-8")).digest()[:8],
            "big",
        ) % (2**63 - 1)
        self.assertEqual(
            deterministic_batch_seed(
                base_seed=42, global_epoch=1, filename=filename
            ),
            expected,
        )
        self.assertEqual(
            deterministic_batch_seed(
                base_seed=42, global_epoch=1, filename=filename
            ),
            deterministic_batch_seed(
                base_seed=42, global_epoch=1, filename=filename
            ),
        )
        self.assertNotEqual(
            deterministic_batch_seed(
                base_seed=42, global_epoch=1, filename=filename
            ),
            deterministic_batch_seed(
                base_seed=42, global_epoch=2, filename=filename
            ),
        )

    def test_one_global_epoch_streams_all_splits_and_publishes_auditable_outputs(
        self,
    ) -> None:
        bridge = self.case_root / "bridge"
        lifecycle = _FakeLifecycle(self.built.output_dir, bridge)
        config = replace(self._config("full_flow"), epochs=2)
        options = self._options(bridge, "full")

        def record_checkpoint(path: Path, checkpoint: object) -> None:
            if not isinstance(checkpoint, dict):
                raise AssertionError("checkpoint hook payload must be a dict")
            lifecycle.events.append(
                (
                    "checkpoint",
                    path.name,
                    checkpoint.get("last_completed_shard"),
                    checkpoint.get("phase"),
                )
            )

        result = train_target_state_sharded(
            config,
            options,
            lifecycle=lifecycle,
            hooks=ShardedTrainingHooks(after_checkpoint=record_checkpoint),
        )

        latest = self._load(result.latest_checkpoint)
        best = self._load(result.best_checkpoint)
        required = {
            "model_type",
            "schema_version",
            "training_stage",
            "model_config",
            "model_state_dict",
            "optimizer_state_dict",
            "parent_dataset_sha256",
            "shard_index_sha256",
            "global_epoch",
            "global_step",
            "phase",
            "current_pc_trans_run_id",
            "train_shard_order",
            "next_train_shard_index",
            "validation_shard_index",
            "test_shard_index",
            "last_completed_shard",
            "best_validation_loss",
            "best_epoch",
            "last_validation_metrics",
            "training_protocol",
        }
        self.assertFalse(required - set(latest))
        self.assertFalse(required - set(best))
        self.assertEqual(latest["training_protocol"], TRAINING_PROTOCOL)
        self.assertEqual(latest["resume_protocol"], RESUME_PROTOCOL)
        self.assertEqual(latest["phase"], "complete")
        self.assertEqual(latest["global_epoch"], 2)
        self.assertEqual(latest["global_step"], 4)
        self.assertIn(best["global_step"], {2, 4})
        self.assertEqual(best["training_stage"], TrainingStage.ORACLE_CLEAN.value)
        self.assertIn(self._optimizer_steps(best), ({2}, {4}))
        self.assertEqual(self._optimizer_steps(latest), {4})
        self.assertEqual(
            best["validation_shard_index"],
            len(self.index.shards_for_split("validation")),
        )
        self.assertTrue(best["last_validation_metrics"])
        self.assertEqual(
            latest["test_shard_index"], len(self.index.shards_for_split("test"))
        )
        self.assertTrue(latest["last_test_metrics"])

        expected_consumed = 2 * (
            len(self.index.shards_for_split("train"))
            + len(self.index.shards_for_split("validation"))
        ) + len(self.index.shards_for_split("test"))
        self.assertEqual(len(lifecycle.consumed), expected_consumed)
        self.assertEqual(
            sum(1 for event in lifecycle.events if event[0] == "wait"),
            expected_consumed,
        )
        for position, event in enumerate(lifecycle.events):
            if event[0] != "consume":
                continue
            preceding = lifecycle.events[position - 1]
            self.assertEqual(preceding[0], "checkpoint")
            self.assertEqual(preceding[2], event[2])

        manifest = json.loads(result.model_manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["training_protocol"], TRAINING_PROTOCOL)
        self.assertEqual(manifest["resume_protocol"], RESUME_PROTOCOL)
        self.assertEqual(manifest["parent_dataset_sha256"], self.index.parent_dataset_sha256)
        self.assertEqual(manifest["shard_index_sha256"], self.index.index_sha256)
        self.assertEqual(manifest["global_epochs"], 2)
        self.assertEqual(manifest["global_step"], 4)
        self.assertEqual(manifest["completed_training_shards"], 4)
        self.assertEqual(manifest["shard_counts"], {"train": 2, "validation": 2, "test": 1})
        self.assertEqual(
            manifest["pc_trans"]["per_epoch_run_ids"],
            ["full.e0001", "full.e0002"],
        )
        self.assertEqual(manifest["pc_trans"]["final_test_run_id"], "full.finaltest")

        stage_b = replace(
            config,
            stage=TrainingStage.YOLO_DEPLOYMENT,
            initial_checkpoint_path=result.best_checkpoint,
            require_dataset_manifest=False,
            run_name="stage_b_probe",
        )
        initial, digest = validate_initial_checkpoint(stage_b, map_location="cpu")
        self.assertIsNotNone(initial)
        self.assertEqual(initial["training_stage"], TrainingStage.ORACLE_CLEAN.value)
        self.assertEqual(len(digest or ""), 64)

    def test_crash_after_checkpoint_resumes_by_consuming_without_retraining(
        self,
    ) -> None:
        bridge = self.case_root / "bridge"
        lifecycle = _FakeLifecycle(self.built.output_dir, bridge)
        config = self._config("resume_flow")
        options = self._options(bridge, "resume")
        trained: list[str] = []
        real_train = sharded_trainer._train_materialized_shard

        def recording_train(**kwargs: object):
            trained.append(str(kwargs["filename"]))
            return real_train(**kwargs)  # type: ignore[arg-type]

        crashed = False

        def crash_after_first_train(path: Path, checkpoint: object) -> None:
            nonlocal crashed
            if (
                not crashed
                and isinstance(checkpoint, dict)
                and checkpoint.get("last_completed_phase") == "train"
                and checkpoint.get("next_train_shard_index") == 1
            ):
                crashed = True
                raise _CheckpointCrash("injected after durable shard checkpoint")

        with mock.patch.object(
            sharded_trainer,
            "_train_materialized_shard",
            side_effect=recording_train,
        ):
            with self.assertRaisesRegex(_CheckpointCrash, "injected"):
                train_target_state_sharded(
                    config,
                    options,
                    lifecycle=lifecycle,
                    hooks=ShardedTrainingHooks(after_checkpoint=crash_after_first_train),
                )

            latest_path = config.output_dir / config.run_name / "latest.pt"
            interrupted = self._load(latest_path)
            first_filename = str(interrupted["last_completed_shard"])
            self.assertEqual(interrupted["global_step"], 1)
            self.assertEqual(interrupted["next_train_shard_index"], 1)
            self.assertEqual(self._optimizer_steps(interrupted), {1})
            self.assertTrue(
                lifecycle._active(str(interrupted["last_completed_run_id"]), first_filename).is_file()
            )
            self.assertFalse(any(event[0] == "consume" for event in lifecycle.events))

            event_boundary = len(lifecycle.events)
            resumed = train_target_state_sharded(
                config,
                replace(options, resume_checkpoint=latest_path),
                lifecycle=lifecycle,
            )

        self.assertEqual(trained.count(first_filename), 1)
        self.assertEqual(len(trained), 2)
        resume_events = lifecycle.events[event_boundary:]
        first_resume_consume = next(
            event for event in resume_events if event[0] == "consume"
        )
        self.assertEqual(first_resume_consume[2], first_filename)
        final = self._load(resumed.latest_checkpoint)
        self.assertEqual(final["phase"], "complete")
        self.assertEqual(final["global_step"], 2)
        self.assertEqual(final["completed_training_shards"], 2)
        self.assertEqual(self._optimizer_steps(final), {2})
        self.assertEqual(len(lifecycle.consumed), len(self.index.shards))

    def test_stage_b_initial_checkpoint_lineage_survives_same_run_resume(self) -> None:
        initial_path = self.case_root / "stage_a_best.pt"
        source_config = self._config("source_model")
        source_model = sharded_trainer._new_model(source_config, torch.device("cpu"))
        torch.save(
            {
                "model_type": "temporal_ray_depth_residual",
                "schema_version": 1,
                "training_stage": TrainingStage.ORACLE_CLEAN.value,
                "model_state_dict": source_model.state_dict(),
            },
            initial_path,
        )
        bridge = self.case_root / "stage_b_bridge"
        lifecycle = _FakeLifecycle(self.built.output_dir, bridge)
        config = replace(
            self._config("stage_b_resume"),
            stage=TrainingStage.YOLO_DEPLOYMENT,
            initial_checkpoint_path=initial_path,
            require_dataset_manifest=False,
        )
        options = self._options(bridge, "stageb")
        crashed = False

        def crash_after_first_train(path: Path, checkpoint: object) -> None:
            nonlocal crashed
            if (
                not crashed
                and isinstance(checkpoint, dict)
                and checkpoint.get("last_completed_phase") == "train"
                and checkpoint.get("next_train_shard_index") == 1
            ):
                crashed = True
                raise _CheckpointCrash("stage B lineage resume")

        with self.assertRaisesRegex(_CheckpointCrash, "lineage"):
            train_target_state_sharded(
                config,
                options,
                lifecycle=lifecycle,
                hooks=ShardedTrainingHooks(after_checkpoint=crash_after_first_train),
            )

        latest_path = config.output_dir / config.run_name / "latest.pt"
        resumed = train_target_state_sharded(
            replace(config, initial_checkpoint_path=None),
            replace(options, resume_checkpoint=latest_path),
            lifecycle=lifecycle,
        )
        latest = self._load(resumed.latest_checkpoint)
        manifest = json.loads(resumed.model_manifest.read_text(encoding="utf-8"))
        expected_lineage = {
            "path": str(initial_path.resolve()),
            "sha256": sha256_file(initial_path),
        }
        self.assertEqual(latest["initial_checkpoint_path"], expected_lineage["path"])
        self.assertEqual(latest["initial_checkpoint_sha256"], expected_lineage["sha256"])
        self.assertEqual(manifest["initial_checkpoint"], expected_lineage)
        self.assertTrue(
            manifest["promotion"]["stage_a_initialization_satisfied"]
        )

    def test_resume_rejects_best_and_tampered_progress_boundaries(self) -> None:
        bridge = self.case_root / "bridge"
        lifecycle = _FakeLifecycle(self.built.output_dir, bridge)
        config = self._config("resume_validation_source")
        options = self._options(bridge, "resumecheck")
        snapshots: dict[str, dict[str, object]] = {}

        def capture_boundaries(path: Path, checkpoint: object) -> None:
            if not isinstance(checkpoint, dict):
                raise AssertionError("checkpoint hook payload must be a dict")
            phase = checkpoint.get("phase")
            if (
                phase == "validation"
                and checkpoint.get("validation_shard_index") == 0
                and "validation" not in snapshots
            ):
                snapshots["validation"] = copy.deepcopy(checkpoint)
            if (
                phase == "final_test"
                and checkpoint.get("test_shard_index") == 0
                and "final_test" not in snapshots
            ):
                snapshots["final_test"] = copy.deepcopy(checkpoint)

        result = train_target_state_sharded(
            config,
            options,
            lifecycle=lifecycle,
            hooks=ShardedTrainingHooks(after_checkpoint=capture_boundaries),
        )
        self.assertEqual(set(snapshots), {"validation", "final_test"})

        # A promotion/initialization artifact is deliberately not a same-run
        # recovery authority, even though its model/optimizer tensors are valid.
        with self.assertRaisesRegex(ShardedTrainingError, "checkpoint_role"):
            sharded_trainer.validate_resume_checkpoint(
                result.best_checkpoint, config=config, index=self.index
            )

        complete = self._load(result.latest_checkpoint)
        self.assertEqual(complete["checkpoint_role"], "latest")
        validated = sharded_trainer.validate_resume_checkpoint(
            result.latest_checkpoint, config=config, index=self.index
        )
        self.assertEqual(validated["phase"], "complete")

        invalid: dict[str, tuple[dict[str, object], str]] = {}

        bad_phase = copy.deepcopy(complete)
        bad_phase["phase"] = "unknown"
        invalid["phase"] = (bad_phase, "phase is invalid")

        bad_train_index = copy.deepcopy(complete)
        bad_train_index["next_train_shard_index"] = (
            len(self.index.shards_for_split("train")) + 1
        )
        invalid["train_index"] = (bad_train_index, "outside shard order")

        bad_test_index = copy.deepcopy(complete)
        bad_test_index["test_shard_index"] = 0
        invalid["complete_test_index"] = (
            bad_test_index,
            "complete phase requires the full test prefix",
        )

        bad_identity = copy.deepcopy(complete)
        bad_identity["last_completed_shard"] = str(
            complete["train_shard_order"][0]  # type: ignore[index]
        )
        invalid["last_identity"] = (bad_identity, "last-completed identity")

        missing_validation_accumulator = copy.deepcopy(snapshots["validation"])
        missing_validation_accumulator["evaluation_accumulator"] = None
        invalid["validation_accumulator"] = (
            missing_validation_accumulator,
            "validation resume checkpoint must contain an evaluation accumulator",
        )

        missing_test_accumulator = copy.deepcopy(snapshots["final_test"])
        missing_test_accumulator["evaluation_accumulator"] = None
        invalid["test_accumulator"] = (
            missing_test_accumulator,
            "final_test resume checkpoint must contain an evaluation accumulator",
        )

        bad_step = copy.deepcopy(complete)
        bad_step["global_step"] = int(complete["global_step"]) + 1
        invalid["global_step"] = (bad_step, "global_step does not match")

        bad_complete_count = copy.deepcopy(complete)
        bad_complete_count["completed_training_shards"] = (
            int(complete["completed_training_shards"]) - 1
        )
        invalid["complete_count"] = (
            bad_complete_count,
            "completed_training_shards is inconsistent",
        )

        missing_materialized_path = copy.deepcopy(complete)
        missing_materialized_path["last_materialized_path"] = None
        invalid["materialized_path"] = (
            missing_materialized_path,
            "must name its materialized path",
        )

        bad_optimizer = copy.deepcopy(complete)
        bad_optimizer["optimizer_state_dict"]["param_groups"][0]["lr"] = float(  # type: ignore[index]
            "nan"
        )
        invalid["optimizer_lr"] = (bad_optimizer, "optimizer lr mismatch")

        for name, (checkpoint, expected_error) in invalid.items():
            with self.subTest(name=name):
                path = self.case_root / f"tampered_{name}.pt"
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ShardedTrainingError, expected_error):
                    sharded_trainer.validate_resume_checkpoint(
                        path, config=config, index=self.index
                    )

        stale_copy = self.case_root / "stale_latest_copy.pt"
        torch.save(snapshots["validation"], stale_copy)
        canonical_sha = sha256_file(result.latest_checkpoint)
        lifecycle_event_count = len(lifecycle.events)
        with self.assertRaisesRegex(ShardedTrainingError, "canonical latest.pt"):
            train_target_state_sharded(
                config,
                replace(options, resume_checkpoint=stale_copy),
                lifecycle=lifecycle,
            )
        self.assertEqual(sha256_file(result.latest_checkpoint), canonical_sha)
        self.assertEqual(len(lifecycle.events), lifecycle_event_count)

    def test_multilayer_gru_second_shard_replay_is_bit_exact(self) -> None:
        baseline_config = replace(
            self._config("rng_baseline"),
            gru_layers=2,
        )
        baseline_lifecycle = _FakeLifecycle(
            self.built.output_dir, self.case_root / "baseline_bridge"
        )
        baseline = train_target_state_sharded(
            baseline_config,
            self._options(self.case_root / "baseline_bridge", "rngbase"),
            lifecycle=baseline_lifecycle,
        )

        replay_config = replace(baseline_config, run_name="rng_replay")
        replay_bridge = self.case_root / "replay_bridge"
        replay_lifecycle = _FakeLifecycle(self.built.output_dir, replay_bridge)
        replay_options = self._options(replay_bridge, "rngreplay")
        real_train = sharded_trainer._train_materialized_shard
        initial_executions: list[str] = []

        def fail_after_second_shard_execution(**kwargs: object):
            result = real_train(**kwargs)  # type: ignore[arg-type]
            initial_executions.append(str(kwargs["filename"]))
            if len(initial_executions) == 2:
                raise _CheckpointCrash(
                    "injected after second shard execution before checkpoint"
                )
            return result

        with mock.patch.object(
            sharded_trainer,
            "_train_materialized_shard",
            side_effect=fail_after_second_shard_execution,
        ):
            with self.assertRaisesRegex(_CheckpointCrash, "second shard execution"):
                train_target_state_sharded(
                    replay_config,
                    replay_options,
                    lifecycle=replay_lifecycle,
                )

        replay_latest = replay_config.output_dir / replay_config.run_name / "latest.pt"
        interrupted = self._load(replay_latest)
        self.assertEqual(interrupted["next_train_shard_index"], 1)
        self.assertEqual(interrupted["global_step"], 1)
        self.assertEqual(interrupted["last_completed_shard"], initial_executions[0])
        self.assertNotEqual(initial_executions[0], initial_executions[1])
        self.assertTrue(
            replay_lifecycle._active("rngreplay.e0001", initial_executions[1]).is_file()
        )

        resumed_executions: list[str] = []

        def record_replayed_training(**kwargs: object):
            resumed_executions.append(str(kwargs["filename"]))
            return real_train(**kwargs)  # type: ignore[arg-type]

        with mock.patch.object(
            sharded_trainer,
            "_train_materialized_shard",
            side_effect=record_replayed_training,
        ):
            resumed = train_target_state_sharded(
                replay_config,
                replace(replay_options, resume_checkpoint=replay_latest),
                lifecycle=replay_lifecycle,
            )

        self.assertEqual(resumed_executions, [initial_executions[1]])
        baseline_state = self._load(baseline.latest_checkpoint)["model_state_dict"]
        resumed_state = self._load(resumed.latest_checkpoint)["model_state_dict"]
        self.assertEqual(set(baseline_state), set(resumed_state))  # type: ignore[arg-type]
        for name in baseline_state:  # type: ignore[union-attr]
            with self.subTest(parameter=name):
                self.assertTrue(
                    torch.equal(baseline_state[name], resumed_state[name]),  # type: ignore[index]
                    msg=f"replayed model tensor differs: {name}",
                )

    def test_second_trainer_fails_closed_while_run_lock_is_held(self) -> None:
        bridge = self.case_root / "bridge"
        config = self._config("locked_run")
        options = self._options(bridge, "locked")
        lifecycle = _FakeLifecycle(self.built.output_dir, bridge)
        run_dir = config.output_dir / config.run_name

        with sharded_trainer._exclusive_run_lock(run_dir):
            with self.assertRaisesRegex(
                ShardedTrainingError, "another sharded trainer already owns this run"
            ):
                train_target_state_sharded(
                    config,
                    options,
                    lifecycle=lifecycle,
                )
        self.assertFalse(lifecycle.events)

    def test_mid_shard_failure_keeps_active_and_does_not_advance_or_consume(
        self,
    ) -> None:
        bridge = self.case_root / "bridge"
        lifecycle = _FakeLifecycle(self.built.output_dir, bridge)
        config = self._config("mid_shard_failure")
        options = self._options(bridge, "midfail")

        with mock.patch.object(
            sharded_trainer,
            "_train_materialized_shard",
            side_effect=RuntimeError("injected batch failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected batch failure"):
                train_target_state_sharded(
                    config,
                    options,
                    lifecycle=lifecycle,
                )

        latest = self._load(config.output_dir / config.run_name / "latest.pt")
        self.assertEqual(latest["phase"], "train")
        self.assertEqual(latest["global_step"], 0)
        self.assertEqual(latest["next_train_shard_index"], 0)
        self.assertIsNone(latest["last_completed_shard"])
        self.assertFalse(lifecycle.consumed)
        active_files = list(
            (bridge / "train_cache" / "active" / "midfail.e0001").glob("shard_*.tar")
        )
        self.assertEqual(len(active_files), 1)


if __name__ == "__main__":
    unittest.main()
