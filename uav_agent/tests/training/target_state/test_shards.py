"""Episode-shard builder and safe materializer tests."""

from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image

from datasets.target_state.dataset import (
    build_manifest,
    check_dataset,
    compute_dataset_sha256,
    read_frame_records,
    split_for_episode,
)
from datasets.target_state.sequence import build_sequences
from tests.training.target_state.test_dataset_schema import make_record
from scripts.build_target_state_shards import main as build_shards_main
from training.target_state.shard_runtime import (
    ShardMaterializationError,
    cleanup_materialized_path,
    cleanup_materialized_shard,
    materialize_shard,
)
from training.target_state.shards import (
    ShardFormatError,
    build_target_state_shards,
    load_shard_index,
    plan_target_state_shards,
    sha256_file,
)


def _episode_for(split: str, ordinal: int = 0) -> str:
    found = []
    for index in range(100_000):
        candidate = f"episode_{split}_{index}"
        if split_for_episode(candidate, seed=42) == split:
            found.append(candidate)
            if len(found) > ordinal:
                return found[ordinal]
    raise AssertionError(f"cannot find episode for split {split}")


def _write_parent_dataset(root: Path) -> None:
    for name in ("rgb", "depth", "instance_mask"):
        (root / name).mkdir(parents=True, exist_ok=True)
    episodes = (
        _episode_for("train", 0),
        _episode_for("train", 1),
        _episode_for("validation"),
        _episode_for("test"),
    )
    records = []
    frame_index = 0
    for episode in episodes:
        for local_index in range(5):
            record = make_record(
                frame_index,
                episode_id=episode,
                candidate_id=f"candidate_{episode}",
            )
            sensor = replace(
                record.sensor_input,
                instance_mask_path=f"instance_mask/frame_{frame_index}.png",
            )
            record = replace(record, sensor_input=sensor)
            records.append(record)
            Image.fromarray(np.full((24, 32, 3), 80 + local_index, np.uint8)).save(
                root / sensor.rgb_path
            )
            np.save(
                root / sensor.depth_path,
                np.full((24, 32), 4.0 + local_index, dtype=np.float32),
                allow_pickle=False,
            )
            Image.fromarray(np.full((24, 32), 1, np.uint8)).save(
                root / sensor.instance_mask_path
            )
            frame_index += 1

    # A second logical target shares one synchronized physical capture.  The
    # builder must store each referenced asset path only once in its tar.
    original = records[0]
    duplicate = replace(
        original,
        frame_id="frame_shared_capture",
        detector_prediction=replace(
            original.detector_prediction,
            candidate_id="candidate_shared_capture",
            tracker_id="tracker_shared_capture",
        ),
    )
    records.append(duplicate)
    (root / "frames.jsonl").write_text(
        "".join(
            json.dumps(
                item.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for item in records
        ),
        encoding="utf-8",
    )
    sequences = build_sequences(records, history_size=4, max_history_age_s=2.0)
    dataset_sha = compute_dataset_sha256(root, records)
    manifest = build_manifest(
        records,
        sequences,
        dataset_sha256=dataset_sha,
        split_seed=42,
        generation_commit_sha="testcommit",
    )
    manifest.update(
        {
            "detector_prediction_source": "external_capture_spool_unverified",
            "candidate_id_source": "external_capture_spool_unverified",
            "detector_truth_association": "external_capture_spool_unverified",
            "detector_deployment": None,
            "yolo_model_sha256": "a" * 64,
            "oracle_usage": "offline_training_labels_only",
            "history_size": 4,
            "max_history_age_s": 2.0,
        }
    )
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    report = check_dataset(root, history_size=4, max_history_age_s=2.0)
    if not report.ok:
        raise AssertionError(report.errors)


def _copy_tar_with_bad_member(
    source: Path,
    destination: Path,
    *,
    bad_name: str,
    bad_type: bytes = tarfile.REGTYPE,
    duplicate_existing: bool = False,
) -> None:
    destination.parent.mkdir(parents=True)
    with tarfile.open(source, "r:") as original, tarfile.open(destination, "w") as output:
        for member in original.getmembers():
            stream = original.extractfile(member) if member.isreg() else None
            output.addfile(member, stream)
            if stream is not None:
                stream.close()
        if duplicate_existing:
            payload = b"{}\n"
            bad = tarfile.TarInfo("frames.jsonl")
            bad.size = len(payload)
            output.addfile(bad, io.BytesIO(payload))
        else:
            bad = tarfile.TarInfo(bad_name)
            bad.type = bad_type
            if bad_type == tarfile.REGTYPE:
                payload = b"bad"
                bad.size = len(payload)
                output.addfile(bad, io.BytesIO(payload))
            else:
                bad.linkname = "frames.jsonl"
                output.addfile(bad)


def _copy_tar_with_corrupted_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True)
    corrupted = False
    with tarfile.open(source, "r:") as original, tarfile.open(destination, "w") as output:
        for member in original.getmembers():
            stream = original.extractfile(member) if member.isreg() else None
            if member.isreg() and member.name.startswith("depth/") and not corrupted:
                payload = b"not-a-valid-numpy-array"
                replacement = tarfile.TarInfo(member.name)
                replacement.size = len(payload)
                output.addfile(replacement, io.BytesIO(payload))
                corrupted = True
            else:
                output.addfile(member, stream)
            if stream is not None:
                stream.close()
    if not corrupted:
        raise AssertionError("test archive had no depth asset to corrupt")


class TargetStateShardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.dataset = self.base / "dataset"
        self.dataset.mkdir()
        _write_parent_dataset(self.dataset)

    def _build(self):
        return build_target_state_shards(
            self.dataset,
            self.base / "shards",
            target_shard_size_bytes=1,
            history_size=4,
            max_history_age_s=2.0,
            split_seed=42,
        )

    def test_plan_is_episode_atomic_single_split_deterministic_and_soft_sized(self) -> None:
        first = plan_target_state_shards(
            self.dataset,
            target_shard_size_bytes=1,
            history_size=4,
            max_history_age_s=2.0,
            split_seed=42,
        )
        second = plan_target_state_shards(
            self.dataset,
            target_shard_size_bytes=1,
            history_size=4,
            max_history_age_s=2.0,
            split_seed=42,
        )
        self.assertEqual(first, second)
        all_episodes = [episode for plan in first for episode in plan.episode_ids]
        self.assertEqual(len(all_episodes), len(set(all_episodes)))
        self.assertTrue(all(len(plan.episode_ids) == 1 for plan in first))
        self.assertTrue(all(plan.estimated_size_bytes > 1 for plan in first))
        for plan in first:
            self.assertEqual(
                {split_for_episode(item.episode_id, seed=42) for item in plan.records},
                {plan.split},
            )

    def test_builder_hashes_self_contained_tar_and_materializer_is_idempotent(self) -> None:
        result = self._build()
        index = result.shard_index
        self.assertEqual(index.episode_count, 4)
        self.assertEqual(
            sum(
                len(index.shards_for_split(item))
                for item in ("train", "validation", "test")
            ),
            4,
        )
        self.assertEqual(
            index.parent_dataset_sha256,
            json.loads((self.dataset / "dataset_manifest.json").read_text())[
                "dataset_sha256"
            ],
        )
        for entry in index.shards:
            archive = result.output_dir / entry.filename
            self.assertEqual(archive.stat().st_size, entry.archive_size_bytes)
            self.assertEqual(sha256_file(archive), entry.archive_sha256)
            with tarfile.open(archive, "r:") as stream:
                names = [item.name for item in stream.getmembers()]
                self.assertEqual(len(names), len(set(names)))
                self.assertTrue(
                    {"frames.jsonl", "dataset_manifest.json", "shard_manifest.json"}.issubset(names)
                )
                self.assertTrue(
                    all(
                        not PurePosixPath(name).is_absolute()
                        and ".." not in PurePosixPath(name).parts
                        for name in names
                    )
                )

        entry = index.shards_for_split("train")[0]
        archive = result.output_dir / entry.filename
        materialized_root = self.base / "active" / ".materialized"
        materialized_root.parent.mkdir()
        first = materialize_shard(
            archive,
            index,
            materialized_root=materialized_root,
        )
        second = materialize_shard(
            archive,
            index,
            materialized_root=materialized_root,
        )
        self.assertEqual(first.dataset_root, second.dataset_root)
        self.assertTrue(first.dataset_report.ok)
        self.assertEqual(
            compute_dataset_sha256(
                first.dataset_root,
                read_frame_records(first.dataset_root / "frames.jsonl"),
            ),
            entry.shard_dataset_sha256,
        )
        cleanup_materialized_shard(first)
        self.assertFalse(first.dataset_root.exists())
        cleanup_materialized_path(
            first.dataset_root,
            materialized_root=materialized_root,
        )

    def test_materializer_rejects_traversal_links_devices_fifo_and_duplicates(self) -> None:
        result = self._build()
        index = result.shard_index
        entry = index.shards[0]
        source = result.output_dir / entry.filename
        cases = (
            ("absolute", "/escape", tarfile.REGTYPE, False),
            ("traversal", "../escape", tarfile.REGTYPE, False),
            ("symlink", "bad_symlink", tarfile.SYMTYPE, False),
            ("hardlink", "bad_hardlink", tarfile.LNKTYPE, False),
            ("device", "bad_device", tarfile.CHRTYPE, False),
            ("fifo", "bad_fifo", tarfile.FIFOTYPE, False),
            ("duplicate", "unused", tarfile.REGTYPE, True),
        )
        for name, member_name, member_type, duplicate in cases:
            with self.subTest(name=name):
                active = self.base / f"active_{name}"
                bad_archive = active / entry.filename
                _copy_tar_with_bad_member(
                    source,
                    bad_archive,
                    bad_name=member_name,
                    bad_type=member_type,
                    duplicate_existing=duplicate,
                )
                bad_entry = replace(
                    entry,
                    archive_sha256=sha256_file(bad_archive),
                    archive_size_bytes=bad_archive.stat().st_size,
                )
                bad_index = replace(
                    index,
                    shards=tuple(
                        bad_entry if item.shard_id == entry.shard_id else item
                        for item in index.shards
                    ),
                    index_sha256=sha256(b"test-index").hexdigest(),
                )
                with self.assertRaises(ShardMaterializationError):
                    materialize_shard(bad_archive, bad_index)
                self.assertFalse((active / ".materialized" / bad_archive.stem).exists())
                self.assertFalse((active.parent / "escape").exists())

    def test_failed_full_validation_never_publishes_final_directory(self) -> None:
        result = self._build()
        index = result.shard_index
        entry = index.shards[0]
        active = self.base / "active_corrupt"
        archive = active / entry.filename
        _copy_tar_with_corrupted_asset(result.output_dir / entry.filename, archive)
        corrupt_entry = replace(
            entry,
            archive_sha256=sha256_file(archive),
            archive_size_bytes=archive.stat().st_size,
        )
        corrupt_index = replace(
            index,
            shards=tuple(
                corrupt_entry if item.shard_id == entry.shard_id else item
                for item in index.shards
            ),
            index_sha256=sha256(b"corrupt-index").hexdigest(),
        )

        with self.assertRaises(ShardMaterializationError):
            materialize_shard(archive, corrupt_index)

        materialized_root = active / ".materialized"
        self.assertFalse((materialized_root / archive.stem).exists())
        self.assertFalse(list(materialized_root.glob("*.tmp.*")))

    def test_materializer_scavenges_only_dead_pid_temporary_directory(self) -> None:
        result = self._build()
        entry = result.shard_index.shards[0]
        archive = result.output_dir / entry.filename
        materialized_root = self.base / "orphan_active" / ".materialized"
        materialized_root.mkdir(parents=True)
        orphan = materialized_root / f"{archive.stem}.tmp.2147483647.012345abcdef"
        orphan.mkdir()
        (orphan / "partial.bin").write_bytes(b"partial")

        receipt = materialize_shard(
            archive,
            result.shard_index,
            materialized_root=materialized_root,
        )

        self.assertFalse(orphan.exists())
        self.assertTrue(receipt.dataset_report.ok)

    def test_builder_rejects_symlink_asset_and_cleanup_rejects_escape(self) -> None:
        rgb = next((self.dataset / "rgb").iterdir())
        replacement = self.base / "replacement.jpg"
        shutil.copyfile(rgb, replacement)
        rgb.unlink()
        rgb.symlink_to(replacement)
        with self.assertRaisesRegex(ShardFormatError, "symlink"):
            plan_target_state_shards(
                self.dataset,
                target_shard_size_bytes=1024,
                history_size=4,
                max_history_age_s=2.0,
            )

        materialized_root = self.base / "materialized"
        materialized_root.mkdir()
        outside = self.base / "outside"
        outside.mkdir()
        sentinel = outside / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(ShardMaterializationError):
            cleanup_materialized_path(outside, materialized_root=materialized_root)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_cpu_builder_cli_prints_phase_44_summary(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = build_shards_main(
                [
                    "--dataset-root",
                    str(self.dataset),
                    "--output-dir",
                    str(self.base / "cli_shards"),
                    "--target-shard-size-mib",
                    "0.000001",
                    "--history-size",
                    "4",
                    "--max-history-age-s",
                    "2.0",
                    "--split-seed",
                    "42",
                ]
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["ok"])
        for field in (
            "parent_dataset_sha256",
            "train_shard_count",
            "validation_shard_count",
            "test_shard_count",
            "episode_count",
            "frame_count",
            "sequence_count",
            "total_tar_bytes",
            "shard_index_path",
        ):
            self.assertIn(field, payload)

    def test_index_loader_rejects_symlink_and_fifo_without_blocking(self) -> None:
        result = self._build()
        index_path = result.shard_index.source_path
        linked = self.base / "linked_index.json"
        linked.symlink_to(index_path)
        with self.assertRaises(ShardFormatError):
            load_shard_index(linked)

        fifo = self.base / "index.fifo"
        os.mkfifo(fifo)
        with self.assertRaises(ShardFormatError):
            load_shard_index(fifo)
if __name__ == "__main__":
    unittest.main()
