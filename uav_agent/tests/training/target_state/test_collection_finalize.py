"""PC-side collection archive finalizer tests (CPU only)."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from hashlib import sha256
import io
import json
import os
from pathlib import Path
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
)
from datasets.target_state.sequence import build_sequences
from scripts.finalize_target_state_collection import main as finalize_main
from tests.training.target_state.test_dataset_schema import make_record
from training.target_state.collection_finalize import (
    COLLECTION_INDEX_PROTOCOL,
    COLLECTION_SCHEMA_VERSION,
    COLLECTION_SHARD_PROTOCOL,
    CollectionFinalizationError,
    finalize_target_state_collection,
    load_collection_index,
)
from training.target_state.collection_spool import TargetStateCollectionSpool


_COLLECTION_ID = "collection_test_001"
_YOLO_SHA = "a" * 64
_PROVENANCE = {
    "generation_commit_sha": "testcommit",
    "detector_prediction_source": "external_capture_spool_unverified",
    "candidate_id_source": "external_capture_spool_unverified",
    "detector_truth_association": "external_capture_spool_unverified",
    "oracle_usage": "offline_training_labels_only",
    "detector_deployment": None,
}


class _CollectionFixture:
    def __init__(self, root: Path, *, asset_conflict: bool = False) -> None:
        self.root = root
        self.shards = root / "collection_shards"
        self.shards.mkdir()
        self.index = root / "collection_index.json"
        entries: list[dict[str, object]] = []
        total_records = 0
        total_captures = 0
        for ordinal in (1, 2):
            episode_id = f"episode_{ordinal}"
            start = (ordinal - 1) * 5
            records = [
                make_record(
                    index,
                    episode_id=episode_id,
                    candidate_id=f"candidate_{ordinal}",
                )
                for index in range(start, start + 5)
            ]
            if ordinal == 1:
                source = records[0]
                records.append(
                    replace(
                        source,
                        frame_id="frame_shared_target",
                        detector_prediction=replace(
                            source.detector_prediction,
                            candidate_id="candidate_shared_target",
                            tracker_id="tracker_shared_target",
                        ),
                    )
                )
            if ordinal == 2 and asset_conflict:
                records[0] = replace(
                    records[0],
                    sensor_input=replace(
                        records[0].sensor_input,
                        rgb_path="rgb/frame_0.jpg",
                    ),
                )
            entry = self._write_shard(ordinal, records)
            entries.append(entry)
            total_records += int(entry["record_count"])
            total_captures += int(entry["physical_capture_count"])
        self.payload: dict[str, object] = {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "protocol": COLLECTION_INDEX_PROTOCOL,
            "collection_id": _COLLECTION_ID,
            "scene_seed": 123,
            "split_seed": 42,
            "history_size": 4,
            "max_history_age_s": 2.0,
            "yolo_model_sha256": _YOLO_SHA,
            "status": "completed",
            "shard_count": 2,
            "episode_count": 2,
            "physical_capture_count": total_captures,
            "record_count": total_records,
            "shards": entries,
        }
        self.write_index()

    def _write_shard(self, ordinal: int, records: list) -> dict[str, object]:
        workspace = self.root / f"workspace_{ordinal}"
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir()
        assets: set[str] = set()
        for record in records:
            for relative in (
                record.sensor_input.rgb_path,
                record.sensor_input.depth_path,
            ):
                assets.add(relative)
        for relative in sorted(assets):
            path = workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".jpg":
                Image.fromarray(np.full((24, 32, 3), 70 + ordinal, np.uint8)).save(path)
            else:
                np.save(
                    path,
                    np.full((24, 32), 4.0 + ordinal, np.float32),
                    allow_pickle=False,
                )
        (workspace / "frames.jsonl").write_text(
            "".join(
                json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        sequences = build_sequences(records, history_size=4, max_history_age_s=2.0)
        dataset_sha = compute_dataset_sha256(workspace, records)
        dataset_manifest = build_manifest(
            records,
            sequences,
            dataset_sha256=dataset_sha,
            split_seed=42,
            generation_commit_sha="testcommit",
        )
        dataset_manifest.update(_PROVENANCE)
        dataset_manifest.update(
            {
                "yolo_model_sha256": _YOLO_SHA,
                "history_size": 4,
                "max_history_age_s": 2.0,
            }
        )
        (workspace / "dataset_manifest.json").write_text(
            json.dumps(dataset_manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        episode_ids = sorted({record.episode_id for record in records})
        captures = {
            (record.episode_id, record.sensor_input.rgb_path) for record in records
        }
        collection_manifest = {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "protocol": COLLECTION_SHARD_PROTOCOL,
            "collection_id": _COLLECTION_ID,
            "shard_ordinal": ordinal,
            "episode_ids": episode_ids,
            "episode_count": len(episode_ids),
            "physical_capture_count": len(captures),
            "record_count": len(records),
            "scene_seed": 123,
            "split_seed": 42,
            "history_size": 4,
            "max_history_age_s": 2.0,
            "yolo_model_sha256": _YOLO_SHA,
            "dataset_sha256": dataset_sha,
            **_PROVENANCE,
        }
        (workspace / "collection_manifest.json").write_text(
            json.dumps(collection_manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        filename = f"shard_{_COLLECTION_ID}_{ordinal:06d}.tar"
        archive = self.shards / filename
        with tarfile.open(archive, "w:") as stream:
            for path in sorted(path for path in workspace.rglob("*") if path.is_file()):
                stream.add(path, arcname=path.relative_to(workspace).as_posix())
        payload = archive.read_bytes()
        return {
            "shard_ordinal": ordinal,
            "filename": filename,
            "archive_sha256": sha256(payload).hexdigest(),
            "archive_size_bytes": len(payload),
            "dataset_sha256": dataset_sha,
            "episode_ids": episode_ids,
            "episode_count": len(episode_ids),
            "physical_capture_count": len(captures),
            "record_count": len(records),
        }

    def write_index(self) -> None:
        self.index.write_text(
            json.dumps(self.payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def replace_archive(self, ordinal: int, transform) -> None:
        entry = self.payload["shards"][ordinal - 1]
        archive = self.shards / entry["filename"]
        replacement = self.root / f"replacement_{ordinal}.tar"
        with tarfile.open(archive, "r:") as source, tarfile.open(replacement, "w:") as output:
            for member in source.getmembers():
                source_file = source.extractfile(member) if member.isreg() else None
                output.addfile(member, source_file)
                if source_file is not None:
                    source_file.close()
            transform(output)
        os.replace(replacement, archive)
        payload = archive.read_bytes()
        entry["archive_sha256"] = sha256(payload).hexdigest()
        entry["archive_size_bytes"] = len(payload)
        self.write_index()


class TargetStateCollectionFinalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_finalizes_complete_parent_dataset_atomically_and_preserves_archives(self) -> None:
        fixture = _CollectionFixture(self.root)
        archives_before = {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in fixture.shards.glob("*.tar")
        }
        output = self.root / "complete_parent"
        result = finalize_target_state_collection(fixture.index, fixture.shards, output)
        self.assertEqual(result.record_count, 11)
        self.assertEqual(result.physical_capture_count, 10)
        self.assertTrue(result.dataset_report.ok)
        self.assertEqual(len(read_frame_records(output / "frames.jsonl")), 11)
        manifest = json.loads((output / "dataset_manifest.json").read_text())
        self.assertEqual(manifest["dataset_sha256"], result.dataset_sha256)
        self.assertEqual(manifest["source_collection_id"], _COLLECTION_ID)
        self.assertEqual(manifest["physical_capture_count"], 10)
        self.assertTrue(
            check_dataset(
                output,
                history_size=4,
                max_history_age_s=2.0,
                split_seed=42,
            ).ok
        )
        self.assertEqual(
            archives_before,
            {
                path.name: sha256(path.read_bytes()).hexdigest()
                for path in fixture.shards.glob("*.tar")
            },
        )
        self.assertFalse(any(path.name.startswith(".complete_parent") for path in self.root.iterdir()))

    def test_production_spool_to_pc_pull_to_finalizer_round_trip(self) -> None:
        pc_trans_root = self.root / "pc_trans"
        (pc_trans_root / "pc_trans").mkdir(parents=True)
        (pc_trans_root / "pc_trans" / "cli.py").write_text("# test stub\n")
        bridge = self.root / "bridge"
        pc_trans_config = self.root / "pc_trans_config.json"
        pc_trans_config.write_text(
            json.dumps({"bridge_root": str(bridge)}) + "\n",
            encoding="utf-8",
        )
        pc_archives = self.root / "pc_archives"
        pc_archives.mkdir()

        def seal_and_pull(source: Path) -> None:
            self.assertTrue(source.name.endswith(".tar.tmp"))
            ready = bridge / "collection_spool" / "ready" / source.name[:-4]
            os.replace(source, ready)
            shutil.copy2(ready, pc_archives / ready.name)
            ready.unlink()

        spool = TargetStateCollectionSpool.create(
            pc_trans_root=pc_trans_root,
            pc_trans_config=pc_trans_config,
            bridge_root=bridge,
            collection_session_dir=self.root / "sessions",
            shard_target_size_bytes=1,
            scene_seed=321,
            split_seed=42,
            history_size=4,
            max_history_age_s=2.0,
            yolo_model_sha256=_YOLO_SHA,
            generation_commit_sha="testcommit",
            collection_id="collection_production_roundtrip",
            seal_callback=seal_and_pull,
        )
        rgb = np.full((24, 32, 3), 90, np.uint8)
        depth = np.full((24, 32), 5.0, np.float32)
        for episode_index in range(2):
            episode_id = f"roundtrip_episode_{episode_index}"
            spool.begin_episode(episode_id, episode_index=episode_index)
            for frame_offset in range(5):
                frame_index = episode_index * 5 + frame_offset
                record = make_record(
                    frame_index,
                    episode_id=episode_id,
                    candidate_id=f"roundtrip_candidate_{episode_index}",
                )
                spool.append_capture(
                    (record,),
                    rgb=rgb,
                    depth_m=depth,
                    asset_id=f"capture_{frame_offset}",
                )
            self.assertTrue(spool.complete_episode())
        spool_result = spool.finalize()
        self.assertEqual(spool_result.shard_count, 2)
        self.assertEqual(tuple((bridge / "collection_spool" / "ready").iterdir()), ())
        self.assertEqual(len(tuple(pc_archives.glob("*.tar"))), 2)
        self.assertFalse(any(spool.workspaces_root.iterdir()))

        output = self.root / "production_roundtrip_parent"
        result = finalize_target_state_collection(
            spool_result.collection_index_path,
            pc_archives,
            output,
        )
        self.assertEqual(result.shard_count, 2)
        self.assertEqual(result.episode_count, 2)
        self.assertEqual(result.physical_capture_count, 10)
        self.assertEqual(result.record_count, 10)
        self.assertTrue(result.dataset_report.ok)
        self.assertEqual(len(tuple(pc_archives.glob("*.tar"))), 2)

    def test_cli_reports_success_and_never_offers_archive_deletion(self) -> None:
        fixture = _CollectionFixture(self.root)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = finalize_main(
                [
                    "--collection-index",
                    str(fixture.index),
                    "--shard-dir",
                    str(fixture.shards),
                    "--output-dir",
                    str(self.root / "cli_output"),
                ]
            )
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(stdout.getvalue())["archives_preserved"])

    def test_index_loader_canonicalizes_unordered_entries_with_optional_identity(self) -> None:
        fixture = _CollectionFixture(self.root)
        for entry in fixture.payload["shards"]:
            entry["collection_id"] = _COLLECTION_ID
        fixture.payload["shards"].reverse()
        fixture.write_index()
        loaded = load_collection_index(fixture.index)
        self.assertEqual(
            tuple(entry.shard_ordinal for entry in loaded.shards),
            (1, 2),
        )

    def test_rejects_incomplete_index_identity_totals_and_duplicate_episode(self) -> None:
        cases = ("status", "ordinal", "filename", "totals", "duplicate_episode")
        for case in cases:
            with self.subTest(case=case):
                case_root = self.root / case
                case_root.mkdir()
                fixture = _CollectionFixture(case_root)
                if case == "status":
                    fixture.payload["status"] = "collecting"
                elif case == "ordinal":
                    fixture.payload["shards"][1]["shard_ordinal"] = 3
                elif case == "filename":
                    fixture.payload["shards"][0]["filename"] = "../escape.tar"
                elif case == "totals":
                    fixture.payload["record_count"] += 1
                else:
                    fixture.payload["shards"][1]["episode_ids"] = ["episode_1"]
                fixture.write_index()
                with self.assertRaises(CollectionFinalizationError):
                    load_collection_index(fixture.index)

    def test_rejects_archive_sha_frame_and_asset_conflicts_without_output(self) -> None:
        for case in ("sha", "manifest_identity", "duplicate_frame", "asset_conflict"):
            with self.subTest(case=case):
                case_root = self.root / case
                case_root.mkdir()
                fixture = _CollectionFixture(
                    case_root, asset_conflict=(case == "asset_conflict")
                )
                if case == "sha":
                    archive = next(fixture.shards.glob("*.tar"))
                    with archive.open("ab") as stream:
                        stream.write(b"tamper")
                elif case == "manifest_identity":
                    fixture.payload["scene_seed"] = 999
                    fixture.write_index()
                elif case == "duplicate_frame":
                    # Rebuild shard 2 with a frame ID already present in shard 1,
                    # while keeping its asset path distinct and the shard valid.
                    workspace = case_root / "workspace_2"
                    records = list(read_frame_records(workspace / "frames.jsonl"))
                    records[0] = replace(records[0], frame_id="frame_0")
                    fixture.payload["shards"][1] = fixture._write_shard(2, records)
                    fixture.write_index()
                output = case_root / "output"
                with self.assertRaises(CollectionFinalizationError):
                    finalize_target_state_collection(
                        fixture.index, fixture.shards, output
                    )
                self.assertFalse(output.exists())
                self.assertEqual(len(tuple(fixture.shards.glob("*.tar"))), 2)

    def test_rejects_traversal_links_devices_fifo_duplicates_and_unexpected_files(self) -> None:
        cases = (
            ("absolute", "/escape", tarfile.REGTYPE),
            ("traversal", "../escape", tarfile.REGTYPE),
            ("symlink", "rgb/symlink.jpg", tarfile.SYMTYPE),
            ("hardlink", "rgb/hardlink.jpg", tarfile.LNKTYPE),
            ("device", "rgb/device", tarfile.CHRTYPE),
            ("fifo", "rgb/fifo", tarfile.FIFOTYPE),
            ("duplicate", "frames.jsonl", tarfile.REGTYPE),
            ("unexpected", "secret.txt", tarfile.REGTYPE),
        )
        for case, name, member_type in cases:
            with self.subTest(case=case):
                case_root = self.root / case
                case_root.mkdir()
                fixture = _CollectionFixture(case_root)

                def add_bad_member(stream: tarfile.TarFile) -> None:
                    member = tarfile.TarInfo(name)
                    member.type = member_type
                    if member_type == tarfile.REGTYPE:
                        payload = b"bad"
                        member.size = len(payload)
                        stream.addfile(member, io.BytesIO(payload))
                    else:
                        member.linkname = "frames.jsonl"
                        stream.addfile(member)

                fixture.replace_archive(1, add_bad_member)
                output = case_root / "output"
                with self.assertRaises(CollectionFinalizationError):
                    finalize_target_state_collection(
                        fixture.index, fixture.shards, output
                    )
                self.assertFalse(output.exists())
                self.assertFalse((case_root / "escape").exists())

    def test_symlink_index_archive_and_existing_output_fail_closed(self) -> None:
        fixture = _CollectionFixture(self.root)
        index_link = self.root / "index_link.json"
        index_link.symlink_to(fixture.index)
        with self.assertRaises(CollectionFinalizationError):
            load_collection_index(index_link)
        first_entry = fixture.payload["shards"][0]
        archive = fixture.shards / first_entry["filename"]
        real = fixture.shards / "real.tar"
        archive.rename(real)
        archive.symlink_to(real)
        with self.assertRaises(CollectionFinalizationError):
            finalize_target_state_collection(
                fixture.index, fixture.shards, self.root / "output"
            )
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(CollectionFinalizationError):
            finalize_target_state_collection(fixture.index, fixture.shards, existing)


if __name__ == "__main__":
    unittest.main()
