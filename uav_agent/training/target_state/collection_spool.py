"""Episode-atomic Target State collection spooling through ``pc_trans``.

The producer in this module understands Target State records and collection
manifests.  It deliberately treats ``pc_trans`` as a separate process and uses
only its public CLI to publish a completed opaque tar file.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import time
from typing import Callable, Mapping, Sequence
import uuid

import numpy as np

from datasets.target_state.dataset import check_dataset
from datasets.target_state.schema import TargetStateFrameRecord
from training.target_state.collector import (
    TargetStateCollectionError,
    TargetStateDatasetWriter,
    VerifiedYoloDeployment,
)


COLLECTION_SCHEMA_VERSION = 1
COLLECTION_SHARD_PROTOCOL = "target_state_collection_shard_v1"
COLLECTION_INDEX_PROTOCOL = "target_state_collection_index_v1"

_PROVENANCE_FIELDS = (
    "generation_commit_sha",
    "detector_prediction_source",
    "candidate_id_source",
    "detector_truth_association",
    "oracle_usage",
    "detector_deployment",
)
_SAFE_IDENTIFIER_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
)


class CollectionSpoolError(TargetStateCollectionError):
    """Raised when collection publication or recovery cannot be proven safe."""


@dataclass(frozen=True, slots=True)
class CollectionSpoolResult:
    collection_id: str
    session_dir: Path
    collection_index_path: Path
    shard_count: int
    episode_count: int
    physical_capture_count: int
    record_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "collection_id": self.collection_id,
            "session_dir": str(self.session_dir),
            "collection_index": str(self.collection_index_path),
            "shard_count": self.shard_count,
            "episode_count": self.episode_count,
            "physical_capture_count": self.physical_capture_count,
            "record_count": self.record_count,
        }


SealCallback = Callable[[Path], None]
SleepCallback = Callable[[float], None]


class TargetStateCollectionSpool:
    """Write complete episodes into immutable, resumable collection shards."""

    def __init__(
        self,
        *,
        collection_id: str,
        session_dir: Path,
        pc_trans_root: Path,
        pc_trans_config: Path,
        bridge_root: Path,
        shard_target_size_bytes: int,
        scene_seed: int,
        split_seed: int,
        history_size: int,
        max_history_age_s: float,
        verified_yolo_deployment: VerifiedYoloDeployment | None,
        yolo_model_sha256: str,
        generation_commit_sha: str,
        poll_interval_s: float,
        seal_callback: SealCallback | None,
        sleep_callback: SleepCallback,
    ) -> None:
        self.collection_id = _safe_identifier(collection_id, "collection_id")
        self.session_dir = session_dir
        self.pc_trans_root = pc_trans_root
        self.pc_trans_config = pc_trans_config
        self.bridge_root = bridge_root
        self.shard_target_size_bytes = shard_target_size_bytes
        self.scene_seed = scene_seed
        self.split_seed = split_seed
        self.history_size = history_size
        self.max_history_age_s = max_history_age_s
        self.verified_yolo_deployment = verified_yolo_deployment
        self.yolo_model_sha256 = _digest(yolo_model_sha256, "yolo_model_sha256")
        self.generation_commit_sha = generation_commit_sha
        self.poll_interval_s = poll_interval_s
        self._seal_callback = seal_callback
        self._sleep = sleep_callback

        self.session_path = self.session_dir / "session.json"
        self.shards_path = self.session_dir / "shards.jsonl"
        self.collection_index_path = self.session_dir / "collection_index.json"
        self.lock_path = self.session_dir / "session.lock"
        self.workspaces_root = self.session_dir / "workspaces"
        self.recovery_root = self.session_dir / "recovery"
        self.writing_root = self.bridge_root / "collection_spool" / "writing"
        self.ready_root = self.bridge_root / "collection_spool" / "ready"
        self.pause_flag = self.bridge_root / "control" / "pause_collection.flag"
        self._lock_descriptor = _acquire_session_lock(self.lock_path)

        self._entries: list[dict[str, object]] = []
        self._status = "collecting"
        self._pending: dict[str, object] | None = None
        self._writer: TargetStateDatasetWriter | None = None
        self._workspace: Path | None = None
        self._shard_episode_ids: list[str] = []
        self._shard_physical_capture_count = 0
        self._shard_record_count = 0
        self._active_episode_id: str | None = None
        self._active_episode_index: int | None = None
        self._episode_physical_capture_count = 0
        self._episode_record_count = 0

    @classmethod
    def create(
        cls,
        *,
        pc_trans_root: str | Path,
        pc_trans_config: str | Path,
        bridge_root: str | Path,
        collection_session_dir: str | Path,
        shard_target_size_bytes: int,
        scene_seed: int,
        split_seed: int = 42,
        history_size: int = 6,
        max_history_age_s: float = 2.0,
        verified_yolo_deployment: VerifiedYoloDeployment | None = None,
        yolo_model_sha256: str | None = None,
        generation_commit_sha: str | None = None,
        collection_id: str | None = None,
        poll_interval_s: float = 1.0,
        seal_callback: SealCallback | None = None,
        sleep_callback: SleepCallback = time.sleep,
    ) -> "TargetStateCollectionSpool":
        values = _validate_configuration(
            pc_trans_root=pc_trans_root,
            pc_trans_config=pc_trans_config,
            bridge_root=bridge_root,
            collection_session_dir=collection_session_dir,
            shard_target_size_bytes=shard_target_size_bytes,
            scene_seed=scene_seed,
            split_seed=split_seed,
            history_size=history_size,
            max_history_age_s=max_history_age_s,
            verified_yolo_deployment=verified_yolo_deployment,
            yolo_model_sha256=yolo_model_sha256,
            poll_interval_s=poll_interval_s,
        )
        parent = _ensure_real_directory(values["collection_session_dir"])
        resolved_collection_id = (
            f"collection_{uuid.uuid4().hex}"
            if collection_id is None
            else _safe_identifier(collection_id, "collection_id")
        )
        session_dir = parent / resolved_collection_id
        try:
            session_dir.mkdir(mode=0o750)
        except FileExistsError as exc:
            raise CollectionSpoolError(
                f"collection session already exists: {session_dir}"
            ) from exc
        _ensure_real_directory(session_dir / "workspaces")
        _ensure_real_directory(session_dir / "recovery")
        _initialize_empty_file(session_dir / "shards.jsonl")
        spool = cls(
            collection_id=resolved_collection_id,
            session_dir=session_dir,
            pc_trans_root=values["pc_trans_root"],
            pc_trans_config=values["pc_trans_config"],
            bridge_root=values["bridge_root"],
            shard_target_size_bytes=values["shard_target_size_bytes"],
            scene_seed=values["scene_seed"],
            split_seed=values["split_seed"],
            history_size=values["history_size"],
            max_history_age_s=values["max_history_age_s"],
            verified_yolo_deployment=verified_yolo_deployment,
            yolo_model_sha256=values["yolo_model_sha256"],
            generation_commit_sha=(
                generation_commit_sha
                or os.environ.get("UAV_AGENT_TRAINING_COMMIT_SHA", "nogit")
            ),
            poll_interval_s=values["poll_interval_s"],
            seal_callback=seal_callback,
            sleep_callback=sleep_callback,
        )
        spool._write_index(status="collecting")
        spool._write_session(status="collecting", pending=None)
        return spool

    @classmethod
    def resume(
        cls,
        resume_session: str | Path,
        *,
        pc_trans_root: str | Path,
        pc_trans_config: str | Path,
        bridge_root: str | Path,
        shard_target_size_bytes: int,
        scene_seed: int,
        split_seed: int = 42,
        history_size: int = 6,
        max_history_age_s: float = 2.0,
        verified_yolo_deployment: VerifiedYoloDeployment | None = None,
        yolo_model_sha256: str | None = None,
        poll_interval_s: float = 1.0,
        seal_callback: SealCallback | None = None,
        sleep_callback: SleepCallback = time.sleep,
    ) -> "TargetStateCollectionSpool":
        session_dir = _require_real_directory(resume_session, "resume session")
        values = _validate_configuration(
            pc_trans_root=pc_trans_root,
            pc_trans_config=pc_trans_config,
            bridge_root=bridge_root,
            collection_session_dir=session_dir.parent,
            shard_target_size_bytes=shard_target_size_bytes,
            scene_seed=scene_seed,
            split_seed=split_seed,
            history_size=history_size,
            max_history_age_s=max_history_age_s,
            verified_yolo_deployment=verified_yolo_deployment,
            yolo_model_sha256=yolo_model_sha256,
            poll_interval_s=poll_interval_s,
        )
        session = _read_json_object(session_dir / "session.json", "session journal")
        collection_id = _safe_identifier(session.get("collection_id"), "collection_id")
        if session_dir.name != collection_id:
            raise CollectionSpoolError(
                "resume session directory name does not match collection_id"
            )
        expected_identity: dict[str, object] = {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "collection_id": collection_id,
            "scene_seed": values["scene_seed"],
            "split_seed": values["split_seed"],
            "history_size": values["history_size"],
            "max_history_age_s": values["max_history_age_s"],
            "yolo_model_sha256": values["yolo_model_sha256"],
            "shard_target_size_bytes": values["shard_target_size_bytes"],
            "pc_trans_root": str(values["pc_trans_root"]),
            "pc_trans_config": str(values["pc_trans_config"]),
            "bridge_root": str(values["bridge_root"]),
            "detector_deployment": (
                None
                if verified_yolo_deployment is None
                else verified_yolo_deployment.to_manifest_dict()
            ),
        }
        for field, expected in expected_identity.items():
            if session.get(field) != expected:
                raise CollectionSpoolError(
                    f"resume session {field} does not match requested collection"
                )
        session_status = session.get("status")
        if session_status not in {
            "collecting",
            "paused",
            "sealing",
            "interrupted",
            "completing",
            "completed",
        }:
            raise CollectionSpoolError("session status is invalid")
        if session_status == "completed":
            raise CollectionSpoolError("collection session is already completed")
        generation_commit_sha = session.get("generation_commit_sha")
        if not isinstance(generation_commit_sha, str) or not generation_commit_sha:
            raise CollectionSpoolError("session generation_commit_sha is invalid")
        spool = cls(
            collection_id=collection_id,
            session_dir=session_dir,
            pc_trans_root=values["pc_trans_root"],
            pc_trans_config=values["pc_trans_config"],
            bridge_root=values["bridge_root"],
            shard_target_size_bytes=values["shard_target_size_bytes"],
            scene_seed=values["scene_seed"],
            split_seed=values["split_seed"],
            history_size=values["history_size"],
            max_history_age_s=values["max_history_age_s"],
            verified_yolo_deployment=verified_yolo_deployment,
            yolo_model_sha256=values["yolo_model_sha256"],
            generation_commit_sha=generation_commit_sha,
            poll_interval_s=values["poll_interval_s"],
            seal_callback=seal_callback,
            sleep_callback=sleep_callback,
        )
        locked_session = _read_json_object(
            session_dir / "session.json", "session journal"
        )
        if dict(locked_session) != dict(session):
            spool.close()
            raise CollectionSpoolError(
                "session changed while its process lock was being acquired; retry resume"
            )
        spool._entries = _read_shard_journal(spool.shards_path, collection_id)
        spool._pending = _validate_pending(session.get("pending_shard"), collection_id)
        if session_status == "completing":
            if spool._pending is not None or not spool._entries:
                spool.close()
                raise CollectionSpoolError(
                    "completing session has inconsistent durable state"
                )
            spool._status = "completed"
            spool._write_index(status="completed")
            spool._write_session(status="completed", pending=None)
            spool.close()
            raise CollectionSpoolError(
                "collection completion was recovered; the session is already completed"
            )
        spool._reconcile_session_counters(session)
        spool._recover_pending_or_partial_workspace()
        spool._cleanup_committed_workspaces()
        spool._write_index(status="collecting")
        spool._write_session(status="collecting", pending=spool._pending)
        return spool

    @property
    def next_episode_index(self) -> int:
        return sum(int(entry["episode_count"]) for entry in self._entries)

    @property
    def next_shard_ordinal(self) -> int:
        return len(self._entries) + 1

    @property
    def sealed_episode_count(self) -> int:
        return self.next_episode_index

    @property
    def sealed_physical_capture_count(self) -> int:
        return sum(int(entry["physical_capture_count"]) for entry in self._entries)

    @property
    def sealed_record_count(self) -> int:
        return sum(int(entry["record_count"]) for entry in self._entries)

    @property
    def shard_count(self) -> int:
        return len(self._entries)

    @property
    def pause_requested(self) -> bool:
        return os.path.lexists(self.pause_flag)

    def wait_before_episode(self) -> None:
        if self._active_episode_id is not None:
            raise CollectionSpoolError("cannot pause while an episode is active")
        announced = False
        while self.pause_requested:
            if not announced:
                self._write_session(status="paused", pending=self._pending)
                announced = True
            self._sleep(self.poll_interval_s)
        if announced:
            self._write_session(status="collecting", pending=self._pending)

    def begin_episode(self, episode_id: str, *, episode_index: int) -> None:
        if self._status == "completed":
            raise CollectionSpoolError("collection session is already completed")
        if self._active_episode_id is not None:
            raise CollectionSpoolError("the prior episode has not been completed")
        if self.pause_requested:
            raise CollectionSpoolError(
                "pause_collection.flag exists; call wait_before_episode first"
            )
        expected_index = self.next_episode_index + len(self._shard_episode_ids)
        if isinstance(episode_index, bool) or episode_index != expected_index:
            raise CollectionSpoolError(
                f"episode_index must be the next unsealed index {expected_index}"
            )
        safe_episode_id = _safe_identifier(episode_id, "episode_id")
        sealed_ids = {
            value
            for entry in self._entries
            for value in entry["episode_ids"]  # type: ignore[union-attr]
        }
        if safe_episode_id in sealed_ids or safe_episode_id in self._shard_episode_ids:
            raise CollectionSpoolError(f"duplicate episode_id: {safe_episode_id}")
        if self._writer is None:
            self._open_shard_writer()
        self._active_episode_id = safe_episode_id
        self._active_episode_index = episode_index
        self._episode_physical_capture_count = 0
        self._episode_record_count = 0

    def append(
        self,
        record: TargetStateFrameRecord,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        instance_mask: np.ndarray | None = None,
        asset_id: str | None = None,
    ) -> TargetStateFrameRecord:
        if self._writer is None or self._active_episode_id is None:
            raise CollectionSpoolError("begin_episode must be called before append")
        if record.episode_id != self._active_episode_id:
            raise CollectionSpoolError(
                "record episode_id does not match the active episode"
            )
        stored = self._writer.append(
            record,
            rgb=rgb,
            depth_m=depth_m,
            instance_mask=instance_mask,
            asset_id=asset_id,
        )
        self._episode_record_count += 1
        return stored

    def append_capture(
        self,
        records: Sequence[TargetStateFrameRecord],
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        instance_mask: np.ndarray | None = None,
        asset_id: str | None = None,
    ) -> tuple[TargetStateFrameRecord, ...]:
        stored = tuple(
            self.append(
                record,
                rgb=rgb,
                depth_m=depth_m,
                instance_mask=instance_mask,
                asset_id=asset_id,
            )
            for record in records
        )
        self.record_physical_capture()
        return stored

    def record_physical_capture(self) -> None:
        if self._active_episode_id is None:
            raise CollectionSpoolError(
                "begin_episode must be called before recording a capture"
            )
        self._episode_physical_capture_count += 1

    def complete_episode(self) -> bool:
        """Commit the active episode in-memory and seal only at a boundary.

        Returns ``True`` when this boundary caused a shard to be sealed.
        """

        if self._active_episode_id is None or self._writer is None:
            raise CollectionSpoolError("there is no active episode to complete")
        if self._episode_physical_capture_count <= 0:
            raise CollectionSpoolError("a completed episode must contain a capture")
        if self._episode_record_count <= 0:
            raise CollectionSpoolError("a completed episode must contain a record")
        self._shard_episode_ids.append(self._active_episode_id)
        self._shard_physical_capture_count += self._episode_physical_capture_count
        self._shard_record_count += self._episode_record_count
        self._active_episode_id = None
        self._active_episode_index = None
        self._episode_physical_capture_count = 0
        self._episode_record_count = 0
        should_seal = (
            _directory_size(self._workspace) >= self.shard_target_size_bytes
            or self.pause_requested
        )
        if should_seal:
            self.seal_current_shard()
        return should_seal

    def seal_current_shard(self) -> dict[str, object] | None:
        if self._active_episode_id is not None:
            raise CollectionSpoolError("cannot split an active episode across shards")
        if self._writer is None:
            return None
        if not self._shard_episode_ids or self._workspace is None:
            raise CollectionSpoolError("cannot seal an empty collection shard")
        ordinal = self.next_shard_ordinal
        workspace = self._workspace
        manifest_path, report = self._writer.finalize()
        self._writer = None
        if not report.ok:
            raise CollectionSpoolError(
                "collection shard failed check_dataset: "
                + "; ".join(report.errors[:5])
            )
        dataset_manifest = _read_json_object(manifest_path, "dataset manifest")
        dataset_sha = _digest(
            dataset_manifest.get("dataset_sha256"), "dataset_sha256"
        )
        canonical_report = check_dataset(
            workspace,
            history_size=self.history_size,
            max_history_age_s=self.max_history_age_s,
            split_seed=self.split_seed,
        )
        if (
            not canonical_report.ok
            or canonical_report.dataset_sha256 != dataset_sha
        ):
            raise CollectionSpoolError(
                "finalized shard failed canonical check_dataset: "
                + "; ".join(canonical_report.errors[:5])
            )
        expected_episodes = sorted(self._shard_episode_ids)
        actual_splits = dataset_manifest.get("episode_splits")
        if not isinstance(actual_splits, Mapping) or sorted(actual_splits) != expected_episodes:
            raise CollectionSpoolError(
                "dataset manifest episode IDs do not match completed episodes"
            )
        if dataset_manifest.get("frame_count") != self._shard_record_count:
            raise CollectionSpoolError(
                "dataset manifest frame_count does not match record_count"
            )
        if (
            dataset_manifest.get("physical_capture_count")
            != self._shard_physical_capture_count
        ):
            raise CollectionSpoolError(
                "dataset manifest physical_capture_count does not match collector count"
            )
        collection_manifest: dict[str, object] = {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "protocol": COLLECTION_SHARD_PROTOCOL,
            "collection_id": self.collection_id,
            "shard_ordinal": ordinal,
            "episode_ids": list(self._shard_episode_ids),
            "episode_count": len(self._shard_episode_ids),
            "physical_capture_count": self._shard_physical_capture_count,
            "record_count": self._shard_record_count,
            "scene_seed": self.scene_seed,
            "split_seed": self.split_seed,
            "history_size": self.history_size,
            "max_history_age_s": self.max_history_age_s,
            "yolo_model_sha256": self.yolo_model_sha256,
            "dataset_sha256": dataset_sha,
        }
        for field in _PROVENANCE_FIELDS:
            if field not in dataset_manifest:
                raise CollectionSpoolError(
                    f"dataset manifest is missing provenance field {field}"
                )
            collection_manifest[field] = dataset_manifest[field]
        _atomic_write_json(workspace / "collection_manifest.json", collection_manifest)

        filename = f"shard_{self.collection_id}_{ordinal:06d}.tar"
        tar_tmp = self.writing_root / f"{filename}.tmp"
        base_entry: dict[str, object] = {
            "shard_ordinal": ordinal,
            "filename": filename,
            "dataset_sha256": dataset_sha,
            "episode_ids": list(self._shard_episode_ids),
            "episode_count": len(self._shard_episode_ids),
            "physical_capture_count": self._shard_physical_capture_count,
            "record_count": self._shard_record_count,
        }
        building_pending = {
            "phase": "building_tar",
            "entry": base_entry,
            "workspace_path": str(workspace),
            "tar_tmp_path": str(tar_tmp),
        }
        self._pending = building_pending
        self._write_session(status="sealing", pending=building_pending)
        _write_deterministic_tar(workspace, tar_tmp)
        archive_sha, archive_size = _hash_regular_file(tar_tmp)
        entry = {
            **base_entry,
            "archive_sha256": archive_sha,
            "archive_size_bytes": archive_size,
        }
        pending = {
            "phase": "ready_to_seal",
            "entry": entry,
            "workspace_path": str(workspace),
            "tar_tmp_path": str(tar_tmp),
        }
        self._pending = pending
        self._write_session(status="sealing", pending=pending)
        self._publish_pending(pending)
        return entry

    def finalize(self) -> CollectionSpoolResult:
        if self._active_episode_id is not None:
            raise CollectionSpoolError("cannot finalize in the middle of an episode")
        if self._pending is not None:
            self._recover_pending_or_partial_workspace()
        if self._writer is not None:
            self.seal_current_shard()
        if not self._entries:
            raise CollectionSpoolError("cannot finalize an empty collection")
        self._write_session(status="completing", pending=None)
        self._write_index(status="completed")
        self._status = "completed"
        self._write_session(status="completed", pending=None)
        result = CollectionSpoolResult(
            collection_id=self.collection_id,
            session_dir=self.session_dir,
            collection_index_path=self.collection_index_path,
            shard_count=self.shard_count,
            episode_count=self.sealed_episode_count,
            physical_capture_count=self.sealed_physical_capture_count,
            record_count=self.sealed_record_count,
        )
        self.close()
        return result

    def abort(self) -> None:
        """Retain all partial evidence and mark the resumable session interrupted."""

        try:
            if self._writer is not None:
                self._writer.abort()
                self._writer = None
            if self._status != "completed":
                self._write_session(status="interrupted", pending=self._pending)
        finally:
            self.close()

    def close(self) -> None:
        descriptor = getattr(self, "_lock_descriptor", None)
        if descriptor is not None:
            self._lock_descriptor = None
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def __del__(self) -> None:  # pragma: no cover - best-effort process cleanup
        try:
            self.close()
        except Exception:
            pass

    def _open_shard_writer(self) -> None:
        ordinal = self.next_shard_ordinal
        workspace = self.workspaces_root / f"shard_{ordinal:06d}"
        if workspace.exists() or workspace.is_symlink():
            raise CollectionSpoolError(
                f"next shard workspace already exists: {workspace}"
            )
        workspace.mkdir(mode=0o750)
        self._workspace = workspace
        self._writer = TargetStateDatasetWriter(
            workspace,
            yolo_model_sha256=(
                self.yolo_model_sha256
                if self.verified_yolo_deployment is None
                else None
            ),
            verified_yolo_deployment=self.verified_yolo_deployment,
            split_seed=self.split_seed,
            history_size=self.history_size,
            max_history_age_s=self.max_history_age_s,
            generation_commit_sha=self.generation_commit_sha,
        )

    def _publish_pending(self, pending: Mapping[str, object]) -> None:
        entry_value = pending.get("entry")
        if not isinstance(entry_value, Mapping):
            raise CollectionSpoolError("pending shard has no valid entry")
        entry = _validate_entry(dict(entry_value), self.collection_id)
        tar_tmp = _require_pending_path(
            pending.get("tar_tmp_path"), self.writing_root, "pending tar"
        )
        expected_tar = self.writing_root / f"{entry['filename']}.tmp"
        if tar_tmp != expected_tar:
            raise CollectionSpoolError(
                "pending tar path does not match shard identity"
            )
        workspace = _require_pending_path(
            pending.get("workspace_path"), self.workspaces_root, "pending workspace"
        )
        expected_workspace = self.workspaces_root / (
            f"shard_{int(entry['shard_ordinal']):06d}"
        )
        if workspace != expected_workspace:
            raise CollectionSpoolError(
                "pending workspace path does not match shard identity"
            )
        ready = self.ready_root / str(entry["filename"])
        tmp_exists = os.path.lexists(tar_tmp)
        ready_exists = os.path.lexists(ready)
        if tmp_exists and ready_exists:
            raise CollectionSpoolError(
                "pending shard exists in both writing and ready directories"
            )
        if tmp_exists:
            actual_sha, actual_size = _hash_regular_file(tar_tmp)
            if (
                actual_sha != entry["archive_sha256"]
                or actual_size != entry["archive_size_bytes"]
            ):
                raise CollectionSpoolError("pending tar does not match its journal")
            self._invoke_seal(tar_tmp)
            if os.path.lexists(tar_tmp):
                raise CollectionSpoolError("pc_trans seal left the source tar in writing")
            # The PC can remove ready immediately after a successful seal; the
            # CLI success is the publication acknowledgement in this process.
        elif ready_exists:
            actual_sha, actual_size = _hash_regular_file(ready)
            if (
                actual_sha != entry["archive_sha256"]
                or actual_size != entry["archive_size_bytes"]
            ):
                raise CollectionSpoolError("ready tar does not match its journal")
        else:
            # Absence alone is not a transfer acknowledgement.  Rebuild the
            # deterministic archive from the retained finalized workspace and
            # publish it again.  If a PC already owns the identical filename,
            # its checksum-based pull remains idempotent.
            if not workspace.is_dir() or workspace.is_symlink():
                raise CollectionSpoolError(
                    "pending tar and ready copy are absent, and the finalized "
                    "workspace is unavailable"
                )
            self._validate_pending_workspace(workspace, entry)
            _write_deterministic_tar(workspace, tar_tmp)
            actual_sha, actual_size = _hash_regular_file(tar_tmp)
            if (
                actual_sha != entry["archive_sha256"]
                or actual_size != entry["archive_size_bytes"]
            ):
                raise CollectionSpoolError(
                    "deterministically rebuilt tar does not match its journal"
                )
            self._invoke_seal(tar_tmp)
            if os.path.lexists(tar_tmp):
                raise CollectionSpoolError(
                    "pc_trans seal left the rebuilt source tar in writing"
                )
        self._commit_entry(entry)
        if os.path.lexists(workspace):
            _remove_owned_workspace(workspace, self.workspaces_root)
        self._reset_current_shard()

    def _invoke_seal(self, tar_tmp: Path) -> None:
        if self._seal_callback is not None:
            self._seal_callback(tar_tmp)
            return
        command = [
            sys.executable,
            "-m",
            "pc_trans.cli",
            "--config",
            str(self.pc_trans_config),
            "seal",
            "--src",
            str(tar_tmp),
        ]
        completed = subprocess.run(
            command,
            cwd=self.pc_trans_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise CollectionSpoolError(
                f"pc_trans seal failed with status {completed.returncode}: {detail}"
            )

    def _commit_entry(self, entry: dict[str, object]) -> None:
        ordinal = int(entry["shard_ordinal"])
        if ordinal <= len(self._entries):
            if self._entries[ordinal - 1] != entry:
                raise CollectionSpoolError("committed shard journal entry conflicts")
        else:
            if ordinal != len(self._entries) + 1:
                raise CollectionSpoolError("collection shard ordinal is not contiguous")
            updated_entries = [*self._entries, entry]
            _atomic_write_jsonl(self.shards_path, updated_entries)
            self._entries = updated_entries
        self._pending = None
        self._write_index(status="collecting")
        self._write_session(status="collecting", pending=None)

    def _recover_pending_or_partial_workspace(self) -> None:
        if self._pending is not None:
            phase = self._pending.get("phase")
            if phase == "building_tar":
                entry_value = self._pending.get("entry")
                if not isinstance(entry_value, Mapping):
                    raise CollectionSpoolError("building_tar journal has no entry")
                workspace = _require_pending_path(
                    self._pending.get("workspace_path"),
                    self.workspaces_root,
                    "pending workspace",
                )
                tar_tmp = _require_pending_path(
                    self._pending.get("tar_tmp_path"),
                    self.writing_root,
                    "pending tar",
                )
                ordinal = entry_value.get("shard_ordinal")
                filename = entry_value.get("filename")
                if (
                    isinstance(ordinal, bool)
                    or not isinstance(ordinal, int)
                    or ordinal != self.next_shard_ordinal
                    or filename
                    != f"shard_{self.collection_id}_{ordinal:06d}.tar"
                    or workspace
                    != self.workspaces_root / f"shard_{ordinal:06d}"
                    or tar_tmp != self.writing_root / f"{filename}.tmp"
                ):
                    raise CollectionSpoolError(
                        "building_tar paths do not match shard identity"
                    )
                if not workspace.is_dir() or workspace.is_symlink():
                    raise CollectionSpoolError(
                        "cannot rebuild pending tar without its workspace"
                    )
                self._validate_pending_workspace(workspace, dict(entry_value))
                if os.path.lexists(tar_tmp):
                    _quarantine_incomplete_tar(tar_tmp)
                _write_deterministic_tar(workspace, tar_tmp)
                archive_sha, archive_size = _hash_regular_file(tar_tmp)
                entry = {
                    **dict(entry_value),
                    "archive_sha256": archive_sha,
                    "archive_size_bytes": archive_size,
                }
                self._pending = {
                    **dict(self._pending),
                    "phase": "ready_to_seal",
                    "entry": entry,
                }
                self._write_session(status="sealing", pending=self._pending)
            if self._pending is None or self._pending.get("phase") != "ready_to_seal":
                raise CollectionSpoolError("session pending_shard phase is unsupported")
            self._publish_pending(self._pending)
            return

        workspace = self.workspaces_root / f"shard_{self.next_shard_ordinal:06d}"
        if os.path.lexists(workspace):
            # No durable pending journal means this shard was never published.
            # Preserve it for forensics and restart at the sealed boundary.
            if workspace.is_symlink() or not workspace.is_dir():
                raise CollectionSpoolError(
                    f"unsealed workspace is not a real directory: {workspace}"
                )
            recovered = self.recovery_root / (
                f"{workspace.name}.partial.{uuid.uuid4().hex}"
            )
            os.rename(workspace, recovered)
            _fsync_directory(self.workspaces_root)
            _fsync_directory(self.recovery_root)
        tar_tmp = self.writing_root / (
            f"shard_{self.collection_id}_{self.next_shard_ordinal:06d}.tar.tmp"
        )
        if os.path.lexists(tar_tmp):
            raise CollectionSpoolError(
                "an unjournaled tar remains in collection_spool/writing; "
                "it was retained and requires operator inspection"
            )

    def _validate_pending_workspace(
        self, workspace: Path, entry: Mapping[str, object]
    ) -> None:
        dataset_manifest = _read_json_object(
            workspace / "dataset_manifest.json", "dataset manifest"
        )
        collection_manifest = _read_json_object(
            workspace / "collection_manifest.json", "collection manifest"
        )
        expected = {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "protocol": COLLECTION_SHARD_PROTOCOL,
            "collection_id": self.collection_id,
            "shard_ordinal": entry.get("shard_ordinal"),
            "episode_ids": entry.get("episode_ids"),
            "episode_count": entry.get("episode_count"),
            "physical_capture_count": entry.get("physical_capture_count"),
            "record_count": entry.get("record_count"),
            "scene_seed": self.scene_seed,
            "split_seed": self.split_seed,
            "history_size": self.history_size,
            "max_history_age_s": self.max_history_age_s,
            "yolo_model_sha256": self.yolo_model_sha256,
            "dataset_sha256": entry.get("dataset_sha256"),
        }
        for field, value in expected.items():
            if collection_manifest.get(field) != value:
                raise CollectionSpoolError(
                    f"pending workspace collection_manifest {field} mismatch"
                )
        dataset_sha = _digest(entry.get("dataset_sha256"), "dataset_sha256")
        if dataset_manifest.get("dataset_sha256") != dataset_sha:
            raise CollectionSpoolError(
                "pending workspace dataset manifest SHA mismatch"
            )
        report = check_dataset(
            workspace,
            history_size=self.history_size,
            max_history_age_s=self.max_history_age_s,
            split_seed=self.split_seed,
        )
        if not report.ok or report.dataset_sha256 != dataset_sha:
            raise CollectionSpoolError(
                "pending workspace failed canonical check_dataset: "
                + "; ".join(report.errors[:5])
            )

    def _reconcile_session_counters(self, session: Mapping[str, object]) -> None:
        expected = {
            "next_episode_index": self.next_episode_index,
            "next_shard_ordinal": self.next_shard_ordinal,
            "sealed_episode_count": self.sealed_episode_count,
            "sealed_physical_capture_count": self.sealed_physical_capture_count,
            "sealed_record_count": self.sealed_record_count,
        }
        # The shard journal is authoritative.  A crash after appending it but
        # before replacing session.json legitimately leaves stale counters.
        for field, value in expected.items():
            actual = session.get(field)
            if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
                raise CollectionSpoolError(f"session {field} is invalid")
            if actual > value:
                raise CollectionSpoolError(
                    f"session {field} is ahead of the durable shard journal"
                )

    def _cleanup_committed_workspaces(self) -> None:
        for entry in self._entries:
            workspace = self.workspaces_root / (
                f"shard_{int(entry['shard_ordinal']):06d}"
            )
            if os.path.lexists(workspace):
                _remove_owned_workspace(workspace, self.workspaces_root)

    def _reset_current_shard(self) -> None:
        self._writer = None
        self._workspace = None
        self._shard_episode_ids = []
        self._shard_physical_capture_count = 0
        self._shard_record_count = 0
        self._active_episode_id = None
        self._active_episode_index = None
        self._episode_physical_capture_count = 0
        self._episode_record_count = 0

    def _session_payload(
        self, *, status: str, pending: Mapping[str, object] | None
    ) -> dict[str, object]:
        return {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "collection_id": self.collection_id,
            "scene_seed": self.scene_seed,
            "split_seed": self.split_seed,
            "history_size": self.history_size,
            "max_history_age_s": self.max_history_age_s,
            "yolo_model_sha256": self.yolo_model_sha256,
            "generation_commit_sha": self.generation_commit_sha,
            "detector_deployment": (
                None
                if self.verified_yolo_deployment is None
                else self.verified_yolo_deployment.to_manifest_dict()
            ),
            "shard_target_size_bytes": self.shard_target_size_bytes,
            "pc_trans_root": str(self.pc_trans_root),
            "pc_trans_config": str(self.pc_trans_config),
            "bridge_root": str(self.bridge_root),
            "next_episode_index": self.next_episode_index,
            "next_shard_ordinal": self.next_shard_ordinal,
            "sealed_episode_count": self.sealed_episode_count,
            "sealed_physical_capture_count": self.sealed_physical_capture_count,
            "sealed_record_count": self.sealed_record_count,
            "status": status,
            "pending_shard": None if pending is None else dict(pending),
        }

    def _write_session(
        self, *, status: str, pending: Mapping[str, object] | None
    ) -> None:
        self._status = status
        _atomic_write_json(
            self.session_path, self._session_payload(status=status, pending=pending)
        )

    def _index_payload(self, *, status: str) -> dict[str, object]:
        return {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "protocol": COLLECTION_INDEX_PROTOCOL,
            "collection_id": self.collection_id,
            "scene_seed": self.scene_seed,
            "split_seed": self.split_seed,
            "history_size": self.history_size,
            "max_history_age_s": self.max_history_age_s,
            "yolo_model_sha256": self.yolo_model_sha256,
            "status": status,
            "shard_count": self.shard_count,
            "episode_count": self.sealed_episode_count,
            "physical_capture_count": self.sealed_physical_capture_count,
            "record_count": self.sealed_record_count,
            "shards": self._entries,
        }

    def _write_index(self, *, status: str) -> None:
        _atomic_write_json(
            self.collection_index_path, self._index_payload(status=status)
        )


def _validate_configuration(
    *,
    pc_trans_root: str | Path,
    pc_trans_config: str | Path,
    bridge_root: str | Path,
    collection_session_dir: str | Path,
    shard_target_size_bytes: int,
    scene_seed: int,
    split_seed: int,
    history_size: int,
    max_history_age_s: float,
    verified_yolo_deployment: VerifiedYoloDeployment | None,
    yolo_model_sha256: str | None,
    poll_interval_s: float,
) -> dict[str, object]:
    resolved_pc_root = _require_real_directory(pc_trans_root, "pc_trans root")
    if not (resolved_pc_root / "pc_trans" / "cli.py").is_file():
        raise CollectionSpoolError(
            f"pc_trans root has no pc_trans/cli.py: {resolved_pc_root}"
        )
    resolved_config = _require_regular_file(pc_trans_config, "pc_trans config")
    config_payload = _read_json_object(resolved_config, "pc_trans config")
    configured_bridge = config_payload.get("bridge_root")
    if not isinstance(configured_bridge, str) or not configured_bridge:
        raise CollectionSpoolError("pc_trans config bridge_root is invalid")
    requested_bridge = Path(bridge_root).expanduser()
    if requested_bridge.resolve(strict=False) != Path(configured_bridge).expanduser().resolve(strict=False):
        raise CollectionSpoolError(
            "--bridge-root must exactly match pc_trans config bridge_root"
        )
    resolved_bridge = _ensure_real_directory(requested_bridge)
    if resolved_bridge in {Path("/"), Path("/home"), Path.home().resolve()}:
        raise CollectionSpoolError("bridge_root is dangerously broad")
    _ensure_real_directory(resolved_bridge / "collection_spool" / "writing")
    _ensure_real_directory(resolved_bridge / "collection_spool" / "ready")
    _ensure_real_directory(resolved_bridge / "control")
    if (
        isinstance(shard_target_size_bytes, bool)
        or not isinstance(shard_target_size_bytes, int)
        or shard_target_size_bytes <= 0
    ):
        raise CollectionSpoolError("shard_target_size_bytes must be a positive integer")
    for name, value in (("scene_seed", scene_seed), ("split_seed", split_seed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CollectionSpoolError(f"{name} must be a non-negative integer")
    if isinstance(history_size, bool) or not isinstance(history_size, int) or not 4 <= history_size <= 8:
        raise CollectionSpoolError("history_size must be within [4, 8]")
    if (
        isinstance(max_history_age_s, bool)
        or not isinstance(max_history_age_s, (int, float))
        or not math.isfinite(float(max_history_age_s))
        or float(max_history_age_s) <= 0.0
    ):
        raise CollectionSpoolError("max_history_age_s must be positive and finite")
    if (
        isinstance(poll_interval_s, bool)
        or not isinstance(poll_interval_s, (int, float))
        or not math.isfinite(float(poll_interval_s))
        or float(poll_interval_s) <= 0.0
    ):
        raise CollectionSpoolError("poll_interval_s must be positive and finite")
    if verified_yolo_deployment is not None and not isinstance(
        verified_yolo_deployment, VerifiedYoloDeployment
    ):
        raise CollectionSpoolError(
            "verified_yolo_deployment must be VerifiedYoloDeployment or None"
        )
    resolved_yolo_sha = (
        verified_yolo_deployment.model_sha256
        if verified_yolo_deployment is not None
        else yolo_model_sha256
    )
    return {
        "pc_trans_root": resolved_pc_root,
        "pc_trans_config": resolved_config,
        "bridge_root": resolved_bridge,
        "collection_session_dir": Path(collection_session_dir)
        .expanduser()
        .resolve(strict=False),
        "shard_target_size_bytes": shard_target_size_bytes,
        "scene_seed": scene_seed,
        "split_seed": split_seed,
        "history_size": history_size,
        "max_history_age_s": float(max_history_age_s),
        "poll_interval_s": float(poll_interval_s),
        "yolo_model_sha256": _digest(resolved_yolo_sha, "yolo_model_sha256"),
    }


def _read_shard_journal(path: Path, collection_id: str) -> list[dict[str, object]]:
    source = _require_regular_file(path, "shards journal")
    entries: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise CollectionSpoolError(
                    f"blank line in shards journal at line {line_number}"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CollectionSpoolError(
                    f"invalid shards journal JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise CollectionSpoolError("shards journal entries must be objects")
            entry = _validate_entry(value, collection_id)
            if entry["shard_ordinal"] != len(entries) + 1:
                raise CollectionSpoolError(
                    "shards journal ordinals must be contiguous from one"
                )
            entries.append(entry)
    episodes = [
        episode
        for entry in entries
        for episode in entry["episode_ids"]  # type: ignore[union-attr]
    ]
    if len(episodes) != len(set(episodes)):
        raise CollectionSpoolError("an episode occurs in multiple sealed shards")
    return entries


def _validate_entry(value: dict[str, object], collection_id: str) -> dict[str, object]:
    required = {
        "shard_ordinal",
        "filename",
        "archive_sha256",
        "archive_size_bytes",
        "dataset_sha256",
        "episode_ids",
        "episode_count",
        "physical_capture_count",
        "record_count",
    }
    missing = required - set(value)
    if missing:
        raise CollectionSpoolError(f"shard entry is missing fields: {sorted(missing)}")
    for field in (
        "shard_ordinal",
        "archive_size_bytes",
        "episode_count",
        "physical_capture_count",
        "record_count",
    ):
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise CollectionSpoolError(f"shard entry {field} must be positive")
    ordinal = int(value["shard_ordinal"])
    filename = value["filename"]
    if filename != f"shard_{collection_id}_{ordinal:06d}.tar":
        raise CollectionSpoolError("shard entry filename does not match identity")
    episodes = value["episode_ids"]
    if not isinstance(episodes, list) or not episodes:
        raise CollectionSpoolError("shard entry episode_ids must be a non-empty list")
    normalized_episodes = [
        _safe_identifier(episode, "episode_id") for episode in episodes
    ]
    if len(normalized_episodes) != len(set(normalized_episodes)):
        raise CollectionSpoolError("shard entry episode_ids must be unique")
    if len(normalized_episodes) != value["episode_count"]:
        raise CollectionSpoolError("shard entry episode_count mismatch")
    normalized = dict(value)
    normalized["episode_ids"] = normalized_episodes
    normalized["archive_sha256"] = _digest(
        value["archive_sha256"], "archive_sha256"
    )
    normalized["dataset_sha256"] = _digest(
        value["dataset_sha256"], "dataset_sha256"
    )
    return normalized


def _validate_pending(value: object, collection_id: str) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CollectionSpoolError("session pending_shard must be an object or null")
    if value.get("phase") not in {"building_tar", "ready_to_seal"}:
        raise CollectionSpoolError("session pending_shard phase is invalid")
    if not isinstance(value.get("entry"), dict):
        raise CollectionSpoolError("session pending_shard entry is invalid")
    if value["phase"] == "ready_to_seal":
        value = dict(value)
        value["entry"] = _validate_entry(value["entry"], collection_id)
    for field in ("workspace_path", "tar_tmp_path"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise CollectionSpoolError(f"session pending_shard {field} is invalid")
    return dict(value)


def _write_deterministic_tar(workspace: Path, target: Path) -> None:
    if os.path.lexists(target):
        raise CollectionSpoolError(f"refusing to overwrite collection tar: {target}")
    workspace = _require_real_directory(workspace, "shard workspace")
    paths = sorted(workspace.rglob("*"), key=lambda item: item.relative_to(workspace).as_posix())
    required = {
        "frames.jsonl",
        "dataset_manifest.json",
        "collection_manifest.json",
    }
    actual_files: set[str] = set()
    for path in paths:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise CollectionSpoolError(f"workspace symlink is forbidden: {path}")
        if stat.S_ISREG(info.st_mode):
            actual_files.add(path.relative_to(workspace).as_posix())
        elif not stat.S_ISDIR(info.st_mode):
            raise CollectionSpoolError(f"workspace special file is forbidden: {path}")
    if not required.issubset(actual_files):
        raise CollectionSpoolError(
            f"workspace is missing required files: {sorted(required - actual_files)}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o640)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as raw:
            with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in paths:
                    relative = path.relative_to(workspace).as_posix()
                    archive.add(
                        path,
                        arcname=relative,
                        recursive=False,
                        filter=_normalize_tar_info,
                    )
            raw.flush()
            os.fsync(raw.fileno())
    except Exception:
        os.close(descriptor)
        raise
    else:
        os.close(descriptor)
    _fsync_directory(target.parent)


def _normalize_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.pax_headers = {}
    if info.isdir():
        info.mode = 0o755
    else:
        info.mode = 0o444
    return info


def _directory_size(path: Path | None) -> int:
    if path is None:
        return 0
    total = 0
    for candidate in path.rglob("*"):
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise CollectionSpoolError(f"workspace symlink is forbidden: {candidate}")
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
        elif not stat.S_ISDIR(info.st_mode):
            raise CollectionSpoolError(f"workspace special file is forbidden: {candidate}")
    return total


def _hash_regular_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = sha256()
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise CollectionSpoolError(f"archive must be a non-empty regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest(), info.st_size
    finally:
        os.close(descriptor)


def _quarantine_incomplete_tar(path: Path) -> Path:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CollectionSpoolError(
            f"incomplete tar is not a regular file: {path}"
        )
    retained = path.parent / f".{path.name}.incomplete.{uuid.uuid4().hex}"
    os.rename(path, retained)
    _fsync_directory(path.parent)
    return retained


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o640)
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _atomic_write_jsonl(
    path: Path, entries: Sequence[Mapping[str, object]]
) -> None:
    encoded = b"".join(
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for payload in entries
    )
    temporary = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o640)
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _initialize_empty_file(path: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o640)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _acquire_session_lock(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o640)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CollectionSpoolError("session lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CollectionSpoolError(
                f"collection session is already owned by another process: {path.parent}"
            ) from exc
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _read_json_object(path: Path, field: str) -> Mapping[str, object]:
    source = _require_regular_file(path, field)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionSpoolError(f"cannot read {field}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CollectionSpoolError(f"{field} must contain a JSON object")
    return value


def _require_pending_path(value: object, parent: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CollectionSpoolError(f"{field} path is invalid")
    candidate = Path(value).expanduser().resolve(strict=False)
    resolved_parent = parent.resolve(strict=False)
    if candidate.parent != resolved_parent:
        raise CollectionSpoolError(f"{field} is outside its owned directory")
    return candidate


def _remove_owned_workspace(path: Path, parent: Path) -> None:
    resolved_parent = parent.resolve(strict=True)
    if path.parent.resolve(strict=True) != resolved_parent:
        raise CollectionSpoolError("workspace deletion escaped the session")
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise CollectionSpoolError("workspace deletion target is not a real directory")
    shutil.rmtree(path)
    _fsync_directory(parent)


def _ensure_real_directory(path: str | Path) -> Path:
    return _walk_real_directory(path, create=True, field="directory")


def _require_real_directory(path: str | Path, field: str) -> Path:
    return _walk_real_directory(path, create=False, field=field)


def _require_regular_file(path: str | Path, field: str) -> Path:
    value = Path(os.path.abspath(Path(path).expanduser()))
    parent = _require_real_directory(value.parent, f"{field} parent")
    value = parent / value.name
    try:
        info = value.lstat()
    except OSError as exc:
        raise CollectionSpoolError(f"cannot inspect {field}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CollectionSpoolError(f"{field} must be a non-symlink regular file")
    return value


def _walk_real_directory(
    path: str | Path, *, create: bool, field: str
) -> Path:
    value = Path(os.path.abspath(Path(path).expanduser()))
    current = Path(value.anchor)
    for part in value.parts[1:]:
        current = current / part
        if os.path.lexists(current):
            try:
                info = current.lstat()
            except OSError as exc:
                raise CollectionSpoolError(
                    f"cannot inspect {field} component {current}: {exc}"
                ) from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise CollectionSpoolError(
                    f"{field} contains a non-directory or symlink component: {current}"
                )
        elif create:
            try:
                current.mkdir(mode=0o750)
            except OSError as exc:
                raise CollectionSpoolError(
                    f"cannot create {field} component {current}: {exc}"
                ) from exc
        else:
            raise CollectionSpoolError(f"{field} does not exist: {current}")
    # Re-walk after creation so a concurrently substituted ancestor symlink is
    # not silently accepted by resolve().
    current = Path(value.anchor)
    for part in value.parts[1:]:
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise CollectionSpoolError(
                f"{field} contains an unsafe component: {current}"
            )
    return value


def _safe_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise CollectionSpoolError(f"{field} must be a bounded identifier")
    if not value[0].isalpha() or any(
        character not in _SAFE_IDENTIFIER_CHARACTERS for character in value
    ):
        raise CollectionSpoolError(f"{field} contains unsafe characters")
    return value


def _digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise CollectionSpoolError(f"{field} must be a SHA256 digest")
    return value.lower()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "COLLECTION_INDEX_PROTOCOL",
    "COLLECTION_SCHEMA_VERSION",
    "COLLECTION_SHARD_PROTOCOL",
    "CollectionSpoolError",
    "CollectionSpoolResult",
    "TargetStateCollectionSpool",
]
