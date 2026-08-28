"""Finalize opaque collection archives into one Target State parent dataset.

This module is deliberately CPU-only.  It knows the Target State collection
format, while ``pc_trans`` continues to treat every archive as an opaque file.
Collection archives are immutable inputs and are never removed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
from types import MappingProxyType
from typing import BinaryIO, Mapping, Sequence

from datasets.target_state.dataset import (
    DatasetCheckReport,
    build_manifest,
    check_dataset,
    compute_dataset_sha256,
    read_frame_records,
)
from datasets.target_state.schema import TargetStateFrameRecord
from datasets.target_state.sequence import build_sequences
from training.target_state.shard_runtime import (
    ShardMaterializationError,
    _extract_members,
    _fsync_directory,
    _materialized_files,
    _open_archive_nofollow,
    _safe_member_name,
    _sha256_stream,
)


# Kept equal to the producer constants in ``collection_spool``.  They live
# here as literals as well so importing/finalizing collection data never
# imports an Isaac-facing collection entrypoint.
COLLECTION_SCHEMA_VERSION = 1
COLLECTION_SHARD_PROTOCOL = "target_state_collection_shard_v1"
COLLECTION_INDEX_PROTOCOL = "target_state_collection_index_v1"
_SHA256_LENGTH = 64
_MAX_INDEX_BYTES = 32 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 1_000_000
_ASSET_ROOTS = frozenset({"rgb", "depth", "instance_masks"})
_PROVENANCE_FIELDS = (
    "generation_commit_sha",
    "detector_prediction_source",
    "candidate_id_source",
    "detector_truth_association",
    "oracle_usage",
    "detector_deployment",
)


class CollectionFinalizationError(RuntimeError):
    """Raised when a collection index, archive, or merged dataset is invalid."""


@dataclass(frozen=True, slots=True)
class CollectionArchiveEntry:
    shard_ordinal: int
    filename: str
    archive_sha256: str
    archive_size_bytes: int
    dataset_sha256: str
    episode_ids: tuple[str, ...]
    episode_count: int
    physical_capture_count: int
    record_count: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CollectionArchiveEntry":
        required = {
            "shard_ordinal",
            "archive_sha256",
            "archive_size_bytes",
            "dataset_sha256",
            "episode_ids",
            "episode_count",
            "physical_capture_count",
            "record_count",
        }
        missing = required - set(payload)
        if missing:
            raise CollectionFinalizationError(
                f"collection index shard is missing fields: {sorted(missing)}"
            )
        filename_value = payload.get("filename")
        archive_name_value = payload.get("archive_name")
        if filename_value is None and archive_name_value is None:
            raise CollectionFinalizationError(
                "collection index shard is missing filename"
            )
        if (
            filename_value is not None
            and archive_name_value is not None
            and filename_value != archive_name_value
        ):
            raise CollectionFinalizationError(
                "collection index shard filename/archive_name disagree"
            )
        filename = _string(
            filename_value if filename_value is not None else archive_name_value,
            "shards[].filename",
        )
        _safe_archive_filename(filename)
        episodes_value = payload["episode_ids"]
        if isinstance(episodes_value, (str, bytes)) or not isinstance(
            episodes_value, Sequence
        ):
            raise CollectionFinalizationError("shards[].episode_ids must be an array")
        episodes = tuple(
            _safe_identifier(_string(value, "episode_id"), "episode_id")
            for value in episodes_value
        )
        if not episodes or len(episodes) != len(set(episodes)):
            raise CollectionFinalizationError(
                "shards[].episode_ids must be non-empty and unique"
            )
        entry = cls(
            shard_ordinal=_positive_int(payload["shard_ordinal"], "shard_ordinal"),
            filename=filename,
            archive_sha256=_digest(payload["archive_sha256"], "archive_sha256"),
            archive_size_bytes=_positive_int(
                payload["archive_size_bytes"], "archive_size_bytes"
            ),
            dataset_sha256=_digest(payload["dataset_sha256"], "dataset_sha256"),
            episode_ids=episodes,
            episode_count=_positive_int(payload["episode_count"], "episode_count"),
            physical_capture_count=_positive_int(
                payload["physical_capture_count"], "physical_capture_count"
            ),
            record_count=_positive_int(payload["record_count"], "record_count"),
        )
        if entry.episode_count != len(entry.episode_ids):
            raise CollectionFinalizationError(
                "collection index shard episode_count does not match episode_ids"
            )
        return entry


@dataclass(frozen=True, slots=True)
class TargetStateCollectionIndex:
    collection_id: str
    scene_seed: int
    split_seed: int
    history_size: int
    max_history_age_s: float
    yolo_model_sha256: str
    status: str
    shards: tuple[CollectionArchiveEntry, ...]
    shard_count: int
    episode_count: int
    physical_capture_count: int
    record_count: int
    source_path: Path
    index_sha256: str


@dataclass(frozen=True, slots=True)
class FinalizedTargetStateCollection:
    dataset_root: Path
    collection_id: str
    collection_index_sha256: str
    dataset_sha256: str
    shard_count: int
    episode_count: int
    physical_capture_count: int
    record_count: int
    dataset_report: DatasetCheckReport

    def to_dict(self) -> dict[str, object]:
        return {
            "collection_id": self.collection_id,
            "collection_index_sha256": self.collection_index_sha256,
            "dataset_root": str(self.dataset_root),
            "dataset_sha256": self.dataset_sha256,
            "shard_count": self.shard_count,
            "episode_count": self.episode_count,
            "physical_capture_count": self.physical_capture_count,
            "record_count": self.record_count,
            "sequence_count": self.dataset_report.sequence_count,
            "archives_preserved": True,
            "check_dataset_ok": self.dataset_report.ok,
        }


def load_collection_index(path: str | Path) -> TargetStateCollectionIndex:
    """Read and strictly validate a completed ``collection_index.json``."""

    source_input = Path(path).expanduser()
    descriptor = _open_regular_nofollow(source_input, "collection index")
    try:
        info = os.fstat(descriptor)
        if info.st_size > _MAX_INDEX_BYTES:
            raise CollectionFinalizationError("collection index is unreasonably large")
        raw = _read_all(descriptor, limit=_MAX_INDEX_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > _MAX_INDEX_BYTES:
        raise CollectionFinalizationError("collection index is unreasonably large")
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionFinalizationError(f"invalid collection index JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CollectionFinalizationError("collection index must contain a JSON object")
    required = {
        "schema_version",
        "protocol",
        "collection_id",
        "scene_seed",
        "split_seed",
        "history_size",
        "max_history_age_s",
        "yolo_model_sha256",
        "status",
        "shard_count",
        "episode_count",
        "physical_capture_count",
        "record_count",
        "shards",
    }
    missing = required - set(payload)
    if missing:
        raise CollectionFinalizationError(
            f"collection index is missing fields: {sorted(missing)}"
        )
    if _integer(payload["schema_version"], "schema_version") != COLLECTION_SCHEMA_VERSION:
        raise CollectionFinalizationError("unsupported collection index schema_version")
    if payload["protocol"] != COLLECTION_INDEX_PROTOCOL:
        raise CollectionFinalizationError("unsupported collection index protocol")
    status = _string(payload["status"], "status")
    if status != "completed":
        raise CollectionFinalizationError(
            "collection index status must be 'completed' before finalization"
        )
    collection_id = _safe_identifier(
        _string(payload["collection_id"], "collection_id"), "collection_id"
    )
    history_size = _integer(payload["history_size"], "history_size")
    if not 4 <= history_size <= 8:
        raise CollectionFinalizationError("history_size must be within [4, 8]")
    max_history_age_s = _positive_finite_number(
        payload["max_history_age_s"], "max_history_age_s"
    )
    entries_value = payload["shards"]
    if not isinstance(entries_value, list):
        raise CollectionFinalizationError("collection index shards must be an array")
    for value in entries_value:
        if (
            isinstance(value, Mapping)
            and "collection_id" in value
            and value["collection_id"] != collection_id
        ):
            raise CollectionFinalizationError(
                "collection index shard collection_id does not match index"
            )
    entries = tuple(
        CollectionArchiveEntry.from_mapping(value)
        if isinstance(value, Mapping)
        else _raise_invalid_shard_entry()
        for value in entries_value
    )
    if not entries:
        raise CollectionFinalizationError("collection index must contain at least one shard")
    shard_count = _positive_int(payload["shard_count"], "shard_count")
    episode_count = _positive_int(payload["episode_count"], "episode_count")
    physical_capture_count = _positive_int(
        payload["physical_capture_count"], "physical_capture_count"
    )
    record_count = _positive_int(payload["record_count"], "record_count")
    if shard_count != len(entries):
        raise CollectionFinalizationError("shard_count does not match shards array")
    ordered = tuple(sorted(entries, key=lambda item: item.shard_ordinal))
    if tuple(item.shard_ordinal for item in ordered) != tuple(
        range(1, len(ordered) + 1)
    ):
        raise CollectionFinalizationError(
            "collection shard ordinals must be unique and contiguous from 1"
        )
    all_episodes = tuple(episode for item in ordered for episode in item.episode_ids)
    if len(all_episodes) != len(set(all_episodes)):
        raise CollectionFinalizationError(
            "an episode appears in more than one collection shard"
        )
    expected_totals = (
        sum(item.episode_count for item in ordered),
        sum(item.physical_capture_count for item in ordered),
        sum(item.record_count for item in ordered),
    )
    if expected_totals != (episode_count, physical_capture_count, record_count):
        raise CollectionFinalizationError(
            "collection index aggregate counts do not match shard entries"
        )
    if len(all_episodes) != episode_count:
        raise CollectionFinalizationError(
            "collection index episode_count does not match unique episode IDs"
        )
    for entry in ordered:
        expected_filename = (
            f"shard_{collection_id}_{entry.shard_ordinal:06d}.tar"
        )
        if entry.filename != expected_filename:
            raise CollectionFinalizationError(
                f"collection shard filename/identity mismatch: {entry.filename!r}"
            )
    source = Path(os.path.abspath(source_input))
    return TargetStateCollectionIndex(
        collection_id=collection_id,
        scene_seed=_integer(payload["scene_seed"], "scene_seed"),
        split_seed=_integer(payload["split_seed"], "split_seed"),
        history_size=history_size,
        max_history_age_s=max_history_age_s,
        yolo_model_sha256=_digest(
            payload["yolo_model_sha256"], "yolo_model_sha256"
        ),
        status=status,
        shards=ordered,
        shard_count=shard_count,
        episode_count=episode_count,
        physical_capture_count=physical_capture_count,
        record_count=record_count,
        source_path=source,
        index_sha256=sha256(raw).hexdigest(),
    )


def finalize_target_state_collection(
    collection_index: TargetStateCollectionIndex | str | Path,
    shard_dir: str | Path,
    output_dir: str | Path,
) -> FinalizedTargetStateCollection:
    """Validate all archives and atomically publish one complete parent dataset.

    The archive files are read-only inputs.  They remain untouched on success
    and on every failure path.
    """

    index = (
        collection_index
        if isinstance(collection_index, TargetStateCollectionIndex)
        else load_collection_index(collection_index)
    )
    archives_root = _real_directory(shard_dir, "collection shard directory")
    destination, parent = _prepare_destination(output_dir, archives_root)
    temporary = parent / (
        f".{destination.name}.tmp.{os.getpid()}.{os.urandom(6).hex()}"
    )
    temporary.mkdir(mode=0o700)
    extraction_root = parent / (
        f".{destination.name}.extract.{os.getpid()}.{os.urandom(6).hex()}"
    )
    try:
        extraction_root.mkdir(mode=0o700)
    except OSError as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise CollectionFinalizationError(
            f"cannot create collection extraction workspace: {exc}"
        ) from exc
    all_records: list[TargetStateFrameRecord] = []
    frame_ids: set[str] = set()
    episode_ids: set[str] = set()
    asset_paths: set[str] = set()
    provenance: Mapping[str, object] | None = None
    archive_digests: list[str] = []
    try:
        for entry in index.shards:
            extracted = extraction_root / f"shard_{entry.shard_ordinal:06d}"
            extracted.mkdir(mode=0o700)
            archive = archives_root / entry.filename
            _extract_collection_archive(archive, entry, extracted)
            records, shard_provenance = _validate_extracted_collection_shard(
                extracted, index, entry
            )
            actual_episodes = {record.episode_id for record in records}
            duplicate_episodes = sorted(episode_ids & actual_episodes)
            if duplicate_episodes:
                raise CollectionFinalizationError(
                    "episode appears in multiple collection shards: "
                    + ", ".join(duplicate_episodes[:5])
                )
            duplicate_frames = sorted(
                record.frame_id for record in records if record.frame_id in frame_ids
            )
            if duplicate_frames:
                raise CollectionFinalizationError(
                    "duplicate frame_id across collection shards: "
                    + ", ".join(duplicate_frames[:5])
                )
            assets = _record_asset_paths(records)
            conflicting_assets = sorted(asset_paths & assets)
            if conflicting_assets:
                raise CollectionFinalizationError(
                    "asset path appears in multiple collection shards: "
                    + ", ".join(conflicting_assets[:5])
                )
            if provenance is None:
                provenance = shard_provenance
            elif dict(provenance) != dict(shard_provenance):
                raise CollectionFinalizationError(
                    "collection shards have inconsistent dataset provenance"
                )
            _copy_assets_nofollow(extracted, temporary, sorted(assets))
            all_records.extend(records)
            frame_ids.update(record.frame_id for record in records)
            episode_ids.update(actual_episodes)
            asset_paths.update(assets)
            archive_digests.append(entry.archive_sha256)
            shutil.rmtree(extracted)

        if len(all_records) != index.record_count:
            raise CollectionFinalizationError(
                "merged record count does not match collection index"
            )
        physical_captures = {
            (record.episode_id, record.sensor_input.rgb_path) for record in all_records
        }
        if len(physical_captures) != index.physical_capture_count:
            raise CollectionFinalizationError(
                "merged physical capture count does not match collection index"
            )
        if len(episode_ids) != index.episode_count:
            raise CollectionFinalizationError(
                "merged episode count does not match collection index"
            )
        _write_frames(temporary / "frames.jsonl", all_records)
        sequences = build_sequences(
            all_records,
            history_size=index.history_size,
            max_history_age_s=index.max_history_age_s,
        )
        preliminary = check_dataset(
            temporary,
            sequences=sequences,
            history_size=index.history_size,
            max_history_age_s=index.max_history_age_s,
            split_seed=index.split_seed,
        )
        if not preliminary.ok:
            raise CollectionFinalizationError(
                "merged collection failed pre-manifest check_dataset: "
                + "; ".join(preliminary.errors[:5])
            )
        dataset_sha = compute_dataset_sha256(temporary, all_records)
        if provenance is None:  # pragma: no cover - non-empty index invariant
            raise CollectionFinalizationError("collection has no provenance")
        manifest = build_manifest(
            all_records,
            sequences,
            dataset_sha256=dataset_sha,
            split_seed=index.split_seed,
            generation_commit_sha=_string(
                provenance["generation_commit_sha"], "generation_commit_sha"
            ),
        )
        manifest.update(dict(provenance))
        manifest.update(
            {
                "yolo_model_sha256": index.yolo_model_sha256,
                "history_size": index.history_size,
                "max_history_age_s": index.max_history_age_s,
                "source_collection_id": index.collection_id,
                "source_collection_index_sha256": index.index_sha256,
                "source_collection_shard_count": index.shard_count,
                "source_collection_archive_sha256": archive_digests,
                "source_collection_scene_seed": index.scene_seed,
                "split_seed": index.split_seed,
            }
        )
        _write_json(temporary / "dataset_manifest.json", manifest)
        final_report = check_dataset(
            temporary,
            sequences=sequences,
            history_size=index.history_size,
            max_history_age_s=index.max_history_age_s,
            split_seed=index.split_seed,
        )
        if not final_report.ok or final_report.dataset_sha256 != dataset_sha:
            raise CollectionFinalizationError(
                "finalized collection failed check_dataset: "
                + "; ".join(final_report.errors[:5])
            )
        _fsync_all_files_and_directories(temporary)
        if destination.exists() or destination.is_symlink():
            raise CollectionFinalizationError(
                f"output directory appeared during finalization: {destination}"
            )
        os.rename(temporary, destination)
        _fsync_directory(parent)
        return FinalizedTargetStateCollection(
            dataset_root=destination,
            collection_id=index.collection_id,
            collection_index_sha256=index.index_sha256,
            dataset_sha256=dataset_sha,
            shard_count=index.shard_count,
            episode_count=index.episode_count,
            physical_capture_count=index.physical_capture_count,
            record_count=index.record_count,
            dataset_report=final_report,
        )
    except CollectionFinalizationError:
        raise
    except (OSError, ValueError, TypeError, tarfile.TarError, ShardMaterializationError) as exc:
        raise CollectionFinalizationError(f"cannot finalize collection: {exc}") from exc
    finally:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary, ignore_errors=True)
        if extraction_root.exists() and not extraction_root.is_symlink():
            shutil.rmtree(extraction_root, ignore_errors=True)


def _extract_collection_archive(
    archive: Path,
    entry: CollectionArchiveEntry,
    destination: Path,
) -> None:
    try:
        stream = _open_archive_nofollow(archive)
    except ShardMaterializationError as exc:
        raise CollectionFinalizationError(str(exc)) from exc
    try:
        size = os.fstat(stream.fileno()).st_size
        if size != entry.archive_size_bytes:
            raise CollectionFinalizationError(
                f"archive size mismatch for {entry.filename}"
            )
        if _sha256_stream(stream) != entry.archive_sha256:
            raise CollectionFinalizationError(
                f"archive SHA256 mismatch for {entry.filename}"
            )
        stream.seek(0)
        with tarfile.open(fileobj=stream, mode="r:") as archive_file:
            members = archive_file.getmembers()
            _validate_collection_members(members, archive_size=size)
            _extract_members(archive_file, members, destination)
    except CollectionFinalizationError:
        raise
    except (OSError, tarfile.TarError, ShardMaterializationError) as exc:
        raise CollectionFinalizationError(
            f"cannot safely extract {entry.filename}: {exc}"
        ) from exc
    finally:
        stream.close()


def _validate_collection_members(
    members: list[tarfile.TarInfo], *, archive_size: int
) -> None:
    if not members:
        raise CollectionFinalizationError("collection archive is empty")
    if len(members) > _MAX_ARCHIVE_MEMBERS:
        raise CollectionFinalizationError("collection archive has too many members")
    names: set[str] = set()
    regular_names: set[str] = set()
    total_size = 0
    for member in members:
        try:
            name = _safe_member_name(member)
        except ShardMaterializationError as exc:
            raise CollectionFinalizationError(str(exc)) from exc
        if name in names:
            raise CollectionFinalizationError(f"duplicate tar member: {name}")
        names.add(name)
        if member.isdir():
            if PurePosixPath(name).parts[0] not in _ASSET_ROOTS:
                raise CollectionFinalizationError(
                    f"unexpected collection archive directory: {name}"
                )
            continue
        if not member.isreg():
            raise CollectionFinalizationError(
                f"tar member type is forbidden: {member.name}"
            )
        if getattr(member, "sparse", None):
            raise CollectionFinalizationError("sparse tar members are forbidden")
        if member.size < 0:
            raise CollectionFinalizationError("tar member has negative size")
        total_size += member.size
        regular_names.add(name)
    if total_size > archive_size:
        raise CollectionFinalizationError(
            "tar member sizes exceed the uncompressed archive size"
        )
    for name in names:
        parts = PurePosixPath(name).parts
        for index in range(1, len(parts)):
            if PurePosixPath(*parts[:index]).as_posix() in regular_names:
                raise CollectionFinalizationError(
                    f"tar member descends from a regular file: {name}"
                )
    required = {
        "frames.jsonl",
        "dataset_manifest.json",
        "collection_manifest.json",
    }
    if not required.issubset(regular_names):
        raise CollectionFinalizationError(
            f"collection archive is missing required files: {sorted(required - regular_names)}"
        )
    for name in regular_names - required:
        if PurePosixPath(name).parts[0] not in _ASSET_ROOTS:
            raise CollectionFinalizationError(
                f"unexpected collection archive file: {name}"
            )


def _validate_extracted_collection_shard(
    root: Path,
    index: TargetStateCollectionIndex,
    entry: CollectionArchiveEntry,
) -> tuple[tuple[TargetStateFrameRecord, ...], Mapping[str, object]]:
    manifest = _read_json_object(root / "collection_manifest.json")
    dataset_manifest = _read_json_object(root / "dataset_manifest.json")
    _validate_collection_manifest_types(manifest)
    expected = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "protocol": COLLECTION_SHARD_PROTOCOL,
        "collection_id": index.collection_id,
        "shard_ordinal": entry.shard_ordinal,
        "episode_ids": list(entry.episode_ids),
        "episode_count": entry.episode_count,
        "physical_capture_count": entry.physical_capture_count,
        "record_count": entry.record_count,
        "scene_seed": index.scene_seed,
        "split_seed": index.split_seed,
        "history_size": index.history_size,
        "max_history_age_s": index.max_history_age_s,
        "yolo_model_sha256": index.yolo_model_sha256,
        "dataset_sha256": entry.dataset_sha256,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise CollectionFinalizationError(
                f"collection_manifest {field} does not match collection index"
            )
    if dataset_manifest.get("dataset_sha256") != entry.dataset_sha256:
        raise CollectionFinalizationError(
            "dataset_manifest dataset_sha256 does not match collection index"
        )
    if dataset_manifest.get("history_size") != index.history_size:
        raise CollectionFinalizationError("dataset_manifest history_size mismatch")
    if dataset_manifest.get("max_history_age_s") != index.max_history_age_s:
        raise CollectionFinalizationError(
            "dataset_manifest max_history_age_s mismatch"
        )
    if dataset_manifest.get("yolo_model_sha256") != index.yolo_model_sha256:
        raise CollectionFinalizationError("dataset_manifest YOLO model SHA mismatch")
    provenance: dict[str, object] = {}
    for field in _PROVENANCE_FIELDS:
        if field not in dataset_manifest:
            raise CollectionFinalizationError(
                f"dataset_manifest is missing provenance field: {field}"
            )
        value = dataset_manifest[field]
        if field not in manifest or manifest[field] != value:
            raise CollectionFinalizationError(
                f"collection_manifest provenance mismatch: {field}"
            )
        provenance[field] = value
    try:
        records = read_frame_records(root / "frames.jsonl")
    except (OSError, ValueError) as exc:
        raise CollectionFinalizationError(f"invalid shard frames.jsonl: {exc}") from exc
    if len(records) != entry.record_count:
        raise CollectionFinalizationError(
            "collection shard record_count does not match decoded records"
        )
    actual_episodes = tuple(sorted({record.episode_id for record in records}))
    if actual_episodes != tuple(sorted(entry.episode_ids)):
        raise CollectionFinalizationError(
            "collection shard episode IDs do not match collection index"
        )
    captures = {(record.episode_id, record.sensor_input.rgb_path) for record in records}
    if len(captures) != entry.physical_capture_count:
        raise CollectionFinalizationError(
            "collection shard physical_capture_count does not match decoded records"
        )
    assets = _record_asset_paths(records)
    actual_files = _materialized_files(root)
    expected_files = {
        "frames.jsonl",
        "dataset_manifest.json",
        "collection_manifest.json",
        *assets,
    }
    if actual_files != expected_files:
        raise CollectionFinalizationError(
            "collection shard file set mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    actual_sha = compute_dataset_sha256(root, records)
    if actual_sha != entry.dataset_sha256:
        raise CollectionFinalizationError(
            "collection shard dataset SHA256 mismatch"
        )
    report = check_dataset(
        root,
        history_size=index.history_size,
        max_history_age_s=index.max_history_age_s,
        split_seed=index.split_seed,
    )
    if not report.ok or report.dataset_sha256 != actual_sha:
        raise CollectionFinalizationError(
            "collection shard failed check_dataset: " + "; ".join(report.errors[:5])
        )
    return records, MappingProxyType(provenance)


def _validate_collection_manifest_types(manifest: Mapping[str, object]) -> None:
    """Reject JSON bool/numeric coercions before comparing declared identity."""

    required = {
        "schema_version",
        "protocol",
        "collection_id",
        "shard_ordinal",
        "episode_ids",
        "episode_count",
        "physical_capture_count",
        "record_count",
        "scene_seed",
        "split_seed",
        "history_size",
        "max_history_age_s",
        "yolo_model_sha256",
        "dataset_sha256",
        *_PROVENANCE_FIELDS,
    }
    missing = required - set(manifest)
    if missing:
        raise CollectionFinalizationError(
            f"collection_manifest is missing fields: {sorted(missing)}"
        )
    _integer(manifest["schema_version"], "collection_manifest.schema_version")
    _string(manifest["protocol"], "collection_manifest.protocol")
    _safe_identifier(
        _string(manifest["collection_id"], "collection_manifest.collection_id"),
        "collection_manifest.collection_id",
    )
    for field in (
        "shard_ordinal",
        "episode_count",
        "physical_capture_count",
        "record_count",
    ):
        _positive_int(manifest[field], f"collection_manifest.{field}")
    _integer(manifest["scene_seed"], "collection_manifest.scene_seed")
    _integer(manifest["split_seed"], "collection_manifest.split_seed")
    history_size = _integer(
        manifest["history_size"], "collection_manifest.history_size"
    )
    if not 4 <= history_size <= 8:
        raise CollectionFinalizationError(
            "collection_manifest history_size must be within [4, 8]"
        )
    _positive_finite_number(
        manifest["max_history_age_s"], "collection_manifest.max_history_age_s"
    )
    _digest(
        manifest["yolo_model_sha256"],
        "collection_manifest.yolo_model_sha256",
    )
    _digest(manifest["dataset_sha256"], "collection_manifest.dataset_sha256")
    episodes = manifest["episode_ids"]
    if isinstance(episodes, (str, bytes)) or not isinstance(episodes, list):
        raise CollectionFinalizationError(
            "collection_manifest episode_ids must be an array"
        )
    normalized = tuple(
        _safe_identifier(
            _string(value, "collection_manifest.episode_id"),
            "collection_manifest.episode_id",
        )
        for value in episodes
    )
    if not normalized or len(normalized) != len(set(normalized)):
        raise CollectionFinalizationError(
            "collection_manifest episode_ids must be non-empty and unique"
        )


def _record_asset_paths(records: Sequence[TargetStateFrameRecord]) -> set[str]:
    assets: set[str] = set()
    for record in records:
        sensor = record.sensor_input
        values = (sensor.rgb_path, sensor.depth_path, sensor.instance_mask_path)
        for value in values:
            if value is None:
                continue
            relative = _safe_relative_path(value, "record asset path")
            if PurePosixPath(relative).parts[0] not in _ASSET_ROOTS:
                raise CollectionFinalizationError(
                    f"record asset has unsupported root: {relative}"
                )
            assets.add(relative)
    return assets


def _copy_assets_nofollow(source_root: Path, destination_root: Path, assets: Sequence[str]) -> None:
    for relative in assets:
        source_descriptor = _open_relative_regular(source_root, relative)
        destination = destination_root.joinpath(*PurePosixPath(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            destination_descriptor = os.open(destination, flags, 0o444)
        except OSError:
            os.close(source_descriptor)
            raise
        try:
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                _write_all(destination_descriptor, chunk)
            os.fsync(destination_descriptor)
        finally:
            os.close(destination_descriptor)
            os.close(source_descriptor)


def _open_relative_regular(root: Path, relative: str) -> int:
    parts = PurePosixPath(_safe_relative_path(relative, "asset path")).parts
    read_flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    directory_flags = read_flags | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        read_flags |= os.O_NOFOLLOW
        directory_flags |= os.O_NOFOLLOW
    parent_descriptor = os.open(root, directory_flags)
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(part, directory_flags, dir_fd=parent_descriptor)
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor
        descriptor = os.open(parts[-1], read_flags, dir_fd=parent_descriptor)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise CollectionFinalizationError(
                f"asset is not a regular file: {relative}"
            )
        return descriptor
    except CollectionFinalizationError:
        raise
    except OSError as exc:
        raise CollectionFinalizationError(
            f"cannot safely open asset {relative}: {exc}"
        ) from exc
    finally:
        os.close(parent_descriptor)


def _prepare_destination(output_dir: str | Path, archives_root: Path) -> tuple[Path, Path]:
    candidate = Path(output_dir).expanduser()
    if not candidate.name or candidate.name in {".", ".."}:
        raise CollectionFinalizationError("output directory must name a dataset")
    if candidate.exists() or candidate.is_symlink():
        raise CollectionFinalizationError(
            f"output directory must not already exist: {candidate}"
        )
    parent_input = candidate.parent
    parent_input.mkdir(parents=True, exist_ok=True)
    parent = _real_directory(parent_input, "output parent directory")
    destination = parent / candidate.name
    if destination == archives_root or archives_root in destination.parents:
        raise CollectionFinalizationError(
            "output directory must be outside the collection shard directory"
        )
    return destination, parent


def _real_directory(path: str | Path, field: str) -> Path:
    value = Path(path).expanduser()
    try:
        info = value.lstat()
    except OSError as exc:
        raise CollectionFinalizationError(f"cannot stat {field}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise CollectionFinalizationError(f"{field} must be a non-symlink directory")
    return value.resolve(strict=True)


def _read_json_object(path: Path) -> Mapping[str, object]:
    descriptor = _open_regular_nofollow(path, path.name)
    try:
        raw = _read_all(descriptor, limit=_MAX_INDEX_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > _MAX_INDEX_BYTES:
        raise CollectionFinalizationError(f"{path.name} is unreasonably large")
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionFinalizationError(f"invalid {path.name}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CollectionFinalizationError(f"{path.name} must contain a JSON object")
    return payload


def _open_regular_nofollow(path: Path, field: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CollectionFinalizationError(f"cannot safely open {field}: {exc}") from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise CollectionFinalizationError(f"{field} must be a regular file")
    return descriptor


def _read_all(descriptor: int, *, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total < limit:
        chunk = os.read(descriptor, min(1024 * 1024, limit - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _write_frames(path: Path, records: Sequence[TargetStateFrameRecord]) -> None:
    payload = b"".join(
        (
            json.dumps(
                record.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )
    _write_new_file(path, payload)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
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
    _write_new_file(path, encoded)


def _write_new_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:  # pragma: no cover - defensive OS contract
            raise OSError("short write")
        view = view[written:]


def _fsync_all_files_and_directories(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise CollectionFinalizationError(
                f"final dataset contains a symlink: {path.relative_to(root)}"
            )
        if stat.S_ISREG(info.st_mode):
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif stat.S_ISDIR(info.st_mode):
            directories.append(path)
        else:
            raise CollectionFinalizationError(
                f"final dataset contains a special file: {path.relative_to(root)}"
            )
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        _fsync_directory(directory)


def _safe_archive_filename(value: str) -> str:
    _safe_relative_path(value, "archive filename")
    if PurePosixPath(value).name != value or not value.endswith(".tar"):
        raise CollectionFinalizationError(f"invalid archive filename: {value!r}")
    return value


def _safe_relative_path(value: object, field: str) -> str:
    text = _string(value, field)
    if not text or "\\" in text or "\x00" in text:
        raise CollectionFinalizationError(f"{field} must be a portable relative path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != text
        or path.parts[0].endswith(":")
    ):
        raise CollectionFinalizationError(f"unsafe {field}: {text!r}")
    return text


def _safe_identifier(value: str, field: str) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if (
        not value
        or len(value) > 64
        or value != value.strip()
        or value[0] not in alphabet[:52]
        or any(character not in alphabet for character in value)
    ):
        raise CollectionFinalizationError(f"invalid {field}: {value!r}")
    return value


def _digest(value: object, field: str) -> str:
    text = _string(value, field).lower()
    if len(text) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise CollectionFinalizationError(f"{field} must be a SHA256 digest")
    return text


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise CollectionFinalizationError(f"{field} must be a string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CollectionFinalizationError(f"{field} must be an integer")
    return value


def _positive_int(value: object, field: str) -> int:
    result = _integer(value, field)
    if result <= 0:
        raise CollectionFinalizationError(f"{field} must be positive")
    return result


def _positive_finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CollectionFinalizationError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise CollectionFinalizationError(f"{field} must be finite and positive")
    return result


def _raise_invalid_shard_entry() -> CollectionArchiveEntry:
    raise CollectionFinalizationError("collection index shard must be an object")


__all__ = [
    "CollectionArchiveEntry",
    "CollectionFinalizationError",
    "FinalizedTargetStateCollection",
    "TargetStateCollectionIndex",
    "finalize_target_state_collection",
    "load_collection_index",
]
