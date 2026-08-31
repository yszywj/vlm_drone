"""Deterministic, episode-atomic Target State dataset shard builder.

This module is intentionally CPU-only and does not import Isaac Sim.  A shard
is a self-contained target-state dataset tarball; ``pc_trans`` treats it as an
opaque immutable file.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from datasets.target_state.dataset import (
    build_manifest,
    check_dataset,
    compute_dataset_sha256,
    read_frame_records,
    split_for_episode,
)
from datasets.target_state.schema import TargetStateFrameRecord
from datasets.target_state.sequence import build_sequences


SHARD_SCHEMA_VERSION = 1
SHARD_PROTOCOL = "target_state_episode_sharded_v1"
SPLITS = ("train", "validation", "test")
_SHA256_LENGTH = 64
_MANIFEST_ALLOWANCE_BYTES = 64 * 1024
_PROVENANCE_FIELDS = (
    "generation_commit_sha",
    "detector_prediction_source",
    "candidate_id_source",
    "detector_truth_association",
    "yolo_model_sha256",
    "oracle_usage",
    "detector_deployment",
    "camera_convention",
    "coordinate_convention",
)


class ShardFormatError(RuntimeError):
    """Raised when a shard dataset, plan, or index violates the protocol."""


@dataclass(frozen=True, slots=True)
class ShardIndexEntry:
    shard_id: str
    filename: str
    split: str
    episode_ids: tuple[str, ...]
    episode_count: int
    frame_count: int
    sequence_count: int
    shard_dataset_sha256: str
    archive_sha256: str
    archive_size_bytes: int

    def __post_init__(self) -> None:
        _safe_identifier(self.shard_id, "shard_id")
        _safe_shard_filename(self.filename)
        if self.split not in SPLITS:
            raise ShardFormatError(f"invalid shard split: {self.split!r}")
        episodes = tuple(self.episode_ids)
        if not episodes or len(set(episodes)) != len(episodes):
            raise ShardFormatError("episode_ids must be non-empty and unique")
        for episode_id in episodes:
            _safe_identifier(episode_id, "episode_id")
        object.__setattr__(self, "episode_ids", episodes)
        _nonnegative_int(self.episode_count, "episode_count")
        _nonnegative_int(self.frame_count, "frame_count")
        _nonnegative_int(self.sequence_count, "sequence_count")
        _positive_int(self.archive_size_bytes, "archive_size_bytes")
        if self.episode_count != len(episodes):
            raise ShardFormatError("episode_count does not match episode_ids")
        if self.frame_count == 0:
            raise ShardFormatError("a shard must contain at least one frame")
        object.__setattr__(
            self,
            "shard_dataset_sha256",
            _digest(self.shard_dataset_sha256, "shard_dataset_sha256"),
        )
        object.__setattr__(
            self,
            "archive_sha256",
            _digest(self.archive_sha256, "archive_sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "shard_id": self.shard_id,
            "filename": self.filename,
            "split": self.split,
            "episode_ids": list(self.episode_ids),
            "episode_count": self.episode_count,
            "frame_count": self.frame_count,
            "sequence_count": self.sequence_count,
            "shard_dataset_sha256": self.shard_dataset_sha256,
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ShardIndexEntry":
        required = {
            "shard_id",
            "filename",
            "split",
            "episode_ids",
            "episode_count",
            "frame_count",
            "sequence_count",
            "shard_dataset_sha256",
            "archive_sha256",
            "archive_size_bytes",
        }
        missing = required - set(payload)
        if missing:
            raise ShardFormatError(f"shard index entry missing fields: {sorted(missing)}")
        episodes = payload["episode_ids"]
        if isinstance(episodes, (str, bytes)) or not isinstance(episodes, Sequence):
            raise ShardFormatError("episode_ids must be an array")
        return cls(
            shard_id=_string(payload["shard_id"], "shard_id"),
            filename=_string(payload["filename"], "filename"),
            split=_string(payload["split"], "split"),
            episode_ids=tuple(_string(item, "episode_id") for item in episodes),
            episode_count=_integer(payload["episode_count"], "episode_count"),
            frame_count=_integer(payload["frame_count"], "frame_count"),
            sequence_count=_integer(payload["sequence_count"], "sequence_count"),
            shard_dataset_sha256=_string(
                payload["shard_dataset_sha256"], "shard_dataset_sha256"
            ),
            archive_sha256=_string(payload["archive_sha256"], "archive_sha256"),
            archive_size_bytes=_integer(
                payload["archive_size_bytes"], "archive_size_bytes"
            ),
        )


@dataclass(frozen=True, slots=True)
class TargetStateShardIndex:
    parent_dataset_sha256: str
    split_seed: int
    history_size: int
    max_history_age_s: float
    target_shard_size_bytes: int
    parent_dataset_provenance: Mapping[str, object]
    shards: tuple[ShardIndexEntry, ...]
    index_sha256: str
    source_path: Path
    episode_count: int
    frame_count: int
    sequence_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_dataset_sha256",
            _digest(self.parent_dataset_sha256, "parent_dataset_sha256"),
        )
        object.__setattr__(
            self,
            "index_sha256",
            _digest(self.index_sha256, "index_sha256"),
        )
        if isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int):
            raise ShardFormatError("split_seed must be an integer")
        if not 4 <= self.history_size <= 8:
            raise ShardFormatError("history_size must be within [4, 8]")
        if (
            isinstance(self.max_history_age_s, bool)
            or not isinstance(self.max_history_age_s, (int, float))
            or not isfinite(self.max_history_age_s)
            or self.max_history_age_s <= 0.0
        ):
            raise ShardFormatError("max_history_age_s must be finite and positive")
        _positive_int(self.target_shard_size_bytes, "target_shard_size_bytes")
        if not isinstance(self.parent_dataset_provenance, Mapping):
            raise ShardFormatError("parent_dataset_provenance must be an object")
        entries = tuple(self.shards)
        if not entries:
            raise ShardFormatError("shard index must contain at least one shard")
        if any(not isinstance(item, ShardIndexEntry) for item in entries):
            raise ShardFormatError("shards must contain ShardIndexEntry objects")
        if len({item.shard_id for item in entries}) != len(entries):
            raise ShardFormatError("duplicate shard_id in shard index")
        if len({item.filename for item in entries}) != len(entries):
            raise ShardFormatError("duplicate shard filename in shard index")
        episode_ids = [episode for item in entries for episode in item.episode_ids]
        if len(set(episode_ids)) != len(episode_ids):
            raise ShardFormatError("an episode appears in more than one shard")
        for entry in entries:
            if any(
                split_for_episode(episode, seed=self.split_seed) != entry.split
                for episode in entry.episode_ids
            ):
                raise ShardFormatError(
                    f"shard {entry.shard_id} contains an episode from another split"
                )
        object.__setattr__(self, "shards", entries)
        object.__setattr__(
            self,
            "parent_dataset_provenance",
            MappingProxyType(dict(self.parent_dataset_provenance)),
        )
        _nonnegative_int(self.episode_count, "episode_count")
        _nonnegative_int(self.frame_count, "frame_count")
        _nonnegative_int(self.sequence_count, "sequence_count")
        if self.episode_count != sum(item.episode_count for item in entries):
            raise ShardFormatError("index episode_count does not match shards")
        if self.frame_count != sum(item.frame_count for item in entries):
            raise ShardFormatError("index frame_count does not match shards")
        if self.sequence_count != sum(item.sequence_count for item in entries):
            raise ShardFormatError("index sequence_count does not match shards")

    def entry_for_filename(self, filename: str) -> ShardIndexEntry:
        _safe_shard_filename(filename)
        matches = [item for item in self.shards if item.filename == filename]
        if len(matches) != 1:
            raise ShardFormatError(f"archive is not declared by shard index: {filename}")
        return matches[0]

    def shards_for_split(self, split: str) -> tuple[ShardIndexEntry, ...]:
        if split not in SPLITS:
            raise ShardFormatError(f"invalid split: {split!r}")
        return tuple(item for item in self.shards if item.split == split)

    def to_dict(self) -> dict[str, object]:
        counts = {split: len(self.shards_for_split(split)) for split in SPLITS}
        return {
            "schema_version": SHARD_SCHEMA_VERSION,
            "protocol": SHARD_PROTOCOL,
            "parent_dataset_sha256": self.parent_dataset_sha256,
            "split_seed": self.split_seed,
            "history_size": self.history_size,
            "max_history_age_s": self.max_history_age_s,
            "target_shard_size_bytes": self.target_shard_size_bytes,
            "parent_dataset_provenance": dict(self.parent_dataset_provenance),
            "episode_count": self.episode_count,
            "frame_count": self.frame_count,
            "sequence_count": self.sequence_count,
            "split_shard_counts": counts,
            "shards": [item.to_dict() for item in self.shards],
        }


@dataclass(frozen=True, slots=True)
class ShardPlan:
    shard_id: str
    filename: str
    split: str
    episode_ids: tuple[str, ...]
    records: tuple[TargetStateFrameRecord, ...]
    asset_paths: tuple[str, ...]
    estimated_size_bytes: int


@dataclass(frozen=True, slots=True)
class ShardBuildResult:
    shard_index: TargetStateShardIndex
    output_dir: Path
    total_tar_bytes: int

    def to_dict(self) -> dict[str, object]:
        index = self.shard_index
        return {
            "parent_dataset_sha256": index.parent_dataset_sha256,
            "train_shard_count": len(index.shards_for_split("train")),
            "validation_shard_count": len(index.shards_for_split("validation")),
            "test_shard_count": len(index.shards_for_split("test")),
            "episode_count": index.episode_count,
            "frame_count": index.frame_count,
            "sequence_count": index.sequence_count,
            "total_tar_bytes": self.total_tar_bytes,
            "shard_index_path": str(index.source_path),
            "shard_index_sha256": index.index_sha256,
        }


@dataclass(frozen=True, slots=True)
class _ParentDataset:
    root: Path
    records: tuple[TargetStateFrameRecord, ...]
    manifest: Mapping[str, object]
    dataset_sha256: str


@dataclass(frozen=True, slots=True)
class _Episode:
    episode_id: str
    split: str
    records: tuple[TargetStateFrameRecord, ...]
    asset_paths: tuple[str, ...]
    payload_size_bytes: int


def load_shard_index(path: str | Path) -> TargetStateShardIndex:
    """Load and strictly validate a ``shard_index.json`` control file."""

    source_input = Path(path).expanduser()
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source_input, flags)
    except OSError as exc:
        raise ShardFormatError(f"cannot safely open shard index: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ShardFormatError(
                f"shard index must be a regular file: {source_input}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read()
    finally:
        os.close(descriptor)
    source = Path(os.path.abspath(source_input))
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ShardFormatError(f"invalid shard index JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ShardFormatError("shard index must contain a JSON object")
    if payload.get("schema_version") != SHARD_SCHEMA_VERSION:
        raise ShardFormatError("unsupported shard index schema_version")
    if payload.get("protocol") != SHARD_PROTOCOL:
        raise ShardFormatError("unsupported shard index protocol")
    raw_entries = payload.get("shards")
    if not isinstance(raw_entries, list):
        raise ShardFormatError("shard index shards must be an array")
    provenance = payload.get("parent_dataset_provenance")
    if not isinstance(provenance, Mapping):
        raise ShardFormatError("parent_dataset_provenance must be an object")
    entries = tuple(
        ShardIndexEntry.from_mapping(item)
        if isinstance(item, Mapping)
        else _raise_entry_type()
        for item in raw_entries
    )
    declared_split_counts = payload.get("split_shard_counts")
    if declared_split_counts is not None:
        if not isinstance(declared_split_counts, Mapping):
            raise ShardFormatError("split_shard_counts must be an object")
        expected_split_counts = {
            split: sum(item.split == split for item in entries) for split in SPLITS
        }
        if set(declared_split_counts) != set(SPLITS) or any(
            _integer(declared_split_counts[split], f"split_shard_counts.{split}")
            != expected_split_counts[split]
            for split in SPLITS
        ):
            raise ShardFormatError("split_shard_counts does not match shard entries")
    episode_count = _optional_count(
        payload, "episode_count", sum(item.episode_count for item in entries)
    )
    frame_count = _optional_count(
        payload, "frame_count", sum(item.frame_count for item in entries)
    )
    sequence_count = _optional_count(
        payload, "sequence_count", sum(item.sequence_count for item in entries)
    )
    return TargetStateShardIndex(
        parent_dataset_sha256=_string(
            payload.get("parent_dataset_sha256"), "parent_dataset_sha256"
        ),
        split_seed=_integer(payload.get("split_seed"), "split_seed"),
        history_size=_integer(payload.get("history_size"), "history_size"),
        max_history_age_s=_number(
            payload.get("max_history_age_s"), "max_history_age_s"
        ),
        target_shard_size_bytes=_integer(
            payload.get("target_shard_size_bytes"), "target_shard_size_bytes"
        ),
        parent_dataset_provenance=dict(provenance),
        shards=entries,
        index_sha256=sha256(raw).hexdigest(),
        source_path=source,
        episode_count=episode_count,
        frame_count=frame_count,
        sequence_count=sequence_count,
    )


def plan_target_state_shards(
    dataset_root: str | Path,
    *,
    target_shard_size_bytes: int,
    history_size: int = 6,
    max_history_age_s: float = 2.0,
    split_seed: int = 42,
) -> tuple[ShardPlan, ...]:
    """Validate the parent dataset and return a deterministic soft-size plan."""

    parent = _load_parent_dataset(
        dataset_root,
        history_size=history_size,
        max_history_age_s=max_history_age_s,
        split_seed=split_seed,
    )
    return _plan_parent(
        parent,
        target_shard_size_bytes=target_shard_size_bytes,
        split_seed=split_seed,
    )


def build_target_state_shards(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    target_shard_size_bytes: int,
    history_size: int = 6,
    max_history_age_s: float = 2.0,
    split_seed: int = 42,
) -> ShardBuildResult:
    """Build immutable tar shards and atomically publish ``shard_index.json``."""

    parent = _load_parent_dataset(
        dataset_root,
        history_size=history_size,
        max_history_age_s=max_history_age_s,
        split_seed=split_seed,
    )
    plans = _plan_parent(
        parent,
        target_shard_size_bytes=target_shard_size_bytes,
        split_seed=split_seed,
    )
    destination = _prepare_output_directory(output_dir, parent.root)
    staging = Path(tempfile.mkdtemp(prefix=".target-state-shards.", dir=destination))
    published: list[Path] = []
    published_index: Path | None = None
    entries: list[ShardIndexEntry] = []
    try:
        for plan in plans:
            shard_root = staging / Path(plan.filename).stem
            shard_root.mkdir(mode=0o755)
            _copy_assets(parent.root, shard_root, plan.asset_paths)
            _write_frames(shard_root / "frames.jsonl", plan.records)
            sequences = build_sequences(
                plan.records,
                history_size=history_size,
                max_history_age_s=max_history_age_s,
            )
            shard_dataset_sha = compute_dataset_sha256(shard_root, plan.records)
            dataset_manifest = build_manifest(
                plan.records,
                sequences,
                dataset_sha256=shard_dataset_sha,
                split_seed=split_seed,
                generation_commit_sha=str(
                    parent.manifest.get("generation_commit_sha", "nogit")
                ),
            )
            dataset_manifest.update(_parent_provenance(parent.manifest))
            dataset_manifest.update(
                {
                    "history_size": history_size,
                    "max_history_age_s": max_history_age_s,
                }
            )
            shard_manifest = {
                "schema_version": SHARD_SCHEMA_VERSION,
                "protocol": SHARD_PROTOCOL,
                "shard_id": plan.shard_id,
                "split": plan.split,
                "parent_dataset_sha256": parent.dataset_sha256,
                "shard_dataset_sha256": shard_dataset_sha,
                "episode_ids": list(plan.episode_ids),
                "episode_count": len(plan.episode_ids),
                "frame_count": len(plan.records),
                "sequence_count": len(sequences),
                "history_size": history_size,
                "max_history_age_s": max_history_age_s,
                "generation_commit_sha": str(
                    parent.manifest.get("generation_commit_sha", "nogit")
                ),
            }
            _write_json(shard_root / "dataset_manifest.json", dataset_manifest)
            _write_json(shard_root / "shard_manifest.json", shard_manifest)
            report = check_dataset(
                shard_root,
                sequences=sequences,
                history_size=history_size,
                max_history_age_s=max_history_age_s,
                split_seed=split_seed,
            )
            if not report.ok or report.dataset_sha256 != shard_dataset_sha:
                details = "; ".join(report.errors[:5])
                raise ShardFormatError(
                    f"generated shard {plan.shard_id} failed check_dataset: {details}"
                )
            archive = destination / plan.filename
            _write_deterministic_tar_atomic(shard_root, archive)
            published.append(archive)
            archive_size = archive.stat().st_size
            entries.append(
                ShardIndexEntry(
                    shard_id=plan.shard_id,
                    filename=plan.filename,
                    split=plan.split,
                    episode_ids=plan.episode_ids,
                    episode_count=len(plan.episode_ids),
                    frame_count=len(plan.records),
                    sequence_count=len(sequences),
                    shard_dataset_sha256=shard_dataset_sha,
                    archive_sha256=sha256_file(archive),
                    archive_size_bytes=archive_size,
                )
            )
            shutil.rmtree(shard_root)

        index_payload = {
            "schema_version": SHARD_SCHEMA_VERSION,
            "protocol": SHARD_PROTOCOL,
            "parent_dataset_sha256": parent.dataset_sha256,
            "split_seed": split_seed,
            "history_size": history_size,
            "max_history_age_s": max_history_age_s,
            "target_shard_size_bytes": target_shard_size_bytes,
            "parent_dataset_provenance": _parent_provenance(parent.manifest),
            "episode_count": len({item.episode_id for item in parent.records}),
            "frame_count": len(parent.records),
            "sequence_count": sum(item.sequence_count for item in entries),
            "split_shard_counts": {
                split: sum(item.split == split for item in entries) for split in SPLITS
            },
            "shards": [item.to_dict() for item in entries],
        }
        index_path = destination / "shard_index.json"
        _atomic_write_bytes(index_path, _json_bytes(index_payload), mode=0o444)
        published_index = index_path
        index = load_shard_index(index_path)
        return ShardBuildResult(
            shard_index=index,
            output_dir=destination,
            total_tar_bytes=sum(item.archive_size_bytes for item in entries),
        )
    except Exception:
        if published_index is not None:
            try:
                published_index.unlink()
            except FileNotFoundError:
                pass
        for archive in published:
            try:
                archive.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_parent_dataset(
    dataset_root: str | Path,
    *,
    history_size: int,
    max_history_age_s: float,
    split_seed: int,
) -> _ParentDataset:
    _history_parameters(history_size, max_history_age_s)
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ShardFormatError("split_seed must be an integer")
    root = Path(dataset_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ShardFormatError(f"dataset root is not a directory: {root}")
    manifest_path = root / "dataset_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ShardFormatError(f"cannot read parent dataset manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ShardFormatError("parent dataset manifest must contain an object")
    if manifest.get("history_size") != history_size:
        raise ShardFormatError("requested history_size differs from parent manifest")
    if manifest.get("max_history_age_s") != max_history_age_s:
        raise ShardFormatError(
            "requested max_history_age_s differs from parent manifest"
        )
    records = read_frame_records(root / "frames.jsonl")
    if not records:
        raise ShardFormatError("parent dataset contains no frame records")
    for relative in _asset_paths(records):
        _validate_source_asset(root, relative)
    report = check_dataset(
        root,
        history_size=history_size,
        max_history_age_s=max_history_age_s,
        split_seed=split_seed,
    )
    if not report.ok or report.dataset_sha256 is None:
        raise ShardFormatError(
            "parent dataset failed validation: " + "; ".join(report.errors[:5])
        )
    declared_sha = manifest.get("dataset_sha256")
    if declared_sha != report.dataset_sha256:
        raise ShardFormatError("parent dataset manifest SHA does not match data")
    return _ParentDataset(root, records, dict(manifest), report.dataset_sha256)


def _plan_parent(
    parent: _ParentDataset,
    *,
    target_shard_size_bytes: int,
    split_seed: int,
) -> tuple[ShardPlan, ...]:
    _positive_int(target_shard_size_bytes, "target_shard_size_bytes")
    grouped: dict[str, list[TargetStateFrameRecord]] = {}
    for record in parent.records:
        grouped.setdefault(record.episode_id, []).append(record)
    episodes: list[_Episode] = []
    for episode_id in sorted(grouped):
        records = tuple(
            sorted(grouped[episode_id], key=lambda item: (item.timestamp_s, item.frame_id))
        )
        assets = _asset_paths(records)
        payload_size = sum(_tar_member_cost((parent.root / item).stat().st_size) for item in assets)
        payload_size += _tar_member_cost(
            sum(len(_record_line(item)) for item in records)
        )
        episodes.append(
            _Episode(
                episode_id=episode_id,
                split=split_for_episode(episode_id, seed=split_seed),
                records=records,
                asset_paths=assets,
                payload_size_bytes=payload_size,
            )
        )

    plans: list[ShardPlan] = []
    for split in SPLITS:
        split_episodes = [item for item in episodes if item.split == split]
        bins: list[list[_Episode]] = []
        current: list[_Episode] = []
        current_size = _MANIFEST_ALLOWANCE_BYTES
        for episode in split_episodes:
            if current and current_size + episode.payload_size_bytes > target_shard_size_bytes:
                bins.append(current)
                current = []
                current_size = _MANIFEST_ALLOWANCE_BYTES
            current.append(episode)
            current_size += episode.payload_size_bytes
        if current:
            bins.append(current)
        for ordinal, members in enumerate(bins, start=1):
            shard_id = f"{split}_{ordinal:06d}"
            filename = f"shard_stagea_{split}_{ordinal:06d}.tar"
            records = tuple(item for episode in members for item in episode.records)
            plans.append(
                ShardPlan(
                    shard_id=shard_id,
                    filename=filename,
                    split=split,
                    episode_ids=tuple(item.episode_id for item in members),
                    records=records,
                    asset_paths=tuple(
                        sorted(
                            {
                                path
                                for episode in members
                                for path in episode.asset_paths
                            }
                        )
                    ),
                    estimated_size_bytes=(
                        _MANIFEST_ALLOWANCE_BYTES
                        + sum(item.payload_size_bytes for item in members)
                    ),
                )
            )
    if not plans:
        raise ShardFormatError("no shard plans were generated")
    return tuple(plans)


def _prepare_output_directory(path: str | Path, dataset_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists() and candidate.is_symlink():
        raise ShardFormatError("output directory cannot be a symlink")
    candidate.mkdir(parents=True, exist_ok=True)
    destination = candidate.resolve(strict=True)
    if destination == dataset_root or dataset_root in destination.parents:
        raise ShardFormatError("output directory must be outside the parent dataset")
    if any(destination.iterdir()):
        raise ShardFormatError(f"output directory must be empty: {destination}")
    return destination


def _asset_paths(records: Sequence[TargetStateFrameRecord]) -> tuple[str, ...]:
    result: set[str] = set()
    for record in records:
        paths = [record.sensor_input.rgb_path, record.sensor_input.depth_path]
        if record.sensor_input.instance_mask_path is not None:
            paths.append(record.sensor_input.instance_mask_path)
        for relative in paths:
            _safe_relative_path(relative, "asset path")
            result.add(relative)
    return tuple(sorted(result))


def _validate_source_asset(root: Path, relative: str) -> None:
    parts = PurePosixPath(relative).parts
    current = root
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise ShardFormatError(f"missing asset {relative}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ShardFormatError(f"asset path contains a symlink: {relative}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ShardFormatError(f"asset parent is not a directory: {relative}")
        if index == len(parts) - 1 and not stat.S_ISREG(info.st_mode):
            raise ShardFormatError(f"asset is not a regular file: {relative}")


def _copy_assets(source_root: Path, destination_root: Path, assets: Sequence[str]) -> None:
    for relative in assets:
        destination = destination_root.joinpath(*PurePosixPath(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_fd = _open_asset_nofollow(source_root, relative)
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            destination_fd = os.open(destination, flags, 0o444)
            try:
                with os.fdopen(source_fd, "rb", closefd=False) as source_stream:
                    with os.fdopen(destination_fd, "wb", closefd=False) as target_stream:
                        shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
                        target_stream.flush()
            finally:
                os.close(destination_fd)
        finally:
            os.close(source_fd)


def _open_asset_nofollow(root: Path, relative: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_DIRECTORY"):
        directory_flags = flags | os.O_DIRECTORY
    else:  # pragma: no cover - POSIX platforms used by the project have it
        directory_flags = flags
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
        directory_flags |= os.O_NOFOLLOW
    parent_fd = os.open(root, directory_flags)
    try:
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        result = os.open(parts[-1], flags, dir_fd=parent_fd)
        if not stat.S_ISREG(os.fstat(result).st_mode):
            os.close(result)
            raise ShardFormatError(f"asset is not a regular file: {relative}")
        return result
    except OSError as exc:
        raise ShardFormatError(f"cannot safely open asset {relative}: {exc}") from exc
    finally:
        os.close(parent_fd)


def _write_frames(path: Path, records: Sequence[TargetStateFrameRecord]) -> None:
    data = b"".join(_record_line(item) for item in records)
    path.write_bytes(data)


def _record_line(record: TargetStateFrameRecord) -> bytes:
    return (
        json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_bytes(_json_bytes(payload))


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_deterministic_tar_atomic(source_root: Path, destination: Path) -> None:
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{os.urandom(6).hex()}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    # Keep the staging inode writable while the complete archive is being
    # produced.  The requested read-only host mode is best-effort metadata;
    # archive integrity is provided by the deterministic tar bytes and SHA.
    file_descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(file_descriptor, "wb", closefd=False) as raw:
            with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
                    if source.is_symlink():
                        raise ShardFormatError(f"staging tree contains symlink: {source}")
                    relative = source.relative_to(source_root).as_posix()
                    _safe_relative_path(relative, "tar member")
                    info = tarfile.TarInfo(relative)
                    info.size = source.stat().st_size
                    info.mtime = 0
                    info.mode = 0o444
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with source.open("rb") as stream:
                        archive.addfile(info, stream)
            raw.flush()
            os.fsync(file_descriptor)
        _try_chmod_portable(temporary, 0o444)
        _publish_file_noreplace(temporary, destination)
    finally:
        os.close(file_descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_bytes(path: Path, data: bytes, *, mode: int) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{os.urandom(6).hex()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(descriptor)
        _try_chmod_portable(temporary, mode)
        _publish_file_noreplace(temporary, path)
    finally:
        os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _try_chmod_portable(path: Path, mode: int) -> None:
    """Apply a host-file mode unless the filesystem lacks POSIX modes.

    WSL DrvFS/NTFS mounts may reject ``chmod`` even though writing, syncing,
    linking, and reading the file all work.  Only the errno values that
    explicitly describe an unsupported permission operation are tolerated;
    I/O errors and every other failure remain fatal.
    """

    try:
        os.chmod(path, mode)
    except OSError as exc:
        if exc.errno not in {
            errno.EPERM,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
        }:
            raise


def _publish_file_noreplace(temporary: Path, destination: Path) -> None:
    """Atomically publish a complete same-directory file without clobbering.

    ``os.replace`` would have a check/replace race and could overwrite an
    immutable shard produced by another process.  A hard-link publication is
    atomic, fails when the final name exists, and is followed by unlink+fsync.
    """

    source_info = temporary.stat(follow_symlinks=False)
    published = False
    try:
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite immutable output: {destination}"
            ) from exc
        published = True
        temporary.unlink()
        _fsync_directory(destination.parent)
    except Exception:
        if published:
            try:
                destination_info = destination.stat(follow_symlinks=False)
                if (
                    destination_info.st_dev,
                    destination_info.st_ino,
                ) == (source_info.st_dev, source_info.st_ino):
                    destination.unlink()
                    _fsync_directory(destination.parent)
            except FileNotFoundError:
                pass
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
    finally:
        os.close(descriptor)


def _parent_provenance(manifest: Mapping[str, object]) -> dict[str, object]:
    return {field: manifest[field] for field in _PROVENANCE_FIELDS if field in manifest}


def _tar_member_cost(size: int) -> int:
    return 512 + ((size + 511) // 512) * 512


def _history_parameters(history_size: int, max_history_age_s: float) -> None:
    if isinstance(history_size, bool) or not isinstance(history_size, int):
        raise ShardFormatError("history_size must be an integer")
    if not 4 <= history_size <= 8:
        raise ShardFormatError("history_size must be within [4, 8]")
    if not isinstance(max_history_age_s, (int, float)) or isinstance(
        max_history_age_s, bool
    ):
        raise ShardFormatError("max_history_age_s must be numeric")
    if not isfinite(float(max_history_age_s)) or float(max_history_age_s) <= 0.0:
        raise ShardFormatError("max_history_age_s must be finite and positive")


def _safe_relative_path(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ShardFormatError(f"{field} must be a portable relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise ShardFormatError(f"unsafe {field}: {value!r}")
    return value


def _safe_shard_filename(value: str) -> str:
    _safe_relative_path(value, "shard filename")
    if (
        PurePosixPath(value).name != value
        or not value.startswith("shard_")
        or not value.endswith(".tar")
    ):
        raise ShardFormatError(f"invalid shard filename: {value!r}")
    return value


def _safe_identifier(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 64
        or value != value.strip()
        or value[0] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in value
        )
    ):
        raise ShardFormatError(f"invalid {field}: {value!r}")
    return value


def _digest(value: object, field: str) -> str:
    text = _string(value, field).lower()
    if len(text) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ShardFormatError(f"{field} must be a SHA256 digest")
    return text


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ShardFormatError(f"{field} must be a string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ShardFormatError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShardFormatError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ShardFormatError(f"{field} must be finite")
    return result


def _positive_int(value: object, field: str) -> int:
    result = _integer(value, field)
    if result <= 0:
        raise ShardFormatError(f"{field} must be positive")
    return result


def _nonnegative_int(value: object, field: str) -> int:
    result = _integer(value, field)
    if result < 0:
        raise ShardFormatError(f"{field} must be non-negative")
    return result


def _optional_count(payload: Mapping[str, object], key: str, default: int) -> int:
    return _integer(payload[key], key) if key in payload else default


def _raise_entry_type() -> Any:
    raise ShardFormatError("each shard index entry must be an object")


__all__ = [
    "SHARD_PROTOCOL",
    "SHARD_SCHEMA_VERSION",
    "ShardBuildResult",
    "ShardFormatError",
    "ShardIndexEntry",
    "ShardPlan",
    "TargetStateShardIndex",
    "build_target_state_shards",
    "load_shard_index",
    "plan_target_state_shards",
    "sha256_file",
]
