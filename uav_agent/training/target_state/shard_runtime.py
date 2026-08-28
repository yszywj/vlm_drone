"""Safe, atomic materialization of Target State shard archives."""

from __future__ import annotations

from dataclasses import dataclass
import errno
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
from typing import BinaryIO, Mapping

from datasets.target_state.dataset import (
    DatasetCheckReport,
    check_dataset,
    compute_dataset_sha256,
    read_frame_records,
    split_for_episode,
)
from datasets.target_state.sequence import build_sequences
from datasets.target_state.schema import TargetStateFrameRecord
from training.target_state.shards import (
    SHARD_PROTOCOL,
    SHARD_SCHEMA_VERSION,
    ShardFormatError,
    ShardIndexEntry,
    TargetStateShardIndex,
    load_shard_index,
)


class ShardMaterializationError(RuntimeError):
    """Raised when an archive cannot be safely materialized and validated."""


@dataclass(frozen=True, slots=True)
class MaterializedShard:
    archive_path: Path
    materialized_root: Path
    dataset_root: Path
    entry: ShardIndexEntry
    shard_index_sha256: str
    dataset_report: DatasetCheckReport


def materialize_shard(
    archive_path: str | Path,
    index: TargetStateShardIndex | str | Path,
    *,
    materialized_root: str | Path | None = None,
) -> MaterializedShard:
    """Validate, safely extract, and atomically publish one indexed shard.

    Repeating the call is idempotent: an existing final directory is returned
    only after it passes the same complete validation as a newly extracted one.
    Invalid existing data is never overwritten.
    """

    resolved_index = (
        index if isinstance(index, TargetStateShardIndex) else load_shard_index(index)
    )
    archive_input = Path(archive_path).expanduser()
    try:
        archive_info = archive_input.lstat()
    except OSError as exc:
        raise ShardMaterializationError(f"cannot stat shard archive: {exc}") from exc
    if stat.S_ISLNK(archive_info.st_mode) or not stat.S_ISREG(archive_info.st_mode):
        raise ShardMaterializationError("shard archive must be a non-symlink regular file")
    archive = archive_input.resolve(strict=True)
    try:
        entry = resolved_index.entry_for_filename(archive.name)
    except ShardFormatError as exc:
        raise ShardMaterializationError(str(exc)) from exc
    root = _prepare_materialized_root(
        archive.parent / ".materialized"
        if materialized_root is None
        else Path(materialized_root).expanduser()
    )
    final = root / archive.stem
    _validate_final_location(final, root)
    _cleanup_orphan_temporary_directories(root, archive.stem)

    stream = _open_archive_nofollow(archive)
    try:
        actual_size = os.fstat(stream.fileno()).st_size
        if actual_size != entry.archive_size_bytes:
            raise ShardMaterializationError(
                "archive size does not match shard index: "
                f"expected={entry.archive_size_bytes}, actual={actual_size}"
            )
        actual_archive_sha = _sha256_stream(stream)
        if actual_archive_sha != entry.archive_sha256:
            raise ShardMaterializationError(
                "archive SHA256 does not match shard index"
            )

        if final.exists() or final.is_symlink():
            report = validate_materialized_shard(final, resolved_index, entry)
            return MaterializedShard(
                archive,
                root,
                final,
                entry,
                resolved_index.index_sha256,
                report,
            )

        temporary = root / f"{archive.stem}.tmp.{os.getpid()}.{os.urandom(6).hex()}"
        temporary.mkdir(mode=0o700)
        try:
            stream.seek(0)
            with tarfile.open(fileobj=stream, mode="r:") as tar:
                members = tar.getmembers()
                _validate_tar_members(members, archive_size=actual_size)
                _extract_members(tar, members, temporary)
            report = validate_materialized_shard(
                temporary, resolved_index, entry, expected_directory_name=archive.stem
            )
            _fsync_tree(temporary)
            try:
                os.rename(temporary, final)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                # Another identical materializer may have won the race.  Its
                # result is acceptable only after full validation.
                report = validate_materialized_shard(final, resolved_index, entry)
                shutil.rmtree(temporary)
            _fsync_directory(root)
            return MaterializedShard(
                archive,
                root,
                final,
                entry,
                resolved_index.index_sha256,
                report,
            )
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    except ShardMaterializationError:
        raise
    except (OSError, tarfile.TarError, ValueError, ShardFormatError) as exc:
        raise ShardMaterializationError(f"cannot materialize shard: {exc}") from exc
    finally:
        stream.close()


def validate_materialized_shard(
    dataset_root: str | Path,
    index: TargetStateShardIndex,
    entry: ShardIndexEntry,
    *,
    expected_directory_name: str | None = None,
) -> DatasetCheckReport:
    """Validate manifests, hashes, assets, episode split, and canonical checker."""

    root = Path(dataset_root)
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise ShardMaterializationError(f"cannot stat materialized shard: {exc}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ShardMaterializationError("materialized shard must be a real directory")
    if (
        expected_directory_name is not None
        and expected_directory_name != Path(entry.filename).stem
    ):
        raise ShardMaterializationError("temporary shard directory does not match archive stem")
    # Reject every symlink/special file before any manifest, JSONL, image, or
    # NumPy reader gets a chance to follow or block on it.
    actual_files = _materialized_files(root)

    shard_manifest = _read_json_object(root / "shard_manifest.json")
    dataset_manifest = _read_json_object(root / "dataset_manifest.json")
    if shard_manifest.get("schema_version") != SHARD_SCHEMA_VERSION:
        raise ShardMaterializationError("shard_manifest schema_version mismatch")
    if shard_manifest.get("protocol") != SHARD_PROTOCOL:
        raise ShardMaterializationError("shard_manifest protocol mismatch")
    expected_manifest_values: Mapping[str, object] = {
        "shard_id": entry.shard_id,
        "split": entry.split,
        "parent_dataset_sha256": index.parent_dataset_sha256,
        "shard_dataset_sha256": entry.shard_dataset_sha256,
        "episode_ids": list(entry.episode_ids),
        "episode_count": entry.episode_count,
        "frame_count": entry.frame_count,
        "sequence_count": entry.sequence_count,
        "history_size": index.history_size,
        "max_history_age_s": index.max_history_age_s,
    }
    for field, expected in expected_manifest_values.items():
        if shard_manifest.get(field) != expected:
            raise ShardMaterializationError(
                f"shard_manifest {field} does not match shard index"
            )
    expected_generation = index.parent_dataset_provenance.get(
        "generation_commit_sha"
    )
    if (
        expected_generation is not None
        and shard_manifest.get("generation_commit_sha") != expected_generation
    ):
        raise ShardMaterializationError(
            "shard_manifest generation_commit_sha does not match parent provenance"
        )
    if dataset_manifest.get("dataset_sha256") != entry.shard_dataset_sha256:
        raise ShardMaterializationError(
            "dataset_manifest dataset_sha256 does not match shard index"
        )
    if dataset_manifest.get("history_size") != index.history_size:
        raise ShardMaterializationError("dataset_manifest history_size mismatch")
    if dataset_manifest.get("max_history_age_s") != index.max_history_age_s:
        raise ShardMaterializationError("dataset_manifest max_history_age_s mismatch")
    for field, expected in index.parent_dataset_provenance.items():
        if field not in dataset_manifest or dataset_manifest[field] != expected:
            raise ShardMaterializationError(
                f"dataset_manifest inherited provenance mismatch: {field}"
            )

    try:
        records = read_frame_records(root / "frames.jsonl")
    except (OSError, ValueError) as exc:
        raise ShardMaterializationError(f"invalid frames.jsonl: {exc}") from exc
    actual_episodes = tuple(sorted({item.episode_id for item in records}))
    if actual_episodes != tuple(sorted(entry.episode_ids)):
        raise ShardMaterializationError("materialized episode IDs do not match shard index")
    if any(
        split_for_episode(episode, seed=index.split_seed) != entry.split
        for episode in actual_episodes
    ):
        raise ShardMaterializationError("materialized shard crosses the declared split")
    assets = _record_asset_paths(records)
    expected_files = {
        "frames.jsonl",
        "dataset_manifest.json",
        "shard_manifest.json",
        *assets,
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ShardMaterializationError(
            f"materialized file set mismatch: missing={missing}, unexpected={unexpected}"
        )
    sequences = build_sequences(
        records,
        history_size=index.history_size,
        max_history_age_s=index.max_history_age_s,
    )
    if len(records) != entry.frame_count or len(sequences) != entry.sequence_count:
        raise ShardMaterializationError("materialized frame/sequence counts do not match index")
    actual_dataset_sha = compute_dataset_sha256(root, records)
    if actual_dataset_sha != entry.shard_dataset_sha256:
        raise ShardMaterializationError("materialized shard dataset SHA256 mismatch")
    report = check_dataset(
        root,
        sequences=sequences,
        history_size=index.history_size,
        max_history_age_s=index.max_history_age_s,
        split_seed=index.split_seed,
    )
    if not report.ok or report.dataset_sha256 != actual_dataset_sha:
        raise ShardMaterializationError(
            "materialized shard failed check_dataset: " + "; ".join(report.errors[:5])
        )
    return report


def cleanup_materialized_shard(materialized: MaterializedShard) -> None:
    if not isinstance(materialized, MaterializedShard):
        raise TypeError("materialized must be a MaterializedShard receipt")
    cleanup_materialized_path(
        materialized.dataset_root,
        materialized_root=materialized.materialized_root,
    )


def cleanup_materialized_path(
    path: str | Path,
    *,
    materialized_root: str | Path,
) -> None:
    """Idempotently remove one final direct child after strict boundary checks."""

    root_input = Path(materialized_root).expanduser()
    try:
        root_info = root_input.lstat()
    except FileNotFoundError:
        # A missing root is a valid already-clean state only when the requested
        # path lexically names its direct child.
        root = Path(os.path.abspath(root_input))
        target = Path(os.path.abspath(Path(path).expanduser()))
        _validate_cleanup_target(target, root)
        return
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ShardMaterializationError("materialized_root must be a real directory")
    root = root_input.resolve(strict=True)
    target = Path(os.path.abspath(Path(path).expanduser()))
    _validate_cleanup_target(target, root)
    try:
        info = target.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ShardMaterializationError(
            "cleanup target must be a non-symlink materialized directory"
        )
    shutil.rmtree(target)
    _fsync_directory(root)


def _prepare_materialized_root(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ShardMaterializationError("materialized_root must be a real directory")
    else:
        try:
            path.mkdir(mode=0o755, parents=False)
        except FileExistsError:
            pass
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ShardMaterializationError("materialized_root creation was redirected")
        _fsync_directory(path.parent)
    return path.resolve(strict=True)


def _validate_final_location(path: Path, root: Path) -> None:
    if path.parent != root or path.name.startswith(".") or not path.name.startswith("shard_"):
        raise ShardMaterializationError("unsafe final materialized shard path")
    if path.name.endswith(".tmp") or ".tmp." in path.name:
        raise ShardMaterializationError("temporary name cannot be used as final shard path")


def _validate_cleanup_target(target: Path, root: Path) -> None:
    _validate_final_location(target, root)


def _cleanup_orphan_temporary_directories(root: Path, archive_stem: str) -> None:
    """Remove only dead-PID extraction directories for this exact archive."""

    prefix = f"{archive_stem}.tmp."
    changed = False
    for candidate in root.iterdir():
        if not candidate.name.startswith(prefix):
            continue
        suffix = candidate.name[len(prefix) :]
        pieces = suffix.split(".", 1)
        if (
            len(pieces) != 2
            or not pieces[0].isdigit()
            or len(pieces[1]) != 12
            or any(character not in "0123456789abcdef" for character in pieces[1])
        ):
            raise ShardMaterializationError(
                f"malformed shard extraction temporary entry: {candidate.name}"
            )
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ShardMaterializationError(
                f"unsafe shard extraction temporary entry: {candidate.name}"
            )
        owner_pid = int(pieces[0])
        if owner_pid == os.getpid() or _process_is_alive(owner_pid):
            continue
        shutil.rmtree(candidate)
        changed = True
    if changed:
        _fsync_directory(root)


def _process_is_alive(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _open_archive_nofollow(path: Path) -> BinaryIO:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ShardMaterializationError(f"cannot safely open shard archive: {exc}") from exc
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise ShardMaterializationError("shard archive changed to a non-regular file")
    return os.fdopen(descriptor, "rb")


def _sha256_stream(stream: BinaryIO) -> str:
    stream.seek(0)
    digest = sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def _validate_tar_members(members: list[tarfile.TarInfo], *, archive_size: int) -> None:
    if not members:
        raise ShardMaterializationError("shard archive is empty")
    if len(members) > 1_000_000:
        raise ShardMaterializationError("shard archive has too many members")
    names: set[str] = set()
    regular_names: set[str] = set()
    total_regular_size = 0
    for member in members:
        name = _safe_member_name(member)
        if name in names:
            raise ShardMaterializationError(f"duplicate tar member: {name}")
        names.add(name)
        if member.isdir():
            continue
        if not member.isreg():
            raise ShardMaterializationError(
                f"tar member type is forbidden: {member.name}"
            )
        if getattr(member, "sparse", None):
            raise ShardMaterializationError("sparse tar members are forbidden")
        if member.size < 0:
            raise ShardMaterializationError("tar member has a negative size")
        total_regular_size += member.size
        regular_names.add(name)
    if total_regular_size > archive_size:
        raise ShardMaterializationError("tar member sizes exceed the archive size")
    for name in names:
        parts = PurePosixPath(name).parts
        for index in range(1, len(parts)):
            if PurePosixPath(*parts[:index]).as_posix() in regular_names:
                raise ShardMaterializationError(
                    f"tar member descends from a regular file: {name}"
                )
    required = {"frames.jsonl", "dataset_manifest.json", "shard_manifest.json"}
    if not required.issubset(regular_names):
        raise ShardMaterializationError(
            f"tar archive is missing required files: {sorted(required - regular_names)}"
        )


def _safe_member_name(member: tarfile.TarInfo) -> str:
    raw = member.name.rstrip("/") if member.isdir() else member.name
    if not raw or "\\" in raw or "\x00" in raw:
        raise ShardMaterializationError(f"unsafe tar member path: {member.name!r}")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != raw
        or path.parts[0].endswith(":")
    ):
        raise ShardMaterializationError(f"unsafe tar member path: {member.name!r}")
    return raw


def _extract_members(
    archive: tarfile.TarFile,
    members: list[tarfile.TarInfo],
    destination: Path,
) -> None:
    directories = sorted(
        (item for item in members if item.isdir()),
        key=lambda item: len(PurePosixPath(_safe_member_name(item)).parts),
    )
    for member in directories:
        relative = _safe_member_name(member)
        path = destination.joinpath(*PurePosixPath(relative).parts)
        path.mkdir(parents=True, exist_ok=True, mode=0o755)
    for member in (item for item in members if item.isreg()):
        relative = _safe_member_name(member)
        path = destination.joinpath(*PurePosixPath(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        source = archive.extractfile(member)
        if source is None:
            raise ShardMaterializationError(f"cannot read tar member: {relative}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o444)
        try:
            remaining = member.size
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ShardMaterializationError(
                        f"short tar member payload: {relative}"
                    )
                _write_all(descriptor, chunk)
                remaining -= len(chunk)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
            source.close()


def _read_json_object(path: Path) -> Mapping[str, object]:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ShardMaterializationError(f"manifest is not a regular file: {path.name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ShardMaterializationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ShardMaterializationError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ShardMaterializationError(f"{path.name} must contain a JSON object")
    return payload


def _record_asset_paths(records: tuple[TargetStateFrameRecord, ...]) -> set[str]:
    result: set[str] = set()
    for record in records:
        sensor = record.sensor_input
        paths = [sensor.rgb_path, sensor.depth_path]
        if sensor.instance_mask_path is not None:
            paths.append(sensor.instance_mask_path)
        for relative in paths:
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative:
                raise ShardMaterializationError(f"unsafe record asset path: {relative}")
            result.add(path.as_posix())
    return result


def _materialized_files(root: Path) -> set[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        info = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(info.st_mode):
            raise ShardMaterializationError(
                f"materialized tree contains a symlink: {relative}"
            )
        if stat.S_ISREG(info.st_mode):
            result.add(relative)
        elif not stat.S_ISDIR(info.st_mode):
            raise ShardMaterializationError(
                f"materialized tree contains a special file: {relative}"
            )
    return result


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            directories.append(path)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:  # pragma: no cover - defensive OS contract check
            raise OSError("short write while extracting tar member")
        view = view[written:]


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


__all__ = [
    "MaterializedShard",
    "ShardMaterializationError",
    "cleanup_materialized_path",
    "cleanup_materialized_shard",
    "load_shard_index",
    "materialize_shard",
    "validate_materialized_shard",
]
