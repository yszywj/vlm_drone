"""Offline case evidence: immutable RGB-D copies, bounded disk, no gate changes."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from scripts.inspect_target_state_episode import copy_verified_asset
from training.target_state.sharded_trainer import _fsync_directory
from training.target_state.trainer import sha256_file


@dataclass(frozen=True)
class CaseExportOptions:
    max_bytes: int = 2 * 1024**3
    outlier_error_m: float = 1.0
    episode_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.max_bytes < 0:
            raise ValueError("case asset budget must be non-negative")
        if not math.isfinite(self.outlier_error_m) or self.outlier_error_m <= 0:
            raise ValueError("case outlier threshold must be finite and positive")


def case_tags(row, options):
    tags = []
    if row["no_target"] and row["model_valid"]:
        tags.append("model_false_positive")
    if row["episode_id"] in options.episode_ids:
        tags.append("requested_episode")
    if row["evaluated_visible_target"]:
        if row["model_valid"] and row["model_error_m"] >= options.outlier_error_m:
            tags.append("accepted_position_outlier")
        if row["baseline_valid"] and not row["model_valid"]:
            tags.append("model_only_rejection")
    return tags


def _contained(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe evidence path: {relative}")
    path = root / relative
    if path.is_symlink() or root.resolve() not in path.resolve().parents:
        raise ValueError(f"evidence path escapes root: {relative}")
    return path


def export_cases(rows, sequences, *, dataset_root, output_dir, entry, options):
    """Select per-shard cases before consume; never silently exceed the budget.

    Full temporal windows, not just crops, are copied byte-for-byte. Metadata
    survives even if the RGB-D budget is exhausted. On crash/retry, existing
    assets count towards the budget and must match the verified source bytes.
    """
    asset_root = output_dir / "case_assets"
    if asset_root.is_symlink():
        raise ValueError("case_assets must not be a symlink")
    used = sum(p.stat().st_size for p in asset_root.rglob("*") if p.is_file())
    selected = [(row, sequence, case_tags(row, options))
                for row, sequence in zip(rows, sequences)]
    selected = [item for item in selected if item[2]]
    priority = ("model_false_positive", "requested_episode",
                "accepted_position_outlier", "model_only_rejection")
    selected.sort(key=lambda item: (min(priority.index(t) for t in item[2]),
                                   item[0]["sequence_id"]))
    cases, assets = [], {}
    for row, sequence, tags in selected:
        frames = (*sequence.history, sequence.reference)
        paths = sorted({p for f in frames for p in
                        (f.sensor_input.rgb_path, f.sensor_input.depth_path)})
        copies = []
        additional_bytes = 0
        for relative in paths:
            source = _contained(dataset_root, relative)
            name = (Path("case_assets") / entry.split / entry.filename / relative).as_posix()
            destination = _contained(output_dir, name)
            expected = sha256_file(source)
            size = source.stat().st_size
            if destination.exists():
                if sha256_file(destination) != expected:
                    raise ValueError(f"existing case evidence differs: {destination}")
            else:
                additional_bytes += size
            copies.append((source, destination, name, expected, size))
        complete = additional_bytes == 0 or used + additional_bytes <= options.max_bytes
        if complete:
            for source, destination, name, expected, size in copies:
                copied = copy_verified_asset(source, destination)
                if copied != expected:
                    raise ValueError(f"source evidence changed during copy: {source}")
                # Persist newly created directory entries as well as file data.
                parent = destination.parent
                while parent != output_dir:
                    _fsync_directory(parent)
                    parent = parent.parent
                _fsync_directory(output_dir)
                assets[name] = {"sha256": expected, "size_bytes": size}
            used += additional_bytes
        cases.append({
            "offline_only": True, "tags": tags, "sample": row,
            "archive_sha256": entry.archive_sha256,
            "records": [frame.to_dict() for frame in frames],
            "delta_t_s": list(sequence.delta_t_s),
            "rgbd_export_complete": complete,
            "rgbd_omission_reason": None if complete else "case_asset_budget_exhausted",
            "asset_paths": {relative: name for relative, (_, _, name, _, _) in zip(paths, copies)}
                           if complete else {},
        })
    return {"cases": cases, "assets": assets}


def verify_case_assets(evidence, output_dir):
    """A durable receipt is insufficient if its promised images disappeared."""
    for relative, expected in evidence["assets"].items():
        path = _contained(output_dir, relative)
        if not path.is_file() or path.stat().st_size != expected["size_bytes"]:
            raise ValueError(f"missing or truncated case evidence: {path}")
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"case evidence SHA256 mismatch: {path}")
    for case in evidence["cases"]:
        if case["rgbd_export_complete"]:
            required = {p for frame in case["records"] for p in
                        (frame["sensor_input"]["rgb_path"], frame["sensor_input"]["depth_path"])}
            if required != set(case["asset_paths"]):
                raise ValueError("case evidence does not include the full temporal window")
            if not set(case["asset_paths"].values()).issubset(evidence["assets"]):
                raise ValueError("case asset has no committed digest")


def compare_metrics(actual, expected, path=""):
    """Report replay drift rather than silently treating it as the original run."""
    differences = []
    for key, old in expected.items():
        name = f"{path}.{key}" if path else key
        new = actual.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            differences.extend(compare_metrics(new, old, name))
        elif isinstance(old, float) and isinstance(new, (float, int)):
            if not math.isclose(old, new, rel_tol=1e-4, abs_tol=1e-6):
                differences.append({"field": name, "training": old, "audit": new})
        elif old != new:
            differences.append({"field": name, "training": old, "audit": new})
    return differences
