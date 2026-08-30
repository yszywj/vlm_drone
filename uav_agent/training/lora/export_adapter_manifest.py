#!/usr/bin/env python3
"""Export a verified, backward-compatible manifest for a trained PEFT adapter."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import struct
import sys
from typing import Mapping


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_BASE_FILES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}
_FORBIDDEN_BASE_SHARD_RE = re.compile(
    r"^(?:model-\d{5}-of-\d{5}\.safetensors|pytorch_model-\d{5}-of-\d{5}\.bin)$"
)


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, description: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file: {path}")
    return path


def _resolved_regular_file(value: str | Path, description: str) -> Path:
    requested = Path(value).expanduser()
    if requested.is_symlink():
        raise ValueError(f"{description} cannot be a symlink: {requested}")
    return _regular_file(requested.resolve(), description)


def _read_object(path: Path, description: str) -> dict[str, object]:
    _regular_file(path, description)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {description} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def _non_empty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _sha256_text(value: object, field: str) -> str:
    text = _non_empty_text(value, field).lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _validate_language_target_name(value: object) -> str:
    target = _non_empty_text(value, "adapter target module")
    if re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", target) is None:
        raise ValueError(f"adapter target module is not an exact module name: {target!r}")
    if not target.startswith("model.language_model.") or not any(
        marker in target
        for marker in (
            ".self_attn.",
            ".attention.",
            ".attn.",
            ".mlp.",
            ".feed_forward.",
        )
    ):
        raise ValueError(
            f"adapter target module is not language attention/MLP only: {target!r}"
        )
    return target


def _exact_targets_from_regex(expression: str) -> list[str]:
    """Recover exact safe module names from modeling.exact_target_regex output."""

    if not expression.startswith("^(?:") or not expression.endswith(")$"):
        raise ValueError("adapter target regex must be an anchored exact alternation")
    body = expression[4:-2]
    escaped_names = body.split("|")
    if not escaped_names or any(not item for item in escaped_names):
        raise ValueError("adapter target regex has an empty alternative")
    escaped_name = re.compile(r"[A-Za-z0-9_]+(?:\\\.[A-Za-z0-9_]+)+")
    targets: list[str] = []
    for item in escaped_names:
        if escaped_name.fullmatch(item) is None:
            raise ValueError(
                "adapter target regex must contain exact escaped module names only"
            )
        targets.append(_validate_language_target_name(item.replace(r"\.", ".")))
    if len(targets) != len(set(targets)):
        raise ValueError("adapter target regex contains duplicate module names")
    return sorted(targets)


def _validate_safetensors(path: Path) -> None:
    """Validate the untrusted safetensors envelope without importing torch."""

    _regular_file(path, "adapter weights")
    size = path.stat().st_size
    if size < 10:
        raise ValueError(f"adapter safetensors file is truncated: {path}")
    with path.open("rb") as stream:
        header_size_bytes = stream.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"adapter safetensors header is truncated: {path}")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if header_size <= 1 or header_size > size - 8 or header_size > 100 * 1024 * 1024:
            raise ValueError(f"adapter safetensors header size is invalid: {path}")
        try:
            header = json.loads(stream.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"adapter safetensors header is invalid: {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"adapter safetensors header must be an object: {path}")
    data_size = size - 8 - header_size
    intervals: list[tuple[int, int]] = []
    tensor_count = 0
    for name, descriptor in header.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict):
                raise ValueError(f"invalid safetensors metadata in {path}")
            continue
        tensor_count += 1
        if not isinstance(name, str) or not name or not isinstance(descriptor, dict):
            raise ValueError(f"invalid tensor descriptor in {path}")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"invalid dtype for tensor {name!r} in {path}")
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape
        ):
            raise ValueError(f"invalid shape for tensor {name!r} in {path}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
        ):
            raise ValueError(f"invalid offsets for tensor {name!r} in {path}")
        start, end = offsets
        if start < 0 or end <= start or end > data_size:
            raise ValueError(f"out-of-range offsets for tensor {name!r} in {path}")
        intervals.append((start, end))
    if tensor_count == 0:
        raise ValueError(f"adapter safetensors contains no tensors: {path}")
    intervals.sort()
    if any(
        right_start < left_end
        for (_, left_end), (right_start, _) in zip(intervals, intervals[1:])
    ):
        raise ValueError(f"adapter safetensors tensor ranges overlap: {path}")


def _adapter_metadata(
    config: Mapping[str, object], base_model_name: str
) -> tuple[int, list[str], re.Pattern[str] | None]:
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("adapter_config.json peft_type must be LORA")
    rank = config.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("adapter_config.json r must be a positive integer")
    raw_targets = config.get("target_modules")
    target_regex: re.Pattern[str] | None = None
    if isinstance(raw_targets, list) and raw_targets:
        targets = sorted({_validate_language_target_name(item) for item in raw_targets})
        if len(targets) != len(raw_targets):
            raise ValueError("adapter_config.json target_modules contains duplicates")
    elif isinstance(raw_targets, str) and raw_targets:
        # PEFT serializes a regex target_modules value as a string.  The model
        # loader constructs one anchored regex from an already audited exact
        # language-module set.  A run manifest is needed below to recover and
        # verify those expanded names.
        if len(raw_targets) > 1_000_000:
            raise ValueError("adapter target_modules regex must be bounded and anchored")
        try:
            target_regex = re.compile(raw_targets)
        except re.error as exc:
            raise ValueError(f"adapter target_modules regex is invalid: {exc}") from exc
        targets = _exact_targets_from_regex(raw_targets)
    else:
        raise ValueError(
            "adapter_config.json target_modules must be a non-empty list or anchored regex"
        )
    configured_base = config.get("base_model_name_or_path")
    if configured_base is not None:
        configured_name = Path(_non_empty_text(configured_base, "base_model_name_or_path")).name
        if configured_name != Path(base_model_name).name:
            raise ValueError(
                "base_model_name does not match adapter_config.json: "
                f"{base_model_name!r} != {configured_base!r}"
            )
    return rank, targets, target_regex


def build_manifest(
    adapter_dir: str | Path,
    base_model_name: str,
    *,
    run_manifest: str | Path | None = None,
    training_config: str | Path | None = None,
    base_model_config: str | Path | None = None,
) -> dict[str, object]:
    """Build a verified adapter manifest.

    The original two positional arguments and schema-v1 keys remain intact.
    Optional provenance inputs add fields without changing AdapterRegistry's
    existing contract.
    """

    requested = Path(adapter_dir).expanduser()
    if requested.is_symlink():
        raise ValueError("adapter directory cannot be a symlink")
    directory = requested.resolve()
    if not directory.is_dir():
        raise ValueError(f"adapter directory does not exist: {directory}")
    model_name = _non_empty_text(base_model_name, "base_model_name")
    config_path = directory / "adapter_config.json"
    adapter_config = _read_object(config_path, "adapter config")
    rank, targets, target_regex = _adapter_metadata(adapter_config, model_name)
    adapter_target_expression = adapter_config["target_modules"]

    forbidden = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
        and (
            path.name in _FORBIDDEN_BASE_FILES
            or _FORBIDDEN_BASE_SHARD_RE.fullmatch(path.name)
        )
    )
    if forbidden:
        raise ValueError(
            "adapter directory contains forbidden base weights: " + ", ".join(forbidden)
        )
    weights = tuple(sorted(directory.glob("adapter_model*.safetensors")))
    if not weights:
        raise ValueError("trained adapter_model*.safetensors weights are required")
    for path in weights:
        _validate_safetensors(path)
    files = (config_path, *weights)

    payload: dict[str, object] = {
        "schema_version": 1,
        "base_model_name": model_name,
        "adapter_dir": str(directory),
        "files": {path.name: _hash(path) for path in files},
        "rank": rank,
        "target_modules": targets,
        "adapter_target_expression": adapter_target_expression,
    }

    run: dict[str, object] | None = None
    if run_manifest is not None:
        run_path = _resolved_regular_file(run_manifest, "run manifest")
        run = _read_object(run_path, "run manifest")
        run_id = _non_empty_text(run.get("run_id"), "run_manifest.run_id")
        run_base = _non_empty_text(run.get("base_model_name"), "run_manifest.base_model_name")
        if Path(run_base).name != Path(model_name).name:
            raise ValueError("run manifest base_model_name does not match adapter")
        run_adapter = Path(
            _non_empty_text(run.get("final_adapter_path"), "run_manifest.final_adapter_path")
        ).expanduser().resolve()
        if run_adapter != directory:
            raise ValueError("run manifest final_adapter_path does not match adapter_dir")
        run_rank = run.get("rank")
        if run_rank != rank:
            raise ValueError("run manifest rank does not match adapter_config.json")
        run_targets = run.get("target_modules")
        if not isinstance(run_targets, list) or not run_targets or any(
            not isinstance(item, str) or not item for item in run_targets
        ):
            raise ValueError("run manifest target_modules must be a non-empty string list")
        expanded_targets = sorted(run_targets)
        if len(set(expanded_targets)) != len(expanded_targets):
            raise ValueError("run manifest target_modules contains duplicates")
        if target_regex is None:
            if expanded_targets != targets:
                raise ValueError(
                    "run manifest target_modules do not match adapter_config.json"
                )
        elif expanded_targets != targets or any(
            target_regex.fullmatch(item) is None for item in expanded_targets
        ):
            raise ValueError(
                "run manifest target_modules do not match the adapter target regex"
            )
        targets = expanded_targets
        payload["target_modules"] = targets
        run_expression = run.get("adapter_target_expression")
        if run_expression is not None and run_expression != adapter_target_expression:
            raise ValueError(
                "run manifest adapter_target_expression does not match adapter_config.json"
            )
        payload["training_run_id"] = run_id
        payload["dataset_manifest_sha256"] = _sha256_text(
            run.get("dataset_manifest_sha256"), "run_manifest.dataset_manifest_sha256"
        )

    if training_config is not None:
        training_path = _resolved_regular_file(training_config, "training config")
        training_hash = _hash(training_path)
        if run is not None and run.get("training_config_sha256") not in {None, training_hash}:
            raise ValueError("run manifest training_config_sha256 does not match training config")
        payload["training_config_sha256"] = training_hash
    elif run is not None and run.get("training_config_sha256") is not None:
        payload["training_config_sha256"] = _sha256_text(
            run["training_config_sha256"], "run_manifest.training_config_sha256"
        )

    if base_model_config is not None:
        base_path = _resolved_regular_file(base_model_config, "base model config")
        base_hash = _hash(base_path)
        if run is not None and run.get("base_model_config_sha256") not in {None, base_hash}:
            raise ValueError("run manifest base_model_config_sha256 does not match base config")
        payload["base_model_config_sha256"] = base_hash
    elif run is not None and run.get("base_model_config_sha256") is not None:
        payload["base_model_config_sha256"] = _sha256_text(
            run["base_model_config_sha256"], "run_manifest.base_model_config_sha256"
        )
    return payload


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise ValueError("manifest output cannot be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(payload), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--base-model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--base-model-config", type=Path)
    args = parser.parse_args(argv)
    try:
        payload = build_manifest(
            args.adapter_dir,
            args.base_model_name,
            run_manifest=args.run_manifest,
            training_config=args.training_config,
            base_model_config=args.base_model_config,
        )
        requested_output = args.output.expanduser()
        if requested_output.is_symlink():
            raise ValueError("manifest output cannot be a symlink")
        output = requested_output.resolve()
        adapter_root = Path(str(payload["adapter_dir"]))
        if output.parent == adapter_root and output.name in payload["files"]:
            raise ValueError("manifest output cannot overwrite an adapter artifact")
        _atomic_write(output, payload)
        print(output)
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"adapter manifest error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
