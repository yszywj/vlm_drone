#!/usr/bin/env python3
"""Export provenance only for a real, already-trained adapter directory."""

from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(adapter_dir: str | Path, base_model_name: str) -> dict[str, object]:
    directory = Path(adapter_dir).expanduser().resolve()
    config = directory / "adapter_config.json"
    weights = sorted(directory.glob("*.safetensors"))
    if not config.is_file() or not weights:
        raise ValueError(
            "real adapter_config.json and trained .safetensors are required; "
            "placeholder adapters cannot export a manifest"
        )
    files = [config, *weights]
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("adapter manifest inputs must be regular non-symlink files")
    return {
        "schema_version": 1,
        "base_model_name": base_model_name,
        "adapter_dir": str(directory),
        "files": {path.name: _hash(path) for path in files},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--base-model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = build_manifest(args.adapter_dir, args.base_model_name)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.output)
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"adapter manifest error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
