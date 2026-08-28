from __future__ import annotations

from pathlib import Path

import pytest

from scripts.collect_target_state_dataset import (
    DEFAULT_BRIDGE_ROOT,
    DEFAULT_COLLECTION_SESSION_DIR,
    DEFAULT_PC_TRANS_CONFIG,
    DEFAULT_PC_TRANS_ROOT,
    _parser,
    _validate_storage_arguments,
)


def _arguments(*extra: str):
    return _parser().parse_args(
        [
            "--oracle-label-generation",
            "--acknowledge-privileged-oracle",
            *extra,
        ]
    )


def test_local_storage_remains_default_and_requires_output() -> None:
    args = _arguments()
    assert args.storage_mode == "local"
    with pytest.raises(ValueError, match="--output is required"):
        _validate_storage_arguments(args)

    args = _arguments("--output", "dataset")
    _validate_storage_arguments(args)
    assert args.output == Path("dataset")


def test_collection_spool_defaults_match_server_bridge_contract() -> None:
    args = _arguments("--storage-mode", "collection-spool")
    _validate_storage_arguments(args)
    assert args.pc_trans_root == DEFAULT_PC_TRANS_ROOT
    assert args.pc_trans_config == DEFAULT_PC_TRANS_CONFIG
    assert args.bridge_root == DEFAULT_BRIDGE_ROOT
    assert args.collection_session_dir == DEFAULT_COLLECTION_SESSION_DIR
    assert args.collection_shard_size_mib == 512.0


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--resume-session", "session"), "only with --storage-mode"),
        (
            ("--storage-mode", "collection-spool", "--mode", "external"),
            "requires --mode isaac",
        ),
        (
            ("--storage-mode", "collection-spool", "--output", "dataset"),
            "only for local datasets",
        ),
        (
            (
                "--storage-mode",
                "collection-spool",
                "--collection-shard-size-mib",
                "0",
            ),
            "must be positive and finite",
        ),
    ],
)
def test_storage_mode_rejects_ambiguous_or_invalid_combinations(
    extra: tuple[str, ...], message: str
) -> None:
    args = _arguments(*extra)
    with pytest.raises(ValueError, match=message):
        _validate_storage_arguments(args)

