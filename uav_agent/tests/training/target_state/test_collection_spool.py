"""Episode-boundary and crash-recovery tests for collection spooling."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import tarfile

import numpy as np
import pytest

from tests.training.target_state.test_dataset_schema import make_record
import training.target_state.collection_spool as collection_spool_module
from training.target_state.collection_spool import (
    CollectionSpoolError,
    TargetStateCollectionSpool,
)


_MODEL_SHA256 = "a" * 64


def _bridge_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    """Create only the filesystem contract required by the producer."""

    pc_trans_root = root / "pc_trans"
    (pc_trans_root / "pc_trans").mkdir(parents=True)
    (pc_trans_root / "pc_trans" / "cli.py").write_text(
        "# fake pc_trans CLI for an injected seal callback\n", encoding="utf-8"
    )
    bridge_root = root / "bridge"
    config = root / "pc_trans.json"
    config.write_text(
        json.dumps({"bridge_root": str(bridge_root)}) + "\n", encoding="utf-8"
    )
    sessions = root / "sessions"
    return pc_trans_root, config, bridge_root, sessions


def _moving_sealer(bridge_root: Path, calls: list[Path]):
    def seal(source: Path) -> None:
        calls.append(source)
        ready = bridge_root / "collection_spool" / "ready" / source.name.removesuffix(
            ".tmp"
        )
        os.replace(source, ready)

    return seal


def _create_spool(
    root: Path,
    *,
    collection_id: str,
    target_bytes: int,
    seal_callback,
    sleep_callback=lambda _seconds: None,
) -> tuple[TargetStateCollectionSpool, Path, Path, Path, Path]:
    pc_trans_root, config, bridge_root, sessions = _bridge_fixture(root)
    spool = TargetStateCollectionSpool.create(
        pc_trans_root=pc_trans_root,
        pc_trans_config=config,
        bridge_root=bridge_root,
        collection_session_dir=sessions,
        shard_target_size_bytes=target_bytes,
        scene_seed=17,
        split_seed=42,
        history_size=4,
        max_history_age_s=2.0,
        yolo_model_sha256=_MODEL_SHA256,
        generation_commit_sha="testcommit",
        collection_id=collection_id,
        poll_interval_s=0.001,
        seal_callback=seal_callback,
        sleep_callback=sleep_callback,
    )
    return spool, pc_trans_root, config, bridge_root, sessions


def _resume_spool(
    session: Path,
    *,
    pc_trans_root: Path,
    config: Path,
    bridge_root: Path,
    target_bytes: int,
    seal_callback,
) -> TargetStateCollectionSpool:
    return TargetStateCollectionSpool.resume(
        session,
        pc_trans_root=pc_trans_root,
        pc_trans_config=config,
        bridge_root=bridge_root,
        shard_target_size_bytes=target_bytes,
        scene_seed=17,
        split_seed=42,
        history_size=4,
        max_history_age_s=2.0,
        yolo_model_sha256=_MODEL_SHA256,
        poll_interval_s=0.001,
        seal_callback=seal_callback,
        sleep_callback=lambda _seconds: None,
    )


def _append_capture(
    spool: TargetStateCollectionSpool,
    *,
    episode_id: str,
    capture_index: int,
    target_count: int = 1,
) -> None:
    """Append one physical image that may yield multiple target records."""

    records = []
    for target_index in range(target_count):
        record = make_record(
            capture_index * 10 + target_index,
            episode_id=episode_id,
            assignment_id=f"assignment_{episode_id}",
            candidate_id=f"candidate_{episode_id}_{target_index}",
            tracker_id=f"tracker_{episode_id}_{target_index}",
            instance_id=f"cube_{episode_id}_{target_index}",
        )
        records.append(
            replace(
                record,
                frame_id=f"frame_{episode_id}_{capture_index}_{target_index}",
                timestamp_s=capture_index * 0.2,
            )
        )
    spool.append_capture(
        records,
        rgb=np.full((24, 32, 3), 90 + capture_index, dtype=np.uint8),
        depth_m=np.full((24, 32), 5.0 + capture_index, dtype=np.float32),
        asset_id=f"capture_{episode_id}_{capture_index}",
    )


def _tar_json(path: Path, name: str) -> dict[str, object]:
    with tarfile.open(path, "r:") as archive:
        stream = archive.extractfile(name)
        assert stream is not None
        value = json.load(stream)
    assert isinstance(value, dict)
    return value


def _journal_entries(session: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (session / "shards.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_soft_target_is_checked_only_after_episode_and_manifest_counts_captures(
    tmp_path: Path,
) -> None:
    calls: list[Path] = []
    root = tmp_path / "soft_split"
    # One record already exceeds this soft target.  Publication must still wait
    # for complete_episode(), and each resulting tar must contain a whole episode.
    spool, _pc_root, _config, bridge, _sessions = _create_spool(
        root,
        collection_id="collection_soft",
        target_bytes=1,
        seal_callback=_moving_sealer(root / "bridge", calls),
    )

    spool.begin_episode("episode_0", episode_index=0)
    _append_capture(spool, episode_id="episode_0", capture_index=0, target_count=2)
    assert calls == []
    assert spool.complete_episode() is True

    spool.begin_episode("episode_1", episode_index=1)
    _append_capture(spool, episode_id="episode_1", capture_index=0)
    assert len(calls) == 1
    assert spool.complete_episode() is True
    result = spool.finalize()

    assert result.shard_count == 2
    assert result.episode_count == 2
    assert result.physical_capture_count == 2
    assert result.record_count == 3
    ready = bridge / "collection_spool" / "ready"
    manifests = [
        _tar_json(path, "collection_manifest.json")
        for path in sorted(ready.glob("*.tar"))
    ]
    assert [item["episode_ids"] for item in manifests] == [
        ["episode_0"],
        ["episode_1"],
    ]
    assert manifests[0]["physical_capture_count"] == 1
    assert manifests[0]["record_count"] == 2
    assert manifests[0]["physical_capture_count"] != manifests[0]["record_count"]
    assert all(item["episode_count"] == 1 for item in manifests)


def test_pause_blocks_next_episode_and_mid_episode_pause_forces_boundary_seal(
    tmp_path: Path,
) -> None:
    calls: list[Path] = []
    root = tmp_path / "pause"
    pause_flag = root / "bridge" / "control" / "pause_collection.flag"
    paused_states: list[str] = []

    def release_pause(_seconds: float) -> None:
        session = root / "sessions" / "collection_pause" / "session.json"
        paused_states.append(json.loads(session.read_text(encoding="utf-8"))["status"])
        pause_flag.unlink()

    spool, _pc_root, _config, bridge, _sessions = _create_spool(
        root,
        collection_id="collection_pause",
        target_bytes=1024**3,
        seal_callback=_moving_sealer(root / "bridge", calls),
        sleep_callback=release_pause,
    )
    pause_flag.write_text("quota\n", encoding="utf-8")
    with pytest.raises(CollectionSpoolError, match="wait_before_episode"):
        spool.begin_episode("episode_0", episode_index=0)
    spool.wait_before_episode()
    assert paused_states == ["paused"]
    assert json.loads(spool.session_path.read_text(encoding="utf-8"))["status"] == "collecting"

    spool.begin_episode("episode_0", episode_index=0)
    _append_capture(spool, episode_id="episode_0", capture_index=0)
    pause_flag.write_text("network_down\n", encoding="utf-8")
    # The current episode is complete and durable; pause merely forces a shard
    # boundary before another episode may begin.
    assert calls == []
    assert spool.complete_episode() is True
    assert len(calls) == 1
    assert len(list((bridge / "collection_spool" / "ready").glob("*.tar"))) == 1


def test_failed_seal_retains_tar_workspace_and_resume_commits_once(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pending"

    def fail_seal(_source: Path) -> None:
        raise ConnectionError("simulated transfer service outage")

    spool, pc_root, config, bridge, _sessions = _create_spool(
        root,
        collection_id="collection_pending",
        target_bytes=1,
        seal_callback=fail_seal,
    )
    spool.begin_episode("episode_0", episode_index=0)
    _append_capture(spool, episode_id="episode_0", capture_index=0)
    with pytest.raises(ConnectionError, match="outage"):
        spool.complete_episode()
    spool.abort()

    session = spool.session_dir
    pending = json.loads((session / "session.json").read_text(encoding="utf-8"))[
        "pending_shard"
    ]
    assert pending["phase"] == "ready_to_seal"
    tar_tmp = Path(pending["tar_tmp_path"])
    workspace = Path(pending["workspace_path"])
    assert tar_tmp.is_file()
    assert workspace.is_dir()
    assert _journal_entries(session) == []

    calls: list[Path] = []
    resumed = _resume_spool(
        session,
        pc_trans_root=pc_root,
        config=config,
        bridge_root=bridge,
        target_bytes=1,
        seal_callback=_moving_sealer(bridge, calls),
    )
    assert resumed.next_episode_index == 1
    assert resumed.shard_count == 1
    assert len(_journal_entries(session)) == 1
    assert calls == [tar_tmp]
    assert not workspace.exists()
    assert not tar_tmp.exists()
    assert (bridge / "collection_spool" / "ready" / tar_tmp.name.removesuffix(".tmp")).is_file()


def test_resume_quarantines_unsealed_partial_and_recollects_boundary_without_duplicates(
    tmp_path: Path,
) -> None:
    calls: list[Path] = []
    root = tmp_path / "partial"
    spool, pc_root, config, bridge, _sessions = _create_spool(
        root,
        collection_id="collection_partial",
        target_bytes=1024**3,
        seal_callback=_moving_sealer(root / "bridge", calls),
    )
    spool.begin_episode("episode_0", episode_index=0)
    _append_capture(spool, episode_id="episode_0", capture_index=0)
    assert spool.complete_episode() is False
    partial_workspace = spool.session_dir / "workspaces" / "shard_000001"
    spool.abort()
    assert partial_workspace.is_dir()
    assert _journal_entries(spool.session_dir) == []

    resumed = _resume_spool(
        spool.session_dir,
        pc_trans_root=pc_root,
        config=config,
        bridge_root=bridge,
        target_bytes=1024**3,
        seal_callback=_moving_sealer(bridge, calls),
    )
    assert resumed.next_episode_index == 0
    recovered = list((resumed.session_dir / "recovery").iterdir())
    assert len(recovered) == 1
    assert recovered[0].name.startswith("shard_000001.partial.")

    resumed.begin_episode("episode_0", episode_index=0)
    _append_capture(resumed, episode_id="episode_0", capture_index=0)
    assert resumed.complete_episode() is False
    result = resumed.finalize()
    entries = _journal_entries(resumed.session_dir)
    assert result.episode_count == 1
    assert result.shard_count == 1
    assert len(entries) == 1
    assert entries[0]["episode_ids"] == ["episode_0"]


def test_bridge_and_resume_config_identity_mismatches_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "identity"
    pc_root, config, bridge, sessions = _bridge_fixture(root)
    other_bridge = root / "other_bridge"
    with pytest.raises(CollectionSpoolError, match="exactly match"):
        TargetStateCollectionSpool.create(
            pc_trans_root=pc_root,
            pc_trans_config=config,
            bridge_root=other_bridge,
            collection_session_dir=sessions,
            shard_target_size_bytes=1024,
            scene_seed=17,
            history_size=4,
            yolo_model_sha256=_MODEL_SHA256,
        )

    calls: list[Path] = []
    spool = TargetStateCollectionSpool.create(
        pc_trans_root=pc_root,
        pc_trans_config=config,
        bridge_root=bridge,
        collection_session_dir=sessions,
        shard_target_size_bytes=1024,
        scene_seed=17,
        history_size=4,
        yolo_model_sha256=_MODEL_SHA256,
        collection_id="collection_identity",
        seal_callback=_moving_sealer(bridge, calls),
    )
    spool.abort()
    copied_config = root / "copied_config.json"
    copied_config.write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(CollectionSpoolError, match="pc_trans_config"):
        _resume_spool(
            spool.session_dir,
            pc_trans_root=pc_root,
            config=copied_config,
            bridge_root=bridge,
            target_bytes=1024,
            seal_callback=_moving_sealer(bridge, calls),
        )


def test_building_tar_resume_quarantines_truncation_and_rebuilds_from_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "building_tar"
    calls: list[Path] = []
    spool, pc_root, config, bridge, _sessions = _create_spool(
        root,
        collection_id="collection_building",
        target_bytes=1024**3,
        seal_callback=_moving_sealer(root / "bridge", calls),
    )
    spool.begin_episode("episode_0", episode_index=0)
    _append_capture(spool, episode_id="episode_0", capture_index=0)
    assert spool.complete_episode() is False

    original_writer = collection_spool_module._write_deterministic_tar
    truncated = b"this is not a tar archive"

    def crash_during_tar(_workspace: Path, target: Path) -> None:
        target.write_bytes(truncated)
        raise OSError("simulated crash while writing tar")

    monkeypatch.setattr(
        collection_spool_module, "_write_deterministic_tar", crash_during_tar
    )
    with pytest.raises(OSError, match="simulated crash"):
        spool.seal_current_shard()
    spool.abort()
    pending = json.loads(spool.session_path.read_text(encoding="utf-8"))[
        "pending_shard"
    ]
    assert pending["phase"] == "building_tar"
    tar_tmp = Path(pending["tar_tmp_path"])
    workspace = Path(pending["workspace_path"])
    assert tar_tmp.read_bytes() == truncated
    assert workspace.is_dir()

    monkeypatch.setattr(
        collection_spool_module, "_write_deterministic_tar", original_writer
    )
    resumed = _resume_spool(
        spool.session_dir,
        pc_trans_root=pc_root,
        config=config,
        bridge_root=bridge,
        target_bytes=1024**3,
        seal_callback=_moving_sealer(bridge, calls),
    )
    assert resumed.shard_count == 1
    assert calls == [tar_tmp]
    quarantined = list(
        tar_tmp.parent.glob(f".{tar_tmp.name}.incomplete.*")
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == truncated
    ready = bridge / "collection_spool" / "ready" / tar_tmp.name.removesuffix(".tmp")
    assert _tar_json(ready, "collection_manifest.json")["episode_ids"] == [
        "episode_0"
    ]
    assert not workspace.exists()
    resumed.abort()


def test_ready_to_seal_resume_rebuilds_and_reseals_when_both_copies_disappear(
    tmp_path: Path,
) -> None:
    root = tmp_path / "missing_pending_copies"

    def unavailable(_source: Path) -> None:
        raise ConnectionError("seal acknowledgement was lost")

    spool, pc_root, config, bridge, _sessions = _create_spool(
        root,
        collection_id="collection_missing",
        target_bytes=1,
        seal_callback=unavailable,
    )
    spool.begin_episode("episode_0", episode_index=0)
    _append_capture(spool, episode_id="episode_0", capture_index=0)
    with pytest.raises(ConnectionError, match="acknowledgement"):
        spool.complete_episode()
    spool.abort()
    pending = json.loads(spool.session_path.read_text(encoding="utf-8"))[
        "pending_shard"
    ]
    assert pending["phase"] == "ready_to_seal"
    tar_tmp = Path(pending["tar_tmp_path"])
    workspace = Path(pending["workspace_path"])
    expected_sha = pending["entry"]["archive_sha256"]
    ready = bridge / "collection_spool" / "ready" / pending["entry"]["filename"]
    tar_tmp.unlink()
    assert not tar_tmp.exists()
    assert not ready.exists()
    assert workspace.is_dir()

    calls: list[Path] = []

    def verify_then_seal(source: Path) -> None:
        # A disappearing archive is not treated as a transfer acknowledgement:
        # resume must reconstruct a complete tar before invoking seal again.
        assert _tar_json(source, "collection_manifest.json")["episode_ids"] == [
            "episode_0"
        ]
        calls.append(source)
        os.replace(source, ready)

    resumed = _resume_spool(
        spool.session_dir,
        pc_trans_root=pc_root,
        config=config,
        bridge_root=bridge,
        target_bytes=1,
        seal_callback=verify_then_seal,
    )
    assert calls == [tar_tmp]
    assert resumed.shard_count == 1
    assert resumed.next_episode_index == 1
    assert not workspace.exists()
    assert _journal_entries(resumed.session_dir)[0]["archive_sha256"] == expected_sha
    assert ready.is_file()
    resumed.abort()


def test_public_identifiers_reject_sixty_five_characters(tmp_path: Path) -> None:
    root = tmp_path / "identifier_bounds"
    pc_root, config, bridge, sessions = _bridge_fixture(root)
    with pytest.raises(CollectionSpoolError, match="bounded identifier"):
        TargetStateCollectionSpool.create(
            pc_trans_root=pc_root,
            pc_trans_config=config,
            bridge_root=bridge,
            collection_session_dir=sessions,
            shard_target_size_bytes=1024,
            scene_seed=17,
            history_size=4,
            yolo_model_sha256=_MODEL_SHA256,
            collection_id="c" * 65,
        )

    spool = TargetStateCollectionSpool.create(
        pc_trans_root=pc_root,
        pc_trans_config=config,
        bridge_root=bridge,
        collection_session_dir=sessions,
        shard_target_size_bytes=1024,
        scene_seed=17,
        history_size=4,
        yolo_model_sha256=_MODEL_SHA256,
        collection_id="collection_identifier",
        seal_callback=lambda _source: None,
    )
    with pytest.raises(CollectionSpoolError, match="bounded identifier"):
        spool.begin_episode("e" * 65, episode_index=0)
    spool.abort()


def test_second_concurrent_resume_of_same_session_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "session_lock"
    calls: list[Path] = []
    spool, pc_root, config, bridge, _sessions = _create_spool(
        root,
        collection_id="collection_locked",
        target_bytes=1024,
        seal_callback=_moving_sealer(root / "bridge", calls),
    )
    session = spool.session_dir
    spool.abort()

    first = _resume_spool(
        session,
        pc_trans_root=pc_root,
        config=config,
        bridge_root=bridge,
        target_bytes=1024,
        seal_callback=_moving_sealer(bridge, calls),
    )
    with pytest.raises(CollectionSpoolError, match="already owned"):
        _resume_spool(
            session,
            pc_trans_root=pc_root,
            config=config,
            bridge_root=bridge,
            target_bytes=1024,
            seal_callback=_moving_sealer(bridge, calls),
        )
    first.abort()
