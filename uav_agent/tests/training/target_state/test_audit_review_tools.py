from pathlib import Path

import pytest

from scripts.inspect_target_state_episode import export_episode
from scripts.review_target_state_audit import regate_rows, covariance_stats
from tests.training.target_state.test_shards import _write_parent_dataset
from training.target_state.shards import build_target_state_shards


def test_saved_output_review_blocks_false_recovery_without_using_labels():
    row = dict(detected=False, raw_depth_m=0.0, corrected_depth_m=0.47,
               model_geometry_valid=True, model_valid=True, baseline_valid=False,
               validity_probability=0.995)
    revised = regate_rows([row], minimum=0.2, maximum=200.0)[0]
    assert revised["model_valid"] is False
    assert row["model_valid"] is True  # original evidence is immutable
    row.update(detected=True, raw_depth_m=4.0, corrected_depth_m=4.2)
    assert regate_rows([row], minimum=0.2, maximum=200.0)[0]["model_valid"] is True


def test_covariance_scale_changes_coverage_not_positions():
    rows = [dict(evaluated_visible_target=True, model_valid=True,
                 model_position_world_m=[1.0, 0.0, 0.0], target_position_world_m=[0.0, 0.0, 0.0],
                 position_variance_m2=[0.1, 0.1, 0.1])]
    original = covariance_stats(rows)
    revised = covariance_stats(rows, scale=2.0)
    assert original["coverage_95_percent_ellipsoid"] == 0.0
    assert revised["coverage_95_percent_ellipsoid"] == 1.0
    assert revised["q95_normalized_squared_error"] == pytest.approx(original["q95_normalized_squared_error"] / 2)
    assert rows[0]["position_variance_m2"] == [0.1, 0.1, 0.1]


def test_episode_export_preserves_original_assets_and_archive(tmp_path: Path):
    parent = tmp_path / "parent"
    _write_parent_dataset(parent)
    index = build_target_state_shards(parent, tmp_path / "shards",
        target_shard_size_bytes=1, history_size=4, max_history_age_s=2.0,
        split_seed=42).shard_index
    entry = index.shards_for_split("test")[0]
    archive = tmp_path / "shards" / entry.filename
    destination = tmp_path / "review"
    result = export_episode(archive, index=index, episode_id=entry.episode_ids[0], output_dir=destination)
    assert result["offline_only"] is True
    assert archive.is_file()
    for relative in result["asset_sha256"]:
        assert (destination / "assets" / relative).read_bytes() == (parent / relative).read_bytes()
    assert not (destination / ".materialized" / archive.stem).exists()
    assert export_episode(archive, index=index, episode_id=entry.episode_ids[0], output_dir=destination) == result
    corrupted = destination / "assets" / next(iter(result["asset_sha256"]))
    corrupted.write_bytes(b"user changed this file")
    with pytest.raises(ValueError, match="existing review asset differs"):
        export_episode(archive, index=index, episode_id=entry.episode_ids[0], output_dir=destination)
    assert corrupted.read_bytes() == b"user changed this file"
    assert archive.is_file()
