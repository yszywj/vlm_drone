from copy import deepcopy
import shutil

import pytest

from planning_data import generator
from scripts import generate_planning_roles_curriculum as curriculum
from training.lora.roles_dataset import PlanningRoleSFTDataset


def _parent(tmp_path, count=25):
    parent = tmp_path / "parent"
    manifest = generator.generate_dataset(parent, count=count, tokenizer_path=None)
    snapshot = tmp_path / "snapshot"
    for relative in manifest["source_sha256"]:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generator.ROOT / relative, destination)
    shutil.copyfile(parent / "manifest.json", tmp_path / "v1_dataset_manifest.json")
    return parent, snapshot


def _generate(parent, snapshot, output, **kwargs):
    return curriculum.generate_curriculum(
        parent, output, parent_source_snapshot=snapshot,
        train_per_focus_scale=1, validation_per_focus_scale=1,
        test_per_focus_scale=1, tokenizer_path=None, **kwargs,
    )


def test_curriculum_replays_all_roles_preserves_parent_and_split_ownership(tmp_path):
    parent, snapshot = _parent(tmp_path)
    parent_bytes = {str(path.relative_to(parent)): path.read_bytes() for path in parent.rglob("*") if path.is_file()}
    output = tmp_path / "curriculum"
    manifest = _generate(parent, snapshot, output)
    assert manifest["underlying_tasks"] == 100
    assert manifest["total_samples"] == 800
    assert manifest["curriculum"]["targeted_split_counts"] == {"train": 25, "validation": 25, "test": 25}
    assert len(manifest["curriculum"]["new_held_out_test_task_ids"]) == 25
    assert set(manifest["source_sha256"]) == set(generator._source_hashes())
    assert manifest["curriculum_source_sha256"]["scripts/generate_planning_roles_curriculum.py"]
    assert curriculum.validate_curriculum(output)["all_persisted_answers_replayed"]
    for role in generator.ROLES:
        for split in generator.SPLITS:
            dataset = PlanningRoleSFTDataset(output, role=role, split=split)
            assert len(dataset) == manifest["role_split_counts"][role][split]
    assert {str(path.relative_to(parent)): path.read_bytes() for path in parent.rglob("*") if path.is_file()} == parent_bytes
    tasks = curriculum.read_tasks(output / "tasks.jsonl")
    old_tasks = curriculum.read_tasks(parent / "tasks.jsonl")
    assert tasks[:len(old_tasks)] == old_tasks
    assert all(task["split"] == task["curriculum_partition"] for task in tasks[len(old_tasks):])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _generate(parent, snapshot, output)


@pytest.mark.parametrize("tamper", ["manifest", "data", "source"])
def test_parent_integrity_rejects_tampering_before_output_exists(tmp_path, tamper):
    parent, snapshot = _parent(tmp_path, count=1)
    if tamper == "manifest":
        path, match = parent / "manifest.json", "manifest differs"
    elif tamper == "data":
        path, match = parent / "tasks.jsonl", "JSONL hashes differ"
    else:
        path, match = snapshot / "planning_data/tasks.py", "source snapshot hash mismatch"
    path.write_text(path.read_text() + "\n")
    output = tmp_path / "curriculum"
    with pytest.raises(ValueError, match=match):
        _generate(parent, snapshot, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".curriculum.staging-*"))


def test_duplicate_task_identity_cannot_be_hidden_by_new_ids_or_split(tmp_path):
    parent, snapshot = _parent(tmp_path, count=1)
    _, tasks = curriculum.verify_parent(parent, snapshot)
    duplicate = deepcopy(tasks[0])
    duplicate.update(task_id="new_id", split="test")
    with pytest.raises(ValueError, match="duplicate task semantic_hash"):
        curriculum._check_task_identity(tasks + [duplicate])


def test_failed_current_production_replay_never_publishes_partial_dataset(tmp_path, monkeypatch):
    parent, snapshot = _parent(tmp_path, count=1)
    output = tmp_path / "curriculum"
    def reject(task):
        raise ValueError("current contract rejected label")
    monkeypatch.setattr(generator, "render_task", reject)
    with pytest.raises(ValueError, match="current contract rejected label"):
        _generate(parent, snapshot, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".curriculum.staging-*"))


def test_partition_audit_rejects_moving_a_training_template_to_test():
    tasks = curriculum._build_targeted(
        [], seed=81, train_per_focus_scale=1,
        validation_per_focus_scale=1, test_per_focus_scale=1,
    )
    for task in tasks:
        if task["split"] == "train":
            task["curriculum_partition"] = "test"
            break
    with pytest.raises(ValueError, match="template partition differs"):
        curriculum._audit_curriculum_partitions(tasks, {
            "train_per_focus_scale": 1, "validation_per_focus_scale": 1, "test_per_focus_scale": 1,
        })


def test_token_limits_are_reported_and_oversize_data_is_not_truncated(tmp_path, monkeypatch):
    parent, snapshot = _parent(tmp_path, count=1)
    class TinyTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "prompt" if len(messages) == 2 else "promptanswer"
        def encode(self, text, **kwargs):
            return list(text)
    monkeypatch.setattr(generator, "load_tokenizer", lambda path: TinyTokenizer())
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    output = tmp_path / "length_audited"
    manifest = curriculum.generate_curriculum(
        parent, output, parent_source_snapshot=snapshot,
        train_per_focus_scale=1, validation_per_focus_scale=1, test_per_focus_scale=1,
        tokenizer_path=tokenizer_path, max_length=20, training_max_length=10,
    )
    for role in generator.ROLES:
        for split in generator.SPLITS:
            check = manifest["training_length_audit"]["role_split_counts"][role][split]
            assert check["count"] == check["exceeds_training_max_length"]
            assert check["max_full_tokens"] == 12
    assert curriculum.validate_curriculum(output)["token_lengths_rechecked"]
    too_short = tmp_path / "too_short"
    with pytest.raises(ValueError, match="refusing truncation"):
        curriculum.generate_curriculum(
            parent, too_short, parent_source_snapshot=snapshot,
            train_per_focus_scale=1, validation_per_focus_scale=1, test_per_focus_scale=1,
            tokenizer_path=tokenizer_path, max_length=10,
        )
    assert not too_short.exists()
    assert not list(tmp_path.glob(".too_short.staging-*"))
