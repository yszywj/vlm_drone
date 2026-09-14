from types import SimpleNamespace

import pytest

from training.lora.train_planning_role_lora import EncodedRoleDataset, prepare_role, import_huggingface_datasets


def test_test_split_is_never_accepted_as_training_or_validation():
    for train, validation in (("test", "validation"), ("train", "test")):
        with pytest.raises(ValueError, match="test is held out"):
            prepare_role(SimpleNamespace(train_split=train, validation_split=validation), "fleet_planner")


def test_invalid_role_fails_before_loading_tokenizer():
    with pytest.raises(ValueError, match="unsupported role"):
        prepare_role(None, "runtime_visual")


def test_prepared_training_features_exclude_metadata():
    class Source:
        manifest_sha256 = "a" * 64
        role = "mission_interpreter"
        split = "train"
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return {"sample_id": f"sample_{index}", "private_metadata": "excluded"}
    class Collator:
        def encode_feature(self, row):
            return {"input_ids": [10, 11, 12], "attention_mask": [1, 1, 1], "labels": [-100, 11, 12]}
    prepared = EncodedRoleDataset(Source(), Collator())
    assert prepared.manifest_sha256 == "a" * 64
    assert set(prepared[0]) == {"input_ids", "attention_mask", "labels"}
    assert prepared.sample_ids == ["sample_0", "sample_1"]
    assert prepared.summary() == {"count": 2, "max_full_tokens": 3, "total_full_tokens": 6, "supervised_tokens": 4}


def test_training_bootstrap_loads_installed_distribution_not_local_namesake(tmp_path, monkeypatch):
    import importlib.metadata
    import sys
    installed = tmp_path / "installed" / "datasets"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("class Dataset: pass\n")
    local = tmp_path / "project" / "datasets"
    local.mkdir(parents=True)
    (local / "__init__.py").write_text("local_target_state = True\n")
    monkeypatch.syspath_prepend(str(local.parent))
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: SimpleNamespace(locate_file=lambda value: installed.parent / value))
    try:
        module = import_huggingface_datasets()
        assert hasattr(module, "Dataset")
        assert not hasattr(module, "local_target_state")
        assert import_huggingface_datasets() is module
    finally:
        sys.modules.pop("datasets", None)
