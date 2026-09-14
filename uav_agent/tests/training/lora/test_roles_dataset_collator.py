from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest

from planning_data.generator import generate_dataset
from training.lora.collator import AssistantOnlyDataCollator, FleetPlannerCollatorError, IGNORE_INDEX
from training.lora.roles_dataset import PlanningRoleSFTDataset, PlanningRoleSFTDatasetError


@pytest.fixture(scope="module")
def role_data(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("roles") / "planning_roles"
    generate_dataset(root, count=25, tokenizer_path=None)
    return root


@pytest.mark.parametrize("role,count", [("mission_interpreter", 20), ("fleet_planner", 20), ("spatial_mission", 120)])
def test_loader_preserves_each_production_conversation(role_data, role, count):
    data = PlanningRoleSFTDataset(role_data, role=role)
    original = json.loads((role_data / role / "train.jsonl").read_text().splitlines()[0])
    assert len(data) == data.count == count
    assert data[0]["messages"] == original["messages"]
    assert data[0]["assistant_json"] == original["messages"][-1]["content"]
    assert data[0]["target"] == json.loads(original["messages"][-1]["content"])
    changed = data[0]
    changed["messages"][-1]["content"] = "{}"
    assert data[0]["messages"] == original["messages"]


def test_held_out_and_other_role_corruption_rejected_before_slice(role_data, tmp_path):
    root = tmp_path / "dataset"
    shutil.copytree(role_data, root)
    with (root / "spatial_mission/test.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(PlanningRoleSFTDatasetError, match="hash mismatch"):
        PlanningRoleSFTDataset(root, role="mission_interpreter", max_samples=1)


def test_cross_split_row_rejected_even_with_updated_file_hash(role_data, tmp_path):
    root = tmp_path / "dataset"
    shutil.copytree(role_data, root)
    source = root / "mission_interpreter/train.jsonl"
    rows = source.read_text().splitlines()
    row = json.loads(rows[0])
    row["split"] = "test"
    rows[0] = json.dumps(row, ensure_ascii=False)
    source.write_text("\n".join(rows) + "\n")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["data_sha256"]["mission_interpreter/train.jsonl"] = sha256(source.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(PlanningRoleSFTDatasetError, match="role/split differs"):
        PlanningRoleSFTDataset(root, role="fleet_planner", max_samples=1)


@pytest.mark.parametrize("kwargs", [
    {"role": "unknown"}, {"role": "fleet_planner", "split": "test_reassignment"},
    {"role": "fleet_planner", "max_samples": True},
])
def test_invalid_selection_rejected(role_data, kwargs):
    with pytest.raises(PlanningRoleSFTDatasetError):
        PlanningRoleSFTDataset(role_data, **kwargs)


class BoundaryMergeTokenizer:
    """A tokenizer whose newline+opening brace is one BPE token."""

    pad_token_id = 0
    merge_token_id = 999999

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        result = []
        position = 0
        while position < len(text):
            if text[position:position + 2] == "\n{":
                result.append(self.merge_token_id)
                position += 2
            else:
                result.append(ord(text[position]) + 1)
                position += 1
        return result

    def decode(self, tokens):
        return "".join("\n{" if item == self.merge_token_id else chr(item - 1) for item in tokens)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        rendered = "".join(f"<{message['role']}>\n{message['content']}<end>\n" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>\n"
        return self.encode(rendered) if tokenize else rendered


def feature():
    return {"sample_id": "bpe", "messages": [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "request"},
        {"role": "assistant", "content": '{"goal":"前往坐标"}'},
    ]}


def test_bpe_boundary_preserves_generation_prompt_and_complete_json_answer():
    tokenizer = BoundaryMergeTokenizer()
    sample = feature()
    collator = AssistantOnlyDataCollator(tokenizer, model_max_length=1024)
    encoded = collator.encode_feature(sample)
    prompt = tokenizer.apply_chat_template(sample["messages"][:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(sample["messages"], tokenize=True, add_generation_prompt=False)
    assert full[:len(prompt)] != prompt  # reproduces the real Qwen boundary case
    assert encoded["input_ids"][:len(prompt)] == prompt
    assert encoded["labels"][:len(prompt)] == [IGNORE_INDEX] * len(prompt)
    assert tokenizer.decode(encoded["labels"][len(prompt):]) == sample["messages"][-1]["content"] + "<end>\n"
    assert tokenizer.decode(encoded["input_ids"]) == tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=False)


def test_bpe_fallback_still_rejects_inconsistent_template_tokenization():
    class BadTemplate(BoundaryMergeTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            result = super().apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
            if tokenize and len(messages) == 3:
                result[0] += 1
            return result
    with pytest.raises(FleetPlannerCollatorError, match="cannot be proven"):
        AssistantOnlyDataCollator(BadTemplate(), model_max_length=1024).encode_feature(feature())


def test_whole_answer_rejected_instead_of_silently_truncated():
    tokenizer = BoundaryMergeTokenizer()
    sample = feature()
    encoded = AssistantOnlyDataCollator(tokenizer, model_max_length=1024).encode_feature(sample)
    with pytest.raises(FleetPlannerCollatorError, match="refusing truncation"):
        AssistantOnlyDataCollator(tokenizer, model_max_length=len(encoded["input_ids"]) - 1).encode_feature(sample)


def test_preencoded_batch_matches_live_encoding_and_masks_padding():
    tokenizer = BoundaryMergeTokenizer()
    collator = AssistantOnlyDataCollator(tokenizer, model_max_length=1024, pad_to_multiple_of=8)
    samples = [feature(), feature()]
    samples[1]["messages"][-1]["content"] = "{}"
    prepared = [collator.encode_feature(item) for item in samples]
    prepared[0]["ignored_metadata"] = "must never be forwarded to the model"
    cached_batch = collator(prepared)
    live_batch = collator(samples)
    assert set(cached_batch) == {"input_ids", "attention_mask", "labels"}
    for key in cached_batch:
        assert cached_batch[key].tolist() == live_batch[key].tolist()
    assert cached_batch["labels"][1, -1].item() == IGNORE_INDEX
    assert cached_batch["attention_mask"][1, -1].item() == 0


@pytest.mark.parametrize("mutation", ["empty_answer", "missing_answer_token", "prompt_loss", "wrong_label", "too_long", "masked_input"])
def test_preencoded_corruption_rejected(mutation):
    collator = AssistantOnlyDataCollator(BoundaryMergeTokenizer(), model_max_length=1024)
    encoded = collator.encode_feature(feature())
    broken = deepcopy(encoded)
    if mutation == "empty_answer":
        broken["labels"] = [IGNORE_INDEX] * len(broken["labels"])
    elif mutation == "missing_answer_token":
        broken["labels"][-1] = IGNORE_INDEX
    elif mutation == "prompt_loss":
        broken["labels"][0] = broken["input_ids"][0]
    elif mutation == "wrong_label":
        broken["labels"][-1] += 1
    elif mutation == "too_long":
        collator.model_max_length = len(broken["input_ids"]) - 1
    else:
        broken["attention_mask"][-1] = 0
    with pytest.raises(FleetPlannerCollatorError, match="invalid cached"):
        collator([broken])
