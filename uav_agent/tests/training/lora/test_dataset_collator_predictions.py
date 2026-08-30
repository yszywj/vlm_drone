from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fleet_data.generator import FleetDatasetGenerator
from fleet_data.validator import PATCH_OUTPUT_KIND, PLAN_OUTPUT_KIND
from training.lora.collator import (
    IGNORE_INDEX,
    AssistantOnlyDataCollator,
    FleetPlannerCollatorError,
)
from training.lora.dataset import (
    FleetPlannerSFTDataset,
    FleetPlannerSFTDatasetError,
    canonical_json,
)
from training.lora.generate_fleet_planner_predictions import (
    PredictionRuntime,
    generate_predictions,
    generate_raw_output,
    prediction_diagnostics,
    strict_json_object,
)


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "fleet_planner_v1"
    FleetDatasetGenerator().write(root, seed=42)
    return root


class FakeChatTokenizer:
    pad_token_id = 0

    @staticmethod
    def _render(messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
        text = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | list[int]:
        rendered = self._render(messages, add_generation_prompt)
        return [ord(character) + 1 for character in rendered] if tokenize else rendered


def test_dataset_uses_canonical_production_plan_and_patch_json(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path)
    train = FleetPlannerSFTDataset(root, split="train", max_samples=1)
    assert len(train) == train.count == 1
    record = train[0]
    assert record["output_kind"] == PLAN_OUTPUT_KIND
    assert json.loads(record["input_json"]) == record["request"]
    assert json.loads(record["assistant_json"]) == record["target"]
    assert record["input_json"] == canonical_json(record["request"])
    assert record["assistant_json"] == canonical_json(record["target"])
    assert "```" not in record["assistant_json"]
    assert "移动目标" in record["input_json"]

    reassignment = FleetPlannerSFTDataset(root, split="test_reassignment")
    patch = next(
        item for item in reassignment if item["output_kind"] == PATCH_OUTPUT_KIND
    )
    assert "replacement_assignments" in patch["target"]
    assert json.loads(patch["assistant_json"]) == patch["target"]


def test_dataset_validates_whole_root_before_safe_limit(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    # Corrupt an unused held-out split.  A train smoke slice must still fail.
    with (root / "test_conflict.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(FleetPlannerSFTDatasetError, match="validation failed"):
        FleetPlannerSFTDataset(root, split="train", max_samples=1)

    valid_root = _dataset(tmp_path / "valid")
    with pytest.raises(FleetPlannerSFTDatasetError, match="positive integer"):
        FleetPlannerSFTDataset(valid_root, split="train", max_samples=0)


def test_canonical_json_is_deterministic_and_rejects_nan() -> None:
    assert canonical_json({"中": 2, "a": 1}) == '{"a":1,"中":2}'
    assert canonical_json({"b": [2, 1], "a": {"d": 4, "c": 3}}) == (
        '{"a":{"c":3,"d":4},"b":[2,1]}'
    )
    with pytest.raises(ValueError):
        canonical_json({"bad": float("nan")})


def test_collator_masks_system_user_and_padding_but_not_assistant(
    tmp_path: Path,
) -> None:
    record = FleetPlannerSFTDataset(
        _dataset(tmp_path),
        split="train",
        max_samples=1,
    )[0]
    short = copy.deepcopy(record)
    short["messages"][-1]["content"] = "{}"
    tokenizer = FakeChatTokenizer()
    collator = AssistantOnlyDataCollator(
        tokenizer,
        model_max_length=100_000,
        pad_to_multiple_of=8,
    )
    batch = collator([record, short])
    input_rows = batch["input_ids"].tolist()
    attention_rows = batch["attention_mask"].tolist()
    label_rows = batch["labels"].tolist()

    for feature, input_ids, attention, labels in zip(
        (record, short),
        input_rows,
        attention_rows,
        label_rows,
    ):
        prompt_ids = tokenizer.apply_chat_template(
            feature["messages"][:-1],
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            feature["messages"],
            tokenize=True,
            add_generation_prompt=False,
        )
        assert input_ids[: len(full_ids)] == full_ids
        assert labels[: len(prompt_ids)] == [IGNORE_INDEX] * len(prompt_ids)
        assert labels[len(prompt_ids) : len(full_ids)] == full_ids[len(prompt_ids) :]
        assert all(value == IGNORE_INDEX for value in labels[len(full_ids) :])
        assert attention[: len(full_ids)] == [1] * len(full_ids)
        assert attention[len(full_ids) :] == [0] * (len(attention) - len(full_ids))


def test_collator_fails_closed_if_assistant_boundary_cannot_be_proven() -> None:
    class BadTemplate(FakeChatTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            result = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
            if len(messages) == 3:
                assert isinstance(result, list)
                result[0] += 1
            return result

    feature = {
        "sample_id": "sample",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": "{}"},
        ],
    }
    with pytest.raises(FleetPlannerCollatorError, match="cannot be proven"):
        AssistantOnlyDataCollator(
            BadTemplate(),
            model_max_length=1000,
        )([feature])


def _diagnostic_for(record: dict[str, object], raw: str) -> dict[str, object]:
    return prediction_diagnostics(
        sample_id=str(record["sample_id"]),
        output_kind=str(record["output_kind"]),
        request=record["request"],
        raw_model_output=raw,
        model_kind="base",
        adapter=None,
    )


def test_prediction_diagnostics_valid_malformed_wrong_schema_and_patch(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path)
    plan = FleetPlannerSFTDataset(root, split="train", max_samples=1)[0]
    patch = next(
        item
        for item in FleetPlannerSFTDataset(root, split="test_reassignment")
        if item["output_kind"] == PATCH_OUTPUT_KIND
    )

    valid = _diagnostic_for(plan, str(plan["assistant_json"]))
    assert valid["parse_success"] is True
    assert valid["schema_valid"] is True
    assert valid["parse_error"] is None
    assert valid["schema_error"] is None

    malformed = _diagnostic_for(plan, "```json\n{}\n```")
    assert malformed["parse_success"] is False
    assert malformed["parsed_output"] is None
    assert malformed["schema_valid"] is False
    # There is a gold target in `plan`, but malformed output is never repaired.
    assert malformed["parsed_output"] != plan["target"]

    wrong_schema = _diagnostic_for(plan, '{"schema_version":1}')
    assert wrong_schema["parse_success"] is True
    assert wrong_schema["schema_valid"] is False
    assert wrong_schema["schema_error"]

    valid_patch = _diagnostic_for(patch, str(patch["assistant_json"]))
    assert valid_patch["parse_success"] is True
    assert valid_patch["schema_valid"] is True

    with pytest.raises(ValueError, match="duplicate JSON key"):
        strict_json_object('{"a":1,"a":2}')
    with pytest.raises(ValueError, match="non-finite JSON number"):
        strict_json_object('{"a":1e999}')


class FakePredictionProcessor(FakeChatTokenizer):
    tokenizer = None

    def __init__(self) -> None:
        self.tokenizer = self

    def __call__(self, *, text, return_tensors, padding):
        assert return_tensors == "pt"
        assert padding is True
        return {
            "input_ids": [[ord(character) for character in text[0]]],
            "token_type_ids": [[0 for _ in text[0]]],
        }

    def batch_decode(self, rows, *, skip_special_tokens, **kwargs):
        assert skip_special_tokens is True
        return ["".join(chr(token) for token in rows[0])]


class FakePredictionModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.generation_kwargs: dict[str, object] | None = None

    def generate(self, **kwargs):
        self.generation_kwargs = dict(kwargs)
        prompt = kwargs["input_ids"][0]
        return [prompt + [ord(character) for character in self.response]]


def test_raw_generation_is_deterministic_and_does_not_pass_temperature(
    tmp_path: Path,
) -> None:
    model = FakePredictionModel("{}")
    runtime = PredictionRuntime(
        processor=FakePredictionProcessor(),
        model=model,
        model_kind="base",
        base_model_path=tmp_path,
        adapter_path=None,
    )
    output = generate_raw_output(
        runtime,
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        max_new_tokens=32,
    )
    assert output == "{}"
    assert model.generation_kwargs is not None
    assert model.generation_kwargs["do_sample"] is False
    assert model.generation_kwargs["num_beams"] == 1
    assert "temperature" not in model.generation_kwargs
    assert "token_type_ids" not in model.generation_kwargs


def test_prediction_jsonl_has_fixed_diagnostics_and_no_gold_repair(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path)
    dataset = FleetPlannerSFTDataset(root, split="train")
    responses = iter(
        [str(dataset[0]["assistant_json"]), "not-json"]
    )
    runtime = PredictionRuntime(
        processor=object(),
        model=object(),
        model_kind="adapter",
        base_model_path=tmp_path,
        adapter_path=tmp_path / "adapter",
    )
    output = tmp_path / "predictions" / "train.jsonl"

    def fake_generator(runtime, messages, max_new_tokens):
        assert max_new_tokens == 64
        assert [message["role"] for message in messages] == ["system", "user"]
        return next(responses)

    summary = generate_predictions(
        dataset_root=root,
        split="train",
        output=output,
        runtime=runtime,
        max_new_tokens=64,
        raw_generator=fake_generator,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["sample_count"] == 2
    assert summary["parse_success_count"] == 1
    assert summary["schema_valid_count"] == 1
    assert rows[0]["schema_valid"] is True
    assert rows[1]["parse_success"] is False
    assert rows[1]["parsed_output"] is None
    assert set(rows[0]) == {
        "sample_id",
        "output_kind",
        "raw_model_output",
        "parsed_output",
        "parse_success",
        "parse_error",
        "schema_valid",
        "schema_error",
        "model_kind",
        "adapter",
    }
    for line in output.read_text(encoding="utf-8").splitlines():
        assert line == canonical_json(json.loads(line))
