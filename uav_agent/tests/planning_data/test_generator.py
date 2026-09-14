from copy import deepcopy
import json

import pytest

from planning_data import generator
from planning_data.tasks import generate_task_blueprints


def test_all_families_and_scales_render_through_real_contracts():
    tasks = generate_task_blueprints(25)
    rows = [row for task in tasks for row in generator.render_task(task)]
    assert len(rows) == 200
    assert {row["role"] for row in rows} == set(generator.ROLES)
    for row in rows:
        assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
        assert row["checks"]["blueprint_semantics"]
        assert json.loads(row["messages"][-1]["content"])["schema_version"] in (1, 2, 3)


def test_persisted_answers_replayed_and_independent_audit_rejects_excess_tracking():
    task = next(task for task in generate_task_blueprints(25) if task["family"] == "search_track")
    candidates = {row["sample_id"]: row for row in generator.render_task(task)}
    generator.render_task(task, candidates)
    row = next(row for row in candidates.values() if row["role"] == "spatial_mission")
    answer = json.loads(row["messages"][-1]["content"])
    track = next(step for step in answer["steps"] if step["skill"] == "TRACK")
    track["args"]["duration_s"] += 5
    row["messages"][-1]["content"] = generator.canonical(answer)
    with pytest.raises(ValueError, match="blueprint local check failed"):
        generator.render_task(task, candidates)


def test_persisted_model_inputs_must_match_real_production_prompt():
    task = generate_task_blueprints(1)[0]
    candidates = {row["sample_id"]: row for row in generator.render_task(task)}
    row = next(iter(candidates.values()))
    row["messages"][0]["content"] += " invented extra prompt"
    with pytest.raises(ValueError, match="mismatched persisted messages"):
        generator.render_task(task, candidates)


def test_generation_disk_replay_and_tamper_detection(tmp_path):
    output = tmp_path / "data"
    manifest = generator.generate_dataset(output, count=25, tokenizer_path=None)
    assert manifest["total_samples"] == 200
    assert generator.validate_dataset(output)["all_persisted_answers_replayed"]
    with pytest.raises(FileExistsError):
        generator.generate_dataset(output, count=1, tokenizer_path=None)
    file = output / "tasks.jsonl"
    file.write_text(file.read_text() + "\n")
    with pytest.raises(ValueError, match="hashes differ"):
        generator.validate_dataset(output)


def test_failed_generation_never_publishes_partial_dataset(tmp_path, monkeypatch):
    def fail(task):
        raise ValueError("bad gold")
    monkeypatch.setattr(generator, "render_task", fail)
    output = tmp_path / "data"
    with pytest.raises(ValueError, match="bad gold"):
        generator.generate_dataset(output, count=1, tokenizer_path=None)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field,value,message", [
    ("distinct_semantic_target_count_distribution", {"200": -1}, "target count distribution"),
    ("validation", {}, "validation metadata"),
    ("total_samples", 999, "total sample count"),
])
def test_manifest_counts_are_checked_independently(tmp_path, field, value, message):
    output = tmp_path / "data"
    manifest = generator.generate_dataset(output, count=1, tokenizer_path=None)
    manifest[field] = value
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        generator.validate_dataset(output)


def test_changed_blueprint_requires_new_semantic_hash():
    task = deepcopy(generate_task_blueprints(1)[0])
    task["assignments"][0]["destination_xyz_m"][0] += 1
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        generator.render_task(task)


def test_full_answer_overflow_is_rejected_instead_of_truncated():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "sp" if len(messages) == 2 else "spabc"
        def encode(self, value, **kwargs):
            return list(value)
    row = generator.render_task(generate_task_blueprints(1)[0])[0]
    assert generator.measure_tokens(row, Tokenizer(), 5)["assistant_tokens"] == 3
    with pytest.raises(ValueError, match="refusing truncation"):
        generator.measure_tokens(row, Tokenizer(), 4)
