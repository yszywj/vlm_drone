#!/usr/bin/env python3
"""Build an audited planning curriculum without changing its parent dataset."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planning_data import generator as gold  # noqa: E402
from planning_data.tasks import instruction_semantic_hash, semantic_hash  # noqa: E402


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tasks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _source_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"source path escapes its root: {relative}")
    return path


def verify_parent(parent: Path, source_snapshot: Path,
                  manifest_snapshot: Path | None = None) -> tuple[dict, list[dict]]:
    """Verify the immutable parent's files against the captured old sources.

    Current production code may have changed; it is intentionally not used as
    evidence of what generated the parent. All parent tasks are subsequently
    rendered and independently audited with current production code.
    """
    parent, source_snapshot = parent.resolve(), source_snapshot.resolve()
    manifest_snapshot = manifest_snapshot or source_snapshot.parent / "v1_dataset_manifest.json"
    manifest_path = parent / "manifest.json"
    if file_digest(manifest_path) != file_digest(manifest_snapshot):
        raise ValueError("parent manifest differs from its immutable snapshot")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_paths = {"tasks.jsonl"} | {
        f"{role}/{split}.jsonl" for role in gold.ROLES for split in gold.SPLITS
    }
    if manifest.get("dataset_schema_version") != 1:
        raise ValueError("unsupported parent dataset schema")
    if set(manifest.get("data_sha256", {})) != expected_paths:
        raise ValueError("parent manifest does not cover all role projections")
    if gold._file_hashes(parent) != manifest["data_sha256"]:
        raise ValueError("parent JSONL hashes differ from its manifest")
    sources = manifest.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("parent lacks a generation source manifest")
    for relative, expected in sources.items():
        if file_digest(_source_path(source_snapshot, relative)) != expected:
            raise ValueError(f"parent source snapshot hash mismatch: {relative}")
    if not all(manifest.get("validation", {}).get(key) is True for key in (
        "all_labels_passed_production_contracts",
        "all_labels_passed_independent_blueprint_audits",
    )):
        raise ValueError("parent lacks successful label validation")
    tasks = read_tasks(parent / "tasks.jsonl")
    if len(tasks) != manifest["underlying_tasks"]:
        raise ValueError("parent task count differs from its manifest")
    if dict(Counter(task["split"] for task in tasks)) != manifest["task_split_counts"]:
        raise ValueError("parent task splits differ from its manifest")
    _check_task_identity(tasks)
    return manifest, tasks


def _check_task_identity(tasks: list[dict]) -> None:
    seen = {key: set() for key in ("task_id", "semantic_hash", "instruction_semantic_hash")}
    regression = gold._regression_instructions()
    for task in tasks:
        if task["split"] not in gold.SPLITS:
            raise ValueError("unknown task split")
        if semantic_hash(task) != task["semantic_hash"]:
            raise ValueError(f"{task['task_id']}: blueprint semantic hash mismatch")
        if instruction_semantic_hash(task) != task["instruction_semantic_hash"]:
            raise ValueError(f"{task['task_id']}: instruction semantic hash mismatch")
        for key, values in seen.items():
            if task[key] in values:
                raise ValueError(f"duplicate task {key}: {task[key]}")
            values.add(task[key])
        if task["instruction"] in regression:
            raise ValueError("held-out regression instruction leaked into curriculum")


def _extra_source_hashes() -> dict[str, dict[str, str]]:
    return {
        "curriculum_source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)): file_digest(Path(__file__))},
        "runtime_source_sha256": {
            name: file_digest(ROOT / name) for name in (
                "models/model_client_factory.py", "models/adapter_registry.py",
                "models/openai_compatible_client.py", "models/schema_order.py",
            ) if (ROOT / name).is_file()
        },
    }


def _curriculum_summary(parent_tasks: list[dict], targeted_tasks: list[dict]) -> dict:
    """Summarize cohort ownership separately from the legacy family strata."""
    return {
        "parent_task_ids": [task["task_id"] for task in parent_tasks],
        "targeted_task_ids": [task["task_id"] for task in targeted_tasks],
        "original_split_counts": dict(Counter(task["split"] for task in parent_tasks)),
        "targeted_split_counts": dict(Counter(task["split"] for task in targeted_tasks)),
        "targeted_focus_split_counts": dict(sorted(Counter(
            f"{task['curriculum_focus']}/{task['scale']}/{task['split']}"
            for task in targeted_tasks
        ).items())),
        "targeted_template_split_counts": dict(sorted(Counter(
            f"{task['curriculum_template_id']}/{task['split']}" for task in targeted_tasks
        ).items())),
        "new_held_out_test_task_ids": [task["task_id"] for task in targeted_tasks if task["split"] == "test"],
        "old_test_use": "previously inspected regression only; never used for training",
        "new_test_use": "held out from training and validation; do not use for checkpoint selection",
        "parent_task_blueprints_preserved": True,
        "parent_role_rows_rerendered_with_current_production": True,
        "new_semantic_hash_matches_parent": 0,
        "new_instruction_semantic_hash_matches_parent": 0,
    }


def _build_targeted(parent_tasks, *, seed, train_per_focus_scale,
                    validation_per_focus_scale, test_per_focus_scale):
    # Import lazily so parent-integrity checks need no curriculum generator.
    from planning_data.targeted_curriculum import audit_targeted_task, build_targeted_tasks
    seen = {task["semantic_hash"] for task in parent_tasks}
    seen_instructions = {task["instruction_semantic_hash"] for task in parent_tasks}
    tasks = []
    for offset, (split, count) in enumerate(zip(gold.SPLITS, (
        train_per_focus_scale, validation_per_focus_scale, test_per_focus_scale,
    ), strict=True)):
        partition = build_targeted_tasks(
            count, seed=seed + offset, partition=split,
            exclude_semantic_hashes=seen, exclude_instruction_semantic_hashes=seen_instructions,
        )
        for task in partition:
            task["task_id"] = f"curriculum_{len(tasks) + 1:06d}"
            task["split"] = split
            audit_targeted_task(task)
            seen.add(task["semantic_hash"])
            seen_instructions.add(task["instruction_semantic_hash"])
            tasks.append(task)
    return tasks


def _audit_curriculum_partitions(tasks: list[dict], parameters: dict) -> None:
    from planning_data.targeted_curriculum import FOCI, SCALES, audit_targeted_task
    expected = {
        f"{focus}/{scale}/{split}": parameters[f"{split}_per_focus_scale"]
        for focus in FOCI for scale in SCALES for split in gold.SPLITS
    }
    actual = Counter(f"{task['curriculum_focus']}/{task['scale']}/{task['split']}" for task in tasks)
    if dict(actual) != expected:
        raise ValueError("targeted focus/scale/split counts differ from generation parameters")
    for task in tasks:
        if task["split"] != task["curriculum_partition"]:
            raise ValueError("targeted template partition differs from task split")
        audit_targeted_task(task)


def generate_curriculum(parent: Path, output: Path, *, parent_source_snapshot: Path,
                        parent_manifest_snapshot: Path | None = None, seed: int = 20260913,
                        train_per_focus_scale: int = 16, validation_per_focus_scale: int = 4,
                        test_per_focus_scale: int = 4,
                        tokenizer_path: Path | None = gold.DEFAULT_TOKENIZER,
                        max_length: int = 16384, training_max_length: int = 8192,
                        progress=None) -> dict:
    parent, output = Path(parent).resolve(), Path(output).resolve()
    parent_source_snapshot = Path(parent_source_snapshot).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    if parent.is_relative_to(output) or output.is_relative_to(parent):
        raise ValueError("parent and output must be separate, non-nested directories")
    for key, value in (
        ("max_length", max_length), ("training_max_length", training_max_length),
        ("train_per_focus_scale", train_per_focus_scale),
        ("validation_per_focus_scale", validation_per_focus_scale),
        ("test_per_focus_scale", test_per_focus_scale),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    parent_manifest, parent_tasks = verify_parent(parent, parent_source_snapshot, parent_manifest_snapshot)
    targeted = _build_targeted(
        parent_tasks, seed=seed, train_per_focus_scale=train_per_focus_scale,
        validation_per_focus_scale=validation_per_focus_scale,
        test_per_focus_scale=test_per_focus_scale,
    )
    generation_parameters = {
        "train_per_focus_scale": train_per_focus_scale,
        "validation_per_focus_scale": validation_per_focus_scale,
        "test_per_focus_scale": test_per_focus_scale,
    }
    _audit_curriculum_partitions(targeted, generation_parameters)
    tasks = parent_tasks + targeted
    _check_task_identity(tasks)
    tokenizer = gold.load_tokenizer(tokenizer_path) if tokenizer_path is not None else None
    source_hashes, extra_sources = gold._source_hashes(), _extra_source_hashes()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    role_counts = {role: Counter() for role in gold.ROLES}
    token_values = {role: defaultdict(list) for role in gold.ROLES}
    training_length_checks = {role: {split: {"count": 0, "exceeds_training_max_length": 0,
                                             "max_full_tokens": 0}
                                    for split in gold.SPLITS} for role in gold.ROLES}
    runtime_closures = 0
    try:
        with ExitStack() as stack:
            task_stream = stack.enter_context((staging / "tasks.jsonl").open("w", encoding="utf-8"))
            streams = {}
            for role in gold.ROLES:
                (staging / role).mkdir()
                for split in gold.SPLITS:
                    streams[role, split] = stack.enter_context((staging / role / f"{split}.jsonl").open("w", encoding="utf-8"))
            for index, task in enumerate(tasks, 1):
                task_stream.write(gold.canonical(task) + "\n")
                for row in gold.render_task(task):
                    if tokenizer is not None:
                        row["token_lengths"] = gold.measure_tokens(row, tokenizer, max_length)
                        for key, value in row["token_lengths"].items():
                            token_values[row["role"]][key].append(value)
                        measured_length = max(row["token_lengths"]["full_tokens"],
                                              row["token_lengths"]["separately_encoded_full_tokens"])
                        check = training_length_checks[row["role"]][row["split"]]
                        check["count"] += 1
                        check["exceeds_training_max_length"] += int(measured_length > training_max_length)
                        check["max_full_tokens"] = max(check["max_full_tokens"], measured_length)
                    role_counts[row["role"]][row["split"]] += 1
                    runtime_closures += int(row["checks"].get("runtime_contract_closure", False))
                    streams[row["role"], row["split"]].write(gold.canonical(row) + "\n")
                if progress is not None and (index % 50 == 0 or index == len(tasks)):
                    progress(f"validated {index}/{len(tasks)} curriculum tasks")
        if source_hashes != gold._source_hashes() or extra_sources != _extra_source_hashes():
            raise ValueError("generation/runtime source changed during curriculum generation")
        manifest = {
            "dataset_name": output.name, "dataset_schema_version": 1,
            "label_origin": "deterministic_program_gold", "seed": seed,
            "underlying_tasks": len(tasks),
            "task_split_counts": dict(Counter(task["split"] for task in tasks)),
            "role_split_counts": {role: dict(values) for role, values in role_counts.items()},
            "total_samples": sum(sum(values.values()) for values in role_counts.values()),
            "stratified_task_counts": dict(sorted(Counter(
                f"{task['family']}/{task['scale']}/{task['split']}" for task in tasks
            ).items())),
            "distinct_semantic_target_count_distribution": dict(sorted(Counter(
                str(len(task["target_catalog"])) for task in tasks
            ).items())),
            "split_group": "underlying_task_all_role_projections", "semantic_duplicates": 0,
            "interpreter_visible_semantic_duplicates": 0, "held_out_regression_instruction_matches": 0,
            "validation": {
                "all_labels_passed_production_contracts": True,
                "all_labels_passed_independent_blueprint_audits": True,
                "runtime_contract_hover_closures": runtime_closures,
                "model_inference_performed": False, "flight_simulation_performed": False,
            },
            "token_audit": {
                "completed": tokenizer is not None, "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
                "tokenizer_sha256": gold._tokenizer_hashes(tokenizer_path),
                "assistant_counting": "separately_encoded_completion_including_end_of_turn_template_suffix",
                "full_length_checks": "both_full_template_and_separate_prompt_completion_encoding",
                "max_full_sequence_tokens": max_length, "truncation_policy": "reject_never_truncate",
                "by_role": gold._summarize_tokens(token_values),
            },
            "training_length_audit": {"completed": tokenizer is not None,
                                      "training_max_length": training_max_length,
                                      "role_split_counts": training_length_checks},
            "source_sha256": source_hashes, "data_sha256": gold._file_hashes(staging),
            **extra_sources,
            "parent": {
                "dataset_path": str(parent), "dataset_manifest_sha256": file_digest(parent / "manifest.json"),
                "data_sha256": parent_manifest["data_sha256"],
                "source_snapshot_path": str(parent_source_snapshot),
                "source_sha256": parent_manifest["source_sha256"],
                "manifest_snapshot_path": str(parent_manifest_snapshot or parent_source_snapshot.parent / "v1_dataset_manifest.json"),
                "immutable_parent_verified": True,
            },
            "curriculum": _curriculum_summary(parent_tasks, targeted),
            "generation_parameters": generation_parameters,
            "limitations": [
                "Positive synthetic instruction planning data; no model inference or flight simulation.",
                "Only mission_interpreter is scheduled for retraining; other role projections support auditing and chain evaluation.",
                "Parent test tasks were used during prior diagnosis and remain a regression cohort.",
                "New test tasks are unseen semantics and held-out language templates; planning families and scales overlap training.",
                "All assistant labels retain the canonical alphabetical JSON object ordering.",
            ],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "README.md").write_text(_readme(manifest), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"destination appeared during generation: {output}")
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_curriculum(directory: Path, *, progress=None) -> dict:
    """Replay all persisted labels, then verify curriculum and parent provenance."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for section, current in _extra_source_hashes().items():
        if manifest.get(section) != current:
            raise ValueError(f"curriculum provenance changed: {section}")
    parent = manifest["parent"]
    parent_path = Path(parent["dataset_path"])
    parent_manifest, parent_tasks = verify_parent(
        parent_path, Path(parent["source_snapshot_path"]), Path(parent["manifest_snapshot_path"]),
    )
    if (file_digest(parent_path / "manifest.json") != parent["dataset_manifest_sha256"]
            or parent_manifest["data_sha256"] != parent["data_sha256"]
            or parent_manifest["source_sha256"] != parent["source_sha256"]):
        raise ValueError("curriculum parent provenance differs from verified parent")
    tasks = read_tasks(directory / "tasks.jsonl")
    if tasks[:len(parent_tasks)] != parent_tasks:
        raise ValueError("parent task blueprints or split ownership changed")
    targeted = tasks[len(parent_tasks):]
    _audit_curriculum_partitions(targeted, manifest["generation_parameters"])
    if _curriculum_summary(parent_tasks, targeted) != manifest["curriculum"]:
        raise ValueError("curriculum cohort metadata differs from its task blueprints")
    result = gold.validate_dataset(directory, progress=progress)
    lengths = manifest["training_length_audit"]
    actual = {role: {split: {"count": 0, "exceeds_training_max_length": 0, "max_full_tokens": 0}
                     for split in gold.SPLITS} for role in gold.ROLES}
    if lengths["completed"] != manifest["token_audit"]["completed"]:
        raise ValueError("training length audit completion differs from token audit")
    if lengths["completed"]:
        for role in gold.ROLES:
            for split in gold.SPLITS:
                for row in read_tasks(directory / role / f"{split}.jsonl"):
                    full = max(row["token_lengths"]["full_tokens"], row["token_lengths"]["separately_encoded_full_tokens"])
                    check = actual[role][split]
                    check["count"] += 1
                    check["exceeds_training_max_length"] += int(full > lengths["training_max_length"])
                    check["max_full_tokens"] = max(check["max_full_tokens"], full)
    if actual != lengths["role_split_counts"]:
        raise ValueError("training length audit differs from persisted token lengths")
    return {**result, "parent_provenance_verified": True, "parent_split_ownership_preserved": True,
            "new_held_out_test_tasks": len(manifest["curriculum"]["new_held_out_test_task_ids"])}


def _readme(manifest: dict) -> str:
    curriculum = manifest["curriculum"]
    return (
        f"# {manifest['dataset_name']}\n\n"
        f"{manifest['underlying_tasks']} 个任务、{manifest['total_samples']} 条角色样本。"
        "所有标签通过当前生产解析器、编译器及独立任务语义审计；未运行模型或飞行仿真。\n\n"
        f"- 原数据任务划分保持不变：`{curriculum['original_split_counts']}`。\n"
        f"- 新定向任务划分：`{curriculum['targeted_split_counts']}`。\n"
        "- 新任务与旧数据全部任务之间、以及新任务自身之间，两种语义哈希均无重复。\n"
        "- 旧测试集只作为已分析的回归集合；新测试集单独报告，不参与训练或 checkpoint 选择。\n"
        "- 所有角色均有完整投影，本轮仅训练任务解释器；Fleet 和 Spatial 权重沿用。\n"
        "- 标签始终按字母序序列化。生产约束解码需使用对应 adapter 的字段排序配置。\n\n"
        "`manifest.json` 保存父数据与旧源码快照哈希、当前生成/推理源码哈希、任务组别及长度审计。"
        f"生成上限 {manifest['token_audit']['max_full_sequence_tokens']} 与训练上限 "
        f"{manifest['training_length_audit']['training_max_length']} 分别核查；不截断超长答案。"
        "`training_length_audit` 给出每个角色/划分是否超过训练长度。\n"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT.parent / "datasets/planning_roles_v1")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "datasets/planning_roles_v2_intent")
    parser.add_argument("--parent-source-snapshot", type=Path)
    parser.add_argument("--parent-manifest-snapshot", type=Path)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--train-per-focus-scale", type=int, default=16)
    parser.add_argument("--validation-per-focus-scale", type=int, default=4)
    parser.add_argument("--test-per-focus-scale", type=int, default=4)
    parser.add_argument("--tokenizer", type=Path, default=gold.DEFAULT_TOKENIZER)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--training-max-length", type=int, default=8192)
    parser.add_argument("--validate-only", type=Path)
    args = parser.parse_args(argv)
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    if args.validate_only:
        result = validate_curriculum(args.validate_only, progress=progress)
    else:
        if args.parent_source_snapshot is None:
            parser.error("--parent-source-snapshot is required for generation")
        result = generate_curriculum(
            args.parent, args.output, parent_source_snapshot=args.parent_source_snapshot,
            parent_manifest_snapshot=args.parent_manifest_snapshot, seed=args.seed,
            train_per_focus_scale=args.train_per_focus_scale,
            validation_per_focus_scale=args.validation_per_focus_scale,
            test_per_focus_scale=args.test_per_focus_scale,
            tokenizer_path=args.tokenizer, max_length=args.max_length,
            training_max_length=args.training_max_length, progress=progress,
        )
        result = {"output": str(args.output.resolve()), "underlying_tasks": result["underlying_tasks"],
                  "role_split_counts": result["role_split_counts"],
                  "targeted_split_counts": result["curriculum"]["targeted_split_counts"],
                  "new_held_out_test_tasks": len(result["curriculum"]["new_held_out_test_task_ids"]),
                  "training_length_audit": result["training_length_audit"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
