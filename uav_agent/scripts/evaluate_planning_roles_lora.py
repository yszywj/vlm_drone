#!/usr/bin/env python3
"""Evaluate base and configured production LoRA decoding on the complete test split.

The variants share prompts and sampling budgets. Adapter-specific decoding is
applied through production routing, so this compares deployed configurations,
not an intervention that changes only model weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adapter_registry import (
    DEFAULT_ADAPTER_CONFIG, AdapterRegistry, AdapterRegistryError, AdapterStatus,
    ModelCallRole,
)
from models.model_client_factory import ModelClientFactory
from models.openai_compatible_client import OpenAICompatibleClient
from training.lora.planning_chain_eval import (
    DEFAULT_MAX_TOKENS, generation_options_audit, prepare_evaluation_options, run_chain,
)
from training.lora.planning_role_eval import ROLES, capture_role_request, score_role_output
from training.lora.roles_dataset import PlanningRoleSFTDataset

CALL_ROLES = dict(zip(ROLES, (
    ModelCallRole.MISSION_INTERPRETATION, ModelCallRole.FLEET_PLAN,
    ModelCallRole.AGENT_SPATIAL_PLAN,
)))
INPUT_METADATA = (
    "sample_id", "task_id", "role", "uav_id", "family", "scale", "split",
    "semantic_hash", "instruction_semantic_hash", "response_schema_sha256",
)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def group_scores(rows, group_key, pass_key):
    groups = {}
    for row in rows:
        key = str(row[group_key])
        entry = groups.setdefault(key, {"passed": 0, "total": 0})
        entry["total"] += 1
        entry["passed"] += bool(row[pass_key])
    for entry in groups.values():
        entry["rate"] = entry["passed"] / entry["total"]
    return groups


def summarize(role_rows, chain_rows):
    result = {
        "isolated": {
            "completed": len(role_rows),
            "by_role": group_scores(role_rows, "role", "passed"),
            "by_family": group_scores(role_rows, "family", "passed"),
            "by_scale": group_scores(role_rows, "scale", "passed"),
            "by_role_scale": group_scores([
                {**row, "role_scale": f"{row['role']}/{row['scale']}"} for row in role_rows
            ], "role_scale", "passed"),
            "structural_by_role": group_scores(role_rows, "role", "structural_pass"),
            "service_errors": sum(row.get("service_error") is not None for row in role_rows),
            "scorer_errors": sum(row.get("scorer_error") is not None for row in role_rows),
            "truncated": sum(row.get("response", {}).get("finish_reason") == "length" for row in role_rows),
            "findings": dict(Counter(f.get("code", "unknown") for row in role_rows for f in row.get("findings", []))),
        },
        "chain": {
            "completed": len(chain_rows),
            "strict_passed": sum(row["strict_entire_task_pass"] for row in chain_rows),
            "passed_with_runtime_completion": sum(row["entire_task_pass_with_runtime_completion"] for row in chain_rows),
            "by_family": group_scores(chain_rows, "family", "strict_entire_task_pass"),
            "by_scale": group_scores(chain_rows, "scale", "strict_entire_task_pass"),
            "local_passed": sum(row["local_pass_count"] for row in chain_rows),
            "local_expected": sum(row["expected_local_count"] for row in chain_rows),
            "local_blocked": sum(row["local_blocked_count"] for row in chain_rows),
            "service_errors": sum(row["client_error_count"] for row in chain_rows),
            "truncated": sum(row["truncated_call_count"] for row in chain_rows),
            "model_calls": sum(row["model_call_count"] for row in chain_rows),
        },
    }
    return result


def prepare(dataset):
    # This checks every data file, split ownership and all generation sources.
    checked = PlanningRoleSFTDataset(dataset, role=ROLES[0], split="test")
    tasks = {task["task_id"]: task for task in read_jsonl(dataset / "tasks.jsonl") if task["split"] == "test"}
    rows = list(checked)
    for role in ROLES[1:]:
        rows.extend(read_jsonl(dataset / role / "test.jsonl"))
    prepared = []
    for row in rows:
        task = tasks[row["task_id"]]
        captured = capture_role_request(task, row["role"], uav_id=row["uav_id"])
        if [message.to_dict() for message in captured["messages"]] != row["messages"][:-1]:
            raise AssertionError(f"production prompt drift: {row['sample_id']}")
        if captured["response_schema_sha256"] != row["response_schema_sha256"]:
            raise AssertionError(f"production schema drift: {row['sample_id']}")
        captured["options"] = replace(captured["options"], temperature=0, top_p=1,
                                       max_tokens=DEFAULT_MAX_TOKENS[row["role"]])
        # No assistant answer enters the inference work item.
        prepared.append((task, {key: row[key] for key in INPUT_METADATA}, captured))
    expected_by_role = {
        role: checked.manifest["role_split_counts"][role].get("test", 0)
        for role in ROLES
    }
    actual_by_role = dict(Counter(row["role"] for row in rows))
    if (
        len(tasks) != checked.manifest["task_split_counts"].get("test", 0)
        or actual_by_role != expected_by_role
    ):
        raise AssertionError("expected complete test split counts from verified manifest")
    if not tasks or not prepared:
        raise AssertionError("evaluation requires a non-empty test split")
    return prepared, list(tasks.values())


def lora_selections(registry):
    selections = {role: registry.resolve(call_role) for role, call_role in CALL_ROLES.items()}
    for role, selection in selections.items():
        if selection.adapter_status is not AdapterStatus.ACTIVE or selection.fallback_used:
            raise AdapterRegistryError(f"LoRA evaluation requires active {role} without base fallback")
    return selections


def build_clients(variant, url, registry):
    """Use production routing for LoRA and the unchanged base decoding policy."""
    if variant == "base":
        return {
            role: OpenAICompatibleClient(
                url, registry.base_model_name, timeout_s=600, max_retries=0,
            )
            for role in ROLES
        }
    if variant != "lora":
        raise ValueError(f"unknown evaluation variant: {variant}")
    lora_selections(registry)
    factory = ModelClientFactory(
        registry, base_url=url, timeout_s=600, max_retries=0,
        client_factory=OpenAICompatibleClient,
    )
    return {role: factory.for_role(call_role) for role, call_role in CALL_ROLES.items()}


def evaluate_role(item, clients):
    task, row, captured = item
    role = row["role"]
    effective_options = prepare_evaluation_options(clients[role], captured["options"])
    started = perf_counter()
    result = {**row, "passed": False, "structural_pass": False, "semantic_pass": False,
              "findings": [], "service_error": None, "scorer_error": None,
              "messages": [message.to_dict() for message in captured["messages"]],
              "generation": {**generation_options_audit(effective_options), "repair_budget": 0},
              "requested_model": clients[role].model}
    try:
        response = clients[role].chat(captured["messages"], options=effective_options)
    except Exception as exc:
        result["service_error"] = {"type": type(exc).__name__, "message": str(exc)}
    else:
        result["response"] = {"content": response.content, "model": response.model,
                              "finish_reason": response.finish_reason, "usage": dict(response.usage)}
        if response.model != clients[role].model:
            raise AssertionError(f"unexpected model routing: {response.model}")
        try:
            result.update(score_role_output(task, role, response.content, uav_id=row["uav_id"]))
        except Exception as exc:
            result["scorer_error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    result["elapsed_s"] = perf_counter() - started
    return result


def evaluate_variant(variant, url, output, prepared, tasks, workers, registry):
    directory = output / variant
    directory.mkdir(exist_ok=True)
    clients = build_clients(variant, url, registry)
    models = {role: client.model for role, client in clients.items()}
    for client in clients.values():
        client.healthcheck()
    role_path = directory / "role_results.jsonl"
    chain_path = directory / "chain_results.jsonl"
    role_rows, chain_rows = read_jsonl(role_path), read_jsonl(chain_path)

    def progress(phase):
        summary = summarize(role_rows, chain_rows)
        write_json(directory / "summary.json", summary)
        write_json(directory / "progress.json", {"variant": variant, "phase": phase, "updated_at": utcnow(),
                   "role_completed": len(role_rows), "role_expected": len(prepared),
                   "chain_completed": len(chain_rows), "chain_expected": len(tasks)})

    def stream_jobs(items, function, key, existing, path, phase, concurrency):
        done = {row[key] for row in existing}
        if len(done) != len(existing):
            raise AssertionError(f"duplicate existing {key}")
        remaining = [item for item in items if (item[1][key] if phase == "isolated" else item[key]) not in done]
        with path.open("a", encoding="utf-8", buffering=1) as stream, ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(function, item) for item in remaining]
            for future in as_completed(futures):
                row = future.result()
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                existing.append(row)
                progress(phase)
                if len(existing) % 20 == 0:
                    print(f"{utcnow()} {variant} {phase} {len(existing)}/{len(items)}", flush=True)

    progress("isolated")
    stream_jobs(prepared, lambda item: evaluate_role(item, clients), "sample_id", role_rows, role_path, "isolated", workers)
    if any(row.get("scorer_error") for row in role_rows):
        progress("scorer_error_requires_review")
        raise RuntimeError(f"{variant}: scorer errors saved; review raw outputs before interpreting scores")

    def chain(task):
        result = run_chain(task, clients, max_tokens=DEFAULT_MAX_TOKENS, repair_budget=0)
        for call in result["calls"]:
            if call["response"] and call["response"]["model"] != models[call["role"]]:
                raise AssertionError("chain model routing mismatch")
        return result

    progress("chain")
    stream_jobs(tasks, chain, "task_id", chain_rows, chain_path, "chain", min(workers, 12))
    progress("completed")
    return summarize(role_rows, chain_rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--lora-url", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--adapter-config", type=Path, default=DEFAULT_ADAPTER_CONFIG)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 32:
        parser.error("workers must be in [1, 32]")
    # Both services are loopback endpoints; never send requests to an HTTP proxy.
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    manifest_path = args.output / "evaluation_manifest.json"
    if args.output.exists() and any(args.output.iterdir()) and not manifest_path.exists():
        raise RuntimeError("output directory is not an evaluation resume directory; choose a fresh directory")
    registry = AdapterRegistry(args.adapter_config)
    selections = lora_selections(registry)
    print(f"{utcnow()} checking dataset and all test production prompts", flush=True)
    prepared, tasks = prepare(args.dataset)
    manifest = {
        "evaluation_schema_version": 2, "dataset": str(args.dataset.resolve()),
        "dataset_manifest_sha256": digest(args.dataset / "manifest.json"),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), ROOT / "training/lora/planning_role_eval.py", ROOT / "training/lora/planning_chain_eval.py",
            ROOT / "models/adapter_registry.py", ROOT / "models/model_client_factory.py",
            ROOT / "models/schema_order.py", ROOT / "models/openai_compatible_client.py", ROOT / "models/base.py",
            ROOT / "training/lora/roles_dataset.py")},
        "base_url": args.base_url, "lora_url": args.lora_url, "base_model": registry.base_model_name,
        "adapter_config": str(registry.config_path), "adapter_config_sha256": digest(registry.config_path),
        "lora_models": {role: selection.effective_model for role, selection in selections.items()},
        "role_routing": {
            "base": {role: {"effective_model": registry.base_model_name,
                             "json_schema_property_order": "preserve"} for role in ROLES},
            "lora": {role: selection.to_dict() for role, selection in selections.items()},
        },
        "comparison_scope": "base decoding versus configured production LoRA decoding; not weights-only",
        "max_tokens": DEFAULT_MAX_TOKENS, "repair_budget": 0,
        "temperature": 0, "top_p": 1, "workers_per_variant": args.workers,
        "expected_isolated_samples_per_variant": len(prepared), "expected_chain_tasks_per_variant": len(tasks),
        "isolated_downstream_conditioning": "gold upstream", "chain_conditioning": "actual upstream model outputs",
        "json_schema_constrained": True, "flight_simulation_performed": False,
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("resume refused: evaluation provenance/configuration changed")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    print(f"{utcnow()} checked; beginning generation", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {variant: pool.submit(evaluate_variant, variant, url, args.output, prepared, tasks, args.workers, registry)
                   for variant, url in (("base", args.base_url), ("lora", args.lora_url))}
        summaries = {variant: future.result() for variant, future in futures.items()}
    write_json(args.output / "comparison.json", {"completed_at": utcnow(), "variants": summaries})
    print(f"{utcnow()} evaluation completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
