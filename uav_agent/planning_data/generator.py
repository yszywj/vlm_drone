"""Offline, production-prompt-aligned datasets for the three planning roles."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from fleet.compiler import FleetAssignmentCompiler
from fleet.llm_planner_v2 import LLMFleetPlannerV2
from fleet.llm_task_interpreter import LLMFleetTaskInterpreter
from models.base import ModelResponse
from planner.dynamic_llm_planner import DynamicLLMPlanner
from planner.schemas import LandingZoneSpec, PlannerWorldContext
from planning_data.local_gold import build_local_gold, validate_local_against_blueprint
from planning_data.tasks import (
    build_fleet_plan, build_fleet_request, build_task_spec, generate_task_blueprints,
    instruction_semantic_hash, semantic_hash,
)
from planning_data.validation import validate_semantic_gold
from target.types import TargetSpec


ROOT = Path(__file__).resolve().parents[1]
ROLES = ("mission_interpreter", "fleet_planner", "spatial_mission")
SPLITS = ("train", "validation", "test")
DEFAULT_TOKENIZER = ROOT.parent / "models/initial_model/Qwen3-VL-4B-Instruct"


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def digest(value: object) -> str:
    return sha256(canonical(value).encode("utf-8")).hexdigest()


class RecordingGoldClient:
    """Replay a program label through the real parser; never call a model."""

    def __init__(self, answer: dict | None = None):
        self.answer = answer
        self.calls: list[dict] = []

    def chat(self, messages, *, options):
        if self.answer is None:
            raise ValueError("gold response was not set")
        content = canonical(self.answer)
        self.calls.append({
            "messages": [message.to_dict() for message in messages] + [
                {"role": "assistant", "content": content},
            ],
            "response_schema_sha256": digest(options.response_format.to_dict()) if options.response_format else None,
            "runtime_default_max_output_tokens": options.max_tokens,
        })
        return ModelResponse(content, "deterministic_program_gold_no_model_call", "stop", {})


class RecordingLocalPlanner:
    source = "dynamic_llm"

    def __init__(self, answer: dict | None = None):
        self.client = RecordingGoldClient(answer)
        self.fixed_answer = answer
        self.planner = DynamicLLMPlanner(
            self.client, ROOT / "prompts/dynamic_skill_planner_v3_system.txt",
            planning_contract="v3", repair_budget=0,
        )

    def plan(self, request):
        self.client.answer = self.fixed_answer if self.fixed_answer is not None else build_local_gold(request).to_dict()
        return self.planner.plan(request)


def _world(task: dict, uav: dict) -> PlannerWorldContext:
    world = task["world"]
    home = tuple(uav["home_xyz_m"])
    name = uav["home_name"]
    return PlannerWorldContext(
        scene_min_xyz_m=tuple(world["scene_min_xyz_m"]),
        scene_max_xyz_m=tuple(world["scene_max_xyz_m"]),
        initial_uav_xyz_m=home, search_regions={},
        landing_zones={name: LandingZoneSpec(name, home[:2], home[2])},
        default_takeoff_altitude_m=world["flight_altitude_m"],
        default_track_duration_s=30.0, search_timeout_s=75.0,
    )


def render_task(task: dict, candidates: dict[str, dict] | None = None) -> list[dict]:
    """Capture real prompts and accept only independently checked answers.

    With candidates supplied, replay their answers instead of generating gold.
    This checks disk data against original blueprints and current production
    prompts, rather than comparing answers to the same answer builder.
    """
    # JSON object ordering must not change the alias directory order after a
    # persisted blueprint is loaded back for validation.
    task = json.loads(canonical(task))
    if semantic_hash(task) != task["semantic_hash"]:
        raise ValueError(f"{task['task_id']}: blueprint semantic hash mismatch")
    if instruction_semantic_hash(task) != task["instruction_semantic_hash"]:
        raise ValueError(f"{task['task_id']}: instruction semantic hash mismatch")
    rows = []

    def sample_id(role, uav_id=None):
        return f"{task['task_id']}/{role}" + (f"/{uav_id}" if uav_id else "")

    def candidate_answer(role, uav_id=None):
        if candidates is None:
            return None
        candidate = candidates[sample_id(role, uav_id)]
        messages = candidate["messages"]
        if [m["role"] for m in messages] != ["system", "user", "assistant"]:
            raise ValueError("role data must contain exactly system/user/assistant messages")
        return json.loads(messages[-1]["content"])

    def add(role, client, checks, uav_id=None):
        if len(client.calls) != 1:
            raise ValueError("gold required repair or more than one model response")
        row = {
            "dataset_schema_version": 1,
            "sample_id": sample_id(role, uav_id), "task_id": task["task_id"],
            "split": task["split"], "role": role, "family": task["family"],
            "scale": task["scale"], "semantic_hash": task["semantic_hash"],
            "instruction_semantic_hash": task["instruction_semantic_hash"],
            "label_origin": "deterministic_program_gold",
            "uav_id": uav_id, "checks": checks, **deepcopy(client.calls[0]),
        }
        if candidates is not None:
            candidate = candidates[row["sample_id"]]
            # Length statistics are checked separately with the actual tokenizer.
            for key, expected in row.items():
                if candidate.get(key) != expected:
                    raise ValueError(f"{row['sample_id']}: mismatched persisted {key}")
        rows.append(row)

    answer = candidate_answer("mission_interpreter")
    interpreter_client = RecordingGoldClient(answer if answer is not None else build_task_spec(task).to_dict())
    spec = LLMFleetTaskInterpreter(
        interpreter_client, uav_alias_catalog=task["uav_aliases"],
        target_alias_catalog=task["target_aliases"], repair_budget=0,
    ).interpret(task["instruction"])
    request = build_fleet_request(task, spec)
    answer = candidate_answer("fleet_planner")
    fleet_client = RecordingGoldClient(answer if answer is not None else build_fleet_plan(task, request).to_dict())
    fleet_planner = LLMFleetPlannerV2(fleet_client, repair_budget=0)
    plan = fleet_planner.plan(request)
    semantic = validate_semantic_gold(task, spec, plan)
    if not semantic["passed"] or fleet_planner.last_semantic_findings:
        raise ValueError(f"{task['task_id']}: semantic gold failed: {semantic}")
    add("mission_interpreter", interpreter_client, {"production_parser": True, "blueprint_semantics": True})
    add("fleet_planner", fleet_client, {"production_parser": True, "blueprint_semantics": True, "all_goals_assigned_once": True})
    target_catalog = {alias: TargetSpec.from_dict(value) for alias, value in task["target_catalog"].items()}
    uavs = {uav["uav_id"]: uav for uav in task["uavs"]}
    assignments = {a["uav_id"]: a for a in task["assignments"]}
    for assignment in plan.assignments:
        uav = uavs[assignment.uav_id]
        local = RecordingLocalPlanner(candidate_answer("spatial_mission", assignment.uav_id))
        result = FleetAssignmentCompiler(local).compile_assignment_v2(
            request, plan, assignment, _world(task, uav), target_catalog=target_catalog,
        )
        if result.compiled_mission is None or not result.semantically_valid:
            raise ValueError(f"{task['task_id']}/{assignment.uav_id}: local compilation failed: {result.validation_report.to_dict()}")
        blueprint = assignments[assignment.uav_id]
        local_audit = validate_local_against_blueprint(
            blueprint, result.planner_output, result.compiled_mission.task_plan,
            tuple(uav["home_xyz_m"]),
            allow_safety_completion=result.planner_request.allow_trusted_safety_completion,
            target_spec=target_catalog.get(blueprint.get("target_alias")),
        )
        if not local_audit["passed"]:
            raise ValueError(f"{task['task_id']}/{assignment.uav_id}: blueprint local check failed: {local_audit}")
        add("spatial_mission", local.client, {
            "production_parser": True, "production_compiler": True,
            "goal_coverage": True, "blueprint_semantics": True,
            "runtime_safety_completion_added": local_audit["runtime_safety_completion_added"],
            "runtime_contract_closure": local_audit["runtime_contract_closure"],
        }, assignment.uav_id)
    return rows


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def measure_tokens(row: dict, tokenizer, max_length: int) -> dict:
    messages = row["messages"]
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if not full_text.startswith(prompt_text) or len(full_text) <= len(prompt_text):
        raise ValueError(f"{row['sample_id']}: chat template does not preserve the assistant prefix")
    prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    full = tokenizer.encode(full_text, add_special_tokens=False)
    # Byte-pair tokenization may merge the final prompt newline with the first
    # answer character. Check both full-template and separate prompt/completion
    # encodings; never assume that tokenized strings preserve token prefixes.
    assistant = tokenizer.encode(full_text[len(prompt_text):], add_special_tokens=False)
    separate_length = len(prompt) + len(assistant)
    if max(len(full), separate_length) > max_length:
        raise ValueError(f"{row['sample_id']}: {max(len(full), separate_length)} full tokens exceeds {max_length}; refusing truncation")
    answer_bytes = len(messages[-1]["content"].encode("utf-8"))
    if row["role"] in {"mission_interpreter", "fleet_planner"} and answer_bytes > 32768:
        raise ValueError(f"{row['sample_id']}: answer exceeds production response byte limit")
    return {
        "prompt_tokens": len(prompt), "assistant_tokens": len(assistant),
        "full_tokens": len(full), "answer_utf8_bytes": answer_bytes,
        "separately_encoded_full_tokens": separate_length,
        "exceeds_runtime_default_output_budget": len(assistant) > row["runtime_default_max_output_tokens"],
    }


def _source_hashes() -> dict[str, str]:
    paths = []
    for pattern in ("planning_data/*.py", "prompts/*planner*.txt", "prompts/fleet_task_interpreter_system.txt", "fleet/*.py", "planner/*.py", "runtime/*.py", "skills/*.py", "models/base.py", "target/types.py", "scripts/generate_planning_roles_dataset.py"):
        paths.extend(ROOT.glob(pattern))
    return {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in sorted(set(paths))}


def _tokenizer_hashes(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    names = ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "chat_template.json", "chat_template.jinja", "special_tokens_map.json")
    return {name: sha256((path / name).read_bytes()).hexdigest() for name in names if (path / name).is_file()}


def _file_hashes(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): sha256(path.read_bytes()).hexdigest() for path in sorted(directory.rglob("*.jsonl"))}


def _regression_instructions() -> set[str]:
    path = ROOT / "configs/benchmarks/fleet_open_scaling_2_to_10.json"
    if not path.exists():
        return set()
    values = set()
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "instruction" and isinstance(item, str):
                    values.add(item)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(json.loads(path.read_text()))
    return values


def _summarize_tokens(token_values: dict) -> dict:
    summaries = {}
    for role, columns in token_values.items():
        summaries[role] = {}
        for key, values in columns.items():
            ordered = sorted(values)
            summaries[role][key] = {"min": min(values), "max": max(values), "p95": ordered[min(len(values)-1, int(len(values)*.95))]}
            if key.startswith("exceeds_"):
                summaries[role][key] = {"count": sum(values), "total": len(values)}
    return summaries


def generate_dataset(output: Path, *, count=1000, seed=42, tokenizer_path: Path | None = DEFAULT_TOKENIZER,
                     max_length=16384, progress: Callable[[str], None] | None = None) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    tasks = generate_task_blueprints(count=count, seed=seed)
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path is not None else None
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    role_counts = {role: Counter() for role in ROLES}
    token_values = {role: defaultdict(list) for role in ROLES}
    task_counts = Counter()
    strata = Counter()
    target_counts = Counter()
    fingerprints = set()
    instruction_fingerprints = set()
    ids = set()
    regression = _regression_instructions()
    runtime_closures = 0
    try:
        with ExitStack() as stack:
            task_stream = stack.enter_context((staging / "tasks.jsonl").open("w", encoding="utf-8"))
            streams = {}
            for role in ROLES:
                (staging / role).mkdir()
                for split in SPLITS:
                    streams[role, split] = stack.enter_context((staging / role / f"{split}.jsonl").open("w", encoding="utf-8"))
            for index, task in enumerate(tasks, 1):
                if task["semantic_hash"] in fingerprints or task["instruction_semantic_hash"] in instruction_fingerprints or task["task_id"] in ids:
                    raise ValueError("duplicate underlying task")
                if task["instruction"] in regression:
                    raise ValueError("held-out regression instruction leaked into dataset")
                fingerprints.add(task["semantic_hash"])
                instruction_fingerprints.add(task["instruction_semantic_hash"])
                ids.add(task["task_id"])
                task_stream.write(canonical(task) + "\n")
                task_counts[task["split"]] += 1
                strata[f"{task['family']}/{task['scale']}/{task['split']}"] += 1
                target_counts[str(len(task["target_catalog"]))] += 1
                for row in render_task(task):
                    if tokenizer is not None:
                        row["token_lengths"] = measure_tokens(row, tokenizer, max_length)
                        for key, value in row["token_lengths"].items():
                            token_values[row["role"]][key].append(value)
                    role_counts[row["role"]][row["split"]] += 1
                    runtime_closures += int(row["checks"].get("runtime_contract_closure", False))
                    streams[row["role"], row["split"]].write(canonical(row) + "\n")
                if progress is not None and (index % 50 == 0 or index == count):
                    progress(f"validated {index}/{count} underlying tasks")
        summaries = _summarize_tokens(token_values)
        manifest = {
            "dataset_name": output.name, "dataset_schema_version": 1,
            "label_origin": "deterministic_program_gold", "seed": seed,
            "underlying_tasks": count, "task_split_counts": dict(task_counts),
            "role_split_counts": {role: dict(values) for role, values in role_counts.items()},
            "total_samples": sum(sum(values.values()) for values in role_counts.values()),
            "stratified_task_counts": dict(sorted(strata.items())),
            "distinct_semantic_target_count_distribution": dict(sorted(target_counts.items())),
            "split_group": "underlying_task_all_role_projections", "semantic_duplicates": 0,
            "interpreter_visible_semantic_duplicates": 0,
            "held_out_regression_instruction_matches": 0,
            "validation": {"all_labels_passed_production_contracts": True, "all_labels_passed_independent_blueprint_audits": True,
                           "runtime_contract_hover_closures": runtime_closures, "model_inference_performed": False, "flight_simulation_performed": False},
            "token_audit": {"completed": tokenizer is not None, "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
                            "tokenizer_sha256": _tokenizer_hashes(tokenizer_path),
                            "assistant_counting": "separately_encoded_completion_including_end_of_turn_template_suffix",
                            "full_length_checks": "both_full_template_and_separate_prompt_completion_encoding",
                            "max_full_sequence_tokens": max_length, "truncation_policy": "reject_never_truncate", "by_role": summaries},
            "source_sha256": _source_hashes(), "data_sha256": _file_hashes(staging),
            "limitations": ["Synthetic positive pilot; no real model success or flight success measured.",
                            "Explicit MUST owner assignments only; no free assignment optimization, failures, replanning or clarification samples.",
                            "Search and search-track pair one distinct target per UAV; mixed tasks vary target count, including zero-target navigation/hover.",
                            "Randomized task-disjoint splits share families/templates/scales; not a held-out-template or held-out-scale evaluation.",
                            "Direct tracking of already locked targets and multi-target serial assignments deferred.",
                            "Existing legacy Fleet V1 SFT loader must be adapted before training these three contracts."],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "README.md").write_text(_readme(manifest), encoding="utf-8")
        # No output directory is published until every sample has passed.
        if output.exists():
            raise FileExistsError(f"destination appeared during generation: {output}")
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


def _readme(manifest: dict) -> str:
    return f"""# 多无人机多任务规划数据集（首批）

共 {manifest['underlying_tasks']} 个独立任务底稿、{manifest['total_samples']} 条角色样本。
覆盖 2 / 4 / 6 / 8 / 10 架无人机、无障碍物世界：坐标导航、原地悬停、搜索、搜索后跟踪、多机混合分工。
`tasks.jsonl` 保存底稿、中文原文、可信实体目录、空间边界、任务归属和划分；每个角色目录包含
`train.jsonl` / `validation.jsonl` / `test.jsonl`。同一个底稿的所有角色样本始终在同一划分。

| 角色 | 输出标签 | train / validation / test |
| --- | --- | --- |
""" + "\n".join(
        f"| {role} | {contract} | " + " / ".join(str(manifest['role_split_counts'][role].get(split, 0)) for split in SPLITS) + " |"
        for role, contract in zip(ROLES, ("FleetTaskSpecV1", "FleetMissionPlanV2", "SkillPlanDraftV3"))
    ) + """

每行 `messages` 是生产规划器实际构造的 system / user 提示与完整 assistant JSON 标签，
采用程序生成并经校验的答案，没有调用 Qwen 生成答案，也没有训练权重。
训练只监督 assistant 答案；sample_id、底稿、checks、token_lengths 等元数据不能加入模型输入。
response_schema_sha256 记录该请求的结构化输出约束，可用任务底稿及对应源码重新构建。

每条标签通过当前生产解析器、Fleet 分配检查及局部编译器；独立检查另从底稿核对目标、归属、
坐标、时长、动作顺序及本机返航降落。manifest.json 含逐角色长度统计、数据/源码哈希和检查范围。
这些检查证明标签符合所列接口和底稿，不代表 Qwen 已会规划或飞行执行已成功，也不验证机间碰撞。

悬停任务只抽取用户表达的 WAIT，不额外声称用户要求返航。当前执行接口仍要求局部计划含
TAKEOFF → HOVER → 本机 home GOTO → LAND；该收尾明确标为 runtime_contract_closure。
它属于模型输出所需的运行契约，不是编译器事后自动补齐。这样十机悬停仍仅占十个 termination goals。
其余任务由原文明确要求返航并在本机起点降落。

这是平衡的正样本初版：MUST 指定归属、每机一个 assignment、每 assignment 至多一个语义目标。
导航/悬停没有语义目标，混合任务的无人机数与目标数不同；纯搜索系列仍一机一目标。
数据没有训练自由分配优化、故障重规划、澄清、跨机同步、已锁定目标的直接追踪。
全部划分共享任务类型与模板池；不能据此宣称未见模板或十机数量泛化。
既有 fleet_open_scaling_2_to_10.json 五档任务作为额外回归测试保留，没有导入本数据集。

使用本地 Qwen tokenizer 的 chat template 对完整提示和答案计数，超长即报错，绝不截尾。
注意 manifest 中 exceeds_runtime_default_output_budget：训练序列可容纳不代表默认推理输出预算足够。
当前旧 Fleet V1 训练入口不能直接读取这些三角色样本；正式开训前需接入新数据契约和拒绝截断的 collator，
再根据 token_audit 与显存测量设置训练长度、服务上下文和各角色输出预算。

从 uav_agent 目录重新生成到一个不存在的新目录：

```bash
/home/amax/miniconda3/envs/qwen_vllm/bin/python scripts/generate_planning_roles_dataset.py --output ../datasets/planning_roles_v2 --count 1000 --seed 42
```

重新读取磁盘数据、重放全部生产检查并核对 tokenizer 长度：

```bash
/home/amax/miniconda3/envs/qwen_vllm/bin/python scripts/generate_planning_roles_dataset.py --validate-only ../datasets/planning_roles_v1
```
"""


def validate_dataset(directory: Path, *, progress: Callable[[str], None] | None = None) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if _file_hashes(directory) != manifest["data_sha256"]:
        raise ValueError("persisted JSONL hashes differ from manifest")
    if _source_hashes() != manifest["source_sha256"]:
        raise ValueError("generator or production source changed; regenerate and review the dataset")
    tasks = [json.loads(line) for line in (directory / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(tasks) != manifest["underlying_tasks"]:
        raise ValueError("task count differs from manifest")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate task IDs")
    if dict(Counter(task["split"] for task in tasks)) != manifest["task_split_counts"]:
        raise ValueError("task split counts differ from manifest")
    strata = Counter(f"{task['family']}/{task['scale']}/{task['split']}" for task in tasks)
    if dict(strata) != manifest["stratified_task_counts"]:
        raise ValueError("task strata differ from manifest")
    if dict(Counter(str(len(task["target_catalog"])) for task in tasks)) != manifest["distinct_semantic_target_count_distribution"]:
        raise ValueError("target count distribution differs from manifest")
    if any(task["instruction"] in _regression_instructions() for task in tasks):
        raise ValueError("held-out regression instruction leaked into dataset")
    fingerprints = set()
    instruction_fingerprints = set()
    by_task = defaultdict(dict)
    counts = {role: Counter() for role in ROLES}
    runtime_closures = 0
    token_values = {role: defaultdict(list) for role in ROLES}
    for role in ROLES:
        for split in SPLITS:
            with (directory / role / f"{split}.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["role"] != role or row["split"] != split or row["sample_id"] in by_task[row["task_id"]]:
                        raise ValueError("row role/split mismatch or duplicate sample ID")
                    by_task[row["task_id"]][row["sample_id"]] = row
                    counts[role][split] += 1
                    runtime_closures += int(row["checks"].get("runtime_contract_closure", False))
    if {role: dict(values) for role, values in counts.items()} != manifest["role_split_counts"]:
        raise ValueError("role counts differ from manifest")
    if sum(sum(values.values()) for values in counts.values()) != manifest["total_samples"]:
        raise ValueError("total sample count differs from manifest")
    if manifest["validation"] != {
        "all_labels_passed_production_contracts": True, "all_labels_passed_independent_blueprint_audits": True,
        "runtime_contract_hover_closures": runtime_closures,
        "model_inference_performed": False, "flight_simulation_performed": False,
    }:
        raise ValueError("validation metadata differs from actual label checks")
    audit = manifest["token_audit"]
    if _tokenizer_hashes(Path(audit["tokenizer_path"]) if audit["completed"] else None) != audit["tokenizer_sha256"]:
        raise ValueError("local tokenizer files changed")
    tokenizer = load_tokenizer(Path(audit["tokenizer_path"])) if audit["completed"] else None
    for index, task in enumerate(tasks, 1):
        if task["semantic_hash"] in fingerprints or task["instruction_semantic_hash"] in instruction_fingerprints:
            raise ValueError("duplicate task semantics across dataset")
        fingerprints.add(task["semantic_hash"])
        instruction_fingerprints.add(task["instruction_semantic_hash"])
        candidates = by_task.pop(task["task_id"])
        expected_ids = {row["sample_id"] for row in render_task(task, candidates)}
        if set(candidates) != expected_ids:
            raise ValueError("missing or extra role projections")
        if tokenizer is not None:
            for row in candidates.values():
                measured = measure_tokens(row, tokenizer, audit["max_full_sequence_tokens"])
                if measured != row["token_lengths"]:
                    raise ValueError(f"{row['sample_id']}: token audit changed")
                for key, value in measured.items():
                    token_values[row["role"]][key].append(value)
        if progress is not None and (index % 50 == 0 or index == len(tasks)):
            progress(f"replayed {index}/{len(tasks)} persisted tasks")
    if by_task:
        raise ValueError("unknown task IDs in role files")
    if _summarize_tokens(token_values) != audit["by_role"]:
        raise ValueError("manifest token length summaries differ from measured lengths")
    return {"passed": True, "underlying_tasks": len(tasks), "samples": manifest["total_samples"],
            "all_persisted_answers_replayed": True, "token_lengths_rechecked": tokenizer is not None}
