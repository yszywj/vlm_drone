"""Independent mission blueprints and production-contract gold labels.

Blueprints are sampled before any natural-language rendering or role output is
constructed. Splits group by a semantic digest which excludes paraphrases, clause
order, task IDs and split labels. No simulator state, model calls or generated
model answers are used to establish the gold goals and owners.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from hashlib import sha256
import json
import random

from fleet.schemas_v2 import validate_fleet_mission_plan_v2
from fleet.task_spec import (
    AssignmentConstraint,
    FleetTaskSpecV1,
    MissionGoal,
    OrderingConstraint,
    SourceEvidence,
    TerminationGoal,
)
from fleet.types import FleetCoordinationPolicy, FleetUavCapability
from fleet.types_v2 import FleetAssignmentV2, FleetMissionPlanV2, FleetMissionRequestV2
from planner.spatial import CircleRegion, PointTarget
from target.types import TargetSpec


FAMILIES = ("navigate", "hover", "search", "search_track", "mixed_parallel")
SCALES = (2, 4, 6, 8, 10)
SPLITS = ("train", "validation", "test")
_COLORS = ("red", "blue", "green", "yellow", "orange", "purple", "cyan", "magenta", "white", "gray")
_SEMANTIC_ASSIGNMENT_FIELDS = (
    "uav_id", "kind", "target_alias", "destination_xyz_m", "search_center_xyz_m",
    "radius_m", "duration_s", "terminal_goal_type", "closure_policy",
)


def semantic_hash(task: Mapping[str, object]) -> str:
    """Hash actual mission semantics, independent of presentation and split."""

    semantic = {
        "family": task["family"],
        "scale": task["scale"],
        "world": task["world"],
        "closure_policy": task["closure_policy"],
        "uavs": sorted(
            [
                {key: uav[key] for key in ("uav_id", "home_name", "home_xyz_m")}
                for uav in task["uavs"]
            ],
            key=lambda uav: uav["uav_id"],
        ),
        "targets": {
            alias: {
                key: value[key]
                for key in ("category", "hard_attributes", "immutable_identity_summary")
            }
            for alias, value in task["target_catalog"].items()
        },
        "assignments": sorted(
            [
                {key: assignment[key] for key in _SEMANTIC_ASSIGNMENT_FIELDS}
                for assignment in task["assignments"]
            ],
            key=lambda assignment: assignment["uav_id"],
        ),
    }
    encoded = json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def instruction_semantic_hash(task: Mapping[str, object]) -> str:
    """Hash only goal facts visible to the mission interpreter.

    Different trusted homes, flight defaults, appearance catalogs or wording
    must not make the same interpreted mission appear novel across splits.
    Explicit navigation altitude remains part of the mission's destination.
    """

    facts = []
    for assignment in task["assignments"]:
        kind = assignment["kind"]
        fact = {
            "uav_id": assignment["uav_id"],
            "kind": kind,
            "terminal_goal_type": assignment["terminal_goal_type"],
        }
        if kind == "navigate":
            fact["destination_xyz_m"] = [float(value) for value in assignment["destination_xyz_m"]]
        if kind in ("search", "search_track"):
            fact["target_alias"] = assignment["target_alias"]
            fact["search_center_xyz_m"] = [float(value) for value in assignment["search_center_xyz_m"]]
            fact["radius_m"] = float(assignment["radius_m"])
        if kind in ("hover", "search_track"):
            fact["duration_s"] = float(assignment["duration_s"])
        facts.append(fact)
    encoded = json.dumps(
        sorted(facts, key=lambda item: item["uav_id"]),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _number(value: float) -> str:
    return f"{float(value):g}"


def _duration(value: float, variant: int) -> str:
    # Use only exactly terminating minute conversions to avoid rounded labels.
    if variant == 2 and value in (15, 30, 45, 60):
        return f"{_number(value / 60)}分钟"
    return f"{_number(value)}秒"


def _render_assignment(assignment: dict, uav: dict, target_names: dict, variant: int) -> str:
    name = uav["display_name"]
    kind = assignment["kind"]
    if kind == "navigate":
        point = ",".join(_number(value) for value in assignment["destination_xyz_m"])
        action = (
            f"飞往世界坐标({point})米的指定位置",
            f"到达世界坐标({point})米处",
            f"导航至世界坐标({point})米",
            f"前往世界坐标({point})米的航点",
        )[variant]
    elif kind == "hover":
        duration = _duration(assignment["duration_s"], variant)
        action = (
            f"起飞后在原地悬停{duration}",
            f"起飞并保持当前位置悬停{duration}",
            f"起飞后执行原地悬停，持续{duration}",
            f"起飞后原地保持悬停状态{duration}",
        )[variant]
    else:
        x, y, _ = assignment["search_center_xyz_m"]
        radius = assignment["radius_m"]
        radius_text = (
            f"{_number(radius * 100)}厘米" if variant == 2 else f"{_number(radius)}米"
        )
        target = target_names[assignment["target_alias"]]
        action = (
            f"在以世界坐标({_number(x)},{_number(y)})米为中心、半径{radius_text}的圆形区域搜索{target}",
            f"搜索{target}，搜索范围为世界坐标({_number(x)},{_number(y)})米周围半径{radius_text}的圆",
            f"搜寻{target}，限定在中心为世界坐标({_number(x)},{_number(y)})米、半径{radius_text}的圆形区域内",
            f"在以世界坐标({_number(x)},{_number(y)})米为中心、半径{radius_text}的圆形区域内寻找{target}",
        )[variant]
        if kind == "search_track":
            duration = _duration(assignment["duration_s"], variant)
            action += (
                f"，找到后跟踪该目标{duration}",
                f"；发现后继续跟踪{duration}",
                f"，随后对找到的目标跟踪{duration}",
                f"，搜到目标后跟踪{duration}",
            )[variant]
    if assignment["terminal_goal_type"] == "RETURN_HOME_AND_LAND":
        action += (
            "，完成后返回自己的起点并降落",
            "；随后回到自己的起飞点并降落",
            "，结束后各自返航到起点并降落",
            "；执行完毕后返回自身起飞位置并降落",
        )[variant]
    return (f"{name}：{action}。" if variant == 2 else f"{name}{action}。")


def _render_task(task: dict) -> None:
    variant = int(task["language_variant"])
    uavs = {uav["uav_id"]: uav for uav in task["uavs"]}
    target_names = {
        alias: f"目标{alias.removeprefix('target_').upper()}"
        for alias in task["target_catalog"]
    }
    quotes = {
        assignment["uav_id"]: _render_assignment(
            assignment, uavs[assignment["uav_id"]], target_names, variant
        )
        for assignment in task["assignments"]
    }
    headers = (
        f"共有{task['scale']}架无人机，以下任务同时开始，各机严格执行自己的任务：",
        f"请让{task['scale']}架无人机并行执行以下安排，不要交换无人机与任务的对应关系：",
        f"为{task['scale']}架无人机安排如下任务，所有无人机并行执行，必须遵守逐机绑定：",
        f"本次出动{task['scale']}架无人机，各机独立并行工作，任务分工如下：",
    )
    instruction = headers[variant] + "".join(quotes[uav_id] for uav_id in task["clause_order"])
    task["instruction"] = instruction
    task["source_quotes"] = quotes


def _sample_task(family: str, scale: int, rng: random.Random) -> dict:
    altitude = float(rng.choice((8, 10, 12, 15)))
    homes = [
        (x + rng.randint(-2, 2), y + rng.randint(-2, 2))
        for x in (-60, -30, 0, 30, 60)
        for y in (-40, 40)
    ]
    rng.shuffle(homes)
    uavs = []
    targets = {}
    uav_aliases = {}
    target_aliases = {}
    for index in range(scale):
        letter = chr(ord("a") + index)
        uav_id = f"uav_{letter}"
        display_name = f"无人机{letter.upper()}"
        x, y = homes[index]
        uavs.append({
            "uav_id": uav_id,
            "display_name": display_name,
            "home_name": f"home_{uav_id}",
            "home_xyz_m": [float(x), float(y), 0.0],
        })
        uav_aliases[uav_id] = uav_id
        uav_aliases[display_name] = uav_id
        alias = f"target_{letter}"
        target_aliases[alias] = alias
        target_aliases[f"目标{letter.upper()}"] = alias
        targets[alias] = TargetSpec(
            original_description=f"{_COLORS[index]} cube",
            category="cube",
            hard_attributes=(f"color={_COLORS[index]}",),
            immutable_identity_summary=f"{_COLORS[index]} cube",
        ).to_dict()
    if family == "mixed_parallel":
        kinds = list(rng.sample(FAMILIES[:4], min(scale, 4)))
        hover_limit = max(1, scale // 4)
        while len(kinds) < scale:
            eligible = [
                kind for kind in FAMILIES[:4]
                if kind != "hover" or kinds.count("hover") < hover_limit
            ]
            kinds.append(rng.choice(eligible))
        rng.shuffle(kinds)
    else:
        kinds = [family] * scale
    target_order = list(targets)
    rng.shuffle(target_order)
    assignments = []
    for index, (uav, kind) in enumerate(zip(uavs, kinds, strict=True)):
        home_x, home_y, _ = uav["home_xyz_m"]
        # Keep each search/navigation area near its own separated home cell.
        center_x = home_x + rng.randint(-4, 4)
        center_y = home_y + (1 if home_y < 0 else -1) * rng.randint(12, 20)
        assignments.append({
            "uav_id": uav["uav_id"],
            "kind": kind,
            "target_alias": target_order[index] if kind in ("search", "search_track") else None,
            "destination_xyz_m": [center_x, center_y, altitude] if kind == "navigate" else None,
            "search_center_xyz_m": [center_x, center_y, 0.0] if kind in ("search", "search_track") else None,
            "radius_m": float(rng.choice((4, 5, 6))) if kind in ("search", "search_track") else None,
            "duration_s": float(rng.choice((5, 10, 15, 20, 25, 30, 40, 45, 50, 60))) if kind in ("hover", "search_track") else None,
            "terminal_goal_type": "WAIT" if kind == "hover" else "RETURN_HOME_AND_LAND",
            "closure_policy": "runtime_contract_home_and_land" if kind == "hover" else "explicit_return_home_and_land",
        })
    clause_order = [uav["uav_id"] for uav in uavs]
    rng.shuffle(clause_order)
    used_targets = {
        assignment["target_alias"] for assignment in assignments
        if assignment["target_alias"] is not None
    }
    targets = {alias: value for alias, value in targets.items() if alias in used_targets}
    target_aliases = {
        alias: canonical for alias, canonical in target_aliases.items()
        if canonical in used_targets
    }
    task = {
        "family": family,
        "scale": scale,
        "language_variant": rng.randrange(4),
        "uavs": uavs,
        "uav_aliases": uav_aliases,
        "target_aliases": target_aliases,
        "target_catalog": targets,
        "assignments": assignments,
        "world": {
            "scene_min_xyz_m": [-100.0, -100.0, 0.0],
            "scene_max_xyz_m": [100.0, 100.0, 40.0],
            "flight_altitude_m": altitude,
        },
        "closure_policy": (
            "runtime_contract_home_and_land" if family == "hover"
            else "mixed_explicit_and_runtime_contract" if family == "mixed_parallel"
            else "explicit_return_home_and_land"
        ),
        "clause_order": clause_order,
    }
    _render_task(task)
    task["semantic_hash"] = semantic_hash(task)
    task["instruction_semantic_hash"] = instruction_semantic_hash(task)
    return task


def _assign_splits(tasks: list[dict]) -> None:
    groups = defaultdict(list)
    for task in tasks:
        groups[(task["family"], task["scale"])].append(task)
    proportions = {"train": 0.8, "validation": 0.1, "test": 0.1}
    desired = {
        "train": len(tasks) * 8 // 10,
        "validation": len(tasks) // 10,
        "test": len(tasks) - len(tasks) * 8 // 10 - len(tasks) // 10,
    }
    allocations = {
        key: {split: int(len(values) * proportions[split]) for split in SPLITS}
        for key, values in groups.items()
    }
    remaining = {
        split: desired[split] - sum(counts[split] for counts in allocations.values())
        for split in SPLITS
    }
    for key, values in sorted(groups.items()):
        counts = allocations[key]
        while sum(counts.values()) < len(values):
            eligible = [split for split in SPLITS if remaining[split] > 0]
            chosen = max(eligible, key=lambda split: (
                len(values) * proportions[split] - counts[split], remaining[split],
            ))
            counts[chosen] += 1
            remaining[chosen] -= 1
        # Digest ordering groups all semantic duplicates before assignment.
        # Generation rejects duplicates altogether; paraphrases must retain the
        # resulting split instead of being randomly re-split by downstream roles.
        ordered = sorted(values, key=lambda item: item["semantic_hash"])
        offset = 0
        for split in SPLITS:
            for task in ordered[offset:offset + counts[split]]:
                task["split"] = split
            offset += counts[split]


def generate_task_blueprints(count: int = 1000, seed: int = 42) -> list[dict]:
    """Sample balanced missions, unique by semantics, with task-level 80/10/10 splits."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    strata = [(family, scale) for family in FAMILIES for scale in SCALES]
    tasks = []
    seen = set()
    seen_instructions = set()
    for index in range(count):
        family, scale = strata[index % len(strata)]
        for _ in range(100):
            task = _sample_task(family, scale, rng)
            if (
                task["semantic_hash"] not in seen
                and task["instruction_semantic_hash"] not in seen_instructions
            ):
                break
        else:
            raise RuntimeError("could not sample a unique task blueprint")
        task["task_id"] = f"task_{index + 1:06d}"
        seen.add(task["semantic_hash"])
        seen_instructions.add(task["instruction_semantic_hash"])
        tasks.append(task)
    _assign_splits(tasks)
    return tasks


def _goal_ids(assignment: Mapping[str, object]) -> tuple[str, ...]:
    prefix = f"goal_{assignment['uav_id']}"
    kind = assignment["kind"]
    if kind == "navigate":
        result = [f"{prefix}_navigate"]
    elif kind == "hover":
        result = [f"{prefix}_wait"]
    elif kind == "search":
        result = [f"{prefix}_search"]
    elif kind == "search_track":
        result = [f"{prefix}_search", f"{prefix}_track"]
    else:
        raise ValueError(f"unsupported blueprint assignment kind: {kind}")
    if assignment["terminal_goal_type"] == "RETURN_HOME_AND_LAND":
        result.append(f"{prefix}_home")
    elif not (kind == "hover" and assignment["terminal_goal_type"] == "WAIT"):
        raise ValueError("unsupported blueprint terminal goal")
    return tuple(result)


def build_task_spec(task: dict) -> FleetTaskSpecV1:
    """Compile independent blueprint semantics to a strictly parsed TaskSpec."""

    goals = []
    termination_goals = []
    constraints = []
    ordering = []
    evidence = []
    for assignment in task["assignments"]:
        uav_id = assignment["uav_id"]
        kind = assignment["kind"]
        goal_ids = _goal_ids(assignment)
        refs = (f"evidence_{uav_id}",)
        evidence.append(SourceEvidence(refs[0], task["source_quotes"][uav_id]))
        if kind == "navigate":
            goals.append(MissionGoal(
                goal_ids[0], "NAVIGATE", None,
                PointTarget("WORLD_ENU", tuple(assignment["destination_xyz_m"])),
                None, None, "MUST", refs,
            ))
        elif kind == "hover":
            termination_goals.append(TerminationGoal(
                goal_ids[0], "WAIT", uav_id, assignment["duration_s"], "MUST", refs,
            ))
        else:
            goals.append(MissionGoal(
                goal_ids[0], "SEARCH_TARGET", assignment["target_alias"],
                CircleRegion("WORLD_ENU", tuple(assignment["search_center_xyz_m"]), assignment["radius_m"]),
                None, None, "MUST", refs,
            ))
            if kind == "search_track":
                goals.append(MissionGoal(
                    goal_ids[1], "TRACK_TARGET", assignment["target_alias"], None,
                    assignment["duration_s"], None, "MUST", refs,
                ))
        if assignment["terminal_goal_type"] == "RETURN_HOME_AND_LAND":
            termination_goals.append(TerminationGoal(
                goal_ids[-1], "RETURN_HOME_AND_LAND", uav_id, None, "MUST", refs,
            ))
        constraints.append(AssignmentConstraint(
            f"binding_{uav_id}", uav_id, goal_ids, "MUST", refs,
        ))
        for index, (before, after) in enumerate(zip(goal_ids, goal_ids[1:])):
            ordering.append(OrderingConstraint(
                f"order_{uav_id}_{index + 1}", before, after, "MUST", refs,
            ))
    spec = FleetTaskSpecV1(
        source_text=task["instruction"],
        goals=tuple(goals),
        assignment_constraints=tuple(constraints),
        ordering_constraints=tuple(ordering),
        termination_goals=tuple(termination_goals),
        source_evidence=tuple(evidence),
    )
    return FleetTaskSpecV1.from_dict(
        spec.to_dict(),
        trusted_uav_ids=tuple(uav["uav_id"] for uav in task["uavs"]),
        trusted_target_aliases=tuple(task["target_catalog"]),
        supported_coordinate_frames=("WORLD_ENU",),
        expected_source_text=task["instruction"],
    )


def build_fleet_request(task: dict, spec: FleetTaskSpecV1 | None = None) -> FleetMissionRequestV2:
    """Project trusted semantic inventory without injecting target coordinates."""

    task_spec = build_task_spec(task) if spec is None else spec
    task_spec = FleetTaskSpecV1.from_dict(
        task_spec.to_dict(),
        trusted_uav_ids=tuple(uav["uav_id"] for uav in task["uavs"]),
        trusted_target_aliases=tuple(task["target_catalog"]),
        supported_coordinate_frames=("WORLD_ENU",),
        expected_source_text=task["instruction"],
    )
    request = FleetMissionRequestV2(
        fleet_mission_id=f"fleet_{task['task_id']}",
        fleet_plan_version=1,
        task_spec=task_spec,
        uav_inventory=tuple(FleetUavCapability(
            uav_id=uav["uav_id"],
            display_name=uav["display_name"],
            available=True,
            home_name=uav["home_name"],
            max_speed_mps=6.0,
            max_altitude_m=task["world"]["scene_max_xyz_m"][2],
            camera_modalities=("RGB",),
            payload_capabilities=(),
            remaining_energy_ratio=1.0,
        ) for uav in task["uavs"]),
        trusted_fleet_state=(),
        coordination_policy=FleetCoordinationPolicy(minimum_uav_separation_m=5.0),
    )
    return FleetMissionRequestV2.from_dict(request.to_dict())


def build_fleet_plan(task: dict, request: FleetMissionRequestV2) -> FleetMissionPlanV2:
    """Assign every blueprint goal, including WAIT and return/landing, once."""

    if request.fleet_mission_id != f"fleet_{task['task_id']}":
        raise ValueError("request belongs to a different blueprint mission")
    if request.task_spec.source_text != task["instruction"]:
        raise ValueError("request source does not match blueprint instruction")
    plan = FleetMissionPlanV2(
        fleet_mission_id=request.fleet_mission_id,
        fleet_plan_version=request.fleet_plan_version,
        assignments=tuple(FleetAssignmentV2(
            assignment_id=f"assignment_{assignment['uav_id']}",
            uav_id=assignment["uav_id"],
            goal_ids=_goal_ids(assignment),
            priority=100,
            start_policy="PARALLEL",
        ) for assignment in task["assignments"]),
        coordination_policy=request.coordination_policy,
    )
    plan = FleetMissionPlanV2.from_dict(plan.to_dict(), request=request)
    validate_fleet_mission_plan_v2(plan, request)
    if plan.semantic_findings(request):
        raise ValueError("gold blueprint assignment has semantic findings")
    if set(request.task_spec.all_goal_ids) != {
        goal_id for assignment in plan.assignments for goal_id in assignment.goal_ids
    }:
        raise ValueError("gold blueprint assignment does not cover every goal")
    return plan


__all__ = [
    "FAMILIES", "SCALES", "SPLITS", "generate_task_blueprints", "semantic_hash",
    "instruction_semantic_hash",
    "build_task_spec", "build_fleet_request", "build_fleet_plan",
]
