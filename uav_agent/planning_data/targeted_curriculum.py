"""Fresh, independently audited task blueprints for interpreter weaknesses.

This module never reads evaluation cases or model outputs.  It samples new
mission semantics, then renders partition-specific language.  The caller owns
task IDs and persisted splits; the requested partition selects a disjoint
template pool and an explicit-coordinate fractional part.  Thus changing only
homes, presentation, or metadata cannot manufacture cross-partition novelty.

Labels remain the existing trusted builders' responsibility.  In particular a
WAIT-only assignment has one goal and no ordering edge, and every search-track
assignment retains SEARCH -> TRACK -> RETURN_HOME_AND_LAND.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from math import dist, isfinite
import random
import re

from planning_data.tasks import SCALES, _sample_task, instruction_semantic_hash, semantic_hash


FOCI = ("mixed_wait", "axes_zero", "mixed_ownership", "duration_contrast", "minute_conversion")
PARTITIONS = ("train", "validation", "test")
_FRACTIONS = {"train": 0.25, "validation": 0.5, "test": 0.75}
_TEMPLATES = {"train": (0, 1), "validation": (2, 3), "test": (4, 5)}
_HEADERS = (
    "本轮{scale}架无人机并行作业，各条安排只属于该条指定的无人机：",
    "下面列出{scale}架无人机的独立任务，请同时启动，保持逐机归属：",
    "给{scale}架无人机下达以下分工，各机并行执行自己的整条指令：",
    "任务清单涉及{scale}架无人机，清单次序不表示机间先后，全部并行执行：",
    "请同时派出{scale}架无人机，按下面点名的分工各自行动：",
    "此次{scale}架无人机一起开始执行，各项任务以所写机名为准：",
)
_COORDINATE = re.compile(r"世界坐标\(([-+\d., ]+)\)米")
_DURATION = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(秒|分钟)")
_RADIUS = re.compile(r"半径(\d+(?:\.\d+)?)(米|厘米)")


def _number(value: float) -> str:
    return f"{value:g}"


def _duration_text(assignment: dict) -> str:
    value = Decimal(str(assignment["duration_s"]))
    unit = assignment["curriculum_duration_unit"]
    if unit == "分钟":
        value /= 60
    return f"{value.normalize():f}{unit}"


def _render_clause(assignment: dict, template: int) -> str:
    name = "无人机" + assignment["uav_id"].removeprefix("uav_").upper()
    kind = assignment["kind"]
    if kind == "navigate":
        point = ",".join(_number(v) for v in assignment["destination_xyz_m"])
        coordinate = f"世界坐标({point})米"
        action = (
            f"前往{coordinate}处的指定航点",
            f"导航到{coordinate}给定的位置",
            f"飞到{coordinate}所标的位置",
            f"以{coordinate}为目的地并到达该点",
            f"抵达指定位置，位置为{coordinate}",
            f"飞往给定航点，其位置是{coordinate}",
        )[template]
    elif kind == "hover":
        duration = _duration_text(assignment)
        action = (
            f"起飞后保持原位悬停{duration}",
            f"起飞后在当前位置悬停，时间为{duration}",
            f"起飞，然后原地悬停{duration}",
            f"起飞后原地悬停，持续时间设为{duration}",
            f"起飞并在原处悬停{duration}",
            f"起飞后将原地悬停保持{duration}",
        )[template]
    else:
        x, y, _ = assignment["search_center_xyz_m"]
        coordinate = f"世界坐标({_number(x)},{_number(y)})米"
        radius = _number(assignment["radius_m"])
        target = "目标" + assignment["target_alias"].removeprefix("target_").upper()
        action = (
            f"在中心为{coordinate}、半径{radius}米的圆形区域搜索{target}",
            f"搜寻{target}，范围是以{coordinate}为中心、半径{radius}米的圆形区域",
            f"寻找{target}，搜索区域为中心位于{coordinate}且半径{radius}米的圆",
            f"在以{coordinate}为圆心、半径{radius}米的圆内搜索{target}",
            f"查找{target}，仅在以{coordinate}为中心、半径{radius}米的圆形范围内搜索",
            f"搜索{target}，指定圆形区域的圆心是{coordinate}，半径{radius}米",
        )[template]
        if kind == "search_track":
            duration = _duration_text(assignment)
            action += (
                f"，发现后跟踪该目标{duration}",
                f"；找到它以后继续跟踪{duration}",
                f"，搜到该目标后再跟踪{duration}",
                f"；随后跟踪找到的目标，持续{duration}",
                f"，目标一旦找到便跟踪{duration}",
                f"；找到后对同一目标跟踪{duration}",
            )[template]
    if kind != "hover":
        action += (
            "，执行完后返回自己的起点并降落",
            "；完成后回到自己的起飞点并降落",
            "，任务完成后各自返航到起飞点并降落",
            "；结束后返回本机起飞位置并降落",
            "，做完后回到自身起点并降落",
            "；最后返航至自己的起飞点并降落",
        )[template]
    return (f"{name}负责{action}。" if template % 2 else f"{name}：{action}。")


def _assignment_kinds(focus: str, scale: int, ordinal: int, rng: random.Random) -> list[str]:
    if focus == "mixed_wait":
        kinds = ["hover"] * max(1, scale // 3)
        kinds += ["search_track", "navigate", "search"] * scale
    elif focus == "mixed_ownership":
        kinds = ["search_track", "navigate", "hover", "search"]
        kinds += ["search_track", "search_track", "navigate"] * scale
    elif focus in {"duration_contrast", "minute_conversion"}:
        kinds = ["search_track", "search_track", "hover", "navigate", "search"]
        kinds += ["search_track", "hover", "navigate"] * scale
    elif ordinal % 3 == 0:
        kinds = ["search_track"] * scale
    elif ordinal % 3 == 1:
        kinds = ["navigate"] * scale
    else:
        kinds = ["search", "search_track", "navigate"] * scale
    kinds = kinds[:scale]
    rng.shuffle(kinds)
    return kinds


def _sample(focus: str, scale: int, ordinal: int, partition: str, rng: random.Random) -> dict:
    # Reuse the established trusted world/home sampler; no old task is loaded.
    task = _sample_task("search_track", scale, rng)
    kinds = _assignment_kinds(focus, scale, ordinal, rng)
    fraction = _FRACTIONS[partition]
    altitude = task["world"]["flight_altitude_m"]
    target_ids = [f"target_{chr(ord('a') + index)}" for index in range(scale)]
    # A nonzero cyclic shift prevents owner-letter copying from becoming gold.
    shift = rng.randrange(1, scale)
    target_ids = target_ids[shift:] + target_ids[:shift]
    rng.shuffle(task["clause_order"])
    natural_order = [uav["uav_id"] for uav in task["uavs"]]
    if task["clause_order"] == natural_order:
        task["clause_order"] = natural_order[1:] + natural_order[:1]
    axis_slots = [(0, -20), (0, 20), (-40, 0), (40, 0), (0, -60), (0, 60), (-80, 0), (80, 0), (-20, 0), (20, 0)]
    # Shuffle owners over axis locations, retaining paired signs for scale two.
    axis_slots = axis_slots[:scale]
    axis_jitter = rng.randint(-4, 4)
    axis_slots = [
        ((x + (axis_jitter if x > 0 else -axis_jitter)) if x else 0,
         (y + (axis_jitter if y > 0 else -axis_jitter)) if y else 0)
        for x, y in axis_slots
    ]
    if ordinal % 2:
        axis_slots = [(y, x) for x, y in axis_slots]
    rng.shuffle(axis_slots)
    assignments = []
    for index, (old, kind) in enumerate(zip(task["assignments"], kinds, strict=True)):
        x, y, _ = old["search_center_xyz_m"]
        x += fraction
        if focus == "axes_zero":
            x, y = axis_slots[index]
            if x:
                x += fraction if x > 0 else -fraction
            if y:
                y += fraction if y > 0 else -fraction
        has_target = kind in {"search", "search_track"}
        assignments.append({
            "uav_id": old["uav_id"], "kind": kind,
            "target_alias": target_ids[index] if has_target else None,
            "destination_xyz_m": [float(x), float(y), altitude] if kind == "navigate" else None,
            "search_center_xyz_m": [float(x), float(y), 0.0] if has_target else None,
            "radius_m": float(rng.choice((3, 4, 5))) if has_target else None,
            "duration_s": float(rng.choice((10, 15, 20, 25, 30, 40, 45, 50, 60))) if kind in {"hover", "search_track"} else None,
            "terminal_goal_type": "WAIT" if kind == "hover" else "RETURN_HOME_AND_LAND",
            "closure_policy": "runtime_contract_home_and_land" if kind == "hover" else "explicit_return_home_and_land",
            "curriculum_duration_unit": "秒" if kind in {"hover", "search_track"} else None,
        })
    trackers = [a for a in assignments if a["kind"] == "search_track"]
    if focus == "duration_contrast":
        for assignment, duration in zip(trackers, (20.0, 120.0), strict=False):
            assignment["duration_s"] = duration
    elif focus == "minute_conversion":
        minute_durations = (30.0, 45.0, 90.0, 120.0, 15.0, 60.0, 75.0, 105.0, 150.0)
        first = minute_durations[ordinal % len(minute_durations)]
        trackers[0].update(duration_s=first, curriculum_duration_unit="分钟")
        trackers[1].update(duration_s=20.0 if first != 20 else 120.0)
        for index, assignment in enumerate(trackers[2:], 1):
            duration = minute_durations[(ordinal + index) % len(minute_durations)]
            assignment.update(duration_s=duration, curriculum_duration_unit="分钟" if index % 2 else "秒")
        for assignment in assignments:
            if assignment["kind"] == "hover":
                assignment.update(duration_s=rng.choice((15.0, 30.0, 45.0, 60.0)), curriculum_duration_unit="分钟")
    used_targets = {a["target_alias"] for a in assignments if a["target_alias"] is not None}
    task["target_catalog"] = {alias: value for alias, value in task["target_catalog"].items() if alias in used_targets}
    task["target_aliases"] = {alias: value for alias, value in task["target_aliases"].items() if value in used_targets}
    task["assignments"] = assignments
    unique_kinds = set(kinds)
    task["family"] = kinds[0] if len(unique_kinds) == 1 else "mixed_parallel"
    task["closure_policy"] = (
        "mixed_explicit_and_runtime_contract" if "hover" in kinds else "explicit_return_home_and_land"
    )
    template = _TEMPLATES[partition][ordinal % 2]
    header = _HEADERS[template].format(scale=scale)
    quotes = {a["uav_id"]: _render_clause(a, template) for a in assignments}
    task.update(
        language_variant=template + 4,
        curriculum_focus=focus,
        curriculum_partition=partition,
        curriculum_template_id=f"targeted_{template}",
        curriculum_header=header,
        source_quotes=quotes,
        instruction=header + "".join(quotes[uav_id] for uav_id in task["clause_order"]),
    )
    task["semantic_hash"] = semantic_hash(task)
    task["instruction_semantic_hash"] = instruction_semantic_hash(task)
    return task


def audit_targeted_task(task: dict) -> None:
    """Reject text/blueprint disagreement without invoking the text renderer.

    This validates numeric facts by parsing evidence, including unit conversion,
    coordinate order, owners, targets, and the presence/absence of follow-on
    actions.  Full production parsing and independent plan validation are still
    performed by planning_data.generator.render_task after assigning ID/split.
    """
    partition = task["curriculum_partition"]
    if partition not in PARTITIONS or task["curriculum_focus"] not in FOCI:
        raise ValueError("unknown curriculum partition or focus")
    if task.get("split", partition) != partition:
        raise ValueError("persisted split differs from curriculum partition")
    allowed_templates = {f"targeted_{index}" for index in _TEMPLATES[partition]}
    if task["curriculum_template_id"] not in allowed_templates:
        raise ValueError("template belongs to a different partition")
    uavs = {uav["uav_id"]: uav for uav in task["uavs"]}
    owners = [assignment["uav_id"] for assignment in task["assignments"]]
    if len(owners) != task["scale"] or len(set(owners)) != len(owners) or set(owners) != set(uavs):
        raise ValueError("assignments must cover every UAV exactly once")
    if sorted(task["clause_order"]) != sorted(owners) or set(task["source_quotes"]) != set(owners):
        raise ValueError("clause order and quotes must cover every UAV exactly once")
    if task["instruction"] != task["curriculum_header"] + "".join(task["source_quotes"][owner] for owner in task["clause_order"]):
        raise ValueError("instruction contains missing or extra clauses")
    circles = []
    for assignment in task["assignments"]:
        owner, kind = assignment["uav_id"], assignment["kind"]
        quote = task["source_quotes"][owner]
        expected_name = "无人机" + owner.removeprefix("uav_").upper()
        if re.findall(r"无人机[A-J]", quote) != [expected_name] or task["instruction"].count(quote) != 1:
            raise ValueError("source evidence does not identify its unique owner")
        coordinates = [tuple(float(value.strip()) for value in match.split(",")) for match in _COORDINATE.findall(quote)]
        durations = [Decimal(value) * (60 if unit == "分钟" else 1) for value, unit in _DURATION.findall(quote)]
        targets = re.findall(r"目标[A-J]", quote)
        has_track = bool(re.search(r"跟踪|追踪", quote))
        has_hover = "悬停" in quote
        has_search = bool(re.search(r"搜索|搜寻|寻找|查找", quote))
        has_navigation = bool(re.search(r"前往|导航|飞到|到达|抵达|飞往", quote))
        if (has_track, has_hover, has_search, has_navigation) != (
            kind == "search_track", kind == "hover", kind in {"search", "search_track"}, kind == "navigate",
        ):
            raise ValueError("source action types differ from blueprint")
        expected_duration = [] if assignment["duration_s"] is None else [Decimal(str(assignment["duration_s"]))]
        if durations != expected_duration:
            raise ValueError("source duration or units differ from blueprint")
        if kind == "hover":
            if coordinates or targets or "降落" in quote or "起飞" not in quote:
                raise ValueError("WAIT evidence invents a spatial/target/landing goal")
            if assignment["terminal_goal_type"] != "WAIT" or not 1 <= assignment["duration_s"] <= 60:
                raise ValueError("WAIT blueprint violates the runtime contract")
        else:
            if "降落" not in quote or assignment["terminal_goal_type"] != "RETURN_HOME_AND_LAND":
                raise ValueError("explicit home-and-land termination is missing")
            point = assignment["destination_xyz_m"] if kind == "navigate" else assignment["search_center_xyz_m"]
            expected_coordinates = tuple(point if kind == "navigate" else point[:2])
            if coordinates != [expected_coordinates]:
                raise ValueError("source coordinate axes or values differ from blueprint")
            if not all(isfinite(value) for value in point):
                raise ValueError("coordinates must be finite")
            if not all(task["world"]["scene_min_xyz_m"][axis] <= value <= task["world"]["scene_max_xyz_m"][axis] for axis, value in enumerate(point)):
                raise ValueError("coordinate is outside the world")
            expected_targets = [] if kind == "navigate" else ["目标" + assignment["target_alias"].removeprefix("target_").upper()]
            if targets != expected_targets:
                raise ValueError("source target ownership differs from blueprint")
            radii = [float(value) / (100 if unit == "厘米" else 1) for value, unit in _RADIUS.findall(quote)]
            if radii != ([] if kind == "navigate" else [assignment["radius_m"]]):
                raise ValueError("source search radius differs from blueprint")
            if kind in {"search", "search_track"}:
                radius = assignment["radius_m"]
                if not all(task["world"]["scene_min_xyz_m"][axis] <= point[axis] - radius <= point[axis] + radius <= task["world"]["scene_max_xyz_m"][axis] for axis in range(2)):
                    raise ValueError("search circle is outside the world")
                circles.append((point, radius))
            if kind == "search_track" and not 1 <= assignment["duration_s"] <= 600:
                raise ValueError("TRACK duration violates the runtime contract")
    used_targets = {assignment["target_alias"] for assignment in task["assignments"] if assignment["target_alias"] is not None}
    if used_targets != set(task["target_catalog"]) or used_targets != set(task["target_aliases"].values()):
        raise ValueError("target catalog differs from actually assigned targets")
    if any(dist(a, b) - ra - rb < 5 for index, (a, ra) in enumerate(circles) for b, rb in circles[index + 1:]):
        raise ValueError("search circles are insufficiently separated")
    if task["semantic_hash"] != semantic_hash(task) or task["instruction_semantic_hash"] != instruction_semantic_hash(task):
        raise ValueError("blueprint hashes are stale")


def build_targeted_tasks(
    count_per_focus_scale: int = 16,
    *,
    seed: int = 20260914,
    partition: str = "train",
    exclude_semantic_hashes: Iterable[str] = (),
    exclude_instruction_semantic_hashes: Iterable[str] = (),
) -> list[dict]:
    """Return 5 foci x 5 scales x count fresh blueprints, without IDs/splits.

    Pass hashes from ALL prior tasks, including all validation/test tasks, in
    the exclusion sets.  The caller must retain ``curriculum_partition`` as the
    eventual split and validate every rendered label with ``render_task``.
    Calling each partition separately guarantees fresh language and explicit
    geometry; it is not evidence of generalization to unbounded task families.
    """
    if isinstance(count_per_focus_scale, bool) or not isinstance(count_per_focus_scale, int) or count_per_focus_scale <= 0:
        raise ValueError("count_per_focus_scale must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if partition not in PARTITIONS:
        raise ValueError("partition must be train, validation, or test")
    rng = random.Random(seed)
    seen = set(exclude_semantic_hashes)
    seen_instructions = set(exclude_instruction_semantic_hashes)
    tasks = []
    for focus in FOCI:
        for scale in SCALES:
            for ordinal in range(count_per_focus_scale):
                for _ in range(100):
                    task = _sample(focus, scale, ordinal, partition, rng)
                    audit_targeted_task(task)
                    if task["semantic_hash"] not in seen and task["instruction_semantic_hash"] not in seen_instructions:
                        break
                else:
                    raise RuntimeError("could not sample unique targeted task semantics")
                seen.add(task["semantic_hash"])
                seen_instructions.add(task["instruction_semantic_hash"])
                tasks.append(task)
    return tasks
