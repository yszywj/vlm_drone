"""Trusted config-to-Fleet request assembly without target ground truth leakage."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
import re
from typing import TYPE_CHECKING

from common.ids import generate_routing_id
from fleet.types import (
    AssignmentFailurePolicy,
    FleetCoordinationPolicy,
    FleetMissionRequest,
    FleetStartPolicy,
    FleetTargetRequest,
    FleetUavCapability,
    RouteConflictPolicy,
    TargetClaimPolicy,
)
from fleet.task_spec import FleetTaskSpecV1, validate_task_spec_trust
from fleet.types_v2 import FleetMissionRequestV2, TrustedFleetStateEvidence
from planner.schemas import LandingZoneSpec, PlannerWorldContext
from planner.spatial import CircleRegion, CoordinateFrame, RegionSpec
from target.types import TargetSpec

if TYPE_CHECKING:
    from configs.schema import AppConfig, TargetConfig, UavConfig
    from fleet.types import FleetMissionPlan
    from fleet.types_v2 import FleetMissionPlanV2


class FleetRequestBuildError(ValueError):
    pass


# LandingZoneSpec's public default remains useful for isolated/far-apart UAVs,
# but Fleet recovery needs a tighter bound whenever home disks approach the
# configured separation envelope.  The margin absorbs floating-point and
# discrete controller-step error instead of relying on exact equality at the
# airspace threshold.
DEFAULT_HOME_HORIZONTAL_TOLERANCE_M = 0.75
MIN_HOME_HORIZONTAL_TOLERANCE_M = 0.10
MIN_HOME_SEPARATION_MARGIN_M = 0.10
HOME_SEPARATION_MARGIN_RATIO = 0.05


@dataclass(frozen=True, slots=True)
class ExplicitAssignmentDirective:
    uav_id: str
    target_alias: str
    search_region: RegionSpec
    track_duration_s: float
    priority: int = 100
    start_policy: FleetStartPolicy = FleetStartPolicy.PARALLEL


def build_fleet_inventory(config: "AppConfig") -> tuple[FleetUavCapability, ...]:
    """Project only safe capability fields; no controller/PID values escape."""

    max_altitude = float(config.scene.size_xyz_m[2])
    return tuple(
        FleetUavCapability(
            uav_id=uav.id,
            display_name=uav.display_name or uav.id,
            available=True,
            home_name=uav.home_name or f"home_{uav.id}",
            max_speed_mps=uav.max_speed_mps,
            max_altitude_m=max_altitude,
            camera_modalities=("RGB",),
            payload_capabilities=(),
            remaining_energy_ratio=1.0,
        )
        for uav in config.uavs
    )


def build_target_catalog(config: "AppConfig") -> dict[str, TargetSpec]:
    """Build semantic identity from configured appearance, never GT position/motion."""

    result: dict[str, TargetSpec] = {}
    for target in config.targets:
        appearance = target.appearance
        alias = target.semantic_alias or target.id
        hard = (
            f"color={appearance.color_name}",
            f"shape={appearance.shape}",
        )
        description = f"{appearance.color_name} {appearance.shape.lower()} {alias}"
        result[target.id] = TargetSpec(
            original_description=description,
            category=appearance.shape.lower(),
            hard_attributes=hard,
            immutable_identity_summary=description,
        )
    return result


def parse_explicit_assignment_instruction(
    instruction: str,
    config: "AppConfig",
    *,
    require_all_targets: bool = True,
) -> tuple[ExplicitAssignmentDirective, ...]:
    """Parse only the documented explicit demo grammar.

    This is intentionally not a general Chinese mission parser.  It recognizes
    clauses with a trusted UAV name, WORLD coordinate, radius, target alias,
    and tracking duration so the deterministic Scripted Fleet baseline can be
    invoked from the documented CLI without consulting target ground truth.
    """

    if not isinstance(instruction, str) or not instruction.strip():
        raise FleetRequestBuildError("instruction must be a non-empty string")
    clauses = tuple(part.strip() for part in re.split(r"[；;]", instruction) if part.strip())
    directives: list[ExplicitAssignmentDirective] = []
    seen_uavs: set[str] = set()
    seen_targets: set[str] = set()
    for clause in clauses:
        uav = _match_uav(clause, config)
        target = _match_target(clause, config)
        if uav is None or target is None:
            continue
        coordinate_match = re.search(
            r"世界坐标\s*([负零〇一二两三四五六七八九十百千万点.\-+\d]+)\s*[、,，]\s*"
            r"([负零〇一二两三四五六七八九十百千万点.\-+\d]+)"
            r"(?:\s*[、,，]\s*([负零〇一二两三四五六七八九十百千万点.\-+\d]+))?"
            r"\s*附近\s*([零〇一二两三四五六七八九十百千万点.\d]+)\s*米",
            clause,
        )
        duration_match = re.search(
            r"跟踪[^；;]*?([零〇一二两三四五六七八九十百千万点.\d]+)\s*秒",
            clause,
        )
        if coordinate_match is None or duration_match is None:
            raise FleetRequestBuildError(
                "explicit scripted clause must contain WORLD x/y[/z], radius, and track seconds: "
                + clause
            )
        x = _number(coordinate_match.group(1))
        y = _number(coordinate_match.group(2))
        z = 0.0 if coordinate_match.group(3) is None else _number(coordinate_match.group(3))
        radius = _number(coordinate_match.group(4))
        duration = _number(duration_match.group(1))
        if radius <= 0.0 or duration <= 0.0:
            raise FleetRequestBuildError("radius and tracking duration must be positive")
        if uav.id in seen_uavs:
            raise FleetRequestBuildError(f"UAV appears in multiple active clauses: {uav.id}")
        if target.id in seen_targets:
            raise FleetRequestBuildError(f"target appears in multiple clauses: {target.id}")
        seen_uavs.add(uav.id)
        seen_targets.add(target.id)
        directives.append(
            ExplicitAssignmentDirective(
                uav_id=uav.id,
                target_alias=target.id,
                search_region=CircleRegion(
                    CoordinateFrame.WORLD_ENU,
                    (x, y, z),
                    radius,
                ),
                track_duration_s=duration,
            )
        )
    if require_all_targets:
        missing = sorted({target.id for target in config.targets} - seen_targets)
        if missing:
            raise FleetRequestBuildError(
                "explicit instruction is missing configured targets: " + ", ".join(missing)
            )
    if not directives:
        raise FleetRequestBuildError("no explicit UAV-target clauses were recognized")
    return tuple(directives)


def build_fleet_mission_request(
    config: "AppConfig",
    instruction: str,
    *,
    directives: tuple[ExplicitAssignmentDirective, ...] | None = None,
    parse_explicit: bool = False,
    fleet_mission_id: str | None = None,
    fleet_plan_version: int = 1,
) -> FleetMissionRequest:
    if directives is None and parse_explicit:
        directives = parse_explicit_assignment_instruction(instruction, config)
    directive_by_target = {
        directive.target_alias: directive for directive in (directives or ())
    }
    catalog = build_target_catalog(config)
    target_requests: list[FleetTargetRequest] = []
    for target in config.targets:
        directive = directive_by_target.get(target.id)
        target_requests.append(
            FleetTargetRequest(
                target_alias=target.id,
                target_spec=catalog[target.id],
                requested_uav_id=None if directive is None else directive.uav_id,
                search_region=None if directive is None else directive.search_region,
                track_duration_s=(
                    None if directive is None else directive.track_duration_s
                ),
                priority=None if directive is None else directive.priority,
                start_policy=None if directive is None else directive.start_policy,
                required=True,
            )
        )
    unknown = sorted(set(directive_by_target) - set(catalog))
    if unknown:
        raise FleetRequestBuildError("directives contain unknown targets: " + ", ".join(unknown))
    policy = FleetCoordinationPolicy(
        target_claim_policy=TargetClaimPolicy(config.fleet.target_claim_policy),
        minimum_uav_separation_m=config.fleet.minimum_uav_separation_m,
        route_conflict_policy=RouteConflictPolicy(config.fleet.route_conflict_policy),
        assignment_failure_policy=AssignmentFailurePolicy(
            config.fleet.assignment_failure_policy
        ),
    )
    return FleetMissionRequest(
        fleet_mission_id=fleet_mission_id or generate_routing_id("fleet_mission"),
        fleet_plan_version=fleet_plan_version,
        original_instruction=instruction,
        uav_inventory=build_fleet_inventory(config),
        target_requests=tuple(target_requests),
        coordination_policy=policy,
    )


def build_fleet_mission_request_v2(
    config: "AppConfig",
    task_spec: FleetTaskSpecV1,
    *,
    trusted_fleet_state: tuple[TrustedFleetStateEvidence, ...] = (),
    fleet_mission_id: str | None = None,
    fleet_plan_version: int = 1,
    supported_coordinate_frames: tuple[CoordinateFrame | str, ...] = (
        CoordinateFrame.WORLD_ENU,
        CoordinateFrame.HOME_ENU,
        CoordinateFrame.UAV_START_FLU,
    ),
) -> FleetMissionRequestV2:
    """Build a V2 request from a trusted TaskSpec without fixed grammar parsing."""

    if not isinstance(task_spec, FleetTaskSpecV1):
        raise TypeError("task_spec must be a FleetTaskSpecV1")
    inventory = build_fleet_inventory(config)
    validate_task_spec_trust(
        task_spec,
        trusted_uav_ids=tuple(item.uav_id for item in inventory),
        trusted_target_aliases=tuple(build_target_catalog(config)),
        supported_coordinate_frames=supported_coordinate_frames,
    )
    policy = FleetCoordinationPolicy(
        target_claim_policy=TargetClaimPolicy(config.fleet.target_claim_policy),
        minimum_uav_separation_m=config.fleet.minimum_uav_separation_m,
        route_conflict_policy=RouteConflictPolicy(config.fleet.route_conflict_policy),
        assignment_failure_policy=AssignmentFailurePolicy(
            config.fleet.assignment_failure_policy
        ),
    )
    return FleetMissionRequestV2(
        fleet_mission_id=fleet_mission_id or generate_routing_id("fleet_mission"),
        fleet_plan_version=fleet_plan_version,
        task_spec=task_spec,
        uav_inventory=inventory,
        trusted_fleet_state=trusted_fleet_state,
        coordination_policy=policy,
    )


def build_agent_world_contexts(
    config: "AppConfig",
    plan: "FleetMissionPlan",
) -> dict[str, PlannerWorldContext]:
    """Build per-UAV trusted context without any target initial/motion state."""

    size_x, size_y, size_z = config.scene.size_xyz_m
    uavs = {uav.id: uav for uav in config.uavs}
    landing_tolerances = derive_safe_home_landing_tolerances(config, plan)
    contexts: dict[str, PlannerWorldContext] = {}
    for assignment in plan.assignments:
        try:
            uav = uavs[assignment.uav_id]
        except KeyError:
            raise FleetRequestBuildError(
                f"plan references unknown configured UAV: {assignment.uav_id}"
            ) from None
        home_name = uav.home_name or f"home_{uav.id}"
        contexts[uav.id] = PlannerWorldContext(
            scene_min_xyz_m=(-size_x / 2.0, -size_y / 2.0, 0.0),
            scene_max_xyz_m=(size_x / 2.0, size_y / 2.0, size_z),
            initial_uav_xyz_m=uav.initial_position_xyz_m,
            search_regions={},
            landing_zones={
                home_name: LandingZoneSpec(
                    name=home_name,
                    position_xy_m=(
                        uav.initial_position_xyz_m[0],
                        uav.initial_position_xyz_m[1],
                    ),
                    ground_altitude_m=uav.initial_position_xyz_m[2],
                    description=f"trusted launch/recovery zone for {uav.id}",
                    horizontal_tolerance_m=landing_tolerances[uav.id],
                )
            },
            default_takeoff_altitude_m=min(10.0, size_z),
            default_track_duration_s=assignment.track_duration_s,
            search_timeout_s=config.search.timeout_s,
            goto_timeout_s=120.0,
            land_timeout_s=60.0,
        )
    return contexts


def build_agent_world_contexts_v2(
    config: "AppConfig",
    request: FleetMissionRequestV2,
    plan: "FleetMissionPlanV2",
) -> dict[str, PlannerWorldContext]:
    """Build one trusted, target-GT-free context for each V2 Assignment.

    Unlike the v1 helper, this projection does not assume that every
    Assignment contains SEARCH and TRACK.  A small positive default tracking
    duration is retained only because :class:`PlannerWorldContext` has a
    backwards-compatible required field; the focused V2 prompt carries the
    actual Goal durations and never presents that default as a user Goal.
    """

    from fleet.schemas_v2 import validate_fleet_mission_plan_v2
    from fleet.task_spec import GoalType, MissionGoal

    validate_fleet_mission_plan_v2(plan, request)
    size_x, size_y, size_z = config.scene.size_xyz_m
    uavs = {uav.id: uav for uav in config.uavs}
    landing_tolerances = derive_safe_home_landing_tolerances(config, plan)
    contexts: dict[str, PlannerWorldContext] = {}
    for assignment in plan.assignments:
        try:
            uav = uavs[assignment.uav_id]
        except KeyError:
            raise FleetRequestBuildError(
                f"plan references unknown configured UAV: {assignment.uav_id}"
            ) from None
        home_name = uav.home_name or f"home_{uav.id}"
        assigned_goals = tuple(request.task_spec.goal(item) for item in assignment.goal_ids)
        track_durations = tuple(
            float(goal.duration_s)
            for goal in assigned_goals
            if (
                isinstance(goal, MissionGoal)
                and goal.goal_type is GoalType.TRACK_TARGET
                and goal.duration_s is not None
            )
        )
        default_track_duration_s = (
            max(track_durations)
            if track_durations
            else max(1.0, float(config.planner.min_track_duration_s))
        )
        contexts[uav.id] = PlannerWorldContext(
            scene_min_xyz_m=(-size_x / 2.0, -size_y / 2.0, 0.0),
            scene_max_xyz_m=(size_x / 2.0, size_y / 2.0, size_z),
            initial_uav_xyz_m=uav.initial_position_xyz_m,
            search_regions={},
            landing_zones={
                home_name: LandingZoneSpec(
                    name=home_name,
                    position_xy_m=(
                        uav.initial_position_xyz_m[0],
                        uav.initial_position_xyz_m[1],
                    ),
                    ground_altitude_m=uav.initial_position_xyz_m[2],
                    description=f"trusted launch/recovery zone for {uav.id}",
                    horizontal_tolerance_m=landing_tolerances[uav.id],
                )
            },
            default_takeoff_altitude_m=min(10.0, size_z),
            default_track_duration_s=default_track_duration_s,
            search_timeout_s=config.search.timeout_s,
            goto_timeout_s=120.0,
            land_timeout_s=60.0,
        )
    return contexts


def derive_safe_home_landing_tolerances(
    config: "AppConfig",
    plan: "FleetMissionPlan",
    *,
    default_tolerance_m: float = DEFAULT_HOME_HORIZONTAL_TOLERANCE_M,
    minimum_tolerance_m: float = MIN_HOME_HORIZONTAL_TOLERANCE_M,
) -> dict[str, float]:
    """Derive per-UAV landing disks that cannot overlap separation limits.

    For every active home pair, the conservative invariant is::

        nominal_distance - tolerance_a - tolerance_b
            >= minimum_uav_separation + safety_margin

    The derivation is deterministic and runs during pure-Python preparation,
    before Isaac Sim can be imported or an environment can be started.
    """

    for name, value in (
        ("default_tolerance_m", default_tolerance_m),
        ("minimum_tolerance_m", minimum_tolerance_m),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or value <= 0.0
        ):
            raise FleetRequestBuildError(f"{name} must be greater than zero")
    if minimum_tolerance_m > default_tolerance_m:
        raise FleetRequestBuildError(
            "minimum_tolerance_m must not exceed default_tolerance_m"
        )

    configured_uavs = {uav.id: uav for uav in config.uavs}
    active_uav_ids = tuple(
        sorted({assignment.uav_id for assignment in plan.assignments})
    )
    try:
        active_uavs = tuple(configured_uavs[uav_id] for uav_id in active_uav_ids)
    except KeyError as exc:
        raise FleetRequestBuildError(
            f"plan references unknown configured UAV: {exc.args[0]}"
        ) from None

    home_names: dict[str, str] = {}
    for uav in active_uavs:
        home_name = uav.home_name or f"home_{uav.id}"
        previous = home_names.get(home_name)
        if previous is not None:
            raise FleetRequestBuildError(
                "active UAVs must use distinct landing-zone IDs: "
                f"{previous}, {uav.id} -> {home_name}"
            )
        home_names[home_name] = uav.id

    minimum_separation = float(config.fleet.minimum_uav_separation_m)
    safety_margin = max(
        MIN_HOME_SEPARATION_MARGIN_M,
        minimum_separation * HOME_SEPARATION_MARGIN_RATIO,
    )
    tolerances = {
        uav.id: float(default_tolerance_m) for uav in active_uavs
    }
    for index, first in enumerate(active_uavs):
        for second in active_uavs[index + 1 :]:
            first_xy = first.initial_position_xyz_m[:2]
            second_xy = second.initial_position_xyz_m[:2]
            distance = hypot(
                float(second_xy[0]) - float(first_xy[0]),
                float(second_xy[1]) - float(first_xy[1]),
            )
            usable_tolerance_budget = (
                distance - minimum_separation - safety_margin
            )
            minimum_budget = 2.0 * float(minimum_tolerance_m)
            if usable_tolerance_budget + 1e-12 < minimum_budget:
                required_distance = (
                    minimum_separation + safety_margin + minimum_budget
                )
                raise FleetRequestBuildError(
                    "Fleet homes are too close for safe landing: "
                    f"{first.id}/{second.id} distance={distance:.3f} m, "
                    f"required>={required_distance:.3f} m for "
                    f"minimum_separation={minimum_separation:.3f} m"
                )
            pair_tolerance = usable_tolerance_budget / 2.0
            tolerances[first.id] = min(
                tolerances[first.id], pair_tolerance
            )
            tolerances[second.id] = min(
                tolerances[second.id], pair_tolerance
            )

    # Recheck the derived result as an executable invariant, keeping future
    # edits to the allocation rule fail-closed.
    for index, first in enumerate(active_uavs):
        for second in active_uavs[index + 1 :]:
            distance = hypot(
                float(second.initial_position_xyz_m[0])
                - float(first.initial_position_xyz_m[0]),
                float(second.initial_position_xyz_m[1])
                - float(first.initial_position_xyz_m[1]),
            )
            worst_case_distance = (
                distance - tolerances[first.id] - tolerances[second.id]
            )
            required = minimum_separation + safety_margin
            if worst_case_distance + 1e-12 < required:
                raise FleetRequestBuildError(
                    "derived landing tolerances violate Fleet separation"
                )
    return dict(sorted(tolerances.items()))


def _match_uav(clause: str, config: "AppConfig") -> "UavConfig | None":
    matches = [
        uav
        for uav in config.uavs
        if uav.id in clause or (uav.display_name is not None and uav.display_name in clause)
    ]
    if len(matches) > 1:
        raise FleetRequestBuildError("clause names multiple UAVs: " + clause)
    return None if not matches else matches[0]


def _match_target(clause: str, config: "AppConfig") -> "TargetConfig | None":
    matches = [
        target
        for target in config.targets
        if target.id in clause
        or target.id.replace("_", "") in clause
        or (target.semantic_alias is not None and target.semantic_alias in clause)
    ]
    if len(matches) > 1:
        raise FleetRequestBuildError("clause names multiple targets: " + clause)
    return None if not matches else matches[0]


_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _number(text: str) -> float:
    normalized = text.strip()
    try:
        return float(normalized)
    except ValueError:
        pass
    sign = -1.0 if normalized.startswith("负") else 1.0
    if sign < 0:
        normalized = normalized[1:]
    if "点" in normalized:
        whole, fraction = normalized.split("点", 1)
        fraction_digits = "".join(str(_DIGITS[char]) for char in fraction)
        return sign * (_chinese_integer(whole) + float("0." + fraction_digits))
    return sign * float(_chinese_integer(normalized))


def _chinese_integer(text: str) -> int:
    if not text:
        return 0
    if all(char in _DIGITS for char in text):
        return int("".join(str(_DIGITS[char]) for char in text))
    total = 0
    section = 0
    number = 0
    for char in text:
        if char in _DIGITS:
            number = _DIGITS[char]
            continue
        unit = _UNITS.get(char)
        if unit is None:
            raise FleetRequestBuildError(f"unsupported numeric token: {text}")
        if unit == 10000:
            section = (section + number) * unit
            total += section
            section = 0
            number = 0
        else:
            if number == 0:
                number = 1
            section += number * unit
            number = 0
    return total + section + number


__all__ = [
    "ExplicitAssignmentDirective",
    "FleetRequestBuildError",
    "build_agent_world_contexts",
    "build_agent_world_contexts_v2",
    "build_fleet_inventory",
    "build_fleet_mission_request",
    "build_fleet_mission_request_v2",
    "build_target_catalog",
    "derive_safe_home_landing_tolerances",
    "parse_explicit_assignment_instruction",
]
