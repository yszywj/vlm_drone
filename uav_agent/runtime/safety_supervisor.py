"""Deterministic, side-effect-free safety checks for compiled missions.

The supervisor reports policy decisions only.  It deliberately has no
reference to a SkillManager, an LLM, or a UAV controller, so the future
MissionAgent remains responsible for acting on ``CANCEL_AND_LAND``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from math import isfinite
from numbers import Real

from planner.schemas import CompiledMission
from skills.manager import TaskPlan, TaskStep
from skills.motion_types import (
    MotionPolicy,
    MotionPolicyValidationError,
    YawMode,
)
from skills.types import Observation, SkillName


class SafetyAction(str, Enum):
    """Action requested from the future MissionAgent."""

    CONTINUE = "CONTINUE"
    CANCEL_AND_LAND = "CANCEL_AND_LAND"
    ABORT = "ABORT"


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    """One immutable safety verdict with a human-readable reason."""

    action: SafetyAction
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.action, SafetyAction):
            raise TypeError("SafetyDecision.action must be a SafetyAction")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("SafetyDecision.reason must be a non-empty string")


class SafetySupervisor:
    """Validate plans and observations without performing control actions.

    ``position_margin_m`` is a runtime numerical tolerance around the scene
    bounds.  Preflight goals must remain inside the exact configured bounds;
    a live pose only triggers a boundary response after it exceeds the bound
    by more than this margin.
    """

    MIN_TRACK_DURATION_S = 1.0
    MAX_TRACK_DURATION_S = 600.0

    def __init__(
        self,
        scene_min_xyz_m: Sequence[float],
        scene_max_xyz_m: Sequence[float],
        max_mission_time_s: float = 900.0,
        position_margin_m: float = 0.0,
        max_safe_altitude_m: float | None = None,
    ) -> None:
        self._scene_min = _finite_vector3(scene_min_xyz_m, "scene_min_xyz_m")
        self._scene_max = _finite_vector3(scene_max_xyz_m, "scene_max_xyz_m")
        if any(
            lower >= upper
            for lower, upper in zip(self._scene_min, self._scene_max)
        ):
            raise ValueError(
                "scene_min_xyz_m must be strictly below scene_max_xyz_m "
                "on every axis"
            )

        self._max_mission_time_s = _positive_finite(
            max_mission_time_s,
            "max_mission_time_s",
        )
        self._position_margin_m = _nonnegative_finite(
            position_margin_m,
            "position_margin_m",
        )
        safe_altitude = (
            self._scene_max[2]
            if max_safe_altitude_m is None
            else _finite_number(max_safe_altitude_m, "max_safe_altitude_m")
        )
        if safe_altitude < self._scene_min[2]:
            raise ValueError(
                "max_safe_altitude_m must not be below the scene minimum Z"
        )
        self._max_safe_altitude_m = safe_altitude
        self._last_observation_timestamp: float | None = None
        self._last_mission_elapsed_s: float | None = None

    def preflight(
        self,
        compiled_mission: CompiledMission | TaskPlan,
    ) -> SafetyDecision:
        """Check a mission before dispatching any Skill.

        A bare ``TaskPlan`` is accepted as a convenience for callers that have
        already discarded planner metadata.  Invalid preflight input is an
        ``ABORT`` because no flight has started and no safe execution decision
        can be made from it.
        """

        try:
            if isinstance(compiled_mission, CompiledMission):
                task_plan = compiled_mission.task_plan
            elif isinstance(compiled_mission, TaskPlan):
                task_plan = compiled_mission
            else:
                return _abort("preflight input must be a CompiledMission or TaskPlan")
        except Exception as exc:
            return _abort(
                "compiled mission is structurally invalid: "
                f"{_exception_text(exc)}"
            )

        decision = self._validate_task_plan(task_plan)
        if decision.action is SafetyAction.CONTINUE:
            # A successful preflight begins a fresh observation timeline.
            self._last_observation_timestamp = None
            self._last_mission_elapsed_s = None
        return decision

    def evaluate(
        self,
        observation: Observation,
        *,
        mission_elapsed_s: float,
    ) -> SafetyDecision:
        """Evaluate one immutable runtime sample and return a policy verdict."""

        if not isinstance(observation, Observation):
            return _abort("observation must be an Observation")

        try:
            observation.validate()
        except Exception as exc:
            return _abort(f"observation is invalid: {_exception_text(exc)}")

        try:
            elapsed = _nonnegative_finite(
                mission_elapsed_s,
                "mission_elapsed_s",
            )
        except (TypeError, ValueError) as exc:
            return _abort(f"mission timing is invalid: {exc}")

        timestamp = float(observation.timestamp)
        if (
            self._last_observation_timestamp is not None
            and timestamp < self._last_observation_timestamp
        ):
            return _abort(
                "observation timestamp moved backwards: "
                f"{timestamp:g} < {self._last_observation_timestamp:g}"
            )
        if (
            self._last_mission_elapsed_s is not None
            and elapsed < self._last_mission_elapsed_s
        ):
            return _abort(
                "mission elapsed time moved backwards: "
                f"{elapsed:g} < {self._last_mission_elapsed_s:g}"
            )

        # A time-rollback sample never rewinds either trusted monotonic anchor.
        self._last_observation_timestamp = timestamp
        self._last_mission_elapsed_s = elapsed

        pose = observation.uav_pose
        position = (float(pose.x), float(pose.y), float(pose.z))
        if not self._point_in_runtime_bounds(position):
            return _cancel_and_land(
                "UAV position is outside the scene bounds: "
                f"({position[0]:g}, {position[1]:g}, {position[2]:g})"
            )
        if position[2] > self._max_safe_altitude_m:
            return _cancel_and_land(
                "UAV altitude exceeds max_safe_altitude_m: "
                f"{position[2]:g} > {self._max_safe_altitude_m:g}"
            )
        if elapsed > self._max_mission_time_s:
            return _cancel_and_land(
                "mission exceeded max_mission_time_s: "
                f"{elapsed:g} > {self._max_mission_time_s:g}"
            )
        return _continue("safety checks passed")

    def reset(self) -> None:
        """Clear the per-mission monotonic timestamp history."""

        self._last_observation_timestamp = None
        self._last_mission_elapsed_s = None

    def _validate_task_plan(self, task_plan: TaskPlan) -> SafetyDecision:
        if not isinstance(task_plan, TaskPlan):
            return _abort("compiled mission does not contain a TaskPlan")
        try:
            steps = task_plan.steps
        except Exception as exc:
            return _abort(
                "TaskPlan is structurally invalid: "
                f"{_exception_text(exc)}"
            )
        if (
            not isinstance(steps, tuple)
            or not steps
            or not all(isinstance(step, TaskStep) for step in steps)
        ):
            return _abort("TaskPlan steps are structurally invalid")

        snapshots: list[TaskStep] = []
        for index, step in enumerate(steps):
            try:
                skill = step.skill
                params = step.params
                if not isinstance(skill, SkillName):
                    return _abort(f"TaskPlan step {index} contains an unknown Skill")
                if skill is SkillName.REACQUIRE:
                    return _abort("TaskPlan must not contain explicit REACQUIRE")
                if not isinstance(params, Mapping):
                    return _abort(f"TaskPlan step {index} params must be a mapping")
                # TaskStep performs a defensive deep copy, insulating all later
                # checks from mutable input and surfacing damaged mappings here.
                snapshots.append(TaskStep(skill, params))
            except Exception as exc:
                return _abort(
                    f"TaskPlan step {index} is structurally invalid: "
                    f"{_exception_text(exc)}"
                )

        if snapshots[-1].skill is not SkillName.LAND:
            return _abort("TaskPlan final step must be LAND")

        # Re-run the canonical structure/field-name validator because frozen
        # TaskPlan objects can still contain mutable parameter dictionaries.
        try:
            TaskPlan.from_dicts([step.to_dict() for step in snapshots])
        except Exception as exc:
            return _abort(f"TaskPlan structure is invalid: {_exception_text(exc)}")

        try:
            for index, step in enumerate(snapshots):
                _validate_finite_tree(step.params, f"step[{index}].params")
                self._validate_step(index, step)
        except (TypeError, ValueError) as exc:
            return _abort(f"TaskPlan safety validation failed: {exc}")
        return _continue("preflight checks passed")

    def _validate_step(self, index: int, step: TaskStep) -> None:
        params = step.params
        prefix = f"step[{index}] {step.skill.value}"

        if "timeout" in params:
            if params["timeout"] is None:
                if step.skill is not SkillName.TRACK:
                    raise ValueError(f"{prefix} timeout must not be None")
            else:
                _positive_finite(params["timeout"], f"{prefix} timeout")

        if step.skill is SkillName.TAKEOFF:
            altitude = _positive_finite(
                _required(params, "target_altitude", prefix),
                f"{prefix} target_altitude",
            )
            self._require_flight_altitude(altitude, f"{prefix} target_altitude")
            _validate_optional_positive_fields(
                params,
                prefix,
                ("tolerance", "climb_speed"),
            )
            _validate_vertical_yaw(params, prefix)
            return

        if step.skill is SkillName.GOTO:
            position = _finite_vector3(
                _required(params, "position", prefix),
                f"{prefix} position",
            )
            self._require_point_in_plan_bounds(position, f"{prefix} position")
            _validate_optional_positive_fields(params, prefix, ("tolerance",))
            if "motion_policy" in params:
                motion_policy = _validate_motion_policy(
                    params["motion_policy"],
                    prefix,
                )
                if motion_policy.look_at_point is not None:
                    look_at_point = _finite_vector3(
                        motion_policy.look_at_point,
                        f"{prefix} motion_policy.look_at_point",
                    )
                    self._require_point_in_scene_bounds(
                        look_at_point,
                        f"{prefix} motion_policy.look_at_point",
                    )
            return

        if step.skill is SkillName.SEARCH:
            center = _finite_vector3(
                _required(params, "center", prefix),
                f"{prefix} center",
            )
            self._require_point_in_plan_bounds(center, f"{prefix} center")
            radius = _positive_finite(
                _required(params, "radius", prefix),
                f"{prefix} radius",
            )
            altitude = _positive_finite(
                _required(params, "search_altitude", prefix),
                f"{prefix} search_altitude",
            )
            self._require_flight_altitude(altitude, f"{prefix} search_altitude")
            if (
                center[0] - radius < self._scene_min[0]
                or center[0] + radius > self._scene_max[0]
                or center[1] - radius < self._scene_min[1]
                or center[1] + radius > self._scene_max[1]
            ):
                raise ValueError(f"{prefix} search radius leaves the scene bounds")
            _validate_optional_positive_fields(
                params,
                prefix,
                ("transit_speed", "scan_yaw_rate"),
            )
            description = _required(params, "target_description", prefix)
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"{prefix} target_description must be non-empty")
            return

        if step.skill is SkillName.TRACK:
            duration = _positive_finite(
                _required(params, "track_duration", prefix),
                f"{prefix} track_duration",
            )
            if not self.MIN_TRACK_DURATION_S <= duration <= self.MAX_TRACK_DURATION_S:
                raise ValueError(
                    f"{prefix} track_duration must be between "
                    f"{self.MIN_TRACK_DURATION_S:g} and "
                    f"{self.MAX_TRACK_DURATION_S:g} seconds"
                )
            if "desired_altitude" in params:
                altitude = _positive_finite(
                    params["desired_altitude"],
                    f"{prefix} desired_altitude",
                )
                self._require_flight_altitude(
                    altitude,
                    f"{prefix} desired_altitude",
                )
            _validate_optional_positive_fields(
                params,
                prefix,
                (
                    "desired_distance",
                    "max_speed",
                    "max_target_lost_time",
                    "timeout",
                ),
                allow_none=("timeout",),
            )
            target_id = params.get("target_id", "$SEARCH.result.target_id")
            if not isinstance(target_id, str) or not target_id.strip():
                raise ValueError(f"{prefix} target_id must be non-empty")
            return

        if step.skill is SkillName.LAND:
            altitude = _finite_number(
                params.get("ground_altitude", 0.0),
                f"{prefix} ground_altitude",
            )
            if not self._scene_min[2] <= altitude <= self._scene_max[2]:
                raise ValueError(f"{prefix} ground_altitude is outside scene Z bounds")
            if altitude > self._max_safe_altitude_m:
                raise ValueError(f"{prefix} ground_altitude exceeds max_safe_altitude_m")
            _validate_optional_positive_fields(
                params,
                prefix,
                ("tolerance", "descent_speed"),
            )
            _validate_vertical_yaw(params, prefix)

    def _require_flight_altitude(self, altitude: float, name: str) -> None:
        if not self._scene_min[2] < altitude <= self._scene_max[2]:
            raise ValueError(f"{name} is outside valid flight altitude bounds")
        if altitude > self._max_safe_altitude_m:
            raise ValueError(f"{name} exceeds max_safe_altitude_m")

    def _require_point_in_plan_bounds(
        self,
        point: tuple[float, float, float],
        name: str,
    ) -> None:
        if any(
            coordinate < lower or coordinate > upper
            for coordinate, lower, upper in zip(
                point,
                self._scene_min,
                self._scene_max,
            )
        ):
            raise ValueError(f"{name} is outside the scene bounds")
        if point[2] > self._max_safe_altitude_m:
            raise ValueError(f"{name} exceeds max_safe_altitude_m")

    def _require_point_in_scene_bounds(
        self,
        point: tuple[float, float, float],
        name: str,
    ) -> None:
        if any(
            coordinate < lower or coordinate > upper
            for coordinate, lower, upper in zip(
                point,
                self._scene_min,
                self._scene_max,
            )
        ):
            raise ValueError(f"{name} is outside the scene bounds")

    def _point_in_runtime_bounds(
        self,
        point: tuple[float, float, float],
    ) -> bool:
        margin = self._position_margin_m
        return all(
            lower - margin <= coordinate <= upper + margin
            for coordinate, lower, upper in zip(
                point,
                self._scene_min,
                self._scene_max,
            )
        )


def _required(params: Mapping[str, object], name: str, prefix: str) -> object:
    if name not in params:
        raise ValueError(f"{prefix} is missing {name}")
    return params[name]


def _validate_optional_positive_fields(
    params: Mapping[str, object],
    prefix: str,
    names: Sequence[str],
    *,
    allow_none: Sequence[str] = (),
) -> None:
    nullable = frozenset(allow_none)
    for name in names:
        if name not in params or (name in nullable and params[name] is None):
            continue
        _positive_finite(params[name], f"{prefix} {name}")


def _validate_vertical_yaw(params: Mapping[str, object], prefix: str) -> None:
    raw_mode = params.get("yaw_mode", YawMode.KEEP_CURRENT)
    mode = _yaw_mode(raw_mode, f"{prefix} yaw_mode")
    if mode not in {YawMode.KEEP_CURRENT, YawMode.FIXED}:
        raise ValueError(f"{prefix} yaw_mode must be KEEP_CURRENT or FIXED")
    if mode is YawMode.FIXED:
        if params.get("yaw_value") is None:
            raise ValueError(f"{prefix} FIXED yaw_mode requires yaw_value")
        _finite_number(params["yaw_value"], f"{prefix} yaw_value")


def _validate_motion_policy(value: object, prefix: str) -> MotionPolicy:
    if isinstance(value, MotionPolicy):
        policy = value
    elif isinstance(value, Mapping):
        allowed = {field.name for field in fields(MotionPolicy)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                f"{prefix} motion_policy contains unknown fields: "
                + ", ".join(str(name) for name in unknown)
            )
        params = dict(value)
        if "yaw_mode" in params:
            params["yaw_mode"] = _yaw_mode(
                params["yaw_mode"],
                f"{prefix} motion_policy.yaw_mode",
            )
        if isinstance(params.get("look_at_point"), list):
            params["look_at_point"] = tuple(params["look_at_point"])
        try:
            policy = MotionPolicy(**params)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{prefix} motion_policy is invalid: {exc}") from exc
    else:
        raise TypeError(f"{prefix} motion_policy must be MotionPolicy or mapping")

    try:
        policy.validate()
    except (MotionPolicyValidationError, TypeError, ValueError) as exc:
        raise ValueError(f"{prefix} motion_policy is invalid: {exc}") from exc
    return policy


def _yaw_mode(value: object, name: str) -> YawMode:
    if isinstance(value, YawMode):
        return value
    if isinstance(value, str):
        try:
            return YawMode[value.upper()]
        except KeyError as exc:
            raise ValueError(f"{name} is unknown") from exc
    raise TypeError(f"{name} must be a YawMode or string")


def _validate_finite_tree(
    value: object,
    name: str,
    *,
    _ancestors: set[int] | None = None,
    _depth: int = 0,
) -> None:
    """Reject non-finite, cyclic, deep, or opaque plan parameter values."""

    if _depth > 32:
        raise ValueError(f"{name} exceeds the maximum parameter nesting depth")

    if value is None or isinstance(value, (str, bool, Enum)):
        return
    if isinstance(value, Real):
        _finite_number(value, name)
        return
    is_mapping = isinstance(value, Mapping)
    is_record = is_dataclass(value) and not isinstance(value, type)
    is_sequence = isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )
    if is_mapping or is_record or is_sequence:
        ancestors = set() if _ancestors is None else _ancestors
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{name} contains a cyclic value")
        ancestors.add(identity)
        try:
            if is_mapping:
                for key, child in value.items():  # type: ignore[union-attr]
                    if not isinstance(key, str):
                        raise TypeError(f"{name} keys must be strings")
                    _validate_finite_tree(
                        child,
                        f"{name}.{key}",
                        _ancestors=ancestors,
                        _depth=_depth + 1,
                    )
            elif is_record:
                for field in fields(value):
                    _validate_finite_tree(
                        getattr(value, field.name),
                        f"{name}.{field.name}",
                        _ancestors=ancestors,
                        _depth=_depth + 1,
                    )
            else:
                for index, child in enumerate(value):  # type: ignore[arg-type]
                    _validate_finite_tree(
                        child,
                        f"{name}[{index}]",
                        _ancestors=ancestors,
                        _depth=_depth + 1,
                    )
        finally:
            ancestors.remove(identity)
        return
    raise TypeError(f"{name} contains unsupported value type {type(value).__name__}")


def _finite_vector3(value: object, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{name} must contain exactly three finite numbers")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            f"{name} must contain exactly three finite numbers"
        ) from exc
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three finite numbers")
    parsed = tuple(
        _finite_number(component, f"{name}[{index}]")
        for index, component in enumerate(values)
    )
    return parsed[0], parsed[1], parsed[2]


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _positive_finite(value: object, name: str) -> float:
    parsed = _finite_number(value, name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _nonnegative_finite(value: object, name: str) -> float:
    parsed = _finite_number(value, name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _exception_text(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or type(exc).__name__


def _continue(reason: str) -> SafetyDecision:
    return SafetyDecision(SafetyAction.CONTINUE, reason)


def _cancel_and_land(reason: str) -> SafetyDecision:
    return SafetyDecision(SafetyAction.CANCEL_AND_LAND, reason)


def _abort(reason: str) -> SafetyDecision:
    return SafetyDecision(SafetyAction.ABORT, reason)


__all__ = ["SafetyAction", "SafetyDecision", "SafetySupervisor"]
