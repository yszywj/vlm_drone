"""Deterministic, side-effect-free safety checks for compiled missions.

The supervisor reports policy decisions only.  It deliberately has no
reference to a SkillManager, an LLM, or a UAV controller, so the future
MissionAgent remains responsible for acting on ``CANCEL_AND_LAND``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from math import hypot, isfinite, pi
from numbers import Real

from common.ids import validate_routing_id
from planner.schemas import CompiledMission
from runtime.plan_validator import PlannerLimits
from skills.plan import RecoveryPolicy, StepOutputRef, TaskPlan, TaskStep
from skills.motion_types import (
    MotionPolicy,
    MotionPolicyValidationError,
    YawMode,
)
from skills.types import Observation, SkillName


_ALLOWED_COMPILED_PARAMS: Mapping[SkillName, frozenset[str]] = {
    SkillName.TAKEOFF: frozenset(
        {"target_altitude", "tolerance", "climb_speed", "yaw_mode", "yaw_value", "timeout"}
    ),
    SkillName.GOTO: frozenset(
        {"position", "tolerance", "motion_policy", "timeout"}
    ),
    SkillName.HOVER: frozenset(
        {
            "mode",
            "duration_s",
            "max_wait_s",
            "position_tolerance_m",
            "max_correction_speed_mps",
            "reason_code",
            "motion_policy",
        }
    ),
    SkillName.SEARCH: frozenset(
        {
            "center",
            "radius",
            "target_description",
            "search_altitude",
            "transit_speed",
            "scan_yaw_rate",
            "timeout",
        }
    ),
    SkillName.TRACK: frozenset(
        {
            "target_id",
            "desired_distance",
            "desired_altitude",
            "max_speed",
            "max_target_lost_time",
            "timeout",
            "track_duration",
        }
    ),
    SkillName.INSPECT: frozenset(
        {
            "candidate_id",
            "desired_observation_distance_m",
            "viewpoint_change_rad",
            "max_duration_s",
            "approach_policy",
        }
    ),
    SkillName.LAND: frozenset(
        {
            "ground_altitude",
            "tolerance",
            "descent_speed",
            "yaw_mode",
            "yaw_value",
            "timeout",
            "expected_position_xy",
            "zone_tolerance_m",
        }
    ),
}


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
        planner_limits: PlannerLimits | None = None,
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
        self._max_track_distance_m = hypot(
            self._scene_max[0] - self._scene_min[0],
            self._scene_max[1] - self._scene_min[1],
        )
        if not isfinite(self._max_track_distance_m):
            raise ValueError("scene horizontal span must be finite")
        if planner_limits is None:
            planner_limits = PlannerLimits()
        if not isinstance(planner_limits, PlannerLimits):
            raise TypeError("planner_limits must be a PlannerLimits")
        self._planner_limits = planner_limits
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
                step_id = step.step_id
                skill = step.skill
                params = step.params
                recovery = step.recovery
                if not isinstance(skill, SkillName):
                    return _abort(f"TaskPlan step {index} contains an unknown Skill")
                if skill is SkillName.REACQUIRE:
                    return _abort("TaskPlan must not contain explicit REACQUIRE")
                if not isinstance(params, Mapping):
                    return _abort(f"TaskPlan step {index} params must be a mapping")
                # TaskStep performs a defensive deep copy, insulating all later
                # checks from mutable input and surfacing damaged mappings here.
                snapshots.append(TaskStep(step_id, skill, params, recovery))
            except Exception as exc:
                return _abort(
                    f"TaskPlan step {index} is structurally invalid: "
                    f"{_exception_text(exc)}"
                )

        try:
            structure_decision = self._validate_plan_structure(tuple(snapshots))
        except Exception as exc:
            return _abort(
                "TaskPlan structure is invalid: "
                f"{_exception_text(exc)}"
            )
        if structure_decision.action is not SafetyAction.CONTINUE:
            return structure_decision

        try:
            for index, step in enumerate(snapshots):
                _validate_finite_tree(step.params, f"step[{index}].params")
                self._validate_step(index, step)
        except (TypeError, ValueError) as exc:
            return _abort(f"TaskPlan safety validation failed: {exc}")
        return _continue("preflight checks passed")

    def _validate_plan_structure(
        self,
        steps: tuple[TaskStep, ...],
    ) -> SafetyDecision:
        limits = self._planner_limits
        if not 2 <= len(steps) <= limits.max_plan_steps:
            return _abort(
                f"TaskPlan must contain 2-{limits.max_plan_steps} steps"
            )
        if steps[0].skill is not SkillName.TAKEOFF:
            return _abort("TaskPlan first step must be TAKEOFF")
        if steps[-1].skill is not SkillName.LAND:
            return _abort("TaskPlan final step must be LAND")

        ids = [step.step_id for step in steps]
        if len(ids) != len(set(ids)):
            return _abort("TaskPlan step ids must be unique")
        counts = {
            skill: sum(step.skill is skill for step in steps)
            for skill in SkillName
        }
        if counts[SkillName.TAKEOFF] != 1:
            return _abort("TaskPlan must contain exactly one TAKEOFF")
        if counts[SkillName.LAND] != 1:
            return _abort("TaskPlan must contain exactly one LAND")
        if counts[SkillName.GOTO] > limits.max_goto_calls:
            return _abort("TaskPlan GOTO call count exceeds planner limit")
        if counts[SkillName.SEARCH] > limits.max_search_calls:
            return _abort("TaskPlan SEARCH call count exceeds planner limit")
        if counts[SkillName.TRACK] > limits.max_track_calls:
            return _abort("TaskPlan TRACK call count exceeds planner limit")
        if counts[SkillName.REACQUIRE]:
            return _abort("TaskPlan must not contain explicit REACQUIRE")

        previous: dict[str, SkillName] = {}
        total_recovery_attempts = 0
        for index, step in enumerate(steps):
            prefix = f"TaskPlan step {index} ({step.step_id})"
            target = step.params.get("target_id")
            if (
                step.skill is SkillName.TRACK
                and SkillName.SEARCH not in previous.values()
            ):
                return _abort(f"{prefix} TRACK must appear after SEARCH")
            if (
                step.skill is SkillName.INSPECT
                and SkillName.SEARCH not in previous.values()
            ):
                return _abort(f"{prefix} INSPECT must appear after SEARCH")
            if isinstance(target, StepOutputRef):
                if step.skill is not SkillName.TRACK:
                    return _abort(
                        f"{prefix} contains a StepOutputRef outside TRACK"
                    )
                referenced_skill = previous.get(target.step_id)
                if referenced_skill is None:
                    return _abort(f"{prefix} target reference is not backward")
                if referenced_skill is not SkillName.SEARCH:
                    return _abort(f"{prefix} target reference must point to SEARCH")

            recovery = step.recovery
            if recovery is not None:
                if step.skill is not SkillName.TRACK:
                    return _abort(f"{prefix} recovery is only valid on TRACK")
                if not isinstance(recovery, RecoveryPolicy):
                    return _abort(f"{prefix} recovery policy is structurally invalid")
                if recovery.skill is not SkillName.REACQUIRE:
                    return _abort(f"{prefix} recovery Skill must be REACQUIRE")
                if (
                    recovery.max_attempts
                    > limits.max_reacquire_attempts_per_track
                ):
                    return _abort(f"{prefix} recovery attempts exceed planner limit")
                if not 3.0 <= recovery.search_radius_m <= 20.0:
                    return _abort(f"{prefix} recovery radius is outside 3-20 m")
                if not 5.0 <= recovery.timeout_s <= 60.0:
                    return _abort(f"{prefix} recovery timeout is outside 5-60 s")
                total_recovery_attempts += recovery.max_attempts
            previous[step.step_id] = step.skill

        if total_recovery_attempts > limits.max_total_reacquire_attempts:
            return _abort("TaskPlan total recovery attempt budget exceeds planner limit")
        return _continue("TaskPlan structure is safe")

    def _validate_step(self, index: int, step: TaskStep) -> None:
        params = step.params
        prefix = f"step[{index}] {step.skill.value}"
        allowed = _ALLOWED_COMPILED_PARAMS.get(step.skill)
        if allowed is None:
            raise ValueError(f"{prefix} is not a supported top-level Skill")
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ValueError(
                f"{prefix} contains unknown compiled params: "
                + ", ".join(unknown)
            )

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

        if step.skill is SkillName.HOVER:
            from skills.hover import HoverMode

            raw_mode = _required(params, "mode", prefix)
            try:
                mode = (
                    raw_mode
                    if isinstance(raw_mode, HoverMode)
                    else HoverMode(raw_mode)
                )
            except (TypeError, ValueError):
                raise ValueError(f"{prefix} mode must be TIMED") from None
            if mode is not HoverMode.TIMED:
                raise ValueError(f"{prefix} only TIMED mode is plan-safe")
            duration = _positive_finite(
                _required(params, "duration_s", prefix),
                f"{prefix} duration_s",
            )
            if not 1.0 <= duration <= 60.0:
                raise ValueError(f"{prefix} duration_s must be between 1 and 60")
            max_wait = _positive_finite(
                _required(params, "max_wait_s", prefix),
                f"{prefix} max_wait_s",
            )
            if max_wait < duration or max_wait > 60.0:
                raise ValueError(
                    f"{prefix} max_wait_s must cover duration_s and not exceed 60"
                )
            position_tolerance = _positive_finite(
                _required(params, "position_tolerance_m", prefix),
                f"{prefix} position_tolerance_m",
            )
            if position_tolerance > 1.0:
                raise ValueError(
                    f"{prefix} position_tolerance_m exceeds the trusted bound"
                )
            correction_speed = _positive_finite(
                _required(params, "max_correction_speed_mps", prefix),
                f"{prefix} max_correction_speed_mps",
            )
            if correction_speed > 0.5:
                raise ValueError(
                    f"{prefix} max_correction_speed_mps exceeds the trusted bound"
                )
            reason_code = _required(params, "reason_code", prefix)
            try:
                validated_reason = validate_routing_id(
                    reason_code,
                    f"{prefix} reason_code",
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(str(exc)) from None
            if validated_reason != "PLANNED_HOVER":
                raise ValueError(f"{prefix} reason_code is not a planned HOVER")
            motion_policy = _validate_motion_policy(
                _required(params, "motion_policy", prefix),
                prefix,
            )
            if motion_policy.yaw_mode not in {
                YawMode.KEEP_CURRENT,
                YawMode.FIXED,
            }:
                raise ValueError(
                    f"{prefix} motion_policy yaw must be KEEP_CURRENT or FIXED"
                )
            if (
                motion_policy.max_speed is None
                or motion_policy.max_speed > correction_speed
            ):
                raise ValueError(
                    f"{prefix} motion_policy.max_speed exceeds correction speed"
                )
            if (
                motion_policy.max_yaw_rate is None
                or motion_policy.max_yaw_rate > 1.0
            ):
                raise ValueError(
                    f"{prefix} motion_policy.max_yaw_rate exceeds the trusted bound"
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

        if step.skill is SkillName.INSPECT:
            candidate_id = _required(params, "candidate_id", prefix)
            try:
                validate_routing_id(candidate_id, f"{prefix} candidate_id")
            except (TypeError, ValueError) as exc:
                raise ValueError(str(exc)) from None
            distance = _positive_finite(
                _required(
                    params,
                    "desired_observation_distance_m",
                    prefix,
                ),
                f"{prefix} desired_observation_distance_m",
            )
            if not 2.0 <= distance <= 20.0:
                raise ValueError(
                    f"{prefix} desired_observation_distance_m must be between "
                    "2 and 20"
                )
            angle = _finite_number(
                _required(params, "viewpoint_change_rad", prefix),
                f"{prefix} viewpoint_change_rad",
            )
            if abs(angle) <= 1e-9 or abs(angle) > pi / 2.0:
                raise ValueError(
                    f"{prefix} viewpoint_change_rad must be non-zero and no "
                    "greater than pi/2"
                )
            duration = _positive_finite(
                _required(params, "max_duration_s", prefix),
                f"{prefix} max_duration_s",
            )
            if duration > 60.0:
                raise ValueError(f"{prefix} max_duration_s exceeds 60")
            from skills.inspect import InspectApproachPolicy

            raw_approach = _required(params, "approach_policy", prefix)
            try:
                approach = (
                    raw_approach
                    if isinstance(raw_approach, InspectApproachPolicy)
                    else InspectApproachPolicy(raw_approach)
                )
            except (TypeError, ValueError):
                raise ValueError(
                    f"{prefix} approach_policy must be "
                    "MAINTAIN_ALTITUDE_ORBIT"
                ) from None
            if approach is not InspectApproachPolicy.MAINTAIN_ALTITUDE_ORBIT:
                raise ValueError(f"{prefix} approach_policy is unsupported")
            return

        if step.skill is SkillName.TRACK:
            duration = _positive_finite(
                _required(params, "track_duration", prefix),
                f"{prefix} track_duration",
            )
            if not (
                self._planner_limits.min_track_duration_s
                <= duration
                <= self._planner_limits.max_track_duration_s
            ):
                raise ValueError(
                    f"{prefix} track_duration must be between "
                    f"{self._planner_limits.min_track_duration_s:g} and "
                    f"{self._planner_limits.max_track_duration_s:g} seconds"
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
            if (
                "desired_distance" in params
                and float(params["desired_distance"]) > self._max_track_distance_m
            ):
                raise ValueError(
                    f"{prefix} desired_distance exceeds the scene scale"
                )
            target_id = _required(params, "target_id", prefix)
            if isinstance(target_id, StepOutputRef):
                if target_id.field != "target_id":
                    raise ValueError(f"{prefix} target reference field is invalid")
            elif target_id != "$SEARCH.result.target_id":
                raise ValueError(
                    f"{prefix} target_id must be a validated StepOutputRef "
                    "or the legacy SEARCH placeholder"
                )
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
                ("tolerance", "descent_speed", "zone_tolerance_m"),
            )
            expected_xy = params.get("expected_position_xy")
            if expected_xy is not None:
                expected_x, expected_y = _finite_vector2(
                    expected_xy,
                    f"{prefix} expected_position_xy",
                )
                if not (
                    self._scene_min[0] <= expected_x <= self._scene_max[0]
                    and self._scene_min[1] <= expected_y <= self._scene_max[1]
                ):
                    raise ValueError(
                        f"{prefix} expected_position_xy is outside scene XY bounds"
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
        if abs(float(params["yaw_value"])) > 2.0 * pi:
            raise ValueError(f"{prefix} yaw_value exceeds one full rotation")
    elif params.get("yaw_value") is not None:
        raise ValueError(f"{prefix} yaw_value is only valid with FIXED yaw")


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
    if (
        policy.yaw_mode is YawMode.FIXED
        and policy.yaw_value is not None
        and abs(float(policy.yaw_value)) > 2.0 * pi
    ):
        raise ValueError(
            f"{prefix} motion_policy.yaw_value exceeds one full rotation"
        )
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


def _finite_vector2(value: object, name: str) -> tuple[float, float]:
    if (
        isinstance(value, (str, bytes, bytearray, Mapping))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise ValueError(f"{name} must contain exactly two finite numbers")
    return (
        _finite_number(value[0], f"{name}[0]"),
        _finite_number(value[1], f"{name}[1]"),
    )


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
