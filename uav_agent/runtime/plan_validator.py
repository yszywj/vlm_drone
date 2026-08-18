"""Trusted compilation of legacy intents and constrained dynamic Skill plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import hypot, isfinite, radians
from numbers import Real

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id
from planner.schemas import (
    CompiledMission,
    LandingZoneSpec,
    MissionIntent,
    PlannerWorldContext,
    SearchRegionSpec,
    SkillPlanDraft,
    SkillPlanDraftV2,
    migrate_plan_v1_to_v2,
)
from planner.policy import PlannerLimits, PlannerPolicy, TargetLostAction
from planner.symbolic_checker import SymbolicPlanChecker
from skills.motion_types import MotionPolicy, YawMode
from skills.plan import RecoveryPolicy, StepOutputRef, TaskPlan, TaskPlanError, TaskStep
from skills.types import SkillName


class PlanValidationError(ValueError):
    """Raised when planner output cannot be compiled without unsafe inference."""


class _MissionState(Enum):
    ON_GROUND = "ON_GROUND"
    AIRBORNE_NO_TARGET = "AIRBORNE_NO_TARGET"
    AIRBORNE_TARGET_AVAILABLE = "AIRBORNE_TARGET_AVAILABLE"
    LANDED = "LANDED"


@dataclass(frozen=True, slots=True)
class _TrustedWorld:
    scene_min: tuple[float, float, float]
    scene_max: tuple[float, float, float]
    initial_uav: tuple[float, float, float]
    default_altitude: float
    default_track_duration: float
    search_timeout: float
    goto_timeout: float
    land_timeout: float


_DYNAMIC_SOURCES = frozenset({"dynamic_scripted", "dynamic_llm"})
_LEGACY_SOURCES = frozenset({"scripted", "llm"})
_ALL_SOURCES = _DYNAMIC_SOURCES | _LEGACY_SOURCES

_DYNAMIC_ARGUMENTS: Mapping[SkillName, frozenset[str]] = {
    SkillName.TAKEOFF: frozenset({"altitude_m", "yaw_mode", "yaw_deg"}),
    SkillName.GOTO: frozenset(
        {"destination", "altitude_m", "yaw_mode", "yaw_deg"}
    ),
    SkillName.HOVER: frozenset({"duration_s", "yaw_mode", "yaw_deg"}),
    SkillName.SEARCH: frozenset(
        {"region", "target_description", "altitude_m"}
    ),
    SkillName.INSPECT: frozenset(
        {
            "candidate_id",
            "desired_observation_distance_m",
            "viewpoint_change_deg",
            "max_duration_s",
            "approach_policy",
        }
    ),
    SkillName.TRACK: frozenset(
        {
            "target_ref",
            "duration_s",
            "desired_altitude_m",
            "desired_distance_m",
            "on_target_lost",
        }
    ),
    SkillName.LAND: frozenset({"zone", "yaw_mode", "yaw_deg"}),
}


def _trusted_inspect_candidate_ids(value: object) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("trusted_inspect_candidate_ids must be a sequence")
    result = tuple(
        validate_routing_id(candidate_id, "trusted_inspect_candidate_ids entry")
        for candidate_id in value
    )
    if len(result) != len(set(result)):
        raise PlanValidationError(
            "trusted_inspect_candidate_ids must not contain duplicates"
        )
    return frozenset(result)


def _require_trusted_inspect_candidates(
    draft: SkillPlanDraft,
    trusted_candidate_ids: frozenset[str],
) -> None:
    for step in draft.steps:
        if step.skill != "INSPECT":
            continue
        candidate_id = step.args.get("candidate_id")
        if candidate_id not in trusted_candidate_ids:
            raise PlanValidationError(
                f"step {step.id} INSPECT candidate_id is not authorized by "
                "trusted runtime CandidateBank input"
            )


class PlanValidator:
    """Resolve semantic planner output to a bounded executable ``TaskPlan``.

    ``MissionIntent`` retains the original deterministic six-step compiler.
    ``SkillPlanDraft`` follows a separate fail-closed path: the model chooses a
    bounded linear sequence, while this class owns coordinates, motion limits,
    timeouts, reference resolution and recovery budgets.
    """

    MAX_TARGET_DESCRIPTION_CHARS = 256
    TRUSTED_GOTO_MAX_SPEED_MPS = 2.0
    TRUSTED_TRACK_MAX_SPEED_MPS = 2.0
    TRUSTED_TARGET_LOST_TIME_S = 2.0
    TRUSTED_TRACK_TIMEOUT_GRACE_S = 5.0
    DEFAULT_DESIRED_DISTANCE_M = 6.0
    MIN_PLANNED_HOVER_DURATION_S = 1.0
    MAX_PLANNED_HOVER_DURATION_S = 60.0
    TRUSTED_HOVER_POSITION_TOLERANCE_M = 0.25
    TRUSTED_HOVER_MAX_CORRECTION_SPEED_MPS = 0.5
    TRUSTED_HOVER_MAX_YAW_RATE_RAD_S = 1.0
    DEFAULT_INSPECT_OBSERVATION_DISTANCE_M = 4.0
    MIN_INSPECT_OBSERVATION_DISTANCE_M = 2.0
    MAX_INSPECT_OBSERVATION_DISTANCE_M = 20.0
    DEFAULT_INSPECT_VIEWPOINT_CHANGE_DEG = 45.0
    MAX_INSPECT_VIEWPOINT_CHANGE_DEG = 90.0
    DEFAULT_INSPECT_MAX_DURATION_S = 15.0
    MAX_INSPECT_MAX_DURATION_S = 60.0

    def __init__(
        self,
        limits: PlannerLimits | None = None,
        policy: PlannerPolicy | None = None,
    ) -> None:
        if limits is None:
            limits = PlannerLimits()
        if not isinstance(limits, PlannerLimits):
            raise TypeError("limits must be a PlannerLimits")
        if policy is None:
            policy = PlannerPolicy()
        if not isinstance(policy, PlannerPolicy):
            raise TypeError("policy must be a PlannerPolicy")
        policy.validate_against(limits)
        self._limits = limits
        self._policy = policy
        self._symbolic_checker = SymbolicPlanChecker()

    @property
    def limits(self) -> PlannerLimits:
        # ``PlanValidator`` had no constructor before dynamic planning.  Keep
        # older lightweight test/adapter subclasses that override ``__init__``
        # without ``super()`` on the trusted default limits.
        limits = getattr(self, "_limits", None)
        return PlannerLimits() if limits is None else limits

    @property
    def policy(self) -> PlannerPolicy:
        policy = getattr(self, "_policy", None)
        if policy is None:
            policy = PlannerPolicy()
        policy.validate_against(self.limits)
        return policy

    def validate_and_compile(
        self,
        planner_output: MissionIntent | SkillPlanDraft | SkillPlanDraftV2,
        context: PlannerWorldContext,
        *,
        source: str,
        mission_id: str | None = None,
        uav_id: str | None = None,
        plan_version: int | None = None,
        trusted_inspect_candidate_ids: Sequence[str] = (),
    ) -> CompiledMission:
        """Dispatch to the legacy or dynamic compiler based on output type.

        ``INSPECT`` is runtime-revision-only.  Candidate IDs must be supplied
        explicitly by the trusted coordinator after CandidateBank validation;
        ordinary initial planning leaves this allow-list empty and fails
        closed before compilation.
        """

        if not isinstance(context, PlannerWorldContext):
            raise TypeError("context must be a PlannerWorldContext")
        if not isinstance(source, str) or source not in _ALL_SOURCES:
            raise PlanValidationError(
                "source must be scripted, llm, dynamic_scripted, or dynamic_llm"
            )
        trusted_inspect_ids = _trusted_inspect_candidate_ids(
            trusted_inspect_candidate_ids
        )
        supplied = (
            mission_id is not None,
            uav_id is not None,
            plan_version is not None,
        )
        if any(supplied) and not all(supplied):
            raise PlanValidationError(
                "mission_id, uav_id, and plan_version must be supplied together"
            )
        trusted_mission_id = (
            "mission_legacy"
            if mission_id is None
            else validate_mission_id(mission_id)
        )
        trusted_uav_id = "uav_1" if uav_id is None else validate_uav_id(uav_id)
        trusted_plan_version = 1 if plan_version is None else plan_version
        if isinstance(trusted_plan_version, bool) or not isinstance(
            trusted_plan_version, int
        ) or trusted_plan_version <= 0:
            raise PlanValidationError("plan_version must be a positive integer")
        if isinstance(planner_output, MissionIntent):
            if trusted_inspect_ids:
                raise PlanValidationError(
                    "trusted INSPECT candidates are invalid for MissionIntent"
                )
            if source not in _LEGACY_SOURCES:
                raise PlanValidationError(
                    "MissionIntent requires source 'scripted' or 'llm'"
                )
            return self._compile_legacy(
                planner_output,
                context,
                source,
                mission_id=trusted_mission_id,
                uav_id=trusted_uav_id,
                plan_version=trusted_plan_version,
            )
        if isinstance(planner_output, SkillPlanDraftV2):
            if source not in _DYNAMIC_SOURCES:
                raise PlanValidationError(
                    "SkillPlanDraftV2 requires source 'dynamic_scripted' or "
                    "'dynamic_llm'"
                )
            if mission_id is None:
                raise PlanValidationError(
                    "SkillPlanDraftV2 requires trusted mission_id, uav_id, and "
                    "plan_version values"
                )
            if mission_id is not None and (
                planner_output.mission_id != trusted_mission_id
                or planner_output.uav_id != trusted_uav_id
                or planner_output.plan_version != trusted_plan_version
            ):
                raise PlanValidationError(
                    "schema-v2 routing IDs do not match trusted runtime values"
                )
            semantic_draft = planner_output.to_v1()
            result = getattr(
                self,
                "_symbolic_checker",
                SymbolicPlanChecker(),
            ).check(
                semantic_draft,
                world_context=context,
                limits=self.limits,
                policy=self.policy,
            )
            if not result.valid:
                issue = result.issues[0]
                location = "" if issue.step_id is None else f" at {issue.step_id}"
                raise PlanValidationError(
                    f"symbolic plan invalid [{issue.code.value}]{location}: "
                    f"{issue.message}"
                )
            _require_trusted_inspect_candidates(
                semantic_draft,
                trusted_inspect_ids,
            )
            return self._compile_dynamic(
                semantic_draft,
                context,
                source,
                mission_id=planner_output.mission_id,
                uav_id=planner_output.uav_id,
                plan_version=planner_output.plan_version,
                public_output=planner_output,
            )
        if isinstance(planner_output, SkillPlanDraft):
            if source not in _DYNAMIC_SOURCES:
                raise PlanValidationError(
                    "SkillPlanDraft requires source 'dynamic_scripted' or "
                    "'dynamic_llm'"
                )
            result = getattr(
                self,
                "_symbolic_checker",
                SymbolicPlanChecker(),
            ).check(
                planner_output,
                world_context=context,
                limits=self.limits,
                policy=self.policy,
            )
            if not result.valid:
                issue = result.issues[0]
                location = "" if issue.step_id is None else f" at {issue.step_id}"
                raise PlanValidationError(
                    f"symbolic plan invalid [{issue.code.value}]{location}: "
                    f"{issue.message}"
                )
            _require_trusted_inspect_candidates(
                planner_output,
                trusted_inspect_ids,
            )
            # Parsing schema v1 remains supported, but routing it into a new
            # mission is always performed through the explicit adapter.
            public_output: SkillPlanDraft | SkillPlanDraftV2 = (
                migrate_plan_v1_to_v2(
                    planner_output,
                    mission_id=trusted_mission_id,
                    uav_id=trusted_uav_id,
                    plan_version=trusted_plan_version,
                )
            )
            return self._compile_dynamic(
                planner_output,
                context,
                source,
                mission_id=trusted_mission_id,
                uav_id=trusted_uav_id,
                plan_version=trusted_plan_version,
                public_output=public_output,
            )
        raise TypeError(
            "planner_output must be MissionIntent, SkillPlanDraft, or "
            "SkillPlanDraftV2"
        )

    def _compile_legacy(
        self,
        intent: MissionIntent,
        context: PlannerWorldContext,
        source: str,
        *,
        mission_id: str,
        uav_id: str,
        plan_version: int,
    ) -> CompiledMission:
        """Compile the original MissionIntent to its unchanged six-step plan."""

        world = self._trusted_world(context)
        description = _target_description(intent.target_description)
        search_region = _search_region(context, intent.search_region)
        landing_zone = _landing_zone(context, intent.landing_zone)
        takeoff_altitude = _positive_finite_number(
            world.default_altitude
            if intent.takeoff_altitude_m is None
            else intent.takeoff_altitude_m,
            "takeoff_altitude_m",
        )
        track_duration = self._track_duration(intent.track_duration_s)
        self._require_flight_altitude(takeoff_altitude, world, "takeoff altitude")
        if takeoff_altitude < world.initial_uav[2]:
            raise PlanValidationError(
                "takeoff altitude must not be below the initial UAV altitude"
            )

        center, approach, radius = self._trusted_search_geometry(
            search_region,
            world,
            require_disk_in_bounds=False,
        )
        landing_xy, ground_altitude, landing_tolerance = self._trusted_landing_geometry(
            landing_zone,
            world,
        )
        if ground_altitude > takeoff_altitude:
            raise PlanValidationError(
                "landing ground altitude must not exceed the flight altitude"
            )

        # Keep the deterministic six-step shape and legacy target placeholder;
        # trusted LAND geometry is appended without exposing it to the model.
        raw_plan: list[dict[str, object]] = [
            {
                "skill": "TAKEOFF",
                "target_altitude": takeoff_altitude,
                "timeout": world.goto_timeout,
            },
            {
                "skill": "GOTO",
                "position": list(approach),
                "timeout": world.goto_timeout,
            },
            {
                "skill": "SEARCH",
                "center": list(center),
                "radius": radius,
                "target_description": description,
                "search_altitude": takeoff_altitude,
                "timeout": world.search_timeout,
            },
            {
                "skill": "TRACK",
                "target_id": "$SEARCH.result.target_id",
                "desired_altitude": takeoff_altitude,
                "track_duration": track_duration,
            },
            {
                "skill": "GOTO",
                "position": [landing_xy[0], landing_xy[1], takeoff_altitude],
                "timeout": world.goto_timeout,
            },
            {
                "skill": "LAND",
                "ground_altitude": ground_altitude,
                "expected_position_xy": list(landing_xy),
                "zone_tolerance_m": landing_tolerance,
                "timeout": world.land_timeout,
            },
        ]
        task_plan = self._task_plan_from_dicts(
            raw_plan,
            mission_id=mission_id,
            uav_id=uav_id,
            plan_version=plan_version,
        )
        return CompiledMission(
            planner_output=intent,
            task_plan=task_plan,
            source=source,
            compiler_notes=(
                "step_06: trusted landing geometry attached for zone "
                f"{intent.landing_zone!r}",
            ),
        )

    def _compile_dynamic(
        self,
        draft: SkillPlanDraft,
        context: PlannerWorldContext,
        source: str,
        *,
        mission_id: str,
        uav_id: str,
        plan_version: int,
        public_output: SkillPlanDraft | SkillPlanDraftV2,
    ) -> CompiledMission:
        world = self._trusted_world(context)
        self._require_unambiguous_named_locations(context)
        steps = tuple(draft.steps)
        limits = self.limits
        # Cross-step ordering/count/reference rules were already accepted by
        # SymbolicPlanChecker in validate_and_compile().  Compilation below
        # owns only trusted geometry, numeric policy and Goal construction.
        skill_names = tuple(_skill_name(step.skill) for step in steps)

        state = _MissionState.ON_GROUND
        current_altitude = world.initial_uav[2]
        search_step_id: str | None = None
        compiled_steps: list[TaskStep] = []
        total_recovery_attempts = 0
        compiler_notes: list[str] = []

        for index, (draft_step, skill) in enumerate(zip(steps, skill_names)):
            step_id = draft_step.id
            args = draft_step.args
            if not isinstance(args, Mapping):
                raise PlanValidationError(f"step {step_id} args must be a mapping")
            self._reject_unknown_args(step_id, skill, args)
            recovery = getattr(draft_step, "recovery", None)
            assert recovery is None or skill is SkillName.TRACK

            params: dict[str, object]
            compiled_recovery: RecoveryPolicy | None = None
            if state is _MissionState.ON_GROUND:
                if skill is not SkillName.TAKEOFF:
                    raise PlanValidationError(
                        f"illegal state transition: ON_GROUND + {skill.value}"
                    )
            elif state is _MissionState.LANDED:
                raise PlanValidationError("no step is allowed after LAND")

            if skill is SkillName.TAKEOFF:
                altitude = self._dynamic_altitude(
                    args.get("altitude_m", world.default_altitude),
                    world,
                    "TAKEOFF altitude_m",
                )
                if altitude < world.initial_uav[2]:
                    raise PlanValidationError(
                        "TAKEOFF altitude_m must not be below initial UAV altitude"
                    )
                yaw_mode, yaw_value = _vertical_yaw_args(
                    args,
                    default=YawMode.KEEP_CURRENT,
                    prefix=f"step {step_id} TAKEOFF",
                )
                params = {
                    "target_altitude": altitude,
                    "yaw_mode": yaw_mode,
                    "yaw_value": yaw_value,
                    "timeout": world.goto_timeout,
                }
                current_altitude = altitude
                state = _MissionState.AIRBORNE_NO_TARGET
                compiler_notes.append(
                    f"{step_id}: altitude/timeouts bounded by trusted world context"
                )

            elif skill is SkillName.GOTO:
                self._require_airborne(state, skill)
                destination = _required_non_empty_string(
                    args,
                    "destination",
                    f"step {step_id} GOTO",
                )
                altitude = self._dynamic_altitude(
                    args.get("altitude_m", current_altitude),
                    world,
                    f"step {step_id} GOTO altitude_m",
                )
                position, face_point = self._resolve_destination(
                    destination,
                    altitude,
                    context,
                    world,
                )
                motion_policy = _goto_motion_policy(
                    args,
                    look_at_point=face_point,
                    max_speed=self.TRUSTED_GOTO_MAX_SPEED_MPS,
                    prefix=f"step {step_id} GOTO",
                )
                params = {
                    "position": position,
                    "motion_policy": motion_policy,
                    "timeout": world.goto_timeout,
                }
                current_altitude = altitude
                compiler_notes.append(
                    f"{step_id}: destination {destination!r} resolved by trusted context"
                )

            elif skill is SkillName.HOVER:
                self._require_airborne(state, skill)
                duration = _bounded_finite_number(
                    _required_value(
                        args,
                        "duration_s",
                        f"step {step_id} HOVER",
                    ),
                    f"step {step_id} HOVER duration_s",
                    minimum=self.MIN_PLANNED_HOVER_DURATION_S,
                    maximum=self.MAX_PLANNED_HOVER_DURATION_S,
                )
                yaw_mode, yaw_value = _vertical_yaw_args(
                    args,
                    default=YawMode.KEEP_CURRENT,
                    prefix=f"step {step_id} HOVER",
                )
                motion_policy = MotionPolicy(
                    max_speed=self.TRUSTED_HOVER_MAX_CORRECTION_SPEED_MPS,
                    max_yaw_rate=self.TRUSTED_HOVER_MAX_YAW_RATE_RAD_S,
                    yaw_mode=yaw_mode,
                    yaw_value=yaw_value,
                )
                try:
                    motion_policy.validate()
                except (TypeError, ValueError) as exc:
                    raise PlanValidationError(
                        f"step {step_id} HOVER motion policy is invalid: {exc}"
                    ) from exc
                from skills.hover import HoverMode

                params = {
                    "mode": HoverMode.TIMED,
                    "duration_s": duration,
                    "max_wait_s": duration,
                    "position_tolerance_m": (
                        self.TRUSTED_HOVER_POSITION_TOLERANCE_M
                    ),
                    "max_correction_speed_mps": (
                        self.TRUSTED_HOVER_MAX_CORRECTION_SPEED_MPS
                    ),
                    "reason_code": "PLANNED_HOVER",
                    "motion_policy": motion_policy,
                }
                compiler_notes.append(
                    f"{step_id}: supervisory-only HOVER fields supplied by "
                    "trusted compiler"
                )

            elif skill is SkillName.SEARCH:
                if state is not _MissionState.AIRBORNE_NO_TARGET:
                    raise PlanValidationError(
                        f"illegal state transition: {state.value} + SEARCH"
                    )
                region_name = _required_non_empty_string(
                    args,
                    "region",
                    f"step {step_id} SEARCH",
                )
                region = _search_region(context, region_name)
                center, _approach, radius = self._trusted_search_geometry(
                    region,
                    world,
                    require_disk_in_bounds=True,
                )
                altitude = self._dynamic_altitude(
                    args.get("altitude_m", current_altitude),
                    world,
                    f"step {step_id} SEARCH altitude_m",
                )
                description = _target_description(
                    _required_non_empty_string(
                        args,
                        "target_description",
                        f"step {step_id} SEARCH",
                    )
                )
                params = {
                    "center": center,
                    "radius": radius,
                    "target_description": description,
                    "search_altitude": altitude,
                    "timeout": world.search_timeout,
                }
                current_altitude = altitude
                search_step_id = step_id
                state = _MissionState.AIRBORNE_TARGET_AVAILABLE
                compiler_notes.append(
                    f"{step_id}: region center/radius resolved by trusted context"
                )

            elif skill is SkillName.INSPECT:
                if state is not _MissionState.AIRBORNE_TARGET_AVAILABLE:
                    raise PlanValidationError("INSPECT must appear after SEARCH")
                raw_candidate_id = _required_value(
                    args,
                    "candidate_id",
                    f"step {step_id} INSPECT",
                )
                try:
                    candidate_id = validate_routing_id(
                        raw_candidate_id,
                        "candidate_id",
                    )
                except (TypeError, ValueError) as exc:
                    raise PlanValidationError(str(exc)) from None
                distance = _bounded_finite_number(
                    args.get(
                        "desired_observation_distance_m",
                        self.DEFAULT_INSPECT_OBSERVATION_DISTANCE_M,
                    ),
                    (
                        f"step {step_id} INSPECT "
                        "desired_observation_distance_m"
                    ),
                    minimum=self.MIN_INSPECT_OBSERVATION_DISTANCE_M,
                    maximum=self.MAX_INSPECT_OBSERVATION_DISTANCE_M,
                )
                viewpoint_degrees = _bounded_finite_number(
                    args.get(
                        "viewpoint_change_deg",
                        self.DEFAULT_INSPECT_VIEWPOINT_CHANGE_DEG,
                    ),
                    f"step {step_id} INSPECT viewpoint_change_deg",
                    minimum=-self.MAX_INSPECT_VIEWPOINT_CHANGE_DEG,
                    maximum=self.MAX_INSPECT_VIEWPOINT_CHANGE_DEG,
                )
                if abs(viewpoint_degrees) <= 1e-9:
                    raise PlanValidationError(
                        f"step {step_id} INSPECT viewpoint_change_deg must be "
                        "non-zero"
                    )
                max_duration = _bounded_finite_number(
                    args.get(
                        "max_duration_s",
                        self.DEFAULT_INSPECT_MAX_DURATION_S,
                    ),
                    f"step {step_id} INSPECT max_duration_s",
                    minimum=1.0,
                    maximum=self.MAX_INSPECT_MAX_DURATION_S,
                )
                from skills.inspect import InspectApproachPolicy

                raw_approach = args.get(
                    "approach_policy",
                    InspectApproachPolicy.MAINTAIN_ALTITUDE_ORBIT.value,
                )
                try:
                    approach_policy = InspectApproachPolicy(raw_approach)
                except (TypeError, ValueError):
                    raise PlanValidationError(
                        f"step {step_id} INSPECT approach_policy must be "
                        "MAINTAIN_ALTITUDE_ORBIT"
                    ) from None
                params = {
                    "candidate_id": candidate_id,
                    "desired_observation_distance_m": distance,
                    "viewpoint_change_rad": radians(viewpoint_degrees),
                    "max_duration_s": max_duration,
                    "approach_policy": approach_policy,
                }
                compiler_notes.append(
                    f"{step_id}: candidate semantics compiled without world "
                    "geometry"
                )

            elif skill is SkillName.TRACK:
                if state is not _MissionState.AIRBORNE_TARGET_AVAILABLE:
                    raise PlanValidationError("TRACK must appear after SEARCH")
                if search_step_id is None:
                    raise PlanValidationError("TRACK has no preceding SEARCH output")
                target_ref = _parse_target_ref(
                    _required_value(args, "target_ref", f"step {step_id} TRACK"),
                    prefix=f"step {step_id} TRACK",
                )
                if target_ref.step_id != search_step_id:
                    raise PlanValidationError(
                        "TRACK target_ref must reference the one preceding SEARCH step"
                    )
                duration = self._track_duration(
                    args.get("duration_s", world.default_track_duration)
                )
                desired_altitude = self._dynamic_altitude(
                    args.get("desired_altitude_m", current_altitude),
                    world,
                    f"step {step_id} TRACK desired_altitude_m",
                )
                desired_distance = _positive_finite_number(
                    args.get(
                        "desired_distance_m",
                        self.DEFAULT_DESIRED_DISTANCE_M,
                    ),
                    f"step {step_id} TRACK desired_distance_m",
                )
                max_distance = hypot(
                    world.scene_max[0] - world.scene_min[0],
                    world.scene_max[1] - world.scene_min[1],
                )
                if not isfinite(max_distance) or desired_distance > max_distance:
                    raise PlanValidationError(
                        f"step {step_id} TRACK desired_distance_m exceeds "
                        "the trusted scene scale"
                    )
                params = {
                    "target_id": target_ref,
                    "desired_distance": desired_distance,
                    "desired_altitude": desired_altitude,
                    "max_speed": self.TRUSTED_TRACK_MAX_SPEED_MPS,
                    "max_target_lost_time": self.TRUSTED_TARGET_LOST_TIME_S,
                    "track_duration": duration,
                    "timeout": duration + self.TRUSTED_TRACK_TIMEOUT_GRACE_S,
                }
                current_altitude = desired_altitude
                compiled_recovery, recovery_note = self._compile_track_recovery(
                    args=args,
                    recovery=recovery,
                    step_id=step_id,
                )
                compiler_notes.append(f"{step_id}: {recovery_note}")
                if compiled_recovery is not None:
                    total_recovery_attempts += compiled_recovery.max_attempts
                    if (
                        total_recovery_attempts
                        > limits.max_total_reacquire_attempts
                    ):
                        raise PlanValidationError(
                            "total REACQUIRE attempt budget exceeds planner limit"
                        )

            elif skill is SkillName.LAND:
                self._require_airborne(state, skill)
                zone_name = _required_non_empty_string(
                    args,
                    "zone",
                    f"step {step_id} LAND",
                )
                zone = _landing_zone(context, zone_name)
                landing_xy, ground_altitude, landing_tolerance = (
                    self._trusted_landing_geometry(
                    zone,
                    world,
                    )
                )
                if ground_altitude > current_altitude:
                    raise PlanValidationError(
                        "landing ground altitude must not exceed current flight altitude"
                    )
                self._validate_landing_precondition(
                    steps,
                    index,
                    zone_name,
                    context,
                    world,
                )
                yaw_mode, yaw_value = _vertical_yaw_args(
                    args,
                    default=YawMode.KEEP_CURRENT,
                    prefix=f"step {step_id} LAND",
                )
                params = {
                    "ground_altitude": ground_altitude,
                    "expected_position_xy": landing_xy,
                    "zone_tolerance_m": landing_tolerance,
                    "yaw_mode": yaw_mode,
                    "yaw_value": yaw_value,
                    "timeout": world.land_timeout,
                }
                state = _MissionState.LANDED
                compiler_notes.append(
                    f"{step_id}: trusted landing geometry attached for zone "
                    f"{zone_name!r}"
                )

            else:  # pragma: no cover - guarded by _skill_name and catalog set
                raise PlanValidationError(f"unsupported top-level Skill: {skill.value}")

            compiled_steps.append(
                TaskStep(step_id, skill, params, compiled_recovery)
            )

        if state is not _MissionState.LANDED:
            raise PlanValidationError("dynamic plan did not end in LANDED state")
        try:
            task_plan = TaskPlan(
                tuple(compiled_steps),
                mission_id=mission_id,
                uav_id=uav_id,
                plan_version=plan_version,
            )
        except TaskPlanError as exc:
            raise PlanValidationError(f"compiled TaskPlan is invalid: {exc}") from exc
        return CompiledMission(
            planner_output=public_output,
            task_plan=task_plan,
            source=source,
            compiler_notes=tuple(compiler_notes),
        )

    def _trusted_world(self, context: PlannerWorldContext) -> _TrustedWorld:
        scene_min = _finite_vector3(context.scene_min_xyz_m, "scene_min_xyz_m")
        scene_max = _finite_vector3(context.scene_max_xyz_m, "scene_max_xyz_m")
        if any(lower >= upper for lower, upper in zip(scene_min, scene_max)):
            raise PlanValidationError(
                "each scene_min_xyz_m component must be smaller than scene_max_xyz_m"
            )
        initial_uav = _finite_vector3(
            context.initial_uav_xyz_m,
            "initial_uav_xyz_m",
        )
        _require_point_in_bounds(
            initial_uav,
            scene_min,
            scene_max,
            "initial UAV position",
        )
        return _TrustedWorld(
            scene_min=scene_min,
            scene_max=scene_max,
            initial_uav=initial_uav,
            default_altitude=_positive_finite_number(
                context.default_takeoff_altitude_m,
                "default_takeoff_altitude_m",
            ),
            default_track_duration=_positive_finite_number(
                context.default_track_duration_s,
                "default_track_duration_s",
            ),
            search_timeout=_positive_finite_number(
                context.search_timeout_s,
                "search_timeout_s",
            ),
            goto_timeout=_positive_finite_number(
                context.goto_timeout_s,
                "goto_timeout_s",
            ),
            land_timeout=_positive_finite_number(
                context.land_timeout_s,
                "land_timeout_s",
            ),
        )

    def _trusted_search_geometry(
        self,
        region: SearchRegionSpec,
        world: _TrustedWorld,
        *,
        require_disk_in_bounds: bool = True,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
        center = _finite_vector3(region.center_xyz_m, "search center")
        approach = _finite_vector3(region.approach_xyz_m, "search approach")
        radius = _positive_finite_number(region.radius_m, "search radius")
        _require_point_in_bounds(
            center,
            world.scene_min,
            world.scene_max,
            "search center",
        )
        _require_point_in_bounds(
            approach,
            world.scene_min,
            world.scene_max,
            "search approach",
        )
        if require_disk_in_bounds and (
            center[0] - radius < world.scene_min[0]
            or center[0] + radius > world.scene_max[0]
            or center[1] - radius < world.scene_min[1]
            or center[1] + radius > world.scene_max[1]
        ):
            raise PlanValidationError("search radius leaves the scene bounds")
        return center, approach, radius

    def _trusted_landing_geometry(
        self,
        zone: LandingZoneSpec,
        world: _TrustedWorld,
    ) -> tuple[tuple[float, float], float, float]:
        landing_xy = _finite_vector2(zone.position_xy_m, "landing position_xy_m")
        if not (
            world.scene_min[0] <= landing_xy[0] <= world.scene_max[0]
            and world.scene_min[1] <= landing_xy[1] <= world.scene_max[1]
        ):
            raise PlanValidationError("landing position is outside the scene XY bounds")
        ground_altitude = _finite_number(
            zone.ground_altitude_m,
            "landing ground_altitude_m",
        )
        if not world.scene_min[2] <= ground_altitude <= world.scene_max[2]:
            raise PlanValidationError(
                "landing ground altitude is outside the scene Z bounds"
            )
        horizontal_tolerance = _positive_finite_number(
            zone.horizontal_tolerance_m,
            "landing horizontal_tolerance_m",
        )
        return landing_xy, ground_altitude, horizontal_tolerance

    def _dynamic_altitude(
        self,
        value: object,
        world: _TrustedWorld,
        name: str,
    ) -> float:
        altitude = _positive_finite_number(value, name)
        self._require_flight_altitude(altitude, world, name)
        return altitude

    @staticmethod
    def _require_flight_altitude(
        altitude: float,
        world: _TrustedWorld,
        name: str,
    ) -> None:
        if not world.scene_min[2] < altitude <= world.scene_max[2]:
            raise PlanValidationError(f"{name} is outside the scene Z bounds")

    def _track_duration(self, value: object) -> float:
        duration = _positive_finite_number(value, "track_duration_s")
        limits = self.limits
        if not (
            limits.min_track_duration_s
            <= duration
            <= limits.max_track_duration_s
        ):
            raise PlanValidationError(
                "track_duration_s must be between "
                f"{limits.min_track_duration_s:g} and "
                f"{limits.max_track_duration_s:g} seconds"
            )
        return duration

    @staticmethod
    def _require_airborne(state: _MissionState, skill: SkillName) -> None:
        if state not in {
            _MissionState.AIRBORNE_NO_TARGET,
            _MissionState.AIRBORNE_TARGET_AVAILABLE,
        }:
            raise PlanValidationError(
                f"illegal state transition: {state.value} + {skill.value}"
            )

    @staticmethod
    def _reject_unknown_args(
        step_id: str,
        skill: SkillName,
        args: Mapping[str, object],
    ) -> None:
        allowed = _DYNAMIC_ARGUMENTS.get(skill)
        if allowed is None:
            raise PlanValidationError(f"unsupported top-level Skill: {skill.value}")
        if any(not isinstance(key, str) for key in args):
            raise PlanValidationError(f"step {step_id} args keys must be strings")
        unknown = sorted(set(args) - allowed)
        if unknown:
            raise PlanValidationError(
                f"step {step_id} {skill.value} contains unknown args: "
                + ", ".join(unknown)
            )

    def _resolve_destination(
        self,
        destination: str,
        altitude: float,
        context: PlannerWorldContext,
        world: _TrustedWorld,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        if destination in context.search_regions:
            center, approach, _radius = self._trusted_search_geometry(
                _search_region(context, destination),
                world,
                require_disk_in_bounds=False,
            )
            point = (approach[0], approach[1], altitude)
            face_point = center
        elif destination in context.landing_zones:
            landing_xy, ground, _tolerance = self._trusted_landing_geometry(
                _landing_zone(context, destination),
                world,
            )
            point = (landing_xy[0], landing_xy[1], altitude)
            face_point = (landing_xy[0], landing_xy[1], ground)
        else:
            navigation_points = getattr(context, "navigation_points", {})
            if not isinstance(navigation_points, Mapping) or destination not in navigation_points:
                raise PlanValidationError(f"unknown destination: {destination}")
            spec = navigation_points[destination]
            raw_position = getattr(spec, "position_xyz_m", None)
            if raw_position is None and isinstance(spec, Mapping):
                raw_position = spec.get("position_xyz_m")
            trusted_position = _finite_vector3(
                raw_position,
                f"navigation point {destination!r}",
            )
            point = (trusted_position[0], trusted_position[1], altitude)
            face_point = trusted_position
        _require_point_in_bounds(
            point,
            world.scene_min,
            world.scene_max,
            f"destination {destination!r}",
        )
        _require_point_in_bounds(
            face_point,
            world.scene_min,
            world.scene_max,
            f"destination {destination!r} FACE_POINT target",
        )
        return point, face_point

    @staticmethod
    def _require_unambiguous_named_locations(
        context: PlannerWorldContext,
    ) -> None:
        groups = (
            set(context.search_regions),
            set(context.landing_zones),
            set(context.navigation_points),
        )
        ambiguous = sorted(
            (groups[0] & groups[1])
            | (groups[0] & groups[2])
            | (groups[1] & groups[2])
        )
        if ambiguous:
            raise PlanValidationError(
                "named locations are ambiguous across world-context categories: "
                + ", ".join(ambiguous)
            )

    def _compile_track_recovery(
        self,
        *,
        args: Mapping[str, object],
        recovery: object | None,
        step_id: str,
    ) -> tuple[RecoveryPolicy | None, str]:
        """Resolve model intent to one explicit trusted recovery policy."""

        raw_action = args.get("on_target_lost")
        if raw_action is None:
            # A pre-policy hand-written plan containing only ``recovery`` is
            # interpreted exactly as it was before this protocol field existed.
            action = (
                TargetLostAction.REACQUIRE
                if recovery is not None
                else self.policy.default_on_target_lost
            )
        else:
            try:
                action = TargetLostAction(str(raw_action))
            except ValueError:
                raise PlanValidationError(
                    f"step {step_id} on_target_lost must be REACQUIRE or FAIL"
                ) from None

        if action is TargetLostAction.FAIL:
            if recovery is not None:
                # SymbolicPlanChecker is the public authority for this error;
                # this branch protects direct/internal compiler reuse.
                raise PlanValidationError(
                    f"step {step_id} recovery conflicts with on_target_lost=FAIL"
                )
            note = (
                "recovery explicitly disabled by on_target_lost=FAIL"
                if raw_action is not None
                else "recovery disabled by trusted default policy"
            )
            return None, note

        if recovery is None:
            return self._default_recovery_policy(), (
                "recovery injected from trusted default policy"
            )

        attempts = _nonnegative_integer(
            getattr(recovery, "max_attempts", None),
            f"step {step_id} recovery.max_attempts",
        )
        if attempts == 0:
            return None, (
                "recovery disabled by deprecated max_attempts=0 compatibility"
            )
        return self._compile_recovery(recovery, step_id), (
            "recovery explicitly enabled with bounded overrides"
        )

    def _default_recovery_policy(self) -> RecoveryPolicy:
        policy = self.policy
        try:
            return RecoveryPolicy(
                skill=SkillName.REACQUIRE,
                max_attempts=policy.default_reacquire_max_attempts,
                search_radius_m=policy.default_reacquire_search_radius_m,
                timeout_s=policy.default_reacquire_timeout_s,
            )
        except (TypeError, ValueError, TaskPlanError) as exc:
            raise PlanValidationError(
                f"trusted default recovery policy is invalid: {exc}"
            ) from exc

    def _compile_recovery(
        self,
        recovery: object,
        step_id: str,
    ) -> RecoveryPolicy:
        skill = _skill_name(getattr(recovery, "skill", None))
        if skill is not SkillName.REACQUIRE:
            raise PlanValidationError(
                f"step {step_id} recovery.skill must be REACQUIRE"
            )
        max_attempts = _nonnegative_integer(
            getattr(recovery, "max_attempts", None),
            f"step {step_id} recovery.max_attempts",
        )
        if max_attempts > self.limits.max_reacquire_attempts_per_track:
            raise PlanValidationError(
                f"step {step_id} recovery attempts exceed planner limit"
            )
        radius_value = getattr(recovery, "search_radius_m", None)
        timeout_value = getattr(recovery, "timeout_s", None)
        if radius_value is None:
            radius_value = self.policy.default_reacquire_search_radius_m
        if timeout_value is None:
            timeout_value = self.policy.default_reacquire_timeout_s
        radius = _finite_number(radius_value, "recovery.search_radius_m")
        timeout = _finite_number(timeout_value, "recovery.timeout_s")
        if not 3.0 <= radius <= 20.0:
            raise PlanValidationError("recovery.search_radius_m must be between 3 and 20")
        if not 5.0 <= timeout <= 60.0:
            raise PlanValidationError("recovery.timeout_s must be between 5 and 60")
        try:
            return RecoveryPolicy(
                skill=SkillName.REACQUIRE,
                max_attempts=max_attempts,
                search_radius_m=radius,
                timeout_s=timeout,
            )
        except (TypeError, ValueError, TaskPlanError) as exc:
            raise PlanValidationError(f"invalid recovery policy: {exc}") from exc

    def _validate_landing_precondition(
        self,
        draft_steps: tuple[object, ...],
        index: int,
        zone_name: str,
        context: PlannerWorldContext,
        world: _TrustedWorld,
    ) -> None:
        # LandSkill locks the current XY; it does not navigate to the zone.
        if index == 1:
            zone = _landing_zone(context, zone_name)
            landing_xy, _ground, tolerance = self._trusted_landing_geometry(
                zone,
                world,
            )
            if _same_xy(world.initial_uav, landing_xy, tolerance):
                return
            raise PlanValidationError(
                "TAKEOFF -> LAND is only valid when initial XY is in that zone"
            )
        previous = draft_steps[index - 1]
        if _skill_name(previous.skill) is not SkillName.GOTO:
            raise PlanValidationError(
                "LAND must be immediately preceded by GOTO to the same zone"
            )
        previous_destination = previous.args.get("destination")
        if previous_destination != zone_name:
            raise PlanValidationError(
                "LAND must be immediately preceded by GOTO to the same zone"
            )

    @staticmethod
    def _task_plan_from_dicts(
        raw_plan: Sequence[Mapping[str, object]],
        *,
        mission_id: str,
        uav_id: str,
        plan_version: int,
    ) -> TaskPlan:
        try:
            return TaskPlan.from_dicts(
                raw_plan,
                mission_id=mission_id,
                uav_id=uav_id,
                plan_version=plan_version,
            )
        except TaskPlanError as exc:
            raise PlanValidationError(f"compiled TaskPlan is invalid: {exc}") from exc


def _skill_name(value: object) -> SkillName:
    if isinstance(value, SkillName):
        return value
    if isinstance(value, str):
        try:
            return SkillName(value.strip().upper())
        except ValueError as exc:
            raise PlanValidationError(f"unknown Skill: {value}") from exc
    raise PlanValidationError("Skill name must be a string or SkillName")


def _search_region(context: PlannerWorldContext, name: object) -> SearchRegionSpec:
    if not isinstance(name, str) or not name.strip():
        raise PlanValidationError("search region name must be a non-empty string")
    try:
        region = context.search_regions[name]
    except KeyError as exc:
        raise PlanValidationError(f"unknown search_region: {name}") from exc
    if not isinstance(region, SearchRegionSpec):
        raise PlanValidationError(f"search region {name!r} has an invalid specification")
    return region


def _landing_zone(context: PlannerWorldContext, name: object) -> LandingZoneSpec:
    if not isinstance(name, str) or not name.strip():
        raise PlanValidationError("landing zone name must be a non-empty string")
    try:
        zone = context.landing_zones[name]
    except KeyError as exc:
        raise PlanValidationError(f"unknown landing_zone: {name}") from exc
    if not isinstance(zone, LandingZoneSpec):
        raise PlanValidationError(f"landing zone {name!r} has an invalid specification")
    return zone


def _target_description(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanValidationError("target_description must be non-empty")
    description = value.strip()
    if len(description) > PlanValidator.MAX_TARGET_DESCRIPTION_CHARS:
        raise PlanValidationError(
            "target_description must contain at most "
            f"{PlanValidator.MAX_TARGET_DESCRIPTION_CHARS} characters"
        )
    return description


def _required_value(
    args: Mapping[str, object],
    name: str,
    prefix: str,
) -> object:
    if name not in args:
        raise PlanValidationError(f"{prefix} is missing required arg {name}")
    return args[name]


def _required_non_empty_string(
    args: Mapping[str, object],
    name: str,
    prefix: str,
) -> str:
    value = _required_value(args, name, prefix)
    if not isinstance(value, str) or not value.strip():
        raise PlanValidationError(f"{prefix} {name} must be a non-empty string")
    return value.strip()


def _yaw_mode(value: object, name: str) -> YawMode:
    if isinstance(value, YawMode):
        return value
    if isinstance(value, str):
        try:
            return YawMode[value.strip().upper()]
        except KeyError as exc:
            raise PlanValidationError(f"{name} is unknown") from exc
    raise PlanValidationError(f"{name} must be a string")


def _vertical_yaw_args(
    args: Mapping[str, object],
    *,
    default: YawMode,
    prefix: str,
) -> tuple[YawMode, float | None]:
    mode = _yaw_mode(args.get("yaw_mode", default), f"{prefix} yaw_mode")
    if mode not in {YawMode.KEEP_CURRENT, YawMode.FIXED}:
        raise PlanValidationError(
            f"{prefix} yaw_mode must be KEEP_CURRENT or FIXED"
        )
    yaw_deg = args.get("yaw_deg")
    if mode is YawMode.FIXED:
        if yaw_deg is None:
            raise PlanValidationError(f"{prefix} FIXED yaw_mode requires yaw_deg")
        return mode, radians(_bounded_yaw_degrees(yaw_deg, f"{prefix} yaw_deg"))
    if yaw_deg is not None:
        raise PlanValidationError(f"{prefix} yaw_deg is only valid with FIXED")
    return mode, None


def _goto_motion_policy(
    args: Mapping[str, object],
    *,
    look_at_point: tuple[float, float, float],
    max_speed: float,
    prefix: str,
) -> MotionPolicy:
    mode = _yaw_mode(
        args.get("yaw_mode", YawMode.COURSE_ALIGNED),
        f"{prefix} yaw_mode",
    )
    if mode not in {
        YawMode.KEEP_CURRENT,
        YawMode.COURSE_ALIGNED,
        YawMode.FACE_POINT,
        YawMode.FIXED,
    }:
        raise PlanValidationError(f"{prefix} yaw_mode is unsupported")
    yaw_deg = args.get("yaw_deg")
    yaw_value: float | None = None
    if mode is YawMode.FIXED:
        if yaw_deg is None:
            raise PlanValidationError(f"{prefix} FIXED yaw_mode requires yaw_deg")
        yaw_value = radians(_bounded_yaw_degrees(yaw_deg, f"{prefix} yaw_deg"))
    elif yaw_deg is not None:
        raise PlanValidationError(f"{prefix} yaw_deg is only valid with FIXED")
    policy = MotionPolicy(
        max_speed=max_speed,
        yaw_mode=mode,
        yaw_value=yaw_value,
        look_at_point=look_at_point if mode is YawMode.FACE_POINT else None,
    )
    try:
        policy.validate()
    except (TypeError, ValueError) as exc:
        raise PlanValidationError(f"{prefix} motion policy is invalid: {exc}") from exc
    return policy


def _parse_target_ref(value: object, *, prefix: str) -> StepOutputRef:
    if isinstance(value, StepOutputRef):
        ref = value
    elif isinstance(value, str):
        if not value.startswith("$") or value.count(".") != 1:
            raise PlanValidationError(
                f"{prefix} target_ref must be $<SEARCH step id>.target_id"
            )
        step_id, field = value[1:].split(".", 1)
        try:
            ref = StepOutputRef(step_id=step_id, field=field)
        except (TypeError, ValueError, TaskPlanError) as exc:
            raise PlanValidationError(f"{prefix} target_ref is invalid: {exc}") from exc
    else:
        raise PlanValidationError(
            f"{prefix} target_ref must be $<SEARCH step id>.target_id"
        )
    if ref.field != "target_id":
        raise PlanValidationError(f"{prefix} target_ref field must be target_id")
    return ref


def _bounded_yaw_degrees(value: object, name: str) -> float:
    degrees = _finite_number(value, name)
    if not -360.0 <= degrees <= 360.0:
        raise PlanValidationError(f"{name} must be between -360 and 360 degrees")
    return degrees


def _same_xy(
    point: Sequence[float],
    xy: Sequence[float],
    tolerance: float = 1e-6,
) -> bool:
    return abs(point[0] - xy[0]) <= tolerance and abs(point[1] - xy[1]) <= tolerance


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlanValidationError(f"{name} must be an integer between 0 and 2")
    return value


def _finite_number(
    value: object,
    name: str,
    *,
    error_type: type[ValueError] = PlanValidationError,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise error_type(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise error_type(f"{name} must be a finite number") from exc
    if not isfinite(parsed):
        raise error_type(f"{name} must be a finite number")
    return parsed


def _positive_finite_number(
    value: object,
    name: str,
    *,
    error_type: type[ValueError] = PlanValidationError,
) -> float:
    parsed = _finite_number(value, name, error_type=error_type)
    if parsed <= 0.0:
        raise error_type(f"{name} must be greater than zero")
    return parsed


def _bounded_finite_number(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    parsed = _finite_number(value, name)
    if not minimum <= parsed <= maximum:
        raise PlanValidationError(
            f"{name} must be between {minimum:g} and {maximum:g}"
        )
    return parsed


def _finite_vector(
    value: object,
    size: int,
    name: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise PlanValidationError(f"{name} must contain exactly {size} numbers")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise PlanValidationError(
            f"{name} must contain exactly {size} numbers"
        ) from exc
    if len(values) != size:
        raise PlanValidationError(f"{name} must contain exactly {size} numbers")
    return tuple(_finite_number(component, name) for component in values)


def _finite_vector2(value: object, name: str) -> tuple[float, float]:
    parsed = _finite_vector(value, 2, name)
    return parsed[0], parsed[1]


def _finite_vector3(value: object, name: str) -> tuple[float, float, float]:
    parsed = _finite_vector(value, 3, name)
    return parsed[0], parsed[1], parsed[2]


def _require_point_in_bounds(
    point: Sequence[float],
    scene_min: Sequence[float],
    scene_max: Sequence[float],
    name: str,
) -> None:
    if any(
        coordinate < lower or coordinate > upper
        for coordinate, lower, upper in zip(point, scene_min, scene_max)
    ):
        raise PlanValidationError(f"{name} is outside the scene bounds")


__all__ = ["PlanValidationError", "PlannerLimits", "PlanValidator"]
