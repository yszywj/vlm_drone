"""Trusted compilation of an accepted obstacle route into a complete TaskPlan.

The model-facing obstacle revision protocol deliberately contains semantic
steps, not executable world geometry.  This module is the narrow bridge to the
runtime representation: the accepted route is referenced by ID, while later
empty-argument steps may only reuse already compiled parameters from the
interrupted plan.  It never resolves a named place or invents coordinates.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite, pi
from numbers import Real

from planner.obstacle_revision import (
    ObstacleReplacementStep,
    ObstacleRouteRevisionDraft,
)
from planner.spatial import (
    NamedLocationTarget,
    PointTarget,
    RouteTarget,
    spatial_target_from_dict,
)
from planner.spatial_resolver import SpatialResolver
from skills.hover import HoverMode
from skills.motion_types import MotionPolicy, YawMode
from skills.plan import (
    RecoveryPolicy,
    StepOutputRef,
    TaskPlan,
    TaskPlanError,
    TaskStep,
)
from skills.types import SkillName


_LEGACY_SEARCH_TARGET_REF = "$SEARCH.result.target_id"
_ROUTE_SUBSTITUTES_INTERRUPTED = frozenset(
    {SkillName.GOTO, SkillName.FOLLOW_ROUTE}
)


class ObstacleReplacementCompilationError(ValueError):
    """Raised before an ungrounded or structurally unsafe suffix is published."""


def _bounded_positive(
    value: object,
    name: str,
    *,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized) or not 0.0 < normalized <= maximum:
        raise ValueError(f"{name} must be within (0, {maximum:g}]")
    return normalized


@dataclass(frozen=True, slots=True)
class TrustedFollowRouteDefaults:
    """Bounded runtime defaults unavailable for model control."""

    tolerance_m: float = 0.75
    timeout_s: float = 120.0
    max_speed_mps: float = 2.0
    max_yaw_rate_rad_s: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tolerance_m",
            _bounded_positive(self.tolerance_m, "tolerance_m", maximum=5.0),
        )
        object.__setattr__(
            self,
            "timeout_s",
            _bounded_positive(self.timeout_s, "timeout_s", maximum=900.0),
        )
        object.__setattr__(
            self,
            "max_speed_mps",
            _bounded_positive(self.max_speed_mps, "max_speed_mps", maximum=5.0),
        )
        object.__setattr__(
            self,
            "max_yaw_rate_rad_s",
            _bounded_positive(
                self.max_yaw_rate_rad_s,
                "max_yaw_rate_rad_s",
                maximum=pi,
            ),
        )

    def task_params(self, route_ref: str) -> dict[str, object]:
        policy = MotionPolicy(
            max_speed=self.max_speed_mps,
            max_yaw_rate=self.max_yaw_rate_rad_s,
            yaw_mode=YawMode.COURSE_ALIGNED,
        )
        policy.validate()
        return {
            "route_ref": route_ref,
            "tolerance_m": self.tolerance_m,
            "timeout_s": self.timeout_s,
            "motion_policy": policy,
        }


class ObstacleReplacementCompiler:
    """Compile one accepted route proposal without changing trusted geometry."""

    def __init__(
        self,
        defaults: TrustedFollowRouteDefaults | None = None,
    ) -> None:
        if defaults is None:
            defaults = TrustedFollowRouteDefaults()
        if not isinstance(defaults, TrustedFollowRouteDefaults):
            raise TypeError("defaults must be TrustedFollowRouteDefaults")
        self._defaults = defaults

    @property
    def defaults(self) -> TrustedFollowRouteDefaults:
        return self._defaults

    def compile(
        self,
        draft: ObstacleRouteRevisionDraft,
        interrupted_plan: TaskPlan,
        interrupted_index: int,
        *,
        spatial_resolver: SpatialResolver | None = None,
    ) -> TaskPlan:
        """Return a full, routed ``plan_version + 1`` replacement.

        ``args={}`` means "reuse one remaining trusted compiled template of
        this Skill".  The model may reorder or omit suffix Skills, but cannot
        invent their controller parameters. Explicit GOTO/LAND arguments are
        accepted only when a trusted spatial resolver proves that they select
        an existing remaining compiled objective. A planned HOVER may choose a
        bounded duration while all hold-control parameters remain trusted.

        A route replaces an interrupted GOTO or FOLLOW_ROUTE; other
        interrupted Skills remain eligible to be resumed after the detour.
        A model ``target_continuation`` is consumed here and never becomes a
        controller parameter. ``REACQUIRE`` reuses only the interrupted
        TRACK's trusted RecoveryPolicy; it is never emitted as a TaskStep.
        """

        if not isinstance(draft, ObstacleRouteRevisionDraft):
            raise TypeError("draft must be an ObstacleRouteRevisionDraft")
        if not isinstance(interrupted_plan, TaskPlan):
            raise TypeError("interrupted_plan must be a TaskPlan")
        if isinstance(interrupted_index, bool) or not isinstance(
            interrupted_index, int
        ):
            raise TypeError("interrupted_index must be an integer")
        if not 0 <= interrupted_index < len(interrupted_plan.steps):
            raise ObstacleReplacementCompilationError(
                "interrupted_index is outside interrupted_plan"
            )

        self._validate_envelope(draft, interrupted_plan, interrupted_index)
        if spatial_resolver is not None and not isinstance(
            spatial_resolver, SpatialResolver
        ):
            raise TypeError("spatial_resolver must be a SpatialResolver or None")
        prefix = interrupted_plan.steps[:interrupted_index]
        model_steps = draft.replacement_steps
        first = model_steps[0]
        follow_route = TaskStep(
            first.step_id,
            SkillName.FOLLOW_ROUTE,
            self._defaults.task_params(draft.route_draft.route_id),
        )

        # Identity mappings preserve references into the completed prefix.
        step_id_map = {step.step_id: step.step_id for step in prefix}
        compiled_suffix: list[TaskStep] = [follow_route]
        matched_original_indices: list[int] = []
        interrupted_skill = interrupted_plan.steps[interrupted_index].skill
        template_start = interrupted_index + (
            1 if interrupted_skill in _ROUTE_SUBSTITUTES_INTERRUPTED else 0
        )
        used_template_indices: set[int] = set()

        for model_position, model_step in enumerate(model_steps[1:], start=1):
            skill = self._skill(model_step)
            if model_step.args and skill is SkillName.HOVER:
                compiled_suffix.append(self._compile_explicit_hover(model_step))
                continue
            continuation = model_step.target_continuation
            if continuation is not None:
                if model_position != 1:
                    raise ObstacleReplacementCompilationError(
                        "target_continuation must be the first step after FOLLOW_ROUTE"
                    )
                match_index = self._match_target_continuation_template(
                    model_step,
                    interrupted_plan,
                    interrupted_index=interrupted_index,
                    used_template_indices=used_template_indices,
                )
            elif model_step.args:
                match_index = self._match_explicit_template(
                    model_step,
                    interrupted_plan,
                    template_start=template_start,
                    used_template_indices=used_template_indices,
                    spatial_resolver=spatial_resolver,
                )
            else:
                match_index = self._next_template_index(
                    interrupted_plan,
                    template_start=template_start,
                    skill=skill,
                    used_template_indices=used_template_indices,
                )
            used_template_indices.add(match_index)
            template = interrupted_plan.steps[match_index]
            step_id_map[template.step_id] = model_step.step_id
            compiled = self._copy_template(
                model_step,
                template,
                step_id_map=step_id_map,
                prior_steps=(*prefix, *compiled_suffix),
            )
            compiled_suffix.append(compiled)
            matched_original_indices.append(match_index)

        self._validate_terminal_rejoin(
            interrupted_plan,
            interrupted_index=interrupted_index,
            matched_original_indices=tuple(matched_original_indices),
        )
        result_steps = (*prefix, *compiled_suffix)
        self._validate_full_ids(result_steps)
        self._validate_replacement_track_refs(
            result_steps,
            replacement_start=interrupted_index,
        )
        try:
            return TaskPlan(
                tuple(result_steps),
                mission_id=interrupted_plan.mission_id,
                uav_id=interrupted_plan.uav_id,
                plan_version=interrupted_plan.plan_version + 1,
            )
        except TaskPlanError as exc:
            raise ObstacleReplacementCompilationError(
                f"compiled obstacle replacement is invalid: {exc}"
            ) from exc

    @staticmethod
    def _validate_envelope(
        draft: ObstacleRouteRevisionDraft,
        plan: TaskPlan,
        interrupted_index: int,
    ) -> None:
        if draft.mission_id != plan.mission_id:
            raise ObstacleReplacementCompilationError("mission_id mismatch")
        if draft.uav_id != plan.uav_id:
            raise ObstacleReplacementCompilationError("uav_id mismatch")
        if draft.base_plan_version != plan.plan_version:
            raise ObstacleReplacementCompilationError("base_plan_version mismatch")
        if draft.new_plan_version != plan.plan_version + 1:
            raise ObstacleReplacementCompilationError(
                "new_plan_version must advance interrupted plan by exactly one"
            )
        interrupted = plan.steps[interrupted_index]
        if draft.replace_from_step_id != interrupted.step_id:
            raise ObstacleReplacementCompilationError(
                "replace_from_step_id does not identify the interrupted step"
            )
        if interrupted_index == 0:
            raise ObstacleReplacementCompilationError(
                "FOLLOW_ROUTE cannot replace the initial ground step"
            )
        prefix = plan.steps[:interrupted_index]
        if (
            prefix[0].skill is not SkillName.TAKEOFF
            or sum(step.skill is SkillName.TAKEOFF for step in prefix) != 1
            or any(step.skill is SkillName.LAND for step in prefix)
        ):
            raise ObstacleReplacementCompilationError(
                "completed prefix does not establish one airborne mission"
            )
        if plan.steps[-1].skill is not SkillName.LAND:
            raise ObstacleReplacementCompilationError(
                "interrupted plan is not terminated by LAND"
            )

        replacements = draft.replacement_steps
        if (
            replacements[-1].skill != SkillName.LAND.value
            or sum(step.skill == SkillName.LAND.value for step in replacements) != 1
        ):
            raise ObstacleReplacementCompilationError(
                "replacement_steps must terminate with exactly one LAND"
            )
        first = replacements[0]
        if set(first.args) != {"route_ref"}:
            raise ObstacleReplacementCompilationError(
                "first FOLLOW_ROUTE args may contain only route_ref"
            )
        if first.args["route_ref"] != draft.route_draft.route_id:
            raise ObstacleReplacementCompilationError(
                "first FOLLOW_ROUTE route_ref does not match route_draft"
            )

    @staticmethod
    def _skill(step: ObstacleReplacementStep) -> SkillName:
        try:
            return SkillName(step.skill)
        except ValueError as exc:  # The transport type normally catches this.
            raise ObstacleReplacementCompilationError(
                f"unsupported replacement skill: {step.skill}"
            ) from exc

    @staticmethod
    def _next_template_index(
        plan: TaskPlan,
        *,
        template_start: int,
        skill: SkillName,
        used_template_indices: set[int],
    ) -> int:
        for index in range(template_start, len(plan.steps)):
            if (
                index not in used_template_indices
                and plan.steps[index].skill is skill
            ):
                return index
        raise ObstacleReplacementCompilationError(
            f"no remaining trusted {skill.value} step can supply args={{}}"
        )

    @classmethod
    def _match_target_continuation_template(
        cls,
        model_step: ObstacleReplacementStep,
        plan: TaskPlan,
        *,
        interrupted_index: int,
        used_template_indices: set[int],
    ) -> int:
        """Resolve one model intent to an existing trusted SEARCH/TRACK step."""

        action = model_step.target_continuation
        interrupted = plan.steps[interrupted_index]
        if action == "RESTART_SEARCH":
            if interrupted.skill is SkillName.SEARCH:
                source_index = interrupted_index
            elif interrupted.skill is SkillName.TRACK:
                target = interrupted.params.get("target_id")
                if isinstance(target, StepOutputRef):
                    source_index = next(
                        (
                            index
                            for index, candidate in enumerate(
                                plan.steps[:interrupted_index]
                            )
                            if candidate.step_id == target.step_id
                            and candidate.skill is SkillName.SEARCH
                        ),
                        -1,
                    )
                elif target == _LEGACY_SEARCH_TARGET_REF:
                    source_index = next(
                        (
                            index
                            for index in range(interrupted_index - 1, -1, -1)
                            if plan.steps[index].skill is SkillName.SEARCH
                        ),
                        -1,
                    )
                else:
                    source_index = -1
                if source_index < 0:
                    raise ObstacleReplacementCompilationError(
                        "RESTART_SEARCH cannot find the interrupted TRACK's "
                        "trusted source SEARCH"
                    )
            else:
                raise ObstacleReplacementCompilationError(
                    "RESTART_SEARCH requires an interrupted SEARCH or TRACK"
                )
            if source_index in used_template_indices:
                raise ObstacleReplacementCompilationError(
                    "RESTART_SEARCH source template was already reused"
                )
            return source_index

        if action not in {"CONTINUE_TRACK", "REACQUIRE"}:
            raise ObstacleReplacementCompilationError(
                "unsupported target_continuation"
            )
        if (
            model_step.skill != SkillName.TRACK.value
            or interrupted.skill is not SkillName.TRACK
        ):
            raise ObstacleReplacementCompilationError(
                f"{action} requires an interrupted TRACK"
            )
        if interrupted_index in used_template_indices:
            raise ObstacleReplacementCompilationError(
                "interrupted TRACK template was already reused"
            )
        if action == "REACQUIRE":
            recovery = interrupted.recovery
            if (
                recovery is None
                or recovery.skill is not SkillName.REACQUIRE
                or recovery.max_attempts <= 0
            ):
                raise ObstacleReplacementCompilationError(
                    "REACQUIRE requires the interrupted TRACK's trusted "
                    "bounded RecoveryPolicy"
                )
        return interrupted_index

    @staticmethod
    def _compile_explicit_hover(
        model_step: ObstacleReplacementStep,
    ) -> TaskStep:
        if set(model_step.args) != {"duration_s"}:
            raise ObstacleReplacementCompilationError(
                "explicit HOVER args may contain only duration_s"
            )
        duration = _bounded_positive(
            model_step.args["duration_s"],
            "HOVER.duration_s",
            maximum=60.0,
        )
        policy = MotionPolicy(
            max_speed=0.5,
            max_yaw_rate=1.0,
            yaw_mode=YawMode.KEEP_CURRENT,
        )
        policy.validate()
        return TaskStep(
            model_step.step_id,
            SkillName.HOVER,
            {
                "mode": HoverMode.TIMED,
                "duration_s": duration,
                "max_wait_s": duration,
                "position_tolerance_m": 0.25,
                "max_correction_speed_mps": 0.5,
                "reason_code": "PLANNED_HOVER",
                "motion_policy": policy,
            },
        )

    @classmethod
    def _match_explicit_template(
        cls,
        model_step: ObstacleReplacementStep,
        plan: TaskPlan,
        *,
        template_start: int,
        used_template_indices: set[int],
        spatial_resolver: SpatialResolver | None,
    ) -> int:
        skill = cls._skill(model_step)
        if skill not in {SkillName.GOTO, SkillName.LAND}:
            raise ObstacleReplacementCompilationError(
                f"explicit args are unsupported for replacement {skill.value}; "
                "use args={} to reuse trusted compiled parameters"
            )
        if spatial_resolver is None:
            raise ObstacleReplacementCompilationError(
                f"explicit {skill.value} requires a trusted SpatialResolver"
            )
        if skill is SkillName.GOTO:
            if set(model_step.args) != {"target"}:
                raise ObstacleReplacementCompilationError(
                    "explicit GOTO args may contain only target"
                )
            try:
                target = spatial_target_from_dict(model_step.args["target"])
                resolved = spatial_resolver.resolve_target(target)
            except (TypeError, ValueError) as exc:
                raise ObstacleReplacementCompilationError(
                    f"explicit GOTO target is unresolved: {exc}"
                ) from exc
            if isinstance(resolved, RouteTarget) or not isinstance(
                resolved, PointTarget
            ):
                raise ObstacleReplacementCompilationError(
                    "explicit GOTO must resolve to one trusted point"
                )
            for index in range(template_start, len(plan.steps)):
                if index in used_template_indices:
                    continue
                template = plan.steps[index]
                if template.skill is not SkillName.GOTO:
                    continue
                position = template.params.get("position")
                if not _is_xyz(position):
                    continue
                if isinstance(target, NamedLocationTarget):
                    matches = _same_xy(position, resolved.xyz_m)
                else:
                    matches = _same_xyz(position, resolved.xyz_m)
                if matches:
                    return index
            raise ObstacleReplacementCompilationError(
                "explicit GOTO does not select a remaining trusted objective"
            )

        if set(model_step.args) != {"zone"}:
            raise ObstacleReplacementCompilationError(
                "explicit LAND args may contain only zone"
            )
        zone = model_step.args["zone"]
        if not isinstance(zone, str) or not zone.strip():
            raise ObstacleReplacementCompilationError(
                "explicit LAND zone must be a non-empty string"
            )
        try:
            resolved_zone = spatial_resolver.resolve_target(
                NamedLocationTarget(zone.strip())
            )
        except (TypeError, ValueError) as exc:
            raise ObstacleReplacementCompilationError(
                f"explicit LAND zone is unresolved: {exc}"
            ) from exc
        if not isinstance(resolved_zone, PointTarget):  # pragma: no cover
            raise ObstacleReplacementCompilationError(
                "explicit LAND zone did not resolve to a point"
            )
        for index in range(template_start, len(plan.steps)):
            if index in used_template_indices:
                continue
            template = plan.steps[index]
            if template.skill is not SkillName.LAND:
                continue
            expected_xy = template.params.get("expected_position_xy")
            ground = template.params.get("ground_altitude")
            if (
                _is_xy(expected_xy)
                and isinstance(ground, Real)
                and not isinstance(ground, bool)
                and isfinite(float(ground))
                and _same_xy(expected_xy, resolved_zone.xyz_m)
                and abs(float(ground) - resolved_zone.xyz_m[2]) <= 1e-6
            ):
                return index
        raise ObstacleReplacementCompilationError(
            "explicit LAND does not select the remaining trusted landing zone"
        )

    @staticmethod
    def _copy_template(
        model_step: ObstacleReplacementStep,
        template: TaskStep,
        *,
        step_id_map: dict[str, str],
        prior_steps: tuple[TaskStep, ...],
    ) -> TaskStep:
        params = deepcopy(dict(template.params))
        recovery: RecoveryPolicy | None = deepcopy(template.recovery)
        if template.skill is SkillName.TRACK:
            raw_ref = params.get("target_id")
            if isinstance(raw_ref, StepOutputRef):
                mapped_step_id = step_id_map.get(raw_ref.step_id)
                if mapped_step_id is None:
                    raise ObstacleReplacementCompilationError(
                        f"TRACK template {template.step_id} references removed "
                        f"SEARCH step {raw_ref.step_id}"
                    )
                params["target_id"] = StepOutputRef(mapped_step_id, raw_ref.field)
            elif raw_ref == _LEGACY_SEARCH_TARGET_REF:
                if not any(step.skill is SkillName.SEARCH for step in prior_steps):
                    raise ObstacleReplacementCompilationError(
                        f"TRACK template {template.step_id} has no retained prior SEARCH"
                    )
            else:
                raise ObstacleReplacementCompilationError(
                    f"TRACK template {template.step_id} does not use a trusted "
                    "SEARCH StepOutputRef"
                )
        try:
            return TaskStep(
                model_step.step_id,
                template.skill,
                params,
                recovery,
            )
        except TaskPlanError as exc:
            raise ObstacleReplacementCompilationError(
                f"could not reuse trusted step {template.step_id}: {exc}"
            ) from exc

    @staticmethod
    def _validate_terminal_rejoin(
        plan: TaskPlan,
        *,
        interrupted_index: int,
        matched_original_indices: tuple[int, ...],
    ) -> None:
        final_index = len(plan.steps) - 1
        if not matched_original_indices or matched_original_indices[-1] != final_index:
            raise ObstacleReplacementCompilationError(
                "replacement LAND must reuse the original terminal LAND"
            )
        terminal_goto = next(
            (
                index
                for index in range(final_index - 1, -1, -1)
                if plan.steps[index].skill is SkillName.GOTO
            ),
            None,
        )
        if terminal_goto is None:
            raise ObstacleReplacementCompilationError(
                "original terminal LAND has no trusted GOTO approach"
            )
        if terminal_goto == interrupted_index:
            # The accepted FOLLOW_ROUTE is the trusted replacement for this
            # GOTO and its critic has already checked rejoin geometry.
            required = tuple(range(terminal_goto + 1, final_index + 1))
        else:
            required = tuple(range(terminal_goto, final_index + 1))
        retained_terminal = tuple(
            index
            for index in matched_original_indices
            if index >= required[0]
        )
        if retained_terminal != required:
            raise ObstacleReplacementCompilationError(
                "replacement omits or reorders the trusted terminal approach"
            )
        first_terminal_position = matched_original_indices.index(required[0])
        if matched_original_indices[first_terminal_position:] != required:
            raise ObstacleReplacementCompilationError(
                "replacement places mission work after the trusted terminal approach"
            )

    @staticmethod
    def _validate_full_ids(steps: tuple[TaskStep, ...]) -> None:
        ids = tuple(step.step_id for step in steps)
        if len(ids) != len(set(ids)):
            raise ObstacleReplacementCompilationError(
                "replacement step ID duplicates the completed prefix"
            )

    @staticmethod
    def _validate_replacement_track_refs(
        steps: tuple[TaskStep, ...],
        *,
        replacement_start: int,
    ) -> None:
        for index, step in enumerate(steps[replacement_start:], replacement_start):
            if step.skill is not SkillName.TRACK:
                continue
            target = step.params.get("target_id")
            if target == _LEGACY_SEARCH_TARGET_REF:
                if not any(
                    candidate.skill is SkillName.SEARCH
                    for candidate in steps[:index]
                ):
                    raise ObstacleReplacementCompilationError(
                        f"TRACK step {step.step_id} has no prior SEARCH"
                    )
                continue
            if not isinstance(target, StepOutputRef):
                raise ObstacleReplacementCompilationError(
                    f"TRACK step {step.step_id} target_id is not a StepOutputRef"
                )
            source_index = next(
                (
                    candidate_index
                    for candidate_index, candidate in enumerate(steps[:index])
                    if candidate.step_id == target.step_id
                ),
                None,
            )
            if source_index is None:
                raise ObstacleReplacementCompilationError(
                    f"TRACK step {step.step_id} references a missing prior step"
                )
            if steps[source_index].skill is not SkillName.SEARCH:
                raise ObstacleReplacementCompilationError(
                    f"TRACK step {step.step_id} reference does not point to SEARCH"
                )


def compile_obstacle_replacement(
    draft: ObstacleRouteRevisionDraft,
    interrupted_plan: TaskPlan,
    interrupted_index: int,
    *,
    defaults: TrustedFollowRouteDefaults | None = None,
    spatial_resolver: SpatialResolver | None = None,
) -> TaskPlan:
    """Functional convenience wrapper around :class:`ObstacleReplacementCompiler`."""

    return ObstacleReplacementCompiler(defaults).compile(
        draft,
        interrupted_plan,
        interrupted_index,
        spatial_resolver=spatial_resolver,
    )


def _is_xyz(value: object) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 3
        and all(
            isinstance(item, Real)
            and not isinstance(item, bool)
            and isfinite(float(item))
            for item in value
        )
    )


def _is_xy(value: object) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(
            isinstance(item, Real)
            and not isinstance(item, bool)
            and isfinite(float(item))
            for item in value
        )
    )


def _same_xy(left: object, right: object, *, tolerance: float = 1e-6) -> bool:
    if not isinstance(left, (tuple, list)) or not isinstance(
        right, (tuple, list)
    ):
        return False
    if len(left) < 2 or len(right) < 2:
        return False
    try:
        return all(
            abs(float(left[index]) - float(right[index])) <= tolerance
            for index in range(2)
        )
    except (TypeError, ValueError):
        return False


def _same_xyz(left: object, right: object, *, tolerance: float = 1e-6) -> bool:
    if not _is_xyz(left) or not _is_xyz(right):
        return False
    return all(
        abs(float(left[index]) - float(right[index])) <= tolerance
        for index in range(3)  # type: ignore[index]
    )


__all__ = [
    "ObstacleReplacementCompilationError",
    "ObstacleReplacementCompiler",
    "TrustedFollowRouteDefaults",
    "compile_obstacle_replacement",
]
