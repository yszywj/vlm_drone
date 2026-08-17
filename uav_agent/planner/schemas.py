"""Pure-Python schemas at the boundary between planning and Skill execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
import re
from types import MappingProxyType

from planner.text_safety import reject_forbidden_planner_text
from skills.plan import TaskPlan


_STEP_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_TARGET_REF_PATTERN = re.compile(
    r"^\$(?P<step_id>[a-z][a-z0-9_]{0,31})\.target_id$"
)
_TOP_LEVEL_SKILLS = frozenset({"TAKEOFF", "GOTO", "SEARCH", "TRACK", "LAND"})
_YAW_MODES = {
    "TAKEOFF": frozenset({"KEEP_CURRENT", "FIXED"}),
    "GOTO": frozenset(
        {"KEEP_CURRENT", "COURSE_ALIGNED", "FACE_POINT", "FIXED"}
    ),
    "LAND": frozenset({"KEEP_CURRENT", "FIXED"}),
}


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


def _positive_number(value: object, field_name: str) -> float:
    normalized = _finite_number(value, field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be greater than zero")
    return normalized


def _finite_vector(
    value: object,
    length: int,
    field_name: str,
) -> tuple[float, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != length
    ):
        raise ValueError(f"{field_name} must contain exactly {length} numbers")
    return tuple(
        _finite_number(component, f"{field_name}[{index}]")
        for index, component in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class SearchRegionSpec:
    """Named search geometry supplied by trusted world configuration."""

    name: str
    center_xyz_m: tuple[float, float, float]
    radius_m: float
    approach_xyz_m: tuple[float, float, float]
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(
            self,
            "center_xyz_m",
            _finite_vector(self.center_xyz_m, 3, "center_xyz_m"),
        )
        object.__setattr__(self, "radius_m", _positive_number(self.radius_m, "radius_m"))
        object.__setattr__(
            self,
            "approach_xyz_m",
            _finite_vector(self.approach_xyz_m, 3, "approach_xyz_m"),
        )
        object.__setattr__(
            self,
            "description",
            _string(self.description, "description"),
        )


@dataclass(frozen=True, slots=True)
class LandingZoneSpec:
    """Named landing geometry supplied by trusted world configuration."""

    name: str
    position_xy_m: tuple[float, float]
    ground_altitude_m: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(
            self,
            "position_xy_m",
            _finite_vector(self.position_xy_m, 2, "position_xy_m"),
        )
        object.__setattr__(
            self,
            "ground_altitude_m",
            _finite_number(self.ground_altitude_m, "ground_altitude_m"),
        )
        object.__setattr__(
            self,
            "description",
            _string(self.description, "description"),
        )


@dataclass(frozen=True, slots=True)
class NavigationPointSpec:
    """Trusted geometry for an optional named navigation point.

    Only ``name`` and ``description`` are exposed to a model.  The coordinate
    remains behind the dynamic plan compiler.
    """

    name: str
    position_xyz_m: tuple[float, float, float]
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(
            self,
            "position_xyz_m",
            _finite_vector(self.position_xyz_m, 3, "position_xyz_m"),
        )
        object.__setattr__(
            self,
            "description",
            _string(self.description, "description"),
        )


@dataclass(frozen=True, slots=True)
class PlannerWorldContext:
    """Trusted, non-oracle world facts available to a mission planner."""

    scene_min_xyz_m: tuple[float, float, float]
    scene_max_xyz_m: tuple[float, float, float]
    initial_uav_xyz_m: tuple[float, float, float]
    search_regions: Mapping[str, SearchRegionSpec]
    landing_zones: Mapping[str, LandingZoneSpec]
    default_takeoff_altitude_m: float
    default_track_duration_s: float
    search_timeout_s: float
    goto_timeout_s: float = 120.0
    land_timeout_s: float = 60.0
    navigation_points: Mapping[str, NavigationPointSpec] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        scene_min = _finite_vector(self.scene_min_xyz_m, 3, "scene_min_xyz_m")
        scene_max = _finite_vector(self.scene_max_xyz_m, 3, "scene_max_xyz_m")
        if any(lower >= upper for lower, upper in zip(scene_min, scene_max)):
            raise ValueError(
                "scene_min_xyz_m must be strictly less than scene_max_xyz_m "
                "on every axis"
            )

        object.__setattr__(self, "scene_min_xyz_m", scene_min)
        object.__setattr__(self, "scene_max_xyz_m", scene_max)
        object.__setattr__(
            self,
            "initial_uav_xyz_m",
            _finite_vector(self.initial_uav_xyz_m, 3, "initial_uav_xyz_m"),
        )
        object.__setattr__(
            self,
            "search_regions",
            _readonly_spec_mapping(
                self.search_regions,
                SearchRegionSpec,
                "search_regions",
            ),
        )
        object.__setattr__(
            self,
            "landing_zones",
            _readonly_spec_mapping(
                self.landing_zones,
                LandingZoneSpec,
                "landing_zones",
            ),
        )
        object.__setattr__(
            self,
            "navigation_points",
            _readonly_spec_mapping(
                self.navigation_points,
                NavigationPointSpec,
                "navigation_points",
            ),
        )
        object.__setattr__(
            self,
            "default_takeoff_altitude_m",
            _positive_number(
                self.default_takeoff_altitude_m,
                "default_takeoff_altitude_m",
            ),
        )
        object.__setattr__(
            self,
            "default_track_duration_s",
            _positive_number(
                self.default_track_duration_s,
                "default_track_duration_s",
            ),
        )
        object.__setattr__(
            self,
            "search_timeout_s",
            _positive_number(self.search_timeout_s, "search_timeout_s"),
        )
        object.__setattr__(
            self,
            "goto_timeout_s",
            _positive_number(self.goto_timeout_s, "goto_timeout_s"),
        )
        object.__setattr__(
            self,
            "land_timeout_s",
            _positive_number(self.land_timeout_s, "land_timeout_s"),
        )


@dataclass(frozen=True, slots=True)
class MissionIntent:
    """Model-facing task intent without low-level coordinates or timeouts."""

    target_description: str
    search_region: str
    track_duration_s: float
    landing_zone: str
    takeoff_altitude_m: float | None = None

    _REQUIRED_FIELDS = frozenset(
        {
            "target_description",
            "search_region",
            "track_duration_s",
            "landing_zone",
        }
    )
    _OPTIONAL_FIELDS = frozenset({"takeoff_altitude_m"})

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_description",
            _non_empty_string(self.target_description, "target_description"),
        )
        object.__setattr__(
            self,
            "search_region",
            _non_empty_string(self.search_region, "search_region"),
        )
        object.__setattr__(
            self,
            "track_duration_s",
            _positive_number(self.track_duration_s, "track_duration_s"),
        )
        object.__setattr__(
            self,
            "landing_zone",
            _non_empty_string(self.landing_zone, "landing_zone"),
        )
        if self.takeoff_altitude_m is not None:
            object.__setattr__(
                self,
                "takeoff_altitude_m",
                _finite_number(self.takeoff_altitude_m, "takeoff_altitude_m"),
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MissionIntent:
        """Parse an exact, JSON-like planner response without ignoring fields."""

        if not isinstance(data, Mapping):
            raise TypeError("MissionIntent input must be a mapping")
        if any(not isinstance(key, str) for key in data):
            raise TypeError("MissionIntent field names must be strings")

        keys = frozenset(data)
        allowed = cls._REQUIRED_FIELDS | cls._OPTIONAL_FIELDS
        unknown = keys - allowed
        if unknown:
            raise ValueError(
                "MissionIntent contains unknown fields: " + ", ".join(sorted(unknown))
            )
        missing = cls._REQUIRED_FIELDS - keys
        if missing:
            raise ValueError(
                "MissionIntent is missing required fields: "
                + ", ".join(sorted(missing))
            )

        return cls(
            target_description=data["target_description"],
            search_region=data["search_region"],
            track_duration_s=data["track_duration_s"],
            landing_zone=data["landing_zone"],
            takeoff_altitude_m=data.get("takeoff_altitude_m"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible mapping."""

        return {
            "target_description": self.target_description,
            "search_region": self.search_region,
            "track_duration_s": self.track_duration_s,
            "landing_zone": self.landing_zone,
            "takeoff_altitude_m": self.takeoff_altitude_m,
        }


def _exact_mapping_fields(
    data: object,
    *,
    type_name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{type_name} input must be a mapping")
    if any(not isinstance(key, str) for key in data):
        raise TypeError(f"{type_name} field names must be strings")
    keys = frozenset(data)
    unknown = keys - required - optional
    if unknown:
        raise ValueError(
            f"{type_name} contains unknown fields: " + ", ".join(sorted(unknown))
        )
    missing = required - keys
    if missing:
        raise ValueError(
            f"{type_name} is missing required fields: "
            + ", ".join(sorted(missing))
        )
    return data


def _bounded_number(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    normalized = _finite_number(value, field_name)
    if strictly_positive and normalized <= 0.0:
        raise ValueError(f"{field_name} must be greater than zero")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field_name} must be at least {minimum:g}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{field_name} must be at most {maximum:g}")
    return normalized


def _bounded_integer(
    value: object,
    field_name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{field_name} must be between {minimum} and {maximum}"
        )
    return value


@dataclass(frozen=True, slots=True)
class RecoveryDraft:
    """Model-selected, bounded TRACK recovery policy."""

    skill: str
    max_attempts: int
    search_radius_m: float | None = None
    timeout_s: float | None = None

    _REQUIRED_FIELDS = frozenset({"skill", "max_attempts"})
    _OPTIONAL_FIELDS = frozenset({"search_radius_m", "timeout_s"})

    def __post_init__(self) -> None:
        skill = _non_empty_string(self.skill, "recovery.skill")
        if skill != "REACQUIRE":
            raise ValueError("recovery.skill must be REACQUIRE")
        object.__setattr__(self, "skill", skill)
        object.__setattr__(
            self,
            "max_attempts",
            _bounded_integer(
                self.max_attempts,
                "recovery.max_attempts",
                minimum=0,
                maximum=2,
            ),
        )
        if self.search_radius_m is not None:
            object.__setattr__(
                self,
                "search_radius_m",
                _bounded_number(
                    self.search_radius_m,
                    "recovery.search_radius_m",
                    minimum=3.0,
                    maximum=20.0,
                ),
            )
        if self.timeout_s is not None:
            object.__setattr__(
                self,
                "timeout_s",
                _bounded_number(
                    self.timeout_s,
                    "recovery.timeout_s",
                    minimum=5.0,
                    maximum=60.0,
                ),
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RecoveryDraft:
        parsed = _exact_mapping_fields(
            data,
            type_name="RecoveryDraft",
            required=cls._REQUIRED_FIELDS,
            optional=cls._OPTIONAL_FIELDS,
        )
        return cls(
            skill=parsed["skill"],
            max_attempts=parsed["max_attempts"],
            search_radius_m=parsed.get("search_radius_m"),
            timeout_s=parsed.get("timeout_s"),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "skill": self.skill,
            "max_attempts": self.max_attempts,
        }
        if self.search_radius_m is not None:
            result["search_radius_m"] = self.search_radius_m
        if self.timeout_s is not None:
            result["timeout_s"] = self.timeout_s
        return result


_STEP_ARGUMENT_FIELDS: dict[
    str,
    tuple[frozenset[str], frozenset[str]],
] = {
    "TAKEOFF": (
        frozenset(),
        frozenset({"altitude_m", "yaw_mode", "yaw_deg"}),
    ),
    "GOTO": (
        frozenset({"destination"}),
        frozenset({"altitude_m", "yaw_mode", "yaw_deg"}),
    ),
    "SEARCH": (
        frozenset({"region", "target_description"}),
        frozenset({"altitude_m"}),
    ),
    "TRACK": (
        frozenset({"target_ref", "duration_s"}),
        frozenset({"desired_altitude_m", "desired_distance_m"}),
    ),
    "LAND": (
        frozenset({"zone"}),
        frozenset({"yaw_mode", "yaw_deg"}),
    ),
}


def _validated_step_args(skill: str, value: object) -> Mapping[str, object]:
    required, optional = _STEP_ARGUMENT_FIELDS[skill]
    data = _exact_mapping_fields(
        value,
        type_name=f"{skill} args",
        required=required,
        optional=optional,
    )
    normalized: dict[str, object] = {}

    string_fields = {
        "destination",
        "region",
        "target_description",
        "target_ref",
        "zone",
    }
    finite_number_fields = {
        "altitude_m",
        "duration_s",
        "desired_altitude_m",
        "desired_distance_m",
    }
    for key, raw in data.items():
        if key in string_fields:
            text = _non_empty_string(raw, f"{skill}.args.{key}")
            if key == "target_description":
                if len(text) > 256:
                    raise ValueError(
                        "SEARCH.args.target_description must contain at most "
                        "256 characters"
                    )
                reject_forbidden_planner_text(
                    text,
                    "SEARCH.args.target_description",
                )
            normalized[key] = text
        elif key in finite_number_fields:
            # World-dependent ranges and mission-policy bounds belong to the
            # trusted PlanValidator.  This model-output boundary only enforces
            # JSON type/finite-number structure.
            normalized[key] = _finite_number(raw, f"{skill}.args.{key}")
        elif key == "yaw_deg":
            normalized[key] = _bounded_number(
                raw,
                f"{skill}.args.yaw_deg",
                minimum=-360.0,
                maximum=360.0,
            )
        elif key == "yaw_mode":
            yaw_mode = _non_empty_string(raw, f"{skill}.args.yaw_mode")
            if yaw_mode not in _YAW_MODES[skill]:
                allowed = ", ".join(sorted(_YAW_MODES[skill]))
                raise ValueError(
                    f"{skill}.args.yaw_mode must be one of: {allowed}"
                )
            normalized[key] = yaw_mode
        else:  # pragma: no cover - exact allow-list above makes this unreachable
            raise AssertionError(f"unhandled {skill} argument: {key}")

    if "yaw_deg" in normalized and normalized.get("yaw_mode") != "FIXED":
        raise ValueError(f"{skill}.args.yaw_deg is only allowed for FIXED yaw")
    if normalized.get("yaw_mode") == "FIXED" and "yaw_deg" not in normalized:
        raise ValueError(f"{skill}.args.yaw_deg is required for FIXED yaw")
    if skill == "TRACK":
        reference = str(normalized["target_ref"])
        if _TARGET_REF_PATTERN.fullmatch(reference) is None:
            raise ValueError(
                "TRACK.args.target_ref must use $<step_id>.target_id"
            )
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class PlanStepDraft:
    """One strictly allow-listed step selected by a dynamic planner."""

    id: str
    skill: str
    args: Mapping[str, object]
    recovery: RecoveryDraft | None = None

    _REQUIRED_FIELDS = frozenset({"id", "skill", "args"})
    _OPTIONAL_FIELDS = frozenset({"recovery"})

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError("step.id must be a string")
        step_id = self.id
        if _STEP_ID_PATTERN.fullmatch(step_id) is None:
            raise ValueError("step.id must match ^[a-z][a-z0-9_]{0,31}$")
        object.__setattr__(self, "id", step_id)

        skill = _non_empty_string(self.skill, "step.skill")
        if skill == "REACQUIRE":
            raise ValueError("REACQUIRE is recovery-only and cannot be top-level")
        if skill not in _TOP_LEVEL_SKILLS:
            raise ValueError(f"unknown top-level Skill: {skill}")
        object.__setattr__(self, "skill", skill)
        object.__setattr__(self, "args", _validated_step_args(skill, self.args))

        if self.recovery is not None:
            if not isinstance(self.recovery, RecoveryDraft):
                raise TypeError("step.recovery must be a RecoveryDraft or None")
            if skill != "TRACK":
                raise ValueError("recovery is only allowed on TRACK steps")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PlanStepDraft:
        parsed = _exact_mapping_fields(
            data,
            type_name="PlanStepDraft",
            required=cls._REQUIRED_FIELDS,
            optional=cls._OPTIONAL_FIELDS,
        )
        recovery = (
            None
            if "recovery" not in parsed
            else RecoveryDraft.from_dict(parsed["recovery"])
        )
        return cls(
            id=parsed["id"],
            skill=parsed["skill"],
            args=parsed["args"],
            recovery=recovery,
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.id,
            "skill": self.skill,
            "args": dict(self.args),
        }
        if self.recovery is not None:
            result["recovery"] = self.recovery.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class SkillPlanDraft:
    """Finite, linear, high-level plan produced by a dynamic planner."""

    schema_version: int
    steps: tuple[PlanStepDraft, ...]

    _REQUIRED_FIELDS = frozenset({"schema_version", "steps"})

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise TypeError("schema_version must be the integer 1")
        if self.schema_version != 1:
            raise ValueError("schema_version must equal 1")
        if isinstance(self.steps, (str, bytes)) or not isinstance(
            self.steps, Sequence
        ):
            raise TypeError("steps must be an array of PlanStepDraft values")
        steps = tuple(self.steps)
        if not 2 <= len(steps) <= 10:
            raise ValueError("steps must contain between 2 and 10 entries")
        if any(not isinstance(step, PlanStepDraft) for step in steps):
            raise TypeError("steps must contain only PlanStepDraft values")
        step_ids = tuple(step.id for step in steps)
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("step ids must be unique")

        # References are plan structure rather than WorldContext semantics, so
        # reject forward and non-SEARCH references at this boundary as well as
        # in the trusted compiler's defense-in-depth checks.
        previous: dict[str, str] = {}
        for step in steps:
            if step.skill == "TRACK":
                match = _TARGET_REF_PATTERN.fullmatch(
                    str(step.args["target_ref"])
                )
                assert match is not None  # validated by PlanStepDraft
                source_id = match.group("step_id")
                if source_id not in previous:
                    raise ValueError("TRACK target_ref must point to a prior step")
                if previous[source_id] != "SEARCH":
                    raise ValueError("TRACK target_ref must point to a SEARCH step")
            previous[step.id] = step.skill

        object.__setattr__(self, "steps", steps)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SkillPlanDraft:
        parsed = _exact_mapping_fields(
            data,
            type_name="SkillPlanDraft",
            required=cls._REQUIRED_FIELDS,
        )
        raw_steps = parsed["steps"]
        if isinstance(raw_steps, (str, bytes)) or not isinstance(
            raw_steps, Sequence
        ):
            raise TypeError("SkillPlanDraft.steps must be an array")
        return cls(
            schema_version=parsed["schema_version"],
            steps=tuple(PlanStepDraft.from_dict(step) for step in raw_steps),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "steps": [step.to_dict() for step in self.steps],
        }


PlannerOutput = MissionIntent | SkillPlanDraft


@dataclass(frozen=True, slots=True)
class PlannerRequest:
    instruction: str
    world_context: PlannerWorldContext

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instruction",
            _non_empty_string(self.instruction, "instruction"),
        )
        if not isinstance(self.world_context, PlannerWorldContext):
            raise TypeError("world_context must be a PlannerWorldContext")


@dataclass(frozen=True, slots=True, init=False)
class CompiledMission:
    """Validated planner output paired with its executable Skill plan.

    ``CompiledMission(intent, task_plan, source)`` and the legacy ``intent=``
    keyword remain accepted.  New code should use ``planner_output=``.
    """

    planner_output: PlannerOutput
    task_plan: TaskPlan
    source: str
    compiler_notes: tuple[str, ...] = ()

    def __init__(
        self,
        planner_output: PlannerOutput | None = None,
        task_plan: TaskPlan | None = None,
        source: str | None = None,
        compiler_notes: Sequence[str] = (),
        *,
        intent: MissionIntent | None = None,
    ) -> None:
        if planner_output is not None and intent is not None:
            raise TypeError("provide planner_output or intent, not both")
        output = intent if planner_output is None else planner_output
        if not isinstance(output, (MissionIntent, SkillPlanDraft)):
            raise TypeError(
                "planner_output must be a MissionIntent or SkillPlanDraft"
            )
        if not isinstance(task_plan, TaskPlan):
            raise TypeError("task_plan must be a skills.plan.TaskPlan")
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        if source not in {"scripted", "llm", "dynamic_scripted", "dynamic_llm"}:
            raise ValueError(
                "source must be scripted, llm, dynamic_scripted, or dynamic_llm"
            )
        if isinstance(compiler_notes, (str, bytes)) or not isinstance(
            compiler_notes, Sequence
        ):
            raise TypeError("compiler_notes must be a sequence of strings")
        notes = tuple(
            _string(note, f"compiler_notes[{index}]")
            for index, note in enumerate(compiler_notes)
        )
        object.__setattr__(self, "planner_output", output)
        object.__setattr__(self, "task_plan", task_plan)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "compiler_notes", notes)

    @property
    def intent(self) -> MissionIntent | None:
        if isinstance(self.planner_output, MissionIntent):
            return self.planner_output
        return None

    @property
    def skill_plan_draft(self) -> SkillPlanDraft | None:
        if isinstance(self.planner_output, SkillPlanDraft):
            return self.planner_output
        return None


def _readonly_spec_mapping(
    value: object,
    expected_type: (
        type[SearchRegionSpec]
        | type[LandingZoneSpec]
        | type[NavigationPointSpec]
    ),
    field_name: str,
) -> (
    Mapping[str, SearchRegionSpec]
    | Mapping[str, LandingZoneSpec]
    | Mapping[str, NavigationPointSpec]
):
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    snapshot: dict[
        str,
        SearchRegionSpec | LandingZoneSpec | NavigationPointSpec,
    ] = {}
    for key, spec in value.items():
        normalized_key = _non_empty_string(key, f"{field_name} key")
        if not isinstance(spec, expected_type):
            raise TypeError(
                f"{field_name}[{normalized_key!r}] must be a "
                f"{expected_type.__name__}"
            )
        snapshot[normalized_key] = spec
    return MappingProxyType(snapshot)
