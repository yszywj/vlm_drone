"""Strict routed model-output values for Spatial Contract V3.

V3 is a parallel contract, not a collection of optional fields added to V2.
The initial runtime default therefore remains :class:`SkillPlanDraftV2`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from types import MappingProxyType

from common.ids import validate_mission_id, validate_uav_id
from planner.schemas import PlanStepDraftV2, RecoveryDraft
from planner.spatial import (
    RegionSpec,
    SpatialAssumption,
    SpatialTarget,
    region_spec_from_dict,
    spatial_target_from_dict,
)
from skills.search_strategy import SearchEntryPolicy, SearchStrategySpec
from target.types import TargetSpec


class SpatialPlanValidationError(ValueError):
    """Raised at the exact V3 parsing boundary."""


def _exact(
    value: object,
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} field names must be strings")
    keys = frozenset(value)
    unknown, missing = keys - required - optional, required - keys
    if unknown:
        raise SpatialPlanValidationError(
            f"{name} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise SpatialPlanValidationError(
            f"{name} is missing fields: {', '.join(sorted(missing))}"
        )
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise SpatialPlanValidationError(f"{name} must be a finite number")
    if positive and result <= 0.0:
        raise SpatialPlanValidationError(f"{name} must be greater than zero")
    return result


def _point(value: object, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise SpatialPlanValidationError(f"{name} must contain exactly three numbers")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _text(value: object, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result or len(result) > maximum:
        raise SpatialPlanValidationError(
            f"{name} must contain between 1 and {maximum} characters"
        )
    return result


def _yaw_args(data: Mapping[str, object], *, skill: str) -> dict[str, object]:
    result: dict[str, object] = {}
    if "altitude_m" in data:
        result["altitude_m"] = _finite(data["altitude_m"], f"{skill}.args.altitude_m", positive=True)
    if "yaw_mode" in data:
        value = _text(data["yaw_mode"], f"{skill}.args.yaw_mode", maximum=32)
        if value not in {"KEEP_CURRENT", "COURSE_ALIGNED", "FACE_POINT", "FIXED"}:
            raise SpatialPlanValidationError(f"unsupported {skill} yaw_mode")
        result["yaw_mode"] = value
    if "yaw_deg" in data:
        result["yaw_deg"] = _finite(data["yaw_deg"], f"{skill}.args.yaw_deg")
    if "yaw_deg" in result and result.get("yaw_mode") != "FIXED":
        raise SpatialPlanValidationError(f"{skill}.args.yaw_deg requires FIXED yaw_mode")
    if result.get("yaw_mode") == "FIXED" and "yaw_deg" not in result:
        raise SpatialPlanValidationError(f"{skill}.args FIXED yaw_mode requires yaw_deg")
    return result


def _goto_args(value: object) -> Mapping[str, object]:
    data = _exact(
        value,
        name="GOTO V3 args",
        required=frozenset({"target"}),
        optional=frozenset({"altitude_m", "yaw_mode", "yaw_deg"}),
    )
    return MappingProxyType({"target": spatial_target_from_dict(data["target"]), **_yaw_args(data, skill="GOTO")})


def _search_args(value: object) -> Mapping[str, object]:
    data = _exact(
        value,
        name="SEARCH V3 args",
        required=frozenset(
            {
                "region", "strategy", "entry_policy", "target_description",
                "search_altitude_m", "timeout_s",
            }
        ),
        optional=frozenset(
            {
                "transit_speed_mps", "scan_yaw_rate_rad_s",
                "user_anchor_xyz_m", "model_selected_entry_xyz_m",
            }
        ),
    )
    try:
        policy = SearchEntryPolicy(data["entry_policy"])
    except (TypeError, ValueError) as exc:
        raise SpatialPlanValidationError("unsupported SEARCH entry_policy") from exc
    if policy is SearchEntryPolicy.USER_ANCHOR and "user_anchor_xyz_m" not in data:
        raise SpatialPlanValidationError(
            "USER_ANCHOR entry_policy requires user_anchor_xyz_m"
        )
    if policy is SearchEntryPolicy.MODEL_SELECTED and "model_selected_entry_xyz_m" not in data:
        raise SpatialPlanValidationError(
            "MODEL_SELECTED entry_policy requires model_selected_entry_xyz_m"
        )
    if "user_anchor_xyz_m" in data and policy is not SearchEntryPolicy.USER_ANCHOR:
        raise SpatialPlanValidationError(
            "user_anchor_xyz_m is only allowed for USER_ANCHOR"
        )
    if (
        "model_selected_entry_xyz_m" in data
        and policy is not SearchEntryPolicy.MODEL_SELECTED
    ):
        raise SpatialPlanValidationError(
            "model_selected_entry_xyz_m is only allowed for MODEL_SELECTED"
        )
    result: dict[str, object] = {
        "region": region_spec_from_dict(data["region"]),
        "strategy": SearchStrategySpec.from_dict(data["strategy"]),
        "entry_policy": policy,
        "target_description": _text(data["target_description"], "SEARCH.args.target_description"),
        "search_altitude_m": _finite(data["search_altitude_m"], "SEARCH.args.search_altitude_m", positive=True),
        "timeout_s": _finite(data["timeout_s"], "SEARCH.args.timeout_s", positive=True),
    }
    if "transit_speed_mps" in data:
        result["transit_speed_mps"] = _finite(data["transit_speed_mps"], "SEARCH.args.transit_speed_mps", positive=True)
    if "scan_yaw_rate_rad_s" in data:
        result["scan_yaw_rate_rad_s"] = _finite(data["scan_yaw_rate_rad_s"], "SEARCH.args.scan_yaw_rate_rad_s", positive=True)
    for name in ("user_anchor_xyz_m", "model_selected_entry_xyz_m"):
        if name in data:
            result[name] = _point(data[name], f"SEARCH.args.{name}")
    return MappingProxyType(result)


def _jsonify(value: object) -> object:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, SearchEntryPolicy):
        return value.value
    return value


@dataclass(frozen=True, slots=True)
class PlanStepDraftV3:
    id: str
    uav_id: str
    skill: str
    args: Mapping[str, object]
    recovery: RecoveryDraft | None = None

    def __post_init__(self) -> None:
        if self.skill in {"GOTO", "SEARCH"}:
            # Reuse the routed V2 step for stable id/uav/skill/recovery checks,
            # while V3 owns the independent spatial argument parser.
            placeholder = (
                {"destination": "v3_spatial_placeholder"}
                if self.skill == "GOTO"
                else {"region": "v3_spatial_placeholder", "target_description": "placeholder"}
            )
            identity = PlanStepDraftV2(
                id=self.id,
                uav_id=self.uav_id,
                skill=self.skill,
                args=placeholder,
                recovery=self.recovery,
            )
            if self.recovery is not None:
                raise SpatialPlanValidationError("recovery is only allowed on TRACK steps")
            parsed_args = _goto_args(self.args) if self.skill == "GOTO" else _search_args(self.args)
            object.__setattr__(self, "id", identity.id)
            object.__setattr__(self, "uav_id", identity.uav_id)
            object.__setattr__(self, "skill", identity.skill)
            object.__setattr__(self, "args", parsed_args)
            return
        legacy = PlanStepDraftV2(
            id=self.id,
            uav_id=self.uav_id,
            skill=self.skill,
            args=self.args,
            recovery=self.recovery,
        )
        object.__setattr__(self, "id", legacy.id)
        object.__setattr__(self, "uav_id", legacy.uav_id)
        object.__setattr__(self, "skill", legacy.skill)
        object.__setattr__(self, "args", legacy.args)
        object.__setattr__(self, "recovery", legacy.recovery)

    @classmethod
    def from_dict(cls, value: object) -> PlanStepDraftV3:
        data = _exact(
            value,
            name="PlanStepDraftV3",
            required=frozenset({"id", "uav_id", "skill", "args"}),
            optional=frozenset({"recovery"}),
        )
        return cls(
            id=data["id"],  # type: ignore[arg-type]
            uav_id=data["uav_id"],  # type: ignore[arg-type]
            skill=data["skill"],  # type: ignore[arg-type]
            args=data["args"],  # type: ignore[arg-type]
            recovery=None if "recovery" not in data else RecoveryDraft.from_dict(data["recovery"]),  # type: ignore[arg-type]
        )

    @property
    def spatial_target(self) -> SpatialTarget | None:
        value = self.args.get("target")
        return value if self.skill == "GOTO" else None  # type: ignore[return-value]

    @property
    def region(self) -> RegionSpec | None:
        value = self.args.get("region")
        return value if self.skill == "SEARCH" else None  # type: ignore[return-value]

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.id,
            "uav_id": self.uav_id,
            "skill": self.skill,
            "args": {key: _jsonify(value) for key, value in self.args.items()},
        }
        if self.recovery is not None:
            result["recovery"] = self.recovery.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class SkillPlanDraftV3:
    schema_version: int
    mission_id: str
    uav_id: str
    plan_version: int
    assumptions: tuple[SpatialAssumption, ...]
    steps: tuple[PlanStepDraftV3, ...]
    target_spec: TargetSpec | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != 3
        ):
            raise SpatialPlanValidationError("schema_version must equal integer 3")
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        if isinstance(self.plan_version, bool) or not isinstance(self.plan_version, int) or self.plan_version <= 0:
            raise SpatialPlanValidationError("plan_version must be a positive integer")
        if isinstance(self.assumptions, (str, bytes)) or not isinstance(self.assumptions, Sequence):
            raise TypeError("assumptions must be an array")
        assumptions = tuple(self.assumptions)
        if len(assumptions) > 32 or any(not isinstance(item, SpatialAssumption) for item in assumptions):
            raise SpatialPlanValidationError("assumptions must contain at most 32 SpatialAssumption values")
        object.__setattr__(self, "assumptions", assumptions)
        if isinstance(self.steps, (str, bytes)) or not isinstance(self.steps, Sequence):
            raise TypeError("steps must be an array")
        steps = tuple(self.steps)
        if not 2 <= len(steps) <= 10 or any(not isinstance(step, PlanStepDraftV3) for step in steps):
            raise SpatialPlanValidationError("steps must contain 2 to 10 PlanStepDraftV3 values")
        if any(step.uav_id != self.uav_id for step in steps):
            raise SpatialPlanValidationError("every step.uav_id must equal top-level uav_id")
        if len({step.id for step in steps}) != len(steps):
            raise SpatialPlanValidationError("step ids must be unique")
        object.__setattr__(self, "steps", steps)
        if self.target_spec is not None and not isinstance(self.target_spec, TargetSpec):
            raise TypeError("target_spec must be a TargetSpec or None")
        if (
            self.target_spec is not None
            and self.target_spec.mutable_appearance_notes
        ):
            raise SpatialPlanValidationError(
                "initial target_spec.mutable_appearance_notes must be empty"
            )

    @classmethod
    def from_dict(cls, value: object) -> SkillPlanDraftV3:
        data = _exact(
            value,
            name="SkillPlanDraftV3",
            required=frozenset({"schema_version", "mission_id", "uav_id", "plan_version", "assumptions", "steps"}),
            optional=frozenset({"target_spec"}),
        )
        assumptions = data["assumptions"]
        steps = data["steps"]
        if isinstance(assumptions, (str, bytes)) or not isinstance(assumptions, Sequence):
            raise TypeError("assumptions must be an array")
        if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
            raise TypeError("steps must be an array")
        result = cls(
            schema_version=data["schema_version"],  # type: ignore[arg-type]
            mission_id=data["mission_id"],  # type: ignore[arg-type]
            uav_id=data["uav_id"],  # type: ignore[arg-type]
            plan_version=data["plan_version"],  # type: ignore[arg-type]
            assumptions=tuple(SpatialAssumption.from_dict(item) for item in assumptions),
            steps=tuple(PlanStepDraftV3.from_dict(item) for item in steps),
            target_spec=None if "target_spec" not in data else TargetSpec.from_dict(data["target_spec"]),  # type: ignore[arg-type]
        )
        if any(step.skill == "INSPECT" for step in result.steps):
            raise SpatialPlanValidationError(
                "INSPECT is unavailable in an initial V3 plan without a trusted candidate revision"
            )
        return result

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": 3,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "assumptions": [item.to_dict() for item in self.assumptions],
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.target_spec is not None:
            result["target_spec"] = self.target_spec.to_dict()
        return result


__all__ = [
    "PlanStepDraftV3", "SkillPlanDraftV3", "SpatialPlanValidationError",
]
