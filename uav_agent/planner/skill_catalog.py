"""Stable, model-visible contracts for the supported high-level Skills.

The catalog is deliberately smaller than the runtime Goal dataclasses.  It is
an allow-list for semantic planning, not an introspection view of the flight
controller.  In particular it contains no coordinates, target truth, speed,
timeout, or low-level control parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from planner.text_safety import reject_forbidden_planner_text


_ARGUMENT_TYPES = frozenset({"string", "number", "integer"})
_MODEL_VISIBLE_ARGUMENTS: dict[str, frozenset[str]] = {
    "TAKEOFF": frozenset({"altitude_m", "yaw_mode", "yaw_deg"}),
    "GOTO": frozenset({"destination", "altitude_m", "yaw_mode", "yaw_deg"}),
    "SEARCH": frozenset({"region", "target_description", "altitude_m"}),
    "TRACK": frozenset(
        {
            "target_ref",
            "duration_s",
            "desired_altitude_m",
            "desired_distance_m",
            "on_target_lost",
        }
    ),
    "REACQUIRE": frozenset(
        {"max_attempts", "search_radius_m", "timeout_s"}
    ),
    "LAND": frozenset({"zone", "yaw_mode", "yaw_deg"}),
}


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _finite_bound(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


@dataclass(frozen=True, slots=True)
class SkillArgumentSpec:
    """One model-visible argument in a :class:`SkillContract`."""

    name: str
    description: str
    value_type: str
    required: bool = True
    allowed_values: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    condition: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, "name"))
        object.__setattr__(
            self,
            "description",
            _non_empty(self.description, "description"),
        )
        reject_forbidden_planner_text(
            self.description,
            f"{self.name} argument description",
        )
        value_type = _non_empty(self.value_type, "value_type")
        if value_type not in _ARGUMENT_TYPES:
            raise ValueError(
                "value_type must be one of: " + ", ".join(sorted(_ARGUMENT_TYPES))
            )
        object.__setattr__(self, "value_type", value_type)
        if not isinstance(self.required, bool):
            raise TypeError("required must be a bool")

        if not isinstance(self.allowed_values, tuple):
            object.__setattr__(self, "allowed_values", tuple(self.allowed_values))
        normalized_values = tuple(
            _non_empty(value, "allowed_values entry")
            for value in self.allowed_values
        )
        if len(normalized_values) != len(set(normalized_values)):
            raise ValueError("allowed_values must be unique")
        object.__setattr__(self, "allowed_values", normalized_values)
        for index, value in enumerate(normalized_values):
            reject_forbidden_planner_text(
                value,
                f"{self.name} allowed_values[{index}]",
            )

        minimum = (
            None
            if self.minimum is None
            else _finite_bound(self.minimum, "minimum")
        )
        maximum = (
            None
            if self.maximum is None
            else _finite_bound(self.maximum, "maximum")
        )
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("minimum must not exceed maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        if self.condition is not None:
            object.__setattr__(
                self,
                "condition",
                _non_empty(self.condition, "condition"),
            )
            reject_forbidden_planner_text(
                self.condition,
                f"{self.name} condition",
            )

    def to_prompt_dict(self) -> dict[str, object]:
        """Return a fresh, deterministic JSON-compatible representation."""

        result: dict[str, object] = {
            "name": self.name,
            "description": self.description,
            "type": self.value_type,
            "required": self.required,
        }
        if self.allowed_values:
            result["allowed_values"] = list(self.allowed_values)
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        if self.condition is not None:
            result["condition"] = self.condition
        return result


@dataclass(frozen=True, slots=True)
class SkillContract:
    """The high-level portion of one Skill contract visible to the model."""

    name: str
    description: str
    top_level_allowed: bool
    recovery_only: bool
    arguments: tuple[SkillArgumentSpec, ...]
    preconditions: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = _non_empty(self.name, "name").upper()
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "description",
            _non_empty(self.description, "description"),
        )
        reject_forbidden_planner_text(self.description, f"{name} description")
        if not isinstance(self.top_level_allowed, bool):
            raise TypeError("top_level_allowed must be a bool")
        if not isinstance(self.recovery_only, bool):
            raise TypeError("recovery_only must be a bool")
        if self.recovery_only and self.top_level_allowed:
            raise ValueError("a recovery-only Skill cannot be top-level")

        arguments = tuple(self.arguments)
        if any(not isinstance(item, SkillArgumentSpec) for item in arguments):
            raise TypeError("arguments must contain only SkillArgumentSpec values")
        argument_names = tuple(item.name for item in arguments)
        if len(argument_names) != len(set(argument_names)):
            raise ValueError("Skill argument names must be unique")
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(
            self,
            "preconditions",
            tuple(_non_empty(value, "precondition") for value in self.preconditions),
        )
        object.__setattr__(
            self,
            "outputs",
            tuple(_non_empty(value, "output") for value in self.outputs),
        )
        for index, value in enumerate(self.preconditions):
            reject_forbidden_planner_text(
                value,
                f"{name} precondition[{index}]",
            )
        for index, value in enumerate(self.outputs):
            reject_forbidden_planner_text(value, f"{name} output[{index}]")

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "top_level_allowed": self.top_level_allowed,
            "recovery_only": self.recovery_only,
            "arguments": [item.to_prompt_dict() for item in self.arguments],
            "preconditions": list(self.preconditions),
            "outputs": list(self.outputs),
        }


@dataclass(frozen=True, slots=True)
class SkillCatalog:
    """Immutable, ordered collection of model-visible Skill contracts."""

    skills: tuple[SkillContract, ...]

    def __post_init__(self) -> None:
        skills = tuple(self.skills)
        if not skills:
            raise ValueError("SkillCatalog must contain at least one Skill")
        if any(not isinstance(item, SkillContract) for item in skills):
            raise TypeError("skills must contain only SkillContract values")
        names = tuple(item.name for item in skills)
        if len(names) != len(set(names)):
            raise ValueError("Skill names must be unique")
        unknown_skills = sorted(set(names) - set(_MODEL_VISIBLE_ARGUMENTS))
        if unknown_skills:
            raise ValueError(
                "SkillCatalog contains unsupported v1 Skills: "
                + ", ".join(unknown_skills)
            )
        for skill in skills:
            expected_recovery_only = skill.name == "REACQUIRE"
            if skill.recovery_only is not expected_recovery_only:
                raise ValueError(
                    f"{skill.name} has an invalid recovery_only capability"
                )
            if skill.top_level_allowed is expected_recovery_only:
                raise ValueError(
                    f"{skill.name} has an invalid top_level_allowed capability"
                )
            allowed_arguments = _MODEL_VISIBLE_ARGUMENTS[skill.name]
            exposed = {argument.name for argument in skill.arguments}
            unknown_arguments = sorted(exposed - allowed_arguments)
            if unknown_arguments:
                raise ValueError(
                    f"{skill.name} exposes unsupported low-level arguments: "
                    + ", ".join(unknown_arguments)
                )
            for argument in skill.arguments:
                reject_forbidden_planner_text(
                    argument.description,
                    f"{skill.name}.{argument.name} description",
                )
                if argument.condition is not None:
                    reject_forbidden_planner_text(
                        argument.condition,
                        f"{skill.name}.{argument.name} condition",
                    )
        object.__setattr__(self, "skills", skills)

    def __iter__(self):
        return iter(self.skills)

    def get(self, name: str) -> SkillContract:
        normalized = _non_empty(name, "name").upper()
        for contract in self.skills:
            if contract.name == normalized:
                return contract
        raise KeyError(normalized)

    def to_prompt_dict(self) -> dict[str, object]:
        """Return contracts in registration order for reproducible prompts."""

        return {"skills": [skill.to_prompt_dict() for skill in self.skills]}


def _argument(
    name: str,
    description: str,
    value_type: str,
    *,
    required: bool = True,
    allowed_values: tuple[str, ...] = (),
    minimum: float | None = None,
    maximum: float | None = None,
    condition: str | None = None,
) -> SkillArgumentSpec:
    return SkillArgumentSpec(
        name=name,
        description=description,
        value_type=value_type,
        required=required,
        allowed_values=allowed_values,
        minimum=minimum,
        maximum=maximum,
        condition=condition,
    )


def build_default_skill_catalog() -> SkillCatalog:
    """Build the stable v1 catalog for the six implemented Skills."""

    return SkillCatalog(
        skills=(
            SkillContract(
                name="TAKEOFF",
                description="从地面起飞到指定安全高度；必须是第一步且最多一次。",
                top_level_allowed=True,
                recovery_only=False,
                arguments=(
                    _argument(
                        "altitude_m",
                        "安全起飞高度；省略时使用可信世界默认值。",
                        "number",
                        required=False,
                    ),
                    _argument(
                        "yaw_mode",
                        "起飞时机头策略。",
                        "string",
                        required=False,
                        allowed_values=("KEEP_CURRENT", "FIXED"),
                    ),
                    _argument(
                        "yaw_deg",
                        "固定机头航向角（度）。",
                        "number",
                        required=False,
                        minimum=-360.0,
                        maximum=360.0,
                        condition="only allowed and required when yaw_mode is FIXED",
                    ),
                ),
                preconditions=("UAV is on the ground", "must be the first step"),
                outputs=("takeoff_complete",),
            ),
            SkillContract(
                name="GOTO",
                description="飞往可信 WorldContext 中的具名地点；不得提供坐标。",
                top_level_allowed=True,
                recovery_only=False,
                arguments=(
                    _argument("destination", "具名目的地。", "string"),
                    _argument(
                        "altitude_m",
                        "飞行高度；省略时由可信编译器选择。",
                        "number",
                        required=False,
                    ),
                    _argument(
                        "yaw_mode",
                        "移动期间机头策略。",
                        "string",
                        required=False,
                        allowed_values=(
                            "KEEP_CURRENT",
                            "COURSE_ALIGNED",
                            "FACE_POINT",
                            "FIXED",
                        ),
                    ),
                    _argument(
                        "yaw_deg",
                        "固定机头航向角（度）。",
                        "number",
                        required=False,
                        minimum=-360.0,
                        maximum=360.0,
                        condition="only allowed and required when yaw_mode is FIXED",
                    ),
                ),
                preconditions=("UAV is airborne", "destination is a trusted name"),
                outputs=("goal_reached",),
            ),
            SkillContract(
                name="SEARCH",
                description="在一个具名搜索区域内寻找单个任务目标；最多一次。",
                top_level_allowed=True,
                recovery_only=False,
                arguments=(
                    _argument("region", "具名搜索区域。", "string"),
                    _argument(
                        "target_description",
                        "需要寻找的目标语义描述。",
                        "string",
                    ),
                    _argument(
                        "altitude_m",
                        "搜索高度；省略时由可信编译器选择。",
                        "number",
                        required=False,
                    ),
                ),
                preconditions=("UAV is airborne", "no prior SEARCH in this plan"),
                outputs=("target_id",),
            ),
            SkillContract(
                name="TRACK",
                description=(
                    "跟踪先前 SEARCH 输出的目标。目标丢失时，省略 "
                    "on_target_lost 将继承 trusted_planner_policy 的默认动作；"
                    "用户明确要求不重新搜索、丢失即失败时使用 FAIL。"
                ),
                top_level_allowed=True,
                recovery_only=False,
                arguments=(
                    _argument(
                        "target_ref",
                        "格式为 $<先前SEARCH步骤id>.target_id。",
                        "string",
                    ),
                    _argument("duration_s", "跟踪持续时间。", "number"),
                    _argument(
                        "desired_altitude_m",
                        "期望跟踪高度；省略时由可信编译器选择。",
                        "number",
                        required=False,
                    ),
                    _argument(
                        "desired_distance_m",
                        "期望跟踪距离；省略时由可信编译器选择。",
                        "number",
                        required=False,
                    ),
                    _argument(
                        "on_target_lost",
                        "目标丢失动作；省略时继承可信 Planner Policy。",
                        "string",
                        required=False,
                        allowed_values=("REACQUIRE", "FAIL"),
                    ),
                ),
                preconditions=("a prior SEARCH target reference is available",),
                outputs=("track_complete", "last_seen_state_on_loss"),
            ),
            SkillContract(
                name="REACQUIRE",
                description="在 TRACK 丢失目标时进行有界恢复；只能用作 recovery。",
                top_level_allowed=False,
                recovery_only=True,
                arguments=(
                    _argument(
                        "max_attempts",
                        "该 TRACK 的最大恢复尝试次数。",
                        "integer",
                        minimum=1,
                        maximum=2,
                    ),
                    _argument(
                        "search_radius_m",
                        "局部重搜索半径。",
                        "number",
                        required=False,
                        minimum=3,
                        maximum=20,
                    ),
                    _argument(
                        "timeout_s",
                        "每次恢复的超时。",
                        "number",
                        required=False,
                        minimum=5,
                        maximum=60,
                    ),
                ),
                preconditions=("TRACK returned TARGET_LOST",),
                outputs=("target_id",),
            ),
            SkillContract(
                name="LAND",
                description="在可信 WorldContext 中的具名降落区垂直降落；必须是最后一步。",
                top_level_allowed=True,
                recovery_only=False,
                arguments=(
                    _argument("zone", "具名降落区。", "string"),
                    _argument(
                        "yaw_mode",
                        "降落期间机头策略。",
                        "string",
                        required=False,
                        allowed_values=("KEEP_CURRENT", "FIXED"),
                    ),
                    _argument(
                        "yaw_deg",
                        "固定机头航向角（度）。",
                        "number",
                        required=False,
                        minimum=-360.0,
                        maximum=360.0,
                        condition="only allowed and required when yaw_mode is FIXED",
                    ),
                ),
                preconditions=(
                    "UAV is airborne",
                    "must be the final step",
                    "the previous GOTO destination must equal this LAND zone",
                ),
                outputs=("land_complete",),
            ),
        )
    )


__all__ = [
    "SkillArgumentSpec",
    "SkillCatalog",
    "SkillContract",
    "build_default_skill_catalog",
]
