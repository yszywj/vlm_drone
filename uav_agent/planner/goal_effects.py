"""Small symbolic effect model for freely ordered top-level Skills.

This is intentionally not a second flight controller.  It describes only the
facts needed to decide whether a finite plan contains a plausible path to a
semantic goal.  Geometry and current physical safety remain authoritative in
``PlanValidator`` and ``SafetySupervisor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GoalFact(str, Enum):
    ON_GROUND = "ON_GROUND"
    AIRBORNE = "AIRBORNE"
    TARGET_CANDIDATE_AVAILABLE = "TARGET_CANDIDATE_AVAILABLE"
    CONFIRMED_TARGET_AVAILABLE = "CONFIRMED_TARGET_AVAILABLE"
    AT_DESTINATION = "AT_DESTINATION"
    VALID_LANDING_ZONE = "VALID_LANDING_ZONE"
    LANDED = "LANDED"
    TRACKED_DURATION = "TRACKED_DURATION"
    WAITED_DURATION = "WAITED_DURATION"
    REPORT_AVAILABLE = "REPORT_AVAILABLE"


@dataclass(frozen=True, slots=True)
class SkillEffect:
    """Protocol-v1 preconditions and effects for one top-level Skill."""

    schema_version: int
    skill: str
    preconditions: frozenset[GoalFact]
    effects: frozenset[GoalFact]
    possible_effects: frozenset[GoalFact] = frozenset()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("SkillEffect.schema_version must equal 1")
        if not isinstance(self.skill, str) or not self.skill:
            raise ValueError("SkillEffect.skill must be a non-empty string")
        for name in ("preconditions", "effects", "possible_effects"):
            values = frozenset(getattr(self, name))
            if any(not isinstance(item, GoalFact) for item in values):
                raise TypeError(f"SkillEffect.{name} must contain GoalFact values")
            object.__setattr__(self, name, values)


DEFAULT_SKILL_EFFECTS: dict[str, SkillEffect] = {
    "TAKEOFF": SkillEffect(
        1,
        "TAKEOFF",
        frozenset({GoalFact.ON_GROUND}),
        frozenset({GoalFact.AIRBORNE}),
    ),
    "GOTO": SkillEffect(
        1,
        "GOTO",
        frozenset({GoalFact.AIRBORNE}),
        frozenset({GoalFact.AT_DESTINATION}),
    ),
    "HOVER": SkillEffect(
        1,
        "HOVER",
        frozenset({GoalFact.AIRBORNE}),
        frozenset({GoalFact.WAITED_DURATION}),
    ),
    "SEARCH": SkillEffect(
        1,
        "SEARCH",
        frozenset({GoalFact.AIRBORNE}),
        frozenset(),
        frozenset({GoalFact.TARGET_CANDIDATE_AVAILABLE}),
    ),
    "INSPECT": SkillEffect(
        1,
        "INSPECT",
        frozenset({GoalFact.TARGET_CANDIDATE_AVAILABLE}),
        frozenset(),
        frozenset({GoalFact.CONFIRMED_TARGET_AVAILABLE}),
    ),
    "TRACK": SkillEffect(
        1,
        "TRACK",
        frozenset({GoalFact.CONFIRMED_TARGET_AVAILABLE}),
        frozenset({GoalFact.TRACKED_DURATION}),
    ),
    "LAND": SkillEffect(
        1,
        "LAND",
        frozenset({GoalFact.AIRBORNE, GoalFact.VALID_LANDING_ZONE}),
        frozenset({GoalFact.LANDED, GoalFact.ON_GROUND}),
    ),
    # REPORT is a semantic/runtime action and may be introduced by a future
    # catalog.  Modelling it now keeps GoalType.REPORT honest without forcing
    # the current controller catalog to advertise it.
    "REPORT": SkillEffect(
        1,
        "REPORT",
        frozenset(),
        frozenset({GoalFact.REPORT_AVAILABLE}),
    ),
}


def skill_effect(skill: object) -> SkillEffect | None:
    """Return the effect model for a string or Enum-like Skill value."""

    value = getattr(skill, "value", skill)
    if not isinstance(value, str):
        return None
    return DEFAULT_SKILL_EFFECTS.get(value.upper())


__all__ = ["DEFAULT_SKILL_EFFECTS", "GoalFact", "SkillEffect", "skill_effect"]
