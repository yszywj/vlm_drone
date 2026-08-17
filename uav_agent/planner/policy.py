"""Trusted policy shared by dynamic planning and compilation.

The language model may select values inside these bounds, but it never owns
the bounds or the default lost-target behaviour.  Keeping the types in the
planner package avoids making model-facing code depend on the runtime layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _bounded_number(
    value: object,
    field_name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    if not minimum <= normalized <= maximum:
        raise ValueError(
            f"{field_name} must be between {minimum:g} and {maximum:g}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class PlannerLimits:
    """Immutable v1 limits for an untrusted dynamic Skill draft."""

    max_plan_steps: int = 10
    max_goto_calls: int = 5
    max_search_calls: int = 1
    max_track_calls: int = 2
    max_reacquire_attempts_per_track: int = 2
    max_total_reacquire_attempts: int = 4
    min_track_duration_s: float = 1.0
    max_track_duration_s: float = 600.0

    def __post_init__(self) -> None:
        for name in (
            "max_plan_steps",
            "max_goto_calls",
            "max_search_calls",
            "max_track_calls",
            "max_reacquire_attempts_per_track",
            "max_total_reacquire_attempts",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(getattr(self, name), name),
            )

        if self.max_plan_steps < 2:
            raise ValueError("max_plan_steps must be at least 2")
        hard_caps = {
            "max_plan_steps": 10,
            "max_goto_calls": 5,
            "max_track_calls": 2,
            "max_reacquire_attempts_per_track": 2,
            "max_total_reacquire_attempts": 4,
        }
        for name, hard_cap in hard_caps.items():
            if getattr(self, name) > hard_cap:
                raise ValueError(f"{name} must not exceed {hard_cap} in planner v1")
        if self.max_search_calls != 1:
            raise ValueError("max_search_calls must be 1 in dynamic planner v1")
        if (
            self.max_reacquire_attempts_per_track
            > self.max_total_reacquire_attempts
        ):
            raise ValueError(
                "max_reacquire_attempts_per_track must not exceed "
                "max_total_reacquire_attempts"
            )

        minimum = _bounded_number(
            self.min_track_duration_s,
            "min_track_duration_s",
            minimum=0.0,
            maximum=float("inf"),
        )
        maximum = _bounded_number(
            self.max_track_duration_s,
            "max_track_duration_s",
            minimum=0.0,
            maximum=float("inf"),
        )
        if minimum <= 0.0 or maximum <= 0.0:
            raise ValueError("track duration limits must be greater than zero")
        if minimum > maximum:
            raise ValueError(
                "min_track_duration_s must not exceed max_track_duration_s"
            )
        object.__setattr__(self, "min_track_duration_s", minimum)
        object.__setattr__(self, "max_track_duration_s", maximum)

    @classmethod
    def from_config(cls, config: object) -> "PlannerLimits":
        """Construct limits from the public ``PlannerConfig`` contract."""

        names = tuple(cls.__dataclass_fields__)
        try:
            values = {name: getattr(config, name) for name in names}
        except AttributeError as exc:
            raise TypeError("config must expose every PlannerLimits field") from exc
        return cls(**values)


class TargetLostAction(str, Enum):
    """Supported bounded actions when TRACK loses its target."""

    REACQUIRE = "REACQUIRE"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class PlannerPolicy:
    """Trusted defaults used when a TRACK draft omits lost-target policy."""

    default_on_target_lost: TargetLostAction = TargetLostAction.REACQUIRE
    default_reacquire_max_attempts: int = 2
    default_reacquire_search_radius_m: float = 10.0
    default_reacquire_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        action = self.default_on_target_lost
        if not isinstance(action, TargetLostAction):
            if not isinstance(action, str):
                raise ValueError(
                    "default_on_target_lost must be REACQUIRE or FAIL"
                )
            try:
                action = TargetLostAction(action)
            except ValueError:
                raise ValueError(
                    "default_on_target_lost must be REACQUIRE or FAIL"
                ) from None
        object.__setattr__(self, "default_on_target_lost", action)

        attempts = _positive_integer(
            self.default_reacquire_max_attempts,
            "default_reacquire_max_attempts",
        )
        if attempts > 2:
            raise ValueError(
                "default_reacquire_max_attempts must not exceed 2 in planner v1"
            )
        object.__setattr__(self, "default_reacquire_max_attempts", attempts)
        object.__setattr__(
            self,
            "default_reacquire_search_radius_m",
            _bounded_number(
                self.default_reacquire_search_radius_m,
                "default_reacquire_search_radius_m",
                minimum=3.0,
                maximum=20.0,
            ),
        )
        object.__setattr__(
            self,
            "default_reacquire_timeout_s",
            _bounded_number(
                self.default_reacquire_timeout_s,
                "default_reacquire_timeout_s",
                minimum=5.0,
                maximum=60.0,
            ),
        )

    def validate_against(self, limits: PlannerLimits) -> "PlannerPolicy":
        """Fail closed if trusted defaults exceed the configured budgets."""

        if not isinstance(limits, PlannerLimits):
            raise TypeError("limits must be a PlannerLimits")
        attempts = self.default_reacquire_max_attempts
        if attempts > limits.max_reacquire_attempts_per_track:
            raise ValueError(
                "default_reacquire_max_attempts exceeds the per-TRACK limit"
            )
        default_total = attempts * limits.max_track_calls
        if default_total > limits.max_total_reacquire_attempts:
            raise ValueError(
                "default REACQUIRE budget exceeds max_total_reacquire_attempts"
            )
        return self

    @classmethod
    def from_config(
        cls,
        config: object,
        limits: PlannerLimits | None = None,
    ) -> "PlannerPolicy":
        names = tuple(cls.__dataclass_fields__)
        try:
            values = {name: getattr(config, name) for name in names}
        except AttributeError as exc:
            raise TypeError("config must expose every PlannerPolicy field") from exc
        policy = cls(**values)
        if limits is not None:
            policy.validate_against(limits)
        return policy

    def to_prompt_dict(self) -> dict[str, object]:
        """Return the minimal policy projection safe for model input."""

        return {
            "default_on_target_lost": self.default_on_target_lost.value,
            "allowed_on_target_lost": [
                TargetLostAction.REACQUIRE.value,
                TargetLostAction.FAIL.value,
            ],
        }


__all__ = ["PlannerLimits", "PlannerPolicy", "TargetLostAction"]
