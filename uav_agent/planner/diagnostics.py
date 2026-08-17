"""Stable, sanitized diagnostics for one high-level planning attempt."""

from __future__ import annotations

from dataclasses import dataclass

from planner.schemas import MissionIntent, PlannerOutput, SkillPlanDraft


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty when provided")
    return normalized


@dataclass(frozen=True, slots=True)
class PlannerDiagnostics:
    """Small metadata record which deliberately excludes raw model output."""

    model_calls: int
    repair_used: bool
    repair_succeeded: bool
    initial_output_valid: bool
    final_output_valid: bool
    initial_error_code: str | None
    initial_error_message: str | None
    structured_output_enabled: bool

    def __post_init__(self) -> None:
        if isinstance(self.model_calls, bool) or not isinstance(self.model_calls, int):
            raise TypeError("model_calls must be an integer")
        if self.model_calls < 0:
            raise ValueError("model_calls must be non-negative")
        for field_name in (
            "repair_used",
            "repair_succeeded",
            "initial_output_valid",
            "final_output_valid",
            "structured_output_enabled",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        object.__setattr__(
            self,
            "initial_error_code",
            _optional_text(self.initial_error_code, "initial_error_code"),
        )
        object.__setattr__(
            self,
            "initial_error_message",
            _optional_text(self.initial_error_message, "initial_error_message"),
        )
        if self.repair_succeeded and not self.repair_used:
            raise ValueError("repair_succeeded requires repair_used")
        if self.repair_used and self.model_calls < 2:
            raise ValueError("repair_used requires at least two model calls")
        if self.repair_used and self.initial_output_valid:
            raise ValueError("repair cannot be used after a valid initial output")
        if self.initial_output_valid and self.initial_error_code is not None:
            raise ValueError("valid initial output cannot have an error code")
        if not self.initial_output_valid and self.model_calls > 0:
            if self.initial_error_code is None:
                raise ValueError("invalid initial output requires an error code")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_calls": self.model_calls,
            "repair_used": self.repair_used,
            "repair_succeeded": self.repair_succeeded,
            "initial_output_valid": self.initial_output_valid,
            "final_output_valid": self.final_output_valid,
            "initial_error_code": self.initial_error_code,
            "initial_error_message": self.initial_error_message,
            "structured_output_enabled": self.structured_output_enabled,
        }


@dataclass(frozen=True, slots=True)
class PlannerExecution:
    """A successful Planner output paired with sanitized diagnostics."""

    output: PlannerOutput
    diagnostics: PlannerDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.output, (MissionIntent, SkillPlanDraft)):
            raise TypeError("output must be a MissionIntent or SkillPlanDraft")
        if not isinstance(self.diagnostics, PlannerDiagnostics):
            raise TypeError("diagnostics must be PlannerDiagnostics")


__all__ = ["PlannerDiagnostics", "PlannerExecution"]
