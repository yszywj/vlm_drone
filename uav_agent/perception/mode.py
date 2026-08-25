"""Explicit, fail-closed target-perception launch modes.

The public mode is intentionally smaller than the backend catalog.  ``disabled``
remains a backend for targetless missions, but it is not a selectable target
perception mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from perception.runtime import PerceptionRuntimeProfile


class TargetPerceptionMode(str, Enum):
    """Operator-visible target-perception modes."""

    ORACLE = "oracle"
    YOLO = "yolo"


class TargetPerceptionModeError(ValueError):
    """Raised when mode, profile, backend, and acknowledgement disagree."""


@dataclass(frozen=True, slots=True)
class ResolvedTargetPerceptionMode:
    """The single audited mapping used at the Fleet launch boundary."""

    mode: TargetPerceptionMode
    runtime_profile: PerceptionRuntimeProfile
    backend: str
    privileged: bool
    requires_oracle_acknowledgement: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("schema_version must equal integer 1")
        if not isinstance(self.mode, TargetPerceptionMode):
            raise TypeError("mode must be a TargetPerceptionMode")
        if not isinstance(self.runtime_profile, PerceptionRuntimeProfile):
            raise TypeError("runtime_profile must be a PerceptionRuntimeProfile")
        if not isinstance(self.backend, str) or not self.backend:
            raise TypeError("backend must be a non-empty string")
        if not isinstance(self.privileged, bool):
            raise TypeError("privileged must be bool")
        if not isinstance(self.requires_oracle_acknowledgement, bool):
            raise TypeError("requires_oracle_acknowledgement must be bool")

        canonical = _canonical_mode(self.mode)
        if (
            self.runtime_profile is not canonical.runtime_profile
            or self.backend != canonical.backend
            or self.privileged is not canonical.privileged
            or self.requires_oracle_acknowledgement
            is not canonical.requires_oracle_acknowledgement
        ):
            raise TargetPerceptionModeError(
                f"non-canonical fields for target perception mode {self.mode.value!r}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "runtime_profile": self.runtime_profile.value.lower(),
            "backend": self.backend,
            "privileged": self.privileged,
            "requires_oracle_acknowledgement": (
                self.requires_oracle_acknowledgement
            ),
        }


@dataclass(frozen=True, slots=True)
class _CanonicalMode:
    runtime_profile: PerceptionRuntimeProfile
    backend: str
    privileged: bool
    requires_oracle_acknowledgement: bool


def _canonical_mode(mode: TargetPerceptionMode) -> _CanonicalMode:
    if mode is TargetPerceptionMode.ORACLE:
        return _CanonicalMode(
            runtime_profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
            backend="oracle_evaluation",
            privileged=True,
            requires_oracle_acknowledgement=True,
        )
    if mode is TargetPerceptionMode.YOLO:
        return _CanonicalMode(
            runtime_profile=PerceptionRuntimeProfile.PRODUCTION,
            backend="ultralytics_service",
            privileged=False,
            requires_oracle_acknowledgement=False,
        )
    raise TargetPerceptionModeError(f"unsupported target perception mode: {mode!r}")


def _coerce_mode(value: TargetPerceptionMode | str) -> TargetPerceptionMode:
    if isinstance(value, TargetPerceptionMode):
        return value
    if not isinstance(value, str):
        raise TypeError("mode must be a TargetPerceptionMode or string")
    try:
        return TargetPerceptionMode(value.strip().lower())
    except ValueError as exc:
        raise TargetPerceptionModeError(
            "target perception mode must be 'oracle' or 'yolo'"
        ) from exc


def _coerce_profile(
    value: PerceptionRuntimeProfile | str,
) -> PerceptionRuntimeProfile:
    if isinstance(value, PerceptionRuntimeProfile):
        return value
    if not isinstance(value, str):
        raise TypeError("runtime_profile must be a PerceptionRuntimeProfile or string")
    try:
        return PerceptionRuntimeProfile(value.strip().upper())
    except ValueError as exc:
        raise TargetPerceptionModeError(
            f"unsupported perception runtime profile: {value!r}"
        ) from exc


def resolve_target_perception_mode(
    mode: TargetPerceptionMode | str,
    *,
    runtime_profile: PerceptionRuntimeProfile | str | None = None,
    backend: str | None = None,
    acknowledge_privileged_oracle: bool | None = None,
) -> ResolvedTargetPerceptionMode:
    """Resolve an explicit mode and reject every supplied conflicting switch.

    Optional arguments let pure callers obtain the canonical mapping while the
    Fleet CLI supplies all three values to enforce launch consistency.
    """

    selected = _coerce_mode(mode)
    canonical = _canonical_mode(selected)

    if runtime_profile is not None:
        supplied_profile = _coerce_profile(runtime_profile)
        if supplied_profile is not canonical.runtime_profile:
            raise TargetPerceptionModeError(
                f"target perception mode {selected.value!r} requires "
                "runtime profile "
                f"{canonical.runtime_profile.value.lower()!r}, got "
                f"{supplied_profile.value.lower()!r}"
            )
    if backend is not None:
        if not isinstance(backend, str) or not backend.strip():
            raise TypeError("backend must be a non-empty string or None")
        supplied_backend = backend.strip()
        if supplied_backend != canonical.backend:
            raise TargetPerceptionModeError(
                f"target perception mode {selected.value!r} requires YAML "
                f"target_perception.backend={canonical.backend!r}, got "
                f"{supplied_backend!r}; the backend is never changed silently"
            )
    if acknowledge_privileged_oracle is not None:
        if not isinstance(acknowledge_privileged_oracle, bool):
            raise TypeError("acknowledge_privileged_oracle must be bool or None")
        if canonical.requires_oracle_acknowledgement and not (
            acknowledge_privileged_oracle
        ):
            raise TargetPerceptionModeError(
                "oracle target perception requires "
                "--acknowledge-privileged-oracle"
            )
        if not canonical.requires_oracle_acknowledgement and (
            acknowledge_privileged_oracle
        ):
            raise TargetPerceptionModeError(
                "--acknowledge-privileged-oracle is forbidden in yolo mode"
            )

    return ResolvedTargetPerceptionMode(
        mode=selected,
        runtime_profile=canonical.runtime_profile,
        backend=canonical.backend,
        privileged=canonical.privileged,
        requires_oracle_acknowledgement=(
            canonical.requires_oracle_acknowledgement
        ),
    )


__all__ = [
    "ResolvedTargetPerceptionMode",
    "TargetPerceptionMode",
    "TargetPerceptionModeError",
    "resolve_target_perception_mode",
]
