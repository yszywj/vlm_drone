"""Fail-closed factory for independent target-perception backends."""

from __future__ import annotations

from collections.abc import Iterable

from configs.schema import AppConfig
from perception.base import PerceptionBackend
from perception.oracle import OraclePerception
from perception.runtime import (
    GuardedPerceptionBackend,
    PerceptionBoundaryError,
    PerceptionRuntimeProfile,
)
from perception.vision_backend import (
    DisabledTargetPerceptionBackend,
    VisionPerceptionBackend,
)
from skills.types import SkillName


class TargetPerceptionConfigurationError(PerceptionBoundaryError):
    """Raised before runtime when profile/backend switches conflict."""


class TargetPerceptionUnavailableError(TargetPerceptionConfigurationError):
    """Raised when a target-dependent plan selected the disabled backend."""


_TARGET_SKILLS = frozenset(
    {SkillName.SEARCH, SkillName.INSPECT, SkillName.TRACK, SkillName.REACQUIRE}
)


def build_target_perception_backend(
    config: AppConfig,
    *,
    runtime_profile: PerceptionRuntimeProfile,
    acknowledge_privileged_oracle: bool,
    uav_id: str,
) -> PerceptionBackend:
    """Construct exactly the configured backend without any fallback path."""

    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    if not isinstance(runtime_profile, PerceptionRuntimeProfile):
        raise TypeError("runtime_profile must be a PerceptionRuntimeProfile")
    if not isinstance(acknowledge_privileged_oracle, bool):
        raise TypeError("acknowledge_privileged_oracle must be bool")
    backend_name = config.target_perception.backend

    if runtime_profile is PerceptionRuntimeProfile.ORACLE_EVALUATION:
        if backend_name != "oracle_evaluation":
            raise TargetPerceptionConfigurationError(
                "ORACLE_EVALUATION requires target_perception.backend="
                "oracle_evaluation"
            )
        if not acknowledge_privileged_oracle:
            raise TargetPerceptionConfigurationError(
                "Oracle target perception requires explicit acknowledgement"
            )
        return GuardedPerceptionBackend(
            OraclePerception(uav_id=uav_id, target_id="target"),
            profile=runtime_profile,
            acknowledge_privileged_oracle=True,
        )

    if acknowledge_privileged_oracle:
        raise TargetPerceptionConfigurationError(
            "Oracle acknowledgement is invalid in production"
        )
    if backend_name == "oracle_evaluation":
        raise TargetPerceptionConfigurationError(
            "production profile forbids target_perception.backend=oracle_evaluation"
        )
    if backend_name == "ultralytics_service":
        return VisionPerceptionBackend(config.target_perception, uav_id=uav_id)
    if backend_name == "disabled":
        return DisabledTargetPerceptionBackend(uav_id=uav_id)
    # AppConfig can be constructed directly rather than through load_config;
    # repeat the closed-set check at this trust boundary.
    raise TargetPerceptionConfigurationError(
        f"unknown target_perception backend: {backend_name!r}"
    )


def validate_target_perception_preflight(
    backend_name: str,
    skills: Iterable[SkillName | str],
) -> None:
    """Reject target-dependent plans when target perception is disabled."""

    if backend_name != "disabled":
        return
    required: list[str] = []
    for value in skills:
        try:
            skill = value if isinstance(value, SkillName) else SkillName(str(value).upper())
        except ValueError as exc:
            raise ValueError(f"unknown Skill in target-perception preflight: {value!r}") from exc
        if skill in _TARGET_SKILLS:
            required.append(skill.value)
    if required:
        raise TargetPerceptionUnavailableError(
            "target_perception.backend=disabled cannot execute target Skills: "
            + ", ".join(required)
        )


__all__ = [
    "TargetPerceptionConfigurationError",
    "TargetPerceptionUnavailableError",
    "build_target_perception_backend",
    "validate_target_perception_preflight",
]
