"""Perception capability boundary for production and privileged runs.

The Stage-0 Skills still consume privileged ``oracle_target_*`` fields.  That
legacy ideal-control path is supported only behind the explicitly named
``ORACLE_EVALUATION`` profile.  The default ``PRODUCTION`` profile rejects
both an Oracle backend and any observation that happens to contain privileged
fields, including observations emitted by a mislabelled third-party backend.

This module is pure Python and deliberately knows nothing about Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from common.provenance import is_privileged_oracle_source
from perception.base import PerceptionBackend
from skills.types import Observation


class PerceptionCapability(str, Enum):
    """The strongest information a perception backend may expose."""

    VISION = "VISION"
    PRIVILEGED_ORACLE = "PRIVILEGED_ORACLE"


class PerceptionRuntimeProfile(str, Enum):
    """Runtime information policy.

    ``PRODUCTION`` is the default and is appropriate for learned perception,
    deployment, and real vehicles.  ``ORACLE_EVALUATION`` is only for upper
    bounds, regression tests, label generation, and expert-policy rollouts.
    """

    PRODUCTION = "PRODUCTION"
    ORACLE_EVALUATION = "ORACLE_EVALUATION"


class PerceptionBoundaryError(RuntimeError):
    """Raised when privileged data attempts to cross a runtime boundary."""


_ORACLE_FIELDS = (
    "oracle_target_id",
    "oracle_target_visible",
    "oracle_target_pose",
    "oracle_target_velocity",
)


def observation_contains_oracle_data(observation: Observation) -> bool:
    """Return whether any privileged observation field is populated."""

    if not isinstance(observation, Observation):
        raise TypeError("observation must be an Observation")
    legacy_oracle = any(
        getattr(observation, name, None) is not None
        for name in _ORACLE_FIELDS
    )
    estimate = getattr(observation, "target_estimate", None)
    return legacy_oracle or (
        estimate is not None
        and is_privileged_oracle_source(estimate.source)
    )


def validate_observation_access(
    observation: Observation,
    profile: PerceptionRuntimeProfile = PerceptionRuntimeProfile.PRODUCTION,
) -> None:
    """Validate one observation against the selected information policy."""

    if not isinstance(observation, Observation):
        raise TypeError("perception backend must return an Observation")
    if not isinstance(profile, PerceptionRuntimeProfile):
        raise TypeError("profile must be a PerceptionRuntimeProfile")
    observation.validate()
    if (
        profile is PerceptionRuntimeProfile.PRODUCTION
        and observation_contains_oracle_data(observation)
    ):
        populated_fields = [
            name for name in _ORACLE_FIELDS if getattr(observation, name) is not None
        ]
        if (
            observation.target_estimate is not None
            and is_privileged_oracle_source(
                observation.target_estimate.source
            )
        ):
            populated_fields.append(
                "target_estimate.source="
                f"{observation.target_estimate.source}"
            )
        populated = ", ".join(populated_fields)
        raise PerceptionBoundaryError(
            "production Agent Runtime rejects privileged Oracle fields: "
            f"{populated}"
        )


@dataclass(frozen=True, slots=True)
class GuardedPerceptionBackend:
    """Apply a declared runtime policy to every backend observation.

    Oracle use requires *both* the Oracle-specific profile and the explicit
    acknowledgement flag.  This two-part opt-in prevents a production config
    from silently selecting ground truth merely by swapping backend objects.
    """

    backend: PerceptionBackend
    profile: PerceptionRuntimeProfile = PerceptionRuntimeProfile.PRODUCTION
    acknowledge_privileged_oracle: bool = False
    _declared_capability: PerceptionCapability = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, PerceptionBackend):
            raise TypeError("backend must satisfy PerceptionBackend")
        if not isinstance(self.profile, PerceptionRuntimeProfile):
            raise TypeError("profile must be a PerceptionRuntimeProfile")
        if not isinstance(self.acknowledge_privileged_oracle, bool):
            raise TypeError("acknowledge_privileged_oracle must be bool")

        capability = getattr(
            self.backend,
            "capability",
            PerceptionCapability.VISION,
        )
        if not isinstance(capability, PerceptionCapability):
            raise TypeError(
                "backend.capability must be a PerceptionCapability when declared"
            )
        object.__setattr__(self, "_declared_capability", capability)
        self._validate_capability_policy(capability)

    def _validate_capability_policy(
        self,
        capability: PerceptionCapability,
    ) -> None:
        if capability is PerceptionCapability.PRIVILEGED_ORACLE:
            if self.profile is not PerceptionRuntimeProfile.ORACLE_EVALUATION:
                raise PerceptionBoundaryError(
                    "privileged Oracle backend is forbidden in the default "
                    "PRODUCTION runtime profile"
                )
            if not self.acknowledge_privileged_oracle:
                raise PerceptionBoundaryError(
                    "ORACLE_EVALUATION requires explicit "
                    "acknowledge_privileged_oracle=True"
                )
        elif self.acknowledge_privileged_oracle:
            raise PerceptionBoundaryError(
                "privileged acknowledgement is invalid for a vision backend"
            )

    @property
    def capability(self) -> PerceptionCapability:
        value = getattr(
            self.backend,
            "capability",
            PerceptionCapability.VISION,
        )
        if not isinstance(value, PerceptionCapability):  # guarded at construction
            raise PerceptionBoundaryError("backend capability changed at runtime")
        if value is not self._declared_capability:
            raise PerceptionBoundaryError(
                "backend capability changed after the runtime policy was established"
            )
        self._validate_capability_policy(value)
        return value

    def observe(self, frame: object) -> Observation:
        capability = self.capability
        observation = self.backend.observe(frame)
        validate_observation_access(observation, self.profile)
        if (
            self.profile is PerceptionRuntimeProfile.ORACLE_EVALUATION
            and capability is not PerceptionCapability.PRIVILEGED_ORACLE
            and observation_contains_oracle_data(observation)
        ):
            raise PerceptionBoundaryError(
                "only a backend declaring PRIVILEGED_ORACLE may emit Oracle fields"
            )
        return observation

    def get_observation(self, frame: object) -> Observation:
        """Compatibility alias for runtimes using ``get_*`` naming."""

        return self.observe(frame)


__all__ = [
    "GuardedPerceptionBackend",
    "PerceptionBoundaryError",
    "PerceptionCapability",
    "PerceptionRuntimeProfile",
    "observation_contains_oracle_data",
    "validate_observation_access",
]
