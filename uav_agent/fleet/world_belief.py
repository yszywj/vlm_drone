"""Sanitized fleet summary shared with coordination logic and local planners."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType

from common.ids import validate_mission_id, validate_routing_id, validate_uav_id


_FORBIDDEN_KEYS = frozenset(
    {
        "camera_rgb",
        "rgb",
        "image",
        "images",
        "oracle_target_pose",
        "oracle_target_velocity",
        "oracle_target_visible",
        "oracle_target_id",
    }
)


def _safe_copy(value: object, path: str = "belief") -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, nested in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"{path} keys must be strings")
            normalized_key = raw_key.casefold()
            if (
                normalized_key in _FORBIDDEN_KEYS
                or normalized_key.startswith("oracle_target_")
            ):
                raise ValueError(f"{path}.{raw_key} is not allowed in FleetWorldBelief")
            result[raw_key] = _safe_copy(nested, f"{path}.{raw_key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_copy(item, f"{path}[]") for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _safe_copy(value.to_dict(), path)
    raise TypeError(f"{path} contains unsupported value {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class AgentFleetSummary:
    uav_id: str
    assignment_id: str
    status: str
    plan_version: int
    current_region: str | None = None
    altitude_layer: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(self, "assignment_id", validate_routing_id(self.assignment_id, "assignment_id"))
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("status must be a non-empty string")
        object.__setattr__(self, "status", self.status.strip())
        if isinstance(self.plan_version, bool) or not isinstance(self.plan_version, int) or self.plan_version <= 0:
            raise ValueError("plan_version must be a positive integer")
        for name in ("current_region", "altitude_layer"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty string or None")
            if isinstance(value, str):
                object.__setattr__(self, name, value.strip())

    def to_dict(self) -> dict[str, object]:
        return {
            "uav_id": self.uav_id,
            "assignment_id": self.assignment_id,
            "status": self.status,
            "plan_version": self.plan_version,
            "current_region": self.current_region,
            "altitude_layer": self.altitude_layer,
        }


@dataclass(frozen=True, slots=True)
class FleetWorldBelief:
    fleet_mission_id: str
    fleet_plan_version: int
    timestamp_s: float
    agents: Mapping[str, AgentFleetSummary]
    target_claims: Mapping[str, object] = field(default_factory=dict)
    airspace: Mapping[str, object] = field(default_factory=dict)
    events: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fleet_mission_id", validate_mission_id(self.fleet_mission_id))
        if isinstance(self.fleet_plan_version, bool) or not isinstance(self.fleet_plan_version, int) or self.fleet_plan_version <= 0:
            raise ValueError("fleet_plan_version must be a positive integer")
        if (
            isinstance(self.timestamp_s, bool)
            or not isinstance(self.timestamp_s, (int, float))
            or not isfinite(float(self.timestamp_s))
            or self.timestamp_s < 0
        ):
            raise ValueError("timestamp_s must be a finite non-negative number")
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        normalized: dict[str, AgentFleetSummary] = {}
        for raw_id, summary in self.agents.items():
            uav_id = validate_uav_id(raw_id)
            if not isinstance(summary, AgentFleetSummary) or summary.uav_id != uav_id:
                raise ValueError("agents must map UAV IDs to matching AgentFleetSummary")
            normalized[uav_id] = summary
        object.__setattr__(self, "agents", MappingProxyType(dict(sorted(normalized.items()))))
        object.__setattr__(self, "target_claims", MappingProxyType(_safe_copy(self.target_claims, "target_claims")))
        object.__setattr__(self, "airspace", MappingProxyType(_safe_copy(self.airspace, "airspace")))
        object.__setattr__(self, "events", tuple(MappingProxyType(_safe_copy(event, "events")) for event in self.events))

    def local_safety_summary(self, uav_id: str) -> tuple[dict[str, object], ...]:
        """Return only other UAV region/layer/assignment metadata, never images."""

        uav_id = validate_uav_id(uav_id)
        return tuple(
            {
                "uav_id": other_id,
                "assignment_id": summary.assignment_id,
                "current_region": summary.current_region,
                "altitude_layer": summary.altitude_layer,
                "status": summary.status,
            }
            for other_id, summary in self.agents.items()
            if other_id != uav_id
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "fleet_mission_id": self.fleet_mission_id,
            "fleet_plan_version": self.fleet_plan_version,
            "timestamp_s": self.timestamp_s,
            "agents": {uav_id: summary.to_dict() for uav_id, summary in self.agents.items()},
            "target_claims": deepcopy(dict(self.target_claims)),
            "airspace": deepcopy(dict(self.airspace)),
            "events": [deepcopy(dict(event)) for event in self.events],
        }


__all__ = ["AgentFleetSummary", "FleetWorldBelief"]
