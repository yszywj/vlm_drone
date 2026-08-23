"""Routing-safe OR fusion for independent obstacle hazard reporters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import Iterable

from common.ids import (
    validate_mission_id,
    validate_routing_id,
    validate_uav_id,
)
from common.obstacle_types import (
    IDEAL_CAMERA_OBSTACLE_SOURCE,
    ObstacleObservation,
)


class HazardSource(str, Enum):
    IDEAL_CAMERA = IDEAL_CAMERA_OBSTACLE_SOURCE
    FUTURE_LOW_LEVEL_DETECTOR = "future_low_level_detector"
    QWEN_VISUAL_REVIEW = "qwen_visual_review"


@dataclass(frozen=True, slots=True)
class HazardReport:
    """One source's bounded report, prior to route-level fusion."""

    source: HazardSource
    hazard_detected: bool
    geometry_grounded: bool = False
    obstacle_ids: tuple[str, ...] = ()
    visible_obstacle_ids: tuple[str, ...] = ()
    active_corridor_intersection: bool = False
    minimum_ttc_s: float | None = None
    minimum_distance_m: float | None = None
    imminent_collision: bool = False
    privileged: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source, HazardSource):
            try:
                object.__setattr__(self, "source", HazardSource(self.source))
            except (TypeError, ValueError):
                raise ValueError("source must be a supported HazardSource") from None
        for name in (
            "hazard_detected",
            "geometry_grounded",
            "active_corridor_intersection",
            "imminent_collision",
            "privileged",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        object.__setattr__(
            self,
            "obstacle_ids",
            _obstacle_ids(self.obstacle_ids, "obstacle_ids"),
        )
        object.__setattr__(
            self,
            "visible_obstacle_ids",
            _obstacle_ids(self.visible_obstacle_ids, "visible_obstacle_ids"),
        )
        object.__setattr__(
            self,
            "minimum_ttc_s",
            _optional_nonnegative(self.minimum_ttc_s, "minimum_ttc_s"),
        )
        object.__setattr__(
            self,
            "minimum_distance_m",
            _optional_nonnegative(self.minimum_distance_m, "minimum_distance_m"),
        )
        if self.geometry_grounded and not self.obstacle_ids:
            raise ValueError("geometry_grounded reports must identify an obstacle")
        if self.geometry_grounded and not self.hazard_detected:
            raise ValueError("geometry_grounded requires hazard_detected")
        if self.imminent_collision and not self.hazard_detected:
            raise ValueError("imminent_collision requires hazard_detected")

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.value,
            "privileged": self.privileged,
            "hazard_detected": self.hazard_detected,
            "geometry_grounded": self.geometry_grounded,
            "obstacle_ids": list(self.obstacle_ids),
            "visible_obstacle_ids": list(self.visible_obstacle_ids),
            "active_corridor_intersection": self.active_corridor_intersection,
            "minimum_ttc_s": self.minimum_ttc_s,
            "minimum_distance_m": self.minimum_distance_m,
            "imminent_collision": self.imminent_collision,
        }


@dataclass(frozen=True, slots=True)
class HazardFusionResult:
    mission_id: str
    uav_id: str
    plan_version: int
    timestamp_s: float
    reports: tuple[HazardReport, ...]
    low_level_hazard_detected: bool
    qwen_hazard_detected: bool
    should_hold: bool
    geometry_grounded: bool
    imminent_collision: bool
    obstacle_ids: tuple[str, ...]
    visible_obstacle_ids: tuple[str, ...]
    minimum_ttc_s: float | None
    braking_distance_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        if (
            isinstance(self.plan_version, bool)
            or not isinstance(self.plan_version, int)
            or self.plan_version <= 0
        ):
            raise ValueError("plan_version must be a positive integer")
        object.__setattr__(self, "timestamp_s", _nonnegative(self.timestamp_s, "timestamp_s"))
        reports = tuple(self.reports)
        if any(not isinstance(report, HazardReport) for report in reports):
            raise TypeError("reports must contain HazardReport values")
        object.__setattr__(self, "reports", reports)
        expected_hold = self.low_level_hazard_detected or self.qwen_hazard_detected
        if self.should_hold != expected_hold:
            raise ValueError("should_hold must equal low_level OR qwen hazard")
        if self.geometry_grounded and not self.should_hold:
            raise ValueError("geometry_grounded must describe an active hazard")
        object.__setattr__(
            self,
            "obstacle_ids",
            _obstacle_ids(self.obstacle_ids, "obstacle_ids"),
        )
        object.__setattr__(
            self,
            "visible_obstacle_ids",
            _obstacle_ids(self.visible_obstacle_ids, "visible_obstacle_ids"),
        )
        object.__setattr__(
            self,
            "minimum_ttc_s",
            _optional_nonnegative(self.minimum_ttc_s, "minimum_ttc_s"),
        )
        object.__setattr__(
            self,
            "braking_distance_m",
            _nonnegative(self.braking_distance_m, "braking_distance_m"),
        )

    @property
    def can_generate_route(self) -> bool:
        return self.should_hold and self.geometry_grounded

    def to_dict(self) -> dict[str, object]:
        return {
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "timestamp_s": self.timestamp_s,
            "sources": [report.source.value for report in self.reports],
            "source_reports": [report.to_dict() for report in self.reports],
            "low_level_hazard_detected": self.low_level_hazard_detected,
            "qwen_hazard_detected": self.qwen_hazard_detected,
            "should_hold": self.should_hold,
            "geometry_grounded": self.geometry_grounded,
            "can_generate_route": self.can_generate_route,
            "imminent_collision": self.imminent_collision,
            "obstacle_ids": list(self.obstacle_ids),
            "visible_obstacle_ids": list(self.visible_obstacle_ids),
            "minimum_ttc_s": self.minimum_ttc_s,
            "braking_distance_m": self.braking_distance_m,
        }


class HazardFusion:
    """Fuse independent sources while preserving stop/route distinctions."""

    def __init__(
        self,
        *,
        path_blocked_ttc_s: float = 5.0,
        imminent_collision_ttc_s: float = 1.5,
        maximum_deceleration_mps2: float = 3.0,
        braking_margin_m: float = 0.5,
    ) -> None:
        self.path_blocked_ttc_s = _positive(
            path_blocked_ttc_s, "path_blocked_ttc_s"
        )
        self.imminent_collision_ttc_s = _positive(
            imminent_collision_ttc_s, "imminent_collision_ttc_s"
        )
        if self.imminent_collision_ttc_s > self.path_blocked_ttc_s:
            raise ValueError(
                "imminent_collision_ttc_s must not exceed path_blocked_ttc_s"
            )
        self.maximum_deceleration_mps2 = _positive(
            maximum_deceleration_mps2, "maximum_deceleration_mps2"
        )
        self.braking_margin_m = _nonnegative(braking_margin_m, "braking_margin_m")

    def report_from_observation(
        self,
        observation: ObstacleObservation,
        *,
        uav_speed_mps: float,
    ) -> HazardReport:
        if not isinstance(observation, ObstacleObservation):
            raise TypeError("observation must be an ObstacleObservation")
        speed = _nonnegative(uav_speed_mps, "uav_speed_mps")
        braking_distance = speed * speed / (2.0 * self.maximum_deceleration_mps2)
        braking_distance += self.braking_margin_m
        visible_ids = tuple(item.obstacle_id for item in observation.visible_obstacles)
        hazards = tuple(
            item
            for item in observation.visible_obstacles
            if item.active_corridor_intersection
            and (
                (
                    item.time_to_collision_s is not None
                    and item.time_to_collision_s <= self.path_blocked_ttc_s
                )
                or item.depth_m <= braking_distance
            )
        )
        ttc_values = tuple(
            item.time_to_collision_s
            for item in hazards
            if item.time_to_collision_s is not None
        )
        min_ttc = min(ttc_values) if ttc_values else None
        minimum_distance = min((item.depth_m for item in hazards), default=None)
        imminent = any(
            (
                item.time_to_collision_s is not None
                and item.time_to_collision_s <= self.imminent_collision_ttc_s
            )
            or item.depth_m <= braking_distance
            for item in hazards
        )
        return HazardReport(
            source=HazardSource.IDEAL_CAMERA,
            hazard_detected=bool(hazards),
            geometry_grounded=bool(hazards),
            obstacle_ids=tuple(item.obstacle_id for item in hazards),
            visible_obstacle_ids=visible_ids,
            active_corridor_intersection=any(
                item.active_corridor_intersection
                for item in observation.visible_obstacles
            ),
            minimum_ttc_s=min_ttc,
            minimum_distance_m=minimum_distance,
            imminent_collision=imminent,
            privileged=observation.privileged,
        )

    @staticmethod
    def qwen_report(
        *,
        hazard_detected: bool,
        geometry_grounded: bool = False,
        obstacle_ids: Iterable[str] = (),
        imminent_collision: bool = False,
    ) -> HazardReport:
        return HazardReport(
            source=HazardSource.QWEN_VISUAL_REVIEW,
            hazard_detected=hazard_detected,
            geometry_grounded=geometry_grounded,
            obstacle_ids=tuple(obstacle_ids),
            visible_obstacle_ids=tuple(obstacle_ids),
            active_corridor_intersection=hazard_detected,
            imminent_collision=imminent_collision,
            privileged=False,
        )

    @staticmethod
    def low_level_report(
        *,
        hazard_detected: bool,
        geometry_grounded: bool = False,
        obstacle_ids: Iterable[str] = (),
        visible_obstacle_ids: Iterable[str] = (),
        minimum_ttc_s: float | None = None,
        imminent_collision: bool = False,
    ) -> HazardReport:
        return HazardReport(
            source=HazardSource.FUTURE_LOW_LEVEL_DETECTOR,
            hazard_detected=hazard_detected,
            geometry_grounded=geometry_grounded,
            obstacle_ids=tuple(obstacle_ids),
            visible_obstacle_ids=tuple(visible_obstacle_ids),
            active_corridor_intersection=hazard_detected,
            minimum_ttc_s=minimum_ttc_s,
            imminent_collision=imminent_collision,
            privileged=False,
        )

    def fuse(
        self,
        reports: Iterable[HazardReport],
        *,
        mission_id: str,
        uav_id: str,
        plan_version: int,
        timestamp_s: float,
        uav_speed_mps: float = 0.0,
    ) -> HazardFusionResult:
        normalized_mission = validate_mission_id(mission_id)
        normalized_uav = validate_uav_id(uav_id)
        if (
            isinstance(plan_version, bool)
            or not isinstance(plan_version, int)
            or plan_version <= 0
        ):
            raise ValueError("plan_version must be a positive integer")
        timestamp = _nonnegative(timestamp_s, "timestamp_s")
        speed = _nonnegative(uav_speed_mps, "uav_speed_mps")
        materialized = tuple(reports)
        if any(not isinstance(report, HazardReport) for report in materialized):
            raise TypeError("reports must contain HazardReport values")

        low_level = any(
            report.hazard_detected
            and report.source
            in {
                HazardSource.IDEAL_CAMERA,
                HazardSource.FUTURE_LOW_LEVEL_DETECTOR,
            }
            for report in materialized
        )
        qwen = any(
            report.hazard_detected
            and report.source is HazardSource.QWEN_VISUAL_REVIEW
            for report in materialized
        )
        should_hold = low_level or qwen
        geometry_grounded = any(
            report.hazard_detected and report.geometry_grounded
            for report in materialized
        )
        hazard_reports = tuple(report for report in materialized if report.hazard_detected)
        ttc_values = tuple(
            report.minimum_ttc_s
            for report in hazard_reports
            if report.minimum_ttc_s is not None
        )
        braking_distance = speed * speed / (2.0 * self.maximum_deceleration_mps2)
        braking_distance += self.braking_margin_m
        return HazardFusionResult(
            mission_id=normalized_mission,
            uav_id=normalized_uav,
            plan_version=plan_version,
            timestamp_s=timestamp,
            reports=materialized,
            low_level_hazard_detected=low_level,
            qwen_hazard_detected=qwen,
            should_hold=should_hold,
            geometry_grounded=geometry_grounded,
            imminent_collision=any(
                report.imminent_collision for report in hazard_reports
            ),
            obstacle_ids=_ordered_unique(
                obstacle_id
                for report in hazard_reports
                for obstacle_id in report.obstacle_ids
            ),
            visible_obstacle_ids=_ordered_unique(
                obstacle_id
                for report in materialized
                for obstacle_id in report.visible_obstacle_ids
            ),
            minimum_ttc_s=min(ttc_values) if ttc_values else None,
            braking_distance_m=braking_distance,
        )

    evaluate = fuse


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _obstacle_ids(value: Iterable[str], field_name: str) -> tuple[str, ...]:
    result = tuple(
        validate_routing_id(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


def _nonnegative(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite non-negative number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return result


def _positive(value: object, field_name: str) -> float:
    result = _nonnegative(value, field_name)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be greater than zero")
    return result


def _optional_nonnegative(value: object, field_name: str) -> float | None:
    return None if value is None else _nonnegative(value, field_name)


__all__ = [
    "HazardFusion",
    "HazardFusionResult",
    "HazardReport",
    "HazardSource",
]
