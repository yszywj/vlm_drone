"""Strict, JSON-safe records for temporal ray-depth training data.

The three top-level sections intentionally separate deployable sensor input,
real detector output, and privileged training labels.  Only collection and
offline training code may construct or read ``TargetTrainingLabel``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from pathlib import PurePosixPath
from typing import Mapping, Sequence

from common.ids import validate_routing_id, validate_uav_id


SCHEMA_VERSION = 1


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    if len(value) > 512:
        raise ValueError(f"{field} is too long")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _nonnegative(value: object, field: str) -> float:
    result = _finite(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _probability(value: object, field: str) -> float:
    result = _finite(value, field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return result


def _vector(value: object, field: str, length: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a {length}-number sequence")
    if len(value) != length:
        raise ValueError(f"{field} must contain {length} values")
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _relative_path(value: object, field: str, suffixes: tuple[str, ...]) -> str:
    text = _text(value, field)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe dataset-relative path")
    if path.suffix.lower() not in suffixes:
        raise ValueError(f"{field} must end with one of {suffixes}")
    return path.as_posix()


def _bbox(value: object) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    bbox = _vector(value, "bbox_xyxy_normalized", 4)
    x1, y1, x2, y2 = bbox
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("bbox_xyxy_normalized must be ordered within [0, 1]")
    return bbox  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CameraFrameInput:
    fx: float
    fy: float
    cx: float
    cy: float
    position_world_m: tuple[float, float, float]
    orientation_world_wxyz: tuple[float, float, float, float]
    resolution_wh_px: tuple[int, int]

    def __post_init__(self) -> None:
        fx, fy = _finite(self.fx, "camera.fx"), _finite(self.fy, "camera.fy")
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("camera fx/fy must be positive")
        width, height = self.resolution_wh_px
        width = _positive_int(width, "camera.width")
        height = _positive_int(height, "camera.height")
        cx, cy = _finite(self.cx, "camera.cx"), _finite(self.cy, "camera.cy")
        if not (0.0 <= cx < width and 0.0 <= cy < height):
            raise ValueError("camera principal point must lie inside the image")
        position = _vector(self.position_world_m, "camera.position_world_m", 3)
        quaternion = _vector(
            self.orientation_world_wxyz,
            "camera.orientation_world_wxyz",
            4,
        )
        norm = sum(item * item for item in quaternion) ** 0.5
        if norm <= 1e-12:
            raise ValueError("camera quaternion must have non-zero norm")
        object.__setattr__(self, "fx", fx)
        object.__setattr__(self, "fy", fy)
        object.__setattr__(self, "cx", cx)
        object.__setattr__(self, "cy", cy)
        object.__setattr__(self, "position_world_m", position)
        object.__setattr__(
            self,
            "orientation_world_wxyz",
            tuple(item / norm for item in quaternion),
        )
        object.__setattr__(self, "resolution_wh_px", (width, height))

    def to_dict(self) -> dict[str, object]:
        return {
            "intrinsics": {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy},
            "position_world_m": list(self.position_world_m),
            "orientation_world_wxyz": list(self.orientation_world_wxyz),
            "resolution_wh_px": list(self.resolution_wh_px),
        }


@dataclass(frozen=True, slots=True)
class UavFrameInput:
    position_world_m: tuple[float, float, float]
    orientation_world_wxyz: tuple[float, float, float, float]
    linear_velocity_world_mps: tuple[float, float, float]
    angular_velocity_body_radps: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_world_m", _vector(self.position_world_m, "uav.position_world_m", 3))
        quaternion = _vector(self.orientation_world_wxyz, "uav.orientation_world_wxyz", 4)
        norm = sum(item * item for item in quaternion) ** 0.5
        if norm <= 1e-12:
            raise ValueError("uav quaternion must have non-zero norm")
        object.__setattr__(self, "orientation_world_wxyz", tuple(item / norm for item in quaternion))
        object.__setattr__(self, "linear_velocity_world_mps", _vector(self.linear_velocity_world_mps, "uav.linear_velocity_world_mps", 3))
        object.__setattr__(self, "angular_velocity_body_radps", _vector(self.angular_velocity_body_radps, "uav.angular_velocity_body_radps", 3))

    def to_dict(self) -> dict[str, object]:
        return {
            "position_world_m": list(self.position_world_m),
            "orientation_world_wxyz": list(self.orientation_world_wxyz),
            "linear_velocity_world_mps": list(self.linear_velocity_world_mps),
            "angular_velocity_body_radps": list(self.angular_velocity_body_radps),
        }


@dataclass(frozen=True, slots=True)
class SensorInput:
    camera: CameraFrameInput
    uav: UavFrameInput
    rgb_path: str
    depth_path: str
    instance_mask_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.camera, CameraFrameInput) or not isinstance(self.uav, UavFrameInput):
            raise TypeError("sensor_input camera/uav types are invalid")
        object.__setattr__(self, "rgb_path", _relative_path(self.rgb_path, "rgb_path", (".jpg", ".jpeg", ".png")))
        object.__setattr__(self, "depth_path", _relative_path(self.depth_path, "depth_path", (".npy", ".npz", ".png", ".tiff")))
        if self.instance_mask_path is not None:
            object.__setattr__(self, "instance_mask_path", _relative_path(self.instance_mask_path, "instance_mask_path", (".png", ".npy", ".npz")))

    def to_dict(self) -> dict[str, object]:
        return {
            "camera": self.camera.to_dict(),
            "uav": self.uav.to_dict(),
            "rgb_path": self.rgb_path,
            "depth_path": self.depth_path,
            "instance_mask_path": self.instance_mask_path,
        }


@dataclass(frozen=True, slots=True)
class DetectorPrediction:
    detected: bool
    bbox_xyxy_normalized: tuple[float, float, float, float] | None
    confidence: float | None
    tracker_id: str | None
    candidate_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.detected, bool):
            raise TypeError("detector.detected must be bool")
        bbox = _bbox(self.bbox_xyxy_normalized)
        confidence = None if self.confidence is None else _probability(self.confidence, "detector.confidence")
        tracker = None if self.tracker_id is None else validate_routing_id(self.tracker_id, "tracker_id")
        candidate = None if self.candidate_id is None else validate_routing_id(self.candidate_id, "candidate_id")
        if self.detected and (bbox is None or confidence is None):
            raise ValueError("detected=True requires bbox and confidence")
        if not self.detected and any(value is not None for value in (bbox, confidence, tracker)):
            raise ValueError("detected=False cannot carry bbox/confidence/tracker_id")
        object.__setattr__(self, "bbox_xyxy_normalized", bbox)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "tracker_id", tracker)
        object.__setattr__(self, "candidate_id", candidate)

    def to_dict(self) -> dict[str, object]:
        return {
            "detected": self.detected,
            "bbox_xyxy_normalized": None if self.bbox_xyxy_normalized is None else list(self.bbox_xyxy_normalized),
            "confidence": self.confidence,
            "tracker_id": self.tracker_id,
            "candidate_id": self.candidate_id,
        }


@dataclass(frozen=True, slots=True)
class TargetTrainingLabel:
    position_world_m: tuple[float, float, float]
    velocity_world_mps: tuple[float, float, float]
    center_pixel_uv: tuple[float, float] | None
    visible: bool
    occlusion_ratio: float
    color_name: str | None = None
    instance_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.visible, bool):
            raise TypeError("target_label.visible must be bool")
        object.__setattr__(self, "position_world_m", _vector(self.position_world_m, "target_label.position_world_m", 3))
        object.__setattr__(self, "velocity_world_mps", _vector(self.velocity_world_mps, "target_label.velocity_world_mps", 3))
        center = None if self.center_pixel_uv is None else _vector(self.center_pixel_uv, "target_label.center_pixel_uv", 2)
        if self.visible and center is None:
            raise ValueError("visible target label requires center_pixel_uv")
        if not self.visible and center is not None:
            raise ValueError("invisible target label cannot carry center_pixel_uv")
        object.__setattr__(self, "center_pixel_uv", center)
        object.__setattr__(self, "occlusion_ratio", _probability(self.occlusion_ratio, "target_label.occlusion_ratio"))
        if self.color_name is not None:
            object.__setattr__(self, "color_name", _text(self.color_name, "target_label.color_name").casefold())
        if self.instance_id is not None:
            object.__setattr__(
                self,
                "instance_id",
                validate_routing_id(self.instance_id, "target_label.instance_id"),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "position_world_m": list(self.position_world_m),
            "velocity_world_mps": list(self.velocity_world_mps),
            "center_pixel_uv": None if self.center_pixel_uv is None else list(self.center_pixel_uv),
            "visible": self.visible,
            "occlusion_ratio": self.occlusion_ratio,
            "color_name": self.color_name,
            "instance_id": self.instance_id,
        }


@dataclass(frozen=True, slots=True)
class TargetStateFrameRecord:
    frame_id: str
    episode_id: str
    assignment_id: str
    uav_id: str
    timestamp_s: float
    sensor_input: SensorInput
    detector_prediction: DetectorPrediction
    training_label: TargetTrainingLabel | None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        object.__setattr__(self, "frame_id", validate_routing_id(self.frame_id, "frame_id"))
        object.__setattr__(self, "episode_id", validate_routing_id(self.episode_id, "episode_id"))
        object.__setattr__(self, "assignment_id", validate_routing_id(self.assignment_id, "assignment_id"))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(self, "timestamp_s", _nonnegative(self.timestamp_s, "timestamp_s"))
        if not isinstance(self.sensor_input, SensorInput):
            raise TypeError("sensor_input must be SensorInput")
        if not isinstance(self.detector_prediction, DetectorPrediction):
            raise TypeError("detector_prediction must be DetectorPrediction")
        if self.training_label is not None and not isinstance(
            self.training_label, TargetTrainingLabel
        ):
            raise TypeError("training_label must be TargetTrainingLabel or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frame_id": self.frame_id,
            "episode_id": self.episode_id,
            "assignment_id": self.assignment_id,
            "uav_id": self.uav_id,
            "timestamp_s": self.timestamp_s,
            "sensor_input": self.sensor_input.to_dict(),
            "detector_prediction": self.detector_prediction.to_dict(),
            "training_label": (
                None if self.training_label is None else self.training_label.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TargetStateFrameRecord":
        if not isinstance(payload, Mapping):
            raise TypeError("frame record must be a mapping")
        expected = {
            "schema_version", "frame_id", "episode_id", "assignment_id", "uav_id",
            "timestamp_s", "sensor_input", "detector_prediction", "training_label",
        }
        if set(payload) != expected:
            raise ValueError(f"frame fields mismatch: missing={sorted(expected-set(payload))}, unknown={sorted(set(payload)-expected)}")
        sensor = payload["sensor_input"]
        detector = payload["detector_prediction"]
        label = payload["training_label"]
        if not isinstance(sensor, Mapping) or not isinstance(detector, Mapping):
            raise TypeError("sensor_input and detector_prediction must be mappings")
        if label is not None and not isinstance(label, Mapping):
            raise TypeError("training_label must be a mapping or null")
        camera = sensor.get("camera")
        uav = sensor.get("uav")
        if not isinstance(camera, Mapping) or not isinstance(uav, Mapping):
            raise TypeError("sensor camera/uav must be mappings")
        intrinsics = camera.get("intrinsics")
        if not isinstance(intrinsics, Mapping):
            raise TypeError("camera.intrinsics must be a mapping")
        return cls(
            schema_version=payload["schema_version"],
            frame_id=payload["frame_id"],
            episode_id=payload["episode_id"],
            assignment_id=payload["assignment_id"],
            uav_id=payload["uav_id"],
            timestamp_s=payload["timestamp_s"],
            sensor_input=SensorInput(
                camera=CameraFrameInput(
                    fx=intrinsics.get("fx"), fy=intrinsics.get("fy"),
                    cx=intrinsics.get("cx"), cy=intrinsics.get("cy"),
                    position_world_m=camera.get("position_world_m"),
                    orientation_world_wxyz=camera.get("orientation_world_wxyz"),
                    resolution_wh_px=camera.get("resolution_wh_px"),
                ),
                uav=UavFrameInput(
                    position_world_m=uav.get("position_world_m"),
                    orientation_world_wxyz=uav.get("orientation_world_wxyz"),
                    linear_velocity_world_mps=uav.get("linear_velocity_world_mps"),
                    angular_velocity_body_radps=uav.get("angular_velocity_body_radps"),
                ),
                rgb_path=sensor.get("rgb_path"),
                depth_path=sensor.get("depth_path"),
                instance_mask_path=sensor.get("instance_mask_path"),
            ),
            detector_prediction=DetectorPrediction(
                detected=detector.get("detected"),
                bbox_xyxy_normalized=detector.get("bbox_xyxy_normalized"),
                confidence=detector.get("confidence"),
                tracker_id=detector.get("tracker_id"),
                candidate_id=detector.get("candidate_id"),
            ),
            training_label=(
                None
                if label is None
                else TargetTrainingLabel(
                    position_world_m=label.get("position_world_m"),
                    velocity_world_mps=label.get("velocity_world_mps"),
                    center_pixel_uv=label.get("center_pixel_uv"),
                    visible=label.get("visible"),
                    occlusion_ratio=label.get("occlusion_ratio"),
                    color_name=label.get("color_name"),
                    instance_id=label.get("instance_id"),
                )
            ),
        )
