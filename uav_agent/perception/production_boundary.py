"""Pure-Python production information-boundary checks and audit metadata."""

from __future__ import annotations

from collections.abc import Mapping

from common.ids import validate_uav_id
from perception.runtime import PerceptionRuntimeProfile
from perception.target_query import ALLOWED_TARGET_QUERY_FIELDS


_PRODUCTION_ESTIMATE_SOURCES = frozenset(
    {
        "ultralytics_service",
        "yolo26_botsort",
        "yoloe26_botsort",
        "rgbd_depth_geometry",
        "temporal_ray_depth",
        "kalman_prediction",
    }
)


def validate_production_estimate_source(source: object) -> str:
    """Fail closed on an unknown or privileged production provenance label."""

    if not isinstance(source, str) or not source.strip():
        raise PermissionError("production TargetEstimate.source must be non-empty")
    normalized = source.strip()
    if normalized not in _PRODUCTION_ESTIMATE_SOURCES:
        raise PermissionError(
            "production TargetEstimate.source is not in the visual-chain allowlist: "
            f"{normalized!r}"
        )
    return normalized


def build_target_perception_startup_audit(
    *,
    runtime_profile: PerceptionRuntimeProfile | str,
    target_perception_mode: str,
    backend_by_uav: Mapping[str, str],
    privileged: bool,
    yolo_model_sha256_by_uav: Mapping[str, str],
    temporal_model_sha256: str | None = None,
) -> dict[str, object]:
    """Build one bounded startup record without target/query contents."""

    profile = (
        runtime_profile.value
        if isinstance(runtime_profile, PerceptionRuntimeProfile)
        else str(runtime_profile).strip().upper()
    )
    mode = str(target_perception_mode).strip().lower()
    if profile != PerceptionRuntimeProfile.PRODUCTION.value:
        raise PermissionError("production audit requires runtime_profile=PRODUCTION")
    if mode != "yolo":
        raise PermissionError("production audit requires target_perception_mode=yolo")
    if privileged is not False:
        raise PermissionError("production audit requires privileged=false")
    if not isinstance(backend_by_uav, Mapping) or not backend_by_uav:
        raise ValueError("backend_by_uav must be a non-empty mapping")
    backends = {
        validate_uav_id(raw_uav): str(raw_backend).strip()
        for raw_uav, raw_backend in backend_by_uav.items()
    }
    if any(value != "ultralytics_service" for value in backends.values()):
        raise PermissionError(
            "production YOLO audit requires ultralytics_service for every UAV"
        )
    if not isinstance(yolo_model_sha256_by_uav, Mapping):
        raise TypeError("yolo_model_sha256_by_uav must be a mapping")
    hashes: dict[str, str] = {}
    for raw_uav, raw_sha in yolo_model_sha256_by_uav.items():
        uav_id = validate_uav_id(raw_uav)
        if not isinstance(raw_sha, str):
            raise TypeError("YOLO model SHA256 values must be strings")
        sha = raw_sha.strip().casefold()
        if len(sha) != 64 or any(
            character not in "0123456789abcdef" for character in sha
        ):
            raise ValueError("YOLO model SHA256 must contain exactly 64 hex characters")
        hashes[uav_id] = sha
    if set(hashes) != set(backends):
        raise ValueError("YOLO model SHA256 coverage must match backend_by_uav")
    if temporal_model_sha256 is not None:
        temporal = temporal_model_sha256.strip().casefold()
        if len(temporal) != 64 or any(
            character not in "0123456789abcdef" for character in temporal
        ):
            raise ValueError(
                "temporal_model_sha256 must contain exactly 64 hex characters"
            )
    else:
        temporal = None
    unique_hashes = tuple(sorted(set(hashes.values())))
    return {
        "runtime_profile": profile.lower(),
        "target_perception_mode": mode,
        "backend_by_uav": dict(sorted(backends.items())),
        "privileged": False,
        "allowed_target_query_fields": list(ALLOWED_TARGET_QUERY_FIELDS),
        "yolo_model_sha256": (
            unique_hashes[0]
            if len(unique_hashes) == 1
            else dict(sorted(hashes.items()))
        ),
        "temporal_model_sha256": temporal,
    }


__all__ = [
    "build_target_perception_startup_audit",
    "validate_production_estimate_source",
]
