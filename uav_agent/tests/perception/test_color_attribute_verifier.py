from __future__ import annotations

import numpy as np
import pytest

from configs.schema import TargetColorAttributeConfig
from env.camera_types import CameraIntrinsics, CameraSample
from perception.attribute_types import (
    AttributeDecision,
    AttributeEvidence,
    AttributeObservation,
    AttributeRequirement,
)
from perception.attribute_verifier import (
    AttributeRouteMismatch,
    AttributeVerificationRoute,
)
from perception.color_attribute_verifier import RgbdColorAttributeVerifier
from perception.runtime import PerceptionRuntimeProfile
from runtime.frame_store import FrameStore
from yolo_service.protocol import TrackDetection


def _requirement(
    *, tracker_id: int = 7, expected: str = "red"
) -> AttributeRequirement:
    return AttributeRequirement(
        mission_id="mission_a",
        uav_id="uav_a",
        assignment_id="assignment_a",
        candidate_id="candidate_a",
        tracker_id=tracker_id,
        attribute_name="color",
        expected_value=expected,
    )


def _detection(
    *, tracker_id: int = 7, bbox: tuple[float, float, float, float] = (0.1, 0.1, 0.9, 0.9)
) -> TrackDetection:
    return TrackDetection(
        track_id=tracker_id,
        class_id=0,
        class_name="cube",
        confidence=0.9,
        bbox_xyxy_normalized=bbox,
    )


def _route(*, tracker_id: int = 7) -> AttributeVerificationRoute:
    return AttributeVerificationRoute(
        mission_id="mission_a",
        uav_id="uav_a",
        assignment_id="assignment_a",
        candidate_id="candidate_a",
        tracker_id=tracker_id,
    )


def _sample(
    color: tuple[int, int, int],
    *,
    timestamp_s: float = 1.0,
    depth: bool = True,
    size: int = 20,
) -> CameraSample:
    rgb = np.empty((size, size, 3), dtype=np.uint8)
    rgb[:] = color
    depth_plane = (
        np.full((size, size), 10.0, dtype=np.float32) if depth else None
    )
    return CameraSample(
        timestamp_s=timestamp_s,
        rgb=rgb,
        depth_to_image_plane_m=depth_plane,
        camera_position_world_m=(0.0, 0.0, 0.0),
        camera_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsics=CameraIntrinsics(
            fx=10.0,
            fy=10.0,
            cx=size / 2,
            cy=size / 2,
            width=size,
            height=size,
        ),
    )


@pytest.mark.parametrize(
    ("rgb", "expected", "decision", "observed"),
    [
        ((255, 0, 0), "red", AttributeDecision.MATCH, "red"),
        # Hue near 360 exercises red wrap-around.
        ((255, 0, 12), "red", AttributeDecision.MATCH, "red"),
        ((0, 0, 255), "red", AttributeDecision.MISMATCH, "blue"),
        ((0, 0, 255), "blue", AttributeDecision.MATCH, "blue"),
    ],
)
def test_rgbd_color_classification(
    rgb: tuple[int, int, int],
    expected: str,
    decision: AttributeDecision,
    observed: str,
) -> None:
    result = RgbdColorAttributeVerifier().verify(
        requirement=_requirement(expected=expected),
        detection=_detection(),
        route=_route(),
        camera_sample=_sample(rgb),
    )
    assert result.decision is decision
    assert result.observed_value == observed
    assert result.source == "hsv_depth_mask"
    assert result.confidence == pytest.approx(1.0)
    assert result.valid_sample_ratio == pytest.approx(1.0)


@pytest.mark.parametrize("rgb", [(128, 128, 128), (255, 255, 255), (0, 0, 0)])
def test_achromatic_and_dark_pixels_are_unknown_not_negative(
    rgb: tuple[int, int, int],
) -> None:
    result = RgbdColorAttributeVerifier().verify(
        requirement=_requirement(),
        detection=_detection(),
        route=_route(),
        camera_sample=_sample(rgb),
    )
    assert result.decision is AttributeDecision.PENDING
    assert result.observed_value == "unknown"
    assert result.reason_code == "achromatic_or_dark"


def test_missing_depth_and_small_bbox_remain_pending() -> None:
    verifier = RgbdColorAttributeVerifier()
    no_depth = verifier.verify(
        requirement=_requirement(),
        detection=_detection(),
        route=_route(),
        camera_sample=_sample((255, 0, 0), depth=False),
    )
    tiny = verifier.verify(
        requirement=_requirement(),
        detection=_detection(bbox=(0.4, 0.4, 0.5, 0.5)),
        route=_route(),
        camera_sample=_sample((255, 0, 0)),
    )
    assert no_depth.decision is AttributeDecision.PENDING
    assert no_depth.reason_code == "depth_unavailable"
    assert tiny.decision is AttributeDecision.PENDING
    assert tiny.reason_code == "bbox_too_small"


def test_depth_mask_removes_different_depth_background() -> None:
    sample = _sample((0, 255, 0), size=30)
    rgb = sample.rgb.copy()
    depth = sample.depth_to_image_plane_m.copy()  # type: ignore[union-attr]
    rgb[10:20, 10:20] = (255, 0, 0)
    depth[:] = 20.0
    depth[10:20, 10:20] = 10.0
    masked_sample = CameraSample(
        timestamp_s=sample.timestamp_s,
        rgb=rgb,
        depth_to_image_plane_m=depth,
        camera_position_world_m=sample.camera_position_world_m,
        camera_orientation_world_wxyz=sample.camera_orientation_world_wxyz,
        intrinsics=sample.intrinsics,
    )
    result = RgbdColorAttributeVerifier(
        roi_inset_ratio=0.0,
        min_valid_pixel_ratio=0.1,
    ).verify(
        requirement=_requirement(),
        detection=_detection(bbox=(0.1, 0.1, 0.9, 0.9)),
        route=_route(),
        camera_sample=masked_sample,
    )
    assert result.decision is AttributeDecision.MATCH
    assert result.observed_value == "red"
    assert result.valid_sample_ratio < 0.25


def test_frame_ref_reads_both_channels_from_one_store_entry_and_checks_route() -> None:
    store = FrameStore()
    ref = store.add_sample(
        uav_id="uav_a",
        frame_id="frame_a",
        sample=_sample((255, 0, 0)),
    )
    result = RgbdColorAttributeVerifier(store).verify(
        requirement=_requirement(),
        detection=_detection(),
        route=_route(),
        frame_ref=ref,
    )
    assert result.decision is AttributeDecision.MATCH

    other_ref = FrameStore().add_sample(
        uav_id="uav_b",
        frame_id="frame_b",
        sample=_sample((255, 0, 0)),
    )
    with pytest.raises(AttributeRouteMismatch):
        RgbdColorAttributeVerifier(FrameStore()).verify(
            requirement=_requirement(),
            detection=_detection(),
            route=_route(),
            frame_ref=other_ref,
        )


def test_tracker_route_mismatch_fails_closed() -> None:
    with pytest.raises(AttributeRouteMismatch):
        RgbdColorAttributeVerifier().verify(
            requirement=_requirement(tracker_id=7),
            detection=_detection(tracker_id=8),
            route=_route(tracker_id=7),
            camera_sample=_sample((255, 0, 0)),
        )


def test_verifiers_build_from_public_color_config() -> None:
    config = TargetColorAttributeConfig()
    verifier = RgbdColorAttributeVerifier.from_config(config)
    assert verifier.supported_values == ("red", "blue")


def test_overlapping_hue_ranges_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        RgbdColorAttributeVerifier(
            hue_ranges_deg={
                "red": ((0.0, 30.0),),
                "blue": ((20.0, 50.0),),
            }
        )


def test_attribute_protocol_is_scalar_only_strict_and_rejects_production_oracle() -> None:
    observation = RgbdColorAttributeVerifier().verify(
        requirement=_requirement(),
        detection=_detection(),
        route=_route(),
        camera_sample=_sample((255, 0, 0)),
    )
    payload = observation.to_dict()
    assert payload["schema_version"] == 1
    assert not ({"rgb", "depth", "crop", "base64", "path"} & payload.keys())
    assert AttributeObservation.from_dict(payload) == observation
    with pytest.raises(ValueError, match="unknown fields"):
        AttributeObservation.from_dict({**payload, "rgb": "forbidden"})
    with pytest.raises(ValueError, match="Oracle"):
        AttributeEvidence(
            mission_id="mission_a",
            uav_id="uav_a",
            assignment_id="assignment_a",
            candidate_id="candidate_a",
            tracker_id=7,
            timestamp_s=1.0,
            attribute_name="color",
            expected_value="red",
            observed_value="red",
            decision=AttributeDecision.MATCH,
            confidence=0.9,
            observation_count=3,
            duration_s=0.5,
            valid_sample_ratio=0.8,
            source="oracle_color",
            reason_code="test",
            runtime_profile=PerceptionRuntimeProfile.PRODUCTION,
        )
    bad_unknown = payload.copy()
    bad_unknown["observed_value"] = "unknown"
    bad_unknown["decision"] = "MISMATCH"
    with pytest.raises(ValueError, match="unknown cannot"):
        AttributeObservation.from_dict(bad_unknown)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_attribute_protocol_rejects_nonfinite_values(bad: float) -> None:
    payload = RgbdColorAttributeVerifier().verify(
        requirement=_requirement(),
        detection=_detection(),
        route=_route(),
        camera_sample=_sample((255, 0, 0)),
    ).to_dict()
    payload["confidence"] = bad
    with pytest.raises(ValueError, match="finite"):
        AttributeObservation.from_dict(payload)
