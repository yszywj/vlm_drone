from types import SimpleNamespace

import pytest
import torch

from perception.ray_measurement_gate import ray_measurement_gate
from perception.temporal_ray_depth import TemporalRayDepthResolver, TemporalMeasurementRejected
from tests.perception.test_temporal_ray_depth import _artifact, _frames
from runtime.frame_store import FrameStore


def gate_arguments():
    return dict(reference_detected=True, raw_depth_m=4.0, anchor_uv_px=(16.0, 12.0),
                corrected_uv_px=(16.0, 12.0), corrected_depth_m=4.2,
                image_size_wh=(32.0, 24.0), validity_probability=0.99,
                minimum_depth_m=0.2, maximum_depth_m=200.0)


@pytest.mark.parametrize("updates,accepted", [
    ({}, True), ({"reference_detected": False}, False),
    ({"raw_depth_m": 0.0, "corrected_depth_m": 0.47}, False),
    ({"raw_depth_m": float("nan")}, False),
    ({"raw_depth_m": 0.19}, False), ({"corrected_depth_m": 0.19}, False),
    ({"corrected_depth_m": 201.0}, False),
    ({"corrected_uv_px": (32.0, 12.0)}, False),
    ({"corrected_uv_px": (16.0, -1.0)}, False),
    ({"validity_probability": 0.49}, False),
    ({"validity_probability": float("nan")}, False),
    ({"anchor_uv_px": (0.0, 0.0), "corrected_uv_px": (0.0, 0.0)}, True),
])
def test_scalar_and_tensor_gate_are_identical(updates, accepted):
    arguments = gate_arguments() | updates
    scalar = ray_measurement_gate(**arguments)
    assert bool(scalar.accepted) is accepted
    tensor_args = {key: (tuple(torch.tensor([v]) for v in value)
                         if isinstance(value, tuple) else torch.tensor([value]))
                   for key, value in arguments.items()}
    tensor = ray_measurement_gate(**tensor_args, finite=torch.isfinite)
    assert tuple(bool(value.item()) for value in tensor) == tuple(bool(value) for value in scalar)


def resolver_fixture(tmp_path):
    checkpoint, digest = _artifact(tmp_path)
    store = FrameStore(max_frames=16, max_bytes=2_000_000, max_age_s=10.0)
    resolver = TemporalRayDepthResolver(store, checkpoint_path=checkpoint,
        expected_sha256=digest, history_size=4, max_history_age_s=2.0,
        roi_size_px=32, deterministic_fallback=True)
    resolver.reset(uav_id="uav_1", assignment_id="assignment_gate")
    _, candidate = _frames(5, store=store)
    return resolver, candidate


def test_explicit_network_rejection_cannot_be_undone_by_fallback(tmp_path):
    resolver, candidate = resolver_fixture(tmp_path)
    with torch.no_grad():
        resolver._model.validity_head.bias.fill_(-3.0)
    with pytest.raises(TemporalMeasurementRejected, match="invalid_probability"):
        resolver.resolve(candidate, timestamp_s=0.8)
    assert resolver.statistics.fallback_total == 0
    assert resolver.statistics.successes == 0


@pytest.mark.parametrize("invalid_reference", ["missing", "zero_depth"])
def test_reference_failure_is_rejected_before_inference(tmp_path, monkeypatch, invalid_reference):
    resolver, candidate = resolver_fixture(tmp_path)
    if invalid_reference == "missing":
        roi, features, missing = resolver._build_model_inputs(candidate)
        missing[0, -1] = True
        monkeypatch.setattr(resolver, "_build_model_inputs", lambda _: (roi, features, missing))
    else:
        monkeypatch.setattr(resolver._fallback, "resolve", lambda *args, **kwargs:
            SimpleNamespace(raw_depth_m=0.0, corrected_depth_m=0.0, pixel_uv=(0.0, 0.0)))
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid reference must not invoke the network")
    monkeypatch.setattr(resolver._model, "forward", forbidden)
    with pytest.raises(TemporalMeasurementRejected, match="reference_measurement_unavailable"):
        resolver.resolve(candidate, timestamp_s=0.8)
    assert resolver.statistics.fallback_total == 0


def test_out_of_image_correction_cannot_fallback(tmp_path):
    resolver, candidate = resolver_fixture(tmp_path)
    with torch.no_grad():
        resolver._model.delta_uv_head.bias.fill_(100.0)
    with pytest.raises(TemporalMeasurementRejected, match="geometry_out_of_bounds"):
        resolver.resolve(candidate, timestamp_s=0.8)
    assert resolver.statistics.fallback_total == 0
