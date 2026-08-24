from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

from common.target_estimate import TargetEstimate
from perception.evaluation import (
    TargetEvaluationError,
    TargetEvaluationMode,
    TargetEstimateEvaluator,
    TargetGroundTruth,
)
from perception.target_perception_coordinator import TargetPerceptionMetrics
from scripts.run_dynamic_visual_mission import _run_until_terminal


def _estimate(
    *,
    timestamp_s: float = 1.0,
    source: str = "yolo26_botsort",
    position: tuple[float, float, float] | None = (4.0, 6.0, 3.0),
    velocity: tuple[float, float, float] | None = (2.0, 0.0, 0.0),
) -> TargetEstimate:
    return TargetEstimate(
        timestamp_s=timestamp_s,
        target_id="target_1",
        candidate_id="candidate_1",
        tracker_id="track_1",
        visible=True,
        confirmed=True,
        predicted_only=False,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox_xyxy_normalized=(0.1, 0.2, 0.4, 0.8),
        position_world_m=position,
        velocity_world_mps=velocity,
        measurement_age_s=0.0,
        source=source,
    )


def _truth(timestamp_s: float, *, x: float = 1.0) -> TargetGroundTruth:
    return TargetGroundTruth(
        timestamp_s=timestamp_s,
        position_world_m=(x, 2.0, 3.0),
        velocity_world_mps=(1.0, 0.0, 0.0),
    )


def _evaluator(metrics: TargetPerceptionMetrics) -> TargetEstimateEvaluator:
    return TargetEstimateEvaluator(
        metrics,
        mode=TargetEvaluationMode.ORACLE_GROUND_TRUTH,
        allowed_estimate_sources=frozenset(
            {"yolo26_botsort", "yoloe26_botsort", "kalman_prediction"}
        ),
    )


def test_evaluator_is_write_only_and_records_rmse() -> None:
    metrics = TargetPerceptionMetrics()
    evaluator = _evaluator(metrics)

    result = evaluator.evaluate(_estimate(), _truth(1.0))

    assert result is None
    assert evaluator.matched_samples == 1
    values = metrics.to_dict()
    assert values["position_rmse_m"] == 5.0
    assert values["velocity_rmse_mps"] == 1.0
    # The evaluator label is not an Observation and has no target-estimate or
    # camera fields that a Skill could consume.
    assert not hasattr(_truth(1.0), "target_estimate")
    assert not hasattr(_truth(1.0), "camera_rgb")


def test_delayed_estimate_uses_matching_historical_truth_and_is_deduplicated() -> None:
    metrics = TargetPerceptionMetrics()
    evaluator = _evaluator(metrics)
    evaluator.evaluate(None, _truth(1.0, x=1.0))
    delayed = _estimate(timestamp_s=1.0, position=(4.0, 2.0, 3.0))

    evaluator.evaluate(delayed, _truth(2.0, x=100.0))
    evaluator.evaluate(delayed, _truth(3.0, x=200.0))

    assert evaluator.matched_samples == 1
    assert metrics.to_dict()["position_rmse_m"] == 3.0


def test_position_and_velocity_samples_are_independently_optional() -> None:
    metrics = TargetPerceptionMetrics()
    evaluator = _evaluator(metrics)
    evaluator.evaluate(
        _estimate(position=(2.0, 2.0, 3.0), velocity=None),
        _truth(1.0),
    )
    evaluator.evaluate(
        _estimate(timestamp_s=2.0, position=None, velocity=(3.0, 0.0, 0.0)),
        _truth(2.0),
    )

    values = metrics.to_dict()
    assert values["position_rmse_m"] == 1.0
    assert values["velocity_rmse_mps"] == 2.0


def test_provenance_and_evaluator_authority_fail_closed() -> None:
    metrics = TargetPerceptionMetrics()
    with pytest.raises(PermissionError, match="explicit oracle-ground-truth"):
        TargetEstimateEvaluator(
            metrics,
            mode="oracle_ground_truth",  # type: ignore[arg-type]
            allowed_estimate_sources=frozenset({"yolo26_botsort"}),
        )

    evaluator = _evaluator(metrics)
    with pytest.raises(PermissionError, match="not authorized"):
        evaluator.evaluate(
            _estimate(source="oracle_evaluation"),
            _truth(1.0),
        )
    with pytest.raises(PermissionError, match="evaluator-only"):
        metrics.record_evaluator_error(
            position_error_m=1.0,
            velocity_error_mps=1.0,
            evaluator_mode=False,
        )


def test_ground_truth_rejects_simulator_arrays_and_spoofed_provenance() -> None:
    with pytest.raises(TargetEvaluationError, match="three-number tuple"):
        TargetGroundTruth(
            timestamp_s=1.0,
            position_world_m=np.asarray((1.0, 2.0, 3.0)),  # type: ignore[arg-type]
            velocity_world_mps=(0.0, 0.0, 0.0),
        )
    with pytest.raises(TargetEvaluationError, match="provenance"):
        TargetGroundTruth(
            timestamp_s=1.0,
            position_world_m=(1.0, 2.0, 3.0),
            velocity_world_mps=(0.0, 0.0, 0.0),
            provenance="oracle_target_pose",
        )


def test_rmse_accumulates_vector_error_magnitudes() -> None:
    metrics = TargetPerceptionMetrics()
    evaluator = _evaluator(metrics)
    evaluator.evaluate(
        _estimate(timestamp_s=1.0, position=(4.0, 2.0, 3.0)),
        _truth(1.0),
    )
    evaluator.evaluate(
        _estimate(timestamp_s=2.0, position=(5.0, 2.0, 3.0)),
        _truth(2.0),
    )
    assert metrics.to_dict()["position_rmse_m"] == pytest.approx(
        sqrt((3.0**2 + 4.0**2) / 2.0)
    )


def test_main_loop_wires_evaluator_as_side_channel_not_agent_input() -> None:
    from types import SimpleNamespace

    running = SimpleNamespace(
        status=SimpleNamespace(value="RUNNING"),
        active_skill="SEARCH",
    )
    terminal = SimpleNamespace(
        status=SimpleNamespace(value="SUCCEEDED"),
        active_skill=None,
    )
    estimate = _estimate()
    observation = SimpleNamespace(
        timestamp=1.0,
        uav_pose=SimpleNamespace(x=0.0, y=0.0, z=1.0),
        uav_velocity=(0.0, 0.0, 0.0),
        target_estimate=estimate,
    )
    frame = SimpleNamespace(
        observation=SimpleNamespace(camera_timestamp_s=1.0),
        target_position_m=np.asarray((1.0, 2.0, 3.0)),
        target_velocity_mps=np.asarray((1.0, 0.0, 0.0)),
    )

    class Agent:
        def __init__(self) -> None:
            self.received = None

        def snapshot(self) -> object:
            return running

        def tick(self, value: object) -> object:
            self.received = value
            return terminal

    class WriteOnlyEvaluator:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def evaluate(self, value: object, truth: object) -> None:
            self.calls.append((value, truth))
            return None

    agent = Agent()
    evaluator = WriteOnlyEvaluator()
    result = _run_until_terminal(
        simulation_app=SimpleNamespace(is_running=lambda: True),
        debug_visualizer=None,
        environment=SimpleNamespace(
            step=lambda: True,
            get_evaluator_frame=lambda: frame,
        ),
        perception=SimpleNamespace(observe=lambda value: observation),
        agent=agent,
        manager=SimpleNamespace(
            active_planned_step_id=None,
            active_invocation=None,
            task_plan=None,
            transition_log=(),
        ),
        clock=SimpleNamespace(now=lambda: 1.0),
        task_start_s=10.0,
        max_sim_time_s=20.0,
        shutdown_guard_s=10.0,
        debug_ground_truth=True,
        injections=(),
        visual_runtime=None,
        target_estimate_evaluator=evaluator,
    )

    assert result.snapshot is terminal
    assert agent.received is observation
    assert evaluator.calls[0][0] is estimate
    assert isinstance(evaluator.calls[0][1], TargetGroundTruth)
    assert evaluator.calls[0][1] is not agent.received


def test_production_default_does_not_read_evaluator_ground_truth() -> None:
    from types import SimpleNamespace

    running = SimpleNamespace(
        status=SimpleNamespace(value="RUNNING"),
        active_skill="SEARCH",
    )
    terminal = SimpleNamespace(
        status=SimpleNamespace(value="SUCCEEDED"),
        active_skill=None,
    )
    observation = SimpleNamespace(
        timestamp=1.0,
        uav_pose=SimpleNamespace(x=0.0, y=0.0, z=1.0),
        uav_velocity=(0.0, 0.0, 0.0),
        target_estimate=None,
    )

    class Agent:
        def snapshot(self) -> object:
            return running

        def tick(self, value: object) -> object:
            assert value is observation
            return terminal

    def forbidden_evaluator_read() -> object:
        raise AssertionError("production default accessed evaluator truth")

    result = _run_until_terminal(
        simulation_app=SimpleNamespace(is_running=lambda: True),
        debug_visualizer=None,
        environment=SimpleNamespace(
            step=lambda: True,
            get_skill_observation=lambda **kwargs: observation,
            get_evaluator_frame=forbidden_evaluator_read,
        ),
        perception=SimpleNamespace(observe=lambda value: value),
        agent=Agent(),
        manager=SimpleNamespace(
            active_planned_step_id=None,
            active_invocation=None,
            task_plan=None,
            transition_log=(),
        ),
        clock=SimpleNamespace(now=lambda: 1.0),
        task_start_s=0.0,
        max_sim_time_s=20.0,
        shutdown_guard_s=10.0,
        debug_ground_truth=False,
        injections=(),
        visual_runtime=None,
        production_target_perception=True,
    )

    assert result.snapshot is terminal
