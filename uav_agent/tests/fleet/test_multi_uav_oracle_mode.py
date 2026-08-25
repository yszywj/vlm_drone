from __future__ import annotations

from pathlib import Path

import pytest

from configs.loader import load_config
from perception.factory import build_target_perception_runtime
from perception.mode import resolve_target_perception_mode
from perception.oracle import OraclePerception
from perception.runtime_provider import OracleTargetPerceptionRuntime
from target import TargetSpec


ROOT = Path(__file__).resolve().parents[2]


class _AssignmentEnvironment:
    def __init__(self) -> None:
        self.assignments = {
            "uav_a": "target_i",
            "uav_b": "target_j",
        }
        self.created: list[tuple[str, str]] = []
        self.evaluator_calls: list[tuple[str, str]] = []

    def make_oracle_perception(self, uav_id: str) -> OraclePerception:
        target_id = self.assignments[uav_id]
        self.created.append((uav_id, target_id))
        return OraclePerception(uav_id=uav_id, target_id=target_id)

    def get_evaluator_frame(self, uav_id: str, target_id: str) -> object:
        self.evaluator_calls.append((uav_id, target_id))
        if self.assignments.get(uav_id) != target_id:
            raise PermissionError("evaluator target is outside Assignment")
        raise RuntimeError("no Camera frame was published in this pure factory test")


def test_two_oracle_runtimes_are_distinct_and_assignment_scoped() -> None:
    config = load_config(ROOT / "configs/multi_uav_oracle.yaml")
    mode = resolve_target_perception_mode(
        "oracle",
        acknowledge_privileged_oracle=True,
    )
    environment = _AssignmentEnvironment()
    runtime_a = build_target_perception_runtime(
        config,
        resolved_mode=mode,
        environment=environment,
        uav_id="uav_a",
    )
    runtime_b = build_target_perception_runtime(
        config,
        resolved_mode=mode,
        environment=environment,
        uav_id="uav_b",
    )
    assert isinstance(runtime_a, OracleTargetPerceptionRuntime)
    assert isinstance(runtime_b, OracleTargetPerceptionRuntime)
    assert runtime_a is not runtime_b
    assert environment.created == [
        ("uav_a", "target_i"),
        ("uav_b", "target_j"),
    ]

    spec_a = TargetSpec("red cube", category="cube")
    spec_b = TargetSpec("blue cube", category="cube")
    runtime_a.reset(
        mission_id="fleet_mission_1",
        assignment_id="assignment_a",
        uav_id="uav_a",
        target_alias="target_i",
        target_spec=spec_a,
    )
    runtime_b.reset(
        mission_id="fleet_mission_1",
        assignment_id="assignment_b",
        uav_id="uav_b",
        target_alias="target_j",
        target_spec=spec_b,
    )
    assert runtime_a.target_id == "target_i"
    assert runtime_b.target_id == "target_j"

    with pytest.raises(PermissionError, match="rebound"):
        runtime_a.reset(
            mission_id="fleet_mission_1",
            assignment_id="assignment_wrong",
            uav_id="uav_a",
            target_alias="target_j",
            target_spec=spec_b,
        )
    with pytest.raises(ValueError, match="does not match"):
        runtime_b.reset(
            mission_id="fleet_mission_1",
            assignment_id="assignment_wrong_uav",
            uav_id="uav_a",
            target_alias="target_j",
            target_spec=spec_b,
        )
    assert environment.evaluator_calls == []

    runtime_a.close()
    runtime_b.reset(
        mission_id="fleet_mission_1",
        assignment_id="assignment_b_repair",
        uav_id="uav_b",
        target_alias="target_j",
        target_spec=spec_b,
    )
    assert runtime_b.target_id == "target_j"
    runtime_b.close()

