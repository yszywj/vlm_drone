from __future__ import annotations

from types import SimpleNamespace

from perception.target_query import TargetQuerySpec
from perception.runtime import PerceptionRuntimeProfile
from target import TargetSpec
from tests.test_mission_agent import make_harness


def _project(hidden_truth: object) -> TargetQuerySpec:
    # hidden_truth deliberately exists in the caller's scene/evaluator scope;
    # the production projection has no parameter or field through which to
    # consume it.
    del hidden_truth
    return TargetQuerySpec.from_assignment_semantics(
        target_alias="target_red",
        target_spec=TargetSpec(
            "moving red cube",
            category="cube",
            hard_attributes=("color=red",),
        ),
        detector_class_id=0,
        detector_class_name="cube",
    )


def test_changing_only_hidden_truth_cannot_change_production_query() -> None:
    truth_a = SimpleNamespace(
        position_world_m=(1.0, 2.0, 0.5),
        velocity_world_mps=(1.0, 0.0, 0.0),
        motion_seed=7,
        prim_path="/World/Target_A",
    )
    truth_b = SimpleNamespace(
        position_world_m=(999.0, -500.0, 30.0),
        velocity_world_mps=(-10.0, 4.0, 0.0),
        motion_seed=9999,
        prim_path="/World/SecretTarget",
    )
    assert _project(truth_a) == _project(truth_b)


def test_query_schema_cannot_carry_planner_or_skill_truth_geometry() -> None:
    query = _project(object())
    public_names = set(query.__slots__)
    assert public_names == {
        "target_alias",
        "detector_class_id",
        "detector_class_name",
        "hard_attributes",
        "soft_description",
    }
    assert public_names.isdisjoint(
        {
            "position_world_m",
            "velocity_world_mps",
            "motion_seed",
            "prim_path",
            "initial_region",
        }
    )


def test_agent_output_is_invariant_to_an_external_hidden_truth_object() -> None:
    first = make_harness(
        perception_runtime_profile=PerceptionRuntimeProfile.PRODUCTION,
        acknowledge_privileged_oracle=False,
    )
    second = make_harness(
        perception_runtime_profile=PerceptionRuntimeProfile.PRODUCTION,
        acknowledge_privileged_oracle=False,
    )
    first.start()
    second.start()
    hidden_a = SimpleNamespace(
        position_world_m=(1.0, 2.0, 0.5),
        motion_seed=7,
        prim_path="/World/Target_A",
    )
    hidden_b = SimpleNamespace(
        position_world_m=(900.0, -800.0, 30.0),
        motion_seed=999,
        prim_path="/World/Target_B",
    )
    # The only legal runtime inputs are kept identical.  Merely changing an
    # evaluator-side object cannot alter the Agent snapshot.
    del hidden_a, hidden_b
    snapshot_a = first.tick(1.0)
    snapshot_b = second.tick(1.0)
    assert snapshot_a.status == snapshot_b.status
    assert snapshot_a.task_status == snapshot_b.task_status
    assert snapshot_a.active_skill == snapshot_b.active_skill
    assert snapshot_a.target == snapshot_b.target
    assert snapshot_a.feedback == snapshot_b.feedback
