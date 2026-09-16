"""Configuration, role boundaries and lifecycle checks for opt-in Fleet recovery."""
from dataclasses import replace
import json
from types import SimpleNamespace
import pytest

from configs.loader import load_config
from configs.schema import FleetRecoveryConfig
from fleet.llm_planner_v2 import LLMFleetPlannerV2
from models.runtime_deadline import DeadlineModelClient
from scripts.run_fleet_mission import (
    FleetLaunchConfigurationError, _build_fleet_recovery_controller,
    parse_args, prepare_fleet_mission,
)
from tests.fleet.test_fleet_planner_v2 import _QueuedClient, _payload, _request
from tests.fleet.test_model_request_dispatcher import _dispatcher


def test_old_configuration_defaults_off_and_new_demo_changes_only_recovery():
    old = load_config("configs/multi_uav_demo.yaml")
    new = load_config("configs/multi_uav_local_repair_demo.yaml")
    assert not old.fleet_recovery.enabled
    assert new.fleet_recovery.enabled
    assert new.uavs == old.uavs and new.planner == old.planner
    assert new.target_perception == old.target_perception
    assert _build_fleet_recovery_controller(SimpleNamespace(config=old),
        broker=None, geometry_provider=None, replan_boundary=None) is None


@pytest.mark.parametrize("flags, expected", [
    (["--runtime-program", "graph", "--fleet-planner", "llm"], "linear Spatial V3"),
    (["--fleet-planner", "scripted"], "LLM Fleet/Spatial V3"),
])
def test_unsupported_enabled_paths_fail_before_model_or_simulator(flags, expected):
    args = parse_args(["--config", "configs/multi_uav_local_repair_demo.yaml",
        "--instruction", "visit assigned point", *flags])
    with pytest.raises(FleetLaunchConfigurationError, match=expected):
        prepare_fleet_mission(args)


@pytest.mark.parametrize("values", [
    {"enabled":"true"}, {"mode":"GRAPH"}, {"request_timeout_s":float("nan")},
    {"retry_cooldown_s":0}, {"max_local_attempts":True},
    {"max_reassign_attempts":9}, {"max_suffix_steps":11},
    {"request_timeout_s":100, "episode_timeout_s":90},
])
def test_recovery_config_rejects_invalid_or_unsupported_budgets(values):
    with pytest.raises((ValueError, TypeError)):
        FleetRecoveryConfig(**values)


def test_expired_model_pipeline_cannot_start_another_http_call():
    clock = SimpleNamespace(now=10.0)
    calls = []
    class Client:
        def chat(self, messages, *, options=None):
            calls.append(messages)
            clock.now = 20.0  # real HTTP finishes after owner-revocable deadline
            return object()
    client = DeadlineModelClient(Client(), 15.0, clock=lambda:clock.now)
    with pytest.raises(TimeoutError, match="during HTTP"):
        client.chat(("first",))
    with pytest.raises(TimeoutError, match="before HTTP"):
        client.chat(("retry",))
    assert calls == [("first",)]


def test_single_assignment_recovery_restricts_both_prompt_and_output_schema():
    request = _request()
    client = _QueuedClient([json.dumps(_payload(request))])
    planner = LLMFleetPlannerV2(client, maximum_assignments=1, repair_budget=0)
    planner.external_dependencies = ({"dependency_id":"order_ab","state":"CONFIRMED","version":1},)
    planner.plan(request)
    messages, options = client.calls[0]
    payload = json.loads(messages[1].content)
    schema = options.response_format.schema
    assert schema["properties"]["assignments"]["minItems"] == 1
    assert schema["properties"]["assignments"]["maxItems"] == 1
    assert payload["planner_limits"]["maximum_assignments"] == 1
    assert payload["external_dependencies_read_only"][0]["dependency_id"] == "order_ab"


def test_rejected_visual_candidate_does_not_poison_spare_assignment_binding():
    dispatcher, broker, workers, _ = _dispatcher(("uav_a",))
    try:
        first = dispatcher.prepare_worker_for("uav_a", assignment_id="assignment_rejected")
        second = dispatcher.prepare_worker_for("uav_a", assignment_id="assignment_accepted")
        assert first is not second
        assert dispatcher._assignment_ids == {}
        assert dispatcher._facades == {}
        assert not workers["uav_a"].submitted
        official = dispatcher.worker_for("uav_a", assignment_id="assignment_accepted")
        assert official is not None
        assert dispatcher._assignment_ids == {"uav_a":"assignment_accepted"}
        with pytest.raises(ValueError, match="another visual assignment"):
            dispatcher.prepare_worker_for("uav_a", assignment_id="assignment_conflict")
    finally:
        dispatcher.close()
