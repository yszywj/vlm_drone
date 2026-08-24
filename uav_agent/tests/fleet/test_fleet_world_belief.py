from __future__ import annotations

import pytest

from fleet.world_belief import AgentFleetSummary, FleetWorldBelief


def _agents() -> dict[str, AgentFleetSummary]:
    return {
        "uav_a": AgentFleetSummary(
            uav_id="uav_a",
            assignment_id="assignment_a_i",
            status="RUNNING",
            plan_version=1,
            current_region="CIRCLE",
        )
    }


@pytest.mark.parametrize(
    "forbidden_key",
    (
        "oracle_target_pose",
        "oracle_target_xyz_m",
        "ORACLE_TARGET_PRIVATE_STATE",
    ),
)
def test_world_belief_rejects_every_oracle_target_side_channel(
    forbidden_key: str,
) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        FleetWorldBelief(
            fleet_mission_id="fleet_mission_belief",
            fleet_plan_version=1,
            timestamp_s=1.0,
            agents=_agents(),
            events=({"nested": {forbidden_key: [1.0, 2.0, 3.0]}},),
        )


def test_world_belief_rejects_nonfinite_time_and_invalid_summary_labels() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        FleetWorldBelief(
            fleet_mission_id="fleet_mission_belief",
            fleet_plan_version=1,
            timestamp_s=float("nan"),
            agents=_agents(),
        )
    with pytest.raises(ValueError, match="current_region"):
        AgentFleetSummary(
            uav_id="uav_a",
            assignment_id="assignment_a_i",
            status="RUNNING",
            plan_version=1,
            current_region=" ",
        )
