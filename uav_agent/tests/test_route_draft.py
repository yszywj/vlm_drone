from __future__ import annotations

import unittest

from planner.route_types import (
    AvoidanceStrategy,
    AvoidanceStrategyType,
    RouteConstraints,
    RouteContractError,
    RouteDraft,
    RouteWaypoint,
)
from planner.spatial import CoordinateFrame
from planner.spatial_resolver import FramePose
from planner.route_critic import RouteCritique, RouteCriticStatus
from planner.route_types import RouteState
from runtime.route_registry import RouteRegistry, RouteRegistryError


class RouteDraftTest(unittest.TestCase):
    def test_round_trip_keeps_explicit_frame_and_model_coordinates(self) -> None:
        route = RouteDraft(
            "route_1",
            CoordinateFrame.UAV_HOLD_FLU,
            (
                RouteWaypoint("wp_1", (1.5, 3.5, 1.0)),
                RouteWaypoint("wp_2", (7.0, 3.5, 1.0)),
                RouteWaypoint("wp_3", (9.0, 0.0, 0.0)),
            ),
        )
        self.assertEqual(RouteDraft.from_dict(route.to_dict()), route)
        self.assertGreater(route.length_m, 0.0)

    def test_rejects_bare_nonfinite_or_duplicate_waypoints(self) -> None:
        with self.assertRaisesRegex(RouteContractError, "2..16"):
            RouteDraft("route_1", CoordinateFrame.WORLD_ENU, (RouteWaypoint("wp_1", (1, 2, 3)),))
        with self.assertRaisesRegex(ValueError, "finite"):
            RouteWaypoint("wp_1", (1.0, float("nan"), 3.0))
        with self.assertRaisesRegex(RouteContractError, "unique"):
            RouteDraft(
                "route_1",
                CoordinateFrame.WORLD_ENU,
                (RouteWaypoint("wp_1", (0, 0, 0)), RouteWaypoint("wp_1", (1, 0, 0))),
            )

    def test_constraints_are_finite_and_bounded(self) -> None:
        constraints = RouteConstraints()
        self.assertEqual(constraints.max_waypoints, 5)
        with self.assertRaisesRegex(RouteContractError, "max_waypoints"):
            RouteConstraints(max_waypoints=100)

    def test_avoidance_strategy_has_no_hidden_reasoning_or_waypoints(self) -> None:
        strategy = AvoidanceStrategy(
            AvoidanceStrategyType.BYPASS_LEFT,
            "original_goto_target",
            ("LEFT_CLEARANCE_VISIBLE",),
        )
        self.assertEqual(strategy.to_dict()["strategy"], "BYPASS_LEFT")
        self.assertNotIn("chain_of_thought", strategy.to_dict())

    def test_registry_preserves_raw_proposal_and_state(self) -> None:
        route = RouteDraft(
            "route_9",
            CoordinateFrame.UAV_HOLD_FLU,
            (RouteWaypoint("wp_1", (1, 2, 0)), RouteWaypoint("wp_2", (3, 2, 0))),
        )
        registry = RouteRegistry(max_records=2)
        raw = {"route_draft": route.to_dict(), "model_note": "left"}
        registry.register(
            route,
            frame_snapshot=FramePose((10, 20, 5), 0.5),
            raw_proposal=raw,
            plan_version=2,
            proposal_timestamp_s=12.0,
        )
        raw["model_note"] = "mutated"
        critique = RouteCritique(RouteCriticStatus.ACCEPT, "route_9", (), 2.0, 4.0)
        accepted = registry.record_critique("route_9", critique)
        self.assertEqual(accepted.state, RouteState.ACCEPTED)
        self.assertEqual(accepted.raw_proposal["model_note"], "left")
        self.assertEqual(registry.transition("route_9", "EXECUTING").state, RouteState.EXECUTING)
        with self.assertRaisesRegex(RouteRegistryError, "illegal"):
            registry.transition("route_9", RouteState.REJECTED)

    def test_accepted_route_can_be_rejected_when_atomic_publication_fails(self) -> None:
        route = RouteDraft(
            "route_publish_failure",
            CoordinateFrame.UAV_HOLD_FLU,
            (RouteWaypoint("wp_1", (1, 2, 0)), RouteWaypoint("wp_2", (3, 2, 0))),
        )
        registry = RouteRegistry()
        registry.register(
            route,
            frame_snapshot=FramePose((10, 20, 5), 0.5),
            raw_proposal={"route_draft": route.to_dict()},
            plan_version=2,
            proposal_timestamp_s=12.0,
        )
        registry.record_critique(
            route.route_id,
            RouteCritique(RouteCriticStatus.ACCEPT, route.route_id, (), 2.0, 4.0),
        )

        rejected = registry.transition(route.route_id, RouteState.REJECTED)

        self.assertEqual(rejected.state, RouteState.REJECTED)


if __name__ == "__main__":
    unittest.main()
