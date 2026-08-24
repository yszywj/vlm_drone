"""Deterministic Spatial V3 baseline used by Fleet planner demos/tests."""

from __future__ import annotations

import json

from planner.base import MissionPlanner
from planner.schemas import PlannerRequest
from planner.schemas_v3 import SkillPlanDraftV3


class ScriptedAssignmentSpatialPlanner(MissionPlanner):
    """Compile the focused JSON created by FleetAssignmentCompiler.

    This planner has no natural-language parser and cannot see another
    assignment.  It exists only as a deterministic local Spatial V3 baseline.
    """

    source = "dynamic_scripted"

    def plan(self, request: PlannerRequest) -> SkillPlanDraftV3:
        if not isinstance(request, PlannerRequest):
            raise TypeError("request must be a PlannerRequest")
        try:
            payload = json.loads(request.instruction)
            target_spec = payload["target_spec"]
            region = payload["search_region"]
            duration = payload["track_duration_s"]
            home = payload["return_home"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("focused Fleet assignment instruction is invalid") from exc
        shape = region.get("shape") if isinstance(region, dict) else None
        strategy = {
            "CIRCLE": "SPIRAL_OUT",
            "RECTANGLE": "LAWNMOWER",
            "POLYGON": "LAWNMOWER",
            "SECTOR": "SECTOR_SWEEP",
            "CORRIDOR": "CORRIDOR_FOLLOW",
            "RELATIONAL": "RANDOM_COVERAGE",
        }.get(shape)
        if strategy is None:
            raise ValueError(f"unsupported assignment RegionSpec shape: {shape}")
        return SkillPlanDraftV3.from_dict(
            {
                "schema_version": 3,
                "mission_id": request.mission_id,
                "uav_id": request.uav_id,
                "plan_version": request.plan_version,
                "assumptions": [],
                "target_spec": target_spec,
                "steps": [
                    {
                        "id": "takeoff_1",
                        "uav_id": request.uav_id,
                        "skill": "TAKEOFF",
                        "args": {
                            "altitude_m": request.world_context.default_takeoff_altitude_m
                        },
                    },
                    {
                        "id": "search_1",
                        "uav_id": request.uav_id,
                        "skill": "SEARCH",
                        "args": {
                            "region": region,
                            "strategy": {"kind": strategy, "spacing_m": 4.0},
                            "entry_policy": "START_IN_PLACE_IF_INSIDE",
                            "target_description": target_spec["original_description"],
                            "search_altitude_m": request.world_context.default_takeoff_altitude_m,
                            "timeout_s": request.world_context.search_timeout_s,
                        },
                    },
                    {
                        "id": "track_1",
                        "uav_id": request.uav_id,
                        "skill": "TRACK",
                        "args": {
                            "target_ref": "$search_1.target_id",
                            "duration_s": duration,
                        },
                    },
                    {
                        "id": "goto_home",
                        "uav_id": request.uav_id,
                        "skill": "GOTO",
                        "args": {
                            "target": {"kind": "NAMED_LOCATION", "name": home}
                        },
                    },
                    {
                        "id": "land_1",
                        "uav_id": request.uav_id,
                        "skill": "LAND",
                        "args": {"zone": home},
                    },
                ],
            }
        )


class RoutedPreplannedSpatialPlanner(MissionPlanner):
    """Replay one already validated assignment draft with fresh Agent routing.

    Fleet entry points use this after all LLM/local planning succeeds before
    ``SimulationApp`` starts.  ``MissionAgent`` still owns the final trusted
    mission ID and validation boundary, while no second model request is made.
    """

    def __init__(self, draft: SkillPlanDraftV3, *, source: str) -> None:
        if not isinstance(draft, SkillPlanDraftV3):
            raise TypeError("draft must be a SkillPlanDraftV3")
        if source not in {"dynamic_scripted", "dynamic_llm"}:
            raise ValueError("source must be dynamic_scripted or dynamic_llm")
        self._draft = draft
        self.source = source

    def plan(self, request: PlannerRequest) -> SkillPlanDraftV3:
        if not isinstance(request, PlannerRequest):
            raise TypeError("request must be a PlannerRequest")
        try:
            focused = json.loads(request.instruction)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("preplanned assignment requires focused JSON instruction") from exc
        searches = [step for step in self._draft.steps if step.skill == "SEARCH"]
        tracks = [step for step in self._draft.steps if step.skill == "TRACK"]
        if (
            focused.get("target_spec") != (
                None if self._draft.target_spec is None else self._draft.target_spec.to_dict()
            )
            or len(searches) != 1
            or focused.get("search_region") != searches[0].region.to_dict()
            or len(tracks) != 1
            or focused.get("track_duration_s") != tracks[0].args.get("duration_s")
        ):
            raise ValueError("focused assignment differs from the preplanned Spatial V3 draft")
        data = self._draft.to_dict()
        data["mission_id"] = request.mission_id
        data["uav_id"] = request.uav_id
        data["plan_version"] = request.plan_version
        for step in data["steps"]:
            step["uav_id"] = request.uav_id
        return SkillPlanDraftV3.from_dict(data)


__all__ = ["RoutedPreplannedSpatialPlanner", "ScriptedAssignmentSpatialPlanner"]
