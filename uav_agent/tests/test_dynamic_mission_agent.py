"""Pure-Python MissionAgent coverage for constrained dynamic Skill plans.

The tests exercise the real validator, safety supervisor, SkillManager, and
TargetManager with deterministic Skills.  They intentionally do not import
Isaac Sim, connect to a model service, or use image/oracle ground-truth data.
The explicit ORACLE_EVALUATION runtime profile only authorizes the Stage-0
SEARCH/REACQUIRE result shortcut used by these scripted Skills.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import unittest

import numpy as np

from agents.mission_agent import AgentStatus, MissionAgent
from env.kinematic_uav import KinematicUAV, UAVState
from perception.runtime import PerceptionRuntimeProfile
from planner.base import MissionPlanner
from planner.mission_program import MissionEdge, MissionNode, ProgramEvent
from planner.program_patch import ProgramPatch
from planner.schemas import (
    CompiledMission,
    LandingZoneSpec,
    MissionIntent,
    PlannerOutput,
    PlannerRequest,
    PlannerWorldContext,
    SearchRegionSpec,
    SkillPlanDraft,
)
from planner.spatial import CircleRegion, CoordinateFrame
from runtime.plan_validator import PlanValidator
from runtime.safety_supervisor import SafetySupervisor
from skills.base import Skill
from skills.hover import HoverSkill
from skills.manager import SkillManager
from skills.plan import TaskPlan
from skills.search_strategy import (
    SearchEntryPolicy,
    SearchStrategySpec,
    SearchStrategyType,
)
from skills.types import (
    Observation,
    SkillContext,
    SkillGoal,
    SkillName,
    SkillResultCode,
    SkillStatus,
)
from target.target_manager import TargetManager
from target.types import TargetLifecycle


class ManualClock:
    def __init__(self) -> None:
        self.time_s = 0.0

    def now(self) -> float:
        return self.time_s

    def set(self, time_s: float) -> None:
        self.time_s = float(time_s)


class FakeCamera:
    def get_rgb(self) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros(3, dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )


class StaticPlanner(MissionPlanner):
    """Return a defensive copy of one legacy or dynamic planner output."""

    def __init__(self, output: PlannerOutput, *, source: str) -> None:
        self._output = output
        self.source = source
        self.calls = 0

    def plan(self, request: PlannerRequest) -> PlannerOutput:
        if not isinstance(request, PlannerRequest):
            raise TypeError("request must be a PlannerRequest")
        self.calls += 1
        if isinstance(self._output, SkillPlanDraft):
            return SkillPlanDraft.from_dict(self._output.to_dict())
        return MissionIntent.from_dict(self._output.to_dict())


@dataclass(frozen=True, slots=True)
class ScriptedOutcome:
    status: SkillStatus
    code: SkillResultCode
    data: dict[str, object] = field(default_factory=dict)


class ScriptedSkill(Skill):
    """Consume exactly one deterministic terminal outcome per external tick."""

    goal_type = SkillGoal

    def __init__(self, *outcomes: ScriptedOutcome) -> None:
        super().__init__()
        self._outcomes = deque(outcomes)

    def _on_start(self, goal: SkillGoal, context: SkillContext) -> None:
        del goal, context

    def _on_tick(self, observation: Observation) -> None:
        del observation
        if not self._outcomes:
            raise AssertionError("ScriptedSkill has no queued outcome")
        outcome = self._outcomes.popleft()
        if outcome.status is SkillStatus.SUCCEEDED:
            self._succeed(outcome.code, "scripted success", outcome.data)
            return
        if outcome.status is SkillStatus.FAILED:
            self._fail(outcome.code, "scripted failure", outcome.data)
            return
        raise AssertionError("ScriptedOutcome must be terminal")


def succeeded(
    code: SkillResultCode,
    data: dict[str, object] | None = None,
) -> ScriptedOutcome:
    return ScriptedOutcome(SkillStatus.SUCCEEDED, code, data or {})


def failed(
    code: SkillResultCode,
    data: dict[str, object] | None = None,
) -> ScriptedOutcome:
    return ScriptedOutcome(SkillStatus.FAILED, code, data or {})


def world_context() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
        initial_uav_xyz_m=(0.0, 0.0, 0.0),
        search_regions={
            "search_area": SearchRegionSpec(
                name="search_area",
                center_xyz_m=(20.0, 20.0, 0.0),
                radius_m=10.0,
                approach_xyz_m=(20.0, 5.0, 10.0),
                description="known semantic search area",
            )
        },
        landing_zones={
            "home": LandingZoneSpec(
                name="home",
                position_xy_m=(0.0, 0.0),
                ground_altitude_m=0.0,
                description="home landing pad",
            )
        },
        default_takeoff_altitude_m=10.0,
        default_track_duration_s=30.0,
        search_timeout_s=60.0,
        goto_timeout_s=120.0,
        land_timeout_s=60.0,
    )


def draft(*steps: dict[str, object]) -> SkillPlanDraft:
    return SkillPlanDraft.from_dict(
        {"schema_version": 1, "steps": list(steps)}
    )


def step(
    step_id: str,
    skill: str,
    args: dict[str, object],
    *,
    recovery: dict[str, object] | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "id": step_id,
        "skill": skill,
        "args": args,
    }
    if recovery is not None:
        value["recovery"] = recovery
    return value


def legacy_intent() -> MissionIntent:
    return MissionIntent(
        target_description="moving red target",
        search_region="search_area",
        track_duration_s=30.0,
        landing_zone="home",
        takeoff_altitude_m=10.0,
    )


def default_outcomes() -> dict[SkillName, list[ScriptedOutcome]]:
    return {
        SkillName.TAKEOFF: [
            succeeded(SkillResultCode.TAKEOFF_COMPLETE)
        ],
        SkillName.GOTO: [
            succeeded(SkillResultCode.GOAL_REACHED),
            succeeded(SkillResultCode.GOAL_REACHED),
            succeeded(SkillResultCode.GOAL_REACHED),
        ],
        SkillName.SEARCH: [
            succeeded(
                SkillResultCode.TARGET_FOUND,
                {"target_id": "target_0"},
            )
        ],
        SkillName.TRACK: [
            succeeded(SkillResultCode.TRACK_COMPLETE)
        ],
        SkillName.REACQUIRE: [
            succeeded(
                SkillResultCode.TARGET_FOUND,
                {"target_id": "target_0"},
            )
        ],
        SkillName.LAND: [succeeded(SkillResultCode.LAND_COMPLETE)],
    }


@dataclass(slots=True)
class Harness:
    agent: MissionAgent
    planner: StaticPlanner
    manager: SkillManager
    target: TargetManager
    clock: ManualClock
    context: PlannerWorldContext

    def start(self) -> None:
        self.agent.start("execute the bounded mission", self.context)

    def tick(self, timestamp: float):
        self.clock.set(timestamp)
        return self.agent.tick(
            Observation(
                uav_id="uav_1",
                timestamp=float(timestamp),
                uav_pose=UAVState(0.0, 0.0, 10.0, 0.0),
                uav_velocity=np.zeros(3, dtype=np.float64),
                camera_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            )
        )


def make_harness(
    output: PlannerOutput,
    *,
    source: str,
    outcome_overrides: dict[SkillName, list[ScriptedOutcome]] | None = None,
    validator: PlanValidator | None = None,
    runtime_program: str = "linear",
) -> Harness:
    configured = default_outcomes()
    if outcome_overrides is not None:
        configured.update(outcome_overrides)
    registry = {
        name: ScriptedSkill(*outcomes)
        for name, outcomes in configured.items()
    }
    # Supervisory HOVER is an internal runtime transition rather than a
    # planner step, so keep the real continuously-commanded implementation in
    # this otherwise deterministic registry.
    registry[SkillName.HOVER] = HoverSkill()
    clock = ManualClock()
    manager = SkillManager(
        SkillContext(
            uav=KinematicUAV(
                UAVState(0.0, 0.0, 0.0, 0.0),
                max_speed_mps=5.0,
                max_yaw_rate_rad_s=2.0,
            ),
            camera=FakeCamera(),
            perception=None,
            clock=clock,
            uav_id="uav_1",
        ),
        registry=registry,
    )
    planner = StaticPlanner(output, source=source)
    target = TargetManager()
    context = world_context()
    agent = MissionAgent(
        planner=planner,
        validator=PlanValidator() if validator is None else validator,
        safety=SafetySupervisor(
            context.scene_min_xyz_m,
            context.scene_max_xyz_m,
            max_mission_time_s=300.0,
            max_safe_altitude_m=25.0,
        ),
        skill_manager=manager,
        target_manager=target,
        clock=clock,
        perception_runtime_profile=(
            PerceptionRuntimeProfile.ORACLE_EVALUATION
        ),
        acknowledge_privileged_oracle=True,
        runtime_program=runtime_program,
    )
    return Harness(agent, planner, manager, target, clock, context)


def run_to_terminal(harness: Harness, *, start_at: int = 1):
    snapshot = harness.agent.snapshot()
    for timestamp in range(start_at, start_at + 20):
        snapshot = harness.tick(float(timestamp))
        if snapshot.status in {
            AgentStatus.SUCCEEDED,
            AgentStatus.FAILED,
            AgentStatus.CANCELED,
        }:
            return snapshot
    raise AssertionError("scripted mission did not terminate")


class DynamicMissionAgentTests(unittest.TestCase):
    def test_agent_adopts_only_the_manager_installed_next_plan_version(self) -> None:
        plan = draft(
            step("takeoff", "TAKEOFF", {}),
            step("return_home", "GOTO", {"destination": "home"}),
            step("land", "LAND", {"zone": "home"}),
        )
        harness = make_harness(plan, source="dynamic_scripted")
        harness.start()
        harness.tick(1.0)  # TAKEOFF -> GOTO
        current = harness.manager.task_plan
        assert current is not None
        replacement = TaskPlan(
            current.steps,
            mission_id=current.mission_id,
            uav_id=current.uav_id,
            plan_version=2,
        )

        with self.assertRaisesRegex(Exception, "atomically installed"):
            harness.agent.adopt_runtime_task_plan(replacement)

        harness.clock.set(1.1)
        harness.manager.interrupt_with_hover("PATH_BLOCKED")
        harness.tick(2.0)  # establish the hold on a fresh Observation
        harness.manager.replace_interrupted_step_and_suffix(replacement)
        harness.tick(3.0)  # release HOVER and atomically install replacement

        adopted = harness.agent.adopt_runtime_task_plan(replacement)

        self.assertEqual(adopted.plan_version, 2)
        self.assertEqual(harness.manager.task_plan.plan_version, 2)

    def test_agent_adopts_atomically_published_graph_program_plan(self) -> None:
        plan = draft(
            step("takeoff", "TAKEOFF", {}),
            step("return_home", "GOTO", {"destination": "home"}),
            step("land", "LAND", {"zone": "home"}),
        )
        harness = make_harness(
            plan,
            source="dynamic_scripted",
            runtime_program="graph",
        )
        harness.start()
        harness.tick(1.0)  # TAKEOFF -> GOTO
        current = harness.manager.task_plan
        assert current is not None
        current_step, land_step = current.steps[1:]
        patch = ProgramPatch(
            mission_id=current.mission_id,
            uav_id=current.uav_id,
            base_plan_version=1,
            new_plan_version=2,
            replace_from_node_id=current_step.step_id,
            replacement_nodes=(
                MissionNode(current_step.step_id, current_step),
                MissionNode(land_step.step_id, land_step),
            ),
            replacement_edges=(
                MissionEdge(
                    current_step.step_id,
                    land_step.step_id,
                    ProgramEvent.SUCCESS,
                ),
            ),
            reason_codes=("PATH_BLOCKED",),
        )

        harness.clock.set(1.1)
        harness.manager.interrupt_with_hover("PATH_BLOCKED")
        harness.tick(2.0)  # establish HOVER
        harness.manager.replace_interrupted_program_suffix(patch)
        self.assertEqual(
            harness.manager.graph_task_plan_for_adoption().plan_version,
            1,
        )

        before_adoption = harness.tick(3.0)  # publish v2; successor is not ticked
        self.assertEqual(before_adoption.plan_version, 1)
        published = harness.manager.graph_task_plan_for_adoption()
        self.assertEqual(published.plan_version, 2)

        adopted = harness.agent.adopt_runtime_task_plan(published)

        self.assertEqual(adopted.plan_version, 2)
        self.assertEqual(harness.manager.program_snapshot.plan_version, 2)

    def test_unadopted_graph_program_version_fails_closed_before_motion_tick(self) -> None:
        plan = draft(
            step("takeoff", "TAKEOFF", {}),
            step("return_home", "GOTO", {"destination": "home"}),
            step("land", "LAND", {"zone": "home"}),
        )
        harness = make_harness(
            plan,
            source="dynamic_scripted",
            runtime_program="graph",
        )
        harness.start()
        harness.tick(1.0)
        current = harness.manager.task_plan
        assert current is not None
        current_step, land_step = current.steps[1:]
        patch = ProgramPatch(
            mission_id=current.mission_id,
            uav_id=current.uav_id,
            base_plan_version=1,
            new_plan_version=2,
            replace_from_node_id=current_step.step_id,
            replacement_nodes=(
                MissionNode(current_step.step_id, current_step),
                MissionNode(land_step.step_id, land_step),
            ),
            replacement_edges=(
                MissionEdge(
                    current_step.step_id,
                    land_step.step_id,
                    ProgramEvent.SUCCESS,
                ),
            ),
            reason_codes=("PATH_BLOCKED",),
        )
        harness.clock.set(1.1)
        harness.manager.interrupt_with_hover("PATH_BLOCKED")
        harness.tick(2.0)
        harness.manager.replace_interrupted_program_suffix(patch)
        harness.tick(3.0)  # publish v2, intentionally do not adopt it

        failed_closed = harness.tick(4.0)

        self.assertIs(failed_closed.status, AgentStatus.RUNNING)
        self.assertEqual(failed_closed.active_skill, "LAND")
        self.assertIn(
            "without MissionAgent adoption",
            failed_closed.last_error or "",
        )

    def test_consecutive_search_fallback_preserves_one_target_lifecycle(self) -> None:
        class FallbackValidator(PlanValidator):
            def validate_and_compile(
                self,
                planner_output,
                context,
                *,
                source,
                mission_id=None,
                uav_id=None,
                plan_version=None,
                **kwargs,
            ):
                del context, kwargs
                plan = TaskPlan.from_dicts(
                    [
                        {
                            "id": "takeoff",
                            "skill": "TAKEOFF",
                            "target_altitude": 10.0,
                        },
                        {
                            "id": "search_near",
                            "skill": "SEARCH",
                            "region": CircleRegion(
                                CoordinateFrame.WORLD_ENU,
                                (10.0, 0.0, 0.0),
                                5.0,
                            ),
                            "strategy": SearchStrategySpec(
                                SearchStrategyType.PERIMETER_V1
                            ),
                            "entry_policy": SearchEntryPolicy.NEAREST_POINT,
                            "search_altitude_m": 10.0,
                            "timeout_s": 20.0,
                            "target_description": "moving red target",
                        },
                        {
                            "id": "search_far",
                            "skill": "SEARCH",
                            "region": CircleRegion(
                                CoordinateFrame.WORLD_ENU,
                                (20.0, 0.0, 0.0),
                                5.0,
                            ),
                            "strategy": SearchStrategySpec(
                                SearchStrategyType.PERIMETER_V1
                            ),
                            "entry_policy": SearchEntryPolicy.NEAREST_POINT,
                            "search_altitude_m": 10.0,
                            "timeout_s": 20.0,
                            "target_description": "moving red target",
                        },
                        {
                            "id": "track",
                            "skill": "TRACK",
                            "target_id": "$search_far.target_id",
                            "track_duration": 5.0,
                        },
                        {"id": "land", "skill": "LAND"},
                    ],
                    mission_id=mission_id,
                    uav_id=uav_id,
                    plan_version=plan_version,
                )
                return CompiledMission(planner_output, plan, source)

        for runtime_program in ("linear", "graph"):
            with self.subTest(runtime_program=runtime_program):
                harness = make_harness(
                    legacy_intent(),
                    source="scripted",
                    validator=FallbackValidator(),
                    runtime_program=runtime_program,
                    outcome_overrides={
                        SkillName.SEARCH: [
                            failed(
                                SkillResultCode.SEARCH_EXHAUSTED,
                                {"coverage_ratio": 1.0},
                            ),
                            succeeded(
                                SkillResultCode.TARGET_FOUND,
                                {"target_id": "target_0"},
                            ),
                        ]
                    },
                )
                harness.start()
                searching = harness.tick(1.0)
                self.assertIs(
                    searching.target.lifecycle,
                    TargetLifecycle.SEARCHING,
                )
                second_region = harness.tick(2.0)
                self.assertIs(
                    second_region.target.lifecycle,
                    TargetLifecycle.SEARCHING,
                )
                self.assertEqual(
                    harness.manager.active_planned_step_id,
                    "search_far",
                )
                tracking = harness.tick(3.0)
                self.assertIs(
                    tracking.target.lifecycle,
                    TargetLifecycle.TRACKING,
                )
                self.assertEqual(tracking.target.target_id, "target_0")
                self.assertEqual(
                    [event.reason for event in harness.target.events()],
                    [
                        "search_started",
                        "target_locked_by_oracle_evaluation",
                        "tracking_started",
                    ],
                )

    def test_navigation_without_search_never_fabricates_target_process(self) -> None:
        plan = draft(
            step("takeoff", "TAKEOFF", {}),
            step("return_home", "GOTO", {"destination": "home"}),
            step("land", "LAND", {"zone": "home"}),
        )
        harness = make_harness(plan, source="dynamic_scripted")

        harness.start()
        self.assertIs(
            harness.target.lifecycle,
            TargetLifecycle.UNINITIALIZED,
        )
        harness.tick(1.0)
        self.assertIs(
            harness.target.lifecycle,
            TargetLifecycle.UNINITIALIZED,
        )
        landing = harness.tick(2.0)
        self.assertEqual(landing.active_skill, "LAND")
        self.assertIs(
            landing.target.lifecycle,
            TargetLifecycle.UNINITIALIZED,
        )
        final = run_to_terminal(harness, start_at=3)

        self.assertIs(final.status, AgentStatus.SUCCEEDED)
        self.assertIs(final.target.lifecycle, TargetLifecycle.TERMINATED)
        self.assertIsNone(final.target.target_id)
        self.assertIsNone(final.target.source)
        self.assertEqual(final.target.description, "uninitialized")
        states = [event.new_state for event in harness.target.events()]
        self.assertEqual(states, [TargetLifecycle.TERMINATED])
        self.assertNotIn(TargetLifecycle.SEARCHING, states)
        self.assertNotIn(TargetLifecycle.CANDIDATE, states)
        self.assertNotIn(TargetLifecycle.LOCKED, states)

    def test_search_then_goto_locks_before_later_track_entry(self) -> None:
        plan = draft(
            step("takeoff", "TAKEOFF", {}),
            step("approach", "GOTO", {"destination": "search_area"}),
            step(
                "search",
                "SEARCH",
                {
                    "region": "search_area",
                    "target_description": "moving red target",
                },
            ),
            step("reposition", "GOTO", {"destination": "search_area"}),
            step(
                "track",
                "TRACK",
                {"target_ref": "$search.target_id", "duration_s": 10.0},
            ),
            step("return_home", "GOTO", {"destination": "home"}),
            step("land", "LAND", {"zone": "home"}),
        )
        harness = make_harness(plan, source="dynamic_scripted")
        harness.start()

        harness.tick(1.0)  # TAKEOFF -> approach
        harness.tick(2.0)  # approach -> SEARCH
        search_result = harness.tick(3.0)  # SEARCH -> reposition GOTO

        self.assertEqual(search_result.active_skill, "GOTO")
        self.assertIs(search_result.target.lifecycle, TargetLifecycle.LOCKED)
        self.assertEqual(search_result.target.target_id, "target_0")
        self.assertNotIn(
            TargetLifecycle.TRACKING,
            [event.new_state for event in harness.target.events()],
        )

        track_entry = harness.tick(4.0)  # reposition -> TRACK
        self.assertEqual(track_entry.active_skill, "TRACK")
        self.assertIs(
            track_entry.target.lifecycle,
            TargetLifecycle.TRACKING,
        )
        self.assertEqual(
            [event.new_state for event in harness.target.events()][:3],
            [
                TargetLifecycle.SEARCHING,
                TargetLifecycle.LOCKED,
                TargetLifecycle.TRACKING,
            ],
        )
        self.assertIs(run_to_terminal(harness, start_at=5).status, AgentStatus.SUCCEEDED)

    def test_returned_compiled_plan_cannot_mutate_agent_snapshot(self) -> None:
        plan = draft(
            step("takeoff", "TAKEOFF", {}),
            step("approach", "GOTO", {"destination": "search_area"}),
            step(
                "search",
                "SEARCH",
                {
                    "region": "search_area",
                    "target_description": "moving red target",
                },
            ),
            step("return_home", "GOTO", {"destination": "home"}),
            step("land", "LAND", {"zone": "home"}),
        )
        harness = make_harness(plan, source="dynamic_scripted")

        compiled = harness.agent.start("execute", harness.context)
        compiled.task_plan.steps[2].params["target_description"] = "tampered"
        harness.tick(1.0)
        entered_search = harness.tick(2.0)

        self.assertEqual(entered_search.active_skill, "SEARCH")
        self.assertEqual(
            entered_search.target.description,
            "moving red target",
        )

    def test_search_without_track_returns_home_without_tracking_state(self) -> None:
        plan = draft(
            step("takeoff", "TAKEOFF", {}),
            step("approach", "GOTO", {"destination": "search_area"}),
            step(
                "search",
                "SEARCH",
                {
                    "region": "search_area",
                    "target_description": "moving red target",
                },
            ),
            step("return_home", "GOTO", {"destination": "home"}),
            step("land", "LAND", {"zone": "home"}),
        )
        harness = make_harness(plan, source="dynamic_scripted")
        harness.start()
        harness.tick(1.0)
        harness.tick(2.0)

        locked = harness.tick(3.0)
        self.assertEqual(locked.active_skill, "GOTO")
        self.assertIs(locked.target.lifecycle, TargetLifecycle.LOCKED)

        final = run_to_terminal(harness, start_at=4)
        states = [event.new_state for event in harness.target.events()]
        self.assertIs(final.status, AgentStatus.SUCCEEDED)
        self.assertEqual(
            states,
            [
                TargetLifecycle.SEARCHING,
                TargetLifecycle.LOCKED,
                TargetLifecycle.TERMINATED,
            ],
        )
        self.assertNotIn(TargetLifecycle.TRACKING, states)

    def test_dynamic_bounded_recovery_updates_target_lifecycle(self) -> None:
        plan = draft(
            step("takeoff", "TAKEOFF", {}),
            step("approach", "GOTO", {"destination": "search_area"}),
            step(
                "search",
                "SEARCH",
                {
                    "region": "search_area",
                    "target_description": "moving red target",
                },
            ),
            step(
                "track",
                "TRACK",
                {"target_ref": "$search.target_id", "duration_s": 20.0},
                recovery={
                    "skill": "REACQUIRE",
                    "max_attempts": 1,
                    "search_radius_m": 8.0,
                    "timeout_s": 15.0,
                },
            ),
            step("return_home", "GOTO", {"destination": "home"}),
            step("land", "LAND", {"zone": "home"}),
        )
        lost_data = {
            "target_id": "target_0",
            "last_seen_position": (7.0, 8.0, 0.0),
            "last_seen_velocity": (0.5, 0.0, 0.0),
            "last_seen_time": 3.5,
            "tracking_duration": 2.0,
        }
        harness = make_harness(
            plan,
            source="dynamic_scripted",
            outcome_overrides={
                SkillName.TRACK: [
                    failed(SkillResultCode.TARGET_LOST, lost_data),
                    succeeded(SkillResultCode.TRACK_COMPLETE),
                ]
            },
        )
        harness.start()
        harness.tick(1.0)
        harness.tick(2.0)
        harness.tick(3.0)  # SEARCH -> TRACK

        lost = harness.tick(4.0)  # TRACK -> internal REACQUIRE
        self.assertEqual(lost.active_skill, "REACQUIRE")
        self.assertIs(
            lost.target.lifecycle,
            TargetLifecycle.REACQUIRING,
        )
        self.assertEqual(lost.target.last_seen_position, (7.0, 8.0, 0.0))
        self.assertEqual(harness.manager.recovery_attempts, {"track": 1})

        recovered = harness.tick(5.0)  # REACQUIRE -> same TRACK step
        self.assertEqual(recovered.active_skill, "TRACK")
        self.assertIs(
            recovered.target.lifecycle,
            TargetLifecycle.TRACKING,
        )
        self.assertEqual(
            [event.new_state for event in harness.target.events()][-4:],
            [
                TargetLifecycle.LOST,
                TargetLifecycle.REACQUIRING,
                TargetLifecycle.LOCKED,
                TargetLifecycle.TRACKING,
            ],
        )
        recovery_records = [
            record
            for record in harness.manager.transition_log
            if record.new_skill is SkillName.REACQUIRE
        ]
        self.assertEqual(len(recovery_records), 1)
        self.assertEqual(recovery_records[0].old_step_id, "track")
        self.assertEqual(recovery_records[0].new_step_id, "track")
        self.assertEqual(recovery_records[0].recovery_attempt, 1)
        self.assertIs(run_to_terminal(harness, start_at=6).status, AgentStatus.SUCCEEDED)

    def test_legacy_mission_intent_still_runs_fixed_six_step_plan(self) -> None:
        harness = make_harness(legacy_intent(), source="scripted")

        compiled = harness.agent.start("legacy search mission", harness.context)
        final = run_to_terminal(harness)

        self.assertIs(compiled.intent, compiled.planner_output)
        self.assertIsNone(compiled.skill_plan_draft)
        self.assertEqual(compiled.source, "scripted")
        self.assertEqual(
            [step.skill for step in compiled.task_plan.steps],
            [
                SkillName.TAKEOFF,
                SkillName.GOTO,
                SkillName.SEARCH,
                SkillName.TRACK,
                SkillName.GOTO,
                SkillName.LAND,
            ],
        )
        self.assertEqual(harness.planner.calls, 1)
        self.assertIs(final.status, AgentStatus.SUCCEEDED)
        self.assertIs(final.target.lifecycle, TargetLifecycle.TERMINATED)

    def test_two_track_segments_are_locked_during_intervening_goto(self) -> None:
        plan = draft(
            step("takeoff", "TAKEOFF", {}),
            step("approach", "GOTO", {"destination": "search_area"}),
            step(
                "search",
                "SEARCH",
                {
                    "region": "search_area",
                    "target_description": "moving red target",
                },
            ),
            step(
                "track_1",
                "TRACK",
                {"target_ref": "$search.target_id", "duration_s": 5.0},
            ),
            step("reposition", "GOTO", {"destination": "search_area"}),
            step(
                "track_2",
                "TRACK",
                {"target_ref": "$search.target_id", "duration_s": 5.0},
            ),
            step("return_home", "GOTO", {"destination": "home"}),
            step("land", "LAND", {"zone": "home"}),
        )
        harness = make_harness(
            plan,
            source="dynamic_scripted",
            outcome_overrides={
                SkillName.TRACK: [
                    succeeded(SkillResultCode.TRACK_COMPLETE),
                    succeeded(SkillResultCode.TRACK_COMPLETE),
                ]
            },
        )
        harness.start()
        harness.tick(1.0)
        harness.tick(2.0)
        harness.tick(3.0)  # SEARCH -> TRACK 1

        between = harness.tick(4.0)  # TRACK 1 -> GOTO
        self.assertEqual(between.active_skill, "GOTO")
        self.assertIs(between.target.lifecycle, TargetLifecycle.LOCKED)

        second = harness.tick(5.0)  # GOTO -> TRACK 2
        self.assertEqual(second.active_skill, "TRACK")
        self.assertIs(second.target.lifecycle, TargetLifecycle.TRACKING)
        self.assertEqual(
            [event.new_state for event in harness.target.events()][-3:],
            [
                TargetLifecycle.TRACKING,
                TargetLifecycle.LOCKED,
                TargetLifecycle.TRACKING,
            ],
        )
        self.assertIs(
            run_to_terminal(harness, start_at=6).status,
            AgentStatus.SUCCEEDED,
        )


if __name__ == "__main__":
    unittest.main()
