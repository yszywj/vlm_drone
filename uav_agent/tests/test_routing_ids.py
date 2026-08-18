"""Pure-Python coverage for schema-v2 and single-UAV routing boundaries."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import unittest

import numpy as np

from agents.mission_agent import (
    AgentStatus,
    MissionAgent,
    MissionAgentError,
    MissionAgentSnapshot,
)
from common.ids import (
    validate_invocation_id,
    validate_mission_id,
    validate_plan_id,
    validate_request_id,
    validate_review_id,
    validate_routing_id,
    validate_uav_id,
)
from env.kinematic_uav import KinematicUAV, UAVState
from models.base import ChatMessage, GenerationOptions, ModelResponse
from planner.base import MissionPlanner, PlannerOutputError
from planner.dynamic_llm_planner import DynamicLLMPlanner
from planner.json_schema import build_skill_plan_v2_json_schema
from planner.policy import PlannerLimits
from planner.schemas import (
    LandingZoneSpec,
    MissionIntent,
    PlannerRequest,
    PlannerWorldContext,
    SearchRegionSpec,
    SkillPlanDraft,
    SkillPlanDraftV2,
    migrate_plan_v1_to_v2,
)
from planner.skill_catalog import build_default_skill_catalog
from runtime.plan_validator import PlanValidationError, PlanValidator
from runtime.safety_supervisor import SafetySupervisor
from skills.base import Skill
from skills.land import LandGoal
from skills.manager import SkillManager, SkillManagerError
from skills.plan import TaskPlan
from skills.takeoff import TakeoffGoal
from skills.types import (
    Observation,
    SkillContext,
    SkillGoal,
    SkillInvocation,
    SkillName,
    SkillResultCode,
)
from target.target_manager import TargetManager


PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "dynamic_skill_planner_system.txt"
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value


class _Camera:
    def get_rgb(self) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])


class _InstantSuccessSkill(Skill):
    goal_type = SkillGoal

    def __init__(self, code: SkillResultCode) -> None:
        super().__init__()
        self._code = code

    def _on_tick(self, observation: Observation) -> None:
        self._succeed(self._code, "done")


class _FakeModelClient:
    def __init__(self, *outputs: dict[str, object]) -> None:
        self._outputs = deque(outputs)
        self.options: list[GenerationOptions | None] = []

    def chat(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        options: GenerationOptions | None = None,
    ) -> ModelResponse:
        self.options.append(options)
        return ModelResponse(
            json.dumps(self._outputs.popleft()),
            "fake",
            "stop",
            {},
        )


class _IntentPlanner(MissionPlanner):
    source = "scripted"

    def plan(self, request: PlannerRequest) -> MissionIntent:
        return MissionIntent("target", "search_area", 1.0, "home", 10.0)


def _world() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-20, -20, 0),
        scene_max_xyz_m=(20, 20, 20),
        initial_uav_xyz_m=(0, 0, 0),
        search_regions={
            "search_area": SearchRegionSpec(
                "search_area", (5, 5, 0), 3, (5, 2, 10), "search sector"
            )
        },
        landing_zones={
            "home": LandingZoneSpec("home", (0, 0), description="home pad")
        },
        default_takeoff_altitude_m=10,
        default_track_duration_s=10,
        search_timeout_s=30,
    )


def _uav() -> KinematicUAV:
    return KinematicUAV(UAVState(0, 0, 0, 0), 5.0, 1.0)


def _v1_plan() -> SkillPlanDraft:
    return SkillPlanDraft.from_dict(
        {
            "schema_version": 1,
            "steps": [
                {"id": "takeoff", "skill": "TAKEOFF", "args": {}},
                {
                    "id": "goto_home",
                    "skill": "GOTO",
                    "args": {"destination": "home"},
                },
                {"id": "land", "skill": "LAND", "args": {"zone": "home"}},
            ],
        }
    )


def _v2_dict(*, mission_id: str = "mission_001", uav_id: str = "uav_1") -> dict[str, object]:
    migrated = migrate_plan_v1_to_v2(
        _v1_plan(), mission_id=mission_id, uav_id=uav_id, plan_version=1
    )
    return migrated.to_dict()


class RoutingIdValidationTests(unittest.TestCase):
    def test_routed_runtime_types_require_explicit_uav_id(self) -> None:
        with self.assertRaisesRegex(TypeError, "uav_id"):
            SkillContext(  # type: ignore[call-arg]
                _uav(),
                _Camera(),
                None,
                _Clock(),
            )
        with self.assertRaisesRegex(TypeError, "uav_id"):
            Observation(  # type: ignore[call-arg]
                timestamp=0.0,
                uav_pose=UAVState(0.0, 0.0, 0.0, 0.0),
                uav_velocity=np.zeros(3),
                camera_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            )
        with self.assertRaisesRegex(TypeError, "uav_id"):
            MissionAgentSnapshot(  # type: ignore[call-arg]
                status=AgentStatus.IDLE,
                task_status="IDLE",
                active_skill=None,
                target=TargetManager().snapshot(),
                feedback=None,
                last_error=None,
            )

    def test_all_routing_ids_share_one_strict_validator(self) -> None:
        validators = (
            validate_uav_id,
            validate_mission_id,
            validate_plan_id,
            validate_review_id,
            validate_request_id,
            validate_invocation_id,
        )
        for validator in validators:
            self.assertEqual(validator("A.b-c_1"), "A.b-c_1")
            for invalid in ("", "1bad", " bad", "bad ", "a/b", "a" * 65):
                with self.subTest(validator=validator.__name__, invalid=invalid):
                    with self.assertRaises(ValueError):
                        validator(invalid)
            with self.assertRaises(TypeError):
                validator(123)
        self.assertEqual(validate_routing_id("uav_1", "custom"), "uav_1")

    def test_context_observation_manager_and_invocation_are_uav_bound(self) -> None:
        clock = _Clock()
        context = SkillContext(
            _uav(), _Camera(), None, clock, uav_id="uav_alpha"
        )
        manager = SkillManager(
            context,
            registry={SkillName.TAKEOFF: _InstantSuccessSkill(SkillResultCode.TAKEOFF_COMPLETE)},
        )
        self.assertEqual(manager.uav_id, "uav_alpha")
        bad_invocation = SkillInvocation(
            mission_id="mission_1",
            uav_id="uav_beta",
            plan_version=1,
            step_id="takeoff",
            invocation_id="invocation_1",
            skill_name=SkillName.TAKEOFF,
            goal=TakeoffGoal(10),
        )
        with self.assertRaisesRegex(SkillManagerError, "uav_id"):
            manager.invoke(bad_invocation)
        bad_plan = TaskPlan.from_dicts(
            [{"skill": "TAKEOFF", "target_altitude": 10}],
            mission_id="mission_1",
            uav_id="uav_beta",
        )
        with self.assertRaisesRegex(SkillManagerError, "uav_id"):
            manager.start_task(bad_plan)

    def test_transitions_and_execution_reports_are_fully_routed(self) -> None:
        clock = _Clock()
        manager = SkillManager(
            SkillContext(_uav(), _Camera(), None, clock, uav_id="uav_7"),
            registry={
                SkillName.TAKEOFF: _InstantSuccessSkill(
                    SkillResultCode.TAKEOFF_COMPLETE
                ),
                SkillName.LAND: _InstantSuccessSkill(SkillResultCode.LAND_COMPLETE),
            },
        )
        plan = TaskPlan.from_dicts(
            [
                {"id": "takeoff", "skill": "TAKEOFF", "target_altitude": 10},
                {"id": "land", "skill": "LAND"},
            ],
            mission_id="mission_7",
            uav_id="uav_7",
            plan_version=3,
        )
        manager.start_task(plan)
        observation = Observation(
            0.0,
            UAVState(0, 0, 0, 0),
            np.zeros(3),
            np.zeros((4, 4, 3), dtype=np.uint8),
            uav_id="uav_7",
        )
        manager.tick(observation)
        report = manager.execution_reports[0]
        self.assertEqual(report.uav_id, "uav_7")
        self.assertEqual(report.mission_id, "mission_7")
        self.assertEqual(report.plan_version, 3)
        self.assertEqual(report.step_id, "takeoff")
        self.assertTrue(report.invocation_id)
        for transition in manager.transition_log:
            self.assertEqual(transition.uav_id, "uav_7")
            self.assertEqual(transition.mission_id, "mission_7")
            self.assertEqual(transition.plan_version, 3)
            self.assertIn("uav_id", transition.to_dict())

        mismatched = Observation(
            1.0,
            UAVState(0, 0, 0, 0),
            np.zeros(3),
            np.zeros((4, 4, 3), dtype=np.uint8),
            uav_id="uav_other",
        )
        with self.assertRaisesRegex(SkillManagerError, "uav_id"):
            manager.tick(mismatched)

    def test_agent_snapshot_has_public_routing_id(self) -> None:
        snapshot = MissionAgentSnapshot(
            status=AgentStatus.IDLE,
            task_status="IDLE",
            active_skill=None,
            target=TargetManager().snapshot(),
            feedback=None,
            last_error=None,
            uav_id="uav_9",
        )
        self.assertEqual(snapshot.uav_id, "uav_9")
        self.assertEqual(snapshot.to_dict()["uav_id"], "uav_9")

    def test_agent_rejects_a_frame_routed_to_another_uav(self) -> None:
        clock = _Clock()
        manager = SkillManager(
            SkillContext(_uav(), _Camera(), None, clock, uav_id="uav_alpha"),
            registry={
                SkillName.TAKEOFF: _InstantSuccessSkill(
                    SkillResultCode.TAKEOFF_COMPLETE
                ),
                SkillName.GOTO: _InstantSuccessSkill(SkillResultCode.GOAL_REACHED),
                SkillName.SEARCH: _InstantSuccessSkill(
                    SkillResultCode.TARGET_FOUND
                ),
                SkillName.TRACK: _InstantSuccessSkill(
                    SkillResultCode.TRACK_COMPLETE
                ),
                SkillName.REACQUIRE: _InstantSuccessSkill(
                    SkillResultCode.TARGET_FOUND
                ),
                SkillName.LAND: _InstantSuccessSkill(SkillResultCode.LAND_COMPLETE),
            },
        )
        world = _world()
        agent = MissionAgent(
            _IntentPlanner(),
            PlanValidator(),
            SafetySupervisor(world.scene_min_xyz_m, world.scene_max_xyz_m),
            manager,
            TargetManager(),
            clock,
            uav_id="uav_alpha",
        )
        agent.start("find target", world)
        routed_snapshot = agent.snapshot()
        self.assertEqual(routed_snapshot.uav_id, "uav_alpha")
        self.assertEqual(routed_snapshot.skill_report["uav_id"], "uav_alpha")
        wrong_frame = Observation(
            0.0,
            UAVState(0, 0, 0, 0),
            np.zeros(3),
            np.zeros((4, 4, 3), dtype=np.uint8),
            uav_id="uav_beta",
        )
        with self.assertRaisesRegex(MissionAgentError, "uav_id"):
            agent.tick(wrong_frame)


class PlannerSchemaV2Tests(unittest.TestCase):
    def test_explicit_v1_migration_adds_every_routing_field(self) -> None:
        migrated = migrate_plan_v1_to_v2(
            _v1_plan(), mission_id="mission_001", uav_id="uav_1", plan_version=4
        )
        self.assertIsInstance(migrated, SkillPlanDraftV2)
        encoded = migrated.to_dict()
        self.assertEqual(encoded["schema_version"], 2)
        self.assertEqual(encoded["mission_id"], "mission_001")
        self.assertEqual(encoded["uav_id"], "uav_1")
        self.assertEqual(encoded["plan_version"], 4)
        self.assertTrue(all(step["uav_id"] == "uav_1" for step in encoded["steps"]))

    def test_v2_schema_consts_are_trusted_and_steps_require_uav_id(self) -> None:
        schema = build_skill_plan_v2_json_schema(
            world_context=_world(),
            skill_catalog=build_default_skill_catalog(),
            limits=PlannerLimits(),
            mission_id="mission_001",
            uav_id="uav_1",
            plan_version=2,
        )
        self.assertEqual(schema["properties"]["schema_version"]["const"], 2)
        self.assertEqual(schema["properties"]["mission_id"]["const"], "mission_001")
        self.assertIn("target_spec", schema["required"])
        self.assertFalse(
            schema["properties"]["target_spec"]["additionalProperties"]
        )
        variants = schema["properties"]["steps"]["items"]["oneOf"]
        self.assertTrue(all("uav_id" in variant["required"] for variant in variants))
        # The deployed vLLM grammar backend crashes on uniqueItems.  Duplicate
        # semantic entries remain rejected by the strict TargetSpec parser.
        self.assertNotIn("uniqueItems", json.dumps(schema, sort_keys=True))

    def test_qwen_v2_requires_explicit_safe_target_spec_before_takeoff(self) -> None:
        missing = _v2_dict()
        missing.pop("target_spec")
        repaired = _v2_dict()
        repaired["target_spec"] = {
            "original_description": "red moving vehicle",
            "category": "vehicle",
            "hard_attributes": ["red body"],
            "soft_attributes": [],
            "negative_constraints": [],
            "relation_constraints": [],
            "query_ladder": ["red vehicle"],
            "inspection_questions": ["Is the body red?"],
            "immutable_identity_summary": "red moving vehicle",
            "mutable_appearance_notes": [],
        }
        client = _FakeModelClient(missing, repaired)
        planner = DynamicLLMPlanner(client, PROMPT_PATH)
        request = PlannerRequest(
            "return home after finding the red moving vehicle",
            _world(),
            mission_id="mission_001",
            uav_id="uav_1",
            plan_version=1,
        )

        result = planner.plan(request)

        self.assertEqual(len(client.options), 2)
        self.assertEqual(result.target_spec.category, "vehicle")
        self.assertEqual(
            result.target_spec.immutable_identity_summary,
            "red moving vehicle",
        )

        unsafe = _v2_dict()
        unsafe["target_spec"]["immutable_identity_summary"] = (
            "oracle_target_pose=(1,2,3)"
        )
        bad_client = _FakeModelClient(unsafe, unsafe)
        with self.assertRaises(PlannerOutputError):
            DynamicLLMPlanner(bad_client, PROMPT_PATH).plan(request)

    def test_qwen_routing_mismatch_is_repaired_once_then_rejected(self) -> None:
        bad = _v2_dict(mission_id="mission_wrong", uav_id="uav_1")
        client = _FakeModelClient(bad, bad)
        planner = DynamicLLMPlanner(client, PROMPT_PATH)
        request = PlannerRequest(
            "return home",
            _world(),
            mission_id="mission_001",
            uav_id="uav_1",
            plan_version=1,
        )
        with self.assertRaises(PlannerOutputError):
            planner.plan(request)
        self.assertEqual(len(client.options), 2)
        response_schema = client.options[0].response_format.schema
        self.assertEqual(response_schema["properties"]["schema_version"]["const"], 2)

    def test_validator_rejects_runtime_mismatch_and_compiles_v2_ids(self) -> None:
        draft = SkillPlanDraftV2.from_dict(_v2_dict())
        validator = PlanValidator()
        with self.assertRaisesRegex(PlanValidationError, "requires trusted"):
            validator.validate_and_compile(
                draft,
                _world(),
                source="dynamic_llm",
            )
        with self.assertRaisesRegex(PlanValidationError, "routing IDs"):
            validator.validate_and_compile(
                draft,
                _world(),
                source="dynamic_llm",
                mission_id="mission_001",
                uav_id="uav_other",
                plan_version=1,
            )
        compiled = validator.validate_and_compile(
            draft,
            _world(),
            source="dynamic_llm",
            mission_id="mission_001",
            uav_id="uav_1",
            plan_version=1,
        )
        self.assertEqual(compiled.task_plan.mission_id, "mission_001")
        self.assertEqual(compiled.task_plan.uav_id, "uav_1")
        self.assertEqual(compiled.task_plan.plan_version, 1)


if __name__ == "__main__":
    unittest.main()
