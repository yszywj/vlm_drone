from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from types import MappingProxyType
import unittest

from models import AsyncModelResult, ModelProtocolError, ModelResponse
from planner.policy import PlannerLimits
from planner.revision import (
    PlanRevisionDraft,
    PlanRevisionRequest,
    QwenPlanRevisionPlanner,
    RevisionErrorCode,
    RevisionLimits,
    RevisionValidationError,
    RevisionValidator,
    build_plan_revision_json_schema,
    replace_plan_suffix,
)
from planner.schemas import (
    LandingZoneSpec,
    NavigationPointSpec,
    PlannerWorldContext,
    SearchRegionSpec,
    SkillPlanDraftV2,
)
from planner.skill_catalog import build_default_skill_catalog
from runtime.plan_validator import PlanValidator
from runtime.events import EventSeverity, MissionEvent, MissionEventType
from runtime.world_belief import CandidateSummary, QwenRequestStatus, WorldBelief
from target.types import TargetLifecycle, TargetSnapshot


def world() -> PlannerWorldContext:
    return PlannerWorldContext(
        scene_min_xyz_m=(-50.0, -50.0, 0.0),
        scene_max_xyz_m=(50.0, 50.0, 30.0),
        initial_uav_xyz_m=(0.0, 0.0, 0.0),
        search_regions={
            "search_area": SearchRegionSpec(
                name="search_area",
                center_xyz_m=(20.0, 20.0, 0.0),
                radius_m=10.0,
                approach_xyz_m=(20.0, 10.0, 10.0),
            )
        },
        landing_zones={
            "home": LandingZoneSpec(
                name="home",
                position_xy_m=(0.0, 0.0),
            )
        },
        navigation_points={
            "observation_point": NavigationPointSpec(
                name="observation_point",
                position_xyz_m=(10.0, 5.0, 10.0),
            )
        },
        default_takeoff_altitude_m=10.0,
        default_track_duration_s=20.0,
        search_timeout_s=60.0,
    )


def step(
    step_id: str,
    skill: str,
    args: dict[str, object],
    *,
    uav_id: str = "uav_1",
) -> dict[str, object]:
    return {"id": step_id, "uav_id": uav_id, "skill": skill, "args": args}


def original_plan() -> SkillPlanDraftV2:
    return SkillPlanDraftV2.from_dict(
        {
            "schema_version": 2,
            "mission_id": "mission_1",
            "uav_id": "uav_1",
            "plan_version": 1,
            "steps": [
                step("takeoff_1", "TAKEOFF", {"altitude_m": 10.0}),
                step(
                    "goto_search",
                    "GOTO",
                    {"destination": "search_area", "altitude_m": 10.0},
                ),
                step(
                    "search_1",
                    "SEARCH",
                    {
                        "region": "search_area",
                        "target_description": "red moving target",
                        "altitude_m": 10.0,
                    },
                ),
                step(
                    "track_1",
                    "TRACK",
                    {"target_ref": "$search_1.target_id", "duration_s": 20.0},
                ),
                step(
                    "goto_home",
                    "GOTO",
                    {"destination": "home", "altitude_m": 10.0},
                ),
                step("land_home", "LAND", {"zone": "home"}),
            ],
        }
    )


def revision_dict(
    *,
    replace_from: str = "track_1",
    mission_id: str = "mission_1",
    uav_id: str = "uav_1",
    base_version: int = 1,
    new_version: int = 2,
    steps: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if steps is None:
        steps = [
            step(
                "track_revised",
                "TRACK",
                {"target_ref": "$search_1.target_id", "duration_s": 15.0},
                uav_id=uav_id,
            ),
            step(
                "goto_home_revised",
                "GOTO",
                {"destination": "home", "altitude_m": 10.0},
                uav_id=uav_id,
            ),
            step(
                "land_home_revised",
                "LAND",
                {"zone": "home"},
                uav_id=uav_id,
            ),
        ]
    return {
        "schema_version": 2,
        "mission_id": mission_id,
        "uav_id": uav_id,
        "base_plan_version": base_version,
        "new_plan_version": new_version,
        "replace_from_step_id": replace_from,
        "steps": steps,
        "reason_codes": ["PATH_BLOCKED"],
    }


class PlanRevisionSchemaTest(unittest.TestCase):
    def test_strict_round_trip_is_immutable_and_defensive(self) -> None:
        source = revision_dict()
        draft = PlanRevisionDraft.from_dict(source)
        source["steps"][0]["args"]["duration_s"] = 999  # type: ignore[index]
        serialized = draft.to_dict()
        self.assertEqual(serialized["steps"][0]["args"]["duration_s"], 15.0)
        serialized["steps"][0]["args"]["duration_s"] = 888  # type: ignore[index]
        self.assertEqual(draft.to_dict()["steps"][0]["args"]["duration_s"], 15.0)
        with self.assertRaises(FrozenInstanceError):
            draft.new_plan_version = 7  # type: ignore[misc]

    def test_unknown_missing_version_jump_and_step_routing_are_rejected(self) -> None:
        examples = []
        unknown = revision_dict()
        unknown["extra"] = True
        examples.append(unknown)
        missing = revision_dict()
        del missing["reason_codes"]
        examples.append(missing)
        examples.append(revision_dict(new_version=3))
        wrong_step = revision_dict()
        wrong_step["steps"][0]["uav_id"] = "uav_2"  # type: ignore[index]
        examples.append(wrong_step)
        duplicate_reasons = revision_dict()
        duplicate_reasons["reason_codes"] = ["PATH_BLOCKED", "PATH_BLOCKED"]
        examples.append(duplicate_reasons)
        for value in examples:
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                PlanRevisionDraft.from_dict(value)

    def test_json_schema_is_strict_and_runtime_bound(self) -> None:
        schema = build_plan_revision_json_schema(
            world_context=world(),
            skill_catalog=build_default_skill_catalog(),
            limits=PlannerLimits(),
            mission_id="mission_1",
            uav_id="uav_1",
            base_plan_version=1,
            new_plan_version=2,
            replaceable_step_ids=("track_1", "goto_home", "land_home"),
        )
        self.assertFalse(schema["additionalProperties"])
        properties = schema["properties"]
        self.assertEqual(properties["schema_version"]["const"], 2)
        self.assertEqual(properties["mission_id"]["const"], "mission_1")
        self.assertEqual(properties["uav_id"]["const"], "uav_1")
        self.assertEqual(properties["base_plan_version"]["const"], 1)
        self.assertEqual(properties["new_plan_version"]["const"], 2)
        self.assertEqual(
            properties["replace_from_step_id"]["enum"],
            ["track_1", "goto_home", "land_home"],
        )
        self.assertEqual(properties["steps"]["minItems"], 1)
        self.assertNotIn("uniqueItems", json.dumps(schema, sort_keys=True))
        ordinary_skills = {
            variant["properties"]["skill"]["const"]
            for variant in properties["steps"]["items"]["oneOf"]
        }
        self.assertNotIn("INSPECT", ordinary_skills)
        for variant in properties["steps"]["items"]["oneOf"]:
            self.assertEqual(
                variant["properties"]["uav_id"]["const"],
                "uav_1",
            )
            self.assertIn("uav_id", variant["required"])

        inspect_schema = build_plan_revision_json_schema(
            world_context=world(),
            skill_catalog=build_default_skill_catalog(),
            limits=PlannerLimits(),
            mission_id="mission_1",
            uav_id="uav_1",
            base_plan_version=1,
            new_plan_version=2,
            replaceable_step_ids=("search_1", "track_1", "goto_home", "land_home"),
            trusted_inspect_candidate_id="candidate_1",
        )
        inspect_variant = next(
            variant
            for variant in inspect_schema["properties"]["steps"]["items"]["oneOf"]
            if variant["properties"]["skill"]["const"] == "INSPECT"
        )
        self.assertEqual(
            inspect_variant["properties"]["args"]["properties"]["candidate_id"],
            {"type": "string", "const": "candidate_1"},
        )


class QwenPlanRevisionPlannerTest(unittest.TestCase):
    def _runtime_request(self) -> PlanRevisionRequest:
        event = MissionEvent(
            event_id="event_path_blocked",
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
            timestamp_s=20.0,
            event_type=MissionEventType.PATH_BLOCKED,
            severity=EventSeverity.WARNING,
            payload={
                "source": "test_injection",
                "oracle_target_pose": [9.0, 8.0, 7.0],
            },
        )
        plan = original_plan()
        belief = WorldBelief(
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
            current_step_id="track_1",
            current_skill="TRACK",
            skill_feedback={"progress": 0.3},
            target_spec=plan.target_spec,
            target_snapshot=None,
            candidate_summaries=(),
            recent_events=(event,),
            qwen_request_status=QwenRequestStatus(),
            latest_frame_ref=None,
            mission_elapsed_s=20.0,
            plan_id="plan_1",
        )
        return PlanRevisionRequest(
            original_instruction="遇到持续阻塞时安全调整后缀并返回 home 降落",
            original_plan=plan,
            current_step_id="track_1",
            completed_step_ids=("takeoff_1", "goto_search", "search_1"),
            completed_step_outputs={
                "search_1": {
                    "target_id": "target_1",
                    "oracle_target_pose": [1.0, 2.0, 3.0],
                }
            },
            replaceable_step_ids=("track_1", "goto_home", "land_home"),
            world_belief=belief,
            trigger_event=event,
        )

    def _inspect_runtime_request(self) -> PlanRevisionRequest:
        plan = original_plan()
        event = MissionEvent(
            event_id="event_inspect_candidate",
            mission_id=plan.mission_id,
            uav_id=plan.uav_id,
            plan_version=plan.plan_version,
            timestamp_s=20.0,
            event_type=MissionEventType.PLAN_REVISION_REQUESTED,
            severity=EventSeverity.INFO,
            payload={
                "action": "INSPECT",
                "candidate_id": "candidate_1",
                "source": "qwen_vl",
            },
        )
        belief = WorldBelief(
            mission_id=plan.mission_id,
            uav_id=plan.uav_id,
            plan_version=plan.plan_version,
            current_step_id="search_1",
            current_skill="SEARCH",
            skill_feedback={"phase": "WAITING_FOR_REVIEW"},
            target_spec=plan.target_spec,
            target_snapshot=None,
            candidate_summaries=(
                CandidateSummary("candidate_1", 0.8, 20.0, "qwen_vl"),
            ),
            recent_events=(event,),
            qwen_request_status=QwenRequestStatus(),
            latest_frame_ref=None,
            mission_elapsed_s=20.0,
            plan_id="plan_1",
        )
        return PlanRevisionRequest(
            original_instruction="检查可信候选后继续任务并返回 home",
            original_plan=plan,
            current_step_id="search_1",
            completed_step_ids=("takeoff_1", "goto_search"),
            completed_step_outputs={},
            replaceable_step_ids=(
                "search_1",
                "track_1",
                "goto_home",
                "land_home",
            ),
            world_belief=belief,
            trigger_event=event,
            trusted_inspect_candidate_id="candidate_1",
        )

    def test_second_stage_request_is_text_only_bounded_and_has_no_truth(self) -> None:
        planner = QwenPlanRevisionPlanner(
            world_context=world(),
            skill_catalog=build_default_skill_catalog(),
            limits=PlannerLimits(),
        )
        runtime = self._runtime_request()

        request = planner.build_async_request(
            runtime,
            request_id="request_revision_1",
            review_id="review_revision_1",
        )

        self.assertEqual(request.uav_id, "uav_1")
        self.assertEqual(request.plan_version, 1)
        self.assertEqual(request.frame_id, "event_path_blocked")
        self.assertEqual(request.options.temperature, 0.0)
        self.assertEqual(request.options.max_tokens, 1024)
        self.assertIsNotNone(request.options.response_format)
        self.assertEqual(
            request.options.response_format.name,
            "qwen_plan_revision_v2",
        )
        self.assertTrue(all(isinstance(message.content, str) for message in request.messages))
        prompt = request.messages[1].content
        self.assertNotIn("9.0", prompt)
        self.assertNotIn("oracle_target_pose", prompt)
        self.assertNotIn("data:image", prompt)
        payload = json.loads(prompt)
        self.assertEqual(
            payload["trusted_revision"]["new_plan_version"],
            2,
        )
        self.assertNotIn(
            "INSPECT",
            {item["name"] for item in payload["skill_catalog"]["skills"]},
        )

    def test_oracle_target_snapshot_is_removed_from_revision_prompt(self) -> None:
        runtime = self._runtime_request()
        privileged = TargetSnapshot(
            target_id="oracle_secret_target",
            description=runtime.world_belief.target_spec.description,
            lifecycle=TargetLifecycle.TRACKING,
            confidence=1.0,
            last_seen_position=(12.0, 34.0, 0.5),
            last_seen_velocity=(1.0, 0.0, 0.0),
            last_seen_time_s=19.5,
            source="oracle",
        )
        belief = runtime.world_belief
        guarded_belief = WorldBelief(
            mission_id=belief.mission_id,
            uav_id=belief.uav_id,
            plan_version=belief.plan_version,
            current_step_id=belief.current_step_id,
            current_skill=belief.current_skill,
            skill_feedback=belief.skill_feedback,
            target_spec=belief.target_spec,
            target_snapshot=privileged,
            candidate_summaries=belief.candidate_summaries,
            recent_events=belief.recent_events,
            qwen_request_status=belief.qwen_request_status,
            latest_frame_ref=belief.latest_frame_ref,
            mission_elapsed_s=belief.mission_elapsed_s,
            plan_id=belief.plan_id,
        )
        guarded_request = PlanRevisionRequest(
            original_instruction=runtime.original_instruction,
            original_plan=runtime.original_plan,
            current_step_id=runtime.current_step_id,
            completed_step_ids=runtime.completed_step_ids,
            completed_step_outputs=runtime.completed_step_outputs,
            replaceable_step_ids=runtime.replaceable_step_ids,
            world_belief=guarded_belief,
            trigger_event=runtime.trigger_event,
        )
        planner = QwenPlanRevisionPlanner(
            world_context=world(),
            skill_catalog=build_default_skill_catalog(),
            limits=PlannerLimits(),
        )
        request = planner.build_async_request(
            guarded_request,
            request_id="request_revision_oracle",
            review_id="review_revision_oracle",
        )
        prompt = request.messages[1].content
        self.assertNotIn("oracle_secret_target", prompt)
        self.assertNotIn("12.0", prompt)
        self.assertIsNone(json.loads(prompt)["world_belief_summary"]["target_state"])

    def test_oracle_candidate_summary_and_inspect_event_are_removed_from_prompt(self) -> None:
        runtime = self._runtime_request()
        belief = runtime.world_belief
        guarded_belief = WorldBelief(
            mission_id=belief.mission_id,
            uav_id=belief.uav_id,
            plan_version=belief.plan_version,
            current_step_id=belief.current_step_id,
            current_skill=belief.current_skill,
            skill_feedback=belief.skill_feedback,
            target_spec=belief.target_spec,
            target_snapshot=belief.target_snapshot,
            candidate_summaries=(
                CandidateSummary(
                    "oracle_candidate_secret",
                    1.0,
                    19.5,
                    "oracle_evaluation",
                ),
                CandidateSummary("qwen_candidate_1", 0.7, 19.0, "qwen_vl"),
            ),
            recent_events=belief.recent_events,
            qwen_request_status=belief.qwen_request_status,
            latest_frame_ref=belief.latest_frame_ref,
            mission_elapsed_s=belief.mission_elapsed_s,
            plan_id=belief.plan_id,
        )
        oracle_inspect = MissionEvent(
            event_id="event_oracle_inspect",
            mission_id=belief.mission_id,
            uav_id=belief.uav_id,
            plan_version=belief.plan_version,
            timestamp_s=20.0,
            event_type=MissionEventType.PLAN_REVISION_REQUESTED,
            severity=EventSeverity.WARNING,
            payload={
                "action": "INSPECT",
                "candidate_id": "oracle_candidate_secret",
                "source": "oracle_evaluation",
                "oracle_target_pose": [1.0, 2.0, 3.0],
            },
        )
        guarded_request = PlanRevisionRequest(
            original_instruction=runtime.original_instruction,
            original_plan=runtime.original_plan,
            current_step_id=runtime.current_step_id,
            completed_step_ids=runtime.completed_step_ids,
            completed_step_outputs=runtime.completed_step_outputs,
            replaceable_step_ids=runtime.replaceable_step_ids,
            world_belief=guarded_belief,
            trigger_event=oracle_inspect,
        )
        planner = QwenPlanRevisionPlanner(
            world_context=world(),
            skill_catalog=build_default_skill_catalog(),
            limits=PlannerLimits(),
        )
        request = planner.build_async_request(
            guarded_request,
            request_id="request_revision_oracle_candidate",
            review_id="review_revision_oracle_candidate",
        )
        prompt = request.messages[1].content
        self.assertNotIn("oracle_candidate_secret", prompt)
        self.assertNotIn("oracle_evaluation", prompt)
        self.assertNotIn("oracle_target_pose", prompt)
        payload = json.loads(prompt)
        self.assertEqual(
            payload["world_belief_summary"]["candidates"],
            [
                {
                    "candidate_id": "qwen_candidate_1",
                    "lifecycle": None,
                    "source": "qwen_vl",
                    "confidence": 0.7,
                }
            ],
        )
        self.assertNotIn("candidate_id", payload["trigger_event"])

    def test_strict_result_parser_binds_route_and_version(self) -> None:
        planner = QwenPlanRevisionPlanner(
            world_context=world(),
            skill_catalog=build_default_skill_catalog(),
            limits=PlannerLimits(),
        )
        runtime = self._runtime_request()
        request = planner.build_async_request(
            runtime,
            request_id="request_revision_1",
            review_id="review_revision_1",
        )
        output = revision_dict()
        result = AsyncModelResult(
            request_id=request.request_id,
            review_id=request.review_id,
            mission_id=request.mission_id,
            uav_id=request.uav_id,
            plan_version=request.plan_version,
            observation_timestamp_s=request.observation_timestamp_s,
            frame_id=request.frame_id,
            response=ModelResponse(
                content=json.dumps(output),
                model="fake-qwen",
                finish_reason="stop",
                usage={},
            ),
            error_code=None,
            error_message=None,
        )

        parsed = planner.parse_async_result(
            result,
            revision_request=runtime,
            expected_request_id=request.request_id,
            expected_review_id=request.review_id,
        )
        self.assertEqual(parsed.new_plan_version, 2)

        stale = AsyncModelResult(
            request_id=request.request_id,
            review_id=request.review_id,
            mission_id=request.mission_id,
            uav_id=request.uav_id,
            plan_version=request.plan_version,
            observation_timestamp_s=request.observation_timestamp_s,
            frame_id=request.frame_id,
            response=result.response,
            error_code=None,
            error_message=None,
            stale=True,
        )
        with self.assertRaises(ModelProtocolError):
            planner.parse_async_result(
                stale,
                revision_request=runtime,
                expected_request_id=request.request_id,
                expected_review_id=request.review_id,
            )

    def test_inspect_result_must_echo_the_candidate_bank_identifier(self) -> None:
        planner = QwenPlanRevisionPlanner(
            world_context=world(),
            skill_catalog=build_default_skill_catalog(),
            limits=PlannerLimits(),
        )
        runtime = self._inspect_runtime_request()
        request = planner.build_async_request(
            runtime,
            request_id="request_inspect_1",
            review_id="review_inspect_1",
        )
        prompt_payload = json.loads(request.messages[1].content)
        prompt_inspect = next(
            item
            for item in prompt_payload["skill_catalog"]["skills"]
            if item["name"] == "INSPECT"
        )
        prompt_candidate = next(
            item
            for item in prompt_inspect["arguments"]
            if item["name"] == "candidate_id"
        )
        self.assertEqual(prompt_candidate["allowed_values"], ["candidate_1"])
        self.assertEqual(
            prompt_payload["trigger_event"]["candidate_id"],
            "candidate_1",
        )
        variants = request.options.response_format.schema["properties"]["steps"]["items"]["oneOf"]  # type: ignore[union-attr]
        inspect_variant = next(
            variant
            for variant in variants
            if variant["properties"]["skill"]["const"] == "INSPECT"
        )
        self.assertEqual(
            inspect_variant["properties"]["args"]["properties"]["candidate_id"]["const"],
            "candidate_1",
        )

        output = revision_dict(
            replace_from="search_1",
            steps=[
                step(
                    "search_1",
                    "SEARCH",
                    {
                        "region": "search_area",
                        "target_description": "red moving target",
                        "altitude_m": 10.0,
                    },
                ),
                step(
                    "inspect_1",
                    "INSPECT",
                    {"candidate_id": "candidate_wrong"},
                ),
                step("goto_home", "GOTO", {"destination": "home"}),
                step("land_home", "LAND", {"zone": "home"}),
            ],
        )
        result = AsyncModelResult(
            request_id=request.request_id,
            review_id=request.review_id,
            mission_id=request.mission_id,
            uav_id=request.uav_id,
            plan_version=request.plan_version,
            observation_timestamp_s=request.observation_timestamp_s,
            frame_id=request.frame_id,
            response=ModelResponse(
                content=json.dumps(output),
                model="fake-qwen",
                finish_reason="stop",
                usage={},
            ),
            error_code=None,
            error_message=None,
        )
        with self.assertRaisesRegex(ModelProtocolError, "CandidateBank identifier"):
            planner.parse_async_result(
                result,
                revision_request=runtime,
                expected_request_id=request.request_id,
                expected_review_id=request.review_id,
            )


class PlanRevisionValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = world()
        self.original = original_plan()
        self.compiler = PlanValidator()
        self.compiled_original = self.compiler.validate_and_compile(
            self.original,
            self.world,
            source="dynamic_scripted",
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=1,
        )
        self.validator = RevisionValidator(self.compiler)

    def validate(self, draft: PlanRevisionDraft, **overrides: object):
        values = {
            "original": self.compiled_original,
            "world_context": self.world,
            "current_step_id": "track_1",
            "completed_step_ids": ("takeoff_1", "goto_search", "search_1"),
            "completed_step_outputs": {
                "search_1": {"target_id": "target_1"}
            },
            "revision_count": 0,
            "now_s": 20.0,
            "expected_new_plan_version": 2,
            "source": "dynamic_llm",
        }
        values.update(overrides)
        return self.validator.validate_and_apply(draft, **values)

    def test_valid_revision_atomically_replaces_current_suffix(self) -> None:
        original_before = self.original.to_dict()
        result = self.validate(PlanRevisionDraft.from_dict(revision_dict()))

        self.assertEqual(result.revised_plan.plan_version, 2)
        self.assertEqual(
            [item.id for item in result.revised_plan.steps],
            [
                "takeoff_1",
                "goto_search",
                "search_1",
                "track_revised",
                "goto_home_revised",
                "land_home_revised",
            ],
        )
        self.assertEqual(result.compiled_mission.task_plan.plan_version, 2)
        self.assertIs(result.revised_plan.target_spec, self.original.target_spec)
        self.assertEqual(result.revision_count, 1)
        self.assertEqual(result.added_step_count, 0)
        self.assertEqual(self.original.to_dict(), original_before)

    def test_revision_validator_requires_exact_trusted_inspect_candidate(self) -> None:
        inspect_revision = PlanRevisionDraft.from_dict(
            revision_dict(
                steps=[
                    step(
                        "inspect_runtime",
                        "INSPECT",
                        {"candidate_id": "candidate_1"},
                    ),
                    step("goto_home_new", "GOTO", {"destination": "home"}),
                    step("land_home_new", "LAND", {"zone": "home"}),
                ]
            )
        )
        for trusted in (None, "candidate_other"):
            with self.subTest(trusted=trusted), self.assertRaises(
                RevisionValidationError
            ) as raised:
                self.validate(
                    inspect_revision,
                    trusted_inspect_candidate_id=trusted,
                )
            self.assertEqual(
                raised.exception.code,
                RevisionErrorCode.INSPECT_CANDIDATE_UNTRUSTED,
            )

        accepted = self.validate(
            inspect_revision,
            trusted_inspect_candidate_id="candidate_1",
        )
        self.assertEqual(
            accepted.compiled_mission.task_plan.steps[3].params["candidate_id"],
            "candidate_1",
        )

    def test_completed_prefix_and_outputs_are_immutable(self) -> None:
        outputs = {"search_1": {"target_id": "target_1", "scores": [1, 2]}}
        result = self.validate(
            PlanRevisionDraft.from_dict(revision_dict()),
            completed_step_outputs=outputs,
        )
        outputs["search_1"]["target_id"] = "rewritten"
        self.assertEqual(
            result.completed_step_outputs["search_1"]["target_id"],
            "target_1",
        )
        self.assertIsInstance(result.completed_step_outputs, MappingProxyType)
        with self.assertRaises(TypeError):
            result.completed_step_outputs["search_1"] = {}  # type: ignore[index]

        with self.assertRaises(RevisionValidationError) as raised:
            self.validate(
                PlanRevisionDraft.from_dict(revision_dict()),
                completed_step_ids=("takeoff_1", "goto_search"),
            )
        self.assertEqual(
            raised.exception.code,
            RevisionErrorCode.COMPLETED_PREFIX_MISMATCH,
        )

    def test_completed_prefix_cannot_be_replaced(self) -> None:
        draft = PlanRevisionDraft.from_dict(
            revision_dict(replace_from="search_1")
        )
        before = self.original.to_dict()
        with self.assertRaises(RevisionValidationError) as raised:
            self.validate(draft)
        self.assertEqual(raised.exception.code, RevisionErrorCode.REPLACE_STEP_INVALID)
        self.assertEqual(self.original.to_dict(), before)

    def test_stale_wrong_uav_and_trusted_version_are_rejected(self) -> None:
        examples = (
            (
                PlanRevisionDraft.from_dict(
                    revision_dict(base_version=2, new_version=3)
                ),
                RevisionErrorCode.STALE_REVISION,
                {},
            ),
            (
                PlanRevisionDraft.from_dict(
                    revision_dict(
                        mission_id="mission_1",
                        uav_id="uav_2",
                    )
                ),
                RevisionErrorCode.ROUTING_MISMATCH,
                {},
            ),
            (
                PlanRevisionDraft.from_dict(revision_dict()),
                RevisionErrorCode.VERSION_MISMATCH,
                {"expected_new_plan_version": 3},
            ),
        )
        for draft, code, overrides in examples:
            with self.subTest(code=code), self.assertRaises(
                RevisionValidationError
            ) as raised:
                self.validate(draft, **overrides)
            self.assertEqual(raised.exception.code, code)

    def test_invalid_suffix_does_not_mutate_original(self) -> None:
        invalid = PlanRevisionDraft.from_dict(
            revision_dict(
                steps=[step("land_too_early", "LAND", {"zone": "home"})]
            )
        )
        original_before = self.original.to_dict()
        with self.assertRaises(RevisionValidationError) as raised:
            self.validate(invalid)
        self.assertEqual(
            raised.exception.code,
            RevisionErrorCode.SYMBOLIC_PLAN_INVALID,
        )
        self.assertEqual(self.original.to_dict(), original_before)

    def test_revision_cannot_drift_immutable_target_identity(self) -> None:
        changed_target = PlanRevisionDraft.from_dict(
            revision_dict(
                replace_from="search_1",
                steps=[
                    step(
                        "search_changed",
                        "SEARCH",
                        {
                            "region": "search_area",
                            "target_description": "different blue target",
                            "altitude_m": 10.0,
                        },
                    ),
                    step(
                        "track_changed",
                        "TRACK",
                        {
                            "target_ref": "$search_changed.target_id",
                            "duration_s": 10.0,
                        },
                    ),
                    step(
                        "goto_home_changed",
                        "GOTO",
                        {"destination": "home", "altitude_m": 10.0},
                    ),
                    step("land_home_changed", "LAND", {"zone": "home"}),
                ],
            )
        )
        with self.assertRaises(RevisionValidationError) as raised:
            self.validator.validate_and_apply(
                changed_target,
                original=self.original,
                world_context=self.world,
                current_step_id="search_1",
                completed_step_ids=("takeoff_1", "goto_search"),
            )
        self.assertEqual(
            raised.exception.code,
            RevisionErrorCode.TARGET_IDENTITY_MUTATION,
        )

    def test_revision_count_cooldown_and_added_step_budgets(self) -> None:
        draft = PlanRevisionDraft.from_dict(revision_dict())
        with self.assertRaises(RevisionValidationError) as raised:
            self.validate(draft, revision_count=3)
        self.assertEqual(
            raised.exception.code,
            RevisionErrorCode.REVISION_BUDGET_EXCEEDED,
        )

        with self.assertRaises(RevisionValidationError) as raised:
            self.validate(draft, revision_count=1)
        self.assertEqual(raised.exception.code, RevisionErrorCode.REVISION_COOLDOWN)

        with self.assertRaises(RevisionValidationError) as raised:
            self.validate(
                draft,
                revision_count=1,
                now_s=20.0,
                last_revision_timestamp_s=18.0,
            )
        self.assertEqual(raised.exception.code, RevisionErrorCode.REVISION_COOLDOWN)

        strict = RevisionValidator(
            self.compiler,
            revision_limits=RevisionLimits(
                max_plan_revisions=3,
                cooldown_s=0.0,
                max_added_steps_per_revision=0,
                max_total_plan_steps=10,
            ),
        )
        expanded = PlanRevisionDraft.from_dict(
            revision_dict(
                replace_from="goto_home",
                steps=[
                    step(
                        "goto_observation",
                        "GOTO",
                        {"destination": "observation_point", "altitude_m": 10.0},
                    ),
                    step(
                        "goto_home_new",
                        "GOTO",
                        {"destination": "home", "altitude_m": 10.0},
                    ),
                    step("land_home_new", "LAND", {"zone": "home"}),
                ],
            )
        )
        with self.assertRaises(RevisionValidationError) as raised:
            strict.validate_and_apply(
                expanded,
                original=self.original,
                world_context=self.world,
                current_step_id="track_1",
                completed_step_ids=("takeoff_1", "goto_search", "search_1"),
                now_s=20.0,
            )
        self.assertEqual(
            raised.exception.code,
            RevisionErrorCode.ADDED_STEP_BUDGET_EXCEEDED,
        )

    def test_circular_completed_output_is_rejected_without_mutation(self) -> None:
        circular: dict[str, object] = {}
        circular["self"] = circular
        before = self.original.to_dict()
        with self.assertRaises(RevisionValidationError) as raised:
            self.validate(
                PlanRevisionDraft.from_dict(revision_dict()),
                completed_step_outputs={"search_1": circular},
            )
        self.assertEqual(
            raised.exception.code,
            RevisionErrorCode.COMPLETED_OUTPUT_INVALID,
        )
        self.assertEqual(self.original.to_dict(), before)

    def test_land_started_forbids_ordinary_revision(self) -> None:
        draft = PlanRevisionDraft.from_dict(
            revision_dict(
                replace_from="land_home",
                steps=[step("land_new", "LAND", {"zone": "home"})],
            )
        )
        with self.assertRaises(RevisionValidationError) as raised:
            self.validator.validate_and_apply(
                draft,
                original=self.original,
                world_context=self.world,
                current_step_id="land_home",
                completed_step_ids=(
                    "takeoff_1",
                    "goto_search",
                    "search_1",
                    "track_1",
                    "goto_home",
                ),
            )
        self.assertEqual(
            raised.exception.code,
            RevisionErrorCode.REVISION_DURING_LAND,
        )

    def test_safety_preflight_callback_and_object_are_supported(self) -> None:
        draft = PlanRevisionDraft.from_dict(revision_dict())
        seen = []

        def callback(compiled):
            seen.append(compiled.task_plan.plan_version)
            return True

        accepted = self.validate(draft, safety_preflight=callback)
        self.assertEqual(accepted.revised_plan.plan_version, 2)
        self.assertEqual(seen, [2])

        class RejectingSupervisor:
            class Decision:
                action = "ABORT"

            def preflight(self, compiled):
                del compiled
                return self.Decision()

        with self.assertRaises(RevisionValidationError) as raised:
            self.validate(draft, safety_preflight=RejectingSupervisor())
        self.assertEqual(
            raised.exception.code,
            RevisionErrorCode.SAFETY_PREFLIGHT_REJECTED,
        )

    def test_atomic_function_accepts_future_suffix_only(self) -> None:
        draft = PlanRevisionDraft.from_dict(
            revision_dict(
                replace_from="goto_home",
                steps=[
                    step(
                        "goto_home_new",
                        "GOTO",
                        {"destination": "home", "altitude_m": 10.0},
                    ),
                    step("land_home_new", "LAND", {"zone": "home"}),
                ],
            )
        )
        result = replace_plan_suffix(
            self.original,
            draft,
            current_step_id="track_1",
        )
        self.assertEqual(result.steps[3].id, "track_1")
        self.assertEqual(result.steps[4].id, "goto_home_new")


if __name__ == "__main__":
    unittest.main()
