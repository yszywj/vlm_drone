"""Strict schema-v2 plan-suffix revisions and trusted atomic validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from numbers import Real
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from common.ids import (
    validate_mission_id,
    validate_request_id,
    validate_review_id,
    validate_routing_id,
    validate_uav_id,
)
from common.provenance import is_privileged_oracle_source
from models import (
    AsyncModelRequest,
    AsyncModelResult,
    ChatMessage,
    GenerationOptions,
    JsonSchemaResponseFormat,
    ModelProtocolError,
)
from planner.json_schema import build_skill_plan_v2_json_schema
from planner.policy import PlannerLimits, PlannerPolicy
from planner.prompt_builder import build_dynamic_skill_planner_messages
from planner.schemas import (
    CompiledMission,
    PlanStepDraftV2,
    PlannerWorldContext,
    SkillPlanDraftV2,
)
from planner.skill_catalog import SkillCatalog, revision_planner_catalog
from planner.symbolic_checker import SymbolicPlanChecker

if TYPE_CHECKING:
    from runtime.events import MissionEvent
    from runtime.world_belief import WorldBelief


_STEP_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_REASON_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def _positive_version(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _step_id(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _STEP_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must match ^[a-z][a-z0-9_]{{0,31}}$"
        )
    return value


def _finite_timestamp(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return normalized


@dataclass(frozen=True, slots=True)
class PlanRevisionDraft:
    """Untrusted schema-v2 replacement for one current-or-future suffix."""

    schema_version: int
    mission_id: str
    uav_id: str
    base_plan_version: int
    new_plan_version: int
    replace_from_step_id: str
    steps: tuple[PlanStepDraftV2, ...]
    reason_codes: tuple[str, ...]

    _REQUIRED_FIELDS = frozenset(
        {
            "schema_version",
            "mission_id",
            "uav_id",
            "base_plan_version",
            "new_plan_version",
            "replace_from_step_id",
            "steps",
            "reason_codes",
        }
    )

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version,
            int,
        ):
            raise TypeError("schema_version must be the integer 2")
        if self.schema_version != 2:
            raise ValueError("schema_version must equal 2")
        object.__setattr__(
            self,
            "mission_id",
            validate_mission_id(self.mission_id),
        )
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        base_version = _positive_version(
            self.base_plan_version,
            "base_plan_version",
        )
        new_version = _positive_version(
            self.new_plan_version,
            "new_plan_version",
        )
        if new_version != base_version + 1:
            raise ValueError(
                "new_plan_version must equal base_plan_version + 1"
            )
        object.__setattr__(self, "base_plan_version", base_version)
        object.__setattr__(self, "new_plan_version", new_version)
        object.__setattr__(
            self,
            "replace_from_step_id",
            _step_id(self.replace_from_step_id, "replace_from_step_id"),
        )

        if isinstance(self.steps, (str, bytes)) or not isinstance(
            self.steps,
            Sequence,
        ):
            raise TypeError("steps must be an array of PlanStepDraftV2 values")
        steps = tuple(self.steps)
        if not 1 <= len(steps) <= 10:
            raise ValueError("steps must contain between 1 and 10 entries")
        if any(not isinstance(step, PlanStepDraftV2) for step in steps):
            raise TypeError("steps must contain only PlanStepDraftV2 values")
        if any(step.uav_id != self.uav_id for step in steps):
            raise ValueError("every revision step.uav_id must equal uav_id")
        object.__setattr__(self, "steps", steps)

        if isinstance(self.reason_codes, (str, bytes)) or not isinstance(
            self.reason_codes,
            Sequence,
        ):
            raise TypeError("reason_codes must be an array of strings")
        reason_codes = tuple(self.reason_codes)
        if not 1 <= len(reason_codes) <= 8:
            raise ValueError("reason_codes must contain between 1 and 8 entries")
        for code in reason_codes:
            if not isinstance(code, str) or _REASON_CODE_PATTERN.fullmatch(code) is None:
                raise ValueError(
                    "each reason code must match ^[A-Z][A-Z0-9_]{0,63}$"
                )
        if len(reason_codes) != len(set(reason_codes)):
            raise ValueError("reason_codes must not contain duplicates")
        object.__setattr__(self, "reason_codes", reason_codes)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PlanRevisionDraft:
        if not isinstance(data, Mapping):
            raise TypeError("PlanRevisionDraft input must be a mapping")
        if any(not isinstance(key, str) for key in data):
            raise TypeError("PlanRevisionDraft field names must be strings")
        keys = frozenset(data)
        unknown = keys - cls._REQUIRED_FIELDS
        missing = cls._REQUIRED_FIELDS - keys
        if unknown:
            raise ValueError(
                "PlanRevisionDraft contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        if missing:
            raise ValueError(
                "PlanRevisionDraft is missing required fields: "
                + ", ".join(sorted(missing))
            )
        raw_steps = data["steps"]
        if isinstance(raw_steps, (str, bytes)) or not isinstance(
            raw_steps,
            Sequence,
        ):
            raise TypeError("PlanRevisionDraft.steps must be an array")
        raw_reasons = data["reason_codes"]
        if isinstance(raw_reasons, (str, bytes)) or not isinstance(
            raw_reasons,
            Sequence,
        ):
            raise TypeError("PlanRevisionDraft.reason_codes must be an array")
        return cls(
            schema_version=data["schema_version"],
            mission_id=data["mission_id"],
            uav_id=data["uav_id"],
            base_plan_version=data["base_plan_version"],
            new_plan_version=data["new_plan_version"],
            replace_from_step_id=data["replace_from_step_id"],
            steps=tuple(PlanStepDraftV2.from_dict(step) for step in raw_steps),
            reason_codes=tuple(raw_reasons),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "base_plan_version": self.base_plan_version,
            "new_plan_version": self.new_plan_version,
            "replace_from_step_id": self.replace_from_step_id,
            "steps": [step.to_dict() for step in self.steps],
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class RevisionLimits:
    """Trusted bounded resource policy for runtime plan revision."""

    max_plan_revisions: int = 3
    cooldown_s: float = 5.0
    max_added_steps_per_revision: int = 3
    max_total_plan_steps: int = 10

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_plan_revisions, bool)
            or not isinstance(self.max_plan_revisions, int)
            or not 1 <= self.max_plan_revisions <= 3
        ):
            raise ValueError("max_plan_revisions must be between 1 and 3")
        cooldown = _finite_timestamp(self.cooldown_s, "cooldown_s")
        object.__setattr__(self, "cooldown_s", cooldown)
        if (
            isinstance(self.max_added_steps_per_revision, bool)
            or not isinstance(self.max_added_steps_per_revision, int)
            or self.max_added_steps_per_revision < 0
        ):
            raise ValueError(
                "max_added_steps_per_revision must be a non-negative integer"
            )
        if (
            isinstance(self.max_total_plan_steps, bool)
            or not isinstance(self.max_total_plan_steps, int)
            or not 2 <= self.max_total_plan_steps <= 10
        ):
            raise ValueError("max_total_plan_steps must be between 2 and 10")


class RevisionErrorCode(str, Enum):
    ROUTING_MISMATCH = "ROUTING_MISMATCH"
    STALE_REVISION = "STALE_REVISION"
    VERSION_MISMATCH = "VERSION_MISMATCH"
    CURRENT_STEP_INVALID = "CURRENT_STEP_INVALID"
    REPLACE_STEP_INVALID = "REPLACE_STEP_INVALID"
    COMPLETED_PREFIX_MISMATCH = "COMPLETED_PREFIX_MISMATCH"
    COMPLETED_OUTPUT_INVALID = "COMPLETED_OUTPUT_INVALID"
    REVISION_DURING_LAND = "REVISION_DURING_LAND"
    REVISION_BUDGET_EXCEEDED = "REVISION_BUDGET_EXCEEDED"
    REVISION_COOLDOWN = "REVISION_COOLDOWN"
    ADDED_STEP_BUDGET_EXCEEDED = "ADDED_STEP_BUDGET_EXCEEDED"
    TOTAL_STEP_BUDGET_EXCEEDED = "TOTAL_STEP_BUDGET_EXCEEDED"
    TARGET_IDENTITY_MUTATION = "TARGET_IDENTITY_MUTATION"
    INSPECT_CANDIDATE_UNTRUSTED = "INSPECT_CANDIDATE_UNTRUSTED"
    SYMBOLIC_PLAN_INVALID = "SYMBOLIC_PLAN_INVALID"
    PLAN_COMPILE_FAILED = "PLAN_COMPILE_FAILED"
    SAFETY_PREFLIGHT_REJECTED = "SAFETY_PREFLIGHT_REJECTED"


class RevisionValidationError(ValueError):
    """Stable fail-closed revision rejection for trusted fallback handling."""

    def __init__(self, code: RevisionErrorCode, message: str) -> None:
        if not isinstance(code, RevisionErrorCode):
            raise TypeError("code must be a RevisionErrorCode")
        super().__init__(message)
        self.code = code


class _PlanValidatorProtocol(Protocol):
    @property
    def limits(self) -> PlannerLimits: ...

    @property
    def policy(self) -> PlannerPolicy: ...

    def validate_and_compile(
        self,
        planner_output: object,
        context: PlannerWorldContext,
        *,
        source: str,
        mission_id: str,
        uav_id: str,
        plan_version: int,
        trusted_inspect_candidate_ids: Sequence[str] = (),
    ) -> CompiledMission: ...


SafetyPreflight = Callable[[CompiledMission], object] | object


@dataclass(frozen=True, slots=True)
class ValidatedPlanRevision:
    """An accepted compiled suffix replacement plus immutable prior outputs."""

    revision: PlanRevisionDraft
    revised_plan: SkillPlanDraftV2
    compiled_mission: CompiledMission
    completed_step_outputs: Mapping[str, object]
    replace_from_index: int
    added_step_count: int
    revision_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.revision, PlanRevisionDraft):
            raise TypeError("revision must be a PlanRevisionDraft")
        if not isinstance(self.revised_plan, SkillPlanDraftV2):
            raise TypeError("revised_plan must be a SkillPlanDraftV2")
        if not isinstance(self.compiled_mission, CompiledMission):
            raise TypeError("compiled_mission must be a CompiledMission")
        object.__setattr__(
            self,
            "completed_step_outputs",
            _freeze_json_mapping(self.completed_step_outputs),
        )


def build_plan_revision_json_schema(
    *,
    world_context: PlannerWorldContext,
    skill_catalog: SkillCatalog,
    limits: PlannerLimits,
    mission_id: str,
    uav_id: str,
    base_plan_version: int,
    new_plan_version: int,
    replaceable_step_ids: Sequence[str],
    trusted_inspect_candidate_id: str | None = None,
) -> dict[str, object]:
    """Build a strict revision-only schema bound to trusted runtime values."""

    trusted_mission_id = validate_mission_id(mission_id)
    trusted_uav_id = validate_uav_id(uav_id)
    base_version = _positive_version(base_plan_version, "base_plan_version")
    new_version = _positive_version(new_plan_version, "new_plan_version")
    if new_version != base_version + 1:
        raise ValueError("new_plan_version must equal base_plan_version + 1")
    if isinstance(replaceable_step_ids, (str, bytes)) or not isinstance(
        replaceable_step_ids,
        Sequence,
    ):
        raise TypeError("replaceable_step_ids must be a sequence")
    replaceable = tuple(
        _step_id(value, f"replaceable_step_ids[{index}]")
        for index, value in enumerate(replaceable_step_ids)
    )
    if not replaceable:
        raise ValueError("replaceable_step_ids must not be empty")
    if len(replaceable) != len(set(replaceable)):
        raise ValueError("replaceable_step_ids must be unique")
    trusted_candidate_id = (
        None
        if trusted_inspect_candidate_id is None
        else validate_routing_id(
            trusted_inspect_candidate_id,
            "trusted_inspect_candidate_id",
        )
    )

    full_plan_schema = build_skill_plan_v2_json_schema(
        world_context=world_context,
        skill_catalog=skill_catalog,
        limits=limits,
        mission_id=trusted_mission_id,
        uav_id=trusted_uav_id,
        plan_version=new_version,
        _trusted_inspect_candidate_id=trusted_candidate_id,
    )
    steps_schema = deepcopy(full_plan_schema["properties"]["steps"])
    steps_schema["minItems"] = 1
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "integer", "const": 2},
            "mission_id": {"type": "string", "const": trusted_mission_id},
            "uav_id": {"type": "string", "const": trusted_uav_id},
            "base_plan_version": {"type": "integer", "const": base_version},
            "new_plan_version": {"type": "integer", "const": new_version},
            "replace_from_step_id": {
                "type": "string",
                "enum": list(replaceable),
            },
            "steps": steps_schema,
            # Do not emit JSON-Schema ``uniqueItems``: the deployed vLLM
            # grammar backend fails that keyword with HTTP 500.  The strict
            # PlanRevisionDraft parser below independently rejects duplicates.
            "reason_codes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "string",
                    "pattern": "^[A-Z][A-Z0-9_]{0,63}$",
                },
            },
        },
        "required": [
            "schema_version",
            "mission_id",
            "uav_id",
            "base_plan_version",
            "new_plan_version",
            "replace_from_step_id",
            "steps",
            "reason_codes",
        ],
        "additionalProperties": False,
    }


@dataclass(frozen=True, slots=True)
class PlanRevisionRequest:
    """Safe text-only input for the second, independent planner call.

    The triggering visual review is intentionally not allowed to carry a
    replacement plan.  It can only cause the main thread to construct this
    request from the authoritative active plan, WorldBelief and MissionEvent.
    Pixel arrays and event payloads never enter this value.
    """

    original_instruction: str
    original_plan: SkillPlanDraftV2
    current_step_id: str
    completed_step_ids: tuple[str, ...]
    completed_step_outputs: Mapping[str, object]
    replaceable_step_ids: tuple[str, ...]
    world_belief: "WorldBelief"
    trigger_event: "MissionEvent"
    trusted_inspect_candidate_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.original_instruction, str):
            raise TypeError("original_instruction must be a string")
        instruction = self.original_instruction.strip()
        if not instruction or len(instruction) > 4096:
            raise ValueError(
                "original_instruction must contain between 1 and 4096 characters"
            )
        object.__setattr__(self, "original_instruction", instruction)
        if not isinstance(self.original_plan, SkillPlanDraftV2):
            raise TypeError("original_plan must be a SkillPlanDraftV2")
        current = _step_id(self.current_step_id, "current_step_id")
        object.__setattr__(self, "current_step_id", current)
        completed = _validated_step_id_sequence(
            self.completed_step_ids,
            "completed_step_ids",
            allow_empty=True,
        )
        replaceable = _validated_step_id_sequence(
            self.replaceable_step_ids,
            "replaceable_step_ids",
            allow_empty=False,
        )
        object.__setattr__(self, "completed_step_ids", completed)
        object.__setattr__(self, "replaceable_step_ids", replaceable)

        plan_ids = tuple(step.id for step in self.original_plan.steps)
        if current not in plan_ids:
            raise ValueError("current_step_id is not present in original_plan")
        current_index = plan_ids.index(current)
        if completed != plan_ids[:current_index]:
            raise ValueError(
                "completed_step_ids must equal the immutable completed prefix"
            )
        if any(step_id not in plan_ids for step_id in replaceable):
            raise ValueError("replaceable_step_ids contains an unknown step")
        if any(plan_ids.index(step_id) < current_index for step_id in replaceable):
            raise ValueError(
                "replaceable_step_ids may contain only current or future steps"
            )
        if current not in replaceable:
            raise ValueError("replaceable_step_ids must include current_step_id")

        outputs = _freeze_json_mapping(self.completed_step_outputs)
        if set(outputs) - set(completed):
            raise ValueError(
                "completed_step_outputs may name only completed-prefix steps"
            )
        object.__setattr__(self, "completed_step_outputs", outputs)

        belief = self.world_belief
        event = self.trigger_event
        for value, label in ((belief, "world_belief"), (event, "trigger_event")):
            if not callable(getattr(value, "to_dict", None)):
                raise TypeError(f"{label} must provide to_dict()")
        _require_revision_route(
            belief,
            mission_id=self.original_plan.mission_id,
            uav_id=self.original_plan.uav_id,
            plan_version=self.original_plan.plan_version,
            label="world_belief",
        )
        _require_revision_route(
            event,
            mission_id=self.original_plan.mission_id,
            uav_id=self.original_plan.uav_id,
            plan_version=self.original_plan.plan_version,
            label="trigger_event",
        )
        if getattr(belief, "current_step_id", None) != current:
            raise ValueError("world_belief current_step_id does not match request")
        belief_target_spec = getattr(belief, "target_spec", None)
        if not isinstance(belief_target_spec, type(self.original_plan.target_spec)):
            raise TypeError("world_belief must contain a TargetSpec")
        if (
            belief_target_spec.original_description
            != self.original_plan.target_spec.original_description
            or belief_target_spec.immutable_identity_summary
            != self.original_plan.target_spec.immutable_identity_summary
        ):
            raise ValueError("world_belief immutable target identity has drifted")
        trusted_candidate_id = self.trusted_inspect_candidate_id
        if trusted_candidate_id is not None:
            trusted_candidate_id = validate_routing_id(
                trusted_candidate_id,
                "trusted_inspect_candidate_id",
            )
            payload = getattr(event, "payload", None)
            if (
                not isinstance(payload, Mapping)
                or payload.get("action") != "INSPECT"
                or payload.get("candidate_id") != trusted_candidate_id
                or is_privileged_oracle_source(payload.get("source"))
            ):
                raise ValueError(
                    "trusted_inspect_candidate_id must match a non-Oracle "
                    "trusted trigger event"
                )
        object.__setattr__(
            self,
            "trusted_inspect_candidate_id",
            trusted_candidate_id,
        )

    @property
    def anchor_frame_id(self) -> str:
        latest = getattr(self.world_belief, "latest_frame_ref", None)
        if latest is not None:
            return validate_routing_id(latest.frame_id, "frame_id")
        return validate_routing_id(self.trigger_event.event_id, "event_id")

    @property
    def anchor_timestamp_s(self) -> float:
        latest = getattr(self.world_belief, "latest_frame_ref", None)
        if latest is not None:
            return _finite_timestamp(latest.timestamp_s, "frame timestamp")
        return _finite_timestamp(
            self.trigger_event.timestamp_s,
            "event timestamp",
        )


class QwenPlanRevisionPlanner:
    """Create and parse an asynchronous, text-only suffix-revision request.

    This object never performs HTTP itself.  ``build_async_request`` is called
    by the MissionAgent/main thread, the returned request is submitted to an
    ``AsyncModelWorker``, and ``parse_async_result`` is later called while
    polling.  Consequently no simulation tick waits for Qwen.
    """

    SYSTEM_PROMPT = (
        "You are a constrained UAV plan-revision planner. Return exactly one "
        "JSON object matching the supplied PlanRevisionDraft schema. Replace "
        "only the requested current-or-future suffix; never alter the completed "
        "prefix, routing IDs, plan versions, immutable target identity, or "
        "completed outputs. Use only the supplied Skill Catalog and named "
        "locations. Do not output world coordinates, target truth, Oracle data, "
        "images, velocities, controller commands, waypoints, PID gains, or "
        "Markdown. A visual review recommendation is not itself a plan. If the "
        "trusted trigger action is INSPECT, retain the current SEARCH step "
        "unchanged and place exactly that candidate_id's INSPECT immediately "
        "after it; never invent or alter a candidate ID."
    )

    def __init__(
        self,
        *,
        world_context: PlannerWorldContext,
        skill_catalog: SkillCatalog,
        limits: PlannerLimits,
        policy: PlannerPolicy | None = None,
        max_tokens: int = 1024,
    ) -> None:
        if not isinstance(world_context, PlannerWorldContext):
            raise TypeError("world_context must be a PlannerWorldContext")
        if not isinstance(skill_catalog, SkillCatalog):
            raise TypeError("skill_catalog must be a SkillCatalog")
        if not isinstance(limits, PlannerLimits):
            raise TypeError("limits must be PlannerLimits")
        selected_policy = PlannerPolicy() if policy is None else policy
        if not isinstance(selected_policy, PlannerPolicy):
            raise TypeError("policy must be PlannerPolicy or None")
        selected_policy.validate_against(limits)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if not 128 <= max_tokens <= 8192:
            raise ValueError("max_tokens must be between 128 and 8192")
        self._world_context = world_context
        self._skill_catalog = skill_catalog
        self._limits = limits
        self._policy = selected_policy
        self._max_tokens = max_tokens

    def build_async_request(
        self,
        revision_request: PlanRevisionRequest,
        *,
        request_id: str,
        review_id: str,
    ) -> AsyncModelRequest:
        if not isinstance(revision_request, PlanRevisionRequest):
            raise TypeError("revision_request must be a PlanRevisionRequest")
        request_id = validate_request_id(request_id)
        review_id = validate_review_id(review_id)
        plan = revision_request.original_plan
        new_version = plan.plan_version + 1
        schema = build_plan_revision_json_schema(
            world_context=self._world_context,
            skill_catalog=self._skill_catalog,
            limits=self._limits,
            mission_id=plan.mission_id,
            uav_id=plan.uav_id,
            base_plan_version=plan.plan_version,
            new_plan_version=new_version,
            replaceable_step_ids=revision_request.replaceable_step_ids,
            trusted_inspect_candidate_id=(
                revision_request.trusted_inspect_candidate_id
            ),
        )
        payload = self._prompt_payload(revision_request)
        return AsyncModelRequest(
            request_id=request_id,
            review_id=review_id,
            mission_id=plan.mission_id,
            uav_id=plan.uav_id,
            plan_version=plan.plan_version,
            observation_timestamp_s=revision_request.anchor_timestamp_s,
            frame_id=revision_request.anchor_frame_id,
            messages=(
                ChatMessage("system", self.SYSTEM_PROMPT),
                ChatMessage(
                    "user",
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            ),
            options=GenerationOptions(
                temperature=0.0,
                max_tokens=self._max_tokens,
                response_format=JsonSchemaResponseFormat(
                    "qwen_plan_revision_v2",
                    schema,
                ),
            ),
        )

    def parse_async_result(
        self,
        result: AsyncModelResult,
        *,
        revision_request: PlanRevisionRequest,
        expected_request_id: str,
        expected_review_id: str,
    ) -> PlanRevisionDraft:
        if not isinstance(result, AsyncModelResult):
            raise TypeError("result must be an AsyncModelResult")
        if not isinstance(revision_request, PlanRevisionRequest):
            raise TypeError("revision_request must be a PlanRevisionRequest")
        expected_request_id = validate_request_id(expected_request_id)
        expected_review_id = validate_review_id(expected_review_id)
        plan = revision_request.original_plan
        expected = {
            "request_id": expected_request_id,
            "review_id": expected_review_id,
            "mission_id": plan.mission_id,
            "uav_id": plan.uav_id,
            "plan_version": plan.plan_version,
            "frame_id": revision_request.anchor_frame_id,
        }
        mismatched = [
            name for name, value in expected.items()
            if getattr(result, name) != value
        ]
        if (
            mismatched
            or result.stale
            or abs(
                result.observation_timestamp_s
                - revision_request.anchor_timestamp_s
            )
            > 1e-9
        ):
            raise ModelProtocolError("plan revision response routing is stale")
        if result.response is None:
            raise ModelProtocolError(
                "plan revision model request did not return a response"
            )
        try:
            decoded = json.loads(
                result.response.content,
                parse_constant=_reject_revision_json_constant,
                object_pairs_hook=_strict_revision_object,
            )
            if not isinstance(decoded, Mapping):
                raise TypeError("revision output must be a JSON object")
            revision = PlanRevisionDraft.from_dict(decoded)
        except (json.JSONDecodeError, TypeError, ValueError):
            raise ModelProtocolError(
                "plan revision response violates strict JSON/schema"
            ) from None
        if (
            revision.mission_id != plan.mission_id
            or revision.uav_id != plan.uav_id
            or revision.base_plan_version != plan.plan_version
            or revision.new_plan_version != plan.plan_version + 1
            or revision.replace_from_step_id
            not in revision_request.replaceable_step_ids
        ):
            raise ModelProtocolError("plan revision response metadata mismatch")
        trusted_candidate_id = revision_request.trusted_inspect_candidate_id
        for step in revision.steps:
            if step.skill != "INSPECT":
                continue
            if (
                trusted_candidate_id is None
                or step.args.get("candidate_id") != trusted_candidate_id
            ):
                raise ModelProtocolError(
                    "plan revision INSPECT candidate_id is not the trusted "
                    "CandidateBank identifier"
                )
        return revision

    def _prompt_payload(
        self,
        revision_request: PlanRevisionRequest,
    ) -> dict[str, object]:
        plan = revision_request.original_plan
        # Reuse the canonical dynamic projection so world names/descriptions,
        # catalog text and planner limits receive the same leakage guards as
        # initial planning.  Its task field is then replaced with revision-only
        # instructions.
        canonical_messages = build_dynamic_skill_planner_messages(
            revision_request.original_instruction,
            self._world_context,
            self._skill_catalog,
            self._limits,
            self.SYSTEM_PROMPT,
            self._policy,
            mission_id=plan.mission_id,
            uav_id=plan.uav_id,
            plan_version=plan.plan_version + 1,
        )
        raw_payload = json.loads(canonical_messages[1].content)
        assert isinstance(raw_payload, dict)
        # The shared initial prompt builder deliberately hides INSPECT.  A
        # revision may expose it only after the coordinator has bound one
        # CandidateBank ID, and the advertised argument is an exact singleton.
        raw_payload["skill_catalog"] = revision_planner_catalog(
            self._skill_catalog,
            trusted_inspect_candidate_id=(
                revision_request.trusted_inspect_candidate_id
            ),
        ).to_prompt_dict()
        raw_payload["task"] = (
            "Create one PlanRevisionDraft for the trusted current-or-future suffix."
        )
        raw_payload["trusted_revision"] = {
            "schema_version": 2,
            "mission_id": plan.mission_id,
            "uav_id": plan.uav_id,
            "base_plan_version": plan.plan_version,
            "new_plan_version": plan.plan_version + 1,
            "current_step_id": revision_request.current_step_id,
            "replaceable_step_ids": list(revision_request.replaceable_step_ids),
        }
        raw_payload["original_plan"] = plan.to_dict()
        raw_payload["runtime_target_spec"] = (
            revision_request.world_belief.target_spec.to_dict()
        )
        raw_payload["completed_steps"] = [
            {
                "id": step_id,
                "skill": next(
                    step.skill for step in plan.steps if step.id == step_id
                ),
                # Values may contain internal geometry.  The model only needs
                # to know which immutable output fields exist.
                "output_fields": sorted(
                    set(
                        revision_request.completed_step_outputs.get(step_id, {})
                    )
                    & {"target_id", "status", "result_code"}
                )
                if isinstance(
                    revision_request.completed_step_outputs.get(step_id),
                    Mapping,
                )
                else [],
            }
            for step_id in revision_request.completed_step_ids
        ]
        raw_payload["world_belief_summary"] = _safe_world_belief_summary(
            revision_request.world_belief
        )
        raw_payload["trigger_event"] = _safe_event_summary(
            revision_request.trigger_event,
            trusted_inspect_candidate_id=(
                revision_request.trusted_inspect_candidate_id
            ),
        )
        return raw_payload


def _validated_step_id_sequence(
    value: object,
    field_name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    result = tuple(
        _step_id(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    if not allow_empty and not result:
        raise ValueError(f"{field_name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


def _require_revision_route(
    value: object,
    *,
    mission_id: str,
    uav_id: str,
    plan_version: int,
    label: str,
) -> None:
    if (
        getattr(value, "mission_id", None) != mission_id
        or getattr(value, "uav_id", None) != uav_id
        or getattr(value, "plan_version", None) != plan_version
    ):
        raise ValueError(f"{label} routing IDs do not match original_plan")


def _safe_world_belief_summary(value: object) -> dict[str, object]:
    target = getattr(value, "target_snapshot", None)
    target_summary = None
    target_source = getattr(target, "source", None)
    target_is_privileged = is_privileged_oracle_source(target_source)
    if target is not None and not target_is_privileged:
        target_summary = {
            "target_id": getattr(target, "target_id", None),
            "lifecycle": getattr(
                getattr(target, "lifecycle", None),
                "value",
                str(getattr(target, "lifecycle", "")),
            ),
            "confidence": getattr(target, "confidence", None),
        }
    candidates = []
    for candidate in tuple(getattr(value, "candidate_summaries", ())):
        source = getattr(candidate, "source", None)
        if is_privileged_oracle_source(source):
            continue
        candidates.append(
            {
                "candidate_id": getattr(candidate, "candidate_id", None),
                "lifecycle": getattr(candidate, "lifecycle", None),
                "source": getattr(candidate, "source", None),
                "confidence": getattr(candidate, "confidence", None),
            }
        )
    events = [
        {
            "event_id": event.event_id,
            "event_type": getattr(event.event_type, "value", str(event.event_type)),
            "severity": getattr(event.severity, "value", str(event.severity)),
            "timestamp_s": event.timestamp_s,
        }
        for event in tuple(getattr(value, "recent_events", ()))
    ]
    return {
        "current_step_id": getattr(value, "current_step_id", None),
        "current_skill": getattr(value, "current_skill", None),
        "target_state": target_summary,
        "candidates": candidates,
        "recent_events": events,
        "mission_elapsed_s": getattr(value, "mission_elapsed_s", 0.0),
    }


def _safe_event_summary(
    value: object,
    *,
    trusted_inspect_candidate_id: str | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "event_id": value.event_id,
        "event_type": getattr(value.event_type, "value", str(value.event_type)),
        "severity": getattr(value.severity, "value", str(value.severity)),
        "timestamp_s": value.timestamp_s,
    }
    # Event payloads are otherwise opaque because they can contain internal
    # geometry.  This one semantic route is required to make a trusted
    # SEARCH->INSPECT handoff expressible; neither field carries coordinates.
    payload = getattr(value, "payload", None)
    if (
        trusted_inspect_candidate_id is not None
        and isinstance(payload, Mapping)
        and payload.get("action") == "INSPECT"
        and payload.get("candidate_id") == trusted_inspect_candidate_id
        and not is_privileged_oracle_source(payload.get("source"))
    ):
        try:
            candidate_id = validate_routing_id(
                payload.get("candidate_id"),
                "candidate_id",
            )
        except (TypeError, ValueError):
            pass
        else:
            summary["action"] = "INSPECT"
            summary["candidate_id"] = candidate_id
    return summary


def _reject_revision_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _strict_revision_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field is forbidden")
        result[key] = value
    return result


def replace_plan_suffix(
    original_plan: SkillPlanDraftV2,
    revision: PlanRevisionDraft,
    *,
    current_step_id: str,
) -> SkillPlanDraftV2:
    """Return a new plan while leaving the original plan completely untouched."""

    if not isinstance(original_plan, SkillPlanDraftV2):
        raise TypeError("original_plan must be a SkillPlanDraftV2")
    if not isinstance(revision, PlanRevisionDraft):
        raise TypeError("revision must be a PlanRevisionDraft")
    _current_index, replace_index = _replacement_indexes(
        original_plan,
        revision,
        current_step_id=current_step_id,
    )

    try:
        return SkillPlanDraftV2(
            schema_version=2,
            mission_id=original_plan.mission_id,
            uav_id=original_plan.uav_id,
            plan_version=revision.new_plan_version,
            steps=original_plan.steps[:replace_index] + revision.steps,
            target_spec=original_plan.target_spec,
        )
    except (TypeError, ValueError):
        raise RevisionValidationError(
            RevisionErrorCode.SYMBOLIC_PLAN_INVALID,
            "revision does not produce a structurally valid complete plan",
        ) from None


def _replacement_indexes(
    original_plan: SkillPlanDraftV2,
    revision: PlanRevisionDraft,
    *,
    current_step_id: str,
) -> tuple[int, int]:
    """Validate the immutable envelope and locate current/replacement steps."""

    current_step_id = _step_id(current_step_id, "current_step_id")
    if (
        revision.mission_id != original_plan.mission_id
        or revision.uav_id != original_plan.uav_id
    ):
        raise RevisionValidationError(
            RevisionErrorCode.ROUTING_MISMATCH,
            "revision routing IDs do not match the active plan",
        )
    if revision.base_plan_version != original_plan.plan_version:
        raise RevisionValidationError(
            RevisionErrorCode.STALE_REVISION,
            "revision base_plan_version is stale",
        )
    if revision.new_plan_version != original_plan.plan_version + 1:
        raise RevisionValidationError(
            RevisionErrorCode.VERSION_MISMATCH,
            "revision new_plan_version does not equal the trusted next version",
        )

    original_ids = tuple(step.id for step in original_plan.steps)
    if len(original_ids) != len(set(original_ids)):
        raise RevisionValidationError(
            RevisionErrorCode.CURRENT_STEP_INVALID,
            "active plan step IDs are ambiguous",
        )
    indexes = {step_id: index for index, step_id in enumerate(original_ids)}
    if current_step_id not in indexes:
        raise RevisionValidationError(
            RevisionErrorCode.CURRENT_STEP_INVALID,
            "current_step_id is not present in the active plan",
        )
    replace_index = indexes.get(revision.replace_from_step_id)
    if replace_index is None or replace_index < indexes[current_step_id]:
        raise RevisionValidationError(
            RevisionErrorCode.REPLACE_STEP_INVALID,
            "replace_from_step_id must identify the current or a future step",
        )
    if original_plan.steps[indexes[current_step_id]].skill == "LAND":
        raise RevisionValidationError(
            RevisionErrorCode.REVISION_DURING_LAND,
            "ordinary plan revision is forbidden after LAND has started",
        )

    return indexes[current_step_id], replace_index


apply_plan_revision_atomically = replace_plan_suffix


class RevisionValidator:
    """Validate a suffix replacement fully before publishing any new plan."""

    def __init__(
        self,
        plan_validator: _PlanValidatorProtocol | None = None,
        *,
        revision_limits: RevisionLimits | None = None,
        safety_preflight: SafetyPreflight | None = None,
        symbolic_checker: SymbolicPlanChecker | None = None,
    ) -> None:
        if plan_validator is None:
            # Lazy import keeps planner module import order acyclic while still
            # providing a convenient default trusted compiler.
            from runtime.plan_validator import PlanValidator

            plan_validator = PlanValidator()
        if not callable(getattr(plan_validator, "validate_and_compile", None)):
            raise TypeError("plan_validator must provide validate_and_compile")
        limits = getattr(plan_validator, "limits", None)
        policy = getattr(plan_validator, "policy", None)
        if not isinstance(limits, PlannerLimits) or not isinstance(
            policy,
            PlannerPolicy,
        ):
            raise TypeError("plan_validator must expose PlannerLimits and PlannerPolicy")
        if revision_limits is None:
            revision_limits = RevisionLimits(max_total_plan_steps=limits.max_plan_steps)
        if not isinstance(revision_limits, RevisionLimits):
            raise TypeError("revision_limits must be a RevisionLimits")
        if revision_limits.max_total_plan_steps > limits.max_plan_steps:
            raise ValueError(
                "revision total step budget cannot exceed PlannerLimits"
            )
        if symbolic_checker is None:
            symbolic_checker = SymbolicPlanChecker()
        if not isinstance(symbolic_checker, SymbolicPlanChecker):
            raise TypeError("symbolic_checker must be a SymbolicPlanChecker")
        _preflight_callable(safety_preflight, allow_none=True)

        self._plan_validator = plan_validator
        self._revision_limits = revision_limits
        self._safety_preflight = safety_preflight
        self._symbolic_checker = symbolic_checker

    @property
    def revision_limits(self) -> RevisionLimits:
        return self._revision_limits

    def validate_and_apply(
        self,
        revision: PlanRevisionDraft,
        *,
        original: SkillPlanDraftV2 | CompiledMission,
        world_context: PlannerWorldContext,
        current_step_id: str,
        completed_step_ids: Sequence[str] | None = None,
        completed_step_outputs: Mapping[str, object] | None = None,
        revision_count: int = 0,
        now_s: float = 0.0,
        last_revision_timestamp_s: float | None = None,
        expected_new_plan_version: int | None = None,
        source: str = "dynamic_llm",
        safety_preflight: SafetyPreflight | None = None,
        trusted_inspect_candidate_id: str | None = None,
    ) -> ValidatedPlanRevision:
        """Validate, compile and preflight a revision without partial mutation."""

        if not isinstance(revision, PlanRevisionDraft):
            raise TypeError("revision must be a PlanRevisionDraft")
        if not isinstance(world_context, PlannerWorldContext):
            raise TypeError("world_context must be a PlannerWorldContext")
        original_plan = _semantic_plan(original)
        trusted_candidate_id = (
            None
            if trusted_inspect_candidate_id is None
            else validate_routing_id(
                trusted_inspect_candidate_id,
                "trusted_inspect_candidate_id",
            )
        )
        current_step_id = _step_id(current_step_id, "current_step_id")
        original_indexes = {
            step.id: index for index, step in enumerate(original_plan.steps)
        }
        current_index = original_indexes.get(current_step_id)
        if current_index is None:
            raise RevisionValidationError(
                RevisionErrorCode.CURRENT_STEP_INVALID,
                "current_step_id is not present in the active plan",
            )

        expected_completed = tuple(
            step.id for step in original_plan.steps[:current_index]
        )
        if completed_step_ids is None:
            completed = expected_completed
        else:
            if isinstance(completed_step_ids, (str, bytes)) or not isinstance(
                completed_step_ids,
                Sequence,
            ):
                raise TypeError("completed_step_ids must be a sequence")
            completed = tuple(
                _step_id(value, f"completed_step_ids[{index}]")
                for index, value in enumerate(completed_step_ids)
            )
        if completed != expected_completed:
            raise RevisionValidationError(
                RevisionErrorCode.COMPLETED_PREFIX_MISMATCH,
                "completed_step_ids must equal the immutable completed prefix",
            )

        outputs = {} if completed_step_outputs is None else completed_step_outputs
        try:
            frozen_outputs = _freeze_json_mapping(outputs)
        except (TypeError, ValueError):
            raise RevisionValidationError(
                RevisionErrorCode.COMPLETED_OUTPUT_INVALID,
                "completed step outputs must be finite JSON-compatible values",
            ) from None
        unknown_output_steps = set(frozen_outputs) - set(completed)
        if unknown_output_steps:
            raise RevisionValidationError(
                RevisionErrorCode.COMPLETED_OUTPUT_INVALID,
                "completed step outputs may only name completed-prefix steps",
            )

        if isinstance(revision_count, bool) or not isinstance(revision_count, int):
            raise TypeError("revision_count must be an integer")
        if revision_count < 0:
            raise ValueError("revision_count must be non-negative")
        if revision_count >= self._revision_limits.max_plan_revisions:
            raise RevisionValidationError(
                RevisionErrorCode.REVISION_BUDGET_EXCEEDED,
                "maximum plan revision count has been reached",
            )
        now = _finite_timestamp(now_s, "now_s")
        if revision_count > 0 and last_revision_timestamp_s is None:
            raise RevisionValidationError(
                RevisionErrorCode.REVISION_COOLDOWN,
                "last revision timestamp is required after the first revision",
            )
        if last_revision_timestamp_s is not None:
            last_revision = _finite_timestamp(
                last_revision_timestamp_s,
                "last_revision_timestamp_s",
            )
            if last_revision > now:
                raise RevisionValidationError(
                    RevisionErrorCode.REVISION_COOLDOWN,
                    "revision clock moved backwards",
                )
            if revision_count > 0 and now - last_revision < self._revision_limits.cooldown_s:
                raise RevisionValidationError(
                    RevisionErrorCode.REVISION_COOLDOWN,
                    "plan revision cooldown has not elapsed",
                )

        expected_version = (
            original_plan.plan_version + 1
            if expected_new_plan_version is None
            else _positive_version(
                expected_new_plan_version,
                "expected_new_plan_version",
            )
        )
        _validated_current_index, replace_index = _replacement_indexes(
            original_plan,
            revision,
            current_step_id=current_step_id,
        )
        if revision.new_plan_version != expected_version:
            raise RevisionValidationError(
                RevisionErrorCode.VERSION_MISMATCH,
                "revision does not echo the trusted new plan version",
            )
        old_suffix_length = len(original_plan.steps) - replace_index
        added_steps = max(0, len(revision.steps) - old_suffix_length)
        if added_steps > self._revision_limits.max_added_steps_per_revision:
            raise RevisionValidationError(
                RevisionErrorCode.ADDED_STEP_BUDGET_EXCEEDED,
                "revision exceeds the per-revision added-step budget",
            )
        projected_step_count = replace_index + len(revision.steps)
        if projected_step_count > self._revision_limits.max_total_plan_steps:
            raise RevisionValidationError(
                RevisionErrorCode.TOTAL_STEP_BUDGET_EXCEEDED,
                "revised plan exceeds the overall task step budget",
            )
        revised_plan = replace_plan_suffix(
            original_plan,
            revision,
            current_step_id=current_step_id,
        )
        original_inspect_steps = {
            step.id: step.args.get("candidate_id")
            for step in original_plan.steps
            if step.skill == "INSPECT"
        }
        for step in revised_plan.steps:
            if step.skill != "INSPECT":
                continue
            candidate_id = step.args.get("candidate_id")
            retained_trusted_step = (
                original_inspect_steps.get(step.id) == candidate_id
            )
            newly_authorized_step = (
                trusted_candidate_id is not None
                and candidate_id == trusted_candidate_id
            )
            if not retained_trusted_step and not newly_authorized_step:
                raise RevisionValidationError(
                    RevisionErrorCode.INSPECT_CANDIDATE_UNTRUSTED,
                    "revision INSPECT candidate_id is not authorized by a "
                    "trusted CandidateBank trigger",
                )
        original_search_descriptions = {
            step.args.get("target_description")
            for step in original_plan.steps
            if step.skill == "SEARCH"
        }
        allowed_search_descriptions = (
            original_search_descriptions
            if original_search_descriptions
            else {original_plan.target_spec.original_description}
        )
        for step in revised_plan.steps:
            if (
                step.skill == "SEARCH"
                and step.args.get("target_description")
                not in allowed_search_descriptions
            ):
                raise RevisionValidationError(
                    RevisionErrorCode.TARGET_IDENTITY_MUTATION,
                    "revision cannot change the immutable target description",
                )

        symbolic = self._symbolic_checker.check(
            revised_plan.to_v1(),
            world_context=world_context,
            limits=self._plan_validator.limits,
            policy=self._plan_validator.policy,
        )
        if not symbolic.valid:
            issue = symbolic.issues[0]
            raise RevisionValidationError(
                RevisionErrorCode.SYMBOLIC_PLAN_INVALID,
                f"revised plan failed symbolic validation [{issue.code.value}]",
            )
        try:
            compiled = self._plan_validator.validate_and_compile(
                revised_plan,
                world_context,
                source=source,
                mission_id=revised_plan.mission_id,
                uav_id=revised_plan.uav_id,
                plan_version=revised_plan.plan_version,
                trusted_inspect_candidate_ids=tuple(
                    sorted(
                        {
                            str(step.args["candidate_id"])
                            for step in revised_plan.steps
                            if step.skill == "INSPECT"
                        }
                    )
                ),
            )
        except Exception:
            raise RevisionValidationError(
                RevisionErrorCode.PLAN_COMPILE_FAILED,
                "revised plan failed trusted compilation",
            ) from None

        selected_preflight = (
            self._safety_preflight
            if safety_preflight is None
            else safety_preflight
        )
        if selected_preflight is not None:
            _run_preflight(selected_preflight, compiled)

        return ValidatedPlanRevision(
            revision=revision,
            revised_plan=revised_plan,
            compiled_mission=compiled,
            completed_step_outputs=frozen_outputs,
            replace_from_index=replace_index,
            added_step_count=added_steps,
            revision_count=revision_count + 1,
        )

    def validate_revision(self, *args: object, **kwargs: object) -> ValidatedPlanRevision:
        """Compatibility alias for :meth:`validate_and_apply`."""

        return self.validate_and_apply(*args, **kwargs)  # type: ignore[arg-type]


def _semantic_plan(value: SkillPlanDraftV2 | CompiledMission) -> SkillPlanDraftV2:
    if isinstance(value, SkillPlanDraftV2):
        return value
    if not isinstance(value, CompiledMission):
        raise TypeError("original must be a SkillPlanDraftV2 or CompiledMission")
    if not isinstance(value.planner_output, SkillPlanDraftV2):
        raise TypeError("compiled original must contain a SkillPlanDraftV2")
    plan = value.planner_output
    task_plan = value.task_plan
    if (
        task_plan.mission_id != plan.mission_id
        or task_plan.uav_id != plan.uav_id
        or task_plan.plan_version != plan.plan_version
    ):
        raise RevisionValidationError(
            RevisionErrorCode.ROUTING_MISMATCH,
            "compiled original routing envelope is inconsistent",
        )
    return plan


def _preflight_callable(
    value: SafetyPreflight | None,
    *,
    allow_none: bool,
) -> Callable[[CompiledMission], object] | None:
    if value is None:
        if allow_none:
            return None
        raise TypeError("safety_preflight must not be None")
    preflight = getattr(value, "preflight", None)
    if callable(preflight):
        return preflight
    if callable(value):
        return value
    raise TypeError("safety_preflight must be callable or expose preflight()")


def _run_preflight(value: SafetyPreflight, compiled: CompiledMission) -> None:
    callback = _preflight_callable(value, allow_none=False)
    assert callback is not None
    try:
        decision = callback(compiled)
    except Exception:
        raise RevisionValidationError(
            RevisionErrorCode.SAFETY_PREFLIGHT_REJECTED,
            "safety preflight failed",
        ) from None
    if decision is None or decision is True:
        return
    if decision is False:
        accepted = False
    else:
        action = getattr(decision, "action", decision)
        action_value = getattr(action, "value", action)
        accepted = action_value == "CONTINUE"
    if not accepted:
        raise RevisionValidationError(
            RevisionErrorCode.SAFETY_PREFLIGHT_REJECTED,
            "safety preflight did not return CONTINUE",
        )


def _freeze_json_mapping(value: object) -> Mapping[str, object]:
    frozen = _freeze_json(value, active_containers=set(), depth=0)
    if not isinstance(frozen, Mapping):
        raise TypeError("completed_step_outputs must be a mapping")
    return frozen


def _freeze_json(
    value: object,
    *,
    active_containers: set[int],
    depth: int,
) -> object:
    if depth > 64:
        raise ValueError("completed outputs are nested too deeply")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("completed outputs must not contain NaN or Infinity")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("completed output object keys must be strings")
        container_id = id(value)
        if container_id in active_containers:
            raise ValueError("completed outputs must not contain cycles")
        active_containers.add(container_id)
        try:
            return MappingProxyType(
                {
                    str(key): _freeze_json(
                        item,
                        active_containers=active_containers,
                        depth=depth + 1,
                    )
                    for key, item in value.items()
                }
            )
        finally:
            active_containers.remove(container_id)
    if isinstance(value, (list, tuple)):
        container_id = id(value)
        if container_id in active_containers:
            raise ValueError("completed outputs must not contain cycles")
        active_containers.add(container_id)
        try:
            return tuple(
                _freeze_json(
                    item,
                    active_containers=active_containers,
                    depth=depth + 1,
                )
                for item in value
            )
        finally:
            active_containers.remove(container_id)
    raise TypeError("completed outputs must contain only JSON-compatible values")


__all__ = [
    "PlanRevisionDraft",
    "PlanRevisionRequest",
    "QwenPlanRevisionPlanner",
    "RevisionErrorCode",
    "RevisionLimits",
    "RevisionValidationError",
    "RevisionValidator",
    "ValidatedPlanRevision",
    "apply_plan_revision_atomically",
    "build_plan_revision_json_schema",
    "replace_plan_suffix",
]
