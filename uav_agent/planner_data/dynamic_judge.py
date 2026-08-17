"""Layered, Gold-grounded judging for constrained dynamic Skill plans.

The dataset still stores the planner-v1 semantic Gold fields.  This module
projects a dynamic plan onto those fields only *after* checking the complete
Skill sequence, so an answer cannot score as correct merely by containing five
recoverable strings and numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

from planner.dynamic_llm_planner import DynamicLLMPlanner
from planner.policy import PlannerLimits, PlannerPolicy
from planner.schemas import MissionIntent, PlannerWorldContext, SkillPlanDraft
from planner.skill_catalog import SkillCatalog, build_default_skill_catalog
from planner.symbolic_checker import SymbolicPlanChecker
from runtime.plan_validator import PlanValidator
from tasks.intent_judge import IntentJudge, IntentJudgeResult
from tasks.schemas import GoldPlannerSpec, PlannerWorldCase
from tasks.target_ontology import TargetOntology


_GOLD_SKILL_SEQUENCE = (
    "TAKEOFF",
    "GOTO",
    "SEARCH",
    "TRACK",
    "GOTO",
    "LAND",
)


@dataclass(frozen=True, slots=True)
class DynamicPlanJudgeResult:
    """One six-layer assessment of a dynamic model response."""

    schema_valid: bool
    catalog_valid: bool
    symbolic_valid: bool
    compile_success: bool
    semantic_match: bool
    minimal_plan_match: bool
    skill_sequence_match: bool
    lost_target_policy_match: bool
    intent_result: IntentJudgeResult
    error_codes: tuple[str, ...]
    compiler_notes: tuple[str, ...] = ()
    default_recovery_injected: bool = False
    explicit_fail: bool = False

    def __post_init__(self) -> None:
        for name in (
            "schema_valid",
            "catalog_valid",
            "symbolic_valid",
            "compile_success",
            "semantic_match",
            "minimal_plan_match",
            "skill_sequence_match",
            "lost_target_policy_match",
            "default_recovery_injected",
            "explicit_fail",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if not isinstance(self.intent_result, IntentJudgeResult):
            raise TypeError("intent_result must be an IntentJudgeResult")
        codes = tuple(self.error_codes)
        notes = tuple(self.compiler_notes)
        if any(not isinstance(value, str) or not value for value in codes):
            raise TypeError("error_codes must contain non-empty strings")
        if any(not isinstance(value, str) or not value for value in notes):
            raise TypeError("compiler_notes must contain non-empty strings")
        object.__setattr__(self, "error_codes", tuple(dict.fromkeys(codes)))
        object.__setattr__(self, "compiler_notes", notes)

    @property
    def output_valid(self) -> bool:
        return all(
            (
                self.schema_valid,
                self.catalog_valid,
                self.symbolic_valid,
                self.compile_success,
            )
        )

    @property
    def exact_match(self) -> bool:
        return (
            self.output_valid
            and self.intent_result.exact_match
            and self.minimal_plan_match
        )

    @property
    def primary_error_code(self) -> str | None:
        return self.error_codes[0] if self.error_codes else None

    def to_dict(self) -> dict[str, object]:
        """Return both dynamic metrics and legacy field-metric aliases."""

        intent = self.intent_result
        return {
            "schema_valid": self.schema_valid,
            "catalog_valid": self.catalog_valid,
            "symbolic_valid": self.symbolic_valid,
            "compile_success": self.compile_success,
            "output_valid": self.output_valid,
            "exact_match": self.exact_match,
            "semantic_match": self.semantic_match,
            "minimal_plan_match": self.minimal_plan_match,
            "skill_sequence_match": self.skill_sequence_match,
            "lost_target_policy_match": self.lost_target_policy_match,
            "target_match": intent.target_match,
            "search_region_match": intent.search_region_match,
            "track_duration_match": intent.track_duration_match,
            "landing_zone_match": intent.landing_zone_match,
            "takeoff_altitude_match": intent.takeoff_altitude_match,
            "track_duration_error_s": intent.track_duration_error_s,
            "takeoff_altitude_error_m": intent.takeoff_altitude_error_m,
            "default_recovery_injected": self.default_recovery_injected,
            "explicit_fail": self.explicit_fail,
            "compiler_notes": list(self.compiler_notes),
            "error_codes": list(self.error_codes),
        }


def build_gold_dynamic_draft(gold: GoldPlannerSpec) -> SkillPlanDraft:
    """Build the deterministic minimal dynamic plan implied by planner-v1 Gold.

    Step identifiers are intentionally arbitrary.  Evaluation canonicalizes
    them through references and never requires a model to reproduce these IDs.
    The Gold draft omits recovery because planner-v1 has no lost-target-policy
    annotation; trusted compiler defaults are therefore semantically neutral.
    """

    if not isinstance(gold, GoldPlannerSpec):
        raise TypeError("gold must be a GoldPlannerSpec")
    takeoff_args: dict[str, object] = {}
    if gold.takeoff_altitude_m is not None:
        takeoff_args["altitude_m"] = gold.takeoff_altitude_m
    return SkillPlanDraft.from_dict(
        {
            "schema_version": 1,
            "steps": [
                {"id": "gold_takeoff", "skill": "TAKEOFF", "args": takeoff_args},
                {
                    "id": "gold_goto_search",
                    "skill": "GOTO",
                    "args": {"destination": gold.search_region},
                },
                {
                    "id": "gold_search",
                    "skill": "SEARCH",
                    "args": {
                        "region": gold.search_region,
                        "target_description": gold.target_description,
                    },
                },
                {
                    "id": "gold_track",
                    "skill": "TRACK",
                    "args": {
                        "target_ref": "$gold_search.target_id",
                        "duration_s": gold.track_duration_s,
                    },
                },
                {
                    "id": "gold_goto_land",
                    "skill": "GOTO",
                    "args": {"destination": gold.landing_zone},
                },
                {
                    "id": "gold_land",
                    "skill": "LAND",
                    "args": {"zone": gold.landing_zone},
                },
            ],
        }
    )


class DynamicPlanJudge:
    """Check schema, Catalog, symbols, compilation, semantics and minimality."""

    def __init__(
        self,
        ontology: TargetOntology | None = None,
        *,
        skill_catalog: SkillCatalog | None = None,
        limits: PlannerLimits | None = None,
        policy: PlannerPolicy | None = None,
        validator: PlanValidator | None = None,
    ) -> None:
        self.ontology = ontology or TargetOntology.load_default()
        if not isinstance(self.ontology, TargetOntology):
            raise TypeError("ontology must be a TargetOntology")
        self.intent_judge = IntentJudge(self.ontology)
        self.skill_catalog = skill_catalog or build_default_skill_catalog()
        if not isinstance(self.skill_catalog, SkillCatalog):
            raise TypeError("skill_catalog must be a SkillCatalog")
        self.limits = limits or PlannerLimits()
        if not isinstance(self.limits, PlannerLimits):
            raise TypeError("limits must be a PlannerLimits")
        self.policy = policy or PlannerPolicy()
        if not isinstance(self.policy, PlannerPolicy):
            raise TypeError("policy must be a PlannerPolicy")
        self.policy.validate_against(self.limits)
        if validator is None:
            validator = PlanValidator(limits=self.limits, policy=self.policy)
        if not isinstance(validator, PlanValidator):
            raise TypeError("validator must be a PlanValidator")
        if validator.limits != self.limits or validator.policy != self.policy:
            raise ValueError(
                "validator limits and policy must match the dynamic judge"
            )
        self.validator = validator
        self.symbolic_checker = SymbolicPlanChecker()

    def judge(
        self,
        *,
        gold: GoldPlannerSpec,
        world: PlannerWorldCase,
        world_context: PlannerWorldContext,
        raw_output: str | None = None,
        draft: SkillPlanDraft | None = None,
        source: str = "dynamic_llm",
    ) -> DynamicPlanJudgeResult:
        if not isinstance(gold, GoldPlannerSpec):
            raise TypeError("gold must be a GoldPlannerSpec")
        if not isinstance(world, PlannerWorldCase):
            raise TypeError("world must be a PlannerWorldCase")
        if not isinstance(world_context, PlannerWorldContext):
            raise TypeError("world_context must be a PlannerWorldContext")
        if source not in {"dynamic_scripted", "dynamic_llm"}:
            raise ValueError("source must be dynamic_scripted or dynamic_llm")

        schema_valid = False
        catalog_valid = False
        symbolic_valid = False
        compile_success = False
        compiler_notes: tuple[str, ...] = ()
        errors: list[str] = []

        parsed = draft
        if parsed is not None and not isinstance(parsed, SkillPlanDraft):
            raise TypeError("draft must be a SkillPlanDraft or None")
        if parsed is None:
            try:
                parsed = DynamicLLMPlanner._parse_plan_draft(raw_output or "")
            except (
                json.JSONDecodeError,
                OverflowError,
                RecursionError,
                TypeError,
                ValueError,
            ):
                errors.append("SCHEMA_INVALID")
        if parsed is not None:
            schema_valid = True
            try:
                self._validate_catalog(parsed)
            except (TypeError, ValueError):
                errors.append("CATALOG_INVALID")
            else:
                catalog_valid = True

        if parsed is not None and catalog_valid:
            symbolic = self.symbolic_checker.check(
                parsed,
                world_context=world_context,
                limits=self.limits,
                policy=self.policy,
            )
            symbolic_valid = symbolic.valid
            if not symbolic.valid:
                errors.extend(issue.code.value for issue in symbolic.issues)

        compiled = None
        if parsed is not None and symbolic_valid:
            try:
                compiled = self.validator.validate_and_compile(
                    parsed,
                    world_context,
                    source=source,
                )
            except (TypeError, ValueError):
                errors.append("COMPILE_FAILED")
            else:
                compile_success = True
                compiler_notes = tuple(compiled.compiler_notes)

        prediction = self._project_intent(parsed)
        intent_result = self.intent_judge.judge(
            gold=gold,
            predicted=prediction,
            world=world,
            parse_error=(ValueError("invalid dynamic plan") if prediction is None else None),
        )
        required_path_match = self._required_semantic_path_match(parsed, gold)
        sequence_match = bool(
            parsed is not None
            and tuple(step.skill for step in parsed.steps)
            == _GOLD_SKILL_SEQUENCE
        )
        minimal_match = self._minimal_plan_match(parsed, gold)
        semantic_match = bool(intent_result.semantic_match and required_path_match)
        lost_policy_match = self._lost_target_policy_match(parsed)

        if parsed is not None:
            if not required_path_match:
                errors.append("SKILL_SEQUENCE_MISMATCH")
            if not minimal_match:
                errors.append("MINIMAL_PLAN_MISMATCH")
            errors.extend(intent_result.error_codes)

        explicit_fail = bool(
            parsed is not None
            and any(
                step.skill == "TRACK"
                and step.args.get("on_target_lost") == "FAIL"
                for step in parsed.steps
            )
        )
        default_recovery_injected = any(
            "recovery injected from trusted default policy" in note
            for note in compiler_notes
        )
        return DynamicPlanJudgeResult(
            schema_valid=schema_valid,
            catalog_valid=catalog_valid,
            symbolic_valid=symbolic_valid,
            compile_success=compile_success,
            semantic_match=semantic_match,
            minimal_plan_match=minimal_match,
            skill_sequence_match=sequence_match,
            lost_target_policy_match=lost_policy_match,
            intent_result=intent_result,
            error_codes=tuple(errors),
            compiler_notes=compiler_notes,
            default_recovery_injected=default_recovery_injected,
            explicit_fail=explicit_fail,
        )

    def _validate_catalog(self, draft: SkillPlanDraft) -> None:
        contracts = {contract.name: contract for contract in self.skill_catalog}
        for step in draft.steps:
            contract = contracts.get(step.skill)
            if (
                contract is None
                or not contract.top_level_allowed
                or contract.recovery_only
            ):
                raise ValueError("top-level Skill is absent from active Catalog")
            DynamicLLMPlanner._validate_catalog_arguments(
                step.args,
                contract.arguments,
                prefix=f"step {step.id}",
            )
            if step.recovery is None:
                continue
            recovery = contracts.get(step.recovery.skill)
            if (
                recovery is None
                or recovery.top_level_allowed
                or not recovery.recovery_only
            ):
                raise ValueError("recovery Skill is absent from active Catalog")
            values = step.recovery.to_dict()
            values.pop("skill", None)
            DynamicLLMPlanner._validate_catalog_arguments(
                values,
                recovery.arguments,
                prefix=f"step {step.id} recovery",
            )

    @staticmethod
    def _project_intent(draft: SkillPlanDraft | None) -> MissionIntent | None:
        if draft is None:
            return None
        takeoff = next((step for step in draft.steps if step.skill == "TAKEOFF"), None)
        search = next((step for step in draft.steps if step.skill == "SEARCH"), None)
        track = next((step for step in draft.steps if step.skill == "TRACK"), None)
        land = next((step for step in draft.steps if step.skill == "LAND"), None)
        if any(step is None for step in (takeoff, search, track, land)):
            return None
        try:
            return MissionIntent(
                target_description=search.args["target_description"],
                search_region=search.args["region"],
                track_duration_s=track.args["duration_s"],
                landing_zone=land.args["zone"],
                takeoff_altitude_m=takeoff.args.get("altitude_m"),
            )
        except (TypeError, ValueError, KeyError):
            return None

    @staticmethod
    def _required_semantic_path_match(
        draft: SkillPlanDraft | None,
        gold: GoldPlannerSpec,
    ) -> bool:
        if draft is None:
            return False
        steps = draft.steps
        search_indexes = [
            index
            for index, step in enumerate(steps)
            if step.skill == "SEARCH"
            and step.args.get("region") == gold.search_region
        ]
        track_indexes = [
            index for index, step in enumerate(steps) if step.skill == "TRACK"
        ]
        goto_search = any(
            step.skill == "GOTO"
            and step.args.get("destination") == gold.search_region
            for step in steps
        )
        goto_land = any(
            step.skill == "GOTO"
            and step.args.get("destination") == gold.landing_zone
            for step in steps
        )
        land_indexes = [
            index
            for index, step in enumerate(steps)
            if step.skill == "LAND" and step.args.get("zone") == gold.landing_zone
        ]
        if not (
            search_indexes
            and track_indexes
            and land_indexes
            and goto_search
            and goto_land
        ):
            return False
        return min(search_indexes) < min(track_indexes) < max(land_indexes)

    @staticmethod
    def _minimal_plan_match(
        draft: SkillPlanDraft | None,
        gold: GoldPlannerSpec,
    ) -> bool:
        if draft is None:
            return False
        if tuple(step.skill for step in draft.steps) != _GOLD_SKILL_SEQUENCE:
            return False
        return (
            draft.steps[1].args.get("destination") == gold.search_region
            and draft.steps[4].args.get("destination") == gold.landing_zone
        )

    @staticmethod
    def _lost_target_policy_match(draft: SkillPlanDraft | None) -> bool:
        if draft is None:
            return False
        tracks = [step for step in draft.steps if step.skill == "TRACK"]
        if not tracks:
            return False
        # planner_v1 Gold has no lost-target annotation.  Inherited trusted
        # defaults, explicit REACQUIRE and explicit FAIL are all neutral here;
        # conflicts are rejected by the symbolic layer.
        return all(
            step.args.get("on_target_lost") in {None, "REACQUIRE", "FAIL"}
            for step in tracks
        )


__all__ = [
    "DynamicPlanJudge",
    "DynamicPlanJudgeResult",
    "build_gold_dynamic_draft",
]
