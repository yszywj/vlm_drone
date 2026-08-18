"""Offline, instruction-grounded evaluation for Planner dataset examples.

This module is deliberately independent of Isaac Sim.  Legacy predictions are
compared directly through :class:`tasks.intent_judge.IntentJudge`; dynamic
predictions additionally pass the production Catalog, shared symbolic checker
and trusted compiler before their semantics are scored.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import csv
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time

from models.base import ChatMessage, GenerationOptions, ModelClient, ModelResponse
from common.ids import validate_mission_id, validate_uav_id
from planner.llm_planner import LLMPlanner
from planner.dynamic_llm_planner import DynamicLLMPlanner
from planner.policy import PlannerLimits, PlannerPolicy
from planner.schemas import (
    MissionIntent,
    PlannerRequest,
    PlannerWorldContext,
    SkillPlanDraft,
    SkillPlanDraftV2,
)
from planner.skill_catalog import SkillCatalog, build_default_skill_catalog
from runtime.plan_validator import PlanValidator
from tasks.intent_judge import IntentErrorCode, IntentJudge, IntentJudgeResult
from tasks.schemas import PlannerWorldCase
from tasks.target_ontology import TargetOntology

from .renderers import (
    RendererConfigError,
    load_world_cases as _load_renderer_world_cases,
    world_case_to_runtime_context as _renderer_world_case_to_runtime_context,
)
from .schemas import PLANNER_DATASET_SPLITS, PlannerDatasetSample
from .dynamic_judge import (
    DynamicPlanJudge,
    DynamicPlanJudgeResult,
    build_gold_dynamic_draft,
)


DEFAULT_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "mission_planner_system.txt"
)
DEFAULT_DYNAMIC_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "dynamic_skill_planner_system.txt"
)
DEFAULT_WORLD_CONTEXTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "planner_v1"
    / "world_contexts.yaml"
)


class PlannerEvaluationError(RuntimeError):
    """Raised for an invalid evaluation request or corrupt resume output."""


class PlannerEvaluationErrorCode(str, Enum):
    """Stable Planner-only error categories used by ``errors.csv``."""

    PLANNER_REQUEST_FAILED = "PLANNER_REQUEST_FAILED"
    PLANNER_OUTPUT_INVALID = "PLANNER_OUTPUT_INVALID"
    PLANNER_SEMANTIC_MISMATCH = "PLANNER_SEMANTIC_MISMATCH"
    TARGET_DESCRIPTION_MISMATCH = "TARGET_DESCRIPTION_MISMATCH"
    SEARCH_REGION_MISMATCH = "SEARCH_REGION_MISMATCH"
    TRACK_DURATION_MISMATCH = "TRACK_DURATION_MISMATCH"
    LANDING_ZONE_MISMATCH = "LANDING_ZONE_MISMATCH"
    TAKEOFF_ALTITUDE_MISMATCH = "TAKEOFF_ALTITUDE_MISMATCH"
    UNKNOWN_TARGET_DESCRIPTION = "UNKNOWN_TARGET_DESCRIPTION"


@dataclass(frozen=True, slots=True)
class PlannerEvaluationRun:
    """Completed evaluation output and its portable artifact directory."""

    run_dir: Path
    summary: Mapping[str, object]
    predictions: tuple[Mapping[str, object], ...]


class _RecordingModelClient:
    """Transparent per-sample recorder around an existing model client."""

    def __init__(self, client: ModelClient) -> None:
        if not callable(getattr(client, "chat", None)):
            raise TypeError("model_client must provide chat()")
        self._client = client
        self.calls = 0
        self.responses: list[str] = []
        self.options: list[GenerationOptions | None] = []

    def reset(self) -> None:
        self.calls = 0
        self.responses.clear()
        self.options.clear()

    def healthcheck(self) -> None:
        healthcheck = getattr(self._client, "healthcheck", None)
        if callable(healthcheck):
            healthcheck()

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions | None = None,
    ) -> ModelResponse:
        # Count attempted calls as model calls, including transport failures.
        self.calls += 1
        self.options.append(options)
        response = self._client.chat(messages, options=options)
        if isinstance(response, ModelResponse):
            self.responses.append(response.content)
        return response


def load_planner_dataset_split(
    dataset_root: str | os.PathLike[str],
    split: str,
) -> tuple[PlannerDatasetSample, ...]:
    """Load and strictly schema-parse one public JSONL split."""

    if split not in PLANNER_DATASET_SPLITS:
        raise PlannerEvaluationError(f"unknown Planner dataset split {split!r}")
    path = Path(dataset_root).expanduser() / f"{split}.jsonl"
    samples: list[PlannerDatasetSample] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise PlannerEvaluationError(
                        f"{path.name}:{line_number}: blank JSONL line"
                    )
                try:
                    raw = _strict_json_loads(line)
                    sample = PlannerDatasetSample.from_dict(raw)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise PlannerEvaluationError(
                        f"{path.name}:{line_number}: invalid Planner sample: "
                        f"{type(exc).__name__}"
                    ) from None
                if sample.split != split:
                    raise PlannerEvaluationError(
                        f"{path.name}:{line_number}: split field does not match filename"
                    )
                samples.append(sample)
    except OSError:
        raise PlannerEvaluationError(
            f"could not read Planner dataset split {split!r}"
        ) from None
    return tuple(samples)


def load_planner_world_cases(
    path: str | os.PathLike[str] = DEFAULT_WORLD_CONTEXTS_PATH,
) -> Mapping[str, PlannerWorldCase]:
    """Load worlds through the dataset renderer's single strict resource path."""

    # Keep evaluation, generation, and runtime prompt construction on the same
    # duplicate-key-rejecting YAML loader and schema projection.
    try:
        return _load_renderer_world_cases(path)
    except RendererConfigError as exc:
        raise PlannerEvaluationError(
            f"could not load Planner world contexts: {exc}"
        ) from None


def planner_world_case_to_runtime_context(
    world: PlannerWorldCase,
) -> PlannerWorldContext:
    """Delegate to the renderer's prompt-equivalent runtime projection."""

    return _renderer_world_case_to_runtime_context(world)


def _mission_id_for_sample(sample_id: str) -> str:
    """Derive a stable trusted route without exposing dataset prose."""

    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    digest = sha256(sample_id.encode("utf-8")).hexdigest()[:20]
    return validate_mission_id(f"mission_{digest}")


class PlannerDatasetEvaluator:
    """Evaluate scripted Gold or production LLM Planner outputs sample by sample."""

    def __init__(
        self,
        *,
        planner: str,
        world_cases: Mapping[str, PlannerWorldCase],
        ontology: TargetOntology | None = None,
        model_client: ModelClient | None = None,
        system_prompt_path: str | os.PathLike[str] | None = None,
        skill_catalog: SkillCatalog | None = None,
        planner_limits: PlannerLimits | None = None,
        planner_policy: PlannerPolicy | None = None,
        validator: PlanValidator | None = None,
        logger: object | None = None,
        uav_id: str = "uav_1",
    ) -> None:
        planner_modes = {"scripted", "llm", "dynamic_scripted", "dynamic_llm"}
        if planner not in planner_modes:
            raise ValueError(
                "planner must be scripted, llm, dynamic_scripted, or dynamic_llm"
            )
        if not isinstance(world_cases, Mapping) or not world_cases:
            raise ValueError("world_cases must be a non-empty mapping")
        checked_worlds: dict[str, PlannerWorldCase] = {}
        for context_id, world in world_cases.items():
            if not isinstance(context_id, str) or not context_id:
                raise TypeError("world_cases keys must be non-empty strings")
            if not isinstance(world, PlannerWorldCase):
                raise TypeError("world_cases values must be PlannerWorldCase objects")
            if context_id != world.context_id:
                raise ValueError("world_cases key must match world.context_id")
            checked_worlds[context_id] = world

        self.planner_mode = planner
        self.uav_id = validate_uav_id(uav_id)
        self.world_cases = checked_worlds
        self.ontology = ontology or TargetOntology.load_default()
        self.judge = IntentJudge(self.ontology)
        self.logger = logger
        self._recording_client: _RecordingModelClient | None = None
        self._llm_planner: LLMPlanner | None = None
        self._dynamic_llm_planner: DynamicLLMPlanner | None = None
        self._dynamic_judge: DynamicPlanJudge | None = None
        self._skill_catalog = skill_catalog or build_default_skill_catalog()
        self._planner_limits = planner_limits or PlannerLimits()
        self._planner_policy = planner_policy or PlannerPolicy()
        self._planner_policy.validate_against(self._planner_limits)
        if validator is None:
            validator = PlanValidator(
                limits=self._planner_limits,
                policy=self._planner_policy,
            )
        self._validator = validator

        if system_prompt_path is None:
            system_prompt_path = (
                DEFAULT_DYNAMIC_SYSTEM_PROMPT_PATH
                if planner.startswith("dynamic_")
                else DEFAULT_SYSTEM_PROMPT_PATH
            )

        if planner in {"llm", "dynamic_llm"}:
            if model_client is None:
                raise ValueError("model_client is required for LLM evaluation")
            self._recording_client = _RecordingModelClient(model_client)
            if planner == "llm":
                self._llm_planner = LLMPlanner(
                    self._recording_client,
                    system_prompt_path,
                    logger=logger,
                )
            else:
                self._dynamic_llm_planner = self._build_dynamic_llm_planner(
                    self._recording_client,
                    system_prompt_path,
                )
        elif model_client is not None:
            raise ValueError("scripted evaluation must not receive a model_client")

        if planner.startswith("dynamic_"):
            self._dynamic_judge = DynamicPlanJudge(
                self.ontology,
                skill_catalog=self._skill_catalog,
                limits=self._planner_limits,
                policy=self._planner_policy,
                validator=self._validator,
            )

    def _build_dynamic_llm_planner(
        self,
        client: ModelClient,
        system_prompt_path: str | os.PathLike[str],
    ) -> DynamicLLMPlanner:
        """Construct the production dynamic Planner with trusted policy."""

        return DynamicLLMPlanner(
            client,
            system_prompt_path,
            skill_catalog=self._skill_catalog,
            planner_limits=self._planner_limits,
            planner_policy=self._planner_policy,
            logger=self.logger,
        )

    def evaluate(
        self,
        samples: Iterable[PlannerDatasetSample],
        *,
        output_root: str | os.PathLike[str] | None = None,
        run_dir: str | os.PathLike[str] | None = None,
        start_index: int = 0,
        limit: int | None = None,
        resume: bool = False,
    ) -> PlannerEvaluationRun:
        """Evaluate a slice and atomically refresh aggregate artifacts.

        A prediction is appended immediately after each sample, so interruption
        loses at most the in-flight request.  With ``resume=True`` every sample
        ID already present in ``predictions.jsonl`` is skipped without invoking
        the model again.
        """

        all_samples = tuple(samples)
        if any(not isinstance(sample, PlannerDatasetSample) for sample in all_samples):
            raise TypeError("samples must contain PlannerDatasetSample objects")
        sample_ids = [sample.sample_id for sample in all_samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise PlannerEvaluationError("evaluation samples contain duplicate sample IDs")
        if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
            raise ValueError("start_index must be a non-negative integer")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        ):
            raise ValueError("limit must be a positive integer or None")
        selected = all_samples[start_index:]
        if limit is not None:
            selected = selected[:limit]
        if not selected and not resume:
            raise PlannerEvaluationError("the selected evaluation slice is empty")

        selected_split = self._single_split(all_samples or selected)
        directory = self._resolve_run_dir(
            output_root=output_root,
            run_dir=run_dir,
            split=selected_split,
            resume=resume,
        )
        predictions_path = directory / "predictions.jsonl"
        previous = self._load_existing_predictions(predictions_path) if resume else ()
        if any(record.get("planner") != self.planner_mode for record in previous):
            raise PlannerEvaluationError("resume output belongs to another planner mode")
        if any(record.get("split") != selected_split for record in previous):
            raise PlannerEvaluationError("resume output belongs to another dataset split")
        completed_ids = {str(record["sample_id"]) for record in previous}
        known_ids = set(sample_ids)
        unexpected = completed_ids - known_ids
        if unexpected:
            raise PlannerEvaluationError(
                "resume output contains sample IDs outside the supplied dataset"
            )

        terminal_path = directory / "terminal.log"
        self._append_terminal(
            terminal_path,
            "RUN_RESUMED" if resume else "RUN_STARTED",
            f"planner={self.planner_mode} split={selected_split}",
        )
        mode = "a" if resume else "w"
        with predictions_path.open(mode, encoding="utf-8", newline="\n") as stream:
            for sample in selected:
                if sample.sample_id in completed_ids:
                    self._append_terminal(
                        terminal_path,
                        "SKIP",
                        f"sample_id={sample.sample_id} already_complete=true",
                    )
                    continue
                record = self._evaluate_one(sample)
                serialized = _json_dumps(record)
                stream.write(serialized + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                completed_ids.add(sample.sample_id)
                judge = record["judge_result"]
                assert isinstance(judge, Mapping)
                self._append_terminal(
                    terminal_path,
                    "SAMPLE",
                    (
                        f"sample_id={sample.sample_id} "
                        f"valid={str(judge['output_valid']).lower()} "
                        f"semantic_match={str(judge['semantic_match']).lower()} "
                        f"model_calls={record['model_calls']}"
                    ),
                )

        records = self._load_existing_predictions(predictions_path)
        summary = aggregate_planner_predictions(records)
        self._atomic_json(directory / "summary.json", summary)
        self._write_errors_csv(directory / "errors.csv", records)
        self._write_field_metrics_csv(directory / "field_metrics.csv", records)
        self._append_terminal(
            terminal_path,
            "RUN_COMPLETE",
            (
                f"num_samples={summary['num_samples']} "
                f"exact_match_rate={summary['exact_match_rate']:.6f} "
                f"semantic_match_rate={summary['semantic_match_rate']:.6f}"
            ),
        )
        self._safe_log(
            "info",
            f"Planner evaluation complete: {summary['num_samples']} samples",
        )
        return PlannerEvaluationRun(
            run_dir=directory,
            summary=dict(summary),
            predictions=tuple(dict(record) for record in records),
        )

    def _evaluate_one(self, sample: PlannerDatasetSample) -> Mapping[str, object]:
        try:
            world = self.world_cases[sample.world_context_id]
        except KeyError:
            raise PlannerEvaluationError(
                f"sample {sample.sample_id!r} refers to an unknown world context"
            ) from None
        self.ontology.validate_gold_spec(sample.gold)
        expected = sample.gold.to_expected_intent()
        started_ns = time.perf_counter_ns()

        if self.planner_mode.startswith("dynamic_"):
            return self._evaluate_dynamic_one(
                sample,
                world=world,
                expected=expected,
                started_ns=started_ns,
            )

        if self.planner_mode == "scripted":
            raw = _json_dumps(expected.to_dict())
            result = self.judge.judge(
                gold=sample.gold,
                predicted=expected,
                world=world,
            )
            latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            return self._record(
                planner=self.planner_mode,
                sample=sample,
                expected=expected,
                initial_raw=raw,
                final_raw=raw,
                initial_prediction=expected,
                final_prediction=expected,
                initial_judge=result,
                final_judge=result,
                model_calls=0,
                latency_ms=latency_ms,
                request_failed=False,
            )

        assert self._recording_client is not None
        assert self._llm_planner is not None
        recorder = self._recording_client
        recorder.reset()
        predicted: MissionIntent | None = None
        request_failed = False
        try:
            predicted = self._llm_planner.plan(
                PlannerRequest(
                    instruction=sample.metadata.instruction,
                    world_context=planner_world_case_to_runtime_context(world),
                )
            )
        except Exception:
            # Model and parse failures are sample-local.  Exception details are
            # intentionally not persisted because upstream transports can carry
            # credential-bearing diagnostics.
            request_failed = len(recorder.responses) < recorder.calls

        initial_raw = recorder.responses[0] if recorder.responses else ""
        if recorder.calls <= 1:
            final_raw = initial_raw
        else:
            # A failed repair transport has no final response; do not make the
            # initial invalid response look like a completed repair response.
            final_raw = recorder.responses[1] if len(recorder.responses) > 1 else ""
        initial_prediction = self._strict_parse_or_none(initial_raw)
        initial_judge = self.judge.judge(
            gold=sample.gold,
            predicted=initial_prediction,
            world=world,
            parse_error=(
                ValueError("invalid Planner output")
                if initial_prediction is None
                else None
            ),
        )
        final_judge = self.judge.judge(
            gold=sample.gold,
            predicted=predicted,
            world=world,
            parse_error=(
                ValueError("invalid Planner output") if predicted is None else None
            ),
        )
        latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        if recorder.calls > 2:
            raise PlannerEvaluationError("LLMPlanner exceeded the two-call contract")
        if any(
            options is None or options.temperature != 0.0
            for options in recorder.options
        ):
            raise PlannerEvaluationError("LLMPlanner did not use temperature 0.0")
        return self._record(
            planner=self.planner_mode,
            sample=sample,
            expected=expected,
            initial_raw=initial_raw,
            final_raw=final_raw,
            initial_prediction=initial_prediction,
            final_prediction=predicted,
            initial_judge=initial_judge,
            final_judge=final_judge,
            model_calls=recorder.calls,
            latency_ms=latency_ms,
            request_failed=request_failed,
        )

    def _evaluate_dynamic_one(
        self,
        sample: PlannerDatasetSample,
        *,
        world: PlannerWorldCase,
        expected: MissionIntent,
        started_ns: int,
    ) -> Mapping[str, object]:
        assert self._dynamic_judge is not None
        context = planner_world_case_to_runtime_context(world)
        gold_draft = build_gold_dynamic_draft(sample.gold)

        if self.planner_mode == "dynamic_scripted":
            raw = _json_dumps(gold_draft.to_dict())
            result = self._dynamic_judge.judge(
                gold=sample.gold,
                world=world,
                world_context=context,
                raw_output=raw,
                draft=gold_draft,
                source="dynamic_scripted",
            )
            latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            return self._dynamic_record(
                planner=self.planner_mode,
                sample=sample,
                expected=expected,
                gold_draft=gold_draft,
                initial_raw=raw,
                final_raw=raw,
                initial_draft=gold_draft,
                final_draft=gold_draft,
                initial_result=result,
                final_result=result,
                model_calls=0,
                latency_ms=latency_ms,
                request_failed=False,
                repair_requested=False,
                repair_succeeded=False,
                diagnostics=None,
            )

        assert self._recording_client is not None
        assert self._dynamic_llm_planner is not None
        recorder = self._recording_client
        recorder.reset()
        predicted: SkillPlanDraftV2 | None = None
        diagnostics: object | None = None
        request_failed = False
        request = PlannerRequest(
            instruction=sample.metadata.instruction,
            world_context=context,
            mission_id=_mission_id_for_sample(sample.sample_id),
            uav_id=self.uav_id,
            plan_version=1,
        )
        try:
            plan_with_diagnostics = getattr(
                self._dynamic_llm_planner,
                "plan_with_diagnostics",
                None,
            )
            if callable(plan_with_diagnostics):
                execution = plan_with_diagnostics(request)
                predicted = execution.output
                diagnostics = execution.diagnostics
            else:  # Compatibility during rollout; production exposes diagnostics.
                predicted = self._dynamic_llm_planner.plan(request)
        except Exception:
            request_failed = len(recorder.responses) < recorder.calls
            diagnostics = getattr(
                self._dynamic_llm_planner,
                "last_diagnostics",
                None,
            )

        initial_raw = recorder.responses[0] if recorder.responses else ""
        final_raw = (
            initial_raw
            if recorder.calls <= 1
            else (recorder.responses[1] if len(recorder.responses) > 1 else "")
        )
        initial_draft = self._strict_parse_dynamic_or_none(
            initial_raw,
            request=request,
        )
        final_draft = predicted or self._strict_parse_dynamic_or_none(
            final_raw,
            request=request,
        )
        initial_result = self._dynamic_judge.judge(
            gold=sample.gold,
            world=world,
            world_context=context,
            raw_output=("" if initial_draft is None else None),
            draft=(
                None if initial_draft is None else initial_draft.to_v1()
            ),
            source="dynamic_llm",
        )
        final_result = self._dynamic_judge.judge(
            gold=sample.gold,
            world=world,
            world_context=context,
            raw_output=("" if final_draft is None else None),
            draft=(None if final_draft is None else final_draft.to_v1()),
            source="dynamic_llm",
        )

        if recorder.calls > 2:
            raise PlannerEvaluationError(
                "DynamicLLMPlanner exceeded the two-call contract"
            )
        if any(
            options is None or options.temperature != 0.0
            for options in recorder.options
        ):
            raise PlannerEvaluationError(
                "DynamicLLMPlanner did not use temperature 0.0"
            )
        if any(
            getattr(options, "response_format", None) is None
            for options in recorder.options
        ):
            raise PlannerEvaluationError(
                "DynamicLLMPlanner did not enable structured output"
            )

        if diagnostics is not None:
            model_calls = getattr(diagnostics, "model_calls", None)
            if model_calls != recorder.calls:
                raise PlannerEvaluationError(
                    "DynamicLLMPlanner diagnostics model_calls is inconsistent"
                )
            repair_requested = bool(getattr(diagnostics, "repair_used", False))
            repair_succeeded = bool(
                getattr(diagnostics, "repair_succeeded", False)
            )
            if not bool(
                getattr(diagnostics, "structured_output_enabled", False)
            ):
                raise PlannerEvaluationError(
                    "DynamicLLMPlanner diagnostics disabled structured output"
                )
        else:
            repair_requested = recorder.calls == 2
            repair_succeeded = repair_requested and final_result.output_valid

        latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        return self._dynamic_record(
            planner=self.planner_mode,
            sample=sample,
            expected=expected,
            gold_draft=gold_draft,
            initial_raw=initial_raw,
            final_raw=final_raw,
            initial_draft=initial_draft,
            final_draft=final_draft,
            initial_result=initial_result,
            final_result=final_result,
            model_calls=recorder.calls,
            latency_ms=latency_ms,
            request_failed=request_failed,
            repair_requested=repair_requested,
            repair_succeeded=repair_succeeded,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _strict_parse_dynamic_or_none(
        raw: str,
        *,
        request: PlannerRequest,
    ) -> SkillPlanDraftV2 | None:
        try:
            draft = DynamicLLMPlanner._parse_plan_draft_v2(raw)
        except (
            json.JSONDecodeError,
            OverflowError,
            RecursionError,
            TypeError,
            ValueError,
        ):
            return None
        if (
            draft.mission_id != request.mission_id
            or draft.uav_id != request.uav_id
            or draft.plan_version != request.plan_version
        ):
            return None
        return draft

    @staticmethod
    def _dynamic_record(
        *,
        planner: str,
        sample: PlannerDatasetSample,
        expected: MissionIntent,
        gold_draft: SkillPlanDraft,
        initial_raw: str,
        final_raw: str,
        initial_draft: SkillPlanDraft | SkillPlanDraftV2 | None,
        final_draft: SkillPlanDraft | SkillPlanDraftV2 | None,
        initial_result: DynamicPlanJudgeResult,
        final_result: DynamicPlanJudgeResult,
        model_calls: int,
        latency_ms: float,
        request_failed: bool,
        repair_requested: bool,
        repair_succeeded: bool,
        diagnostics: object | None,
    ) -> Mapping[str, object]:
        error_codes = (
            (PlannerEvaluationErrorCode.PLANNER_REQUEST_FAILED.value,)
            if request_failed
            else final_result.error_codes
        )
        diagnostics_dict = (
            diagnostics.to_dict()
            if diagnostics is not None
            and callable(getattr(diagnostics, "to_dict", None))
            else None
        )
        initial_mapping = initial_result.to_dict()
        final_mapping = final_result.to_dict()
        return {
            "planner": planner,
            "sample_id": sample.sample_id,
            "split": sample.split,
            "instruction": sample.metadata.instruction,
            "gold_intent": expected.to_dict(),
            "gold_dynamic_plan": gold_draft.to_dict(),
            "initial_model_output": initial_raw,
            "final_model_output": final_raw,
            "raw_model_output": final_raw,
            "initial_parsed_prediction": (
                initial_draft.to_dict() if initial_draft is not None else None
            ),
            "parsed_prediction": (
                final_draft.to_dict() if final_draft is not None else None
            ),
            "initial_judge_result": initial_mapping,
            "judge_result": final_mapping,
            "initial_schema_valid": initial_result.schema_valid,
            "initial_catalog_valid": initial_result.catalog_valid,
            "initial_symbolic_valid": initial_result.symbolic_valid,
            "initial_compile_success": initial_result.compile_success,
            "initial_semantic_match": initial_result.semantic_match,
            "initial_minimal_plan_match": initial_result.minimal_plan_match,
            "final_schema_valid": final_result.schema_valid,
            "final_catalog_valid": final_result.catalog_valid,
            "final_symbolic_valid": final_result.symbolic_valid,
            "final_compile_success": final_result.compile_success,
            "final_semantic_match": final_result.semantic_match,
            "final_minimal_plan_match": final_result.minimal_plan_match,
            "semantic_match": final_result.semantic_match,
            "minimal_plan_match": final_result.minimal_plan_match,
            "initial_error_code": (
                diagnostics_dict.get("initial_error_code")
                if diagnostics_dict is not None
                and diagnostics_dict.get("initial_error_code") is not None
                else initial_result.primary_error_code
            ),
            "final_error_code": final_result.primary_error_code,
            "model_calls": model_calls,
            "repair_requested": repair_requested,
            "repair_succeeded": repair_succeeded,
            "request_failed": request_failed,
            "structured_output_enabled": bool(
                diagnostics_dict.get("structured_output_enabled")
                if diagnostics_dict is not None
                else model_calls > 0
            )
            if model_calls else False,
            "planner_diagnostics": diagnostics_dict,
            "default_recovery_injected": (
                final_result.default_recovery_injected
            ),
            "explicit_fail": final_result.explicit_fail,
            "latency_ms": round(float(latency_ms), 6),
            "error_codes": list(error_codes),
        }

    @staticmethod
    def _strict_parse_or_none(raw: str) -> MissionIntent | None:
        try:
            return LLMPlanner._parse_intent(raw)
        except (json.JSONDecodeError, OverflowError, RecursionError, TypeError, ValueError):
            return None

    @staticmethod
    def _record(
        *,
        planner: str,
        sample: PlannerDatasetSample,
        expected: MissionIntent,
        initial_raw: str,
        final_raw: str,
        initial_prediction: MissionIntent | None,
        final_prediction: MissionIntent | None,
        initial_judge: IntentJudgeResult,
        final_judge: IntentJudgeResult,
        model_calls: int,
        latency_ms: float,
        request_failed: bool,
    ) -> Mapping[str, object]:
        repair_requested = model_calls == 2
        repair_succeeded = repair_requested and final_judge.output_valid
        error_codes = _evaluation_error_codes(
            final_judge,
            request_failed=request_failed,
        )
        return {
            "planner": planner,
            "sample_id": sample.sample_id,
            "split": sample.split,
            "instruction": sample.metadata.instruction,
            "gold_intent": expected.to_dict(),
            "initial_model_output": initial_raw,
            "final_model_output": final_raw,
            # Compatibility with the checklist's minimal record schema.
            "raw_model_output": final_raw,
            "initial_parsed_prediction": (
                initial_prediction.to_dict() if initial_prediction is not None else None
            ),
            "parsed_prediction": (
                final_prediction.to_dict() if final_prediction is not None else None
            ),
            "initial_judge_result": initial_judge.to_dict(),
            "judge_result": final_judge.to_dict(),
            "model_calls": model_calls,
            "repair_requested": repair_requested,
            "repair_succeeded": repair_succeeded,
            "request_failed": request_failed,
            "latency_ms": round(float(latency_ms), 6),
            "error_codes": list(error_codes),
        }

    @staticmethod
    def _single_split(samples: Sequence[PlannerDatasetSample]) -> str:
        if not samples:
            return "unknown"
        splits = {sample.split for sample in samples}
        if len(splits) != 1:
            raise PlannerEvaluationError("one evaluation run must contain one split")
        return next(iter(splits))

    def _resolve_run_dir(
        self,
        *,
        output_root: str | os.PathLike[str] | None,
        run_dir: str | os.PathLike[str] | None,
        split: str,
        resume: bool,
    ) -> Path:
        if resume and run_dir is None:
            raise PlannerEvaluationError("resume requires an explicit run_dir")
        if run_dir is not None:
            directory = Path(run_dir).expanduser()
            if resume:
                if not directory.is_dir():
                    raise PlannerEvaluationError("resume run_dir does not exist")
            elif directory.exists() and any(directory.iterdir()):
                raise PlannerEvaluationError("run_dir already contains output")
            directory.mkdir(parents=True, exist_ok=True)
            return directory.resolve()
        root = (
            Path(output_root).expanduser()
            if output_root is not None
            else Path.cwd() / "outputs" / "planner_eval"
        )
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        base_name = f"{timestamp}-{self.planner_mode}-{_safe_slug(split)}"
        for suffix in range(1000):
            name = base_name if suffix == 0 else f"{base_name}-{suffix}"
            candidate = root / name
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            return candidate.resolve()
        raise PlannerEvaluationError("could not allocate a unique evaluation run")

    @staticmethod
    def _load_existing_predictions(path: Path) -> tuple[Mapping[str, object], ...]:
        if not path.exists():
            return ()
        records: list[Mapping[str, object]] = []
        seen: set[str] = set()
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        raise PlannerEvaluationError(
                            f"predictions.jsonl:{line_number}: blank line"
                        )
                    record = _strict_json_loads(line)
                    if not isinstance(record, Mapping):
                        raise PlannerEvaluationError(
                            f"predictions.jsonl:{line_number}: record is not an object"
                        )
                    sample_id = record.get("sample_id")
                    if not isinstance(sample_id, str) or not sample_id:
                        raise PlannerEvaluationError(
                            f"predictions.jsonl:{line_number}: invalid sample_id"
                        )
                    if sample_id in seen:
                        raise PlannerEvaluationError(
                            "predictions.jsonl contains duplicate sample IDs"
                        )
                    seen.add(sample_id)
                    # Aggregation validates the remaining fields before reports
                    # are rewritten, making corrupt resumes fail closed.
                    records.append(dict(record))
        except (json.JSONDecodeError, ValueError):
            raise PlannerEvaluationError("predictions.jsonl contains invalid JSON") from None
        except OSError:
            raise PlannerEvaluationError("could not read predictions.jsonl") from None
        aggregate_planner_predictions(records)
        return tuple(records)

    @staticmethod
    def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
        _atomic_text(path, _json_dumps(value) + "\n")

    @staticmethod
    def _write_errors_csv(
        path: Path,
        records: Sequence[Mapping[str, object]],
    ) -> None:
        samples_by_error: dict[str, list[str]] = {}
        for record in records:
            sample_id = str(record["sample_id"])
            codes = record.get("error_codes", [])
            if not isinstance(codes, list):
                raise PlannerEvaluationError("prediction error_codes must be a list")
            for code in codes:
                if not isinstance(code, str):
                    raise PlannerEvaluationError("prediction error code must be text")
                samples_by_error.setdefault(code, []).append(sample_id)
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("error_code", "count", "sample_ids"))
        for code in sorted(samples_by_error):
            ids = samples_by_error[code]
            writer.writerow((code, len(ids), ";".join(ids)))
        _atomic_text(path, output.getvalue())

    @staticmethod
    def _write_field_metrics_csv(
        path: Path,
        records: Sequence[Mapping[str, object]],
    ) -> None:
        fields = (
            ("target", "target_match"),
            ("search_region", "search_region_match"),
            ("track_duration", "track_duration_match"),
            ("landing_zone", "landing_zone_match"),
            ("takeoff_altitude", "takeoff_altitude_match"),
        )
        dynamic = bool(
            records
            and isinstance(records[0].get("planner"), str)
            and str(records[0]["planner"]).startswith("dynamic_")
        )
        if dynamic:
            fields += (
                ("skill_sequence", "skill_sequence_match"),
                ("lost_target_policy", "lost_target_policy_match"),
            )
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("stage", "field", "correct", "total", "accuracy"))
        for stage, judge_key in (
            ("initial", "initial_judge_result"),
            ("final", "judge_result"),
        ):
            for field_name, match_key in fields:
                correct = sum(
                    bool(_judge_mapping(record, judge_key)[match_key])
                    for record in records
                )
                total = len(records)
                writer.writerow(
                    (stage, field_name, correct, total, _rate(correct, total))
                )
        _atomic_text(path, output.getvalue())

    @staticmethod
    def _append_terminal(path: Path, section: str, message: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"[{section}] {message}\n")

    def _safe_log(self, level: str, message: str) -> None:
        if self.logger is None:
            return
        try:
            method = getattr(self.logger, level, None)
            if callable(method):
                method(message)
            elif callable(self.logger):
                self.logger(message)
        except Exception:
            pass


def aggregate_planner_predictions(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate final and initial judge metrics from JSON-compatible records."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence")
    total = len(records)
    planners = {record.get("planner") for record in records}
    splits = {record.get("split") for record in records}
    planner_modes = {"scripted", "llm", "dynamic_scripted", "dynamic_llm"}
    if records and (
        len(planners) != 1
        or not all(
            isinstance(value, str) and value in planner_modes
            for value in planners
        )
    ):
        raise PlannerEvaluationError("predictions mix or omit planner modes")
    if records and (
        len(splits) != 1
        or not all(isinstance(value, str) and value in PLANNER_DATASET_SPLITS for value in splits)
    ):
        raise PlannerEvaluationError("predictions mix or omit dataset splits")
    final_judges = [_judge_mapping(record, "judge_result") for record in records]
    initial_judges = [
        _judge_mapping(record, "initial_judge_result") for record in records
    ]

    def count(judges: Sequence[Mapping[str, object]], key: str) -> int:
        return sum(bool(judge[key]) for judge in judges)

    repairs = sum(bool(record.get("repair_requested")) for record in records)
    repair_successes = sum(bool(record.get("repair_succeeded")) for record in records)
    latency_values: list[float] = []
    model_call_values: list[int] = []
    for record in records:
        value = record.get("latency_ms")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PlannerEvaluationError("prediction latency_ms must be numeric")
        value = float(value)
        if value < 0.0 or not _finite(value):
            raise PlannerEvaluationError("prediction latency_ms must be finite")
        latency_values.append(value)

        model_calls = record.get("model_calls")
        if (
            isinstance(model_calls, bool)
            or not isinstance(model_calls, int)
            or not 0 <= model_calls <= 2
        ):
            raise PlannerEvaluationError(
                "prediction model_calls must be an integer between 0 and 2"
            )
        model_call_values.append(model_calls)

    planner_mode = next(iter(planners)) if planners else None
    dynamic = isinstance(planner_mode, str) and planner_mode.startswith("dynamic_")

    def dynamic_count(key: str) -> int:
        if not dynamic:
            return 0
        count_value = 0
        for record in records:
            value = record.get(key)
            if not isinstance(value, bool):
                raise PlannerEvaluationError(
                    f"dynamic prediction {key} must be boolean"
                )
            count_value += int(value)
        return count_value

    sorted_latency = sorted(latency_values)
    p95_latency = 0.0
    if sorted_latency:
        # Nearest-rank percentile: deterministic and defined for one sample.
        rank = max(1, (95 * len(sorted_latency) + 99) // 100)
        p95_latency = sorted_latency[rank - 1]

    return {
        "schema_version": "planner_eval_v1",
        "planner": planner_mode,
        "split": next(iter(splits)) if splits else None,
        "num_samples": total,
        "output_valid_rate": _rate(count(final_judges, "output_valid"), total),
        "exact_match_rate": _rate(count(final_judges, "exact_match"), total),
        "semantic_match_rate": _rate(count(final_judges, "semantic_match"), total),
        "target_accuracy": _rate(count(final_judges, "target_match"), total),
        "search_region_accuracy": _rate(
            count(final_judges, "search_region_match"), total
        ),
        "track_duration_accuracy": _rate(
            count(final_judges, "track_duration_match"), total
        ),
        "landing_zone_accuracy": _rate(
            count(final_judges, "landing_zone_match"), total
        ),
        "takeoff_altitude_accuracy": _rate(
            count(final_judges, "takeoff_altitude_match"), total
        ),
        "initial_output_valid_rate": _rate(
            count(initial_judges, "output_valid"), total
        ),
        "initial_exact_match_rate": _rate(
            count(initial_judges, "exact_match"), total
        ),
        "initial_semantic_match_rate": _rate(
            count(initial_judges, "semantic_match"), total
        ),
        "repair_request_rate": _rate(repairs, total),
        "repair_success_rate": _rate(repair_successes, repairs),
        "mean_model_calls": (
            round(sum(model_call_values) / total, 6) if total else 0.0
        ),
        "mean_latency_ms": (
            round(sum(latency_values) / total, 6) if total else 0.0
        ),
        "p95_latency_ms": round(p95_latency, 6),
        "initial_schema_valid_rate": _rate(
            dynamic_count("initial_schema_valid"), total
        ),
        "initial_catalog_valid_rate": _rate(
            dynamic_count("initial_catalog_valid"), total
        ),
        "initial_symbolic_valid_rate": _rate(
            dynamic_count("initial_symbolic_valid"), total
        ),
        "initial_compile_success_rate": _rate(
            dynamic_count("initial_compile_success"), total
        ),
        "final_schema_valid_rate": _rate(
            dynamic_count("final_schema_valid"), total
        ),
        "final_catalog_valid_rate": _rate(
            dynamic_count("final_catalog_valid"), total
        ),
        "final_symbolic_valid_rate": _rate(
            dynamic_count("final_symbolic_valid"), total
        ),
        "final_compile_success_rate": _rate(
            dynamic_count("final_compile_success"), total
        ),
        "minimal_plan_match_rate": _rate(
            dynamic_count("minimal_plan_match"), total
        ),
        "default_recovery_injected_rate": _rate(
            dynamic_count("default_recovery_injected"), total
        ),
        "explicit_fail_rate": _rate(dynamic_count("explicit_fail"), total),
    }


def _judge_mapping(
    record: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    if not isinstance(record, Mapping):
        raise PlannerEvaluationError("prediction record must be an object")
    judge = record.get(key)
    if not isinstance(judge, Mapping):
        raise PlannerEvaluationError(f"prediction {key} must be an object")
    required = {
        "output_valid",
        "exact_match",
        "semantic_match",
        "target_match",
        "search_region_match",
        "track_duration_match",
        "landing_zone_match",
        "takeoff_altitude_match",
    }
    if not required <= set(judge):
        raise PlannerEvaluationError(f"prediction {key} is missing metrics")
    for field_name in required:
        if not isinstance(judge[field_name], bool):
            raise PlannerEvaluationError(
                f"prediction {key}.{field_name} must be boolean"
            )
    return judge


def _evaluation_error_codes(
    result: IntentJudgeResult,
    *,
    request_failed: bool,
) -> tuple[str, ...]:
    if request_failed:
        return (PlannerEvaluationErrorCode.PLANNER_REQUEST_FAILED.value,)
    if not result.output_valid:
        return (PlannerEvaluationErrorCode.PLANNER_OUTPUT_INVALID.value,)
    codes: list[str] = []
    if not result.semantic_match:
        codes.append(PlannerEvaluationErrorCode.PLANNER_SEMANTIC_MISMATCH.value)
    source = set(result.error_codes)
    mapping = (
        (
            IntentErrorCode.UNKNOWN_TARGET_DESCRIPTION.value,
            PlannerEvaluationErrorCode.UNKNOWN_TARGET_DESCRIPTION.value,
        ),
        (
            IntentErrorCode.TARGET_MISMATCH.value,
            PlannerEvaluationErrorCode.TARGET_DESCRIPTION_MISMATCH.value,
        ),
        (
            IntentErrorCode.SEARCH_REGION_MISMATCH.value,
            PlannerEvaluationErrorCode.SEARCH_REGION_MISMATCH.value,
        ),
        (
            IntentErrorCode.TRACK_DURATION_MISMATCH.value,
            PlannerEvaluationErrorCode.TRACK_DURATION_MISMATCH.value,
        ),
        (
            IntentErrorCode.LANDING_ZONE_MISMATCH.value,
            PlannerEvaluationErrorCode.LANDING_ZONE_MISMATCH.value,
        ),
        (
            IntentErrorCode.TAKEOFF_ALTITUDE_MISMATCH.value,
            PlannerEvaluationErrorCode.TAKEOFF_ALTITUDE_MISMATCH.value,
        ),
    )
    for judge_code, evaluation_code in mapping:
        if judge_code in source:
            codes.append(evaluation_code)
    return tuple(codes)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 12) if denominator else 0.0


def _finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _json_dumps(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _strict_json_loads(text: str) -> object:
    """Parse standards-compliant JSON without silent duplicate-key overwrite."""

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON number is not allowed: {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not _finite(parsed):
            raise ValueError("non-finite JSON number is not allowed")
        return parsed

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        parse_float=finite_float,
        object_pairs_hook=unique_object,
    )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    return slug or "split"


__all__ = [
    "DEFAULT_DYNAMIC_SYSTEM_PROMPT_PATH",
    "DEFAULT_SYSTEM_PROMPT_PATH",
    "DEFAULT_WORLD_CONTEXTS_PATH",
    "PlannerDatasetEvaluator",
    "PlannerEvaluationError",
    "PlannerEvaluationErrorCode",
    "PlannerEvaluationRun",
    "aggregate_planner_predictions",
    "load_planner_dataset_split",
    "load_planner_world_cases",
    "planner_world_case_to_runtime_context",
]
