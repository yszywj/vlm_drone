#!/usr/bin/env python3
"""Run the routed dynamic mission with opt-in asynchronous Qwen vision.

This is an experimental Isaac standalone entry point.  It deliberately fails
closed when production perception is selected because this repository does
not yet ship a real detector/tracker.  The retained Oracle execution path is
available only through the two-part ``oracle_evaluation`` acknowledgement.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum
import json
from math import isfinite
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = Path(
    os.environ.get(
        "UAV_AGENT_CONDA_ENV",
        "/home/amax/miniconda3/envs/r_isaac_sim",
    )
).expanduser()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep module import pure Python.  In particular, no Isaac-backed environment
# module is imported until main() has created SimulationApp.
from common.ids import (  # noqa: E402
    generate_routing_id,
    validate_mission_id,
    validate_uav_id,
)
from configs.loader import ConfigError, load_config  # noqa: E402


def _build_oracle_evaluation_backend(uav_id: str) -> object:
    """Build the explicitly privileged backend with one trusted UAV route."""

    from perception import (
        GuardedPerceptionBackend,
        OraclePerception,
        PerceptionRuntimeProfile,
    )

    trusted_uav_id = validate_uav_id(uav_id)
    return GuardedPerceptionBackend(
        OraclePerception(uav_id=trusted_uav_id, target_id="target"),
        profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
        acknowledge_privileged_oracle=True,
    )


class LaunchConfigurationError(ValueError):
    """Raised before Isaac/Qwen startup when an opt-in is incomplete."""


@dataclass(frozen=True, slots=True)
class TestInjectionSpec:
    """One explicitly synthetic, simulation-time event trigger."""

    __test__ = False

    event_type: str
    trigger_at_s: float
    flag_name: str

    def __post_init__(self) -> None:
        if self.event_type not in {
            "PATH_BLOCKED",
            "SKILL_PROGRESS_STALLED",
            "TARGET_IDENTITY_UNCERTAIN",
        }:
            raise ValueError("unsupported test-injection event type")
        if (
            isinstance(self.trigger_at_s, bool)
            or not isinstance(self.trigger_at_s, (int, float))
            or not isfinite(float(self.trigger_at_s))
            or float(self.trigger_at_s) < 0.0
        ):
            raise ValueError("trigger_at_s must be finite and non-negative")
        object.__setattr__(self, "trigger_at_s", float(self.trigger_at_s))


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug-visualization",
        action="store_true",
        help=(
            "draw search geometry, camera frustum, planned route, "
            "and skill-colored executed trajectory in the GUI viewport"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="unified YAML config (default: %(default)s)",
    )
    parser.add_argument(
        "--planner",
        choices=("scripted", "llm", "dynamic_scripted", "dynamic_llm"),
        default="dynamic_llm",
        help="high-level planner (default: %(default)s)",
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI-compatible API key (never printed or persisted)",
    )
    parser.add_argument(
        "--uav-id",
        default="uav_1",
        type=validate_uav_id,
        help="trusted single-Agent routing ID (default: %(default)s)",
    )
    parser.add_argument("--takeoff-altitude", type=_positive_float, default=10.0)
    parser.add_argument("--track-duration", type=_positive_float, default=30.0)
    parser.add_argument("--max-sim-time", type=_positive_float, default=360.0)
    parser.add_argument("--start-altitude", type=_finite_float, default=0.0)

    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=None)
    parser.add_argument("--debug-ground-truth", action="store_true")

    parser.add_argument(
        "--enable-qwen-vision",
        action="store_true",
        help="enable asynchronous RGB review; disabled regardless of YAML without this flag",
    )
    parser.add_argument(
        "--vision-review-mode",
        choices=("shadow", "gate"),
        default=None,
        help="override qwen_visual_review.mode from config",
    )
    parser.add_argument(
        "--acknowledge-vision-gate",
        action="store_true",
        help="explicitly authorize gate-mode semantic control effects",
    )
    parser.add_argument(
        "--perception-runtime-profile",
        choices=("production", "oracle_evaluation"),
        default="production",
        help="production fails closed until a real detector/tracker is installed",
    )
    parser.add_argument(
        "--acknowledge-privileged-oracle",
        action="store_true",
        help="second explicit opt-in required with oracle_evaluation",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/dynamic_visual_missions",
        help="parent directory for bounded image-free run logs",
    )

    parser.add_argument(
        "--inject-path-blocked-at-s",
        type=_nonnegative_float,
        default=None,
        metavar="SECONDS",
        help="inject PATH_BLOCKED with source=test_injection",
    )
    parser.add_argument(
        "--inject-progress-stall-at-s",
        type=_nonnegative_float,
        default=None,
        metavar="SECONDS",
        help="inject SKILL_PROGRESS_STALLED with source=test_injection",
    )
    parser.add_argument(
        "--inject-identity-conflict-at-s",
        type=_nonnegative_float,
        default=None,
        metavar="SECONDS",
        help="inject TARGET_IDENTITY_UNCERTAIN with source=test_injection",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def build_test_injection_specs(
    args: argparse.Namespace,
) -> tuple[TestInjectionSpec, ...]:
    pairs = (
        (
            "PATH_BLOCKED",
            args.inject_path_blocked_at_s,
            "--inject-path-blocked-at-s",
        ),
        (
            "SKILL_PROGRESS_STALLED",
            args.inject_progress_stall_at_s,
            "--inject-progress-stall-at-s",
        ),
        (
            "TARGET_IDENTITY_UNCERTAIN",
            args.inject_identity_conflict_at_s,
            "--inject-identity-conflict-at-s",
        ),
    )
    return tuple(
        TestInjectionSpec(event_type, value, flag)
        for event_type, value, flag in pairs
        if value is not None
    )


def validate_launch_args(
    args: argparse.Namespace,
    *,
    configured_review_mode: str = "shadow",
) -> tuple[str, tuple[TestInjectionSpec, ...]]:
    """Return the effective review mode or reject an unsafe combination."""

    if not isinstance(args.instruction, str) or not args.instruction.strip():
        raise LaunchConfigurationError("--instruction must not be empty")
    mode = args.vision_review_mode or configured_review_mode
    if mode not in {"shadow", "gate"}:
        raise LaunchConfigurationError("visual review mode must be shadow or gate")
    if args.acknowledge_vision_gate and mode != "gate":
        raise LaunchConfigurationError(
            "--acknowledge-vision-gate is only valid with gate mode"
        )
    if mode == "gate" and not args.enable_qwen_vision:
        raise LaunchConfigurationError("gate mode requires --enable-qwen-vision")
    if mode == "gate" and not args.acknowledge_vision_gate:
        raise LaunchConfigurationError(
            "gate mode requires --acknowledge-vision-gate"
        )

    oracle_selected = args.perception_runtime_profile == "oracle_evaluation"
    if oracle_selected != bool(args.acknowledge_privileged_oracle):
        if oracle_selected:
            raise LaunchConfigurationError(
                "oracle_evaluation requires --acknowledge-privileged-oracle"
            )
        raise LaunchConfigurationError(
            "Oracle acknowledgement is invalid in production profile"
        )
    if not oracle_selected:
        raise LaunchConfigurationError(
            "production visual geometry is unavailable: no real detector/tracker "
            "is implemented; select oracle_evaluation and explicitly acknowledge "
            "the privileged evaluation boundary"
        )

    injections = build_test_injection_specs(args)
    if injections and not args.enable_qwen_vision:
        raise LaunchConfigurationError(
            "test-injection review events require --enable-qwen-vision"
        )
    return mode, injections


def make_test_injection_event(
    spec: TestInjectionSpec,
    *,
    mission_id: str,
    uav_id: str,
    plan_version: int,
    timestamp_s: float,
) -> object:
    """Build one routed event with an unambiguous synthetic source label."""

    from runtime.events import EventSeverity, MissionEvent, MissionEventType

    return MissionEvent(
        event_id=generate_routing_id("event"),
        mission_id=validate_mission_id(mission_id),
        uav_id=validate_uav_id(uav_id),
        plan_version=plan_version,
        timestamp_s=timestamp_s,
        event_type=MissionEventType(spec.event_type),
        severity=EventSeverity.WARNING,
        payload={
            "source": "test_injection",
            "trigger_flag": spec.flag_name,
            "trigger_at_s": spec.trigger_at_s,
        },
    )


def startup_fields(
    *,
    uav_id: str,
    mission_id: str,
    planner: str,
    review_enabled: bool,
    review_mode: str,
    perception_profile: str,
    oracle_acknowledged: bool,
) -> tuple[tuple[str, object], ...]:
    """Stable launch metadata used by both the terminal and unit tests."""

    return (
        ("uav_id", validate_uav_id(uav_id)),
        ("mission_id", validate_mission_id(mission_id)),
        ("planner", planner),
        (
            "qwen_visual_review",
            f"enabled:{review_mode}" if review_enabled else "disabled",
        ),
        ("perception_runtime_profile", perception_profile),
        ("oracle_acknowledged", oracle_acknowledged),
    )


def _json_compatible(value: object) -> object:
    if isinstance(value, Enum):
        return _json_compatible(value.value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    raise TypeError(f"manifest value {type(value).__name__} is not JSON-compatible")


def _read_git_commit() -> str | None:
    """Best-effort read-only provenance; never mutates the repository."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT.parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else None


def visual_manifest_context(
    *,
    args: argparse.Namespace,
    config: object,
    mission_id: str,
    review_mode: str,
    model_name: str,
) -> dict[str, object]:
    """Return bounded launch provenance without credentials or image data."""

    blocks = {
        name: getattr(config, name)
        for name in (
            "model_worker",
            "qwen_visual_review",
            "plan_revision",
            "frame_store",
            "debug_images",
        )
    }
    return {
        "mission_id": validate_mission_id(mission_id),
        "uav_id": validate_uav_id(args.uav_id),
        "planner": args.planner,
        "model": model_name,
        "git_commit": _read_git_commit(),
        "perception_runtime_profile": args.perception_runtime_profile,
        "oracle_acknowledged": bool(args.acknowledge_privileged_oracle),
        "qwen_visual_review_mode": review_mode,
        "configuration": _json_compatible(blocks),
    }


@dataclass(slots=True)
class _VisualRuntime:
    coordinator: object
    worker: object
    event_bus: object
    output_parent: Path
    model_name: str
    revision_coordinator: object | None = None
    seen_review_ids: frozenset[str] = frozenset()
    seen_event_ids: frozenset[str] = frozenset()
    seen_revision_keys: frozenset[str] = frozenset()
    transition_cursor: int = 0
    logger: object | None = None
    run_dir: Path | None = None
    hover_started_at_s: float | None = None
    manifest_context: dict[str, object] = field(default_factory=dict)
    terminal_manifest: dict[str, object] = field(default_factory=dict)

    def begin_logging(
        self,
        mission_id: str,
        *,
        manifest_context: dict[str, object] | None = None,
    ) -> None:
        from experiments.sparse_mission_logger import SparseMissionLogger

        self.run_dir = self.output_parent.expanduser() / validate_mission_id(mission_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_context = dict(manifest_context or {})
        try:
            self.logger = SparseMissionLogger(self.run_dir)
        except Exception as exc:
            # Persistent logging is observational.  A filesystem failure after
            # takeoff must never stop the control loop or strand the vehicle.
            self.logger = None
            print(
                f"[Logging] sparse logs disabled: {type(exc).__name__}",
                file=sys.stderr,
            )
        self._write_manifest()

    def _disable_failed_logger(self, exc: Exception) -> None:
        logger = self.logger
        self.logger = None
        if logger is not None:
            try:
                logger.close()
            except Exception:
                pass
        print(
            f"[Logging] sparse log append disabled: {type(exc).__name__}",
            file=sys.stderr,
        )

    def emit_new_reviews(
        self,
        *,
        step_id: str | None,
        skill: str,
    ) -> None:
        from experiments.sparse_mission_logger import QwenReviewLogRecord

        records = self.coordinator.records
        for record in records:
            if record.review_id in self.seen_review_ids:
                continue
            payload = record.to_dict()
            print(
                "[QwenVisualReview] "
                f"mission_id={record.mission_id} uav_id={record.uav_id} "
                f"plan_version={record.plan_version} "
                f"step_id={step_id or 'none'} skill={skill} payload="
                + json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
            )
            if self.logger is not None:
                usage = dict(record.token_usage)
                try:
                    self.logger.log_qwen_review(
                        QwenReviewLogRecord(
                            review_id=record.review_id,
                            request_id=record.request_id,
                            mission_id=record.mission_id,
                            uav_id=record.uav_id,
                            plan_version=record.plan_version,
                            step_id=step_id,
                            frame_id=record.frame_id,
                            observation_timestamp_s=record.observation_timestamp_s,
                            decision=record.decision or record.error_code or "UNKNOWN",
                            bbox_xyxy_normalized=record.bbox_xyxy_normalized,
                            prompt_tokens=int(
                                usage.get("prompt_tokens", usage.get("input_tokens", 0))
                            ),
                            completion_tokens=int(
                                usage.get("completion_tokens", usage.get("output_tokens", 0))
                            ),
                            total_tokens=int(usage.get("total_tokens", 0)),
                            latency_s=record.latency_s,
                            stale=record.stale,
                            accepted=record.accepted_for_control,
                            timeout=record.error_code == "TIMEOUT",
                        )
                    )
                except Exception as exc:
                    self._disable_failed_logger(exc)
        self.seen_review_ids = frozenset(record.review_id for record in records)

    def log_injection(self, event: object, *, skill: str, step_id: str) -> None:
        payload = event.to_dict()
        print(
            "[TestInjection] "
            f"mission_id={event.mission_id} uav_id={event.uav_id} "
            f"plan_version={event.plan_version} step_id={step_id} "
            f"skill={skill} payload="
            + json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        )

    def emit_new_events(self, *, skill: str, step_id: str) -> None:
        from experiments.sparse_mission_logger import MissionEventLogRecord
        from runtime.events import json_payload_to_dict

        events = self.event_bus.recent()
        for event in events:
            if event.event_id in self.seen_event_ids:
                continue
            payload = json_payload_to_dict(event.payload)
            source = payload.get("source", "runtime")
            record = MissionEventLogRecord(
                timestamp_s=event.timestamp_s,
                mission_id=event.mission_id,
                uav_id=event.uav_id,
                plan_version=event.plan_version,
                step_id=step_id,
                skill=skill,
                event=event.event_type.value,
                status=event.severity.value,
                reason=f"source={source}",
            )
            print(record.to_terminal_line())
            if self.logger is not None:
                try:
                    self.logger.log_mission_event(record)
                except Exception as exc:
                    self._disable_failed_logger(exc)
        self.seen_event_ids = frozenset(event.event_id for event in events)

    def emit_new_transitions(self, manager: object) -> None:
        from experiments.sparse_mission_logger import SkillTransitionLogRecord
        from skills.types import SkillName

        transitions = manager.transition_log
        for transition in transitions[self.transition_cursor :]:
            step_id = transition.new_step_id or transition.old_step_id or "none"
            record = SkillTransitionLogRecord(
                timestamp_s=transition.timestamp,
                mission_id=transition.mission_id,
                uav_id=transition.uav_id,
                plan_version=transition.plan_version,
                step_id=step_id,
                old_skill=(
                    "NONE" if transition.old_skill is None else transition.old_skill.value
                ),
                new_skill=(
                    "NONE" if transition.new_skill is None else transition.new_skill.value
                ),
                old_status=(
                    "NONE" if transition.old_status is None else transition.old_status.name
                ),
                result_code=(
                    "NONE"
                    if transition.result_code is None
                    else transition.result_code.name
                ),
                reason=transition.reason,
            )
            terminal_skill = record.new_skill
            if terminal_skill == "NONE":
                terminal_skill = record.old_skill
            print(
                "[SkillTransition] "
                f"mission_id={record.mission_id} uav_id={record.uav_id} "
                f"plan_version={record.plan_version} step_id={record.step_id} "
                f"skill={terminal_skill} old_skill={record.old_skill} "
                f"new_skill={record.new_skill} reason={record.reason}"
            )
            if self.logger is not None:
                try:
                    self.logger.log_skill_transition(record)
                    if (
                        transition.new_skill is SkillName.HOVER
                        and transition.old_skill is not SkillName.HOVER
                        and self.hover_started_at_s is None
                    ):
                        self.hover_started_at_s = transition.timestamp
                        self.logger.record_supervisory_hover_started()
                    elif (
                        transition.old_skill is SkillName.HOVER
                        and transition.new_skill is not SkillName.HOVER
                        and self.hover_started_at_s is not None
                    ):
                        self.logger.record_supervisory_hover_duration(
                            max(0.0, transition.timestamp - self.hover_started_at_s)
                        )
                        self.hover_started_at_s = None
                except Exception as exc:
                    self._disable_failed_logger(exc)
            self.transition_cursor += 1

    def emit_new_revisions(self, *, skill: str, step_id: str) -> None:
        coordinator = self.revision_coordinator
        if coordinator is None:
            return
        records = tuple(coordinator.records)
        seen = set(self.seen_revision_keys)
        for record in records:
            key = f"{record.request_id}:{record.outcome}:{record.timestamp_s}"
            if key in seen:
                continue
            print(
                "[PlanRevision] "
                f"mission_id={record.mission_id} uav_id={record.uav_id} "
                f"plan_version={record.new_plan_version or record.old_plan_version} "
                f"step_id={record.new_step_id or step_id} skill={skill} "
                f"outcome={record.outcome} request_id={record.request_id}"
            )
            if self.logger is not None and record.outcome == "ACCEPTED":
                try:
                    self.logger.record_plan_revision()
                except Exception as exc:
                    self._disable_failed_logger(exc)
            seen.add(key)
        self.seen_revision_keys = frozenset(seen)

    def set_terminal_manifest(
        self,
        *,
        agent_status: str,
        task_status: str,
        plan_version: int,
        guard_error: str | None,
    ) -> None:
        self.terminal_manifest = {
            "agent_status": str(agent_status),
            "task_status": str(task_status),
            "final_plan_version": int(plan_version),
            "guard_error": guard_error,
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        run_dir = self.run_dir
        if run_dir is None:
            return
        stats: dict[str, object]
        if self.logger is None:
            stats = {
                "qwen_visual_reviews": {
                    "count": 0,
                    "accepted": 0,
                    "stale": 0,
                    "timeout": 0,
                },
                "plan_revisions": 0,
                "supervisory_hover": {"count": 0, "total_time_s": 0.0},
                "debug_images": {"count": 0, "bytes": 0},
                "dropped_log_records": 0,
            }
        else:
            try:
                stats = self.logger.snapshot().to_manifest_dict()
            except Exception:
                return
        payload = {
            "schema_version": 1,
            **self.manifest_context,
            **self.terminal_manifest,
            **stats,
        }
        temporary = run_dir / ".run_manifest.json.tmp"
        destination = run_dir / "run_manifest.json"
        try:
            temporary.write_text(
                json.dumps(
                    _json_compatible(payload),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        except Exception as exc:
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass
            print(
                f"[Logging] run manifest disabled: {type(exc).__name__}",
                file=sys.stderr,
            )

    def close(self, *, timeout_s: float) -> None:
        close_error: Exception | None = None
        try:
            self.worker.close(timeout_s=timeout_s)
        except Exception as exc:  # pragma: no cover - integration shutdown path
            close_error = exc
        try:
            if self.logger is not None:
                self._write_manifest()
                self.logger.close()
        finally:
            if close_error is not None:
                raise close_error


def _create_visual_runtime(
    *,
    args: argparse.Namespace,
    config: object,
    manager: object,
    target_manager: object,
    candidate_bank: object,
    frame_store: object,
    review_mode: str,
    await_revision_completion: bool,
) -> _VisualRuntime:
    from agents.visual_review_coordinator import VisualReviewCoordinator
    from models import AsyncModelWorker, OpenAICompatibleClient
    from perception import QwenVLMVerifier, VisualReviewGate
    from runtime import MissionEventBus, ReviewScheduler

    visual_client = OpenAICompatibleClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout_s=config.model_worker.request_timeout_s,
        max_images_per_request=config.qwen_visual_review.max_recent_frames,
    )
    worker = AsyncModelWorker(visual_client, uav_id=args.uav_id)
    scheduler = ReviewScheduler(
        intervals_s={
            "GOTO": config.qwen_visual_review.goto_interval_s,
            "SEARCH": config.qwen_visual_review.search_interval_s,
            "INSPECT": config.qwen_visual_review.inspect_interval_s,
            "TRACK": config.qwen_visual_review.track_interval_s,
        }
    )
    verifier = QwenVLMVerifier(
        max_image_side_px=config.qwen_visual_review.max_image_side_px,
        jpeg_quality=config.qwen_visual_review.jpeg_quality,
    )
    event_bus = MissionEventBus(max_events=256)
    coordinator = VisualReviewCoordinator(
        uav_id=args.uav_id,
        scheduler=scheduler,
        frame_store=frame_store,
        worker=worker,
        verifier=verifier,
        gate=VisualReviewGate(mode=review_mode, min_consistent_matches=2),
        skill_manager=manager,
        target_manager=target_manager,
        candidate_bank=candidate_bank,
        event_bus=event_bus,
        max_recent_frames=config.qwen_visual_review.max_recent_frames,
        review_timeout_s=config.qwen_visual_review.blocking_hover_timeout_s,
        hover_position_tolerance_m=(
            config.qwen_visual_review.hover_position_tolerance_m
        ),
        hover_max_correction_speed_mps=(
            config.qwen_visual_review.hover_max_correction_speed_mps
        ),
        blocking_timeout_fallback=(
            config.qwen_visual_review.blocking_timeout_fallback
        ),
        max_result_age_s=config.frame_store.max_age_s,
        await_revision_completion=await_revision_completion,
    )
    return _VisualRuntime(
        coordinator=coordinator,
        worker=worker,
        event_bus=event_bus,
        output_parent=Path(args.output_dir),
        model_name=visual_client.model,
    )


def _run_until_terminal(
    *,
    simulation_app: object,
    debug_visualizer: object | None,
    environment: object,
    perception: object,
    agent: object,
    manager: object,
    clock: object,
    task_start_s: float,
    max_sim_time_s: float,
    shutdown_guard_s: float,
    debug_ground_truth: bool,
    injections: tuple[TestInjectionSpec, ...],
    visual_runtime: _VisualRuntime | None,
) -> object:
    """Advance on fresh Camera samples; all HTTP work stays in the worker."""

    from scripts.run_llm_oracle_pipeline import _LoopResult, _print_ground_truth

    snapshot = agent.snapshot()
    fired: set[str] = set()
    cancel_requested = False
    landing_deadline_s: float | None = None
    next_debug_time_s = task_start_s

    while simulation_app.is_running() and snapshot.status.value == "RUNNING":
        now = clock.now()
        if snapshot.active_skill == "LAND" and landing_deadline_s is None:
            landing_deadline_s = now + shutdown_guard_s
        if not cancel_requested and now - task_start_s > max_sim_time_s:
            print(
                "[MissionAgent] external max-sim-time reached; requesting fail-safe LAND",
                file=sys.stderr,
            )
            cancel_requested = True
            snapshot = agent.cancel()
            landing_deadline_s = now + shutdown_guard_s
        if landing_deadline_s is not None and now > landing_deadline_s:
            return _LoopResult(
                snapshot,
                "fail-safe LAND exceeded its separate shutdown guard",
            )

        # environment.step() is the fresh-camera gate.  No coordinator/Agent
        # call occurs on rendering ticks that did not produce a new RGB sample.
        if not environment.step():
            continue
        frame = environment.get_evaluator_frame()
        observation = perception.observe(frame)
        if debug_ground_truth and observation.timestamp >= next_debug_time_s:
            _print_ground_truth(frame, observation.timestamp)
            next_debug_time_s = observation.timestamp + 1.0

        elapsed_s = max(0.0, clock.now() - task_start_s)
        if visual_runtime is not None:
            for spec in injections:
                if spec.flag_name in fired or elapsed_s < spec.trigger_at_s:
                    continue
                current = agent.snapshot()
                assert current.mission_id is not None
                assert current.plan_version is not None
                event = make_test_injection_event(
                    spec,
                    mission_id=current.mission_id,
                    uav_id=current.uav_id,
                    plan_version=current.plan_version,
                    timestamp_s=observation.timestamp,
                )
                agent.submit_review_event(event)
                step_id = manager.active_planned_step_id or "none"
                visual_runtime.log_injection(
                    event,
                    skill=current.active_skill or "NONE",
                    step_id=step_id,
                )
                fired.add(spec.flag_name)

        # MissionAgent feeds the same fresh frame to the non-blocking visual
        # coordinator only after hard Safety permits it, then advances Manager.
        snapshot = agent.tick(observation)
        if debug_visualizer is not None:
            # 如果运行时计划修订成功，更新灰色计划路线和目标点。
            revision_coordinator = (
                None
                if visual_runtime is None
                else visual_runtime.revision_coordinator
            )
            accepted_revision = (
                None
                if revision_coordinator is None
                else revision_coordinator.latest_accepted_revision
            )
            if (
                accepted_revision is not None
                and accepted_revision.compiled_mission.task_plan.plan_version
                != debug_visualizer.plan_version
            ):
                debug_visualizer.set_plan(
                    accepted_revision.compiled_mission
                )

            debug_visualizer.update(
                uav_pose=observation.uav_pose,
                camera_position_m=observation.camera_position_m,
                camera_orientation_wxyz=(
                    observation.camera_orientation_wxyz
                ),
                active_skill=snapshot.active_skill,
                active_step_id=manager.active_planned_step_id,
                target_lifecycle=snapshot.target.lifecycle,
            )
        if visual_runtime is not None:
            visual_runtime.emit_new_reviews(
                step_id=manager.active_planned_step_id,
                skill=snapshot.active_skill or "NONE",
            )
            visual_runtime.emit_new_events(
                skill=snapshot.active_skill or "NONE",
                step_id=manager.active_planned_step_id or "none",
            )
            visual_runtime.emit_new_transitions(manager)
            visual_runtime.emit_new_revisions(
                skill=snapshot.active_skill or "NONE",
                step_id=manager.active_planned_step_id or "none",
            )

    if snapshot.status.value == "RUNNING":
        return _LoopResult(
            snapshot,
            "SimulationApp stopped before the mission reached a terminal state",
        )
    return _LoopResult(snapshot)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if Path(sys.prefix).resolve() != EXPECTED_ENV.resolve():
        print(
            "error: run this standalone demo through "
            "./python.sh scripts/run_dynamic_visual_mission.py",
            file=sys.stderr,
        )
        return 2

    try:
        config = load_config(args.config)
        config = replace(config, uav=replace(config.uav, id=args.uav_id))
        review_mode, injections = validate_launch_args(
            args,
            configured_review_mode=config.qwen_visual_review.mode,
        )
    except (ConfigError, LaunchConfigurationError) as exc:
        print(f"launch configuration error: {exc}", file=sys.stderr)
        return 2
    if not 0.0 <= args.start_altitude <= config.scene.size_xyz_m[2]:
        print("error: --start-altitude is outside scene Z bounds", file=sys.stderr)
        return 2
    if not 0.0 < args.takeoff_altitude <= config.scene.size_xyz_m[2]:
        print("error: --takeoff-altitude is outside scene Z bounds", file=sys.stderr)
        return 2
    if args.start_altitude > args.takeoff_altitude:
        print("error: --start-altitude exceeds --takeoff-altitude", file=sys.stderr)
        return 2

    # Pure-Python planning and policy setup precede SimulationApp.
    from models import OpenAICompatibleClient
    from planner import (
        DynamicLLMPlanner,
        LLMPlanner,
        MissionIntent,
        PlannerPolicy,
        QwenPlanRevisionPlanner,
        RevisionLimits,
        RevisionValidator,
        ScriptedDynamicPlanner,
        ScriptedPlanner,
        SkillPlanDraft,
        build_default_skill_catalog,
    )
    from runtime import PlanValidator, PlannerLimits, SafetySupervisor
    from runtime.world_context_builder import (
        WorldContextBuildError,
        build_planner_world_context,
    )
    from scripts.run_llm_oracle_pipeline import (
        IsaacSimulationClock,
        _CountingModelClient,
        _best_effort_interrupt_land,
        _landing_zone_name,
        _print_plan,
        _print_runtime_summary,
        _random_target_spawn,
        _shutdown_guard_s,
        _standard_dynamic_draft_data,
    )

    try:
        world_context = build_planner_world_context(
            config,
            takeoff_altitude_m=args.takeoff_altitude,
            track_duration_s=args.track_duration,
            start_altitude_m=args.start_altitude,
            land_timeout_s=max(60.0, 3.0 * config.scene.size_xyz_m[2]),
        )
    except WorldContextBuildError as exc:
        print(f"world context error: {exc}", file=sys.stderr)
        return 2

    planner_limits = PlannerLimits.from_config(config.planner)
    planner_policy = PlannerPolicy.from_config(config.planner, planner_limits)
    counting_client: _CountingModelClient | None = None
    selected_model: str | None = None
    if args.planner == "scripted":
        planner = ScriptedPlanner(
            MissionIntent(
                target_description="moving target",
                search_region="search_area",
                track_duration_s=args.track_duration,
                landing_zone="home",
                takeoff_altitude_m=args.takeoff_altitude,
            )
        )
    elif args.planner == "dynamic_scripted":
        planner = ScriptedDynamicPlanner(
            SkillPlanDraft.from_dict(_standard_dynamic_draft_data(args))
        )
    else:
        planner_client = OpenAICompatibleClient(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
        )
        selected_model = planner_client.model
        counting_client = _CountingModelClient(planner_client)
        if args.planner == "llm":
            planner = LLMPlanner(
                counting_client,
                PROJECT_ROOT / "prompts" / "mission_planner_system.txt",
            )
        else:
            planner = DynamicLLMPlanner(
                counting_client,
                PROJECT_ROOT / "prompts" / "dynamic_skill_planner_system.txt",
                planner_limits=planner_limits,
                planner_policy=planner_policy,
            )

    validator = PlanValidator(planner_limits, planner_policy)
    shutdown_guard_s = _shutdown_guard_s(
        max(args.takeoff_altitude, config.scene.size_xyz_m[2]),
        args.start_altitude,
    )
    safety = SafetySupervisor(
        world_context.scene_min_xyz_m,
        world_context.scene_max_xyz_m,
        max_mission_time_s=args.max_sim_time + shutdown_guard_s + 1.0,
        position_margin_m=0.25,
        max_safe_altitude_m=world_context.scene_max_xyz_m[2],
        planner_limits=planner_limits,
    )

    # Standalone ordering boundary: this is the first Isaac import.  All
    # env/simple_uav_search_env and omni/pxr-backed imports remain below the
    # successfully constructed SimulationApp.
    from isaacsim import SimulationApp

    headless = (
        config.simulation.headless
        if args.headless is None
        else args.headless
    )

    if args.debug_visualization and headless:
        print(
            "error: --debug-visualization requires --no-headless",
            file=sys.stderr,
        )
        return 2

    simulation_app = SimulationApp({"headless": headless})

    if args.debug_visualization:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.util.debug_draw")
        simulation_app.update()
    environment = None
    agent = None
    perception = None
    clock = None
    visual_runtime: _VisualRuntime | None = None
    debug_visualizer = None
    interrupted = False
    try:
        from agents.mission_agent import MissionAgent
        from agents.plan_revision_coordinator import PlanRevisionCoordinator
        from env.simple_uav_search_env import SimpleUavSearchEnv
        from perception import (
            CandidateBank,
            OracleEvaluationCandidateResolver,
            PerceptionRuntimeProfile,
        )
        from runtime import FrameStore
        from skills.inspect import InspectSkill
        from skills.manager import SkillManager, create_default_skill_registry
        from target.target_manager import TargetManager

        environment = SimpleUavSearchEnv(config)
        environment.setup()
        if not headless:
            environment.configure_overview_viewport()
        configured_start = config.uav.initial_position_xyz_m
        environment.set_uav_pose(
            (configured_start[0], configured_start[1], args.start_altitude)
        )
        target_spawn = _random_target_spawn(config)
        environment.set_target_pose(target_spawn)
        if args.debug_ground_truth:
            print(f"[GroundTruth] target_spawn_m={target_spawn}")

        profile = PerceptionRuntimeProfile.ORACLE_EVALUATION
        print(
            "[Perception] PRIVILEGED ORACLE_EVALUATION explicitly enabled; "
            "geometry_source=oracle_evaluation, never qwen_vl"
        )
        perception = _build_oracle_evaluation_backend(args.uav_id)
        clock = IsaacSimulationClock(environment)
        context = environment.make_skill_context(clock, perception=perception)
        frame_store = FrameStore(
            max_frames=config.frame_store.max_frames,
            max_bytes=config.frame_store.max_bytes,
            max_age_s=config.frame_store.max_age_s,
        )
        candidate_bank = CandidateBank(uav_id=args.uav_id)

        def oracle_candidate_position(
            resolved_uav_id: str,
            candidate_id: str,
            timestamp_s: float,
        ) -> object:
            # This provider exists only inside the explicitly acknowledged
            # ORACLE_EVALUATION process.  Neither Qwen nor MissionAgent sees
            # the returned world coordinate.
            del candidate_id, timestamp_s
            if resolved_uav_id != args.uav_id:
                raise ValueError("candidate resolver UAV route mismatch")
            return environment.get_evaluator_frame().target_position_m

        candidate_resolver = OracleEvaluationCandidateResolver(
            oracle_candidate_position,
            profile=profile,
            acknowledge_privileged_oracle=True,
        )
        inspect_skill = InspectSkill(
            candidate_bank=candidate_bank,
            candidate_resolver=candidate_resolver,
            frame_store=frame_store,
        )
        manager = SkillManager(
            context,
            registry=create_default_skill_registry(
                transit_yaw_mode=config.search.transit_yaw_mode,
                inspect_skill=inspect_skill,
            ),
        )
        target_manager = TargetManager()
        revision_enabled = bool(
            args.enable_qwen_vision
            and review_mode == "gate"
            and config.plan_revision.enabled
            and args.planner in {"dynamic_scripted", "dynamic_llm"}
        )
        if args.enable_qwen_vision:
            visual_runtime = _create_visual_runtime(
                args=args,
                config=config,
                manager=manager,
                target_manager=target_manager,
                candidate_bank=candidate_bank,
                frame_store=frame_store,
                review_mode=review_mode,
                await_revision_completion=revision_enabled,
            )

        agent_kwargs: dict[str, object] = {}
        if visual_runtime is not None:
            agent_kwargs["visual_review_coordinator"] = visual_runtime.coordinator
        if revision_enabled:
            assert visual_runtime is not None
            revision_limits = RevisionLimits(
                max_plan_revisions=config.plan_revision.max_revisions,
                cooldown_s=config.plan_revision.cooldown_s,
                max_added_steps_per_revision=3,
                max_total_plan_steps=planner_limits.max_plan_steps,
            )
            revision_validator = RevisionValidator(
                validator,
                revision_limits=revision_limits,
                safety_preflight=safety,
            )
            revision_planner = QwenPlanRevisionPlanner(
                world_context=world_context,
                skill_catalog=build_default_skill_catalog(),
                limits=planner_limits,
                policy=planner_policy,
            )
            plan_revision_coordinator = PlanRevisionCoordinator(
                uav_id=args.uav_id,
                planner=revision_planner,
                worker=visual_runtime.worker,
                validator=revision_validator,
                skill_manager=manager,
                candidate_bank=candidate_bank,
                world_context=world_context,
                safety_preflight=safety,
                original_instruction=args.instruction,
                clock=clock.now,
                request_timeout_s=config.model_worker.request_timeout_s,
            )
            visual_runtime.revision_coordinator = plan_revision_coordinator
            agent_kwargs["plan_revision_coordinator"] = plan_revision_coordinator
        agent = MissionAgent(
            planner,
            validator,
            safety,
            manager,
            target_manager,
            clock,
            perception_runtime_profile=profile,
            acknowledge_privileged_oracle=True,
            **agent_kwargs,
        )

        task_start_s = clock.now()
        compiled = agent.start(args.instruction, world_context)
        if args.debug_visualization:
            from visualization import MissionDebugDraw

            debug_visualizer = MissionDebugDraw(
                world_context=world_context,
                camera_config=config.camera,
            )
            debug_visualizer.set_plan(compiled)
        launch_snapshot = agent.snapshot()
        assert launch_snapshot.mission_id is not None
        for key, value in startup_fields(
            uav_id=args.uav_id,
            mission_id=launch_snapshot.mission_id,
            planner=args.planner,
            review_enabled=args.enable_qwen_vision,
            review_mode=review_mode,
            perception_profile=args.perception_runtime_profile,
            oracle_acknowledged=args.acknowledge_privileged_oracle,
        ):
            print(f"[Launch] {key}={value}")
        _print_plan(compiled)
        if visual_runtime is not None:
            visual_runtime.begin_logging(
                launch_snapshot.mission_id,
                manifest_context=visual_manifest_context(
                    args=args,
                    config=config,
                    mission_id=launch_snapshot.mission_id,
                    review_mode=review_mode,
                    model_name=visual_runtime.model_name,
                ),
            )
            visual_runtime.emit_new_transitions(manager)

        try:
            loop_result = _run_until_terminal(
                simulation_app=simulation_app,
                debug_visualizer=debug_visualizer,
                environment=environment,
                perception=perception,
                agent=agent,
                manager=manager,
                clock=clock,
                task_start_s=task_start_s,
                max_sim_time_s=args.max_sim_time,
                shutdown_guard_s=shutdown_guard_s,
                debug_ground_truth=args.debug_ground_truth,
                injections=injections,
                visual_runtime=visual_runtime,
            )
        except KeyboardInterrupt:
            interrupted = True
            loop_result = _best_effort_interrupt_land(
                simulation_app=simulation_app,
                environment=environment,
                oracle=perception,
                agent=agent,
                clock=clock,
                shutdown_guard_s=shutdown_guard_s,
            )

        final_pose = environment.uav_controller.get_pose()
        final_xyz = (final_pose.x, final_pose.y, final_pose.z)
        home = world_context.landing_zones[_landing_zone_name(compiled)]
        _, _, checks_passed = _print_runtime_summary(
            args=args,
            compiled=compiled,
            selected_model=(
                selected_model
                if selected_model is not None
                else (None if visual_runtime is None else visual_runtime.model_name)
            ),
            model_calls=(0 if counting_client is None else counting_client.chat_calls),
            manager=manager,
            target_manager=target_manager,
            snapshot=loop_result.snapshot,
            final_xyz_m=final_xyz,
            home_xy_m=home.position_xy_m,
            ground_altitude_m=home.ground_altitude_m,
            elapsed_s=max(0.0, clock.now() - task_start_s),
            guard_error=loop_result.guard_error,
        )
        if visual_runtime is not None:
            terminal_snapshot = loop_result.snapshot
            visual_runtime.emit_new_transitions(manager)
            visual_runtime.emit_new_revisions(
                skill=terminal_snapshot.active_skill or "NONE",
                step_id=manager.active_planned_step_id or "none",
            )
            visual_runtime.set_terminal_manifest(
                agent_status=terminal_snapshot.status.value,
                task_status=terminal_snapshot.task_status,
                plan_version=terminal_snapshot.plan_version or 1,
                guard_error=loop_result.guard_error,
            )
        if interrupted:
            return 130
        return 0 if checks_passed else 1
    except KeyboardInterrupt:
        if all(value is not None for value in (agent, environment, perception, clock)):
            _best_effort_interrupt_land(
                simulation_app=simulation_app,
                environment=environment,
                oracle=perception,
                agent=agent,
                clock=clock,
                shutdown_guard_s=shutdown_guard_s,
            )
        return 130
    except Exception as exc:
        # Never include prompts, API headers, base64 image content, or raw
        # evaluator frames in this error boundary.
        print(f"dynamic visual pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            if visual_runtime is not None:
                try:
                    visual_runtime.close(
                        timeout_s=config.model_worker.request_timeout_s + 1.0
                    )
                except Exception as exc:
                    print(
                        f"visual worker shutdown failed: {type(exc).__name__}",
                        file=sys.stderr,
                    )
            if environment is not None:
                if debug_visualizer is not None:
                    try:
                        debug_visualizer.close()
                    except Exception as exc:
                        print(
                            f"debug visualization shutdown failed: "
                            f"{type(exc).__name__}",
                            file=sys.stderr,
                        )
                environment.close()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
