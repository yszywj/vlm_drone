"""Non-blocking, fail-closed coordination of Qwen MissionProgram patches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import Protocol

from common.ids import generate_routing_id, validate_uav_id
from models import AsyncModelResult
from planner.mission_program import ProgramActionOp, ProgramEvent
from planner.program_patch import ProgramPatch
from planner.program_patch_planner import (
    ProgramPatchRequest,
    QwenProgramPatchPlanner,
)
from runtime.program_executor import ProgramEventDispatch, ProgramExecutorSnapshot


class _Worker(Protocol):
    uav_id: str

    def submit(self, request: object) -> None: ...

    def poll(self, **kwargs: object) -> AsyncModelResult | None: ...


class _GraphManager(Protocol):
    uav_id: str

    @property
    def program_snapshot(self) -> ProgramExecutorSnapshot | None: ...

    @property
    def task_status(self) -> object: ...

    def graph_program_for_revision(self) -> object: ...

    def dispatch_program_event(
        self,
        event: ProgramEvent,
        *,
        expected_plan_version: int,
        defer_observation_timestamp_s: float | None = None,
    ) -> ProgramEventDispatch: ...

    def replace_interrupted_program_suffix(self, patch: ProgramPatch) -> object: ...

    def cancel_task(self) -> object: ...


class ProgramPatchCoordinatorState(str, Enum):
    IDLE = "IDLE"
    AWAITING_MODEL = "AWAITING_MODEL"
    ACCEPTED = "ACCEPTED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"


@dataclass(frozen=True, slots=True)
class ProgramPatchCoordinatorRecord:
    request_id: str
    mission_id: str
    uav_id: str
    base_plan_version: int
    new_plan_version: int | None
    current_node_id: str
    event: str
    submitted_timestamp_s: float
    completed_timestamp_s: float | None
    outcome: str
    error_code: str | None
    accepted_patch: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "base_plan_version": self.base_plan_version,
            "new_plan_version": self.new_plan_version,
            "current_node_id": self.current_node_id,
            "event": self.event,
            "submitted_timestamp_s": self.submitted_timestamp_s,
            "completed_timestamp_s": self.completed_timestamp_s,
            "outcome": self.outcome,
            "error_code": self.error_code,
            "accepted_patch": self.accepted_patch,
        }


@dataclass(frozen=True, slots=True)
class ProgramPatchCoordinatorSnapshot:
    state: ProgramPatchCoordinatorState
    request_id: str | None
    mission_id: str | None
    base_plan_version: int | None
    current_node_id: str | None
    submitted_timestamp_s: float | None
    deadline_timestamp_s: float | None
    error_code: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "request_id": self.request_id,
            "mission_id": self.mission_id,
            "base_plan_version": self.base_plan_version,
            "current_node_id": self.current_node_id,
            "submitted_timestamp_s": self.submitted_timestamp_s,
            "deadline_timestamp_s": self.deadline_timestamp_s,
            "error_code": self.error_code,
        }


class ProgramPatchCoordinator:
    """Own one ``PATH_BLOCKED -> HOVER -> Qwen patch`` transaction.

    All model work runs behind an asynchronous worker.  A missing, stale, or
    malformed response never resumes the interrupted movement: the coordinator
    requests trusted cancel-and-land instead.
    """

    def __init__(
        self,
        *,
        uav_id: str,
        planner: QwenProgramPatchPlanner,
        worker: _Worker,
        skill_manager: _GraphManager,
        request_timeout_s: float = 20.0,
        max_records: int = 32,
        logger: Callable[[ProgramPatchCoordinatorRecord], None] | None = None,
    ) -> None:
        self._uav_id = validate_uav_id(uav_id)
        if not isinstance(planner, QwenProgramPatchPlanner):
            raise TypeError("planner must be a QwenProgramPatchPlanner")
        if validate_uav_id(getattr(worker, "uav_id", None)) != self._uav_id:
            raise ValueError("worker.uav_id must match coordinator uav_id")
        if not callable(getattr(worker, "submit", None)) or not callable(
            getattr(worker, "poll", None)
        ):
            raise TypeError("worker must provide submit and poll")
        if validate_uav_id(getattr(skill_manager, "uav_id", None)) != self._uav_id:
            raise ValueError("SkillManager.uav_id must match coordinator uav_id")
        for method in (
            "graph_program_for_revision",
            "dispatch_program_event",
            "replace_interrupted_program_suffix",
            "cancel_task",
        ):
            if not callable(getattr(skill_manager, method, None)):
                raise TypeError(f"skill_manager must provide {method}()")
        timeout = _timestamp(request_timeout_s, "request_timeout_s")
        if timeout <= 0.0 or timeout > 300.0:
            raise ValueError("request_timeout_s must be within (0, 300]")
        if (
            isinstance(max_records, bool)
            or not isinstance(max_records, int)
            or not 1 <= max_records <= 1024
        ):
            raise ValueError("max_records must be within [1, 1024]")
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable or None")
        self._planner = planner
        self._worker = worker
        self._manager = skill_manager
        self._timeout_s = timeout
        self._max_records = max_records
        self._logger = logger
        self._state = ProgramPatchCoordinatorState.IDLE
        self._request: ProgramPatchRequest | None = None
        self._request_id: str | None = None
        self._submitted_timestamp_s: float | None = None
        self._deadline_timestamp_s: float | None = None
        self._error_code: str | None = None
        self._records: list[ProgramPatchCoordinatorRecord] = []

    @property
    def uav_id(self) -> str:
        return self._uav_id

    @property
    def records(self) -> tuple[ProgramPatchCoordinatorRecord, ...]:
        return tuple(self._records)

    @property
    def is_inflight(self) -> bool:
        return self._state is ProgramPatchCoordinatorState.AWAITING_MODEL

    def begin(
        self,
        *,
        expected_plan_version: int,
        observation_timestamp_s: float,
        frame_id: str,
        defer_observation_timestamp_s: float | None = None,
    ) -> ProgramPatchCoordinatorSnapshot:
        if self._state is not ProgramPatchCoordinatorState.IDLE:
            raise RuntimeError("ProgramPatchCoordinator is already active")
        snapshot = self._manager.program_snapshot
        if snapshot is None or snapshot.current_node_id is None:
            raise RuntimeError("ProgramPatch requires a non-terminal graph runtime")
        if expected_plan_version != snapshot.plan_version:
            raise RuntimeError("PATH_BLOCKED trigger plan version is stale")
        program = self._manager.graph_program_for_revision()
        handler = next(
            (
                candidate
                for candidate in program.event_handlers
                if candidate.on is ProgramEvent.PATH_BLOCKED
            ),
            None,
        )
        if (
            handler is None
            or len(handler.actions) != 2
            or handler.actions[0].op is not ProgramActionOp.HOLD
            or handler.actions[1].op is not ProgramActionOp.REPLAN_CURRENT_ROUTE
        ):
            raise RuntimeError(
                "PATH_BLOCKED does not select a HOLD/QWEN ProgramPatch handler"
            )
        timestamp = _timestamp(observation_timestamp_s, "observation_timestamp_s")
        request = ProgramPatchRequest(
            program=program,
            current_node_id=snapshot.current_node_id,
            completed_node_ids=snapshot.completed_node_ids,
            trigger_event=ProgramEvent.PATH_BLOCKED,
            observation_timestamp_s=timestamp,
            frame_id=frame_id,
        )
        dispatch = self._manager.dispatch_program_event(
            ProgramEvent.PATH_BLOCKED,
            expected_plan_version=expected_plan_version,
            defer_observation_timestamp_s=defer_observation_timestamp_s,
        )
        if (
            dispatch.plan_version != snapshot.plan_version
            or dispatch.current_node_id != snapshot.current_node_id
            or dispatch.actions != handler.actions
        ):
            self._fail_closed("PROGRAM_EVENT_DISPATCH_CHANGED")
            return self.snapshot()

        request_id = generate_routing_id("request_program_patch")
        self._request = request
        self._request_id = request_id
        self._submitted_timestamp_s = timestamp
        self._deadline_timestamp_s = timestamp + self._timeout_s
        try:
            model_request = self._planner.build_async_request(
                request,
                request_id=request_id,
            )
            self._worker.submit(model_request)
        except Exception as exc:
            self._fail_closed(
                "PROGRAM_PATCH_SUBMIT_FAILED",
                completed_timestamp_s=timestamp,
                detail=str(exc),
            )
            return self.snapshot()
        self._state = ProgramPatchCoordinatorState.AWAITING_MODEL
        return self.snapshot()

    def tick(self, *, timestamp_s: float) -> ProgramPatchCoordinatorSnapshot:
        if self._state is not ProgramPatchCoordinatorState.AWAITING_MODEL:
            return self.snapshot()
        now = _timestamp(timestamp_s, "timestamp_s")
        assert self._request is not None
        assert self._request_id is not None
        assert self._deadline_timestamp_s is not None
        result = self._worker.poll(
            expected_request_id=self._request_id,
            include_stale=True,
        )
        if result is None:
            if now >= self._deadline_timestamp_s:
                self._state = ProgramPatchCoordinatorState.TIMED_OUT
                self._fail_closed(
                    "PROGRAM_PATCH_TIMEOUT",
                    completed_timestamp_s=now,
                    preserve_state=True,
                )
            return self.snapshot()
        try:
            patch = self._planner.parse_async_result(
                result,
                request=self._request,
                expected_request_id=self._request_id,
            )
            self._manager.replace_interrupted_program_suffix(patch)
        except Exception as exc:
            self._fail_closed(
                "PROGRAM_PATCH_REJECTED",
                completed_timestamp_s=now,
                detail=str(exc),
            )
            return self.snapshot()
        self._state = ProgramPatchCoordinatorState.ACCEPTED
        self._error_code = None
        self._append_record(
            completed_timestamp_s=now,
            outcome="ACCEPTED",
            error_code=None,
            patch=patch,
        )
        return self.snapshot()

    def snapshot(self) -> ProgramPatchCoordinatorSnapshot:
        request = self._request
        return ProgramPatchCoordinatorSnapshot(
            state=self._state,
            request_id=self._request_id,
            mission_id=None if request is None else request.program.mission_id,
            base_plan_version=(
                None if request is None else request.program.plan_version
            ),
            current_node_id=(
                None if request is None else request.current_node_id
            ),
            submitted_timestamp_s=self._submitted_timestamp_s,
            deadline_timestamp_s=self._deadline_timestamp_s,
            error_code=self._error_code,
        )

    def reset(self) -> None:
        if (
            self._state is ProgramPatchCoordinatorState.AWAITING_MODEL
            and getattr(self._manager.task_status, "name", None) == "RUNNING"
        ):
            raise RuntimeError("cannot reset an in-flight ProgramPatch request")
        self._state = ProgramPatchCoordinatorState.IDLE
        self._request = None
        self._request_id = None
        self._submitted_timestamp_s = None
        self._deadline_timestamp_s = None
        self._error_code = None

    def _fail_closed(
        self,
        error_code: str,
        *,
        completed_timestamp_s: float | None = None,
        detail: str | None = None,
        preserve_state: bool = False,
    ) -> None:
        del detail  # Never retain arbitrary model/transport text in sparse audit.
        self._error_code = error_code
        if not preserve_state:
            self._state = ProgramPatchCoordinatorState.FAILED
        try:
            self._manager.cancel_task()
        except Exception:
            # Manager-owned HOVER timeout is the second fail-closed boundary;
            # never attempt an untrusted resume if cancel itself cannot start.
            pass
        if self._request is not None and self._request_id is not None:
            self._append_record(
                completed_timestamp_s=completed_timestamp_s,
                outcome=self._state.value,
                error_code=error_code,
                patch=None,
            )

    def _append_record(
        self,
        *,
        completed_timestamp_s: float | None,
        outcome: str,
        error_code: str | None,
        patch: ProgramPatch | None,
    ) -> None:
        request = self._request
        request_id = self._request_id
        submitted = self._submitted_timestamp_s
        if request is None or request_id is None or submitted is None:
            return
        record = ProgramPatchCoordinatorRecord(
            request_id=request_id,
            mission_id=request.program.mission_id,
            uav_id=request.program.uav_id,
            base_plan_version=request.program.plan_version,
            new_plan_version=None if patch is None else patch.new_plan_version,
            current_node_id=request.current_node_id,
            event=request.trigger_event.value,
            submitted_timestamp_s=submitted,
            completed_timestamp_s=completed_timestamp_s,
            outcome=outcome,
            error_code=error_code,
            accepted_patch=None if patch is None else patch.to_dict(),
        )
        self._records.append(record)
        if len(self._records) > self._max_records:
            del self._records[: len(self._records) - self._max_records]
        if self._logger is not None:
            self._logger(record)


def _timestamp(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


__all__ = [
    "ProgramPatchCoordinator",
    "ProgramPatchCoordinatorRecord",
    "ProgramPatchCoordinatorSnapshot",
    "ProgramPatchCoordinatorState",
]
