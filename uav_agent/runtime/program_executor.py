"""Pure-Python executor state for validated experimental MissionPrograms."""

from __future__ import annotations

from dataclasses import dataclass

from planner.mission_program import (
    MissionProgram,
    MissionProgramError,
    ProgramAction,
    ProgramEvent,
)
from planner.program_patch import ProgramPatch, apply_program_patch
from skills.plan import TaskStep


@dataclass(frozen=True, slots=True)
class ProgramExecutorSnapshot:
    mission_id: str
    uav_id: str
    plan_version: int
    current_node_id: str | None
    completed_node_ids: tuple[str, ...]
    terminal: bool


@dataclass(frozen=True, slots=True)
class ProgramEventDispatch:
    """Version-pinned resolution of one externally supplied graph event.

    Preparing a dispatch is side-effect free.  A caller may therefore establish
    supervisory HOVER and validate every downstream action before committing a
    static edge.  Reusing the token after either a patch or another graph event
    is rejected by :meth:`ProgramExecutor.commit_dispatch`.
    """

    mission_id: str
    uav_id: str
    plan_version: int
    current_node_id: str
    event: ProgramEvent
    actions: tuple[ProgramAction, ...]
    target_node_id: str | None


class ProgramExecutor:
    """Advance graph control flow for a bound :class:`SkillManager` runtime.

    The executor owns only immutable routing state.  ``SkillManager`` remains
    the sole owner of Skill lifecycles and calls :meth:`handle_event` after a
    terminal Skill result has been validated.  A node is completed once its
    terminal event is consumed, regardless of whether that event represents
    success or a handled failure branch.
    """

    def __init__(self, program: MissionProgram) -> None:
        if not isinstance(program, MissionProgram):
            raise TypeError("program must be a MissionProgram")
        self._program = program
        self._current_node_id: str | None = program.entry_node_id
        self._completed: list[str] = []

    @property
    def current_step(self) -> TaskStep | None:
        if self._current_node_id is None:
            return None
        return self._program.node(self._current_node_id).step

    @property
    def program(self) -> MissionProgram:
        return self._program

    def has_transition(self, event: ProgramEvent | str) -> bool:
        """Return whether the current node has an edge for ``event``.

        This is intentionally non-mutating so the Manager can apply the
        precedence ``specific event -> generic SUCCESS/FAILURE`` atomically.
        """

        if self._current_node_id is None:
            return False
        try:
            normalized = event if isinstance(event, ProgramEvent) else ProgramEvent(event)
        except (TypeError, ValueError):
            raise MissionProgramError("unsupported MissionProgram event") from None
        return any(
            edge.source_node_id == self._current_node_id and edge.on is normalized
            for edge in self._program.edges
        )

    def prepare_dispatch(
        self,
        event: ProgramEvent | str,
        *,
        expected_plan_version: int,
    ) -> ProgramEventDispatch:
        """Resolve a handler or edge without mutating executor state.

        External runtime events must always carry the caller's observed plan
        version.  This closes the race in which a late ``PATH_BLOCKED`` result
        could otherwise be applied to a newly published program suffix.
        """

        if self._current_node_id is None:
            raise MissionProgramError("the MissionProgram is already terminal")
        if (
            isinstance(expected_plan_version, bool)
            or not isinstance(expected_plan_version, int)
            or expected_plan_version <= 0
        ):
            raise MissionProgramError(
                "expected_plan_version must be a positive integer"
            )
        if expected_plan_version != self._program.plan_version:
            raise MissionProgramError("external MissionProgram event is stale")
        try:
            normalized = (
                event if isinstance(event, ProgramEvent) else ProgramEvent(event)
            )
        except (TypeError, ValueError):
            raise MissionProgramError("unsupported MissionProgram event") from None

        handler = next(
            (
                candidate
                for candidate in self._program.event_handlers
                if candidate.on is normalized
            ),
            None,
        )
        matches = tuple(
            edge
            for edge in self._program.edges
            if edge.source_node_id == self._current_node_id
            and edge.on is normalized
        )
        if handler is not None and matches:
            raise MissionProgramError(
                "MissionProgram event is ambiguous between handler and edge"
            )
        if handler is None and not matches:
            raise MissionProgramError(
                "MissionProgram event has no handler or current-node edge"
            )
        return ProgramEventDispatch(
            mission_id=self._program.mission_id,
            uav_id=self._program.uav_id,
            plan_version=self._program.plan_version,
            current_node_id=self._current_node_id,
            event=normalized,
            actions=() if handler is None else handler.actions,
            target_node_id=None if not matches else matches[0].target_node_id,
        )

    def commit_dispatch(self, dispatch: ProgramEventDispatch) -> TaskStep | None:
        """Commit an unchanged, edge-backed dispatch token exactly once."""

        if not isinstance(dispatch, ProgramEventDispatch):
            raise TypeError("dispatch must be a ProgramEventDispatch")
        snapshot = self.snapshot()
        if (
            dispatch.mission_id != snapshot.mission_id
            or dispatch.uav_id != snapshot.uav_id
            or dispatch.plan_version != snapshot.plan_version
            or dispatch.current_node_id != snapshot.current_node_id
        ):
            raise MissionProgramError("MissionProgram dispatch token is stale")
        if dispatch.actions or dispatch.target_node_id is None:
            raise MissionProgramError(
                "only a static edge dispatch can be committed by the executor"
            )
        step = self.handle_event(dispatch.event)
        if step is None or step.step_id != dispatch.target_node_id:
            raise MissionProgramError("MissionProgram dispatch target changed")
        return step

    def handle_event(self, event: ProgramEvent | str) -> TaskStep | None:
        if self._current_node_id is None:
            raise MissionProgramError("the MissionProgram is already terminal")
        try:
            normalized = event if isinstance(event, ProgramEvent) else ProgramEvent(event)
        except (TypeError, ValueError):
            raise MissionProgramError("unsupported MissionProgram event") from None
        source = self._current_node_id
        if source not in self._completed:
            self._completed.append(source)
        matches = [
            edge
            for edge in self._program.edges
            if edge.source_node_id == source and edge.on is normalized
        ]
        if not matches:
            self._current_node_id = None
            return None
        self._current_node_id = matches[0].target_node_id
        return self.current_step

    def apply_patch(self, patch: ProgramPatch) -> None:
        if self._current_node_id is None:
            raise MissionProgramError("cannot patch a terminal MissionProgram")
        candidate = apply_program_patch(
            self._program,
            patch,
            completed_node_ids=frozenset(self._completed),
        )
        if patch.replace_from_node_id != self._current_node_id:
            raise MissionProgramError("runtime patches must begin at the current node")
        self._program = candidate

    def snapshot(self) -> ProgramExecutorSnapshot:
        return ProgramExecutorSnapshot(
            mission_id=self._program.mission_id,
            uav_id=self._program.uav_id,
            plan_version=self._program.plan_version,
            current_node_id=self._current_node_id,
            completed_node_ids=tuple(self._completed),
            terminal=self._current_node_id is None,
        )


__all__ = [
    "ProgramEventDispatch",
    "ProgramExecutor",
    "ProgramExecutorSnapshot",
]
