"""A callable planning capability; never an executable UAV Skill.

The host binds the planning service to trusted configuration and model routing.
The caller supplies only an instruction. Initial global planning returns an
assignment proposal; local compilation and execution remain host responsibilities.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import json

from common.ids import generate_routing_id
from fleet.strict_json import strict_json_object_loads


class PlanningToolInputError(ValueError):
    """Tool arguments do not match the public instruction-only contract."""


class FleetPlanningTool:
    """Expose a host-owned FleetPlanningService as the ``plan_fleet`` tool."""

    name = "plan_fleet"

    def __init__(self, service: object) -> None:
        if not callable(getattr(service, "plan", None)):
            raise TypeError("planning service must provide plan()")
        self._service = service

    @staticmethod
    def tool_definition() -> dict[str, object]:
        """Return a fresh chat tool definition; adapters stay host-controlled."""
        return {
            "type": "function",
            "function": {
                "name": FleetPlanningTool.name,
                "description": (
                    "Propose a global UAV mission assignment from the original "
                    "instruction using the host's trusted scene and planning models. "
                    "Returns interpreted goals, assignments and coverage findings. "
                    "Does not compile local flight plans or execute the mission."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 8192,
                            "description": "Complete original mission instruction, preserving all constraints.",
                        },
                    },
                    "required": ["instruction"],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _instruction(arguments: object) -> str:
        if isinstance(arguments, str):
            try:
                argument_bytes = arguments.encode("utf-8")
            except UnicodeError as exc:
                raise PlanningToolInputError("tool arguments must be valid UTF-8 text") from exc
            if len(argument_bytes) > 65_536:
                raise PlanningToolInputError("tool arguments exceed 64 KiB")
            try:
                arguments = strict_json_object_loads(arguments)
            except (TypeError, ValueError, RecursionError) as exc:
                raise PlanningToolInputError("tool arguments must be a unique-key JSON object") from exc
        if not isinstance(arguments, Mapping) or set(arguments) != {"instruction"}:
            raise PlanningToolInputError("plan_fleet accepts only the instruction field")
        instruction = arguments["instruction"]
        if (
            not isinstance(instruction, str)
            or not instruction.strip()
            or len(instruction) > 8192
            or "\x00" in instruction
        ):
            raise PlanningToolInputError("instruction must contain 1..8192 characters without NUL")
        return instruction

    def invoke(
        self,
        arguments: object,
        *,
        audit_context: MutableMapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Plan once through the service's bounded retries, without executing.

        Invalid arguments raise before any model call. Planning failures return a
        bounded error type; raw proposals and diagnostics stay in the host audit.
        """
        instruction = self._instruction(arguments)
        audit = {} if audit_context is None else audit_context
        mission_id = generate_routing_id("fleet_mission")
        audit.update({
            "fleet_mission_id": mission_id,
            "raw_instruction": instruction,
            "source_text": instruction.strip(),
        })
        envelope = {
            "schema_version": 1,
            "tool": self.name,
            "fleet_mission_id": mission_id,
            "execution_started": False,
            "executable": False,
            "local_compilation_performed": False,
        }
        try:
            result = self._service.plan(
                instruction, fleet_mission_id=mission_id, audit_context=audit,
            )
            payload = result.to_dict()
        except Exception as exc:
            # Exception messages can contain model text, endpoint URLs or keys.
            # Detailed proposal data is intentionally confined to the host audit.
            return {
                **envelope,
                "status": "planning_failed",
                "error_type": type(exc).__name__,
                "failed_stage": audit.get("planning_stage", "mission_interpretation"),
            }
        completeness = payload["completeness"]
        return {
            **envelope,
            "status": "proposal_ready" if completeness["assignment_complete"] else "incomplete_assignment",
            "task_spec": payload["task_spec"],
            "fleet_plan": payload["plan"],
            "completeness": completeness,
            "semantic_findings": payload["semantic_findings"],
        }

    def handle_tool_call(
        self,
        tool_call: Mapping[str, object],
        *,
        audit_context: MutableMapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Dispatch one chat function call and return its correlated tool reply.

        The application explicitly opts into this entry point. Flight mission
        creation does not depend on a language model deciding to call this tool.
        """
        if not isinstance(tool_call, Mapping):
            raise PlanningToolInputError("tool_call must be an object")
        call_id = tool_call.get("id")
        if not isinstance(call_id, str) or not call_id.strip() or len(call_id) > 256:
            raise PlanningToolInputError("tool_call requires a bounded id")
        function = tool_call.get("function")
        if (
            tool_call.get("type") != "function"
            or not isinstance(function, Mapping)
            or function.get("name") != self.name
        ):
            raise PlanningToolInputError("unsupported tool call")
        try:
            result = self.invoke(function.get("arguments"), audit_context=audit_context)
        except PlanningToolInputError as exc:
            result = {
                "status": "invalid_arguments", "message": str(exc),
                "execution_started": False, "executable": False,
            }
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result, ensure_ascii=False, allow_nan=False),
        }


__all__ = ["FleetPlanningTool", "PlanningToolInputError"]
