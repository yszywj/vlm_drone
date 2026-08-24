"""Construct isolated OpenAI-compatible clients for trusted model call roles."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import Lock
from time import perf_counter
from typing import Any

from common.ids import validate_routing_id
from models.adapter_registry import AdapterRegistry, AdapterSelection, ModelCallRole
from models.base import ChatMessage, GenerationOptions, ModelResponse
from models.openai_compatible_client import OpenAICompatibleClient


SelectionLogger = Callable[[dict[str, object]], None]
CallLogger = Callable[[dict[str, object]], None]
ClientFactory = Callable[..., OpenAICompatibleClient]


class _RecordedModelClient:
    """Per-role client wrapper that records one row for every real chat call."""

    def __init__(
        self,
        client: object,
        selection: AdapterSelection,
        *,
        next_call_id: Callable[[], str],
        call_logger: CallLogger,
        fleet_mission_id: str | None,
        assignment_id: str | None,
        uav_id: str | None,
    ) -> None:
        self._client = client
        self._selection = selection
        self._next_call_id = next_call_id
        self._call_logger = call_logger
        self._routing = {
            "fleet_mission_id": fleet_mission_id,
            "assignment_id": assignment_id,
            "uav_id": uav_id,
        }

    @property
    def model(self) -> object:
        return getattr(self._client, "model", self._selection.effective_model)

    def healthcheck(self) -> None:
        healthcheck = getattr(self._client, "healthcheck", None)
        if not callable(healthcheck):
            raise TypeError("model client must provide healthcheck()")
        healthcheck()

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: GenerationOptions | None = None,
    ) -> ModelResponse:
        chat = getattr(self._client, "chat", None)
        if not callable(chat):
            raise TypeError("model client must provide chat()")
        call_id = self._next_call_id()
        started = perf_counter()
        response: object | None = None
        error_code: str | None = None
        try:
            response = chat(messages, options=options)
            return response  # type: ignore[return-value]
        except Exception as exc:
            error_code = type(exc).__name__
            raise
        finally:
            usage = response.usage if isinstance(response, ModelResponse) else {}
            finish_reason = (
                response.finish_reason if isinstance(response, ModelResponse) else None
            )
            if response is not None and not isinstance(response, ModelResponse):
                error_code = "INVALID_MODEL_RESPONSE"
            self._call_logger(
                {
                    "call_id": call_id,
                    "call_role": self._selection.call_role.value,
                    **self._routing,
                    "requested_adapter": self._selection.requested_adapter,
                    "adapter_status": self._selection.adapter_status.value,
                    "effective_model": self._selection.effective_model,
                    "fallback_used": self._selection.fallback_used,
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "latency_s": max(0.0, perf_counter() - started),
                    "finish_reason": finish_reason,
                    "error_code": error_code,
                    "stale_reasons": [],
                }
            )


class ModelClientFactory:
    """Never mutates a shared client's model field; each call gets a new client."""

    def __init__(
        self,
        registry: AdapterRegistry,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
        max_retries: int = 2,
        client_factory: ClientFactory = OpenAICompatibleClient,
        selection_logger: SelectionLogger | None = None,
        call_logger: CallLogger | None = None,
        call_id_prefix: str = "model_call",
        **client_options: Any,
    ) -> None:
        if not isinstance(registry, AdapterRegistry):
            raise TypeError("registry must be an AdapterRegistry")
        if not callable(client_factory):
            raise TypeError("client_factory must be callable")
        if selection_logger is not None and not callable(selection_logger):
            raise TypeError("selection_logger must be callable")
        if call_logger is not None and not callable(call_logger):
            raise TypeError("call_logger must be callable")
        call_id_prefix = validate_routing_id(call_id_prefix, "call_id_prefix")
        # Keep room for the underscore plus the fixed eight-digit sequence so
        # every generated ID remains inside the shared 64-character contract.
        if len(call_id_prefix) > 55:
            raise ValueError("call_id_prefix must contain at most 55 characters")
        self._registry = registry
        self._base_url = base_url
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._client_factory = client_factory
        self._selection_logger = selection_logger
        self._call_logger = call_logger
        self._call_id_prefix = call_id_prefix
        self._client_options = dict(client_options)
        self._call_lock = Lock()
        self._call_sequence = 0

    def selection_for_role(self, role: ModelCallRole | str) -> AdapterSelection:
        return self._registry.resolve(role)

    def for_role(
        self,
        role: ModelCallRole | str,
        *,
        fleet_mission_id: str | None = None,
        assignment_id: str | None = None,
        uav_id: str | None = None,
    ) -> OpenAICompatibleClient:
        selection = self.selection_for_role(role)
        if self._selection_logger is not None:
            self._selection_logger(selection.to_dict())
        client = self._client_factory(
            base_url=self._base_url,
            model=selection.effective_model,
            api_key=self._api_key,
            timeout_s=self._timeout_s,
            max_retries=self._max_retries,
            **self._client_options,
        )
        if self._call_logger is None:
            return client
        return _RecordedModelClient(
            client,
            selection,
            next_call_id=self._next_call_id,
            call_logger=self._call_logger,
            fleet_mission_id=fleet_mission_id,
            assignment_id=assignment_id,
            uav_id=uav_id,
        )  # type: ignore[return-value]

    def _next_call_id(self) -> str:
        with self._call_lock:
            self._call_sequence += 1
            return f"{self._call_id_prefix}_{self._call_sequence:08d}"


__all__ = ["ModelClientFactory"]
