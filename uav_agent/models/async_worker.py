"""Bounded, per-UAV background execution for synchronous model clients."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real
import threading

from common.ids import (
    validate_mission_id,
    validate_request_id,
    validate_review_id,
    validate_routing_id,
    validate_uav_id,
)
from models.base import (
    ChatMessage,
    GenerationOptions,
    ModelClient,
    ModelClientError,
    ModelProtocolError,
    ModelResponse,
)


def _validate_plan_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("plan_version must be an integer")
    if value <= 0:
        raise ValueError("plan_version must be greater than zero")
    return value


def _validate_timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("observation_timestamp_s must be a finite number")
    timestamp = float(value)
    if not isfinite(timestamp) or timestamp < 0.0:
        raise ValueError(
            "observation_timestamp_s must be finite and non-negative"
        )
    return timestamp


@dataclass(frozen=True, slots=True)
class AsyncModelRequest:
    """One immutable routed model request submitted from the main thread."""

    request_id: str
    review_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    observation_timestamp_s: float
    frame_id: str
    messages: Sequence[ChatMessage]
    options: GenerationOptions | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", validate_request_id(self.request_id))
        object.__setattr__(self, "review_id", validate_review_id(self.review_id))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(self, "plan_version", _validate_plan_version(self.plan_version))
        object.__setattr__(
            self,
            "observation_timestamp_s",
            _validate_timestamp(self.observation_timestamp_s),
        )
        object.__setattr__(
            self,
            "frame_id",
            validate_routing_id(self.frame_id, "frame_id"),
        )
        if isinstance(self.messages, (str, bytes)) or not isinstance(
            self.messages,
            Sequence,
        ):
            raise TypeError("messages must be a sequence of ChatMessage objects")
        messages = tuple(self.messages)
        if not messages:
            raise ValueError("messages must not be empty")
        for index, message in enumerate(messages):
            if not isinstance(message, ChatMessage):
                raise TypeError(f"messages[{index}] must be a ChatMessage")
        object.__setattr__(self, "messages", messages)
        if self.options is not None and not isinstance(
            self.options,
            GenerationOptions,
        ):
            raise TypeError("options must be GenerationOptions or None")


@dataclass(frozen=True, slots=True)
class AsyncModelResult:
    """Routed completion metadata without retaining request image bytes."""

    request_id: str
    review_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    observation_timestamp_s: float
    frame_id: str
    response: ModelResponse | None
    error_code: str | None
    error_message: str | None
    stale: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", validate_request_id(self.request_id))
        object.__setattr__(self, "review_id", validate_review_id(self.review_id))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(self, "plan_version", _validate_plan_version(self.plan_version))
        object.__setattr__(
            self,
            "observation_timestamp_s",
            _validate_timestamp(self.observation_timestamp_s),
        )
        object.__setattr__(
            self,
            "frame_id",
            validate_routing_id(self.frame_id, "frame_id"),
        )
        if self.response is not None and not isinstance(self.response, ModelResponse):
            raise TypeError("response must be a ModelResponse or None")
        if self.error_code is not None and (
            not isinstance(self.error_code, str) or not self.error_code
        ):
            raise TypeError("error_code must be a non-empty string or None")
        if self.error_message is not None and not isinstance(self.error_message, str):
            raise TypeError("error_message must be a string or None")
        if (self.response is None) == (self.error_code is None):
            raise ValueError(
                "result must contain exactly one of response or error_code"
            )
        if self.error_code is None and self.error_message is not None:
            raise ValueError("successful result must not contain error_message")
        if not isinstance(self.stale, bool):
            raise TypeError("stale must be a boolean")

    @property
    def succeeded(self) -> bool:
        return self.response is not None and self.error_code is None


class AsyncModelWorker:
    """Execute at most one model request at a time for one bound UAV.

    Only the newest pending request is retained.  Submitting while another
    request is running marks the active response stale without attempting the
    unsafe operation of killing its HTTP call.
    """

    def __init__(
        self,
        client: ModelClient,
        *,
        uav_id: str,
        max_completed_results: int = 16,
    ) -> None:
        if not callable(getattr(client, "chat", None)):
            raise TypeError("client must provide a callable chat method")
        if (
            isinstance(max_completed_results, bool)
            or not isinstance(max_completed_results, int)
        ):
            raise TypeError("max_completed_results must be an integer")
        if max_completed_results <= 0:
            raise ValueError("max_completed_results must be greater than zero")

        self._client = client
        self.uav_id = validate_uav_id(uav_id)
        self._max_completed_results = max_completed_results
        self._condition = threading.Condition()
        self._pending: AsyncModelRequest | None = None
        self._active_request_id: str | None = None
        self._stale_request_ids: set[str] = set()
        self._completed: deque[AsyncModelResult] = deque()
        self._discarded_result_count = 0
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"qwen-worker-{self.uav_id}",
            daemon=True,
        )
        self._thread.start()

    @property
    def is_busy(self) -> bool:
        with self._condition:
            return self._active_request_id is not None or self._pending is not None

    @property
    def is_closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def discarded_result_count(self) -> int:
        with self._condition:
            return self._discarded_result_count

    def submit(self, request: AsyncModelRequest) -> None:
        """Submit without waiting for the synchronous model call."""

        if not isinstance(request, AsyncModelRequest):
            raise TypeError("request must be an AsyncModelRequest")
        if request.uav_id != self.uav_id:
            raise ValueError(
                f"request uav_id {request.uav_id!r} does not match worker "
                f"uav_id {self.uav_id!r}"
            )
        with self._condition:
            if self._closed:
                raise RuntimeError("model worker is closed")
            if self._active_request_id is not None:
                self._stale_request_ids.add(self._active_request_id)
            # A pending request has never crossed the HTTP boundary and can be
            # safely superseded without generating a synthetic result.
            self._pending = request
            self._condition.notify()

    def poll(
        self,
        *,
        expected_request_id: str | None = None,
        expected_review_id: str | None = None,
        minimum_observation_timestamp_s: float | None = None,
        include_stale: bool = False,
    ) -> AsyncModelResult | None:
        """Return one ready result without blocking, discarding stale entries."""

        if expected_request_id is not None:
            expected_request_id = validate_request_id(expected_request_id)
        if expected_review_id is not None:
            expected_review_id = validate_review_id(expected_review_id)
        if minimum_observation_timestamp_s is not None:
            minimum_observation_timestamp_s = _validate_timestamp(
                minimum_observation_timestamp_s
            )

        with self._condition:
            while self._completed:
                result = self._completed.popleft()
                stale = (
                    result.stale
                    or (
                        expected_request_id is not None
                        and result.request_id != expected_request_id
                    )
                    or (
                        expected_review_id is not None
                        and result.review_id != expected_review_id
                    )
                    or (
                        minimum_observation_timestamp_s is not None
                        and result.observation_timestamp_s
                        < minimum_observation_timestamp_s
                    )
                )
                if stale and not include_stale:
                    self._discarded_result_count += 1
                    continue
                return replace(result, stale=stale)
            return None

    def discard_stale_results(
        self,
        *,
        expected_request_id: str | None = None,
        expected_review_id: str | None = None,
        minimum_observation_timestamp_s: float | None = None,
    ) -> int:
        """Discard currently queued stale results and return their count."""

        if expected_request_id is not None:
            expected_request_id = validate_request_id(expected_request_id)
        if expected_review_id is not None:
            expected_review_id = validate_review_id(expected_review_id)
        if minimum_observation_timestamp_s is not None:
            minimum_observation_timestamp_s = _validate_timestamp(
                minimum_observation_timestamp_s
            )
        discarded = 0
        with self._condition:
            retained: deque[AsyncModelResult] = deque()
            while self._completed:
                result = self._completed.popleft()
                stale = (
                    result.stale
                    or (
                        expected_request_id is not None
                        and result.request_id != expected_request_id
                    )
                    or (
                        expected_review_id is not None
                        and result.review_id != expected_review_id
                    )
                    or (
                        minimum_observation_timestamp_s is not None
                        and result.observation_timestamp_s
                        < minimum_observation_timestamp_s
                    )
                )
                if stale:
                    discarded += 1
                else:
                    retained.append(result)
            self._completed = retained
            self._discarded_result_count += discarded
        return discarded

    def close(self, timeout_s: float | None = None) -> None:
        """Stop accepting work and wait for any active HTTP call to return."""

        if timeout_s is not None:
            if isinstance(timeout_s, bool) or not isinstance(timeout_s, Real):
                raise TypeError("timeout_s must be a finite number or None")
            timeout_s = float(timeout_s)
            if not isfinite(timeout_s) or timeout_s < 0.0:
                raise ValueError("timeout_s must be finite and non-negative")
        if threading.current_thread() is self._thread:
            raise RuntimeError("model worker cannot close itself")
        with self._condition:
            if not self._closed:
                self._closed = True
                self._pending = None
                self._condition.notify_all()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("model worker did not close before the deadline")

    def __enter__(self) -> AsyncModelWorker:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._pending is None and self._closed:
                    return
                request = self._pending
                self._pending = None
                assert request is not None
                self._active_request_id = request.request_id

            response: ModelResponse | None = None
            error_code: str | None = None
            error_message: str | None = None
            try:
                candidate = self._client.chat(
                    request.messages,
                    options=request.options,
                )
                if not isinstance(candidate, ModelResponse):
                    raise ModelProtocolError(
                        "model client returned an invalid response object"
                    )
                response = candidate
            except ModelClientError as exc:
                # Keep the worker/model boundary stable and non-secret.  The
                # concrete exception text may contain response bodies, URLs,
                # or transport headers and therefore must not enter runtime
                # review logs.
                error_code = "MODEL_REQUEST_FAILED"
                error_message = type(exc).__name__
            except Exception:
                # Never retain an arbitrary exception, whose text or traceback
                # may include a request payload, base64 image, or API key.
                error_code = "MODEL_REQUEST_FAILED"
                error_message = "unexpected model worker failure"

            with self._condition:
                stale = request.request_id in self._stale_request_ids
                self._stale_request_ids.discard(request.request_id)
                self._active_request_id = None
                result = AsyncModelResult(
                    request_id=request.request_id,
                    review_id=request.review_id,
                    mission_id=request.mission_id,
                    uav_id=request.uav_id,
                    plan_version=request.plan_version,
                    observation_timestamp_s=request.observation_timestamp_s,
                    frame_id=request.frame_id,
                    response=response,
                    error_code=error_code,
                    error_message=error_message,
                    stale=stale,
                )
                if len(self._completed) >= self._max_completed_results:
                    self._completed.popleft()
                    self._discarded_result_count += 1
                self._completed.append(result)
                self._condition.notify_all()


# A descriptive alias for callers that reserve this worker for visual review.
AsyncVisionWorker = AsyncModelWorker


__all__ = [
    "AsyncModelRequest",
    "AsyncModelResult",
    "AsyncModelWorker",
    "AsyncVisionWorker",
]
