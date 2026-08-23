"""Asynchronous Qwen provider for SEARCH macro observation points.

The provider has one narrow authority: after a completed SEARCH viewpoint it
may propose one additional WORLD_ENU point inside the already trusted region,
or declare coverage exhausted.  It cannot emit velocity, yaw-rate, controller
gain, or a replacement mission plan.  HTTP execution remains in the shared
``AsyncModelWorker`` and every poll from SearchSkill is non-blocking.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite
from numbers import Real
from typing import Protocol

from common.ids import (
    generate_routing_id,
    validate_mission_id,
    validate_request_id,
    validate_review_id,
    validate_routing_id,
    validate_uav_id,
)
from models import (
    AsyncModelRequest,
    AsyncModelResult,
    ChatMessage,
    GenerationOptions,
    ImageURLContentPart,
    JsonSchemaResponseFormat,
    ModelProtocolError,
    TextContentPart,
    encode_rgb_to_data_url,
)
from planner.region_compiler import RegionCompiler
from planner.spatial import RegionSpec
from skills.search_strategy import NextBestViewPollResult, NextBestViewRequest


class NextBestViewWorker(Protocol):
    """Structural subset of AsyncModelWorker used by the provider."""

    def submit(self, request: AsyncModelRequest) -> None: ...

    def poll(
        self,
        *,
        expected_request_id: str | None = None,
        expected_review_id: str | None = None,
        minimum_observation_timestamp_s: float | None = None,
        include_stale: bool = False,
    ) -> AsyncModelResult | None: ...


@dataclass(frozen=True, slots=True)
class NextBestViewRouting:
    """Current trusted route projected into one model request."""

    mission_id: str
    plan_version: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        if isinstance(self.plan_version, bool) or not isinstance(
            self.plan_version,
            int,
        ):
            raise TypeError("plan_version must be an integer")
        if self.plan_version <= 0:
            raise ValueError("plan_version must be greater than zero")


@dataclass(frozen=True, slots=True)
class NextBestViewProposalRecord:
    """Image-free audit record for one completed Qwen proposal."""

    request_id: str
    review_id: str
    mission_id: str
    uav_id: str
    plan_version: int
    frame_id: str
    observation_timestamp_s: float
    proposal_index: int
    proposal: Mapping[str, object] | None
    decision: str | None
    viewpoint_xyz_m: tuple[float, float, float] | None
    error_code: str | None
    token_usage: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", validate_request_id(self.request_id))
        object.__setattr__(self, "review_id", validate_review_id(self.review_id))
        object.__setattr__(self, "mission_id", validate_mission_id(self.mission_id))
        object.__setattr__(self, "uav_id", validate_uav_id(self.uav_id))
        object.__setattr__(self, "frame_id", validate_routing_id(self.frame_id, "frame_id"))
        if isinstance(self.plan_version, bool) or not isinstance(self.plan_version, int):
            raise TypeError("plan_version must be an integer")
        if self.plan_version <= 0:
            raise ValueError("plan_version must be greater than zero")
        if isinstance(self.proposal_index, bool) or not isinstance(
            self.proposal_index,
            int,
        ):
            raise TypeError("proposal_index must be an integer")
        if self.proposal_index < 0:
            raise ValueError("proposal_index must be non-negative")
        _finite_nonnegative(
            self.observation_timestamp_s,
            "observation_timestamp_s",
        )
        if self.proposal is not None:
            # Round-trip validation produces an owned, image-free JSON object.
            try:
                owned = json.loads(
                    json.dumps(self.proposal, allow_nan=False, sort_keys=True)
                )
            except (TypeError, ValueError, OverflowError):
                raise TypeError("proposal must be a finite JSON object") from None
            if not isinstance(owned, dict):
                raise TypeError("proposal must be a JSON object or None")
            if _contains_image_payload(owned):
                raise ValueError("proposal must not contain image data")
            object.__setattr__(self, "proposal", owned)
        if self.decision not in {None, "NEXT_VIEW", "EXHAUSTED"}:
            raise ValueError("decision is unsupported")
        if self.viewpoint_xyz_m is not None:
            object.__setattr__(
                self,
                "viewpoint_xyz_m",
                _point(self.viewpoint_xyz_m, "viewpoint_xyz_m"),
            )
        if (self.error_code is None) == (self.decision is None):
            raise ValueError(
                "record must contain exactly one of a decision or error_code"
            )
        if self.error_code is not None and (
            not isinstance(self.error_code, str) or not self.error_code
        ):
            raise TypeError("error_code must be a non-empty string or None")
        usage: dict[str, int] = {}
        for key, value in self.token_usage.items():
            if not isinstance(key, str):
                raise TypeError("token_usage keys must be strings")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError("token_usage values must be non-negative integers")
            usage[key] = value
        object.__setattr__(self, "token_usage", usage)

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "review_id": self.review_id,
            "mission_id": self.mission_id,
            "uav_id": self.uav_id,
            "plan_version": self.plan_version,
            "frame_id": self.frame_id,
            "observation_timestamp_s": self.observation_timestamp_s,
            "proposal_index": self.proposal_index,
            "proposal": None if self.proposal is None else dict(self.proposal),
            "decision": self.decision,
            "viewpoint_xyz_m": self.viewpoint_xyz_m,
            "error_code": self.error_code,
            "token_usage": dict(self.token_usage),
        }


@dataclass(frozen=True, slots=True)
class _PendingRequest:
    request_id: str
    review_id: str
    routing: NextBestViewRouting
    frame_id: str
    observation_timestamp_s: float
    region: RegionSpec
    search_altitude_m: float
    visited_viewpoints_xyz_m: tuple[tuple[float, float, float], ...]


class QwenNextBestViewProvider:
    """Build, submit and validate one strict visual macro-view request."""

    SYSTEM_PROMPT = (
        "Select at most one next macro observation point for SEARCH using only "
        "the supplied current RGB frame and bounded search context. The point "
        "must be in the provided resolved WORLD_ENU region, must use exactly "
        "search_altitude_m, and must differ from every visited point. If no "
        "useful new view remains, choose EXHAUSTED. Emit exactly one JSON "
        "object matching the schema. Never emit velocities, yaw rates, control "
        "gains, low-level commands, target ground-truth coordinates, a route, "
        "or a replacement mission plan."
    )

    def __init__(
        self,
        *,
        worker: NextBestViewWorker,
        uav_id: str,
        routing_context: Callable[[], NextBestViewRouting],
        max_image_side_px: int = 1024,
        jpeg_quality: int = 80,
    ) -> None:
        if not callable(getattr(worker, "submit", None)) or not callable(
            getattr(worker, "poll", None)
        ):
            raise TypeError("worker must provide submit() and poll()")
        if not callable(routing_context):
            raise TypeError("routing_context must be callable")
        if isinstance(max_image_side_px, bool) or not isinstance(
            max_image_side_px,
            int,
        ):
            raise TypeError("max_image_side_px must be an integer")
        if max_image_side_px <= 0:
            raise ValueError("max_image_side_px must be greater than zero")
        if (
            isinstance(jpeg_quality, bool)
            or not isinstance(jpeg_quality, int)
            or not 1 <= jpeg_quality <= 95
        ):
            raise ValueError("jpeg_quality must be an integer within [1, 95]")
        self._worker = worker
        self._uav_id = validate_uav_id(uav_id)
        self._routing_context = routing_context
        self._max_image_side_px = max_image_side_px
        self._jpeg_quality = jpeg_quality
        self._pending: _PendingRequest | None = None
        self._records: list[NextBestViewProposalRecord] = []

    @property
    def records(self) -> tuple[NextBestViewProposalRecord, ...]:
        return tuple(self._records)

    @property
    def request_in_flight(self) -> bool:
        return self._pending is not None

    def submit_next_best_view(self, request: NextBestViewRequest) -> None:
        """Encode and enqueue one request; never wait for network/model I/O."""

        if not isinstance(request, NextBestViewRequest):
            raise TypeError("request must be a NextBestViewRequest")
        if self._pending is not None:
            raise RuntimeError("a next-best-view request is already in flight")
        routing = self._routing_context()
        if not isinstance(routing, NextBestViewRouting):
            raise TypeError("routing_context must return NextBestViewRouting")

        request_id = generate_routing_id("request_nbv")
        review_id = generate_routing_id("review_nbv")
        frame_id = generate_routing_id("frame_nbv")
        schema = build_next_best_view_json_schema(
            request_id=request_id,
            mission_id=routing.mission_id,
            uav_id=self._uav_id,
            plan_version=routing.plan_version,
            frame_id=frame_id,
            observation_timestamp_s=request.observation_timestamp_s,
        )
        payload: dict[str, object] = {
            "routing": {
                "request_id": request_id,
                "mission_id": routing.mission_id,
                "uav_id": self._uav_id,
                "plan_version": routing.plan_version,
                "frame_id": frame_id,
                "observation_timestamp_s": request.observation_timestamp_s,
            },
            "coordinate_contract": {
                "output_frame": "WORLD_ENU",
                "search_altitude_m": request.search_altitude_m,
                "output_is_macro_observation_point": True,
                "controller_rate_output_forbidden": True,
            },
            "target_description": request.target_description,
            "resolved_search_region": request.region.to_dict(),
            "current_observation": {
                "uav_position_xyz_m": list(request.uav_position_xyz_m),
                "uav_yaw_rad": request.uav_yaw_rad,
                "camera_position_m": (
                    None
                    if request.camera_position_m is None
                    else list(request.camera_position_m)
                ),
                "camera_orientation_wxyz": (
                    None
                    if request.camera_orientation_wxyz is None
                    else list(request.camera_orientation_wxyz)
                ),
            },
            "coverage": {
                "coverage_ratio": request.coverage_ratio,
                "visited_viewpoints_xyz_m": [
                    list(point) for point in request.visited_viewpoints_xyz_m
                ],
                "max_viewpoints": request.max_viewpoints,
                "remaining_viewpoint_budget": max(
                    0,
                    request.max_viewpoints
                    - len(request.visited_viewpoints_xyz_m),
                ),
            },
        }
        image = ImageURLContentPart(
            encode_rgb_to_data_url(
                request.camera_rgb,
                max_side_px=self._max_image_side_px,
                jpeg_quality=self._jpeg_quality,
            )
        )
        async_request = AsyncModelRequest(
            request_id=request_id,
            review_id=review_id,
            mission_id=routing.mission_id,
            uav_id=self._uav_id,
            plan_version=routing.plan_version,
            observation_timestamp_s=request.observation_timestamp_s,
            frame_id=frame_id,
            messages=(
                ChatMessage("system", self.SYSTEM_PROMPT),
                ChatMessage(
                    "user",
                    (
                        TextContentPart(
                            json.dumps(
                                payload,
                                ensure_ascii=False,
                                allow_nan=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        ),
                        image,
                    ),
                ),
            ),
            options=GenerationOptions(
                temperature=0.0,
                max_tokens=512,
                response_format=JsonSchemaResponseFormat(
                    "next_best_view_v1",
                    schema,
                ),
            ),
        )
        pending = _PendingRequest(
            request_id=request_id,
            review_id=review_id,
            routing=routing,
            frame_id=frame_id,
            observation_timestamp_s=request.observation_timestamp_s,
            region=request.region,
            search_altitude_m=request.search_altitude_m,
            visited_viewpoints_xyz_m=request.visited_viewpoints_xyz_m,
        )
        # Publish expectation only after request construction succeeds, but
        # before submit so a worker rejection can cleanly clear the state.
        self._pending = pending
        try:
            self._worker.submit(async_request)
        except Exception:
            self._pending = None
            raise

    def poll_next_best_view(self) -> NextBestViewPollResult:
        """Poll once without waiting and validate any completed proposal."""

        pending = self._pending
        if pending is None:
            raise RuntimeError("no next-best-view request is in flight")
        result = self._worker.poll(
            expected_request_id=pending.request_id,
            expected_review_id=pending.review_id,
            minimum_observation_timestamp_s=pending.observation_timestamp_s,
        )
        if result is None:
            return NextBestViewPollResult(completed=False)
        self._pending = None
        if (
            result.stale
            or result.request_id != pending.request_id
            or result.review_id != pending.review_id
            or result.mission_id != pending.routing.mission_id
            or result.uav_id != self._uav_id
            or result.plan_version != pending.routing.plan_version
            or result.frame_id != pending.frame_id
            or abs(
                result.observation_timestamp_s
                - pending.observation_timestamp_s
            )
            > 1e-9
        ):
            self._append_error_record(pending, "STALE_OR_MISROUTED_RESULT")
            raise ModelProtocolError(
                "next-best-view async result is stale or misrouted"
            )
        if result.response is None:
            self._append_error_record(
                pending,
                result.error_code or "MODEL_ERROR",
            )
            raise ModelProtocolError(
                "next-best-view request failed: "
                + (result.error_code or "MODEL_ERROR")
            )
        try:
            proposal = _strict_json_object(result.response.content)
            decision, waypoint = self._validate_proposal(pending, proposal)
        except (TypeError, ValueError, ModelProtocolError):
            self._append_error_record(
                pending,
                "INVALID_MODEL_PROPOSAL",
                token_usage=result.response.usage,
            )
            raise ModelProtocolError(
                "next-best-view response violates its strict spatial contract"
            ) from None
        self._records.append(
            NextBestViewProposalRecord(
                request_id=pending.request_id,
                review_id=pending.review_id,
                mission_id=pending.routing.mission_id,
                uav_id=self._uav_id,
                plan_version=pending.routing.plan_version,
                frame_id=pending.frame_id,
                observation_timestamp_s=pending.observation_timestamp_s,
                proposal_index=len(self._records),
                proposal=proposal,
                decision=decision,
                viewpoint_xyz_m=waypoint,
                error_code=None,
                token_usage=result.response.usage,
            )
        )
        return NextBestViewPollResult(
            completed=True,
            viewpoint_xyz_m=waypoint,
        )

    def cancel_pending_next_best_view(self) -> None:
        """Forget the expectation; AsyncModelWorker will discard it as stale."""

        self._pending = None

    def _validate_proposal(
        self,
        pending: _PendingRequest,
        proposal: Mapping[str, object],
    ) -> tuple[str, tuple[float, float, float] | None]:
        _exact_keys(
            proposal,
            {
                "schema_version",
                "routing",
                "decision",
                "coordinate_frame",
                "viewpoint_xyz_m",
                "rationale",
            },
            "proposal",
        )
        if proposal["schema_version"] != 1:
            raise ModelProtocolError("schema_version mismatch")
        routing = proposal["routing"]
        if not isinstance(routing, Mapping):
            raise ModelProtocolError("routing must be an object")
        _exact_keys(
            routing,
            {
                "request_id",
                "mission_id",
                "uav_id",
                "plan_version",
                "frame_id",
                "observation_timestamp_s",
            },
            "routing",
        )
        expected = {
            "request_id": pending.request_id,
            "mission_id": pending.routing.mission_id,
            "uav_id": self._uav_id,
            "plan_version": pending.routing.plan_version,
            "frame_id": pending.frame_id,
        }
        if any(routing.get(key) != value for key, value in expected.items()):
            raise ModelProtocolError("routing metadata mismatch")
        timestamp = _finite_nonnegative(
            routing.get("observation_timestamp_s"),
            "routing.observation_timestamp_s",
        )
        if abs(timestamp - pending.observation_timestamp_s) > 1e-9:
            raise ModelProtocolError("observation timestamp mismatch")
        if proposal["coordinate_frame"] != "WORLD_ENU":
            raise ModelProtocolError("coordinate_frame must be WORLD_ENU")
        rationale = proposal["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ModelProtocolError("rationale must be a non-empty string")
        if len(rationale) > 256:
            raise ModelProtocolError("rationale is too long")
        decision = proposal["decision"]
        if decision == "EXHAUSTED":
            if proposal["viewpoint_xyz_m"] is not None:
                raise ModelProtocolError(
                    "EXHAUSTED must set viewpoint_xyz_m to null"
                )
            return decision, None
        if decision != "NEXT_VIEW":
            raise ModelProtocolError("decision is unsupported")
        raw_point = proposal["viewpoint_xyz_m"]
        waypoint = RegionCompiler.validate_adaptive_waypoint(
            pending.region,
            raw_point,  # type: ignore[arg-type]
            search_altitude_m=pending.search_altitude_m,
        )
        if any(
            sum((a - b) ** 2 for a, b in zip(waypoint, prior)) ** 0.5 <= 1e-6
            for prior in pending.visited_viewpoints_xyz_m
        ):
            raise ModelProtocolError("viewpoint duplicates a visited point")
        return decision, waypoint

    def _append_error_record(
        self,
        pending: _PendingRequest,
        error_code: str,
        *,
        token_usage: Mapping[str, int] | None = None,
    ) -> None:
        self._records.append(
            NextBestViewProposalRecord(
                request_id=pending.request_id,
                review_id=pending.review_id,
                mission_id=pending.routing.mission_id,
                uav_id=self._uav_id,
                plan_version=pending.routing.plan_version,
                frame_id=pending.frame_id,
                observation_timestamp_s=pending.observation_timestamp_s,
                proposal_index=len(self._records),
                proposal=None,
                decision=None,
                viewpoint_xyz_m=None,
                error_code=error_code,
                token_usage={} if token_usage is None else token_usage,
            )
        )


def build_next_best_view_json_schema(
    *,
    request_id: str,
    mission_id: str,
    uav_id: str,
    plan_version: int,
    frame_id: str,
    observation_timestamp_s: float,
) -> dict[str, object]:
    """Build a routed strict schema for one macro-view decision."""

    request_id = validate_request_id(request_id)
    mission_id = validate_mission_id(mission_id)
    uav_id = validate_uav_id(uav_id)
    frame_id = validate_routing_id(frame_id, "frame_id")
    if isinstance(plan_version, bool) or not isinstance(plan_version, int):
        raise TypeError("plan_version must be an integer")
    if plan_version <= 0:
        raise ValueError("plan_version must be greater than zero")
    timestamp = _finite_nonnegative(
        observation_timestamp_s,
        "observation_timestamp_s",
    )
    routing = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "request_id": {"const": request_id},
            "mission_id": {"const": mission_id},
            "uav_id": {"const": uav_id},
            "plan_version": {"const": plan_version},
            "frame_id": {"const": frame_id},
            "observation_timestamp_s": {"const": timestamp},
        },
        "required": [
            "request_id",
            "mission_id",
            "uav_id",
            "plan_version",
            "frame_id",
            "observation_timestamp_s",
        ],
    }

    def variant(decision: str, viewpoint: dict[str, object]) -> dict[str, object]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "schema_version": {"const": 1},
                "routing": routing,
                "decision": {"const": decision},
                "coordinate_frame": {"const": "WORLD_ENU"},
                "viewpoint_xyz_m": viewpoint,
                "rationale": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 256,
                },
            },
            "required": [
                "schema_version",
                "routing",
                "decision",
                "coordinate_frame",
                "viewpoint_xyz_m",
                "rationale",
            ],
        }

    return {
        "oneOf": [
            variant(
                "NEXT_VIEW",
                {
                    "type": "array",
                    "prefixItems": [
                        {"type": "number"},
                        {"type": "number"},
                        {"type": "number"},
                    ],
                    "minItems": 3,
                    "maxItems": 3,
                },
            ),
            variant("EXHAUSTED", {"type": "null"}),
        ]
    }


def _strict_json_object(text: str) -> dict[str, object]:
    if not isinstance(text, str):
        raise TypeError("model response content must be a string")
    try:
        value = json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        raise ModelProtocolError("response is not strict JSON") from None
    if not isinstance(value, dict):
        raise ModelProtocolError("response must be a JSON object")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise ModelProtocolError(f"{name} fields do not match the schema")


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _point(value: object, name: str) -> tuple[float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise ValueError(f"{name} must contain exactly three finite numbers")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{name}[{index}] must be a finite number")
        number = float(item)
        if not isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(number)
    return result[0], result[1], result[2]


def _contains_image_payload(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in {
                "image",
                "images",
                "image_url",
                "rgb",
                "pixels",
            }:
                return True
            if _contains_image_payload(item):
                return True
    elif isinstance(value, list):
        return any(_contains_image_payload(item) for item in value)
    elif isinstance(value, str):
        lowered = value.casefold()
        return lowered.startswith("data:image/") or "base64," in lowered
    return False


__all__ = [
    "NextBestViewProposalRecord",
    "NextBestViewRouting",
    "QwenNextBestViewProvider",
    "build_next_best_view_json_schema",
]
