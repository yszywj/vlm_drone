"""Ultralytics/BoT-SORT adapter with strict single-stream ownership."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import replace
from importlib import metadata
from math import isfinite
from pathlib import Path
import platform
from threading import Lock
from time import perf_counter
from typing import Any, Callable

import numpy as np

from common.ids import validate_mission_id, validate_uav_id
from yolo_service.config import (
    ModelFamily,
    YoloServiceConfig,
    YoloServiceConfigurationError,
    YoloServiceSettings,
    file_sha256,
)
from yolo_service.protocol import (
    SCHEMA_VERSION,
    TimingMs,
    TrackDetection,
    TrackRequest,
    TrackResponse,
)


class YoloEngineError(RuntimeError):
    """Base class for explicit service errors; never triggers Oracle fallback."""

    code = "YOLO_ENGINE_ERROR"


class YoloDependencyUnavailable(YoloEngineError):
    code = "YOLO_DEPENDENCY_UNAVAILABLE"


class UnsupportedTargetQuery(YoloEngineError):
    code = "UNSUPPORTED_TARGET_CATEGORY"


class StreamConflictError(YoloEngineError):
    code = "ACTIVE_STREAM_CONFLICT"


class StreamBusyError(YoloEngineError):
    code = "STREAM_BUSY"


class StreamSequenceError(YoloEngineError):
    code = "INVALID_STREAM_SEQUENCE"


class ImageValidationError(YoloEngineError):
    code = "INVALID_IMAGE"


class ResultParsingError(YoloEngineError):
    code = "INVALID_MODEL_RESULT"


def rgb_to_bgr_once(image_rgb: np.ndarray) -> np.ndarray:
    """Perform the sole explicit RGB-to-BGR channel swap in direct-array use.

    The HTTP endpoint decodes JPEG directly into BGR and therefore must not call
    this function.  Direct in-process camera integrations call it once before
    invoking :meth:`UltralyticsEngine.track`.
    """

    image = _validate_image_array(image_rgb, name="image_rgb")
    return np.ascontiguousarray(image[:, :, ::-1])


def _validate_image_array(image: object, *, name: str = "image_bgr") -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ImageValidationError(f"{name} must be a NumPy array")
    if image.dtype != np.uint8:
        raise ImageValidationError(f"{name} dtype must be uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageValidationError(f"{name} must have shape HxWx3")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ImageValidationError(f"{name} dimensions must be non-zero")
    return image


def _numpy(value: object, name: str) -> np.ndarray:
    current = value
    if hasattr(current, "detach"):
        current = current.detach()  # type: ignore[union-attr]
    if hasattr(current, "cpu"):
        current = current.cpu()  # type: ignore[union-attr]
    if hasattr(current, "numpy"):
        current = current.numpy()  # type: ignore[union-attr]
    try:
        array = np.asarray(current)
    except Exception as exc:
        raise ResultParsingError(f"could not convert {name} to an array") from exc
    return array


def _model_names(value: object) -> dict[int, str]:
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = enumerate(value)
    else:
        raise ResultParsingError("model names must be a mapping or sequence")
    names: dict[int, str] = {}
    for key, raw_name in items:
        try:
            class_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ResultParsingError("model names contains a non-integer key") from exc
        if class_id < 0 or not isinstance(raw_name, str) or not raw_name.strip():
            raise ResultParsingError("model names contains an invalid entry")
        names[class_id] = raw_name.strip()
    if not names:
        raise ResultParsingError("model names cannot be empty")
    return names


def parse_ultralytics_result(
    result: object,
    *,
    image_shape_hw: tuple[int, int],
    model_names: object | None = None,
    allowed_class_ids: frozenset[int] | None = None,
) -> tuple[TrackDetection, ...]:
    """Convert one duck-typed Ultralytics ``Results`` into wire values."""

    if (
        not isinstance(image_shape_hw, tuple)
        or len(image_shape_hw) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in image_shape_hw)
    ):
        raise ValueError("image_shape_hw must be (height, width) positive integers")
    height, width = image_shape_hw
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()
    xyxy_raw = getattr(boxes, "xyxy", None)
    confidence_raw = getattr(boxes, "conf", None)
    class_raw = getattr(boxes, "cls", None)
    track_raw = getattr(boxes, "id", None)
    if xyxy_raw is None or confidence_raw is None or class_raw is None:
        raise ResultParsingError("model result boxes is missing xyxy/conf/cls")
    # A detector box without a BoT-SORT ID is not exposed as a tracked target.
    if track_raw is None:
        return ()

    xyxy = _numpy(xyxy_raw, "boxes.xyxy")
    confidence = _numpy(confidence_raw, "boxes.conf").reshape(-1)
    classes = _numpy(class_raw, "boxes.cls").reshape(-1)
    tracks = _numpy(track_raw, "boxes.id").reshape(-1)
    if xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise ResultParsingError("boxes.xyxy must have shape Nx4")
    count = xyxy.shape[0]
    if any(len(values) != count for values in (confidence, classes, tracks)):
        raise ResultParsingError("model result arrays have inconsistent lengths")
    if count > 10_000:
        raise ResultParsingError("model returned more than 10000 boxes")

    names_source = model_names
    if names_source is None:
        names_source = getattr(result, "names", None)
    names = _model_names(names_source)
    detections: list[TrackDetection] = []
    seen_track_ids: set[int] = set()
    for index in range(count):
        row = xyxy[index].astype(np.float64, copy=False)
        conf = float(confidence[index])
        cls_value = float(classes[index])
        track_value = float(tracks[index])
        if not np.all(np.isfinite(row)) or not all(
            isfinite(value) for value in (conf, cls_value, track_value)
        ):
            raise ResultParsingError("model result contains NaN or Inf")
        class_id = int(cls_value)
        track_id = int(track_value)
        if cls_value != class_id or class_id < 0:
            raise ResultParsingError("model result contains an invalid class ID")
        if track_value != track_id or track_id < 0:
            raise ResultParsingError("model result contains an invalid track ID")
        if allowed_class_ids is not None and class_id not in allowed_class_ids:
            continue
        if class_id not in names:
            raise ResultParsingError(f"model names has no entry for class {class_id}")
        if track_id in seen_track_ids:
            raise ResultParsingError("model result contains duplicate track IDs")
        seen_track_ids.add(track_id)
        if not 0.0 <= conf <= 1.0:
            raise ResultParsingError("model confidence must be within [0, 1]")
        x1 = min(max(float(row[0]) / width, 0.0), 1.0)
        y1 = min(max(float(row[1]) / height, 0.0), 1.0)
        x2 = min(max(float(row[2]) / width, 0.0), 1.0)
        y2 = min(max(float(row[3]) / height, 0.0), 1.0)
        if x1 >= x2 or y1 >= y2:
            # Ultralytics may surface fully clipped boxes.  They carry no usable
            # image geometry and are discarded instead of fabricating a box.
            continue
        detections.append(
            TrackDetection(
                track_id=track_id,
                class_id=class_id,
                class_name=names[class_id],
                confidence=conf,
                bbox_xyxy_normalized=(x1, y1, x2, y2),
            )
        )
    return tuple(detections)


def parse_ultralytics_results(
    results: object,
    *,
    image_shape_hw: tuple[int, int],
    model_names: object | None = None,
    allowed_class_ids: frozenset[int] | None = None,
) -> tuple[TrackDetection, ...]:
    """Strictly accept the single-frame list returned by ``model.track``."""

    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise ResultParsingError("model.track must return a result sequence")
    if len(results) != 1:
        raise ResultParsingError("model.track must return exactly one frame result")
    return parse_ultralytics_result(
        results[0],
        image_shape_hw=image_shape_hw,
        model_names=model_names,
        allowed_class_ids=allowed_class_ids,
    )


class UltralyticsEngine:
    """Single-process, single-active-stream BoT-SORT engine.

    ``model`` is an explicit test seam.  Production startup always constructs
    :class:`YoloServiceConfig`, verifies local paths, and imports Ultralytics
    only inside this constructor.
    """

    def __init__(
        self,
        config: YoloServiceConfig | None = None,
        *,
        model: object | None = None,
        model_path: str | Path | None = None,
        settings: YoloServiceSettings | None = None,
        model_family: str | ModelFamily | None = None,
        model_factory: Callable[[ModelFamily, Path], object] | None = None,
        clock: Callable[[], float] = perf_counter,
        max_seen_frame_ids: int = 4096,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        if isinstance(max_seen_frame_ids, bool) or not isinstance(max_seen_frame_ids, int):
            raise TypeError("max_seen_frame_ids must be an integer")
        if max_seen_frame_ids <= 0:
            raise ValueError("max_seen_frame_ids must be greater than zero")

        if config is not None:
            if not isinstance(config, YoloServiceConfig):
                raise TypeError("config must be a YoloServiceConfig")
            if settings is not None or model_path is not None or model_family is not None:
                raise ValueError("config cannot be combined with settings/path overrides")
            resolved_settings = config.settings
            resolved_path: Path | None = config.model_path
        else:
            resolved_settings = settings or YoloServiceSettings()
            if model_family is not None:
                resolved_settings = replace(
                    resolved_settings,
                    model_family=ModelFamily.parse(model_family),
                )
            resolved_path = (
                Path(model_path).expanduser().resolve() if model_path is not None else None
            )

        if model is None:
            if resolved_path is None:
                raise YoloServiceConfigurationError(
                    "an explicit local model_path is required"
                )
            # This creates the validated resolved config and fails before the
            # Ultralytics constructor could interpret a missing path as a model
            # name and auto-download it.
            resolved_config = YoloServiceConfig(resolved_path, resolved_settings)
            resolved_path = resolved_config.model_path
            resolved_settings = resolved_config.settings
            factory = model_factory or self._default_model_factory
            model = factory(resolved_settings.model_family, resolved_path)
        elif model_factory is not None:
            raise ValueError("model_factory cannot be supplied with an injected model")

        self._settings = resolved_settings
        self._model_path = resolved_path
        self._model = model
        self._clock = clock
        self._max_seen = max_seen_frame_ids
        self._active_stream_id: str | None = None
        self._last_timestamp_s: float | None = None
        self._seen_frame_order: deque[str] = deque()
        self._seen_frame_ids: set[str] = set()
        self._last_yoloe_prompts: tuple[str, ...] | None = None
        self._state_lock = Lock()
        self._inference_lock = Lock()
        # Validate model names now, so /health never advertises a malformed
        # detector as ready.
        self._names = _model_names(getattr(self._model, "names", None))

    @staticmethod
    def _default_model_factory(model_family: ModelFamily, model_path: Path) -> object:
        if not model_path.is_file():
            raise YoloServiceConfigurationError(
                f"model path is not an existing file: {model_path}"
            )
        try:
            from ultralytics import YOLO, YOLOE
        except ImportError as exc:
            raise YoloDependencyUnavailable(
                "ultralytics is required in the isolated yolo_perception environment"
            ) from exc
        constructor = YOLO if model_family is ModelFamily.YOLO else YOLOE
        try:
            return constructor(str(model_path))
        except Exception as exc:
            raise YoloEngineError(f"failed to load local model {model_path}: {exc}") from exc

    @property
    def active_stream_id(self) -> str | None:
        with self._state_lock:
            return self._active_stream_id

    @property
    def settings(self) -> YoloServiceSettings:
        return self._settings

    def _validate_query(self, request: TrackRequest) -> None:
        query = request.target_query
        if self._settings.model_family is ModelFamily.YOLO:
            if query.text_prompts:
                raise UnsupportedTargetQuery(
                    "ordinary YOLO accepts audited class_ids only; text_prompts require YOLOE"
                )
            unknown = sorted(set(query.class_ids) - set(self._names))
            if unknown:
                raise UnsupportedTargetQuery(
                    "requested class_ids are absent from model.names: "
                    + ", ".join(str(value) for value in unknown)
                )
        else:
            if query.class_ids:
                raise UnsupportedTargetQuery(
                    "YOLOE text mode accepts text_prompts instead of closed-set class_ids"
                )
            if not query.text_prompts:
                raise UnsupportedTargetQuery("YOLOE requires at least one text prompt")

    def _set_yoloe_prompts_if_changed(self, prompts: tuple[str, ...]) -> None:
        if self._settings.model_family is not ModelFamily.YOLOE:
            return
        if prompts == self._last_yoloe_prompts:
            return
        setter = getattr(self._model, "set_classes", None)
        if not callable(setter):
            raise YoloEngineError("configured YOLOE model has no set_classes method")
        try:
            setter(list(prompts))
        except Exception as exc:
            raise YoloEngineError(f"YOLOE set_classes failed: {exc}") from exc
        self._last_yoloe_prompts = prompts
        self._names = _model_names(getattr(self._model, "names", None))

    def _reserve_sequence(self, request: TrackRequest) -> None:
        with self._state_lock:
            if (
                self._active_stream_id is not None
                and self._active_stream_id != request.stream_id
            ):
                raise StreamConflictError(
                    "this process already owns active stream "
                    f"{self._active_stream_id!r}; reset it before using "
                    f"{request.stream_id!r}"
                )
            if request.frame_id in self._seen_frame_ids:
                raise StreamSequenceError(
                    f"duplicate frame_id {request.frame_id!r} in active stream"
                )
            if (
                self._last_timestamp_s is not None
                and request.timestamp_s <= self._last_timestamp_s
            ):
                raise StreamSequenceError(
                    "stream timestamps must increase strictly"
                )
            self._active_stream_id = request.stream_id

    def _commit_sequence(self, request: TrackRequest) -> None:
        with self._state_lock:
            self._last_timestamp_s = request.timestamp_s
            self._seen_frame_ids.add(request.frame_id)
            self._seen_frame_order.append(request.frame_id)
            while len(self._seen_frame_order) > self._max_seen:
                expired = self._seen_frame_order.popleft()
                self._seen_frame_ids.discard(expired)

    def track(
        self,
        request: TrackRequest,
        image_bgr: np.ndarray,
        *,
        decode_ms: float = 0.0,
    ) -> TrackResponse:
        if not isinstance(request, TrackRequest):
            raise TypeError("request must be a TrackRequest")
        image = _validate_image_array(image_bgr)
        if image.shape[1] > self._settings.max_image_width_px:
            raise ImageValidationError("image width exceeds configured maximum")
        if image.shape[0] > self._settings.max_image_height_px:
            raise ImageValidationError("image height exceeds configured maximum")
        if not isinstance(decode_ms, (int, float)) or isinstance(decode_ms, bool):
            raise TypeError("decode_ms must be a finite non-negative number")
        decode_ms = float(decode_ms)
        if not isfinite(decode_ms) or decode_ms < 0.0:
            raise ValueError("decode_ms must be finite and non-negative")
        self._validate_query(request)
        if not self._inference_lock.acquire(blocking=False):
            raise StreamBusyError("only one inference may be active in this process")
        try:
            # Stream ownership and the persistent Ultralytics tracker are one
            # lifecycle boundary.  Reserve only after taking the inference
            # lock so reset_stream() cannot clear ownership between the
            # reservation and model.track(), and a rejected concurrent call
            # cannot mutate the active stream.
            self._reserve_sequence(request)
            self._set_yoloe_prompts_if_changed(request.target_query.text_prompts)
            kwargs: dict[str, object] = {
                "persist": True,
                "tracker": self._settings.tracker_path,
                "conf": self._settings.confidence_threshold,
                "iou": self._settings.iou_threshold,
                "imgsz": self._settings.image_size_px,
                "device": self._settings.device,
                "verbose": False,
            }
            if self._settings.model_family is ModelFamily.YOLO:
                kwargs["classes"] = list(request.target_query.class_ids)
            started = self._clock()
            try:
                results = self._model.track(image, **kwargs)  # type: ignore[attr-defined]
            except Exception as exc:
                raise YoloEngineError(f"Ultralytics track failed: {exc}") from exc
            finished = self._clock()
            elapsed_ms = max(0.0, (finished - started) * 1000.0)
            detections = parse_ultralytics_results(
                results,
                image_shape_hw=(image.shape[0], image.shape[1]),
                model_names=getattr(self._model, "names", self._names),
                allowed_class_ids=(
                    frozenset(request.target_query.class_ids)
                    if self._settings.model_family is ModelFamily.YOLO
                    else None
                ),
            )
            self._commit_sequence(request)
            return TrackResponse(
                schema_version=SCHEMA_VERSION,
                request_id=request.request_id,
                mission_id=request.mission_id,
                uav_id=request.uav_id,
                stream_id=request.stream_id,
                frame_id=request.frame_id,
                timestamp_s=request.timestamp_s,
                detections=detections,
                timing_ms=TimingMs(
                    decode=decode_ms,
                    inference=elapsed_ms,
                    tracking=0.0,
                    total=decode_ms + elapsed_ms,
                ),
            )
        finally:
            self._inference_lock.release()

    def _reset_underlying_tracker(self) -> None:
        predictor = getattr(self._model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if trackers is not None:
            for tracker in tuple(trackers):
                reset = getattr(tracker, "reset", None)
                if callable(reset):
                    reset()
        # Ultralytics creates a fresh predictor/tracker on the next call.  This
        # is the important separation boundary for persist=True state.
        if predictor is not None:
            try:
                setattr(self._model, "predictor", None)
            except Exception as exc:  # pragma: no cover - defensive vendor API
                raise YoloEngineError(f"could not release tracker state: {exc}") from exc

    def reset_stream(self, stream_id: str) -> None:
        parts = stream_id.split(":") if isinstance(stream_id, str) else []
        if len(parts) != 2:
            raise ValueError("stream_id must equal '<mission_id>:<uav_id>'")
        try:
            validate_mission_id(parts[0])
            validate_uav_id(parts[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid stream_id: {exc}") from exc
        if not self._inference_lock.acquire(blocking=False):
            raise StreamBusyError("cannot reset while inference is active")
        try:
            with self._state_lock:
                if self._active_stream_id not in (None, stream_id):
                    raise StreamConflictError(
                        f"cannot reset {stream_id!r}; active stream is "
                        f"{self._active_stream_id!r}"
                    )
                self._reset_underlying_tracker()
                self._active_stream_id = None
                self._last_timestamp_s = None
                self._seen_frame_order.clear()
                self._seen_frame_ids.clear()
                self._last_yoloe_prompts = None
        finally:
            self._inference_lock.release()

    def model_info(self) -> dict[str, object]:
        torch_version: str | None = None
        cuda_available = False
        gpu_name: str | None = None
        try:
            import torch

            torch_version = str(torch.__version__)
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                gpu_name = str(torch.cuda.get_device_name(0))
        except ImportError:
            pass
        try:
            ultralytics_version = metadata.version("ultralytics")
        except metadata.PackageNotFoundError:
            ultralytics_version = None
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "ready",
            "python_version": platform.python_version(),
            "ultralytics_version": ultralytics_version,
            "torch_version": torch_version,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "model_family": self._settings.model_family.value,
            "model_path": str(self._model_path) if self._model_path is not None else "<injected>",
            "model_sha256": (
                file_sha256(self._model_path) if self._model_path is not None else None
            ),
            "model_names": dict(self._names),
            "device": self._settings.device,
            "tracker_path": self._settings.tracker_path,
            "active_stream_id": self.active_stream_id,
        }


__all__ = [
    "ImageValidationError",
    "ResultParsingError",
    "StreamBusyError",
    "StreamConflictError",
    "StreamSequenceError",
    "UltralyticsEngine",
    "UnsupportedTargetQuery",
    "YoloDependencyUnavailable",
    "YoloEngineError",
    "parse_ultralytics_result",
    "parse_ultralytics_results",
    "rgb_to_bgr_once",
]
