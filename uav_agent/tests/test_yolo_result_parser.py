"""Model-free tests for Ultralytics result parsing and stream ownership."""

from __future__ import annotations

from threading import Event, Thread
from types import SimpleNamespace
import unittest

import numpy as np

from yolo_service.config import ModelFamily, YoloServiceSettings
from yolo_service.engine import (
    ResultParsingError,
    StreamBusyError,
    StreamConflictError,
    StreamSequenceError,
    UltralyticsEngine,
    UnsupportedTargetQuery,
    parse_ultralytics_results,
    rgb_to_bgr_once,
)
from yolo_service.protocol import TargetQuery, TrackRequest


def _results(*, track_id: int = 7, class_id: int = 0):
    boxes = SimpleNamespace(
        xyxy=np.asarray([[30.0, 25.0, 42.0, 71.0]], dtype=np.float32),
        conf=np.asarray([0.86], dtype=np.float32),
        cls=np.asarray([class_id], dtype=np.float32),
        id=np.asarray([track_id], dtype=np.float32),
    )
    return [SimpleNamespace(boxes=boxes, names={0: "person", 2: "car"})]


class FakeTracker:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class FakeModel:
    def __init__(self) -> None:
        self.names = {0: "person", 2: "car"}
        self.calls: list[tuple[np.ndarray, dict[str, object]]] = []
        self.set_classes_calls: list[tuple[str, ...]] = []
        self.tracker = FakeTracker()
        self.predictor = SimpleNamespace(trackers=[self.tracker])

    def track(self, image: np.ndarray, **kwargs: object):
        self.calls.append((image.copy(), kwargs))
        return _results()

    def set_classes(self, prompts: list[str]) -> None:
        self.set_classes_calls.append(tuple(prompts))
        self.names = {index: prompt for index, prompt in enumerate(prompts)}


class ReservationBarrierEngine(UltralyticsEngine):
    """Expose the reservation/inference boundary without timing sleeps."""

    def __init__(self, *, model: FakeModel, blocked_stream_id: str) -> None:
        super().__init__(model=model)
        self.blocked_stream_id = blocked_stream_id
        self.reserved = Event()
        self.release_reservation = Event()

    def _reserve_sequence(self, request: TrackRequest) -> None:
        super()._reserve_sequence(request)
        if request.stream_id == self.blocked_stream_id:
            self.reserved.set()
            if not self.release_reservation.wait(timeout=2.0):
                raise RuntimeError("test reservation barrier timed out")


def _request(
    *,
    index: int = 1,
    mission_id: str = "mission_1",
    query: TargetQuery | None = None,
) -> TrackRequest:
    return TrackRequest(
        1,
        f"request_{index}",
        mission_id,
        "uav_1",
        f"{mission_id}:uav_1",
        f"frame_{index}",
        float(index),
        query or TargetQuery(class_ids=(0,)),
    )


class YoloResultParserTest(unittest.TestCase):
    def test_pixel_boxes_are_normalized_without_model_objects(self) -> None:
        parsed = parse_ultralytics_results(
            _results(),
            image_shape_hw=(100, 100),
        )
        self.assertEqual(len(parsed), 1)
        detection = parsed[0]
        self.assertEqual(detection.track_id, 7)
        self.assertEqual(detection.class_name, "person")
        self.assertAlmostEqual(detection.bbox_xyxy_normalized[0], 0.3)
        self.assertAlmostEqual(detection.bbox_xyxy_normalized[3], 0.71)
        self.assertFalse(hasattr(detection, "tensor"))

    def test_malformed_and_untracked_model_results_fail_closed(self) -> None:
        untracked = _results()
        untracked[0].boxes.id = None
        self.assertEqual(
            parse_ultralytics_results(untracked, image_shape_hw=(100, 100)),
            (),
        )
        malformed = _results()
        malformed[0].boxes.conf[0] = np.nan
        with self.assertRaisesRegex(ResultParsingError, "NaN"):
            parse_ultralytics_results(malformed, image_shape_hw=(100, 100))

    def test_rgb_to_bgr_happens_once_before_engine(self) -> None:
        fake = FakeModel()
        engine = UltralyticsEngine(model=fake)
        rgb = np.asarray([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        bgr = rgb_to_bgr_once(rgb)
        engine.track(_request(), bgr)
        np.testing.assert_array_equal(fake.calls[0][0], [[[3, 2, 1], [6, 5, 4]]])
        self.assertEqual(fake.calls[0][1]["persist"], True)
        self.assertEqual(fake.calls[0][1]["classes"], [0])

    def test_yolo_rejects_text_and_never_calls_set_classes(self) -> None:
        fake = FakeModel()
        engine = UltralyticsEngine(model=fake)
        with self.assertRaisesRegex(UnsupportedTargetQuery, "YOLOE"):
            engine.track(
                _request(query=TargetQuery(text_prompts=("person",))),
                np.zeros((100, 100, 3), dtype=np.uint8),
            )
        self.assertEqual(fake.set_classes_calls, [])

    def test_yoloe_only_reencodes_changed_prompts(self) -> None:
        fake = FakeModel()
        settings = YoloServiceSettings(model_family=ModelFamily.YOLOE)
        engine = UltralyticsEngine(model=fake, settings=settings)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        query = TargetQuery(text_prompts=("person",))
        engine.track(_request(index=1, query=query), image)
        engine.track(_request(index=2, query=query), image)
        engine.track(
            _request(index=3, query=TargetQuery(text_prompts=("car",))),
            image,
        )
        self.assertEqual(fake.set_classes_calls, [("person",), ("car",)])
        self.assertNotIn("classes", fake.calls[0][1])

    def test_stream_time_duplicate_and_mission_isolation(self) -> None:
        fake = FakeModel()
        engine = UltralyticsEngine(model=fake)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        engine.track(_request(index=2), image)
        with self.assertRaises(StreamSequenceError):
            engine.track(_request(index=2), image)
        with self.assertRaises(StreamSequenceError):
            engine.track(_request(index=1), image)
        with self.assertRaises(StreamConflictError):
            engine.track(_request(index=3, mission_id="mission_2"), image)
        tracker = fake.tracker
        engine.reset_stream("mission_1:uav_1")
        self.assertEqual(tracker.reset_calls, 1)
        engine.track(_request(index=3, mission_id="mission_2"), image)

    def test_reservation_is_atomic_with_inference_and_reset(self) -> None:
        fake = FakeModel()
        stream_a = "mission_1:uav_1"
        engine = ReservationBarrierEngine(
            model=fake,
            blocked_stream_id=stream_a,
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        thread_errors: list[BaseException] = []

        def track_a() -> None:
            try:
                engine.track(_request(index=1), image)
            except BaseException as exc:  # pragma: no cover - asserted below
                thread_errors.append(exc)

        worker = Thread(target=track_a, name="track-mission-1")
        worker.start()
        try:
            self.assertTrue(engine.reserved.wait(timeout=1.0))
            self.assertEqual(engine.active_stream_id, stream_a)

            # Both operations arrive after reservation but before model.track.
            # Neither may clear or replace ownership while the tracker is live.
            with self.assertRaises(StreamBusyError):
                engine.reset_stream(stream_a)
            with self.assertRaises(StreamBusyError):
                engine.track(_request(index=1, mission_id="mission_2"), image)
            self.assertEqual(engine.active_stream_id, stream_a)
        finally:
            engine.release_reservation.set()
            worker.join(timeout=2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(thread_errors, [])
        self.assertEqual(len(fake.calls), 1)

        # Once inference releases the lock, reset is the explicit tracker
        # separation boundary and the next mission may reserve its own stream.
        tracker = fake.tracker
        engine.reset_stream(stream_a)
        self.assertEqual(tracker.reset_calls, 1)
        engine.track(_request(index=2, mission_id="mission_2"), image)
        self.assertEqual(engine.active_stream_id, "mission_2:uav_1")


if __name__ == "__main__":
    unittest.main()
