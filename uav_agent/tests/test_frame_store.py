from __future__ import annotations

import unittest

import numpy as np

from runtime.frame_store import FrameRef, FrameStore


def _rgb(value: int, *, height: int = 2, width: int = 2) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


class FrameStoreTest(unittest.TestCase):
    def test_pinned_inflight_frame_survives_ring_pressure_until_release(self) -> None:
        store = FrameStore(max_frames=2, max_total_bytes=24, max_frame_age_s=5)
        pending = store.add_frame(
            uav_id="uav_1", frame_id="frame_pending", timestamp_s=1, rgb=_rgb(1)
        )
        store.pin(pending)
        store.add_frame(
            uav_id="uav_1", frame_id="frame_2", timestamp_s=7, rgb=_rgb(2)
        )
        latest = store.add_frame(
            uav_id="uav_1", frame_id="frame_3", timestamp_s=8, rgb=_rgb(3)
        )

        self.assertTrue(store.contains(pending))
        self.assertTrue(store.contains(latest))
        self.assertLessEqual(len(store), 2)
        self.assertLessEqual(store.total_bytes, 24)

        store.unpin(pending)
        self.assertFalse(store.contains(pending))
        with self.assertRaisesRegex(ValueError, "not pinned"):
            store.unpin(pending)

    def test_frame_count_limit_evicts_oldest(self) -> None:
        store = FrameStore(max_frames=2, max_total_bytes=100, max_frame_age_s=20)
        first = store.add_frame(
            uav_id="uav_1", frame_id="frame_1", timestamp_s=1, rgb=_rgb(1)
        )
        second = store.add_frame(
            uav_id="uav_1", frame_id="frame_2", timestamp_s=2, rgb=_rgb(2)
        )
        third = store.add_frame(
            uav_id="uav_1", frame_id="frame_3", timestamp_s=3, rgb=_rgb(3)
        )

        self.assertEqual(len(store), 2)
        self.assertIsNone(store.get_frame(first))
        self.assertIsNotNone(store.get_frame(second))
        self.assertIsNotNone(store.get_frame(third))
        self.assertEqual(store.refs(), (second, third))

    def test_total_byte_limit_evicts_oldest(self) -> None:
        # Each 2x2 RGB frame occupies 12 bytes.
        store = FrameStore(max_frames=10, max_total_bytes=20, max_frame_age_s=20)
        first = store.add_frame(
            uav_id="uav_1", frame_id="frame_1", timestamp_s=1, rgb=_rgb(1)
        )
        second = store.add_frame(
            uav_id="uav_1", frame_id="frame_2", timestamp_s=2, rgb=_rgb(2)
        )

        self.assertEqual(len(store), 1)
        self.assertEqual(store.total_bytes, 12)
        self.assertIsNone(store.get_frame(first))
        self.assertEqual(store.latest_ref(uav_id="uav_1"), second)

    def test_age_limit_and_explicit_clock_advance(self) -> None:
        store = FrameStore(max_frames=10, max_total_bytes=100, max_frame_age_s=5)
        first = store.add_frame(
            uav_id="uav_1", frame_id="frame_1", timestamp_s=1, rgb=_rgb(1)
        )
        boundary = store.add_frame(
            uav_id="uav_1", frame_id="frame_2", timestamp_s=6, rgb=_rgb(2)
        )
        self.assertTrue(store.contains(first))  # exactly max age remains valid

        self.assertEqual(store.evict_expired(timestamp_s=6.01), 1)
        self.assertFalse(store.contains(first))
        self.assertTrue(store.contains(boundary))
        with self.assertRaisesRegex(ValueError, "older"):
            store.add_frame(
                uav_id="uav_1",
                frame_id="stale_frame",
                timestamp_s=0,
                rgb=_rgb(0),
            )

    def test_pixels_are_defensive_and_never_part_of_frame_ref(self) -> None:
        source = _rgb(7)
        store = FrameStore(max_frames=2, max_total_bytes=100, max_frame_age_s=20)
        ref = store.add_frame(
            uav_id="uav_1", frame_id="frame_1", timestamp_s=1, rgb=source
        )
        source[:] = 99

        retained = store.get_frame(ref)
        assert retained is not None
        self.assertTrue(np.all(retained == 7))
        retained[:] = 11
        self.assertTrue(np.all(store.get_frame(ref) == 7))

        read_only = store.get_frame(ref, copy=False)
        assert read_only is not None
        self.assertFalse(read_only.flags.writeable)
        self.assertEqual(
            ref.to_dict(),
            {
                "uav_id": "uav_1",
                "frame_id": "frame_1",
                "timestamp_s": 1.0,
                "width": 2,
                "height": 2,
            },
        )
        self.assertNotIn("rgb", ref.to_dict())

    def test_uav_namespaces_do_not_cross(self) -> None:
        store = FrameStore(max_frames=4, max_total_bytes=100, max_frame_age_s=20)
        first = store.add_frame(
            uav_id="uav_1", frame_id="frame_1", timestamp_s=1, rgb=_rgb(1)
        )
        second = store.add_frame(
            uav_id="uav_2", frame_id="frame_1", timestamp_s=1, rgb=_rgb(2)
        )
        self.assertEqual(store.refs(uav_id="uav_1"), (first,))
        self.assertEqual(store.refs(uav_id="uav_2"), (second,))
        self.assertTrue(np.all(store.get_frame(first) == 1))
        self.assertTrue(np.all(store.get_frame(second) == 2))

        forged = FrameRef("uav_2", "frame_1", 2, 2, 2)
        self.assertIsNone(store.get_frame(forged))

    def test_age_watermarks_are_isolated_per_uav(self) -> None:
        store = FrameStore(max_frames=4, max_bytes=100, max_age_s=5)
        first = store.add_frame(
            uav_id="uav_1", frame_id="frame_1", timestamp_s=1, rgb=_rgb(1)
        )
        second = store.add_frame(
            uav_id="uav_2", frame_id="frame_2", timestamp_s=100, rgb=_rgb(2)
        )
        self.assertTrue(store.contains(first))
        self.assertTrue(store.contains(second))
        self.assertEqual(store.max_bytes, 100)
        self.assertEqual(store.max_age_s, 5.0)

    def test_invalid_or_unbounded_inputs_are_rejected(self) -> None:
        for kwargs in (
            {"max_frames": 0},
            {"max_total_bytes": True},
            {"max_frame_age_s": float("inf")},
            {"max_frame_age_s": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(
                (TypeError, ValueError)
            ):
                FrameStore(**kwargs)

        store = FrameStore(max_frames=2, max_total_bytes=12, max_frame_age_s=20)
        invalid_frames = (
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2, 4), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.float32),
        )
        for index, frame in enumerate(invalid_frames):
            with self.subTest(index=index), self.assertRaises(
                (TypeError, ValueError)
            ):
                store.add_frame(
                    uav_id="uav_1",
                    frame_id=f"frame_{index}",
                    timestamp_s=index,
                    rgb=frame,
                )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            store.add_frame(
                uav_id="uav_1",
                frame_id="large_frame",
                timestamp_s=1,
                rgb=_rgb(0, height=3, width=2),
            )
        with self.assertRaises(ValueError):
            store.add_frame(
                uav_id="bad id",
                frame_id="frame_1",
                timestamp_s=1,
                rgb=_rgb(0),
            )


if __name__ == "__main__":
    unittest.main()
