from __future__ import annotations

import json
import unittest

from datasets.target_state.schema import (
    CameraFrameInput,
    DetectorPrediction,
    SensorInput,
    TargetStateFrameRecord,
    TargetTrainingLabel,
    UavFrameInput,
)
from datasets.target_state.sequence import TargetStateSequence, build_sequences


def make_record(
    index: int,
    *,
    uav_id: str = "uav_1",
    assignment_id: str = "assignment_1",
    episode_id: str = "episode_1",
    candidate_id: str = "candidate_1",
    detected: bool = True,
    tracker_id: str | None = "tracker_1",
    instance_id: str | None = None,
) -> TargetStateFrameRecord:
    return TargetStateFrameRecord(
        frame_id=f"frame_{index}",
        episode_id=episode_id,
        assignment_id=assignment_id,
        uav_id=uav_id,
        timestamp_s=index * 0.2,
        sensor_input=SensorInput(
            camera=CameraFrameInput(
                fx=200.0,
                fy=201.0,
                cx=15.5,
                cy=11.5,
                position_world_m=(1.0, 2.0, 3.0),
                orientation_world_wxyz=(2.0, 0.0, 0.0, 0.0),
                resolution_wh_px=(32, 24),
            ),
            uav=UavFrameInput(
                position_world_m=(1.0, 2.0, 2.5),
                orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
                linear_velocity_world_mps=(0.1, 0.0, 0.0),
                angular_velocity_body_radps=(0.0, 0.0, 0.01),
            ),
            rgb_path=f"rgb/frame_{index}.jpg",
            depth_path=f"depth/frame_{index}.npy",
        ),
        detector_prediction=DetectorPrediction(
            detected=detected,
            bbox_xyxy_normalized=(0.25, 0.25, 0.75, 0.75) if detected else None,
            confidence=0.8 if detected else None,
            tracker_id=tracker_id if detected else None,
            candidate_id=candidate_id,
        ),
        training_label=TargetTrainingLabel(
            position_world_m=(5.0 + index * 0.1, 1.0, 0.5),
            velocity_world_mps=(0.5, 0.0, 0.0),
            center_pixel_uv=(16.0, 12.0),
            visible=True,
            occlusion_ratio=0.2,
            color_name="Red",
            instance_id=instance_id,
        ),
    )


class TargetStateSchemaTest(unittest.TestCase):
    def test_strict_json_round_trip_keeps_sections_separate(self) -> None:
        record = make_record(1)
        payload = json.loads(json.dumps(record.to_dict(), allow_nan=False))
        restored = TargetStateFrameRecord.from_dict(payload)

        self.assertEqual(restored, record)
        self.assertEqual(set(payload["sensor_input"]), {
            "camera", "uav", "rgb_path", "depth_path", "instance_mask_path"
        })
        self.assertNotIn("position_world_m", payload["detector_prediction"])
        self.assertNotIn("training_label", payload["sensor_input"])
        self.assertEqual(restored.sensor_input.camera.orientation_world_wxyz, (1.0, 0.0, 0.0, 0.0))

    def test_non_finite_and_unsafe_paths_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            TargetTrainingLabel(
                position_world_m=(float("nan"), 0.0, 0.0),
                velocity_world_mps=(0.0, 0.0, 0.0),
                center_pixel_uv=(1.0, 1.0),
                visible=True,
                occlusion_ratio=0.0,
            )
        with self.assertRaisesRegex(ValueError, "safe dataset-relative"):
            SensorInput(
                camera=make_record(1).sensor_input.camera,
                uav=make_record(1).sensor_input.uav,
                rgb_path="../secret.jpg",
                depth_path="depth/a.npy",
            )

    def test_missed_detection_may_retain_candidate_identity_only(self) -> None:
        prediction = DetectorPrediction(
            detected=False,
            bbox_xyxy_normalized=None,
            confidence=None,
            tracker_id=None,
            candidate_id="candidate_1",
        )
        self.assertEqual(prediction.candidate_id, "candidate_1")
        with self.assertRaisesRegex(ValueError, "cannot carry"):
            DetectorPrediction(
                detected=False,
                bbox_xyxy_normalized=None,
                confidence=0.2,
                tracker_id=None,
                candidate_id="candidate_1",
            )


class TargetStateSequenceTest(unittest.TestCase):
    def test_sequence_contains_exact_delta_missing_and_tracker_switch_masks(self) -> None:
        records = [make_record(index) for index in range(7)]
        records[2] = make_record(2, detected=False)
        records[4] = make_record(4, tracker_id="tracker_2")
        records[5] = make_record(5, tracker_id="tracker_2")
        records[6] = make_record(6, tracker_id="tracker_2")

        sequences = build_sequences(records, history_size=6, max_history_age_s=2.0)

        self.assertEqual(len(sequences), 1)
        sequence = sequences[0]
        for actual, expected in zip(sequence.delta_t_s, (1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(sequence.missing_mask, (False, False, True, False, False, False, False))
        self.assertEqual(sequence.tracker_id_changed, (False, False, True, True, True, False, False))

    def test_candidate_target_switch_never_mixes_instances_in_one_window(self) -> None:
        records = []
        for index in range(14):
            records.append(
                make_record(
                    index,
                    instance_id="cube_0" if index < 7 else "cube_1",
                    tracker_id="tracker_1" if index < 4 else "tracker_2",
                )
            )

        sequences = build_sequences(records, history_size=6, max_history_age_s=2.0)

        self.assertEqual(len(sequences), 2)
        instance_windows = []
        for sequence in sequences:
            frames = (*sequence.history, sequence.reference)
            self.assertEqual(
                {frame.detector_prediction.candidate_id for frame in frames},
                {"candidate_1"},
            )
            instance_ids = {
                frame.training_label.instance_id
                for frame in frames
                if frame.training_label is not None
            }
            self.assertEqual(len(instance_ids), 1)
            instance_windows.append(instance_ids.pop())
        self.assertEqual(instance_windows, ["cube_0", "cube_1"])
        self.assertIn(True, sequences[0].tracker_id_changed)

    def test_sequence_rejects_cross_uav_assignment_and_candidate(self) -> None:
        history = tuple(make_record(index) for index in range(4))
        reference = make_record(4, uav_id="uav_2")
        with self.assertRaisesRegex(ValueError, "uav_id"):
            TargetStateSequence(
                sequence_id="sequence_1",
                history=history,
                reference=reference,
                delta_t_s=(0.8, 0.6, 0.4, 0.2, 0.0),
                missing_mask=(False,) * 5,
                tracker_id_changed=(False,) * 5,
            )

        mixed = (*history[:3], make_record(3, candidate_id="candidate_2"))
        with self.assertRaisesRegex(ValueError, "mix candidates"):
            TargetStateSequence(
                sequence_id="sequence_2",
                history=mixed,
                reference=make_record(4),
                delta_t_s=(0.8, 0.6, 0.4, 0.2, 0.0),
                missing_mask=(False,) * 5,
                tracker_id_changed=(False,) * 5,
            )


if __name__ == "__main__":
    unittest.main()
