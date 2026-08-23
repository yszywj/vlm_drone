from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.sparse_mission_logger import (
    ModelCallKind,
    ModelProposalLogRecord,
    QwenReviewLogRecord,
    RunManifestMetadata,
    SparseMissionLogger,
)


def _review(review_id: str, **overrides: object) -> QwenReviewLogRecord:
    values: dict[str, object] = {
        "review_id": review_id,
        "mission_id": "mission_1",
        "uav_id": "uav_1",
        "plan_version": 1,
        "frame_id": f"frame_{review_id[-1]}",
        "observation_timestamp_s": 1.0,
        "decision": "NO_RELEVANT_CHANGE",
        "bbox_xyxy_normalized": None,
        "accepted": True,
    }
    values.update(overrides)
    return QwenReviewLogRecord(**values)  # type: ignore[arg-type]


def _proposal(
    proposal_id: str,
    kind: ModelCallKind,
    index: int,
) -> ModelProposalLogRecord:
    return ModelProposalLogRecord(
        proposal_id=proposal_id,
        mission_id="mission_1",
        uav_id="uav_1",
        plan_version=2,
        timestamp_s=2.0 + index,
        call_kind=kind,
        proposal_index=index,
        route_id="route_1",
        proposal={
            "route_id": "route_1",
            "frame": "UAV_HOLD_FLU",
            "waypoints": [
                {"waypoint_id": "wp_1", "xyz_m": [2, 3, 0]},
                {"waypoint_id": "wp_2", "xyz_m": [8, 3, 0]},
            ],
        },
        critique={
            "status": "ACCEPT" if index else "REVISE",
            "route_id": "route_1",
            "violations": [] if index else [{"type": "INSUFFICIENT_CLEARANCE"}],
            "route_length_m": 20.0 - index,
            "minimum_clearance_m": 0.5 + index,
        },
        shadow_strict_critique={
            "status": "ACCEPT" if index else "REVISE",
            "route_id": "route_1",
            "violations": [] if index else [{"type": "PATH_INTERSECTS_OBSTACLE"}],
            "route_length_m": 20.0 - index,
            "minimum_clearance_m": 0.5 + index,
        },
        final_proposal=bool(index),
        latency_s=0.8 + index,
    )


class SparseMissionMetricsV3Test(unittest.TestCase):
    def test_split_model_counts_route_hold_search_and_manifest_fields(self) -> None:
        metadata = RunManifestMetadata(
            experiment_mode="qwen_critic_sim",
            route_planner_backend="qwen",
            planning_contract="spatial_v3",
            runtime_program="linear",
            route_validation_mode="critic_sim",
            obstacle_perception_mode="ideal_camera",
            prompt_schema_versions={"initial_planner": 3, "visual_review": 2},
            git_commit="a" * 40,
        )
        with tempfile.TemporaryDirectory() as temporary:
            with SparseMissionLogger(
                temporary,
                manifest_metadata=metadata,
            ) as logger:
                logger.record_initial_planner_model_call()
                logger.log_qwen_review(_review("review_1"))
                logger.log_qwen_review(
                    _review(
                        "review_2",
                        accepted=False,
                        stale=True,
                        error_code="STALE",
                        stale_reasons=("plan_version_changed",),
                    )
                )
                logger.log_qwen_review(
                    _review(
                        "review_3",
                        accepted=False,
                        error_code="INVALID_JSON",
                    )
                )
                # A valid shadow-mode result is parsed successfully even
                # though it is intentionally not accepted for control.
                logger.log_qwen_review(_review("review_4", accepted=False))
                logger.log_model_proposal(
                    _proposal("proposal_0", ModelCallKind.ROUTE_PLANNER, 0)
                )
                logger.log_model_proposal(
                    _proposal("proposal_1", ModelCallKind.ROUTE_REPAIR, 1)
                )
                logger.record_plan_revision_model_call()
                logger.record_plan_revision()
                logger.set_final_plan_version(3)
                logger.record_hold_metrics(
                    trigger_source="ideal_camera_obstacle_perception",
                    hazard_detection_latency_s=0.12,
                    hold_establishment_latency_s=0.35,
                )
                logger.record_route_metrics(
                    planning_latency_s=1.4,
                    route_length_m=18.5,
                    minimum_clearance_m=2.0,
                )
                logger.record_route_metrics(minimum_clearance_m=1.25)
                logger.record_path_position((0.0, 0.0, 0.0))
                logger.record_path_position((3.0, 4.0, 0.0))
                logger.record_collision()
                logger.record_invalid_waypoints(2)
                logger.record_shadow_strict_route_validity(False)
                logger.record_search_metrics(
                    region_shape="RECTANGLE",
                    search_strategy="LAWNMOWER",
                    coverage_ratio=0.75,
                    visited_viewpoint_count=6,
                    target_detection_time_s=9.2,
                )
                summary = logger.snapshot().to_manifest_dict()

            expected = {
                "initial_planner_model_calls": 1,
                "visual_review_model_calls": 4,
                "visual_review_valid_results": 2,
                "visual_review_stale_results": 1,
                "visual_review_parse_errors": 1,
                "route_planner_model_calls": 1,
                "classical_route_planner_calls": 0,
                "route_repair_model_calls": 1,
                "plan_revision_model_calls": 1,
                "hold_trigger_source": "ideal_camera_obstacle_perception",
                "hazard_detection_latency_s": 0.12,
                "hold_establishment_latency_s": 0.35,
                "route_planning_latency_s": 1.4,
                "route_repair_count": 1,
                "route_length_m": 18.5,
                "path_length_m": 5.0,
                "minimum_route_clearance_m": 1.25,
                "collision_count": 1,
                "invalid_waypoint_count": 2,
                "shadow_strict_route_valid": False,
                "route_validity_source": "shadow_strict_route_valid",
                "plan_revision_count": 1,
                "final_plan_version": 3,
                "region_shape": "RECTANGLE",
                "search_strategy": "LAWNMOWER",
                "coverage_ratio": 0.75,
                "visited_viewpoint_count": 6,
                "target_detection_time_s": 9.2,
                "planning_contract": "spatial_v3",
                "experiment_mode": "qwen_critic_sim",
                "route_planner_backend": "qwen",
                "runtime_program": "linear",
                "route_validation_mode": "critic_sim",
                "obstacle_perception_mode": "ideal_camera",
                "git_commit": "a" * 40,
            }
            for key, value in expected.items():
                self.assertEqual(summary[key], value, key)
            self.assertEqual(
                summary["prompt_schema_versions"],
                {"initial_planner": 3, "visual_review": 2},
            )

            proposals = [
                json.loads(line)
                for line in (Path(temporary) / "model_proposals.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([item["proposal_id"] for item in proposals], ["proposal_0", "proposal_1"])
            self.assertEqual(proposals[0]["critique"]["status"], "REVISE")
            self.assertEqual(
                proposals[0]["shadow_strict_critique"]["status"],
                "REVISE",
            )
            self.assertTrue(proposals[1]["final_proposal"])
            serialized = json.dumps(proposals).casefold()
            self.assertNotIn("base64,", serialized)
            self.assertNotIn("data:image/", serialized)

    def test_proposal_stream_rejects_images_and_manifest_values_are_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "image"):
            ModelProposalLogRecord(
                proposal_id="proposal_bad",
                mission_id="mission_1",
                uav_id="uav_1",
                plan_version=1,
                timestamp_s=1.0,
                call_kind=ModelCallKind.ROUTE_PLANNER,
                proposal_index=0,
                proposal={"image_url": "data:image/jpeg;base64,AAAA"},
            )
        with self.assertRaisesRegex(ValueError, "git_commit"):
            RunManifestMetadata(git_commit="dirty")
        with self.assertRaisesRegex(ValueError, "experiment_mode"):
            RunManifestMetadata(experiment_mode="qwen_maybe")
        with self.assertRaisesRegex(ValueError, "route_planner_backend"):
            RunManifestMetadata(route_planner_backend="silent_fallback")
        with tempfile.TemporaryDirectory() as temporary:
            with SparseMissionLogger(temporary) as logger:
                default_stats = logger.snapshot().to_manifest_dict()
            self.assertNotIn("git_commit", default_stats)
            merged = {"git_commit": "b" * 40, **default_stats}
            self.assertEqual(merged["git_commit"], "b" * 40)
        with tempfile.TemporaryDirectory() as temporary:
            with SparseMissionLogger(temporary) as logger:
                logger.record_shadow_strict_route_validity(True)
                logger.record_shadow_strict_route_validity(True)
                logger.record_shadow_strict_route_validity(False)
                self.assertFalse(
                    logger.snapshot().shadow_strict_route_valid
                )
        with self.assertRaisesRegex(ValueError, "coverage_ratio"):
            with tempfile.TemporaryDirectory() as temporary:
                with SparseMissionLogger(temporary) as logger:
                    logger.record_search_metrics(
                        region_shape="CIRCLE",
                        search_strategy="PERIMETER",
                        coverage_ratio=1.1,
                        visited_viewpoint_count=1,
                        target_detection_time_s=None,
                    )

    def test_route_proposal_derives_latency_length_clearance_and_repair_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with SparseMissionLogger(temporary) as logger:
                logger.log_model_proposal(
                    _proposal("proposal_0", ModelCallKind.ROUTE_PLANNER, 0)
                )
                logger.log_model_proposal(
                    _proposal("proposal_1", ModelCallKind.ROUTE_REPAIR, 1)
                )
                stats = logger.snapshot()
        self.assertEqual(stats.route_planner_model_calls, 1)
        self.assertEqual(stats.route_repair_model_calls, 1)
        self.assertEqual(stats.route_repair_count, 1)
        self.assertEqual(stats.route_planning_latency_s, 1.8)
        self.assertEqual(stats.route_length_m, 19.0)
        # Rejected counterexamples are logged but do not poison the executed
        # route clearance metric.
        self.assertEqual(stats.minimum_route_clearance_m, 1.5)

    def test_classical_proposal_is_not_counted_as_qwen_model_call(self) -> None:
        source = _proposal(
            "proposal_classical",
            ModelCallKind.CLASSICAL_ROUTE_PLANNER,
            1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            with SparseMissionLogger(temporary) as logger:
                logger.log_model_proposal(source)
                stats = logger.snapshot()
        self.assertEqual(stats.classical_route_planner_calls, 1)
        self.assertEqual(stats.route_planner_model_calls, 0)
        self.assertEqual(stats.route_repair_model_calls, 0)
        self.assertTrue(stats.shadow_strict_route_valid)

    def test_next_best_view_has_its_own_model_call_counter(self) -> None:
        record = ModelProposalLogRecord(
            proposal_id="proposal_nbv",
            mission_id="mission_1",
            uav_id="uav_1",
            plan_version=2,
            timestamp_s=4.0,
            call_kind=ModelCallKind.NEXT_BEST_VIEW,
            proposal_index=0,
            proposal={
                "decision": "NEXT_VIEW",
                "coordinate_frame": "WORLD_ENU",
                "viewpoint_xyz_m": [2.0, 3.0, 5.0],
            },
            final_proposal=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            with SparseMissionLogger(temporary) as logger:
                logger.log_model_proposal(record)
                stats = logger.snapshot()
        self.assertEqual(stats.next_best_view_model_calls, 1)
        self.assertEqual(stats.plan_revision_model_calls, 0)
        self.assertEqual(stats.route_planner_model_calls, 0)


if __name__ == "__main__":
    unittest.main()
