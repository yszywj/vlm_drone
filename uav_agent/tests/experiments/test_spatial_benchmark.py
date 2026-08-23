from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.spatial_benchmark import (
    ExperimentMode,
    SpatialBenchmarkAggregator,
    SpatialBenchmarkError,
    SpatialEpisodeResult,
    aggregate_spatial_manifests,
    experiment_mode_profile,
    infer_experiment_mode,
    resolve_experiment_profile,
)


def _episode(
    run_id: str,
    mode: ExperimentMode,
    **overrides: object,
) -> SpatialEpisodeResult:
    values: dict[str, object] = {
        "run_id": run_id,
        "experiment_mode": mode,
        "mission_success": True,
        "collision_count": 0,
        "shadow_strict_route_valid": True,
        "route_repair_count": 1,
        "search_coverage_ratio": 0.8,
        "path_length_m": 20.0,
        "planning_latency_s": 2.0,
        "unseen_spatial_instruction": False,
    }
    values.update(overrides)
    return SpatialEpisodeResult(**values)  # type: ignore[arg-type]


class ExperimentModeTest(unittest.TestCase):
    def test_five_profiles_have_one_explicit_mapping_each(self) -> None:
        self.assertEqual(len(tuple(ExperimentMode)), 5)
        expected = {
            ExperimentMode.SCRIPTED_BASELINE: (
                "dynamic_scripted",
                "v2",
                "strict",
                "none",
            ),
            ExperimentMode.CLASSICAL_BASELINE: (
                "dynamic_scripted",
                "v2",
                "strict",
                "classical",
            ),
            ExperimentMode.QWEN_OPEN_SIM: (
                "dynamic_llm",
                "v3",
                "open_sim",
                "qwen",
            ),
            ExperimentMode.QWEN_CRITIC_SIM: (
                "dynamic_llm",
                "v3",
                "critic_sim",
                "qwen",
            ),
            ExperimentMode.QWEN_STRICT: (
                "dynamic_llm",
                "v3",
                "strict",
                "qwen",
            ),
        }
        for mode, values in expected.items():
            with self.subTest(mode=mode):
                profile = experiment_mode_profile(mode)
                self.assertEqual(
                    (
                        profile.planner,
                        profile.planning_contract,
                        profile.route_validation_mode,
                        profile.route_planner_backend,
                    ),
                    values,
                )

    def test_explicit_conflicts_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            SpatialBenchmarkError,
            "conflicts with --experiment-mode qwen_open_sim",
        ):
            resolve_experiment_profile(
                ExperimentMode.QWEN_OPEN_SIM,
                route_validation_mode="strict",
            )
        with self.assertRaisesRegex(SpatialBenchmarkError, "planning-contract"):
            resolve_experiment_profile(
                ExperimentMode.CLASSICAL_BASELINE,
                planning_contract="v3",
            )

    def test_legacy_mapping_is_deterministic(self) -> None:
        self.assertIs(
            infer_experiment_mode(
                planner="dynamic_scripted",
                route_validation_mode="strict",
            ),
            ExperimentMode.SCRIPTED_BASELINE,
        )
        self.assertIs(
            infer_experiment_mode(
                planner="dynamic_llm",
                route_validation_mode="critic_sim",
            ),
            ExperimentMode.QWEN_CRITIC_SIM,
        )


class SpatialBenchmarkAggregatorTest(unittest.TestCase):
    def test_open_sim_acceptance_is_not_route_validity(self) -> None:
        manifest = {
            "mission_id": "mission_open",
            "experiment_mode": "qwen_open_sim",
            "mission_success": True,
            "collision_count": 0,
            "route_repair_count": 0,
            # These are deliberately tempting but must be ignored.
            "route_validation_mode": "open_sim",
            "route_state": "ACCEPTED",
            "accepted_route_count": 1,
            "critique": {"status": "ACCEPT"},
        }
        result = SpatialEpisodeResult.from_manifest(manifest)
        self.assertIsNone(result.shadow_strict_route_valid)
        aggregator = SpatialBenchmarkAggregator()
        aggregator.add(result)
        metrics = aggregator.summary()["modes"]["qwen_open_sim"]
        self.assertIsNone(metrics["route_validity_rate"])
        self.assertEqual(metrics["route_validity_evaluated_count"], 0)

    def test_aggregates_required_metrics_per_mode_and_overall(self) -> None:
        aggregator = SpatialBenchmarkAggregator()
        aggregator.add(
            _episode(
                "run_1",
                ExperimentMode.QWEN_CRITIC_SIM,
                unseen_spatial_instruction=True,
            )
        )
        aggregator.add(
            _episode(
                "run_2",
                ExperimentMode.QWEN_CRITIC_SIM,
                mission_success=False,
                collision_count=2,
                shadow_strict_route_valid=False,
                route_repair_count=3,
                search_coverage_ratio=0.4,
                path_length_m=30.0,
                planning_latency_s=4.0,
                unseen_spatial_instruction=True,
            )
        )
        # Make every condition visible in the output even though empty modes
        # would also be emitted with explicit zero denominators.
        for index, mode in enumerate(
            (
                ExperimentMode.SCRIPTED_BASELINE,
                ExperimentMode.CLASSICAL_BASELINE,
                ExperimentMode.QWEN_OPEN_SIM,
                ExperimentMode.QWEN_STRICT,
            ),
            start=3,
        ):
            aggregator.add(_episode(f"run_{index}", mode))

        summary = aggregator.summary()
        self.assertEqual(set(summary["modes"]), {mode.value for mode in ExperimentMode})
        critic = summary["modes"]["qwen_critic_sim"]
        self.assertEqual(critic["mission_success_rate"], 0.5)
        self.assertEqual(critic["collision_rate"], 0.5)
        self.assertEqual(critic["collision_count"], 2)
        self.assertEqual(critic["route_validity_rate"], 0.5)
        self.assertEqual(critic["average_repair_count"], 2.0)
        self.assertAlmostEqual(critic["average_search_coverage"], 0.6)
        self.assertEqual(critic["average_path_length_m"], 25.0)
        self.assertEqual(critic["average_planning_latency_s"], 3.0)
        self.assertEqual(critic["unseen_spatial_success_rate"], 0.5)
        self.assertEqual(summary["route_validity_source"], "shadow_strict_route_valid")

    def test_manifest_files_and_atomic_summary_are_json_safe(self) -> None:
        manifest = {
            "mission_id": "mission_1",
            "experiment_mode": "classical_baseline",
            "agent_status": "SUCCEEDED",
            "task_status": "SUCCEEDED",
            "guard_error": None,
            "collision_count": 0,
            "shadow_strict_route_valid": True,
            "route_repair_count": 0,
            "search": {"coverage_ratio": 0.75},
            "route_length_m": 12.0,
            "route_planning_latency_s": 0.1,
            "unseen_spatial_instruction": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "run_manifest.json"
            source.write_text(json.dumps(manifest), encoding="utf-8")
            summary = aggregate_spatial_manifests((source,))
            self.assertEqual(summary["episode_count"], 1)

            aggregator = SpatialBenchmarkAggregator()
            aggregator.add_manifest(manifest)
            destination = aggregator.write_summary(root / "summary.json")
            decoded = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(decoded["episode_count"], 1)
            self.assertFalse((root / ".summary.json.tmp").exists())

    def test_duplicate_and_invalid_values_fail_closed(self) -> None:
        aggregator = SpatialBenchmarkAggregator()
        row = _episode("same", ExperimentMode.QWEN_STRICT)
        aggregator.add(row)
        with self.assertRaisesRegex(SpatialBenchmarkError, "duplicate"):
            aggregator.add(row)
        with self.assertRaisesRegex(SpatialBenchmarkError, "within"):
            _episode(
                "bad",
                ExperimentMode.QWEN_STRICT,
                search_coverage_ratio=1.1,
            )


if __name__ == "__main__":
    unittest.main()
