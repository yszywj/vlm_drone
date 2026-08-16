from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from experiments.plotter import ExperimentPlotter, _best_eval_index


def _write(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ExperimentPlotterTest(unittest.TestCase):
    def test_eval_best_uses_full_checkpoint_tie_break(self) -> None:
        rows = [
            {
                "mission_success_rate": "0.8",
                "false_lock_rate": "0.2",
                "collision_rate": "0.0",
                "safety_abort_rate": "0.0",
                "mean_mission_time_s": "40",
            },
            {
                "mission_success_rate": "0.8",
                "false_lock_rate": "0.1",
                "collision_rate": "0.5",
                "safety_abort_rate": "0.5",
                "mean_mission_time_s": "99",
            },
        ]
        self.assertEqual(_best_eval_index(rows), 1)

    def test_generates_six_png_files_from_csv_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            metrics = run / "metrics"
            _write(
                metrics / "train_metrics.csv",
                ["global_step", "mission_success_rate_100", "episode_return_mean", "policy_loss", "value_loss"],
                [
                    {"global_step": 10, "mission_success_rate_100": 0.2, "episode_return_mean": 1.0, "policy_loss": 0.5, "value_loss": 0.8},
                    {"global_step": 20, "mission_success_rate_100": 0.6, "episode_return_mean": 2.0, "policy_loss": 0.3, "value_loss": 0.4},
                ],
            )
            _write(
                metrics / "eval_metrics.csv",
                ["global_step", "mission_success_rate"],
                [{"global_step": 10, "mission_success_rate": 0.4}, {"global_step": 20, "mission_success_rate": 0.7}],
            )
            _write(
                metrics / "final_metrics.csv",
                ["num_test_episodes", "mission_success_rate", "mission_success_ci95_low", "mission_success_ci95_high"],
                [{"num_test_episodes": 20, "mission_success_rate": 0.7, "mission_success_ci95_low": 0.48, "mission_success_ci95_high": 0.85}],
            )
            episode_fields = [
                "phase", "takeoff_success", "goto_search_success", "search_success",
                "correct_target_locked", "track_success", "return_success",
                "landing_success", "mission_success_strict",
            ]
            _write(
                metrics / "episode_metrics.csv",
                episode_fields,
                [
                    {key: ("test" if key == "phase" else True) for key in episode_fields},
                    {key: ("test" if key == "phase" else key not in {"track_success", "mission_success_strict"}) for key in episode_fields},
                ],
            )
            _write(
                metrics / "failure_cases.csv",
                ["failure_reason"],
                [{"failure_reason": "TARGET_NOT_FOUND"}, {"failure_reason": "UNKNOWN_ERROR"}],
            )

            generated = ExperimentPlotter(run).generate_all()

            self.assertEqual(len(generated), 6)
            self.assertEqual(
                {path.name for path in generated},
                {
                    "train_success_rate.png", "eval_success_rate.png", "final_success_rate.png",
                    "stage_success_rate.png", "failure_breakdown.png", "training_curve.png",
                },
            )
            for path in generated:
                self.assertGreater(path.stat().st_size, 100)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_does_not_create_empty_plot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generated = ExperimentPlotter(temporary).generate_all()
            self.assertEqual(generated, ())


if __name__ == "__main__":
    unittest.main()
