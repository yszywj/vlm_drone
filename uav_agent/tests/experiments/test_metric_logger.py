from __future__ import annotations

from contextlib import contextmanager
import csv
import json
from pathlib import Path
import struct
import tempfile
import unittest

from experiments.evaluator import (
    BestCheckpointRequiredError,
    EvaluationError,
    Evaluator,
    aggregate_episode_metrics,
    compute_mission_success_strict,
    wilson_score_interval_95,
)
from experiments.metric_logger import (
    DuplicateEpisodeError,
    FinalMetricsAlreadyLoggedError,
    MetricLogger,
    MetricSchemaError,
    ScalarEventWriter,
)
from experiments.schemas import (
    EPISODE_METRIC_FIELDS,
    EVAL_METRIC_FIELDS,
    FAILURE_CASE_FIELDS,
    FINAL_METRIC_FIELDS,
    TRAIN_METRIC_FIELDS,
)


def _successful_episode(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "run_id": "run_1",
        "phase": "train",
        "seed": 42,
        "global_step": 100,
        "episode_id": 1,
        "scenario_id": "scenario_1",
        "takeoff_success": True,
        "goto_search_success": True,
        "search_success": True,
        "correct_target_locked": True,
        "false_target_lock": False,
        "track_success": True,
        "reacquire_triggered": False,
        "reacquire_success": None,
        "return_success": True,
        "landing_success": True,
        "collision": False,
        "out_of_bounds": False,
        "safety_abort": False,
        "timeout": False,
        "mission_sim_time_s": 12.5,
        "episode_return": 3.5,
    }
    values.update(overrides)
    return values


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _final_metrics(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "run_id": "run_1",
        "best_checkpoint_step": 500,
        "num_test_episodes": 20,
        "mission_success_rate": 0.5,
        "mission_success_ci95_low": 0.3,
        "mission_success_ci95_high": 0.7,
    }
    values.update(overrides)
    return values


class MetricLoggerTest(unittest.TestCase):
    def test_creates_five_fixed_csv_files_with_one_header(self) -> None:
        expected = {
            "train_metrics.csv": TRAIN_METRIC_FIELDS,
            "eval_metrics.csv": EVAL_METRIC_FIELDS,
            "episode_metrics.csv": EPISODE_METRIC_FIELDS,
            "failure_cases.csv": FAILURE_CASE_FIELDS,
            "final_metrics.csv": FINAL_METRIC_FIELDS,
        }
        with tempfile.TemporaryDirectory() as temporary:
            logger = MetricLogger(temporary, tensorboard_enabled=False)
            logger.close()
            for name, fields in expected.items():
                path = Path(temporary) / "metrics" / name
                with path.open("r", encoding="utf-8", newline="") as stream:
                    self.assertEqual(tuple(next(csv.reader(stream))), fields)
                    self.assertEqual(stream.read(), "")

    def test_append_resume_does_not_repeat_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with MetricLogger(temporary, tensorboard_enabled=False) as logger:
                logger.log_train(10, 1, {"episode_return_mean": 2.0})
            with MetricLogger(temporary, tensorboard_enabled=False) as logger:
                logger.log_train(20, 2, {"episode_return_mean": 3.0})

            path = Path(temporary) / "metrics" / "train_metrics.csv"
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("timestamp,global_step,update,"), 1)
            rows = _read_csv(path)
            self.assertEqual([row["global_step"] for row in rows], ["10", "20"])

    def test_resume_requires_global_step_to_continue_increasing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with MetricLogger(temporary, tensorboard_enabled=False) as logger:
                logger.log_train(10, 1, {})
                logger.log_eval(10, {})
            with MetricLogger(temporary, tensorboard_enabled=False) as logger:
                with self.assertRaises(MetricSchemaError):
                    logger.log_train(10, 2, {})
                with self.assertRaises(MetricSchemaError):
                    logger.log_eval(9, {})
                logger.log_train(11, 2, {})
                logger.log_eval(11, {})

    def test_missing_measurements_are_empty_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with MetricLogger(temporary, tensorboard_enabled=False) as logger:
                logger.log_train(10, 1, {"learning_rate": None})
            row = _read_csv(Path(temporary) / "metrics" / "train_metrics.csv")[0]
            self.assertEqual(row["learning_rate"], "")
            self.assertEqual(row["policy_loss"], "")
            self.assertEqual(row["episode_return_mean"], "")

    def test_episode_key_is_unique_in_one_process_and_after_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with MetricLogger(temporary, tensorboard_enabled=False) as logger:
                logger.log_episode(_successful_episode())
                with self.assertRaises(DuplicateEpisodeError):
                    logger.log_episode(_successful_episode())
            with MetricLogger(temporary, tensorboard_enabled=False) as logger:
                with self.assertRaises(DuplicateEpisodeError):
                    logger.log_episode(_successful_episode())
                different_phase = _successful_episode(phase="validation")
                self.assertTrue(logger.log_episode(different_phase))

    def test_strict_success_is_recomputed_and_false_lock_fails(self) -> None:
        claimed = _successful_episode(
            mission_success_strict=True,
            false_target_lock=True,
            failure_reason="FALSE_TARGET_LOCK",
        )
        self.assertFalse(compute_mission_success_strict(claimed))
        with tempfile.TemporaryDirectory() as temporary:
            with MetricLogger(temporary, tensorboard_enabled=False) as logger:
                self.assertFalse(logger.log_episode(claimed))
            episode = _read_csv(Path(temporary) / "metrics" / "episode_metrics.csv")[0]
            failure = _read_csv(Path(temporary) / "metrics" / "failure_cases.csv")[0]
            self.assertEqual(episode["mission_success_strict"], "False")
            self.assertEqual(failure["failure_reason"], "FALSE_TARGET_LOCK")

    def test_strict_success_requires_measured_safety_exclusions(self) -> None:
        incomplete = _successful_episode()
        incomplete.pop("collision")
        self.assertFalse(compute_mission_success_strict(incomplete))

    def test_failed_episode_without_reason_is_honestly_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with MetricLogger(temporary, tensorboard_enabled=False) as logger:
                logger.log_episode(_successful_episode(landing_success=False))
            episode = _read_csv(Path(temporary) / "metrics" / "episode_metrics.csv")[0]
            failure = _read_csv(Path(temporary) / "metrics" / "failure_cases.csv")[0]
            self.assertEqual(episode["failure_reason"], "UNKNOWN_ERROR")
            self.assertEqual(failure["failure_reason"], "UNKNOWN_ERROR")

    def test_invalid_failure_record_does_not_partially_write_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with MetricLogger(temporary, tensorboard_enabled=False) as logger:
                with self.assertRaises(MetricSchemaError):
                    logger.log_episode(
                        _successful_episode(
                            landing_success=False,
                            failure_reason="LAND_FAILED",
                            message={"not": "a scalar"},
                        )
                    )
            self.assertEqual(
                _read_csv(Path(temporary) / "metrics" / "episode_metrics.csv"),
                [],
            )
            self.assertEqual(
                _read_csv(Path(temporary) / "metrics" / "failure_cases.csv"),
                [],
            )

    def test_failure_csv_rejects_success_and_unknown_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with MetricLogger(temporary, tensorboard_enabled=False) as logger:
                with self.assertRaises(MetricSchemaError):
                    logger.log_failure(
                        {
                            "run_id": "r",
                            "phase": "test",
                            "episode_id": 1,
                            "failure_reason": "TARGET_NOT_FOUND",
                            "mission_success_strict": True,
                        }
                    )
                with self.assertRaises(MetricSchemaError):
                    logger.log_failure(_successful_episode(failure_reason="UNKNOWN_ERROR"))
                with self.assertRaises(MetricSchemaError):
                    logger.log_failure(
                        {
                            "run_id": "r",
                            "phase": "test",
                            "episode_id": 3,
                            "failure_reason": "UNKNOWN_ERROR",
                            "mission_success_strict": " YES ",
                        }
                    )
                with self.assertRaises(MetricSchemaError):
                    logger.log_failure(
                        {
                            "run_id": "r",
                            "phase": "test",
                            "episode_id": 2,
                            "failure_reason": "MADE_UP_REASON",
                        }
                    )
            self.assertEqual(
                _read_csv(Path(temporary) / "metrics" / "failure_cases.csv"), []
            )

    def test_final_csv_allows_exactly_one_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with MetricLogger(temporary, run_id="run_1", tensorboard_enabled=False) as logger:
                logger.log_final(_final_metrics(run_id=""))
                with self.assertRaises(FinalMetricsAlreadyLoggedError):
                    logger.log_final(_final_metrics())
            with MetricLogger(temporary, tensorboard_enabled=False) as logger:
                with self.assertRaises(FinalMetricsAlreadyLoggedError):
                    logger.log_final(_final_metrics())
            rows = _read_csv(Path(temporary) / "metrics" / "final_metrics.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "run_1")

    def test_final_csv_requires_best_test_contract(self) -> None:
        invalid = (
            _final_metrics(run_id=""),
            _final_metrics(best_checkpoint_step=-1),
            _final_metrics(num_test_episodes=0),
            _final_metrics(mission_success_rate=None),
            _final_metrics(mission_success_ci95_low=0.6),
        )
        for index, row in enumerate(invalid):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                with MetricLogger(temporary, tensorboard_enabled=False) as logger:
                    with self.assertRaises(MetricSchemaError):
                        logger.log_final(row)

    def test_flush_makes_rows_visible_before_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = MetricLogger(temporary, tensorboard_enabled=False)
            logger.log_train(1, 1, {"fps": 100.0})
            logger.flush()
            rows = _read_csv(Path(temporary) / "metrics" / "train_metrics.csv")
            self.assertEqual(rows[0]["fps"], "100.0")
            logger.close()

    def test_tensorboard_file_is_real_tfrecord_and_scalar_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with MetricLogger(temporary) as logger:
                logger.log_train(10, 1, {"episode_return_mean": 2.5})
                logger.log_eval(10, {"mission_success_rate": 0.75})
                logger.flush()
                writer = logger.tensorboard
                self.assertIsInstance(writer, ScalarEventWriter)
                for forbidden in (
                    "add_image",
                    "add_images",
                    "add_video",
                    "add_histogram",
                    "add_graph",
                    "add_embedding",
                    "add_text",
                ):
                    self.assertFalse(hasattr(writer, forbidden))

            paths = list((Path(temporary) / "tensorboard").glob("events.out.tfevents.*"))
            self.assertEqual(len(paths), 1)
            payloads = self._read_tfrecord_payloads(paths[0])
            self.assertGreaterEqual(len(payloads), 3)
            encoded = b"".join(payloads)
            self.assertIn(b"brain.Event:2", encoded)
            self.assertIn(b"train/episode_return", encoded)
            self.assertIn(b"eval/mission_success_rate", encoded)
            self.assertNotIn(b"image", encoded.lower())
            self.assertNotIn(b"histogram", encoded.lower())

    @staticmethod
    def _read_tfrecord_payloads(path: Path) -> list[bytes]:
        stream = path.open("rb")
        payloads: list[bytes] = []
        with stream:
            while length_bytes := stream.read(8):
                if len(length_bytes) != 8:
                    raise AssertionError("truncated TFRecord length")
                length_crc = stream.read(4)
                payload = stream.read(struct.unpack("<Q", length_bytes)[0])
                payload_crc = stream.read(4)
                if len(length_crc) != 4 or len(payload_crc) != 4:
                    raise AssertionError("truncated TFRecord checksum")
                payloads.append(payload)
        return payloads


class _FakeRunner:
    def __init__(self, active: list[bool]) -> None:
        self.active = active
        self.calls: list[tuple[int, str, bool]] = []

    def run_episode(self, *, seed: int, phase: str, deterministic: bool) -> dict[str, object]:
        if not self.active[0]:
            raise AssertionError("episode ran outside the no-gradient context")
        self.calls.append((seed, phase, deterministic))
        return _successful_episode(
            run_id="eval_run",
            phase=phase,
            episode_id=seed,
            seed=seed,
            reacquire_triggered=seed % 2 == 0,
            reacquire_success=seed % 4 == 0,
            track_success=seed != 12,
            failure_reason="TRACK_FAILED" if seed == 12 else None,
        )


class EvaluatorTest(unittest.TestCase):
    def test_fixed_seeds_deterministic_no_grad_and_conditional_reacquire_rate(self) -> None:
        active = [False]

        @contextmanager
        def guard():
            active[0] = True
            try:
                yield
            finally:
                active[0] = False

        runner = _FakeRunner(active)
        evaluator = Evaluator(
            runner,
            validation_seeds=(10, 11, 12, 13),
            test_seeds=(20, 21),
            interval_steps=100,
            inference_context_factory=guard,
        )
        self.assertTrue(evaluator.should_evaluate(100))
        result = evaluator.evaluate(global_step=100, checkpoint_step=80)
        self.assertFalse(evaluator.should_evaluate(100))
        self.assertEqual(
            runner.calls,
            [(10, "validation", True), (11, "validation", True), (12, "validation", True), (13, "validation", True)],
        )
        self.assertEqual(result["num_episodes"], 4)
        self.assertEqual(result["mission_success_rate"], 0.75)
        # Only seeds 10 and 12 triggered reacquisition; only seed 12 is divisible by 4.
        self.assertEqual(result["reacquire_success_rate"], 0.5)

    def test_untriggered_reacquisition_is_empty_not_zero(self) -> None:
        def runner(*, seed: int, phase: str, deterministic: bool) -> dict[str, object]:
            return _successful_episode(
                run_id="eval_run", phase=phase, episode_id=seed, reacquire_triggered=False
            )

        evaluator = Evaluator(runner, validation_seeds=(1,), test_seeds=(2,))
        self.assertIsNone(evaluator.evaluate(global_step=1)["reacquire_success_rate"])

    def test_unmeasured_stage_rate_is_empty_not_zero(self) -> None:
        def runner(*, seed: int, phase: str, deterministic: bool) -> dict[str, object]:
            result = _successful_episode(run_id="eval_run", phase=phase, episode_id=seed)
            result.pop("search_success")
            return result

        evaluator = Evaluator(runner, validation_seeds=(1,), test_seeds=(2,))
        result = evaluator.evaluate(global_step=1)
        self.assertIsNone(result["search_success_rate"])
        self.assertEqual(result["mission_success_rate"], 0.0)

    def test_partially_missing_stage_or_risk_metric_is_rejected(self) -> None:
        for field in ("search_success", "landing_success", "false_target_lock", "collision"):
            with self.subTest(field=field):
                first = _successful_episode(episode_id=1)
                second = _successful_episode(episode_id=2)
                second.pop(field)
                with self.assertRaisesRegex(EvaluationError, field):
                    aggregate_episode_metrics((first, second))

    def test_partially_missing_mean_metric_is_rejected(self) -> None:
        for field in ("mission_sim_time_s", "episode_return"):
            with self.subTest(field=field):
                first = _successful_episode(episode_id=1)
                second = _successful_episode(episode_id=2)
                second.pop(field)
                with self.assertRaisesRegex(EvaluationError, field):
                    aggregate_episode_metrics((first, second))

        all_unmeasured = []
        for episode_id in (1, 2):
            row = _successful_episode(episode_id=episode_id)
            row.pop("mission_sim_time_s")
            row.pop("episode_return")
            all_unmeasured.append(row)
        summary = aggregate_episode_metrics(all_unmeasured)
        self.assertIsNone(summary["mean_mission_time_s"])
        self.assertIsNone(summary["mean_episode_return"])

    def test_partial_reacquire_metadata_is_rejected(self) -> None:
        first = _successful_episode(
            episode_id=1,
            reacquire_triggered=True,
            reacquire_success=True,
        )
        second = _successful_episode(episode_id=2)
        second.pop("reacquire_triggered")
        with self.assertRaisesRegex(EvaluationError, "reacquire_triggered"):
            aggregate_episode_metrics((first, second))

        first.pop("reacquire_success")
        second["reacquire_triggered"] = False
        with self.assertRaisesRegex(EvaluationError, "reacquire_success"):
            aggregate_episode_metrics((first, second))

    def test_failed_aggregate_does_not_commit_validation_state(self) -> None:
        def runner(*, seed: int, phase: str, deterministic: bool) -> dict[str, object]:
            row = _successful_episode(episode_id=seed)
            if seed == 2:
                row.pop("collision")
            return row

        evaluator = Evaluator(
            runner,
            validation_seeds=(1, 2),
            test_seeds=(3,),
            interval_steps=20,
        )
        with self.assertRaises(EvaluationError):
            evaluator.evaluate(global_step=20)
        self.assertEqual(evaluator.last_validation_episodes, ())
        self.assertTrue(evaluator.should_evaluate(20))

    def test_runner_cannot_override_evaluator_owned_episode_metadata(self) -> None:
        def runner(*, seed: int, phase: str, deterministic: bool) -> dict[str, object]:
            return _successful_episode(seed=999, phase="train", episode_id="spoofed")

        evaluator = Evaluator(runner, validation_seeds=(7,), test_seeds=(8,))
        evaluator.evaluate(global_step=1)
        episode = evaluator.last_validation_episodes[0]
        self.assertEqual(episode["seed"], 7)
        self.assertEqual(episode["phase"], "validation")
        self.assertEqual(episode["episode_id"], 0)

    def test_eval_checkpoint_step_is_validated(self) -> None:
        evaluator = Evaluator(lambda **_: _successful_episode(), validation_seeds=(1,), test_seeds=(2,))
        with self.assertRaises(ValueError):
            evaluator.evaluate(global_step=1, checkpoint_step=-1)

    def test_final_test_requires_and_loads_best_then_computes_wilson_ci(self) -> None:
        loaded: list[Path] = []

        def runner(*, seed: int, phase: str, deterministic: bool) -> dict[str, object]:
            return _successful_episode(
                run_id="eval_run",
                phase=phase,
                episode_id=seed,
                landing_success=seed != 22,
                failure_reason="LAND_FAILED" if seed == 22 else None,
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            best = root / "checkpoints" / "best"
            latest = root / "checkpoints" / "latest"
            best.mkdir(parents=True)
            latest.mkdir(parents=True)
            (best / "checkpoint_meta.json").write_text(
                json.dumps({"global_step": 500}), encoding="utf-8"
            )
            evaluator = Evaluator(
                runner,
                validation_seeds=(10,),
                test_seeds=(20, 21, 22, 23),
                checkpoint_loader=loaded.append,
            )
            with self.assertRaises(BestCheckpointRequiredError):
                evaluator.run_final_test(latest, best_checkpoint_step=500, run_id="run_1")
            result = evaluator.run_final_test(best, run_id="run_1")

        self.assertEqual(loaded, [best])
        self.assertEqual(result["best_checkpoint_step"], 500)
        self.assertEqual(result["mission_success_rate"], 0.75)
        low, high = wilson_score_interval_95(3, 4)
        self.assertAlmostEqual(result["mission_success_ci95_low"], low)
        self.assertAlmostEqual(result["mission_success_ci95_high"], high)

    def test_validation_and_test_seed_sets_must_be_disjoint(self) -> None:
        with self.assertRaises(ValueError):
            Evaluator(lambda **_: {}, validation_seeds=(1, 2), test_seeds=(2, 3))
        with self.assertRaisesRegex(ValueError, "training, validation, and test"):
            Evaluator(
                lambda **_: {},
                training_seeds=(1, 2),
                validation_seeds=(2, 3),
                test_seeds=(4, 5),
            )


if __name__ == "__main__":
    unittest.main()
