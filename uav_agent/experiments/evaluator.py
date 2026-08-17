"""Deterministic, scalar-only evaluation helpers.

This module deliberately has no Isaac Sim dependency.  An application supplies a
small episode runner and, for final testing, a checkpoint loader.  Mission
success is defined exactly once here and is reused by validation, final testing,
and :mod:`experiments.metric_logger`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Mapping, Protocol, Sequence

if TYPE_CHECKING:
    from tasks.intent_judge import IntentJudgeResult

try:
    from .schemas import MetricPhase
except ImportError:  # pragma: no cover - compatibility during partial installs
    from enum import Enum

    class MetricPhase(str, Enum):
        TRAIN = "train"
        VALIDATION = "validation"
        TEST = "test"


STRICT_SUCCESS_FIELDS = (
    "takeoff_success",
    "goto_search_success",
    "search_success",
    "correct_target_locked",
    "track_success",
    "return_success",
    "landing_success",
)

STRICT_FAILURE_FIELDS = (
    "false_target_lock",
    "collision",
    "out_of_bounds",
    "safety_abort",
    "timeout",
)


class EvaluationError(RuntimeError):
    """Base class for deterministic evaluation errors."""


class BestCheckpointRequiredError(EvaluationError):
    """Raised when final testing is not given the selected ``best`` checkpoint."""


class EpisodeRunner(Protocol):
    """Minimal pure-Python interface required by :class:`Evaluator`."""

    def run_episode(
        self,
        *,
        seed: int,
        phase: str,
        deterministic: bool,
    ) -> Mapping[str, object]:
        """Run one episode without updating model parameters."""


@dataclass(frozen=True, slots=True)
class BestCheckpoint:
    """Explicit token proving which checkpoint must be used for final testing."""

    path: Path
    step: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("best checkpoint step must be a non-negative integer")


def _metric_value(metrics: Mapping[str, object] | object, name: str) -> object:
    if isinstance(metrics, Mapping):
        return metrics.get(name)
    return getattr(metrics, name, None)


def parse_metric_bool(value: object) -> bool | None:
    """Parse persisted booleans while preserving an unmeasured ``None`` state."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
    return None


def _is_true(value: object) -> bool:
    return parse_metric_bool(value) is True


def _is_false(value: object) -> bool:
    return parse_metric_bool(value) is False


def compute_mission_success_strict(
    metrics: Mapping[str, object] | object,
) -> bool:
    """Return the single canonical full-mission success decision.

    Missing or unmeasured values never become an implicit success.  Reacquisition
    is intentionally not a mandatory field: it is conditional on target loss,
    whereas all seven normal mission stages and all five safety exclusions are
    unconditional.
    """

    return all(_is_true(_metric_value(metrics, name)) for name in STRICT_SUCCESS_FIELDS) and all(
        _is_false(_metric_value(metrics, name)) for name in STRICT_FAILURE_FIELDS
    )


def compute_instruction_grounded_success(
    execution_metrics: Mapping[str, object],
    intent_result: "IntentJudgeResult",
) -> bool:
    """Require both Gold-intent agreement and strict execution success.

    This versioned, independent metric intentionally leaves
    :func:`compute_mission_success_strict` and its existing producers unchanged.
    It must only be used when an episode has an independently authored Gold
    specification and a corresponding :class:`tasks.IntentJudgeResult`.
    """

    from tasks.intent_judge import IntentJudgeResult

    if not isinstance(execution_metrics, Mapping):
        raise TypeError("execution_metrics must be a mapping")
    if not isinstance(intent_result, IntentJudgeResult):
        raise TypeError("intent_result must be an IntentJudgeResult")
    return intent_result.semantic_match and compute_mission_success_strict(
        execution_metrics
    )


def wilson_score_interval_95(successes: int, total: int) -> tuple[float | None, float | None]:
    """Compute a two-sided 95% Wilson score interval for a binomial rate."""

    if isinstance(successes, bool) or isinstance(total, bool):
        raise TypeError("successes and total must be integers")
    if not isinstance(successes, int) or not isinstance(total, int):
        raise TypeError("successes and total must be integers")
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("expected 0 <= successes <= total")
    if total == 0:
        return None, None

    z = 1.959963984540054
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


# A shorter alias is useful for callers that already encode the confidence level
# in surrounding configuration.
wilson_interval_95 = wilson_score_interval_95
wilson_confidence_interval = wilson_score_interval_95


@contextmanager
def _default_inference_context():
    """Use torch inference mode when available, otherwise remain dependency-free."""

    try:
        import torch
    except ImportError:
        yield
        return
    with torch.inference_mode():
        yield


def _as_mapping(value: Mapping[str, object] | object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return dict(asdict(value))
    try:
        return dict(vars(value))
    except TypeError as exc:
        raise EvaluationError("episode runner must return a mapping or dataclass") from exc


def _finite_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _rate(episodes: Sequence[Mapping[str, object]], name: str) -> float | None:
    measured = [parse_metric_bool(episode.get(name)) for episode in episodes]
    measured_count = sum(value is not None for value in measured)
    if measured_count == 0:
        return None
    if measured_count != len(episodes):
        raise EvaluationError(
            f"metric {name!r} is missing from some, but not all, evaluation episodes"
        )
    complete = [value for value in measured if value is not None]
    return sum(complete) / len(complete)


def _conditional_reacquire_rate(
    episodes: Sequence[Mapping[str, object]],
) -> float | None:
    trigger_values = [
        parse_metric_bool(episode.get("reacquire_triggered")) for episode in episodes
    ]
    measured_count = sum(value is not None for value in trigger_values)
    if measured_count == 0:
        return None
    if measured_count != len(episodes):
        raise EvaluationError(
            "metric 'reacquire_triggered' is missing from some, but not all, evaluation episodes"
        )
    triggered = [
        episode
        for episode, was_triggered in zip(episodes, trigger_values)
        if was_triggered is True
    ]
    if not triggered:
        return None
    outcomes = [parse_metric_bool(episode.get("reacquire_success")) for episode in triggered]
    if any(value is None for value in outcomes):
        raise EvaluationError(
            "metric 'reacquire_success' is required for every triggered reacquisition episode"
        )
    return sum(value is True for value in outcomes) / len(outcomes)


def _mean_measured(episodes: Sequence[Mapping[str, object]], name: str) -> float | None:
    values = [_finite_number(episode.get(name)) for episode in episodes]
    measured_count = sum(value is not None for value in values)
    if measured_count == 0:
        return None
    if measured_count != len(episodes):
        raise EvaluationError(
            f"metric {name!r} is missing from some, but not all, evaluation episodes"
        )
    complete = [value for value in values if value is not None]
    return fmean(complete)


def aggregate_episode_metrics(
    episodes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate episode records using the canonical strict-success decision."""

    strict_successes = sum(compute_mission_success_strict(episode) for episode in episodes)
    return {
        "num_episodes": len(episodes),
        "mission_success_rate": strict_successes / len(episodes) if episodes else None,
        "takeoff_success_rate": _rate(episodes, "takeoff_success"),
        "goto_search_success_rate": _rate(episodes, "goto_search_success"),
        "search_success_rate": _rate(episodes, "search_success"),
        "correct_lock_rate": _rate(episodes, "correct_target_locked"),
        "false_lock_rate": _rate(episodes, "false_target_lock"),
        "track_success_rate": _rate(episodes, "track_success"),
        "reacquire_success_rate": _conditional_reacquire_rate(episodes),
        "return_success_rate": _rate(episodes, "return_success"),
        "landing_success_rate": _rate(episodes, "landing_success"),
        "collision_rate": _rate(episodes, "collision"),
        "safety_abort_rate": _rate(episodes, "safety_abort"),
        "mean_mission_time_s": _mean_measured(episodes, "mission_sim_time_s"),
        "mean_episode_return": _mean_measured(episodes, "episode_return"),
    }


class Evaluator:
    """Run fixed-seed validation and best-checkpoint final testing.

    The runner may be an object implementing :class:`EpisodeRunner` or a callable
    with the same keyword-only arguments.  Evaluation is wrapped by an injectable
    inference/no-gradient context and always requests deterministic execution.
    """

    def __init__(
        self,
        episode_runner: EpisodeRunner | Callable[..., Mapping[str, object]],
        *,
        training_seeds: Sequence[int] | None = None,
        validation_seeds: Sequence[int] | None = None,
        test_seeds: Sequence[int] | None = None,
        num_validation_episodes: int = 50,
        num_test_episodes: int = 200,
        interval_steps: int = 20_000,
        deterministic: bool = True,
        checkpoint_loader: Callable[[Path], None] | None = None,
        inference_context_factory: Callable[[], ContextManager[object]] | None = None,
    ) -> None:
        if isinstance(interval_steps, bool) or not isinstance(interval_steps, int) or interval_steps <= 0:
            raise ValueError("interval_steps must be a positive integer")
        if not deterministic:
            raise ValueError("evaluation must be deterministic (exploration disabled)")
        if isinstance(num_validation_episodes, bool) or num_validation_episodes <= 0:
            raise ValueError("num_validation_episodes must be positive")
        if isinstance(num_test_episodes, bool) or num_test_episodes <= 0:
            raise ValueError("num_test_episodes must be positive")

        self._episode_runner = episode_runner
        self._training_seeds = (
            () if training_seeds is None else self._normalize_seeds(training_seeds, "training")
        )
        self._validation_seeds = self._normalize_seeds(
            validation_seeds
            if validation_seeds is not None
            else range(10_000, 10_000 + num_validation_episodes),
            "validation",
        )
        self._test_seeds = self._normalize_seeds(
            test_seeds if test_seeds is not None else range(20_000, 20_000 + num_test_episodes),
            "test",
        )
        validation_set = set(self._validation_seeds)
        test_set = set(self._test_seeds)
        training_set = set(self._training_seeds)
        if validation_set.intersection(test_set):
            raise ValueError("validation and test seeds must be disjoint")
        if training_set.intersection(validation_set) or training_set.intersection(test_set):
            raise ValueError("training, validation, and test seeds must be disjoint")

        self.interval_steps = int(interval_steps)
        self.deterministic = True
        self._checkpoint_loader = checkpoint_loader
        self._inference_context_factory = inference_context_factory or _default_inference_context
        self._last_evaluated_step: int | None = None
        self._last_validation_episodes: tuple[dict[str, object], ...] = ()
        self._last_test_episodes: tuple[dict[str, object], ...] = ()

    @staticmethod
    def _normalize_seeds(seeds: Sequence[int], label: str) -> tuple[int, ...]:
        normalized = tuple(seeds)
        if not normalized:
            raise ValueError(f"{label} seeds must not be empty")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in normalized):
            raise TypeError(f"{label} seeds must contain only integers")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{label} seeds must be unique")
        return normalized

    @property
    def training_seeds(self) -> tuple[int, ...]:
        return self._training_seeds

    @property
    def validation_seeds(self) -> tuple[int, ...]:
        return self._validation_seeds

    @property
    def test_seeds(self) -> tuple[int, ...]:
        return self._test_seeds

    @property
    def last_validation_episodes(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(row) for row in self._last_validation_episodes)

    @property
    def last_test_episodes(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(row) for row in self._last_test_episodes)

    def should_evaluate(self, global_step: int) -> bool:
        if isinstance(global_step, bool) or not isinstance(global_step, int):
            raise TypeError("global_step must be an integer")
        return (
            global_step > 0
            and global_step % self.interval_steps == 0
            and global_step != self._last_evaluated_step
        )

    def evaluate(
        self,
        *,
        global_step: int = 0,
        checkpoint_step: int | None = None,
    ) -> dict[str, object]:
        """Run fixed validation seeds exactly once for ``global_step``."""

        if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
            raise ValueError("global_step must be a non-negative integer")
        if checkpoint_step is not None and (
            isinstance(checkpoint_step, bool)
            or not isinstance(checkpoint_step, int)
            or checkpoint_step < 0
        ):
            raise ValueError("checkpoint_step must be a non-negative integer or None")
        if global_step == self._last_evaluated_step:
            raise EvaluationError(f"validation already ran at global_step={global_step}")
        episodes = self._run_episodes(self._validation_seeds, MetricPhase.VALIDATION.value)
        summary = aggregate_episode_metrics(episodes)
        # Commit observable evaluator state only after every aggregate passes
        # completeness validation.
        self._last_validation_episodes = tuple(episodes)
        self._last_evaluated_step = global_step
        summary.update(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "global_step": global_step,
                "checkpoint_step": checkpoint_step if checkpoint_step is not None else global_step,
            }
        )
        return summary

    def run_final_test(
        self,
        checkpoint: BestCheckpoint | str | Path,
        *,
        best_checkpoint_step: int | None = None,
        checkpoint_step: int | None = None,
        run_id: str = "",
    ) -> dict[str, object]:
        """Load the selected ``best`` checkpoint and run the untouched test seeds."""

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id is required for final test metrics")
        if best_checkpoint_step is not None and checkpoint_step is not None:
            if best_checkpoint_step != checkpoint_step:
                raise BestCheckpointRequiredError("conflicting best checkpoint steps")
        selected = self._coerce_best_checkpoint(
            checkpoint,
            best_checkpoint_step if best_checkpoint_step is not None else checkpoint_step,
        )
        loader = self._checkpoint_loader
        if loader is None:
            candidate = getattr(self._episode_runner, "load_checkpoint", None)
            if callable(candidate):
                loader = candidate
        if loader is None:
            raise BestCheckpointRequiredError(
                "final testing requires a checkpoint_loader so the best checkpoint is loaded"
            )
        loader(selected.path)

        episodes = self._run_episodes(self._test_seeds, MetricPhase.TEST.value)
        summary = aggregate_episode_metrics(episodes)
        self._last_test_episodes = tuple(episodes)
        successes = sum(compute_mission_success_strict(episode) for episode in episodes)
        ci_low, ci_high = wilson_score_interval_95(successes, len(episodes))
        return {
            "run_id": run_id,
            "best_checkpoint_step": selected.step,
            "num_test_episodes": len(episodes),
            "mission_success_rate": successes / len(episodes),
            "mission_success_ci95_low": ci_low,
            "mission_success_ci95_high": ci_high,
            "search_success_rate": summary["search_success_rate"],
            "correct_lock_rate": summary["correct_lock_rate"],
            "false_lock_rate": summary["false_lock_rate"],
            "track_success_rate": summary["track_success_rate"],
            "reacquire_success_rate": summary["reacquire_success_rate"],
            "return_success_rate": summary["return_success_rate"],
            "landing_success_rate": summary["landing_success_rate"],
            "collision_rate": summary["collision_rate"],
            "safety_abort_rate": summary["safety_abort_rate"],
            "mean_mission_time_s": summary["mean_mission_time_s"],
            "mean_episode_return": summary["mean_episode_return"],
        }

    def _coerce_best_checkpoint(
        self,
        checkpoint: BestCheckpoint | str | Path,
        step: int | None,
    ) -> BestCheckpoint:
        if isinstance(checkpoint, BestCheckpoint):
            selected = checkpoint
            if step is not None and step != selected.step:
                raise BestCheckpointRequiredError("conflicting best checkpoint steps")
        else:
            path = Path(checkpoint)
            if path.name != "best":
                raise BestCheckpointRequiredError(
                    "final test checkpoint directory must be the selected 'best' directory"
                )
            resolved_step = step if step is not None else self._read_checkpoint_step(path)
            if resolved_step is None:
                raise BestCheckpointRequiredError(
                    "best checkpoint step is required or must exist in checkpoint_meta.json"
                )
            selected = BestCheckpoint(path=path, step=resolved_step)

        if selected.path.name != "best":
            raise BestCheckpointRequiredError("BestCheckpoint.path must name the 'best' directory")
        if not selected.path.is_dir():
            raise BestCheckpointRequiredError(f"best checkpoint does not exist: {selected.path}")
        return selected

    @staticmethod
    def _read_checkpoint_step(path: Path) -> int | None:
        metadata_path = path / "checkpoint_meta.json"
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        for key in ("global_step", "step", "checkpoint_step"):
            value = payload.get(key) if isinstance(payload, Mapping) else None
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    def _run_episodes(self, seeds: Sequence[int], phase: str) -> list[dict[str, object]]:
        episodes: list[dict[str, object]] = []
        with self._inference_context_factory():
            for index, seed in enumerate(seeds):
                runner = getattr(self._episode_runner, "run_episode", self._episode_runner)
                if not callable(runner):
                    raise EvaluationError("episode_runner is not callable")
                raw = runner(seed=seed, phase=phase, deterministic=True)
                episode = _as_mapping(raw)
                # These are evaluator-owned trust-boundary fields.  A buggy or
                # adversarial runner cannot relabel test data as training data.
                episode["seed"] = seed
                episode["phase"] = phase
                episode["episode_id"] = index
                episode["mission_success_strict"] = compute_mission_success_strict(episode)
                episodes.append(episode)
        return episodes

__all__ = [
    "BestCheckpoint",
    "BestCheckpointRequiredError",
    "EpisodeRunner",
    "EvaluationError",
    "Evaluator",
    "STRICT_FAILURE_FIELDS",
    "STRICT_SUCCESS_FIELDS",
    "aggregate_episode_metrics",
    "compute_instruction_grounded_success",
    "compute_mission_success_strict",
    "parse_metric_bool",
    "wilson_interval_95",
    "wilson_confidence_interval",
    "wilson_score_interval_95",
]
