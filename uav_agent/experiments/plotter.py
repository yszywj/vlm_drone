"""Generate the small, fixed PNG report set from persisted CSV files."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Mapping


class PlotError(RuntimeError):
    """Raised when persisted metrics cannot be plotted safely."""


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        raise PlotError(f"could not read metrics CSV {path}: {exc}") from exc


def _number(row: Mapping[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw is None or not str(raw).strip():
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _bool(row: Mapping[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def _best_eval_index(rows: list[Mapping[str, str]]) -> int | None:
    """Apply the same complete ordering used by ``CheckpointManager``."""

    candidates: list[tuple[tuple[float, ...], int]] = []
    for index, row in enumerate(rows):
        success = _number(row, "mission_success_rate")
        if success is None:
            continue
        false_lock = _number(row, "false_lock_rate")
        collision = _number(row, "collision_rate")
        safety_abort = _number(row, "safety_abort_rate")
        mission_time = _number(row, "mean_mission_time_s")
        candidates.append(
            (
                (
                    -success,
                    1.0 if false_lock is None else false_lock,
                    1.0 if collision is None else collision,
                    1.0 if safety_abort is None else safety_abort,
                    math.inf if mission_time is None else mission_time,
                    float(index),
                ),
                index,
            )
        )
    return min(candidates)[1] if candidates else None


class ExperimentPlotter:
    """Read one run directory and render only the approved PNG figures."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.metrics_dir = self.run_dir / "metrics"
        self.figures_dir = self.run_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _pyplot():
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise PlotError("matplotlib is required to generate experiment figures") from exc
        return plt

    @staticmethod
    def _series(
        rows: Iterable[Mapping[str, str]], x_key: str, y_key: str
    ) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            x = _number(row, x_key)
            y = _number(row, y_key)
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        return xs, ys

    def _save(self, fig: object, name: str) -> Path:
        path = self.figures_dir / name
        fig.savefig(path, dpi=130, bbox_inches="tight", format="png")
        self._pyplot().close(fig)
        return path

    def plot_train_success_rate(self) -> Path | None:
        rows = _rows(self.metrics_dir / "train_metrics.csv")
        xs, ys = self._series(rows, "global_step", "mission_success_rate_100")
        if not xs:
            return None
        plt = self._pyplot()
        fig, axis = plt.subplots(figsize=(6.4, 4.0))
        axis.plot(xs, ys, color="#3366cc", linewidth=2)
        axis.set(xlabel="Global step", ylabel="Strict success rate", ylim=(0.0, 1.0))
        axis.set_title("Training success rate (rolling 100 episodes)")
        axis.grid(alpha=0.25)
        return self._save(fig, "train_success_rate.png")

    def plot_eval_success_rate(self) -> Path | None:
        rows = _rows(self.metrics_dir / "eval_metrics.csv")
        xs, ys = self._series(rows, "global_step", "mission_success_rate")
        if not xs:
            return None
        # The series omits malformed rows, while the best selector works on
        # complete records. Rebuild aligned rows so both indices describe the
        # same checkpoint.
        aligned_rows = [
            row
            for row in rows
            if _number(row, "global_step") is not None
            and _number(row, "mission_success_rate") is not None
        ]
        best_index = _best_eval_index(aligned_rows)
        if best_index is None:
            return None
        plt = self._pyplot()
        fig, axis = plt.subplots(figsize=(6.4, 4.0))
        axis.plot(xs, ys, marker="o", color="#109618")
        axis.scatter([xs[best_index]], [ys[best_index]], color="#dc3912", zorder=3)
        axis.annotate(
            f"best={ys[best_index]:.3f}",
            (xs[best_index], ys[best_index]),
            xytext=(8, 8),
            textcoords="offset points",
        )
        axis.set(xlabel="Global step", ylabel="Strict validation success", ylim=(0.0, 1.0))
        axis.set_title("Validation success rate")
        axis.grid(alpha=0.25)
        return self._save(fig, "eval_success_rate.png")

    def plot_final_success_rate(self) -> Path | None:
        rows = _rows(self.metrics_dir / "final_metrics.csv")
        if not rows:
            return None
        row = rows[-1]
        rate = _number(row, "mission_success_rate")
        low = _number(row, "mission_success_ci95_low")
        high = _number(row, "mission_success_ci95_high")
        episodes = _number(row, "num_test_episodes")
        if rate is None or low is None or high is None or episodes is None:
            return None
        plt = self._pyplot()
        fig, axis = plt.subplots(figsize=(5.4, 4.2))
        axis.bar(["Full mission"], [rate], color="#3366cc", width=0.5)
        axis.errorbar(
            [0],
            [rate],
            yerr=[[max(0.0, rate - low)], [max(0.0, high - rate)]],
            fmt="none",
            color="black",
            capsize=5,
        )
        axis.text(0, min(0.97, rate + 0.04), f"{rate:.1%}\n95% CI [{low:.1%}, {high:.1%}]\nn={int(episodes)}", ha="center")
        axis.set(ylabel="Strict test success", ylim=(0.0, 1.0))
        axis.set_title("Final test using best checkpoint")
        axis.grid(axis="y", alpha=0.25)
        return self._save(fig, "final_success_rate.png")

    def plot_stage_success_rate(self) -> Path | None:
        rows = [
            row
            for row in _rows(self.metrics_dir / "episode_metrics.csv")
            if row.get("phase") == "test"
        ]
        if not rows:
            return None
        stages = (
            ("Takeoff", "takeoff_success"),
            ("Goto Search", "goto_search_success"),
            ("Search", "search_success"),
            ("Correct Lock", "correct_target_locked"),
            ("Track", "track_success"),
            ("Return", "return_success"),
            ("Land", "landing_success"),
            ("Full Mission", "mission_success_strict"),
        )
        values = [sum(_bool(row, key) for row in rows) / len(rows) for _, key in stages]
        plt = self._pyplot()
        fig, axis = plt.subplots(figsize=(8.2, 4.4))
        axis.bar([label for label, _ in stages], values, color="#ff9900")
        axis.set(ylabel="Success rate", ylim=(0.0, 1.0))
        axis.set_title("Test-stage success rates")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        return self._save(fig, "stage_success_rate.png")

    def plot_failure_breakdown(self) -> Path | None:
        rows = _rows(self.metrics_dir / "failure_cases.csv")
        counts: dict[str, int] = {}
        for row in rows:
            reason = row.get("failure_reason", "").strip() or "UNKNOWN_ERROR"
            counts[reason] = counts.get(reason, 0) + 1
        if not counts:
            return None
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        plt = self._pyplot()
        fig, axis = plt.subplots(figsize=(8.0, max(3.5, 0.35 * len(ordered))))
        axis.barh([item[0] for item in reversed(ordered)], [item[1] for item in reversed(ordered)], color="#dc3912")
        axis.set(xlabel="Failed episodes", title="Failure breakdown")
        axis.grid(axis="x", alpha=0.25)
        return self._save(fig, "failure_breakdown.png")

    def plot_training_curve(self) -> Path | None:
        rows = _rows(self.metrics_dir / "train_metrics.csv")
        candidates = (
            ("episode_return_mean", "Episode return"),
            ("policy_loss", "Policy loss"),
            ("value_loss", "Value loss"),
            ("train_loss", "Train loss"),
            ("validation_loss", "Validation loss"),
        )
        series: list[tuple[str, list[float], list[float]]] = []
        for key, label in candidates:
            xs, ys = self._series(rows, "global_step", key)
            if xs:
                series.append((label, xs, ys))
        if not series:
            return None
        plt = self._pyplot()
        fig, axes = plt.subplots(len(series), 1, figsize=(7.0, max(3.2, 2.5 * len(series))), squeeze=False)
        for axis, (label, xs, ys) in zip(axes[:, 0], series):
            axis.plot(xs, ys, linewidth=1.8)
            axis.set(xlabel="Global step", ylabel=label)
            axis.grid(alpha=0.25)
        fig.suptitle("Training curves")
        return self._save(fig, "training_curve.png")

    def generate_all(self) -> tuple[Path, ...]:
        """Generate every figure whose source CSV contains suitable data."""

        generated = (
            self.plot_train_success_rate(),
            self.plot_eval_success_rate(),
            self.plot_final_success_rate(),
            self.plot_stage_success_rate(),
            self.plot_failure_breakdown(),
            self.plot_training_curve(),
        )
        return tuple(path for path in generated if path is not None)


__all__ = ["ExperimentPlotter", "PlotError"]
