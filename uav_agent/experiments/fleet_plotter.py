"""The four approved scalar-only fleet summary figures."""

from __future__ import annotations

import csv
from math import hypot, isfinite
from pathlib import Path
from typing import Mapping

from .fleet_result_recorder import DEFAULT_MAX_RUN_BYTES


class FleetPlotError(RuntimeError):
    """Raised when bounded scalar metrics cannot be plotted."""


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(row: Mapping[str, str], field: str) -> float | None:
    try:
        result = float(row.get(field, ""))
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _truth(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _directory_size(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


class FleetPlotter:
    """Generate at most four PNGs, exclusively from persisted scalar CSVs."""

    FIGURE_NAMES = (
        "mission_timeline.png",
        "xy_path.png",
        "separation.png",
        "goal_completion.png",
    )

    def __init__(
        self,
        run_dir: str | Path,
        *,
        enabled: bool = True,
        max_run_bytes: int = DEFAULT_MAX_RUN_BYTES,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.metrics_dir = self.run_dir / "metrics" if (self.run_dir / "metrics").is_dir() else self.run_dir
        self.figures_dir = self.run_dir / "figures"
        self.enabled = bool(enabled)
        if isinstance(max_run_bytes, bool) or not isinstance(max_run_bytes, int) or max_run_bytes <= 0:
            raise ValueError("max_run_bytes must be a positive integer")
        self.max_run_bytes = max_run_bytes

    @staticmethod
    def _pyplot():
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise FleetPlotError("matplotlib is required for summary figures") from exc
        return plt

    def _save(self, figure: object, filename: str) -> Path | None:
        if filename not in self.FIGURE_NAMES:
            raise ValueError("unsupported fleet figure name")
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        path = self.figures_dir / filename
        figure.savefig(path, dpi=110, bbox_inches="tight", format="png")
        self._pyplot().close(figure)
        if _directory_size(self.run_dir) > self.max_run_bytes:
            path.unlink(missing_ok=True)
            try:
                self.figures_dir.rmdir()
            except OSError:
                pass
            return None
        return path

    def plot_mission_timeline(self) -> Path | None:
        rows = _rows(self.metrics_dir / "skill_executions.csv")
        segments = []
        for row in rows:
            start = _number(row, "start_time_s")
            end = _number(row, "end_time_s")
            if start is not None and end is not None and end >= start:
                segments.append((row.get("uav_id", "unknown"), row.get("skill_name", "skill"), start, end))
        if not segments:
            return None
        uavs = sorted({item[0] for item in segments})
        plt = self._pyplot()
        fig, axis = plt.subplots(figsize=(8.2, max(3.0, 0.65 * len(uavs))))
        colors = plt.get_cmap("tab20")
        for index, uav_id in enumerate(uavs):
            for color_index, (_, skill, start, end) in enumerate(item for item in segments if item[0] == uav_id):
                axis.barh(index, end - start, left=start, height=0.55, color=colors(color_index % 20), alpha=0.85)
                if end - start >= 1.0:
                    axis.text((start + end) / 2.0, index, skill, ha="center", va="center", fontsize=7)
        axis.set_yticks(range(len(uavs)), labels=uavs)
        axis.set(xlabel="Simulation time (s)", title="Mission timeline")
        axis.grid(axis="x", alpha=0.25)
        return self._save(fig, "mission_timeline.png")

    def plot_xy_path(self) -> Path | None:
        rows = _rows(self.metrics_dir / "state_samples_1hz.csv")
        paths: dict[str, tuple[list[float], list[float]]] = {}
        for row in rows:
            x = _number(row, "x_m")
            y = _number(row, "y_m")
            if x is None or y is None:
                continue
            xs, ys = paths.setdefault(row.get("uav_id", "unknown"), ([], []))
            xs.append(x)
            ys.append(y)
        if not any(xs for xs, _ in paths.values()):
            return None
        plt = self._pyplot()
        fig, axis = plt.subplots(figsize=(6.5, 6.0))
        for uav_id, (xs, ys) in sorted(paths.items()):
            axis.plot(xs, ys, label=uav_id, linewidth=1.8)
            axis.scatter(xs[:1], ys[:1], marker="o", s=25)
            axis.scatter(xs[-1:], ys[-1:], marker="x", s=35)
        axis.set(xlabel="World X (m)", ylabel="World Y (m)", title="1 Hz XY paths")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
        axis.legend()
        return self._save(fig, "xy_path.png")

    def plot_separation(self) -> Path | None:
        rows = _rows(self.metrics_dir / "state_samples_1hz.csv")
        explicit: list[tuple[float, float]] = []
        positions: dict[float, list[tuple[float, float]]] = {}
        for row in rows:
            timestamp = _number(row, "timestamp_s")
            separation = _number(row, "minimum_inter_uav_distance_m")
            if timestamp is not None and separation is not None:
                explicit.append((timestamp, separation))
            x, y = _number(row, "x_m"), _number(row, "y_m")
            if timestamp is not None and x is not None and y is not None:
                positions.setdefault(timestamp, []).append((x, y))
        series = explicit
        if not series:
            for timestamp, points in sorted(positions.items()):
                if len(points) >= 2:
                    distances = [
                        hypot(a[0] - b[0], a[1] - b[1])
                        for index, a in enumerate(points)
                        for b in points[index + 1 :]
                    ]
                    if distances:
                        series.append((timestamp, min(distances)))
        if not series:
            return None
        plt = self._pyplot()
        fig, axis = plt.subplots(figsize=(7.2, 4.0))
        axis.plot([item[0] for item in series], [item[1] for item in series], color="#dc3912")
        axis.set(xlabel="Simulation time (s)", ylabel="Minimum separation (m)", title="Fleet separation")
        axis.grid(alpha=0.25)
        return self._save(fig, "separation.png")

    def plot_goal_completion(self) -> Path | None:
        rows = _rows(self.metrics_dir / "goal_metrics.csv")
        if not rows:
            return None
        totals: dict[str, list[int]] = {}
        for row in rows:
            goal_type = row.get("goal_type", "UNKNOWN") or "UNKNOWN"
            counts = totals.setdefault(goal_type, [0, 0])
            counts[1] += 1
            counts[0] += int(_truth(row.get("completed")))
        labels = sorted(totals)
        values = [totals[label][0] / totals[label][1] for label in labels]
        plt = self._pyplot()
        fig, axis = plt.subplots(figsize=(max(6.0, 0.8 * len(labels)), 4.0))
        axis.bar(labels, values, color="#109618")
        axis.set(ylabel="Completion rate", ylim=(0.0, 1.0), title="Goal completion")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        return self._save(fig, "goal_completion.png")

    def generate_all(self) -> tuple[Path, ...]:
        if not self.enabled:
            return ()
        generated = (
            self.plot_mission_timeline(),
            self.plot_xy_path(),
            self.plot_separation(),
            self.plot_goal_completion(),
        )
        return tuple(path for path in generated if path is not None)


__all__ = ["FleetPlotError", "FleetPlotter"]
