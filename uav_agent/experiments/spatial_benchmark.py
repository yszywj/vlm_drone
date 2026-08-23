"""Pure-Python aggregation for the five spatial-planning experiment modes.

The benchmark deliberately separates *execution policy* from route validity.
In particular, an ``open_sim`` route may be executed after structural checks,
but it contributes to ``route_validity_rate`` only when an independent STRICT
shadow evaluation populated ``shadow_strict_route_valid``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from numbers import Integral, Real
import os
from pathlib import Path
from typing import TypeAlias


class SpatialBenchmarkError(ValueError):
    """Raised when experiment configuration or a manifest is ambiguous."""


class ExperimentMode(str, Enum):
    """Closed set of comparable experiment conditions."""

    SCRIPTED_BASELINE = "scripted_baseline"
    CLASSICAL_BASELINE = "classical_baseline"
    QWEN_OPEN_SIM = "qwen_open_sim"
    QWEN_CRITIC_SIM = "qwen_critic_sim"
    QWEN_STRICT = "qwen_strict"


@dataclass(frozen=True, slots=True)
class ExperimentModeProfile:
    """Exact launcher controls implied by one benchmark mode."""

    mode: ExperimentMode
    planner: str
    planning_contract: str
    route_validation_mode: str
    route_planner_backend: str

    @property
    def uses_qwen_route_planner(self) -> bool:
        return self.route_planner_backend == "qwen"


_MODE_PROFILES = {
    ExperimentMode.SCRIPTED_BASELINE: ExperimentModeProfile(
        ExperimentMode.SCRIPTED_BASELINE,
        planner="dynamic_scripted",
        planning_contract="v2",
        route_validation_mode="strict",
        route_planner_backend="none",
    ),
    ExperimentMode.CLASSICAL_BASELINE: ExperimentModeProfile(
        ExperimentMode.CLASSICAL_BASELINE,
        planner="dynamic_scripted",
        planning_contract="v2",
        route_validation_mode="strict",
        route_planner_backend="classical",
    ),
    ExperimentMode.QWEN_OPEN_SIM: ExperimentModeProfile(
        ExperimentMode.QWEN_OPEN_SIM,
        planner="dynamic_llm",
        planning_contract="v3",
        route_validation_mode="open_sim",
        route_planner_backend="qwen",
    ),
    ExperimentMode.QWEN_CRITIC_SIM: ExperimentModeProfile(
        ExperimentMode.QWEN_CRITIC_SIM,
        planner="dynamic_llm",
        planning_contract="v3",
        route_validation_mode="critic_sim",
        route_planner_backend="qwen",
    ),
    ExperimentMode.QWEN_STRICT: ExperimentModeProfile(
        ExperimentMode.QWEN_STRICT,
        planner="dynamic_llm",
        planning_contract="v3",
        route_validation_mode="strict",
        route_planner_backend="qwen",
    ),
}


def experiment_mode_profile(mode: ExperimentMode | str) -> ExperimentModeProfile:
    """Return the immutable launch mapping for ``mode``."""

    try:
        normalized = mode if isinstance(mode, ExperimentMode) else ExperimentMode(mode)
    except (TypeError, ValueError):
        raise SpatialBenchmarkError("unsupported experiment_mode") from None
    return _MODE_PROFILES[normalized]


def resolve_experiment_profile(
    mode: ExperimentMode | str,
    *,
    planner: str | None = None,
    planning_contract: str | None = None,
    route_validation_mode: str | None = None,
) -> ExperimentModeProfile:
    """Resolve a mode and reject every explicitly conflicting low-level flag.

    Callers should pass ``None`` for low-level values that were not explicitly
    supplied by the user.  This lets the profile replace legacy parser defaults
    without silently overriding a contradictory command line.
    """

    profile = experiment_mode_profile(mode)
    supplied = {
        "planner": planner,
        "planning_contract": planning_contract,
        "route_validation_mode": route_validation_mode,
    }
    for name, value in supplied.items():
        expected = getattr(profile, name)
        if value is not None and value != expected:
            flag = "--" + name.replace("_", "-")
            raise SpatialBenchmarkError(
                f"{flag}={value} conflicts with --experiment-mode "
                f"{profile.mode.value} (requires {expected})"
            )
    return profile


def infer_experiment_mode(
    *,
    planner: str,
    route_validation_mode: str,
) -> ExperimentMode:
    """Map a legacy launch into one of the five auditable conditions."""

    if planner in {"scripted", "dynamic_scripted"}:
        return ExperimentMode.SCRIPTED_BASELINE
    if planner in {"llm", "dynamic_llm"}:
        by_validation = {
            "open_sim": ExperimentMode.QWEN_OPEN_SIM,
            "critic_sim": ExperimentMode.QWEN_CRITIC_SIM,
            "strict": ExperimentMode.QWEN_STRICT,
        }
        try:
            return by_validation[route_validation_mode]
        except KeyError:
            raise SpatialBenchmarkError(
                "unsupported route_validation_mode for Qwen experiment"
            ) from None
    raise SpatialBenchmarkError("planner does not map to a benchmark mode")


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise SpatialBenchmarkError(f"{name} must be a boolean")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SpatialBenchmarkError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise SpatialBenchmarkError(f"{name} must be a non-negative integer")
    return normalized


def _optional_nonnegative_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SpatialBenchmarkError(f"{name} must be a finite non-negative number or null")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise SpatialBenchmarkError(f"{name} must be a finite non-negative number or null")
    return normalized


def _optional_ratio(value: object, name: str) -> float | None:
    normalized = _optional_nonnegative_float(value, name)
    if normalized is not None and normalized > 1.0:
        raise SpatialBenchmarkError(f"{name} must be within [0, 1] or null")
    return normalized


@dataclass(frozen=True, slots=True)
class SpatialEpisodeResult:
    """One image-free episode row consumed by the benchmark aggregator."""

    run_id: str
    experiment_mode: ExperimentMode
    mission_success: bool
    collision_count: int
    shadow_strict_route_valid: bool | None
    route_repair_count: int
    search_coverage_ratio: float | None
    path_length_m: float | None
    planning_latency_s: float | None
    unseen_spatial_instruction: bool

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise SpatialBenchmarkError("run_id must be a non-empty string")
        if len(self.run_id) > 256 or any(ch in self.run_id for ch in "\x00\r\n"):
            raise SpatialBenchmarkError("run_id is invalid")
        object.__setattr__(self, "run_id", self.run_id.strip())
        if not isinstance(self.experiment_mode, ExperimentMode):
            try:
                object.__setattr__(self, "experiment_mode", ExperimentMode(self.experiment_mode))
            except (TypeError, ValueError):
                raise SpatialBenchmarkError("unsupported experiment_mode") from None
        object.__setattr__(self, "mission_success", _bool(self.mission_success, "mission_success"))
        object.__setattr__(self, "collision_count", _nonnegative_int(self.collision_count, "collision_count"))
        if self.shadow_strict_route_valid is not None:
            object.__setattr__(
                self,
                "shadow_strict_route_valid",
                _bool(self.shadow_strict_route_valid, "shadow_strict_route_valid"),
            )
        object.__setattr__(self, "route_repair_count", _nonnegative_int(self.route_repair_count, "route_repair_count"))
        object.__setattr__(self, "search_coverage_ratio", _optional_ratio(self.search_coverage_ratio, "search_coverage_ratio"))
        object.__setattr__(self, "path_length_m", _optional_nonnegative_float(self.path_length_m, "path_length_m"))
        object.__setattr__(self, "planning_latency_s", _optional_nonnegative_float(self.planning_latency_s, "planning_latency_s"))
        object.__setattr__(
            self,
            "unseen_spatial_instruction",
            _bool(self.unseen_spatial_instruction, "unseen_spatial_instruction"),
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, object],
        *,
        run_id: str | None = None,
    ) -> "SpatialEpisodeResult":
        """Parse a run manifest without treating execution acceptance as validity."""

        if not isinstance(manifest, Mapping):
            raise SpatialBenchmarkError("manifest must be a mapping")
        raw_mode = manifest.get("experiment_mode")
        if not isinstance(raw_mode, str):
            raise SpatialBenchmarkError("manifest.experiment_mode is required")
        resolved_run_id = run_id or manifest.get("mission_id")
        if not isinstance(resolved_run_id, str):
            raise SpatialBenchmarkError("run_id or manifest.mission_id is required")

        raw_success = manifest.get("mission_success")
        if raw_success is None:
            # Older manifests have trustworthy terminal states but no derived
            # success bit.  Collision and guard failures remain exclusions.
            raw_success = bool(
                manifest.get("agent_status") == "SUCCEEDED"
                and manifest.get("task_status") == "SUCCEEDED"
                and manifest.get("guard_error") is None
                and manifest.get("collision_count", 0) == 0
            )

        search = manifest.get("search")
        coverage = (
            search.get("coverage_ratio")
            if isinstance(search, Mapping)
            else manifest.get("coverage_ratio")
        )
        path_length = manifest.get("path_length_m")
        if path_length is None:
            path_length = manifest.get("route_length_m")
        latency = manifest.get("planning_latency_s")
        if latency is None:
            latency = manifest.get("route_planning_latency_s")

        # Deliberately do not inspect critique.status, route state, accepted
        # proposal count, or route_validation_mode here.  Only this independent
        # STRICT shadow result can enter the route-validity numerator.
        strict_valid = manifest.get("shadow_strict_route_valid")
        return cls(
            run_id=resolved_run_id,
            experiment_mode=raw_mode,  # type: ignore[arg-type]
            mission_success=raw_success,  # type: ignore[arg-type]
            collision_count=manifest.get("collision_count", 0),  # type: ignore[arg-type]
            shadow_strict_route_valid=strict_valid,  # type: ignore[arg-type]
            route_repair_count=manifest.get("route_repair_count", 0),  # type: ignore[arg-type]
            search_coverage_ratio=coverage,  # type: ignore[arg-type]
            path_length_m=path_length,  # type: ignore[arg-type]
            planning_latency_s=latency,  # type: ignore[arg-type]
            unseen_spatial_instruction=manifest.get(
                "unseen_spatial_instruction", False
            ),  # type: ignore[arg-type]
        )


ManifestSource: TypeAlias = Mapping[str, object] | str | Path


def _mean(values: Iterable[float]) -> float | None:
    items = tuple(values)
    return None if not items else sum(items) / len(items)


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _aggregate_rows(rows: tuple[SpatialEpisodeResult, ...]) -> dict[str, object]:
    route_rows = tuple(row for row in rows if row.shadow_strict_route_valid is not None)
    unseen_rows = tuple(row for row in rows if row.unseen_spatial_instruction)
    return {
        "episode_count": len(rows),
        "mission_success_rate": _rate(sum(row.mission_success for row in rows), len(rows)),
        "collision_rate": _rate(sum(row.collision_count > 0 for row in rows), len(rows)),
        "collision_count": sum(row.collision_count for row in rows),
        "route_validity_rate": _rate(
            sum(row.shadow_strict_route_valid is True for row in route_rows),
            len(route_rows),
        ),
        "route_validity_evaluated_count": len(route_rows),
        "average_repair_count": _mean(float(row.route_repair_count) for row in rows),
        "average_search_coverage": _mean(
            row.search_coverage_ratio
            for row in rows
            if row.search_coverage_ratio is not None
        ),
        "average_path_length_m": _mean(
            row.path_length_m for row in rows if row.path_length_m is not None
        ),
        "average_planning_latency_s": _mean(
            row.planning_latency_s
            for row in rows
            if row.planning_latency_s is not None
        ),
        "unseen_spatial_success_rate": _rate(
            sum(row.mission_success for row in unseen_rows),
            len(unseen_rows),
        ),
        "unseen_spatial_episode_count": len(unseen_rows),
    }


class SpatialBenchmarkAggregator:
    """Accumulate unique episodes and emit per-mode/overall metrics."""

    def __init__(self) -> None:
        self._rows: list[SpatialEpisodeResult] = []
        self._keys: set[tuple[ExperimentMode, str]] = set()

    @property
    def episodes(self) -> tuple[SpatialEpisodeResult, ...]:
        return tuple(self._rows)

    def add(self, result: SpatialEpisodeResult) -> None:
        if not isinstance(result, SpatialEpisodeResult):
            raise TypeError("result must be SpatialEpisodeResult")
        key = (result.experiment_mode, result.run_id)
        if key in self._keys:
            raise SpatialBenchmarkError(
                f"duplicate episode for {result.experiment_mode.value}: {result.run_id}"
            )
        self._keys.add(key)
        self._rows.append(result)

    def add_manifest(
        self,
        manifest: Mapping[str, object],
        *,
        run_id: str | None = None,
    ) -> None:
        self.add(SpatialEpisodeResult.from_manifest(manifest, run_id=run_id))

    def summary(self) -> dict[str, object]:
        rows = tuple(self._rows)
        modes = {
            mode.value: _aggregate_rows(
                tuple(row for row in rows if row.experiment_mode is mode)
            )
            for mode in ExperimentMode
        }
        return {
            "schema_version": 1,
            "route_validity_source": "shadow_strict_route_valid",
            "episode_count": len(rows),
            "modes": modes,
            "overall": _aggregate_rows(rows),
        }

    def write_summary(self, path: str | Path) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    self.summary(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination


def aggregate_spatial_manifests(
    sources: Iterable[ManifestSource],
) -> dict[str, object]:
    """Load manifest mappings/files and return the deterministic summary."""

    aggregator = SpatialBenchmarkAggregator()
    for source in sources:
        if isinstance(source, Mapping):
            aggregator.add_manifest(source)
            continue
        path = Path(source).expanduser()
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SpatialBenchmarkError(f"cannot read manifest {path}: {type(exc).__name__}") from None
        if not isinstance(decoded, dict):
            raise SpatialBenchmarkError(f"manifest {path} must contain a JSON object")
        aggregator.add_manifest(decoded, run_id=str(path.resolve()))
    return aggregator.summary()


__all__ = [
    "ExperimentMode",
    "ExperimentModeProfile",
    "SpatialBenchmarkAggregator",
    "SpatialBenchmarkError",
    "SpatialEpisodeResult",
    "aggregate_spatial_manifests",
    "experiment_mode_profile",
    "infer_experiment_mode",
    "resolve_experiment_profile",
]
