"""Five-seed production YOLO/Isaac evaluation orchestration helpers.

This module deliberately does not import Isaac Sim.  Every episode is executed
by :mod:`scripts.run_fleet_mission` with a complete, validated production
configuration.  The only per-episode mutation is the declared experiment and
target motion seed; no runtime implementation is copied into this evaluator.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import signal
import subprocess
from typing import Callable

import yaml

from configs.loader import load_config
from experiments.fleet_batch_evaluator import load_fleet_episode_metrics


DEFAULT_FIXED_SEEDS: tuple[int, ...] = (101, 211, 307, 401, 503)
DEFAULT_EXPECTED_TRACK_DURATION_S = 20.0
DEFAULT_EPISODE_TIMEOUT_S = 600.0
_PERCEPTION_COUNT_FIELDS: tuple[str, ...] = (
    "yolo_requests",
    "yolo_successful_responses",
    "yolo_timeouts",
    "yolo_stale_results",
    "detections_total",
    "candidates_total",
    "candidates_confirmed",
    "color_observations",
    "color_matches",
    "color_mismatches",
    "color_pending",
    "depth_resolution_attempts",
    "depth_resolution_successes",
    "depth_resolution_failures",
    "measurement_created",
    "measurement_rejected",
    "track_id_switches",
)


class YoloFixedSeedEvaluationError(RuntimeError):
    """Raised for an invalid evaluation contract or missing run artifact."""


def _strict_seed(value: object, name: str = "seed") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def parse_fixed_seeds(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse and validate at least five unique, deterministic episode seeds."""

    raw: Sequence[object]
    if isinstance(value, str):
        pieces = tuple(item.strip() for item in value.split(","))
        if any(not item for item in pieces):
            raise ValueError("fixed seeds must be comma-separated integers")
        try:
            raw = tuple(int(item) for item in pieces)
        except ValueError as exc:
            raise ValueError("fixed seeds must be comma-separated integers") from exc
    else:
        raw = tuple(value)
    seeds = tuple(_strict_seed(item) for item in raw)
    if len(seeds) < 5:
        raise ValueError("real YOLO evaluation requires at least five fixed seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("fixed evaluation seeds must be unique")
    return seeds


def _yaml_object(path: Path) -> dict[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise YoloFixedSeedEvaluationError(
            f"could not read complete runtime config {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise YoloFixedSeedEvaluationError("runtime config must contain an object")
    return value


def _single_target_mapping(config: dict[str, object]) -> dict[str, object]:
    legacy = config.get("target")
    fleet = config.get("targets")
    if isinstance(legacy, Mapping) and fleet is None:
        # Work on the mutable object returned by safe_load.
        return legacy  # type: ignore[return-value]
    if isinstance(fleet, list) and len(fleet) == 1 and isinstance(fleet[0], dict):
        return fleet[0]
    raise YoloFixedSeedEvaluationError(
        "fixed-seed YOLO evaluation requires exactly one configured target"
    )


def _expected_model_sha256(config: Mapping[str, object]) -> str:
    perception = config.get("target_perception")
    detector = perception.get("detector") if isinstance(perception, Mapping) else None
    value = detector.get("expected_model_sha256") if isinstance(detector, Mapping) else None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.casefold())
    ):
        raise YoloFixedSeedEvaluationError(
            "production detector.expected_model_sha256 must be a 64-character digest"
        )
    return value.casefold()


def materialize_seed_config(
    source: str | Path,
    destination: str | Path,
    *,
    seed: int,
) -> dict[str, object]:
    """Write one complete config whose sole stochastic override is its seed.

    A generated file is intentionally retained with the evaluation artifacts,
    rather than using a partial YAML overlay or a hidden runtime override.
    ``load_config`` validates the materialized AppConfig before Isaac is ever
    launched.
    """

    seed = _strict_seed(seed)
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    raw = _yaml_object(source_path)
    target = _single_target_mapping(raw)
    motion = target.get("motion")
    if not isinstance(motion, dict):
        raise YoloFixedSeedEvaluationError("configured target.motion must be an object")
    motion["seed"] = seed
    experiment = raw.get("experiment")
    if not isinstance(experiment, dict):
        raise YoloFixedSeedEvaluationError("config.experiment must be an object")
    experiment["seed"] = seed

    perception = raw.get("target_perception")
    if not isinstance(perception, Mapping):
        raise YoloFixedSeedEvaluationError("config.target_perception must be an object")
    if str(perception.get("backend", "")).casefold() != "ultralytics_service":
        raise YoloFixedSeedEvaluationError(
            "fixed-seed production evaluation requires ultralytics_service"
        )
    geometry = perception.get("geometry")
    if not isinstance(geometry, Mapping) or geometry.get("mode") not in {
        "isaac_depth",
        "temporal_ray_depth",
    }:
        raise YoloFixedSeedEvaluationError(
            "production evaluation requires isaac_depth or temporal_ray_depth geometry"
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(destination_path)
    finally:
        temporary.unlink(missing_ok=True)
    # Full schema validation here makes configuration failure pre-Isaac and
    # ensures the retained YAML is independently replayable.
    loaded = load_config(destination_path)
    if len(loaded.targets) != 1 or loaded.targets[0].motion.seed != seed:
        raise YoloFixedSeedEvaluationError("materialized target seed did not round-trip")
    if loaded.experiment.seed != seed:
        raise YoloFixedSeedEvaluationError("materialized experiment seed did not round-trip")
    return raw


def build_runtime_command(
    *,
    project_root: str | Path,
    config: str | Path,
    output_root: str | Path,
    instruction: str,
    max_sim_time_s: float,
) -> tuple[str, ...]:
    """Return the one official production runtime invocation for an episode."""

    if not isinstance(max_sim_time_s, (int, float)) or isinstance(max_sim_time_s, bool):
        raise TypeError("max_sim_time_s must be numeric")
    if not isfinite(float(max_sim_time_s)) or float(max_sim_time_s) <= 0.0:
        raise ValueError("max_sim_time_s must be finite and positive")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be non-empty")
    root = Path(project_root).expanduser().resolve()
    return (
        str(root / "python.sh"),
        "scripts/run_fleet_mission.py",
        "--config",
        str(Path(config).expanduser().resolve()),
        "--mission-interpreter",
        "scripted",
        "--fleet-planner",
        "scripted",
        "--local-planner",
        "dynamic_scripted",
        "--planning-contract",
        "v3",
        "--runtime-program",
        "linear",
        "--target-perception-mode",
        "yolo",
        "--perception-runtime-profile",
        "production",
        "--headless",
        "--max-sim-time",
        f"{float(max_sim_time_s):g}",
        "--output-root",
        str(Path(output_root).expanduser().resolve()),
        "--no-summary-figures",
        "--instruction",
        instruction.strip(),
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise YoloFixedSeedEvaluationError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise YoloFixedSeedEvaluationError(f"{path} must contain an object")
    return value


def discover_episode_run(seed_output_root: str | Path) -> Path:
    """Find the exactly-one managed run emitted beneath one seed directory."""

    root = Path(seed_output_root).expanduser().resolve()
    candidates = sorted(
        path.parent
        for path in root.rglob("summary.json")
        if (path.parent / "run_manifest.json").is_file()
    )
    if len(candidates) != 1:
        raise YoloFixedSeedEvaluationError(
            f"expected exactly one managed run below {root}, found {len(candidates)}"
        )
    return candidates[0]


def _perception_counts(summary: Mapping[str, object]) -> dict[str, int]:
    rows = summary.get("perception_by_uav")
    mappings = (
        tuple(value for value in rows.values() if isinstance(value, Mapping))
        if isinstance(rows, Mapping)
        else ()
    )
    result: dict[str, int] = {}
    for name in _PERCEPTION_COUNT_FIELDS:
        total = 0
        for row in mappings:
            value = row.get(name, 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            total += int(value)
        result[name] = total
    return result


def _failure_case(run_dir: Path) -> dict[str, str]:
    path = run_dir / "metrics" / "failure_cases.csv"
    if not path.is_file():
        path = run_dir / "failure_cases.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    return dict(rows[0]) if rows else {}


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return 0.0
    try:
        result = float(value)
    except ValueError:
        return 0.0
    return result if isfinite(result) else 0.0


def _model_sha256_values(summary: Mapping[str, object]) -> tuple[str, ...]:
    values: set[str] = set()
    services = summary.get("yolo_services_by_uav")
    if isinstance(services, Mapping):
        values.update(
            str(row.get("model_sha256"))
            for row in services.values()
            if isinstance(row, Mapping)
            and isinstance(row.get("model_sha256"), str)
        )
    startup_hashes = summary.get("yolo_model_sha256_by_uav")
    if isinstance(startup_hashes, Mapping):
        values.update(
            str(value) for value in startup_hashes.values() if isinstance(value, str)
        )
    return tuple(sorted(values))


def summarize_episode_run(
    run_dir: str | Path,
    *,
    seed: int,
    process_exit_code: int,
    expected_track_duration_s: float = DEFAULT_EXPECTED_TRACK_DURATION_S,
    preflight_verified_model_sha256: str | None = None,
) -> dict[str, object]:
    """Extract auditable perception and mission stages from one real run."""

    seed = _strict_seed(seed)
    run = Path(run_dir).expanduser().resolve()
    summary = _read_json(run / "summary.json")
    normalized = load_fleet_episode_metrics(run)
    counts = _perception_counts(summary)
    detection_success = counts["detections_total"] > 0
    color_confirmed = counts["color_matches"] > 0
    color_decisions = counts["color_matches"] + counts["color_mismatches"]
    geometry_success = counts["measurement_created"] > 0
    search_success = _bool(summary.get("search_success"))
    track_duration_s = _float(summary.get("valid_track_duration_s"))
    track_success = track_duration_s + 0.25 >= float(expected_track_duration_s)
    landing_success = _bool(summary.get("landing_success"))
    summary_strict_success = _bool(normalized.get("strict_success"))
    strict_success = summary_strict_success and process_exit_code == 0
    failure = _failure_case(run)
    if strict_success:
        failure_stage = None
        failure_reason = None
    elif process_exit_code != 0 and summary_strict_success:
        # A wrapper or post-runtime process failure must not be hidden by a
        # previously durable successful mission summary.  os._exit preserves
        # the official run_fleet_mission code, and subprocess exposes it here.
        failure_stage = "PROCESS_EXIT"
        failure_reason = f"PROCESS_EXIT_{process_exit_code}"
    else:
        inferred = next(
            stage
            for passed, stage in (
                (detection_success, "DETECTION"),
                (color_confirmed, "COLOR_CONFIRMATION"),
                (geometry_success, "GEOMETRY_3D"),
                (search_success, "SEARCH"),
                (track_success, "TRACK"),
                (landing_success, "LAND"),
                (False, "RUNTIME"),
            )
            if not passed
        )
        failure_stage = failure.get("stage") or inferred
        failure_reason = (
            failure.get("code")
            or str(normalized.get("failure_reason") or summary.get("last_error") or summary.get("status"))
        )
    model_sha256 = list(_model_sha256_values(summary))
    model_sha256_source = "runtime_summary"
    if not model_sha256 and preflight_verified_model_sha256 is not None:
        model_sha256 = [preflight_verified_model_sha256]
        model_sha256_source = "preflight_verified_identity"
    return {
        "seed": seed,
        "run_dir": str(run),
        "process_exit_code": int(process_exit_code),
        "status": str(summary.get("status", normalized.get("status", "UNKNOWN"))),
        "strict_success": strict_success,
        "model_sha256": model_sha256,
        "model_sha256_source": model_sha256_source,
        "detection": {
            "success": detection_success,
            "requests": counts["yolo_requests"],
            "successful_responses": counts["yolo_successful_responses"],
            "timeouts": counts["yolo_timeouts"],
            "stale_results": counts["yolo_stale_results"],
            "detections": counts["detections_total"],
            "candidates": counts["candidates_total"],
            "time_to_first_detection_s": summary.get("time_to_first_detection_s"),
        },
        "color": {
            "confirmed": color_confirmed,
            "observations": counts["color_observations"],
            "matches": counts["color_matches"],
            "mismatches": counts["color_mismatches"],
            "pending": counts["color_pending"],
            "sensor_match_rate": (
                counts["color_matches"] / color_decisions if color_decisions else None
            ),
        },
        "geometry_3d": {
            "success": geometry_success,
            "resolution_attempts": counts["depth_resolution_attempts"],
            "resolution_successes": counts["depth_resolution_successes"],
            "resolution_failures": counts["depth_resolution_failures"],
            "measurements_created": counts["measurement_created"],
            "measurements_rejected": counts["measurement_rejected"],
        },
        "search": {
            "success": search_success,
            "time_to_lock_s": summary.get("time_to_lock_s"),
        },
        "track": {
            "success": track_success,
            "required_duration_s": float(expected_track_duration_s),
            "valid_duration_s": track_duration_s,
            "target_lost_count": int(_float(summary.get("target_lost_count"))),
            "track_id_switches": counts["track_id_switches"],
        },
        "land": {
            "success": landing_success,
            "return_success": _bool(summary.get("return_success")),
        },
        "failure": {
            "stage": failure_stage,
            "reason": failure_reason,
            "message": (
                f"runtime subprocess exited with code {process_exit_code}"
                if failure_stage == "PROCESS_EXIT"
                else failure.get("message") or summary.get("last_error")
            ),
        },
        # This evaluator intentionally never reads evaluator frames or Oracle
        # state.  These are sensor/control outcomes, not GT error metrics.
        "oracle_metrics_used": False,
    }


def launch_failure_episode(
    *,
    seed: int,
    process_exit_code: int,
    reason: str,
    stage: str = "LAUNCH",
) -> dict[str, object]:
    """Create a structurally explicit record if no managed run was emitted."""

    return {
        "seed": _strict_seed(seed),
        "run_dir": None,
        "process_exit_code": int(process_exit_code),
        "status": "FAILED_TO_EMIT_RUN",
        "strict_success": False,
        "model_sha256": [],
        "detection": {"success": False},
        "color": {"confirmed": False},
        "geometry_3d": {"success": False},
        "search": {"success": False},
        "track": {"success": False},
        "land": {"success": False},
        "failure": {"stage": stage, "reason": reason, "message": reason},
        "oracle_metrics_used": False,
    }


def aggregate_episode_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate all fixed seeds without dropping failed episodes."""

    if not records:
        raise ValueError("at least one episode record is required")

    def stage_rate(section: str, field: str) -> float:
        return sum(
            _bool(row.get(section, {}).get(field))
            for row in records
            if isinstance(row.get(section), Mapping)
        ) / len(records)

    failures = Counter(
        str(failure.get("stage") or "UNKNOWN")
        for row in records
        if not _bool(row.get("strict_success"))
        and isinstance((failure := row.get("failure")), Mapping)
    )
    color_matches = sum(
        int(_float(section.get("matches")))
        for row in records
        if isinstance((section := row.get("color")), Mapping)
    )
    color_mismatches = sum(
        int(_float(section.get("mismatches")))
        for row in records
        if isinstance((section := row.get("color")), Mapping)
    )
    color_decisions = color_matches + color_mismatches
    detector_requests = sum(
        int(_float(section.get("requests")))
        for row in records
        if isinstance((section := row.get("detection")), Mapping)
    )
    detector_responses = sum(
        int(_float(section.get("successful_responses")))
        for row in records
        if isinstance((section := row.get("detection")), Mapping)
    )
    detections = sum(
        int(_float(section.get("detections")))
        for row in records
        if isinstance((section := row.get("detection")), Mapping)
    )
    depth_attempts = sum(
        int(_float(section.get("resolution_attempts")))
        for row in records
        if isinstance((section := row.get("geometry_3d")), Mapping)
    )
    depth_successes = sum(
        int(_float(section.get("resolution_successes")))
        for row in records
        if isinstance((section := row.get("geometry_3d")), Mapping)
    )
    measurements_created = sum(
        int(_float(section.get("measurements_created")))
        for row in records
        if isinstance((section := row.get("geometry_3d")), Mapping)
    )
    model_hashes = sorted(
        {
            str(value)
            for row in records
            for value in (
                row.get("model_sha256", ())
                if isinstance(row.get("model_sha256"), Sequence)
                and not isinstance(row.get("model_sha256"), str)
                else ()
            )
        }
    )
    return {
        "schema_version": 1,
        "episode_count": len(records),
        "fixed_seeds": [int(row["seed"]) for row in records],
        "strict_success_rate": sum(_bool(row.get("strict_success")) for row in records)
        / len(records),
        "stage_success_rate": {
            "detection": stage_rate("detection", "success"),
            "color_confirmation": stage_rate("color", "confirmed"),
            "geometry_3d": stage_rate("geometry_3d", "success"),
            "search": stage_rate("search", "success"),
            "track": stage_rate("track", "success"),
            "land": stage_rate("land", "success"),
        },
        "color_sensor_match_rate": (
            color_matches / color_decisions if color_decisions else None
        ),
        "color_matches": color_matches,
        "color_mismatches": color_mismatches,
        "detector_response_rate": (
            detector_responses / detector_requests if detector_requests else None
        ),
        "geometry_resolution_success_rate": (
            depth_successes / depth_attempts if depth_attempts else None
        ),
        "totals": {
            "detector_requests": detector_requests,
            "detector_successful_responses": detector_responses,
            "detections": detections,
            "depth_resolution_attempts": depth_attempts,
            "depth_resolution_successes": depth_successes,
            "measurements_created": measurements_created,
        },
        "failure_stage_counts": dict(sorted(failures.items())),
        "model_sha256": model_hashes,
        "oracle_metrics_used": False,
        "episodes": [dict(row) for row in records],
    }


def write_json_atomic(path: str | Path, value: Mapping[str, object]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EvaluationLaunchResult:
    exit_code: int
    records: tuple[Mapping[str, object], ...]
    summary: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _BoundedProcessResult:
    returncode: int
    timed_out: bool = False


def _run_bounded_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_s: float,
) -> _BoundedProcessResult:
    """Run one conda/Isaac tree with a wall-clock timeout and group cleanup."""

    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    try:
        return _BoundedProcessResult(
            returncode=process.wait(timeout=timeout_s),
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return _BoundedProcessResult(returncode=124, timed_out=True)


def run_fixed_seed_evaluation(
    *,
    project_root: str | Path,
    source_config: str | Path,
    evaluation_root: str | Path,
    seeds: Sequence[int],
    instruction: str,
    max_sim_time_s: float,
    expected_track_duration_s: float = DEFAULT_EXPECTED_TRACK_DURATION_S,
    episode_timeout_s: float = DEFAULT_EPISODE_TIMEOUT_S,
    command_runner: Callable[..., object] | None = None,
) -> EvaluationLaunchResult:
    """Materialize, launch, and aggregate all episodes through one runtime."""

    fixed_seeds = parse_fixed_seeds(tuple(seeds))
    if (
        isinstance(expected_track_duration_s, bool)
        or not isinstance(expected_track_duration_s, (int, float))
        or not isfinite(float(expected_track_duration_s))
        or float(expected_track_duration_s) <= 0.0
    ):
        raise ValueError("expected_track_duration_s must be finite and positive")
    if (
        isinstance(episode_timeout_s, bool)
        or not isinstance(episode_timeout_s, (int, float))
        or not isfinite(float(episode_timeout_s))
        or float(episode_timeout_s) <= 0.0
    ):
        raise ValueError("episode_timeout_s must be finite and positive")
    root = Path(project_root).expanduser().resolve()
    source = Path(source_config).expanduser().resolve()
    output = Path(evaluation_root).expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise YoloFixedSeedEvaluationError(
            f"evaluation output root already exists; refusing to overwrite: {output}"
        ) from exc
    config_dir = output / "configs"
    run_root = output / "runtime_runs"
    records: list[Mapping[str, object]] = []
    source_raw = _yaml_object(source)
    expected_model_sha256 = _expected_model_sha256(source_raw)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_config": str(source),
        "source_config_sha256": file_sha256(source),
        "fixed_seeds": list(fixed_seeds),
        "instruction": instruction,
        "max_sim_time_s": float(max_sim_time_s),
        "episode_timeout_s": float(episode_timeout_s),
        "expected_track_duration_s": float(expected_track_duration_s),
        "runtime_entrypoint": "scripts/run_fleet_mission.py",
        "target_perception_mode": "yolo",
        "perception_runtime_profile": "production",
        "expected_model_sha256": expected_model_sha256,
        "oracle_metrics_used": False,
    }
    write_json_atomic(output / "evaluation_manifest.json", manifest)
    any_process_failure = False
    for seed in fixed_seeds:
        config = config_dir / f"seed_{seed}.yaml"
        seed_output = run_root / f"seed_{seed}"
        materialize_seed_config(source, config, seed=seed)
        command = build_runtime_command(
            project_root=root,
            config=config,
            output_root=seed_output,
            instruction=instruction,
            max_sim_time_s=max_sim_time_s,
        )
        completed = (
            _run_bounded_command(
                command,
                cwd=root,
                timeout_s=float(episode_timeout_s),
            )
            if command_runner is None
            else command_runner(command, cwd=root, check=False)
        )
        return_code = getattr(completed, "returncode", None)
        if isinstance(return_code, bool) or not isinstance(return_code, int):
            raise TypeError("command_runner result must expose integer returncode")
        timed_out = bool(getattr(completed, "timed_out", False))
        any_process_failure = any_process_failure or return_code != 0
        try:
            run_dir = discover_episode_run(seed_output)
            record = summarize_episode_run(
                run_dir,
                seed=seed,
                process_exit_code=return_code,
                expected_track_duration_s=expected_track_duration_s,
                preflight_verified_model_sha256=expected_model_sha256,
            )
            if timed_out:
                record = {
                    **record,
                    "status": "EPISODE_TIMEOUT",
                    "strict_success": False,
                    "failure": {
                        "stage": "EPISODE_TIMEOUT",
                        "reason": "WALL_CLOCK_TIMEOUT",
                        "message": (
                            "episode exceeded wall-clock timeout "
                            f"of {float(episode_timeout_s):g}s"
                        ),
                    },
                }
        except Exception as exc:
            timeout_reason = (
                "episode exceeded wall-clock timeout "
                f"of {float(episode_timeout_s):g}s"
            )
            record = launch_failure_episode(
                seed=seed,
                process_exit_code=return_code,
                reason=(
                    timeout_reason
                    if timed_out
                    else f"{type(exc).__name__}: {exc}"
                ),
                stage="EPISODE_TIMEOUT" if timed_out else "LAUNCH",
            )
            any_process_failure = True
        records.append(record)
        any_process_failure = any_process_failure or not _bool(
            record.get("strict_success")
        )
        with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
        write_json_atomic(output / "summary.partial.json", aggregate_episode_records(records))
    summary = aggregate_episode_records(records)
    write_json_atomic(output / "summary.json", summary)
    (output / "summary.partial.json").unlink(missing_ok=True)
    return EvaluationLaunchResult(
        exit_code=1 if any_process_failure else 0,
        records=tuple(records),
        summary=summary,
    )


__all__ = [
    "DEFAULT_EPISODE_TIMEOUT_S",
    "DEFAULT_EXPECTED_TRACK_DURATION_S",
    "DEFAULT_FIXED_SEEDS",
    "EvaluationLaunchResult",
    "YoloFixedSeedEvaluationError",
    "aggregate_episode_records",
    "build_runtime_command",
    "discover_episode_run",
    "file_sha256",
    "launch_failure_episode",
    "materialize_seed_config",
    "parse_fixed_seeds",
    "run_fixed_seed_evaluation",
    "summarize_episode_run",
    "write_json_atomic",
]
