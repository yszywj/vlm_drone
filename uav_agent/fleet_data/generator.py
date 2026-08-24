"""Deterministic Gold-first Fleet Planner v1 pilot dataset generator.

Every model input is the production :class:`fleet.types.FleetMissionRequest`
JSON contract. Gold outputs are likewise produced by production value objects
instead of maintaining a second, training-only Fleet schema.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import random
import shutil
import tempfile
from typing import Mapping

from fleet.types import (
    AssignmentFailurePolicy,
    FleetAssignment,
    FleetCoordinationPolicy,
    FleetMissionPlan,
    FleetMissionRequest,
    FleetPlanPatch,
    FleetStartPolicy,
    FleetTargetRequest,
    FleetUavCapability,
    RouteConflictPolicy,
    TargetClaimPolicy,
)
from planner.spatial import (
    CircleRegion,
    CoordinateFrame,
    CorridorRegion,
    PolygonRegion,
    RectangleRegion,
    RegionSpec,
    SectorRegion,
)
from target.types import TargetSpec

from fleet_data.validator import (
    FLEET_DATASET_CONTRACT,
    FLEET_DATASET_GENERATION_SOURCE,
    FLEET_DATASET_MANIFEST_SCHEMA_VERSION,
    FLEET_DATASET_SAMPLES_PER_SPLIT,
    FLEET_DATASET_SCENARIOS,
    FLEET_DATASET_SPLIT_SCENARIOS,
    FLEET_DATASET_SPLITS,
    validate_sample,
)


_SCENARIOS = (
    ("explicit_assignment", "circle"),
    ("natural_alias", "rectangle"),
    ("sector_region", "sector"),
    ("polygon_region", "polygon"),
    ("corridor_region", "corridor"),
    ("different_track_duration", "circle"),
    ("surplus_uav", "rectangle"),
    ("more_tasks_than_uavs", "sector"),
    ("unavailable_uav", "polygon"),
    ("capability_mismatch", "corridor"),
    ("duplicate_target_request", "circle"),
    ("similar_targets", "rectangle"),
    ("overlapping_regions_auto_assignment", "polygon"),
    ("failed_assignment_reassignment", "corridor"),
)

_POLICY = FleetCoordinationPolicy(
    target_claim_policy=TargetClaimPolicy.EXCLUSIVE,
    minimum_uav_separation_m=5.0,
    route_conflict_policy=RouteConflictPolicy.LOWER_PRIORITY_HOLDS,
    assignment_failure_policy=AssignmentFailurePolicy.REPORT_AND_REPLAN,
)


def _region(kind: str, offset: float) -> RegionSpec:
    if kind == "circle":
        return CircleRegion(CoordinateFrame.WORLD_ENU, (offset, 2.0, 0.0), 8.0)
    if kind == "rectangle":
        return RectangleRegion(
            CoordinateFrame.WORLD_ENU,
            (offset + 6.0, 1.0, 0.0),
            12.0,
            10.0,
        )
    if kind == "sector":
        return SectorRegion(
            CoordinateFrame.WORLD_ENU,
            (offset, 0.0, 0.0),
            7.5,
            75.0,
            (0.0, 12.0),
        )
    if kind == "polygon":
        return PolygonRegion(
            CoordinateFrame.WORLD_ENU,
            (
                (offset, 0.0, 0.0),
                (offset + 8.0, 1.0, 0.0),
                (offset + 4.0, 7.0, 0.0),
            ),
        )
    if kind == "corridor":
        return CorridorRegion(
            CoordinateFrame.WORLD_ENU,
            ((offset, 0.0, 0.0), (offset + 15.0, 5.0, 0.0)),
            2.0,
        )
    raise ValueError(f"unsupported private generator region kind: {kind}")


def _inventory(
    *,
    surplus: bool = False,
    unavailable_a: bool = False,
    current_assignment_id: str | None = None,
) -> tuple[FleetUavCapability, ...]:
    result = [
        FleetUavCapability(
            uav_id="uav_a",
            display_name=(
                "无人机A（第一架无人机/左边那架无人机/速度较快的无人机）"
            ),
            available=not unavailable_a,
            home_name="home_a",
            max_speed_mps=12.0,
            max_altitude_m=120.0,
            camera_modalities=("RGB",),
            payload_capabilities=("fast",),
            remaining_energy_ratio=0.9,
            current_assignment_id=current_assignment_id,
        ),
        FleetUavCapability(
            uav_id="uav_b",
            display_name="无人机B（第二架/带高分辨率相机的无人机）",
            available=True,
            home_name="home_b",
            max_speed_mps=10.0,
            max_altitude_m=120.0,
            camera_modalities=("RGB", "THERMAL"),
            payload_capabilities=("high_resolution_camera",),
            remaining_energy_ratio=0.85,
        ),
    ]
    if surplus:
        result.append(
            FleetUavCapability(
                uav_id="uav_c",
                display_name="无人机C（备用无人机）",
                available=True,
                home_name="home_c",
                max_speed_mps=8.0,
                max_altitude_m=100.0,
                camera_modalities=("RGB",),
                remaining_energy_ratio=1.0,
            )
        )
    return tuple(result)


def _target_spec(
    target_alias: str,
    *,
    similar_index: int | None = None,
    required_capability: str | None = None,
) -> TargetSpec:
    description = (
        f"相似的移动目标{similar_index}"
        if similar_index is not None
        else f"移动目标{target_alias.removeprefix('target_')}"
    )
    hard_attributes = (
        ()
        if required_capability is None
        else (f"required_payload:{required_capability}",)
    )
    return TargetSpec(
        description,
        category="moving_target",
        hard_attributes=hard_attributes,
        immutable_identity_summary=description,
    )


def _instruction(scenario: str) -> str:
    if scenario == "natural_alias":
        return (
            "第一架无人机，也就是左边那架无人机和速度较快的无人机，"
            "去M区跟踪目标i；带高分辨率相机的无人机去N区跟踪目标j。"
        )
    if scenario == "overlapping_regions_auto_assignment":
        return "两片搜索区域重叠，请自行选择合适的无人机分别寻找两个目标。"
    if scenario == "duplicate_target_request":
        return "让无人机A和无人机B都去寻找同一个目标i，但不要产生重复目标占用。"
    if scenario == "failed_assignment_reassignment":
        return "assignment_old 执行失败，请把目标i重新分配给仍可用的无人机。"
    if scenario == "more_tasks_than_uavs":
        return "两架无人机需要处理三个目标，无法执行的任务请明确保留为未分配。"
    if scenario == "unavailable_uav":
        return "无人机A不可用，请把原任务重新分配给无人机B，其余任务明确说明。"
    if scenario == "capability_mismatch":
        return "目标i需要高分辨率相机，请不要分配给能力不满足的无人机A。"
    return "无人机A去M附近找目标i；无人机B去N附近找目标j。"


def _sample(index: int, scenario: str, region_type: str) -> dict[str, object]:
    more_tasks = scenario == "more_tasks_than_uavs"
    duplicate_request = scenario == "duplicate_target_request"
    surplus = scenario == "surplus_uav"
    unavailable = scenario == "unavailable_uav"
    capability = scenario == "capability_mismatch"
    reassignment = scenario == "failed_assignment_reassignment"
    # A UAV that is unavailable before planning remains an ordinary
    # FleetMissionPlan decomposition case.  Only an execution-time assignment
    # failure uses the FleetPlanPatch contract.
    patch_output = reassignment
    auto = scenario == "overlapping_regions_auto_assignment"
    similar = scenario == "similar_targets"

    first_region = _region(region_type, float(index * 3))
    second_region = _region(
        "circle" if region_type == "corridor" else region_type,
        float(index * 3 + (2 if auto else 20)),
    )
    third_region = _region("circle", float(index * 3 + 40))
    target_count = 1 if duplicate_request else (3 if more_tasks else 2)
    regions = (first_region, second_region, third_region)
    target_requests: list[FleetTargetRequest] = []
    for target_index in range(target_count):
        alias = f"target_{chr(ord('i') + target_index)}"
        required_capability = (
            "high_resolution_camera"
            if capability and target_index == 0
            else None
        )
        requested_uav_id: str | None
        if (
            capability
            or reassignment
            or unavailable
            or auto
            or duplicate_request
            or scenario == "natural_alias"
        ):
            requested_uav_id = None
        elif target_index == 0:
            requested_uav_id = "uav_a"
        elif target_index == 1:
            requested_uav_id = "uav_b"
        else:
            requested_uav_id = None
        duration = (
            8.0
            if scenario == "different_track_duration" and target_index == 0
            else 25.0
            if scenario == "different_track_duration" and target_index == 1
            else 10.0
            if target_index == 0
            else 15.0
        )
        target_requests.append(
            FleetTargetRequest(
                target_alias=alias,
                target_spec=_target_spec(
                    alias,
                    similar_index=target_index + 1 if similar else None,
                    required_capability=required_capability,
                ),
                requested_uav_id=requested_uav_id,
                search_region=regions[target_index],
                track_duration_s=duration,
                required=True,
            )
        )

    old_assignment = "assignment_old" if reassignment else None
    assumptions = (
        ("assignment_old is terminal and must be replaced",)
        if reassignment
        else ("uav_a is unavailable before planning",)
        if unavailable
        else ()
    )
    request = FleetMissionRequest(
        fleet_mission_id=f"fleet_mission_dataset_{index:03d}",
        fleet_plan_version=1,
        original_instruction=_instruction(scenario),
        uav_inventory=_inventory(
            surplus=surplus,
            unavailable_a=unavailable or reassignment,
            current_assignment_id=old_assignment,
        ),
        target_requests=tuple(target_requests),
        coordination_policy=_POLICY,
        assumptions=assumptions,
    )

    unassigned: list[str] = []
    if unavailable or capability or reassignment:
        selected = (("uav_b", 0),)
        if target_count > 1:
            unassigned.append("target_j: no additional eligible UAV")
    elif duplicate_request:
        selected = (("uav_a", 0),)
        unassigned.append(
            "duplicate request for target_i rejected by EXCLUSIVE claim"
        )
    else:
        selected = (("uav_a", 0), ("uav_b", 1))
        if more_tasks:
            unassigned.append("target_k: no free UAV")

    assignments = tuple(
        FleetAssignment(
            assignment_id=f"assignment_{index:03d}_{assignment_index + 1}",
            uav_id=uav_id,
            target_alias=target_requests[target_index].target_alias,
            target_spec=target_requests[target_index].target_spec,
            search_region=target_requests[target_index].search_region,
            track_duration_s=target_requests[target_index].track_duration_s,
            priority=assignment_index,
            start_policy=FleetStartPolicy.PARALLEL,
        )
        for assignment_index, (uav_id, target_index) in enumerate(selected)
    )

    sample: dict[str, object] = {
        "sample_id": f"fleet_sample_{index:06d}",
        "input": request.to_dict(),
        "output_kind": (
            "fleet_plan_patch" if patch_output else "fleet_mission_plan"
        ),
        "metadata": {
            "language": "zh",
            "task_count": target_count,
            "uav_count": len(request.uav_inventory),
            "target_count": target_count,
            "difficulty": (
                "reassignment"
                if patch_output
                else "conflict"
                if unassigned
                else "explicit_assignment"
            ),
            "scenario_type": scenario,
        },
    }
    if patch_output:
        patch = FleetPlanPatch(
            fleet_mission_id=request.fleet_mission_id,
            base_fleet_plan_version=request.fleet_plan_version,
            new_fleet_plan_version=request.fleet_plan_version + 1,
            replacement_assignments=assignments,
            coordination_policy=request.coordination_policy,
            reason_codes=("ASSIGNMENT_FAILED", "target_j: UNASSIGNED"),
        )
        sample["fleet_plan_patch"] = patch.to_dict()
    else:
        plan = FleetMissionPlan(
            fleet_mission_id=request.fleet_mission_id,
            fleet_plan_version=request.fleet_plan_version,
            assignments=assignments,
            coordination_policy=request.coordination_policy,
            assumptions=request.assumptions,
            unassigned_requirements=tuple(unassigned),
        )
        sample["output"] = plan.to_dict()
    validate_sample(sample)
    return sample


class FleetDatasetGenerator:
    """Create a deterministic pilot set covering routing and conflict cases."""

    def generate(
        self,
        *,
        seed: int = 42,
    ) -> Mapping[str, tuple[Mapping[str, object], ...]]:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        samples_by_scenario = {
            scenario: _sample(index + 1, scenario, region_kind)
            for index, (scenario, region_kind) in enumerate(_SCENARIOS)
        }
        if set(samples_by_scenario) != FLEET_DATASET_SCENARIOS:
            raise RuntimeError(
                "generator scenario catalog diverged from the Fleet dataset contract"
            )
        result: dict[str, tuple[Mapping[str, object], ...]] = {}
        for split_index, split in enumerate(FLEET_DATASET_SPLITS):
            rows = [
                samples_by_scenario[scenario]
                for scenario in FLEET_DATASET_SPLIT_SCENARIOS[split]
            ]
            if len(rows) != FLEET_DATASET_SAMPLES_PER_SPLIT:
                raise RuntimeError(
                    f"split {split} must contain exactly "
                    f"{FLEET_DATASET_SAMPLES_PER_SPLIT} scenarios"
                )
            random.Random(seed + split_index).shuffle(rows)
            result[split] = tuple(rows)
        return result

    def write(
        self,
        output_dir: str | Path,
        *,
        seed: int = 42,
        overwrite: bool = False,
    ) -> dict[str, object]:
        destination = Path(output_dir).expanduser().resolve()
        if destination.exists() and not overwrite:
            raise FileExistsError(f"dataset output already exists: {destination}")
        if destination.exists() and overwrite:
            manifest_path = destination / "manifest.json"
            try:
                previous_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise FileExistsError(
                    "--overwrite only replaces a directory previously created by "
                    f"this generator: {destination}"
                ) from exc
            if (
                not isinstance(previous_manifest, Mapping)
                or previous_manifest.get("schema_version") not in {1, 2}
                or previous_manifest.get("generation_source")
                != "deterministic_gold_template"
            ):
                raise FileExistsError(
                    "--overwrite refused because the existing directory has no valid "
                    "Fleet Planner generator provenance"
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=".fleet-dataset-", dir=destination.parent)
        )
        backup_container: Path | None = None
        try:
            split_samples = self.generate(seed=seed)
            hashes: dict[str, str] = {}
            for split, samples in split_samples.items():
                path = temporary / f"{split}.jsonl"
                encoded = "".join(
                    json.dumps(
                        sample,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                    )
                    + "\n"
                    for sample in samples
                )
                path.write_text(encoded, encoding="utf-8")
                hashes[path.name] = sha256(encoded.encode("utf-8")).hexdigest()
            manifest = {
                "schema_version": FLEET_DATASET_MANIFEST_SCHEMA_VERSION,
                "seed": seed,
                "split_counts": {
                    name: len(rows) for name, rows in split_samples.items()
                },
                "sha256": hashes,
                "generation_source": FLEET_DATASET_GENERATION_SOURCE,
                "dataset_contract": FLEET_DATASET_CONTRACT,
            }
            (temporary / "manifest.json").write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if destination.exists():
                backup_container = Path(
                    tempfile.mkdtemp(
                        prefix=".fleet-dataset-backup-",
                        dir=destination.parent,
                    )
                )
                destination.replace(backup_container / "previous")
            try:
                temporary.replace(destination)
            except Exception:
                if backup_container is not None:
                    (backup_container / "previous").replace(destination)
                raise
            if backup_container is not None:
                shutil.rmtree(backup_container)
                backup_container = None
            return manifest
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            if backup_container is not None:
                previous = backup_container / "previous"
                if previous.exists() and not destination.exists():
                    previous.replace(destination)
                shutil.rmtree(backup_container, ignore_errors=True)


def generate_fleet_dataset(
    output_dir: str | Path,
    *,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, object]:
    return FleetDatasetGenerator().write(
        output_dir,
        seed=seed,
        overwrite=overwrite,
    )


__all__ = ["FleetDatasetGenerator", "generate_fleet_dataset"]
