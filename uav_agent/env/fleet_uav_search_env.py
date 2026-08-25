"""Multi-UAV environment coordinator with an Isaac-independent import boundary.

Isaac modules are imported only by :meth:`FleetUavSearchEnv.setup`, after a
caller has created ``SimulationApp``.  Inventory, Prim paths, assignment
isolation, cache semantics, and the tick barrier are therefore pure-Python
testable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import cos, isfinite, radians, sin
import re
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from common.ids import validate_routing_id, validate_uav_id
from configs.schema import AppConfig


def uav_prim_path(uav_id: str) -> str:
    return f"/World/UAVs/{_prim_segment(validate_uav_id(uav_id))}"


def camera_prim_path(uav_id: str) -> str:
    return f"{uav_prim_path(uav_id)}/Camera"


def target_prim_path(target_id: str) -> str:
    return f"/World/Targets/{_prim_segment(validate_routing_id(target_id, 'target_id'))}"


def _prim_segment(routing_id: str) -> str:
    """Map a routing ID to one valid USD Prim path segment."""

    return re.sub(r"[^A-Za-z0-9_]", "_", routing_id)


@dataclass(frozen=True)
class FleetPoseSnapshot:
    """All vehicle and target states captured at one simulation tick."""

    tick_index: int
    timestamp_s: float
    uav_states: Mapping[str, object]
    uav_velocities_mps: Mapping[str, object]
    target_states: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            isinstance(self.tick_index, bool)
            or not isinstance(self.tick_index, int)
            or self.tick_index < 0
        ):
            raise ValueError("tick_index must be non-negative")
        if (
            isinstance(self.timestamp_s, bool)
            or not isinstance(self.timestamp_s, (int, float))
            or not isfinite(float(self.timestamp_s))
            or self.timestamp_s < 0.0
        ):
            raise ValueError("timestamp_s must be a finite non-negative number")
        if set(self.uav_velocities_mps) != set(self.uav_states):
            raise ValueError("uav_velocities_mps keys must match uav_states")
        velocities: dict[str, np.ndarray] = {}
        for key, value in self.uav_velocities_mps.items():
            velocity = np.asarray(value, dtype=np.float64)
            if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
                raise ValueError(
                    f"uav_velocities_mps[{key!r}] must contain three finite values"
                )
            velocities[key] = velocity.copy()
            velocities[key].setflags(write=False)
        object.__setattr__(
            self,
            "uav_states",
            MappingProxyType({key: deepcopy(value) for key, value in self.uav_states.items()}),
        )
        object.__setattr__(
            self,
            "uav_velocities_mps",
            MappingProxyType(velocities),
        )
        object.__setattr__(
            self,
            "target_states",
            MappingProxyType(
                {key: deepcopy(value) for key, value in self.target_states.items()}
            ),
        )


class FleetUavSearchEnv:
    """Own a fleet scene, synchronized caches, and deterministic agent ticks."""

    _MAX_CAMERA_RENDER_CATCHUP_STEPS = 12

    def __init__(
        self,
        config: AppConfig,
        *,
        assignments: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig")
        self.config = config
        self.world: object | None = None
        self.scene: object | None = None
        self.uav_controllers: dict[str, object] = {}
        self.target_motions: dict[str, object] = {}
        self.camera_sensors: dict[str, object] = {}
        self.latest_agent_observations: dict[str, object] = {}
        self.latest_evaluator_frames: dict[tuple[str, str], object] = {}
        self._last_camera_timestamps_s: dict[str, float] = {}
        active_camera_frequencies = {
            config.camera_profiles[item.camera_profile].frequency_hz
            for item in config.uavs
        }
        if len(active_camera_frequencies) != 1:
            raise ValueError(
                "FleetUavSearchEnv v1 requires every active Camera to use the same frequency"
            )
        public_camera_frequency_hz = next(iter(active_camera_frequencies))
        self._camera_publish_period_s = 1.0 / public_camera_frequency_hz
        self._camera_sync_timeout_s = max(
            2.0,
            4.0 * self._camera_publish_period_s,
        )
        self._camera_sync_wait_started_s: float | None = None
        # A Camera reset/invalidation can leave renderer metadata one or more
        # frames behind World.current_time while its annotators refill.  This
        # explicit warm-up epoch may withhold such a stale frame; publication
        # always requires exact world/Camera timestamp agreement.
        self._camera_warmup_pending = False
        self._fleet_pose_snapshot: FleetPoseSnapshot | None = None
        self._tick_index = 0
        self._last_tick_order: tuple[str, ...] = ()
        self._assignments: dict[str, str] = {}
        # ``None`` means legacy/default routing.  An explicit empty mapping is
        # materially different: it is used by targetless V2 missions and must
        # not silently regain target/Oracle authority through truthiness.
        self.set_assignments(
            self._default_assignments() if assignments is None else assignments
        )

        self.uav_prim_paths = MappingProxyType(
            {item.id: uav_prim_path(item.id) for item in config.uavs}
        )
        self.camera_prim_paths = MappingProxyType(
            {item.id: camera_prim_path(item.id) for item in config.uavs}
        )
        self.target_prim_paths = MappingProxyType(
            {item.id: target_prim_path(item.id) for item in config.targets}
        )
        if len(set(self.uav_prim_paths.values())) != len(self.uav_prim_paths):
            raise ValueError("UAV Prim paths must be unique")
        if len(set(self.camera_prim_paths.values())) != len(self.camera_prim_paths):
            raise ValueError("Camera Prim paths must be unique")
        if len(set(self.target_prim_paths.values())) != len(self.target_prim_paths):
            raise ValueError("Target Prim paths must be unique")

    @property
    def uav_ids(self) -> tuple[str, ...]:
        return tuple(sorted(item.id for item in self.config.uavs))

    @property
    def target_ids(self) -> tuple[str, ...]:
        return tuple(sorted(item.id for item in self.config.targets))

    @property
    def assignments(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._assignments))

    @property
    def last_tick_order(self) -> tuple[str, ...]:
        return self._last_tick_order

    def set_assignments(self, assignments: Mapping[str, str]) -> None:
        if not isinstance(assignments, Mapping):
            raise TypeError("assignments must be a mapping of uav_id to target_id")
        known_uavs = {item.id for item in self.config.uavs}
        known_targets = {item.id for item in self.config.targets}
        normalized: dict[str, str] = {}
        for raw_uav_id, raw_target_id in assignments.items():
            uav_id = validate_uav_id(raw_uav_id)
            target_id = validate_routing_id(raw_target_id, "target_id")
            if uav_id not in known_uavs:
                raise ValueError(f"assignment references unknown UAV {uav_id!r}")
            if target_id not in known_targets:
                raise ValueError(f"assignment references unknown target {target_id!r}")
            normalized[uav_id] = target_id
        if (
            self.config.fleet.target_claim_policy == "EXCLUSIVE"
            and len(set(normalized.values())) != len(normalized)
        ):
            raise ValueError("EXCLUSIVE target assignments must not share a target")
        self._assignments = normalized
        self.latest_evaluator_frames.clear()

    def _default_assignments(self) -> Mapping[str, str]:
        if len(self.config.uavs) == 1 and len(self.config.targets) == 1:
            return {self.config.uavs[0].id: self.config.targets[0].id}
        return {}

    def setup(self) -> None:
        """Construct the shared Isaac World and every configured entity."""

        if self.world is not None:
            raise RuntimeError("environment has already been set up")
        # Delayed imports preserve the required SimulationApp startup order.
        from isaacsim.core.api import World

        from env.kinematic_uav import KinematicUAV, UAVState
        from env.moving_target import MovingTarget
        from env.scene import UavSearchScene

        simulation = self.config.simulation
        world = World(
            physics_dt=simulation.physics_dt_s,
            rendering_dt=simulation.rendering_dt_s,
            stage_units_in_meters=simulation.stage_units_in_meters,
        )
        # Publish ownership immediately so a failure anywhere in scene/sensor
        # construction can be cleaned by ``close``.  Keeping the scene's live
        # Camera mapping also preserves Cameras built before a later entity
        # fails, allowing them to be destroyed deterministically.
        self.world = world
        try:
            scene = UavSearchScene(world, self.config)
            self.scene = scene
            self.camera_sensors = scene.camera_sensors
            scene.build()
            world.reset()
            for sensor in scene.camera_sensors.values():
                sensor.enable_depth()
            # Isaac initializes render products and annotators sequentially.
            # Reset all Camera cadence clocks only after every depth annotator
            # is attached, otherwise equal-frequency Cameras can remain one
            # renderer tick out of phase for the entire mission.
            for sensor in scene.camera_sensors.values():
                sensor.synchronize_acquisition_clock()

            for uav in self.config.uavs:
                self.uav_controllers[uav.id] = KinematicUAV(
                    initial_state=UAVState(*uav.initial_position_xyz_m, yaw=0.0),
                    max_speed_mps=uav.max_speed_mps,
                    max_yaw_rate_rad_s=radians(uav.max_yaw_rate_deg_s),
                    pose_writer=(
                        lambda position, orientation, uav_id=uav.id: (
                            scene.set_uav_pose_for(uav_id, position, orientation)
                        )
                    ),
                )
            for target in self.config.targets:
                motion = target.motion
                half_extent = tuple(
                    value / 2.0 for value in target.appearance.size_xyz_m
                )
                self.target_motions[target.id] = MovingTarget(
                    mode=motion.mode,
                    initial_position_xyz_m=self._initial_target_position(target.id),
                    bounds_min_xyz_m=motion.region.min_xyz_m,
                    bounds_max_xyz_m=motion.region.max_xyz_m,
                    speed_mps=motion.speed_mps,
                    max_speed_mps=target.max_speed_mps,
                    direction_change_interval_s=motion.direction_change_interval_s,
                    seed=motion.seed,
                    initial_heading_rad=radians(motion.initial_heading_deg),
                    pose_writer=(
                        lambda position, orientation, target_id=target.id: (
                            scene.set_target_pose_for(
                                target_id, position, orientation
                            )
                        )
                    ),
                    obstacle_registry=scene.obstacle_registry,
                    target_half_extent_xyz_m=half_extent,
                )
            self._invalidate_caches()
        except BaseException as setup_error:
            try:
                self.close()
            except Exception as cleanup_error:
                setup_error.add_note(
                    "Fleet environment cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise

    def get_agent_view(self, uav_id: str) -> object:
        from env.observation_types import AgentView

        controller = self._require_uav_controller(uav_id)
        return AgentView(
            observe=lambda: self.get_agent_observation(uav_id),
            move_toward=controller.move_toward,
            set_velocity=controller.set_velocity,
            rotate_yaw=controller.rotate_yaw,
            stop=controller.stop,
            distance_to_goal=controller.distance_to_goal,
            heading_error=controller.heading_error,
            goal_reached=controller.goal_reached,
        )

    def make_skill_context(
        self,
        uav_id: str,
        clock: object,
        perception: object | None = None,
    ) -> object:
        from skills.types import SkillContext
        from perception.runtime import PerceptionCapability

        normalized = self._known_uav_id(uav_id)
        if (
            perception is not None
            and getattr(perception, "capability", None)
            is PerceptionCapability.PRIVILEGED_ORACLE
        ):
            # Runtime policy wrappers (for example GuardedPerceptionBackend)
            # intentionally expose the capability while retaining the bound
            # Oracle as ``backend``.  Validate routing against that producer,
            # but inject the wrapper itself into SkillContext.
            routed_perception = getattr(perception, "backend", perception)
            assigned_target = self._assignments.get(normalized)
            if (
                getattr(routed_perception, "uav_id", None) != normalized
                or getattr(routed_perception, "target_id", None) != assigned_target
            ):
                raise PermissionError(
                    "OraclePerception must be bound to this UAV's assigned target"
                )
        return SkillContext(
            uav=self._require_uav_controller(normalized),
            camera=self._require_camera_sensor(normalized),
            perception=perception,
            clock=clock,
            uav_id=normalized,
        )

    def make_oracle_perception(self, uav_id: str) -> object:
        """Create an evaluator-only Oracle bound to exactly one assignment."""

        from perception.oracle import OraclePerception

        normalized = self._known_uav_id(uav_id)
        try:
            target_id = self._assignments[normalized]
        except KeyError as exc:
            raise RuntimeError(
                f"UAV {normalized!r} has no target assignment for Oracle evaluation"
            ) from exc
        return OraclePerception(uav_id=normalized, target_id=target_id)

    def get_agent_observation(self, uav_id: str) -> object:
        normalized = self._known_uav_id(uav_id)
        try:
            return deepcopy(self.latest_agent_observations[normalized])
        except KeyError as exc:
            raise RuntimeError(
                f"no synchronized Camera observation is available for {normalized!r}"
            ) from exc

    def get_skill_observation(
        self,
        uav_id: str,
        *,
        include_oracle: bool = False,
    ) -> object:
        """Adapt only one UAV's synchronized agent-facing Camera snapshot."""

        if not isinstance(include_oracle, bool):
            raise TypeError("include_oracle must be bool")
        if include_oracle:
            raise PermissionError(
                "Fleet Oracle data is assignment-scoped; use "
                "get_evaluator_frame(uav_id, target_id) with OraclePerception"
            )
        from skills.types import Observation

        normalized = self._known_uav_id(uav_id)
        agent = self.get_agent_observation(normalized)
        return Observation(
            uav_id=normalized,
            timestamp=float(agent.camera_timestamp_s),
            uav_pose=agent.uav_state,
            uav_velocity=np.asarray(agent.uav_velocity_mps).copy(),
            camera_rgb=np.asarray(agent.rgb).copy(),
            camera_position_m=np.asarray(agent.camera_position_m).copy(),
            camera_orientation_wxyz=np.asarray(
                agent.camera_orientation_wxyz
            ).copy(),
        )

    def get_target_perception_input(self, uav_id: str) -> object:
        """Return one atomic, production-safe RGB-D perception input.

        Both values are projected from the same published Fleet Camera batch.
        This method deliberately never touches the evaluator-frame cache.
        """

        from perception.runtime_bridge import SynchronizedTargetPerceptionInput
        from skills.types import Observation

        normalized = self._known_uav_id(uav_id)
        agent = self.get_agent_observation(normalized)
        camera_sample = getattr(agent, "camera_sample", None)
        if camera_sample is None:
            raise RuntimeError(
                f"no synchronized RGB-D CameraSample is available for {normalized!r}"
            )
        base = Observation(
            uav_id=normalized,
            timestamp=float(agent.camera_timestamp_s),
            uav_pose=agent.uav_state,
            uav_velocity=np.asarray(agent.uav_velocity_mps).copy(),
            camera_rgb=np.asarray(camera_sample.rgb).copy(),
            camera_position_m=np.asarray(
                camera_sample.camera_position_world_m
            ).copy(),
            camera_orientation_wxyz=np.asarray(
                camera_sample.camera_orientation_world_wxyz
            ).copy(),
        )
        return SynchronizedTargetPerceptionInput(
            base_observation=base,
            camera_sample=camera_sample,
        )

    def get_evaluator_frame(self, uav_id: str, target_id: str) -> object:
        normalized_uav = self._known_uav_id(uav_id)
        normalized_target = self._known_target_id(target_id)
        assigned = self._assignments.get(normalized_uav)
        if assigned != normalized_target:
            raise PermissionError(
                f"Oracle frame for UAV {normalized_uav!r} is restricted to its "
                f"assigned target {assigned!r}"
            )
        try:
            return deepcopy(
                self.latest_evaluator_frames[(normalized_uav, normalized_target)]
            )
        except KeyError as exc:
            raise RuntimeError("no synchronized evaluator frame is available") from exc

    def get_target_state(self, target_id: str) -> object:
        """Return evaluator/debug truth; never inject this API into an Agent."""

        return deepcopy(self._require_target_motion(target_id).get_pose())

    def get_fleet_pose_snapshot(self) -> FleetPoseSnapshot:
        if self._fleet_pose_snapshot is None:
            raise RuntimeError("no fleet pose snapshot is available")
        snapshot = self._fleet_pose_snapshot
        return FleetPoseSnapshot(
            tick_index=snapshot.tick_index,
            timestamp_s=snapshot.timestamp_s,
            uav_states=snapshot.uav_states,
            uav_velocities_mps=snapshot.uav_velocities_mps,
            target_states=snapshot.target_states,
        )

    def tick_uavs(
        self,
        agent_ticks: Mapping[str, Callable[[], object]],
        *,
        global_safety_check: Callable[[FleetPoseSnapshot], object] | None = None,
    ) -> Mapping[str, object]:
        """Advance one shared tick, then invoke every agent in sorted ID order."""

        if not isinstance(agent_ticks, Mapping):
            raise TypeError("agent_ticks must be a mapping")
        supplied = set(agent_ticks)
        expected = set(self.uav_ids)
        if supplied != expected:
            missing = sorted(expected - supplied)
            unknown = sorted(supplied - expected)
            raise ValueError(
                "agent_ticks must cover every UAV; "
                f"missing={missing}, unknown={unknown}"
            )
        if any(not callable(callback) for callback in agent_ticks.values()):
            raise TypeError("every agent_ticks value must be callable")
        if global_safety_check is not None and not callable(global_safety_check):
            raise TypeError("global_safety_check must be callable or None")
        # Never retain a successful prior barrier's order if this barrier or
        # one of its callbacks fails before the complete sorted batch finishes.
        self._last_tick_order = ()
        snapshot, _ = self._advance_snapshot_barrier()
        if global_safety_check is not None:
            global_safety_check(snapshot)
        results: dict[str, object] = {}
        order = self.uav_ids
        for uav_id in order:
            results[uav_id] = agent_ticks[uav_id]()
        self._last_tick_order = order
        return MappingProxyType(results)

    def step(self) -> bool:
        """Advance one barrier tick without ticking MissionAgents."""

        _, new_observation_count = self._advance_snapshot_barrier()
        self._last_tick_order = ()
        return new_observation_count > 0

    def _advance_snapshot_barrier(self) -> tuple[FleetPoseSnapshot, int]:
        world = self._require_world()
        dt_s = self.config.simulation.physics_dt_s
        # Kinematic entities write their next pose directly to USD.  Every
        # writer must finish before the one render that produces this tick's
        # Camera frames; rendering first would pair old pixels with new poses.
        for target_id in self.target_ids:
            self._require_target_motion(target_id).step(dt_s)
        for uav_id in self.uav_ids:
            self._require_uav_controller(uav_id).step(dt_s)
        world.step(render=True)
        self._tick_index += 1
        simulation_timestamp_s = self._simulation_timestamp_s(world)
        # Capture every state from the now-rendered USD tick before touching
        # any Camera cache; no MissionAgent is allowed to run inside this
        # barrier, so these values and the following frames share one state.
        uav_states = {
            uav_id: self._require_uav_controller(uav_id).get_pose()
            for uav_id in self.uav_ids
        }
        uav_velocities_mps = {
            uav_id: self._require_uav_controller(uav_id).get_velocity()
            for uav_id in self.uav_ids
        }
        target_states = {
            target_id: self._require_target_motion(target_id).get_pose()
            for target_id in self.target_ids
        }
        samples = self._capture_camera_batch(simulation_timestamp_s)
        # The bounded frozen-world drain makes any published Camera batch's
        # timestamp equal to this authoritative simulation pose barrier.
        snapshot = FleetPoseSnapshot(
            tick_index=self._tick_index,
            timestamp_s=simulation_timestamp_s,
            uav_states=uav_states,
            uav_velocities_mps=uav_velocities_mps,
            target_states=target_states,
        )
        new_observation_count = self._publish_observation_batch(snapshot, samples)
        # Publish the pose barrier only after the complete Camera/evaluator
        # batch has been constructed.  A projection or sensor failure must not
        # expose a new pose snapshot alongside a partial old/new Camera batch.
        self._fleet_pose_snapshot = snapshot
        return snapshot, new_observation_count

    def _refresh_all_observations(self, snapshot: FleetPoseSnapshot) -> int:
        """Capture and atomically publish one batch for an existing barrier.

        This compatibility hook is used by the singleton adapter.  It applies
        the same renderer-ID rules as the main Fleet barrier and never rewrites
        a Camera frame's renderer-provided timestamp.
        """

        samples = self._capture_camera_batch(snapshot.timestamp_s)
        return self._publish_observation_batch(snapshot, samples)

    def _capture_camera_batch(
        self,
        simulation_timestamp_s: float,
    ) -> dict[str, object]:
        """Return a complete new same-render Camera batch or no batch.

        Isaac Cameras acquire every renderer frame.  This method atomically
        downsamples that stream to CameraConfig.frequency_hz, drains temporary
        per-RenderProduct skew without advancing physics, and publishes only a
        complete batch at the current simulation timestamp.  Warm-up uses a
        bounded watchdog and no renderer timestamp is ever relabelled.
        """

        from env.camera_types import CameraFrameNotReady

        rendering_dt_s = self.config.simulation.rendering_dt_s
        tolerance_s = max(1e-7, rendering_dt_s * 1e-5)

        # Public delivery cadence belongs to the simulation clock, not to the
        # asynchronously completed Camera cache.  Basing this decision on a
        # three-tick-stale Camera timestamp silently reduces a configured
        # 10 Hz stream to roughly 6--7 Hz.
        if self._last_camera_timestamps_s:
            published_timestamps = set(self._last_camera_timestamps_s.values())
            if len(published_timestamps) != 1:
                raise RuntimeError(
                    "Fleet Camera publication history has inconsistent timestamps"
                )
            last_timestamp_s = next(iter(published_timestamps))
            if simulation_timestamp_s < last_timestamp_s - tolerance_s:
                raise RuntimeError(
                    "Fleet simulation timestamp moved behind the last published "
                    "Camera frame"
                )
            if (
                simulation_timestamp_s - last_timestamp_s + tolerance_s
                < self._camera_publish_period_s
            ):
                self._camera_sync_wait_started_s = None
                return {}

        world = self.world
        render = getattr(world, "render", None)
        can_drain_renderer = callable(render)
        previous_metadata_timestamps: dict[str, float] = {}
        metadata: dict[str, tuple[float, tuple[int, int] | None]] | None = None
        not_ready_reason: str | None = None

        # Isaac's RenderProducts complete independently.  Do not reject a
        # transient cross-Camera skew before giving the frozen-world drain a
        # chance to catch every product up to the same simulation timestamp.
        for attempt in range(self._MAX_CAMERA_RENDER_CATCHUP_STEPS + 1):
            candidate: dict[str, tuple[float, tuple[int, int] | None]] = {}
            not_ready_reason = None
            for uav_id in self.uav_ids:
                sensor = self._require_camera_sensor(uav_id)
                get_metadata = getattr(sensor, "get_render_metadata", None)
                if not callable(get_metadata):
                    raise TypeError(
                        "every Fleet Camera must provide get_render_metadata()"
                    )
                try:
                    timestamp_s, render_frame_id = get_metadata()
                except CameraFrameNotReady as exc:
                    not_ready_reason = f"{uav_id}: {exc}"
                    break
                timestamp = float(timestamp_s)
                if timestamp > simulation_timestamp_s + tolerance_s:
                    self._validate_camera_world_lag(
                        timestamp,
                        simulation_timestamp_s,
                    )
                previous_timestamp_s = previous_metadata_timestamps.get(uav_id)
                if (
                    previous_timestamp_s is not None
                    and timestamp < previous_timestamp_s - tolerance_s
                ):
                    raise RuntimeError(
                        "Fleet Camera renderer timestamp moved backwards while "
                        "draining the render pipeline: "
                        f"uav={uav_id!r}, previous={previous_timestamp_s!r}, "
                        f"current={timestamp!r}"
                    )
                published_timestamp_s = self._last_camera_timestamps_s.get(uav_id)
                if (
                    published_timestamp_s is not None
                    and timestamp < published_timestamp_s - tolerance_s
                ):
                    raise RuntimeError(
                        "Fleet Camera renderer timestamp moved behind its last "
                        f"publication: uav={uav_id!r}, "
                        f"last={published_timestamp_s!r}, current={timestamp!r}"
                    )
                previous_metadata_timestamps[uav_id] = timestamp
                candidate[uav_id] = (timestamp, render_frame_id)

            if not_ready_reason is None and all(
                abs(simulation_timestamp_s - value[0]) <= tolerance_s
                for value in candidate.values()
            ):
                metadata = candidate
                break

            if attempt >= self._MAX_CAMERA_RENDER_CATCHUP_STEPS or not can_drain_renderer:
                if self._camera_warmup_pending or not_ready_reason is not None:
                    reason = not_ready_reason or (
                        "reset Camera frame is still stale: "
                        f"world={simulation_timestamp_s!r}, "
                        f"camera_by_uav="
                        f"{ {key: value[0] for key, value in candidate.items()}!r}"
                    )
                    return self._withhold_camera_batch(
                        simulation_timestamp_s,
                        reason,
                    )
                if candidate and len({value[0] for value in candidate.values()}) != 1:
                    self._validate_same_camera_render(candidate, source="metadata")
                stale_timestamp_s = (
                    min(value[0] for value in candidate.values())
                    if candidate
                    else float("nan")
                )
                self._validate_camera_world_lag(
                    stale_timestamp_s,
                    simulation_timestamp_s,
                )
                raise RuntimeError("Fleet Camera synchronization failed")  # pragma: no cover

            assert callable(render)
            render()
            rendered_world_timestamp_s = self._simulation_timestamp_s(world)
            if abs(rendered_world_timestamp_s - simulation_timestamp_s) > tolerance_s:
                raise RuntimeError(
                    "World.render() advanced simulation time while synchronizing "
                    "the Fleet Camera barrier"
                )

        if metadata is None:  # pragma: no cover - loop exits above or returns/raises
            raise RuntimeError("Fleet Camera synchronization produced no metadata")
        camera_timestamp_s = self._validate_same_camera_render(
            metadata,
            source="metadata",
        )
        self._validate_camera_world_lag(
            camera_timestamp_s,
            simulation_timestamp_s,
        )

        # The frame is due and satisfies the renderer barrier; copy its RGB-D
        # payload atomically only now.
        samples: dict[str, object] = {}
        for uav_id in self.uav_ids:
            try:
                sample = self._require_camera_sensor(uav_id).get_sample()
            except CameraFrameNotReady as exc:
                return self._withhold_camera_batch(
                    simulation_timestamp_s,
                    f"{uav_id}: {exc}",
                )
            samples[uav_id] = sample

        sample_metadata = {
            uav_id: (
                float(sample.timestamp_s),
                getattr(sample, "render_frame_id", None),
            )
            for uav_id, sample in samples.items()
        }
        camera_timestamp_s = self._validate_same_camera_render(
            sample_metadata,
            source="samples",
        )
        if sample_metadata != metadata:
            raise RuntimeError(
                "Fleet Camera renderer metadata changed while copying one atomic batch"
            )
        self._validate_camera_world_lag(
            camera_timestamp_s,
            simulation_timestamp_s,
        )

        new_sample_flags = {
            uav_id: self._last_camera_timestamps_s.get(uav_id)
            != float(sample.timestamp_s)
            for uav_id, sample in samples.items()
        }
        if not any(new_sample_flags.values()):
            self._camera_sync_wait_started_s = None
            return {}
        if not all(new_sample_flags.values()):
            return self._withhold_camera_batch(
                simulation_timestamp_s,
                "only part of the Fleet Camera batch is new",
            )
        self._camera_sync_wait_started_s = None
        self._camera_warmup_pending = False
        return samples

    def _withhold_camera_batch(
        self,
        simulation_timestamp_s: float,
        reason: str,
    ) -> dict[str, object]:
        if self._camera_sync_wait_started_s is None:
            self._camera_sync_wait_started_s = float(simulation_timestamp_s)
            return {}
        waited_s = float(simulation_timestamp_s) - self._camera_sync_wait_started_s
        if waited_s >= self._camera_sync_timeout_s:
            raise RuntimeError(
                "Fleet Camera synchronization timed out after "
                f"{waited_s:.3f}s: {reason}"
            )
        return {}

    def _validate_same_camera_render(
        self,
        metadata: Mapping[str, tuple[float, tuple[int, int] | None]],
        *,
        source: str,
    ) -> float:
        if not metadata:
            raise RuntimeError("Fleet Camera batch cannot be empty")
        timestamps = tuple(value[0] for value in metadata.values())
        tolerance_s = max(
            1e-7,
            self.config.simulation.rendering_dt_s * 1e-5,
        )
        if max(timestamps) - min(timestamps) > tolerance_s:
            raise RuntimeError(
                f"fleet Camera {source} are not from the same render timestamp: "
                f"{sorted(timestamps)!r}"
            )
        render_frame_ids = tuple(value[1] for value in metadata.values())
        if any(value is None for value in render_frame_ids):
            if not all(value is None for value in render_frame_ids):
                raise RuntimeError(
                    f"fleet Camera {source} have only partial renderer frame IDs"
                )
        elif len(set(render_frame_ids)) != 1:
            raise RuntimeError(
                f"fleet Camera {source} are not from the same renderer frame ID: "
                f"{sorted(render_frame_ids)!r}"
            )
        return timestamps[0]

    def _validate_camera_world_lag(
        self,
        camera_timestamp_s: float,
        simulation_timestamp_s: float,
    ) -> None:
        lag_s = float(simulation_timestamp_s) - float(camera_timestamp_s)
        tolerance_s = max(
            1e-7,
            self.config.simulation.rendering_dt_s * 1e-5,
        )
        if abs(lag_s) > tolerance_s:
            raise RuntimeError(
                "Fleet Camera timestamp is outside the current renderer barrier: "
                f"world={simulation_timestamp_s!r}, camera={camera_timestamp_s!r}, "
                f"lag={lag_s!r}, allowed_absolute_error={tolerance_s!r}"
            )

    def _publish_observation_batch(
        self,
        snapshot: FleetPoseSnapshot,
        samples: Mapping[str, object],
    ) -> int:
        from env.observation_types import AgentObservation, EvaluatorFrame

        if not samples:
            return 0
        if set(samples) != set(self.uav_ids):
            raise RuntimeError("Camera observation batch must cover every UAV")
        timestamps = {float(sample.timestamp_s) for sample in samples.values()}
        timestamp_tolerance_s = max(
            1e-7,
            self.config.simulation.rendering_dt_s * 1e-5,
        )
        if any(
            abs(timestamp - float(snapshot.timestamp_s)) > timestamp_tolerance_s
            for timestamp in timestamps
        ):
            raise RuntimeError(
                "Camera observation batch timestamp does not match FleetPoseSnapshot"
            )

        scene = self._require_scene()
        pending_agent_observations: dict[str, object] = {}
        pending_evaluator_frames: dict[tuple[str, str], object] = {}
        pending_timestamps: dict[str, float] = {}
        for uav_id, sample in samples.items():
            observation = AgentObservation(
                rgb=np.asarray(sample.rgb).copy(),
                uav_state=deepcopy(snapshot.uav_states[uav_id]),
                uav_velocity_mps=np.asarray(
                    snapshot.uav_velocities_mps[uav_id]
                ).copy(),
                camera_position_m=np.asarray(sample.camera_position_world_m).copy(),
                camera_orientation_wxyz=np.asarray(
                    sample.camera_orientation_world_wxyz
                ).copy(),
                camera_timestamp_s=float(sample.timestamp_s),
                camera_sample=sample,
            )
            pending_agent_observations[uav_id] = observation
            target_id = self._assignments.get(uav_id)
            if target_id is not None:
                target_state = snapshot.target_states[target_id]
                target_position = np.asarray(
                    [target_state.x, target_state.y, target_state.z], dtype=np.float64
                )
                target_motion = self._require_target_motion(target_id)
                pending_evaluator_frames[(uav_id, target_id)] = EvaluatorFrame(
                    observation=observation,
                    target_position_m=target_position,
                    target_orientation_wxyz=_yaw_quaternion(target_state.yaw),
                    target_state=deepcopy(target_state),
                    target_velocity_mps=target_motion.get_velocity(),
                    target_projection=scene.world_to_image_for(uav_id, target_position),
                )
            pending_timestamps[uav_id] = float(sample.timestamp_s)
        self.latest_agent_observations.update(pending_agent_observations)
        self.latest_evaluator_frames.update(pending_evaluator_frames)
        self._last_camera_timestamps_s.update(pending_timestamps)
        return len(samples)

    @staticmethod
    def _simulation_timestamp_s(world: object) -> float:
        """Read the authoritative simulation time immediately after render."""

        timestamp = getattr(world, "current_time", None)
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float, np.number))
            or not isfinite(float(timestamp))
            or float(timestamp) < 0.0
        ):
            raise RuntimeError(
                "World.current_time must be a finite non-negative simulation timestamp"
            )
        return float(timestamp)

    def reset(self, *, target_seeds: Mapping[str, int] | None = None) -> None:
        world = self._require_world()
        supplied_seeds = {} if target_seeds is None else dict(target_seeds)
        unknown = set(supplied_seeds) - set(self.target_ids)
        if unknown:
            raise ValueError(f"target_seeds contains unknown targets: {sorted(unknown)}")
        for target_id, seed in supplied_seeds.items():
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError(
                    f"target_seeds[{target_id!r}] must be a non-negative integer"
                )
        self._invalidate_caches()
        world.reset()
        for uav in self.config.uavs:
            self._require_uav_controller(uav.id).set_pose(
                *uav.initial_position_xyz_m, yaw=0.0
            )
        for target in self.config.targets:
            self._require_target_motion(target.id).reset(
                position_m=self._initial_target_position(target.id),
                seed=supplied_seeds.get(target.id),
            )
        self._invalidate_caches()

    def _invalidate_caches(self) -> None:
        self._invalidate_observation_caches()
        self._fleet_pose_snapshot = None
        self._tick_index = 0
        self._last_tick_order = ()

    def _invalidate_observation_caches(self) -> None:
        """Drop synchronized Camera/evaluator data without rewinding fleet time."""

        self.latest_agent_observations.clear()
        self.latest_evaluator_frames.clear()
        self._last_camera_timestamps_s.clear()
        self._camera_sync_wait_started_s = None
        self._camera_warmup_pending = True
        for sensor in self.camera_sensors.values():
            sensor.invalidate_frame()

    def close(self) -> None:
        if self.world is None:
            return
        world = self.world
        first_error: Exception | None = None
        try:
            world.stop()
        except Exception as exc:
            first_error = exc
        for sensor in tuple(self.camera_sensors.values()):
            try:
                sensor.destroy()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        try:
            type(world).clear_instance()
        except Exception as exc:
            if first_error is None:
                first_error = exc
        finally:
            self.world = None
            self.scene = None
            self.uav_controllers.clear()
            self.target_motions.clear()
            # A destroyed Camera cannot service ``invalidate_frame``.  Clear
            # the mapping first, then reset all pure-Python cache metadata.
            self.camera_sensors.clear()
            self._invalidate_caches()
        if first_error is not None:
            raise first_error

    def _initial_target_position(self, target_id: str) -> np.ndarray:
        target = next(item for item in self.config.targets if item.id == target_id)
        return (
            np.asarray(target.initial_region.min_xyz_m, dtype=np.float64)
            + np.asarray(target.initial_region.max_xyz_m, dtype=np.float64)
        ) / 2.0

    def _known_uav_id(self, uav_id: str) -> str:
        normalized = validate_uav_id(uav_id)
        if normalized not in self.uav_ids:
            raise KeyError(f"unknown uav_id: {normalized}")
        return normalized

    def _known_target_id(self, target_id: str) -> str:
        normalized = validate_routing_id(target_id, "target_id")
        if normalized not in self.target_ids:
            raise KeyError(f"unknown target_id: {normalized}")
        return normalized

    def _require_world(self) -> object:
        if self.world is None:
            raise RuntimeError("environment must be set up before ticking")
        return self.world

    def _require_scene(self) -> object:
        if self.scene is None:
            raise RuntimeError("environment must be set up before reading scene data")
        return self.scene

    def _require_uav_controller(self, uav_id: str) -> object:
        normalized = self._known_uav_id(uav_id)
        try:
            return self.uav_controllers[normalized]
        except KeyError as exc:
            raise RuntimeError("environment must be set up before controlling UAVs") from exc

    def _require_target_motion(self, target_id: str) -> object:
        normalized = self._known_target_id(target_id)
        try:
            return self.target_motions[normalized]
        except KeyError as exc:
            raise RuntimeError("environment must be set up before reading targets") from exc

    def _require_camera_sensor(self, uav_id: str) -> object:
        normalized = self._known_uav_id(uav_id)
        try:
            return self.camera_sensors[normalized]
        except KeyError as exc:
            raise RuntimeError("environment must be set up before reading Cameras") from exc


def _yaw_quaternion(yaw_rad: float) -> np.ndarray:
    return np.asarray([cos(yaw_rad / 2.0), 0.0, 0.0, sin(yaw_rad / 2.0)])


__all__ = [
    "FleetPoseSnapshot",
    "FleetUavSearchEnv",
    "camera_prim_path",
    "target_prim_path",
    "uav_prim_path",
]
