"""Fail-closed factory for independent target-perception backends."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace

from configs.schema import AppConfig, TargetPerceptionConfig
from perception.base import PerceptionBackend
from perception.mode import ResolvedTargetPerceptionMode, TargetPerceptionMode
from perception.runtime import (
    GuardedPerceptionBackend,
    PerceptionBoundaryError,
    PerceptionRuntimeProfile,
)
from perception.vision_backend import (
    DisabledTargetPerceptionBackend,
    VisionPerceptionBackend,
)
from skills.types import SkillName
from runtime.frame_store import FrameStore


class TargetPerceptionConfigurationError(PerceptionBoundaryError):
    """Raised before runtime when profile/backend switches conflict."""


class TargetPerceptionUnavailableError(TargetPerceptionConfigurationError):
    """Raised when a target-dependent plan selected the disabled backend."""


_TARGET_SKILLS = frozenset(
    {SkillName.SEARCH, SkillName.INSPECT, SkillName.TRACK, SkillName.REACQUIRE}
)


def build_target_perception_backend(
    config: AppConfig,
    *,
    runtime_profile: PerceptionRuntimeProfile,
    acknowledge_privileged_oracle: bool,
    uav_id: str,
) -> PerceptionBackend:
    """Construct exactly the configured backend without any fallback path."""

    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    if not isinstance(runtime_profile, PerceptionRuntimeProfile):
        raise TypeError("runtime_profile must be a PerceptionRuntimeProfile")
    if not isinstance(acknowledge_privileged_oracle, bool):
        raise TypeError("acknowledge_privileged_oracle must be bool")
    backend_name = config.target_perception.backend

    if runtime_profile is PerceptionRuntimeProfile.ORACLE_EVALUATION:
        # Keep the privileged implementation out of ordinary production
        # import paths as well as out of constructed YOLO runtime objects.
        from perception.oracle import OraclePerception

        if backend_name != "oracle_evaluation":
            raise TargetPerceptionConfigurationError(
                "ORACLE_EVALUATION requires target_perception.backend="
                "oracle_evaluation"
            )
        if not acknowledge_privileged_oracle:
            raise TargetPerceptionConfigurationError(
                "Oracle target perception requires explicit acknowledgement"
            )
        return GuardedPerceptionBackend(
            OraclePerception(uav_id=uav_id, target_id="target"),
            profile=runtime_profile,
            acknowledge_privileged_oracle=True,
        )

    if acknowledge_privileged_oracle:
        raise TargetPerceptionConfigurationError(
            "Oracle acknowledgement is invalid in production"
        )
    if backend_name == "oracle_evaluation":
        raise TargetPerceptionConfigurationError(
            "production profile forbids target_perception.backend=oracle_evaluation"
        )
    if backend_name == "ultralytics_service":
        return VisionPerceptionBackend(config.target_perception, uav_id=uav_id)
    if backend_name == "disabled":
        return DisabledTargetPerceptionBackend(uav_id=uav_id)
    # AppConfig can be constructed directly rather than through load_config;
    # repeat the closed-set check at this trust boundary.
    raise TargetPerceptionConfigurationError(
        f"unknown target_perception backend: {backend_name!r}"
    )


def validate_target_perception_preflight(
    backend_name: str,
    skills: Iterable[SkillName | str],
) -> None:
    """Reject target-dependent plans when target perception is disabled."""

    if backend_name != "disabled":
        return
    required: list[str] = []
    for value in skills:
        try:
            skill = value if isinstance(value, SkillName) else SkillName(str(value).upper())
        except ValueError as exc:
            raise ValueError(f"unknown Skill in target-perception preflight: {value!r}") from exc
        if skill in _TARGET_SKILLS:
            required.append(skill.value)
    if required:
        raise TargetPerceptionUnavailableError(
            "target_perception.backend=disabled cannot execute target Skills: "
            + ", ".join(required)
            + "; select --target-perception-mode oracle with "
            "--acknowledge-privileged-oracle, or select "
            "--target-perception-mode yolo with a matching production config"
        )


def build_target_candidate_resolver(
    config: TargetPerceptionConfig,
    *,
    frame_store: FrameStore,
) -> object:
    """Build the configured production geometry resolver without fallback.

    This is the single construction boundary shared by Fleet, the standalone
    visual mission, and :class:`TargetPerceptionCoordinator`.  In particular,
    selecting ``temporal_ray_depth`` can never silently construct the
    deterministic resolver as the primary implementation.
    """

    if not isinstance(config, TargetPerceptionConfig):
        raise TypeError("config must be a TargetPerceptionConfig")
    if not isinstance(frame_store, FrameStore):
        raise TypeError("frame_store must be a FrameStore")
    geometry = config.geometry
    if geometry.mode == "isaac_depth":
        from perception.depth_geometry import DepthCandidateResolver

        return DepthCandidateResolver(
            frame_store,
            sampling_strategy=geometry.depth_anchor,
            patch_radius_px=geometry.depth_patch_radius_px,
            min_depth_m=geometry.min_depth_m,
            max_depth_m=geometry.max_depth_m,
        )
    if geometry.mode == "temporal_ray_depth":
        from perception.temporal_ray_depth import TemporalRayDepthResolver

        temporal = geometry.temporal_ray_depth
        if temporal.checkpoint_path is None or temporal.expected_sha256 is None:
            raise TargetPerceptionConfigurationError(
                "temporal_ray_depth artifact identity is incomplete"
            )
        return TemporalRayDepthResolver(
            frame_store,
            checkpoint_path=temporal.checkpoint_path,
            expected_sha256=temporal.expected_sha256,
            manifest_path=temporal.manifest_path,
            history_size=temporal.history_size,
            max_history_age_s=temporal.max_history_age_s,
            roi_size_px=temporal.roi_size_px,
            use_rgb=temporal.use_rgb,
            use_depth=temporal.use_depth,
            deterministic_fallback=temporal.deterministic_fallback,
            device=temporal.device,
            min_depth_m=geometry.min_depth_m,
            max_depth_m=geometry.max_depth_m,
            sampling_strategy=geometry.depth_anchor,
            patch_radius_px=geometry.depth_patch_radius_px,
        )
    raise TargetPerceptionConfigurationError(
        "ultralytics_service requires isaac_depth or temporal_ray_depth geometry"
    )


def preflight_temporal_ray_depth(config: AppConfig) -> Mapping[str, object] | None:
    """Validate and dry-run a temporal artifact before the first Isaac import."""

    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    geometry = config.target_perception.geometry
    if geometry.mode != "temporal_ray_depth":
        return None
    temporal = geometry.temporal_ray_depth
    if temporal.checkpoint_path is None or temporal.expected_sha256 is None:
        raise TargetPerceptionConfigurationError(
            "temporal_ray_depth requires checkpoint_path and expected_sha256"
        )
    try:
        resolver = build_target_candidate_resolver(
            config.target_perception,
            frame_store=FrameStore(
                max_frames=temporal.history_size + 2,
                max_bytes=1024,
                max_age_s=temporal.max_history_age_s,
            ),
        )
    except Exception as exc:
        raise TargetPerceptionConfigurationError(
            "temporal ray-depth preflight failed closed before Isaac import: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    artifact = resolver.artifact_info
    return {
        "checkpoint_path": str(artifact.checkpoint_path),
        "checkpoint_sha256": artifact.checkpoint_sha256,
        "manifest_path": str(artifact.manifest_path),
        "history_size": artifact.history_size,
        "max_history_age_s": artifact.max_history_age_s,
        "roi_size_px": artifact.roi_size_px,
        "camera_convention": "camera_optical_x_right_y_down_z_forward",
        "coordinate_convention": "world_flu_x_forward_y_left_z_up",
        "training_stage": "yolo_deployment",
        "promotion": "passed",
        "dry_run": "passed",
    }


def yolo_service_url_for_uav(config: AppConfig, uav_id: str) -> str:
    """Resolve one isolated worker URL without cross-UAV fallback."""

    from common.ids import validate_uav_id

    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    normalized = validate_uav_id(uav_id)
    service = config.target_perception.yolo_service
    if service.per_uav_urls:
        try:
            return service.per_uav_urls[normalized]
        except KeyError as exc:
            raise TargetPerceptionConfigurationError(
                f"no YOLO service URL configured for active UAV {normalized!r}"
            ) from exc
    if len(config.uavs) > 1:
        raise TargetPerceptionConfigurationError(
            "multi-UAV YOLO requires distinct per_uav_urls entries"
        )
    return service.url


def preflight_fleet_yolo_services(
    config: AppConfig,
    active_uav_ids: Sequence[str],
    *,
    client_factory: Callable[..., object] | None = None,
) -> Mapping[str, Mapping[str, object]]:
    """Check every production worker before the first Isaac import."""

    from common.ids import validate_uav_id
    from perception.yolo_client import (
        YoloClientError,
        YoloClientUnavailable,
        YoloServiceClient,
        validate_yolo_model_identity,
    )

    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    if isinstance(active_uav_ids, (str, bytes)) or not isinstance(
        active_uav_ids, Sequence
    ):
        raise TypeError("active_uav_ids must be a sequence of UAV IDs")
    if config.target_perception.backend != "ultralytics_service":
        raise TargetPerceptionConfigurationError(
            "YOLO service preflight requires backend=ultralytics_service"
        )
    detector = config.target_perception.detector
    confirmation_mode = config.target_perception.confirmation.mode
    if (
        detector.model_family == "yolo"
        and confirmation_mode != "class_track_attribute_or_qwen"
    ):
        raise TargetPerceptionConfigurationError(
            "production YOLO preflight requires "
            "target_perception.confirmation.mode="
            "'class_track_attribute_or_qwen'; "
            f"actual_confirmation_mode={confirmation_mode!r}. "
            "Class/track-only confirmation is forbidden because it can lock "
            "a target before temporal attribute evidence is established."
        )
    factory = YoloServiceClient if client_factory is None else client_factory
    if not callable(factory):
        raise TypeError("client_factory must be callable")
    uav_ids = tuple(validate_uav_id(value) for value in active_uav_ids)
    if not uav_ids:
        raise ValueError("active_uav_ids must not be empty")
    if len(set(uav_ids)) != len(uav_ids):
        raise ValueError("active_uav_ids must be unique")
    configured_uavs = {item.id for item in config.uavs}
    unknown_uavs = sorted(set(uav_ids) - configured_uavs)
    if unknown_uavs:
        raise TargetPerceptionConfigurationError(
            "YOLO preflight received UAVs absent from config.uavs: "
            + ", ".join(unknown_uavs)
        )
    urls = tuple(yolo_service_url_for_uav(config, value) for value in uav_ids)
    if len(uav_ids) > 1 and len(set(urls)) != len(urls):
        raise TargetPerceptionConfigurationError(
            "parallel UAVs must not share one YOLO service URL"
        )

    service = config.target_perception.yolo_service
    if detector.model_family == "yolo" and (
        detector.expected_model_family != "yolo"
        or dict(detector.expected_model_names) != {0: "cube"}
        or detector.expected_model_sha256 is None
    ):
        raise TargetPerceptionConfigurationError(
            "production YOLO preflight requires an explicit trained-model "
            "identity contract: expected_model_family=yolo, "
            "expected_model_names={0: 'cube'}, and expected_model_sha256"
        )
    expected_family = detector.expected_model_family or detector.model_family
    expected_names = dict(detector.expected_model_names)
    result: dict[str, Mapping[str, object]] = {}
    for uav_id, url in zip(uav_ids, urls, strict=True):
        client = factory(
            base_url=url,
            request_timeout_s=service.request_timeout_s,
            jpeg_quality=service.jpeg_quality,
        )
        try:
            health = client.health()
            info = client.model_info()
        except YoloClientUnavailable as exc:
            raise YoloClientUnavailable(
                "YOLO worker preflight failed closed: "
                f"worker_url={url}; "
                f"expected_model_sha256={detector.expected_model_sha256!r}; "
                "actual_model_sha256='<unavailable>'; "
                "actual_model_names='<unavailable>'; "
                f"cause={type(exc).__name__}: {exc}"
            ) from exc
        except YoloClientError as exc:
            raise TargetPerceptionConfigurationError(
                "YOLO worker preflight failed closed: "
                f"worker_url={url}; "
                f"expected_model_sha256={detector.expected_model_sha256!r}; "
                "actual_model_sha256='<unavailable>'; "
                "actual_model_names='<unavailable>'; "
                f"cause={type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(health, Mapping) or (
            health.get("schema_version") != 1
            or health.get("status") != "ok"
            or health.get("ready") is not True
        ):
            raise TargetPerceptionConfigurationError(
                f"YOLO service for {uav_id} returned an invalid health response: "
                f"worker_url={url}; "
                f"expected_model_sha256={detector.expected_model_sha256!r}; "
                f"actual_model_sha256={info.model_sha256!r}; "
                f"actual_model_names={dict(info.names)!r}; ready={health.get('ready')!r}"
            )
        try:
            validate_yolo_model_identity(
                info,
                expected_model_family=expected_family,
                expected_model_names=expected_names,
                expected_model_sha256=detector.expected_model_sha256,
                worker_url=url,
            )
        except YoloClientError as exc:
            raise TargetPerceptionConfigurationError(
                f"YOLO service for {uav_id} failed model identity validation: {exc}"
            ) from exc
        names = dict(info.names)
        result[uav_id] = {
            "url": url,
            "model_family": info.model_family,
            "model_names": names,
            "model_sha256": getattr(info, "model_sha256", None),
            "ready": True,
        }
    return result


def build_target_perception_runtime(
    config: AppConfig,
    *,
    resolved_mode: ResolvedTargetPerceptionMode,
    environment: object,
    uav_id: str,
    attribute_evidence_sink: Callable[[object], None] | None = None,
    candidate_transition_sink: Callable[[object], None] | None = None,
) -> object:
    """Compatibility dispatcher for explicitly split capability builders.

    New production entrypoints should call
    :func:`build_yolo_target_perception_runtime` directly so an environment
    capability is never in scope during YOLO construction.
    """

    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    if not isinstance(resolved_mode, ResolvedTargetPerceptionMode):
        raise TypeError("resolved_mode must be ResolvedTargetPerceptionMode")
    if resolved_mode.backend != config.target_perception.backend:
        raise TargetPerceptionConfigurationError(
            "resolved target perception mode does not match YAML backend"
        )

    if resolved_mode.mode is TargetPerceptionMode.ORACLE:
        return build_oracle_target_perception_runtime(
            config,
            resolved_mode=resolved_mode,
            environment=environment,
            uav_id=uav_id,
        )

    if resolved_mode.mode is TargetPerceptionMode.YOLO:
        return build_yolo_target_perception_runtime(
            config,
            resolved_mode=resolved_mode,
            uav_id=uav_id,
            attribute_evidence_sink=attribute_evidence_sink,
            candidate_transition_sink=candidate_transition_sink,
        )

    raise TargetPerceptionConfigurationError(
        f"unsupported target perception mode: {resolved_mode.mode!r}"
    )


def build_oracle_target_perception_runtime(
    config: AppConfig,
    *,
    resolved_mode: ResolvedTargetPerceptionMode,
    environment: object,
    uav_id: str,
) -> object:
    """Construct the sole runtime allowed to receive evaluator capability."""

    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    if not isinstance(resolved_mode, ResolvedTargetPerceptionMode):
        raise TypeError("resolved_mode must be ResolvedTargetPerceptionMode")
    if (
        resolved_mode.mode is not TargetPerceptionMode.ORACLE
        or resolved_mode.backend != "oracle_evaluation"
        or config.target_perception.backend != "oracle_evaluation"
    ):
        raise TargetPerceptionConfigurationError(
            "Oracle runtime builder requires explicit oracle_evaluation mode"
        )
    make_oracle = getattr(environment, "make_oracle_perception", None)
    evaluator = getattr(environment, "get_evaluator_frame", None)
    if not callable(make_oracle) or not callable(evaluator):
        raise TargetPerceptionConfigurationError(
            "Oracle runtime requires assignment-scoped evaluator APIs"
        )
    raw = make_oracle(uav_id)
    guarded = GuardedPerceptionBackend(
        raw,
        profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
        acknowledge_privileged_oracle=True,
    )
    from perception.runtime_provider import OracleTargetPerceptionRuntime

    return OracleTargetPerceptionRuntime(
        uav_id=uav_id,
        oracle_backend=guarded,
        frame_provider=evaluator,
    )


def build_yolo_target_perception_runtime(
    config: AppConfig,
    *,
    resolved_mode: ResolvedTargetPerceptionMode,
    uav_id: str,
    attribute_evidence_sink: Callable[[object], None] | None = None,
    candidate_transition_sink: Callable[[object], None] | None = None,
) -> object:
    """Construct production vision without any environment/evaluator input."""

    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    if not isinstance(resolved_mode, ResolvedTargetPerceptionMode):
        raise TypeError("resolved_mode must be ResolvedTargetPerceptionMode")
    if (
        resolved_mode.mode is not TargetPerceptionMode.YOLO
        or resolved_mode.backend != "ultralytics_service"
        or config.target_perception.backend != "ultralytics_service"
    ):
        raise TargetPerceptionConfigurationError(
            "YOLO runtime builder requires production ultralytics_service mode"
        )

    from perception.candidate_bank import CandidateBank
    from perception.class_aliases import ClassAliasMapper
    from perception.runtime_bridge import CoordinatedVisionPerceptionBackend
    from perception.runtime_provider import YoloTargetPerceptionRuntime
    from perception.semantic_fusion import TemporalRgbdAttributeSemanticProvider
    from perception.target_perception_coordinator import TargetPerceptionCoordinator
    from runtime.frame_store import FrameStore

    service = config.target_perception.yolo_service
    per_uav_config = replace(
        config.target_perception,
        yolo_service=replace(
            service,
            url=yolo_service_url_for_uav(config, uav_id),
        ),
    )
    frame_config = config.frame_store
    frame_store = FrameStore(
        max_frames=frame_config.max_frames,
        max_bytes=frame_config.max_bytes,
        max_age_s=frame_config.max_age_s,
    )
    expected_names = dict(per_uav_config.detector.expected_model_names)
    if per_uav_config.detector.model_family == "yolo" and len(expected_names) != 1:
        raise TargetPerceptionConfigurationError(
            "production YOLO runtime requires exactly one configured "
            "expected_model_names entry"
        )
    if expected_names:
        expected_class_id, expected_class_name = next(iter(expected_names.items()))
    else:
        # Open-vocabulary YOLOE assigns the active prompt at reset time.  Its
        # legacy semantic adapter still needs inert construction defaults;
        # they are not a detector-class authorization decision.
        expected_class_id, expected_class_name = 0, "cube"
    alias_mapper = ClassAliasMapper.from_yaml(
        per_uav_config.detector.class_aliases_path
    )

    def resolve_detector_class(category: str) -> tuple[int, str]:
        if per_uav_config.detector.model_family == "yoloe":
            return 0, category
        resolved = alias_mapper.resolve(category, expected_names)
        return resolved.class_id, resolved.class_name
    semantic_provider = (
        TemporalRgbdAttributeSemanticProvider.from_target_perception_config(
            per_uav_config,
            mission_id="mission_unbound",
            uav_id=uav_id,
            assignment_id="assignment_unbound",
            frame_store=frame_store,
            expected_class_name=expected_class_name,
            expected_class_id=expected_class_id,
        )
    )
    candidate_bank = CandidateBank(uav_id=uav_id)
    coordinator = TargetPerceptionCoordinator(
        per_uav_config,
        frame_store=frame_store,
        candidate_bank=candidate_bank,
        semantic_evidence_provider=semantic_provider,
    )
    vision = VisionPerceptionBackend(per_uav_config, uav_id=uav_id)
    return YoloTargetPerceptionRuntime(
        uav_id=uav_id,
        attribute_evidence_sink=attribute_evidence_sink,
        candidate_transition_sink=candidate_transition_sink,
        detector_class_resolver=resolve_detector_class,
        bridge=CoordinatedVisionPerceptionBackend(
            uav_id=uav_id,
            coordinator=coordinator,
            vision_backend=vision,
        ),
    )


__all__ = [
    "TargetPerceptionConfigurationError",
    "TargetPerceptionUnavailableError",
    "build_target_candidate_resolver",
    "build_target_perception_backend",
    "build_oracle_target_perception_runtime",
    "build_target_perception_runtime",
    "build_yolo_target_perception_runtime",
    "preflight_fleet_yolo_services",
    "preflight_temporal_ray_depth",
    "validate_target_perception_preflight",
    "yolo_service_url_for_uav",
]
