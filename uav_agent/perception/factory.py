"""Fail-closed factory for independent target-perception backends."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace

from configs.schema import AppConfig
from perception.base import PerceptionBackend
from perception.oracle import OraclePerception
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
    from perception.yolo_client import YoloServiceClient

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
    result: dict[str, Mapping[str, object]] = {}
    for uav_id, url in zip(uav_ids, urls, strict=True):
        client = factory(
            base_url=url,
            request_timeout_s=service.request_timeout_s,
            jpeg_quality=service.jpeg_quality,
        )
        health = client.health()
        if not isinstance(health, Mapping) or (
            health.get("schema_version") != 1
            or health.get("status") != "ok"
            or health.get("ready") is not True
        ):
            raise TargetPerceptionConfigurationError(
                f"YOLO service for {uav_id} returned an invalid health response"
            )
        info = client.model_info()
        if info.model_family != "yolo":
            raise TargetPerceptionConfigurationError(
                f"YOLO service for {uav_id} must report model_family='yolo'"
            )
        names = dict(info.names)
        normalized_names = {
            class_id: name.strip().casefold()
            for class_id, name in names.items()
            if isinstance(class_id, int) and isinstance(name, str)
        }
        if normalized_names != {0: "cube"} or len(names) != 1:
            raise TargetPerceptionConfigurationError(
                f"YOLO service for {uav_id} must expose exactly class 0='cube'"
            )
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
) -> object:
    """Create one provider; all Oracle/YOLO construction branches live here."""

    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    if not isinstance(resolved_mode, ResolvedTargetPerceptionMode):
        raise TypeError("resolved_mode must be ResolvedTargetPerceptionMode")
    if resolved_mode.backend != config.target_perception.backend:
        raise TargetPerceptionConfigurationError(
            "resolved target perception mode does not match YAML backend"
        )

    if resolved_mode.mode is TargetPerceptionMode.ORACLE:
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

    if resolved_mode.mode is TargetPerceptionMode.YOLO:
        from perception.runtime_bridge import CoordinatedVisionPerceptionBackend
        from perception.runtime_provider import YoloTargetPerceptionRuntime
        from perception.candidate_bank import CandidateBank
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
        semantic_provider = (
            TemporalRgbdAttributeSemanticProvider.from_target_perception_config(
                per_uav_config,
                # observe() is unavailable before the public runtime's first
                # Assignment reset, so these valid inert IDs carry no data.
                mission_id="mission_unbound",
                uav_id=uav_id,
                assignment_id="assignment_unbound",
                frame_store=frame_store,
                expected_class_name="cube",
                expected_class_id=0,
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
            bridge=CoordinatedVisionPerceptionBackend(
                uav_id=uav_id,
                coordinator=coordinator,
                vision_backend=vision,
            ),
        )

    raise TargetPerceptionConfigurationError(
        f"unsupported target perception mode: {resolved_mode.mode!r}"
    )


__all__ = [
    "TargetPerceptionConfigurationError",
    "TargetPerceptionUnavailableError",
    "build_target_perception_backend",
    "build_target_perception_runtime",
    "preflight_fleet_yolo_services",
    "validate_target_perception_preflight",
    "yolo_service_url_for_uav",
]
