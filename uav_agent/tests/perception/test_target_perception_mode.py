from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from perception.mode import (
    ResolvedTargetPerceptionMode,
    TargetPerceptionMode,
    TargetPerceptionModeError,
    resolve_target_perception_mode,
)
from perception.runtime import PerceptionRuntimeProfile


def test_explicit_modes_have_one_fixed_mapping() -> None:
    oracle = resolve_target_perception_mode(TargetPerceptionMode.ORACLE)
    assert oracle.to_dict() == {
        "schema_version": 1,
        "mode": "oracle",
        "runtime_profile": "oracle_evaluation",
        "backend": "oracle_evaluation",
        "privileged": True,
        "requires_oracle_acknowledgement": True,
    }

    yolo = resolve_target_perception_mode("yolo")
    assert yolo.to_dict() == {
        "schema_version": 1,
        "mode": "yolo",
        "runtime_profile": "production",
        "backend": "ultralytics_service",
        "privileged": False,
        "requires_oracle_acknowledgement": False,
    }


@pytest.mark.parametrize(
    ("mode", "profile", "backend", "acknowledged", "message"),
    (
        ("oracle", "production", "oracle_evaluation", True, "runtime profile"),
        ("oracle", None, "ultralytics_service", True, "YAML"),
        ("oracle", None, "disabled", True, "YAML"),
        ("oracle", None, "oracle_evaluation", False, "acknowledge"),
        ("yolo", "oracle_evaluation", "ultralytics_service", False, "runtime profile"),
        ("yolo", None, "oracle_evaluation", False, "YAML"),
        ("yolo", None, "disabled", False, "YAML"),
        ("yolo", None, "ultralytics_service", True, "forbidden"),
    ),
)
def test_conflicting_switches_fail_closed(
    mode: str,
    profile: str | None,
    backend: str,
    acknowledged: bool,
    message: str,
) -> None:
    with pytest.raises(TargetPerceptionModeError, match=message):
        resolve_target_perception_mode(
            mode,
            runtime_profile=profile,
            backend=backend,
            acknowledge_privileged_oracle=acknowledged,
        )


def test_resolved_contract_cannot_be_forged_or_mutated() -> None:
    resolved = resolve_target_perception_mode("oracle")
    with pytest.raises(FrozenInstanceError):
        resolved.backend = "ultralytics_service"  # type: ignore[misc]
    with pytest.raises(TargetPerceptionModeError, match="non-canonical"):
        ResolvedTargetPerceptionMode(
            mode=TargetPerceptionMode.ORACLE,
            runtime_profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
            backend="disabled",
            privileged=True,
            requires_oracle_acknowledgement=True,
        )
    with pytest.raises(ValueError, match="schema_version"):
        ResolvedTargetPerceptionMode(
            mode=TargetPerceptionMode.ORACLE,
            runtime_profile=PerceptionRuntimeProfile.ORACLE_EVALUATION,
            backend="oracle_evaluation",
            privileged=True,
            requires_oracle_acknowledgement=True,
            schema_version=2,
        )
