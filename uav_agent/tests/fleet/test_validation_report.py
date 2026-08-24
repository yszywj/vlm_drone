from __future__ import annotations

from copy import deepcopy

import pytest

from runtime.validation_codes import ValidationCode
from runtime.validation_report import (
    RecoveryRecommendation,
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
)


def _finding(
    severity: ValidationSeverity = ValidationSeverity.RECOVERABLE_SEMANTIC_ERROR,
) -> ValidationFinding:
    return ValidationFinding(
        schema_version=1,
        finding_id="finding_goal_track",
        timestamp=1.25,
        stage="LOCAL_GOAL_VALIDATION",
        scope="ASSIGNMENT",
        severity=severity,
        code=(
            ValidationCode.GOAL_NOT_COVERED
            if severity is ValidationSeverity.RECOVERABLE_SEMANTIC_ERROR
            else ValidationCode.OUT_OF_BOUNDS_GOTO
        ),
        message="required goal is not covered",
        mission_id="mission_validation",
        assignment_id="assignment_a",
        uav_id="uav_a",
        goal_id="goal_track",
        step_id=None,
        proposal_id="proposal_1",
        evidence_refs=("evidence_1",),
        recommended_action=RecoveryRecommendation.REPAIR_LOCAL_PLAN,
    )


def _report(finding: ValidationFinding) -> ValidationReport:
    return ValidationReport(
        schema_version=1,
        report_id="report_validation",
        timestamp=1.25,
        stage="LOCAL_GOAL_VALIDATION",
        mission_id="mission_validation",
        assignment_id="assignment_a",
        uav_id="uav_a",
        findings=(finding,),
    )


def test_validation_report_round_trip_is_strict_and_lossless() -> None:
    report = _report(_finding())
    restored = ValidationReport.from_dict(report.to_dict())

    assert restored == report
    assert restored.executable
    assert not restored.semantically_valid
    assert not restored.accepted


def test_hard_action_block_is_never_executable() -> None:
    report = _report(_finding(ValidationSeverity.HARD_ACTION_BLOCK))

    assert report.hard_blocked
    assert not report.executable
    assert not report.accepted


@pytest.mark.parametrize("target", ["report", "finding"])
def test_unknown_json_fields_are_rejected(target: str) -> None:
    report = _report(_finding())
    payload = report.to_dict()
    if target == "report":
        payload["hidden_reasoning"] = "forbidden"
    else:
        finding = deepcopy(payload["findings"][0])
        finding["oracle_target_pose"] = [1, 2, 3]
        payload["findings"][0] = finding

    with pytest.raises(ValueError, match="unknown fields"):
        ValidationReport.from_dict(payload)


def test_nonfinite_timestamp_and_mismatched_routing_are_rejected() -> None:
    payload = _report(_finding()).to_dict()
    payload["timestamp"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ValidationReport.from_dict(payload)

    payload = _report(_finding()).to_dict()
    payload["findings"][0]["mission_id"] = "mission_other"
    with pytest.raises(ValueError, match="does not match"):
        ValidationReport.from_dict(payload)
