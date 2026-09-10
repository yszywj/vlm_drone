"""Temporal target-state dataset schema.

This package is collection/training-only.  Production perception modules must
never import it because :class:`TargetTrainingLabel` contains simulator truth.
"""

from datasets.target_state.schema import (
    CameraFrameInput,
    DetectorPrediction,
    SensorInput,
    TargetStateFrameRecord,
    TargetTrainingLabel,
    UavFrameInput,
)
from datasets.target_state.sequence import TargetStateSequence

__all__ = [
    "CameraFrameInput",
    "DetectorPrediction",
    "SensorInput",
    "TargetStateFrameRecord",
    "TargetStateSequence",
    "TargetTrainingLabel",
    "UavFrameInput",
]

