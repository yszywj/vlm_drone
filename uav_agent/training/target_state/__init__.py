"""Temporal ray-depth residual model training components.

Production code may reuse the network definition, but privileged labels stay in
``datasets.target_state`` and are never imported by perception runtime modules.
"""

from training.target_state.config import (
    TargetStateTrainingConfig,
    TrainingStage,
    load_training_config,
)
from training.target_state.model import TemporalRayDepthNet, TemporalRayDepthOutput

__all__ = [
    "TargetStateTrainingConfig",
    "TemporalRayDepthNet",
    "TemporalRayDepthOutput",
    "TrainingStage",
    "load_training_config",
]
