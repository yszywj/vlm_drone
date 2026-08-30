"""Temporal ray-depth residual model training components.

Production code may reuse the network definition, but privileged labels stay in
``datasets.target_state`` and are never imported by perception runtime modules.

The collection finalizer and shard builder are CPU-only utilities.  Keep the
PyTorch-backed model exports lazy so importing either utility does not require a
training environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from training.target_state.config import (
    TargetStateTrainingConfig,
    TrainingStage,
    load_training_config,
)

if TYPE_CHECKING:
    from training.target_state.model import (
        TemporalRayDepthNet,
        TemporalRayDepthOutput,
    )


_LAZY_MODEL_EXPORTS = frozenset({"TemporalRayDepthNet", "TemporalRayDepthOutput"})


def __getattr__(name: str) -> Any:
    """Load PyTorch model types only when a caller requests one of them."""

    if name not in _LAZY_MODEL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from training.target_state.model import (
        TemporalRayDepthNet,
        TemporalRayDepthOutput,
    )

    # Cache both exports so later attribute access has normal module semantics
    # and never repeats the model import.
    globals().update(
        {
            "TemporalRayDepthNet": TemporalRayDepthNet,
            "TemporalRayDepthOutput": TemporalRayDepthOutput,
        }
    )
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_MODEL_EXPORTS)

__all__ = [
    "TargetStateTrainingConfig",
    "TemporalRayDepthNet",
    "TemporalRayDepthOutput",
    "TrainingStage",
    "load_training_config",
]
