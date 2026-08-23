"""Isaac Sim environments.

Isaac-dependent modules are intentionally not imported here. A standalone
entry point must create ``SimulationApp`` before importing them.
"""

# This registry is deliberately simulator-independent and safe to import here.
from env.obstacle_registry import ObstacleRegistry

__all__ = ["ObstacleRegistry"]
