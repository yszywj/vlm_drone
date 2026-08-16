"""Pure-Python tests for the lightweight experiment-output subsystem.

``unittest discover -s tests`` imports this directory as the top-level package
``experiments``.  Extend that package's search path to the production directory
so test discovery cannot accidentally shadow ``experiments.metric_logger`` and
the other modules under test.  When imported as ``tests.experiments`` this is a
harmless additional package path.
"""

from pathlib import Path


_PRODUCTION_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
if str(_PRODUCTION_EXPERIMENTS) not in __path__:
    __path__.append(str(_PRODUCTION_EXPERIMENTS))
