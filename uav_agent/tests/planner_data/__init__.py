"""Pure-Python tests for Planner dataset tooling.

``unittest discover -s tests`` imports this directory as top-level
``planner_data``.  Extend the package path to the production modules so test
discovery cannot shadow the implementation under test.
"""

from pathlib import Path


_PRODUCTION_PLANNER_DATA = Path(__file__).resolve().parents[2] / "planner_data"
if str(_PRODUCTION_PLANNER_DATA) not in __path__:
    __path__.append(str(_PRODUCTION_PLANNER_DATA))
