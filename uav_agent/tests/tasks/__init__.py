"""Pure-Python tests for independent Gold task semantics.

``unittest discover -s tests`` imports this directory as top-level ``tasks``.
Extend its path so that production submodules remain importable in that mode.
"""

from pathlib import Path


_PRODUCTION_TASKS = Path(__file__).resolve().parents[2] / "tasks"
if str(_PRODUCTION_TASKS) not in __path__:
    __path__.append(str(_PRODUCTION_TASKS))
