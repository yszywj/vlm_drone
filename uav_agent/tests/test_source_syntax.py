from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SourceSyntaxTest(unittest.TestCase):
    def test_all_project_python_sources_parse(self) -> None:
        for path in sorted(PROJECT_ROOT.rglob("*.py")):
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


if __name__ == "__main__":
    unittest.main()
