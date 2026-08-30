"""Import-boundary tests for CPU-only Target State dataset utilities."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class TargetStatePackageImportTest(unittest.TestCase):
    def test_collection_tools_import_when_torch_is_unavailable(self) -> None:
        repository_root = Path(__file__).resolve().parents[3]
        environment = os.environ.copy()
        current_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(repository_root) + (
            os.pathsep + current_pythonpath if current_pythonpath else ""
        )
        script = textwrap.dedent(
            """
            import builtins
            import sys

            real_import = builtins.__import__

            def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "torch" or name.startswith("torch."):
                    raise ModuleNotFoundError("torch intentionally unavailable in this test")
                return real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = import_without_torch

            from training.target_state.collection_finalize import (
                finalize_target_state_collection,
            )
            from training.target_state.shards import build_target_state_shards

            assert callable(finalize_target_state_collection)
            assert callable(build_target_state_shards)
            assert "torch" not in sys.modules
            print("OK")
            """
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertEqual(completed.stdout.strip(), "OK")

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "training-environment compatibility requires torch",
    )
    def test_package_still_exports_model_types_in_training_environment(self) -> None:
        from training.target_state import TemporalRayDepthNet, TemporalRayDepthOutput
        from training.target_state.model import (
            TemporalRayDepthNet as DirectTemporalRayDepthNet,
        )
        from training.target_state.model import (
            TemporalRayDepthOutput as DirectTemporalRayDepthOutput,
        )

        self.assertIs(TemporalRayDepthNet, DirectTemporalRayDepthNet)
        self.assertIs(TemporalRayDepthOutput, DirectTemporalRayDepthOutput)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
