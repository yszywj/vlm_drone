from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.smoke_test import run_smoke, verify_smoke_run


class SmokeOutputTest(unittest.TestCase):
    def test_shell_launcher_preserves_python_exit_and_appends_terminal(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        launcher = project_root / "scripts" / "run_with_output.sh"
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            environment = dict(os.environ)
            environment["RUN_DIR"] = str(run_dir)
            completed = subprocess.run(
                [
                    "bash",
                    str(launcher),
                    "-c",
                    "import sys; print('shell-out'); print('shell-err', file=sys.stderr); sys.exit(7)",
                ],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 7)
            self.assertIn("shell-out", completed.stdout)
            self.assertIn("shell-err", completed.stdout)
            self.assertEqual((run_dir / "exit_code.txt").read_text(encoding="utf-8"), "7\n")
            terminal = (run_dir / "logs" / "terminal.log").read_text(encoding="utf-8")
            self.assertIn("shell-out", terminal)
            self.assertIn("shell-err", terminal)

            environment["VLM_DRONE_RESUME"] = "1"
            resumed = subprocess.run(
                ["bash", str(launcher), "-c", "print('resumed')"],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(resumed.returncode, 0)
            terminal = (run_dir / "logs" / "terminal.log").read_text(encoding="utf-8")
            self.assertIn("================ RUN RESUMED ================", terminal)
            self.assertIn("resumed", terminal)

    def test_complete_output_flow_is_small_and_has_no_raw_artifacts(self) -> None:
        before_heavy = {name for name in sys.modules if name.startswith(("isaacsim", "omni."))}
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, summary = run_smoke(temporary, min_free_space_gb_before_start=0.0)

            verified = verify_smoke_run(run_dir)
            self.assertGreaterEqual(verified["size_bytes"], summary["size_bytes"])
            self.assertLess(summary["size_mib"], 20.0)
            with (run_dir / "metrics" / "episode_metrics.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                episodes = list(csv.DictReader(stream))
            self.assertEqual(sum(row["phase"] == "train" for row in episodes), 20)
            self.assertEqual(sum(row["phase"] == "validation" for row in episodes), 10)
            self.assertEqual(sum(row["phase"] == "test" for row in episodes), 20)
            self.assertTrue(any(row["mission_success_strict"] == "True" for row in episodes))
            self.assertTrue(any(row["mission_success_strict"] == "False" for row in episodes))
            self.assertEqual((run_dir / "exit_code.txt").read_text(encoding="utf-8"), "0\n")
            terminal = (run_dir / "logs" / "terminal.log").read_text(encoding="utf-8")
            self.assertIn("[TRAIN] update=10", terminal)
            self.assertIn("[EVAL]", terminal)
            self.assertIn("[FINISHED]", terminal)
            for forbidden in ("videos", "images", "frames", "trajectories", "observations"):
                self.assertFalse((run_dir / forbidden).exists())
        after_heavy = {name for name in sys.modules if name.startswith(("isaacsim", "omni."))}
        self.assertEqual(after_heavy, before_heavy)


if __name__ == "__main__":
    unittest.main()
