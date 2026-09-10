from copy import deepcopy
import os
from pathlib import Path
import subprocess

import pytest
import torch
import yaml

from scripts import fit_target_state_quality_cached as quality

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ["quality_cached_power025_50k.yaml", "quality_cached_power050_50k.yaml"]
LAUNCHER = ROOT / "scripts/train_target_state_quality_mild_50k.sh"


@pytest.mark.parametrize("filename,power", zip(CONFIGS, [.25, .5]))
def test_mild_config_changes_only_quality_weight_and_output_name(filename, power):
    base = ROOT / "configs/target_state"
    original = yaml.safe_load((base / "quality_cached_50k.yaml").read_text())
    candidate = yaml.safe_load((base / filename).read_text())
    assert candidate["experiment_name"] != original["experiment_name"]
    assert candidate["quality_fit"]["class_balance_power"] == power
    normalized = deepcopy(candidate)
    normalized["experiment_name"] = original["experiment_name"]
    normalized["quality_fit"]["class_balance_power"] = original["quality_fit"]["class_balance_power"]
    assert normalized == original
    candidate.pop("base_config")
    quality.validate_config(candidate)
    assert quality.THRESHOLD == .5 and quality.ERROR_LIMIT_M == 1.


@pytest.mark.parametrize("power", [.25, .5])
def test_real_class_counts_give_expected_mild_weight(power):
    labels = torch.tensor([1.] * 13368 + [0.] * 39)
    weights, stats = quality.quality_weights(labels, power)
    ratio = (13368/39)**power
    assert weights[-1]/weights[0] == pytest.approx(ratio)
    assert stats["negative_loss_weight_fraction"] == pytest.approx(39*ratio/(13368+39*ratio))
    assert stats["negative_loss_weight_fraction"] < .06


@pytest.fixture
def fake_launcher(tmp_path):
    # Isolated fixture: the shell entry point never reaches real data/training.
    app = tmp_path / "app with spaces"
    scripts = app / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / LAUNCHER.name
    script.write_bytes(LAUNCHER.read_bytes())
    interpreter = app / "python.sh"
    interpreter.write_text('''#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$QUALITY_LAUNCH_TEST_LOG"
if [[ -n "${QUALITY_LAUNCH_TEST_FAIL_CONFIG:-}" && "$*" == *"$QUALITY_LAUNCH_TEST_FAIL_CONFIG"* ]]; then
  if [[ "$QUALITY_LAUNCH_TEST_FAIL_PHASE" == check && "$*" == *--dry-run* ]] ||
     [[ "$QUALITY_LAUNCH_TEST_FAIL_PHASE" == train && "$*" != *--dry-run* ]]; then
    exit 7
  fi
fi
''')
    interpreter.chmod(0o755)
    log = tmp_path / "calls.txt"
    env = {**os.environ, "QUALITY_LAUNCH_TEST_LOG":str(log)}
    def run(args=(), fail_config="", fail_phase=""):
        current = {**env, "QUALITY_LAUNCH_TEST_FAIL_CONFIG":fail_config, "QUALITY_LAUNCH_TEST_FAIL_PHASE":fail_phase}
        result = subprocess.run(["bash", str(script), *args], cwd=tmp_path, env=current, text=True, capture_output=True, timeout=10)
        return result, log.read_text().splitlines() if log.exists() else []
    return run


def test_launcher_checks_both_then_trains_in_order(fake_launcher):
    result, calls = fake_launcher()
    assert result.returncode == 0
    expected = [f"scripts/fit_target_state_quality_cached.py --config configs/target_state/{name}" for name in CONFIGS]
    assert calls == [f"{c} --dry-run" for c in expected] + expected


def test_dry_run_never_starts_training(fake_launcher):
    result, calls = fake_launcher(["--dry-run"])
    assert result.returncode == 0 and len(calls) == 2
    assert all(c.endswith(" --dry-run") for c in calls)
    assert "completed;" not in result.stdout


@pytest.mark.parametrize("args", [["--fit-only"], ["--dry-run", "--dry-run"], ["unexpected"]])
def test_invalid_launcher_arguments_stop_before_any_command(fake_launcher, args):
    result, calls = fake_launcher(args)
    assert result.returncode == 2 and calls == []


@pytest.mark.parametrize("phase,filename,count", [("check",CONFIGS[0],1), ("check",CONFIGS[1],2), ("train",CONFIGS[0],3)])
def test_launcher_stops_on_failure_without_claiming_completion(fake_launcher, phase, filename, count):
    result, calls = fake_launcher(fail_config=filename, fail_phase=phase)
    assert result.returncode == 7 and len(calls) == count
    assert "Both quality-weight comparisons completed" not in result.stdout
