from __future__ import annotations

import os
from pathlib import Path
import subprocess
from uuid import uuid4

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_MODEL = Path(
    "/home/amax/ry/vlm_drones/models/initial_model/Qwen3-VL-4B-Instruct"
)


@pytest.mark.qwen_lora_integration
def test_real_qwen3vl_one_optimizer_step(tmp_path: Path) -> None:
    """Opt-in proof that the real local 4B checkpoint can train one LoRA step."""

    if os.environ.get("RUN_QWEN_LORA_INTEGRATION") != "1":
        pytest.skip("set RUN_QWEN_LORA_INTEGRATION=1 in the qwen_lora environment")
    if not LOCAL_MODEL.is_dir():
        pytest.skip(f"local Qwen3-VL checkpoint is absent: {LOCAL_MODEL}")

    run_id = f"pytest_smoke_{uuid4().hex[:12]}"
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "QWEN_LORA_SMOKE_RUN_ID": run_id,
            "QWEN_LORA_SMOKE_OUTPUT_ROOT": str(tmp_path / "outputs"),
            "QWEN_LORA_SMOKE_ADAPTER_ROOT": str(tmp_path / "adapters"),
        }
    )
    subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/run_fleet_planner_lora_smoke.sh")],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    adapter = tmp_path / "adapters" / run_id
    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_model.safetensors").stat().st_size > 0
    assert (adapter / "adapter_manifest.json").is_file()
