from __future__ import annotations

from pathlib import Path
import sys

from scripts import run_fleet_planner_demo


_ROOT = Path(__file__).resolve().parents[2]
_INSTRUCTION = (
    "无人机A前往世界坐标二十、三十附近十五米范围搜索并跟踪目标i二十秒；"
    "无人机B前往世界坐标负二十五、十附近十二米范围搜索并跟踪目标j二十秒；"
    "完成后分别返回各自起点降落"
)


def test_scripted_demo_prints_all_required_sections_without_isaac(capsys) -> None:
    before = {name for name in sys.modules if name.startswith(("isaacsim", "omni"))}
    code = run_fleet_planner_demo.main(
        [
            "--config",
            str(_ROOT / "configs/multi_uav_demo.yaml"),
            "--fleet-planner",
            "scripted",
            "--local-planner",
            "dynamic_scripted",
            "--planning-contract",
            "v3",
            "--adapter-config",
            str(_ROOT / "configs/adapters.json"),
            "--instruction",
            _INSTRUCTION,
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    for heading in (
        "FleetMissionPlan",
        "Assignment summary",
        "Per-agent local plan",
        "Adapter selection",
        "Fleet Planner diagnostics",
        "Per-agent planner diagnostics",
    ):
        assert f"=== {heading} ===" in output
    assert '"effective_model": "Qwen3-VL-4B-Instruct"' in output
    assert '"fallback_used": true' in output
    after = {name for name in sys.modules if name.startswith(("isaacsim", "omni"))}
    assert after == before


def test_scripted_demo_preserves_both_routing_relationships() -> None:
    arguments = run_fleet_planner_demo._parser().parse_args(
        [
            "--config",
            str(_ROOT / "configs/multi_uav_demo.yaml"),
            "--adapter-config",
            str(_ROOT / "configs/adapters.json"),
            "--instruction",
            _INSTRUCTION,
        ]
    )
    result = run_fleet_planner_demo.run(arguments)
    summaries = result["assignment_summary"]
    assert [(row["uav_id"], row["target_alias"]) for row in summaries] == [
        ("uav_a", "target_i"),
        ("uav_b", "target_j"),
    ]
    assert summaries[0]["search_region"]["center_xyz_m"] == [20.0, 30.0, 0.0]
    assert summaries[1]["search_region"]["center_xyz_m"] == [-25.0, 10.0, 0.0]
