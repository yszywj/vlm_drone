"""Pure-Python tests for the Stage-1 planner world-context adapter."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import unittest

from configs.loader import (
    AppConfig,
    SceneConfig,
    SearchConfig,
    TargetRegionConfig,
    UavConfig,
    load_config,
)
from planner.schemas import MissionIntent, PlannerWorldContext
from runtime.plan_validator import PlanValidator
from runtime.world_context_builder import (
    LANDING_ZONE_NAME,
    SEARCH_REGION_NAME,
    WorldContextBuildError,
    build_planner_world_context,
)
from skills.types import SkillName


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineWorldContextBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(PROJECT_ROOT / "configs" / "default.yaml")

    def build(
        self,
        config: AppConfig | None = None,
        **overrides: object,
    ) -> PlannerWorldContext:
        options: dict[str, object] = {
            "takeoff_altitude_m": 10.0,
            "track_duration_s": 30.0,
        }
        options.update(overrides)
        return build_planner_world_context(
            self.config if config is None else config,
            **options,  # type: ignore[arg-type]
        )

    def test_default_config_builds_named_search_area_and_home(self) -> None:
        context = self.build()

        self.assertEqual(context.scene_min_xyz_m, (-50.0, -50.0, 0.0))
        self.assertEqual(context.scene_max_xyz_m, (50.0, 50.0, 30.0))
        self.assertEqual(context.initial_uav_xyz_m, (0.0, 0.0, 10.0))
        self.assertEqual(set(context.search_regions), {SEARCH_REGION_NAME})
        self.assertEqual(set(context.landing_zones), {LANDING_ZONE_NAME})

        search = context.search_regions[SEARCH_REGION_NAME]
        self.assertEqual(search.name, "search_area")
        self.assertEqual(search.center_xyz_m, (10.0, 0.0, 0.5))
        self.assertEqual(search.radius_m, 25.0)
        self.assertEqual(search.approach_xyz_m, (-15.0, 0.0, 10.0))
        self.assertTrue(search.description)

        home = context.landing_zones[LANDING_ZONE_NAME]
        self.assertEqual(home.name, "home")
        self.assertEqual(home.position_xy_m, (0.0, 0.0))
        self.assertEqual(home.ground_altitude_m, 0.0)
        self.assertTrue(home.description)

    def test_explicit_runtime_overrides_are_reflected(self) -> None:
        context = self.build(
            start_altitude_m=2.0,
            takeoff_altitude_m=12.0,
            track_duration_s=45.0,
            goto_timeout_s=75.0,
            land_timeout_s=40.0,
        )

        self.assertEqual(context.initial_uav_xyz_m, (0.0, 0.0, 2.0))
        self.assertEqual(
            context.search_regions[SEARCH_REGION_NAME].approach_xyz_m,
            (-15.0, 0.0, 12.0),
        )
        self.assertEqual(context.default_takeoff_altitude_m, 12.0)
        self.assertEqual(context.default_track_duration_s, 45.0)
        self.assertEqual(context.goto_timeout_s, 75.0)
        self.assertEqual(context.land_timeout_s, 40.0)

    def test_configured_search_timeout_and_home_xy_are_used(self) -> None:
        config = replace(
            self.config,
            uav=replace(
                self.config.uav,
                initial_position_xyz_m=(-7.0, 8.0, 3.0),
            ),
            search=replace(self.config.search, timeout_s=91.0),
        )

        context = self.build(
            config,
            start_altitude_m=3.0,
            takeoff_altitude_m=11.0,
        )

        self.assertEqual(context.initial_uav_xyz_m, (-7.0, 8.0, 3.0))
        self.assertEqual(
            context.landing_zones[LANDING_ZONE_NAME].position_xy_m,
            (-7.0, 8.0),
        )
        self.assertEqual(context.search_timeout_s, 91.0)

    def test_live_target_motion_configuration_cannot_influence_context(self) -> None:
        baseline = self.build()
        # The builder deliberately consumes only target.initial_region.  Even a
        # replaced motion record cannot alter planner geometry or descriptions.
        changed = replace(
            self.config,
            target=replace(self.config.target, motion=object()),  # type: ignore[arg-type]
        )

        rebuilt = self.build(changed)

        self.assertEqual(rebuilt, baseline)

    def test_context_compiles_to_the_expected_six_step_plan(self) -> None:
        context = self.build(start_altitude_m=0.0)
        intent = MissionIntent(
            target_description="moving target",
            search_region=SEARCH_REGION_NAME,
            track_duration_s=30.0,
            landing_zone=LANDING_ZONE_NAME,
            takeoff_altitude_m=None,
        )

        compiled = PlanValidator().validate_and_compile(
            intent,
            context,
            source="scripted",
        )

        self.assertEqual(
            tuple(step.skill for step in compiled.task_plan.steps),
            (
                SkillName.TAKEOFF,
                SkillName.GOTO,
                SkillName.SEARCH,
                SkillName.TRACK,
                SkillName.GOTO,
                SkillName.LAND,
            ),
        )
        entries = compiled.task_plan.to_dicts()
        self.assertEqual(entries[1]["position"], [-15.0, 0.0, 10.0])
        self.assertEqual(entries[4]["position"], [0.0, 0.0, 10.0])

    def test_builder_output_contains_only_static_named_region_data(self) -> None:
        context = self.build()
        rendered = repr(context)

        self.assertNotIn("target_pose", rendered)
        self.assertNotIn("target_velocity", rendered)
        self.assertNotIn("EvaluatorFrame", rendered)
        self.assertNotIn("camera_rgb", rendered)
        self.assertEqual(
            {
                field.name
                for field in fields(context.search_regions[SEARCH_REGION_NAME])
            },
            {
                "name",
                "center_xyz_m",
                "radius_m",
                "approach_xyz_m",
                "description",
            },
        )

    def test_invalid_scalar_overrides_are_rejected(self) -> None:
        cases = (
            ("takeoff bool", {"takeoff_altitude_m": True}),
            ("takeoff zero", {"takeoff_altitude_m": 0.0}),
            ("takeoff infinity", {"takeoff_altitude_m": float("inf")}),
            ("takeoff above scene", {"takeoff_altitude_m": 31.0}),
            ("track bool", {"track_duration_s": False}),
            ("track below minimum", {"track_duration_s": 0.5}),
            ("track above maximum", {"track_duration_s": 601.0}),
            ("track nan", {"track_duration_s": float("nan")}),
            ("start bool", {"start_altitude_m": True}),
            ("start below scene", {"start_altitude_m": -0.1}),
            ("start above scene", {"start_altitude_m": 31.0}),
            ("start above takeoff", {"start_altitude_m": 11.0}),
            ("goto timeout zero", {"goto_timeout_s": 0.0}),
            ("land timeout nan", {"land_timeout_s": float("nan")}),
        )
        for label, overrides in cases:
            with self.subTest(label=label), self.assertRaises(
                WorldContextBuildError
            ):
                self.build(**overrides)

    def test_invalid_scene_and_uav_geometry_are_rejected(self) -> None:
        cases = (
            (
                "non-positive scene",
                replace(
                    self.config,
                    scene=SceneConfig(size_xyz_m=(0.0, 100.0, 30.0)),
                ),
            ),
            (
                "non-finite scene",
                replace(
                    self.config,
                    scene=SceneConfig(
                        size_xyz_m=(100.0, float("inf"), 30.0)
                    ),
                ),
            ),
            (
                "uav x outside",
                replace(
                    self.config,
                    uav=UavConfig(
                        initial_position_xyz_m=(51.0, 0.0, 10.0),
                        max_speed_mps=self.config.uav.max_speed_mps,
                        max_yaw_rate_deg_s=self.config.uav.max_yaw_rate_deg_s,
                    ),
                ),
            ),
        )
        for label, config in cases:
            with self.subTest(label=label), self.assertRaises(
                WorldContextBuildError
            ):
                self.build(config)

    def test_invalid_initial_target_region_is_rejected(self) -> None:
        cases = (
            (
                "reversed",
                TargetRegionConfig(
                    min_xyz_m=(12.0, -1.0, 0.5),
                    max_xyz_m=(11.0, 1.0, 0.5),
                ),
            ),
            (
                "outside x",
                TargetRegionConfig(
                    min_xyz_m=(49.0, -1.0, 0.5),
                    max_xyz_m=(51.0, 1.0, 0.5),
                ),
            ),
            (
                "non-finite",
                TargetRegionConfig(
                    min_xyz_m=(9.0, float("nan"), 0.5),
                    max_xyz_m=(11.0, 1.0, 0.5),
                ),
            ),
        )
        for label, region in cases:
            config = replace(
                self.config,
                target=replace(self.config.target, initial_region=region),
            )
            with self.subTest(label=label), self.assertRaises(
                WorldContextBuildError
            ):
                self.build(config)

    def test_search_disk_must_fit_inside_scene(self) -> None:
        config = replace(
            self.config,
            search=SearchConfig(
                radius_m=41.0,
                timeout_s=self.config.search.timeout_s,
                transit_yaw_mode=self.config.search.transit_yaw_mode,
            ),
        )

        with self.assertRaisesRegex(WorldContextBuildError, "radius"):
            self.build(config)

    def test_search_config_numbers_are_strictly_validated(self) -> None:
        for label, search in (
            (
                "radius bool",
                replace(self.config.search, radius_m=True),
            ),
            (
                "radius infinity",
                replace(self.config.search, radius_m=float("inf")),
            ),
            (
                "timeout zero",
                replace(self.config.search, timeout_s=0.0),
            ),
        ):
            config = replace(self.config, search=search)
            with self.subTest(label=label), self.assertRaises(
                WorldContextBuildError
            ):
                self.build(config)

    def test_wrong_config_type_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            build_planner_world_context(  # type: ignore[arg-type]
                object(),
                takeoff_altitude_m=10.0,
                track_duration_s=30.0,
            )

    def test_builder_module_has_no_simulator_or_live_pipeline_imports(self) -> None:
        source_path = PROJECT_ROOT / "runtime" / "world_context_builder.py"
        source = source_path.read_text(encoding="utf-8")

        for forbidden in (
            "isaacsim",
            "SimpleUavSearchEnv",
            "EvaluatorFrame",
            "Observation",
            "target.motion",
            "target_position",
            "target_velocity",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
