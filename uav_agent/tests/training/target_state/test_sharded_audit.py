"""Offline audit metrics and commit-before-consume crash recovery."""
from pathlib import Path
from dataclasses import replace
import json
import tempfile
import unittest
from unittest import mock

import torch

from scripts import audit_target_state_sharded as audit
from tests.training.target_state.test_shards import _write_parent_dataset
from tests.training.target_state.test_sharded_trainer import _FakeLifecycle
from training.target_state.config import TargetStateTrainingConfig, TrainingStage
from training.target_state.shards import build_target_state_shards
from scripts.target_state_audit_evidence import CaseExportOptions, compare_metrics
from tests.training.target_state.test_data_and_trainer import (
    _missing_depth_evaluation_batch, _FixedEvaluationModel,
)
from training.target_state.trainer import TargetStateEvaluationAccumulator


class AuditSummaryTest(unittest.TestCase):
    def test_detailed_failure_flags_match_official_gate_and_ignore_truth(self):
        one = _missing_depth_evaluation_batch()
        n = 9
        batch = {k: v.repeat(n, *([1] * (v.ndim - 1))) for k, v in one.items()}
        batch["missing_mask"][:, -1] = False
        batch["raw_depth_m"].fill_(4)
        batch["reference_sensor_consistent"] = torch.ones(n, dtype=torch.bool)
        batch["validity_supervision_mask"] = torch.ones(n, dtype=torch.bool)
        batch["missing_mask"][1, -1] = True
        batch["raw_depth_m"][2] = 0
        batch["reference_sensor_consistent"][3] = False
        batch["anchor_uv_px"][4, 0] = -1
        output = _FixedEvaluationModel(validity_logit=8.0)(
            batch["roi_rgbd"], batch["geometry"], batch["missing_mask"])
        output.delta_uv_px[5, 0] = 100
        output.depth_residual_m[6] = -4
        output.measurement_valid_logit[7] = -8
        # Impossible true centre and unknown supervision must NOT gate output.
        batch["target_position_world_m"][8] = torch.tensor([5, 0, -100])
        batch["validity_supervision_mask"][8] = False
        rows = audit.diagnostic_rows(batch, output, maximum_depth_m=200)
        self.assertEqual([r["model_failure_reason"] for r in rows], [
            None, "detector_miss", "raw_depth_invalid_or_out_of_range",
            "rgbd_consistency_rejected", "reference_anchor_or_image_invalid",
            "corrected_pixel_invalid_or_out_of_image", "corrected_depth_invalid_or_out_of_range",
            "validity_head_rejected", None])
        self.assertFalse(rows[8]["offline_target_in_output_domain"])
        self.assertTrue(rows[8]["model_valid"])
        self.assertTrue(rows[2]["failure_flags"]["corrected_depth_invalid_or_out_of_range"])
        for r in rows:
            r["detected"] = r["reference_detected"]
        summary = audit.summarize_rows(rows)
        self.assertEqual(sum(summary["model_failure_reasons_ordered"].values()), 7)
        self.assertEqual(summary["model_only_failed_count"], 3)
        acc = TargetStateEvaluationAccumulator(200)
        acc.add_batch(batch=batch, output=output)
        state = acc.state_dict()
        self.assertEqual([r["model_valid"] for r in rows], (~torch.cat(state["model_failures"])).tolist())
        self.assertEqual([r["baseline_valid"] for r in rows], (~torch.cat(state["baseline_failures"])).tolist())
        # Mutating all privileged masks/coordinates leaves sensor gate identical.
        batch["target_present_mask"].fill_(False)
        batch["measurement_valid"].fill_(True)
        batch["target_position_world_m"].zero_()
        changed = audit.diagnostic_rows(batch, output, maximum_depth_m=200)
        self.assertEqual([r["model_valid"] for r in rows], [r["model_valid"] for r in changed])

    def test_metric_replay_reports_count_changes_and_tolerates_roundoff(self):
        original = {"model": {"accepted": 2690, "p95": .31909504}}
        self.assertEqual(compare_metrics({"model": {"accepted": 2690, "p95": .31909505}}, original), [])
        differences = compare_metrics({"model": {"accepted": 2691, "p95": .31909504}}, original)
        self.assertEqual(differences[0]["field"], "model.accepted")

    def test_rejected_errors_are_excluded_and_regressions_are_paired(self):
        def row(model, baseline, error, **extra):
            return dict(evaluated_visible_target=True, no_target=False,
                        model_valid=model, baseline_valid=baseline,
                        model_geometry_valid=True, detected=True, raw_depth_m=4.0,
                        model_error_m=error, baseline_error_m=error + 0.5, **extra)
        good = row(True, True, 0.1)
        missed = row(False, False, 30.0)
        missed.update(detected=False, raw_depth_m=0.0, model_geometry_valid=False)
        regression = row(False, True, 20.0)
        recovery = row(True, False, 0.2)
        negative = row(True, True, 100.0)
        negative.update(evaluated_visible_target=False, no_target=True)
        report = audit.summarize_rows([good, missed, regression, recovery, negative])
        self.assertEqual(report["visible_target_count"], 4)
        self.assertEqual(report["model_failed_count"], 2)
        self.assertEqual(report["baseline_failed_count"], 2)
        self.assertEqual(report["model_only_failed_count"], 1)
        self.assertEqual(report["baseline_only_failed_count"], 1)
        self.assertEqual(report["both_failed_count"], 1)
        self.assertEqual(report["model_false_positive_count"], 1)
        self.assertLess(report["model_accepted_only"]["p95_m"], 0.21)
        self.assertEqual(report["both_accepted"]["model"]["count"], 1)
        self.assertEqual(report["model_failure_reasons_ordered"],
                         {"detector_miss": 1, "validity_head_rejected": 1})

    def test_empty_subsets_are_not_reported_as_zero_error(self):
        report = audit.summarize_rows([])
        self.assertIsNone(report["model_accepted_only"]["p95_m"])
        self.assertEqual(report["model_accepted_only"]["count"], 0)


class AuditLifecycleTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        parent = root / "parent"
        _write_parent_dataset(parent)
        index = build_target_state_shards(
            parent, root / "shards", target_shard_size_bytes=1,
            history_size=4, max_history_age_s=2.0, split_seed=42,
        ).shard_index
        config = TargetStateTrainingConfig(
            dataset_root=parent, output_dir=root / "training",
            stage=TrainingStage.ORACLE_CLEAN, history_size=4,
            roi_size_px=32, roi_feature_dim=8, geometry_feature_dim=8,
            hidden_dim=8, gru_layers=1, num_workers=0, device="cpu",
        )
        options = audit.ShardedTrainingOptions(
            shard_index_path=index.source_path, pc_trans_root=root / "unused",
            pc_trans_config=root / "unused.json", bridge_root=root / "bridge",
            run_id_prefix="audit_fixture", wait_timeout_s=0,
        )
        self.lifecycle = _FakeLifecycle(root / "shards", root / "bridge")
        self.output = root / "audit"
        self.entry = index.shards_for_split("validation")[0]
        self.kwargs = dict(
            split="validation", config=config, options=options, index=index,
            lifecycle=self.lifecycle,
            model=audit._new_model(config, torch.device("cpu")).eval(),
            device=torch.device("cpu"), output_dir=self.output, contract_sha="fixture",
        )

    def test_commit_then_consume_and_resume_without_reinference(self):
        consume = self.lifecycle.consume
        def interrupted_consume(run_id, filename, *, delete):
            receipt = self.output / "receipts" / (filename + ".pt")
            self.assertTrue(receipt.is_file())
            raise RuntimeError("simulated interruption after evidence commit")
        with mock.patch.object(self.lifecycle, "consume", side_effect=interrupted_consume):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                audit.audit_split(**self.kwargs)
        with mock.patch.object(audit, "evaluate_archive", side_effect=AssertionError("must resume receipt")):
            report = audit.audit_split(**self.kwargs)
            repeated = audit.audit_split(**self.kwargs)
        self.assertEqual(report, repeated)
        self.assertEqual(report["diagnostics"]["sample_count"], self.entry.sequence_count)
        diagnostics = report["diagnostics"]
        self.assertAlmostEqual(
            report["evaluation_metrics"]["model"]["measurement_failure_rate"],
            diagnostics["model_failed_count"] / diagnostics["visible_target_count"],
            places=6,
        )
        self.assertTrue((self.lifecycle.source / self.entry.filename).is_file())
        self.assertFalse(self.lifecycle._active("audit_fixture.validation", self.entry.filename).exists())
        self.assertEqual(self.lifecycle.consume, consume)

    def test_failed_evidence_commit_does_not_consume_archive(self):
        with mock.patch.object(audit, "save_receipt", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                audit.audit_split(**self.kwargs)
        self.assertFalse(self.lifecycle.consumed)
        self.assertTrue(self.lifecycle._active("audit_fixture.validation", self.entry.filename).is_file())
        self.assertTrue((self.lifecycle.source / self.entry.filename).is_file())

    def test_mismatched_receipt_is_not_reused(self):
        audit.audit_split(**self.kwargs)
        self.kwargs["contract_sha"] = "different-model"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            audit.audit_split(**self.kwargs)

    def enable_case_exports(self, budget=1024**2):
        self.kwargs["case_options"] = CaseExportOptions(
            max_bytes=budget, episode_ids=tuple(self.entry.episode_ids))

    def test_case_export_retains_full_window_and_resume_checks_assets(self):
        self.enable_case_exports()
        report = audit.audit_split(**self.kwargs)
        self.assertEqual(report["case_exports"]["rgbd_complete_count"], self.entry.sequence_count)
        self.assertGreater(report["case_exports"]["unique_asset_bytes"], 0)
        evidence = json.loads((self.output / "validation_cases.json").read_text())
        self.assertEqual(len(evidence["cases"][0]["records"]), 5)
        self.assertEqual(len(evidence["assets"]), 10)  # RGB + depth, no oracle masks
        for case in evidence["cases"]:
            for relative, copied in case["asset_paths"].items():
                self.assertEqual((self.output / copied).read_bytes(),
                                 (self.kwargs["config"].dataset_root / relative).read_bytes())
        self.assertEqual(report, audit.audit_split(**self.kwargs))
        (self.output / next(iter(evidence["assets"]))).write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "truncated case evidence"):
            audit.audit_split(**self.kwargs)
        self.assertTrue((self.lifecycle.source / self.entry.filename).is_file())

    def test_exhausted_asset_budget_keeps_metadata_without_claiming_replayable(self):
        self.enable_case_exports(budget=0)
        report = audit.audit_split(**self.kwargs)
        self.assertEqual(report["case_exports"]["rgbd_complete_count"], 0)
        self.assertEqual(report["case_exports"]["rgbd_omitted_budget_count"], self.entry.sequence_count)
        self.assertEqual(report["case_exports"]["unique_asset_bytes"], 0)
        self.assertFalse((self.output / "case_assets").exists())

    def test_case_write_or_verify_failure_never_consumes_source(self):
        self.enable_case_exports()
        with mock.patch.object(audit, "export_cases", side_effect=OSError("evidence disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                audit.audit_split(**self.kwargs)
        self.assertFalse(self.lifecycle.consumed)
        with mock.patch.object(audit, "verify_case_assets", side_effect=ValueError("bad digest")):
            with self.assertRaisesRegex(ValueError, "bad digest"):
                audit.audit_split(**self.kwargs)
        self.assertFalse(self.lifecycle.consumed)
        self.assertTrue(self.lifecycle._active("audit_fixture.validation", self.entry.filename).is_file())
        with mock.patch.object(audit, "evaluate_archive", side_effect=AssertionError("must reuse receipt")):
            report = audit.audit_split(**self.kwargs)
        self.assertGreater(report["case_exports"]["rgbd_complete_count"], 0)

    def test_geometry_v2_fixture_replay_matches_official_counts(self):
        self.kwargs["config"] = replace(self.kwargs["config"],
            supervision_protocol="projected_center_v2", reference_guard_protocol="rgbd_consistency_v1")
        self.enable_case_exports()
        report = audit.audit_split(**self.kwargs)
        self.assertEqual(report["diagnostics"]["sample_count"], self.entry.sequence_count)
        rows = json.loads((self.output / "validation_samples.json").read_text())["samples"]
        self.assertEqual(report["evaluation_metrics"]["model"]["accepted_measurement_count"],
                         sum(r["model_valid"] and r["evaluated_visible_target"] for r in rows))
        self.assertIn("offline_label_subgroups_overlapping", report["diagnostics"])


if __name__ == "__main__":
    unittest.main()
