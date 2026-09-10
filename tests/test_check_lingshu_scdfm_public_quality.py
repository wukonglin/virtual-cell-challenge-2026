import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.check_lingshu_scdfm_public_quality import check_public_quality


def fixture():
    metric = {"model_centroid_mse": 0.5, "control_centroid_mse": 1.0,
              "model_mmd2": -0.01, "control_mmd2": 0.1,
              "clamped_high_fraction": 0.01, "clamped_low_fraction": 0.7}
    return {"schema": "vcc-lingshu-scdfm-training-summary-v1", "status": "completed",
            "challenge_treated_used": False, "public_model_quality_evaluated": True,
            "official_challenge_metrics_evaluated": False,
            "best_score": 0.5, "best_step": 200, "last_step": 2000,
            "deployment_rollout_diagnostics": {
                "role": "diagnostic_only_after_checkpoint_selection",
                "checkpoint_selection_affected": False, "challenge_treated_used": False,
                "official_challenge_metrics_evaluated": False, "ode_steps": 20,
                "kind_metrics": {"context": copy.deepcopy(metric), "target": copy.deepcopy(metric)}}}


class PublicQualityPreflightTests(unittest.TestCase):
    def check(self, summary):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training_summary.json"
            path.write_text(json.dumps(summary))
            return check_public_quality(path)

    def test_passes_at_high_clamp_boundary_and_reports_ratios(self):
        result = self.check(fixture())
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["metric_ratios"]["context"]["centroid_mse"], 0.5)
        self.assertAlmostEqual(result["metric_ratios"]["target"]["mmd2"], -0.1)
        self.assertEqual(len(result["training_summary_sha256"]), 64)

    def test_strict_improvement_is_recomputed_for_both_kinds(self):
        for kind in ("context", "target"):
            for metric in ("centroid_mse", "mmd2"):
                for worsening in (0, 0.1):
                    with self.subTest(kind=kind, metric=metric, worsening=worsening):
                        summary = fixture()
                        rollout = summary["deployment_rollout_diagnostics"]
                        rollout[f"improves_control_{metric}_each_kind"] = True
                        scores = rollout["kind_metrics"][kind]
                        scores[f"model_{metric}"] = scores[f"control_{metric}"] + worsening
                        result = self.check(summary)
                        self.assertEqual(result["status"], "fail")
                        self.assertIn(f"{kind} {metric} does not improve controls", result["errors"])

    def test_rejects_incomplete_nonpublic_or_selection_affected_summary(self):
        changes = [lambda s: s.update(status="running"),
                   lambda s: s.update(challenge_treated_used=True),
                   lambda s: s.update(public_model_quality_evaluated=False),
                   lambda s: s.update(official_challenge_metrics_evaluated=True),
                   lambda s: s.update(best_step=0),
                   lambda s: s.update(best_step=2001),
                   lambda s: s.update(best_score=-0.1)]
        for key, value in (("role", "checkpoint_selection"), ("checkpoint_selection_affected", True),
                           ("challenge_treated_used", True), ("official_challenge_metrics_evaluated", True),
                           ("ode_steps", 10)):
            changes.append(lambda s, key=key, value=value: s["deployment_rollout_diagnostics"].update({key: value}))
        for change in changes:
            with self.subTest(change=change):
                summary = fixture()
                change(summary)
                self.assertEqual(self.check(summary)["status"], "fail")

    def test_requires_both_exact_kinds_and_valid_clamps(self):
        for changes in ({"clamped_high_fraction": 0.010001}, {"clamped_high_fraction": -0.1},
                        {"clamped_low_fraction": 1.1}, {"clamped_low_fraction": -0.1}):
            summary = fixture()
            summary["deployment_rollout_diagnostics"]["kind_metrics"]["context"].update(changes)
            self.assertEqual(self.check(summary)["status"], "fail")
        for kind in ("context", "target"):
            summary = fixture()
            del summary["deployment_rollout_diagnostics"]["kind_metrics"][kind]
            self.assertEqual(self.check(summary)["status"], "fail")
        summary = fixture()
        summary["deployment_rollout_diagnostics"]["kind_metrics"]["extra"] = {}
        self.assertEqual(self.check(summary)["status"], "fail")

    def test_rejects_nonfinite_negative_mse_and_non_numeric_values(self):
        for value in (float("nan"), float("inf"), -0.5, None, True, "0.2"):
            for key in ("model_centroid_mse", "control_centroid_mse"):
                with self.subTest(key=key, value=value):
                    summary = fixture()
                    summary["deployment_rollout_diagnostics"]["kind_metrics"]["target"][key] = value
                    self.assertEqual(self.check(summary)["status"], "fail")

    def test_signed_unbiased_mmd_and_zero_denominator_are_supported(self):
        for control in (0, -0.005):
            summary = fixture()
            summary["deployment_rollout_diagnostics"]["kind_metrics"]["context"]["control_mmd2"] = control
            result = self.check(summary)
            self.assertEqual(result["status"], "pass", result)
            if control == 0:
                self.assertIsNone(result["metric_ratios"]["context"]["mmd2"])

    def test_aggregates_failures_without_crashing_on_invalid_objects(self):
        summary = fixture()
        for scores in summary["deployment_rollout_diagnostics"]["kind_metrics"].values():
            scores.update(model_centroid_mse=2, model_mmd2=2)
        self.assertEqual(len(self.check(summary)["errors"]), 4)
        for malformed in (None, [], {}, {"deployment_rollout_diagnostics": []}):
            self.assertEqual(self.check(malformed)["status"], "fail")

    def test_cli_writes_report_and_failure_exit_prevents_next_stage(self):
        script = Path(__file__).resolve().parents[1] / "scripts/check_lingshu_scdfm_public_quality.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, output = root / "summary.json", root / "checks/preflight.json"
            for value, expected in ((fixture(), 0), ({}, 1)):
                summary.write_text(json.dumps(value))
                run = subprocess.run([sys.executable, str(script), "--training-summary", str(summary),
                                      "--output", str(output)], capture_output=True, text=True)
                self.assertEqual(run.returncode, expected, run.stderr)
                self.assertEqual(json.loads(output.read_text()), json.loads(run.stdout))
            original = summary.read_bytes()
            run = subprocess.run([sys.executable, str(script), "--training-summary", str(summary),
                                  "--output", str(summary)], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(summary.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
