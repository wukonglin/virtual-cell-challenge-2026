import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.compare_lingshu_scdfm_effect_v2 import ARMS, METHOD, compare


def fixture(root):
    records = {}
    for index, arm in enumerate(ARMS):
        p = {"schema": "vcc-lingshu-scdfm-group-effect-training-provenance-v2", "method": METHOD,
             "challenge_treated_used": False, "independent_test": False,
             "selection_and_diagnostics_share_public_development_groups": True,
             "cache_sha256": "a" * 64, "embeddings_sha256": "b" * 64,
             "code_sha256": {"trainer.py": "c" * 64}, "upstream": {"commit": "d" * 40, "tree": "e" * 40},
             "embedding_receipt": {"sha256": "f" * 64, "npz_sha256": "b" * 64},
             "model_config": {"feature_mode": arm, "gene_count": 512},
             "training_config": {"debug_test_features": False, "steps": 4000, "seed": 1},
             "feature_mapping": {"mode": arm, "challenge_expression_or_labels_used": False}}
        metric = {"model_centroid_mse": 0.3 + index * 0.1, "control_centroid_mse": 1.0,
                  "model_mmd2": 0.01 + index * 0.01, "control_mmd2": 0.1,
                  "clamped_high_fraction": 0.001, "clamped_low_fraction": 0.2}
        s = {"schema": "vcc-lingshu-scdfm-training-summary-v1", "method": METHOD, "feature_mode": arm,
             "status": "completed", "challenge_treated_used": False, "independent_test": False,
             "selection_and_diagnostics_share_public_development_groups": True,
             "public_model_quality_evaluated": True, "official_challenge_metrics_evaluated": False,
             "best_step": 400 * (index + 1), "last_step": 4000, "best_score": 0.2,
             "deployment_rollout_diagnostics": {
                 "role": "diagnostic_only_after_checkpoint_selection", "checkpoint_selection_affected": False,
                 "challenge_treated_used": False, "official_challenge_metrics_evaluated": False,
                 "independent_test": False, "selection_used_same_public_development_groups": True,
                 "ode_steps": 20, "ode_solver": "euler", "cells_per_group": 16, "seed": 10, "noise_std": 0,
                 "groups": [{"kind": kind, "context": "K562" if kind == "target" else "RPE1", "target": "GENE1"}
                            for kind in ("context", "target")],
                 "kind_metrics": {kind: copy.deepcopy(metric) for kind in ("context", "target")}}}
        records[arm] = s, p
    save(root, records)
    return records


def save(root, records):
    for arm, (summary, provenance) in records.items():
        fit = root / arm / "fit"
        fit.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(provenance).encode()
        (fit / "training_provenance.json").write_bytes(payload)
        summary["training_provenance_sha256"] = hashlib.sha256(payload).hexdigest()
        (fit / "training_summary.json").write_text(json.dumps(summary))


class EffectComparisonTests(unittest.TestCase):
    def test_favorable_comparison_is_development_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            report = compare(root)
            self.assertEqual(report["status"], "pass", report)
            self.assertIn("not independent test", report["lingshu_benefit_claim"])
            self.assertAlmostEqual(report["true_minus_comparator"]["constant"]["target"]["centroid_mse"], -0.2)
            self.assertEqual(report["arms"]["constant"]["best_step"], 1200)

    def test_any_tied_or_worse_metric_prevents_benefit_claim(self):
        for arm in ("shuffled", "constant"):
            for kind in ("context", "target"):
                for metric in ("centroid_mse", "mmd2"):
                    with self.subTest(arm=arm, kind=kind, metric=metric), tempfile.TemporaryDirectory() as temporary:
                        root = Path(temporary)
                        records = fixture(root)
                        scores = records[arm][0]["deployment_rollout_diagnostics"]["kind_metrics"][kind]
                        scores[f"model_{metric}"] = records["true"][0]["deployment_rollout_diagnostics"]["kind_metrics"][kind][f"model_{metric}"]
                        save(root, records)
                        report = compare(root)
                        self.assertEqual(report["status"], "pass")
                        self.assertEqual(report["lingshu_benefit_claim"], "not established")

    def test_incomparable_config_hash_labels_or_groups_fail(self):
        mutations = [lambda s, p: p.update(cache_sha256="0" * 64),
                     lambda s, p: p.update(embeddings_sha256="0" * 64),
                     lambda s, p: p["code_sha256"].update(**{"trainer.py": "0" * 64}),
                     lambda s, p: p["upstream"].update(commit="0" * 40),
                     lambda s, p: p["model_config"].update(gene_count=64),
                     lambda s, p: p["training_config"].update(seed=2),
                     lambda s, p: s.update(feature_mode="true"),
                     lambda s, p: p["feature_mapping"].update(mode="constant"),
                     lambda s, p: s["deployment_rollout_diagnostics"]["groups"][0].update(target="GENE2"),
                     lambda s, p: s["deployment_rollout_diagnostics"].update(seed=11)]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                records = fixture(root)
                mutation(*records["shuffled"])
                save(root, records)
                report = compare(root)
                self.assertEqual(report["status"], "fail", report)
                self.assertEqual(report["lingshu_benefit_claim"], "not established")

    def test_incomplete_nonpublic_nonfinite_or_tampered_inputs_fail(self):
        mutations = [lambda s, p: s.update(status="running"),
                     lambda s, p: p.update(challenge_treated_used=True),
                     lambda s, p: s.update(independent_test=True),
                     lambda s, p: p["training_config"].update(debug_test_features=True),
                     lambda s, p: s["deployment_rollout_diagnostics"]["kind_metrics"]["target"].update(model_mmd2=float("nan"))]
        for mutation in mutations:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                records = fixture(root)
                mutation(*records["true"])
                save(root, records)
                self.assertEqual(compare(root)["status"], "fail")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            path = root / "true/fit/training_provenance.json"
            path.write_text(path.read_text() + " ")
            self.assertIn("digest mismatch", " ".join(compare(root)["errors"]))

    def test_quality_failure_is_reported_but_not_comparison_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = fixture(root)
            records["true"][0]["deployment_rollout_diagnostics"]["kind_metrics"]["context"]["model_mmd2"] = 0.2
            save(root, records)
            report = compare(root)
            self.assertEqual(report["status"], "pass", report)
            self.assertEqual(report["arms"]["true"]["public_quality_status"], "fail")
            self.assertTrue(report["arms"]["true"]["public_quality_errors"])
            self.assertEqual(report["lingshu_benefit_claim"], "not established")

    def test_cli_reports_missing_arm_without_overwriting_input(self):
        script = Path(__file__).resolve().parents[1] / "scripts/compare_lingshu_scdfm_effect_v2.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            (root / "constant/fit/training_summary.json").unlink()
            output = root / "comparison.json"
            run = subprocess.run([sys.executable, str(script), "--root", str(root), "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(output.read_text()), json.loads(run.stdout))
            source = root / "true/fit/training_summary.json"
            original = source.read_bytes()
            run = subprocess.run([sys.executable, str(script), "--root", str(root), "--output", str(source)], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
