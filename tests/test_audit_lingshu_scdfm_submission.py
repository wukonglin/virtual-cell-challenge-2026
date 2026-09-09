import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_lingshu_scdfm_submission import MODEL_ID, REVISION, audit, digest


def write_json(path, value):
    path.write_text(json.dumps(value))


def fixture(root):
    (root / "fit").mkdir()
    for name in ("fit/best.pt", "lingshu_embeddings.npz", "prediction.h5ad", "prediction.vcc"):
        (root / name).write_bytes(name.encode())
    e = {"status": "complete", "model_id": MODEL_ID, "model_revision": REVISION,
         "embedding_dimension": 3584, "challenge_treated_data_used": False,
         "npz": {"sha256": digest(root / "lingshu_embeddings.npz")}}
    write_json(root / "lingshu_embeddings.json", e)
    p = {"schema": "vcc-lingshu-scdfm-training-provenance-v1", "challenge_treated_used": False,
         "training_config": {"debug_test_features": False}, "model_config": {"gene_count": 512},
         "embeddings_sha256": e["npz"]["sha256"], "embedding_receipt_sha256": digest(root / "lingshu_embeddings.json"),
         "embedding_receipt": {"sha256": digest(root / "lingshu_embeddings.json")}}
    write_json(root / "fit/training_provenance.json", p)
    metric = {"model_centroid_mse": 0.5, "control_centroid_mse": 1.0,
              "model_mmd2": -0.01, "control_mmd2": 0.1,
              "clamped_high_fraction": 0.01, "clamped_low_fraction": 0.7}
    rollout = {"checkpoint_selection_affected": False, "challenge_treated_used": False, "ode_steps": 20,
               "kind_metrics": {"context": copy.deepcopy(metric), "target": copy.deepcopy(metric)}}
    write_json(root / "fit/best_rollout_diagnostics.json", rollout)
    s = {"schema": "vcc-lingshu-scdfm-training-summary-v1", "status": "completed", "challenge_treated_used": False,
         "public_model_quality_evaluated": True, "official_challenge_metrics_evaluated": False,
         "best_checkpoint_sha256": digest(root / "fit/best.pt"), "training_provenance_sha256": digest(root / "fit/training_provenance.json"),
         "best_score": 0.5, "best_step": 200, "last_step": 2000, "deployment_rollout_diagnostics": rollout}
    write_json(root / "fit/training_summary.json", s)
    g = {"schema": "vcc-lingshu-scdfm-generation-receipt-v1", "status": "completed", "challenge_treated_used": False,
         "test_subset": False, "manual_target_effects": False, "modeled_gene_count": 512, "normalization_gene_count": 18077,
         "shape": [360000, 18533], "nnz": 5000000, "checkpoint_sha256": s["best_checkpoint_sha256"],
         "training_summary_sha256": digest(root / "fit/training_summary.json"), "prediction_sha256": digest(root / "prediction.h5ad"),
         "embeddings_sha256": e["npz"]["sha256"], "clamping": {"modeled_values": 360000 * 512, "clamped_low": 100000000, "clamped_high": 0},
         "group_receipts": [{"context": context, "target_gene": f"T{i}", "library_sum_ratio": 1.0,
                             "input_library_sum": 1000, "output_library_sum": 1000} for context in "ABC" for i in range(300)]}
    write_json(root / "prediction.h5ad.receipt.json", g)
    for name, dry in (("prep_validation", True), ("prep_package", False)):
        write_json(root / f"{name}.json", {"dry_run": dry, "verified_targets": True, "n_cells": 360000,
                   "n_genes": 18533, "normalization": "counts-preserved", "cells_per_context": {c: 120000 for c in "ABC"},
                   "nnz": g["nnz"], "input": str(root / "prediction.h5ad"), "output": None if dry else str(root / "prediction.vcc")})
    (root / "prediction.vcc.sha256").write_text(digest(root / "prediction.vcc") + "  prediction.vcc\n")
    return s, p, g


class AuditLingshuSubmissionTests(unittest.TestCase):
    def test_passes_and_only_reports_high_low_clamping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            result = audit(root)
            self.assertEqual(result["status"], "pass", result)
            self.assertGreater(result["generation_low_clamp_fraction"], 0.5)

    def test_failed_or_nonfinite_improvement_is_rejected(self):
        for value in (0.1, 0.2, float("nan")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                s, _, g = fixture(root)
                s["deployment_rollout_diagnostics"]["kind_metrics"]["target"]["model_mmd2"] = value
                write_json(root / "fit/best_rollout_diagnostics.json", s["deployment_rollout_diagnostics"])
                write_json(root / "fit/training_summary.json", s)
                g["training_summary_sha256"] = digest(root / "fit/training_summary.json")
                write_json(root / "prediction.h5ad.receipt.json", g)
                self.assertEqual(audit(root)["status"], "fail")

    def test_test_output_bad_clamp_library_and_duplicate_groups_fail(self):
        changes = [lambda g: g.update(test_subset=True),
                   lambda g: g["clamping"].update(clamped_high=2000000),
                   lambda g: g["group_receipts"][0].update(library_sum_ratio=2.1, output_library_sum=2100),
                   lambda g: g["group_receipts"].__setitem__(0, g["group_receipts"][1])]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, _, g = fixture(root)
                change(g)
                write_json(root / "prediction.h5ad.receipt.json", g)
                self.assertEqual(audit(root)["status"], "fail")

    def test_tampered_payload_and_failed_prep_fail(self):
        for name in ("fit/best.pt", "prediction.h5ad", "prediction.vcc", "lingshu_embeddings.npz", "prep_validation.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture(root)
                (root / name).write_text("{}")
                self.assertEqual(audit(root)["status"], "fail")

    def test_synthetic_training_is_rejected_even_with_updated_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            s, p, g = fixture(root)
            p["training_config"]["debug_test_features"] = True
            write_json(root / "fit/training_provenance.json", p)
            s["training_provenance_sha256"] = digest(root / "fit/training_provenance.json")
            write_json(root / "fit/training_summary.json", s)
            g["training_summary_sha256"] = digest(root / "fit/training_summary.json")
            write_json(root / "prediction.h5ad.receipt.json", g)
            self.assertIn("Synthetic", audit(root)["errors"][0])


if __name__ == "__main__":
    unittest.main()
