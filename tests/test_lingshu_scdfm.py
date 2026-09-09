"""Focused data-leakage, numerical and output tests for the new PAD adapter."""
from __future__ import annotations

import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from lingshu_scdfm_model import load_cache, load_embeddings, mmd2_unbiased, integrate, LingshuScDFM, authenticate_embedding_receipt
from generate_lingshu_scdfm import replace_modeled_counts, StreamingH5AD
from train_lingshu_scdfm import validation_groups


def cache_arrays():
    return {"gene_names": np.array(["G1", "G3"]), "gene_indices": np.array([0, 2]),
            "normalization_gene_names": np.array(["G1", "G2", "G3"]),
            "train_x": np.array([[1, 2], [2, 1]], dtype=np.float32),
            "train_control": np.array([[1, 1], [1, 1]], dtype=np.float32),
            "train_targets": np.array(["T1", "T1"]), "train_contexts": np.array(["k562", "k562"]),
            "val_x": np.array([[1, 2], [2, 1]], dtype=np.float32),
            "val_control": np.array([[1, 1], [1, 1]], dtype=np.float32),
            "val_targets": np.array(["T1", "T2"]), "val_contexts": np.array(["rpe1", "k562"]),
            "val_kind": np.array(["context", "target"]),
            "metadata_json": np.array(json.dumps({"challenge_treated_used": False,
                "normalization": {"space": "log1p_library_normalized_full_axis", "target_sum": 10000,
                                  "full_axis_scope": "shared_public_challenge_gene_axis"}}))}


class CacheContractTests(unittest.TestCase):
    def test_valid_cache_and_both_validation_kinds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.npz"
            np.savez(path, **cache_arrays())
            data, digest = load_cache(path)
            self.assertEqual(len(digest), 64)
            self.assertEqual({key[0] for key, _ in validation_groups(data, 1)}, {"context", "target"})

    def test_rejects_context_or_target_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            for label, values in (("val_contexts", ["k562", "k562"]), ("val_targets", ["T1", "T1"])):
                data = cache_arrays()
                data[label] = np.array(values)
                path = Path(folder) / "cache.npz"
                np.savez(path, **data)
                with self.assertRaisesRegex(ValueError, "leaks"):
                    load_cache(path)

    def test_rejects_missing_normalization_axis_and_nonfinite_expression(self):
        with tempfile.TemporaryDirectory() as folder:
            for bad in ("axis", "nan"):
                data = cache_arrays()
                if bad == "axis":
                    del data["normalization_gene_names"]
                else:
                    data["train_x"][0, 0] = np.nan
                path = Path(folder) / "cache.npz"
                np.savez(path, **data)
                with self.assertRaises(ValueError):
                    load_cache(path)

    def test_rejects_duplicate_target_embeddings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "features.npz"
            np.savez(path, gene_names=np.array(["A", "B"]), embeddings=np.ones((2, 4), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "identical embedding"):
                load_embeddings(path)

    def test_embedding_receipt_rejects_changed_identity_bytes_and_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            npz = root / "embeddings.npz"
            np.savez(npz, gene_names=np.array(["A", "B"]), embeddings=np.eye(2, 3584, dtype=np.float32))
            features, digest = load_embeddings(npz)
            receipt = {"schema": "vcc-lingshu-frozen-gene-embeddings-v1", "status": "complete",
                       "model_id": "lingshu-medical-mllm/Lingshu-7B", "model_revision": "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9",
                       "embedding_dimension": 3584, "gene_count": 2, "gene_names": ["A", "B"],
                       "role": "frozen_symbol_prompt_language_features", "challenge_treated_data_used": False,
                       "generated_annotations_used": False, "pooling": "last_hidden_state_attention_masked_mean_float32_then_l2",
                       "npz": {"sha256": digest, "size_bytes": npz.stat().st_size, "keys": ["gene_names", "embeddings"]}}
            path = root / "receipt.json"
            path.write_text(json.dumps(receipt))
            authenticate_embedding_receipt(path, npz, features, digest)
            for key, value in (("model_revision", "wrong"), ("embedding_dimension", 512), ("gene_names", ["B", "A"])):
                changed = dict(receipt)
                changed[key] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    authenticate_embedding_receipt(path, npz, features, digest)
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "NPZ bytes"):
                authenticate_embedding_receipt(path, npz, features, "0" * 64)


class NumericalTests(unittest.TestCase):
    def test_mmd_gradient_finite_for_constant_target(self):
        prediction = torch.ones(3, 4, requires_grad=True)
        value = mmd2_unbiased(prediction, torch.zeros(3, 4))
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_integrator_matches_constant_velocity_and_is_repeatable(self):
        class Constant(torch.nn.Module):
            def forward(self, x, control, features, time):
                return torch.ones_like(x) * 2
        control = torch.ones(2, 3)
        outputs = [integrate(Constant(), control, torch.zeros(2, 4), steps=4, noise_std=0,
                             generator=torch.Generator().manual_seed(7)) for _ in range(2)]
        self.assertTrue(torch.equal(outputs[0], torch.full((2, 3), 3.0)))
        self.assertTrue(torch.equal(*outputs))

    def test_emitter_preserves_unmodeled_counts_and_integer_domain(self):
        raw = sparse.csr_matrix([[2, 11, 3, 17], [7, 19, 5, 23]], dtype=np.int32)
        log = np.log1p(np.array([[4, 6], [8, 10]]) * 10000 / np.array([33, 54])[:, None])
        result, stats = replace_modeled_counts(raw, log, np.array([0, 2]), np.array([33, 54]), np.random.default_rng(0))
        np.testing.assert_array_equal(result[:, [1, 3]].toarray(), raw[:, [1, 3]].toarray())
        np.testing.assert_array_equal(result[:, [0, 2]].toarray(), [[4, 6], [8, 10]])
        self.assertEqual(result.dtype, np.int32)
        self.assertEqual(stats["clamped_low"], 0)

    def test_writer_roundtrips_csr_and_axis(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prediction.h5ad"
            writer = StreamingH5AD(path, 3, 4)
            blocks = [sparse.csr_matrix([[0, 1, 0, 2], [3, 0, 0, 0]]), sparse.csr_matrix([[0, 0, 4, 0]])]
            try:
                for block in blocks:
                    writer.append(block)
                writer.finish(pd.DataFrame({"target_gene": ["A", "B", "C"]}, index=["a", "b", "c"]),
                              pd.DataFrame(index=["g1", "g2", "g3", "g4"]), {"test": True})
            finally:
                writer.close()
            obj = ad.read_h5ad(path)
            np.testing.assert_array_equal(obj.X.toarray(), sparse.vstack(blocks).toarray())
            self.assertEqual(obj.obs_names.tolist(), ["a", "b", "c"])


@unittest.skipUnless(importlib.util.find_spec("timm") and importlib.util.find_spec("torchvision"), "official PAD dependencies unavailable")
class OfficialPADIntegration(unittest.TestCase):
    def test_official_pad_backward_projection_trainable(self):
        source = SCRIPTS.parent / "external/scdfm_lingshu_2cf6bca1"
        if not source.exists():
            self.skipTest("Pinned official source unavailable")
        torch.set_num_threads(1)
        config = {"gene_count": 4, "embedding_dimension": 6, "hidden_size": 16, "heads": 2, "layers": 1, "dropout": 0.0}
        model = LingshuScDFM(config, source)
        output = model(torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 6), torch.rand(2))
        self.assertEqual(tuple(output.shape), (2, 4))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(model.target_projection[0].weight.grad).all())
        self.assertGreater(float(model.target_projection[0].weight.grad.abs().sum()), 0)
        self.assertTrue(type(model.pad).__module__.endswith("models.origin.model"))

    def test_train_resume_matches_uninterrupted_and_controls_only_generation(self):
        from train_lingshu_scdfm import parser as train_parser, train
        from generate_lingshu_scdfm import parser as generation_parser, generate
        source = SCRIPTS.parent / "external/scdfm_lingshu_2cf6bca1"
        if not source.exists():
            self.skipTest("Pinned official source unavailable")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(root / "cache.npz", **cache_arrays())
            np.savez(root / "features.npz", gene_names=np.array(["T1", "T2"]),
                     embeddings=np.eye(2, 6, dtype=np.float32))
            common = ["--cache", str(root / "cache.npz"), "--embeddings", str(root / "features.npz"),
                      "--source", str(source), "--validate-every", "1", "--batch-size", "2",
                      "--validation-batch-size", "2", "--hidden-size", "16", "--heads", "2",
                      "--layers", "1", "--cpu", "--debug-test-features"]
            with redirect_stdout(io.StringIO()):
                train(train_parser().parse_args(common + ["--output-dir", str(root / "first"), "--steps", "2"]))
                train(train_parser().parse_args(common + ["--output-dir", str(root / "resumed"), "--steps", "4",
                                                          "--resume", str(root / "first/last.pt")]))
                train(train_parser().parse_args(common + ["--output-dir", str(root / "full"), "--steps", "4"]))
            left = torch.load(root / "resumed/last.pt", weights_only=True)
            right = torch.load(root / "full/last.pt", weights_only=True)
            for key in left["model_state"]:
                self.assertTrue(torch.equal(left["model_state"][key], right["model_state"][key]), key)
            self.assertEqual(left["validation"], right["validation"])
            pd.DataFrame({"gene_name": ["G1", "G2", "G3"]}).to_csv(root / "genes.csv", index=False)
            pd.DataFrame({"target_gene": ["T1", "T2"]}).to_csv(root / "targets.csv", index=False)
            raw = sparse.csr_matrix([[10, 20, 30], [20, 30, 40]], dtype=np.int32)
            ad.AnnData(raw, obs=pd.DataFrame({"context": ["A", "A"], "target_gene": ["non-targeting"] * 2},
                                           index=["control1", "control2"]),
                       var=pd.DataFrame(index=["G1", "G2", "G3"])).write_h5ad(root / "controls.h5ad")
            args = generation_parser().parse_args([
                "--checkpoint", str(root / "full/best.pt"), "--embeddings", str(root / "features.npz"),
                "--source", str(source), "--controls", f"A={root / 'controls.h5ad'}",
                "--gene-axis", str(root / "genes.csv"), "--targets", str(root / "targets.csv"),
                "--output", str(root / "prediction.h5ad"), "--cells-per-target", "2", "--batch-size", "2",
                "--ode-steps", "2", "--cpu", "--test-subset"])
            with redirect_stdout(io.StringIO()):
                receipt = generate(args)
            result = ad.read_h5ad(root / "prediction.h5ad")
            self.assertEqual(result.shape, (4, 3))
            self.assertEqual(receipt["status"], "completed")
            self.assertTrue(receipt["test_subset"])
            expected = raw[result.obs["source_control_row"].to_numpy(), 1].toarray()
            np.testing.assert_array_equal(result.X[:, 1].toarray(), expected)


if __name__ == "__main__":
    unittest.main()
