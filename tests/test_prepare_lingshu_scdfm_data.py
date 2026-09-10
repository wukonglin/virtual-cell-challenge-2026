import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scripts.prepare_lingshu_scdfm_data import normalize_source_log1p, prepare_cache, target_split


class PrepareLingshuScdfmDataTests(unittest.TestCase):
    def test_normalizes_full_axis_without_claiming_raw_counts(self):
        abundance = np.array([[1.25, 2.5, 20.0], [12.0, 0.0, 0.5]])
        expected = np.log1p(10000 * abundance / abundance.sum(axis=1, keepdims=True))
        actual = normalize_source_log1p(sparse.csr_matrix(np.log1p(abundance)))
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        self.assertEqual(actual.dtype, np.float32)
        # Source-only genes cannot alter the denominator used at inference.
        shared = normalize_source_log1p(np.log1p(abundance), np.array([0, 1]))
        np.testing.assert_allclose(shared[:, :2], np.log1p(10000 * abundance[:, :2] / abundance[:, :2].sum(axis=1, keepdims=True)), rtol=1e-6)
        with self.assertRaises(ValueError):
            normalize_source_log1p(np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            normalize_source_log1p(np.array([[np.inf, 1.0]]))

    def fixture(self, root: Path) -> argparse.Namespace:
        targets = [f"T{index}" for index in range(5)]
        labels = ["non-targeting"] * 4 + [target for target in targets for _ in range(4)]
        abundance = np.random.default_rng(51).uniform(0.1, 10, (len(labels), 4))
        for context in ["k562", "rpe1"]:
            obs = pd.DataFrame({"target_gene": labels, "cell_type": context, "gem_group": "b1"},
                               index=[f"{context}_{i}" for i in range(len(labels))])
            matrix = np.log1p(abundance).astype(np.float32)
            if context == "rpe1":
                matrix[:, 2] = np.log1p(1e4)
            ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=["G2", "G1", "G3", "G4"])).write_h5ad(root / f"{context}.h5ad")
        with (root / "genes.csv").open("w") as stream:
            stream.write("gene_name\nG1\nG2\nG3\nG4\nG_missing\n")
        with (root / "targets.csv").open("w") as stream:
            stream.write("target_gene\nCHALLENGE_ONLY\nT2\n")
        return argparse.Namespace(k562=root / "k562.h5ad", rpe1=root / "rpe1.h5ad",
                                  official_genes=root / "genes.csv", official_targets=root / "targets.csv",
                                  gene_count=2, max_cells_per_target=3, target_holdout_fraction=0.2,
                                  seed=20260909, block_rows=3, output_npz=root / "cache.npz",
                                  output_json=root / "receipt.json", output_genes_csv=root / "embedding_genes.csv")

    def test_cache_schema_holdouts_and_validation_blind_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.fixture(root)
            receipt = prepare_cache(args)
            with np.load(args.output_npz, allow_pickle=False) as cache:
                self.assertEqual(cache["train_x"].shape, (12, 2))
                self.assertEqual(cache["val_x"].shape, (18, 2))
                for name in ["train_x", "train_control", "val_x", "val_control"]:
                    self.assertEqual(cache[name].dtype, np.float32)
                self.assertEqual(cache["train_targets"].dtype.kind, "U")
                self.assertEqual(cache["gene_indices"].dtype, np.int64)
                self.assertTrue(np.all(np.diff(cache["gene_indices"]) > 0))
                target_val = cache["val_targets"][cache["val_kind"] == "target"]
                self.assertFalse(set(target_val) & set(cache["train_targets"]))
                self.assertEqual(set(cache["train_contexts"]), {"K562"})
                self.assertEqual(set(cache["val_contexts"][cache["val_kind"] == "context"]), {"RPE1"})
                metadata = json.loads(cache["metadata_json"].item())
                self.assertFalse(metadata["challenge_treated_used"])
                self.assertFalse(metadata["normalization"]["raw_counts_reconstructed"])
                first_genes = cache["gene_names"].copy()
                self.assertEqual(len(cache["normalization_gene_names"]), 4)
            # Alter ONLY heldout target and whole heldout context values: gene
            # selection must stay identical, demonstrating no validation fit.
            heldout = set(receipt["split"]["heldout_targets"])
            k = ad.read_h5ad(args.k562)
            rows = k.obs.target_gene.isin(heldout).to_numpy()
            k.X[rows, 0] = 10.0
            k.write_h5ad(args.k562)
            r = ad.read_h5ad(args.rpe1)
            r.X[:, 1] = 15.0
            r.write_h5ad(args.rpe1)
            args.output_npz = root / "cache2.npz"
            args.output_json = root / "receipt2.json"
            args.output_genes_csv = root / "embedding_genes2.csv"
            prepare_cache(args)
            with np.load(args.output_npz, allow_pickle=False) as cache:
                np.testing.assert_array_equal(first_genes, cache["gene_names"])
            with args.output_genes_csv.open() as stream:
                gene_rows = list(csv.reader(stream))
            self.assertEqual(gene_rows[0], ["gene_name"])
            self.assertIn(["CHALLENGE_ONLY"], gene_rows)
            self.assertIn(["non-targeting"], gene_rows)
            with self.assertRaises(FileExistsError):
                prepare_cache(args)

    def test_rejects_other_context_and_duplicate_official_genes(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            k = ad.read_h5ad(args.k562)
            k.obs["cell_type"] = "hepg2"
            k.write_h5ad(args.k562)
            with self.assertRaisesRegex(ValueError, "Expected only k562"):
                prepare_cache(args)

    def test_target_split_is_order_invariant(self):
        targets = [f"T{i}" for i in range(10)]
        self.assertEqual(target_split(targets, 7), target_split(targets[::-1], 7))
        train, heldout = target_split(targets, 7)
        self.assertEqual(len(heldout), 2)
        self.assertFalse(set(train) & set(heldout))


if __name__ == "__main__":
    unittest.main()
