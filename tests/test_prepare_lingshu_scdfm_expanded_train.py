"""Training-only expansion preserves parent validation and normalization."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd

from scripts.prepare_lingshu_scdfm_data import normalize_source_log1p, prepare_cache, sha256_file
from scripts.prepare_lingshu_scdfm_expanded_train import (
    AXES, expanded_rows, parse_args, prepare, split_half_diagnostic, verify_existing,
)


class ExpandedTrainingTests(unittest.TestCase):
    def fixture(self, root):
        labels = ["non-targeting"] * 12 + [f"T{i}" for i in range(6) for _ in range(10)]
        genes = ["G2", "G1", "G3", "PUBLIC_ONLY"]
        abundance = np.random.default_rng(551).uniform(0.1, 10, (len(labels), len(genes)))
        for context in ("k562", "rpe1"):
            obs = pd.DataFrame({"target_gene": labels, "cell_type": context,
                                "gem_group": [f"batch{i % 2}" for i in range(len(labels))]},
                               index=[f"{context}_{i}" for i in range(len(labels))])
            ad.AnnData(np.log1p(abundance).astype(np.float32), obs=obs,
                       var=pd.DataFrame(index=genes)).write_h5ad(root / f"{context}.h5ad")
        (root / "official_genes.csv").write_text("gene_name\nG1\nG2\nG3\nCHALLENGE_ONLY\n")
        (root / "targets.csv").write_text("target_gene\nT2\n")
        (root / "public_genes.csv").write_text("\n".join(genes) + "\n")
        base_args = argparse.Namespace(k562=root / "k562.h5ad", rpe1=root / "rpe1.h5ad",
            official_genes=root / "official_genes.csv", official_targets=root / "targets.csv",
            gene_count=2, max_cells_per_target=3, target_holdout_fraction=0.2, seed=20260909,
            block_rows=4, output_npz=root / "base.npz", output_json=root / "base.json",
            output_genes_csv=root / "embedding_genes.csv")
        prepare_cache(base_args)
        return parse_args(["--base-cache", str(base_args.output_npz), "--k562", str(base_args.k562),
            "--public-genes", str(root / "public_genes.csv"), "--output-npz", str(root / "expanded.npz"),
            "--output-json", str(root / "expanded.json"), "--max-train-cells", "8",
            "--max-controls-per-batch", "8", "--block-rows", "5"])

    def test_expands_only_training_freezes_all_validation_and_axes(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            before = sha256_file(args.base_cache)
            real_read = ad.read_h5ad
            with mock.patch("scripts.prepare_lingshu_scdfm_expanded_train.ad.read_h5ad", wraps=real_read) as reader:
                receipt = prepare(args)
            self.assertEqual([call.args[0] for call in reader.call_args_list], [args.k562])
            self.assertEqual(sha256_file(args.base_cache), before)
            with np.load(args.base_cache, allow_pickle=False) as parent, np.load(args.output_npz, allow_pickle=False) as new:
                self.assertEqual(parent["train_x"].shape, (12, 2))
                self.assertEqual(new["train_x"].shape, (32, 2))
                for name in set(AXES) | {key for key in parent.files if key.startswith("val_")}:
                    self.assertEqual(parent[name].dtype, new[name].dtype)
                    self.assertEqual(parent[name].tobytes(), new[name].tobytes())
                self.assertEqual(set(parent["train_targets"]), set(new["train_targets"]))
                self.assertTrue(set(parent["train_source_rows"]) <= set(new["train_source_rows"]))
                source = real_read(args.k562)
                indices = {gene: i for i, gene in enumerate(source.var_names)}
                norm = np.asarray([indices[name] for name in parent["normalization_gene_names"]])
                columns = np.asarray([indices[name] for name in parent["gene_names"]])
                expected = normalize_source_log1p(source.X[new["train_source_rows"]], norm)[:, columns]
                np.testing.assert_array_equal(new["train_x"], expected)
                self.assertTrue(np.all(source.obs.target_gene.to_numpy()[new["train_control_source_rows"]] == "non-targeting"))
                self.assertTrue(np.all(source.obs.gem_group.to_numpy()[new["train_control_source_rows"]] ==
                                       source.obs.gem_group.to_numpy()[new["train_source_rows"]]))
            self.assertFalse(receipt["challenge_treated_used"])
            self.assertFalse(receipt["training_gene_selection"])
            self.assertFalse(receipt["expansion"]["training_only_noise_diagnostic"]["used_for_selection_or_threshold_tuning"])
            # Portable verification must not open either source expression file.
            with mock.patch("scripts.prepare_lingshu_scdfm_expanded_train.ad.read_h5ad", side_effect=AssertionError("source read")):
                checked = verify_existing(args.base_cache, args.output_npz, args.output_json)
            self.assertEqual(checked["status"], "pass")
            self.assertEqual(checked["training_targets"], 4)
            with self.assertRaises(FileExistsError):
                prepare(args)

    def test_source_hash_failure_precedes_expression_read(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            args.k562.write_bytes(args.k562.read_bytes() + b"changed")
            with mock.patch("scripts.prepare_lingshu_scdfm_expanded_train.ad.read_h5ad", side_effect=AssertionError("source read")):
                with self.assertRaisesRegex(ValueError, "source hash"):
                    prepare(args)
            self.assertFalse(args.output_npz.exists())

    def test_wrong_public_axis_and_reduced_cap_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            args.public_genes.write_text("G1\nG2\nG3\nPUBLIC_ONLY\n")
            with self.assertRaisesRegex(ValueError, "gene CSV"):
                prepare(args)
            args.public_genes.write_text("G2\nG1\nG3\nPUBLIC_ONLY\n")
            args.max_train_cells = 2
            with self.assertRaisesRegex(ValueError, "discard parent"):
                prepare(args)

    def test_verifier_rejects_modified_validation_even_with_updated_npz_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            receipt = prepare(args)
            with np.load(args.output_npz, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["val_x"][0, 0] += 0.125
            with args.output_npz.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            receipt["artifacts"]["cache"]["sha256"] = sha256_file(args.output_npz)
            args.output_json.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "Frozen parent array changed: val_x"):
                verify_existing(args.base_cache, args.output_npz, args.output_json)

    def test_nested_sampling_is_deterministic_and_retains_parent_rows(self):
        labels = np.asarray(["A"] * 10 + ["B"] * 10)
        parent = np.asarray([1, 3, 12, 17], dtype=np.int64)
        rows = expanded_rows(labels, ["A", "B"], parent, 7, 51)
        np.testing.assert_array_equal(rows, expanded_rows(labels, ["B", "A"], parent, 7, 51))
        self.assertTrue(set(parent) <= set(rows))
        self.assertEqual(len(rows), 14)

    def test_split_half_is_deterministic_training_only_description(self):
        x = np.arange(32, dtype=np.float32).reshape(8, 4)
        controls = x - 2
        labels = np.asarray(["A"] * 4 + ["B"] * 4)
        diagnostic = split_half_diagnostic(x, controls, labels, 7)
        self.assertEqual(diagnostic["equal_target_mean_mse"], 0)
        self.assertEqual(diagnostic, split_half_diagnostic(x, controls, labels, 7))


if __name__ == "__main__":
    unittest.main()
