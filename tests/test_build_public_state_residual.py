"""Unit tests for leakage-safe public STATE residual construction."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_public_state_residual import (  # noqa: E402
    combine_centered_sources,
    count_reliability,
    load_aligned_atlas,
    read_control_statistics,
    validate_frozen_file,
)


class PublicStateResidualTests(unittest.TestCase):
    def test_held_targets_are_zero_and_do_not_enter_source_center(self) -> None:
        k562 = np.asarray([[1.0, 2.0], [3.0, 6.0], [100.0, 200.0]], dtype=np.float32)
        rpe1 = np.asarray([[2.0, 4.0], [4.0, 8.0], [200.0, 400.0]], dtype=np.float32)
        reliability = np.ones(3, dtype=np.float32)
        direct = np.asarray([True, True, False])
        combined, joint, centers = combine_centered_sources(
            k562, rpe1, reliability, reliability, direct
        )
        np.testing.assert_allclose(centers["k562_center"], [2.0, 4.0])
        np.testing.assert_allclose(centers["rpe1_center"], [3.0, 6.0])
        np.testing.assert_allclose(combined[0], [-1.0, -2.0])
        np.testing.assert_allclose(combined[1], [1.0, 2.0])
        np.testing.assert_array_equal(combined[2], 0)
        np.testing.assert_array_equal(joint, 1)

    def test_named_atlas_alignment_is_order_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atlas.npz"
            np.savez_compressed(
                path,
                effects=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                target_names=np.asarray(["B", "A"]),
                gene_names=np.asarray(["g2", "g0", "g1"]),
                target_cell_counts=np.asarray([20, 10]),
            )
            effects, counts, axes = load_aligned_atlas(path, ["A", "B"], ["g0", "g2"])
            np.testing.assert_array_equal(effects, [[5, 4], [2, 1]])
            np.testing.assert_array_equal(counts, [10, 20])
            self.assertEqual(axes, {"source_targets": 2, "source_genes": 3})

    def test_controls_firewall_rejects_treated_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not_controls_only.h5ad"
            obs = pd.DataFrame(
                {"target_gene": pd.Categorical(["non-targeting", "GENE1"])},
                index=["c0", "c1"],
            )
            var = pd.DataFrame(
                {"is_state_support": [True, False]}, index=["g0", "g1"]
            )
            ad.AnnData(sp.csr_matrix([[1, 0], [0, 1]]), obs=obs, var=var).write_h5ad(path)
            with self.assertRaisesRegex(RuntimeError, "leaked"):
                read_control_statistics(path, chunk_size=1)

    def test_controls_require_integer_counts_and_honor_context_and_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controls.h5ad"
            obs = pd.DataFrame(
                {
                    "target_gene": pd.Categorical(["non-targeting"]),
                    "context": pd.Categorical(["HepG2"]),
                },
                index=["c0"],
            )
            var = pd.DataFrame(
                {
                    "is_state_support": [True, False],
                    "is_common_public_gene": [True, True],
                },
                index=["g0", "g1"],
            )
            data = ad.AnnData(
                sp.csr_matrix(np.asarray([[1.0, 1.5]], dtype=np.float32)),
                obs=obs,
                var=var,
            )
            data.uns["public_validation"] = {
                "schema": "vcc-public-validation-data-v1",
                "role": "controls-only-generator-input",
                "sealed_treated_profiles_present": False,
            }
            data.write_h5ad(path)
            with self.assertRaisesRegex(RuntimeError, "non-integer"):
                read_control_statistics(
                    path,
                    chunk_size=1,
                    expected_context="HepG2",
                    require_public_role=True,
                )

            data.X = sp.csr_matrix(np.asarray([[1, 2]], dtype=np.int32))
            data.write_h5ad(path)
            with self.assertRaisesRegex(RuntimeError, "requested context"):
                read_control_statistics(
                    path,
                    chunk_size=1,
                    expected_context="Jurkat",
                    require_public_role=True,
                )

    def test_frozen_file_validation_rejects_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.csv"
            path.write_text("gene_name\ng0\n", encoding="utf-8")
            import hashlib

            recorded = {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            validate_frozen_file(path, recorded, "Synthetic")
            path.write_text("gene_name\ng1\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash changed"):
                validate_frozen_file(path, recorded, "Synthetic")

    def test_count_reliability_is_monotonic_and_bounded(self) -> None:
        values = count_reliability(np.asarray([0, 60, 600]), 60.0)
        np.testing.assert_allclose(values, [0.0, 0.5, 10.0 / 11.0])
        self.assertTrue(np.all(np.diff(values) > 0))


if __name__ == "__main__":
    unittest.main()
