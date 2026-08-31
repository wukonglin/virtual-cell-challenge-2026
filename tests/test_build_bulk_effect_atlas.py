"""Focused tests for pseudobulk effect-atlas construction."""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_bulk_effect_atlas import (  # noqa: E402
    GLOBAL_BATCH,
    REQUIRED_NPZ_KEYS,
    build_effect_atlas,
    inspect_matrix,
    parse_condition_name,
    resolve_bulk_schema,
    resolve_gene_axis,
)


def make_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "target_col": "target_gene",
        "guide_col": "guide_id",
        "batch_col": "batch",
        "cell_count_col": "n_cells",
        "control_selection_col": "selected_control",
        "gene_col": "gene_name",
        "control": "non-targeting",
        "min_cells": 1,
        "target_sum": 100.0,
        "chunk_rows": 2,
        "matrix_semantics": "auto",
        "count_reconstruction_tolerance": 0.02,
        "missing_control_batch": "global",
        "max_targets": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def write_synthetic_bulk(path: Path) -> None:
    matrix = np.asarray(
        [
            [1.2, 0.8, 2.0],
            [2.0, 1.0, 4.0],
            [3.1, 0.9, 2.0],
            [4.0, 2.0, 6.0],
            [1.2, 0.8, 8.0],
            [9.0, 9.0, 9.0],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {
            "target_gene": [
                "non-targeting",
                "non-targeting",
                "T1",
                "T1",
                "T2",
                "non-targeting",
            ],
            "guide_id": ["nt1", "nt2", "g1", "g2", "g3", "bad"],
            "batch": ["b1", "b2", "b1", "b2", "b3", "b1"],
            "n_cells": [5.0, 10.0, 10.0, 20.0, 30.0, np.nan],
            "selected_control": [True, True, False, False, False, False],
        },
        index=pd.Index(
            [
                "0_non-targeting_non-targeting_non-targeting",
                "1_non-targeting_non-targeting_non-targeting",
                "2_T1_P1_ENSG1",
                "3_T1_P2_ENSG1",
                "4_T2_P1P2_ENSG2",
                "5_non-targeting_non-targeting_non-targeting",
            ],
            name="condition",
        ),
    )
    var = pd.DataFrame(
        {"gene_name": ["g0", "g0", "g1"]},
        index=pd.Index(["id0", "id0b", "id1"], name="gene_id"),
    )
    ad.AnnData(X=matrix, obs=obs, var=var).write_h5ad(path)


class ConditionParsingTests(unittest.TestCase):
    def test_parser_preserves_guide_underscores(self) -> None:
        parsed = parse_condition_name(
            "10598_ZNF735_NM_001159524_ENSG00000223614", "non-targeting"
        )
        self.assertEqual(parsed, ("ZNF735", "NM_001159524", "ENSG00000223614"))

    def test_parser_recognizes_control_encoding(self) -> None:
        parsed = parse_condition_name(
            "10747_non-targeting_non-targeting_non-targeting", "non-targeting"
        )
        self.assertEqual(parsed, ("non-targeting",) * 3)


class BulkAtlasIntegrationTests(unittest.TestCase):
    def test_cell_weighted_guide_pooling_and_batch_matching(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bulk.h5ad"
            write_synthetic_bulk(path)
            data = ad.read_h5ad(path, backed="r")
            try:
                args = make_args()
                schema = resolve_bulk_schema(data, args)
                genes = resolve_gene_axis(data, args)
                semantics, matrix_report = inspect_matrix(data, schema.cell_counts, args)
                payload, report = build_effect_atlas(
                    data, schema, genes, semantics, args
                )
            finally:
                data.file.close()

        self.assertEqual(semantics, "per-cell-mean")
        self.assertEqual(matrix_report["reconstructed_noninteger_values"], 0)
        self.assertEqual(tuple(payload), REQUIRED_NPZ_KEYS)
        np.testing.assert_array_equal(payload["gene_names"], np.asarray(["g0", "g1"]))
        np.testing.assert_array_equal(payload["target_names"], np.asarray(["T1", "T2"]))
        np.testing.assert_array_equal(payload["target_cell_counts"], np.asarray([30, 30]))
        np.testing.assert_array_equal(payload["batch_names"], np.asarray(["b1", "b2", "b3"]))
        np.testing.assert_array_equal(
            payload["target_batch_counts"], np.asarray([[10, 20, 0], [0, 0, 30]])
        )
        np.testing.assert_array_equal(payload["control_batch_counts"], np.asarray([5, 10, 0]))

        target_one_profile = np.log1p(100.0 * np.asarray([160.0, 140.0]) / 300.0)
        matched_profile = np.log1p(100.0 * np.asarray([80.0, 100.0]) / 180.0)
        np.testing.assert_allclose(
            payload["effects"][0], target_one_profile - matched_profile, atol=1e-6
        )
        np.testing.assert_allclose(
            payload["matched_control_profiles"][0], matched_profile, atol=1e-6
        )
        np.testing.assert_allclose(
            payload["matched_control_profiles"][1], matched_profile, atol=1e-6
        )
        self.assertEqual(report["guides_per_target"]["targets_with_multiple_guides"], 1)
        self.assertEqual(report["missing_control_batches_using_global_mean"], ["b3"])
        self.assertEqual(genes.duplicate_symbols, {"g0": 2})

    def test_missing_control_batch_can_be_a_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bulk.h5ad"
            write_synthetic_bulk(path)
            data = ad.read_h5ad(path, backed="r")
            try:
                args = make_args(missing_control_batch="error")
                schema = resolve_bulk_schema(data, args)
                genes = resolve_gene_axis(data, args)
                semantics, _ = inspect_matrix(data, schema.cell_counts, args)
                with self.assertRaisesRegex(ValueError, "no controls"):
                    build_effect_atlas(data, schema, genes, semantics, args)
            finally:
                data.file.close()

    def test_absent_batch_column_uses_global_control_stratum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bulk.h5ad"
            write_synthetic_bulk(path)
            data = ad.read_h5ad(path, backed="r")
            try:
                args = make_args(batch_col="none")
                schema = resolve_bulk_schema(data, args)
                genes = resolve_gene_axis(data, args)
                semantics, _ = inspect_matrix(data, schema.cell_counts, args)
                payload, _ = build_effect_atlas(data, schema, genes, semantics, args)
            finally:
                data.file.close()
        self.assertEqual(schema.batch_source, "global-fallback")
        np.testing.assert_array_equal(payload["batch_names"], np.asarray([GLOBAL_BATCH]))
        np.testing.assert_array_equal(payload["control_batch_counts"], np.asarray([15]))

    def test_integer_matrix_auto_resolves_to_summed_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bulk.h5ad"
            write_synthetic_bulk(path)
            data = ad.read_h5ad(path)
            data.X = np.rint(data.X).astype(np.int32)
            data.write_h5ad(path)
            backed = ad.read_h5ad(path, backed="r")
            try:
                args = make_args()
                schema = resolve_bulk_schema(backed, args)
                semantics, _ = inspect_matrix(backed, schema.cell_counts, args)
            finally:
                backed.file.close()
        self.assertEqual(semantics, "summed-counts")

    def test_negative_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bulk.h5ad"
            write_synthetic_bulk(path)
            data = ad.read_h5ad(path)
            data.X[0, 0] = -1.0
            data.write_h5ad(path)
            backed = ad.read_h5ad(path, backed="r")
            try:
                args = make_args()
                schema = resolve_bulk_schema(backed, args)
                with self.assertRaisesRegex(ValueError, "negative"):
                    inspect_matrix(backed, schema.cell_counts, args)
            finally:
                backed.file.close()

    def test_sparse_backed_bulk_matrix_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bulk.h5ad"
            write_synthetic_bulk(path)
            data = ad.read_h5ad(path)
            data.X = sp.csr_matrix(data.X)
            data.write_h5ad(path)
            backed = ad.read_h5ad(path, backed="r")
            try:
                args = make_args()
                schema = resolve_bulk_schema(backed, args)
                genes = resolve_gene_axis(backed, args)
                semantics, _ = inspect_matrix(backed, schema.cell_counts, args)
                payload, _ = build_effect_atlas(
                    backed, schema, genes, semantics, args
                )
            finally:
                backed.file.close()
        self.assertEqual(semantics, "per-cell-mean")
        self.assertEqual(payload["effects"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
