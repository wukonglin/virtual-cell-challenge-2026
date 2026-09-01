"""Integration tests for sealed public scoring-view preparation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_public_scoring_views as module  # noqa: E402


def _data(
    matrix: np.ndarray,
    labels: list[str],
    prefix: str,
    *,
    sparse: bool = True,
    role: str | None = None,
) -> ad.AnnData:
    obs = pd.DataFrame(
        {"target_gene": pd.Categorical(labels), "context": "synthetic"},
        index=[f"{prefix}{index}" for index in range(len(labels))],
    )
    var = pd.DataFrame(index=["g0", "g1", "g2"])
    values = sp.csr_matrix(matrix, dtype=np.int32) if sparse else np.asarray(matrix)
    data = ad.AnnData(values, obs=obs, var=var)
    if role is not None:
        data.uns["public_validation"] = {
            "schema": "vcc-public-validation-data-v1",
            "role": role,
            "sealed_treated_profiles_present": role == "sealed-scorer-input",
        }
    return data


class PublicScoringViewTests(unittest.TestCase):
    def test_controls_are_appended_and_common_axis_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = [f"T{index:03d}" for index in range(300)]
            prediction = root / "prediction.h5ad"
            controls = root / "controls.h5ad"
            truth = root / "truth.h5ad"
            common = root / "common.csv"
            output_prediction = root / "pred_view.h5ad"
            output_truth = root / "truth_view.h5ad"
            report = root / "report.json"
            _data(
                np.tile(np.asarray([[1, 2, 3]], dtype=np.int32), (300, 1)),
                targets,
                "p",
            ).write_h5ad(prediction)
            _data(
                np.asarray([[4, 5, 6], [7, 8, 9]]),
                ["non-targeting"] * 2,
                "c",
                role="controls-only-generator-input",
            ).write_h5ad(controls)
            _data(
                np.vstack(
                    [
                        np.asarray([[4, 5, 6], [7, 8, 9]]),
                        np.tile(np.asarray([[10, 11, 12]]), (300, 1)),
                    ]
                ),
                ["non-targeting"] * 2 + targets,
                "r",
                role="sealed-scorer-input",
            ).write_h5ad(truth)
            common.write_text("gene_name\ng2\ng0\n", encoding="utf-8")
            common.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "schema": "vcc-public-validation-gene-axis-v1",
                        "contract": {"expression_matrix_accessed": False},
                        "provenance": {"output_csv": module.describe_file(common)},
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "prepare_public_scoring_views.py",
                str(prediction),
                str(controls),
                str(truth),
                str(common),
                str(output_prediction),
                str(output_truth),
                str(report),
                "--chunk-rows",
                "37",
            ]
            with mock.patch.object(sys, "argv", argv):
                module.main()
            pred_view = ad.read_h5ad(output_prediction)
            truth_view = ad.read_h5ad(output_truth)
            self.assertEqual(pred_view.shape, (302, 2))
            self.assertEqual(truth_view.shape, (302, 2))
            self.assertEqual(pred_view.var_names.tolist(), ["g2", "g0"])
            self.assertEqual(
                set(pred_view.obs["target_gene"].astype(str)),
                set(targets) | {"non-targeting"},
            )
            np.testing.assert_array_equal(pred_view.X[:2].toarray(), [[6, 4], [9, 7]])
            self.assertFalse(pred_view.uns["public_scoring"]["treated_truth_profiles_present"])
            self.assertTrue(truth_view.uns["public_scoring"]["treated_truth_profiles_present"])
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["context"], "synthetic")
            self.assertTrue(
                payload["contract"]["prediction_controls_match_sealed_truth_exactly"]
            )

    def test_dense_backed_matrix_supports_a_reordered_axis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.h5ad"
            _data(
                np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
                ["T0", "T1"],
                "d",
                sparse=False,
            ).write_h5ad(path)
            source = ad.read_h5ad(path, backed="r")
            try:
                observed = module._copy_matrix_on_axis(
                    source, np.asarray([2, 0]), chunk_rows=1
                )
            finally:
                source.file.close()
            np.testing.assert_array_equal(observed.toarray(), [[3, 1], [6, 4]])

    def test_fractional_sparse_counts_are_rejected_before_integer_cast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fractional.h5ad"
            data = _data(np.asarray([[1, 2, 3]]), ["T0"], "f")
            data.X = sp.csr_matrix(np.asarray([[1.0, 2.5, 3.0]], dtype=np.float32))
            data.write_h5ad(path)
            source = ad.read_h5ad(path, backed="r")
            try:
                with self.assertRaisesRegex(RuntimeError, "not integer-like"):
                    module._copy_matrix_on_axis(
                        source, np.asarray([0, 1, 2]), chunk_rows=1
                    )
            finally:
                source.file.close()


if __name__ == "__main__":
    unittest.main()
