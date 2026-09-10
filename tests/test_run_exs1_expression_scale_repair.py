"""Focused synthetic tests for the truth-blind EXS-1 CPU command."""

from __future__ import annotations

import json
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

from exs1_local_io import sha256_file  # noqa: E402
from run_exs1_expression_scale_repair import (  # noqa: E402
    MSE_METRIC,
    NMAE_METRIC,
    PDS_METRIC,
    attenuate_count_row,
    evaluate_registered_gate,
    main,
)


def _write_inputs(root: Path) -> tuple[Path, Path]:
    genes = pd.DataFrame(
        index=pd.Index(
            ["g0", "g1", "g2", "g3"],
            name="gene_name",
        )
    )
    controls = ad.AnnData(
        X=sp.csr_matrix(
            np.asarray(
                [
                    [8, 0, 2, 0],
                    [0, 6, 0, 6],
                    [1, 1, 1, 1],
                ],
                dtype=np.int32,
            )
        ),
        obs=pd.DataFrame(
            {
                "context": ["HepG2"] * 3,
                "target_gene": ["non-targeting"] * 3,
            },
            index=pd.Index(
                ["control-0", "control-1", "control-2"],
                name="cell_id",
            ),
        ),
        var=genes.copy(),
    )
    controls.uns["public_validation"] = {
        "schema": "vcc-public-validation-data-v1",
        "role": "controls-only-generator-input",
        "sealed_treated_profiles_present": False,
    }
    prediction = ad.AnnData(
        X=sp.csr_matrix(
            np.asarray(
                [
                    [0, 10, 0, 10],
                    [6, 0, 3, 0],
                    [0, 0, 5, 5],
                    [4, 4, 0, 0],
                ],
                dtype=np.int32,
            )
        ),
        obs=pd.DataFrame(
            {
                "context": ["HepG2"] * 4,
                "target_gene": [
                    "GENE_A",
                    "GENE_A",
                    "GENE_B",
                    "GENE_B",
                ],
                "source_control_row": [0, 1, 1, 2],
                "source_control_cell_id": [
                    "control-0",
                    "control-1",
                    "control-1",
                    "control-2",
                ],
                "kept_annotation": ["a", "b", "c", "d"],
            },
            index=pd.Index(
                ["pred-0", "pred-1", "pred-2", "pred-3"],
                name="cell_id",
            ),
        ),
        var=genes.copy(),
    )
    prediction_path = root / "prediction.h5ad"
    controls_path = root / "controls.h5ad"
    prediction.write_h5ad(prediction_path)
    controls.write_h5ad(controls_path)
    return prediction_path, controls_path


def _transform_argv(
    prediction: Path,
    controls: Path,
    output: Path,
) -> list[str]:
    return [
        "transform",
        "--prediction",
        str(prediction),
        "--prediction-expected-sha256",
        sha256_file(prediction),
        "--controls-only",
        str(controls),
        "--controls-expected-sha256",
        sha256_file(controls),
        "--output-dir",
        str(output),
        "--attenuation",
        "0",
        "--attenuation",
        "0.5",
        "--attenuation",
        "1",
        "--chunk-rows",
        "2",
    ]


class EXS1RowTests(unittest.TestCase):
    def test_largest_remainder_ties_use_ascending_gene_index(
        self,
    ) -> None:
        prediction = sp.csr_matrix(
            np.asarray([[1, 0, 1, 0]], dtype=np.int32)
        )
        control = sp.csr_matrix(
            np.asarray([[0, 1, 0, 1]], dtype=np.int32)
        )
        first = attenuate_count_row(
            prediction,
            control,
            attenuation=0.5,
            library_policy="keep_prediction",
            n_genes=4,
        )
        second = attenuate_count_row(
            prediction,
            control,
            attenuation=0.5,
            library_policy="keep_prediction",
            n_genes=4,
        )
        expected = np.asarray([[1, 1, 0, 0]], dtype=np.int32)
        np.testing.assert_array_equal(first.toarray(), expected)
        np.testing.assert_array_equal(
            first.toarray(), second.toarray()
        )

    def test_exact_identity_endpoints(self) -> None:
        prediction = sp.csr_matrix(
            np.asarray([[7, 0, 2, 1]], dtype=np.int32)
        )
        control = sp.csr_matrix(
            np.asarray([[0, 3, 0, 2]], dtype=np.int32)
        )
        identity = attenuate_count_row(
            prediction,
            control,
            attenuation=1.0,
            library_policy="keep_prediction",
            n_genes=4,
        )
        source = attenuate_count_row(
            prediction,
            control,
            attenuation=0.0,
            library_policy="match_source_control",
            n_genes=4,
        )
        np.testing.assert_array_equal(
            identity.toarray(), prediction.toarray()
        )
        np.testing.assert_array_equal(
            source.toarray(), control.toarray()
        )

    def test_registered_gate_is_exact_and_finite(self) -> None:
        passing = {
            MSE_METRIC: 0.9,
            PDS_METRIC: 0.564526198439242 + 0.9e-6,
            NMAE_METRIC: 1.017691308794254,
        }
        self.assertTrue(
            evaluate_registered_gate(passing)["passed"]
        )
        for metric, value in (
            (MSE_METRIC, 0.9000001),
            (PDS_METRIC, 0.564526198439242 + 1.1e-6),
            (NMAE_METRIC, 1.017691308794255),
        ):
            failing = dict(passing)
            failing[metric] = value
            self.assertFalse(
                evaluate_registered_gate(failing)["passed"]
            )
        nonfinite = dict(passing)
        nonfinite[MSE_METRIC] = float("nan")
        with self.assertRaisesRegex(
            RuntimeError, "non-finite"
        ):
            evaluate_registered_gate(nonfinite)


class EXS1TransformTests(unittest.TestCase):
    def test_transform_preserves_axes_libraries_and_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_path, controls_path = _write_inputs(root)
            first_output = root / "exs1-first"
            second_output = root / "exs1-second"
            main(
                _transform_argv(
                    prediction_path,
                    controls_path,
                    first_output,
                )
            )
            main(
                _transform_argv(
                    prediction_path,
                    controls_path,
                    second_output,
                )
            )

            first = json.loads(
                (first_output / "experiment.json").read_text(
                    encoding="utf-8"
                )
            )
            second = json.loads(
                (second_output / "experiment.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                first["status"], "transformed_unscored"
            )
            self.assertEqual(
                first["gate_result"]["passed"], None
            )
            self.assertFalse(
                first["firewall"][
                    "transform_parser_accepts_treated_truth"
                ]
            )
            self.assertFalse(
                first["firewall"][
                    "treated_truth_profiles_read"
                ]
            )
            self.assertEqual(len(first["arms"]), 6)
            self.assertEqual(
                {
                    name: arm["output_prediction"][
                        "logical_csr_sha256"
                    ]
                    for name, arm in first["arms"].items()
                },
                {
                    name: arm["output_prediction"][
                        "logical_csr_sha256"
                    ]
                    for name, arm in second["arms"].items()
                },
            )
            for arm in first["arms"].values():
                self.assertEqual(
                    arm["scores"],
                    {
                        MSE_METRIC: None,
                        PDS_METRIC: None,
                        NMAE_METRIC: None,
                    },
                )
                self.assertIsNone(arm["gate"]["passed"])

            source_prediction = ad.read_h5ad(prediction_path)
            source_controls = ad.read_h5ad(controls_path)
            identity = ad.read_h5ad(
                first_output
                / "arms"
                / "attenuation_10000bp__keep_prediction"
                / "prediction.h5ad"
            )
            control_endpoint = ad.read_h5ad(
                first_output
                / "arms"
                / "attenuation_0000bp__match_source_control"
                / "prediction.h5ad"
            )
            np.testing.assert_array_equal(
                identity.X.toarray(),
                source_prediction.X.toarray(),
            )
            paired_rows = source_prediction.obs[
                "source_control_row"
            ].to_numpy(dtype=np.int64)
            np.testing.assert_array_equal(
                control_endpoint.X.toarray(),
                source_controls.X[paired_rows, :].toarray(),
            )
            for output in (identity, control_endpoint):
                self.assertEqual(
                    output.obs_names.tolist(),
                    source_prediction.obs_names.tolist(),
                )
                self.assertEqual(
                    output.var_names.tolist(),
                    source_prediction.var_names.tolist(),
                )
                self.assertEqual(
                    output.obs["target_gene"].astype(str).tolist(),
                    source_prediction.obs[
                        "target_gene"
                    ].astype(str).tolist(),
                )

            middle_keep = ad.read_h5ad(
                first_output
                / "arms"
                / "attenuation_5000bp__keep_prediction"
                / "prediction.h5ad"
            )
            middle_control = ad.read_h5ad(
                first_output
                / "arms"
                / "attenuation_5000bp__match_source_control"
                / "prediction.h5ad"
            )
            np.testing.assert_array_equal(
                np.asarray(
                    middle_keep.X.sum(axis=1)
                ).reshape(-1),
                np.asarray(
                    source_prediction.X.sum(axis=1)
                ).reshape(-1),
            )
            np.testing.assert_array_equal(
                np.asarray(
                    middle_control.X.sum(axis=1)
                ).reshape(-1),
                np.asarray(
                    source_controls.X[paired_rows, :].sum(
                        axis=1
                    )
                ).reshape(-1),
            )

    def test_hash_overwrite_and_firewall_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_path, controls_path = _write_inputs(root)

            wrong_hash_output = root / "wrong-hash"
            argv = _transform_argv(
                prediction_path,
                controls_path,
                wrong_hash_output,
            )
            argv[argv.index("--prediction-expected-sha256") + 1] = (
                "0" * 64
            )
            with self.assertRaisesRegex(
                RuntimeError, "SHA-256 mismatch"
            ):
                main(argv)
            self.assertFalse(wrong_hash_output.exists())

            existing = root / "existing"
            existing.mkdir()
            sentinel = existing / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "Refusing to overwrite"
            ):
                main(
                    _transform_argv(
                        prediction_path,
                        controls_path,
                        existing,
                    )
                )
            self.assertEqual(
                sentinel.read_text(encoding="utf-8"), "keep"
            )

            bad_controls = ad.read_h5ad(controls_path)
            bad_controls.obs["target_gene"] = [
                "non-targeting",
                "TREATED",
                "non-targeting",
            ]
            bad_controls_path = root / "bad-controls.h5ad"
            bad_controls.write_h5ad(bad_controls_path)
            bad_output = root / "bad-control-output"
            with self.assertRaisesRegex(
                RuntimeError,
                "Controls-only input contains treated cells",
            ):
                main(
                    _transform_argv(
                        prediction_path,
                        bad_controls_path,
                        bad_output,
                    )
                )
            self.assertFalse(bad_output.exists())


if __name__ == "__main__":
    unittest.main()
