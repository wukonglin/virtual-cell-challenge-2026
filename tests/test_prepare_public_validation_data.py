"""Integration tests for the public-validation data firewall."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from infer_state_effect_prior import describe_file  # noqa: E402
from prepare_public_validation_data import main  # noqa: E402


class PublicValidationDataIntegrationTests(unittest.TestCase):
    def test_controls_and_treated_truth_are_separated_and_duplicates_are_summed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.h5ad"
            panel_csv = root / "panel.csv"
            panel_json = root / "panel.json"
            genes_csv = root / "common.csv"
            genes_json = root / "common.json"
            state_genes = root / "state_genes.csv"
            controls_path = root / "controls.h5ad"
            truth_path = root / "truth.h5ad"
            report_path = root / "report.json"

            source = ad.AnnData(
                X=np.asarray(
                    [
                        [1, 2, 3, 0],
                        [2, 1, 1, 0],
                        [4, 5, 0, 2],
                        [5, 4, 0, 3],
                    ],
                    dtype=np.int32,
                ),
                obs=pd.DataFrame(
                    {
                        "gene": ["non-targeting", "non-targeting", "T1", "T1"],
                        "gem_group": ["b0", "b0", "b0", "b0"],
                        "sgID_AB": ["nt0", "nt1", "sg0", "sg1"],
                    },
                    index=[f"source_{index}" for index in range(4)],
                ),
                var=pd.DataFrame(
                    {"gene_name": ["g0", "g0", "g1", "T1"]},
                    index=["e0", "e1", "e2", "e3"],
                ),
            )
            source.write_h5ad(source_path)
            pd.DataFrame(
                {
                    "target_gene": ["T1"],
                    "validation_route": ["direct"],
                    "hepg2_treated_cells": [2],
                }
            ).to_csv(panel_csv, index=False)
            panel_json.write_text(
                json.dumps(
                    {
                        "schema": "vcc-public-validation-manifest-v1",
                        "provenance": {"output_csv": describe_file(panel_csv)},
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame({"gene_name": ["g0", "g1"]}).to_csv(genes_csv, index=False)
            genes_json.write_text(
                json.dumps(
                    {
                        "schema": "vcc-public-validation-gene-axis-v1",
                        "provenance": {"output_csv": describe_file(genes_csv)},
                    }
                ),
                encoding="utf-8",
            )
            state_genes.write_text("g0\ng1\nT1\n", encoding="utf-8")

            argv = [
                "prepare_public_validation_data.py",
                str(source_path),
                "HepG2",
                str(controls_path),
                str(truth_path),
                str(report_path),
                "--panel-csv",
                str(panel_csv),
                "--panel-json",
                str(panel_json),
                "--common-genes",
                str(genes_csv),
                "--common-genes-json",
                str(genes_json),
                "--state-genes",
                str(state_genes),
                "--chunk-rows",
                "2",
                "--expected-panel-targets",
                "1",
                "--expected-common-genes",
                "2",
            ]
            with patch.object(sys, "argv", argv):
                main()

            controls = ad.read_h5ad(controls_path)
            truth = ad.read_h5ad(truth_path)
            controls_sealed = controls.uns["public_validation"][
                "sealed_treated_profiles_present"
            ]
            truth_sealed = truth.uns["public_validation"][
                "sealed_treated_profiles_present"
            ]
            self.assertIsInstance(controls_sealed, np.bool_)
            self.assertIsInstance(truth_sealed, np.bool_)
            self.assertFalse(bool(controls_sealed))
            self.assertTrue(bool(truth_sealed))
            self.assertEqual(controls.shape, (2, 3))
            self.assertEqual(truth.shape, (4, 3))
            self.assertEqual(controls.var_names.tolist(), ["g0", "g1", "T1"])
            self.assertTrue(
                (controls.obs["target_gene"].astype(str) == "non-targeting").all()
            )
            self.assertEqual(
                set(truth.obs["target_gene"].astype(str)), {"non-targeting", "T1"}
            )
            self.assertEqual(
                set(controls.obs["validation_route"].astype(str)), {"control"}
            )
            self.assertEqual(
                set(truth.obs["validation_route"].astype(str)), {"control", "direct"}
            )
            np.testing.assert_array_equal(
                controls.X.toarray(), np.asarray([[3, 3, 0], [3, 1, 0]])
            )
            truth_controls = (
                truth.obs["target_gene"].astype(str).to_numpy() == "non-targeting"
            )
            np.testing.assert_array_equal(
                truth.X[truth_controls].toarray(), controls.X.toarray()
            )
            np.testing.assert_array_equal(
                truth.obs.loc[truth_controls, "source_cell_index"].to_numpy(),
                controls.obs["source_cell_index"].to_numpy(),
            )
            self.assertEqual(int(controls.var["is_common_public_gene"].sum()), 2)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["firewall"]["controls_contains_treated_cells"])
            self.assertEqual(report["duplicate_symbols_collapsed_by_sum"], {"g0": 2})

            collision_path = root / "collision.h5ad"
            collision_argv = argv.copy()
            collision_argv[3] = str(collision_path)
            collision_argv[4] = str(collision_path)
            collision_argv[5] = str(root / "collision.json")
            with patch.object(sys, "argv", collision_argv):
                with self.assertRaisesRegex(RuntimeError, "outputs must be distinct"):
                    main()
            self.assertFalse(collision_path.exists())


if __name__ == "__main__":
    unittest.main()
