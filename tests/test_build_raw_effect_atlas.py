"""Integration tests for raw-count effect-atlas construction."""

from __future__ import annotations

import json
import os
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

from build_raw_effect_atlas import main  # noqa: E402


class RawEffectAtlasIntegrationTests(unittest.TestCase):
    def test_output_pair_rolls_back_if_json_commit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "raw.h5ad"
            output_npz = root / "atlas.npz"
            output_json = root / "atlas.json"
            data = ad.AnnData(
                X=np.asarray(
                    [[1, 3], [2, 2], [8, 1], [7, 1]],
                    dtype=np.int32,
                ),
                obs=pd.DataFrame(
                    {
                        "gene": ["non-targeting", "non-targeting", "T1", "T1"],
                        "batch": ["b0", "b0", "b0", "b0"],
                    },
                    index=[f"cell_{index}" for index in range(4)],
                ),
                var=pd.DataFrame(
                    {"gene_name": ["g0", "g1"]},
                    index=["ensembl_0", "ensembl_1"],
                ),
            )
            data.write_h5ad(input_path)
            argv = [
                "build_raw_effect_atlas.py",
                str(input_path),
                str(output_npz),
                str(output_json),
                "--batch-col",
                "batch",
                "--min-cells",
                "2",
                "--device",
                "cpu",
            ]
            original_replace = os.replace
            replace_calls = 0

            def fail_second_replace(source: str | Path, destination: str | Path) -> None:
                nonlocal replace_calls
                replace_calls += 1
                if replace_calls == 2:
                    raise OSError("injected JSON commit failure")
                original_replace(source, destination)

            with patch.object(sys, "argv", argv):
                with patch(
                    "build_raw_effect_atlas.os.replace",
                    side_effect=fail_second_replace,
                ):
                    with self.assertRaisesRegex(OSError, "injected JSON commit failure"):
                        main()

            self.assertFalse(output_npz.exists())
            self.assertFalse(output_json.exists())
            self.assertEqual(list(root.glob(".*.partial.*")), [])

    def test_duplicate_gene_symbols_are_collapsed_by_sum_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "raw.h5ad"
            output_npz = root / "atlas.npz"
            output_json = root / "atlas.json"
            data = ad.AnnData(
                X=np.asarray(
                    [
                        [1, 2, 3],
                        [2, 1, 3],
                        [8, 2, 0],
                        [8, 2, 0],
                    ],
                    dtype=np.int32,
                ),
                obs=pd.DataFrame(
                    {
                        "gene": ["non-targeting", "non-targeting", "T1", "T1"],
                        "batch": ["b0", "b0", "b0", "b0"],
                    },
                    index=[f"cell_{index}" for index in range(4)],
                ),
                var=pd.DataFrame(
                    {"gene_name": ["g0", "g0", "g1"]},
                    index=["ensembl_0", "ensembl_1", "ensembl_2"],
                ),
            )
            data.write_h5ad(input_path)

            argv = [
                "build_raw_effect_atlas.py",
                str(input_path),
                str(output_npz),
                str(output_json),
                "--batch-col",
                "batch",
                "--min-cells",
                "2",
                "--chunk-rows",
                "2",
                "--device",
                "cpu",
            ]
            with patch.object(sys, "argv", argv):
                main()

            with np.load(output_npz, allow_pickle=False) as archive:
                self.assertEqual(archive["gene_names"].tolist(), ["g0", "g1"])
                self.assertEqual(archive["target_names"].tolist(), ["T1"])
                self.assertEqual(archive["effects"].shape, (1, 2))
                self.assertGreater(float(archive["effects"][0, 0]), 0)
                self.assertLess(float(archive["effects"][0, 1]), 0)
                np.testing.assert_array_equal(
                    archive["gene_first_positions"], np.asarray([0, 2])
                )
                np.testing.assert_array_equal(
                    archive["gene_duplicate_positions"], np.asarray([1])
                )
                np.testing.assert_array_equal(
                    archive["gene_duplicate_destinations"], np.asarray([0])
                )

            report = json.loads(output_json.read_text(encoding="utf-8"))
            contract = report["contract"]
            self.assertEqual(contract["input_gene_columns"], 3)
            self.assertEqual(contract["genes"], 2)
            self.assertEqual(contract["collapsed_duplicate_columns"], 1)
            self.assertEqual(contract["duplicate_symbols_collapsed_by_sum"], {"g0": 2})
            self.assertEqual(
                contract["duplicate_gene_sources"]["g0"],
                [
                    {"source_column": 0, "source_var_name": "ensembl_0"},
                    {"source_column": 1, "source_var_name": "ensembl_1"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
