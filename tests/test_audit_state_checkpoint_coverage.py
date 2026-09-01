"""Focused tests for the restricted STATE perturbation-map coverage audit."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_state_checkpoint_coverage import (  # noqa: E402
    SCHEMA,
    UnsafeCheckpointError,
    atomic_json,
    build_report,
    load_perturbation_map,
)


class UnexpectedCheckpointObject:
    pass


class RestrictedLoadTests(unittest.TestCase):
    @staticmethod
    def write_official_style_map(path: Path) -> None:
        torch.save(
            {
                np.str_("GENE1"): torch.tensor([1.0, 0.0, 0.0]),
                np.str_("GENE2"): torch.tensor([0.0, 1.0, 0.0]),
                np.str_("non-targeting"): torch.zeros(3),
            },
            path,
        )

    def test_official_numpy_string_keys_load_with_local_safe_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pert_onehot_map.pt"
            self.write_official_style_map(path)
            before = set(torch.serialization.get_safe_globals())
            payload, declared = load_perturbation_map(path)
            after = set(torch.serialization.get_safe_globals())

            self.assertEqual(set(payload), {"GENE1", "GENE2", "non-targeting"})
            self.assertEqual({tensor.numel() for tensor in payload.values()}, {3})
            self.assertIn("numpy.dtype", declared)
            self.assertEqual(before, after)

    def test_unknown_global_is_rejected_before_torch_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unsafe.pt"
            torch.save({"bad": UnexpectedCheckpointObject()}, path)
            with torch.serialization.safe_globals([UnexpectedCheckpointObject]):
                with mock.patch(
                    "audit_state_checkpoint_coverage.torch.load",
                    side_effect=AssertionError("torch.load must not run"),
                ) as loader:
                    with self.assertRaisesRegex(
                        UnsafeCheckpointError, "unsupported globals"
                    ):
                        load_perturbation_map(path)
            loader.assert_not_called()

    def test_non_string_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad_key.pt"
            torch.save({1: torch.ones(3)}, path)
            with self.assertRaisesRegex(RuntimeError, "non-string key"):
                load_perturbation_map(path)

    def test_non_string_numpy_dtype_is_not_allowlisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "integer_numpy_key.pt"
            torch.save({np.int64(1): torch.ones(3)}, path)
            with self.assertRaisesRegex(
                UnsafeCheckpointError, "Restricted weights-only load failed"
            ):
                load_perturbation_map(path)


class CoverageReportTests(unittest.TestCase):
    def test_report_contains_hashes_overlap_and_missing_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pert_map = root / "pert_onehot_map.pt"
            pert_counts = root / "pert_counts.csv"
            RestrictedLoadTests.write_official_style_map(pert_map)
            pert_counts.write_text(
                "target_gene\nGENE2\nMISSING\nGENE1\nGENE2\n",
                encoding="utf-8",
            )

            report = build_report(pert_map, pert_counts)
            self.assertEqual(report["schema"], SCHEMA)
            self.assertEqual(report["embedding_map"]["entries"], 3)
            self.assertEqual(report["embedding_map"]["dimensions"], [3])
            self.assertEqual(report["challenge_targets"]["rows"], 4)
            self.assertEqual(report["challenge_targets"]["unique_genes"], 3)
            self.assertEqual(report["challenge_targets"]["duplicate_rows"], 1)
            self.assertEqual(
                report["challenge_targets"]["duplicate_genes"], ["GENE2"]
            )
            self.assertEqual(report["coverage"]["overlap_genes"], 2)
            self.assertEqual(report["coverage"]["missing_genes"], ["MISSING"])
            self.assertEqual(report["coverage"]["missing_gene_count"], 1)
            self.assertFalse(report["coverage"]["full_target_coverage"])
            self.assertEqual(len(report["inputs"]["perturbation_map"]["sha256"]), 64)
            self.assertEqual(len(report["coverage"]["overlap_genes_sha256"]), 64)

    def test_atomic_json_writes_once_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "report.json"
            atomic_json(output, {"schema": SCHEMA})
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], SCHEMA)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                atomic_json(output, {"schema": "changed"})


if __name__ == "__main__":
    unittest.main()
