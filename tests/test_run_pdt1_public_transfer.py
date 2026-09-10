"""Focused synthetic tests for the registered CPU-only PDT-1 experiment.

No real g100 or sealed HepG2 artifact is available to these tests. They exercise
arm construction, scorer cross-checking, authentication, the registered gates,
oracle isolation, and atomic receipt publication using small synthetic arrays.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_pds_reachability as pds  # noqa: E402
import run_pdt1_public_transfer as pdt1  # noqa: E402


def synthetic_arm_fixture() -> tuple[
    tuple[str, ...], tuple[str, ...], np.ndarray, pdt1.EffectAtlas
]:
    perturbations = (pds.CONTROL_LABEL, "T3", "T1", "T2", "T4")
    genes = ("T1", "gB", "T2", "gA", "unmapped", "T4")
    control = np.full(len(genes), 10.0)
    effects = np.asarray(
        [
            [3.0, 0.3, -0.3, 0.5, 1.0, -0.2],
            [1.0, 0.1, -0.1, 0.2, 2.0, -0.1],
            [2.0, 0.2, -2.0, 0.4, 3.0, -0.4],
            [4.0, 0.4, -0.4, 0.8, 4.0, -4.0],
        ]
    )
    g100 = np.vstack([control, control[None, :] + effects])
    atlas = pdt1.EffectAtlas(
        effects=np.asarray(
            [
                [11.0, 101.0, 12.0, 999.0],
                [21.0, 201.0, 22.0, 999.0],
                [31.0, 301.0, 32.0, 999.0],
            ]
        ),
        target_names=("T1", "T2", "T3"),
        gene_names=("gA", "T1", "gB", "atlas_only"),
        npz_keys=pdt1.ATLAS_REQUIRED_KEYS,
    )
    return perturbations, genes, g100, atlas


def gate_arm(pds_value: float, anti_fraction: float) -> dict[str, object]:
    return {
        "pds_cosine": pds_value,
        "matched_cosine": {"anti_correlated_fraction": anti_fraction},
    }


class RegisteredConstantsTests(unittest.TestCase):
    def test_constants_match_next_route_decision(self) -> None:
        self.assertEqual(pdt1.SHARED_REMOVAL_FRACTION, 1.05)
        self.assertEqual(pdt1.SHUFFLE_SEED, 20260901)
        self.assertEqual(pdt1.REGISTERED_A0_PDS, 0.564526198439242)
        self.assertEqual(pdt1.REGISTERED_A1_PDS, 0.576667)
        self.assertEqual(pdt1.REGISTERED_PDS_FLOOR, 0.6263)
        self.assertEqual(pdt1.REGISTERED_SHUFFLE_MARGIN, 0.0400)
        self.assertEqual(pdt1.REGISTERED_ANTICORRELATED_CEILING, 0.300)
        self.assertEqual(pdt1.REGISTERED_TARGET_COUNT, 300)
        self.assertEqual(pdt1.REGISTERED_GENE_COUNT, 7_107)
        self.assertEqual(
            pdt1.REGISTERED_CELL_EVAL_COMMIT,
            "5e64833518a6603a0301cbe28185d49c30f4a986",
        )

    def test_non_oracle_constructor_cannot_accept_truth(self) -> None:
        parameters = inspect.signature(pdt1.build_non_oracle_arms).parameters
        self.assertEqual(tuple(parameters), ("perturbations", "genes", "g100_means", "atlas"))


class ArmConstructionTests(unittest.TestCase):
    def test_all_registered_non_oracle_arms_have_exact_semantics(self) -> None:
        perturbations, genes, g100, atlas = synthetic_arm_fixture()
        result = pdt1.build_non_oracle_arms(perturbations, genes, g100, atlas)
        self.assertEqual(tuple(result.means), ("A0", "A1", "A2", "A3", "A4"))
        np.testing.assert_array_equal(result.means["A0"], g100)

        control = g100[0]
        expected_a1_effects = pds.remove_shared(
            g100[1:] - control,
            pds.shared_direction(g100[1:] - control),
            1.05,
        )
        np.testing.assert_allclose(result.means["A1"][1:] - control, expected_a1_effects)

        # A2 maps K562 gB/gA in target-name space even though perturbations are not
        # alphabetically ordered. Panel genes T1/T2/T4 and the unmapped gene stay g100.
        row = {name: index for index, name in enumerate(perturbations)}
        expected = {
            "T1": (12.0, 11.0),
            "T2": (22.0, 21.0),
            "T3": (32.0, 31.0),
        }
        for target, (g_b, g_a) in expected.items():
            self.assertEqual(result.means["A2"][row[target], 1], control[1] + g_b)
            self.assertEqual(result.means["A2"][row[target], 3], control[3] + g_a)
            np.testing.assert_array_equal(
                result.means["A2"][row[target], [0, 2, 4, 5]],
                g100[row[target], [0, 2, 4, 5]],
            )
        np.testing.assert_array_equal(result.means["A2"][row["T4"]], g100[row["T4"]])

        expected_a3_effects = pds.remove_shared(
            result.means["A2"][1:] - control,
            pds.shared_direction(result.means["A2"][1:] - control),
            1.05,
        )
        np.testing.assert_allclose(result.means["A3"][1:] - control, expected_a3_effects)

        # Sorted overlap is T1,T2,T3 and the registered permutation is [T3,T2,T1].
        self.assertEqual(result.means["A4"][row["T1"], 1], control[1] + 32.0)
        self.assertEqual(result.means["A4"][row["T2"], 1], control[1] + 22.0)
        self.assertEqual(result.means["A4"][row["T3"], 1], control[1] + 12.0)
        np.testing.assert_array_equal(result.means["A4"][row["T4"]], g100[row["T4"]])

        for arm in result.means.values():
            np.testing.assert_array_equal(arm[0], control)
            self.assertFalse(arm.flags.writeable)
        with self.assertRaises(TypeError):
            result.means["A5"] = g100
        self.assertFalse(result.audit["truth_available_to_constructor"])
        self.assertTrue(result.audit["non_oracle_arrays_write_protected"])
        self.assertEqual(result.audit["panel_target_gene_coordinates_transferred"], 0)
        self.assertEqual(result.audit["unmapped_targets_retain_g100"], 1)

    def test_registered_shuffle_is_reproducible(self) -> None:
        fixture = synthetic_arm_fixture()
        first = pdt1.build_non_oracle_arms(*fixture)
        second = pdt1.build_non_oracle_arms(*fixture)
        np.testing.assert_array_equal(first.means["A4"], second.means["A4"])
        self.assertEqual(first.audit["a4_assignment_sha256"], second.audit["a4_assignment_sha256"])
        self.assertFalse(np.array_equal(first.means["A4"], first.means["A2"]))

    def test_oracle_is_an_isolated_copy(self) -> None:
        truth = np.arange(12, dtype=np.float64).reshape(3, 4)
        oracle = pdt1.build_oracle_arm(truth)
        np.testing.assert_array_equal(oracle, truth)
        oracle[0, 0] = -1.0
        self.assertEqual(truth[0, 0], 0.0)


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.perturbations = (pds.CONTROL_LABEL, "A", "B", "C")
        self.genes = ("A", "B", "C", "x", "y", "z")
        self.truth = np.zeros((4, 6), dtype=np.float64)
        self.truth[1:, 3:] = np.eye(3)

    def test_oracle_scores_one_and_has_no_anticorrelated_targets(self) -> None:
        def independent_scorer(means: np.ndarray) -> float:
            metrics = pdt1.independent_arm_metrics(
                means, self.truth, self.perturbations, self.genes
            )
            return float(metrics["independent_pds"])

        result = pdt1.score_arm(
            self.truth, self.truth, self.perturbations, self.genes, independent_scorer
        )
        self.assertEqual(result["pds_cosine"], 1.0)
        self.assertEqual(result["independent_pds"], 1.0)
        self.assertEqual(result["matched_cosine"]["anti_correlated_fraction"], 0.0)
        self.assertEqual(result["ranked_gene_count"], 3)

    def test_scorer_disagreement_fails_closed(self) -> None:
        with self.assertRaisesRegex(pdt1.PDT1Error, "disagrees"):
            pdt1.score_arm(
                self.truth,
                self.truth,
                self.perturbations,
                self.genes,
                lambda _: 0.9,
            )


class RegisteredGateTests(unittest.TestCase):
    def test_one_qualifying_arm_passes_all_three_gates(self) -> None:
        metrics = {
            "A2": gate_arm(0.6263, 0.299),
            "A3": gate_arm(0.60, 0.10),
            "A4": gate_arm(0.5863, 0.50),
        }
        decision = pdt1.registered_decision(metrics)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["status"], "PASS")
        self.assertEqual(decision["passing_arms"], ["A2"])

    def test_anti_correlation_ceiling_is_strict(self) -> None:
        metrics = {
            "A2": gate_arm(0.70, 0.300),
            "A3": gate_arm(0.50, 0.10),
            "A4": gate_arm(0.55, 0.50),
        }
        decision = pdt1.registered_decision(metrics)
        self.assertFalse(decision["passed"])
        self.assertFalse(
            decision["arm_evaluations"]["A2"]["checks"][
                "anti_correlated_fraction_below_0_300"
            ]
        )

    def test_shuffle_margin_and_pds_floor_are_both_required(self) -> None:
        metrics = {
            "A2": gate_arm(0.6263, 0.10),  # only 0.0263 above A4
            "A3": gate_arm(0.6200, 0.10),  # margin clears, PDS floor does not
            "A4": gate_arm(0.6000, 0.50),
        }
        decision = pdt1.registered_decision(metrics)
        self.assertFalse(decision["passed"])
        self.assertFalse(
            decision["arm_evaluations"]["A2"]["checks"]["pds_at_least_a4_plus_0_0400"]
        )
        self.assertFalse(
            decision["arm_evaluations"]["A3"]["checks"]["pds_at_least_0_6263"]
        )

    def test_registered_reproduction_is_fail_closed(self) -> None:
        valid = {
            "A0": {"pds_cosine": pdt1.REGISTERED_A0_PDS},
            "A1": {"pds_cosine": 0.5766666666666667},
            "A5": {"pds_cosine": 1.0},
        }
        report = pdt1.validate_registered_reproductions(valid, pdt1.REGISTERED_A0_PDS)
        self.assertTrue(all(report["checks"].values()))
        invalid = {**valid, "A5": {"pds_cosine": 0.999}}
        with self.assertRaisesRegex(pdt1.PDT1Error, "reproduction failed"):
            pdt1.validate_registered_reproductions(invalid, pdt1.REGISTERED_A0_PDS)


class AuthenticationTests(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_file_hash_mismatch_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"authenticated")
            record = pdt1.authenticate_regular_file(
                source, self._sha256(source), label="synthetic"
            )
            self.assertEqual(record.size_bytes, len(b"authenticated"))
            with self.assertRaisesRegex(pdt1.PDT1Error, "SHA-256 mismatch"):
                pdt1.authenticate_regular_file(source, "0" * 64, label="synthetic")
            alias = root / "alias.bin"
            alias.symlink_to(source)
            with self.assertRaisesRegex(pdt1.PDT1Error, "cannot open authenticated"):
                pdt1.authenticate_regular_file(alias, self._sha256(source), label="alias")

    def test_authenticated_path_mutation_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.bin"
            path.write_bytes(b"first")
            record = pdt1.authenticate_regular_file(path, self._sha256(path), label="input")
            path.write_bytes(b"second-longer")
            with self.assertRaisesRegex(pdt1.PDT1Error, "changed after hashing"):
                pdt1.require_unchanged(record, label="input")

    def test_axis_hash_is_order_sensitive_and_authenticated_csv_is_exact(self) -> None:
        self.assertNotEqual(pdt1.axis_sha256(("A", "B")), pdt1.axis_sha256(("B", "A")))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "genes.csv"
            path.write_text("gene_name\ng1\ng2\n", encoding="utf-8")
            record = pdt1.authenticate_regular_file(path, self._sha256(path), label="genes")
            self.assertEqual(
                pdt1.read_authenticated_csv_axis(record, "gene_name", label="genes"),
                ("g1", "g2"),
            )

    def _write_atlas(self, root: Path) -> tuple[Path, Path]:
        atlas_path = root / "atlas.npz"
        effects = np.arange(12, dtype=np.float32).reshape(3, 4)
        np.savez(
            atlas_path,
            effects=effects,
            matched_control_profiles=np.zeros_like(effects),
            target_names=np.asarray(["T1", "T2", "T3"]),
            gene_names=np.asarray(["g1", "g2", "g3", "g4"]),
            target_cell_counts=np.asarray([10, 20, 30]),
            batch_names=np.asarray(["b1", "b2"]),
            target_batch_counts=np.asarray([[10, 0], [10, 10], [0, 30]]),
            control_batch_counts=np.asarray([40, 40]),
            target_sum=np.asarray([50_000.0]),
        )
        receipt_path = root / "atlas.json"
        receipt = {
            "schema": pdt1.ATLAS_SCHEMA,
            "output": {
                "bytes": atlas_path.stat().st_size,
                "sha256": self._sha256(atlas_path),
                "npz_keys": list(pdt1.ATLAS_REQUIRED_KEYS),
            },
            "contract": {
                "effect_space": pdt1.ATLAS_EFFECT_SPACE,
                "normalization_target_sum": 50_000.0,
                "targets": 3,
                "genes": 4,
            },
        }
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return atlas_path, receipt_path

    def test_atlas_npz_and_receipt_are_cross_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            atlas_path, receipt_path = self._write_atlas(Path(temporary))
            atlas_record = pdt1.authenticate_regular_file(
                atlas_path, self._sha256(atlas_path), label="atlas"
            )
            receipt_record = pdt1.authenticate_regular_file(
                receipt_path, self._sha256(receipt_path), label="receipt"
            )
            atlas, receipt = pdt1.load_authenticated_atlas(atlas_record, receipt_record)
            self.assertEqual(atlas.effects.shape, (3, 4))
            self.assertEqual(atlas.target_names, ("T1", "T2", "T3"))
            self.assertEqual(receipt["schema"], pdt1.ATLAS_SCHEMA)

            forged_path = Path(temporary) / "forged.json"
            forged = json.loads(receipt_path.read_text(encoding="utf-8"))
            forged["output"]["sha256"] = "0" * 64
            forged_path.write_text(json.dumps(forged), encoding="utf-8")
            forged_record = pdt1.authenticate_regular_file(
                forged_path, self._sha256(forged_path), label="forged receipt"
            )
            with self.assertRaisesRegex(pdt1.PDT1Error, "does not bind"):
                pdt1.load_authenticated_atlas(atlas_record, forged_record)


class ReceiptWriterTests(unittest.TestCase):
    def test_atomic_writer_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt.json"
            first = {"schema": pdt1.SCHEMA, "status": "completed"}
            pdt1.write_json_no_overwrite(output, first)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), first)
            with self.assertRaises(FileExistsError):
                pdt1.write_json_no_overwrite(output, {"replacement": True})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), first)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
