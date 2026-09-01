"""Deterministic unit tests for the scorer-aware STATE count adapter."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_state_scorer_aware_counts import (  # noqa: E402
    aggregate_count_mass_qc,
    combine_log_fold,
    emit_exact_library_counts,
    emit_independent_counts,
    expected_counts,
    load_pseudobulk_bundle,
    load_response_bundle,
    parse_args,
    validate_args,
)
from generate_state_direct_counts import transform_raw_row  # noqa: E402


class ScorerAwarePolicyTests(unittest.TestCase):
    def test_target_force_is_off_by_default(self) -> None:
        with patch.object(sys, "argv", ["generator", "--model-dir", "model"]):
            args = parse_args()
        self.assertEqual(args.target_policy, "off")
        self.assertEqual(args.count_emission, "round")
        self.assertEqual(args.pseudobulk_blend_weight, 0.0)

    def test_target_force_requires_a_strict_knockdown_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = [
                "generator",
                "--model-dir",
                "model",
                "--output-h5ad",
                str(root / "candidate.h5ad"),
                "--output-json",
                str(root / "candidate.json"),
                "--target-policy",
                "force",
                "--target-max-remaining-fraction",
                "1.0",
                "--count-emission",
                "exact-largest-remainder",
            ]
            with patch.object(sys, "argv", argv):
                args = parse_args()
            with self.assertRaisesRegex(RuntimeError, "remaining fraction below one"):
                validate_args(args)

    def test_target_clamp_does_not_redistribute_counts(self) -> None:
        base = np.asarray([[10.0, 20.0, 30.0]])
        log_fold = np.zeros_like(base)
        unclamped, _ = expected_counts(
            base,
            log_fold,
            None,
            pseudobulk_weight=0.0,
            zero_induction_scale=0.0,
            depth_policy="absolute",
            target_index=1,
            target_policy="off",
            target_remaining_fraction=0.2,
        )
        clamped, _ = expected_counts(
            base,
            log_fold,
            None,
            pseudobulk_weight=0.0,
            zero_induction_scale=0.0,
            depth_policy="absolute",
            target_index=1,
            target_policy="clamp",
            target_remaining_fraction=0.2,
        )
        np.testing.assert_array_equal(unclamped, base)
        np.testing.assert_array_equal(clamped[:, [0, 2]], base[:, [0, 2]])
        self.assertEqual(float(clamped[0, 1]), 4.0)
        self.assertEqual(float(clamped.sum()), 44.0)

    def test_zero_induction_requires_explicit_pseudobulk_mass(self) -> None:
        base = np.asarray([[0.0, 2.0, 0.0]])
        log_fold = np.zeros_like(base)
        mean = np.asarray([3.0, 4.0, 5.0])
        disabled, disabled_qc = expected_counts(
            base,
            log_fold,
            mean,
            pseudobulk_weight=0.5,
            zero_induction_scale=0.0,
            depth_policy="absolute",
            target_index=1,
            target_policy="off",
            target_remaining_fraction=0.2,
        )
        enabled, enabled_qc = expected_counts(
            base,
            log_fold,
            mean,
            pseudobulk_weight=0.5,
            zero_induction_scale=1.0,
            depth_policy="absolute",
            target_index=1,
            target_policy="off",
            target_remaining_fraction=0.2,
        )
        np.testing.assert_array_equal(disabled, np.asarray([[0.0, 3.0, 0.0]]))
        np.testing.assert_array_equal(enabled, np.asarray([[1.5, 3.0, 2.5]]))
        self.assertEqual(disabled_qc["induced_zero_entries"], 0)
        self.assertEqual(enabled_qc["induced_zero_entries"], 2)

    def test_common_and_learned_effects_are_additive(self) -> None:
        state = np.zeros((2, 2), dtype=np.float32)
        support_to_current = np.asarray([0, 2], dtype=np.int64)
        common = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
        learned = np.asarray([0.4, -0.2, 0.1], dtype=np.float32)
        combined, _ = combine_log_fold(
            state,
            support_to_current,
            common,
            learned,
            state_weight=0.0,
            state_clip=0.6,
            state_bounding="tanh",
            common_weight=2.0,
            learned_weight=0.5,
            combined_clip=5.0,
        )
        expected = 2.0 * common + 0.5 * learned
        np.testing.assert_allclose(combined, np.tile(expected, (2, 1)), atol=1e-7)

    def test_emission_cap_drops_counts_without_redistribution(self) -> None:
        expectation = np.asarray([[4.8, 3.2, 2.6]])
        matrix, qc = emit_independent_counts(
            expectation,
            np.random.default_rng(7),
            emission="round",
            nb_dispersion=20.0,
            max_genes_per_cell=2,
            target_index=2,
            target_policy="off",
            target_remaining_fraction=0.2,
            base_target_counts=np.asarray([3.0]),
        )
        np.testing.assert_array_equal(matrix.toarray(), np.asarray([[5, 3, 0]]))
        self.assertEqual(qc["dropped_nnz_without_redistribution"], 1)
        self.assertEqual(qc["dropped_counts_without_redistribution"], 3)
        self.assertEqual(qc["emitted_count_total_before_cap"], 11)
        self.assertEqual(qc["emitted_count_total_after_cap"], 8)
        self.assertEqual(int(matrix.sum()), 8)

    def test_stochastic_emission_is_seed_deterministic(self) -> None:
        expectation = np.asarray([[0.2, 1.7, 4.5], [2.1, 0.9, 3.3]])
        kwargs = dict(
            emission="stochastic-round",
            nb_dispersion=20.0,
            max_genes_per_cell=3,
            target_index=0,
            target_policy="off",
            target_remaining_fraction=0.2,
            base_target_counts=np.asarray([1.0, 2.0]),
        )
        first, _ = emit_independent_counts(
            expectation, np.random.default_rng(42), **kwargs
        )
        second, _ = emit_independent_counts(
            expectation, np.random.default_rng(42), **kwargs
        )
        np.testing.assert_array_equal(first.toarray(), second.toarray())

    def test_exact_emission_preserves_library_and_redistributes_cap_mass(self) -> None:
        base = sp.csr_matrix(np.asarray([[5, 3, 2, 7]], dtype=np.int32))
        emitted, qc = emit_exact_library_counts(
            base,
            np.zeros((1, 4), dtype=np.float32),
            np.asarray([0, 1, 2], dtype=np.int64),
            np.random.default_rng(7),
            emission="exact-largest-remainder",
            max_genes_per_cell=3,
            target_current_index=1,
            target_policy="off",
            target_remaining_fraction=0.2,
        )
        np.testing.assert_array_equal(emitted.toarray(), np.asarray([[6, 4, 0, 7]]))
        self.assertEqual(int(emitted.sum()), int(base.sum()))
        self.assertEqual(qc["dropped_counts_without_redistribution"], 0)
        self.assertEqual(qc["redistributed_shared_counts"], 2)
        self.assertEqual(qc["library_size_mismatches"], 0)
        self.assertEqual(qc["current_only_count_mismatches"], 0)
        self.assertLessEqual(int(np.diff(emitted.indptr).max()), 3)

    def test_exact_emission_matches_scored_allocator_at_zero_residual(self) -> None:
        base = sp.csr_matrix(np.asarray([[10, 20, 0, 4]], dtype=np.int32))
        support_to_current = np.asarray([0, 1, 2], dtype=np.int64)
        log_fold = np.asarray([[0.2, -0.1, 0.0, 0.0]], dtype=np.float32)
        emitted, qc = emit_exact_library_counts(
            base,
            log_fold,
            support_to_current,
            np.random.default_rng(11),
            emission="exact-largest-remainder",
            max_genes_per_cell=4,
            target_current_index=1,
            target_policy="force",
            target_remaining_fraction=0.2,
        )

        support_delta = log_fold[:, support_to_current].copy()
        support_delta[:, 1] = np.log(0.2)
        current_to_support = np.asarray([0, 1, 2, -1], dtype=np.int64)
        indices, values, _ = transform_raw_row(
            base.indices,
            base.data,
            support_delta[0],
            current_to_support,
            4,
            "largest-remainder",
            np.random.default_rng(11),
        )
        expected = sp.csr_matrix(
            (values, indices, np.asarray([0, len(indices)])), shape=base.shape
        )
        np.testing.assert_array_equal(emitted.toarray(), expected.toarray())
        self.assertEqual(int(emitted.sum()), int(base.sum()))
        np.testing.assert_allclose(qc["final_target_delta_values"], np.log(0.2))
        self.assertGreater(qc["precomposition_expected_count_total"], 0)
        self.assertTrue(
            np.isfinite(qc["composition_normalization_values"]).all()
        )

    def test_exact_emission_does_not_activate_source_zero_genes(self) -> None:
        base = sp.csr_matrix(np.asarray([[5, 0, 7]], dtype=np.int32))
        emitted, _ = emit_exact_library_counts(
            base,
            np.asarray([[0.0, 0.6, 0.0]], dtype=np.float32),
            np.asarray([0, 1, 2], dtype=np.int64),
            np.random.default_rng(3),
            emission="exact-largest-remainder",
            max_genes_per_cell=3,
            target_current_index=0,
            target_policy="off",
            target_remaining_fraction=0.2,
        )
        self.assertEqual(int(emitted[0, 1]), 0)
        self.assertEqual(int(emitted.sum()), int(base.sum()))

    def test_exact_emission_rejects_unrepresentable_current_only_effects(self) -> None:
        base = sp.csr_matrix(np.asarray([[5, 3, 7]], dtype=np.int32))
        with self.assertRaisesRegex(
            RuntimeError, "cannot silently discard current-only effects"
        ):
            emit_exact_library_counts(
                base,
                np.asarray([[0.0, 0.0, 0.2]], dtype=np.float32),
                np.asarray([0, 1], dtype=np.int64),
                np.random.default_rng(3),
                emission="exact-largest-remainder",
                max_genes_per_cell=3,
                target_current_index=0,
                target_policy="off",
                target_remaining_fraction=0.2,
            )


class CountMassGateTests(unittest.TestCase):
    @staticmethod
    def group(context: str, source: int, before: int, after: int) -> dict[str, object]:
        dropped = before - after
        return {
            "context": context,
            "count_mass": {
                "source_input_count_total": source,
                "model_expected_count_total": float(before),
                "emitted_count_total_before_cap": before,
                "emitted_count_total_after_cap": after,
                "dropped_count_total": dropped,
                "dropped_count_fraction": dropped / before,
                "absolute_library_drift_fraction": abs(after - source) / source,
                "expectation_comparison_kind": "synthetic-test",
            },
        }

    def test_mass_gate_reports_exact_context_and_overall_totals(self) -> None:
        groups = [
            self.group("A", 100, 100, 95),
            self.group("B", 100, 100, 98),
            self.group("C", 100, 100, 97),
        ]
        report = aggregate_count_mass_qc(
            groups,
            max_dropped_count_fraction=0.10,
            max_absolute_library_drift_fraction=0.15,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["violations"], [])
        self.assertEqual(report["overall"]["source_input_count_total"], 300)
        self.assertEqual(report["overall"]["dropped_count_total"], 10)
        self.assertAlmostEqual(report["overall"]["dropped_count_fraction"], 10 / 300)
        self.assertEqual(report["by_context"]["B"]["emitted_count_total_after_cap"], 98)

    def test_mass_gate_fails_a_context_even_when_overall_is_within_threshold(self) -> None:
        groups = [
            self.group("A", 100, 100, 75),
            self.group("B", 1_000, 1_000, 1_000),
            self.group("C", 1_000, 1_000, 1_000),
        ]
        report = aggregate_count_mass_qc(
            groups,
            max_dropped_count_fraction=0.10,
            max_absolute_library_drift_fraction=0.15,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(
            any(value.startswith("A:dropped_count_fraction=") for value in report["violations"])
        )
        self.assertTrue(
            any(
                value.startswith("A:absolute_library_drift_fraction=")
                for value in report["violations"]
            )
        )


class AuxiliaryArtifactTests(unittest.TestCase):
    def test_response_subset_is_aligned_and_missing_targets_are_zero(self) -> None:
        challenge_genes = ["g0", "g1", "g2"]
        challenge_targets = ["t0", "t1"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "response.npz"
            np.savez_compressed(
                path,
                genes=np.asarray(["g2", "g0"]),
                contexts=np.asarray(["B"]),
                targets=np.asarray(["t1"]),
                common_log_fold=np.asarray([[0.3, 0.1]], dtype=np.float32),
                learned_log_fold=np.asarray([[[0.8, 0.4]]], dtype=np.float32),
            )
            bundle = load_response_bundle(path, challenge_genes, challenge_targets)
        common_b, learned_b = bundle.group(1, 1, len(challenge_genes))
        np.testing.assert_allclose(common_b, np.asarray([0.1, 0.0, 0.3]))
        np.testing.assert_allclose(learned_b, np.asarray([0.4, 0.0, 0.8]))
        _, missing = bundle.group(0, 0, len(challenge_genes))
        np.testing.assert_array_equal(missing, np.zeros(3))
        self.assertTrue(bundle.learned_available[1, 1])
        self.assertFalse(bundle.learned_available[0, 0])

    def test_pseudobulk_subset_is_aligned_and_broadcast_across_contexts(self) -> None:
        challenge_genes = ["g0", "g1", "g2"]
        challenge_targets = ["t0", "t1"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mean.npz"
            np.savez_compressed(
                path,
                genes=np.asarray([b"g1", b"g2"]),
                targets=np.asarray([b"t0"]),
                pseudobulk_mean=np.asarray([[2.0, 5.0]], dtype=np.float32),
            )
            bundle = load_pseudobulk_bundle(path, challenge_genes, challenge_targets)
        for context_index in range(3):
            np.testing.assert_array_equal(
                bundle.group(context_index, 0), np.asarray([0.0, 2.0, 5.0])
            )
            self.assertIsNone(bundle.group(context_index, 1))


if __name__ == "__main__":
    unittest.main()
