"""Deterministic unit tests for the scorer-aware STATE count adapter."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_state_scorer_aware_counts import (  # noqa: E402
    combine_log_fold,
    emit_independent_counts,
    expected_counts,
    load_pseudobulk_bundle,
    load_response_bundle,
    parse_args,
)


class ScorerAwarePolicyTests(unittest.TestCase):
    def test_target_force_is_off_by_default(self) -> None:
        with patch.object(sys, "argv", ["generator", "--model-dir", "model"]):
            args = parse_args()
        self.assertEqual(args.target_policy, "off")
        self.assertEqual(args.count_emission, "round")
        self.assertEqual(args.pseudobulk_blend_weight, 0.0)

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
