"""Unit and adversarial tests for scripts/diagnose_pds_reachability.py.

The numeric core is tested against closed-form expectations and against invariants
that must hold for any correct implementation of a per-row cosine ranking. The
preset restatement is tested against the pinned cell-eval2 configuration so the
module cannot silently drift from what the competition actually scores at.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

VCC2026_CONFIG = REPO_ROOT / "external" / "cell-eval2" / "src" / "cell_eval2" / "configs" / "vcc2026.yaml"

import diagnose_pds_reachability as pds  # noqa: E402


class UnitRowsTest(unittest.TestCase):
    def test_normalises_each_row(self):
        out = pds.unit_rows(np.array([[3.0, 4.0], [0.0, 2.0]]))
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0])

    def test_leaves_zero_row_at_zero(self):
        out = pds.unit_rows(np.array([[0.0, 0.0], [1.0, 0.0]]))
        np.testing.assert_array_equal(out[0], [0.0, 0.0])

    def test_rejects_non_matrix(self):
        with self.assertRaises(pds.DiagnosticError):
            pds.unit_rows(np.zeros(3))


class MidrankPdsTest(unittest.TestCase):
    def test_perfect_match_scores_one(self):
        similarity = np.eye(4)
        np.testing.assert_allclose(pds.midrank_pds(similarity), np.ones(4))

    def test_fully_tied_row_scores_one_half(self):
        # A no-information prediction ties every competitor; the mid-rank rule must
        # score it exactly 0.5 rather than an alphabetical index.
        similarity = np.ones((5, 5))
        np.testing.assert_allclose(pds.midrank_pds(similarity), np.full(5, 0.5))

    def test_worst_case_scores_zero(self):
        n = 4
        similarity = np.ones((n, n))
        np.fill_diagonal(similarity, 0.0)  # every competitor strictly nearer
        np.testing.assert_allclose(pds.midrank_pds(similarity), np.zeros(n))

    def test_partial_tie_takes_the_block_midpoint(self):
        # Row 0: one strictly nearer competitor and two exact ties -> rank 1 + 1.0 = 2.
        similarity = np.array(
            [[0.5, 0.9, 0.5, 0.5],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        )
        self.assertAlmostEqual(float(pds.midrank_pds(similarity)[0]), 1.0 - 2.0 / 3.0)

    def test_mean_of_a_permutation_is_one_half(self):
        # A fully constant prediction self-corrects: the ranks are a bijection.
        rng = np.random.default_rng(0)
        base = rng.standard_normal((6, 6))
        similarity = np.repeat(base[:1], 6, axis=0)
        self.assertAlmostEqual(float(pds.midrank_pds(similarity).mean()), 0.5)

    def test_rejects_non_square(self):
        with self.assertRaises(pds.DiagnosticError):
            pds.midrank_pds(np.zeros((2, 3)))


class AmplitudeInvarianceTest(unittest.TestCase):
    """The load-bearing property: a per-row cosine ranking is scale free."""

    def test_uniform_rescale_is_bit_identical(self):
        rng = np.random.default_rng(20260901)
        real = rng.standard_normal((25, 60))
        pred = 0.1 * real + rng.standard_normal((25, 60))
        base = pds.midrank_pds(pds.cosine_similarity_matrix(pred, real))
        for factor in (0.25, 1.0, 4.0, 1000.0):
            scaled = pds.midrank_pds(pds.cosine_similarity_matrix(factor * pred, real))
            np.testing.assert_array_equal(scaled, base)

    def test_per_row_rescale_is_also_inert(self):
        rng = np.random.default_rng(7)
        real = rng.standard_normal((15, 40))
        pred = rng.standard_normal((15, 40))
        weights = rng.uniform(0.1, 10.0, size=(15, 1))
        base = pds.midrank_pds(pds.cosine_similarity_matrix(pred, real))
        scaled = pds.midrank_pds(pds.cosine_similarity_matrix(weights * pred, real))
        np.testing.assert_array_equal(scaled, base)


class SharedRemovalTest(unittest.TestCase):
    def test_full_removal_is_orthogonal_to_the_axis(self):
        rng = np.random.default_rng(3)
        effects = rng.standard_normal((12, 30))
        axis = pds.shared_direction(effects)
        stripped = pds.remove_shared(effects, axis, 1.0)
        np.testing.assert_allclose(stripped @ axis, np.zeros(12), atol=1e-12)

    def test_zero_removal_is_the_identity(self):
        rng = np.random.default_rng(4)
        effects = rng.standard_normal((8, 20))
        axis = pds.shared_direction(effects)
        np.testing.assert_allclose(pds.remove_shared(effects, axis, 0.0), effects)

    def test_shared_direction_is_unit_norm(self):
        rng = np.random.default_rng(5)
        axis = pds.shared_direction(rng.standard_normal((10, 25)) + 3.0)
        self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0)


class StructureTest(unittest.TestCase):
    def test_separation_z_is_zero_for_an_uninformative_prediction(self):
        similarity = np.full((10, 10), 0.3)
        structure = pds.similarity_structure(similarity)
        self.assertTrue(np.isnan(structure.separation_z))
        self.assertAlmostEqual(structure.pds, 0.5)

    def test_independent_error_beats_correlated_error_at_equal_accuracy(self):
        """Two predictions carry the SAME per-target directional accuracy. One has
        isotropic error; the other's error is a single shared direction. The shared
        error must score worse, which is what lets the diagnostic attribute a
        shortfall to error structure rather than to missing signal.
        """
        rng = np.random.default_rng(20260901)
        n, d, accuracy = 60, 400, 0.08
        real = rng.standard_normal((n, d))
        signal = pds.unit_rows(real)

        independent = pds.simulate_directional_prediction(real, accuracy, rng)
        confound = pds.unit_rows(rng.standard_normal((1, d)))
        correlated = accuracy * signal + np.sqrt(1 - accuracy**2) * np.repeat(confound, n, axis=0)

        iso = pds.similarity_structure(pds.cosine_similarity_matrix(independent, real))
        cor = pds.similarity_structure(pds.cosine_similarity_matrix(correlated, real))

        self.assertAlmostEqual(iso.matched_mean, cor.matched_mean, delta=0.02)
        self.assertGreater(iso.pds, cor.pds)
        self.assertGreater(iso.separation_z, cor.separation_z)

    def test_heterogeneous_reference_scores_below_the_homogeneous_one(self):
        """The correction that matters: a homogeneous reference discards per-target
        dispersion and therefore flatters the candidate's shortfall. With the same MEAN
        accuracy but a realistic spread that leaves some targets anti-correlated, the
        independent-error reference must score materially lower.
        """
        # Shaped after the real HepG2 measurement: 300 targets, mean accuracy 0.0248,
        # standard deviation 0.0646, so about a third of targets fall at or below zero.
        rng = np.random.default_rng(20260901)
        n, d = 300, 3000
        real = rng.standard_normal((n, d))
        spread = 0.0248 + 0.0646 * rng.standard_normal(n)
        self.assertGreater(float((spread <= 0).mean()), 0.25)
        homogeneous = pds.simulate_directional_prediction(real, float(spread.mean()), rng)
        heterogeneous = pds.simulate_directional_prediction(real, spread, rng)

        hom = pds.similarity_structure(pds.cosine_similarity_matrix(homogeneous, real)).pds
        het = pds.similarity_structure(pds.cosine_similarity_matrix(heterogeneous, real)).pds
        self.assertGreater(hom, het)
        self.assertGreater(hom - het, 0.10)

    def test_negative_accuracy_is_preserved_not_clipped(self):
        """A target predicted backwards must stay backwards in the reference."""
        rng = np.random.default_rng(5)
        real = rng.standard_normal((40, 200))
        wanted = np.full(40, -0.3)
        simulated = pds.simulate_directional_prediction(real, wanted, rng)
        measured = (pds.unit_rows(simulated) * pds.unit_rows(real)).sum(axis=1)
        self.assertAlmostEqual(float(measured.mean()), -0.3, delta=0.03)

    def test_matched_accuracy_is_recovered_by_the_simulator(self):
        rng = np.random.default_rng(11)
        real = rng.standard_normal((80, 300))
        simulated = pds.simulate_directional_prediction(real, 0.2, rng)
        measured = (pds.unit_rows(simulated) * pds.unit_rows(real)).sum(axis=1).mean()
        self.assertAlmostEqual(float(measured), 0.2, delta=0.02)

    def test_simulator_rejects_an_out_of_range_accuracy(self):
        rng = np.random.default_rng(1)
        with self.assertRaises(pds.DiagnosticError):
            pds.simulate_directional_prediction(np.zeros((3, 4)), 1.5, rng)

    def test_simulator_rejects_a_wrong_length_accuracy_vector(self):
        rng = np.random.default_rng(1)
        with self.assertRaises(pds.DiagnosticError):
            pds.simulate_directional_prediction(np.zeros((3, 4)), np.zeros(5), rng)


class RetrievalAndCollapseTest(unittest.TestCase):
    def test_top1_retrieval_counts_exact_nearest_matches(self):
        similarity = np.array([[0.9, 0.1], [0.7, 0.2]])  # row 0 correct, row 1 wrong
        self.assertEqual(pds.top1_retrieval(similarity), 1)

    def test_top1_retrieval_is_full_for_the_identity(self):
        self.assertEqual(pds.top1_retrieval(np.eye(6)), 6)

    def test_mean_pairwise_cosine_is_one_for_parallel_rows(self):
        rows = np.repeat(np.array([[1.0, 2.0, 3.0]]), 5, axis=0)
        self.assertAlmostEqual(pds.mean_pairwise_cosine(rows), 1.0, places=10)

    def test_mean_pairwise_cosine_is_near_zero_for_random_high_dimensional_rows(self):
        rng = np.random.default_rng(3)
        self.assertAlmostEqual(
            pds.mean_pairwise_cosine(rng.standard_normal((40, 5000))), 0.0, delta=0.02
        )


class EffectiveRankTest(unittest.TestCase):
    def test_orthonormal_rows_have_full_effective_rank(self):
        self.assertAlmostEqual(pds.effective_rank(np.eye(7)), 7.0, places=6)

    def test_rank_one_matrix_has_effective_rank_one(self):
        rng = np.random.default_rng(2)
        outer = np.outer(rng.standard_normal(9), rng.standard_normal(13))
        self.assertAlmostEqual(pds.effective_rank(outer), 1.0, places=6)

    def test_zero_matrix_is_defined(self):
        self.assertEqual(pds.effective_rank(np.zeros((4, 4))), 0.0)


class PanelExclusionTest(unittest.TestCase):
    def test_removes_every_panel_target_gene(self):
        genes = ["A", "B", "C", "D"]
        keep = pds.panel_excluded_mask(genes, ["B", "D"])
        np.testing.assert_array_equal(keep, [True, False, True, False])

    def test_a_target_absent_from_the_gene_axis_removes_nothing(self):
        keep = pds.panel_excluded_mask(["A", "B"], ["ZZZ"])
        np.testing.assert_array_equal(keep, [True, True])


class PresetAgreementTest(unittest.TestCase):
    """The restated preset must not drift from the pinned scorer configuration."""

    def test_discrimination_kwargs_match_the_pinned_vcc2026_config(self):
        if not VCC2026_CONFIG.exists():
            self.skipTest("pinned cell-eval2 checkout is not present")
        text = VCC2026_CONFIG.read_text(encoding="utf-8")
        block = text.split("discrimination:", 1)[1].split("\nde:", 1)[0]
        declared = {}
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, _, value = stripped.partition(":")
            declared[key.strip()] = value.strip()
        for key, expected in pds.DISCRIMINATION_KWARGS.items():
            if key == "control_source":
                continue  # declared at the top level of the config, not in this block
            self.assertIn(key, declared, f"{key} missing from the pinned config")
            self.assertEqual(
                declared[key], "true" if expected is True else str(expected),
                f"{key} drifted from the pinned config",
            )

    def test_top_level_settings_match(self):
        if not VCC2026_CONFIG.exists():
            self.skipTest("pinned cell-eval2 checkout is not present")
        text = VCC2026_CONFIG.read_text(encoding="utf-8")
        self.assertIn(f"bulk_target_sum: {pds.BULK_TARGET_SUM:g}", text)
        self.assertIn(f"control: {pds.CONTROL_LABEL}", text)
        self.assertIn(f"control_source: {pds.DISCRIMINATION_KWARGS['control_source']}", text)


class RepoRelativeTest(unittest.TestCase):
    def test_rejects_a_path_outside_the_repository(self):
        with self.assertRaises(pds.DiagnosticError):
            pds.repo_relative(Path("/etc/hostname"), REPO_ROOT)

    def test_returns_a_posix_relative_path(self):
        self.assertEqual(
            pds.repo_relative(REPO_ROOT / "scripts" / "x.py", REPO_ROOT), "scripts/x.py"
        )


if __name__ == "__main__":
    unittest.main()
