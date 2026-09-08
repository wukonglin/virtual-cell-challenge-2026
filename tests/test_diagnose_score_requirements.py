"""Unit and adversarial tests for scripts/diagnose_score_requirements.py.

The analytic claims carry the decision, so they are tested against closed forms and
against a brute-force amplitude search rather than against themselves.
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

import diagnose_score_requirements as req  # noqa: E402


class AmplitudeAndErrorTest(unittest.TestCase):
    def test_amplitude_ratio_is_the_norm_quotient(self):
        truth = np.array([[3.0, 4.0]])
        self.assertAlmostEqual(req.amplitude_ratio(2.0 * truth, truth), 2.0)

    def test_amplitude_ratio_rejects_zero_truth(self):
        with self.assertRaises(req.RequirementError):
            req.amplitude_ratio(np.ones((1, 2)), np.zeros((1, 2)))

    def test_error_ratio_is_zero_for_a_perfect_prediction(self):
        rng = np.random.default_rng(0)
        truth = rng.standard_normal((10, 30))
        self.assertAlmostEqual(req.expression_error_ratio(truth, truth), 0.0)

    def test_error_ratio_is_one_for_the_control(self):
        """Predicting no change is the no-skill point, and it must read exactly 1."""
        rng = np.random.default_rng(1)
        truth = rng.standard_normal((10, 30))
        self.assertAlmostEqual(req.expression_error_ratio(np.zeros_like(truth), truth), 1.0)

    def test_error_ratio_is_four_for_a_perfectly_wrong_prediction(self):
        rng = np.random.default_rng(2)
        truth = rng.standard_normal((8, 20))
        self.assertAlmostEqual(req.expression_error_ratio(-truth, truth), 4.0)


class FloorTest(unittest.TestCase):
    def test_floor_matches_a_brute_force_amplitude_search(self):
        """The closed form 1 - c^2 at a = c must beat every amplitude on real vectors."""
        rng = np.random.default_rng(20260901)
        truth = rng.standard_normal((200, 400))
        for accuracy in (0.05, 0.2, 0.5):
            direction = req.blend_toward_truth(truth, accuracy, rng)
            realised = float(
                (req.unit_rows(direction) * req.unit_rows(truth)).sum(axis=1).mean()
            )
            norms = np.linalg.norm(truth, axis=1, keepdims=True)
            ratios = {
                a: req.expression_error_ratio(direction * norms * a, truth)
                for a in np.linspace(0.01, 1.5, 150)
            }
            best_amplitude = min(ratios, key=ratios.__getitem__)
            self.assertAlmostEqual(best_amplitude, req.optimal_amplitude(realised), delta=0.03)
            self.assertAlmostEqual(
                min(ratios.values()), req.best_expression_error_ratio(realised), delta=0.02
            )

    def test_floor_is_one_at_zero_accuracy(self):
        self.assertAlmostEqual(req.best_expression_error_ratio(0.0), 1.0)

    def test_floor_is_zero_at_perfect_accuracy(self):
        self.assertAlmostEqual(req.best_expression_error_ratio(1.0), 0.0)

    def test_required_accuracy_inverts_the_floor(self):
        for accuracy in (0.05, 0.2, 0.35, 0.8):
            ratio = req.best_expression_error_ratio(accuracy)
            self.assertAlmostEqual(req.required_accuracy_for_error_ratio(ratio), accuracy)

    def test_floor_rejects_an_out_of_range_accuracy(self):
        with self.assertRaises(req.RequirementError):
            req.best_expression_error_ratio(1.5)

    def test_required_accuracy_rejects_a_ratio_above_one(self):
        with self.assertRaises(req.RequirementError):
            req.required_accuracy_for_error_ratio(1.2)

    def test_low_accuracy_cannot_score_on_expression_error(self):
        """The load-bearing consequence: at the candidate's measured 2.5% accuracy the
        best reachable expression error is still worse than the no-skill baseline, so the
        clamped score is exactly zero however the amplitude is tuned."""
        floor = req.best_expression_error_ratio(0.0248)
        self.assertGreater(floor, req.MSE_BASELINE)
        self.assertEqual(
            req.scaled_from_raw(floor, req.MSE_BASELINE, req.MSE_REPLICATE), 0.0
        )


class ScalingTest(unittest.TestCase):
    def test_baseline_scales_to_zero(self):
        self.assertAlmostEqual(
            req.scaled_from_raw(req.MSE_BASELINE, req.MSE_BASELINE, req.MSE_REPLICATE), 0.0
        )

    def test_replicate_scales_to_one(self):
        self.assertAlmostEqual(
            req.scaled_from_raw(req.MSE_REPLICATE, req.MSE_BASELINE, req.MSE_REPLICATE), 1.0
        )

    def test_worse_than_baseline_clamps_to_zero_rather_than_going_negative(self):
        self.assertEqual(req.scaled_from_raw(4.7389, req.MSE_BASELINE, req.MSE_REPLICATE), 0.0)

    def test_better_than_the_replicate_clamps_to_one(self):
        self.assertEqual(req.scaled_from_raw(0.0, req.MSE_BASELINE, req.MSE_REPLICATE), 1.0)


class BlendTest(unittest.TestCase):
    def test_blend_realises_the_requested_cosine(self):
        rng = np.random.default_rng(4)
        truth = rng.standard_normal((300, 2000))
        for accuracy in (0.0, 0.05, 0.3, 1.0):
            blended = req.blend_toward_truth(truth, accuracy, rng)
            realised = (req.unit_rows(blended) * req.unit_rows(truth)).sum(axis=1).mean()
            self.assertAlmostEqual(float(realised), accuracy, delta=0.03)

    def test_blend_returns_unit_rows(self):
        rng = np.random.default_rng(5)
        blended = req.blend_toward_truth(rng.standard_normal((20, 50)), 0.4, rng)
        np.testing.assert_allclose(np.linalg.norm(blended, axis=1), np.ones(20))

    def test_blend_rejects_a_negative_accuracy(self):
        rng = np.random.default_rng(6)
        with self.assertRaises(req.RequirementError):
            req.blend_toward_truth(np.zeros((3, 4)), -0.1, rng)


class GateTest(unittest.TestCase):
    def test_gate_zeroes_the_least_reliable_rows(self):
        delta = np.array([[1.0], [2.0], [3.0]])
        gated = req.gate_targets(delta, np.array([0.9, 0.1, 0.5]), 1)
        np.testing.assert_array_equal(gated, [[1.0], [0.0], [3.0]])

    def test_gate_with_zero_drop_is_the_identity(self):
        delta = np.arange(6.0).reshape(3, 2)
        np.testing.assert_array_equal(req.gate_targets(delta, np.zeros(3), 0), delta)

    def test_gate_does_not_mutate_its_input(self):
        delta = np.ones((3, 2))
        req.gate_targets(delta, np.array([0.0, 1.0, 2.0]), 2)
        np.testing.assert_array_equal(delta, np.ones((3, 2)))

    def test_nan_reliability_is_dropped_first(self):
        delta = np.array([[1.0], [2.0], [3.0]])
        gated = req.gate_targets(delta, np.array([0.5, np.nan, 0.9]), 1)
        np.testing.assert_array_equal(gated, [[1.0], [0.0], [3.0]])

    def test_gate_rejects_a_mismatched_reliability_length(self):
        with self.assertRaises(req.RequirementError):
            req.gate_targets(np.ones((3, 2)), np.zeros(4), 1)

    def test_gate_rejects_an_out_of_range_drop(self):
        with self.assertRaises(req.RequirementError):
            req.gate_targets(np.ones((3, 2)), np.zeros(3), 4)


class RepoRelativeTest(unittest.TestCase):
    def test_rejects_a_path_outside_the_repository(self):
        with self.assertRaises(req.RequirementError):
            req.repo_relative(Path("/etc/hostname"), REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
