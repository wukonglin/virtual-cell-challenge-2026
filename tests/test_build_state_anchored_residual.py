"""Unit tests for the STATE-anchored public residual builder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_state_anchored_residual import build_gated_residual  # noqa: E402


class GatedResidualTests(unittest.TestCase):
    def test_centering_reliability_detection_and_masks_are_applied(self) -> None:
        learned = np.asarray(
            [
                [3.0, 1.0, 4.0, 0.0, 5.0, 2.0],
                [1.0, 3.0, 0.0, 4.0, 1.0, 2.0],
                [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            ],
            dtype=np.float32,
        )
        direct = np.asarray([True, True, False])
        fallback = ~direct
        reliability = np.asarray([1.0, 0.5, 0.0], dtype=np.float32)
        detection = np.asarray(
            [
                [1.0, 1.0, 1.0, 0.001, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        mutable = np.asarray([True, True, True, True, False, True])
        query = np.asarray([True, True, True, False, True, True])
        panel = np.asarray([False, True, False, False, False, False])
        output, qc = build_gated_residual(
            learned,
            direct,
            fallback,
            reliability,
            detection,
            np.ones_like(detection),
            mutable,
            query,
            panel,
            top_k=2,
            min_context_detection_rate=0.01,
            min_context_mean_count=0.01,
            direct_reliability_power=1.0,
            fallback_reliability=0.0,
            centering="mean",
        )

        self.assertEqual(output.shape, (2, 3, 6))
        self.assertFalse(np.any(output[:, :, 1]))
        self.assertFalse(np.any(output[:, :, 3]))
        self.assertFalse(np.any(output[:, :, 4]))
        self.assertFalse(np.any(output[:, 2]))
        self.assertEqual(np.count_nonzero(output[0, 0]), 2)
        self.assertEqual(np.count_nonzero(output[1, 0]), 2)
        self.assertEqual(np.count_nonzero(output[0, 1]), 2)
        self.assertTrue(np.all(np.count_nonzero(output, axis=2) <= 2))
        np.testing.assert_allclose(qc["target_reliability"], [1.0, 0.5, 0.0])

    def test_top_k_uses_gene_index_as_a_deterministic_tie_breaker(self) -> None:
        learned = np.asarray([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], dtype=np.float32)
        output, _ = build_gated_residual(
            learned,
            np.asarray([True, True]),
            np.asarray([False, False]),
            np.ones(2, dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
            np.ones(3, dtype=bool),
            np.ones(3, dtype=bool),
            np.zeros(3, dtype=bool),
            top_k=2,
            min_context_detection_rate=0.0,
            min_context_mean_count=0.0,
            direct_reliability_power=1.0,
            fallback_reliability=0.0,
            centering="mean",
        )
        self.assertNotEqual(float(output[0, 0, 0]), 0.0)
        self.assertNotEqual(float(output[0, 0, 1]), 0.0)
        self.assertEqual(float(output[0, 0, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
