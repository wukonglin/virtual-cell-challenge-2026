"""Unit tests for the target-generalizing signature model."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_target_signature_model import (  # noqa: E402
    load_esm2_partition,
    make_cluster_split,
    proxy_metrics,
    select_scales,
    weighted_common_effect,
)


class ESM2CoverageTests(unittest.TestCase):
    def test_missing_training_targets_are_filtered_with_stable_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "esm2.pt"
            torch.save(
                {
                    "A": torch.tensor([1.0, 2.0]),
                    "B": torch.tensor([3.0, 4.0]),
                    "P": torch.tensor([5.0, 6.0]),
                },
                path,
            )
            partition = load_esm2_partition(
                path,
                ["A", "missing", "B"],
                ["P", "A"],
            )
        np.testing.assert_array_equal(partition.training_indices, [0, 2])
        np.testing.assert_array_equal(
            partition.training_features,
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            partition.prediction_features,
            np.asarray([[5.0, 6.0], [1.0, 2.0]], dtype=np.float32),
        )
        self.assertEqual(partition.excluded_training_target_names, ("missing",))
        self.assertEqual(
            partition.metadata(),
            {
                "training_targets_before_esm2_filter": 3,
                "training_targets_after_esm2_filter": 2,
                "excluded_training_targets_missing_esm2": 1,
                "excluded_training_target_names": ["missing"],
                "prediction_targets_requested": 2,
                "prediction_targets_missing_esm2": 0,
            },
        )

    def test_missing_prediction_target_remains_a_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "esm2.pt"
            torch.save(
                {
                    "A": torch.tensor([1.0, 2.0]),
                    "B": torch.tensor([3.0, 4.0]),
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "prediction targets"):
                load_esm2_partition(path, ["A", "B", "missing"], ["P"])


class CommonResponseTests(unittest.TestCase):
    def test_self_knockdown_is_removed_from_common_and_residual(self) -> None:
        effects = np.asarray(
            [
                [-4.0, 1.0, 0.0, 0.2],
                [2.0, -3.0, 0.5, 0.1],
                [2.0, 1.0, -5.0, -0.2],
            ],
            dtype=np.float32,
        )
        common, residual = weighted_common_effect(
            effects,
            np.ones(3, dtype=np.float32),
            ["A", "B", "C"],
            ["A", "B", "C", "D"],
        )
        np.testing.assert_allclose(common[:3], np.asarray([2.0, 1.0, 0.25]))
        np.testing.assert_array_equal(np.diag(residual[:, :3]), np.zeros(3))


class ProxyMetricTests(unittest.TestCase):
    def test_perfect_target_specific_predictions_score_perfectly(self) -> None:
        targets = ["T0", "T1", "T2", "T3"]
        genes = targets + ["D0", "D1", "D2", "D3"]
        truth = np.zeros((4, 8), dtype=np.float32)
        for row in range(4):
            truth[row, row] = -3.0
            truth[row, 4 + row] = 1.0
        metrics = proxy_metrics(truth.copy(), truth, targets, genes, top_k=2)
        self.assertAlmostEqual(metrics["pds_cosine_proxy"], 1.0)
        self.assertAlmostEqual(metrics["expression_mse_ratio_proxy"], 0.0)
        self.assertAlmostEqual(metrics["nmae_proxy"], 0.0)
        self.assertAlmostEqual(metrics["direction_fidelity_proxy"], 1.0)

    def test_scale_screen_recovers_exact_grid_point(self) -> None:
        targets = ["T0", "T1", "T2", "T3"]
        genes = targets + ["D0", "D1", "D2", "D3"]
        common = np.asarray([0.0, 0.0, 0.0, 0.0, 0.2, -0.1, 0.1, -0.2])
        residual = np.zeros((4, 8), dtype=np.float32)
        for row in range(4):
            residual[row, 4 + row] = 1.0
        truth = 0.5 * common[None, :] + 0.75 * residual
        selected, _ = select_scales(residual, truth, common, targets, genes)
        self.assertEqual(selected["common_scale"], 0.5)
        self.assertEqual(selected["residual_scale"], 0.75)
        self.assertAlmostEqual(
            selected["metrics"]["expression_mse_ratio_proxy"], 0.0
        )


class SplitTests(unittest.TestCase):
    def test_cluster_holdout_has_no_cluster_overlap(self) -> None:
        features = np.random.default_rng(4).normal(size=(120, 32)).astype(np.float32)
        train, validation, labels = make_cluster_split(features, 8, 0.2, 9)
        self.assertGreater(len(train), len(validation))
        self.assertTrue(set(labels[train]).isdisjoint(set(labels[validation])))


if __name__ == "__main__":
    unittest.main()
