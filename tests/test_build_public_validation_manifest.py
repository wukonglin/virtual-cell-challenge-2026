"""Tests for deterministic public validation-panel construction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_public_validation_manifest import (  # noqa: E402
    balanced_panel_order,
    choose_holdout_clusters,
)
from build_public_validation_gene_axis import shared_axis_in_state_order  # noqa: E402


class PublicPanelSelectionTests(unittest.TestCase):
    def test_shared_gene_axis_preserves_state_order(self) -> None:
        axis = shared_axis_in_state_order(
            ["g3", "g1", "g2", "g0"],
            {"g0", "g1", "g3"},
            {"g1", "g2", "g3"},
        )
        self.assertEqual(axis, ["g3", "g1"])

    def test_balanced_order_is_deterministic_and_round_robins_clusters(self) -> None:
        targets = ["a0", "a1", "a2", "b0", "b1", "c0"]
        labels = np.asarray([0, 0, 0, 1, 1, 2])
        first = balanced_panel_order(targets, labels, seed=17)
        second = balanced_panel_order(targets, labels, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(targets))
        label_by_target = dict(zip(targets, labels.tolist(), strict=True))
        self.assertEqual({label_by_target[target] for target in first[:3]}, {0, 1, 2})

    def test_holdout_selection_uses_complete_clusters_and_hits_exact_total(self) -> None:
        counts = {0: 10, 1: 12, 2: 8, 3: 13}
        selected = choose_holdout_clusters(counts, desired_targets=33)
        self.assertEqual(sum(counts[cluster] for cluster in selected), 33)
        self.assertEqual(selected, choose_holdout_clusters(counts, desired_targets=33))

    def test_holdout_selection_uses_nearest_deterministic_total_if_no_exact_sum(self) -> None:
        counts = {0: 4, 1: 7, 2: 9}
        selected = choose_holdout_clusters(counts, desired_targets=8)
        self.assertEqual(selected, (1,))


if __name__ == "__main__":
    unittest.main()
