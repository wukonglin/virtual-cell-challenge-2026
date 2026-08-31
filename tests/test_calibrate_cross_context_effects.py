"""Focused tests for cross-context effect calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from calibrate_cross_context_effects import (  # noqa: E402
    apply_ridge_route,
    compose_effects,
    fit_ridge_route,
    make_cluster_holdout,
    merge_direct_and_fallback_effects,
    prepare_holdout_residuals,
    reliability_from_counts,
    run,
    scorer_proxy_metrics,
    shrink_measured_residuals,
    weighted_common_response,
)


class ReliabilityTests(unittest.TestCase):
    def test_cell_count_reliability_is_monotonic_and_bounded(self) -> None:
        values = reliability_from_counts(np.asarray([0, 30, 60, 120]), tau=60.0)
        np.testing.assert_allclose(values, np.asarray([0.0, 1 / 3, 0.5, 2 / 3]))
        self.assertTrue(np.all(np.diff(values) > 0))
        self.assertTrue(np.all((0 <= values) & (values < 1)))


class SharedResponseTests(unittest.TestCase):
    def test_fallback_fills_genes_absent_from_a_direct_target_atlas(self) -> None:
        direct = np.asarray(
            [[10.0, 0.0, 30.0], [0.0, 0.0, 0.0]], dtype=np.float32
        )
        fallback = np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        merged = merge_direct_and_fallback_effects(
            direct,
            fallback,
            np.asarray([True, False]),
            np.asarray([True, False, True]),
        )
        np.testing.assert_array_equal(
            merged,
            np.asarray([[10.0, 2.0, 30.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        )

    def test_reliability_shrinks_only_measured_target_gene_coordinates(self) -> None:
        residuals = np.ones((2, 3), dtype=np.float32)
        shrunk = shrink_measured_residuals(
            residuals,
            np.asarray([True, False]),
            np.asarray([True, False, True]),
            np.asarray([0.25, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            shrunk,
            np.asarray([[0.25, 1.0, 0.25], [1.0, 1.0, 1.0]], dtype=np.float32),
        )

    def test_common_response_excludes_each_targets_own_knockdown(self) -> None:
        effects = np.asarray(
            [
                [-9.0, 2.0, 1.0, 0.5],
                [1.0, -8.0, 1.0, 0.5],
                [1.0, 2.0, -7.0, 0.5],
            ],
            dtype=np.float32,
        )
        common = weighted_common_response(
            effects,
            np.ones(3, dtype=np.float32),
            ["A", "B", "C"],
            ["A", "B", "C", "D"],
        )
        np.testing.assert_allclose(common, np.asarray([1.0, 2.0, 1.0, 0.5]))

    def test_routing_and_scaling_do_not_modify_explicit_common(self) -> None:
        common = np.asarray([0.4, -0.2, 0.1], dtype=np.float32)
        source = np.asarray([[1.0, 0.0, -1.0]], dtype=np.float32)
        routed = np.asarray([[2.0, 1.0, -0.5]], dtype=np.float32)
        effects, learned = compose_effects(common, source, routed, 0.75, 0.5)
        np.testing.assert_allclose(effects - learned, common[None, :])


class RouteTests(unittest.TestCase):
    def test_validation_recipient_labels_do_not_leak_into_holdout_preprocessing(self) -> None:
        rng = np.random.default_rng(3)
        targets = [f"T{index}" for index in range(10)]
        genes = targets + [f"G{index}" for index in range(4)]
        source = rng.normal(size=(10, len(genes))).astype(np.float32)
        recipient = rng.normal(size=(10, len(genes))).astype(np.float32)
        reliability = np.linspace(0.2, 0.9, 10, dtype=np.float32)
        train = np.arange(7, dtype=np.int64)
        validation = np.arange(7, 10, dtype=np.int64)

        baseline = prepare_holdout_residuals(
            source,
            recipient,
            reliability,
            reliability[::-1].copy(),
            train,
            validation,
            targets,
            genes,
        )
        changed_recipient = recipient.copy()
        changed_recipient[validation] += 10_000.0
        counterfactual = prepare_holdout_residuals(
            source,
            changed_recipient,
            reliability,
            reliability[::-1].copy(),
            train,
            validation,
            targets,
            genes,
        )
        for baseline_value, counterfactual_value in zip(
            baseline, counterfactual, strict=True
        ):
            np.testing.assert_array_equal(baseline_value, counterfactual_value)

    def test_ridge_route_recovers_a_linear_recipient_residual(self) -> None:
        rng = np.random.default_rng(8)
        source = rng.normal(size=(40, 6)).astype(np.float32)
        transform = rng.normal(size=(6, 6)).astype(np.float32)
        recipient = source @ transform + np.asarray(
            [0.2, -0.3, 0.1, 0.4, -0.2, 0.05], dtype=np.float32
        )
        route = fit_ridge_route(
            source,
            recipient,
            np.ones(6, dtype=bool),
            np.ones(40, dtype=np.float32),
            source_rank=6,
            recipient_rank=6,
            alpha=1e-8,
            seed=4,
        )
        prediction = apply_ridge_route(route, source)
        np.testing.assert_allclose(prediction, recipient, atol=2e-4, rtol=2e-4)

    def test_whole_embedding_clusters_do_not_cross_the_split(self) -> None:
        rng = np.random.default_rng(11)
        centers = np.eye(4, 8, dtype=np.float32) * 8
        features = np.concatenate(
            [center[None, :] + 0.02 * rng.normal(size=(10, 8)) for center in centers]
        ).astype(np.float32)
        train, validation, labels = make_cluster_holdout(features, 4, 0.2, 17)
        self.assertGreater(len(train), len(validation))
        self.assertTrue(set(labels[train]).isdisjoint(set(labels[validation])))


class ProxyMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = ["T0", "T1", "T2"]
        self.genes = self.targets + ["D0", "D1", "D2"]
        self.truth = np.zeros((3, 6), dtype=np.float32)
        for row in range(3):
            self.truth[row, row] = -5.0
            self.truth[row, 3 + row] = 1.0

    def test_pds_masks_every_panel_target_coordinate(self) -> None:
        prediction = self.truth.copy()
        prediction[:, :3] = np.asarray(
            [[100, -50, 20], [-20, 100, -30], [60, 70, -100]], dtype=np.float32
        )
        metrics = scorer_proxy_metrics(
            prediction, self.truth, self.targets, self.genes, top_k=1
        )
        self.assertAlmostEqual(metrics["pds_tie_aware_panel_masked"], 1.0)

    def test_pds_assigns_chance_to_a_complete_tie(self) -> None:
        prediction = np.zeros_like(self.truth)
        prediction[:, 3:] = 1.0
        metrics = scorer_proxy_metrics(
            prediction, self.truth, self.targets, self.genes, top_k=1
        )
        self.assertAlmostEqual(metrics["pds_tie_aware_panel_masked"], 0.5)


def write_atlas(
    path: Path,
    effects: np.ndarray,
    targets: list[str],
    genes: list[str],
    counts: np.ndarray,
    **extra: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        effects=effects.astype(np.float32),
        target_names=np.asarray(targets, dtype="U"),
        gene_names=np.asarray(genes, dtype="U"),
        target_cell_counts=np.asarray(counts, dtype=np.int64),
        **extra,
    )


class EndToEndTests(unittest.TestCase):
    def test_run_writes_contract_masks_hashes_and_context_reduction(self) -> None:
        rng = np.random.default_rng(23)
        paired = [f"E{index}" for index in range(12)]
        challenge = ["C0", "C1"]
        genes = paired + challenge + [f"G{index}" for index in range(6)]
        source_effects = rng.normal(scale=0.2, size=(len(paired), len(genes))).astype(
            np.float32
        )
        recipient_effects = 0.15 + 1.3 * source_effects
        query_targets = ["C0", "Q0", "Q1"]
        query_effects = rng.normal(scale=0.2, size=(3, len(genes))).astype(np.float32)
        fallback_effects = rng.normal(
            scale=0.1, size=(3, len(challenge), len(genes))
        ).astype(np.float32)
        embedding_names = paired + challenge
        embeddings = rng.normal(size=(len(embedding_names), 16)).astype(np.float32)

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            query_path = directory / "query.npz"
            source_path = directory / "source.npz"
            recipient_path = directory / "recipient.npz"
            fallback_path = directory / "fallback.npz"
            targets_path = directory / "targets.csv"
            output_npz = directory / "calibrated.npz"
            output_checkpoint = directory / "calibrated.pt"
            output_json = directory / "calibrated.json"
            write_atlas(
                query_path,
                query_effects,
                query_targets,
                genes,
                np.asarray([120, 80, 90]),
            )
            write_atlas(
                source_path,
                source_effects,
                paired,
                genes,
                np.arange(40, 52),
            )
            write_atlas(
                recipient_path,
                recipient_effects,
                paired,
                genes,
                np.arange(50, 62),
            )
            np.savez_compressed(
                fallback_path,
                effects=fallback_effects,
                target_names=np.asarray(challenge, dtype="U"),
                gene_names=np.asarray(genes, dtype="U"),
                embedding_target_names=np.asarray(embedding_names, dtype="U"),
                esm2_embeddings=embeddings,
            )
            pd.DataFrame({"target_gene": challenge}).to_csv(targets_path, index=False)
            args = argparse.Namespace(
                query_atlas=query_path,
                source_essential_atlas=source_path,
                recipient_essential_atlas=recipient_path,
                fallback_prior=fallback_path,
                challenge_targets=targets_path,
                output_npz=output_npz,
                output_checkpoint=output_checkpoint,
                output_json=output_json,
                target_column="target_gene",
                challenge_genes=None,
                esm2_embeddings=None,
                source_rank=4,
                recipient_rank=4,
                clusters=3,
                validation_fraction=0.25,
                reliability_tau=60.0,
                ridge_grid="0.01,1",
                residual_scale_grid="0.5,1",
                route_scale_grid="0,1",
                top_k=3,
                seed=31,
            )
            report = run(args)

            with np.load(output_npz, allow_pickle=False) as artifact:
                expected_keys = {
                    "effects",
                    "target_names",
                    "gene_names",
                    "target_cell_counts",
                    "genes",
                    "targets",
                    "common_log_fold",
                    "learned_log_fold",
                    "direct_mask",
                    "fallback_mask",
                    "source_reliability",
                }
                self.assertTrue(expected_keys.issubset(artifact.files))
                np.testing.assert_array_equal(artifact["direct_mask"], [True, False])
                np.testing.assert_array_equal(artifact["fallback_mask"], [False, True])
                np.testing.assert_allclose(
                    artifact["effects"],
                    artifact["common_log_fold"][None, :] + artifact["learned_log_fold"],
                )

            parsed = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(
                parsed["input_effect_reductions"]["fallback_prior"],
                "arithmetic-mean-across-3-contexts",
            )
            self.assertEqual(parsed["coverage"]["direct_targets"], 1)
            self.assertEqual(parsed["coverage"]["fallback_targets"], 1)
            self.assertEqual(parsed["selection"]["candidate_count"], 8)
            self.assertEqual(
                parsed["outputs"]["effect_prior"]["sha256"],
                hashlib.sha256(output_npz.read_bytes()).hexdigest(),
            )
            self.assertEqual(report["schema"], parsed["schema"])


if __name__ == "__main__":
    unittest.main()
