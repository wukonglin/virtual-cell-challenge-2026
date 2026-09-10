from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ensemble_state_seed_candidates",
    ROOT / "scripts" / "ensemble_state_seed_candidates.py",
)
assert SPEC is not None and SPEC.loader is not None
ensemble = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ensemble
SPEC.loader.exec_module(ensemble)


def small_plan() -> dict:
    return {
        "schema": ensemble.PLAN_SCHEMA,
        "lineage": ensemble.PLAN_LINEAGE,
        "context": "HepG2",
        "output_tag": "uniform_test_v1",
        "source_tags": ["seed_a", "seed_b"],
        "axes": {"contexts": 1, "targets": 2, "cells_per_target": 2, "genes": 4},
        "method": {
            "name": ensemble.METHOD_NAME,
            "weights": "uniform",
            "integerization": ensemble.INTEGERIZATION,
            "tie_break": ensemble.TIE_BREAK,
            "tie_break_seed": 17,
        },
        "evaluation_policy": {
            "absolute_tolerance": 1e-12,
            "best_individual_seed_selection_allowed": False,
            "fallback_tag": "seed_a",
            "held_target_gate": "all_evaluable_metrics_non_regression",
            "individual_scores_role": "seed_variance_diagnostic_only",
            "primary_baseline_tag": "seed_a",
            "primary_candidate_tag": "uniform_test_v1",
            "primary_gate": (
                "all_and_direct_mse_nmae_reach_non_regression_and_"
                "pds_or_jaccard_strict_improvement"
            ),
            "public_score_opens_after_ensemble_materialization": True,
        },
        "selection_policy": {
            "all_registered_sources_required": True,
            "best_public_score_seed_selection_allowed": False,
            "failed_integrity_source_exclusion_allowed": False,
            "missing_source_fallback_allowed": False,
        },
        "firewall": {
            "challenge_leaderboard_used_for_selection": False,
            "sealed_treated_profiles_read": False,
            "source_selection_uses_metrics": False,
        },
    }


class ExactMeanTests(unittest.TestCase):
    def test_uniform_mean_is_integer_canonical_and_library_exact(self) -> None:
        left = sp.csr_matrix(
            np.asarray([[3, 0, 1, 0], [0, 2, 0, 2]], dtype=np.int32)
        )
        right = sp.csr_matrix(
            np.asarray([[0, 3, 0, 1], [1, 0, 3, 0]], dtype=np.int32)
        )
        observed, qc = ensemble.mean_csr_exact_library(
            [left, right], global_row_start=0, tie_break_seed=17
        )
        self.assertEqual(observed.dtype, np.dtype("int32"))
        self.assertTrue(observed.has_canonical_format)
        self.assertTrue(
            np.array_equal(
                np.asarray(observed.sum(axis=1)).reshape(-1),
                np.asarray([4, 4]),
            )
        )
        self.assertEqual(qc["rounding_increments"], 3)
        again, again_qc = ensemble.mean_csr_exact_library(
            [right, left], global_row_start=0, tie_break_seed=17
        )
        self.assertTrue(np.array_equal(observed.toarray(), again.toarray()))
        self.assertEqual(qc, again_qc)

    def test_source_library_mismatch_fails_closed(self) -> None:
        left = sp.csr_matrix(np.asarray([[1, 1]], dtype=np.int32))
        right = sp.csr_matrix(np.asarray([[1, 2]], dtype=np.int32))
        with self.assertRaisesRegex(RuntimeError, "libraries differ"):
            ensemble.mean_csr_exact_library(
                [left, right], global_row_start=0, tie_break_seed=17
            )

    def test_four_source_output_is_a_floor_or_ceiling_of_the_uniform_mean(self) -> None:
        matrices = [
            sp.csr_matrix(np.asarray([row], dtype=np.int32))
            for row in (
                [4, 0, 0, 0],
                [0, 4, 0, 0],
                [0, 0, 4, 0],
                [1, 1, 1, 1],
            )
        ]
        observed, _ = ensemble.mean_csr_exact_library(
            matrices, global_row_start=11, tie_break_seed=17
        )
        dense = observed.toarray()[0]
        exact = np.mean([matrix.toarray()[0] for matrix in matrices], axis=0)
        self.assertTrue(np.all((dense == np.floor(exact)) | (dense == np.ceil(exact))))
        self.assertEqual(int(dense.sum()), 4)

    def test_plan_rejects_metric_based_seed_selection(self) -> None:
        plan = small_plan()
        plan["selection_policy"]["best_public_score_seed_selection_allowed"] = True
        with self.assertRaisesRegex(RuntimeError, "policy is unsafe"):
            ensemble.validate_plan(plan)


class H5adMaterializationTests(unittest.TestCase):
    def _write_source(self, path: Path, matrix: np.ndarray, *, altered_obs: bool = False) -> None:
        obs = pd.DataFrame(
            {
                "context": ["HepG2"] * 4,
                "target_gene": ["g0", "g0", "g1", "g1"],
                "source_control_row": [0, 1, 0, 1],
                "source_control_cell_id": ["c0", "c1", "c0", "c1"],
            },
            index=pd.Index(["g0|0", "g0|1", "g1|0", "g1|1"], name="cell_id"),
        )
        if altered_obs:
            obs.loc["g1|1", "source_control_row"] = 99
        var = pd.DataFrame(index=pd.Index(["g0", "g1", "g2", "g3"], name="gene"))
        ad.AnnData(X=sp.csr_matrix(matrix, dtype=np.int32), obs=obs, var=var).write_h5ad(
            path, compression="lzf"
        )

    def test_materializes_aligned_groups_and_exact_libraries(self) -> None:
        plan = small_plan()
        with tempfile.TemporaryDirectory() as text:
            root = Path(text)
            first = root / "first.h5ad"
            second = root / "second.h5ad"
            output = root / "mean.h5ad"
            self._write_source(
                first,
                np.asarray(
                    [[3, 0, 1, 0], [0, 2, 0, 2], [2, 1, 0, 1], [0, 1, 3, 0]],
                    dtype=np.int32,
                ),
            )
            self._write_source(
                second,
                np.asarray(
                    [[0, 3, 0, 1], [1, 0, 3, 0], [1, 0, 2, 1], [2, 0, 0, 2]],
                    dtype=np.int32,
                ),
            )
            report = ensemble.materialize_prediction(
                [first, second], output, plan=plan, max_loaded_elements=100
            )
            observed = ad.read_h5ad(output)
            one = ad.read_h5ad(first)
            self.assertEqual(observed.shape, (4, 4))
            self.assertTrue(np.issubdtype(observed.X.dtype, np.integer))
            self.assertTrue(
                np.array_equal(
                    np.asarray(observed.X.sum(axis=1)).reshape(-1),
                    np.asarray(one.X.sum(axis=1)).reshape(-1),
                )
            )
            self.assertEqual(observed.obs_names.tolist(), one.obs_names.tolist())
            self.assertEqual(report["groups"], 2)
            self.assertEqual(len(report["group_qc"]), 2)

    def test_observation_misalignment_fails_closed(self) -> None:
        plan = small_plan()
        with tempfile.TemporaryDirectory() as text:
            root = Path(text)
            first = root / "first.h5ad"
            second = root / "second.h5ad"
            output = root / "mean.h5ad"
            matrix = np.asarray(
                [[3, 0, 1, 0], [0, 2, 0, 2], [2, 1, 0, 1], [0, 1, 3, 0]],
                dtype=np.int32,
            )
            self._write_source(first, matrix)
            self._write_source(second, matrix, altered_obs=True)
            with self.assertRaisesRegex(RuntimeError, "observation metadata differs"):
                ensemble.materialize_prediction(
                    [first, second], output, plan=plan, max_loaded_elements=100
                )

    def test_exact_verifier_rejects_a_same_library_non_mean_output(self) -> None:
        plan = small_plan()
        with tempfile.TemporaryDirectory() as text:
            root = Path(text)
            first = root / "first.h5ad"
            second = root / "second.h5ad"
            output = root / "mean.h5ad"
            matrix_left = np.asarray(
                [[3, 0, 1, 0], [0, 2, 0, 2], [2, 1, 0, 1], [0, 1, 3, 0]],
                dtype=np.int32,
            )
            matrix_right = np.asarray(
                [[0, 3, 0, 1], [1, 0, 3, 0], [1, 0, 2, 1], [2, 0, 0, 2]],
                dtype=np.int32,
            )
            self._write_source(first, matrix_left)
            self._write_source(second, matrix_right)
            ensemble.materialize_prediction(
                [first, second], output, plan=plan, max_loaded_elements=100
            )
            forged = ad.read_h5ad(output)
            dense = forged.X.toarray()
            donor = int(np.flatnonzero(dense[0] > 0)[0])
            recipient = (donor + 1) % dense.shape[1]
            dense[0, donor] -= 1
            dense[0, recipient] += 1
            forged.X = sp.csr_matrix(dense, dtype=np.int32)
            forged.write_h5ad(output, compression="lzf")
            with self.assertRaisesRegex(RuntimeError, "not the registered deterministic mean"):
                ensemble.verify_prediction_exact_mean(
                    [first, second], output, plan=plan
                )

    def test_registered_plan_is_truth_blind_and_fixed_to_four_sources(self) -> None:
        path = ROOT / "configs" / "state" / "anvil_seed_ensemble_v1.json"
        plan = ensemble.validate_plan(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(len(plan["source_tags"]), 4)
        self.assertEqual(plan["method"]["weights"], "uniform")
        self.assertFalse(plan["selection_policy"]["best_public_score_seed_selection_allowed"])
        self.assertFalse(plan["evaluation_policy"]["best_individual_seed_selection_allowed"])
        self.assertFalse(plan["firewall"]["sealed_treated_profiles_read"])


if __name__ == "__main__":
    unittest.main()
