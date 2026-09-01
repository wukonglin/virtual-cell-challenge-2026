"""Tests for leakage-safe paired target-level public bootstrapping."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_public_paired_bootstrap import (  # noqa: E402
    BOOTSTRAP_SEED,
    COMPLETE_RESULT_METRICS,
    DERIVED_MSE_METRIC,
    MSE_DENOMINATOR_METRIC,
    MSE_NUMERATOR_METRIC,
    NMAE_METRIC,
    NULLABLE_COMPLETE_METRICS,
    _read_scored_results,
    bootstrap_mean_difference_summary,
    bootstrap_mse_ratio_summary,
    main,
)


def _manifest(path: Path) -> list[str]:
    targets = ["DIRECT_A", "DIRECT_B", "HELD_A", "HELD_B"]
    pd.DataFrame(
        {
            "target_gene": targets,
            "validation_route": ["direct", "direct", "held_target", "held_target"],
        }
    ).to_csv(path, index=False)
    return targets


def _results(
    path: Path,
    targets: list[str],
    *,
    candidate: bool,
    nmae_targets: tuple[str, ...] = ("DIRECT_A", "HELD_A"),
    nan_fidelity_targets: tuple[str, ...] = (),
    nan_reach_targets: tuple[str, ...] = (),
) -> None:
    rows = []
    for target_index, target in enumerate(targets):
        for metric in COMPLETE_RESULT_METRICS:
            anchor = 0.20 + target_index * 0.01
            if metric == MSE_NUMERATOR_METRIC:
                value = float((1 if candidate else 2) * (target_index + 1))
            elif metric == MSE_DENOMINATOR_METRIC:
                value = float(4 * (target_index + 1))
            elif metric == NULLABLE_COMPLETE_METRICS[0] and target in nan_fidelity_targets:
                value = np.nan
            elif metric == NULLABLE_COMPLETE_METRICS[1] and target in nan_reach_targets:
                value = np.nan
            elif not candidate:
                value = anchor
            else:
                value = anchor + 0.05
            rows.append({"perturbation": target, "metric": metric, "value": value})
        if target in nmae_targets:
            anchor = 0.20 + target_index * 0.01
            value = anchor - 0.05 if candidate else anchor
            rows.append(
                {"perturbation": target, "metric": NMAE_METRIC, "value": value}
            )
        rows.append(
            {"perturbation": target, "metric": "diagnostic_not_scored", "value": 1.0}
        )
    pd.DataFrame(rows).to_csv(path, index=False)


class PairedBootstrapTests(unittest.TestCase):
    def test_bootstrap_mean_difference_is_paired_and_deterministic(self) -> None:
        anchor = np.array([0.2, 0.3, 0.4], dtype=np.float64)
        candidate = anchor - 0.1
        first = bootstrap_mean_difference_summary(
            anchor,
            candidate,
            "lower",
            np.random.default_rng(BOOTSTRAP_SEED),
        )
        second = bootstrap_mean_difference_summary(
            anchor,
            candidate,
            "lower",
            np.random.default_rng(BOOTSTRAP_SEED),
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["observed_mean_raw_candidate_minus_anchor"], -0.1)
        self.assertAlmostEqual(
            first["observed_mean_oriented_candidate_minus_anchor"], 0.1
        )
        self.assertAlmostEqual(
            first["bootstrap_90pct_two_sided_ci"]["lower"], 0.1
        )
        self.assertAlmostEqual(
            first["bootstrap_90pct_one_sided_lower_bound"], 0.1
        )
        self.assertTrue(first["one_sided_lower_bound_above_zero"])

    def test_nullable_metrics_use_paired_axis_with_independent_nanmeans(self) -> None:
        summary = bootstrap_mean_difference_summary(
            np.array([1.0, np.nan, 3.0]),
            np.array([2.0, 4.0, np.nan]),
            "higher",
            np.random.default_rng(BOOTSTRAP_SEED),
        )
        self.assertEqual(summary["anchor_finite_targets"], 2)
        self.assertEqual(summary["candidate_finite_targets"], 2)
        self.assertEqual(summary["finite_target_intersection"], 1)
        self.assertFalse(summary["finite_target_masks_identical"])
        self.assertAlmostEqual(summary["observed_anchor_mean"], 2.0)
        self.assertAlmostEqual(summary["observed_candidate_mean"], 3.0)

    def test_mse_bootstrap_recomputes_ratio_of_sums(self) -> None:
        anchor_num = np.array([2.0, 0.10])
        anchor_den = np.array([8.0, 0.01])
        candidate_num = anchor_num / 2.0
        candidate_den = anchor_den.copy()
        summary = bootstrap_mse_ratio_summary(
            anchor_num,
            anchor_den,
            candidate_num,
            candidate_den,
            np.random.default_rng(BOOTSTRAP_SEED),
        )
        expected_anchor = anchor_num.sum() / anchor_den.sum()
        self.assertAlmostEqual(
            summary["observed_anchor_ratio_of_sums"],
            expected_anchor,
        )
        self.assertNotAlmostEqual(
            expected_anchor,
            np.mean(anchor_num / anchor_den),
        )
        self.assertAlmostEqual(
            summary["observed_oriented_candidate_minus_anchor"],
            expected_anchor / 2.0,
        )

    def test_main_labels_roles_and_respects_matched_nmae_omissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            targets = _manifest(manifest)
            paths = {
                name: root / f"{name}.csv"
                for name in (
                    "hepg2_anchor",
                    "hepg2_candidate",
                    "jurkat_anchor",
                    "jurkat_candidate",
                )
            }
            for name, path in paths.items():
                _results(path, targets, candidate=name.endswith("candidate"))
            output = root / "bootstrap.json"
            argv = [
                "analyze_public_paired_bootstrap.py",
                "--manifest",
                str(manifest),
                "--hepg2-anchor",
                str(paths["hepg2_anchor"]),
                "--hepg2-candidate",
                str(paths["hepg2_candidate"]),
                "--jurkat-anchor",
                str(paths["jurkat_anchor"]),
                "--jurkat-candidate",
                str(paths["jurkat_candidate"]),
                "--output-json",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                main()

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                report["context_roles"], {"HepG2": "primary", "Jurkat": "diagnostic"}
            )
            self.assertEqual(report["configuration"]["bootstrap_resamples"], 10_000)
            self.assertEqual(report["configuration"]["bootstrap_seed"], BOOTSTRAP_SEED)
            self.assertEqual(report["contexts"]["HepG2"]["matched_nmae_targets"], 2)
            direct_nmae = report["contexts"]["HepG2"]["cohorts"]["direct"][
                "metrics"
            ][NMAE_METRIC]
            self.assertEqual(direct_nmae["resampling_axis_targets"], 1)
            self.assertEqual(direct_nmae["omitted_cohort_targets"], 1)
            mse = report["contexts"]["HepG2"]["cohorts"]["all"]["metrics"][
                DERIVED_MSE_METRIC
            ]
            self.assertEqual(mse["aggregation"], "ratio_of_sums")
            self.assertAlmostEqual(mse["observed_anchor_ratio_of_sums"], 0.5)
            self.assertAlmostEqual(mse["observed_candidate_ratio_of_sums"], 0.25)
            self.assertAlmostEqual(
                mse["observed_raw_candidate_minus_anchor"], -0.25
            )
            self.assertAlmostEqual(
                mse["observed_oriented_candidate_minus_anchor"], 0.25
            )
            self.assertEqual(
                mse["orientation_transform"],
                "negative_candidate_minus_anchor",
            )
            self.assertFalse(report["leakage_firewall"]["reads_truth_h5ad"])

    def test_reader_fails_on_missing_complete_axis_and_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets = _manifest(root / "manifest.csv")
            path = root / "results.csv"
            _results(path, targets, candidate=False)
            frame = pd.read_csv(path)
            missing = frame.loc[
                ~(
                    (frame["perturbation"] == "DIRECT_A")
                    & (frame["metric"] == COMPLETE_RESULT_METRICS[0])
                )
            ]
            missing.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "Complete scored and MSE-component"):
                _read_scored_results(path, frozenset(targets))

            _results(path, targets, candidate=False)
            frame = pd.read_csv(path)
            duplicate = pd.concat(
                [frame, frame.loc[[frame.index[0]]]],
                ignore_index=True,
            )
            duplicate.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "Duplicate target/metric"):
                _read_scored_results(path, frozenset(targets))

    def test_main_fails_on_nmae_subset_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            targets = _manifest(manifest)
            paths = {
                name: root / f"{name}.csv"
                for name in (
                    "hepg2_anchor",
                    "hepg2_candidate",
                    "jurkat_anchor",
                    "jurkat_candidate",
                )
            }
            _results(paths["hepg2_anchor"], targets, candidate=False)
            _results(
                paths["hepg2_candidate"],
                targets,
                candidate=True,
                nmae_targets=("DIRECT_B", "HELD_A"),
            )
            _results(paths["jurkat_anchor"], targets, candidate=False)
            _results(paths["jurkat_candidate"], targets, candidate=True)
            argv = [
                "analyze_public_paired_bootstrap.py",
                "--manifest",
                str(manifest),
                "--hepg2-anchor",
                str(paths["hepg2_anchor"]),
                "--hepg2-candidate",
                str(paths["hepg2_candidate"]),
                "--jurkat-anchor",
                str(paths["jurkat_anchor"]),
                "--jurkat-candidate",
                str(paths["jurkat_candidate"]),
                "--output-json",
                str(root / "output.json"),
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(RuntimeError, "NMAE omission subsets differ"):
                    main()

    def test_reader_fails_on_empty_or_nonfinite_nmae_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets = _manifest(root / "manifest.csv")
            path = root / "results.csv"
            _results(path, targets, candidate=False, nmae_targets=())
            with self.assertRaisesRegex(RuntimeError, "NMAE matched omission subset is empty"):
                _read_scored_results(path, frozenset(targets))

            _results(path, targets, candidate=False)
            frame = pd.read_csv(path)
            selector = frame["metric"] == NMAE_METRIC
            frame.loc[selector, "value"] = np.nan
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "NMAE matched omission subset contains"):
                _read_scored_results(path, frozenset(targets))

    def test_reader_accepts_nullable_rows_but_rejects_bad_mse_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets = _manifest(root / "manifest.csv")
            path = root / "results.csv"
            _results(
                path,
                targets,
                candidate=False,
                nan_fidelity_targets=("DIRECT_A",),
                nan_reach_targets=("HELD_A",),
            )
            parsed = _read_scored_results(path, frozenset(targets))
            self.assertEqual(
                int(
                    parsed.loc[
                        parsed["metric"] == NULLABLE_COMPLETE_METRICS[0], "value"
                    ].isna().sum()
                ),
                1,
            )

            bad_mse = pd.read_csv(path)
            selector = (
                (bad_mse["perturbation"] == "DIRECT_A")
                & (bad_mse["metric"] == MSE_NUMERATOR_METRIC)
            )
            bad_mse.loc[selector, "value"] = np.nan
            bad_mse.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "all-finite metric"):
                _read_scored_results(path, frozenset(targets))

            _results(path, targets, candidate=False)
            derived = pd.read_csv(path)
            derived.loc[len(derived)] = {
                "perturbation": "DIRECT_A",
                "metric": DERIVED_MSE_METRIC,
                "value": 0.5,
            }
            derived.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "must not appear as a per-target"):
                _read_scored_results(path, frozenset(targets))

    def test_main_allows_model_dependent_nullable_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            targets = _manifest(manifest)
            paths = {
                name: root / f"{name}.csv"
                for name in (
                    "hepg2_anchor",
                    "hepg2_candidate",
                    "jurkat_anchor",
                    "jurkat_candidate",
                )
            }
            _results(
                paths["hepg2_anchor"],
                targets,
                candidate=False,
                nan_fidelity_targets=("DIRECT_A",),
            )
            _results(
                paths["hepg2_candidate"],
                targets,
                candidate=True,
                nan_fidelity_targets=("DIRECT_B",),
            )
            _results(paths["jurkat_anchor"], targets, candidate=False)
            _results(paths["jurkat_candidate"], targets, candidate=True)
            output = root / "output.json"
            argv = [
                "analyze_public_paired_bootstrap.py",
                "--manifest",
                str(manifest),
                "--hepg2-anchor",
                str(paths["hepg2_anchor"]),
                "--hepg2-candidate",
                str(paths["hepg2_candidate"]),
                "--jurkat-anchor",
                str(paths["jurkat_anchor"]),
                "--jurkat-candidate",
                str(paths["jurkat_candidate"]),
                "--output-json",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                main()
            report = json.loads(output.read_text(encoding="utf-8"))
            metric = report["contexts"]["HepG2"]["cohorts"]["direct"][
                "metrics"
            ][NULLABLE_COMPLETE_METRICS[0]]
            self.assertFalse(metric["finite_target_masks_identical"])
            self.assertEqual(metric["anchor_finite_targets"], 1)
            self.assertEqual(metric["candidate_finite_targets"], 1)


if __name__ == "__main__":
    unittest.main()
