"""Tests for scorer-aligned public candidate comparison."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from summarize_public_validation import (  # noqa: E402
    EMITTED_METRICS,
    FIDELITY_METRIC,
    MSE_DENOMINATOR_METRIC,
    MSE_METRIC,
    MSE_NUMERATOR_METRIC,
    SCORED_METRICS,
    NMAE_METRIC,
    _mean_summary,
    _read_results,
    _validate_cell_eval_sidecars,
    main,
    non_regression_gate,
    oriented_improvement,
    promotion_gate,
)


def write_cell_eval_bundle(
    results_path: Path,
    rows: list[dict[str, object]],
    aggregate_overrides: dict[str, float] | None = None,
) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(results_path, index=False)
    aggregation_rows: list[dict[str, object]] = []
    for metric in sorted(EMITTED_METRICS):
        values = pd.to_numeric(
            frame.loc[frame["metric"] == metric, "value"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        aggregation_rows.append(
            {
                "metric": metric,
                "agg": "mean",
                "n_used": int(np.isfinite(values).sum()),
                "n_rows": int(values.size),
                "n_nan": int((~np.isfinite(values)).sum()),
                "n_null": 0,
                "derived": False,
            }
        )
    aggregation_rows.append(
        {
            "metric": MSE_METRIC,
            "agg": "ratio_of_sums",
            "n_used": 0,
            "n_rows": 0,
            "n_nan": 0,
            "n_null": 0,
            "derived": True,
        }
    )
    pd.DataFrame(aggregation_rows).to_csv(
        results_path.with_name("metric_aggregation.csv"), index=False
    )
    means: dict[str, float] = {}
    for metric in set(SCORED_METRICS) - {MSE_METRIC}:
        values = pd.to_numeric(
            frame.loc[frame["metric"] == metric, "value"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        means[metric] = float(values[np.isfinite(values)].mean())
    numerator = pd.to_numeric(
        frame.loc[frame["metric"] == MSE_NUMERATOR_METRIC, "value"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    denominator = pd.to_numeric(
        frame.loc[frame["metric"] == MSE_DENOMINATOR_METRIC, "value"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    means[MSE_METRIC] = float(numerator.sum() / denominator.sum())
    means.update(aggregate_overrides or {})
    pd.DataFrame([{"statistic": "mean", **means}]).to_csv(
        results_path.with_name("agg_results.csv"), index=False
    )


class PublicValidationSummaryTests(unittest.TestCase):
    def test_improvements_respect_metric_direction(self) -> None:
        self.assertAlmostEqual(oriented_improvement("pds_cosine", 0.7, 0.6), 0.1)
        self.assertAlmostEqual(
            oriented_improvement("expr_mse_unbiased_capped_norm", 0.2, 0.3), 0.1
        )

    def test_promotion_requires_all_non_regression_constraints(self) -> None:
        improvements = {
            "pds_cosine": 0.01,
            "expr_mse_unbiased_capped_norm": 0.0,
            "de_wilcoxon_lfc_nmae": 0.0,
            "de_wilcoxon_direction_fidelity_yield_raw": -0.5,
            "de_wilcoxon_direction_reach_raw": 0.0,
            "de_wilcoxon_sig_jaccard": 0.0,
        }
        self.assertTrue(promotion_gate(improvements, 1e-12)["passed"])
        improvements["de_wilcoxon_lfc_nmae"] = -0.001
        self.assertFalse(promotion_gate(improvements, 1e-12)["passed"])

    def test_jaccard_can_satisfy_discrimination_gate(self) -> None:
        improvements = {
            "pds_cosine": 0.0,
            "expr_mse_unbiased_capped_norm": 0.0,
            "de_wilcoxon_lfc_nmae": 0.0,
            "de_wilcoxon_direction_fidelity_yield_raw": 0.0,
            "de_wilcoxon_direction_reach_raw": 0.0,
            "de_wilcoxon_sig_jaccard": 0.01,
        }
        self.assertTrue(promotion_gate(improvements, 1e-12)["passed"])

    def test_main_accepts_valid_manifest_route_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            baseline_results = root / "baseline" / "results.csv"
            candidate_results = root / "candidate" / "results.csv"
            output = root / "summary.json"
            pd.DataFrame(
                {
                    "target_gene": ["GENE_A", "GENE_B"],
                    "validation_route": ["direct", "held_target"],
                }
            ).to_csv(manifest, index=False)

            rows = [
                {"perturbation": target, "metric": metric, "value": 0.5}
                for target in ("GENE_A", "GENE_B")
                for metric in EMITTED_METRICS
            ]
            write_cell_eval_bundle(baseline_results, rows)
            candidate_rows = [dict(row) for row in rows]
            for row in candidate_rows:
                if row["metric"] == "pds_cosine":
                    row["value"] = 0.6
            write_cell_eval_bundle(candidate_results, candidate_rows)

            argv = [
                "summarize_public_validation.py",
                "--manifest",
                str(manifest),
                "--baseline-label",
                "anchor",
                "--primary-context",
                "HepG2",
                "--non-harm-context",
                "Jurkat",
                "--entry",
                "anchor",
                "HepG2",
                str(baseline_results),
                "--entry",
                "candidate",
                "HepG2",
                str(candidate_results),
                "--entry",
                "anchor",
                "Jurkat",
                str(baseline_results),
                "--entry",
                "candidate",
                "Jurkat",
                str(baseline_results),
                "--output-json",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                main()

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], "vcc-public-validation-comparison-v1")
            self.assertEqual(
                report["cohort_sizes"], {"all": 2, "direct": 1, "held_target": 1}
            )
            experiment_gate = report["comparisons"]["candidate"]["HepG2"][
                "experiment_gate"
            ]
            self.assertEqual(
                experiment_gate,
                {
                    "role": "primary",
                    "all_full_promotion": True,
                    "direct_full_promotion": True,
                    "held_target_all_evaluable_metric_non_regression": True,
                    "passed": True,
                },
            )
            self.assertTrue(
                report["comparisons"]["candidate"]["aggregate_gate"][
                    "passed_all_contexts"
                ]
            )
            non_harm_gate = report["comparisons"]["candidate"]["Jurkat"][
                "experiment_gate"
            ]
            self.assertEqual(non_harm_gate["role"], "non_harm")
            self.assertFalse(
                non_harm_gate["strict_discrimination_improvement_required"]
            )
            self.assertTrue(non_harm_gate["passed"])

    def test_results_require_complete_metrics_and_finite_nmae_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.csv"
            targets = {"GENE_A", "GENE_B"}
            rows = [
                {"perturbation": target, "metric": metric, "value": 0.5}
                for target in targets
                for metric in EMITTED_METRICS
            ]

            complete_metric = "de_wilcoxon_sig_jaccard"
            missing_complete = [
                row
                for row in rows
                if not (
                    row["perturbation"] == "GENE_A"
                    and row["metric"] == complete_metric
                )
            ]
            pd.DataFrame(missing_complete).to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "full target coverage"):
                _read_results(path, targets)

            missing_one_nmae = [
                row
                for row in rows
                if not (
                    row["perturbation"] == "GENE_A"
                    and row["metric"] == NMAE_METRIC
                )
            ]
            pd.DataFrame(missing_one_nmae).to_csv(path, index=False)
            frame = _read_results(path, targets)
            summary = _mean_summary(frame, expected_targets=2)
            self.assertEqual(summary[NMAE_METRIC]["finite_targets"], 1)
            self.assertEqual(summary[NMAE_METRIC]["missing_targets"], 1)

            nonfinite = [dict(row) for row in rows]
            for row in nonfinite:
                if row["perturbation"] == "GENE_A" and row["metric"] == NMAE_METRIC:
                    row["value"] = float("nan")
            pd.DataFrame(nonfinite).to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                _read_results(path, targets)

    def test_mse_uses_ratio_of_sums_and_authenticates_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run" / "results.csv"
            targets = {"GENE_A", "GENE_B"}
            rows = [
                {"perturbation": target, "metric": metric, "value": 0.5}
                for target in sorted(targets)
                for metric in EMITTED_METRICS
            ]
            components = {
                ("GENE_A", MSE_NUMERATOR_METRIC): 1.0,
                ("GENE_B", MSE_NUMERATOR_METRIC): 9.0,
                ("GENE_A", MSE_DENOMINATOR_METRIC): 1.0,
                ("GENE_B", MSE_DENOMINATOR_METRIC): 3.0,
            }
            for row in rows:
                row["value"] = components.get(
                    (str(row["perturbation"]), str(row["metric"])), row["value"]
                )
            write_cell_eval_bundle(path, rows)
            frame = _read_results(path, targets)
            summary = _mean_summary(frame, expected_targets=2)
            self.assertAlmostEqual(summary[MSE_METRIC]["mean"], 2.5)
            self.assertNotAlmostEqual(summary[MSE_METRIC]["mean"], (1.0 + 3.0) / 2.0)
            _validate_cell_eval_sidecars(path, frame, summary, expected_targets=2)

            write_cell_eval_bundle(path, rows, {MSE_METRIC: 2.4})
            with self.assertRaisesRegex(RuntimeError, "authenticated components"):
                _validate_cell_eval_sidecars(path, frame, summary, expected_targets=2)

    def test_direction_nan_counts_match_metric_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run" / "results.csv"
            targets = {"GENE_A", "GENE_B"}
            rows = [
                {"perturbation": target, "metric": metric, "value": 0.5}
                for target in sorted(targets)
                for metric in EMITTED_METRICS
            ]
            for row in rows:
                if (
                    row["perturbation"] == "GENE_A"
                    and row["metric"] == FIDELITY_METRIC
                ):
                    row["value"] = float("nan")
            write_cell_eval_bundle(path, rows)
            frame = _read_results(path, targets)
            summary = _mean_summary(frame, expected_targets=2)
            self.assertEqual(summary[FIDELITY_METRIC]["finite_targets"], 1)
            _validate_cell_eval_sidecars(path, frame, summary, expected_targets=2)

            aggregation_path = path.with_name("metric_aggregation.csv")
            aggregation = pd.read_csv(aggregation_path)
            aggregation.loc[
                aggregation["metric"] == FIDELITY_METRIC, "n_used"
            ] = 2
            aggregation.to_csv(aggregation_path, index=False)
            with self.assertRaisesRegex(RuntimeError, "n_used mismatch"):
                _validate_cell_eval_sidecars(path, frame, summary, expected_targets=2)

    def test_held_target_regression_blocks_experiment_gate(self) -> None:
        improvements = {metric: 0.0 for metric in SCORED_METRICS}
        self.assertTrue(non_regression_gate(improvements, 1e-12)["passed"])
        improvements["de_wilcoxon_direction_fidelity_yield_raw"] = -0.01
        gate = non_regression_gate(improvements, 1e-12)
        self.assertFalse(gate["passed"])
        self.assertFalse(
            gate["metric_not_worse"][
                "de_wilcoxon_direction_fidelity_yield_raw"
            ]
        )

    def test_main_rejects_candidate_nmae_subset_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            baseline_results = root / "baseline" / "results.csv"
            candidate_results = root / "candidate" / "results.csv"
            targets = ["DIRECT_A", "DIRECT_B", "HELD_A", "HELD_B"]
            pd.DataFrame(
                {
                    "target_gene": targets,
                    "validation_route": [
                        "direct",
                        "direct",
                        "held_target",
                        "held_target",
                    ],
                }
            ).to_csv(manifest, index=False)
            rows = [
                {"perturbation": target, "metric": metric, "value": 0.5}
                for target in targets
                for metric in EMITTED_METRICS
            ]
            baseline_rows = [
                row
                for row in rows
                if not (
                    row["perturbation"] == "DIRECT_A"
                    and row["metric"] == NMAE_METRIC
                )
            ]
            candidate_rows = [
                row
                for row in rows
                if not (
                    row["perturbation"] == "DIRECT_B"
                    and row["metric"] == NMAE_METRIC
                )
            ]
            write_cell_eval_bundle(baseline_results, baseline_rows)
            write_cell_eval_bundle(candidate_results, candidate_rows)
            argv = [
                "summarize_public_validation.py",
                "--manifest",
                str(manifest),
                "--baseline-label",
                "anchor",
                "--primary-context",
                "HepG2",
                "--non-harm-context",
                "Jurkat",
                "--entry",
                "anchor",
                "HepG2",
                str(baseline_results),
                "--entry",
                "candidate",
                "HepG2",
                str(candidate_results),
                "--entry",
                "anchor",
                "Jurkat",
                str(baseline_results),
                "--entry",
                "candidate",
                "Jurkat",
                str(candidate_results),
                "--output-json",
                str(root / "summary.json"),
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(RuntimeError, "NMAE target subset differs"):
                    main()

    def test_main_allows_cohort_with_no_evaluable_nmae(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            baseline_results = root / "baseline" / "results.csv"
            candidate_results = root / "candidate" / "results.csv"
            output = root / "summary.json"
            targets = ["DIRECT_A", "DIRECT_B", "HELD_A", "HELD_B"]
            routes = ["direct", "direct", "held_target", "held_target"]
            pd.DataFrame(
                {"target_gene": targets, "validation_route": routes}
            ).to_csv(manifest, index=False)
            rows = [
                {"perturbation": target, "metric": metric, "value": 0.5}
                for target, route in zip(targets, routes, strict=True)
                for metric in EMITTED_METRICS
                if metric != NMAE_METRIC or route == "direct"
            ]
            write_cell_eval_bundle(baseline_results, rows)
            candidate_rows = [dict(row) for row in rows]
            for row in candidate_rows:
                if row["metric"] == "pds_cosine":
                    row["value"] = 0.6
            write_cell_eval_bundle(candidate_results, candidate_rows)
            argv = [
                "summarize_public_validation.py",
                "--manifest",
                str(manifest),
                "--baseline-label",
                "anchor",
                "--primary-context",
                "HepG2",
                "--non-harm-context",
                "Jurkat",
                "--entry",
                "anchor",
                "HepG2",
                str(baseline_results),
                "--entry",
                "candidate",
                "HepG2",
                str(candidate_results),
                "--entry",
                "anchor",
                "Jurkat",
                str(baseline_results),
                "--entry",
                "candidate",
                "Jurkat",
                str(baseline_results),
                "--output-json",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                main()

            report = json.loads(output.read_text(encoding="utf-8"))
            held_entry = report["entries"]["anchor"]["HepG2"]["held_target"]
            self.assertEqual(
                held_entry[NMAE_METRIC],
                {
                    "mean": None,
                    "finite_targets": 0,
                    "missing_targets": 2,
                    "emitted_targets": 0,
                    "aggregation": "finite_mean",
                },
            )
            held_comparison = report["comparisons"]["candidate"]["HepG2"][
                "held_target"
            ]
            self.assertEqual(
                held_comparison["not_evaluable_metrics"], [NMAE_METRIC]
            )
            self.assertNotIn(
                NMAE_METRIC,
                held_comparison["oriented_improvement_positive_is_better"],
            )
            self.assertEqual(
                held_comparison["non_regression_gate"]["not_evaluable_metrics"],
                [NMAE_METRIC],
            )
            self.assertTrue(
                report["comparisons"]["candidate"]["HepG2"]["experiment_gate"][
                    "passed"
                ]
            )

    def test_non_harm_context_regression_blocks_aggregate_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            baseline_results = root / "baseline" / "results.csv"
            primary_results = root / "primary" / "results.csv"
            non_harm_results = root / "non_harm" / "results.csv"
            output = root / "summary.json"
            targets = ("DIRECT", "HELD")
            pd.DataFrame(
                {
                    "target_gene": targets,
                    "validation_route": ("direct", "held_target"),
                }
            ).to_csv(manifest, index=False)
            rows = [
                {"perturbation": target, "metric": metric, "value": 0.5}
                for target in targets
                for metric in EMITTED_METRICS
            ]
            write_cell_eval_bundle(baseline_results, rows)
            primary_rows = [dict(row) for row in rows]
            for row in primary_rows:
                if row["metric"] == "pds_cosine":
                    row["value"] = 0.6
            write_cell_eval_bundle(primary_results, primary_rows)
            non_harm_rows = [dict(row) for row in rows]
            for row in non_harm_rows:
                if (
                    row["perturbation"] == "DIRECT"
                    and row["metric"] == MSE_NUMERATOR_METRIC
                ):
                    row["value"] = 0.6
            write_cell_eval_bundle(non_harm_results, non_harm_rows)
            argv = [
                "summarize_public_validation.py",
                "--manifest",
                str(manifest),
                "--baseline-label",
                "anchor",
                "--primary-context",
                "HepG2",
                "--non-harm-context",
                "Jurkat",
                "--entry",
                "anchor",
                "HepG2",
                str(baseline_results),
                "--entry",
                "candidate",
                "HepG2",
                str(primary_results),
                "--entry",
                "anchor",
                "Jurkat",
                str(baseline_results),
                "--entry",
                "candidate",
                "Jurkat",
                str(non_harm_results),
                "--output-json",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                main()

            report = json.loads(output.read_text(encoding="utf-8"))
            candidate = report["comparisons"]["candidate"]
            self.assertTrue(candidate["HepG2"]["experiment_gate"]["passed"])
            self.assertFalse(candidate["Jurkat"]["experiment_gate"]["passed"])
            self.assertFalse(
                candidate["Jurkat"]["experiment_gate"][
                    "direct_evaluable_metric_non_regression"
                ]
            )
            self.assertFalse(candidate["aggregate_gate"]["passed_all_contexts"])


if __name__ == "__main__":
    unittest.main()
