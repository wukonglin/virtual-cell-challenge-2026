"""Tests for leakage-safe single-context V6 candidate comparisons."""

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

from tests.scoring_receipt_test_utils import (
    descriptor,
    write_v51_receipt,
    write_v6_receipt,
)


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_public_v6_sequential import (  # noqa: E402
    SCHEMA,
    _direction_nan_diagnostics,
    main,
)
from summarize_public_validation import (  # noqa: E402
    EMITTED_METRICS,
    FIDELITY_METRIC,
    MSE_DENOMINATOR_METRIC,
    MSE_METRIC,
    MSE_NUMERATOR_METRIC,
    NMAE_METRIC,
    REACH_METRIC,
    SCORED_METRICS,
)


TARGETS = ("DIRECT_A", "DIRECT_B", "HELD_A", "HELD_B")


def _write_manifest(path: Path) -> None:
    pd.DataFrame(
        {
            "target_gene": TARGETS,
            "validation_route": (
                "direct",
                "direct",
                "held_target",
                "held_target",
            ),
        }
    ).to_csv(path, index=False)


def _write_bundle(
    path: Path,
    *,
    candidate: bool,
    nmae_targets: tuple[str, ...] = ("DIRECT_A", "HELD_A"),
    held_reach_regression: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for target_index, target in enumerate(TARGETS):
        for metric in sorted(EMITTED_METRICS):
            if metric == NMAE_METRIC and target not in nmae_targets:
                continue
            if metric == MSE_NUMERATOR_METRIC:
                value = float((1 if candidate else 2) * (target_index + 1))
            elif metric == MSE_DENOMINATOR_METRIC:
                value = float(4 * (target_index + 1))
            elif metric == NMAE_METRIC:
                value = 0.25 if candidate else 0.30
            elif metric in SCORED_METRICS:
                value = 0.25 if candidate else 0.20
                if (
                    candidate
                    and held_reach_regression
                    and target.startswith("HELD")
                    and metric == REACH_METRIC
                ):
                    value = 0.10
                if metric == FIDELITY_METRIC and target == "DIRECT_A":
                    value = np.nan
            else:
                raise AssertionError(f"Unexpected emitted metric: {metric}")
            rows.append(
                {"perturbation": target, "metric": metric, "value": value}
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)

    aggregation_rows: list[dict[str, object]] = []
    for metric in sorted(EMITTED_METRICS):
        values = pd.to_numeric(
            frame.loc[frame["metric"] == metric, "value"],
            errors="coerce",
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
        path.with_name("metric_aggregation.csv"),
        index=False,
    )

    aggregate: dict[str, float | str] = {"statistic": "mean"}
    for metric in set(SCORED_METRICS) - {MSE_METRIC}:
        values = pd.to_numeric(
            frame.loc[frame["metric"] == metric, "value"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        aggregate[metric] = float(values[np.isfinite(values)].mean())
    numerator = pd.to_numeric(
        frame.loc[frame["metric"] == MSE_NUMERATOR_METRIC, "value"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    denominator = pd.to_numeric(
        frame.loc[frame["metric"] == MSE_DENOMINATOR_METRIC, "value"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    aggregate[MSE_METRIC] = float(numerator.sum() / denominator.sum())
    pd.DataFrame([aggregate]).to_csv(path.with_name("agg_results.csv"), index=False)


def _run(
    root: Path,
    *,
    candidate_nmae_targets: tuple[str, ...] = ("DIRECT_A", "HELD_A"),
    held_reach_regression: bool = False,
    role: str = "primary",
    context: str = "HepG2",
    candidate_cli_label: str = "v6_candidate",
    tamper_candidate_results: bool = False,
    tamper_baseline_prediction_view: bool = False,
    mislabel_baseline_source_prediction: bool = False,
    tamper_manifest_routes: bool = False,
    candidate_config_drift: bool = False,
    candidate_status: str = "passed",
) -> dict[str, object]:
    manifest = root / "manifest.csv"
    baseline = root / "baseline" / "results.csv"
    candidate = root / "candidate" / "results.csv"
    output = root / "comparison.json"
    _write_manifest(manifest)
    _write_bundle(baseline, candidate=False)
    _write_bundle(
        candidate,
        candidate=True,
        nmae_targets=candidate_nmae_targets,
        held_reach_regression=held_reach_regression,
    )
    v51_receipt = write_v51_receipt(
        root / "v51_scoring_receipt.json",
        context=context,
        anchor_results=baseline,
        candidate_results=baseline,
    )
    if mislabel_baseline_source_prediction:
        anchor_views = root / "anchor_views.json"
        candidate_views = root / "v51_alpha010_views.json"
        anchor_payload = json.loads(anchor_views.read_text(encoding="utf-8"))
        candidate_payload = json.loads(candidate_views.read_text(encoding="utf-8"))
        anchor_payload["provenance"]["prediction"] = candidate_payload[
            "provenance"
        ]["prediction"]
        anchor_views.write_text(json.dumps(anchor_payload), encoding="utf-8")
        receipt_payload = json.loads(v51_receipt.read_text(encoding="utf-8"))
        receipt_payload["outputs"]["anchor"]["views_report"] = descriptor(
            anchor_views, mtime=False
        )
        v51_receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    candidate_receipt, candidate_spec = write_v6_receipt(
        root / "candidate" / "scoring_verified.json",
        context=context,
        label="v6_candidate",
        results=candidate,
        expected_targets=len(TARGETS),
        target_remaining_fraction=0.20,
        residual_alpha=0.10,
        manifest=manifest,
    )
    if candidate_config_drift:
        run_meta = candidate.parent / "run_meta.json"
        metadata = json.loads(run_meta.read_text(encoding="utf-8"))
        metadata["config_digest"] = "4" * 64
        run_meta.write_text(json.dumps(metadata), encoding="utf-8")
        receipt_payload = json.loads(candidate_receipt.read_text(encoding="utf-8"))
        receipt_payload["resolved_run"]["config_digest"] = "4" * 64
        receipt_payload["outputs"]["run_meta"] = descriptor(run_meta)
        receipt_payload["status"] = candidate_status
        candidate_receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    elif candidate_status != "passed":
        receipt_payload = json.loads(candidate_receipt.read_text(encoding="utf-8"))
        receipt_payload["status"] = candidate_status
        candidate_receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    if tamper_candidate_results:
        candidate.write_text(
            candidate.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
    if tamper_baseline_prediction_view:
        (root / "anchor_prediction_view.h5ad").write_bytes(b"tampered-view")
    if tamper_manifest_routes:
        frame = pd.read_csv(manifest)
        frame.loc[0, "validation_route"] = "held_target"
        frame.loc[2, "validation_route"] = "direct"
        frame.to_csv(manifest, index=False)
    argv = [
        "compare_public_v6_sequential.py",
        "--manifest",
        str(manifest),
        "--context",
        context,
        "--role",
        role,
        "--baseline-label",
        "anchor",
        "--baseline-results",
        str(baseline),
        "--baseline-scoring-receipt",
        str(v51_receipt),
        "--baseline-receipt-role",
        "anchor",
        "--candidate-label",
        candidate_cli_label,
        "--candidate-results",
        str(candidate),
        "--candidate-scoring-receipt",
        str(candidate_receipt),
        "--candidate-candidate-spec",
        str(candidate_spec),
        "--output-json",
        str(output),
    ]
    with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
        main()
    return json.loads(output.read_text(encoding="utf-8"))


class PublicV6SequentialComparisonTests(unittest.TestCase):
    def test_direction_nan_diagnostics_distinguish_mask_drift(self) -> None:
        diagnostics = _direction_nan_diagnostics(
            np.array([1.0, np.nan, 3.0, np.nan]),
            np.array([1.0, 2.0, np.nan, np.nan]),
        )
        self.assertEqual(diagnostics["both_nan_targets"], 1)
        self.assertEqual(diagnostics["baseline_only_nan_targets"], 1)
        self.assertEqual(diagnostics["candidate_only_nan_targets"], 1)
        self.assertEqual(diagnostics["finite_target_intersection"], 1)
        self.assertEqual(diagnostics["finite_target_union"], 3)
        self.assertFalse(diagnostics["finite_target_masks_identical"])

    def test_p4_gamma_label_requires_factor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "require --factor-contract"):
                _run(
                    Path(temporary),
                    candidate_cli_label="p4_g075_p0_a010",
                )

    def test_primary_comparison_reports_metrics_bootstrap_and_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = _run(Path(temporary))

        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["context"], "HepG2")
        self.assertEqual(report["role"], "primary")
        self.assertEqual(report["configuration"]["bootstrap_resamples"], 10_000)
        self.assertEqual(report["configuration"]["bootstrap_seed"], 20_260_901)
        self.assertEqual(
            report["cohort_sizes"],
            {"all": 4, "direct": 2, "held_target": 2},
        )
        self.assertEqual(report["matched_nmae_subset"]["targets"], 2)
        self.assertEqual(
            report["matched_nmae_subset"]["by_cohort"]["direct"],
            {"included_targets": 1, "omitted_targets": 1},
        )

        all_metrics = report["cohorts"]["all"]["metrics"]
        mse = all_metrics[MSE_METRIC]
        self.assertEqual(mse["aggregation"], "ratio_of_sums")
        self.assertAlmostEqual(mse["observed_anchor_ratio_of_sums"], 0.5)
        self.assertAlmostEqual(mse["observed_baseline_ratio_of_sums"], 0.5)
        self.assertAlmostEqual(mse["observed_candidate_ratio_of_sums"], 0.25)
        self.assertAlmostEqual(
            mse["oriented_improvement_positive_is_better"],
            0.25,
        )
        self.assertTrue(mse["one_sided_lower_bound_above_zero"])

        diagnostics = report["cohorts"]["all"]["direction_nan_diagnostics"]
        self.assertEqual(diagnostics[FIDELITY_METRIC]["baseline_nan_targets"], 1)
        self.assertEqual(diagnostics[FIDELITY_METRIC]["candidate_nan_targets"], 1)
        self.assertTrue(
            diagnostics[FIDELITY_METRIC]["finite_target_masks_identical"]
        )
        self.assertEqual(
            report["cohorts"]["all"]["all_six_metric_summary"][
                "point_estimate_strict_improvement_count"
            ],
            6,
        )
        self.assertTrue(report["primary_promotion_gate"]["passed"])
        self.assertFalse(report["leakage_firewall"]["reads_truth_h5ad"])
        self.assertIn("sha256", report["provenance"]["baseline_results"])

    def test_held_target_regression_blocks_primary_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = _run(Path(temporary), held_reach_regression=True)

        gate = report["primary_promotion_gate"]
        self.assertFalse(gate["held_target_all_evaluable_metric_non_regression"])
        self.assertFalse(gate["passed"])
        held = report["cohorts"]["held_target"]
        self.assertLess(
            held["oriented_improvement_positive_is_better"][REACH_METRIC],
            0.0,
        )

    def test_mismatched_nmae_subset_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "NMAE omission subsets differ"):
                _run(
                    Path(temporary),
                    candidate_nmae_targets=("DIRECT_B", "HELD_A"),
                )

    def test_missing_direct_nmae_blocks_primary_guardrail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            baseline = root / "baseline" / "results.csv"
            candidate = root / "candidate" / "results.csv"
            output = root / "comparison.json"
            _write_manifest(manifest)
            _write_bundle(baseline, candidate=False, nmae_targets=("HELD_A",))
            _write_bundle(candidate, candidate=True, nmae_targets=("HELD_A",))
            v51_receipt = write_v51_receipt(
                root / "v51_scoring_receipt.json",
                context="HepG2",
                anchor_results=baseline,
                candidate_results=baseline,
            )
            candidate_receipt, candidate_spec = write_v6_receipt(
                root / "candidate" / "scoring_verified.json",
                context="HepG2",
                label="v6_candidate",
                results=candidate,
                expected_targets=len(TARGETS),
                target_remaining_fraction=0.20,
                residual_alpha=0.10,
                manifest=manifest,
            )
            argv = [
                "compare_public_v6_sequential.py",
                "--manifest",
                str(manifest),
                "--context",
                "HepG2",
                "--role",
                "primary",
                "--baseline-label",
                "anchor",
                "--baseline-results",
                str(baseline),
                "--baseline-scoring-receipt",
                str(v51_receipt),
                "--baseline-receipt-role",
                "anchor",
                "--candidate-label",
                "v6_candidate",
                "--candidate-results",
                str(candidate),
                "--candidate-scoring-receipt",
                str(candidate_receipt),
                "--candidate-candidate-spec",
                str(candidate_spec),
                "--output-json",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                main()
            report = json.loads(output.read_text(encoding="utf-8"))

        direct = report["cohorts"]["direct"]
        self.assertIn(NMAE_METRIC, direct["not_evaluable_metrics"])
        self.assertFalse(
            report["primary_promotion_gate"][
                "all_required_guardrail_metrics_evaluable"
            ]["direct"][NMAE_METRIC]
        )
        self.assertFalse(report["primary_promotion_gate"]["passed"])

    def test_non_primary_context_is_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = _run(
                Path(temporary),
                role="diagnostic",
                context="Jurkat",
            )
        self.assertFalse(report["primary_promotion_gate"]["applicable"])
        self.assertIsNone(report["primary_promotion_gate"]["passed"])
        self.assertFalse(report["experiment_gate"]["applicable"])
        self.assertIsNone(report["experiment_gate"]["passed"])

    def test_jurkat_non_harm_gate_requires_all_three_cohorts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            passing = _run(
                Path(temporary),
                role="non_harm",
                context="Jurkat",
            )
        self.assertTrue(passing["experiment_gate"]["passed"])
        self.assertEqual(
            passing["experiment_gate"]["cohort_non_regression"],
            {"all": True, "direct": True, "held_target": True},
        )
        with tempfile.TemporaryDirectory() as temporary:
            failing = _run(
                Path(temporary),
                role="non_harm",
                context="Jurkat",
                held_reach_regression=True,
            )
        self.assertFalse(failing["experiment_gate"]["passed"])
        self.assertFalse(
            failing["experiment_gate"]["cohort_non_regression"]["held_target"]
        )

    def test_primary_role_is_reserved_for_hepg2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "pre-registered for HepG2"):
                _run(Path(temporary), role="primary", context="Jurkat")

    def test_mislabeled_or_mutated_candidate_receipt_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "label does not match output tag"):
                _run(Path(temporary), candidate_cli_label="wrong_tag")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "stamped (size|SHA-256) mismatch"):
                _run(Path(temporary), tamper_candidate_results=True)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "prediction_view: stamped .* mismatch"):
                _run(Path(temporary), tamper_baseline_prediction_view=True)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "source_prediction: descriptor path mismatch"):
                _run(Path(temporary), mislabel_baseline_source_prediction=True)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "panel_csv: stamped SHA-256 mismatch"):
                _run(Path(temporary), tamper_manifest_routes=True)

    def test_status_and_shared_configuration_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "scoring did not pass"):
                _run(Path(temporary), candidate_status="failed")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "differ in config_digest"):
                _run(Path(temporary), candidate_config_drift=True)


if __name__ == "__main__":
    unittest.main()
