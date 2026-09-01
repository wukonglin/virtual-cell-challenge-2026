"""Focused tests for deterministic P4 STATE-gamma selection receipts."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import select_public_v6_p4_state_gamma as module  # noqa: E402


class PublicV6P4StateGammaSelectionTests(unittest.TestCase):
    def _factor_receipt(self) -> dict[str, object]:
        return {
            "schema": module.FACTOR_SCHEMA,
            "status": "passed",
            "factor": {
                "context": module.CONTEXT,
                "factor_name": module.FACTOR_NAME,
                "levels_by_arm": {
                    arm: definition[module.FACTOR_NAME]
                    for arm, definition in module.ARMS.items()
                },
                "sole_configuration_difference": module.FACTOR_NAME,
                "shared_spec_identity_sha256": "a" * 64,
            },
            "realized_generation": {
                "shared_generation_identity_sha256": "b" * 64,
            },
            "arms": {
                arm: {
                    "output_tag": definition["label"],
                    module.FACTOR_NAME: definition[module.FACTOR_NAME],
                }
                for arm, definition in module.ARMS.items()
            },
        }

    @staticmethod
    def _cohorts(passed: bool) -> dict[str, object]:
        required_metrics = {
            metric: {"evaluable": True}
            for metric in module.PRIMARY_REQUIRED_GUARDRAILS
        }
        return {
            "all": {
                "metrics": copy.deepcopy(required_metrics),
                "point_estimate_promotion_gate": {"passed": passed},
                "point_estimate_non_regression_gate": {"passed": passed},
            },
            "direct": {
                "metrics": copy.deepcopy(required_metrics),
                "point_estimate_promotion_gate": {"passed": passed},
                "point_estimate_non_regression_gate": {"passed": passed},
            },
            "held_target": {
                "metrics": {},
                "point_estimate_promotion_gate": {"passed": passed},
                "point_estimate_non_regression_gate": {"passed": passed},
            },
        }

    def _write_comparison(
        self,
        *,
        root: Path,
        key: str,
        passed: bool,
        factor_receipt: dict[str, object],
        factor_descriptor: dict[str, object],
        manifest_descriptor: dict[str, object],
    ) -> Path:
        definition = module.COMPARISONS[key]
        baseline_arm = definition["baseline_arm"]
        candidate_arm = definition["candidate_arm"]
        cohorts = self._cohorts(passed)
        primary_gate = module._primary_promotion_gate(cohorts)

        def scoring_artifacts(arm: str) -> tuple[dict[str, object], dict[str, object]]:
            label = module.ARMS[arm]["label"]
            results_path = root / f"{arm}_results.csv"
            receipt_path = root / f"{arm}_scoring_receipt.json"
            aggregate_path = root / f"{arm}_agg_results.csv"
            aggregation_path = root / f"{arm}_metric_aggregation.csv"
            if not results_path.exists():
                results_path.write_text(f"arm,value\n{arm},1\n", encoding="utf-8")
            if not receipt_path.exists():
                receipt_path.write_text(
                    json.dumps({"arm": arm}, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            if not aggregate_path.exists():
                aggregate_path.write_text("statistic,value\nmean,1\n", encoding="utf-8")
            if not aggregation_path.exists():
                aggregation_path.write_text("metric,agg\ntest,mean\n", encoding="utf-8")
            bundle = {
                "kind": "v6",
                "context": module.CONTEXT,
                "label": label,
                "receipt": module._immutable_descriptor(receipt_path),
                "candidate_spec": {"sha256": arm * 16},
                "candidate_configuration": {
                    module.FACTOR_NAME: module.ARMS[arm][module.FACTOR_NAME]
                },
                "outputs": {"results": module._immutable_descriptor(results_path)},
                "software": {"runtime": "test"},
            }
            sidecars = {
                "aggregate_results": module._immutable_descriptor(aggregate_path),
                "metric_aggregation": module._immutable_descriptor(aggregation_path),
                "validated_all_context_means": {"test": float(len(arm))},
                "expected_targets": 300,
            }
            return bundle, sidecars

        baseline_bundle, baseline_sidecars = scoring_artifacts(baseline_arm)
        candidate_bundle, candidate_sidecars = scoring_artifacts(candidate_arm)
        baseline_results = root / f"{baseline_arm}_results.csv"
        candidate_results = root / f"{candidate_arm}_results.csv"
        report = {
            "schema": module.COMPARISON_SCHEMA,
            "context": module.CONTEXT,
            "role": module.ROLE,
            "baseline_label": module.ARMS[definition["baseline_arm"]]["label"],
            "candidate_label": module.ARMS[definition["candidate_arm"]]["label"],
            "configuration": {
                "absolute_tolerance": 1e-12,
                "bootstrap_resamples": 10_000,
                "bootstrap_seed": 20_260_901,
            },
            "cohort_sizes": dict(module.EXPECTED_COHORT_SIZES),
            "matched_nmae_subset": {
                "matched": True,
                "targets": 299,
                "omitted_manifest_targets": 1,
                "target_axis_sha256": "c" * 64,
                "by_cohort": {
                    "all": {"included_targets": 299, "omitted_targets": 1},
                    "direct": {"included_targets": 266, "omitted_targets": 1},
                    "held_target": {"included_targets": 33, "omitted_targets": 0},
                },
            },
            "cohorts": cohorts,
            "primary_promotion_gate": primary_gate,
            "experiment_gate": {"role": module.ROLE, **primary_gate},
            "uncertainty_contract": {
                "point_gate_is_not_overridden_by_bootstrap": True,
                "metric_level_intervals_reported": True,
                "all_six_improvement_counts_reported_per_cohort": True,
                "one_sided_lower_bound_above_zero_is_strong_improvement_evidence": True,
            },
            "provenance": {
                "manifest": manifest_descriptor,
                "baseline_results": module._immutable_descriptor(baseline_results),
                "candidate_results": module._immutable_descriptor(candidate_results),
                "authenticated_scoring_bundles": {
                    "baseline": baseline_bundle,
                    "candidate": candidate_bundle,
                },
                "shared_scoring_identity": {
                    "config_digest": "d" * 64,
                    "source_fingerprint": "e" * 64,
                    "anchor_semantic_identity": "f" * 64,
                    "cell_eval2_git_commit": "1" * 40,
                    "cell_eval2_declared_version": "test",
                    "cell_eval2_runtime_version": "test",
                    "pdex_version": "test",
                },
                "baseline_cell_eval2_sidecars": baseline_sidecars,
                "candidate_cell_eval2_sidecars": candidate_sidecars,
                "factor_contract": module._expected_factor_binding(
                    factor_receipt,
                    factor_descriptor,
                    definition["baseline_arm"],
                    definition["candidate_arm"],
                ),
            },
            "leakage_firewall": {
                "reads_truth_h5ad": False,
                "loads_scoring_view_expression_matrices": False,
                "streams_generated_prediction_counts_for_factor_authentication": True,
                "materializes_full_prediction_expression_matrices": False,
                "writes_or_modifies_input_artifacts": False,
                "authenticates_factor_contract": True,
            },
        }
        path = root / f"{key}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _fixture(
        self,
        root: Path,
        *,
        g075_passed: bool,
        g050_passed: bool,
        incremental_passed: bool | None = None,
    ) -> tuple[Path, dict[str, object], dict[str, Path]]:
        factor_path = root / "factor.json"
        factor_path.write_text("{}\n", encoding="utf-8")
        manifest_path = root / "manifest.csv"
        manifest_path.write_text("target_gene,validation_route\n", encoding="utf-8")
        factor_descriptor = module._immutable_descriptor(factor_path)
        manifest_descriptor = module._immutable_descriptor(manifest_path)
        factor_receipt = self._factor_receipt()
        paths = {
            "g075_vs_g100": self._write_comparison(
                root=root,
                key="g075_vs_g100",
                passed=g075_passed,
                factor_receipt=factor_receipt,
                factor_descriptor=factor_descriptor,
                manifest_descriptor=manifest_descriptor,
            ),
            "g050_vs_g100": self._write_comparison(
                root=root,
                key="g050_vs_g100",
                passed=g050_passed,
                factor_receipt=factor_receipt,
                factor_descriptor=factor_descriptor,
                manifest_descriptor=manifest_descriptor,
            ),
        }
        if incremental_passed is not None:
            paths["g050_vs_g075"] = self._write_comparison(
                root=root,
                key="g050_vs_g075",
                passed=incremental_passed,
                factor_receipt=factor_receipt,
                factor_descriptor=factor_descriptor,
                manifest_descriptor=manifest_descriptor,
            )
        return factor_path, factor_receipt, paths

    def _build(
        self,
        factor_path: Path,
        factor_receipt: dict[str, object],
        paths: dict[str, Path],
    ) -> dict[str, object]:
        with mock.patch.object(
            module,
            "validate_factor_receipt",
            return_value=factor_receipt,
        ):
            return module.build_selection_payload(
                factor_contract=factor_path,
                g075_vs_g100=paths["g075_vs_g100"],
                g050_vs_g100=paths["g050_vs_g100"],
                g050_vs_g075=paths.get("g050_vs_g075"),
            )

    def test_neither_primary_arm_passes_selects_g100(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(
                Path(directory), g075_passed=False, g050_passed=False
            )
            receipt = self._build(*fixture)
        self.assertEqual(receipt["decision"]["selected_arm"], "g100")
        self.assertEqual(receipt["decision"][module.FACTOR_NAME], 1.0)
        self.assertIsNone(receipt["comparisons"]["g050_vs_g075"])

    def test_exactly_one_primary_arm_passes_selects_that_arm(self) -> None:
        for g075_passed, g050_passed, expected in (
            (True, False, "g075"),
            (False, True, "g050"),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                fixture = self._fixture(
                    Path(directory),
                    g075_passed=g075_passed,
                    g050_passed=g050_passed,
                )
                receipt = self._build(*fixture)
                self.assertEqual(receipt["decision"]["selected_arm"], expected)

    def test_both_pass_requires_incremental_and_uses_its_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factor_path, factor_receipt, paths = self._fixture(
                root,
                g075_passed=True,
                g050_passed=True,
            )
            with mock.patch.object(
                module, "validate_factor_receipt", return_value=factor_receipt
            ), self.assertRaisesRegex(RuntimeError, "g050-vs-g075 is required"):
                module.build_selection_payload(
                    factor_contract=factor_path,
                    g075_vs_g100=paths["g075_vs_g100"],
                    g050_vs_g100=paths["g050_vs_g100"],
                )

        for incremental_passed, expected in ((True, "g050"), (False, "g075")):
            with self.subTest(incremental_passed=incremental_passed), tempfile.TemporaryDirectory() as directory:
                fixture = self._fixture(
                    Path(directory),
                    g075_passed=True,
                    g050_passed=True,
                    incremental_passed=incremental_passed,
                )
                receipt = self._build(*fixture)
                self.assertEqual(receipt["decision"]["selected_arm"], expected)
                self.assertEqual(
                    receipt["decision"]["incremental_experiment_gate_passed"],
                    incremental_passed,
                )

    def test_factor_binding_and_exact_labels_are_hard_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            factor_path, factor_receipt, paths = self._fixture(
                Path(directory), g075_passed=False, g050_passed=False
            )
            report = json.loads(paths["g075_vs_g100"].read_text(encoding="utf-8"))
            report["baseline_label"] = "renamed"
            paths["g075_vs_g100"].write_text(json.dumps(report), encoding="utf-8")
            with mock.patch.object(
                module, "validate_factor_receipt", return_value=factor_receipt
            ), self.assertRaisesRegex(RuntimeError, "baseline label differs"):
                module.build_selection_payload(
                    factor_contract=factor_path,
                    g075_vs_g100=paths["g075_vs_g100"],
                    g050_vs_g100=paths["g050_vs_g100"],
                )

        with tempfile.TemporaryDirectory() as directory:
            factor_path, factor_receipt, paths = self._fixture(
                Path(directory), g075_passed=False, g050_passed=False
            )
            report = json.loads(paths["g050_vs_g100"].read_text(encoding="utf-8"))
            report["provenance"]["factor_contract"]["receipt"]["sha256"] = "0" * 64
            paths["g050_vs_g100"].write_text(json.dumps(report), encoding="utf-8")
            with mock.patch.object(
                module, "validate_factor_receipt", return_value=factor_receipt
            ), self.assertRaisesRegex(RuntimeError, "factor-contract binding differs"):
                module.build_selection_payload(
                    factor_contract=factor_path,
                    g075_vs_g100=paths["g075_vs_g100"],
                    g050_vs_g100=paths["g050_vs_g100"],
                )

    def test_gate_is_reproduced_instead_of_trusting_passed_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            factor_path, factor_receipt, paths = self._fixture(
                Path(directory), g075_passed=False, g050_passed=False
            )
            report = json.loads(paths["g075_vs_g100"].read_text(encoding="utf-8"))
            report["experiment_gate"]["passed"] = True
            paths["g075_vs_g100"].write_text(json.dumps(report), encoding="utf-8")
            with mock.patch.object(
                module, "validate_factor_receipt", return_value=factor_receipt
            ), self.assertRaisesRegex(RuntimeError, "experiment gate does not reproduce"):
                module.build_selection_payload(
                    factor_contract=factor_path,
                    g075_vs_g100=paths["g075_vs_g100"],
                    g050_vs_g100=paths["g050_vs_g100"],
                )

    def test_all_comparisons_must_share_evaluation_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            factor_path, factor_receipt, paths = self._fixture(
                Path(directory), g075_passed=False, g050_passed=False
            )
            report = json.loads(paths["g050_vs_g100"].read_text(encoding="utf-8"))
            report["configuration"]["absolute_tolerance"] = 0.0
            paths["g050_vs_g100"].write_text(json.dumps(report), encoding="utf-8")
            with mock.patch.object(
                module, "validate_factor_receipt", return_value=factor_receipt
            ), self.assertRaisesRegex(RuntimeError, "one evaluation identity"):
                module.build_selection_payload(
                    factor_contract=factor_path,
                    g075_vs_g100=paths["g075_vs_g100"],
                    g050_vs_g100=paths["g050_vs_g100"],
                )

    def test_repeated_arm_must_reuse_exact_scored_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            factor_path, factor_receipt, paths = self._fixture(
                Path(directory), g075_passed=False, g050_passed=False
            )
            report = json.loads(paths["g050_vs_g100"].read_text(encoding="utf-8"))
            report["provenance"]["authenticated_scoring_bundles"]["baseline"][
                "candidate_configuration"
            ]["audit_marker"] = "different"
            paths["g050_vs_g100"].write_text(json.dumps(report), encoding="utf-8")
            with mock.patch.object(
                module, "validate_factor_receipt", return_value=factor_receipt
            ), self.assertRaisesRegex(RuntimeError, "different scored artifacts"):
                module.build_selection_payload(
                    factor_contract=factor_path,
                    g075_vs_g100=paths["g075_vs_g100"],
                    g050_vs_g100=paths["g050_vs_g100"],
                )

    def test_sidecar_bytes_are_reauthenticated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factor_path, factor_receipt, paths = self._fixture(
                root, g075_passed=False, g050_passed=False
            )
            (root / "g100_agg_results.csv").write_text(
                "statistic,value\nmean,2\n", encoding="utf-8"
            )
            with mock.patch.object(
                module, "validate_factor_receipt", return_value=factor_receipt
            ), self.assertRaisesRegex(RuntimeError, "aggregate results SHA-256 differs"):
                module.build_selection_payload(
                    factor_contract=factor_path,
                    g075_vs_g100=paths["g075_vs_g100"],
                    g050_vs_g100=paths["g050_vs_g100"],
                )

    def test_validate_rehashes_and_exactly_reproduces_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factor_path, factor_receipt, paths = self._fixture(
                root, g075_passed=False, g050_passed=True
            )
            receipt = self._build(factor_path, factor_receipt, paths)
            receipt_path = root / "selection.json"
            module._atomic_json(receipt_path, receipt)
            with mock.patch.object(
                module, "validate_factor_receipt", return_value=factor_receipt
            ):
                self.assertEqual(
                    module.validate_selection_receipt(receipt_path), receipt
                )

            paths["g050_vs_g100"].write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "size differs"):
                module.validate_selection_receipt(receipt_path)


if __name__ == "__main__":
    unittest.main()
