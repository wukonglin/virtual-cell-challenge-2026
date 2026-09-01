"""Tests for fail-closed VCC 2026 scorer-output authentication."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "external" / "cell-eval2" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import validate_public_cell_eval_run as module  # noqa: E402
import public_candidate_v6_contract as contract  # noqa: E402
from summarize_public_validation import (  # noqa: E402
    FIDELITY_METRIC,
    MSE_DENOMINATOR_METRIC,
    MSE_METRIC,
    MSE_NUMERATOR_METRIC,
    NMAE_METRIC,
    REACH_METRIC,
)


class PublicCellEvalRunValidationTests(unittest.TestCase):
    def _candidate_receipts(
        self,
        root: Path,
        *,
        checks: object,
    ) -> tuple[Path, Path, Path, Path, Path]:
        spec = root / "spec.json"
        verification = root / "generation_verified.json"
        views = root / "views.json"
        run_meta = root / "run_meta.json"
        prediction = root / "prediction.h5ad"
        prediction_view = root / "prediction_view.h5ad"
        truth_view = root / "truth_view.h5ad"
        for path, payload in (
            (prediction, b"prediction"),
            (prediction_view, b"prediction-view"),
            (truth_view, b"truth-view"),
        ):
            path.write_bytes(payload)
        source = root / "source.dat"
        source.write_bytes(b"source")
        source_descriptor = module.describe_file(source)
        inputs = {
            name: dict(source_descriptor)
            for name in contract.STRICT_SPEC_INPUT_KEYS
            if name != "state_config"
        }
        inputs["state_config"] = None
        configuration = {"context": "HepG2", "residual_alpha": 0.1}
        prediction_descriptor = module.describe_file(prediction)
        spec.write_text(
            json.dumps(
                {
                    "schema": module.SPEC_SCHEMA,
                    "context": "HepG2",
                    "output_tag": "candidate",
                    "configuration": configuration,
                    "inputs": inputs,
                    "state_source": {
                        "path": str((root / "state_source").resolve()),
                        "repository_commit": "1" * 40,
                        "repository_tree": "2" * 40,
                        "tracked_worktree_clean": True,
                        "untracked_and_ignored_runtime_files_absent": True,
                        "runtime_isolation": {
                            "mode": "authenticated-clean-worktree-no-bytecode-v1",
                            "pythonpath_first": str((root / "state_source/src").resolve()),
                            "python_dont_write_bytecode": True,
                        },
                    },
                    "provenance_contract": {
                        "schema": contract.STRICT_PROVENANCE_CONTRACT,
                        "mode": "strict",
                        "generator_report_provenance_authenticated": True,
                        "optional_state_config_bound": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        verification.write_text(
            json.dumps(
                {
                    "schema": module.VERIFICATION_SCHEMA,
                    "status": "passed",
                    "context": "HepG2",
                    "output_tag": "candidate",
                    "checks": checks,
                    "validation_mode": "strict-v2",
                    "configuration": configuration,
                    "provenance": {
                        "spec": module.describe_file(spec),
                        "prediction_h5ad": prediction_descriptor,
                    },
                }
            ),
            encoding="utf-8",
        )
        views.write_text(
            json.dumps(
                {
                    "schema": module.VIEWS_SCHEMA,
                    "context": "HepG2",
                    "axes": {},
                    "contract": {
                        "model_read_treated_truth": False,
                        "input_type": "raw-nonnegative-integer-counts",
                    },
                    "provenance": {
                        "common_genes": {},
                        "common_genes_json": {},
                        "controls_only": {},
                        "output_prediction": module.describe_file(prediction_view),
                        "output_truth": module.describe_file(truth_view),
                        "prediction": prediction_descriptor,
                        "sealed_truth": {},
                    },
                }
            ),
            encoding="utf-8",
        )
        run_meta.write_text(
            json.dumps({"source": str(truth_view.resolve())}),
            encoding="utf-8",
        )
        return spec, verification, views, run_meta, source

    def _scorer_files(self, root: Path) -> tuple[Path, Path]:
        targets = [f"T{index:03d}" for index in range(300)]
        manifest = root / "manifest.csv"
        pd.DataFrame(
            {
                "target_gene": targets,
                "validation_route": ["direct"] * 267 + ["held_target"] * 33,
            }
        ).to_csv(manifest, index=False)
        metric_values = {
            "pds_cosine": np.linspace(0.1, 0.4, 300),
            MSE_NUMERATOR_METRIC: np.linspace(0.2, 0.5, 300),
            MSE_DENOMINATOR_METRIC: np.linspace(0.8, 1.1, 300),
            FIDELITY_METRIC: np.linspace(0.3, 0.6, 300),
            REACH_METRIC: np.linspace(0.2, 0.7, 300),
            "de_wilcoxon_sig_jaccard": np.linspace(0.05, 0.2, 300),
        }
        metric_values[FIDELITY_METRIC][3] = np.nan
        metric_values[REACH_METRIC][7] = np.nan
        rows: list[dict[str, object]] = []
        for metric, values in metric_values.items():
            rows.extend(
                {"perturbation": target, "metric": metric, "value": value}
                for target, value in zip(targets, values, strict=True)
            )
        nmae_values = np.linspace(0.4, 0.6, 290)
        rows.extend(
            {"perturbation": target, "metric": NMAE_METRIC, "value": value}
            for target, value in zip(targets[:290], nmae_values, strict=True)
        )
        results = root / "results.csv"
        pd.DataFrame(rows).to_csv(results, index=False)

        aggregate_values = {
            "pds_cosine": float(np.mean(metric_values["pds_cosine"])),
            MSE_METRIC: float(
                metric_values[MSE_NUMERATOR_METRIC].sum()
                / metric_values[MSE_DENOMINATOR_METRIC].sum()
            ),
            NMAE_METRIC: float(nmae_values.mean()),
            FIDELITY_METRIC: float(np.nanmean(metric_values[FIDELITY_METRIC])),
            REACH_METRIC: float(np.nanmean(metric_values[REACH_METRIC])),
            "de_wilcoxon_sig_jaccard": float(
                np.mean(metric_values["de_wilcoxon_sig_jaccard"])
            ),
        }
        pd.DataFrame([{"statistic": "mean", **aggregate_values}]).to_csv(
            root / "agg_results.csv", index=False
        )

        aggregation_rows = []
        for metric, values in metric_values.items():
            missing = int(np.count_nonzero(~np.isfinite(values)))
            aggregation_rows.append(
                {
                    "metric": metric,
                    "agg": "mean",
                    "n_used": int(np.count_nonzero(np.isfinite(values))),
                    "n_rows": 300,
                    "n_nan": missing,
                    "n_null": 0,
                    "derived": False,
                }
            )
        aggregation_rows.extend(
            [
                {
                    "metric": NMAE_METRIC,
                    "agg": "mean",
                    "n_used": 290,
                    "n_rows": 290,
                    "n_nan": 0,
                    "n_null": 0,
                    "derived": False,
                },
                {
                    "metric": MSE_METRIC,
                    "agg": "ratio_of_sums",
                    "n_used": 0,
                    "n_rows": 0,
                    "n_nan": 0,
                    "n_null": 0,
                    "derived": True,
                },
            ]
        )
        pd.DataFrame(aggregation_rows).to_csv(
            root / "metric_aggregation.csv", index=False
        )
        return results, manifest

    def test_ratio_of_sums_nan_cohorts_and_nmae_omission_are_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results, manifest = self._scorer_files(Path(directory))
            metrics, sidecars = module.validate_metric_outputs(results, manifest)
            self.assertEqual(metrics[MSE_METRIC]["aggregation"], "ratio_of_sums")
            self.assertEqual(metrics[MSE_METRIC]["finite_targets"], 300)
            self.assertEqual(metrics[NMAE_METRIC]["finite_targets"], 290)
            self.assertEqual(metrics[FIDELITY_METRIC]["finite_targets"], 299)
            self.assertEqual(metrics[REACH_METRIC]["finite_targets"], 299)
            self.assertIn("metric_aggregation", sidecars)

    def test_per_target_derived_mse_and_bad_n_used_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, manifest = self._scorer_files(root)
            frame = pd.read_csv(results)
            frame.loc[len(frame)] = ["T000", MSE_METRIC, 0.5]
            frame.to_csv(results, index=False)
            with self.assertRaisesRegex(RuntimeError, "must not appear"):
                module.validate_metric_outputs(results, manifest)

            results, manifest = self._scorer_files(root)
            aggregation = pd.read_csv(root / "metric_aggregation.csv")
            aggregation.loc[
                aggregation["metric"] == FIDELITY_METRIC, "n_used"
            ] = 300
            aggregation.to_csv(root / "metric_aggregation.csv", index=False)
            with self.assertRaisesRegex(RuntimeError, "n_used mismatch"):
                module.validate_metric_outputs(results, manifest)

    def test_generation_checks_must_be_present_and_nonempty(self) -> None:
        for checks in (None, {}):
            with self.subTest(checks=checks), tempfile.TemporaryDirectory() as directory:
                spec, verification, views, run_meta, manifest = self._candidate_receipts(
                    Path(directory), checks=checks
                )
                with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                    module.validate_candidate_receipts(
                        spec,
                        verification,
                        views,
                        run_meta,
                        manifest,
                        context="HepG2",
                        output_tag="candidate",
                    )

    def test_generation_check_values_must_be_literal_true(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec, verification, views, run_meta, manifest = self._candidate_receipts(
                Path(directory), checks={"structural": 1}
            )
            with self.assertRaisesRegex(RuntimeError, "failed check"):
                module.validate_candidate_receipts(
                    spec,
                    verification,
                    views,
                    run_meta,
                    manifest,
                    context="HepG2",
                    output_tag="candidate",
                )

    def test_views_prediction_and_run_source_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, verification, views, run_meta, manifest = self._candidate_receipts(
                root,
                checks={
                    "structural": True,
                    "strict_generator_provenance_authenticated": True,
                },
            )
            module.validate_candidate_receipts(
                spec,
                verification,
                views,
                run_meta,
                manifest,
                context="HepG2",
                output_tag="candidate",
            )

            payload = json.loads(views.read_text(encoding="utf-8"))
            payload["provenance"]["prediction"]["sha256"] = "0" * 64
            views.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "prediction descriptor differs"):
                module.validate_candidate_receipts(
                    spec,
                    verification,
                    views,
                    run_meta,
                    manifest,
                    context="HepG2",
                    output_tag="candidate",
                )

    def test_spec_descriptor_and_configuration_are_bound(self) -> None:
        checks = {
            "structural": True,
            "strict_generator_provenance_authenticated": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            spec, verification, views, run_meta, manifest = self._candidate_receipts(
                Path(directory), checks=checks
            )
            payload = json.loads(verification.read_text(encoding="utf-8"))
            payload["configuration"]["residual_alpha"] = 0.2
            verification.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "configuration differs"):
                module.validate_candidate_receipts(
                    spec,
                    verification,
                    views,
                    run_meta,
                    manifest,
                    context="HepG2",
                    output_tag="candidate",
                )
        with tempfile.TemporaryDirectory() as directory:
            spec, verification, views, run_meta, manifest = self._candidate_receipts(
                Path(directory), checks=checks
            )
            payload = json.loads(verification.read_text(encoding="utf-8"))
            payload["provenance"]["spec"]["sha256"] = "0" * 64
            verification.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Generation spec descriptor differs"):
                module.validate_candidate_receipts(
                    spec,
                    verification,
                    views,
                    run_meta,
                    manifest,
                    context="HepG2",
                    output_tag="candidate",
                )


if __name__ == "__main__":
    unittest.main()
