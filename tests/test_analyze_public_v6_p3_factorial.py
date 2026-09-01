"""Tests for the leakage-safe public V6 P3 factorial analyzer."""

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
    write_v51_receipt,
    write_v6_receipt,
)


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_public_v6_p3_factorial import (  # noqa: E402
    SCHEMA,
    _four_arm_nan_diagnostics,
    main,
)
from compare_public_v6_sequential import main as sequential_main  # noqa: E402
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
ARM_LEVELS = {
    "A": {"higher": 0.20, "lower": 0.50},
    "B": {"higher": 0.30, "lower": 0.40},
    "C": {"higher": 0.25, "lower": 0.45},
    "D": {"higher": 0.40, "lower": 0.30},
}


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
    arm: str,
    *,
    nmae_targets: tuple[str, ...] = ("DIRECT_A", "HELD_A"),
    direction_nan_target: str = "DIRECT_A",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    levels = ARM_LEVELS[arm]
    for target_index, target in enumerate(TARGETS):
        denominator = float(10 * (target_index + 1))
        for metric in sorted(EMITTED_METRICS):
            if metric == NMAE_METRIC and target not in nmae_targets:
                continue
            if metric == MSE_NUMERATOR_METRIC:
                value = levels["lower"] * denominator
            elif metric == MSE_DENOMINATOR_METRIC:
                value = denominator
            elif metric == NMAE_METRIC:
                value = levels["lower"]
            elif metric in SCORED_METRICS:
                value = levels["higher"]
                if metric in {FIDELITY_METRIC, REACH_METRIC} and target == direction_nan_target:
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
    context: str = "HepG2",
    arm_d_nmae_targets: tuple[str, ...] = ("DIRECT_A", "HELD_A"),
    swap_new_arm_receipts: bool = False,
) -> dict[str, object]:
    manifest = root / "manifest.csv"
    output = root / "factorial.json"
    _write_manifest(manifest)
    paths: dict[str, Path] = {}
    for arm in ("A", "B", "C", "D"):
        path = root / arm / "results.csv"
        _write_bundle(
            path,
            arm,
            nmae_targets=(
                arm_d_nmae_targets
                if arm == "D"
                else ("DIRECT_A", "HELD_A")
            ),
        )
        paths[arm] = path
    v51_receipt = write_v51_receipt(
        root / "v51_scoring_receipt.json",
        context=context,
        anchor_results=paths["A"],
        candidate_results=paths["B"],
    )
    receipts: dict[str, Path] = {"A": v51_receipt, "B": v51_receipt}
    specs: dict[str, Path] = {}
    for arm, alpha in (("C", 0.0), ("D", 0.10)):
        receipt, spec = write_v6_receipt(
            root / arm / "scoring_verified.json",
            context=context,
            label=(
                "p3_tf040_state_a000"
                if arm == "C"
                else "p3_tf040_p0_a010"
            ),
            results=paths[arm],
            expected_targets=len(TARGETS),
            target_remaining_fraction=0.40,
            residual_alpha=alpha,
            manifest=manifest,
        )
        receipts[arm] = receipt
        specs[arm] = spec
    if swap_new_arm_receipts:
        receipts["C"], receipts["D"] = receipts["D"], receipts["C"]
        specs["C"], specs["D"] = specs["D"], specs["C"]
    argv = [
        "analyze_public_v6_p3_factorial.py",
        "--manifest",
        str(manifest),
        "--context",
        context,
        "--factor-contract-receipt",
        str(root / "factor_contract.json"),
    ]
    for arm in ("A", "B", "C", "D"):
        label = (
            "anchor"
            if arm == "A"
            else "v51_alpha010"
            if arm == "B"
            else "p3_tf040_state_a000"
            if arm == "C"
            else "p3_tf040_p0_a010"
        )
        argv.extend(
            [
                f"--arm-{arm.lower()}-label",
                label,
                f"--arm-{arm.lower()}-results",
                str(paths[arm]),
                f"--arm-{arm.lower()}-scoring-receipt",
                str(receipts[arm]),
            ]
        )
        if arm == "A":
            argv.extend(["--arm-a-receipt-role", "anchor"])
        elif arm == "B":
            argv.extend(["--arm-b-receipt-role", "v51_alpha010"])
        else:
            argv.extend(
                [f"--arm-{arm.lower()}-candidate-spec", str(specs[arm])]
            )
    argv.extend(["--output-json", str(output)])
    factor_contract = {
        "schema": "vcc-public-v6-p3-factor-contract-v1",
        "status": "authenticated",
        "receipt": {
            "path": str((root / "factor_contract.json").resolve()),
            "size_bytes": 1,
            "sha256": "f" * 64,
        },
    }
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch(
            "analyze_public_v6_p3_factorial.authenticate_factor_contract_receipt",
            return_value=factor_contract,
        ),
        redirect_stdout(io.StringIO()),
    ):
        main()
    return json.loads(output.read_text(encoding="utf-8"))


class PublicV6P3FactorialTests(unittest.TestCase):
    def test_four_arm_nan_diagnostics_report_pairwise_mask_drift(self) -> None:
        diagnostics = _four_arm_nan_diagnostics(
            {
                "A": np.array([1.0, np.nan, 3.0]),
                "B": np.array([1.0, 2.0, np.nan]),
                "C": np.array([1.0, np.nan, 3.0]),
                "D": np.array([np.nan, 2.0, 3.0]),
            }
        )
        self.assertEqual(diagnostics["four_way_finite_intersection"], 0)
        self.assertEqual(diagnostics["four_way_finite_union"], 3)
        self.assertFalse(diagnostics["all_four_finite_masks_identical"])
        self.assertTrue(
            diagnostics["pairwise"]["A_vs_C"]["finite_target_masks_identical"]
        )
        self.assertFalse(
            diagnostics["pairwise"]["A_vs_B"]["finite_target_masks_identical"]
        )

    def test_factorial_report_uses_common_draws_and_selects_d(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = _run(Path(temporary))

        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["context"], "HepG2")
        self.assertEqual(report["configuration"]["bootstrap_resamples"], 10_000)
        self.assertEqual(report["configuration"]["bootstrap_seed"], 20_260_901)
        self.assertTrue(
            report["configuration"][
                "common_four_arm_draws_within_each_cohort_metric"
            ]
        )
        self.assertEqual(
            report["cohort_sizes"],
            {"all": 4, "direct": 2, "held_target": 2},
        )
        self.assertEqual(report["matched_nmae_subset"]["targets"], 2)

        pds = report["cohorts"]["all"]["metrics"]["pds_cosine"]
        self.assertTrue(
            pds["common_bootstrap"][
                "same_indices_for_all_four_arms_and_all_contrasts"
            ]
        )
        expected = {
            "C_minus_A": 0.05,
            "D_minus_B": 0.10,
            "B_minus_A": 0.10,
            "D_minus_C": 0.15,
            "interaction_D_minus_C_minus_B_minus_A": 0.05,
        }
        for contrast, expected_value in expected.items():
            summary = pds["contrasts"][contrast]
            self.assertTrue(summary["evaluable"])
            self.assertAlmostEqual(
                summary["observed_oriented_contrast_positive_is_better"],
                expected_value,
            )
            self.assertAlmostEqual(
                summary["bootstrap_90pct_two_sided_ci"]["lower"],
                expected_value,
            )
            self.assertAlmostEqual(
                summary["bootstrap_90pct_two_sided_ci"]["upper"],
                expected_value,
            )

        mse = report["cohorts"]["all"]["metrics"][MSE_METRIC]
        self.assertEqual(mse["aggregation"], "ratio_of_sums")
        self.assertFalse(mse["per_target_derived_rows_used"])
        self.assertAlmostEqual(
            mse["contrasts"]["interaction_D_minus_C_minus_B_minus_A"][
                "observed_oriented_contrast_positive_is_better"
            ],
            0.05,
        )
        fidelity = report["cohorts"]["all"]["metrics"][FIDELITY_METRIC]
        self.assertTrue(
            fidelity["finite_mask_diagnostics"][
                "all_four_finite_masks_identical"
            ]
        )
        self.assertGreater(
            fidelity["common_bootstrap"]["attempted_draws"],
            10_000,
        )

        self.assertFalse(
            report["deployment_comparisons"]["C_vs_B"][
                "primary_promotion_gate"
            ]["passed"]
        )
        self.assertTrue(
            report["deployment_comparisons"]["D_vs_B"][
                "primary_promotion_gate"
            ]["passed"]
        )
        selection = report["deterministic_selection"]
        self.assertEqual(selection["selected_arm"], "D")
        self.assertEqual(selection["selected_label"], "p3_tf040_p0_a010")
        self.assertTrue(selection["requires_jurkat_non_harm_before_final_deployment"])
        self.assertFalse(report["leakage_firewall"]["reads_truth_h5ad"])
        self.assertIn("sha256", report["provenance"]["arms"]["A"]["results"])
        self.assertEqual(
            report["provenance"]["factor_contract"]["status"],
            "authenticated",
        )

    def test_nmae_subset_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "Four-arm NMAE omission subsets differ",
            ):
                _run(
                    Path(temporary),
                    arm_d_nmae_targets=("DIRECT_B", "HELD_A"),
                )

    def test_deployment_gate_matches_the_sequential_comparator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = _run(root)
            sequential_output = root / "d_vs_b.json"
            argv = [
                "compare_public_v6_sequential.py",
                "--manifest",
                str(root / "manifest.csv"),
                "--context",
                "HepG2",
                "--role",
                "primary",
                "--baseline-label",
                "v51_alpha010",
                "--baseline-results",
                str(root / "B" / "results.csv"),
                "--baseline-scoring-receipt",
                str(root / "v51_scoring_receipt.json"),
                "--baseline-receipt-role",
                "v51_alpha010",
                "--candidate-label",
                "p3_tf040_p0_a010",
                "--candidate-results",
                str(root / "D" / "results.csv"),
                "--candidate-scoring-receipt",
                str(root / "D" / "scoring_verified.json"),
                "--candidate-candidate-spec",
                str(root / "D" / "candidate_contract" / "spec.json"),
                "--output-json",
                str(sequential_output),
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                sequential_main()
            sequential = json.loads(sequential_output.read_text(encoding="utf-8"))

        factorial_deployment = report["deployment_comparisons"]["D_vs_B"]
        self.assertEqual(
            factorial_deployment["primary_promotion_gate"],
            sequential["primary_promotion_gate"],
        )
        for cohort in ("all", "direct", "held_target"):
            self.assertEqual(
                factorial_deployment["cohorts"][cohort][
                    "oriented_improvement_positive_is_better"
                ],
                sequential["cohorts"][cohort][
                    "oriented_improvement_positive_is_better"
                ],
            )

    def test_non_primary_context_is_rejected_before_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError, "factorial selection is restricted to HepG2"
            ):
                _run(Path(temporary), context="Jurkat")

    def test_swapped_new_arm_receipts_cannot_select(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "label does not match output tag"):
                _run(Path(temporary), swap_new_arm_receipts=True)


if __name__ == "__main__":
    unittest.main()
