#!/usr/bin/env python3
"""Compare one V6 candidate with its fixed baseline in one public context.

The comparator is intentionally leakage-safe: it authenticates the sealed
route manifest, scorer receipts, candidate contracts, result tables, and
adjacent sidecars. It hashes immutable scoring-view H5AD bytes but never loads
their expression matrices. The candidate and baseline are compared on paired
target axes with a fixed 10,000-draw bootstrap, while official expression MSE
is always reconstructed as a ratio of component sums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from authenticate_public_scoring_bundle import (
    authenticate_scoring_bundle,
    require_shared_scoring_identity,
)
from analyze_public_paired_bootstrap import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DERIVED_MSE_METRIC,
    METRIC_ORIENTATION,
    MSE_DENOMINATOR_METRIC as BOOTSTRAP_MSE_DENOMINATOR_METRIC,
    MSE_NUMERATOR_METRIC as BOOTSTRAP_MSE_NUMERATOR_METRIC,
    NMAE_METRIC as BOOTSTRAP_NMAE_METRIC,
    _aligned_metric_values,
    _atomic_write_json,
    _describe_file,
    _read_manifest,
    _read_scored_results,
    bootstrap_mean_difference_summary,
    bootstrap_mse_ratio_summary,
    require,
)
from summarize_public_validation import (
    COHORT_OPTIONAL_METRICS,
    FIDELITY_METRIC,
    MSE_DENOMINATOR_METRIC,
    MSE_METRIC,
    MSE_NUMERATOR_METRIC,
    NMAE_METRIC,
    REACH_METRIC,
    SCORED_METRICS,
    _mean_summary,
    _read_results,
    _validate_cell_eval_sidecars,
    non_regression_gate,
    oriented_improvement,
    promotion_gate,
)


SCHEMA = "vcc-public-v6-sequential-comparison-v1"
COHORTS = ("all", "direct", "held_target")
DIRECTION_METRICS = (FIDELITY_METRIC, REACH_METRIC)
PRIMARY_CONTEXT = "HepG2"
NON_HARM_CONTEXT = "Jurkat"
PRIMARY_REQUIRED_GUARDRAILS = (MSE_METRIC, NMAE_METRIC, REACH_METRIC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--context", type=str, required=True)
    parser.add_argument(
        "--role",
        choices=("primary", "diagnostic", "non_harm"),
        required=True,
    )
    parser.add_argument("--baseline-label", type=str, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--baseline-scoring-receipt", type=Path, required=True)
    parser.add_argument(
        "--baseline-receipt-role",
        choices=("anchor", "v51_alpha010"),
    )
    parser.add_argument("--baseline-candidate-spec", type=Path)
    parser.add_argument(
        "--baseline-allow-legacy-v1",
        action="store_true",
        help="Permit an immutable pre-v2 V6 baseline spec for read-only analysis.",
    )
    parser.add_argument("--candidate-label", type=str, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--candidate-scoring-receipt", type=Path, required=True)
    parser.add_argument(
        "--candidate-receipt-role",
        choices=("anchor", "v51_alpha010"),
    )
    parser.add_argument("--candidate-candidate-spec", type=Path)
    parser.add_argument(
        "--candidate-allow-legacy-v1",
        action="store_true",
        help="Permit an immutable pre-v2 V6 candidate spec for read-only analysis.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    return parser.parse_args()


def _validate_shared_metric_contract() -> None:
    """Fail closed if the imported summary and bootstrap contracts diverge."""

    require(SCORED_METRICS == METRIC_ORIENTATION, "Metric orientations diverged")
    require(MSE_METRIC == DERIVED_MSE_METRIC, "Derived MSE names diverged")
    require(
        MSE_NUMERATOR_METRIC == BOOTSTRAP_MSE_NUMERATOR_METRIC,
        "MSE numerator names diverged",
    )
    require(
        MSE_DENOMINATOR_METRIC == BOOTSTRAP_MSE_DENOMINATOR_METRIC,
        "MSE denominator names diverged",
    )
    require(NMAE_METRIC == BOOTSTRAP_NMAE_METRIC, "NMAE names diverged")


def _authenticate_results_bundle(
    path: Path,
    targets: frozenset[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Authenticate result axes and the two adjacent cell-eval2 sidecars."""

    summary_frame = _read_results(path, set(targets))
    all_summary = _mean_summary(summary_frame, len(targets))
    sidecars = _validate_cell_eval_sidecars(
        path,
        summary_frame,
        all_summary,
        len(targets),
    )
    bootstrap_frame = _read_scored_results(path, targets)
    return summary_frame, bootstrap_frame, sidecars


def _target_axis_sha256(targets: frozenset[str]) -> str:
    payload = "\n".join(sorted(targets)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _direction_nan_diagnostics(
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    baseline_finite = np.isfinite(baseline)
    candidate_finite = np.isfinite(candidate)
    return {
        "target_axis_size": int(baseline.size),
        "baseline_nan_targets": int((~baseline_finite).sum()),
        "candidate_nan_targets": int((~candidate_finite).sum()),
        "both_nan_targets": int((~baseline_finite & ~candidate_finite).sum()),
        "baseline_only_nan_targets": int((~baseline_finite & candidate_finite).sum()),
        "candidate_only_nan_targets": int((baseline_finite & ~candidate_finite).sum()),
        "finite_target_intersection": int((baseline_finite & candidate_finite).sum()),
        "finite_target_union": int((baseline_finite | candidate_finite).sum()),
        "finite_target_masks_identical": bool(
            np.array_equal(baseline_finite, candidate_finite)
        ),
    }


def _not_evaluable_bootstrap(
    *,
    direction: str,
    cohort_targets: int,
    metric_targets: int,
    baseline_values: np.ndarray | None = None,
    candidate_values: np.ndarray | None = None,
) -> dict[str, Any]:
    baseline_finite = (
        int(np.isfinite(baseline_values).sum())
        if baseline_values is not None
        else 0
    )
    candidate_finite = (
        int(np.isfinite(candidate_values).sum())
        if candidate_values is not None
        else 0
    )
    return {
        "evaluable": False,
        "direction": direction,
        "orientation_transform": (
            "candidate_minus_baseline"
            if direction == "higher"
            else "negative_candidate_minus_baseline"
        ),
        "resampling_axis_targets": int(metric_targets),
        "omitted_cohort_targets": int(cohort_targets - metric_targets),
        "baseline_finite_targets": baseline_finite,
        "candidate_finite_targets": candidate_finite,
        "oriented_improvement_positive_is_better": None,
        "bootstrap_90pct_two_sided_ci": None,
        "bootstrap_90pct_one_sided_lower_bound": None,
        "one_sided_lower_bound_above_zero": None,
        "reason": "no_matched_finite_metric_population",
    }


def _metric_summary(
    *,
    metric: str,
    direction: str,
    cohort_targets: frozenset[str],
    metric_targets: frozenset[str],
    baseline_scored: pd.DataFrame,
    candidate_scored: pd.DataFrame,
    baseline_mean: dict[str, Any],
    candidate_mean: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build one observed comparison and its paired-target uncertainty."""

    if metric == MSE_METRIC:
        baseline_num, candidate_num = _aligned_metric_values(
            baseline_scored,
            candidate_scored,
            MSE_NUMERATOR_METRIC,
            cohort_targets,
        )
        baseline_den, candidate_den = _aligned_metric_values(
            baseline_scored,
            candidate_scored,
            MSE_DENOMINATOR_METRIC,
            cohort_targets,
        )
        summary = bootstrap_mse_ratio_summary(
            baseline_num,
            baseline_den,
            candidate_num,
            candidate_den,
            rng,
        )
        improvement = float(summary["observed_oriented_candidate_minus_anchor"])
        summary.update(
            {
                "evaluable": True,
                "direction": direction,
                "orientation_transform": "negative_candidate_minus_baseline",
                "observed_baseline_ratio_of_sums": float(
                    summary["observed_anchor_ratio_of_sums"]
                ),
                "observed_raw_candidate_minus_baseline": float(
                    summary["observed_raw_candidate_minus_anchor"]
                ),
                "oriented_improvement_positive_is_better": improvement,
                "omitted_cohort_targets": 0,
            }
        )
        return summary, None

    if not metric_targets:
        return (
            _not_evaluable_bootstrap(
                direction=direction,
                cohort_targets=len(cohort_targets),
                metric_targets=0,
            ),
            None,
        )

    baseline_values, candidate_values = _aligned_metric_values(
        baseline_scored,
        candidate_scored,
        metric,
        metric_targets,
    )
    nan_diagnostics = (
        _direction_nan_diagnostics(baseline_values, candidate_values)
        if metric in DIRECTION_METRICS
        else None
    )
    if not np.isfinite(baseline_values).any() or not np.isfinite(candidate_values).any():
        return (
            _not_evaluable_bootstrap(
                direction=direction,
                cohort_targets=len(cohort_targets),
                metric_targets=len(metric_targets),
                baseline_values=baseline_values,
                candidate_values=candidate_values,
            ),
            nan_diagnostics,
        )

    summary = bootstrap_mean_difference_summary(
        baseline_values,
        candidate_values,
        direction,
        rng,
    )
    improvement = oriented_improvement(
        metric,
        float(candidate_mean[metric]["mean"]),
        float(baseline_mean[metric]["mean"]),
    )
    require(
        bool(
            np.isclose(
                improvement,
                float(summary["observed_mean_oriented_candidate_minus_anchor"]),
                rtol=1e-12,
                atol=1e-12,
            )
        ),
        f"Observed and bootstrapped improvements differ: {metric}",
    )
    summary.update(
        {
            "evaluable": True,
            "direction": direction,
            "orientation_transform": (
                "candidate_minus_baseline"
                if direction == "higher"
                else "negative_candidate_minus_baseline"
            ),
            "observed_baseline_mean": float(summary["observed_anchor_mean"]),
            "observed_mean_raw_candidate_minus_baseline": float(
                summary["observed_mean_raw_candidate_minus_anchor"]
            ),
            "oriented_improvement_positive_is_better": float(improvement),
            "omitted_cohort_targets": int(len(cohort_targets) - len(metric_targets)),
        }
    )
    return summary, nan_diagnostics


def _six_metric_summary(
    metrics: dict[str, dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    evaluable = [metric for metric in SCORED_METRICS if metrics[metric]["evaluable"]]
    not_evaluable = [
        metric for metric in SCORED_METRICS if not metrics[metric]["evaluable"]
    ]
    strict = [
        metric
        for metric in evaluable
        if float(metrics[metric]["oriented_improvement_positive_is_better"])
        > tolerance
    ]
    non_regressed = [
        metric
        for metric in evaluable
        if float(metrics[metric]["oriented_improvement_positive_is_better"])
        >= -tolerance
    ]
    supported = [
        metric
        for metric in evaluable
        if metrics[metric]["one_sided_lower_bound_above_zero"] is True
    ]
    unfavorable = [
        metric
        for metric in evaluable
        if float(metrics[metric]["bootstrap_90pct_two_sided_ci"]["upper"])
        < -tolerance
    ]
    uncertain = [
        metric
        for metric in evaluable
        if float(metrics[metric]["bootstrap_90pct_two_sided_ci"]["lower"])
        <= tolerance
        and float(metrics[metric]["bootstrap_90pct_two_sided_ci"]["upper"])
        >= -tolerance
    ]
    return {
        "total_metrics": len(SCORED_METRICS),
        "evaluable_metrics": len(evaluable),
        "not_evaluable_metrics": not_evaluable,
        "point_estimate_strict_improvement_count": len(strict),
        "point_estimate_strict_improvement_metrics": strict,
        "point_estimate_non_regression_count": len(non_regressed),
        "point_estimate_non_regression_metrics": non_regressed,
        "bootstrap_one_sided_90pct_supported_improvement_count": len(supported),
        "bootstrap_one_sided_90pct_supported_improvement_metrics": supported,
        "bootstrap_two_sided_90pct_strict_regression_count": len(unfavorable),
        "bootstrap_two_sided_90pct_strict_regression_metrics": unfavorable,
        "bootstrap_two_sided_90pct_overlaps_zero_count": len(uncertain),
        "bootstrap_two_sided_90pct_overlaps_zero_metrics": uncertain,
    }


def _cohort_comparison(
    *,
    cohort_targets: frozenset[str],
    baseline_summary_frame: pd.DataFrame,
    candidate_summary_frame: pd.DataFrame,
    baseline_scored: pd.DataFrame,
    candidate_scored: pd.DataFrame,
    matched_nmae_targets: frozenset[str],
    rng: np.random.Generator,
    tolerance: float,
) -> dict[str, Any]:
    baseline_cohort = baseline_summary_frame.loc[
        baseline_summary_frame["perturbation"].isin(cohort_targets)
    ]
    candidate_cohort = candidate_summary_frame.loc[
        candidate_summary_frame["perturbation"].isin(cohort_targets)
    ]
    baseline_mean = _mean_summary(baseline_cohort, len(cohort_targets))
    candidate_mean = _mean_summary(candidate_cohort, len(cohort_targets))

    metrics: dict[str, dict[str, Any]] = {}
    direction_diagnostics: dict[str, Any] = {}
    improvements: dict[str, float] = {}
    not_evaluable: list[str] = []
    for metric, direction in SCORED_METRICS.items():
        baseline_value = baseline_mean[metric]["mean"]
        candidate_value = candidate_mean[metric]["mean"]
        if (baseline_value is None) != (candidate_value is None):
            require(
                metric in DIRECTION_METRICS,
                f"Metric evaluability differs outside a direction metric: {metric}",
            )
        metric_targets = (
            cohort_targets & matched_nmae_targets
            if metric == NMAE_METRIC
            else cohort_targets
        )
        summary, diagnostics = _metric_summary(
            metric=metric,
            direction=direction,
            cohort_targets=cohort_targets,
            metric_targets=metric_targets,
            baseline_scored=baseline_scored,
            candidate_scored=candidate_scored,
            baseline_mean=baseline_mean,
            candidate_mean=candidate_mean,
            rng=rng,
        )
        summary["baseline_aggregate"] = baseline_mean[metric]
        summary["candidate_aggregate"] = candidate_mean[metric]
        metrics[metric] = summary
        if diagnostics is not None:
            direction_diagnostics[metric] = diagnostics
        if summary["evaluable"]:
            improvements[metric] = float(
                summary["oriented_improvement_positive_is_better"]
            )
        else:
            require(metric in COHORT_OPTIONAL_METRICS, "Required metric is not evaluable")
            not_evaluable.append(metric)

    not_evaluable_tuple = tuple(not_evaluable)
    point_promotion = promotion_gate(
        improvements,
        tolerance,
        not_evaluable_tuple,
    )
    point_non_regression = non_regression_gate(
        improvements,
        tolerance,
        not_evaluable_tuple,
    )
    return {
        "cohort_targets": int(len(cohort_targets)),
        "metrics": metrics,
        "oriented_improvement_positive_is_better": improvements,
        "direction_nan_diagnostics": direction_diagnostics,
        "not_evaluable_metrics": not_evaluable,
        "point_estimate_promotion_gate": point_promotion,
        "point_estimate_non_regression_gate": point_non_regression,
        "all_six_metric_summary": _six_metric_summary(metrics, tolerance),
    }


def _primary_promotion_gate(
    cohorts: dict[str, Any],
) -> dict[str, Any]:
    all_gate = cohorts["all"]["point_estimate_promotion_gate"]
    direct_gate = cohorts["direct"]["point_estimate_promotion_gate"]
    held_gate = cohorts["held_target"]["point_estimate_non_regression_gate"]
    required_evaluability = {
        cohort: {
            metric: bool(cohorts[cohort]["metrics"][metric]["evaluable"])
            for metric in PRIMARY_REQUIRED_GUARDRAILS
        }
        for cohort in ("all", "direct")
    }
    guardrails_evaluable = all(
        value
        for by_metric in required_evaluability.values()
        for value in by_metric.values()
    )
    passed = bool(
        guardrails_evaluable
        and all_gate["passed"]
        and direct_gate["passed"]
        and held_gate["passed"]
    )
    return {
        "applicable": True,
        "context_required": PRIMARY_CONTEXT,
        "point_estimate_only": True,
        "all_required_guardrail_metrics_evaluable": required_evaluability,
        "all_cohort_mse_nmae_reach_non_regression_and_discrimination_improvement": bool(
            all_gate["passed"]
        ),
        "direct_cohort_mse_nmae_reach_non_regression_and_discrimination_improvement": bool(
            direct_gate["passed"]
        ),
        "held_target_all_evaluable_metric_non_regression": bool(held_gate["passed"]),
        "passed": passed,
    }


def _experiment_gate(
    *,
    role: str,
    context: str,
    cohorts: dict[str, Any],
) -> dict[str, Any]:
    """Apply the pre-registered role-specific deterministic decision gate."""

    if role == "primary":
        return {"role": role, **_primary_promotion_gate(cohorts)}
    if role == "non_harm":
        require(
            context == NON_HARM_CONTEXT,
            f"The non_harm role is pre-registered for {NON_HARM_CONTEXT} only",
        )
        cohort_pass = {
            cohort: bool(cohorts[cohort]["point_estimate_non_regression_gate"]["passed"])
            for cohort in COHORTS
        }
        return {
            "role": role,
            "applicable": True,
            "context_required": NON_HARM_CONTEXT,
            "point_estimate_only": True,
            "strict_discrimination_improvement_required": False,
            "cohort_non_regression": cohort_pass,
            "passed": bool(all(cohort_pass.values())),
        }
    return {
        "role": role,
        "applicable": False,
        "reason": "Diagnostic comparisons cannot promote or reject a candidate",
        "passed": None,
    }


def main() -> None:
    args = parse_args()
    _validate_shared_metric_contract()
    require(not args.output_json.exists(), f"Refusing to overwrite: {args.output_json}")
    require(args.absolute_tolerance >= 0.0, "absolute-tolerance cannot be negative")
    context = args.context.strip()
    baseline_label = args.baseline_label.strip()
    candidate_label = args.candidate_label.strip()
    require(bool(context), "Context cannot be empty")
    require(bool(baseline_label), "Baseline label cannot be empty")
    require(bool(candidate_label), "Candidate label cannot be empty")
    require(baseline_label != candidate_label, "Baseline and candidate labels must differ")
    require(
        args.baseline_results.resolve() != args.candidate_results.resolve(),
        "Baseline and candidate result paths must differ",
    )
    if args.role == "primary":
        require(
            context == PRIMARY_CONTEXT,
            f"The primary role is pre-registered for {PRIMARY_CONTEXT} only",
        )

    manifest = _read_manifest(args.manifest)
    manifest_targets = frozenset(manifest["target_gene"])
    authenticated_bundles = {
        "baseline": authenticate_scoring_bundle(
            receipt_path=args.baseline_scoring_receipt,
            results_path=args.baseline_results,
            context=context,
            label=baseline_label,
            expected_targets=len(manifest_targets),
            receipt_role=args.baseline_receipt_role,
            candidate_spec_path=args.baseline_candidate_spec,
            allow_legacy_v1=args.baseline_allow_legacy_v1,
            manifest_path=args.manifest,
        ),
        "candidate": authenticate_scoring_bundle(
            receipt_path=args.candidate_scoring_receipt,
            results_path=args.candidate_results,
            context=context,
            label=candidate_label,
            expected_targets=len(manifest_targets),
            receipt_role=args.candidate_receipt_role,
            candidate_spec_path=args.candidate_candidate_spec,
            allow_legacy_v1=args.candidate_allow_legacy_v1,
            manifest_path=args.manifest,
        ),
    }
    shared_scoring_identity = require_shared_scoring_identity(authenticated_bundles)
    baseline_summary, baseline_scored, baseline_sidecars = _authenticate_results_bundle(
        args.baseline_results,
        manifest_targets,
    )
    candidate_summary, candidate_scored, candidate_sidecars = _authenticate_results_bundle(
        args.candidate_results,
        manifest_targets,
    )
    baseline_nmae_targets = frozenset(
        baseline_scored.loc[
            baseline_scored["metric"] == NMAE_METRIC,
            "perturbation",
        ]
    )
    candidate_nmae_targets = frozenset(
        candidate_scored.loc[
            candidate_scored["metric"] == NMAE_METRIC,
            "perturbation",
        ]
    )
    require(
        baseline_nmae_targets == candidate_nmae_targets,
        f"{context}: baseline and candidate NMAE omission subsets differ",
    )

    route_targets = {
        "all": manifest_targets,
        "direct": frozenset(
            manifest.loc[manifest["validation_route"] == "direct", "target_gene"]
        ),
        "held_target": frozenset(
            manifest.loc[
                manifest["validation_route"] == "held_target",
                "target_gene",
            ]
        ),
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cohorts = {
        cohort: _cohort_comparison(
            cohort_targets=route_targets[cohort],
            baseline_summary_frame=baseline_summary,
            candidate_summary_frame=candidate_summary,
            baseline_scored=baseline_scored,
            candidate_scored=candidate_scored,
            matched_nmae_targets=baseline_nmae_targets,
            rng=rng,
            tolerance=args.absolute_tolerance,
        )
        for cohort in COHORTS
    }
    experiment_gate = _experiment_gate(
        role=args.role,
        context=context,
        cohorts=cohorts,
    )
    primary_gate = (
        _primary_promotion_gate(cohorts)
        if args.role == "primary"
        else {
            "applicable": False,
            "reason": "Only the pre-registered HepG2 primary context can promote a candidate",
            "passed": None,
        }
    )

    report = {
        "schema": SCHEMA,
        "context": context,
        "role": args.role,
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "metric_orientation": SCORED_METRICS,
        "configuration": {
            "absolute_tolerance": float(args.absolute_tolerance),
            "bootstrap_unit": "paired_target_axis",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_interval_method": "percentile",
            "bootstrap_two_sided_confidence": 0.90,
            "bootstrap_one_sided_lower_confidence": 0.90,
            "nmae_population": "matched_finite_cell_eval2_subset_only",
            "mse_aggregation": (
                "sum(expr_mse_unbiased_capped)/sum(expr_distance_unbiased)"
            ),
        },
        "cohort_sizes": {cohort: len(targets) for cohort, targets in route_targets.items()},
        "matched_nmae_subset": {
            "matched": True,
            "targets": int(len(baseline_nmae_targets)),
            "omitted_manifest_targets": int(
                len(manifest_targets) - len(baseline_nmae_targets)
            ),
            "target_axis_sha256": _target_axis_sha256(baseline_nmae_targets),
            "by_cohort": {
                cohort: {
                    "included_targets": int(len(targets & baseline_nmae_targets)),
                    "omitted_targets": int(len(targets - baseline_nmae_targets)),
                }
                for cohort, targets in route_targets.items()
            },
        },
        "cohorts": cohorts,
        "experiment_gate": experiment_gate,
        "primary_promotion_gate": primary_gate,
        "uncertainty_contract": {
            "point_gate_is_not_overridden_by_bootstrap": True,
            "metric_level_intervals_reported": True,
            "all_six_improvement_counts_reported_per_cohort": True,
            "one_sided_lower_bound_above_zero_is_strong_improvement_evidence": True,
        },
        "provenance": {
            "manifest": _describe_file(args.manifest),
            "baseline_results": _describe_file(args.baseline_results),
            "candidate_results": _describe_file(args.candidate_results),
            "authenticated_scoring_bundles": authenticated_bundles,
            "shared_scoring_identity": shared_scoring_identity,
            "baseline_cell_eval2_sidecars": baseline_sidecars,
            "candidate_cell_eval2_sidecars": candidate_sidecars,
        },
        "leakage_firewall": {
            "reads_manifest": True,
            "reads_cell_eval2_results_csv_only": True,
            "reads_cell_eval2_aggregate_sidecars": True,
            "reads_authenticated_scoring_receipts": True,
            "reads_candidate_specs_and_small_verification_sidecars": True,
            "hashes_prediction_and_truth_scoring_views": True,
            "loads_scoring_view_expression_matrices": False,
            "reads_truth_h5ad": False,
            "reads_prediction_h5ad": False,
            "reads_control_h5ad": False,
            "writes_or_modifies_input_artifacts": False,
        },
    }
    _atomic_write_json(args.output_json, report)
    print(json.dumps({"experiment_gate": experiment_gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
