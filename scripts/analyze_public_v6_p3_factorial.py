#!/usr/bin/env python3
"""Analyze the public V6 P3 target-force two-by-two factorial experiment.

The analyzer is leakage-safe by construction. It authenticates one sealed
route manifest, four scorer receipts, their result bundles, and the immutable
scoring-view hashes without loading expression matrices. All four arms use
the same accepted paired-target bootstrap draws for a given cohort and metric.
Consequently, the reported interaction interval is a genuine four-arm paired
bootstrap, including for the official ratio-of-sums expression MSE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from authenticate_public_scoring_bundle import (
    authenticate_scoring_bundle,
    require_shared_scoring_identity,
)
from build_public_v6_p3_factor_contract import (
    ARM_IDENTITIES as FACTOR_ARM_IDENTITIES,
    authenticate_factor_contract_receipt,
)
from analyze_public_paired_bootstrap import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    MSE_DENOMINATOR_METRIC as BOOTSTRAP_MSE_DENOMINATOR_METRIC,
    MSE_NUMERATOR_METRIC as BOOTSTRAP_MSE_NUMERATOR_METRIC,
    NMAE_METRIC as BOOTSTRAP_NMAE_METRIC,
    _atomic_write_json,
    _describe_file,
    _read_manifest,
    require,
)
from compare_public_v6_sequential import (
    COHORTS,
    DIRECTION_METRICS,
    PRIMARY_CONTEXT,
    _authenticate_results_bundle,
    _primary_promotion_gate,
    _target_axis_sha256,
    _validate_shared_metric_contract,
)
from summarize_public_validation import (
    COHORT_OPTIONAL_METRICS,
    MSE_DENOMINATOR_METRIC,
    MSE_METRIC,
    MSE_NUMERATOR_METRIC,
    NMAE_METRIC,
    SCORED_METRICS,
    _mean_summary,
    non_regression_gate,
    oriented_improvement,
    promotion_gate,
)


SCHEMA = "vcc-public-v6-p3-factorial-v1"
ARM_KEYS = ("A", "B", "C", "D")
CONTRAST_FORMULAS = {
    "C_minus_A": {"C": 1.0, "A": -1.0},
    "D_minus_B": {"D": 1.0, "B": -1.0},
    "B_minus_A": {"B": 1.0, "A": -1.0},
    "D_minus_C": {"D": 1.0, "C": -1.0},
    "interaction_D_minus_C_minus_B_minus_A": {
        "D": 1.0,
        "C": -1.0,
        "B": -1.0,
        "A": 1.0,
    },
}
DEPLOYMENT_PAIRS = {
    "C_vs_B": ("B", "C"),
    "D_vs_B": ("B", "D"),
    "D_vs_C": ("C", "D"),
}
MAXIMUM_BOOTSTRAP_ATTEMPTS_FACTOR = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--context", type=str, required=True)
    parser.add_argument(
        "--factor-contract-receipt",
        type=Path,
        required=True,
        help=(
            "Passing immutable four-arm generation factor contract. It is "
            "re-authenticated before any scoring analysis or selection."
        ),
    )
    for arm in ARM_KEYS:
        lower = arm.lower()
        parser.add_argument(f"--arm-{lower}-label", type=str, required=True)
        parser.add_argument(f"--arm-{lower}-results", type=Path, required=True)
        parser.add_argument(f"--arm-{lower}-scoring-receipt", type=Path, required=True)
        parser.add_argument(
            f"--arm-{lower}-receipt-role",
            choices=("anchor", "v51_alpha010"),
        )
        parser.add_argument(f"--arm-{lower}-candidate-spec", type=Path)
        parser.add_argument(
            f"--arm-{lower}-allow-legacy-v1",
            action="store_true",
            help="Permit an immutable pre-v2 V6 arm for read-only analysis.",
        )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    return parser.parse_args()


def _validate_factorial_metric_contract() -> None:
    """Fail closed if imported scorer contracts have drifted."""

    _validate_shared_metric_contract()
    require(
        MSE_NUMERATOR_METRIC == BOOTSTRAP_MSE_NUMERATOR_METRIC,
        "MSE numerator contracts diverged",
    )
    require(
        MSE_DENOMINATOR_METRIC == BOOTSTRAP_MSE_DENOMINATOR_METRIC,
        "MSE denominator contracts diverged",
    )
    require(NMAE_METRIC == BOOTSTRAP_NMAE_METRIC, "NMAE contracts diverged")


def _arm_cli_inputs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARM_KEYS:
        lower = arm.lower()
        label = str(getattr(args, f"arm_{lower}_label")).strip()
        path = Path(getattr(args, f"arm_{lower}_results"))
        require(bool(label), f"Arm {arm} label cannot be empty")
        arms[arm] = {
            "label": label,
            "results_path": path,
            "scoring_receipt_path": Path(
                getattr(args, f"arm_{lower}_scoring_receipt")
            ),
            "receipt_role": getattr(args, f"arm_{lower}_receipt_role"),
            "candidate_spec_path": getattr(args, f"arm_{lower}_candidate_spec"),
            "allow_legacy_v1": bool(
                getattr(args, f"arm_{lower}_allow_legacy_v1")
            ),
        }
    require(
        len({entry["label"] for entry in arms.values()}) == len(ARM_KEYS),
        "Factorial arm labels must be distinct",
    )
    resolved = [entry["results_path"].resolve() for entry in arms.values()]
    require(
        len(set(resolved)) == len(ARM_KEYS),
        "Factorial result paths must be distinct",
    )
    return arms


def _validate_factor_identities(
    authenticated: dict[str, dict[str, Any]],
) -> None:
    """Bind CLI arms to the pre-registered P3 factor levels."""

    require(
        authenticated["A"]["kind"] == "v51"
        and authenticated["A"].get("receipt_role") == "anchor",
        "Arm A must be the authenticated V5.1 anchor role",
    )
    require(
        authenticated["A"]["label"] == FACTOR_ARM_IDENTITIES["A"]
        and authenticated["B"]["label"] == FACTOR_ARM_IDENTITIES["B"]
        and authenticated["C"]["label"] == FACTOR_ARM_IDENTITIES["C"]
        and authenticated["D"]["label"] == FACTOR_ARM_IDENTITIES["D"],
        "Factorial arm labels differ from the registered factor contract",
    )
    require(
        authenticated["B"]["kind"] == "v51"
        and authenticated["B"].get("receipt_role") == "v51_alpha010",
        "Arm B must be the authenticated V5.1 alpha-0.10 role",
    )
    require(
        authenticated["A"]["receipt"] == authenticated["B"]["receipt"],
        "Arms A and B must come from the same completed V5.1 receipt",
    )
    expected = {
        "C": {"target_remaining_fraction": 0.40, "residual_alpha": 0.00},
        "D": {"target_remaining_fraction": 0.40, "residual_alpha": 0.10},
    }
    for arm, levels in expected.items():
        identity = authenticated[arm]
        require(identity["kind"] == "v6", f"Arm {arm} must be a V6 candidate")
        require(
            identity.get("candidate_spec_mode") == "strict-v2",
            f"Arm {arm} must use a strict v2 candidate spec",
        )
        configuration = identity["candidate_configuration"]
        for name, value in levels.items():
            require(
                float(configuration.get(name, float("nan"))) == value,
                f"Arm {arm} has the wrong {name}",
            )
    c_configuration = dict(authenticated["C"]["candidate_configuration"])
    d_configuration = dict(authenticated["D"]["candidate_configuration"])
    c_configuration.pop("residual_alpha", None)
    d_configuration.pop("residual_alpha", None)
    require(
        c_configuration == d_configuration,
        "Arms C and D differ outside the residual-alpha factor",
    )


def _aligned_arm_values(
    scored_by_arm: dict[str, pd.DataFrame],
    metric: str,
    target_axis: frozenset[str],
) -> dict[str, np.ndarray]:
    require(bool(target_axis), f"{metric}: target axis is empty")
    ordered_targets = sorted(target_axis)
    output: dict[str, np.ndarray] = {}
    for arm in ARM_KEYS:
        frame = scored_by_arm[arm]
        subset = frame.loc[
            (frame["metric"] == metric)
            & (frame["perturbation"].isin(target_axis)),
            ["perturbation", "value"],
        ]
        require(
            set(subset["perturbation"]) == target_axis,
            f"{metric}: arm {arm} target axis is incomplete",
        )
        values = (
            subset.set_index("perturbation")["value"]
            .reindex(ordered_targets)
            .to_numpy(dtype=np.float64)
        )
        require(not np.isinf(values).any(), f"{metric}: arm {arm} contains infinity")
        output[arm] = values
    return output


def _indices_sha256(indices: np.ndarray, targets: frozenset[str]) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(sorted(targets)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.asarray(indices, dtype="<i4", order="C").tobytes(order="C"))
    return digest.hexdigest()


def _common_bootstrap_indices(
    *,
    axis_size: int,
    rng: np.random.Generator,
    validity: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, int]:
    """Draw 10,000 indices that are jointly valid for all four arms."""

    require(axis_size > 0, "Bootstrap axis is empty")
    accepted: list[np.ndarray] = []
    accepted_count = 0
    attempted = 0
    maximum_attempts = BOOTSTRAP_RESAMPLES * MAXIMUM_BOOTSTRAP_ATTEMPTS_FACTOR
    while accepted_count < BOOTSTRAP_RESAMPLES and attempted < maximum_attempts:
        remaining = BOOTSTRAP_RESAMPLES - accepted_count
        batch_size = min(maximum_attempts - attempted, max(1024, remaining))
        indices = rng.integers(
            0,
            axis_size,
            size=(batch_size, axis_size),
            dtype=np.int32,
        )
        valid = np.asarray(validity(indices), dtype=bool)
        require(valid.shape == (batch_size,), "Bootstrap validity mask is malformed")
        if valid.any():
            accepted.append(indices[valid])
            accepted_count += int(valid.sum())
        attempted += batch_size
    require(
        accepted_count >= BOOTSTRAP_RESAMPLES,
        "Could not obtain 10,000 jointly valid four-arm bootstrap draws",
    )
    return np.concatenate(accepted, axis=0)[:BOOTSTRAP_RESAMPLES], attempted


def _interval_summary(samples: np.ndarray) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    require(
        values.shape == (BOOTSTRAP_RESAMPLES,) and np.isfinite(values).all(),
        "Factorial bootstrap statistics are malformed",
    )
    lower, upper = np.quantile(values, [0.05, 0.95])
    one_sided_lower = float(np.quantile(values, 0.10))
    return {
        "bootstrap_90pct_two_sided_ci": {
            "lower": float(lower),
            "upper": float(upper),
        },
        "bootstrap_90pct_one_sided_lower_bound": one_sided_lower,
        "one_sided_lower_bound_above_zero": bool(one_sided_lower > 0.0),
    }


def _four_arm_nan_diagnostics(values_by_arm: dict[str, np.ndarray]) -> dict[str, Any]:
    masks = {arm: np.isfinite(values_by_arm[arm]) for arm in ARM_KEYS}
    stacked = np.stack([masks[arm] for arm in ARM_KEYS], axis=0)
    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(ARM_KEYS):
        for right in ARM_KEYS[left_index + 1 :]:
            pairwise[f"{left}_vs_{right}"] = {
                "finite_target_masks_identical": bool(
                    np.array_equal(masks[left], masks[right])
                ),
                "finite_target_intersection": int(
                    (masks[left] & masks[right]).sum()
                ),
                "finite_target_union": int((masks[left] | masks[right]).sum()),
                "left_only_finite_targets": int(
                    (masks[left] & ~masks[right]).sum()
                ),
                "right_only_finite_targets": int(
                    (~masks[left] & masks[right]).sum()
                ),
            }
    return {
        "target_axis_size": int(stacked.shape[1]),
        "finite_targets_by_arm": {
            arm: int(masks[arm].sum()) for arm in ARM_KEYS
        },
        "nan_targets_by_arm": {
            arm: int((~masks[arm]).sum()) for arm in ARM_KEYS
        },
        "four_way_finite_intersection": int(stacked.all(axis=0).sum()),
        "four_way_finite_union": int(stacked.any(axis=0).sum()),
        "all_four_finite_masks_identical": bool(
            all(np.array_equal(masks["A"], masks[arm]) for arm in ARM_KEYS[1:])
        ),
        "pairwise": pairwise,
    }


def _linear_contrast(values: dict[str, float], weights: dict[str, float]) -> float:
    return float(sum(weights.get(arm, 0.0) * values[arm] for arm in ARM_KEYS))


def _contrast_reports(
    *,
    observed_raw: dict[str, float],
    bootstrapped_raw: dict[str, np.ndarray],
    direction: str,
) -> dict[str, Any]:
    orientation = 1.0 if direction == "higher" else -1.0
    reports: dict[str, Any] = {}
    for name, weights in CONTRAST_FORMULAS.items():
        raw_point = _linear_contrast(observed_raw, weights)
        raw_samples = sum(
            weights.get(arm, 0.0) * bootstrapped_raw[arm]
            for arm in ARM_KEYS
        )
        oriented_samples = orientation * raw_samples
        report = {
            "evaluable": True,
            "formula": " + ".join(
                f"{weight:+g}*{arm}"
                for arm, weight in weights.items()
            ),
            "observed_raw_contrast": float(raw_point),
            "observed_oriented_contrast_positive_is_better": float(
                orientation * raw_point
            ),
            "all_four_arms_use_identical_bootstrap_indices": True,
        }
        report.update(_interval_summary(oriented_samples))
        reports[name] = report
    return reports


def _not_evaluable_contrasts(reason: str) -> dict[str, Any]:
    return {
        name: {
            "evaluable": False,
            "reason": reason,
            "observed_raw_contrast": None,
            "observed_oriented_contrast_positive_is_better": None,
            "bootstrap_90pct_two_sided_ci": None,
            "bootstrap_90pct_one_sided_lower_bound": None,
            "one_sided_lower_bound_above_zero": None,
        }
        for name in CONTRAST_FORMULAS
    }


def _factorial_mean_metric(
    *,
    metric: str,
    direction: str,
    values_by_arm: dict[str, np.ndarray],
    target_axis: frozenset[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    finite_masks = {arm: np.isfinite(values_by_arm[arm]) for arm in ARM_KEYS}
    arm_aggregates = {
        arm: {
            "mean": (
                float(values_by_arm[arm][finite_masks[arm]].mean())
                if finite_masks[arm].any()
                else None
            ),
            "finite_targets": int(finite_masks[arm].sum()),
            "missing_targets": int((~finite_masks[arm]).sum()),
            "aggregation": "finite_mean",
        }
        for arm in ARM_KEYS
    }
    output: dict[str, Any] = {
        "direction": direction,
        "aggregation": "finite_mean",
        "resampling_axis_targets": int(len(target_axis)),
        "target_axis_sha256": _target_axis_sha256(target_axis),
        "arm_aggregates": arm_aggregates,
    }
    if metric in DIRECTION_METRICS:
        output["finite_mask_diagnostics"] = _four_arm_nan_diagnostics(values_by_arm)
    if not all(finite_masks[arm].any() for arm in ARM_KEYS):
        output["contrasts"] = _not_evaluable_contrasts(
            "at_least_one_arm_has_no_finite_metric_value_on_this_cohort"
        )
        output["common_bootstrap"] = {
            "evaluable": False,
            "accepted_draws": 0,
            "reason": "four_arm_joint_nanmean_is_not_evaluable",
        }
        return output

    def validity(indices: np.ndarray) -> np.ndarray:
        valid = np.ones(indices.shape[0], dtype=bool)
        for arm in ARM_KEYS:
            valid &= finite_masks[arm][indices].sum(axis=1) > 0
        return valid

    indices, attempted = _common_bootstrap_indices(
        axis_size=len(target_axis),
        rng=rng,
        validity=validity,
    )
    bootstrap_by_arm: dict[str, np.ndarray] = {}
    for arm in ARM_KEYS:
        sampled = values_by_arm[arm][indices]
        counts = np.isfinite(sampled).sum(axis=1)
        require(np.all(counts > 0), "Accepted nanmean draw became non-evaluable")
        bootstrap_by_arm[arm] = np.nansum(sampled, axis=1) / counts
    observed = {
        arm: float(arm_aggregates[arm]["mean"]) for arm in ARM_KEYS
    }
    output["contrasts"] = _contrast_reports(
        observed_raw=observed,
        bootstrapped_raw=bootstrap_by_arm,
        direction=direction,
    )
    output["common_bootstrap"] = {
        "evaluable": True,
        "accepted_draws": BOOTSTRAP_RESAMPLES,
        "attempted_draws": int(attempted),
        "discarded_draws": int(attempted - BOOTSTRAP_RESAMPLES),
        "indices_sha256": _indices_sha256(indices, target_axis),
        "same_indices_for_all_four_arms_and_all_contrasts": True,
    }
    return output


def _factorial_mse_metric(
    *,
    scored_by_arm: dict[str, pd.DataFrame],
    target_axis: frozenset[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    numerator = _aligned_arm_values(
        scored_by_arm,
        MSE_NUMERATOR_METRIC,
        target_axis,
    )
    denominator = _aligned_arm_values(
        scored_by_arm,
        MSE_DENOMINATOR_METRIC,
        target_axis,
    )
    for arm in ARM_KEYS:
        require(
            np.isfinite(numerator[arm]).all()
            and np.isfinite(denominator[arm]).all(),
            f"MSE components are non-finite in arm {arm}",
        )
        require(
            float(denominator[arm].sum(dtype=np.float64)) > 0.0,
            f"MSE denominator sum is non-positive in arm {arm}",
        )

    def validity(indices: np.ndarray) -> np.ndarray:
        valid = np.ones(indices.shape[0], dtype=bool)
        for arm in ARM_KEYS:
            valid &= denominator[arm][indices].sum(axis=1) > 0.0
        return valid

    indices, attempted = _common_bootstrap_indices(
        axis_size=len(target_axis),
        rng=rng,
        validity=validity,
    )
    observed: dict[str, float] = {}
    bootstrap_by_arm: dict[str, np.ndarray] = {}
    arm_aggregates: dict[str, Any] = {}
    for arm in ARM_KEYS:
        numerator_sum = float(numerator[arm].sum(dtype=np.float64))
        denominator_sum = float(denominator[arm].sum(dtype=np.float64))
        ratio = float(numerator_sum / denominator_sum)
        observed[arm] = ratio
        sampled_denominator = denominator[arm][indices].sum(axis=1)
        require(
            np.all(sampled_denominator > 0.0),
            "Accepted MSE draw has a non-positive denominator",
        )
        bootstrap_by_arm[arm] = (
            numerator[arm][indices].sum(axis=1) / sampled_denominator
        )
        arm_aggregates[arm] = {
            "ratio_of_sums": ratio,
            "numerator_sum": numerator_sum,
            "denominator_sum": denominator_sum,
            "finite_targets": int(len(target_axis)),
            "missing_targets": 0,
            "aggregation": "ratio_of_sums",
        }
    return {
        "direction": "lower",
        "aggregation": "ratio_of_sums",
        "formula": (
            "sum(expr_mse_unbiased_capped)/sum(expr_distance_unbiased)"
        ),
        "per_target_derived_rows_used": False,
        "resampling_axis_targets": int(len(target_axis)),
        "target_axis_sha256": _target_axis_sha256(target_axis),
        "arm_aggregates": arm_aggregates,
        "contrasts": _contrast_reports(
            observed_raw=observed,
            bootstrapped_raw=bootstrap_by_arm,
            direction="lower",
        ),
        "common_bootstrap": {
            "evaluable": True,
            "accepted_draws": BOOTSTRAP_RESAMPLES,
            "attempted_draws": int(attempted),
            "discarded_draws": int(attempted - BOOTSTRAP_RESAMPLES),
            "indices_sha256": _indices_sha256(indices, target_axis),
            "same_indices_for_all_four_arms_and_all_contrasts": True,
        },
    }


def _cohort_factorial(
    *,
    cohort_targets: frozenset[str],
    scored_by_arm: dict[str, pd.DataFrame],
    matched_nmae_targets: frozenset[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    require(bool(cohort_targets), "Factorial cohort target axis is empty")
    metrics: dict[str, Any] = {}
    for metric, direction in SCORED_METRICS.items():
        if metric == MSE_METRIC:
            metrics[metric] = _factorial_mse_metric(
                scored_by_arm=scored_by_arm,
                target_axis=cohort_targets,
                rng=rng,
            )
            continue
        metric_targets = (
            cohort_targets & matched_nmae_targets
            if metric == NMAE_METRIC
            else cohort_targets
        )
        if not metric_targets:
            metrics[metric] = {
                "direction": direction,
                "aggregation": "finite_mean",
                "resampling_axis_targets": 0,
                "target_axis_sha256": _target_axis_sha256(metric_targets),
                "arm_aggregates": {
                    arm: {
                        "mean": None,
                        "finite_targets": 0,
                        "missing_targets": int(len(cohort_targets)),
                        "aggregation": "finite_mean",
                    }
                    for arm in ARM_KEYS
                },
                "contrasts": _not_evaluable_contrasts(
                    "matched_metric_population_is_empty_on_this_cohort"
                ),
                "common_bootstrap": {
                    "evaluable": False,
                    "accepted_draws": 0,
                    "reason": "empty_matched_metric_population",
                },
            }
            continue
        values_by_arm = _aligned_arm_values(
            scored_by_arm,
            metric,
            metric_targets,
        )
        metrics[metric] = _factorial_mean_metric(
            metric=metric,
            direction=direction,
            values_by_arm=values_by_arm,
            target_axis=metric_targets,
            rng=rng,
        )
        metrics[metric]["omitted_cohort_targets"] = int(
            len(cohort_targets) - len(metric_targets)
        )
    return {
        "cohort_targets": int(len(cohort_targets)),
        "metrics": metrics,
    }


def _deployment_cohort(
    *,
    baseline_arm: str,
    candidate_arm: str,
    cohort_targets: frozenset[str],
    summary_by_arm: dict[str, pd.DataFrame],
    tolerance: float,
) -> dict[str, Any]:
    baseline_frame = summary_by_arm[baseline_arm].loc[
        summary_by_arm[baseline_arm]["perturbation"].isin(cohort_targets)
    ]
    candidate_frame = summary_by_arm[candidate_arm].loc[
        summary_by_arm[candidate_arm]["perturbation"].isin(cohort_targets)
    ]
    baseline = _mean_summary(baseline_frame, len(cohort_targets))
    candidate = _mean_summary(candidate_frame, len(cohort_targets))
    improvements: dict[str, float] = {}
    not_evaluable: list[str] = []
    metric_evaluability: dict[str, dict[str, Any]] = {}
    for metric in SCORED_METRICS:
        baseline_mean = baseline[metric]["mean"]
        candidate_mean = candidate[metric]["mean"]
        if (baseline_mean is None) != (candidate_mean is None):
            require(
                metric in DIRECTION_METRICS,
                f"{baseline_arm}->{candidate_arm}: metric evaluability drift: {metric}",
            )
        evaluable = baseline_mean is not None and candidate_mean is not None
        metric_evaluability[metric] = {
            "evaluable": evaluable,
            "baseline_aggregate": baseline[metric],
            "candidate_aggregate": candidate[metric],
        }
        if not evaluable:
            require(metric in COHORT_OPTIONAL_METRICS, "Required metric is not evaluable")
            not_evaluable.append(metric)
            continue
        improvements[metric] = oriented_improvement(
            metric,
            float(candidate_mean),
            float(baseline_mean),
        )
    not_evaluable_tuple = tuple(not_evaluable)
    return {
        "baseline_arm": baseline_arm,
        "candidate_arm": candidate_arm,
        "metrics": metric_evaluability,
        "oriented_improvement_positive_is_better": improvements,
        "not_evaluable_metrics": not_evaluable,
        "point_estimate_promotion_gate": promotion_gate(
            improvements,
            tolerance,
            not_evaluable_tuple,
        ),
        "point_estimate_non_regression_gate": non_regression_gate(
            improvements,
            tolerance,
            not_evaluable_tuple,
        ),
    }


def _deployment_comparison(
    *,
    baseline_arm: str,
    candidate_arm: str,
    route_targets: dict[str, frozenset[str]],
    summary_by_arm: dict[str, pd.DataFrame],
    tolerance: float,
    context: str,
) -> dict[str, Any]:
    cohorts = {
        cohort: _deployment_cohort(
            baseline_arm=baseline_arm,
            candidate_arm=candidate_arm,
            cohort_targets=route_targets[cohort],
            summary_by_arm=summary_by_arm,
            tolerance=tolerance,
        )
        for cohort in COHORTS
    }
    primary_gate = (
        _primary_promotion_gate(cohorts)
        if context == PRIMARY_CONTEXT
        else {
            "applicable": False,
            "passed": None,
            "reason": f"P3 selection is pre-registered for {PRIMARY_CONTEXT} only",
        }
    )
    return {
        "baseline_arm": baseline_arm,
        "candidate_arm": candidate_arm,
        "cohorts": cohorts,
        "primary_promotion_gate": primary_gate,
    }


def _select_arm(
    *,
    context: str,
    deployment: dict[str, Any],
) -> dict[str, Any]:
    if context != PRIMARY_CONTEXT:
        return {
            "applicable": False,
            "selected_arm": None,
            "reason": f"Deterministic P3 selection is reserved for {PRIMARY_CONTEXT}",
        }
    c_passes = bool(deployment["C_vs_B"]["primary_promotion_gate"]["passed"])
    d_passes = bool(deployment["D_vs_B"]["primary_promotion_gate"]["passed"])
    d_beats_c = bool(deployment["D_vs_C"]["primary_promotion_gate"]["passed"])
    if not c_passes and not d_passes:
        selected = "B"
        reason = "Neither C nor D passes the primary deployment gate against B"
    elif c_passes and not d_passes:
        selected = "C"
        reason = "Only C passes the primary deployment gate against B"
    elif d_passes and not c_passes:
        selected = "D"
        reason = "Only D passes the primary deployment gate against B"
    elif d_beats_c:
        selected = "D"
        reason = "C and D pass against B, and D also passes against C"
    else:
        selected = "C"
        reason = "C and D pass against B, but D does not pass against C"
    return {
        "applicable": True,
        "selection_uses_point_estimate_gates_only": True,
        "bootstrap_does_not_override_a_failed_gate": True,
        "C_passes_against_B": c_passes,
        "D_passes_against_B": d_passes,
        "D_passes_against_C": d_beats_c,
        "selected_arm": selected,
        "selected_label": deployment["arm_labels"][selected],
        "reason": reason,
        "requires_jurkat_non_harm_before_final_deployment": selected in {"C", "D"},
    }


def main() -> None:
    args = parse_args()
    _validate_factorial_metric_contract()
    require(not args.output_json.exists(), f"Refusing to overwrite: {args.output_json}")
    require(args.absolute_tolerance >= 0.0, "absolute-tolerance cannot be negative")
    context = str(args.context).strip()
    require(bool(context), "Context cannot be empty")
    arm_inputs = _arm_cli_inputs(args)
    require(context == PRIMARY_CONTEXT, "P3 factorial selection is restricted to HepG2")
    require(
        arm_inputs["A"]["scoring_receipt_path"].resolve()
        == arm_inputs["B"]["scoring_receipt_path"].resolve(),
        "Arms A and B must use the same V5.1 scoring receipt",
    )
    require(
        arm_inputs["C"]["candidate_spec_path"] is not None
        and arm_inputs["D"]["candidate_spec_path"] is not None,
        "Arms C and D require strict candidate specs",
    )
    factor_contract = authenticate_factor_contract_receipt(
        args.factor_contract_receipt,
        expected_manifest=args.manifest,
        expected_v51_scoring_receipt=arm_inputs["A"]["scoring_receipt_path"],
        expected_c_spec=arm_inputs["C"]["candidate_spec_path"],
        expected_d_spec=arm_inputs["D"]["candidate_spec_path"],
    )

    manifest = _read_manifest(args.manifest)
    manifest_targets = frozenset(manifest["target_gene"])
    route_targets = {
        "all": manifest_targets,
        "direct": frozenset(
            manifest.loc[manifest["validation_route"] == "direct", "target_gene"]
        ),
        "held_target": frozenset(
            manifest.loc[
                manifest["validation_route"] == "held_target", "target_gene"
            ]
        ),
    }
    require(
        all(bool(route_targets[cohort]) for cohort in COHORTS),
        "Every factorial cohort must contain at least one target",
    )

    authenticated_arms = {
        arm: authenticate_scoring_bundle(
            receipt_path=arm_inputs[arm]["scoring_receipt_path"],
            results_path=arm_inputs[arm]["results_path"],
            context=context,
            label=arm_inputs[arm]["label"],
            expected_targets=len(manifest_targets),
            receipt_role=arm_inputs[arm]["receipt_role"],
            candidate_spec_path=arm_inputs[arm]["candidate_spec_path"],
            allow_legacy_v1=arm_inputs[arm]["allow_legacy_v1"],
            manifest_path=args.manifest,
        )
        for arm in ARM_KEYS
    }
    _validate_factor_identities(authenticated_arms)
    shared_scoring_identity = require_shared_scoring_identity(authenticated_arms)

    summary_by_arm: dict[str, pd.DataFrame] = {}
    scored_by_arm: dict[str, pd.DataFrame] = {}
    sidecars_by_arm: dict[str, Any] = {}
    nmae_by_arm: dict[str, frozenset[str]] = {}
    for arm in ARM_KEYS:
        summary, scored, sidecars = _authenticate_results_bundle(
            arm_inputs[arm]["results_path"],
            manifest_targets,
        )
        summary_by_arm[arm] = summary
        scored_by_arm[arm] = scored
        sidecars_by_arm[arm] = sidecars
        nmae_by_arm[arm] = frozenset(
            scored.loc[scored["metric"] == NMAE_METRIC, "perturbation"]
        )
    require(
        all(nmae_by_arm[arm] == nmae_by_arm["A"] for arm in ARM_KEYS[1:]),
        "Four-arm NMAE omission subsets differ",
    )
    matched_nmae_targets = nmae_by_arm["A"]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cohorts = {
        cohort: _cohort_factorial(
            cohort_targets=route_targets[cohort],
            scored_by_arm=scored_by_arm,
            matched_nmae_targets=matched_nmae_targets,
            rng=rng,
        )
        for cohort in COHORTS
    }
    deployment = {
        name: _deployment_comparison(
            baseline_arm=pair[0],
            candidate_arm=pair[1],
            route_targets=route_targets,
            summary_by_arm=summary_by_arm,
            tolerance=args.absolute_tolerance,
            context=context,
        )
        for name, pair in DEPLOYMENT_PAIRS.items()
    }
    deployment["arm_labels"] = {
        arm: arm_inputs[arm]["label"] for arm in ARM_KEYS
    }
    selection = _select_arm(context=context, deployment=deployment)

    report = {
        "schema": SCHEMA,
        "context": context,
        "factorial_design": {
            "arms": {
                "A": {
                    "label": arm_inputs["A"]["label"],
                    "target_remaining_fraction": 0.20,
                    "residual_alpha": 0.00,
                },
                "B": {
                    "label": arm_inputs["B"]["label"],
                    "target_remaining_fraction": 0.20,
                    "residual_alpha": 0.10,
                },
                "C": {
                    "label": arm_inputs["C"]["label"],
                    "target_remaining_fraction": 0.40,
                    "residual_alpha": 0.00,
                },
                "D": {
                    "label": arm_inputs["D"]["label"],
                    "target_remaining_fraction": 0.40,
                    "residual_alpha": 0.10,
                },
            },
            "contrasts": {
                "C_minus_A": "target-force effect without residual",
                "D_minus_B": "target-force effect with residual",
                "B_minus_A": "residual effect at target fold 0.20",
                "D_minus_C": "residual effect at target fold 0.40",
                "interaction_D_minus_C_minus_B_minus_A": (
                    "(D-C)-(B-A), equivalently (D-B)-(C-A)"
                ),
            },
        },
        "metric_orientation": SCORED_METRICS,
        "configuration": {
            "absolute_tolerance": float(args.absolute_tolerance),
            "bootstrap_unit": "paired_target_axis",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_interval_method": "percentile",
            "bootstrap_two_sided_confidence": 0.90,
            "bootstrap_one_sided_lower_confidence": 0.90,
            "common_four_arm_draws_within_each_cohort_metric": True,
            "nmae_population": "exact_four_arm_matched_cell_eval2_subset_only",
            "mse_aggregation": (
                "sum(expr_mse_unbiased_capped)/sum(expr_distance_unbiased)"
            ),
        },
        "cohort_sizes": {
            cohort: int(len(targets)) for cohort, targets in route_targets.items()
        },
        "matched_nmae_subset": {
            "matched_across_all_four_arms": True,
            "targets": int(len(matched_nmae_targets)),
            "omitted_manifest_targets": int(
                len(manifest_targets) - len(matched_nmae_targets)
            ),
            "target_axis_sha256": _target_axis_sha256(matched_nmae_targets),
            "by_cohort": {
                cohort: {
                    "included_targets": int(len(targets & matched_nmae_targets)),
                    "omitted_targets": int(len(targets - matched_nmae_targets)),
                }
                for cohort, targets in route_targets.items()
            },
        },
        "cohorts": cohorts,
        "deployment_comparisons": deployment,
        "deterministic_selection": selection,
        "provenance": {
            "manifest": _describe_file(args.manifest),
            "arms": {
                arm: {
                    "label": arm_inputs[arm]["label"],
                    "results": _describe_file(arm_inputs[arm]["results_path"]),
                    "authenticated_scoring_bundle": authenticated_arms[arm],
                    "cell_eval2_sidecars": sidecars_by_arm[arm],
                }
                for arm in ARM_KEYS
            },
            "shared_scoring_identity": shared_scoring_identity,
            "factor_contract": factor_contract,
        },
        "leakage_firewall": {
            "reads_manifest": True,
            "reads_cell_eval2_results_csv_only": True,
            "reads_cell_eval2_aggregate_sidecars": True,
            "reads_authenticated_scoring_receipts": True,
            "requires_authenticated_four_arm_factor_contract": True,
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
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
