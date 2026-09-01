#!/usr/bin/env python3
"""Run a leakage-safe paired bootstrap on public V5.1 scorer outputs.

This analyzer intentionally consumes only the sealed split manifest and the
per-target ``results.csv`` tables emitted by cell-eval2. It never reads truth,
control, or prediction H5AD files. The target is the resampling unit, and every
bootstrap statistic is computed from paired candidate-versus-anchor values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA = "vcc-public-paired-target-bootstrap-v1"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_901
TWO_SIDED_CONFIDENCE = 0.90
ONE_SIDED_CONFIDENCE = 0.90
NMAE_METRIC = "de_wilcoxon_lfc_nmae"
DERIVED_MSE_METRIC = "expr_mse_unbiased_capped_norm"
MSE_NUMERATOR_METRIC = "expr_mse_unbiased_capped"
MSE_DENOMINATOR_METRIC = "expr_distance_unbiased"
METRIC_ORIENTATION = {
    "pds_cosine": "higher",
    DERIVED_MSE_METRIC: "lower",
    NMAE_METRIC: "lower",
    "de_wilcoxon_direction_fidelity_yield_raw": "higher",
    "de_wilcoxon_direction_reach_raw": "higher",
    "de_wilcoxon_sig_jaccard": "higher",
}
PAIRWISE_RESULT_METRICS = tuple(
    metric
    for metric in METRIC_ORIENTATION
    if metric not in {DERIVED_MSE_METRIC, NMAE_METRIC}
)
MSE_COMPONENT_METRICS = (MSE_NUMERATOR_METRIC, MSE_DENOMINATOR_METRIC)
COMPLETE_RESULT_METRICS = PAIRWISE_RESULT_METRICS + MSE_COMPONENT_METRICS
NULLABLE_COMPLETE_METRICS = (
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
)
STRICT_FINITE_COMPLETE_METRICS = tuple(
    metric
    for metric in COMPLETE_RESULT_METRICS
    if metric not in NULLABLE_COMPLETE_METRICS
)
COHORTS = ("all", "direct", "held_target")
CONTEXTS = (
    ("HepG2", "primary"),
    ("Jurkat", "diagnostic"),
)


def require(condition: bool, message: str) -> None:
    """Raise a consistent fail-closed error when an input contract is broken."""

    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hepg2-anchor", type=Path, required=True)
    parser.add_argument("--hepg2-candidate", type=Path, required=True)
    parser.add_argument("--jurkat-anchor", type=Path, required=True)
    parser.add_argument("--jurkat-candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _describe_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _read_manifest(path: Path) -> pd.DataFrame:
    require(path.is_file(), f"Missing manifest: {path}")
    frame = pd.read_csv(path)
    required = {"target_gene", "validation_route"}
    require(required.issubset(frame.columns), "Manifest lacks required route columns")
    require(not frame.empty, "Manifest target axis is empty")
    require(frame["target_gene"].notna().all(), "Manifest target axis contains a null")
    require(
        frame["validation_route"].notna().all(),
        "Manifest validation route contains a null",
    )
    frame = frame.copy()
    frame["target_gene"] = frame["target_gene"].astype(str).str.strip()
    frame["validation_route"] = frame["validation_route"].astype(str).str.strip()
    require(
        frame["target_gene"].str.len().gt(0).all(),
        "Manifest target axis contains an empty identifier",
    )
    require(frame["target_gene"].is_unique, "Manifest target axis has duplicates")
    require(
        set(frame["validation_route"]) == {"direct", "held_target"},
        "Manifest routes must contain nonempty direct and held_target cohorts only",
    )
    return frame[["target_gene", "validation_route"]]


def _read_scored_results(path: Path, manifest_targets: frozenset[str]) -> pd.DataFrame:
    """Read the scored rows and authenticate their component-level axes."""

    require(path.is_file(), f"Missing scorer results: {path}")
    raw = pd.read_csv(path)
    required_columns = {"perturbation", "metric", "value"}
    require(
        required_columns.issubset(raw.columns),
        f"Scorer table lacks required columns: {path}",
    )
    require(
        raw["perturbation"].notna().all(),
        f"Null perturbation identifier: {path}",
    )
    require(raw["metric"].notna().all(), f"Null metric identifier: {path}")
    raw = raw.copy()
    raw["perturbation"] = raw["perturbation"].astype(str).str.strip()
    raw["metric"] = raw["metric"].astype(str).str.strip()
    require(
        raw["perturbation"].str.len().gt(0).all(),
        f"Empty perturbation identifier: {path}",
    )
    require(
        raw["metric"].str.len().gt(0).all(),
        f"Empty metric identifier: {path}",
    )
    require(
        not raw.duplicated(["perturbation", "metric"]).any(),
        f"Duplicate target/metric row: {path}",
    )
    require(
        set(raw["perturbation"]) == manifest_targets,
        f"Scorer target axis does not exactly match the manifest: {path}",
    )
    require(
        not (raw["metric"] == DERIVED_MSE_METRIC).any(),
        f"Derived MSE must not appear as a per-target result: {path}",
    )
    result_metrics = set(COMPLETE_RESULT_METRICS) | {NMAE_METRIC}
    scored = raw.loc[
        raw["metric"].isin(result_metrics),
        ["perturbation", "metric", "value"],
    ].copy()
    require(not scored.empty, f"Scorer table contains no VCC scored metrics: {path}")
    original_values = scored["value"].copy()
    numeric_values = pd.to_numeric(original_values, errors="coerce")
    require(
        not (original_values.notna() & numeric_values.isna()).any(),
        f"Non-numeric scored value: {path}",
    )
    scored["value"] = numeric_values
    require(
        not np.isinf(scored["value"].to_numpy(dtype=np.float64)).any(),
        f"Infinite scored value: {path}",
    )

    observed = set(
        scored[["perturbation", "metric"]].itertuples(index=False, name=None)
    )
    expected_complete = {
        (target, metric)
        for target in manifest_targets
        for metric in COMPLETE_RESULT_METRICS
    }
    observed_complete = {
        pair for pair in observed if pair[1] in COMPLETE_RESULT_METRICS
    }
    require(
        observed_complete == expected_complete,
        f"Complete scored and MSE-component axes differ from the manifest: {path}",
    )
    for metric in STRICT_FINITE_COMPLETE_METRICS:
        values = scored.loc[scored["metric"] == metric, "value"].to_numpy(
            dtype=np.float64
        )
        require(
            np.isfinite(values).all(),
            f"Required all-finite metric contains a non-finite value ({metric}): {path}",
        )
    for metric in NULLABLE_COMPLETE_METRICS:
        values = scored.loc[scored["metric"] == metric, "value"].to_numpy(
            dtype=np.float64
        )
        require(
            np.isfinite(values).any(),
            f"Nullable metric has no finite targets ({metric}): {path}",
        )
    nmae_targets = frozenset(
        scored.loc[scored["metric"] == NMAE_METRIC, "perturbation"]
    )
    require(bool(nmae_targets), f"NMAE matched omission subset is empty: {path}")
    nmae_values = scored.loc[scored["metric"] == NMAE_METRIC, "value"].to_numpy(
        dtype=np.float64
    )
    require(
        np.isfinite(nmae_values).all(),
        f"NMAE matched omission subset contains a non-finite value: {path}",
    )
    expected_rows = len(expected_complete) + len(nmae_targets)
    require(
        len(scored) == expected_rows,
        f"Scored target/metric row count is invalid: {path}",
    )
    return scored


def _aligned_metric_values(
    anchor: pd.DataFrame,
    candidate: pd.DataFrame,
    metric: str,
    target_axis: frozenset[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Align anchor and candidate values to one deterministic target axis."""

    require(bool(target_axis), f"{metric}: target axis is empty")
    ordered_targets = sorted(target_axis)
    output: list[np.ndarray] = []
    for label, frame in (("anchor", anchor), ("candidate", candidate)):
        subset = frame.loc[
            (frame["metric"] == metric)
            & (frame["perturbation"].isin(target_axis)),
            ["perturbation", "value"],
        ]
        require(
            set(subset["perturbation"]) == target_axis,
            f"{metric}: {label} target axis is incomplete",
        )
        values = (
            subset.set_index("perturbation")["value"]
            .reindex(ordered_targets)
            .to_numpy(dtype=np.float64)
        )
        require(not np.isinf(values).any(), f"{metric}: {label} contains infinity")
        output.append(values)
    return output[0], output[1]


def _bootstrap_interval_summary(oriented_samples: np.ndarray) -> dict[str, Any]:
    require(
        oriented_samples.ndim == 1
        and oriented_samples.size == BOOTSTRAP_RESAMPLES,
        "Bootstrap statistic vector has an invalid shape",
    )
    require(np.isfinite(oriented_samples).all(), "Bootstrap statistics are non-finite")
    two_sided_alpha = 1.0 - TWO_SIDED_CONFIDENCE
    lower, upper = np.quantile(
        oriented_samples,
        [two_sided_alpha / 2.0, 1.0 - two_sided_alpha / 2.0],
    )
    one_sided_lower = np.quantile(
        oriented_samples,
        1.0 - ONE_SIDED_CONFIDENCE,
    )
    return {
        "bootstrap_90pct_two_sided_ci": {
            "lower": float(lower),
            "upper": float(upper),
        },
        "bootstrap_90pct_one_sided_lower_bound": float(one_sided_lower),
        "one_sided_lower_bound_above_zero": bool(one_sided_lower > 0.0),
    }


def bootstrap_mean_difference_summary(
    anchor_values: np.ndarray,
    candidate_values: np.ndarray,
    direction: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Bootstrap two nanmeans using the same resampled target multiplicities."""

    anchor = np.asarray(anchor_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    require(
        anchor.ndim == 1 and candidate.ndim == 1,
        "Bootstrap inputs must be vectors",
    )
    require(
        anchor.size == candidate.size and anchor.size > 0,
        "Bootstrap target axes must be nonempty and aligned",
    )
    require(direction in {"higher", "lower"}, "Unknown metric direction")
    require(
        not np.isinf(anchor).any() and not np.isinf(candidate).any(),
        "Bootstrap inputs contain infinity",
    )
    anchor_finite = np.isfinite(anchor)
    candidate_finite = np.isfinite(candidate)
    require(anchor_finite.any(), "Anchor has no finite targets on this cohort")
    require(candidate_finite.any(), "Candidate has no finite targets on this cohort")
    valid_raw_batches: list[np.ndarray] = []
    valid_statistics = 0
    attempted_resamples = 0
    maximum_attempts = BOOTSTRAP_RESAMPLES * 100
    while valid_statistics < BOOTSTRAP_RESAMPLES and attempted_resamples < maximum_attempts:
        remaining = BOOTSTRAP_RESAMPLES - valid_statistics
        batch_size = min(maximum_attempts - attempted_resamples, max(1024, remaining))
        indices = rng.integers(
            0,
            anchor.size,
            size=(batch_size, anchor.size),
            dtype=np.int32,
        )
        sampled_anchor = anchor[indices]
        sampled_candidate = candidate[indices]
        anchor_counts = np.isfinite(sampled_anchor).sum(axis=1)
        candidate_counts = np.isfinite(sampled_candidate).sum(axis=1)
        valid = (anchor_counts > 0) & (candidate_counts > 0)
        if valid.any():
            anchor_means = (
                np.nansum(sampled_anchor[valid], axis=1) / anchor_counts[valid]
            )
            candidate_means = (
                np.nansum(sampled_candidate[valid], axis=1) / candidate_counts[valid]
            )
            valid_raw_batches.append(candidate_means - anchor_means)
            valid_statistics += int(valid.sum())
        attempted_resamples += batch_size
    require(
        valid_statistics >= BOOTSTRAP_RESAMPLES,
        "Could not obtain enough bootstrap draws with finite nanmeans",
    )
    raw_samples = np.concatenate(valid_raw_batches)[:BOOTSTRAP_RESAMPLES]
    oriented_samples = raw_samples if direction == "higher" else -raw_samples
    anchor_mean = float(anchor[anchor_finite].mean())
    candidate_mean = float(candidate[candidate_finite].mean())
    raw_observed = candidate_mean - anchor_mean
    oriented_observed = raw_observed if direction == "higher" else -raw_observed
    summary = {
        "aggregation": "mean",
        "observed_anchor_mean": anchor_mean,
        "observed_candidate_mean": candidate_mean,
        "observed_mean_raw_candidate_minus_anchor": float(raw_observed),
        "observed_mean_oriented_candidate_minus_anchor": float(oriented_observed),
        "resampling_axis_targets": int(anchor.size),
        "anchor_finite_targets": int(anchor_finite.sum()),
        "candidate_finite_targets": int(candidate_finite.sum()),
        "anchor_missing_targets": int((~anchor_finite).sum()),
        "candidate_missing_targets": int((~candidate_finite).sum()),
        "finite_target_intersection": int((anchor_finite & candidate_finite).sum()),
        "finite_target_union": int((anchor_finite | candidate_finite).sum()),
        "finite_target_masks_identical": bool(
            np.array_equal(anchor_finite, candidate_finite)
        ),
        "bootstrap_draws_attempted": int(attempted_resamples),
        "bootstrap_draws_discarded_for_empty_nanmean": int(
            attempted_resamples - valid_statistics
        ),
    }
    summary.update(_bootstrap_interval_summary(oriented_samples))
    return summary


def bootstrap_mse_ratio_summary(
    anchor_numerator: np.ndarray,
    anchor_denominator: np.ndarray,
    candidate_numerator: np.ndarray,
    candidate_denominator: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Bootstrap the derived panel MSE as a paired ratio-of-sums statistic."""

    arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            anchor_numerator,
            anchor_denominator,
            candidate_numerator,
            candidate_denominator,
        )
    )
    size = arrays[0].size
    require(size > 0, "MSE ratio target axis is empty")
    require(
        all(values.ndim == 1 and values.size == size for values in arrays),
        "MSE ratio component axes are not aligned",
    )
    require(
        all(np.isfinite(values).all() for values in arrays),
        "MSE ratio components must remain all-finite",
    )
    anchor_num, anchor_den, candidate_num, candidate_den = arrays
    anchor_den_sum = float(anchor_den.sum())
    candidate_den_sum = float(candidate_den.sum())
    require(anchor_den_sum > 0.0, "Anchor MSE denominator sum is non-positive")
    require(candidate_den_sum > 0.0, "Candidate MSE denominator sum is non-positive")
    anchor_ratio = float(anchor_num.sum() / anchor_den_sum)
    candidate_ratio = float(candidate_num.sum() / candidate_den_sum)

    indices = rng.integers(
        0,
        size,
        size=(BOOTSTRAP_RESAMPLES, size),
        dtype=np.int32,
    )
    anchor_den_bootstrap = anchor_den[indices].sum(axis=1)
    candidate_den_bootstrap = candidate_den[indices].sum(axis=1)
    require(
        np.all(anchor_den_bootstrap > 0.0),
        "A bootstrap resample has a non-positive anchor MSE denominator",
    )
    require(
        np.all(candidate_den_bootstrap > 0.0),
        "A bootstrap resample has a non-positive candidate MSE denominator",
    )
    anchor_bootstrap = anchor_num[indices].sum(axis=1) / anchor_den_bootstrap
    candidate_bootstrap = (
        candidate_num[indices].sum(axis=1) / candidate_den_bootstrap
    )
    raw_samples = candidate_bootstrap - anchor_bootstrap
    oriented_samples = -raw_samples
    summary = {
        "aggregation": "ratio_of_sums",
        "numerator_metric": MSE_NUMERATOR_METRIC,
        "denominator_metric": MSE_DENOMINATOR_METRIC,
        "observed_anchor_ratio_of_sums": anchor_ratio,
        "observed_candidate_ratio_of_sums": candidate_ratio,
        "observed_raw_candidate_minus_anchor": float(candidate_ratio - anchor_ratio),
        "observed_oriented_candidate_minus_anchor": float(
            anchor_ratio - candidate_ratio
        ),
        "resampling_axis_targets": int(size),
        "anchor_numerator_sum": float(anchor_num.sum()),
        "anchor_denominator_sum": anchor_den_sum,
        "candidate_numerator_sum": float(candidate_num.sum()),
        "candidate_denominator_sum": candidate_den_sum,
    }
    summary.update(_bootstrap_interval_summary(oriented_samples))
    return summary


def _analyze_context(
    context: str,
    role: str,
    anchor_path: Path,
    candidate_path: Path,
    manifest: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_targets = frozenset(manifest["target_gene"])
    anchor = _read_scored_results(anchor_path, manifest_targets)
    candidate = _read_scored_results(candidate_path, manifest_targets)
    anchor_nmae = frozenset(
        anchor.loc[anchor["metric"] == NMAE_METRIC, "perturbation"]
    )
    candidate_nmae = frozenset(
        candidate.loc[candidate["metric"] == NMAE_METRIC, "perturbation"]
    )
    require(
        anchor_nmae == candidate_nmae,
        f"{context}: anchor and candidate NMAE omission subsets differ",
    )

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
    cohorts: dict[str, Any] = {}
    for cohort in COHORTS:
        cohort_targets = route_targets[cohort]
        require(bool(cohort_targets), f"{context}|{cohort}: cohort target axis is empty")
        metrics: dict[str, Any] = {}
        for metric, direction in METRIC_ORIENTATION.items():
            if metric == DERIVED_MSE_METRIC:
                anchor_num, candidate_num = _aligned_metric_values(
                    anchor,
                    candidate,
                    MSE_NUMERATOR_METRIC,
                    cohort_targets,
                )
                anchor_den, candidate_den = _aligned_metric_values(
                    anchor,
                    candidate,
                    MSE_DENOMINATOR_METRIC,
                    cohort_targets,
                )
                summary = bootstrap_mse_ratio_summary(
                    anchor_num,
                    anchor_den,
                    candidate_num,
                    candidate_den,
                    rng,
                )
                summary["omitted_cohort_targets"] = 0
            else:
                metric_targets = (
                    cohort_targets & anchor_nmae
                    if metric == NMAE_METRIC
                    else cohort_targets
                )
                require(
                    bool(metric_targets),
                    f"{context}|{cohort}|{metric}: resampling axis is empty",
                )
                anchor_values, candidate_values = _aligned_metric_values(
                    anchor,
                    candidate,
                    metric,
                    metric_targets,
                )
                summary = bootstrap_mean_difference_summary(
                    anchor_values,
                    candidate_values,
                    direction,
                    rng,
                )
                summary["omitted_cohort_targets"] = int(
                    len(cohort_targets) - len(metric_targets)
                )
            summary.update(
                {
                    "direction": direction,
                    "orientation_transform": (
                        "candidate_minus_anchor"
                        if direction == "higher"
                        else "negative_candidate_minus_anchor"
                    ),
                }
            )
            metrics[metric] = summary
        cohorts[cohort] = {
            "cohort_targets": int(len(cohort_targets)),
            "metrics": metrics,
        }

    context_report = {
        "role": role,
        "matched_nmae_targets": int(len(anchor_nmae)),
        "cohorts": cohorts,
    }
    provenance = {
        "anchor_results": _describe_file(anchor_path),
        "candidate_results": _describe_file(candidate_path),
    }
    return context_report, provenance


def main() -> None:
    args = parse_args()
    require(not args.output_json.exists(), f"Refusing to overwrite: {args.output_json}")
    manifest = _read_manifest(args.manifest)
    inputs = {
        "HepG2": (args.hepg2_anchor, args.hepg2_candidate),
        "Jurkat": (args.jurkat_anchor, args.jurkat_candidate),
    }
    all_input_paths = [path.resolve() for pair in inputs.values() for path in pair]
    require(
        len(set(all_input_paths)) == len(all_input_paths),
        "Anchor and candidate input paths must be distinct across contexts",
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    contexts: dict[str, Any] = {}
    scorer_provenance: dict[str, Any] = {}
    for context, role in CONTEXTS:
        anchor_path, candidate_path = inputs[context]
        contexts[context], scorer_provenance[context] = _analyze_context(
            context,
            role,
            anchor_path,
            candidate_path,
            manifest,
            rng,
        )

    report = {
        "schema": SCHEMA,
        "context_roles": {"HepG2": "primary", "Jurkat": "diagnostic"},
        "configuration": {
            "bootstrap_unit": "paired_target_axis",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "interval_method": "percentile",
            "two_sided_confidence": TWO_SIDED_CONFIDENCE,
            "one_sided_lower_confidence": ONE_SIDED_CONFIDENCE,
            "positive_oriented_difference_is_better": True,
            "nmae_population": "matched_finite_cell_eval2_subset_only",
            "nullable_metric_policy": (
                "same_resampled_target_multiplicities_with_independent_nanmeans"
            ),
            "derived_mse_aggregation": {
                "metric": DERIVED_MSE_METRIC,
                "formula": "sum(expr_mse_unbiased_capped)/sum(expr_distance_unbiased)",
                "per_target_derived_rows_expected": False,
            },
        },
        "metric_orientation": METRIC_ORIENTATION,
        "cohort_sizes": {
            "all": int(len(manifest)),
            "direct": int((manifest["validation_route"] == "direct").sum()),
            "held_target": int(
                (manifest["validation_route"] == "held_target").sum()
            ),
        },
        "contexts": contexts,
        "provenance": {
            "manifest": _describe_file(args.manifest),
            "scorer_results": scorer_provenance,
        },
        "leakage_firewall": {
            "reads_manifest": True,
            "reads_cell_eval2_results_csv_only": True,
            "reads_truth_h5ad": False,
            "reads_prediction_h5ad": False,
            "reads_control_h5ad": False,
        },
    }
    _atomic_write_json(args.output_json, report)
    print(json.dumps(report["contexts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
