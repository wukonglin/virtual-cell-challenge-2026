#!/usr/bin/env python3
"""Compare scorer-aligned public validation runs against a fixed STATE anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generate_state_direct_counts import atomic_write_json
from infer_state_effect_prior import describe_file, require


SCHEMA = "vcc-public-validation-comparison-v1"
NMAE_METRIC = "de_wilcoxon_lfc_nmae"
FIDELITY_METRIC = "de_wilcoxon_direction_fidelity_yield_raw"
REACH_METRIC = "de_wilcoxon_direction_reach_raw"
MSE_METRIC = "expr_mse_unbiased_capped_norm"
MSE_NUMERATOR_METRIC = "expr_mse_unbiased_capped"
MSE_DENOMINATOR_METRIC = "expr_distance_unbiased"
SCORED_METRICS = {
    "pds_cosine": "higher",
    MSE_METRIC: "lower",
    NMAE_METRIC: "lower",
    FIDELITY_METRIC: "higher",
    REACH_METRIC: "higher",
    "de_wilcoxon_sig_jaccard": "higher",
}
EMITTED_METRICS = (
    (set(SCORED_METRICS) - {MSE_METRIC})
    | {MSE_NUMERATOR_METRIC, MSE_DENOMINATOR_METRIC}
)
COMPLETE_EMITTED_METRICS = EMITTED_METRICS - {NMAE_METRIC}
NAN_TOLERANT_METRICS = {FIDELITY_METRIC, REACH_METRIC}
COHORT_OPTIONAL_METRICS = {NMAE_METRIC, FIDELITY_METRIC, REACH_METRIC}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-label", type=str, required=True)
    parser.add_argument(
        "--primary-context",
        type=str,
        required=True,
        help="Held-out context used for strict candidate promotion.",
    )
    parser.add_argument(
        "--non-harm-context",
        type=str,
        required=True,
        help="Training-seen context used only for fail-closed non-regression.",
    )
    parser.add_argument(
        "--entry",
        action="append",
        nargs=3,
        metavar=("LABEL", "CONTEXT", "RESULTS_CSV"),
        required=True,
        help="Repeat once for each candidate/context cell-eval2 results.csv.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    return parser.parse_args()


def _read_results(path: Path, targets: set[str]) -> pd.DataFrame:
    require(path.is_file(), f"Missing scorer results: {path}")
    frame = pd.read_csv(path)
    required = {"perturbation", "metric", "value"}
    require(required.issubset(frame.columns), f"Bad scorer table: {path}")
    frame = frame.loc[frame["metric"].astype(str).isin(EMITTED_METRICS)].copy()
    frame["perturbation"] = frame["perturbation"].astype(str)
    frame["metric"] = frame["metric"].astype(str)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    require(set(frame["perturbation"]) == targets, "Scorer target set differs from manifest")
    duplicates = frame.duplicated(["perturbation", "metric"])
    require(not duplicates.any(), "Duplicate target/metric scorer rows")
    expected_complete_pairs = {
        (target, metric) for target in targets for metric in COMPLETE_EMITTED_METRICS
    }
    observed_pairs = set(
        frame[["perturbation", "metric"]].itertuples(index=False, name=None)
    )
    observed_complete_pairs = {
        pair for pair in observed_pairs if pair[1] in COMPLETE_EMITTED_METRICS
    }
    require(
        observed_complete_pairs == expected_complete_pairs,
        "Scorer results lack full target coverage for a required complete metric",
    )
    nmae_targets = set(
        frame.loc[frame["metric"] == NMAE_METRIC, "perturbation"].astype(str)
    )
    require(bool(nmae_targets), "NMAE has no scored targets")
    require(nmae_targets.issubset(targets), "NMAE contains a target outside the manifest")
    require(
        len(frame) == len(expected_complete_pairs) + len(nmae_targets),
        "Scorer results have an invalid scored target/metric row count",
    )
    for metric in EMITTED_METRICS:
        values = frame.loc[frame["metric"] == metric, "value"].to_numpy(dtype=np.float64)
        require(not np.isinf(values).any(), f"Scorer metric contains infinity: {metric}")
        if metric in NAN_TOLERANT_METRICS:
            require(np.isfinite(values).any(), f"Scorer metric has no finite values: {metric}")
        else:
            require(np.isfinite(values).all(), f"Scorer metric is non-finite: {metric}")
    return frame


def _mean_summary(
    frame: pd.DataFrame, expected_targets: int
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    numerator = frame.loc[
        frame["metric"] == MSE_NUMERATOR_METRIC, "value"
    ].to_numpy(dtype=np.float64)
    denominator = frame.loc[
        frame["metric"] == MSE_DENOMINATOR_METRIC, "value"
    ].to_numpy(dtype=np.float64)
    require(
        numerator.size == denominator.size == expected_targets,
        "MSE component axes do not cover the cohort",
    )
    require(
        np.isfinite(numerator).all() and np.isfinite(denominator).all(),
        "MSE components contain a non-finite value",
    )
    numerator_sum = float(numerator.sum(dtype=np.float64))
    denominator_sum = float(denominator.sum(dtype=np.float64))
    require(
        np.isfinite(numerator_sum)
        and np.isfinite(denominator_sum)
        and denominator_sum > 0.0,
        "MSE ratio-of-sums denominator is not positive and finite",
    )
    output[MSE_METRIC] = {
        "mean": float(numerator_sum / denominator_sum),
        "finite_targets": int(expected_targets),
        "missing_targets": 0,
        "numerator_sum": numerator_sum,
        "denominator_sum": denominator_sum,
        "aggregation": "ratio_of_sums",
    }
    for metric in set(SCORED_METRICS) - {MSE_METRIC}:
        values = frame.loc[frame["metric"] == metric, "value"].to_numpy(dtype=np.float64)
        emitted_targets = int(values.size)
        finite = values[np.isfinite(values)]
        if finite.size == 0 and metric in COHORT_OPTIONAL_METRICS:
            output[metric] = {
                "mean": None,
                "finite_targets": 0,
                "missing_targets": int(expected_targets),
                "emitted_targets": emitted_targets,
                "aggregation": "finite_mean",
            }
            continue
        require(values.size > 0, f"Metric has no values: {metric}")
        require(not np.isinf(values).any(), f"Metric contains infinity: {metric}")
        if metric not in COHORT_OPTIONAL_METRICS:
            require(
                values.size == expected_targets and finite.size == expected_targets,
                f"Required finite metric has missing targets: {metric}",
            )
        elif metric != NMAE_METRIC:
            require(values.size == expected_targets, f"Metric axis is incomplete: {metric}")
        output[metric] = {
            "mean": float(finite.mean()),
            "finite_targets": int(finite.size),
            "missing_targets": int(expected_targets - finite.size),
            "emitted_targets": emitted_targets,
            "aggregation": "finite_mean",
        }
    return output


def _validate_cell_eval_sidecars(
    results_path: Path,
    frame: pd.DataFrame,
    all_summary: dict[str, dict[str, Any]],
    expected_targets: int,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    def as_bool(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        normalized = str(value).strip().lower()
        require(normalized in {"true", "false"}, f"Invalid derived flag: {value}")
        return normalized == "true"

    aggregate_path = results_path.with_name("agg_results.csv")
    aggregation_path = results_path.with_name("metric_aggregation.csv")
    require(aggregate_path.is_file(), f"Missing aggregate scorer table: {aggregate_path}")
    require(aggregation_path.is_file(), f"Missing metric aggregation table: {aggregation_path}")

    aggregation = pd.read_csv(aggregation_path)
    aggregation_columns = {
        "metric", "agg", "n_used", "n_rows", "n_nan", "n_null", "derived"
    }
    require(
        aggregation_columns.issubset(aggregation.columns),
        "Bad metric_aggregation.csv schema",
    )
    aggregation["metric"] = aggregation["metric"].astype(str)
    relevant = aggregation.loc[
        aggregation["metric"].isin(EMITTED_METRICS | {MSE_METRIC})
    ].copy()
    require(relevant["metric"].is_unique, "Duplicate metric aggregation rows")
    require(
        set(relevant["metric"]) == EMITTED_METRICS | {MSE_METRIC},
        "Metric aggregation table lacks a required metric",
    )
    relevant = relevant.set_index("metric")
    for metric in EMITTED_METRICS:
        values = frame.loc[frame["metric"] == metric, "value"].to_numpy(dtype=np.float64)
        row = relevant.loc[metric]
        require(str(row["agg"]) == "mean", f"Unexpected aggregation for {metric}")
        require(not as_bool(row["derived"]), f"Raw metric marked derived: {metric}")
        require(int(row["n_rows"]) == values.size, f"n_rows mismatch for {metric}")
        require(
            int(row["n_used"]) == int(np.isfinite(values).sum()),
            f"n_used mismatch for {metric}",
        )
        require(
            int(row["n_nan"]) + int(row["n_null"])
            == int((~np.isfinite(values)).sum()),
            f"missing-value accounting mismatch for {metric}",
        )
    derived = relevant.loc[MSE_METRIC]
    require(str(derived["agg"]) == "ratio_of_sums", "MSE is not ratio_of_sums")
    require(as_bool(derived["derived"]), "MSE metric is not marked derived")
    require(
        all(int(derived[column]) == 0 for column in ("n_used", "n_rows", "n_nan", "n_null")),
        "Derived MSE aggregation counters are not zero",
    )

    aggregate = pd.read_csv(aggregate_path)
    require("statistic" in aggregate.columns, "Bad agg_results.csv schema")
    mean_rows = aggregate.loc[aggregate["statistic"].astype(str) == "mean"]
    require(len(mean_rows) == 1, "agg_results.csv does not contain one mean row")
    mean_row = mean_rows.iloc[0]
    for metric in SCORED_METRICS:
        require(metric in aggregate.columns, f"Aggregate scorer table lacks {metric}")
        expected = all_summary[metric]["mean"]
        require(expected is not None, f"All-context metric is not evaluable: {metric}")
        observed = float(mean_row[metric])
        require(np.isfinite(observed), f"Aggregate scorer metric is non-finite: {metric}")
        require(
            bool(np.isclose(observed, float(expected), rtol=tolerance, atol=tolerance)),
            f"Aggregate scorer mean differs from authenticated components: {metric}",
        )
    return {
        "aggregate_results": describe_file(aggregate_path),
        "metric_aggregation": describe_file(aggregation_path),
        "validated_all_context_means": {
            metric: float(all_summary[metric]["mean"]) for metric in SCORED_METRICS
        },
        "expected_targets": int(expected_targets),
    }


def oriented_improvement(metric: str, candidate: float, baseline: float) -> float:
    require(metric in SCORED_METRICS, f"Unknown scored metric: {metric}")
    return candidate - baseline if SCORED_METRICS[metric] == "higher" else baseline - candidate


def promotion_gate(
    improvements: dict[str, float],
    tolerance: float,
    not_evaluable_metrics: tuple[str, ...] = (),
) -> dict[str, Any]:
    require(
        set(not_evaluable_metrics).issubset(COHORT_OPTIONAL_METRICS),
        "A required cohort metric is non-evaluable",
    )
    mse = improvements["expr_mse_unbiased_capped_norm"] >= -tolerance
    nmae = (
        None
        if NMAE_METRIC in not_evaluable_metrics
        else improvements[NMAE_METRIC] >= -tolerance
    )
    reach = (
        None
        if REACH_METRIC in not_evaluable_metrics
        else improvements[REACH_METRIC] >= -tolerance
    )
    discrimination = (
        improvements["pds_cosine"] > tolerance
        or improvements["de_wilcoxon_sig_jaccard"] > tolerance
    )
    return {
        "mse_not_worse": bool(mse),
        "nmae_not_worse": None if nmae is None else bool(nmae),
        "reach_not_worse": None if reach is None else bool(reach),
        "pds_or_jaccard_improves": bool(discrimination),
        "not_evaluable_metrics": list(not_evaluable_metrics),
        "passed": bool(
            mse
            and (nmae is None or nmae)
            and (reach is None or reach)
            and discrimination
        ),
    }


def non_regression_gate(
    improvements: dict[str, float],
    tolerance: float,
    not_evaluable_metrics: tuple[str, ...] = (),
) -> dict[str, Any]:
    require(
        set(not_evaluable_metrics).issubset(COHORT_OPTIONAL_METRICS),
        "A required cohort metric is non-evaluable",
    )
    checks = {
        metric: bool(improvements[metric] >= -tolerance)
        for metric in SCORED_METRICS
        if metric not in not_evaluable_metrics
    }
    return {
        "metric_not_worse": checks,
        "not_evaluable_metrics": list(not_evaluable_metrics),
        "passed": bool(all(checks.values())),
    }


def main() -> None:
    args = parse_args()
    require(args.manifest.is_file(), f"Missing manifest: {args.manifest}")
    require(args.absolute_tolerance >= 0, "absolute-tolerance cannot be negative")
    require(
        args.primary_context != args.non_harm_context,
        "Primary and non-harm contexts must differ",
    )
    require(not args.output_json.exists(), f"Refusing to overwrite: {args.output_json}")
    manifest = pd.read_csv(args.manifest)
    require(
        {"target_gene", "validation_route"}.issubset(manifest.columns),
        "Manifest lacks route columns",
    )
    manifest["target_gene"] = manifest["target_gene"].astype(str)
    manifest["validation_route"] = manifest["validation_route"].astype(str)
    require(manifest["target_gene"].is_unique, "Manifest target axis has duplicates")
    route_by_target = manifest.set_index("target_gene")["validation_route"].astype(str)
    require(
        set(route_by_target.index.astype(str))
        == set(manifest["target_gene"].astype(str)),
        "Bad route index",
    )
    require(set(route_by_target.values) == {"direct", "held_target"}, "Bad routes")
    target_set = set(route_by_target.index.astype(str))

    entries: dict[str, dict[str, dict[str, Any]]] = {}
    nmae_targets_by_entry: dict[tuple[str, str], frozenset[str]] = {}
    provenance: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for label, context, path_string in args.entry:
        key = (str(label), str(context))
        require(key not in seen, f"Duplicate entry: {key}")
        seen.add(key)
        path = Path(path_string)
        frame = _read_results(path, target_set)
        frame["route"] = frame["perturbation"].map(route_by_target)
        require(not frame["route"].isna().any(), "A scorer target lacks a validation route")
        nmae_targets_by_entry[key] = frozenset(
            frame.loc[frame["metric"] == NMAE_METRIC, "perturbation"].astype(str)
        )
        direct_targets = int((route_by_target == "direct").sum())
        held_targets = int((route_by_target == "held_target").sum())
        all_summary = _mean_summary(frame, len(target_set))
        sidecar_validation = _validate_cell_eval_sidecars(
            path, frame, all_summary, len(target_set)
        )
        cohorts = {
            "all": all_summary,
            "direct": _mean_summary(
                frame.loc[frame["route"] == "direct"], direct_targets
            ),
            "held_target": _mean_summary(
                frame.loc[frame["route"] == "held_target"], held_targets
            ),
        }
        entries.setdefault(str(label), {})[str(context)] = cohorts
        provenance.append(
            {
                "label": str(label),
                "context": str(context),
                "results": describe_file(path),
                **sidecar_validation,
            }
        )

    require(args.baseline_label in entries, "Baseline label is absent")
    contexts = sorted(entries[args.baseline_label])
    require(bool(contexts), "Baseline has no contexts")
    require(
        set(contexts) == {args.primary_context, args.non_harm_context},
        "Scored contexts do not exactly match the configured context roles",
    )
    for label, by_context in entries.items():
        require(sorted(by_context) == contexts, f"{label}: context set differs from baseline")
        for context in contexts:
            require(
                nmae_targets_by_entry[(label, context)]
                == nmae_targets_by_entry[(args.baseline_label, context)],
                f"{label}|{context}: NMAE target subset differs from baseline",
            )

    comparisons: dict[str, Any] = {}
    for label, by_context in entries.items():
        if label == args.baseline_label:
            continue
        comparisons[label] = {}
        context_passes: list[bool] = []
        for context in contexts:
            comparisons[label][context] = {}
            for cohort in ("all", "direct", "held_target"):
                baseline = entries[args.baseline_label][context][cohort]
                candidate = by_context[context][cohort]
                not_evaluable_metrics: list[str] = []
                improvements: dict[str, float] = {}
                for metric in SCORED_METRICS:
                    baseline_mean = baseline[metric]["mean"]
                    candidate_mean = candidate[metric]["mean"]
                    require(
                        (baseline_mean is None) == (candidate_mean is None),
                        f"{label}|{context}|{cohort}: metric evaluability differs",
                    )
                    if baseline_mean is None:
                        require(
                            metric in COHORT_OPTIONAL_METRICS,
                            "A required cohort metric is non-evaluable",
                        )
                        not_evaluable_metrics.append(metric)
                        continue
                    improvements[metric] = oriented_improvement(
                        metric,
                        float(candidate_mean),
                        float(baseline_mean),
                    )
                not_evaluable = tuple(not_evaluable_metrics)
                promotion = promotion_gate(
                    improvements, args.absolute_tolerance, not_evaluable
                )
                non_regression = non_regression_gate(
                    improvements, args.absolute_tolerance, not_evaluable
                )
                comparisons[label][context][cohort] = {
                    "oriented_improvement_positive_is_better": improvements,
                    "not_evaluable_metrics": list(not_evaluable),
                    "promotion_gate": promotion,
                    "non_regression_gate": non_regression,
                }
            if context == args.primary_context:
                all_passed = bool(
                    comparisons[label][context]["all"]["promotion_gate"]["passed"]
                )
                direct_passed = bool(
                    comparisons[label][context]["direct"]["promotion_gate"]["passed"]
                )
                held_passed = bool(
                    comparisons[label][context]["held_target"][
                        "non_regression_gate"
                    ]["passed"]
                )
                context_gate = {
                    "role": "primary",
                    "all_full_promotion": all_passed,
                    "direct_full_promotion": direct_passed,
                    "held_target_all_evaluable_metric_non_regression": held_passed,
                    "passed": bool(all_passed and direct_passed and held_passed),
                }
            else:
                require(
                    context == args.non_harm_context,
                    f"Unexpected unassigned context: {context}",
                )
                all_passed = bool(
                    comparisons[label][context]["all"]["non_regression_gate"][
                        "passed"
                    ]
                )
                direct_passed = bool(
                    comparisons[label][context]["direct"]["non_regression_gate"][
                        "passed"
                    ]
                )
                held_passed = bool(
                    comparisons[label][context]["held_target"][
                        "non_regression_gate"
                    ]["passed"]
                )
                context_gate = {
                    "role": "non_harm",
                    "all_evaluable_metric_non_regression": all_passed,
                    "direct_evaluable_metric_non_regression": direct_passed,
                    "held_target_all_evaluable_metric_non_regression": held_passed,
                    "strict_discrimination_improvement_required": False,
                    "passed": bool(all_passed and direct_passed and held_passed),
                }
            comparisons[label][context]["experiment_gate"] = context_gate
            context_passes.append(context_gate["passed"])
        comparisons[label]["aggregate_gate"] = {
            "contexts_passed": int(sum(context_passes)),
            "contexts_total": len(context_passes),
            "passed_all_contexts": bool(all(context_passes)),
        }

    report = {
        "schema": SCHEMA,
        "baseline_label": args.baseline_label,
        "context_roles": {
            "primary": args.primary_context,
            "non_harm": args.non_harm_context,
        },
        "metric_orientation": SCORED_METRICS,
        "configuration": {"absolute_tolerance": args.absolute_tolerance},
        "cohort_sizes": {
            "all": int(len(manifest)),
            "direct": int((manifest["validation_route"] == "direct").sum()),
            "held_target": int((manifest["validation_route"] == "held_target").sum()),
        },
        "entries": entries,
        "comparisons": comparisons,
        "provenance": {
            "manifest": describe_file(args.manifest),
            "scorer_results": provenance,
        },
        "contract": {
            "metrics_are_not_combined_across_incommensurate_units": True,
            "promotion_requires_expression_and_reach_non_regression": True,
            "promotion_requires_pds_or_jaccard_improvement": True,
            "primary_context_requires_all_and_direct_full_promotion": True,
            "primary_context_requires_held_target_evaluable_metric_non_regression": True,
            "non_harm_context_requires_all_cohort_evaluable_metric_non_regression": True,
            "non_harm_context_requires_direct_cohort_evaluable_metric_non_regression": True,
            "non_harm_context_requires_held_target_evaluable_metric_non_regression": True,
            "non_harm_context_requires_strict_discrimination_improvement": False,
            "cohort_local_nmae_may_be_not_evaluable": True,
            "cohort_local_direction_metrics_may_be_not_evaluable": True,
            "mse_aggregation": "ratio_of_sums",
            "all_context_means_match_cell_eval2_aggregate": True,
        },
    }
    atomic_write_json(args.output_json, report)
    print(json.dumps(report["comparisons"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
