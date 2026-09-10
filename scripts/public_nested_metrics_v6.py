"""Versioned numerical MMD safeguard for nested public validation.

The frozen v4 estimator and output decoder are unchanged. Only signed roundoff
in the mathematically nonnegative biased MMD is canonicalized, with the raw
per-group values and original aggregate retained for audit and exact replay.
"""
from __future__ import annotations

import math

import public_joint_target_source_v3 as v3
import public_sparse_hurdle_v4 as sparse

MMD_ROUNDOFF_TOLERANCE = 1e-12
MMD_KEYS = ("model_mmd2", "control_mmd2")


def policy():
    return {
        "schema": "public-nested-metrics-v6",
        "estimator": "frozen-v4-biased-rbf-mmd2",
        "negative_roundoff_tolerance": MMD_ROUNDOFF_TOLERANCE,
        "canonicalization": "negative values in [-tolerance, 0) become exactly zero",
        "nonfinite_or_more_negative": "reject before aggregation or scalar tracking",
        "raw_evidence": "original fold metrics, per-group raw MMD and nonnegative corrections",
        "aggregation": "frozen-v3 equal batch/target/source macro aggregation",
        "scientific_comparison_inequalities_changed": False,
    }


def canonical_mmd2(value):
    """Return a finite nonnegative float; do not excuse substantive negatives."""
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or value < -MMD_ROUNDOFF_TOLERANCE:
        raise ValueError("MMD must be finite and not below the registered roundoff tolerance")
    return 0.0 if value < 0 else float(value)


def canonicalize_report(original):
    """Copy a fresh frozen-v4 fold report and add traceable MMD corrections.

The original aggregate is evidence, not a validation scalar. All per-group
MMD values are validated before recomputing any new aggregate. Re-wrapping an
already canonicalized report is refused so original evidence cannot be lost.
"""
    if not isinstance(original, dict) or not isinstance(original.get("metrics"), dict):
        raise ValueError("Expected a frozen-v4 fold report with metrics")
    if "raw_metrics" in original or "mmd_roundoff_policy" in original:
        raise ValueError("Refusing to canonicalize an already wrapped report")
    if not isinstance(original.get("groups"), list) or not original["groups"]:
        raise ValueError("Expected nonempty group reports")
    if not isinstance(original.get("targets_report"), list) or not original["targets_report"]:
        raise ValueError("Expected nonempty target reports")
    groups = []
    for group in original["groups"]:
        if not isinstance(group, dict):
            raise ValueError("Expected group metric dictionaries")
        row = dict(group)
        for key in MMD_KEYS:
            raw_key, correction_key = "raw_" + key, key + "_roundoff_correction"
            if raw_key in row or correction_key in row:
                raise ValueError("Refusing to overwrite existing raw MMD evidence")
            canonical = canonical_mmd2(row.get(key))
            raw = float(row[key])
            row.update({raw_key: raw, key: canonical, correction_key: canonical - raw})
        groups.append(row)
    return {**original, "groups": groups, "raw_metrics": dict(original["metrics"]),
            "metrics": v3._aggregate(groups, original["targets_report"]),
            "mmd_roundoff_policy": policy()}


def evaluate_fold(arrays, fold, aligned, model, prior):
    """Drop-in evaluator for both registered inner and outer fold roles."""
    report, outputs = sparse.evaluate_fold(arrays, fold, aligned, model, prior)
    return canonicalize_report(report), outputs
