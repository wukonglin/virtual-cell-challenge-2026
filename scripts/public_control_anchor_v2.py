"""Opt-in public-only nonnegative control-anchored residual diagnostics.

Inputs must already be authenticated/row-authorized by the caller. This module
does not load data or prove its public provenance. Fitting accepts TRAINING group
effects only; prediction accepts target identities and matched-control summaries,
never recipient-treated values. Fixed penalties/gain are not development-tuned.
The explicit positive-part projection is a diagnostic output mapping, not a
count emitter or a learned cell-distribution generator. Legacy runs are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
import os
from pathlib import Path
import socket
from typing import Sequence

import numpy as np

from public_flow_conditioning import align_target_features, ablate_target_features
from public_flow_stable_conditioning import METHOD, fit_stable_target_transform
from public_mean_effect_baseline import balanced_weights, ridge_fit, weighted_standardize, save_arrays, delta_diagnostics
from train_public_flow_pilot import diagnostics, load_go_table, sha256_file, write_new

REPRESENTATION = "log1p_cp10k_shared_measured_genes"
ARMS = ("control", "context", "true", "shuffled")


def policy():
    return {"schema": "public-control-anchor-v2", "arms": list(ARMS),
        "representation": REPRESENTATION, "ridge_lambda": 10.0, "gain": 1.0,
        "shuffle_seed": 20260910, "target_transform": METHOD,
        "context_scale_floor": 0.1, "intercept": "penalized_zero_effect_prior",
        "group_weights": "uniform source then target then matched group",
        "prediction": "positive_part(matched_control + gain * predicted_delta)",
        "projection": "explicit; fraction, magnitude, mean distortion and raw diagnostics required",
        "null_condition": "NTC rows bypass all shifts/projection and are exact matched-control identity",
        "zero_effect": "zero-delta rows bypass shift/projection and preserve exact control values",
        "cross_validation": "whole source held out; overlapping targets only; no evaluation-source fitting",
        "fit_control_pool": "training-source anchor pool serves independent response reference; context pool is independent; held-source anchor is prediction/evaluation only; sham never fits",
        "evaluation_population": "all registered anchor and treated rows; identical across fixed arms",
        "empirical_null": "anchor versus independent sham NTC; deduplicated source and batch; hard gate is not learned null behavior",
        "shift_diagnostics": "projected and raw centroid shifts versus identical anchor/treated means; context-anchor discrepancy reported separately",
        "hyperparameter_selection": False, "count_emitter": False,
        "posttraining": False, "automatic_promotion": False}


def _strings(values: Sequence[str], name, *, empty=False, unique=False):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a string sequence")
    result = tuple(values)
    if (not result and not empty) or any(not isinstance(v, str) or not v or v.strip() != v for v in result):
        raise ValueError(f"Malformed {name}")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"Duplicate {name}")
    return result


def _matrix(value, name):
    value = np.asarray(value)
    if value.dtype.kind not in "iuf" or value.ndim != 2 or not value.size:
        raise ValueError(f"Expected finite numeric matrix: {name}")
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"Expected finite numeric matrix: {name}")
    return result


def _ntc(value, n):
    value = np.asarray(value)
    if value.dtype != np.bool_ or value.shape != (n,):
        raise ValueError("Explicit boolean NTC row mask required")
    return value


def _representation(value):
    if value != REPRESENTATION:
        raise ValueError("Only declared continuous log1p-CP10k shared-axis expression is supported; not raw counts")


@dataclass(frozen=True)
class AnchoredPrediction:
    raw: np.ndarray
    nonnegative: np.ndarray
    applied_delta: np.ndarray
    metrics: dict


def anchor_expression(control, delta, *, is_ntc, representation, gain=1.0):
    """Apply an explicit positive-part mapping; report every alteration.

Delta is a group vector [genes] or row-specific matrix [cells, genes]. Gain must
be finite in [0,1]. No library-size renormalization, exponentiation, rounding,
integer sampling, negative-delta threshold or hidden selection is performed.
"""
    _representation(representation)
    control = _matrix(control, "matched controls")
    if np.any(control < 0):
        raise ValueError("Matched controls must already be nonnegative log1p expression")
    ntc = _ntc(is_ntc, len(control))
    if type(gain) not in (int, float) or not np.isfinite(gain) or not 0 <= gain <= 1:
        raise ValueError("Finite gain in [0,1] required")
    delta = np.asarray(delta)
    if delta.dtype.kind not in "iuf" or delta.shape not in {(control.shape[1],), control.shape}:
        raise ValueError("Delta must align to genes or cells-by-genes")
    delta = np.broadcast_to(delta, control.shape).astype(np.float64, copy=True)
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite residual delta")
    delta[ntc] = 0.0
    delta *= gain
    active = np.any(delta != 0, axis=1)
    raw = control.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        raw[active] = control[active] + delta[active]
    if not np.isfinite(raw).all():
        raise ValueError("Residual arithmetic overflow; refusing nonfinite output")
    predicted = raw.copy()
    projected = raw < 0
    predicted[projected] = 0.0
    correction = predicted - raw
    mean_change = correction.mean(axis=0)
    active_elements = int(np.count_nonzero(~ntc) * control.shape[1])
    metrics = {
        "projection_fraction": float(projected.mean()),
        "projection_fraction_perturbed": float(projected[~ntc].mean()) if active_elements else 0.0,
        "projection_mean_absolute_correction": float(correction.mean()),
        "projection_max_absolute_correction": float(correction.max()),
        "projection_centroid_distortion_rms": float(np.sqrt(np.mean(mean_change ** 2))),
        "raw_negative_fraction": float(projected.mean()),
        "raw_negative_magnitude_mean": float(-raw[projected].mean()) if np.any(projected) else 0.0,
        "raw_negative_rms": float(np.sqrt(np.mean(np.square(np.minimum(raw, 0))))),
        "projected_negative_fraction": float(np.mean(predicted < 0)),
        "ntc_identity_verified": bool(np.array_equal(predicted[ntc], control[ntc])),
        "zero_effect_identity_verified": bool(np.array_equal(predicted[~active], control[~active])),
        "gain": float(gain), "projection_performed": bool(np.any(projected)),
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError("Projection diagnostic overflow")
    for array in (raw, predicted, delta):
        array.setflags(write=False)
    return AnchoredPrediction(raw, predicted, delta, metrics)


def paired_population_diagnostics(prediction, control, treated, *, representation):
    """Report both raw and projected outputs against identical populations.

Only call after fitting. Treated values are used for reporting, not fitting,
calibration, clipping choices or gain selection. Exact batch/source matching and
data-role authorization remain the caller's registered responsibility.
"""
    _representation(representation)
    if not isinstance(prediction, AnchoredPrediction):
        raise ValueError("Explicit anchored prediction required")
    control, treated = _matrix(control, "control population"), _matrix(treated, "treated population")
    if control.shape != prediction.raw.shape or treated.shape[1] != control.shape[1]:
        raise ValueError("Population expression axes disagree")
    if min(len(control), len(treated)) < 2 or np.any(control < 0) or np.any(treated < 0):
        raise ValueError("Finite nonnegative log-expression populations with >=2 cells required")
    return {"raw": diagnostics(prediction.raw, control, treated),
            "nonnegative": diagnostics(prediction.nonnegative, control, treated),
            "projection": dict(prediction.metrics), "used_for_fitting_or_selection": False}


@dataclass(frozen=True)
class SourceFold:
    held_source: str
    fit_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    excluded_missing_target_indices: tuple[int, ...]


def source_holdout_folds(targets, sources):
    targets, sources = _strings(targets, "targets"), _strings(sources, "sources")
    if len(targets) != len(sources) or len(set(sources)) < 2:
        raise ValueError("At least two aligned public sources required for whole-source validation")
    folds = []
    for source in sorted(set(sources)):
        fit = tuple(i for i, s in enumerate(sources) if s != source)
        available = {targets[i] for i in fit}
        admitted = tuple(i for i, s in enumerate(sources) if s == source and targets[i] in available)
        excluded = tuple(i for i, s in enumerate(sources) if s == source and targets[i] not in available)
        folds.append(SourceFold(source, fit, admitted, excluded))
    return tuple(folds)


@dataclass(frozen=True)
class ResidualDecoder:
    arm: str
    training_targets: tuple[str, ...]
    training_sources: tuple[str, ...]
    fold_id: str
    context_mean: np.ndarray
    context_scale: np.ndarray
    coefficients: np.ndarray
    target_features: np.ndarray
    target_provenance: dict

    def predict_delta(self, targets, control_context, *, is_ntc):
        targets = _strings(targets, "prediction targets")
        contexts = _matrix(control_context, "control-derived context")
        ntc = _ntc(is_ntc, len(targets))
        if contexts.shape != (len(targets), len(self.context_mean)):
            raise ValueError("Control context does not match the trained axis")
        lookup = {target: i for i, target in enumerate(self.training_targets)}
        if any(target not in lookup for target, null in zip(targets, ntc) if not null):
            raise ValueError("This pilot requires overlapping training targets; NTC is separately identified")
        result = np.zeros((len(targets), self.coefficients.shape[1]))
        active = np.flatnonzero(~ntc)
        if not len(active) or self.arm == "control":
            return result
        context = (contexts[active] - self.context_mean) / self.context_scale / math.sqrt(len(self.context_mean))
        pieces = [np.ones((len(active), 1))]
        if self.arm in {"true", "shuffled"}:
            pieces.append(self.target_features[[lookup[targets[i]] for i in active]])
        pieces.append(context)
        result[active] = np.column_stack(pieces) @ self.coefficients
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite predicted residual")
        return result


def fit_residual_decoder(targets, sources, control_context, mean_delta, *, table,
                         arm, fold_id, permitted_training_sources, heldout_sources,
                         excluded_targets, representation):
    """Fit only caller-authorized public group effects, never recipient effects."""
    _representation(representation)
    targets, sources = _strings(targets, "training targets"), _strings(sources, "training sources")
    permitted = _strings(permitted_training_sources, "permitted training sources", unique=True)
    heldout = _strings(heldout_sources, "held-out sources", empty=True, unique=True)
    excluded = _strings(excluded_targets, "excluded targets", empty=True, unique=True)
    if not isinstance(fold_id, str) or not fold_id or fold_id.strip() != fold_id:
        raise ValueError("Explicit registered fold id required")
    if arm not in ARMS or len(targets) != len(sources):
        raise ValueError("Invalid arm or aligned group axes")
    if not set(sources) <= set(permitted) or set(sources) & set(heldout) or set(targets) & set(excluded):
        raise ValueError("Unauthorized training source or held-out/excluded training data")
    context, response = _matrix(control_context, "training control context"), _matrix(mean_delta, "training mean effects")
    if len(context) != len(targets) or len(response) != len(targets):
        raise ValueError("Training group axis mismatch")
    weights = balanced_weights(targets, sources)
    mean, scale = weighted_standardize(context, weights)
    xt = (context - mean) / scale / math.sqrt(context.shape[1])
    unique = tuple(sorted(set(targets)))
    features, provenance = np.empty((len(unique), 0)), {}
    if arm in {"true", "shuffled"}:
        aligned = align_target_features(unique, [table])
        if arm == "shuffled":
            aligned = ablate_target_features(aligned, mode="shuffle", seed=20260910,
                                              train_ids=unique, fold_id=fold_id)
        transform = fit_stable_target_transform(aligned, train_ids=unique,
                                                excluded_ids=excluded, fold_id=fold_id)
        features = np.column_stack((transform.transform(aligned), aligned.present, aligned.unknown_target))
        features /= math.sqrt(features.shape[1])
        index = {target: i for i, target in enumerate(unique)}
        xt = np.column_stack((features[[index[t] for t in targets]], xt))
        provenance = transform.provenance()
    design = np.column_stack((np.ones(len(targets)), xt))
    coefficients = np.zeros((design.shape[1], response.shape[1])) if arm == "control" else ridge_fit(design, response, weights)
    for array in (mean, scale, coefficients, features):
        array.setflags(write=False)
    return ResidualDecoder(arm, unique, tuple(sorted(set(sources))), fold_id,
                           mean, scale, coefficients, features, provenance)


def aggregate_group_metrics(groups):
    """Source→target→batch macro; no independent-cell confidence claims."""
    if not groups:
        raise ValueError("No evaluated groups")
    metadata = {"group", "source", "target", "batch", "cells", "anchor_cells", "treated_cells", "ntc_identity_verified",
                "zero_effect_identity_verified", "projection_performed"}
    result = {}
    for key in sorted(set(groups[0]) - metadata):
        per_source = []
        for source in sorted({g["source"] for g in groups}):
            per_target = []
            for target in sorted({g["target"] for g in groups if g["source"] == source}):
                values = [g[key] for g in groups if (g["source"], g["target"]) == (source, target)
                          and g[key] is not None]
                if values:
                    per_target.append(float(np.mean(values)))
            if per_target:
                per_source.append(float(np.mean(per_target)))
        result[key] = float(np.mean(per_source)) if per_source else None
    return result


def run_validation(arrays, table, arm, output_dir, *, excluded_targets, provenance):
    """Two fixed source-held-out fits on an authenticated public-only v2 cache.

The CLI authenticates and validates the cache before this function. This callable
also supports synthetic tests; it does not independently establish provenance.
No hyperparameters/arm/projection are selected; every registered arm is retained.
"""
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    targets = _strings(tuple(arrays["targets"].astype(str)), "cache targets")
    sources = _strings(tuple(arrays["sources"].astype(str)), "cache sources")
    batches = _strings(tuple(arrays["batches"].astype(str)), "cache batches")
    excluded = _strings(excluded_targets, "excluded targets", empty=True, unique=True)
    if len(batches) != len(targets) or set(targets) & set(excluded) or len(set(sources)) != 2:
        raise ValueError("Expected two authorized public sources and no excluded targets")
    folds = source_holdout_folds(targets, sources)
    if any(fold.excluded_missing_target_indices or not fold.evaluation_indices for fold in folds):
        raise ValueError("Registered cohort must be fully common across both sources")
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh diagnostic output directory required")
    output_dir.mkdir(mode=0o700)
    weights, predictions, reports, all_groups, step_events = {}, {}, [], [], []
    hard_null_verified = True
    zero_identity_verified = True
    for number, fold in enumerate(folds):
        prefix = f"fold_{number}_"
        fit = np.asarray(fold.fit_indices, dtype=np.int64)
        held = np.asarray(fold.evaluation_indices, dtype=np.int64)
        fitting_sources = tuple(sorted({sources[i] for i in fit}))
        response = []
        for group in fit:
            # The training source's anchor pool is an independent response
            # reference. Using context controls here would mathematically couple
            # context noise to the negative response noise under the null.
            # No HELD-source anchor, treated or sham values enter this fit.
            treated = arrays["treated"][arrays["group"] == group].astype(np.float64)
            reference = arrays["control"][arrays["control_group"] == group].astype(np.float64)
            response.append(treated.mean(0) - reference.mean(0))
        model = fit_residual_decoder(tuple(targets[i] for i in fit), tuple(sources[i] for i in fit),
            arrays["context"][fit], np.asarray(response), table=table, arm=arm,
            fold_id=provenance["diagnostic_contract_sha256"] + ":held:" + fold.held_source,
            permitted_training_sources=fitting_sources, heldout_sources=(fold.held_source,),
            excluded_targets=excluded, representation=REPRESENTATION)
        held_delta = model.predict_delta(tuple(targets[i] for i in held), arrays["context"][held],
                                         is_ntc=np.zeros(len(held), dtype=bool))
        weights.update({prefix + key: value for key, value in {
            "coefficients": model.coefficients, "context_mean": model.context_mean,
            "context_scale": model.context_scale, "target_features": model.target_features,
            "target_ids": np.asarray(model.training_targets), "training_sources": np.asarray(model.training_sources),
        }.items()})
        groups, raw_rows, positive_rows, prediction_groups, control_rows, treated_rows, treated_groups = [], [], [], [], [], [], []
        nulls, seen_batches = [], set()
        for local, group in enumerate(held):
            ci = np.flatnonzero(arrays["control_group"] == group)
            ti = np.flatnonzero(arrays["group"] == group)
            control, treated = arrays["control"][ci], arrays["treated"][ti]
            prediction = anchor_expression(control, held_delta[local], is_ntc=np.zeros(len(ci), dtype=bool),
                                            representation=REPRESENTATION, gain=0.0 if arm == "control" else 1.0)
            zero_prediction = anchor_expression(control, np.zeros(control.shape[1]),
                is_ntc=np.zeros(len(ci), dtype=bool), representation=REPRESENTATION)
            zero_identity_verified = zero_identity_verified and np.array_equal(zero_prediction.nonnegative, control)
            paired = paired_population_diagnostics(prediction, control, treated, representation=REPRESENTATION)
            control_mean, treated_mean = control.astype(np.float64).mean(0), treated.astype(np.float64).mean(0)
            observed_shift = treated_mean - control_mean
            projected_shift = delta_diagnostics(prediction.nonnegative.mean(0) - control_mean, observed_shift)
            raw_shift = delta_diagnostics(prediction.raw.mean(0) - control_mean, observed_shift)
            context_mean = arrays["context_control"][arrays["context_control_group"] == group].astype(np.float64).mean(0)
            row = {"group": int(group), "source": sources[group], "target": targets[group],
                "batch": batches[group], "cells": len(ci), "anchor_cells": len(ci), "treated_cells": len(ti),
                **paired["nonnegative"], **paired["projection"],
                "raw_model_centroid_mse": paired["raw"]["model_centroid_mse"],
                "raw_model_mmd2": paired["raw"]["model_mmd2"], **projected_shift,
                "raw_predicted_shift_rms": raw_shift["predicted_shift_rms"],
                "raw_shift_cosine": raw_shift["shift_cosine"], "raw_shift_dot_mean": raw_shift["shift_dot_mean"],
                "context_anchor_centroid_mse": float(np.mean((context_mean - control_mean) ** 2))}
            groups.append(row)
            raw_rows.append(prediction.raw)
            positive_rows.append(prediction.nonnegative)
            prediction_groups.extend([int(group)] * len(ci))
            control_rows.extend(ci)
            treated_rows.extend(ti)
            treated_groups.extend([int(group)] * len(ti))
            batch_key = sources[group], batches[group]
            if batch_key not in seen_batches:
                seen_batches.add(batch_key)
                si = np.flatnonzero(arrays["sham_control_group"] == group)
                null_delta = model.predict_delta(("non-targeting",), arrays["context"][group:group + 1],
                                                 is_ntc=np.array([True]))[0]
                null_prediction = anchor_expression(control, null_delta, is_ntc=np.ones(len(ci), dtype=bool),
                                                     representation=REPRESENTATION)
                null = paired_population_diagnostics(null_prediction, control, arrays["sham_control"][si],
                                                       representation=REPRESENTATION)
                identity = np.array_equal(null_prediction.nonnegative, control)
                hard_null_verified = hard_null_verified and identity
                nulls.append({"source": sources[group], "batch": batches[group], "anchor_cells": len(ci),
                    "sham_cells": len(si), "hard_ntc_identity_verified": bool(identity),
                    "ntc_sham_centroid_mse": null["nonnegative"]["model_centroid_mse"],
                    "ntc_sham_mmd2": null["nonnegative"]["model_mmd2"]})
        predictions.update({prefix + key: value for key, value in {
            "raw": np.concatenate(raw_rows), "nonnegative": np.concatenate(positive_rows),
            "delta": held_delta, "group": np.asarray(prediction_groups, dtype=np.int64),
            "control_cache_rows": np.asarray(control_rows, dtype=np.int64),
            "treated_cache_rows": np.asarray(treated_rows, dtype=np.int64),
            "treated_group": np.asarray(treated_groups, dtype=np.int64),
            "held_groups": held, "held_source": np.asarray(fold.held_source),
        }.items()})
        aggregate = aggregate_group_metrics(groups)
        aggregate.update({key: float(np.mean([null[key] for null in nulls]))
                          for key in ("ntc_sham_centroid_mse", "ntc_sham_mmd2")})
        report = {"held_source": fold.held_source, "train_sources": list(fitting_sources),
            "array_prefix": prefix, "fit_group_indices": list(fold.fit_indices),
            "evaluation_group_indices": list(fold.evaluation_indices), "target_count": len(set(model.training_targets)),
            "groups": groups, "aggregate": aggregate, "ntc_sham_groups": nulls,
            "target_transform": model.target_provenance, "hyperparameter_selection": False}
        reports.append(report)
        all_groups.extend(groups)
        event = {"step": number + 1, "outer_fold": number,
                 "deployment_rollout_diagnostics": {"kind_metrics": {"context": aggregate}}}
        step_events.append(event)
        print(json.dumps(event, allow_nan=False), flush=True)
    weights["genes"], predictions["genes"] = arrays["genes"], arrays["genes"]
    weights_sha = save_arrays(output_dir / "weights.npz", weights)
    predictions_sha = save_arrays(output_dir / "predictions.npz", predictions)
    aggregate = aggregate_group_metrics(all_groups)
    aggregate.update({key: float(np.mean([report["aggregate"][key] for report in reports]))
                      for key in ("ntc_sham_centroid_mse", "ntc_sham_mmd2")})
    summary = {"schema": "public-control-anchor-validation-summary-v2", "status": "completed_unpromoted_diagnostic",
        "stage": "validation", "arm": arm, "last_step": len(folds), "policy": policy(),
        "provenance": provenance, "diagnostic_contract_sha256": provenance["diagnostic_contract_sha256"],
        "contract_sha256": provenance["contract_sha256"], "source_manifest_sha256": provenance["source_manifest_sha256"],
        "weights_sha256": weights_sha, "predictions_sha256": predictions_sha, "outer_folds": reports,
        "deployment_rollout_diagnostics": {"kind_metrics": {"context": aggregate}},
        "count_emitter": False, "dev_selection": False, "development_used_for_fit_or_selection": False,
        "H1_treated_read": False, "RPE1_treated_read": False,
        "exact_ntc_identity_verified": bool(hard_null_verified), "learned_null_calibration_evaluated": False,
        "exact_zero_effect_identity_verified": bool(zero_identity_verified),
        "posttraining_performed": False, "promoted": False, "submission_performed": False,
        "zero_shot_scope": "same target, held source/context; not unseen-target prediction",
        "runtime": {"hostname": socket.gethostname(), "numpy_version": np.__version__, "device": "cpu"}}
    write_new(output_dir / "steps.jsonl", b"".join((json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode() for event in step_events))
    write_new(output_dir / "summary.json", (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("cache", "target-features", "feature-receipt", "contract", "source-manifest")
    for name in (*names, "diagnostic-contract", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (*names, "diagnostic-contract"):
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--stage", choices=("validation",), default="validation")
    args = parser.parse_args(argv)
    if socket.gethostname().split(".")[0] not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
        raise RuntimeError("This bounded CPU diagnostic requires an approved BioHPC compute host; never a head node")
    from run_public_control_anchor_suite import validate_diagnostic_contract
    from prepare_public_control_anchor_v2 import load_cache
    inputs = {name.replace("-", "_") + "_sha256": getattr(args, name.replace("-", "_") + "_sha256") for name in names}
    validate_diagnostic_contract(args.diagnostic_contract, args.diagnostic_contract_sha256, inputs)
    arrays, contract = load_cache(args.cache, args.cache_sha256, args.contract, args.contract_sha256,
                                  args.source_manifest, args.source_manifest_sha256)
    table = load_go_table(args.target_features, args.target_features_sha256, args.feature_receipt,
                          args.feature_receipt_sha256, tuple(arrays["targets"].astype(str)))
    run_validation(arrays, table, args.arm, args.output_dir, excluded_targets=contract["excluded_targets"],
        provenance={**inputs, "diagnostic_contract_sha256": args.diagnostic_contract_sha256,
                    "entrypoint_sha256": sha256_file(Path(__file__))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
