"""Explicit-grid kernel ridge with typed outer/inner validation roles.

No data loading, experiment execution, regularization selection or monkeypatching.
The v5 regularization-independent kernel geometry and predictor are reused
read-only. This version makes lambda an explicit, serialized caller decision,
rejects normalization overflow and strengthens row-count and restore dtype checks.
Callers must authenticate exact group labels, expression rows and GO vocabularies;
membership checks here cannot authenticate externally supplied expression labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

import public_target_context_kernel_v5 as v5
import public_joint_target_source_v3 as v3
from public_control_anchor_v2 import _strings, _matrix, _representation, REPRESENTATION
from public_flow_conditioning import AlignedTargets
from public_flow_stable_conditioning import fit_stable_target_transform
from public_mean_effect_baseline import balanced_weights, weighted_standardize

REGULARIZATION_GRID = (0.1, 1.0, 10.0)
ARMS = v5.ARMS
SHUFFLE_SEEDS = v5.SHUFFLE_SEEDS
feature_arm = v5.feature_arm
kernel_matrix = v5.kernel_matrix
_target_features = v5._target_features
MODEL_KEYS = v5.MODEL_KEYS + ("regularization", "validation_kind", "outer_fold_id")
TRANSFORM_KEYS = v5.TRANSFORM_KEYS


def policy():
    base = v5.policy()
    del base["ridge_lambda"]
    return {**base, "schema": "public-target-context-kernel-v6",
        "regularization_grid": list(REGULARIZATION_GRID),
        "regularization_binding": "explicit exact grid value in fit, restore and diagnostic calls; serialized float64",
        "regularization_selection": "caller-owned preregistered inner validation only; this module does not select",
        "dual_system": "(sqrtW K sqrtW+regularization*I) A=sqrtW response; saved dual_coefficients=sqrtW A",
        "training_diagnostics": "weighted head RMS; effective_df=sum eig(S)/(eig(S)+regularization), kernel_trace=trace(S)",
        "validation_kinds": {"outer_joint_target_source": "ordinary JointFold with distinct fitting and held sources",
            "inner_same_source_target": "exact InnerTargetFold with same source and explicit nonempty outer_fold_id"},
        "fit_alignment": "group count equals distinct nonnegative fold.fit_indices; exact declared target roster and source; disjoint held roles",
        "normalization": "reject nonfinite means/scales before division; context scale must remain >=0.1; no silent infinite-scale collapse",
        "restore_dtypes": "real model arrays must match float64 dtype and shape exactly; textual arrays must be Unicode, never object/structured"}


def _regularization(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("Explicit numeric regularization from the registered fixed grid required")
    result = float(value)
    if not math.isfinite(result) or result not in REGULARIZATION_GRID:
        raise ValueError("Regularization must equal one of 0.1, 1.0, 10.0 exactly")
    return result


def decoder_keys(arm):
    feature_arm(arm)
    return MODEL_KEYS + (() if arm in {"control", "context"} else TRANSFORM_KEYS)


@dataclass(frozen=True)
class InnerTargetFold(v3.JointFold):
    """Same-source target-held-out calibration with explicit outer ownership."""
    outer_fold_id: str


def _indices(values, name, *, empty=False):
    if not isinstance(values, tuple) or (not values and not empty):
        raise ValueError(f"Explicit tuple of group indices required: {name}")
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or i < 0 for i in values):
        raise ValueError(f"Nonnegative integer group indices required: {name}")
    if len(set(map(int, values))) != len(values): raise ValueError(f"Duplicate group index: {name}")
    return tuple(map(int, values))


def _scope(fold):
    if type(fold) not in {v3.JointFold, InnerTargetFold}:
        raise ValueError("Only an ordinary outer JointFold or explicit InnerTargetFold is permitted")
    _strings((fold.fold_id, fold.target_fold, fold.fit_source, fold.held_source), "fold labels")
    if type(fold) is InnerTargetFold:
        _strings((fold.outer_fold_id,), "outer fold ownership")
        if fold.fit_source != fold.held_source:
            raise ValueError("InnerTargetFold requires honest same-source calibration labels")
        return "inner_same_source_target", fold.outer_fold_id
    if fold.fit_source == fold.held_source:
        raise ValueError("Ordinary outer JointFold requires distinct sources; use explicit InnerTargetFold")
    return "outer_joint_target_source", ""


def _admit(targets, sources, fold, arm):
    feature_arm(arm)
    kind, outer = _scope(fold)
    targets, sources = _strings(targets, "training targets"), _strings(sources, "training sources")
    fit_targets = _strings(fold.fit_targets, "fitting target roster", unique=True)
    held_targets = _strings(fold.held_targets, "held target roster", unique=True)
    fit = _indices(fold.fit_indices, "fit_indices")
    evaluated = _indices(fold.evaluation_indices, "evaluation_indices")
    held_all = _indices(fold.held_target_indices_all_sources, "held_target_indices_all_sources")
    other = _indices(fold.other_excluded_indices, "other_excluded_indices", empty=True)
    if len(targets) != len(sources) or len(targets) != len(fit):
        raise ValueError("Fitting rows must match fold.fit_indices exactly in length")
    if (set(fit_targets) & set(held_targets) or set(targets) != set(fit_targets)
        or set(sources) != {fold.fit_source} or set(targets) & set(held_targets)
        or set(fit) & set(held_all) or not set(evaluated) <= set(held_all)
        or set(other) & (set(fit) | set(held_all))):
        raise ValueError("Unauthorized target/source/group exclusion role in kernel fit")
    return targets, sources, kind, outer


def _training_design(targets, sources, context, response, *, aligned, arm, fold, representation):
    _representation(representation)
    targets, sources, kind, outer = _admit(targets, sources, fold, arm)
    if not isinstance(aligned, AlignedTargets): raise ValueError("Explicit fold-local target annotations required")
    context, response = _matrix(context, "training control context"), _matrix(response, "training two-head response")
    if len(context) != len(targets) or len(response) != len(targets) or response.shape[1] % 2:
        raise ValueError("Aligned training groups and two equal-width response heads required")
    weights = balanced_weights(targets, sources)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean, scale = weighted_standardize(context, weights)
    if (mean.shape != (context.shape[1],) or scale.shape != mean.shape or not np.isfinite(mean).all()
        or not np.isfinite(scale).all() or np.any(scale < 0.1)):
        raise ValueError("Nonfinite or invalid context mean/scale; refusing normalization collapse")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        c = (context-mean)/scale/math.sqrt(context.shape[1])
    if not np.isfinite(c).all(): raise ValueError("Nonfinite training context normalization")
    transform, z = None, np.empty((len(targets), 0))
    if arm not in {"control", "context"}:
        transform = fit_stable_target_transform(aligned, train_ids=tuple(sorted(fold.fit_targets)),
            excluded_ids=fold.held_targets, fold_id=fold.fold_id)
        if not np.isfinite(transform.mean).all() or not np.isfinite(transform.scale).all() or np.any(transform.scale <= 0):
            raise ValueError("Nonfinite or invalid target normalization")
        unique = v3._subset(aligned, tuple(sorted(fold.fit_targets)))
        features = _target_features(transform, unique)
        lookup = {t: i for i, t in enumerate(unique.target_ids)}
        z = features[[lookup[t] for t in targets]]
    return targets, sources, response, weights, mean, scale, c, z, transform, kind, outer


@dataclass(frozen=True)
class KernelDecoder(v5.KernelDecoder):
    regularization: float
    validation_kind: str
    outer_fold_id: str


def _diagnostics(c, z, weights, coefficients, response, arm, regularization):
    kernel = kernel_matrix(c, c, z, z, arm=arm)
    sw = np.sqrt(weights)
    weighted = sw[:, None]*kernel*sw[None, :]
    if not np.isfinite(weighted).all(): raise ValueError("Nonfinite weighted kernel")
    if arm == "control":
        effective_df, trace, fitted = 0.0, 0.0, np.zeros_like(response)
    else:
        eigenvalues = np.linalg.eigvalsh(weighted)
        tolerance = 1e-10*(1+float(np.max(np.abs(weighted))))
        if not np.isfinite(eigenvalues).all() or eigenvalues.min() < -tolerance:
            raise ValueError("Weighted polynomial kernel is not numerically positive semidefinite")
        effective_df = float(np.sum(eigenvalues/(eigenvalues+regularization)))
        trace = float(np.trace(weighted))
        fitted = kernel @ coefficients
    width = response.shape[1]//2
    result = {"effective_df": effective_df, "kernel_trace": trace}
    for name, sl in (("occupancy", slice(0, width)), ("amplitude", slice(width, None))):
        for label, values in (("response", response), ("fitted", fitted)):
            with np.errstate(over="ignore", invalid="ignore"):
                value = float(np.sqrt(np.sum(weights*np.mean(values[:, sl]**2, axis=1))))
            if not math.isfinite(value): raise ValueError("Nonfinite training head RMS")
            result[f"{name}_{label}_rms"] = value
    if not all(math.isfinite(value) for value in result.values()): raise ValueError("Nonfinite kernel diagnostics")
    return result


def _model(arm, fold, targets, sources, mean, scale, c, z, weights, coefficients, transform, aligned, response,
           metrics, regularization, kind, outer):
    return KernelDecoder(arm, fold.fold_id, tuple(sorted(fold.fit_targets)), fold.fit_source, fold.held_source,
        targets, sources, v5._read_only(mean), v5._read_only(scale), v5._read_only(c), v5._read_only(z),
        v5._read_only(weights), v5._read_only(coefficients), transform, aligned.signature,
        v5._response_hash(response), tuple(sorted(metrics.items())), regularization, kind, outer)


def fit_kernel_decoder(targets, sources, context, response, *, aligned, arm, fold, regularization, representation=REPRESENTATION):
    """Explicit admitted lambda; selection belongs to the authenticated caller."""
    regularization = _regularization(regularization)
    targets, sources, y, weights, mean, scale, c, z, transform, kind, outer = _training_design(
        targets, sources, context, response, aligned=aligned, arm=arm, fold=fold, representation=representation)
    if arm == "control": coefficients = np.zeros_like(y)
    else:
        kernel = kernel_matrix(c, c, z, z, arm=arm)
        sw = np.sqrt(weights)
        system = sw[:, None]*kernel*sw[None, :] + regularization*np.eye(len(weights))
        if not np.isfinite(system).all(): raise ValueError("Nonfinite weighted ridge system")
        try: weighted_dual = np.linalg.solve(system, sw[:, None]*y)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Registered kernel ridge system failed; no fallback lambda") from exc
        coefficients = sw[:, None]*weighted_dual
    if not np.isfinite(coefficients).all(): raise ValueError("Nonfinite dual coefficients")
    metrics = _diagnostics(c, z, weights, coefficients, y, arm, regularization)
    return _model(arm, fold, targets, sources, mean, scale, c, z, weights, coefficients, transform, aligned, y,
                  metrics, regularization, kind, outer)


def training_diagnostics(model, response, *, regularization):
    regularization = _regularization(regularization)
    if type(model) is not KernelDecoder or model.regularization != regularization:
        raise ValueError("Diagnostics require the exact admitted model and regularization")
    response = _matrix(response, "training response for diagnostics")
    if v5._response_hash(response) != model.training_response_sha256:
        raise ValueError("Diagnostics response differs from admitted training data")
    return dict(model.training_metrics)


def decoder_arrays(model):
    if type(model) is not KernelDecoder: raise ValueError("Explicit v6 kernel decoder required")
    _regularization(model.regularization)
    result = v5.decoder_arrays(model)
    result.update(regularization=np.asarray(model.regularization, dtype=np.float64),
        validation_kind=np.asarray(model.validation_kind), outer_fold_id=np.asarray(model.outer_fold_id))
    if set(result) != set(decoder_keys(model.arm)): raise ValueError("Incomplete v6 kernel state")
    return result


def _same_array(actual, expected, key):
    if not isinstance(actual, np.ndarray) or actual.dtype.hasobject or actual.dtype.fields is not None:
        raise ValueError(f"Unsafe saved array dtype: {key}")
    expected = np.asarray(expected)
    if actual.shape != expected.shape: raise ValueError(f"Saved array shape differs: {key}")
    if expected.dtype.kind == "U":
        if actual.dtype.kind != "U": raise ValueError(f"Saved metadata must be Unicode: {key}")
    elif actual.dtype != expected.dtype:
        raise ValueError(f"Saved numeric dtype differs: {key}")
    if not np.array_equal(actual, expected): raise ValueError(f"Saved kernel state mismatch: {key}")


def restore_kernel_decoder(saved, *, arm, fold, aligned, training_targets, training_sources, context, response, regularization):
    """No solve/refit; lambda/scope/dtypes, training state and dual residual bind."""
    regularization = _regularization(regularization)
    if not isinstance(saved, dict) or set(saved) != set(decoder_keys(arm)):
        raise ValueError("Exact v6 kernel state keys required")
    kind, outer = _scope(fold)
    _same_array(saved["regularization"], np.asarray(regularization, dtype=np.float64), "regularization")
    _same_array(saved["validation_kind"], np.asarray(kind), "validation_kind")
    _same_array(saved["outer_fold_id"], np.asarray(outer), "outer_fold_id")
    targets, sources, y, weights, mean, scale, c, z, transform, kind, outer = _training_design(
        training_targets, training_sources, context, response, aligned=aligned, arm=arm, fold=fold, representation=REPRESENTATION)
    raw = saved["dual_coefficients"]
    if not isinstance(raw, np.ndarray) or raw.dtype != np.dtype(np.float64) or raw.shape != y.shape or not np.isfinite(raw).all():
        raise ValueError("Saved dual coefficients require finite float64 and exact shape")
    coefficients = raw.copy()
    if arm == "control":
        if np.any(coefficients != 0): raise ValueError("Control coefficients must be exactly zero")
    else:
        kernel = kernel_matrix(c, c, z, z, arm=arm); sw = np.sqrt(weights)
        system = sw[:, None]*kernel*sw[None, :] + regularization*np.eye(len(weights))
        with np.errstate(over="ignore", invalid="ignore"):
            lhs = system @ (coefficients/sw[:, None]); rhs = sw[:, None]*y
        tolerance = 1e-10*(1+float(np.max(np.abs(rhs))))
        if not np.isfinite(lhs).all() or np.max(np.abs(lhs-rhs)) > tolerance:
            raise ValueError("Saved coefficients fail the admitted-lambda dual residual")
    metrics = _diagnostics(c, z, weights, coefficients, y, arm, regularization)
    model = _model(arm, fold, targets, sources, mean, scale, c, z, weights, coefficients, transform, aligned, y,
                   metrics, regularization, kind, outer)
    for key, expected in decoder_arrays(model).items(): _same_array(saved[key], expected, key)
    return model
