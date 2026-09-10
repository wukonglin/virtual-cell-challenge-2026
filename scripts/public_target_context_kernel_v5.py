"""Fixed dual kernel ridge for the public sparse-hurdle response heads.

No file/data loading or experiment execution. Callers authenticate public rows,
fold-local GO vocabulary and role manifests before passing TRAINING arrays.
The explicit polynomial feature map is used only in synthetic tests, never
materialized here. Saved-state restoration checks the prescribed dual system
without solving another ridge fit. V3/v4 implementation bytes remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

import public_joint_target_source_v3 as v3
from public_control_anchor_v2 import _strings, _matrix, _ntc, _representation, REPRESENTATION
from public_flow_conditioning import AlignedTargets
from public_flow_stable_conditioning import METHOD, fit_stable_target_transform
from public_mean_effect_baseline import balanced_weights, weighted_standardize

RIDGE_LAMBDA = 10.0
SHUFFLE_SEEDS = (20260921, 20260922, 20260923)
ARMS = ("control", "context", "additive", "true", *(f"shuffled_{s}" for s in SHUFFLE_SEEDS))
MODEL_KEYS = ("dual_coefficients", "training_context_features", "training_target_features", "training_weights",
    "training_group_targets", "training_group_sources", "training_targets", "fit_source", "held_source",
    "fold_id", "arm", "context_mean", "context_scale", "feature_signature_json", "training_response_sha256",
    "training_diagnostics_json")
TRANSFORM_KEYS = ("target_mean", "target_scale", "target_transform_provenance_json")


def policy():
    return {"schema": "public-target-context-kernel-v5", "arms": list(ARMS), "ridge_lambda": RIDGE_LAMBDA,
        "representation": REPRESENTATION, "target_transform": METHOD, "context_scale_floor": 0.1,
        "c": "v3 weighted train-standardized control context divided by sqrt(context width)",
        "z": "v3 stable train-standardized GO concatenated with present and unknown masks, divided by sqrt(total target width)",
        "kernels": {"control": "zero predictor, no solve", "context": "1+C+C*C",
            "additive": "1+C+C*C+T", "true_and_shuffles": "1+C+C*C+T+T*C"},
        "C": "c_left @ c_right.T", "T": "z_left @ z_right.T", "products": "Hadamard products",
        "constant": "penalized, same lambda as all other implicit feature coefficients",
        "dual_system": "(sqrtW K sqrtW+10I) A=sqrtW response; saved dual_coefficients=sqrtW A",
        "prediction": "K(test,train) @ dual_coefficients; explicit NTC and control return exact zeros",
        "weights": "equal source, then target, then matched training group",
        "feature_arms": "additive uses the exact true GO packets; three shuffled arms reuse v3 role-contained paired value/mask permutations",
        "target_source_roles": "held-target treated excluded in both sources; only fitting source and fitting targets are admitted",
        "training_diagnostics": "weighted head RMS; effective_df=sum eig(S)/(eig(S)+10), kernel_trace=trace(S), S=sqrtW K sqrtW",
        "kernel_eigenvalues": "signed floating estimates retained; non-PSD beyond norm-scaled numerical tolerance is rejected",
        "restore_tolerance": "dual residual maxabs <= 1e-10*(1+maxabs(sqrtW response))",
        "capacity_matching_claimed": False, "held_source_extrapolation": True,
        "hyperparameter_selection": False, "count_emitter": False, "automatic_promotion": False}


def feature_arm(arm):
    if arm not in ARMS: raise ValueError("Unknown registered kernel arm")
    return "true" if arm == "additive" else arm


def decoder_keys(arm):
    feature_arm(arm)
    return MODEL_KEYS + (() if arm in {"control", "context"} else TRANSFORM_KEYS)


def _needs_target(arm):
    return arm not in {"control", "context"}


def _response_hash(response):
    value = np.ascontiguousarray(response, dtype="<f8")
    digest = hashlib.sha256(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _read_only(value):
    value = np.asarray(value).copy()
    value.setflags(write=False)
    return value


def _target_features(transform, aligned):
    result = np.column_stack((transform.transform(aligned), aligned.present, aligned.unknown_target))
    result /= math.sqrt(result.shape[1])
    if not np.isfinite(result).all(): raise ValueError("Nonfinite target design")
    return result


def kernel_matrix(context_left, context_right, target_left=None, target_right=None, *, arm):
    """Finite exact polynomial cross-kernel; negative off-diagonals are allowed."""
    feature_arm(arm)
    left, right = _matrix(context_left, "left context design"), _matrix(context_right, "right context design")
    if left.shape[1] != right.shape[1]: raise ValueError("Context feature widths disagree")
    if _needs_target(arm):
        zl, zr = _matrix(target_left, "left target design"), _matrix(target_right, "right target design")
        if len(zl) != len(left) or len(zr) != len(right) or zl.shape[1] != zr.shape[1]:
            raise ValueError("Target/context kernel axes disagree")
    elif target_left is not None or target_right is not None:
        # Empty target designs are the explicit serialization convention for
        # control/context arms. Refuse silently ignored nonempty target inputs.
        if any(np.asarray(z).shape != (n, 0) for z, n in ((target_left, len(left)), (target_right, len(right)))):
            raise ValueError("No target feature block belongs to control/context kernel")
    if arm == "control": return np.zeros((len(left), len(right)))
    with np.errstate(over="ignore", invalid="ignore"):
        c = left @ right.T
        result = 1 + c + c*c
        if _needs_target(arm):
            t = zl @ zr.T
            result += t
            if arm != "additive": result += t*c
    if not np.isfinite(result).all(): raise ValueError("Kernel arithmetic overflow")
    return result


def _admit(targets, sources, fold, arm):
    feature_arm(arm)
    targets, sources = _strings(targets, "training targets"), _strings(sources, "training sources")
    if not isinstance(fold, v3.JointFold) or len(targets) != len(sources):
        raise ValueError("Explicit registered joint target/source fold required")
    if (set(fold.fit_targets) & set(fold.held_targets) or fold.fit_source == fold.held_source
        or set(targets) != set(fold.fit_targets) or set(sources) != {fold.fit_source}
        or set(targets) & set(fold.held_targets)
        or set(fold.fit_indices) & set(fold.held_target_indices_all_sources)):
        raise ValueError("Unauthorized target/source/exclusion role in kernel fitting arrays")
    return targets, sources


def _training_design(targets, sources, context, response, *, aligned, arm, fold, representation):
    _representation(representation)
    targets, sources = _admit(targets, sources, fold, arm)
    if not isinstance(aligned, AlignedTargets): raise ValueError("Explicit fold-local target annotations required")
    context, response = _matrix(context, "training control context"), _matrix(response, "training two-head response")
    if len(context) != len(targets) or len(response) != len(targets) or response.shape[1] % 2:
        raise ValueError("Aligned training groups and two equal-width response heads required")
    weights = balanced_weights(targets, sources)
    mean, scale = weighted_standardize(context, weights)
    with np.errstate(over="ignore", invalid="ignore"):
        c = (context-mean)/scale/math.sqrt(context.shape[1])
    if not np.isfinite(c).all(): raise ValueError("Nonfinite training context normalization")
    transform, z = None, np.empty((len(targets), 0))
    if _needs_target(arm):
        transform = fit_stable_target_transform(aligned, train_ids=tuple(sorted(fold.fit_targets)),
            excluded_ids=fold.held_targets, fold_id=fold.fold_id)
        unique = v3._subset(aligned, tuple(sorted(fold.fit_targets)))
        features = _target_features(transform, unique)
        lookup = {t: i for i, t in enumerate(unique.target_ids)}
        z = features[[lookup[t] for t in targets]]
    return targets, sources, response, weights, mean, scale, c, z, transform


@dataclass(frozen=True)
class KernelDecoder:
    arm: str
    fold_id: str
    training_targets: tuple[str, ...]
    training_source: str
    held_source: str
    training_group_targets: tuple[str, ...]
    training_group_sources: tuple[str, ...]
    context_mean: np.ndarray
    context_scale: np.ndarray
    training_context_features: np.ndarray
    training_target_features: np.ndarray
    training_weights: np.ndarray
    dual_coefficients: np.ndarray
    target_transform: object | None
    feature_signature: tuple
    training_response_sha256: str
    training_metrics: tuple[tuple[str, float], ...]

    def predict_delta(self, aligned, control_context, *, is_ntc):
        if not isinstance(aligned, AlignedTargets): raise ValueError("Explicit aligned target annotations required")
        context = _matrix(control_context, "prediction control context")
        ntc = _ntc(is_ntc, len(aligned.target_ids))
        if context.shape != (len(aligned.target_ids), len(self.context_mean)):
            raise ValueError("Prediction target/context axes disagree")
        result = np.zeros((len(context), self.dual_coefficients.shape[1]))
        if self.arm == "control" or np.all(ntc): return result
        with np.errstate(over="ignore", invalid="ignore"):
            c = (context-self.context_mean)/self.context_scale/math.sqrt(context.shape[1])
        if not np.isfinite(c).all(): raise ValueError("Prediction context normalization overflow")
        z = _target_features(self.target_transform, aligned) if self.target_transform is not None else np.empty((len(context), 0))
        kernel = kernel_matrix(c, self.training_context_features, z, self.training_target_features, arm=self.arm)
        with np.errstate(over="ignore", invalid="ignore"):
            result = kernel @ self.dual_coefficients
        result[ntc] = 0
        if not np.isfinite(result).all(): raise ValueError("Kernel prediction overflow")
        return result


def _diagnostics(c, z, weights, coefficients, response, arm):
    kernel = kernel_matrix(c, c, z, z, arm=arm)
    sw = np.sqrt(weights)
    weighted = sw[:, None]*kernel*sw[None, :]
    if arm == "control":
        effective_df, trace, fitted = 0.0, 0.0, np.zeros_like(response)
    else:
        eigenvalues = np.linalg.eigvalsh(weighted)
        tolerance = 1e-10*(1+float(np.max(np.abs(weighted))))
        if not np.isfinite(eigenvalues).all() or eigenvalues.min() < -tolerance:
            raise ValueError("Weighted polynomial kernel is not numerically positive semidefinite")
        effective_df = float(np.sum(eigenvalues/(eigenvalues+RIDGE_LAMBDA)))
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
    if not all(math.isfinite(v) for v in result.values()): raise ValueError("Nonfinite kernel diagnostics")
    return result


def fit_kernel_decoder(targets, sources, context, response, *, aligned, arm, fold, representation=REPRESENTATION):
    """Fit only explicitly admitted public training groups, with fixed lambda10."""
    targets, sources, y, weights, mean, scale, c, z, transform = _training_design(
        targets, sources, context, response, aligned=aligned, arm=arm, fold=fold, representation=representation)
    if arm == "control":
        coefficients = np.zeros_like(y)
    else:
        kernel = kernel_matrix(c, c, z, z, arm=arm)
        sw = np.sqrt(weights)
        system = sw[:, None]*kernel*sw[None, :] + RIDGE_LAMBDA*np.eye(len(weights))
        if not np.isfinite(system).all(): raise ValueError("Nonfinite weighted ridge system")
        try:
            weighted_dual = np.linalg.solve(system, sw[:, None]*y)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Fixed kernel ridge system could not be solved; no adaptive regularization") from exc
        coefficients = sw[:, None]*weighted_dual
    if not np.isfinite(coefficients).all(): raise ValueError("Nonfinite dual coefficients")
    metrics = _diagnostics(c, z, weights, coefficients, y, arm)
    return KernelDecoder(arm, fold.fold_id, tuple(sorted(fold.fit_targets)), fold.fit_source, fold.held_source,
        targets, sources, _read_only(mean), _read_only(scale), _read_only(c), _read_only(z), _read_only(weights),
        _read_only(coefficients), transform, aligned.signature, _response_hash(y), tuple(sorted(metrics.items())))


def training_diagnostics(model, response):
    """Only the exact admitted training response can request these diagnostics."""
    if not isinstance(model, KernelDecoder): raise ValueError("Kernel decoder required")
    response = _matrix(response, "training response for diagnostics")
    if _response_hash(response) != model.training_response_sha256:
        raise ValueError("Diagnostics response differs from the admitted training response")
    return dict(model.training_metrics)


def decoder_arrays(model):
    """Safe, complete model-only serialization; no pickle or implicit response."""
    if not isinstance(model, KernelDecoder): raise ValueError("Kernel decoder required")
    result = {"dual_coefficients": model.dual_coefficients, "training_context_features": model.training_context_features,
        "training_target_features": model.training_target_features, "training_weights": model.training_weights,
        "training_group_targets": np.asarray(model.training_group_targets), "training_group_sources": np.asarray(model.training_group_sources),
        "training_targets": np.asarray(model.training_targets), "fit_source": np.asarray(model.training_source),
        "held_source": np.asarray(model.held_source), "fold_id": np.asarray(model.fold_id), "arm": np.asarray(model.arm),
        "context_mean": model.context_mean, "context_scale": model.context_scale,
        "feature_signature_json": np.asarray(json.dumps(model.feature_signature)),
        "training_response_sha256": np.asarray(model.training_response_sha256),
        "training_diagnostics_json": np.asarray(json.dumps(dict(model.training_metrics), sort_keys=True))}
    if model.target_transform:
        result.update(target_mean=model.target_transform.mean, target_scale=model.target_transform.scale,
            target_transform_provenance_json=np.asarray(json.dumps(model.target_transform.provenance(), sort_keys=True)))
    if set(result) != set(decoder_keys(model.arm)) or any(np.asarray(v).dtype.hasobject for v in result.values()):
        raise ValueError("Unsafe or incomplete kernel state")
    return result


def restore_kernel_decoder(saved, *, arm, fold, aligned, training_targets, training_sources, context, response):
    """Recompute train-only normalization and verify the saved dual system, no fit.

Caller authenticates the containing NPZ/hash and passes this exact unprefixed key
dictionary. Dual residuals are norm-scaled; all other state arrays are exact.
"""
    if not isinstance(saved, dict) or set(saved) != set(decoder_keys(arm)):
        raise ValueError("Exact kernel decoder state keys required")
    targets, sources, y, weights, mean, scale, c, z, transform = _training_design(training_targets, training_sources,
        context, response, aligned=aligned, arm=arm, fold=fold, representation=REPRESENTATION)
    coefficients = _matrix(saved["dual_coefficients"], "saved dual coefficients")
    if coefficients.shape != y.shape: raise ValueError("Saved dual coefficient dimensions differ")
    if arm == "control":
        if np.any(coefficients != 0): raise ValueError("Control dual coefficients must be exactly zero")
    else:
        kernel = kernel_matrix(c, c, z, z, arm=arm)
        sw = np.sqrt(weights)
        system = sw[:, None]*kernel*sw[None, :] + RIDGE_LAMBDA*np.eye(len(weights))
        with np.errstate(over="ignore", invalid="ignore"):
            lhs = system @ (coefficients/sw[:, None]); rhs = sw[:, None]*y
        tolerance = 1e-10*(1+float(np.max(np.abs(rhs))))
        if not np.isfinite(lhs).all() or np.max(np.abs(lhs-rhs)) > tolerance:
            raise ValueError("Saved dual coefficients fail fixed-ridge system residual")
    metrics = _diagnostics(c, z, weights, coefficients, y, arm)
    model = KernelDecoder(arm, fold.fold_id, tuple(sorted(fold.fit_targets)), fold.fit_source, fold.held_source,
        targets, sources, _read_only(mean), _read_only(scale), _read_only(c), _read_only(z), _read_only(weights),
        _read_only(coefficients), transform, aligned.signature, _response_hash(y), tuple(sorted(metrics.items())))
    expected = decoder_arrays(model)
    for key, value in expected.items():
        if not np.array_equal(saved[key], value): raise ValueError(f"Saved kernel normalization/provenance mismatch: {key}")
    return model
