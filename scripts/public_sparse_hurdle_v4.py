"""Output-only sparse hurdle diagnostic on already authenticated public cells.

Frozen v3 target/context design, six arms and ridge penalty are reused unchanged.
Occupancy and positive-log-expression magnitude are modeled separately. Outputs
remain continuous log1p-CP10k values, not raw counts or a production submission.
The caller must authenticate/register all inputs before decoding the v2 cache.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import socket

import numpy as np

import public_joint_target_source_v3 as v3
from public_control_anchor_v2 import _matrix, _strings, _representation, REPRESENTATION
from public_flow_stable_conditioning import fit_stable_target_transform
from public_mean_effect_baseline import balanced_weights, weighted_standardize, save_arrays, delta_diagnostics
from train_public_flow_pilot import diagnostics, load_npz, write_new

ARMS = v3.ARMS
SHUFFLE_SEEDS = v3.SHUFFLE_SEEDS
RANK_SEED = 20260924
OCCUPANCY_CLIP = 4.0
AMPLITUDE_CLIP = math.log(4.0)
DONOR_KINDS = {0: "zero", 1: "retained_anchor", 2: "borrowed_anchor", 3: "matched_context", 4: "training_prior", 5: "unresolved_activation"}
PREDICTION_KEYS = ("nonnegative", "group", "control_cache_rows", "control_original_rows", "treated_cache_rows",
    "treated_original_rows", "treated_group", "held_groups", "held_source", "head_delta", "occupancy_delta",
    "amplitude_delta", "unbounded_requested_occupancy", "requested_occupancy", "realized_occupancy",
    "desired_positive_count", "realized_positive_count", "donor_kind", "donor_original_rows",
    "group_control_mean", "group_treated_mean", "group_predicted_mean", "group_control_occupancy",
    "group_treated_occupancy", "group_predicted_occupancy")
WEIGHT_KEYS = ("coefficients", "context_mean", "context_scale", "fit_targets", "held_targets", "fit_source",
    "held_source", "aligned_target_ids", "aligned_values", "aligned_present", "feature_ids", "feature_signature_json",
    "feature_receipt_json", "positive_prior", "prior_positive_counts", "training_positive_support", "prior_source_ids",
    "prior_original_rows", "prior_pool_kind", "training_group_ids", "training_response")
TRANSFORM_KEYS = ("target_mean", "target_scale", "target_transform_provenance_json")


def policy():
    return {"schema": "public-sparse-hurdle-model-v4", "base_design": v3.policy(), "arms": list(ARMS),
        "representation": REPRESENTATION, "ridge_lambda": 10.0, "interactions": False,
        "heads": "concatenated occupancy-logit and positive-magnitude-log response; same frozen v3 design",
        "q_smoothing": "qtilde=(32*q+0.5)/33 identically for treated and reference regardless of sample size",
        "m_smoothing": "m=(mean_expression+prior/32)/(q+1/32); prior is positive-value mean on unique admitted training anchor+treated rows",
        "unsupported_genes": "zero prior and zero amplitude response; mask saved, no fabricated positive prior",
        "occupancy_prediction": "clip(anchor_q+sigmoid(logit(qtilde_anchor)+clip(delta_occ,-4,4))-qtilde_anchor,0,1)",
        "rounding": "floor(n*requested_q+0.5); unchanged requested count preserves positive support exactly",
        "amplitude_prediction": "multiply positive log1p-CP10k magnitudes by exp(clip(delta_amp,-log(4),log(4)))",
        "rank_seed": RANK_SEED, "ranking": "SHA256(namespace,seed,group_id,gene_id,original_row_id,role), ascending, arm-independent",
        "support": "retain lowest-ranked existing positives when decreasing; activate lowest-ranked zeros when increasing",
        "donors": {str(k): v for k, v in DONOR_KINDS.items()}, "donor_order": "original anchor positives, else independent matched context positives, else positive training prior; deterministic cycle",
        "donor_failure": "leave zero and report unresolved activation; no invented positive value",
        "zero_effect": "per-gene exact zero-head bypass before smoothing; exact NTC identity",
        "fit_roles": "only fitting-source/fitting-target treated; independent training anchor-reference; context never response/prior",
        "primary": "equal-batch target-pooled mean vectors before squared gene error; equal targets per source",
        "occupancy_metric": "equal-batch target-pooled positive fractions before squared gene error",
        "counts_in_metrics": "per-group counts; aggregate count metrics are source/target/batch macro means, not independent-cell totals",
        "empirical_null": "not estimated here; exact NTC bypass is an implementation check, not learned null calibration",
        "hyperparameter_selection": False, "count_emitter": False, "posttraining": False,
        "automatic_promotion": False, "evaluation_claim": "exploratory exposed-public development"}


def _row_ids(values, length, name):
    values = np.asarray(values)
    if values.dtype.kind not in "iu" or values.shape != (length,) or np.any(values < 0) or len(set(map(int, values))) != length:
        raise ValueError(f"Unique nonnegative original row IDs required: {name}")
    if values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
        raise ValueError("Original row IDs exceed int64")
    return values.astype(np.int64, copy=True)


@dataclass(frozen=True)
class PositivePrior:
    values: np.ndarray
    positive_counts: np.ndarray
    source_ids: np.ndarray
    original_rows: np.ndarray
    pool_kind: np.ndarray

    @property
    def supported(self):
        return self.positive_counts > 0


def training_positive_prior(arrays, fold):
    """Mean positives from unique TRAINING anchor+treated raw rows, never context."""
    targets = tuple(arrays["targets"].astype(str)); sources = tuple(arrays["sources"].astype(str))
    if (not isinstance(fold, v3.JointFold) or set(fold.fit_targets) & set(fold.held_targets)
        or fold.fit_source == fold.held_source or set(fold.fit_indices) & set(fold.held_target_indices_all_sources)
        or any(i < 0 or i >= len(targets) or sources[i] != fold.fit_source or targets[i] not in fold.fit_targets for i in fold.fit_indices)):
        raise ValueError("Unauthorized source/target in prior fitting role")
    unique = {}
    for field, labels, rowfield, kind in (("control", "control_group", "control_rows", 1), ("treated", "group", "treated_rows", 2)):
        indices = np.flatnonzero(np.isin(arrays[labels], fold.fit_indices))
        for index in indices:
            group = int(arrays[labels][index]); original = arrays[rowfield][index]
            if not isinstance(original, (int, np.integer)) or original < 0:
                raise ValueError("Invalid original training row identity")
            key = sources[group], int(original)
            value = np.asarray(arrays[field][index], dtype=np.float64)
            if value.ndim != 1 or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError("Invalid training positive-prior expression")
            if key in unique:
                old_kind, old_value = unique[key]
                if kind != old_kind or not np.array_equal(old_value, value):
                    raise ValueError("Original training row identity has conflicting pool or expression")
            else:
                unique[key] = kind, value.copy()
    if not unique:
        raise ValueError("No admitted training cells for positive prior")
    keys = sorted(unique)
    matrix = np.asarray([unique[key][1] for key in keys])
    positive = matrix > 0
    counts = positive.sum(0).astype(np.int64)
    # Scale first: avoid overflow in large positive sums and premature underflow
    # when many equal tiny positives would each round to zero after division.
    maxima = np.max(matrix, axis=0)
    scales = np.where(maxima > 0, maxima, 1)
    values = scales * (np.sum(matrix / scales, axis=0) / np.maximum(counts, 1))
    if not np.isfinite(values).all():
        raise ValueError("Positive-prior arithmetic overflow")
    result = PositivePrior(values, counts, np.asarray([k[0] for k in keys]),
        np.asarray([k[1] for k in keys], dtype=np.int64), np.asarray([unique[k][0] for k in keys], dtype=np.int8))
    for value in (result.values, result.positive_counts, result.source_ids, result.original_rows, result.pool_kind):
        value.setflags(write=False)
    return result


def head_response(treated, reference, prior, *, representation=REPRESENTATION):
    """Identical fixed occupancy smoothing for unequal treated/reference sizes."""
    _representation(representation)
    treated, reference = _matrix(treated, "treated fitting cells"), _matrix(reference, "reference fitting cells")
    prior = np.asarray(prior, dtype=np.float64)
    if treated.shape[1] != reference.shape[1] or prior.shape != (treated.shape[1],) or not np.isfinite(prior).all() or np.any(prior < 0):
        raise ValueError("Positive prior and response gene axes disagree")
    if np.any(treated < 0) or np.any(reference < 0):
        raise ValueError("Nonnegative log-expression required")
    qt, qc = np.mean(treated > 0, axis=0), np.mean(reference > 0, axis=0)
    st, sc = (32 * qt + .5) / 33, (32 * qc + .5) / 33
    occupancy = (np.log(st) - np.log1p(-st)) - (np.log(sc) - np.log1p(-sc))
    mt = (treated.mean(0) + prior / 32) / (qt + 1 / 32)
    mc = (reference.mean(0) + prior / 32) / (qc + 1 / 32)
    support = prior > 0
    if np.any((mt[support] <= 0) | (mc[support] <= 0)):
        raise ValueError("Supported gene has nonpositive smoothed magnitude")
    amplitude = np.zeros_like(prior)
    amplitude[support] = np.log(mt[support]) - np.log(mc[support])
    if not np.isfinite(occupancy).all() or not np.isfinite(amplitude).all():
        raise ValueError("Nonfinite training head response")
    return np.concatenate((occupancy, amplitude))


def training_responses(arrays, fold, prior):
    return np.asarray([head_response(arrays["treated"][arrays["group"] == group],
        arrays["control"][arrays["control_group"] == group], prior.values) for group in fold.fit_indices])


def fit_hurdle_fold(arrays, fold, aligned, arm):
    prior = training_positive_prior(arrays, fold)
    response = training_responses(arrays, fold, prior)
    model = v3.fit_joint_decoder(tuple(str(arrays["targets"][i]) for i in fold.fit_indices),
        tuple(str(arrays["sources"][i]) for i in fold.fit_indices), arrays["context"][list(fold.fit_indices)], response,
        aligned=aligned, arm=arm, fold=fold, representation=REPRESENTATION)
    return model, prior, response


def _rank(indices, original_rows, group_id, gene_id, role):
    def key(index):
        fields = ["public-sparse-hurdle-v4", RANK_SEED, int(group_id), gene_id, int(original_rows[index]), role]
        return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).digest(), int(original_rows[index])
    return sorted(map(int, indices), key=key)


@dataclass(frozen=True)
class HurdlePrediction:
    nonnegative: np.ndarray
    occupancy_delta: np.ndarray
    amplitude_delta: np.ndarray
    unbounded_requested_occupancy: np.ndarray
    requested_occupancy: np.ndarray
    realized_occupancy: np.ndarray
    desired_positive_count: np.ndarray
    realized_positive_count: np.ndarray
    donor_kind: np.ndarray
    donor_original_rows: np.ndarray
    metrics: dict


def sparse_anchor(control, context_control, prior, occupancy_delta, amplitude_delta, *, control_rows,
                  context_rows, group_id, genes, is_ntc=False, representation=REPRESENTATION):
    """Deterministic support changes with explicit donor fallback; no raw counts."""
    _representation(representation)
    control, context = _matrix(control, "anchor controls"), _matrix(context_control, "independent context controls")
    genes = _strings(genes, "output genes", unique=True)
    n, width = control.shape
    if context.shape[1] != width or len(genes) != width or np.any(control < 0) or np.any(context < 0):
        raise ValueError("Nonnegative aligned log-expression populations required")
    cr = _row_ids(control_rows, n, "anchor"); xr = _row_ids(context_rows, len(context), "context")
    if set(cr) & set(xr):
        raise ValueError("Anchor and context original rows must be disjoint")
    if type(group_id) not in (int, np.int64, np.int32) or group_id < 0 or type(is_ntc) is not bool:
        raise ValueError("Explicit group ID and boolean NTC flag required")
    prior, raw_occ, raw_amp = [np.asarray(x, dtype=np.float64) for x in (prior, occupancy_delta, amplitude_delta)]
    if any(x.shape != (width,) or not np.isfinite(x).all() for x in (prior, raw_occ, raw_amp)) or np.any(prior < 0):
        raise ValueError("Finite aligned nonnegative prior and two effect vectors required")
    occ = np.clip(raw_occ, -OCCUPANCY_CLIP, OCCUPANCY_CLIP)
    amp = np.clip(raw_amp, -AMPLITUDE_CLIP, AMPLITUDE_CLIP)
    if is_ntc:
        occ = np.zeros(width); amp = np.zeros(width)
    bypass = (occ == 0) & (amp == 0)
    original_count = np.sum(control > 0, axis=0).astype(np.int64)
    anchor_q = original_count / n
    unbounded = anchor_q.copy()
    # Δoccupancy==0 is exact, including when amplitude alone is active.
    active = occ != 0
    smooth = (32 * anchor_q[active] + .5) / 33
    logits = np.log(smooth) - np.log1p(-smooth) + occ[active]
    shifted = 1 / (1 + np.exp(-logits))
    unbounded[active] = anchor_q[active] + shifted - smooth
    requested = np.clip(unbounded, 0, 1)
    desired = np.floor(n * requested + .5).astype(np.int64)
    desired[bypass] = original_count[bypass]
    output = control.copy()
    kinds = np.where(control > 0, 1, 0).astype(np.int8)
    donor_rows = np.where(control > 0, cr[:, None], -1).astype(np.int64)
    activations = deactivations = unresolved = requested_activations = 0
    for g, gene in enumerate(genes):
        if bypass[g]:
            continue
        positives = np.flatnonzero(control[:, g] > 0)
        count = len(positives)
        if desired[g] < count:
            removed = _rank(positives, cr, group_id, gene, "retain_positive")[int(desired[g]):]
            output[removed, g] = 0; kinds[removed, g] = 0; donor_rows[removed, g] = -1
            deactivations += len(removed)
        elif desired[g] > count:
            zeros = _rank(np.flatnonzero(control[:, g] == 0), cr, group_id, gene, "activate_zero")
            add = zeros[:int(desired[g]) - count]
            requested_activations += len(add)
            if count:
                donor_indices = _rank(positives, cr, group_id, gene, "anchor_donor")
                donor_values, ids, kind = control[donor_indices, g], cr[donor_indices], 2
            else:
                donor_indices = _rank(np.flatnonzero(context[:, g] > 0), xr, group_id, gene, "context_donor")
                if donor_indices:
                    donor_values, ids, kind = context[donor_indices, g], xr[donor_indices], 3
                elif prior[g] > 0:
                    donor_values, ids, kind = np.array([prior[g]]), np.array([-1], dtype=np.int64), 4
                else:
                    donor_values, ids, kind = np.array([]), np.array([], dtype=np.int64), 5
            for number, row in enumerate(add):
                kinds[row, g] = kind
                if len(donor_values):
                    donor = number % len(donor_values)
                    output[row, g] = donor_values[donor]; donor_rows[row, g] = ids[donor]
                    activations += 1
                else:
                    unresolved += 1
        keep = output[:, g] > 0
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            output[keep, g] *= math.exp(float(amp[g]))
        if np.any(output[keep, g] <= 0):
            raise ValueError("Positive-magnitude underflow would silently change support")
    if not np.isfinite(output).all() or np.any(output < 0):
        raise ValueError("Sparse hurdle arithmetic overflow or negative output")
    realized_count = np.sum(output > 0, axis=0).astype(np.int64)
    realized = realized_count / n
    unchanged = desired == original_count
    unchanged_support = all(np.array_equal(output[:, g] > 0, control[:, g] > 0) for g in np.flatnonzero(unchanged))
    metrics = {"unchanged_count_fraction": float(unchanged.mean()),
        "occupancy_clip_fraction": float(np.mean(np.abs(raw_occ) > OCCUPANCY_CLIP)) if not is_ntc else 0.0,
        "amplitude_clip_fraction": float(np.mean(np.abs(raw_amp) > AMPLITUDE_CLIP)) if not is_ntc else 0.0,
        "occupancy_probability_clip_fraction": float(np.mean(unbounded != requested)),
        "requested_realized_occupancy_mse": float(np.mean((requested - realized) ** 2)),
        "requested_realized_occupancy_mae": float(np.mean(np.abs(requested - realized))),
        "unresolved_activation_fraction": unresolved / requested_activations if requested_activations else 0.0,
        "activation_count": activations, "deactivation_count": deactivations,
        "requested_activation_count": requested_activations, "unresolved_activation_count": unresolved,
        "positive_count_shortfall": int(np.sum(desired - realized_count)),
        "unchanged_count_support_verified": bool(unchanged_support),
        "exact_zero_effect_identity_verified": bool(np.array_equal(output[:, bypass], control[:, bypass])),
        "exact_ntc_identity_verified": bool(np.array_equal(output, control)) if is_ntc else True}
    result = HurdlePrediction(output, occ, amp, unbounded, requested, realized, desired, realized_count, kinds, donor_rows, metrics)
    for field in HurdlePrediction.__dataclass_fields__:
        value = getattr(result, field)
        if isinstance(value, np.ndarray): value.setflags(write=False)
    return result


def target_metrics(vectors):
    rows = []
    for source, target in sorted({(r["source"], r["target"]) for r in vectors}):
        selected = [r for r in vectors if (r["source"], r["target"]) == (source, target)]
        if len({r["batch"] for r in selected}) != len(selected):
            raise ValueError("Repeated target/source/batch")
        means = {key: np.mean([r[key] for r in selected], axis=0) for key in
            ("control_mean", "treated_mean", "predicted_mean", "control_occupancy", "treated_occupancy", "predicted_occupancy")}
        row = {"source": source, "target": target, "batches": len(selected)}
        for metric, pred, truth in (("target_pooled_mse", "predicted_mean", "treated_mean"),
            ("control_target_pooled_mse", "control_mean", "treated_mean"),
            ("target_pooled_occupancy_mse", "predicted_occupancy", "treated_occupancy"),
            ("control_target_pooled_occupancy_mse", "control_occupancy", "treated_occupancy")):
            row[metric] = float(np.mean((means[pred] - means[truth]) ** 2))
        rows.append(row)
    return rows


def evaluate_fold(arrays, fold, aligned, model, prior):
    records, vectors, chunks = [], [], {k: [] for k in PREDICTION_KEYS if k not in ("held_groups", "held_source")}
    null_identity = zero_identity = True
    for group in fold.evaluation_indices:
        ci = np.flatnonzero(arrays["control_group"] == group); ti = np.flatnonzero(arrays["group"] == group)
        xi = np.flatnonzero(arrays["context_control_group"] == group)
        control, treated, context = arrays["control"][ci], arrays["treated"][ti], arrays["context_control"][xi]
        target = str(arrays["targets"][group]); source = str(arrays["sources"][group]); batch = str(arrays["batches"][group])
        features = v3._subset(aligned, (target,))
        head = model.predict_delta(features, arrays["context"][group:group+1], is_ntc=np.array([False]))[0]
        width = control.shape[1]
        args = dict(control_rows=arrays["control_rows"][ci], context_rows=arrays["context_control_rows"][xi],
                    group_id=group, genes=tuple(arrays["genes"].astype(str)))
        pred = sparse_anchor(control, context, prior.values, head[:width], head[width:], **args)
        null = sparse_anchor(control, context, prior.values, head[:width], head[width:], is_ntc=True, **args)
        zero = sparse_anchor(control, context, prior.values, np.zeros(width), np.zeros(width), **args)
        null_identity &= np.array_equal(null.nonnegative, control)
        zero_identity &= np.array_equal(zero.nonnegative, control)
        metrics = diagnostics(pred.nonnegative, control, treated)
        cm, tm, pm = control.astype(np.float64).mean(0), treated.astype(np.float64).mean(0), pred.nonnegative.mean(0)
        cq, tq, pq = np.mean(control > 0, axis=0), np.mean(treated > 0, axis=0), pred.realized_occupancy
        vector = {"source": source, "target": target, "batch": batch, "control_mean": cm, "treated_mean": tm,
            "predicted_mean": pm, "control_occupancy": cq, "treated_occupancy": tq, "predicted_occupancy": pq}
        vectors.append(vector)
        records.append({"source": source, "target": target, "batch": batch, "group": group,
            "anchor_cells": len(ci), "treated_cells": len(ti), **metrics, **pred.metrics,
            "control_zero_fraction": float(np.mean(control == 0)), "treated_zero_fraction": float(np.mean(treated == 0)),
            **delta_diagnostics(pm - cm, tm - cm)})
        entry = {"nonnegative": pred.nonnegative, "group": np.full(len(ci), group, dtype=np.int64),
            "control_cache_rows": ci, "control_original_rows": arrays["control_rows"][ci],
            "treated_cache_rows": ti, "treated_original_rows": arrays["treated_rows"][ti],
            "treated_group": np.full(len(ti), group, dtype=np.int64), "head_delta": head,
            **{k: getattr(pred, k) for k in ("occupancy_delta", "amplitude_delta", "unbounded_requested_occupancy",
                "requested_occupancy", "realized_occupancy", "desired_positive_count", "realized_positive_count", "donor_kind", "donor_original_rows")},
            **{"group_" + k: value for k, value in vector.items() if isinstance(value, np.ndarray)}}
        for key, value in entry.items(): chunks[key].append(value)
    population_keys = {"nonnegative", "group", "control_cache_rows", "control_original_rows", "treated_cache_rows",
        "treated_original_rows", "treated_group", "donor_kind", "donor_original_rows"}
    output = {key: np.concatenate(values) if key in population_keys else np.asarray(values) for key, values in chunks.items()}
    output.update(held_groups=np.asarray(fold.evaluation_indices, dtype=np.int64), held_source=np.asarray(fold.held_source))
    targets_report = target_metrics(vectors)
    report = {"fold_id": fold.fold_id, "target_fold": fold.target_fold, "fit_source": fold.fit_source, "held_source": fold.held_source,
        "fit_targets": list(fold.fit_targets), "held_targets": list(fold.held_targets), "fit_group_ids": list(fold.fit_indices),
        "joint_eval_group_ids": list(fold.evaluation_indices), "held_target_group_ids_all_sources": list(fold.held_target_indices_all_sources),
        "groups": records, "targets_report": targets_report, "metrics": v3._aggregate(records, targets_report),
        "exact_ntc_identity_verified": bool(null_identity), "exact_zero_effect_identity_verified": bool(zero_identity)}
    return report, output


def _model_arrays(model, prior, response, fold, aligned, receipt, table):
    result = {"coefficients": model.coefficients, "context_mean": model.context_mean, "context_scale": model.context_scale,
        "fit_targets": np.asarray(model.training_targets), "held_targets": np.asarray(fold.held_targets),
        "fit_source": np.asarray(fold.fit_source), "held_source": np.asarray(fold.held_source),
        "aligned_target_ids": np.asarray(aligned.target_ids), "aligned_values": aligned.values, "aligned_present": aligned.present,
        "feature_ids": np.asarray(table.feature_ids), "feature_signature_json": np.asarray(json.dumps(aligned.signature)),
        "feature_receipt_json": np.asarray(json.dumps(receipt, sort_keys=True)), "positive_prior": prior.values,
        "prior_positive_counts": prior.positive_counts, "training_positive_support": prior.supported,
        "prior_source_ids": prior.source_ids, "prior_original_rows": prior.original_rows, "prior_pool_kind": prior.pool_kind,
        "training_group_ids": np.asarray(fold.fit_indices, dtype=np.int64), "training_response": response}
    if model.target_transform:
        result.update(target_mean=model.target_transform.mean, target_scale=model.target_transform.scale,
            target_transform_provenance_json=np.asarray(json.dumps(model.target_transform.provenance(), sort_keys=True)))
    return result


def _final_metrics(reports):
    groups = [g for r in reports for g in r["groups"]]; targets = [t for r in reports for t in r["targets_report"]]
    if len({(t["source"], t["target"]) for t in targets}) != len(targets):
        raise ValueError("Repeated out-of-fold target/source would bias aggregation")
    sources = [{"source": s, "metrics": v3._aggregate([g for g in groups if g["source"] == s],
        [t for t in targets if t["source"] == s])} for s in sorted({g["source"] for g in groups})]
    return sources, v3._aggregate(groups, targets)


def run_arm(arrays, registration, tables, arm, output_dir, provenance, progress=None):
    if arm not in ARMS:
        raise ValueError("Unknown fixed hurdle arm")
    folds = v3.joint_folds(tuple(arrays["targets"].astype(str)), tuple(arrays["sources"].astype(str)), registration)
    if set(tables) != {f.target_fold for f in folds}:
        raise ValueError("Authenticated fold-local GO tables required")
    if not isinstance(provenance.get("diagnostic_contract_sha256"), str) or len(provenance["diagnostic_contract_sha256"]) != 64:
        raise ValueError("Frozen diagnostic contract required")
    output_dir = Path(output_dir)
    if os.path.lexists(output_dir): raise FileExistsError("Fresh hurdle output directory required")
    output_dir.mkdir(mode=0o700)
    reports, weights, predictions, events = [], {}, {}, []
    for number, fold in enumerate(folds):
        aligned, receipt = v3.role_features(tables[fold.target_fold], fold.fit_targets, fold.held_targets, arm=arm, fold_id=fold.target_fold)
        model, prior, response = fit_hurdle_fold(arrays, fold, aligned, arm)
        report, output = evaluate_fold(arrays, fold, aligned, model, prior)
        prefix = f"fold_{number}_"
        report.update(array_prefix=prefix, feature_receipt=receipt,
            target_transform=model.target_transform.provenance() if model.target_transform else None,
            prior_unique_training_rows=len(prior.original_rows), prior_unique_anchor_rows=int(np.sum(prior.pool_kind == 1)),
            prior_unique_treated_rows=int(np.sum(prior.pool_kind == 2)), unsupported_genes=int(np.sum(~prior.supported)))
        weights.update({prefix + k: val for k, val in _model_arrays(model, prior, response, fold, aligned, receipt, tables[fold.target_fold]).items()})
        predictions.update({prefix + k: val for k, val in output.items()})
        reports.append(report)
        event = {"fold_index": number, "held_source": fold.held_source, "target_fold": fold.target_fold, "metrics": report["metrics"]}
        events.append(event)
        if progress is not None: progress(event)
    sources, aggregate = _final_metrics(reports)
    weights["genes"] = predictions["genes"] = arrays["genes"]
    summary = {"schema": "public-sparse-hurdle-summary-v4", "status": "completed_unpromoted_diagnostic", "stage": "validation",
        "arm": arm, "policy": policy(), "folds": reports, "source_metrics": sources, "aggregate": aggregate,
        "provenance": provenance, "diagnostic_contract_sha256": provenance["diagnostic_contract_sha256"],
        "weights_sha256": save_arrays(output_dir / "weights.npz", weights), "predictions_sha256": save_arrays(output_dir / "predictions.npz", predictions),
        "exact_ntc_identity_verified": all(r["exact_ntc_identity_verified"] for r in reports),
        "exact_zero_effect_identity_verified": all(r["exact_zero_effect_identity_verified"] for r in reports),
        "count_emitter": False, "posttraining_performed": False, "promoted": False, "submission_performed": False,
        "hyperparameter_selection": False, "H1_treated_read": False, "RPE1_treated_read": False,
        "challenge_treated_used": False, "held_roles_used_for_fitting": False,
        "prior_public_development_exposure_acknowledged": True, "learned_null_calibration_evaluated": False,
        "runtime": {"hostname": socket.gethostname(), "device": "cpu", "numpy_version": np.__version__}}
    write_new(output_dir / "steps.jsonl", b"".join((json.dumps(e, sort_keys=True, allow_nan=False) + "\n").encode() for e in events))
    write_new(output_dir / "summary.json", (json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    return summary


def _same_tree(actual, expected, where="metrics"):
    """Strict structure with tiny arithmetic tolerance only for finite floats."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected): raise ValueError(f"Replay structure mismatch: {where}")
        for key in expected: _same_tree(actual[key], expected[key], where + "." + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected): raise ValueError(f"Replay list mismatch: {where}")
        for i, (a, e) in enumerate(zip(actual, expected)): _same_tree(a, e, f"{where}[{i}]")
    elif isinstance(expected, (float, np.floating)):
        if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError(f"Replay numerical mismatch: {where}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Replay value mismatch: {where}")


def verify_artifacts(output_dir, summary, arrays, registration, tables):
    """No ridge refit: replay saved heads, training-only priors and donor outputs."""
    summary = json.loads(json.dumps(summary, allow_nan=False))
    arm = summary["arm"]
    if (arm not in ARMS or summary.get("schema") != "public-sparse-hurdle-summary-v4"
        or summary.get("status") != "completed_unpromoted_diagnostic" or summary.get("stage") != "validation"):
        raise ValueError("Invalid completed hurdle summary")
    # JSON converts integer donor-code dictionary keys to strings.
    _same_tree(summary["policy"], json.loads(json.dumps(policy())), "policy")
    for key in ("exact_ntc_identity_verified", "exact_zero_effect_identity_verified", "prior_public_development_exposure_acknowledged"):
        if summary.get(key) is not True: raise ValueError("Required identity/exposure flag failed")
    for key in ("count_emitter", "posttraining_performed", "promoted", "submission_performed", "hyperparameter_selection",
                "H1_treated_read", "RPE1_treated_read", "challenge_treated_used", "held_roles_used_for_fitting"):
        if summary.get(key) is not False: raise ValueError("Forbidden hurdle activity")
    folds = v3.joint_folds(tuple(arrays["targets"].astype(str)), tuple(arrays["sources"].astype(str)), registration)
    if len(summary["folds"]) != len(folds): raise ValueError("Fold count mismatch")
    prefixes = [f"fold_{i}_" for i in range(len(folds))]
    weights_keys = WEIGHT_KEYS + (() if arm in {"control", "context"} else TRANSFORM_KEYS)
    weights = load_npz(Path(output_dir) / "weights.npz", summary["weights_sha256"], {"genes"} | {p+k for p in prefixes for k in weights_keys})
    predictions = load_npz(Path(output_dir) / "predictions.npz", summary["predictions_sha256"], {"genes"} | {p+k for p in prefixes for k in PREDICTION_KEYS})
    if not np.array_equal(weights["genes"], arrays["genes"]) or not np.array_equal(predictions["genes"], arrays["genes"]):
        raise ValueError("Saved gene axis differs")
    replayed = []
    for prefix, fold, saved_report in zip(prefixes, folds, summary["folds"]):
        aligned, receipt = v3.role_features(tables[fold.target_fold], fold.fit_targets, fold.held_targets, arm=arm, fold_id=fold.target_fold)
        prior = training_positive_prior(arrays, fold); response = training_responses(arrays, fold, prior)
        target_ids = tuple(str(arrays["targets"][i]) for i in fold.fit_indices)
        sources = tuple(str(arrays["sources"][i]) for i in fold.fit_indices)
        cw = balanced_weights(target_ids, sources)
        mean, scale = weighted_standardize(arrays["context"][list(fold.fit_indices)], cw)
        transform = None if arm in {"control", "context"} else fit_stable_target_transform(aligned,
            train_ids=tuple(sorted(fold.fit_targets)), excluded_ids=fold.held_targets, fold_id=fold.fold_id)
        coefficients = weights[prefix+"coefficients"]
        design_width = 1 + len(mean) + (0 if transform is None else aligned.values.shape[1]+aligned.present.shape[1]+1)
        if coefficients.shape != (design_width, 2*len(arrays["genes"])) or not np.isfinite(coefficients).all():
            raise ValueError("Invalid saved two-head coefficients")
        if arm == "control" and np.any(coefficients != 0): raise ValueError("Control coefficients must be exact zero")
        if arm != "control":
            pieces = [np.ones((len(fold.fit_indices), 1))]
            if transform:
                selected = v3._subset(aligned, tuple(sorted(fold.fit_targets)))
                block = np.column_stack((transform.transform(selected), selected.present, selected.unknown_target))
                lookup = {t: i for i, t in enumerate(selected.target_ids)}
                pieces.append(block[[lookup[t] for t in target_ids]] / math.sqrt(block.shape[1]))
            pieces.append((arrays["context"][list(fold.fit_indices)] - mean) / scale / math.sqrt(len(mean)))
            design = np.column_stack(pieces)
            lhs = design.T @ (cw[:, None] * (design @ coefficients)) + 10 * coefficients
            rhs = design.T @ (cw[:, None] * response)
            tolerance = 1e-10 * (1 + float(np.max(np.abs(rhs))))
            if not np.isfinite(lhs).all() or np.max(np.abs(lhs-rhs)) > tolerance:
                raise ValueError("Saved coefficients fail fixed-ridge normal equations")
        decoder = v3.JointDecoder(arm, fold.fold_id, tuple(sorted(fold.fit_targets)), fold.fit_source, fold.held_source,
                                 mean, scale, coefficients, transform)
        expected_weights = _model_arrays(decoder, prior, response, fold, aligned, receipt, tables[fold.target_fold])
        for key, expected in expected_weights.items():
            if not np.array_equal(weights[prefix+key], expected): raise ValueError(f"Saved training state replay mismatch: {key}")
        report, output = evaluate_fold(arrays, fold, aligned, decoder, prior)
        for key, expected in output.items():
            if not np.array_equal(predictions[prefix+key], expected): raise ValueError(f"Saved prediction replay mismatch: {key}")
        report.update(array_prefix=prefix, feature_receipt=receipt,
            target_transform=transform.provenance() if transform else None,
            prior_unique_training_rows=len(prior.original_rows), prior_unique_anchor_rows=int(np.sum(prior.pool_kind == 1)),
            prior_unique_treated_rows=int(np.sum(prior.pool_kind == 2)), unsupported_genes=int(np.sum(~prior.supported)))
        _same_tree(saved_report, json.loads(json.dumps(report)), "fold")
        replayed.append(report)
    sources, aggregate = _final_metrics(replayed)
    _same_tree(summary["source_metrics"], sources, "source_metrics"); _same_tree(summary["aggregate"], aggregate, "aggregate")
    return {"passed": True, "ridge_refits": 0, "saved_head_replay": True, "training_prior_replay": True,
        "deterministic_donor_replay": True, "original_row_mapping_verified": True,
        "pooled_and_batch_metrics_replayed": True, "out_of_fold_groups": sum(len(f.evaluation_indices) for f in folds),
        "out_of_fold_source_targets": sum(len(f.held_targets) for f in folds)}
