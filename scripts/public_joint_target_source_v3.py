"""Fixed public-only joint target/source diagnostic, never a count emitter.

The caller authenticates the existing v2 cache, train-vocabulary GO tables and
metadata registration before calling run_arm. No raw data are opened here.
Held-target treated cells in BOTH sources and every held-source response are
excluded from fitting. Previously exposed public data remain development data.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import socket

import numpy as np

from public_flow_conditioning import AlignedTargets, align_target_features
from public_flow_stable_conditioning import METHOD, fit_stable_target_transform
from public_control_anchor_v2 import (
    REPRESENTATION, _strings, _matrix, _ntc, _representation,
    anchor_expression, paired_population_diagnostics, aggregate_group_metrics,
)
from public_mean_effect_baseline import balanced_weights, weighted_standardize, ridge_fit, delta_diagnostics, save_arrays
from train_public_flow_pilot import write_new

SHUFFLE_SEEDS = (20260921, 20260922, 20260923)
ARMS = ("control", "context", "true", *(f"shuffled_{s}" for s in SHUFFLE_SEEDS))


def policy():
    return {"schema": "public-joint-target-source-model-v3", "arms": list(ARMS),
        "representation": REPRESENTATION, "ridge_lambda": 10.0, "gain": 1.0,
        "context_scale_floor": 0.1, "target_transform": METHOD,
        "intercept": "penalized_zero_effect_prior", "pca": False,
        "fit_weights": "equal source then target then batch",
        "fit_response": "training treated mean minus independent training anchor-reference mean",
        "context": "separate matched control pool; train-only normalization",
        "split": "four target folds, two source directions; held-target treated excluded in BOTH sources",
        "vocabulary": "GO terms fitted on 42 training targets; held annotations restricted to that vocabulary",
        "shuffle": "values and coverage masks move jointly; separate training/held permutations; no cross-role donors",
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "shuffle_rng": "numpy default_rng with SHA256(namespace, seed, target-fold, role); source-independent",
        "prediction": "explicit positive_part(anchor + delta); control arm gain zero; exact NTC/zero bypass",
        "primary": "target_pooled_mse: equal-batch mean vectors FIRST, then squared gene error; target macro per source",
        "secondary": "equal source/target/batch centroid MSE, MMD, projection, sparsity and shift diagnostics",
        "hyperparameter_selection": False, "count_emitter": False,
        "posttraining": False, "automatic_promotion": False,
        "evaluation_claim": "exploratory exposed-public development, not untouched independent test"}


def _subset(aligned, ids):
    ids = _strings(ids, "feature subset", unique=True)
    lookup = {t: i for i, t in enumerate(aligned.target_ids)}
    if not set(ids) <= set(lookup):
        raise ValueError("Feature target absent from registered axis")
    rows = [lookup[t] for t in ids]
    return replace(aligned, target_ids=ids, values=aligned.values[rows], present=aligned.present[rows])


def role_features(table, fit_targets, held_targets, *, arm, fold_id):
    """Return registered features with role-contained, jointly masked shuffles."""
    fit = _strings(fit_targets, "fit targets", unique=True)
    held = _strings(held_targets, "held targets", unique=True)
    if arm not in ARMS or set(fit) & set(held) or not isinstance(fold_id, str) or not fold_id:
        raise ValueError("Invalid feature arm or target roles")
    targets = tuple(sorted((*fit, *held)))
    original = align_target_features(targets, [table])
    donors = {t: t for t in targets}
    if arm.startswith("shuffled_"):
        seed = int(arm.split("_")[1])
        for role, ids in (("fit", fit), ("held", held)):
            ids = tuple(sorted(ids))
            payload = json.dumps(["joint-target-source-v3", seed, fold_id, role], separators=(",", ":"))
            digest = hashlib.sha256(payload.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "big"))
            donors.update(zip(ids, (ids[i] for i in rng.permutation(len(ids)))))
        lookup = {t: i for i, t in enumerate(targets)}
        rows = [lookup[donors[t]] for t in targets]
        operation = "role_shuffle_v3:" + hashlib.sha256(json.dumps(donors, sort_keys=True).encode()).hexdigest()
        aligned = replace(original, values=original.values[rows], present=original.present[rows], operation=operation)
    else:
        aligned = original
    receipt = {"fold_id": fold_id, "arm": arm, "donors": donors,
        "fit_targets": list(fit), "held_targets": list(held),
        "original_covered_targets": [t for i, t in enumerate(targets) if original.present[i].any()],
        "assigned_covered_targets": [t for i, t in enumerate(targets) if aligned.present[i].any()],
        "feature_ids": list(table.feature_ids), "source_fingerprint": table.fingerprint,
        "operation": aligned.operation, "roles_crossed": False,
        "association_changes": {}}
    lookup = {t: i for i, t in enumerate(targets)}
    for role, ids in (("fit", fit), ("held", held)):
        donor_changes = sum(donors[t] != t for t in ids)
        packet_changes = sum(not (np.array_equal(original.values[lookup[t]], aligned.values[lookup[t]])
            and np.array_equal(original.present[lookup[t]], aligned.present[lookup[t]])) for t in ids)
        receipt["association_changes"][role] = {
            "targets": len(ids), "donor_identity_changed": donor_changes,
            "donor_fixed_points": len(ids) - donor_changes,
            "consumed_packet_changed": packet_changes,
            "identical_consumed_packets": len(ids) - packet_changes,
            "changed_donor_but_identical_packet": donor_changes - packet_changes}
    return aligned, receipt


@dataclass(frozen=True)
class JointFold:
    fold_id: str
    target_fold: str
    fit_source: str
    held_source: str
    fit_targets: tuple[str, ...]
    held_targets: tuple[str, ...]
    fit_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    held_target_indices_all_sources: tuple[int, ...]
    other_excluded_indices: tuple[int, ...]


def joint_folds(targets, sources, registration):
    """Validate metadata before any expression arithmetic (also synthetic-safe)."""
    targets, sources = _strings(targets, "group targets"), _strings(sources, "group sources")
    if len(targets) != len(sources) or len(set(sources)) != 2:
        raise ValueError("Two aligned public source axes required")
    universe = set(targets)
    excluded = set(_strings(registration.get("excluded_targets", ()), "excluded targets", empty=True, unique=True))
    if excluded & universe:
        raise ValueError("Excluded target in cache")
    fold_records = registration.get("folds")
    if not isinstance(fold_records, list) or not fold_records:
        raise ValueError("Registered target folds required")
    result, seen_held, seen_names = [], set(), set()
    for record in fold_records:
        name = record["fold_id"]
        if not isinstance(name, str) or not name or name in seen_names:
            raise ValueError("Distinct registered target fold identifiers required")
        seen_names.add(name)
        fit = _strings(record["fit_targets"], "fit target roster", unique=True)
        held = _strings(record["held_targets"], "held target roster", unique=True)
        if set(fit) & set(held) or set(fit) | set(held) != universe or seen_held & set(held):
            raise ValueError("Target fold roles overlap or fail to partition the full target roster")
        seen_held.update(held)
        directions = record.get("directions", [])
        if len(directions) != 2 or {d["held_source"] for d in directions} != set(sources):
            raise ValueError("Both fixed source directions required")
        for d in directions:
            fs, hs = d["fit_source"], d["held_source"]
            if {fs, hs} != set(sources):
                raise ValueError("Fitting and held sources must be distinct")
            fi = tuple(i for i, (t, s) in enumerate(zip(targets, sources)) if s == fs and t in fit)
            ei = tuple(i for i, (t, s) in enumerate(zip(targets, sources)) if s == hs and t in held)
            hi = tuple(i for i, t in enumerate(targets) if t in held)
            # Held-target groups in either source have their own audit role;
            # the remaining exclusion consists of fit targets in held source.
            oi = tuple(i for i in range(len(targets)) if i not in set(fi) | set(hi))
            if {targets[i] for i in fi} != set(fit) or {targets[i] for i in ei} != set(held):
                raise ValueError("Targets must occur in each registered source role")
            for key, expected in (("fit_group_ids", fi), ("joint_eval_group_ids", ei),
                                   ("held_target_group_ids_all_sources", hi), ("other_excluded_group_ids", oi)):
                raw = d.get(key)
                if not isinstance(raw, list) or any(type(i) is not int for i in raw) or tuple(raw) != expected:
                    raise ValueError(f"Registered {key} differs from independently derived roles")
            result.append(JointFold(d["direction_id"], name, fs, hs, fit, held, fi, ei, hi, oi))
    if seen_held != universe or len({f.fold_id for f in result}) != len(result):
        raise ValueError("Each target must be held exactly once and direction IDs unique")
    return tuple(result)


@dataclass(frozen=True)
class JointDecoder:
    arm: str
    fold_id: str
    training_targets: tuple[str, ...]
    training_source: str
    held_source: str
    context_mean: np.ndarray
    context_scale: np.ndarray
    coefficients: np.ndarray
    target_transform: object | None

    def predict_delta(self, aligned, control_context, *, is_ntc):
        """Apply saved transforms to new annotations, never a target-ID lookup."""
        if not isinstance(aligned, AlignedTargets):
            raise ValueError("Explicit aligned target features and coverage masks required")
        context = _matrix(control_context, "prediction context")
        ntc = _ntc(is_ntc, len(aligned.target_ids))
        if context.shape != (len(aligned.target_ids), len(self.context_mean)):
            raise ValueError("Prediction context axis mismatch")
        result = np.zeros((len(context), self.coefficients.shape[1]))
        if self.arm == "control" or np.all(ntc):
            return result
        parts = [np.ones((len(context), 1))]
        if self.target_transform is not None:
            features = np.column_stack((self.target_transform.transform(aligned), aligned.present, aligned.unknown_target))
            parts.append(features / math.sqrt(features.shape[1]))
        parts.append((context - self.context_mean) / self.context_scale / math.sqrt(context.shape[1]))
        result = np.column_stack(parts) @ self.coefficients
        result[ntc] = 0.0
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite predicted effect")
        return result


def fit_joint_decoder(targets, sources, context, response, *, aligned, arm, fold, representation):
    """Low-level fit accepts TRAINING group response arrays only."""
    _representation(representation)
    targets, sources = _strings(targets, "fit targets"), _strings(sources, "fit sources")
    if arm not in ARMS or not isinstance(fold, JointFold) or len(targets) != len(sources):
        raise ValueError("Invalid fixed arm or fold")
    if set(targets) != set(fold.fit_targets) or set(sources) != {fold.fit_source} or set(targets) & set(fold.held_targets):
        raise ValueError("Unauthorized target or source in fitting arrays")
    context, response = _matrix(context, "training context"), _matrix(response, "training effect")
    if len(context) != len(targets) or len(response) != len(targets):
        raise ValueError("Training group axis mismatch")
    weights = balanced_weights(targets, sources)
    mean, scale = weighted_standardize(context, weights)
    pieces, transform = [np.ones((len(context), 1))], None
    if arm not in {"control", "context"}:
        transform = fit_stable_target_transform(aligned, train_ids=tuple(sorted(fold.fit_targets)),
            excluded_ids=fold.held_targets, fold_id=fold.fold_id)
        unique = _subset(aligned, tuple(sorted(fold.fit_targets)))
        features = np.column_stack((transform.transform(unique), unique.present, unique.unknown_target))
        lookup = {t: i for i, t in enumerate(unique.target_ids)}
        pieces.append(features[[lookup[t] for t in targets]] / math.sqrt(features.shape[1]))
    pieces.append((context - mean) / scale / math.sqrt(context.shape[1]))
    design = np.column_stack(pieces)
    coefficients = np.zeros((design.shape[1], response.shape[1])) if arm == "control" else ridge_fit(design, response, weights, regularization=10.0)
    for value in (mean, scale, coefficients):
        value.setflags(write=False)
    return JointDecoder(arm, fold.fold_id, tuple(sorted(fold.fit_targets)), fold.fit_source, fold.held_source,
                        mean, scale, coefficients, transform)


def fit_cache_fold(arrays, fold, aligned, arm):
    """Read only training group effects after independently validated roles."""
    targets, sources = tuple(arrays["targets"].astype(str)), tuple(arrays["sources"].astype(str))
    if any(sources[i] != fold.fit_source or targets[i] not in fold.fit_targets for i in fold.fit_indices):
        raise ValueError("Unauthorized training indices")
    response = []
    for group in fold.fit_indices:
        treated = arrays["treated"][arrays["group"] == group].astype(np.float64)
        reference = arrays["control"][arrays["control_group"] == group].astype(np.float64)
        if min(len(treated), len(reference)) < 2:
            raise ValueError("Insufficient independent training populations")
        response.append(treated.mean(0) - reference.mean(0))
    return fit_joint_decoder(tuple(targets[i] for i in fold.fit_indices), tuple(sources[i] for i in fold.fit_indices),
        arrays["context"][list(fold.fit_indices)], np.asarray(response), aligned=aligned, arm=arm,
        fold=fold, representation=REPRESENTATION)


def pooled_target_metrics(rows):
    """Mean matched batch vectors first; never average batch squared errors."""
    reports = []
    for source, target in sorted({(r["source"], r["target"]) for r in rows}):
        selected = [r for r in rows if (r["source"], r["target"]) == (source, target)]
        if len({r["batch"] for r in selected}) != len(selected):
            raise ValueError("Duplicate target/source/batch in pooled metric")
        control, observed, raw, predicted = [np.mean([r[k] for r in selected], axis=0)
            for k in ("control_mean", "treated_mean", "raw_mean", "predicted_mean")]
        shifts = delta_diagnostics(predicted - control, observed - control)
        reports.append({"source": source, "target": target, "batches": len(selected),
            "target_pooled_mse": float(np.mean((predicted - observed) ** 2)),
            "control_target_pooled_mse": float(np.mean((control - observed) ** 2)),
            "raw_target_pooled_mse": float(np.mean((raw - observed) ** 2)),
            **{"target_pooled_" + k: value for k, value in shifts.items()}})
    return reports


def _aggregate(groups, targets_report):
    result = aggregate_group_metrics(groups)
    for key in sorted(set(targets_report[0]) - {"source", "target", "batches"}):
        per_source = []
        for source in sorted({r["source"] for r in targets_report}):
            values = [r[key] for r in targets_report if r["source"] == source and r[key] is not None]
            if values:
                per_source.append(float(np.mean(values)))
        result[key] = float(np.mean(per_source)) if per_source else None
    return result


def evaluate_joint_fold(arrays, fold, aligned, model):
    """Evaluate the registered held-target/held-source populations after fit."""
    held = np.asarray(fold.evaluation_indices, dtype=np.int64)
    rows, vectors, raw_rows, positive_rows, pgroup, crow, trow, tgroup, deltas = [], [], [], [], [], [], [], [], []
    hard_null, zero_identity = True, True
    for group in held:
        target = str(arrays["targets"][group])
        features = _subset(aligned, (target,))
        delta = model.predict_delta(features, arrays["context"][group:group + 1], is_ntc=np.array([False]))[0]
        ci, ti = np.flatnonzero(arrays["control_group"] == group), np.flatnonzero(arrays["group"] == group)
        control, treated = arrays["control"][ci], arrays["treated"][ti]
        prediction = anchor_expression(control, delta, is_ntc=np.zeros(len(ci), dtype=bool), representation=REPRESENTATION,
                                       gain=0.0 if model.arm == "control" else 1.0)
        null_delta = model.predict_delta(features, arrays["context"][group:group + 1], is_ntc=np.array([True]))[0]
        null = anchor_expression(control, null_delta, is_ntc=np.ones(len(ci), dtype=bool), representation=REPRESENTATION)
        zero = anchor_expression(control, np.zeros(control.shape[1]), is_ntc=np.zeros(len(ci), dtype=bool), representation=REPRESENTATION)
        hard_null &= np.array_equal(null.nonnegative, control)
        zero_identity &= np.array_equal(zero.nonnegative, control)
        paired = paired_population_diagnostics(prediction, control, treated, representation=REPRESENTATION)
        cm, tm = control.astype(np.float64).mean(0), treated.astype(np.float64).mean(0)
        pm, rm = prediction.nonnegative.mean(0), prediction.raw.mean(0)
        meta = {"source": str(arrays["sources"][group]), "target": target, "batch": str(arrays["batches"][group])}
        rows.append({**meta, "group": int(group), "anchor_cells": len(ci), "treated_cells": len(ti),
            **paired["nonnegative"], **paired["projection"],
            "raw_model_centroid_mse": paired["raw"]["model_centroid_mse"], "raw_model_mmd2": paired["raw"]["model_mmd2"],
            "control_zero_fraction": float(np.mean(control == 0)), "treated_zero_fraction": float(np.mean(treated == 0)),
            **delta_diagnostics(pm - cm, tm - cm)})
        vectors.append({**meta, "control_mean": cm, "treated_mean": tm, "raw_mean": rm, "predicted_mean": pm})
        raw_rows.append(prediction.raw); positive_rows.append(prediction.nonnegative); deltas.append(delta)
        pgroup.extend([int(group)] * len(ci)); crow.extend(ci); trow.extend(ti); tgroup.extend([int(group)] * len(ti))
    target_report = pooled_target_metrics(vectors)
    report = {"fold_id": fold.fold_id, "target_fold": fold.target_fold, "fit_source": fold.fit_source, "held_source": fold.held_source,
        "fit_targets": list(fold.fit_targets), "held_targets": list(fold.held_targets),
        "fit_group_ids": list(fold.fit_indices), "joint_eval_group_ids": list(fold.evaluation_indices),
        "held_target_group_ids_all_sources": list(fold.held_target_indices_all_sources),
        "metrics": _aggregate(rows, target_report), "groups": rows, "targets_report": target_report,
        "exact_ntc_identity_verified": bool(hard_null), "exact_zero_effect_identity_verified": bool(zero_identity)}
    predictions = {"raw": np.concatenate(raw_rows), "nonnegative": np.concatenate(positive_rows), "delta": np.asarray(deltas),
        "group": np.asarray(pgroup, dtype=np.int64), "control_cache_rows": np.asarray(crow, dtype=np.int64),
        "treated_cache_rows": np.asarray(trow, dtype=np.int64), "treated_group": np.asarray(tgroup, dtype=np.int64),
        "held_groups": held, "held_source": np.asarray(fold.held_source),
        "group_control_mean": np.asarray([r["control_mean"] for r in vectors]),
        "group_treated_mean": np.asarray([r["treated_mean"] for r in vectors]),
        "group_raw_mean": np.asarray([r["raw_mean"] for r in vectors]),
        "group_predicted_mean": np.asarray([r["predicted_mean"] for r in vectors])}
    return report, predictions


def run_arm(arrays, registration, tables, arm, output_dir, provenance, progress=None):
    """Caller-authenticated fixed arm; fresh safe artifacts, no model selection."""
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    folds = joint_folds(tuple(arrays["targets"].astype(str)), tuple(arrays["sources"].astype(str)), registration)
    if set(tables) != {f.target_fold for f in folds}:
        raise ValueError("Exactly one authenticated fold-local feature table per target fold required")
    if not isinstance(provenance.get("diagnostic_contract_sha256"), str) or len(provenance["diagnostic_contract_sha256"]) != 64:
        raise ValueError("Registered diagnostic contract hash required")
    output_dir = Path(output_dir)
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh diagnostic output directory required")
    output_dir.mkdir(mode=0o700)
    reports, weights, predictions, events = [], {}, {}, []
    for number, fold in enumerate(folds):
        aligned, receipt = role_features(tables[fold.target_fold], fold.fit_targets, fold.held_targets, arm=arm, fold_id=fold.target_fold)
        model = fit_cache_fold(arrays, fold, aligned, arm)
        report, output = evaluate_joint_fold(arrays, fold, aligned, model)
        prefix = f"fold_{number}_"
        report.update({"array_prefix": prefix, "feature_receipt": receipt,
            "target_transform": model.target_transform.provenance() if model.target_transform else None})
        model_arrays = {"coefficients": model.coefficients, "context_mean": model.context_mean, "context_scale": model.context_scale,
            "fit_targets": np.asarray(model.training_targets), "held_targets": np.asarray(fold.held_targets),
            "fit_source": np.asarray(fold.fit_source), "held_source": np.asarray(fold.held_source),
            "aligned_target_ids": np.asarray(aligned.target_ids), "aligned_values": aligned.values, "aligned_present": aligned.present,
            "feature_ids": np.asarray(tables[fold.target_fold].feature_ids),
            "feature_signature_json": np.asarray(json.dumps(aligned.signature)),
            "feature_receipt_json": np.asarray(json.dumps(receipt, sort_keys=True))}
        if model.target_transform:
            model_arrays.update({"target_mean": model.target_transform.mean, "target_scale": model.target_transform.scale,
                "target_transform_provenance_json": np.asarray(json.dumps(model.target_transform.provenance(), sort_keys=True))})
        weights.update({prefix + key: value for key, value in model_arrays.items()})
        predictions.update({prefix + key: value for key, value in output.items()})
        reports.append(report)
        event = {"fold_index": number, "held_source": fold.held_source, "target_fold": fold.target_fold, "metrics": report["metrics"]}
        events.append(event)
        if progress is not None:
            progress(event)
    groups = [g for report in reports for g in report["groups"]]
    target_rows = [t for report in reports for t in report["targets_report"]]
    if len({(t["source"], t["target"]) for t in target_rows}) != len(target_rows):
        raise ValueError("Repeated held-source target would bias aggregation")
    source_metrics = [{"source": source, "metrics": _aggregate([g for g in groups if g["source"] == source],
        [t for t in target_rows if t["source"] == source])} for source in sorted({g["source"] for g in groups})]
    weights["genes"] = predictions["genes"] = arrays["genes"]
    summary = {"schema": "public-joint-target-source-summary-v3", "status": "completed_unpromoted_diagnostic",
        "stage": "validation", "arm": arm, "policy": policy(), "folds": reports, "source_metrics": source_metrics,
        "aggregate": _aggregate(groups, target_rows), "provenance": provenance,
        "diagnostic_contract_sha256": provenance["diagnostic_contract_sha256"],
        "weights_sha256": save_arrays(output_dir / "weights.npz", weights),
        "predictions_sha256": save_arrays(output_dir / "predictions.npz", predictions),
        "exact_ntc_identity_verified": all(r["exact_ntc_identity_verified"] for r in reports),
        "exact_zero_effect_identity_verified": all(r["exact_zero_effect_identity_verified"] for r in reports),
        "count_emitter": False, "posttraining_performed": False, "promoted": False, "submission_performed": False,
        "hyperparameter_selection": False, "H1_treated_read": False, "RPE1_treated_read": False,
        "challenge_treated_used": False, "held_roles_used_for_fitting": False,
        "learned_null_calibration_evaluated": False, "permutation_pvalue_computed": False,
        "prior_public_development_exposure_acknowledged": True,
        "runtime": {"hostname": socket.gethostname(), "numpy_version": np.__version__, "device": "cpu"}}
    write_new(output_dir / "steps.jsonl", b"".join((json.dumps(e, sort_keys=True, allow_nan=False) + "\n").encode() for e in events))
    write_new(output_dir / "summary.json", (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return summary
