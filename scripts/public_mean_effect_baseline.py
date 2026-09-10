"""Registered public-only mean-effect diagnostics; no raw data, clipping or tuning."""
from __future__ import annotations
import argparse
import io
import json
import math
import os
from pathlib import Path
import socket
import numpy as np
from public_flow_conditioning import align_target_features, ablate_target_features, fit_target_transform
from train_public_flow_pilot import (PilotConfig, REPRESENTATION, authenticated_json, diagnostics,
    load_cache, load_go_table, sha256_file, validate_registration, write_new)

ARMS = ("control", "transfer", "true", "constant", "shuffled", "context")
SCHEMA = "public-mean-effect-diagnostic-contract-v1"

def policy():
    return {
        "arms": list(ARMS), "ridge_lambda": 10.0, "shuffle_seed": 20260910,
        "evaluation_seed": 20260910, "representation": REPRESENTATION,
        "fit_response": "per-matched-group treated mean minus control mean",
        "weights": "uniform source then target then matched group",
        "transfer": "same target; mean groups within source then mean sources; missing target forbidden",
        "ridge_design": "penalized intercept + train-only standardized GO and masks / sqrt(width) + train-only standardized control context / sqrt(width)",
        "ridge_objective": "sum normalized group weight * squared delta error + 10 * squared all-coefficient norm",
        "context_scale_floor": 0.1,
        "context_arm": "target block omitted; constant arm retains GO missingness masks",
        "inner_validation": "leave-one-training-source-out; overlap targets only; refit all transforms; descriptive only",
        "development": "fixed arms evaluated once after fitting; no selection; already exposed development context",
        "population_sampling": "per group min(32,controls,treated); identical seed and row sampling for every arm",
        "clipping": False, "posttraining": False, "challenge_2026_treated_allowed": False,
        "promotion": "not performed", "mask_preserving_ablation": True,
    }

def matrix(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not value.size or not np.isfinite(value).all():
        raise ValueError("Expected a finite nonempty matrix")
    return value

def group_means(split):
    means = []
    for group in range(len(split.targets)):
        control = split.control[split.control_group == group].astype(np.float64)
        treated = split.treated[split.group == group].astype(np.float64)
        if min(len(control), len(treated)) < 2:
            raise ValueError("Insufficient matched populations")
        means.append(treated.mean(0) - control.mean(0))
    return {"targets": split.targets, "sources": split.sources,
            "context": matrix(split.context), "delta": matrix(means)}

def subset(train, indices):
    return {key: tuple(value[i] for i in indices) if key in {"targets", "sources"} else value[indices]
            for key, value in train.items()}

def balanced_weights(targets, sources):
    if not targets or len(targets) != len(sources):
        raise ValueError("Aligned nonempty group axes required")
    source_set = set(sources)
    counts = {(s, t): sum(a == s and b == t for a, b in zip(sources, targets)) for s, t in zip(sources, targets)}
    nt = {s: len({t for a, t in zip(sources, targets) if a == s}) for s in source_set}
    return np.array([1 / (len(source_set) * nt[s] * counts[s, t]) for s, t in zip(sources, targets)])

def transfer_delta(train, targets):
    result = []
    for target in targets:
        per_source = []
        for source in sorted(set(train["sources"])):
            indices = [i for i, pair in enumerate(zip(train["targets"], train["sources"])) if pair == (target, source)]
            if indices:
                per_source.append(train["delta"][indices].mean(0))
        if not per_source:
            raise ValueError("Transfer target absent from fitting sources")
        result.append(np.mean(per_source, axis=0))
    return matrix(result)

def weighted_standardize(values, weights):
    values, weights = matrix(values), np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(values),) or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid normalized weights")
    mean = weights @ values
    return mean, np.maximum(np.sqrt(weights @ np.square(values - mean)), 0.1)

def ridge_fit(design, response, weights, regularization=10.0):
    x, y, w = matrix(design), matrix(response), np.asarray(weights, dtype=np.float64)
    if len(x) != len(y) or w.shape != (len(x),) or not np.isfinite(w).all() or np.any(w <= 0) or not np.isclose(w.sum(), 1):
        raise ValueError("Response/weight axis mismatch")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Positive fixed regularization required")
    xw, yw = x * np.sqrt(w[:, None]), y * np.sqrt(w[:, None])
    beta = xw.T @ np.linalg.solve(xw @ xw.T + regularization * np.eye(len(xw)), yw)
    if not np.isfinite(beta).all():
        raise ValueError("Nonfinite coefficients")
    return beta

def target_design(table, train_targets, eval_targets, mode, fold_id):
    train_ids = tuple(sorted(set(train_targets)))
    if not set(eval_targets) <= set(train_ids):
        raise ValueError("Only overlapping targets admitted")
    aligned = align_target_features(train_ids, [table])
    if mode != "true":
        if mode not in {"constant", "shuffled"}:
            raise ValueError("Unknown feature arm")
        aligned = ablate_target_features(aligned, mode="shuffle" if mode == "shuffled" else "constant",
            seed=20260910, train_ids=train_ids, fold_id=fold_id)
    transform = fit_target_transform(aligned, train_ids=train_ids, fold_id=fold_id)
    values = np.column_stack((transform.transform(aligned), aligned.present, aligned.unknown_target))
    values /= math.sqrt(values.shape[1])
    lookup = {target: i for i, target in enumerate(train_ids)}
    return values[[lookup[t] for t in train_targets]], values[[lookup[t] for t in eval_targets]], transform, aligned

def fit_predict(train, targets, contexts, table, arm, fold_id):
    """Fitting receives no recipient-treated values, only its control context."""
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    contexts = matrix(contexts)
    if contexts.shape != (len(targets), train["context"].shape[1]) or not set(targets) <= set(train["targets"]):
        raise ValueError("Recipient axis or overlapping-target admission mismatch")
    if arm == "control":
        return np.zeros((len(targets), train["delta"].shape[1])), {"arm": arm}, {}
    if arm == "transfer":
        return transfer_delta(train, targets), {"arm": arm}, {}
    w = balanced_weights(train["targets"], train["sources"])
    mean, scale = weighted_standardize(train["context"], w)
    xt = (train["context"] - mean) / scale / math.sqrt(train["context"].shape[1])
    xe = (contexts - mean) / scale / math.sqrt(train["context"].shape[1])
    meta = {"arm": arm, "ridge_lambda": 10.0, "intercept_penalized": True,
            "context_scale_floor": 0.1, "train_group_weights": w.tolist()}
    arrays = {"context_mean": mean, "context_scale": scale, "train_group_weights": w}
    if arm != "context":
        tt, te, transform, aligned = target_design(table, train["targets"], targets, arm, fold_id)
        xt, xe = np.column_stack((tt, xt)), np.column_stack((te, xe))
        meta.update(target_transform=transform.provenance(), target_ids=list(aligned.target_ids),
                    coverage_masks_preserved=True, go_covered_targets=int(aligned.present.sum()))
        arrays.update(target_mean=transform.mean, target_scale=transform.scale,
            target_ids=np.asarray(aligned.target_ids), target_values_after_ablation=aligned.values,
            target_present=aligned.present)
    xt, xe = np.column_stack((np.ones(len(xt)), xt)), np.column_stack((np.ones(len(xe)), xe))
    beta = ridge_fit(xt, train["delta"], w)
    arrays["coefficients"] = beta
    return xe @ beta, meta, arrays

def delta_diagnostics(predicted, observed):
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    if predicted.shape != observed.shape or not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("Invalid delta diagnostics")
    norm = np.linalg.norm(predicted) * np.linalg.norm(observed)
    return {"predicted_shift_rms": float(np.sqrt(np.mean(predicted ** 2))),
        "observed_shift_rms": float(np.sqrt(np.mean(observed ** 2))),
        "shift_dot_mean": float(np.mean(predicted * observed)),
        "shift_cosine": float(np.dot(predicted, observed) / norm) if norm > 1e-15 else None}

def aggregate_groups(groups):
    if not groups:
        return {}
    result = {}
    for metric in sorted(set(groups[0]) - {"group", "source", "target", "cells"}):
        source_values = []
        for source in sorted({g["source"] for g in groups}):
            target_values = []
            for target in sorted({g["target"] for g in groups if g["source"] == source}):
                values = [g[metric] for g in groups if (g["source"], g["target"]) == (source, target) and g[metric] is not None]
                if values:
                    target_values.append(float(np.mean(values)))
            if target_values:
                source_values.append(float(np.mean(target_values)))
        result[metric] = float(np.mean(source_values)) if source_values else None
    return result

def inner_validation(train, table, arm, fold_id):
    reports = []
    for source in sorted(set(train["sources"])):
        fit_ids = [i for i, s in enumerate(train["sources"]) if s != source]
        available = {train["targets"][i] for i in fit_ids}
        held_ids = [i for i, s in enumerate(train["sources"]) if s == source]
        admitted = [i for i in held_ids if train["targets"][i] in available]
        row = {"held_source": source, "fit_groups": fit_ids, "admitted_groups": admitted,
            "excluded_missing_training_target_groups": [i for i in held_ids if i not in admitted],
            "used_for_hyperparameter_or_arm_selection": False}
        if not admitted:
            row.update(status="no_overlapping_targets", metrics=None)
        else:
            fit, held = subset(train, fit_ids), subset(train, admitted)
            predicted, meta, _ = fit_predict(fit, held["targets"], held["context"], table, arm, fold_id + ":hold:" + source)
            groups = [{"group": i, "source": source, "target": held["targets"][j],
                "model_centroid_mse": float(np.mean((predicted[j] - held["delta"][j]) ** 2)),
                "control_centroid_mse": float(np.mean(held["delta"][j] ** 2)),
                **delta_diagnostics(predicted[j], held["delta"][j])} for j, i in enumerate(admitted)]
            row.update(status="evaluated", groups=groups, metrics=aggregate_groups(groups), fit=meta)
        reports.append(row)
    return reports

def evaluate(split, delta):
    if delta.shape != (len(split.targets), split.control.shape[1]):
        raise ValueError("Predicted delta axis mismatch")
    rng, groups, preds, labels, cis, tis = np.random.default_rng(20260910), [], [], [], [], []
    for group, (target, source) in enumerate(zip(split.targets, split.sources)):
        ci, ti = np.flatnonzero(split.control_group == group), np.flatnonzero(split.group == group)
        n = min(32, len(ci), len(ti))
        ci, ti = rng.choice(ci, n, replace=False), rng.choice(ti, n, replace=False)
        control, treated = split.control[ci], split.treated[ti]
        predicted = control.astype(np.float64) + delta[group]
        groups.append({"group": group, "source": source, "target": target, "cells": n,
            **diagnostics(predicted, control, treated), **delta_diagnostics(delta[group], treated.mean(0) - control.mean(0))})
        preds.append(predicted)
        labels.extend([group] * n)
        cis.extend(ci)
        tis.extend(ti)
    arrays = {"prediction": np.concatenate(preds), "group": np.asarray(labels, dtype=np.int64),
        "control_cache_rows": np.asarray(cis, dtype=np.int64), "treated_cache_rows": np.asarray(tis, dtype=np.int64), "delta": delta}
    return groups, aggregate_groups(groups), arrays

def save_arrays(path, arrays):
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("Object arrays forbidden")
    payload = io.BytesIO()
    np.savez_compressed(payload, **arrays)
    write_new(path, payload.getvalue())
    return sha256_file(path)

def validate_diagnostic_contract(path, sha256, bindings):
    contract = authenticated_json(path, sha256)
    if contract.get("schema") != SCHEMA or contract.get("policy") != policy():
        raise ValueError("Unrecognized or changed diagnostic policy")
    if contract.get("inputs") != bindings:
        raise ValueError("Diagnostic inputs differ from registration")
    if contract.get("prior_development_exposure_acknowledged") is not True or contract.get("challenge_2026_treated_allowed") is not False:
        raise ValueError("Development exposure and challenge exclusion must be explicit")
    modules = contract.get("code_sha256")
    names = {"public_mean_effect_baseline.py", "train_public_flow_pilot.py", "public_flow_conditioning.py", "build_go_target_features.py", "public_crispri_flow.py"}
    if not isinstance(modules, dict) or set(modules) != names:
        raise ValueError("Pinned code set required")
    for name, expected in modules.items():
        if sha256_file(Path(__file__).with_name(name)) != expected:
            raise ValueError("Diagnostic code differs from registration")
    return contract

def run_baseline(cache, table, arm, output_dir, *, provenance=None):
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh output directory required")
    output_dir.mkdir(mode=0o700)
    train = group_means(cache.train)
    inner = inner_validation(train, table, arm, cache.contract_sha256)
    delta, fitted, arrays = fit_predict(train, cache.dev.targets, cache.dev.context, table, arm, cache.contract_sha256 + ":all_train")
    train_delta, _, _ = fit_predict(train, train["targets"], train["context"], table, arm, cache.contract_sha256 + ":all_train")
    loss = float(np.sum(balanced_weights(train["targets"], train["sources"]) * np.mean((train_delta - train["delta"]) ** 2, axis=1)))
    print(json.dumps({"step": 1, "loss": loss}, allow_nan=False), flush=True)
    arrays.update(genes=np.asarray(cache.genes), train_group_targets=np.asarray(train["targets"]),
        train_group_sources=np.asarray(train["sources"]), train_mean_deltas=train["delta"])
    weights_sha = save_arrays(output_dir / "weights.npz", arrays)
    groups, aggregate, predictions = evaluate(cache.dev, delta)
    predictions.update(genes=np.asarray(cache.genes), group_targets=np.asarray(cache.dev.targets), group_sources=np.asarray(cache.dev.sources))
    prediction_sha = save_arrays(output_dir / "predictions.npz", predictions)
    rollout = {"kind_metrics": {"context": aggregate}}
    summary = {"schema": "public-mean-effect-baseline-summary-v1", "status": "completed_unpromoted_diagnostic",
        "stage": "pretrain", "arm": arm, "last_step": 1, "loss": loss,
        "weights_sha256": weights_sha, "predictions_sha256": prediction_sha,
        "representation": REPRESENTATION, "policy": policy(), "fit": fitted,
        "inner_source_validation": inner, "dev_groups": groups,
        "deployment_rollout_diagnostics": rollout, "provenance": provenance or {},
        "diagnostic_contract_sha256": (provenance or {}).get("diagnostic_contract_sha256"),
        "contract_sha256": cache.contract_sha256, "source_manifest_sha256": cache.source_manifest_sha256,
        "development_used_for_fit_or_selection": False, "development_previously_exposed": True,
        "zero_shot_scope": "same target, held recipient context; not unseen targets",
        "nonnegative_output_guaranteed": False, "count_emitter": False, "full_axis_submission_ready": False,
        "posttraining_performed": False, "mask_preserving_ablation_is_fully_target_blind": False,
        "runtime": {"hostname": socket.gethostname(), "numpy_version": np.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "device": "cpu"}}
    write_new(output_dir / "summary.json", (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    print(json.dumps({"step": 1, "deployment_rollout_diagnostics": rollout}, allow_nan=False), flush=True)
    return summary

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("cache", "target-features", "feature-receipt", "contract", "source-manifest")
    for name in (*names, "diagnostic-contract", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (*names, "diagnostic-contract"):
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--stage", choices=("pretrain",), default="pretrain")
    args = parser.parse_args(argv)
    host = socket.gethostname().split(".")[0]
    if host not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
        raise RuntimeError("This bounded CPU diagnostic requires an approved BioHPC compute host, never a login node")
    bindings = {name.replace("-", "_") + "_sha256": getattr(args, name.replace("-", "_") + "_sha256") for name in names}
    validate_diagnostic_contract(args.diagnostic_contract, args.diagnostic_contract_sha256, bindings)
    cache = load_cache(args.cache, args.cache_sha256, args.contract_sha256, args.source_manifest_sha256)
    registration = validate_registration(cache, PilotConfig(), args.contract, args.source_manifest)
    table = load_go_table(args.target_features, args.target_features_sha256, args.feature_receipt, args.feature_receipt_sha256, cache.train.targets)
    run_baseline(cache, table, args.arm, args.output_dir, provenance={**bindings, **registration,
        "diagnostic_contract_sha256": args.diagnostic_contract_sha256, "entrypoint_sha256": sha256_file(Path(__file__))})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
