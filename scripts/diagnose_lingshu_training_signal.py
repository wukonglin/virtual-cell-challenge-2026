"""Exploratory target-level nested CV using ONLY cached public training rows.

This is not the scDFM model, an independent final test, or a submission route.
The fixed gene axis was selected on public training cells before this diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Bound CPU usage before NumPy imports BLAS. This script runs on BioHPC compute.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np

SEED = 20260909
PENALTIES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
TRAIN_KEYS = ("train_x", "train_control", "train_targets", "train_contexts")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def training_effects(archive):
    """The explicit key allowlist never materializes a val_* member."""
    data = {key: archive[key] for key in TRAIN_KEYS}
    if set(map(str, data["train_contexts"])) != {"K562"}:
        raise ValueError("Only public K562 training rows are allowed")
    names = np.asarray(sorted(set(map(str, data["train_targets"]))))
    effects, counts = [], []
    for name in names:
        rows = data["train_targets"] == name
        treated = data["train_x"][rows].astype(np.float64)
        control = data["train_control"][rows].astype(np.float64)
        effects.append(treated.mean(axis=0) - control.mean(axis=0))
        counts.append(int(rows.sum()))
    values = np.asarray(effects)
    if len(names) < 15 or not np.isfinite(values).all():
        raise ValueError("Too few groups or nonfinite training effects")
    return names, values, counts


def folds(count, nfold, seed):
    permutation = np.random.default_rng(seed).permutation(count)
    return [np.sort(part) for part in np.array_split(permutation, nfold)]


def normalize(train, query):
    center = train.mean(axis=0)
    scale = max(float(np.sqrt(np.mean((train - center) ** 2))), 1e-12)
    return (train - center) / scale, (query - center) / scale


def kernel(left, right, kind):
    dot = left @ right.T / left.shape[1]
    if kind == "linear":
        return dot
    if kind == "rbf":
        distance = np.maximum(0.0, np.mean(left ** 2, axis=1)[:, None]
                              + np.mean(right ** 2, axis=1)[None, :] - 2 * dot)
        return np.exp(-distance)  # fixed gamma 1 / dimension; no tuning
    raise ValueError("Unknown kernel")


def fit_predict(features, effects, query, kind, penalty):
    """All fitted transforms and the intercept use these training rows only."""
    train, test = normalize(features, query)
    train_kernel = kernel(train, train, kind)
    query_kernel = kernel(test, train, kind)
    # Center the feature-space kernel using training rows, including for RBF.
    colmean = train_kernel.mean(axis=0)
    overall = float(colmean.mean())
    train_kernel = train_kernel - colmean[None, :] - colmean[:, None] + overall
    query_kernel = query_kernel - query_kernel.mean(axis=1)[:, None] - colmean[None, :] + overall
    mean_effect = effects.mean(axis=0)
    alpha = np.linalg.solve(train_kernel + len(features) * penalty * np.eye(len(features)),
                            effects - mean_effect)
    return mean_effect + query_kernel @ alpha, mean_effect + train_kernel @ alpha


def choose_penalty(features, effects, kind, seed):
    splits = folds(len(features), 3, seed)
    losses = {penalty: [] for penalty in PENALTIES}
    for heldout in splits:
        train = np.setdiff1d(np.arange(len(features)), heldout)
        for penalty in PENALTIES:
            prediction, _ = fit_predict(features[train], effects[train], features[heldout], kind, penalty)
            losses[penalty].extend(np.mean((prediction - effects[heldout]) ** 2, axis=1).tolist())
    means = {penalty: float(np.mean(values)) for penalty, values in losses.items()}
    # Exact ties favor stronger regularization, not an extra expression criterion.
    selected = min(PENALTIES, key=lambda penalty: (means[penalty], -penalty))
    return selected, means


def error_summary(errors, baseline, seed):
    difference = errors - baseline
    rng = np.random.default_rng(seed)
    bootstrap = difference[rng.integers(0, len(errors), size=(5000, len(errors)))].mean(axis=1)
    return {"mse": float(errors.mean()), "ratio_to_fold_global_mean": float(errors.mean() / baseline.mean()),
            "difference_to_fold_global_mean": float(difference.mean()),
            "exploratory_paired_target_bootstrap_95pct_difference": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "fraction_targets_better_than_fold_global_mean": float(np.mean(difference < 0))}


def evaluate(features_by_mode, effects, names):
    outer = folds(len(names), 5, SEED)
    mean_errors = np.zeros(len(names))
    zero_errors = np.mean(effects ** 2, axis=1)
    predictions = {f"{mode}_{kind}": np.zeros_like(effects)
                   for mode in features_by_mode for kind in ("linear", "rbf")}
    records = {key: [] for key in predictions}
    for fold, heldout in enumerate(outer):
        train = np.setdiff1d(np.arange(len(names)), heldout)
        mean_errors[heldout] = np.mean((effects[heldout] - effects[train].mean(axis=0)) ** 2, axis=1)
        for mode, features in features_by_mode.items():
            for kind in ("linear", "rbf"):
                key = f"{mode}_{kind}"
                penalty, scores = choose_penalty(features[train], effects[train], kind, SEED + 100 + fold)
                prediction, fitted = fit_predict(features[train], effects[train], features[heldout], kind, penalty)
                predictions[key][heldout] = prediction
                weak_prediction, weak_fitted = fit_predict(features[train], effects[train], features[heldout], kind, PENALTIES[0])
                records[key].append({"outer_fold": fold, "heldout_targets": names[heldout].tolist(),
                                     "train_target_count": len(train), "selected_penalty": penalty,
                                     "inner_cv_mse_by_penalty": scores,
                                     "outer_mse": float(np.mean((prediction - effects[heldout]) ** 2)),
                                     "fitted_train_mse": float(np.mean((fitted - effects[train]) ** 2)),
                                     "weakest_penalty_train_mse": float(np.mean((weak_fitted - effects[train]) ** 2)),
                                     "weakest_penalty_outer_mse": float(np.mean((weak_prediction - effects[heldout]) ** 2))})
        print(json.dumps({"completed_outer_fold": fold}), flush=True)
    arms = {}
    for key, prediction in predictions.items():
        errors = np.mean((prediction - effects) ** 2, axis=1)
        arms[key] = error_summary(errors, mean_errors, SEED + 900)
        arms[key]["folds"] = records[key]
        arms[key]["per_target_mse"] = dict(zip(names.tolist(), errors.tolist(), strict=True))
    comparisons = {}
    for kind in ("linear", "rbf"):
        true_error = np.mean((predictions[f"true_{kind}"] - effects) ** 2, axis=1)
        shuffled_error = np.mean((predictions[f"shuffled_{kind}"] - effects) ** 2, axis=1)
        comparison = error_summary(true_error, shuffled_error, SEED + 901)
        comparisons[kind] = {"true_minus_shuffled_mse": comparison["difference_to_fold_global_mean"],
                             "exploratory_paired_target_bootstrap_95pct_difference": comparison["exploratory_paired_target_bootstrap_95pct_difference"]}
    return {"zero_effect_mse": float(zero_errors.mean()), "fold_global_mean_effect_mse": float(mean_errors.mean()),
            "arms": arms, "true_vs_shuffled": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new diagnostic output; no overwrites")
    with np.load(args.cache, allow_pickle=False) as archive:
        names, effects, counts = training_effects(archive)
    with np.load(args.embeddings, allow_pickle=False) as archive:
        feature_names = archive["gene_names"].astype(str)
        features = archive["embeddings"].astype(np.float64)
    norm = np.linalg.norm(features, axis=1)
    if len(set(feature_names)) != len(feature_names) or not np.isfinite(features).all() or not (norm > 0).all():
        raise ValueError("Invalid frozen embeddings")
    features /= norm[:, None]
    mapping = dict(zip(feature_names.tolist(), features, strict=True))
    available = sorted(mapping)
    shuffled = dict(zip(available, np.random.default_rng(SEED).permutation(available), strict=True))
    feature_modes = {"true": np.stack([mapping[name] for name in names]),
                     "shuffled": np.stack([mapping[shuffled[name]] for name in names])}
    result = evaluate(feature_modes, effects, names)
    result.update({"schema": "vcc-lingshu-public-training-signal-diagnostic-v1", "seed": SEED,
                   "read_cache_arrays": list(TRAIN_KEYS), "target_count": len(names),
                   "training_cell_count": sum(counts), "gene_count": effects.shape[1],
                   "target_cell_counts": dict(zip(names.tolist(), counts, strict=True)),
                   "feature_dimension": features.shape[1], "available_frozen_feature_count": len(available),
                   "outer_folds": 5, "inner_folds": 3, "ridge_penalty_grid": list(PENALTIES),
                   "ridge_system": "centered_kernel + n_train * penalty * identity",
                   "rbf_gamma": "1 / feature_dimension_after_fold_training_scalar_rms_normalization",
                   "fit_scope": "outer_training_targets_and_inner_training_targets_only",
                   "shuffle_scope": "same_seed_permutation_of_all_467_frozen_feature_associations_as_v2",
                   "validation_expression_read": False, "challenge_expression_read": False,
                   "challenge_treated_used": False, "independent_final_test": False,
                   "caveats": ["The fixed 512-gene axis was selected using the original public training cells, including targets later withheld by this diagnostic CV.",
                               "Matched control cells can recur across target groups, so target-held-out groups need not be independent at the source-cell level.",
                               "Bootstrap intervals are exploratory and assume exchangeable targets; biological pathways may violate this assumption.",
                               "Only one target partition and one shuffled-feature assignment were tested.",
                               "This ridge probe is not scDFM, does not test cell-level MMD, and cannot establish challenge improvement."],
                   "source_sha256": {"cache": digest(args.cache), "embeddings": digest(args.embeddings),
                                      "script": digest(__file__)}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "zero_effect_mse": result["zero_effect_mse"],
                      "global_mean_mse": result["fold_global_mean_effect_mse"],
                      "arms": {key: {k: v for k, v in arm.items() if k not in ("folds", "per_target_mse")}
                               for key, arm in result["arms"].items()}}), flush=True)


if __name__ == "__main__":
    main()
