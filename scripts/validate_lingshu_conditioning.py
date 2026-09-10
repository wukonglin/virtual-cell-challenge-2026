"""Preregistered train-only nested-CV screen of a regularized Lingshu adapter.

Run on BioHPC compute, not an HPC login node. Only the authenticated cached
public K562 TRAIN_KEYS are materialized. No val_* or challenge arrays are read.
This is an exploratory conditioning screen, NOT scDFM, cell generation, a
replacement for the fixed public MSE/MMD gates, or a challenge submission.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np

from diagnose_lingshu_training_signal import TRAIN_KEYS, digest, folds, training_effects
from lingshu_conditioning_adapter import AdapterSpec, TrainingBasis

SEEDS = (20260910, 20260911, 20260912)
SHUFFLE_SEEDS = (20260920, 20260921, 20260922)
INPUT_RANKS = (8, 32)
EFFECT_RANKS = (4, 16)
PENALTIES = (0.1, 1.0, 10.0)
MEAN_GAINS = (0.0, 0.5, 1.0)
RESIDUAL_GAINS = (0.0, 0.25, 1.0)
CACHE_SHA = "39d04694b579776e8c5fa9d73097287de2927629e1a01e19dbfcb2fc26341e77"
EMBEDDING_SHA = "66d9f299212472102d292cb4e94c69a39f5ed700cd407b4f0fa5ac51d438f4cb"
RECEIPT_SHA = "8918387497f21c67de5cb85423a2e3a4000f6a46c14f897b418b013a4c9bf46a"


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def candidates(mean_gain):
    # The exact constant arm occurs ONCE; no rank/penalty duplicate testing.
    yield AdapterSpec(8, 4, 10.0, mean_gain, 0.0)
    for p in INPUT_RANKS:
        for q in EFFECT_RANKS:
            for penalty in PENALTIES:
                for gain in RESIDUAL_GAINS[1:]:
                    yield AdapterSpec(p, q, penalty, mean_gain, gain)


def complexity(spec):
    return (spec.residual_gain, spec.input_rank, spec.effect_rank, -spec.penalty)


def select_constant(effects, splits):
    errors = {gain: [] for gain in MEAN_GAINS}
    for heldout in splits:
        train = np.setdiff1d(np.arange(len(effects)), heldout)
        mean = effects[train].mean(axis=0)
        for gain in MEAN_GAINS:
            errors[gain].extend(np.mean((gain * mean - effects[heldout]) ** 2, axis=1).tolist())
    scores = {gain: float(np.mean(values)) for gain, values in errors.items()}
    return min(MEAN_GAINS, key=lambda gain: (scores[gain], gain)), scores


def select_adapter(features, effects, splits, mean_gain):
    specs = list(candidates(mean_gain))
    errors = np.zeros((len(specs), len(effects)))
    for heldout in splits:
        train = np.setdiff1d(np.arange(len(effects)), heldout)
        basis = TrainingBasis(features[train], effects[train])
        for index, spec in enumerate(specs):
            prediction = basis.fit(spec).predict_effect(features[heldout])
            # Score ALL 512 genes, not only the retained effect PCs.
            errors[index, heldout] = np.mean((prediction - effects[heldout]) ** 2, axis=1)
    means = errors.mean(axis=1)
    best = min(range(len(specs)), key=lambda i: (means[i], complexity(specs[i])))
    return specs[best], [{"spec": asdict(spec), "inner_mse": float(score)}
                         for spec, score in zip(specs, means, strict=True)]


def feature_arms(features, names):
    # Shuffle only the SAME training-target feature set; no new covariate pool.
    arms = {"true": features}
    mappings = {"true": dict(zip(names.tolist(), names.tolist(), strict=True))}
    for seed in SHUFFLE_SEEDS:
        permutation = np.random.default_rng(seed).permutation(len(names))
        key = f"shuffled_{seed}"
        arms[key] = features[permutation]
        mappings[key] = dict(zip(names.tolist(), names[permutation].tolist(), strict=True))
    return arms, mappings


def paired_comparison(errors, baseline):
    # Average repeated losses per target FIRST: n=146, NOT n=438 independent rows.
    difference = (errors - baseline).mean(axis=0)
    rng = np.random.default_rng(20260930)
    bootstrap = difference[rng.integers(0, len(difference), size=(5000, len(difference)))].mean(axis=1)
    return {"mean_mse_difference": float(difference.mean()),
            "relative_improvement": float(1 - errors.mean() / max(float(baseline.mean()), 1e-30)),
            "repeat_mse_differences": (errors.mean(axis=1) - baseline.mean(axis=1)).tolist(),
            "exploratory_target_bootstrap_95pct": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "bootstrap_unit": "target_after_averaging_repeated_out_of_fold_losses"}


def decision(errors):
    expected = {"true", "constant", "zero", *(f"shuffled_{seed}" for seed in SHUFFLE_SEEDS)}
    if set(errors) != expected:
        raise ValueError("All registered true, zero, constant, and shuffled arms are required")
    shape = np.shape(errors["true"])
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 2:
        raise ValueError("Expected repeat-by-target error matrices")
    errors = {name: np.asarray(value, dtype=np.float64) for name, value in errors.items()}
    if any(value.shape != shape or not np.isfinite(value).all() or (value < 0).any()
           for value in errors.values()):
        raise ValueError("All errors must be aligned, finite and nonnegative")
    comparisons = {name: paired_comparison(errors["true"], value)
                   for name, value in errors.items() if name != "true"}
    baseline = comparisons["constant"]
    checks = {
        "at_least_one_percent_better_than_optimized_constant": baseline["relative_improvement"] >= 0.01,
        "constant_paired_exploratory_interval_below_zero": baseline["exploratory_target_bootstrap_95pct"][1] < 0,
        "better_than_constant_in_each_repeat": all(v < 0 for v in baseline["repeat_mse_differences"]),
        "better_than_zero_effect": comparisons["zero"]["mean_mse_difference"] < 0,
        "better_than_each_shuffle_in_each_repeat": all(
            all(v < 0 for v in value["repeat_mse_differences"])
            for key, value in comparisons.items() if key.startswith("shuffled_")),
        "each_shuffle_paired_exploratory_interval_below_zero": all(
            value["exploratory_target_bootstrap_95pct"][1] < 0
            for key, value in comparisons.items() if key.startswith("shuffled_")),
    }
    return {"checks": checks, "conditioning_screen_passed": all(checks.values()),
            "next_stage": "eligible_for_separate_context_distribution_pilot" if all(checks.values())
                          else "blocked_pending_better_public_conditioning_signal",
            "comparisons": comparisons, "submission_allowed": False,
            "official_public_quality_gates_changed": False}


def evaluate(features, effects, names, *, seeds=SEEDS, outer_folds=5):
    arms, mappings = feature_arms(features, names)
    errors = {name: np.zeros((len(seeds), len(names))) for name in (*arms, "constant", "zero")}
    records = []
    for repeat, seed in enumerate(seeds):
        for fold, heldout in enumerate(folds(len(names), outer_folds, seed)):
            train = np.setdiff1d(np.arange(len(names)), heldout)
            inner = folds(len(train), 3, seed + 100 + fold)
            mean_gain, constant_scores = select_constant(effects[train], inner)
            constant = mean_gain * effects[train].mean(axis=0)
            errors["constant"][repeat, heldout] = np.mean((constant - effects[heldout]) ** 2, axis=1)
            errors["zero"][repeat, heldout] = np.mean(effects[heldout] ** 2, axis=1)
            for mode, values in arms.items():
                spec, candidates_scored = select_adapter(values[train], effects[train], inner, mean_gain)
                adapter = TrainingBasis(values[train], effects[train]).fit(spec)
                prediction = adapter.predict_effect(values[heldout])
                errors[mode][repeat, heldout] = np.mean((prediction - effects[heldout]) ** 2, axis=1)
                records.append({"repeat_seed": seed, "outer_fold": fold, "arm": mode,
                                "train_targets": names[train].tolist(), "heldout_targets": names[heldout].tolist(),
                                "inner_fold_targets": [names[train][part].tolist() for part in inner],
                                "selected_spec": asdict(spec), "constant_inner_scores": constant_scores,
                                "candidate_inner_scores": candidates_scored,
                                "outer_mse": float(errors[mode][repeat, heldout].mean())})
            print(json.dumps({"completed_repeat": repeat, "outer_fold": fold}), flush=True)
    return {"decision": decision(errors), "mse": {key: float(value.mean()) for key, value in errors.items()},
            "per_repeat_mse": {key: value.mean(axis=1).tolist() for key, value in errors.items()},
            "folds": records, "feature_associations": mappings}, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "embeddings", "embedding-receipt", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    hostname = socket.getfqdn()
    if hostname != "cbsuvlaminck3.biohpc.cornell.edu" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use the authorized BioHPC compute server or a Slurm allocation, not a login node")
    hashes = {"cache": digest(args.cache), "embeddings": digest(args.embeddings),
              "embedding_receipt": digest(args.embedding_receipt),
              "validator": digest(__file__),
              "adapter": digest(Path(__file__).with_name("lingshu_conditioning_adapter.py")),
              "train_loader": digest(Path(__file__).with_name("diagnose_lingshu_training_signal.py"))}
    if (hashes["cache"], hashes["embeddings"], hashes["embedding_receipt"]) != (CACHE_SHA, EMBEDDING_SHA, RECEIPT_SHA):
        raise ValueError("Only the authenticated expanded public cache and genuine frozen embeddings are allowed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    contract = {"schema": "vcc-lingshu-conditioning-screen-contract-v4", "source_sha256": hashes,
                "outer_seeds": SEEDS, "outer_folds": 5, "inner_folds": 3, "shuffle_seeds": SHUFFLE_SEEDS,
                "input_ranks": INPUT_RANKS, "effect_ranks": EFFECT_RANKS, "penalties": PENALTIES,
                "mean_gains": MEAN_GAINS, "residual_gains": RESIDUAL_GAINS,
                "selection": "inner_target_cv_all_genes_mse_then_lower_gain_lower_rank_stronger_penalty",
                "constant_selection": "inner_cv_mean_gain_shared_with_all_feature_arms",
                "feature_normalization": "training_center_then_unit_mean_squared_row_norm_no_whitening",
                "ridge_system": "X_pc_transpose_X_pc / n_train + penalty * identity",
                "screen_rule": ["at_least_1pct_relative_mse_improvement_over_optimized_constant",
                                "true_better_than_constant_and_each_of_three_shuffles_in_every_repeat",
                                "paired_exploratory_95pct_interval_upper_bound_below_zero_vs_constant_and_every_shuffle",
                                "true_aggregate_mse_lower_than_zero_effect"],
                "bootstrap": "5000_seed20260930_resamples_of_targets_after_averaging_repeated_losses",
                "read_cache_arrays": [*TRAIN_KEYS, "gene_names"], "validation_expression_read": False,
                "challenge_treated_used": False, "independent_test": False,
                "hostname": hostname, "cpu_threads": 4, "gpu_used": False,
                "caveats": ["512-gene axis was selected on original public training targets before these CV folds.",
                            "Cached sampled controls may recur across target groups; targets/pathways are not independent.",
                            "Repeated target CV and bootstrap intervals are exploratory, not independent final tests.",
                            "Three feature permutations are negative controls, not a permutation significance test.",
                            "Only K562 training targets; no claim of unseen-context or cell-distribution generalization.",
                            "No change to fixed public centroid MSE/MMD/clipping gates; no challenge score is measured."]}
    write_json(args.output_dir / "contract.json", contract)
    started = time.monotonic()
    with np.load(args.cache, allow_pickle=False) as archive:
        names, effects, counts = training_effects(archive)
        gene_names = archive["gene_names"].astype(str)
    if len(gene_names) != effects.shape[1] or len(set(gene_names)) != len(gene_names):
        raise ValueError("Invalid ordered gene axis")
    with np.load(args.embeddings, allow_pickle=False) as archive:
        lookup = dict(zip(archive["gene_names"].astype(str), archive["embeddings"].astype(np.float64), strict=True))
    features = np.stack([lookup[name] for name in names])
    features /= np.linalg.norm(features, axis=1)[:, None]
    result, errors = evaluate(features, effects, names)
    # A final RESEARCH-ONLY fit; selected without any existing public val_* rows.
    inner = folds(len(names), 3, 20260940)
    mean_gain, _ = select_constant(effects, inner)
    spec, scores = select_adapter(features, effects, inner, mean_gain)
    adapter = TrainingBasis(features, effects).fit(spec)
    adapter_path = args.output_dir / "adapter.research-only.npz"
    adapter.save_research_only(adapter_path, metadata={"gene_names": gene_names.tolist(),
        "training_targets": names.tolist(), "source_sha256": hashes,
        "contract_sha256": digest(args.output_dir / "contract.json"),
        "conditioning_screen_passed": result["decision"]["conditioning_screen_passed"],
        "context_scope": "K562_only", "feature_input": "L2_normalized_frozen_Lingshu3584",
        "normalization": "log1p_CP10k_using_fixed_18077_gene_library_axis"})
    with (args.output_dir / "out_of_fold_errors.npz").open("xb") as handle:
        np.savez_compressed(handle, targets=names, **errors)
    result.update({"schema": "vcc-lingshu-conditioning-screen-v4", "execution_status": "completed",
                   "contract_sha256": digest(args.output_dir / "contract.json"),
                   "elapsed_seconds": time.monotonic() - started, "target_count": len(names),
                   "training_cell_count": sum(counts), "gene_count": effects.shape[1],
                   "final_research_spec": asdict(spec), "final_inner_scores": scores,
                   "adapter_sha256": digest(adapter_path),
                   "oof_sha256": digest(args.output_dir / "out_of_fold_errors.npz"),
                   "validation_expression_read": False, "challenge_treated_used": False,
                   "independent_test": False, "submission_allowed": False,
                   "lingshu_benefit_claim": "exploratory_screen_only" if result["decision"]["conditioning_screen_passed"]
                                           else "not_established"})
    write_json(args.output_dir / "result.json", result)
    print(json.dumps({"output": str(args.output_dir), "mse": result["mse"], "decision": result["decision"]}), flush=True)


if __name__ == "__main__":
    main()
