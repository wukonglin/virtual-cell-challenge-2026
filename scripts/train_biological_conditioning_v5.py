"""Public-training-only ESM2/Lingshu/fusion conditioning screen, not a submission.

The Arc artifact has authenticated bytes but incomplete upstream model lineage.
Every family uses the same exact-symbol intersection, folds and tuning budget.
Fusion must beat ESM2 alone AND ESM2 with shuffled Lingshu to establish value.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import socket
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"

import numpy as np
import torch

from authenticate_esm2_target_features import load_authenticated_arc_feature_map, OFFICIAL_SPEC
from diagnose_lingshu_training_signal import TRAIN_KEYS, digest, folds, training_effects
from lingshu_conditioning_adapter import TrainingBasis, matrix
from validate_lingshu_conditioning import (CACHE_SHA, EMBEDDING_SHA, RECEIPT_SHA, candidates,
    complexity, paired_comparison, select_constant, write_json)

SEEDS = (20260951, 20260952, 20260953)
SHUFFLES = (20260960, 20260961, 20260962)
FAMILIES = ("esm2", "lingshu", "fusion")


class BlockBasis:
    """Fold-training-only equal-energy feature blocks, then shared-rank adapter."""

    def __init__(self, blocks, effects):
        if not blocks:
            raise ValueError("At least one feature block required")
        blocks = tuple(matrix(block) for block in blocks)
        if len({len(block) for block in blocks}) != 1:
            raise ValueError("Feature block row counts differ")
        self.centers = tuple(block.mean(axis=0) for block in blocks)
        self.scales = tuple(max(float(np.sqrt(np.mean(np.sum((block - center) ** 2, axis=1)))), 1e-12)
                            for block, center in zip(blocks, self.centers, strict=True))
        self.basis = TrainingBasis(self.transform(blocks), effects)

    def transform(self, blocks):
        if len(blocks) != len(self.centers):
            raise ValueError("Wrong feature blocks")
        normalized = []
        for block, center, scale in zip(blocks, self.centers, self.scales, strict=True):
            block = matrix(block)
            if block.shape[1] != len(center):
                raise ValueError("Feature block width changed")
            normalized.append((block - center) / scale)
        return np.concatenate(normalized, axis=1) / np.sqrt(len(blocks))

    def fit(self, spec):
        return BlockAdapter(self, self.basis.fit(spec))


@dataclass
class BlockAdapter:
    transform: BlockBasis
    core: object

    def predict_effect(self, blocks):
        return self.core.predict_effect(self.transform.transform(blocks))

    def save(self, path, metadata):
        core = self.core
        with Path(path).open("xb") as handle:
            np.savez_compressed(handle,
                schema=np.asarray("vcc-biological-conditioning-research-v5"),
                spec=np.asarray(json.dumps(asdict(core.spec), sort_keys=True)),
                block_widths=np.asarray([len(center) for center in self.transform.centers]),
                block_centers=np.concatenate(self.transform.centers), block_scales=self.transform.scales,
                feature_center=core.feature_center, feature_scale=core.feature_scale,
                feature_basis=core.feature_basis, effect_center=core.effect_center,
                effect_basis=core.effect_basis, coefficients=core.coefficients,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)), submission_allowed=np.asarray(False),
                independent_test=np.asarray(False), requires_authenticated_sidecars=np.asarray(True),
                feature_input=np.asarray("each_frozen_feature_vector_L2_normalized_before_stored_block_transforms"))


def subset(blocks, indices):
    return tuple(block[indices] for block in blocks)


def make_arms(esm2, lingshu, names):
    esm2, lingshu = matrix(esm2), matrix(lingshu)
    if len(esm2) != len(lingshu) or len(names) != len(esm2) or len(set(names)) != len(names):
        raise ValueError("Unique aligned target identities required")
    arms = {"esm2": (esm2,), "lingshu": (lingshu,), "fusion": (esm2, lingshu)}
    associations = {}
    for seed in SHUFFLES:
        perm = np.random.default_rng(seed).permutation(len(names))
        associations[str(seed)] = dict(zip(names.tolist(), names[perm].tolist(), strict=True))
        arms[f"esm2_shuffled_{seed}"] = (esm2[perm],)
        arms[f"lingshu_shuffled_{seed}"] = (lingshu[perm],)
        # Keep the genuine ESM2 signal fixed: test incremental Lingshu value.
        arms[f"fusion_shuffled_lingshu_{seed}"] = (esm2, lingshu[perm])
    return arms, associations


def select(blocks, effects, splits, mean_gain):
    specs = list(candidates(mean_gain))
    errors = np.zeros((len(specs), len(effects)))
    for heldout in splits:
        train = np.setdiff1d(np.arange(len(effects)), heldout)
        basis = BlockBasis(subset(blocks, train), effects[train])
        for index, spec in enumerate(specs):
            prediction = basis.fit(spec).predict_effect(subset(blocks, heldout))
            errors[index, heldout] = np.mean((prediction - effects[heldout]) ** 2, axis=1)
    scores = errors.mean(axis=1)
    best = min(range(len(specs)), key=lambda index: (scores[index], complexity(specs[index])))
    return specs[best], [{"spec": asdict(spec), "inner_mse": float(score)}
                         for spec, score in zip(specs, scores, strict=True)]


def promotion_decisions(errors):
    expected = set(FAMILIES) | {"constant", "zero"}
    for seed in SHUFFLES:
        expected |= {f"esm2_shuffled_{seed}", f"lingshu_shuffled_{seed}", f"fusion_shuffled_lingshu_{seed}"}
    if set(errors) != expected:
        raise ValueError("Missing or extra feature arms")
    shape = np.shape(errors["constant"])
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("Expected repeat-by-target errors")
    if any(np.shape(value) != shape or not np.isfinite(value).all() or (value < 0).any()
           for value in errors.values()):
        raise ValueError("Invalid or inconsistent errors")
    decisions = {}
    for family in FAMILIES:
        controls = ["constant", "zero"]
        controls += (["esm2"] + [f"fusion_shuffled_lingshu_{seed}" for seed in SHUFFLES]
                     if family == "fusion" else [f"{family}_shuffled_{seed}" for seed in SHUFFLES])
        comparisons = {key: paired_comparison(errors[family], errors[key]) for key in controls}
        core = [value for key, value in comparisons.items() if key != "zero"]
        checks = {
            "at_least_one_percent_vs_constant": comparisons["constant"]["relative_improvement"] >= 0.01,
            "every_repeat_beats_each_required_feature_control": all(all(v < 0 for v in item["repeat_mse_differences"]) for item in core),
            "each_paired_exploratory_interval_below_zero": all(item["exploratory_target_bootstrap_95pct"][1] < 0 for item in core),
            "beats_zero_effect": comparisons["zero"]["mean_mse_difference"] < 0,
        }
        decisions[family] = {"checks": checks, "comparisons": comparisons,
                             "conditioning_screen_passed": all(checks.values()), "submission_allowed": False}
    return decisions


def evaluate(arms, effects, names, *, seeds=SEEDS, outer_folds=5):
    errors = {name: np.zeros((len(seeds), len(names))) for name in (*arms, "constant", "zero")}
    records = []
    for repeat, seed in enumerate(seeds):
        for fold, heldout in enumerate(folds(len(names), outer_folds, seed)):
            train = np.setdiff1d(np.arange(len(names)), heldout)
            inner = folds(len(train), 3, seed + 100 + fold)
            mean_gain, baseline_scores = select_constant(effects[train], inner)
            errors["constant"][repeat, heldout] = np.mean((mean_gain * effects[train].mean(axis=0) - effects[heldout]) ** 2, axis=1)
            errors["zero"][repeat, heldout] = np.mean(effects[heldout] ** 2, axis=1)
            for name, blocks in arms.items():
                spec, scores = select(subset(blocks, train), effects[train], inner, mean_gain)
                adapter = BlockBasis(subset(blocks, train), effects[train]).fit(spec)
                prediction = adapter.predict_effect(subset(blocks, heldout))
                errors[name][repeat, heldout] = np.mean((prediction - effects[heldout]) ** 2, axis=1)
                records.append({"repeat_seed": seed, "outer_fold": fold, "arm": name,
                    "train_targets": names[train].tolist(), "heldout_targets": names[heldout].tolist(),
                    "inner_fold_targets": [names[train][part].tolist() for part in inner],
                    "selected_spec": asdict(spec), "candidate_inner_scores": scores,
                    "constant_inner_scores": baseline_scores, "outer_mse": float(errors[name][repeat, heldout].mean())})
            print(json.dumps({"completed_repeat": repeat, "outer_fold": fold}), flush=True)
    return {"decisions": promotion_decisions(errors), "folds": records,
            "mse": {name: float(value.mean()) for name, value in errors.items()},
            "per_repeat_mse": {name: value.mean(axis=1).tolist() for name, value in errors.items()}}, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cache", "esm2-features", "lingshu-embeddings", "lingshu-receipt", "output-dir"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    if socket.getfqdn() != "cbsuvlaminck3.biohpc.cornell.edu" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use BioHPC compute or a Slurm allocation, not a login node")
    torch.set_num_threads(4)
    hashes = {"cache": digest(args.cache), "lingshu": digest(args.lingshu_embeddings),
              "lingshu_receipt": digest(args.lingshu_receipt)}
    if tuple(hashes.values()) != (CACHE_SHA, EMBEDDING_SHA, RECEIPT_SHA):
        raise ValueError("Wrong immutable public cache or Lingshu features")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_names = ("train_biological_conditioning_v5.py", "validate_lingshu_conditioning.py",
                    "lingshu_conditioning_adapter.py", "diagnose_lingshu_training_signal.py",
                    "authenticate_esm2_target_features.py")
    code_hashes = {name: digest(Path(__file__).with_name(name)) for name in source_names}
    with np.load(args.cache, allow_pickle=False) as archive:
        # Only identities are needed to register the common feature panel.
        original_names = np.asarray(sorted(set(map(str, archive["train_targets"]))))
    feature_map, feature_descriptor = load_authenticated_arc_feature_map(args.esm2_features.resolve())
    with np.load(args.lingshu_embeddings, allow_pickle=False) as archive:
        language = dict(zip(archive["gene_names"].astype(str), archive["embeddings"].astype(np.float64), strict=True))
    names = np.asarray([name for name in original_names if name in feature_map and name in language])
    missing = sorted(set(original_names) - set(names))
    if len(names) != 145 or missing != ["TAZ"]:
        raise ValueError("The registered 145-target shared panel changed")
    contract = {"schema": "vcc-biological-conditioning-contract-v5", "source_sha256": hashes,
        "code_sha256": code_hashes, "esm2_descriptor": feature_descriptor,
        "esm2_artifact_sha256": OFFICIAL_SPEC.member_sha256,
        "esm2_provenance": "authenticated_Arc_supplied_opaque_artifact_upstream_model_lineage_unknown",
        "target_names": names.tolist(), "excluded_exact_symbols": missing,
        "alias_mapping": "none_TAZ_is_ambiguous", "outer_seeds": SEEDS, "outer_folds": 5,
        "inner_folds": 3, "shuffle_seeds": SHUFFLES,
        "candidate_specs": [asdict(spec) for spec in candidates(1.0)], "mean_gains": [0, 0.5, 1],
        "feature_normalization": "L2_each_vector_then_fold_train_center_and_row_energy_per_block_equal_block_weight",
        "fusion_negative_control": "genuine_ESM2_fixed_with_Lingshu_associations_permuted",
        "same_total_input_rank_for_all_families": True,
        "screen_rule": "at_least_1pct_vs_constant; every_repeat_and_paired95pct_vs_all_controls; fusion_also_vs_ESM2",
        "read_cache_arrays": [*TRAIN_KEYS, "gene_names"], "validation_expression_read": False,
        "challenge_treated_used": False, "independent_test": False, "submission_allowed": False,
        "caveats": ["Previously selected 512-gene training axis and reused control cells limit independence.",
                    "These target CV results are exploratory development, not whole-family or whole-context confirmation.",
                    "Current Lingshu lineage contains14 targets from globally excluded V7 clusters; V7 test is not authorized by this experiment.",
                    "Feature dimensions do not prove an ESM2 model identity; no feature redistribution is permitted.",
                    "No context, cell-distribution, full-gene or official challenge score is evaluated."]}
    write_json(args.output_dir / "contract.json", contract)
    started = time.monotonic()
    with np.load(args.cache, allow_pickle=False) as archive:
        loaded_names, all_effects, counts = training_effects(archive)
        genes = archive["gene_names"].astype(str)
    index = {name: i for i, name in enumerate(loaded_names)}
    selected = np.asarray([index[name] for name in names])
    effects = all_effects[selected]
    esm2 = np.stack([feature_map[name].numpy().astype(np.float64) for name in names])
    lingshu = np.stack([language[name] for name in names])
    esm2 /= np.linalg.norm(esm2, axis=1)[:, None]
    lingshu /= np.linalg.norm(lingshu, axis=1)[:, None]
    arms, associations = make_arms(esm2, lingshu, names)
    result, errors = evaluate(arms, effects, names)
    inner = folds(len(names), 3, 20260970)
    mean_gain, _ = select_constant(effects, inner)
    exports = {}
    for family in FAMILIES:
        spec, _ = select(arms[family], effects, inner, mean_gain)
        adapter = BlockBasis(arms[family], effects).fit(spec)
        path = args.output_dir / f"{family}.research-only.npz"
        adapter.save(path, {"family": family, "gene_names": genes.tolist(), "training_targets": names.tolist(),
            "ordered_blocks": ["esm2", "lingshu"] if family == "fusion" else [family],
            "input_preprocessing": "L2_normalize_each_frozen_feature_vector_before_block_transform",
            "contract_sha256": digest(args.output_dir / "contract.json"),
            "conditioning_screen_passed": result["decisions"][family]["conditioning_screen_passed"],
            "context_scope": "K562_only", "normalization": "log1p_CP10k_fixed_18077_gene_library_axis"})
        exports[family] = {"file": path.name, "sha256": digest(path), "spec": asdict(spec)}
    with (args.output_dir / "out_of_fold_errors.npz").open("xb") as handle:
        np.savez_compressed(handle, targets=names, **errors)
    result.update(schema="vcc-biological-conditioning-result-v5", execution_status="completed",
        contract_sha256=digest(args.output_dir / "contract.json"), feature_associations=associations,
        exports=exports, oof_sha256=digest(args.output_dir / "out_of_fold_errors.npz"),
        elapsed_seconds=time.monotonic() - started, target_count=len(names), gene_count=len(genes),
        training_cell_count=sum(counts[index] for index in selected), excluded_exact_symbols=missing,
        validation_expression_read=False, challenge_treated_used=False, independent_test=False, submission_allowed=False)
    write_json(args.output_dir / "result.json", result)
    print(json.dumps({"mse": result["mse"], "decisions": result["decisions"],
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
