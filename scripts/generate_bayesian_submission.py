#!/usr/bin/env python3
"""Generate a VCC 2026 raw-count candidate from a Bayesian effect prior.

Each predicted cell starts from a real non-targeting A/B/C control cell.  Only
the sparse set of genes selected by the fitted prior is changed, using binomial
thinning for down-regulation and a control-smoothed Poisson increment for
up-regulation.  This preserves context, library-depth variation, and most
single-cell heterogeneity while producing non-negative integer raw counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


CONTEXTS = ("A", "B", "C")
MAX_OFFICIAL_NNZ = 4_750_000_000
SAFE_INT32_NNZ = 2_140_000_000
STATE_EFFECT_SPACE = "log1p-target-sum-normalized-arithmetic-pseudobulk-delta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prior",
        type=Path,
        default=Path("artifacts/bayesian_prior_h100_v0.npz"),
    )
    parser.add_argument(
        "--controls-dir", type=Path, default=Path("dataset/controls")
    )
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=Path("artifacts/bayesian_baseline_v0.h5ad"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/bayesian_baseline_v0.json"),
    )
    parser.add_argument("--cells-per-group", type=int, default=400)
    parser.add_argument("--max-base-genes", type=int, default=5_780)
    parser.add_argument("--response-sigma", type=float, default=0.15)
    parser.add_argument("--upreg-control-smoothing", type=float, default=0.10)
    parser.add_argument(
        "--effect-baseline-target-sum",
        type=float,
        default=10_000.0,
        help=(
            "Target sum used to interpret log1p-space prior effects. Keep 10000 "
            "for the Bayesian CP10K prior; use 50000 for VCC-aligned STATE effects."
        ),
    )
    parser.add_argument("--minimum-fold", type=float, default=0.55)
    parser.add_argument("--maximum-fold", type=float, default=1.82)
    parser.add_argument("--target-remaining-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--target-limit",
        type=int,
        default=0,
        help="non-production smoke mode; 0 means all official targets",
    )
    parser.add_argument(
        "--compression", choices=("lzf", "gzip", "none"), default="lzf"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def deterministic_seed(base: int, context: str, target: str) -> int:
    token = f"{base}|{context}|{target}".encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return value % (2**63 - 1)


def stratified_control_indices(
    obs: pd.DataFrame,
    n_cells: int,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = obs.groupby("ntc_id", observed=True, sort=True).indices
    keys = sorted(groups)
    base, remainder = divmod(n_cells, len(keys))
    selected: list[int] = []
    for position, key in enumerate(keys):
        count = base + int(position < remainder)
        candidates = np.asarray(groups[key], dtype=np.int64)
        if count > len(candidates):
            raise ValueError(f"NTC stratum {key!r} has too few cells")
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def cap_control_sparsity(
    matrix: sparse.csr_matrix,
    max_genes: int,
    seed: int,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Cap row nnz while exactly preserving each raw library size.

    Removed low-count reads are redistributed over the 128 most abundant
    retained genes.  This keeps the full candidate safely below scipy's signed
    int32 CSR boundary without the severe 300-gene compression used only by the
    earlier schema smoke test.
    """
    matrix = matrix.tocsr(copy=False)
    rng = np.random.default_rng(seed)
    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    indptr = np.zeros(matrix.shape[0] + 1, dtype=np.int64)
    total_redistributed = 0
    capped_cells = 0
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row : row + 2]
        values = np.rint(matrix.data[start:stop]).astype(np.int32, copy=True)
        indices = matrix.indices[start:stop].astype(np.int32, copy=True)
        if len(values) > max_genes:
            capped_cells += 1
            keep = np.argpartition(values, -max_genes)[-max_genes:]
            removed_total = int(values.sum(dtype=np.int64) - values[keep].sum(dtype=np.int64))
            values = values[keep]
            indices = indices[keep]
            if removed_total:
                total_redistributed += removed_total
                n_receivers = min(128, len(values))
                receivers = np.argpartition(values, -n_receivers)[-n_receivers:]
                probabilities = values[receivers].astype(np.float64)
                probabilities /= probabilities.sum()
                values[receivers] += rng.multinomial(removed_total, probabilities).astype(
                    np.int32
                )
        order = np.argsort(indices)
        values = values[order]
        indices = indices[order]
        positive = values > 0
        data_parts.append(values[positive])
        index_parts.append(indices[positive])
        indptr[row + 1] = indptr[row] + int(np.sum(positive))
    if indptr[-1] >= np.iinfo(np.int32).max:
        raise RuntimeError("single-context capped control exceeds int32 CSR boundary")
    result = sparse.csr_matrix(
        (
            np.concatenate(data_parts).astype(np.int32, copy=False),
            np.concatenate(index_parts).astype(np.int32, copy=False),
            indptr.astype(np.int32),
        ),
        shape=matrix.shape,
    )
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    before = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)
    after = np.asarray(result.sum(axis=1)).ravel().astype(np.int64)
    if not np.array_equal(before, after):
        raise RuntimeError("sparsity cap changed one or more library sizes")
    raw_pseudobulk = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    capped_pseudobulk = np.asarray(result.sum(axis=0)).ravel().astype(np.float64)
    cosine = float(
        np.dot(raw_pseudobulk, capped_pseudobulk)
        / max(
            np.linalg.norm(raw_pseudobulk) * np.linalg.norm(capped_pseudobulk),
            1e-12,
        )
    )
    return result, {
        "cells_capped": capped_cells,
        "cells_capped_fraction": capped_cells / matrix.shape[0],
        "redistributed_counts": int(total_redistributed),
        "redistributed_count_fraction": float(
            total_redistributed / max(raw_pseudobulk.sum(), 1.0)
        ),
        "pseudobulk_cosine": cosine,
        "pseudobulk_l1_fraction": float(
            np.abs(capped_pseudobulk - raw_pseudobulk).sum()
            / max(raw_pseudobulk.sum(), 1.0)
        ),
    }


def effect_to_fold(
    effect: np.ndarray,
    baseline_effect_space: np.ndarray,
    selected: np.ndarray,
    target_index: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    genes = np.where(selected)[0]
    baseline = baseline_effect_space[genes]
    shifted = np.maximum(np.expm1(np.log1p(baseline) + effect[genes]), 0.0)
    raw_folds = (shifted + 1e-3) / (baseline + 1e-3)
    folds = np.clip(raw_folds, args.minimum_fold, args.maximum_fold)
    target_position = np.where(genes == target_index)[0]
    if len(target_position) != 1:
        raise RuntimeError("target transcript is missing from selected effects")
    off_target = np.ones(len(genes), dtype=bool)
    off_target[target_position[0]] = False
    low_clipped = int(np.count_nonzero(raw_folds[off_target] < args.minimum_fold))
    high_clipped = int(np.count_nonzero(raw_folds[off_target] > args.maximum_fold))
    off_target_count = int(np.count_nonzero(off_target))
    folds[target_position[0]] = args.target_remaining_fraction
    return genes.astype(np.int32), folds.astype(np.float64), {
        "off_target_effect_genes": off_target_count,
        "off_target_fold_low_clipped": low_clipped,
        "off_target_fold_high_clipped": high_clipped,
        "off_target_fold_clipped_fraction": (
            (low_clipped + high_clipped) / off_target_count if off_target_count else 0.0
        ),
    }


def perturb_block(
    base: sparse.csr_matrix,
    effect: np.ndarray,
    baseline_effect_space: np.ndarray,
    baseline_probabilities: np.ndarray,
    target_index: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    values = base.toarray().astype(np.int32, copy=False)
    original_libraries = values.sum(axis=1, dtype=np.int64)
    selected = effect != 0
    selected[target_index] = True
    genes, folds, fold_qc = effect_to_fold(
        effect, baseline_effect_space, selected, target_index, args
    )
    selected_before = values[:, genes].copy()
    strength = rng.lognormal(
        mean=-0.5 * args.response_sigma**2,
        sigma=args.response_sigma,
        size=values.shape[0],
    )
    strength = np.clip(strength, 0.6, 1.4)

    for gene, fold in zip(genes, folds):
        cell_folds = np.power(fold, strength)
        current = values[:, gene].astype(np.int64)
        if fold <= 1.0:
            values[:, gene] = rng.binomial(current, cell_folds).astype(np.int32)
        else:
            control_expectation = original_libraries * baseline_probabilities[gene]
            rate_base = (
                (1.0 - args.upreg_control_smoothing) * current
                + args.upreg_control_smoothing * control_expectation
            )
            increments = rng.poisson(np.maximum((cell_folds - 1.0) * rate_base, 0.0))
            updated = current + increments
            if np.any(updated > np.iinfo(np.int32).max):
                raise OverflowError("generated count exceeds int32")
            values[:, gene] = updated.astype(np.int32)

    selected_after = values[:, genes]
    realized_delta = selected_after.sum(axis=0, dtype=np.int64) - selected_before.sum(
        axis=0, dtype=np.int64
    )
    changed = selected_after != selected_before
    realized_nonzero = realized_delta != 0
    intended_sign = np.sign(folds - 1.0)
    direction_mask = realized_nonzero & (intended_sign != 0)
    direction_accuracy = (
        float(np.mean(np.sign(realized_delta[direction_mask]) == intended_sign[direction_mask]))
        if np.any(direction_mask)
        else 0.0
    )

    result = sparse.csr_matrix(values, dtype=np.int32)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    generated_libraries = np.asarray(result.sum(axis=1)).ravel().astype(np.int64)
    target_position = int(np.where(genes == target_index)[0][0])
    target_before = selected_before[:, target_position]
    target_after = selected_after[:, target_position]
    return result, {
        **fold_qc,
        "selected_effect_genes": int(len(genes)),
        "library_median_before": float(np.median(original_libraries)),
        "library_median_after": float(np.median(generated_libraries)),
        "library_ratio": float(
            np.median(generated_libraries) / np.median(original_libraries)
        ),
        "target_sum_before": int(target_before.sum()),
        "target_sum_after": int(target_after.sum()),
        "target_remaining_fraction": float(
            target_after.sum() / max(target_before.sum(), 1)
        ),
        "changed_cell_fraction": float(np.mean(np.any(changed, axis=1))),
        "realized_changed_genes": int(np.sum(realized_nonzero)),
        "realized_pseudobulk_l1": int(np.abs(realized_delta).sum()),
        "realized_pseudobulk_l2": float(np.linalg.norm(realized_delta)),
        "intended_direction_accuracy": direction_accuracy,
        "nnz": int(result.nnz),
    }


def validate_candidate(
    prediction: ad.AnnData,
    genes: list[str],
    targets: list[str],
    cells_per_group: int,
) -> dict[str, Any]:
    matrix = prediction.X.tocsr(copy=False)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    group_sizes = prediction.obs.groupby(
        ["context", "target_gene"], observed=True, sort=True
    ).size()
    expected_groups = pd.MultiIndex.from_product(
        [CONTEXTS, targets], names=["context", "target_gene"]
    )
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    expected_shape = (len(CONTEXTS) * len(targets) * cells_per_group, len(genes))
    checks = {
        "shape_exact": prediction.shape == expected_shape,
        "csr": sparse.isspmatrix_csr(matrix),
        "canonical_csr": bool(matrix.has_canonical_format),
        "gene_axis_exact": list(prediction.var_names.astype(str)) == genes,
        "obs_names_unique": bool(prediction.obs_names.is_unique),
        "contexts_exact": set(prediction.obs["context"].astype(str)) == set(CONTEXTS),
        "targets_exact": set(prediction.obs["target_gene"].astype(str)) == set(targets),
        "groups_exact": group_sizes.index.equals(expected_groups),
        "cells_per_group_exact": bool((group_sizes == cells_per_group).all()),
        "no_controls": "non-targeting"
        not in set(prediction.obs["target_gene"].astype(str)),
        "finite": bool(np.isfinite(matrix.data).all()),
        "nonnegative": bool(np.all(matrix.data >= 0)),
        "integer": bool(np.issubdtype(matrix.data.dtype, np.integer))
        or bool(np.equal(matrix.data, np.floor(matrix.data)).all()),
        "no_explicit_zeros": bool(np.all(matrix.data != 0)),
        "all_cells_nonzero": bool(np.all(row_sums > 0)),
        "cell_count_cap": bool(np.all(row_sums <= 1_000_000)),
        "nnz_official_cap": int(matrix.nnz) <= MAX_OFFICIAL_NNZ,
        "nnz_safe_int32": int(matrix.nnz) < SAFE_INT32_NNZ,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "checks": checks,
        "failed_checks": failed,
        "shape": list(prediction.shape),
        "groups": int(len(group_sizes)),
        "nnz": int(matrix.nnz),
        "mean_nnz_per_cell": float(matrix.nnz / prediction.n_obs),
        "library_size": {
            "min": int(row_sums.min()),
            "median": float(np.median(row_sums)),
            "max": int(row_sums.max()),
        },
    }
    if failed:
        raise ValueError(f"candidate validation failed: {failed}")
    return report


def main() -> None:
    args = parse_args()
    for output in (args.output_h5ad, args.output_json):
        if output.exists() and not args.force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
    if args.max_base_genes <= 0:
        raise ValueError("--max-base-genes must be positive")
    if (
        not np.isfinite(args.effect_baseline_target_sum)
        or args.effect_baseline_target_sum <= 0
    ):
        raise ValueError("--effect-baseline-target-sum must be positive and finite")
    if not 0 < args.target_remaining_fraction < 1:
        raise ValueError("--target-remaining-fraction must be in (0,1)")

    started = time.perf_counter()
    prior = np.load(args.prior)
    prior_contexts = prior["contexts"].astype(str).tolist()
    prior_targets = prior["targets"].astype(str).tolist()
    genes = prior["genes"].astype(str).tolist()
    effects = prior["effects"].astype(np.float32)
    effect_space_present = "effect_space" in prior.files
    effect_target_sum_present = "effect_target_sum" in prior.files
    effect_gene_mask_present = "effect_gene_mask" in prior.files
    explicit_state_contract = all(
        (effect_space_present, effect_target_sum_present, effect_gene_mask_present)
    )
    if (
        any((effect_space_present, effect_target_sum_present, effect_gene_mask_present))
        and not explicit_state_contract
    ):
        raise ValueError(
            "effect_space, effect_target_sum, and effect_gene_mask must appear together"
        )
    prior_effect_space = (
        str(prior["effect_space"].item())
        if explicit_state_contract
        else "legacy-log1p-cp10k-delta"
    )
    prior_effect_target_sum = (
        float(prior["effect_target_sum"].item()) if explicit_state_contract else None
    )
    if explicit_state_contract:
        raw_effect_gene_mask = np.asarray(prior["effect_gene_mask"])
        if raw_effect_gene_mask.shape != (len(genes),):
            raise ValueError("prior effect_gene_mask has the wrong shape")
        if not np.isin(raw_effect_gene_mask, (False, True)).all():
            raise ValueError("prior effect_gene_mask must be boolean-like")
        effect_gene_mask = raw_effect_gene_mask.astype(np.bool_)
    else:
        effect_gene_mask = np.ones(len(genes), dtype=np.bool_)
    if explicit_state_contract and not np.isclose(
        prior_effect_target_sum, args.effect_baseline_target_sum, rtol=0.0, atol=1e-8
    ):
        raise ValueError(
            "prior effect_target_sum does not match --effect-baseline-target-sum: "
            f"{prior_effect_target_sum} != {args.effect_baseline_target_sum}"
        )
    if explicit_state_contract and prior_effect_space != STATE_EFFECT_SPACE:
        raise ValueError(f"unsupported explicit prior effect space: {prior_effect_space}")
    if not explicit_state_contract and not np.isclose(
        args.effect_baseline_target_sum, 10_000.0, rtol=0.0, atol=1e-8
    ):
        raise ValueError("legacy Bayesian priors must use a 10,000-count baseline")
    if prior_contexts != list(CONTEXTS):
        raise ValueError("prior context order mismatch")
    official_targets = pd.read_csv(args.controls_dir / "pert_counts.csv")[
        "target_gene"
    ].astype(str).tolist()
    if prior_targets != official_targets:
        raise ValueError("prior target order mismatch")
    expected_prior_shape = (len(CONTEXTS), len(official_targets), len(genes))
    if effects.shape != expected_prior_shape:
        raise ValueError(
            f"prior effect shape {effects.shape} does not equal {expected_prior_shape}"
        )
    if not np.isfinite(effects).all():
        raise ValueError("prior contains non-finite effects")
    if explicit_state_contract:
        if int(np.count_nonzero(effect_gene_mask)) != 18_077:
            raise ValueError("STATE effect_gene_mask must contain 18,077 shared genes")
        if np.count_nonzero(effects[:, :, ~effect_gene_mask]):
            raise ValueError("STATE prior assigns effects outside its normalization axis")
    effects_per_group = np.count_nonzero(effects, axis=2)
    max_prior_effects = int(effects_per_group.max())
    if max_prior_effects > 161:
        raise ValueError(
            f"prior has {max_prior_effects} effects/group; producer contract allows 161"
        )
    if args.target_limit:
        if not 1 <= args.target_limit <= len(official_targets):
            raise ValueError("invalid --target-limit")
        targets = official_targets[: args.target_limit]
        effects = effects[:, : args.target_limit]
    else:
        targets = official_targets
    full_contract = len(targets) == 300 and args.cells_per_group == 400
    theoretical_nnz = (
        args.max_base_genes + max_prior_effects
    ) * len(CONTEXTS) * len(targets) * args.cells_per_group
    if theoretical_nnz >= SAFE_INT32_NNZ:
        raise ValueError(
            f"theoretical nnz upper bound {theoretical_nnz:,} is not int32-safe"
        )
    if len(genes) != 18_533 or len(set(genes)) != len(genes):
        raise ValueError("unexpected prior gene axis")
    gene_index = {gene: i for i, gene in enumerate(genes)}

    context_matrices: list[sparse.csr_matrix] = []
    obs_frames: list[pd.DataFrame] = []
    context_reports: dict[str, Any] = {}
    group_qc: list[dict[str, Any]] = []
    for c, context in enumerate(CONTEXTS):
        control_path = args.controls_dir / f"context_{context}.h5ad"
        control = ad.read_h5ad(control_path)
        if list(control.var_names.astype(str)) != genes:
            raise ValueError(f"{control_path}: gene axis mismatch")
        required_obs = {"context", "target_gene", "ntc_id"}
        if not required_obs.issubset(control.obs.columns):
            raise ValueError(f"{control_path}: missing required obs columns")
        if set(control.obs["context"].astype(str)) != {context}:
            raise ValueError(f"{control_path}: context mismatch")
        matrix = control.X
        if not sparse.issparse(matrix):
            matrix = sparse.csr_matrix(matrix)
        matrix = matrix.tocsr(copy=False)
        if not np.equal(matrix.data, np.floor(matrix.data)).all():
            raise ValueError(f"{control_path}: controls are not raw integer counts")
        capped, cap_qc = cap_control_sparsity(
            matrix, args.max_base_genes, args.seed + c * 100_000
        )
        if full_contract:
            if cap_qc["pseudobulk_cosine"] < 0.995:
                raise RuntimeError(f"{context}: sparsity-cap pseudobulk cosine too low")
            if cap_qc["pseudobulk_l1_fraction"] > 0.06:
                raise RuntimeError(f"{context}: sparsity-cap pseudobulk L1 distortion too high")
            if cap_qc["redistributed_count_fraction"] > 0.03:
                raise RuntimeError(f"{context}: sparsity cap redistributes too many counts")
        raw_gene_sums = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
        total_counts = float(raw_gene_sums.sum())
        effect_space_total_counts = float(raw_gene_sums[effect_gene_mask].sum())
        if effect_space_total_counts <= 0:
            raise RuntimeError(f"{context}: effect normalization axis has zero counts")
        baseline_probabilities = (raw_gene_sums + 0.5) / (
            total_counts + 0.5 * len(genes)
        )
        baseline_effect_space = (
            raw_gene_sums / effect_space_total_counts * args.effect_baseline_target_sum
        )
        baseline_effect_space[~effect_gene_mask] = 0.0

        target_blocks: list[sparse.csr_matrix] = []
        obs_parts: list[pd.DataFrame] = []
        for p, target in enumerate(targets):
            rng = np.random.default_rng(deterministic_seed(args.seed, context, target))
            selected_rows = stratified_control_indices(
                control.obs, args.cells_per_group, rng
            )
            base = capped[selected_rows]
            target_idx = gene_index[target]
            block, qc = perturb_block(
                base,
                effects[c, p].copy(),
                baseline_effect_space,
                baseline_probabilities,
                target_idx,
                rng,
                args,
            )
            if block.shape != (args.cells_per_group, len(genes)):
                raise RuntimeError("generated block has wrong shape")
            target_blocks.append(block)
            qc.update({"context": context, "target_gene": target})
            group_qc.append(qc)
            obs_parts.append(
                pd.DataFrame(
                    {
                        "context": context,
                        "target_gene": target,
                        "source_control_row": selected_rows.astype(str),
                    },
                    index=[
                        f"{context}|{target}|{cell:03d}"
                        for cell in range(args.cells_per_group)
                    ],
                )
            )

        context_matrix = sparse.vstack(target_blocks, format="csr", dtype=np.int32)
        context_matrix.sum_duplicates()
        context_matrix.eliminate_zeros()
        context_matrix.sort_indices()
        context_matrices.append(context_matrix)
        obs_frames.append(pd.concat(obs_parts, axis=0))
        context_reports[context] = {
            "control_nnz": int(matrix.nnz),
            "control_mean_nnz_per_cell": float(matrix.nnz / matrix.shape[0]),
            "capped_control_nnz": int(capped.nnz),
            "capped_mean_nnz_per_cell": float(capped.nnz / capped.shape[0]),
            "prediction_nnz": int(context_matrix.nnz),
            "prediction_mean_nnz_per_cell": float(
                context_matrix.nnz / context_matrix.shape[0]
            ),
            "effect_normalization_count_fraction": float(
                effect_space_total_counts / total_counts
            ),
            "sparsity_cap_qc": cap_qc,
        }
        del target_blocks, context_matrix, capped, matrix, control

    total_nnz = sum(matrix.nnz for matrix in context_matrices)
    if total_nnz >= SAFE_INT32_NNZ:
        raise RuntimeError(
            f"predicted nnz {total_nnz:,} exceeds conservative int32 CSR boundary"
        )
    prediction_matrix = sparse.vstack(
        context_matrices, format="csr", dtype=np.int32
    )
    prediction_matrix.sum_duplicates()
    prediction_matrix.eliminate_zeros()
    prediction_matrix.sort_indices()
    prediction = ad.AnnData(
        X=prediction_matrix,
        obs=pd.concat(obs_frames, axis=0),
        var=pd.DataFrame(index=pd.Index(genes, name="gene_name")),
    )
    prediction.uns["method"] = "ESM2-assisted empirical-Bayes public prior with control-anchored count reconstruction"
    prediction.uns["seed"] = args.seed
    validation = validate_candidate(
        prediction, genes, targets, args.cells_per_group
    )

    ratios = np.asarray([item["library_ratio"] for item in group_qc])
    target_remaining = np.asarray(
        [item["target_remaining_fraction"] for item in group_qc]
    )
    output_group_nonzero = np.asarray([item["nnz"] > 0 for item in group_qc])
    realized_l1 = np.asarray([item["realized_pseudobulk_l1"] for item in group_qc])
    changed_fraction = np.asarray([item["changed_cell_fraction"] for item in group_qc])
    direction_accuracy = np.asarray(
        [item["intended_direction_accuracy"] for item in group_qc]
    )
    fold_clipped_fraction = np.asarray(
        [item["off_target_fold_clipped_fraction"] for item in group_qc]
    )
    target_qc_failures = [
        f"{item['context']}|{item['target_gene']}"
        for item in group_qc
        if item["target_sum_before"] >= 20
        and not (0.0 <= item["target_remaining_fraction"] <= 0.40)
    ]
    realized_failures = [
        f"{item['context']}|{item['target_gene']}"
        for item in group_qc
        if item["realized_pseudobulk_l1"] <= 0
        or item["changed_cell_fraction"] <= 0
    ]
    scientific_qc = {
        "all_output_groups_nonzero": bool(output_group_nonzero.all()),
        "all_groups_have_realized_delta": not realized_failures,
        "realized_delta_failures": realized_failures,
        "target_knockdown_failures": target_qc_failures,
        "library_ratio_quantiles": {
            str(q): float(np.quantile(ratios, q))
            for q in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "target_remaining_fraction_quantiles": {
            str(q): float(np.quantile(target_remaining, q))
            for q in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "groups_with_target_baseline_counts": int(
            np.sum([item["target_sum_before"] > 0 for item in group_qc])
        ),
        "changed_cell_fraction_quantiles": {
            str(q): float(np.quantile(changed_fraction, q))
            for q in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "realized_pseudobulk_l1_quantiles": {
            str(q): float(np.quantile(realized_l1, q))
            for q in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "intended_direction_accuracy_quantiles": {
            str(q): float(np.quantile(direction_accuracy, q))
            for q in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "off_target_fold_clipped_fraction_quantiles": {
            str(q): float(np.quantile(fold_clipped_fraction, q))
            for q in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
    }
    if not scientific_qc["all_output_groups_nonzero"]:
        raise RuntimeError("one or more context-target groups are all zero")
    if realized_failures:
        raise RuntimeError(
            f"{len(realized_failures)} groups have no realized perturbation delta"
        )
    if target_qc_failures:
        raise RuntimeError(
            f"{len(target_qc_failures)} groups failed target-knockdown QC"
        )
    if np.max(np.abs(ratios - 1.0)) > 0.10:
        raise RuntimeError("generated group median library depth differs by more than 10%")

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_h5ad = args.output_h5ad.with_suffix(".tmp.h5ad")
    compression = None if args.compression == "none" else args.compression
    prediction.write_h5ad(temporary_h5ad, compression=compression)
    os.replace(temporary_h5ad, args.output_h5ad)

    report = {
        "artifact_type": "vcc_2026_prediction_candidate",
        "created_at_unix": time.time(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "seed": args.seed,
        "full_official_contract": full_contract,
        "input_prior": str(args.prior),
        "input_prior_sha256": sha256_file(args.prior),
        "input_prior_effect_space": prior_effect_space,
        "input_prior_effect_target_sum": prior_effect_target_sum,
        "input_prior_effect_normalization_genes": int(
            np.count_nonzero(effect_gene_mask)
        ),
        "generation": {
            "cells_per_group": args.cells_per_group,
            "max_base_genes": args.max_base_genes,
            "response_sigma": args.response_sigma,
            "upreg_control_smoothing": args.upreg_control_smoothing,
            "effect_baseline_target_sum": args.effect_baseline_target_sum,
            "effect_normalization_genes": int(np.count_nonzero(effect_gene_mask)),
            "non_target_fold_clip": [args.minimum_fold, args.maximum_fold],
            "target_remaining_fraction": args.target_remaining_fraction,
            "max_prior_effects_per_group": max_prior_effects,
            "theoretical_nnz_upper_bound": theoretical_nnz,
        },
        "validation": validation,
        "scientific_qc": scientific_qc,
        "context_statistics": context_reports,
        "output_h5ad": str(args.output_h5ad),
        "output_h5ad_bytes": args.output_h5ad.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
    }
    temporary_json = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_json, args.output_json)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
