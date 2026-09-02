#!/usr/bin/env python3
"""Fit an ESM2-assisted Bayesian perturbation prior for VCC 2026.

This is deliberately a transfer baseline rather than a direct STATE inference
run.  The published STATE checkpoints have no overlap with the 2026 target
panel.  We instead use the official STATE support data to learn a low-rank
perturbation response basis, predict unseen-target coordinates from ESM2, and
adapt weakly with correlations measured in the three released control sets.

The output contains log1p(CP10K)-space effects only.  A separate generator
anchors those effects to real A/B/C raw-count control cells.
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
from collections import defaultdict
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
import torch

from authenticate_esm2_target_features import (
    OFFICIAL_SPEC as OFFICIAL_ESM2_SPEC,
    TargetFeatureAuthenticationError,
    load_authenticated_arc_feature_map,
    load_restricted_tensor_mapping,
)


CONTEXTS = ("A", "B", "C")
SOURCE_GROUP = {
    "competition_train": "H1",
    "k562": "K562",
    "k562_gwps": "K562",
    "rpe1": "RPE1",
    "jurkat": "Jurkat",
    "hepg2": "HepG2",
}
GROUP_ORDER = ("H1", "HepG2", "Jurkat", "K562", "RPE1")
ESM_ALIASES = {"TAZ": "WWTR1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--effect-cache",
        type=Path,
        default=Path("artifacts/bayes_public_effects_v1.npz"),
    )
    parser.add_argument(
        "--esm2",
        type=Path,
        default=Path("dataset/state_support/extracted/ESM2_pert_features.pt"),
    )
    parser.add_argument(
        "--esm2-expected-sha256",
        default=None,
        help=(
            "Expected SHA-256 for a non-default ESM2 feature map. The exact "
            "registered Arc artifact digest is enforced automatically for "
            "the default filename."
        ),
    )
    parser.add_argument(
        "--controls-dir", type=Path, default=Path("dataset/controls")
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("artifacts/bayesian_prior_h100_v0.npz"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/bayesian_prior_h100_v0.json"),
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--svd-oversample", type=int, default=16)
    parser.add_argument("--svd-power-iterations", type=int, default=2)
    parser.add_argument("--minimum-source-cells", type=int, default=20)
    parser.add_argument("--ridge-context", type=float, default=30.0)
    parser.add_argument("--context-residual-multiplier", type=float, default=0.0)
    parser.add_argument("--context-temperature", type=float, default=0.08)
    parser.add_argument("--context-uniform-floor", type=float, default=0.20)
    parser.add_argument(
        "--generic-scale",
        type=float,
        default=0.30,
        help="negative selects a conservative public leave-target-out calibration",
    )
    parser.add_argument("--direct-target-scale", type=float, default=0.45)
    parser.add_argument("--effect-clip", type=float, default=0.60)
    parser.add_argument("--minimum-abs-effect", type=float, default=0.01)
    parser.add_argument("--top-nonpanel", type=int, default=128)
    parser.add_argument("--top-panel", type=int, default=32)
    parser.add_argument("--control-sample-cells", type=int, default=2048)
    parser.add_argument("--coexpression-z-shift", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def choose_device(name: str, require_cuda: bool) -> torch.device:
    if name == "cpu":
        device = torch.device("cpu")
    elif name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("this production fit requires an allocated CUDA GPU")
    return device


def stable_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    shifted = (values - np.max(values)) / temperature
    weights = np.exp(shifted)
    return weights / weights.sum()


def source_scale_factors(
    effects: np.ndarray,
    targets: np.ndarray,
    sources: np.ndarray,
) -> dict[str, float]:
    """Calibrate only the two technical K562 panels against one another.

    Cross-cell-line amplitude differences may be real biology, so H1, HepG2,
    Jurkat, and RPE1 remain on their harmonized log1p(CP10K) scales.
    """
    rms = np.sqrt(np.mean(np.square(effects, dtype=np.float64), axis=1))
    by_source_target: dict[tuple[str, str], float] = {
        (str(source), str(target)): float(value)
        for source, target, value in zip(sources, targets, rms)
    }
    factors = {str(source): 1.0 for source in np.unique(sources)}
    essential = {
        str(target)
        for target, source in zip(targets, sources)
        if source == "k562"
    }
    gwps = {
        str(target)
        for target, source in zip(targets, sources)
        if source == "k562_gwps"
    }
    overlap = sorted(essential.intersection(gwps))
    ratios = [
        by_source_target[("k562_gwps", target)]
        / max(by_source_target[("k562", target)], 1e-8)
        for target in overlap
    ]
    if len(ratios) >= 10:
        factors["k562"] = float(np.clip(np.median(ratios), 0.50, 2.0))
    return factors


def context_similarity_weights(cache: Any, targets: list[str], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    support_genes = cache["support_genes"].astype(str)
    support_to_current = cache["support_to_current"].astype(np.int64)
    common = support_to_current >= 0
    support_genes = support_genes[common]
    current_means = cache["current_control_log_means"][:, support_to_current[common]]
    source_means = cache["source_control_log_means"][:, common]
    source_names = cache["source_names"].astype(str)

    group_profiles: list[np.ndarray] = []
    for group in GROUP_ORDER:
        indices = [
            i
            for i, source in enumerate(source_names)
            if SOURCE_GROUP[str(source)] == group
        ]
        group_profiles.append(np.mean(source_means[indices], axis=0))
    profiles = np.vstack(group_profiles)

    excluded = np.asarray(
        [
            gene in set(targets)
            or gene.startswith("MT-")
            or gene.startswith("RPL")
            or gene.startswith("RPS")
            for gene in support_genes
        ],
        dtype=bool,
    )
    expressed = np.max(np.vstack([current_means, profiles]), axis=0) > 0.05
    variance = np.var(profiles, axis=0)
    candidate = np.where((~excluded) & expressed)[0]
    if len(candidate) > 4_000:
        candidate = candidate[np.argpartition(variance[candidate], -4_000)[-4_000:]]

    similarities = np.empty((len(CONTEXTS), len(GROUP_ORDER)), dtype=np.float64)
    for c in range(len(CONTEXTS)):
        for d in range(len(GROUP_ORDER)):
            similarities[c, d] = np.corrcoef(
                current_means[c, candidate], profiles[d, candidate]
            )[0, 1]
    weights = np.vstack(
        [
            (1.0 - args.context_uniform_floor)
            * stable_softmax(row, args.context_temperature)
            + args.context_uniform_floor / len(GROUP_ORDER)
            for row in similarities
        ]
    )
    return similarities, weights


def load_esm_features(
    path: Path,
    public_targets: list[str],
    query_targets: list[str],
    *,
    expected_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    if expected_sha256 is None:
        if path.name != OFFICIAL_ESM2_SPEC.extracted_filename:
            raise ValueError(
                "A non-default ESM2 map requires --esm2-expected-sha256"
            )
        embeddings, load_descriptor = load_authenticated_arc_feature_map(path)
    else:
        try:
            payload, load_descriptor = load_restricted_tensor_mapping(
                path,
                label="ESM2 feature map",
                expected_sha256=expected_sha256,
            )
        except TargetFeatureAuthenticationError as error:
            raise ValueError(str(error)) from error
        if type(payload) is not dict:
            raise TypeError("ESM2 feature file must contain a plain gene->tensor dict")
        embeddings = payload

    if not all(isinstance(key, str) and key for key in embeddings):
        raise TypeError("Every ESM2 feature key must be a non-empty string")
    dimensions: set[int] = set()
    for key, value in embeddings.items():
        if type(value) is not torch.Tensor:
            raise TypeError(f"ESM2 feature value must be a plain tensor: {key}")
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise TypeError(f"ESM2 feature must be a CPU strided tensor: {key}")
        if value.dtype != torch.float32 or value.ndim != 1:
            raise TypeError(f"ESM2 feature must be a rank-one float32 tensor: {key}")
        if not bool(torch.isfinite(value).all()) or not bool(torch.count_nonzero(value)):
            raise ValueError(f"ESM2 feature must be finite and nonzero: {key}")
        dimensions.add(int(value.numel()))
    if dimensions != {OFFICIAL_ESM2_SPEC.embedding_dimension}:
        raise ValueError(f"Unexpected ESM2 feature dimensions: {sorted(dimensions)}")

    missing: list[str] = []

    def fetch(name: str) -> np.ndarray | None:
        key = name if name in embeddings else ESM_ALIASES.get(name, name)
        value = embeddings.get(key)
        if value is None:
            missing.append(name)
            return None
        return value.detach().cpu().numpy().astype(np.float32, copy=False)

    public_raw = [fetch(target) for target in public_targets]
    query_raw = [fetch(target) for target in query_targets]
    if any(value is None for value in public_raw):
        raise ValueError(f"missing public-target ESM2 features: {sorted(set(missing))}")
    if any(value is None for value in query_raw):
        raise ValueError(f"missing challenge-target ESM2 features: {sorted(set(missing))}")
    public = np.vstack(public_raw)
    query = np.vstack(query_raw)
    mean = public.mean(axis=0)
    std = public.std(axis=0)
    std[std < 1e-5] = 1.0
    public = (public - mean) / std
    query = (query - mean) / std
    public /= np.maximum(np.linalg.norm(public, axis=1, keepdims=True), 1e-8)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-8)
    return (
        public.astype(np.float32),
        query.astype(np.float32),
        sorted(set(missing)),
        load_descriptor,
    )


def fit_source_latents(
    sample_scores: np.ndarray,
    sample_targets: np.ndarray,
    sample_groups: np.ndarray,
    sample_quality: np.ndarray,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    dict[str, tuple[list[str], np.ndarray, np.ndarray]],
    np.ndarray,
]:
    unique_targets = sorted(set(sample_targets.astype(str)))
    target_index = {target: i for i, target in enumerate(unique_targets)}
    group_index = {group: i for i, group in enumerate(GROUP_ORDER)}
    anchor_targets = {
        target
        for target in unique_targets
        if set(sample_groups[sample_targets == target]) == set(GROUP_ORDER)
    }
    if len(anchor_targets) < 20:
        raise ValueError("too few shared source-context anchors for bias estimation")
    global_scores = np.zeros((len(unique_targets), sample_scores.shape[1]), dtype=np.float64)
    group_bias = np.zeros((len(GROUP_ORDER), sample_scores.shape[1]), dtype=np.float64)

    for _ in range(4):
        for target, ti in target_index.items():
            rows = np.where(sample_targets == target)[0]
            adjusted = sample_scores[rows] - np.vstack(
                [group_bias[group_index[str(sample_groups[row])]] for row in rows]
            )
            weights = sample_quality[rows]
            global_scores[ti] = np.average(adjusted, axis=0, weights=weights)
        for group, gi in group_index.items():
            rows = np.asarray(
                [
                    row
                    for row in np.where(sample_groups == group)[0]
                    if str(sample_targets[row]) in anchor_targets
                ],
                dtype=np.int64,
            )
            residual = np.vstack(
                [
                    sample_scores[row] - global_scores[target_index[str(sample_targets[row])]]
                    for row in rows
                ]
            )
            group_bias[gi] = np.average(
                residual, axis=0, weights=sample_quality[rows]
            )
        group_bias -= group_bias.mean(axis=0, keepdims=True)

    residual_data: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    for group in GROUP_ORDER:
        group_targets: list[str] = []
        group_residuals: list[np.ndarray] = []
        group_weights: list[float] = []
        for target in unique_targets:
            rows = np.where((sample_groups == group) & (sample_targets == target))[0]
            if not len(rows):
                continue
            values = np.vstack(
                [
                    sample_scores[row]
                    - global_scores[target_index[target]]
                    - group_bias[group_index[group]]
                    for row in rows
                ]
            )
            weights = sample_quality[rows]
            group_targets.append(target)
            group_residuals.append(np.average(values, axis=0, weights=weights))
            group_weights.append(float(np.clip(weights.sum(), 0.05, 1.0)))
        residual_data[group] = (
            group_targets,
            np.vstack(group_residuals),
            np.asarray(group_weights, dtype=np.float64),
        )

    target_weights = np.asarray(
        [
            np.clip(sample_quality[sample_targets == target].sum(), 0.05, 1.0)
            for target in unique_targets
        ],
        dtype=np.float64,
    )
    return unique_targets, global_scores.astype(np.float32), target_weights, residual_data, group_bias.astype(np.float32)


def collapse_group_target_rows(
    effects: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    n_cells: np.ndarray,
    quality: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge replicate panels such as K562 essential and K562 GWPS."""
    merged_effects: list[np.ndarray] = []
    merged_targets: list[str] = []
    merged_groups: list[str] = []
    merged_cells: list[float] = []
    merged_quality: list[float] = []
    keys = sorted(set(zip(targets.astype(str), groups.astype(str))))
    for target, group in keys:
        rows = np.where((targets == target) & (groups == group))[0]
        row_weights = quality[rows]
        merged_effects.append(np.average(effects[rows], axis=0, weights=row_weights))
        merged_targets.append(target)
        merged_groups.append(group)
        merged_cells.append(float(n_cells[rows].sum()))
        merged_quality.append(float(np.clip(row_weights.sum(), 0.02, 1.0)))
    return (
        np.vstack(merged_effects).astype(np.float32),
        np.asarray(merged_targets),
        np.asarray(merged_groups),
        np.asarray(merged_cells, dtype=np.float64),
        np.asarray(merged_quality, dtype=np.float64),
    )


def kernel_ridge_predict(
    train_features: np.ndarray,
    train_values: np.ndarray,
    train_weights: np.ndarray,
    query_features: np.ndarray,
    ridge: float,
) -> np.ndarray:
    kernel = train_features @ train_features.T
    cross = query_features @ train_features.T
    sqrt_weight = np.sqrt(train_weights)
    system = kernel * sqrt_weight[:, None] * sqrt_weight[None, :]
    system.flat[:: len(system) + 1] += ridge
    rhs = sqrt_weight[:, None] * train_values
    coefficients = np.linalg.solve(system.astype(np.float64), rhs.astype(np.float64))
    return ((cross * sqrt_weight[None, :]) @ coefficients).astype(np.float32)


def target_cluster_cv(
    features: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> tuple[float, float, str, dict[str, Any]]:
    labels = KMeans(n_clusters=5, n_init=20, random_state=seed).fit_predict(features)
    ridge_grid = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    results: dict[str, Any] = {}
    best_ridge = ridge_grid[0]
    best_error = math.inf
    best_predictions: np.ndarray | None = None
    baseline_error = float(np.sum(weights[:, None] * np.square(values)))
    nearest_predictions = np.zeros_like(values)
    for fold in range(5):
        test = labels == fold
        train = ~test
        nearest = np.argmax(features[test] @ features[train].T, axis=1)
        nearest_predictions[test] = values[train][nearest]
    nearest_error = float(
        np.sum(weights[:, None] * np.square(nearest_predictions - values))
    )

    for ridge in ridge_grid:
        predictions = np.zeros_like(values)
        for fold in range(5):
            test = labels == fold
            train = ~test
            predictions[test] = kernel_ridge_predict(
                features[train],
                values[train],
                weights[train],
                features[test],
                ridge,
            )
        error = float(np.sum(weights[:, None] * np.square(predictions - values)))
        results[str(ridge)] = {
            "weighted_sse": error,
            "relative_to_generic": error / max(baseline_error, 1e-12),
        }
        if error < best_error:
            best_error = error
            best_ridge = ridge
            best_predictions = predictions

    assert best_predictions is not None
    method = "kernel_ridge"
    deployed_predictions = best_predictions
    deployed_error = best_error
    if nearest_error < best_error:
        method = "nearest_esm_neighbor"
        deployed_predictions = nearest_predictions
        deployed_error = nearest_error
    numerator = float(np.sum(weights[:, None] * deployed_predictions * values))
    denominator = float(np.sum(weights[:, None] * np.square(deployed_predictions)))
    optimal_scale = numerator / max(denominator, 1e-12)
    deployed_scale = float(np.clip(0.70 * optimal_scale, 0.0, 0.70))
    if deployed_error >= baseline_error or optimal_scale <= 0:
        method = "generic_only"
        deployed_scale = 0.0
        deployed_error = baseline_error
    cosine = np.sum(deployed_predictions * values, axis=1) / np.maximum(
        np.linalg.norm(deployed_predictions, axis=1)
        * np.linalg.norm(values, axis=1),
        1e-8,
    )
    summary = {
        "fold_sizes": [int(np.sum(labels == fold)) for fold in range(5)],
        "nested_preprocessing": False,
        "promotion_eligible": False,
        "ridge_grid": results,
        "best_ridge": best_ridge,
        "best_relative_sse": best_error / max(baseline_error, 1e-12),
        "nearest_neighbor_relative_sse": nearest_error / max(baseline_error, 1e-12),
        "deployed_method": method,
        "deployed_relative_sse": deployed_error / max(baseline_error, 1e-12),
        "median_latent_cosine": float(np.median(cosine)),
        "optimal_scale": optimal_scale,
        "deployed_target_specific_scale": deployed_scale,
    }
    return float(best_ridge), deployed_scale, method, summary


def calibrate_generic_scale(
    effects: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Calibrate the shared perturbation response without challenge labels."""
    total = np.sum(weights[:, None] * effects, axis=0)
    total_weight = float(weights.sum())
    numerator = 0.0
    denominator = 0.0
    zero_error = 0.0
    for target in sorted(set(targets.astype(str))):
        held_out = targets == target
        held_weight = float(weights[held_out].sum())
        if held_weight <= 0 or total_weight <= held_weight:
            continue
        prediction = (
            total - np.sum(weights[held_out, None] * effects[held_out], axis=0)
        ) / (total_weight - held_weight)
        truth = np.average(effects[held_out], axis=0, weights=weights[held_out])
        target_weight = min(1.0, held_weight)
        numerator += target_weight * float(np.dot(prediction, truth))
        denominator += target_weight * float(np.dot(prediction, prediction))
        zero_error += target_weight * float(np.dot(truth, truth))
    optimum = numerator / max(denominator, 1e-12)
    deployed = float(np.clip(0.70 * optimum, 0.0, 0.70))
    deployed_error = (
        zero_error - 2.0 * deployed * numerator + deployed**2 * denominator
    )
    if optimum <= 0 or deployed_error >= zero_error:
        deployed = 0.0
        deployed_error = zero_error
    return deployed, {
        "optimal_scale": optimum,
        "deployed_scale": deployed,
        "relative_sse": deployed_error / max(zero_error, 1e-12),
    }


def target_balanced_effects(
    effects: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average source contexts within target before estimating a generic effect."""
    names = np.asarray(sorted(set(targets.astype(str))))
    meta_effects: list[np.ndarray] = []
    meta_weights: list[float] = []
    for target in names:
        rows = targets == target
        meta_effects.append(np.average(effects[rows], axis=0, weights=weights[rows]))
        meta_weights.append(float(np.clip(weights[rows].sum(), 0.05, 1.0)))
    return (
        names,
        np.vstack(meta_effects).astype(np.float32),
        np.asarray(meta_weights, dtype=np.float64),
    )


def anchored_group_generics(
    effects: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Estimate comparable context-generic responses from shared targets only."""
    unique_targets = sorted(set(targets.astype(str)))
    anchors = [
        target
        for target in unique_targets
        if set(groups[targets == target].astype(str)) == set(GROUP_ORDER)
    ]
    if len(anchors) < 20:
        raise ValueError("too few five-context targets for anchored generic effects")
    result: list[np.ndarray] = []
    for group in GROUP_ORDER:
        rows = (groups == group) & np.isin(targets, anchors)
        result.append(np.average(effects[rows], axis=0, weights=weights[rows]))
    return np.vstack(result).astype(np.float32), anchors


def control_correlations(
    path: Path,
    targets: list[str],
    genes: list[str],
    sample_cells: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        if list(adata.var_names.astype(str)) != genes:
            raise ValueError(f"{path}: gene axis mismatch")
        rng = np.random.default_rng(seed)
        chosen = np.sort(
            rng.choice(adata.n_obs, size=min(sample_cells, adata.n_obs), replace=False)
        )
        block = adata.X[chosen]
        if sparse.issparse(block):
            block = block.toarray()
        values = torch.as_tensor(np.asarray(block, dtype=np.float32), device=device)
        libraries = values.sum(dim=1, keepdim=True).clamp_min(1.0)
        values = torch.log1p(values * (10_000.0 / libraries))
        means = values.mean(dim=0)
        std = values.std(dim=0, unbiased=True).clamp_min(1e-4)
        standardized = (values - means) / std
        gene_index = {gene: i for i, gene in enumerate(genes)}
        target_indices = torch.as_tensor(
            [gene_index[target] for target in targets], dtype=torch.long, device=device
        )
        target_values = standardized.index_select(1, target_indices)
        correlations = target_values.T @ standardized / max(len(chosen) - 1, 1)
        return (
            correlations.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32),
            std.cpu().numpy().astype(np.float32),
        )
    finally:
        adata.file.close()


def main() -> None:
    args = parse_args()
    for output in (args.output_npz, args.output_json):
        if output.exists() and not args.force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
    if not 0 <= args.context_uniform_floor < 1:
        raise ValueError("--context-uniform-floor must be in [0,1)")
    if args.context_temperature <= 0:
        raise ValueError("--context-temperature must be positive")
    if args.rank < 1 or args.svd_oversample < 0:
        raise ValueError("invalid SVD rank/oversampling")
    if args.ridge_context <= 0:
        raise ValueError("--ridge-context must be positive")
    if args.effect_clip <= 0 or args.minimum_abs_effect < 0:
        raise ValueError("invalid effect clipping/threshold")
    if args.top_nonpanel < 0 or args.top_panel < 0:
        raise ValueError("top-gene limits must be non-negative")
    if args.control_sample_cells < 2:
        raise ValueError("--control-sample-cells must be at least 2")
    if args.generic_scale > 1 or not 0 <= args.direct_target_scale <= 1:
        raise ValueError("effect scales must not exceed 1")
    device = choose_device(args.device, args.require_cuda)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    started = time.perf_counter()
    cache = np.load(args.effect_cache)
    effects = cache["effects"].astype(np.float32)
    sample_targets = cache["sample_targets"].astype(str)
    sample_sources = cache["sample_sources"].astype(str)
    sample_n_cells = cache["sample_n_cells"].astype(np.float64)
    support_genes = cache["support_genes"].astype(str).tolist()
    current_genes = cache["current_genes"].astype(str).tolist()
    support_to_current = cache["support_to_current"].astype(np.int64)
    challenge_targets = pd.read_csv(args.controls_dir / "pert_counts.csv")[
        "target_gene"
    ].astype(str).tolist()
    if len(challenge_targets) != 300 or len(set(challenge_targets)) != 300:
        raise ValueError("unexpected challenge target panel")

    support_index = {gene: i for i, gene in enumerate(support_genes)}
    raw_effects = effects.copy()
    knockdown = np.asarray(
        [
            max(0.0, -float(raw_effects[i, support_index[target]]))
            if target in support_index
            else 0.0
            for i, target in enumerate(sample_targets)
        ]
    )
    scales = source_scale_factors(effects, sample_targets, sample_sources)
    for source, factor in scales.items():
        effects[sample_sources == source] *= factor
    for i, target in enumerate(sample_targets):
        if target in support_index:
            effects[i, support_index[target]] = 0.0
    np.clip(effects, -2.0, 2.0, out=effects)

    keep = sample_n_cells >= args.minimum_source_cells
    effects = effects[keep]
    sample_targets = sample_targets[keep]
    sample_sources = sample_sources[keep]
    sample_n_cells = sample_n_cells[keep]
    knockdown = knockdown[keep]
    sample_groups = np.asarray([SOURCE_GROUP[source] for source in sample_sources])
    sample_quality = (sample_n_cells / (sample_n_cells + 100.0)) * np.clip(
        knockdown / 0.5, 0.20, 1.0
    )
    sample_quality = np.clip(sample_quality, 0.02, 1.0)
    (
        effects,
        sample_targets,
        sample_groups,
        sample_n_cells,
        sample_quality,
    ) = collapse_group_target_rows(
        effects,
        sample_targets,
        sample_groups,
        sample_n_cells,
        sample_quality,
    )

    meta_names, meta_effects, meta_weights = target_balanced_effects(
        effects, sample_targets, sample_quality
    )
    generic = np.average(meta_effects, axis=0, weights=meta_weights)
    auto_generic_scale, generic_cv_report = calibrate_generic_scale(
        meta_effects, meta_names, meta_weights
    )
    generic_scale = (
        auto_generic_scale if args.generic_scale < 0 else float(args.generic_scale)
    )
    group_generics, generic_anchor_targets = anchored_group_generics(
        effects, sample_targets, sample_groups, sample_quality
    )
    gene_scale = np.maximum(
        np.median(np.abs(effects - generic[None, :]), axis=0) / 0.6745,
        0.03,
    ).astype(np.float32)
    whitened = ((effects - generic[None, :]) / gene_scale[None, :]).astype(np.float32)

    matrix = torch.as_tensor(whitened, device=device)
    matrix = matrix * torch.as_tensor(
        np.sqrt(sample_quality)[:, None], dtype=torch.float32, device=device
    )
    q = min(args.rank + args.svd_oversample, matrix.shape[0] - 1, matrix.shape[1])
    _, _, right = torch.pca_lowrank(
        matrix,
        q=q,
        center=False,
        niter=args.svd_power_iterations,
    )
    basis = right[:, : args.rank]
    unweighted = torch.as_tensor(whitened, device=device)
    sample_scores = (unweighted @ basis).cpu().numpy().astype(np.float32)

    (
        public_targets,
        global_scores,
        target_weights,
        residual_data,
        group_bias,
    ) = fit_source_latents(
        sample_scores, sample_targets, sample_groups, sample_quality
    )
    public_features, query_features, missing_esm, esm2_load_descriptor = load_esm_features(
        args.esm2,
        public_targets,
        challenge_targets,
        expected_sha256=args.esm2_expected_sha256,
    )
    ridge, target_scale, transfer_method, cv_report = target_cluster_cv(
        public_features, global_scores, target_weights, args.seed
    )
    if transfer_method == "nearest_esm_neighbor":
        nearest = np.argmax(query_features @ public_features.T, axis=1)
        predicted_global = global_scores[nearest].copy()
    elif transfer_method == "generic_only":
        predicted_global = np.zeros(
            (len(challenge_targets), global_scores.shape[1]), dtype=np.float32
        )
    else:
        predicted_global = kernel_ridge_predict(
            public_features,
            global_scores,
            target_weights,
            query_features,
            ridge,
        )

    similarities, context_weights = context_similarity_weights(
        cache, challenge_targets, args
    )
    context_generic_effects = context_weights @ group_generics
    nearest_similarity = np.max(query_features @ public_features.T, axis=1)
    ood_reliability = np.clip(
        (nearest_similarity - 0.25) / (0.75 - 0.25), 0.20, 1.0
    ).astype(np.float32)
    predicted_global *= ood_reliability[:, None]

    public_feature_index = {target: i for i, target in enumerate(public_targets)}
    residual_predictions: dict[str, np.ndarray] = {}
    for group in GROUP_ORDER:
        group_targets, group_values, group_weights = residual_data[group]
        feature_rows = np.asarray(
            [public_feature_index[target] for target in group_targets], dtype=np.int64
        )
        residual_predictions[group] = kernel_ridge_predict(
            public_features[feature_rows],
            group_values,
            group_weights,
            query_features,
            args.ridge_context,
        )

    context_scores = np.empty(
        (len(CONTEXTS), len(challenge_targets), args.rank), dtype=np.float32
    )
    group_index = {group: i for i, group in enumerate(GROUP_ORDER)}
    challenge_index = {target: i for i, target in enumerate(challenge_targets)}
    direct_target_mask = np.asarray(
        [target in set(sample_targets) for target in challenge_targets], dtype=bool
    )
    direct_full_effects: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
    for c in range(len(CONTEXTS)):
        score = predicted_global.copy()
        for group in GROUP_ORDER:
            weight = context_weights[c, group_index[group]]
            score += weight * group_bias[group_index[group]]
            score += (
                weight
                * args.context_residual_multiplier
                * residual_predictions[group]
                * ood_reliability[:, None]
            )

        for target in set(sample_targets).intersection(challenge_targets):
            rows = np.where(sample_targets == target)[0]
            direct_weights = np.asarray(
                [
                    sample_quality[row]
                    * context_weights[c, group_index[str(sample_groups[row])]]
                    for row in rows
                ]
            )
            if direct_weights.sum() <= 0:
                continue
            direct = np.average(sample_scores[rows], axis=0, weights=direct_weights)
            direct_full = np.average(effects[rows], axis=0, weights=direct_weights)
            direct_n = float(
                np.sum(
                    [
                        sample_n_cells[row]
                        * context_weights[c, group_index[str(sample_groups[row])]]
                        for row in rows
                    ]
                )
            )
            gamma = direct_n / (direct_n + 200.0)
            qi = challenge_index[target]
            score[qi] = gamma * direct + (1.0 - gamma) * score[qi]
            direct_full_effects[(c, qi)] = (direct_full.astype(np.float32), gamma)
        context_scores[c] = score

    if args.coexpression_z_shift != 0:
        correlations: list[np.ndarray] = []
        control_std: list[np.ndarray] = []
        for c, context in enumerate(CONTEXTS):
            corr, std = control_correlations(
                args.controls_dir / f"context_{context}.h5ad",
                challenge_targets,
                current_genes,
                args.control_sample_cells,
                args.seed + c * 10_000,
                device,
            )
            correlations.append(corr)
            control_std.append(std)
        corr_array = np.stack(correlations)
        std_array = np.stack(control_std)
        shared_corr = np.mean(corr_array, axis=0)
        sign_agreement = np.sum(
            np.sign(corr_array) == np.sign(shared_corr)[None, :, :], axis=0
        ) >= 2
    else:
        corr_array = np.empty((0,), dtype=np.float32)
        std_array = np.empty((0,), dtype=np.float32)
        shared_corr = np.empty((0,), dtype=np.float32)
        sign_agreement = np.empty((0,), dtype=bool)

    basis_cpu = basis.cpu().numpy().astype(np.float32)
    current_effects = np.zeros(
        (len(CONTEXTS), len(challenge_targets), len(current_genes)),
        dtype=np.float32,
    )
    challenge_set = set(challenge_targets)
    current_index = {gene: i for i, gene in enumerate(current_genes)}
    common_current = support_to_current >= 0
    common_support_indices = np.where(common_current)[0]
    common_current_indices = support_to_current[common_current]
    nonpanel_mask = np.asarray([gene not in challenge_set for gene in current_genes])
    panel_mask = ~nonpanel_mask
    common_mask_current = np.zeros(len(current_genes), dtype=bool)
    common_mask_current[common_current_indices] = True

    selected_counts = np.zeros((len(CONTEXTS), len(challenge_targets)), dtype=np.int32)
    for c in range(len(CONTEXTS)):
        deployed_scales = np.where(
            direct_target_mask, args.direct_target_scale, target_scale
        ).astype(np.float32)
        if transfer_method == "generic_only":
            reconstructed = np.repeat(
                (generic_scale * context_generic_effects[c])[None, :],
                len(challenge_targets),
                axis=0,
            )
            for p in np.where(direct_target_mask)[0]:
                direct_effect, gamma = direct_full_effects[(c, int(p))]
                reconstructed[p] += (
                    args.direct_target_scale * gamma * (direct_effect - generic)
                )
        else:
            reconstructed = (
                generic_scale * context_generic_effects[c][None, :]
                + deployed_scales[:, None]
                * ((context_scores[c] @ basis_cpu.T) * gene_scale[None, :])
            )
        mapped = np.zeros((len(challenge_targets), len(current_genes)), dtype=np.float32)
        mapped[:, common_current_indices] = reconstructed[:, common_support_indices]

        if args.coexpression_z_shift != 0:
            corr = 0.70 * corr_array[c] + 0.30 * shared_corr
            coexpr = -args.coexpression_z_shift * corr * std_array[c][None, :]
            coexpr *= sign_agreement
            coexpr[:, ~common_mask_current] = 0.0
            mapped += coexpr.astype(np.float32)
        np.clip(mapped, -args.effect_clip, args.effect_clip, out=mapped)

        expression = cache["current_control_log_means"][c]
        expression_gate = np.sqrt(expression / (expression + 0.20))
        for p, target in enumerate(challenge_targets):
            target_idx = current_index[target]
            ranking = np.abs(mapped[p]) * expression_gate
            ranking[target_idx] = -np.inf
            selected = np.zeros(len(current_genes), dtype=bool)

            nonpanel_candidates = np.where(
                nonpanel_mask
                & common_mask_current
                & (np.abs(mapped[p]) >= args.minimum_abs_effect)
            )[0]
            if len(nonpanel_candidates) > args.top_nonpanel:
                nonpanel_candidates = nonpanel_candidates[
                    np.argpartition(
                        ranking[nonpanel_candidates], -args.top_nonpanel
                    )[-args.top_nonpanel:]
                ]
            selected[nonpanel_candidates] = True

            panel_candidates = np.where(
                panel_mask
                & common_mask_current
                & (np.arange(len(current_genes)) != target_idx)
                & (np.abs(mapped[p]) >= args.minimum_abs_effect)
            )[0]
            if len(panel_candidates) > args.top_panel:
                panel_candidates = panel_candidates[
                    np.argpartition(ranking[panel_candidates], -args.top_panel)[
                        -args.top_panel:
                    ]
                ]
            selected[panel_candidates] = True
            current_effects[c, p, selected] = mapped[p, selected]
            current_effects[c, p, target_idx] = math.log(0.20)
            selected_counts[c, p] = int(selected.sum() + 1)

    if not np.isfinite(current_effects).all():
        raise ValueError("non-finite predicted effects")
    current_only = ~common_mask_current
    current_target_indices = np.asarray(
        [current_index[target] for target in challenge_targets]
    )
    forbidden_current_only = current_only.copy()
    forbidden_current_only[current_target_indices] = False
    if np.any(current_effects[:, :, forbidden_current_only] != 0):
        raise RuntimeError("current-only genes unexpectedly received transferred effects")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = args.output_npz.with_suffix(args.output_npz.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary_npz,
        effects=current_effects,
        contexts=np.asarray(CONTEXTS, dtype="U1"),
        targets=np.asarray(challenge_targets, dtype="U64"),
        genes=np.asarray(current_genes, dtype="U64"),
        selected_counts=selected_counts,
        context_similarities=similarities.astype(np.float32),
        context_weights=context_weights.astype(np.float32),
        source_groups=np.asarray(GROUP_ORDER, dtype="U16"),
        nearest_esm_similarity=nearest_similarity.astype(np.float32),
        ood_reliability=ood_reliability,
    )
    os.replace(temporary_npz, args.output_npz)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        gpu_name = torch.cuda.get_device_name(device)
        peak_gpu_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        gpu_name = None
        peak_gpu_bytes = 0
    non_target_values = current_effects[
        np.arange(len(CONTEXTS))[:, None, None],
        np.arange(len(challenge_targets))[None, :, None],
        np.broadcast_to(
            current_target_indices[None, :, None],
            (len(CONTEXTS), len(challenge_targets), 1),
        ),
    ]
    report = {
        "artifact_type": "esm2_bayesian_perturbation_prior",
        "created_at_unix": time.time(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu_name": gpu_name,
        "peak_gpu_bytes": peak_gpu_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "seed": args.seed,
        "input_effect_cache": str(args.effect_cache),
        "input_effect_cache_sha256": sha256_file(args.effect_cache),
        "source_scale_factors": scales,
        "source_rows_before_filter": int(len(raw_effects)),
        "source_rows_after_filter": int(len(effects)),
        "public_unique_targets": len(public_targets),
        "challenge_targets": len(challenge_targets),
        "direct_target_overlap": len(set(sample_targets).intersection(challenge_targets)),
        "rank": args.rank,
        "cross_validation": cv_report,
        "generic_scale": generic_scale,
        "generic_scale_calibration": generic_cv_report,
        "generic_anchor_target_count": len(generic_anchor_targets),
        "generic_context_conditioned": True,
        "target_specific_scale": target_scale,
        "direct_target_scale": args.direct_target_scale,
        "transfer_method": transfer_method,
        "context_residual_multiplier": args.context_residual_multiplier,
        "context_similarity_groups": list(GROUP_ORDER),
        "context_similarities": {
            context: {
                group: float(similarities[c, d])
                for d, group in enumerate(GROUP_ORDER)
            }
            for c, context in enumerate(CONTEXTS)
        },
        "context_weights": {
            context: {
                group: float(context_weights[c, d])
                for d, group in enumerate(GROUP_ORDER)
            }
            for c, context in enumerate(CONTEXTS)
        },
        "nearest_esm_similarity_quantiles": {
            str(q): float(np.quantile(nearest_similarity, q))
            for q in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "missing_esm_features": missing_esm,
        "esm2_artifact_load": esm2_load_descriptor,
        "selected_effects_per_group": {
            "min": int(selected_counts.min()),
            "median": float(np.median(selected_counts)),
            "max": int(selected_counts.max()),
        },
        "target_transcript_log_effect": float(non_target_values[0, 0, 0]),
        "output_npz": str(args.output_npz),
        "output_npz_bytes": args.output_npz.stat().st_size,
        "output_npz_sha256": sha256_file(args.output_npz),
    }
    temporary_json = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_json, args.output_json)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
