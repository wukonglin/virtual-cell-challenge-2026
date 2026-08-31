#!/usr/bin/env python3
"""Calibrate K562 perturbation effects for transfer to a recipient cell line.

The calibrator keeps the panel-wide response separate from target-specific
residuals.  It learns a ridge route between residual PCA spaces using paired
K562 and RPE1 essential-gene perturbations, chooses regularization and residual
scales on whole ESM2 clusters, and then refits the route on all paired targets.

Challenge targets measured in the K562 GWPS query atlas use their same-target
source residual.  Targets absent from that atlas use an ESM2-predicted fallback
effect.  Source cell counts shrink noisy measured residuals toward the explicit
shared response.  Routing and PCA decorrelation never modify that response.

The output implements both the canonical effect-atlas contract and the compact
response contract consumed by the scorer-aware count generator:

``effects,target_names,gene_names,target_cell_counts`` and
``genes,targets,common_log_fold,learned_log_fold``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


SCHEMA = "vcc-cross-context-effect-calibration-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def describe_file(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def parse_float_grid(value: str, name: str, *, nonnegative: bool = True) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid {name}: {value}") from error
    require(bool(values), f"{name} cannot be empty")
    require(all(np.isfinite(values)), f"{name} contains a non-finite value")
    if nonnegative:
        require(all(item >= 0 for item in values), f"{name} must be nonnegative")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_atlas", type=Path, help="K562 GWPS query effect atlas.")
    parser.add_argument(
        "source_essential_atlas", type=Path, help="K562 essential-gene effect atlas."
    )
    parser.add_argument(
        "recipient_essential_atlas", type=Path, help="RPE1 essential-gene effect atlas."
    )
    parser.add_argument(
        "fallback_prior",
        type=Path,
        help="ESM2-predicted fallback effect NPZ covering every challenge target.",
    )
    parser.add_argument("challenge_targets", type=Path, help="Challenge target CSV.")
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("output_checkpoint", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--target-column", default="target_gene")
    parser.add_argument(
        "--challenge-genes",
        type=Path,
        default=None,
        help="Optional challenge gene CSV; otherwise the fallback gene axis is used.",
    )
    parser.add_argument(
        "--esm2-embeddings",
        type=Path,
        default=None,
        help=(
            "Optional NPZ or Torch target-embedding map. If omitted, the fallback NPZ "
            "must contain esm2_embeddings and embedding_target_names."
        ),
    )
    parser.add_argument("--source-rank", type=int, default=64)
    parser.add_argument("--recipient-rank", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--reliability-tau", type=float, default=60.0)
    parser.add_argument(
        "--ridge-grid", default="0.001,0.01,0.1,1,10,100", help="Comma-separated values."
    )
    parser.add_argument(
        "--residual-scale-grid",
        default="0.25,0.5,0.75,1,1.25",
        help="Comma-separated target-residual amplitude values.",
    )
    parser.add_argument(
        "--route-scale-grid",
        default="0,0.25,0.5,0.75,1",
        help="Comma-separated interpolation values from source to routed residual.",
    )
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


@dataclass(frozen=True)
class EffectAtlas:
    effects: np.ndarray
    target_names: tuple[str, ...]
    gene_names: tuple[str, ...]
    target_cell_counts: np.ndarray
    context_reduction: str


@dataclass(frozen=True)
class ResidualPCA:
    mean: np.ndarray
    components: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean[None, :]) @ self.components.T

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        return scores @ self.components + self.mean[None, :]


@dataclass(frozen=True)
class RidgeRoute:
    source_pca: ResidualPCA
    recipient_pca: ResidualPCA
    coefficient: np.ndarray
    intercept: np.ndarray
    gene_mask: np.ndarray
    alpha: float


def _read_string_array(payload: Any, keys: Sequence[str], path: Path) -> tuple[str, ...]:
    for key in keys:
        if key in payload:
            values = tuple(str(value) for value in np.asarray(payload[key]).tolist())
            require(values and all(values), f"Empty string in {key} from {path}")
            require(len(values) == len(set(values)), f"Duplicate values in {key} from {path}")
            return values
    raise ValueError(f"Missing one of {list(keys)} in {path}")


def load_effect_atlas(path: Path, *, require_cell_counts: bool) -> EffectAtlas:
    require(path.is_file(), f"Missing effect atlas: {path}")
    with np.load(path, allow_pickle=False) as payload:
        targets = _read_string_array(payload, ("target_names", "targets"), path)
        genes = _read_string_array(payload, ("gene_names", "genes"), path)
        require("effects" in payload, f"Missing effects in {path}")
        effects = np.asarray(payload["effects"], dtype=np.float32)
        context_reduction = "none"
        if effects.ndim == 3:
            require(
                effects.shape[1:] == (len(targets), len(genes)),
                f"Three-dimensional effects must have shape [context,target,gene] in {path}",
            )
            context_reduction = f"arithmetic-mean-across-{effects.shape[0]}-contexts"
            effects = effects.mean(axis=0, dtype=np.float64).astype(np.float32)
        require(
            effects.shape == (len(targets), len(genes)),
            f"Effect shape mismatch in {path}: {effects.shape}",
        )
        require(np.isfinite(effects).all(), f"Non-finite effects in {path}")
        if "target_cell_counts" in payload:
            counts = np.asarray(payload["target_cell_counts"], dtype=np.int64)
            require(counts.shape == (len(targets),), f"Cell-count shape mismatch in {path}")
            require(np.all(counts >= 0), f"Negative target cell count in {path}")
        else:
            require(not require_cell_counts, f"Missing target_cell_counts in {path}")
            counts = np.zeros(len(targets), dtype=np.int64)
    return EffectAtlas(effects, targets, genes, counts, context_reduction)


def read_named_column(path: Path, preferred_column: str, aliases: Sequence[str]) -> list[str]:
    require(path.is_file(), f"Missing CSV: {path}")
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    accepted = (preferred_column, *aliases)
    for column in accepted:
        if column in table.columns:
            values = table[column].astype(str).tolist()
            break
    else:
        raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
        require(raw.shape[1] == 1, f"Expected one column in {path}")
        values = raw.iloc[:, 0].astype(str).tolist()
        if values and values[0].strip().lower() in {item.lower() for item in accepted}:
            values = values[1:]
    require(values and all(values), f"Empty value in {path}")
    require(len(values) == len(set(values)), f"Duplicate values in {path}")
    return values


def align_effects(
    atlas: EffectAtlas,
    targets: Sequence[str],
    genes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align an atlas and return values plus target/gene availability masks."""

    target_lookup = {name: index for index, name in enumerate(atlas.target_names)}
    gene_lookup = {name: index for index, name in enumerate(atlas.gene_names)}
    target_mask = np.asarray([name in target_lookup for name in targets], dtype=bool)
    gene_mask = np.asarray([name in gene_lookup for name in genes], dtype=bool)
    output = np.zeros((len(targets), len(genes)), dtype=np.float32)
    if bool(target_mask.any()) and bool(gene_mask.any()):
        output_rows = np.flatnonzero(target_mask)
        output_columns = np.flatnonzero(gene_mask)
        source_rows = np.asarray([target_lookup[targets[index]] for index in output_rows])
        source_columns = np.asarray([gene_lookup[genes[index]] for index in output_columns])
        output[np.ix_(output_rows, output_columns)] = atlas.effects[
            np.ix_(source_rows, source_columns)
        ]
    return output, target_mask, gene_mask


def merge_direct_and_fallback_effects(
    direct_effects: np.ndarray,
    fallback_effects: np.ndarray,
    direct_target_mask: np.ndarray,
    direct_gene_mask: np.ndarray,
) -> np.ndarray:
    """Use direct effects only where both their target and gene are available."""

    require(direct_effects.shape == fallback_effects.shape, "Effect-source shape mismatch")
    require(
        direct_target_mask.shape == (direct_effects.shape[0],),
        "Direct target-mask mismatch",
    )
    require(
        direct_gene_mask.shape == (direct_effects.shape[1],),
        "Direct gene-mask mismatch",
    )
    merged = fallback_effects.astype(np.float32, copy=True)
    direct_rows = np.flatnonzero(direct_target_mask)
    direct_columns = np.flatnonzero(direct_gene_mask)
    if len(direct_rows) and len(direct_columns):
        merged[np.ix_(direct_rows, direct_columns)] = direct_effects[
            np.ix_(direct_rows, direct_columns)
        ]
    return merged


def shrink_measured_residuals(
    residuals: np.ndarray,
    direct_target_mask: np.ndarray,
    direct_gene_mask: np.ndarray,
    reliability: np.ndarray,
) -> np.ndarray:
    """Shrink only coordinates whose effects came from the measured atlas."""

    require(direct_target_mask.shape == (residuals.shape[0],), "Target-mask mismatch")
    require(direct_gene_mask.shape == (residuals.shape[1],), "Gene-mask mismatch")
    require(reliability.shape == (residuals.shape[0],), "Reliability shape mismatch")
    require(np.all((0 <= reliability) & (reliability <= 1)), "Invalid reliability")
    output = residuals.astype(np.float32, copy=True)
    direct_rows = np.flatnonzero(direct_target_mask)
    direct_columns = np.flatnonzero(direct_gene_mask)
    if len(direct_rows) and len(direct_columns):
        block = output[np.ix_(direct_rows, direct_columns)]
        block *= reliability[direct_rows, None]
        output[np.ix_(direct_rows, direct_columns)] = block
    return output


def reliability_from_counts(counts: np.ndarray, tau: float) -> np.ndarray:
    require(tau >= 0 and np.isfinite(tau), "reliability-tau must be finite and nonnegative")
    values = np.asarray(counts, dtype=np.float64)
    require(np.all(values >= 0), "Cell counts must be nonnegative")
    if tau == 0:
        return np.ones_like(values, dtype=np.float32)
    return (values / (values + tau)).astype(np.float32)


def weighted_common_response(
    effects: np.ndarray,
    weights: np.ndarray,
    target_names: Sequence[str],
    gene_names: Sequence[str],
) -> np.ndarray:
    """Estimate a shared response while excluding each target's own coordinate."""

    require(effects.shape == (len(target_names), len(gene_names)), "Effect shape mismatch")
    weights64 = np.asarray(weights, dtype=np.float64)
    require(weights64.shape == (len(target_names),), "Weight shape mismatch")
    require(np.all(weights64 >= 0) and weights64.sum() > 0, "Invalid common weights")
    numerator = np.zeros(len(gene_names), dtype=np.float64)
    # Chunking avoids a full float64 copy of a genome-wide atlas.
    for begin in range(0, len(target_names), 512):
        end = min(begin + 512, len(target_names))
        numerator += np.sum(
            effects[begin:end].astype(np.float64)
            * weights64[begin:end, None],
            axis=0,
        )
    denominator = np.full(len(gene_names), weights64.sum(), dtype=np.float64)
    lookup = {name: index for index, name in enumerate(gene_names)}
    for row, target in enumerate(target_names):
        column = lookup.get(target)
        if column is not None:
            numerator[column] -= weights64[row] * float(effects[row, column])
            denominator[column] -= weights64[row]
    require(np.all(denominator > 0), "Too few rows to compute leave-own-out common response")
    return (numerator / denominator).astype(np.float32)


def residualize(
    effects: np.ndarray,
    common: np.ndarray,
    target_names: Sequence[str],
    gene_names: Sequence[str],
) -> np.ndarray:
    residual = np.asarray(effects, dtype=np.float32) - common[None, :]
    lookup = {name: index for index, name in enumerate(gene_names)}
    for row, target in enumerate(target_names):
        column = lookup.get(target)
        if column is not None:
            residual[row, column] = 0.0
    return residual


def prepare_holdout_residuals(
    source_effects: np.ndarray,
    recipient_effects: np.ndarray,
    source_reliability: np.ndarray,
    recipient_reliability: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    target_names: Sequence[str],
    gene_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit shared responses on training rows and prepare a leakage-safe holdout.

    Recipient validation effects are labels and therefore must not contribute to
    either shared-response estimate or either training residual matrix. Source
    validation effects are available inputs, but fitting the source shared
    response on training rows as well keeps all learned preprocessing confined
    to the training partition.
    """

    require(source_effects.shape == recipient_effects.shape, "Holdout effect mismatch")
    require(
        source_effects.shape == (len(target_names), len(gene_names)),
        "Holdout effect-axis mismatch",
    )
    require(
        source_reliability.shape == recipient_reliability.shape == (len(target_names),),
        "Holdout reliability mismatch",
    )
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    require(len(train_indices) >= 3 and len(validation_indices) >= 2, "Invalid holdout split")
    require(
        set(train_indices.tolist()).isdisjoint(validation_indices.tolist()),
        "Holdout indices overlap",
    )
    train_targets = [target_names[index] for index in train_indices]
    validation_targets = [target_names[index] for index in validation_indices]
    source_common = weighted_common_response(
        source_effects[train_indices],
        np.maximum(source_reliability[train_indices], 1e-6),
        train_targets,
        gene_names,
    )
    recipient_common = weighted_common_response(
        recipient_effects[train_indices],
        np.maximum(recipient_reliability[train_indices], 1e-6),
        train_targets,
        gene_names,
    )
    train_source = residualize(
        source_effects[train_indices], source_common, train_targets, gene_names
    ) * source_reliability[train_indices, None]
    train_recipient = residualize(
        recipient_effects[train_indices], recipient_common, train_targets, gene_names
    )
    validation_source = residualize(
        source_effects[validation_indices],
        source_common,
        validation_targets,
        gene_names,
    ) * source_reliability[validation_indices, None]
    return source_common, train_source, train_recipient, validation_source


def _embedding_arrays(payload: Any, path: Path) -> tuple[tuple[str, ...], np.ndarray] | None:
    matrix_key = next(
        (
            key
            for key in ("esm2_embeddings", "target_embeddings", "embeddings", "features")
            if key in payload
        ),
        None,
    )
    if matrix_key is None:
        return None
    names = _read_string_array(
        payload,
        ("embedding_target_names", "esm2_target_names", "target_names", "targets"),
        path,
    )
    matrix = np.asarray(payload[matrix_key], dtype=np.float32)
    require(
        matrix.ndim == 2 and matrix.shape[0] == len(names),
        f"Embedding shape mismatch in {path}",
    )
    require(np.isfinite(matrix).all(), f"Non-finite embeddings in {path}")
    return names, matrix


def load_embedding_map(
    fallback_path: Path,
    external_path: Path | None,
    required_names: Iterable[str],
    *,
    allow_missing: bool = False,
) -> dict[str, np.ndarray]:
    path = external_path or fallback_path
    require(path.is_file(), f"Missing ESM2 embedding input: {path}")
    embedding_map: dict[str, np.ndarray]
    if path.suffix.lower() in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        require(isinstance(payload, dict), f"Expected a target-to-tensor map in {path}")
        embedding_map = {
            str(name): np.asarray(value.detach().float().reshape(-1).numpy(), dtype=np.float32)
            for name, value in payload.items()
            if isinstance(value, torch.Tensor)
        }
    else:
        with np.load(path, allow_pickle=False) as payload:
            arrays = _embedding_arrays(payload, path)
        require(arrays is not None, f"No ESM2 embedding arrays found in {path}")
        names, matrix = arrays
        embedding_map = {name: matrix[index] for index, name in enumerate(names)}
    required = tuple(dict.fromkeys(str(name) for name in required_names))
    missing = [name for name in required if name not in embedding_map]
    require(
        allow_missing or not missing,
        f"Missing ESM2 embeddings for {len(missing)} targets: {missing[:8]}",
    )
    available = [name for name in required if name in embedding_map]
    require(available, "No required ESM2 targets are available")
    dimensions = {int(embedding_map[name].size) for name in available}
    require(len(dimensions) == 1, "ESM2 embeddings have inconsistent dimensions")
    require(
        all(np.isfinite(embedding_map[name]).all() for name in available),
        "Non-finite ESM2 embedding",
    )
    return embedding_map


def make_cluster_holdout(
    features: np.ndarray,
    clusters: int,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold out complete ESM2 clusters rather than randomly selected targets."""

    require(
        features.ndim == 2 and features.shape[0] >= 6,
        "At least six paired targets are required",
    )
    require(clusters >= 2, "clusters must be at least two")
    require(0 < validation_fraction < 0.5, "validation-fraction must be in (0, 0.5)")
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    pca_rank = min(32, normalized.shape[0] - 1, normalized.shape[1])
    cluster_features = PCA(
        n_components=pca_rank,
        svd_solver="full" if pca_rank == min(normalized.shape) else "randomized",
        random_state=seed,
    ).fit_transform(normalized)
    cluster_count = min(clusters, max(2, features.shape[0] // 3), features.shape[0] - 1)
    labels = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=seed,
        n_init=10,
        batch_size=min(512, features.shape[0]),
    ).fit_predict(cluster_features)
    desired = max(2, int(math.ceil(validation_fraction * features.shape[0])))
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    count = 0
    for cluster in rng.permutation(np.unique(labels)):
        cluster_size = int(np.sum(labels == cluster))
        if count + cluster_size >= features.shape[0] - 2:
            continue
        selected.append(int(cluster))
        count += cluster_size
        if count >= desired:
            break
    require(selected, "Unable to select a validation cluster")
    validation = np.flatnonzero(np.isin(labels, selected))
    train = np.flatnonzero(~np.isin(labels, selected))
    require(len(train) >= 3 and len(validation) >= 2, "Invalid cluster holdout")
    require(set(labels[train]).isdisjoint(set(labels[validation])), "Cluster leakage detected")
    return train, validation, labels.astype(np.int64)


def fit_residual_pca(values: np.ndarray, rank: int, seed: int) -> ResidualPCA:
    require(values.ndim == 2 and values.shape[0] >= 2, "Invalid PCA matrix")
    effective_rank = min(rank, values.shape[0] - 1, values.shape[1])
    require(effective_rank >= 1, "PCA rank is zero")
    model = PCA(
        n_components=effective_rank,
        svd_solver="full" if effective_rank == min(values.shape) else "randomized",
        random_state=seed,
    ).fit(values)
    return ResidualPCA(
        mean=np.asarray(model.mean_, dtype=np.float32),
        components=np.asarray(model.components_, dtype=np.float32),
    )


def fit_ridge_route(
    source_residual: np.ndarray,
    recipient_residual: np.ndarray,
    gene_mask: np.ndarray,
    sample_weights: np.ndarray,
    source_rank: int,
    recipient_rank: int,
    alpha: float,
    seed: int,
) -> RidgeRoute:
    require(source_residual.shape == recipient_residual.shape, "Route effect shape mismatch")
    require(gene_mask.shape == (source_residual.shape[1],), "Route gene-mask mismatch")
    require(bool(gene_mask.any()), "No genes are shared by source and recipient atlases")
    require(alpha >= 0 and np.isfinite(alpha), "Ridge alpha must be nonnegative")
    source_pca = fit_residual_pca(source_residual[:, gene_mask], source_rank, seed)
    recipient_pca = fit_residual_pca(
        recipient_residual[:, gene_mask], recipient_rank, seed + 1
    )
    source_scores = source_pca.transform(source_residual[:, gene_mask]).astype(np.float64)
    recipient_scores = recipient_pca.transform(recipient_residual[:, gene_mask]).astype(np.float64)
    coefficient, intercept = fit_weighted_ridge_scores(
        source_scores, recipient_scores, sample_weights, alpha
    )
    return RidgeRoute(
        source_pca=source_pca,
        recipient_pca=recipient_pca,
        coefficient=coefficient,
        intercept=intercept,
        gene_mask=np.asarray(gene_mask, dtype=bool),
        alpha=float(alpha),
    )


def fit_weighted_ridge_scores(
    source_scores: np.ndarray,
    recipient_scores: np.ndarray,
    sample_weights: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a multi-output weighted ridge map between two PCA score spaces."""

    require(source_scores.shape[0] == recipient_scores.shape[0], "Score row mismatch")
    weights = np.asarray(sample_weights, dtype=np.float64)
    require(weights.shape == (source_scores.shape[0],), "Route weight shape mismatch")
    require(np.all(weights > 0), "Route sample weights must be positive")
    weights = weights / weights.mean()
    source_mean = np.average(source_scores, axis=0, weights=weights)
    recipient_mean = np.average(recipient_scores, axis=0, weights=weights)
    centered_source = source_scores - source_mean
    centered_recipient = recipient_scores - recipient_mean
    gram = centered_source.T @ (weights[:, None] * centered_source)
    gram.flat[:: gram.shape[0] + 1] += alpha
    right = centered_source.T @ (weights[:, None] * centered_recipient)
    try:
        coefficient = np.linalg.solve(gram, right)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.pinv(gram) @ right
    intercept = recipient_mean - source_mean @ coefficient
    return (
        coefficient.astype(np.float32),
        intercept.astype(np.float32),
    )


def apply_ridge_route(route: RidgeRoute, source_residual: np.ndarray) -> np.ndarray:
    require(source_residual.ndim == 2, "Source residual must be two-dimensional")
    require(route.gene_mask.shape == (source_residual.shape[1],), "Route gene-mask mismatch")
    source_scores = route.source_pca.transform(source_residual[:, route.gene_mask])
    recipient_scores = source_scores @ route.coefficient + route.intercept[None, :]
    routed = source_residual.astype(np.float32, copy=True)
    routed[:, route.gene_mask] = route.recipient_pca.inverse_transform(recipient_scores)
    return routed


def compose_effects(
    common: np.ndarray,
    source_residual: np.ndarray,
    routed_residual: np.ndarray,
    residual_scale: float,
    route_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    require(source_residual.shape == routed_residual.shape, "Residual shape mismatch")
    require(common.shape == (source_residual.shape[1],), "Common shape mismatch")
    mixed = (1.0 - route_scale) * source_residual + route_scale * routed_residual
    learned = (residual_scale * mixed).astype(np.float32)
    return (common[None, :] + learned).astype(np.float32), learned


def scorer_proxy_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    target_names: Sequence[str],
    gene_names: Sequence[str],
    top_k: int,
) -> dict[str, float]:
    """Approximate the six VCC members in effect space for model selection."""

    require(prediction.shape == truth.shape, "Prediction/truth shape mismatch")
    require(prediction.shape == (len(target_names), len(gene_names)), "Metric axis mismatch")
    require(top_k > 0, "top-k must be positive")
    lookup = {name: index for index, name in enumerate(gene_names)}
    own = np.asarray([lookup.get(name, -1) for name in target_names], dtype=np.int64)
    row_mask = np.ones(prediction.shape, dtype=bool)
    valid = own >= 0
    row_mask[np.flatnonzero(valid), own[valid]] = False

    squared_error = float(np.sum(np.square(prediction - truth), where=row_mask))
    squared_signal = float(np.sum(np.square(truth), where=row_mask))
    mse_ratio = squared_error / max(squared_signal, 1e-12)
    absolute_error = float(np.sum(np.abs(prediction - truth), where=row_mask))
    absolute_signal = float(np.sum(np.abs(truth), where=row_mask))
    nmae = absolute_error / max(absolute_signal, 1e-12)

    panel_mask = np.ones(len(gene_names), dtype=bool)
    panel_mask[own[own >= 0]] = False
    require(bool(panel_mask.any()), "Panel mask removed every gene")
    predicted_panel = prediction[:, panel_mask].astype(np.float64, copy=False)
    true_panel = truth[:, panel_mask].astype(np.float64, copy=False)
    numerator = predicted_panel @ true_panel.T
    denominator = np.linalg.norm(predicted_panel, axis=1)[:, None] * np.linalg.norm(
        true_panel, axis=1
    )[None, :]
    similarities = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    diagonal = np.diag(similarities)
    greater = np.sum(similarities > diagonal[:, None] + 1e-12, axis=1)
    tied = np.sum(np.isclose(similarities, diagonal[:, None], atol=1e-12), axis=1) - 1
    rank_fraction = (greater + 0.5 * tied) / max(len(target_names) - 1, 1)
    pds = float(np.mean(1.0 - rank_fraction))

    fidelities: list[float] = []
    reaches: list[float] = []
    jaccards: list[float] = []
    for row in range(len(target_names)):
        allowed = np.flatnonzero(row_mask[row])
        k = min(top_k, len(allowed))
        require(k > 0, "No genes remain for top-set metrics")
        true_values = np.abs(truth[row, allowed])
        prediction_values = np.abs(prediction[row, allowed])
        true_top = allowed[np.argpartition(true_values, len(allowed) - k)[-k:]]
        predicted_top = allowed[
            np.argpartition(prediction_values, len(allowed) - k)[-k:]
        ]
        fidelities.append(
            float(np.mean(np.sign(prediction[row, true_top]) == np.sign(truth[row, true_top])))
        )
        intersection = len(set(true_top.tolist()) & set(predicted_top.tolist()))
        reaches.append(intersection / k)
        jaccards.append(intersection / max(2 * k - intersection, 1))

    fidelity = float(np.mean(fidelities))
    reach = float(np.mean(reaches))
    jaccard = float(np.mean(jaccards))
    combined = float(
        np.mean(
            [
                np.clip((pds - 0.5) / 0.45, -1.0, 1.0),
                1.0 - np.clip(mse_ratio, 0.0, 1.0),
                1.0 - np.clip(nmae, 0.0, 1.0),
                2.0 * fidelity - 1.0,
                reach,
                jaccard,
            ]
        )
    )
    return {
        "pds_tie_aware_panel_masked": pds,
        "expression_mse_ratio": float(mse_ratio),
        "nmae": float(nmae),
        "direction_fidelity": fidelity,
        "top_set_reach": reach,
        "top_set_jaccard": jaccard,
        "six_member_proxy": combined,
    }


def select_route_hyperparameters(
    train_source: np.ndarray,
    train_recipient: np.ndarray,
    validation_source: np.ndarray,
    validation_truth: np.ndarray,
    train_weights: np.ndarray,
    common: np.ndarray,
    validation_targets: Sequence[str],
    genes: Sequence[str],
    gene_mask: np.ndarray,
    source_rank: int,
    recipient_rank: int,
    ridge_grid: Sequence[float],
    residual_scale_grid: Sequence[float],
    route_scale_grid: Sequence[float],
    top_k: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    source_pca = fit_residual_pca(
        train_source[:, gene_mask], source_rank, seed
    )
    recipient_pca = fit_residual_pca(
        train_recipient[:, gene_mask], recipient_rank, seed + 1
    )
    train_source_scores = source_pca.transform(train_source[:, gene_mask])
    train_recipient_scores = recipient_pca.transform(train_recipient[:, gene_mask])
    for alpha in ridge_grid:
        coefficient, intercept = fit_weighted_ridge_scores(
            train_source_scores,
            train_recipient_scores,
            train_weights,
            float(alpha),
        )
        route = RidgeRoute(
            source_pca=source_pca,
            recipient_pca=recipient_pca,
            coefficient=coefficient,
            intercept=intercept,
            gene_mask=np.asarray(gene_mask, dtype=bool),
            alpha=float(alpha),
        )
        routed = apply_ridge_route(route, validation_source)
        for residual_scale in residual_scale_grid:
            for route_scale in route_scale_grid:
                prediction, _ = compose_effects(
                    common,
                    validation_source,
                    routed,
                    float(residual_scale),
                    float(route_scale),
                )
                metrics = scorer_proxy_metrics(
                    prediction,
                    validation_truth,
                    validation_targets,
                    genes,
                    top_k,
                )
                candidates.append(
                    {
                        "ridge_alpha": float(alpha),
                        "residual_scale": float(residual_scale),
                        "route_scale": float(route_scale),
                        "metrics": metrics,
                    }
                )
    require(candidates, "Hyperparameter grid is empty")

    def selection_key(candidate: dict[str, Any]) -> tuple[float, ...]:
        metrics = candidate["metrics"]
        return (
            metrics["six_member_proxy"],
            -metrics["expression_mse_ratio"],
            -metrics["nmae"],
            metrics["pds_tie_aware_panel_masked"],
            metrics["direction_fidelity"],
            -abs(candidate["residual_scale"] - 1.0),
            -abs(candidate["route_scale"] - 1.0),
            -candidate["ridge_alpha"],
        )

    return max(candidates, key=selection_key), candidates


def _route_checkpoint(route: RidgeRoute) -> dict[str, torch.Tensor | float]:
    return {
        "source_pca_mean": torch.from_numpy(route.source_pca.mean),
        "source_pca_components": torch.from_numpy(route.source_pca.components),
        "recipient_pca_mean": torch.from_numpy(route.recipient_pca.mean),
        "recipient_pca_components": torch.from_numpy(route.recipient_pca.components),
        "ridge_coefficient": torch.from_numpy(route.coefficient),
        "ridge_intercept": torch.from_numpy(route.intercept),
        "route_gene_mask": torch.from_numpy(route.gene_mask),
        "ridge_alpha": route.alpha,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    inputs = (
        args.query_atlas,
        args.source_essential_atlas,
        args.recipient_essential_atlas,
        args.fallback_prior,
        args.challenge_targets,
    )
    for path in inputs:
        require(path.is_file(), f"Missing input: {path}")
    if args.challenge_genes is not None:
        require(
            args.challenge_genes.is_file(),
            f"Missing challenge gene axis: {args.challenge_genes}",
        )
    if args.esm2_embeddings is not None:
        require(args.esm2_embeddings.is_file(), f"Missing ESM2 embeddings: {args.esm2_embeddings}")
    for output in (args.output_npz, args.output_checkpoint, args.output_json):
        require(not output.exists(), f"Refusing to overwrite {output}")
    require(args.source_rank > 0 and args.recipient_rank > 0, "PCA ranks must be positive")
    require(args.top_k > 0, "top-k must be positive")
    ridge_grid = parse_float_grid(args.ridge_grid, "ridge-grid")
    residual_scale_grid = parse_float_grid(args.residual_scale_grid, "residual-scale-grid")
    route_scale_grid = parse_float_grid(args.route_scale_grid, "route-scale-grid")

    query = load_effect_atlas(args.query_atlas, require_cell_counts=True)
    source = load_effect_atlas(args.source_essential_atlas, require_cell_counts=True)
    recipient = load_effect_atlas(args.recipient_essential_atlas, require_cell_counts=True)
    fallback = load_effect_atlas(args.fallback_prior, require_cell_counts=False)
    challenge_targets = read_named_column(
        args.challenge_targets, args.target_column, ("target", "targets", "gene")
    )
    fallback_target_set = set(fallback.target_names)
    missing_fallback = [name for name in challenge_targets if name not in fallback_target_set]
    require(not missing_fallback, f"Fallback misses challenge targets: {missing_fallback[:8]}")
    if args.challenge_genes is None:
        output_genes = list(fallback.gene_names)
    else:
        output_genes = read_named_column(
            args.challenge_genes, "gene_name", ("gene", "genes")
        )

    paired_targets = sorted(set(source.target_names) & set(recipient.target_names))
    require(len(paired_targets) >= 6, "Fewer than six paired essential targets")
    embedding_map = load_embedding_map(
        args.fallback_prior,
        args.esm2_embeddings,
        paired_targets,
        allow_missing=True,
    )
    clustered_paired_indices = np.asarray(
        [index for index, name in enumerate(paired_targets) if name in embedding_map],
        dtype=np.int64,
    )
    missing_embedding_targets = [
        name for name in paired_targets if name not in embedding_map
    ]
    require(
        len(clustered_paired_indices) >= 6,
        "Fewer than six paired targets have ESM2 embeddings",
    )
    paired_features = np.stack(
        [embedding_map[paired_targets[index]] for index in clustered_paired_indices]
    ).astype(np.float32)
    local_train, local_validation, cluster_labels = make_cluster_holdout(
        paired_features,
        args.clusters,
        args.validation_fraction,
        args.seed,
    )
    train_indices = clustered_paired_indices[local_train]
    validation_indices = clustered_paired_indices[local_validation]

    source_aligned, _, source_gene_mask = align_effects(source, paired_targets, output_genes)
    recipient_aligned, _, recipient_gene_mask = align_effects(
        recipient, paired_targets, output_genes
    )
    route_gene_mask = source_gene_mask & recipient_gene_mask
    require(int(route_gene_mask.sum()) >= 2, "Fewer than two route genes overlap")
    source_lookup = {name: index for index, name in enumerate(source.target_names)}
    recipient_lookup = {name: index for index, name in enumerate(recipient.target_names)}
    source_counts = np.asarray(
        [source.target_cell_counts[source_lookup[name]] for name in paired_targets], dtype=np.int64
    )
    recipient_counts = np.asarray(
        [recipient.target_cell_counts[recipient_lookup[name]] for name in paired_targets],
        dtype=np.int64,
    )
    source_reliability = reliability_from_counts(source_counts, args.reliability_tau)
    recipient_reliability = reliability_from_counts(recipient_counts, args.reliability_tau)
    route_weights = np.sqrt(
        np.maximum(source_reliability * recipient_reliability, 1e-6)
    ).astype(np.float32)
    (
        selection_common,
        train_source_residual,
        train_recipient_residual,
        validation_source_residual,
    ) = prepare_holdout_residuals(
        source_aligned,
        recipient_aligned,
        source_reliability,
        recipient_reliability,
        train_indices,
        validation_indices,
        paired_targets,
        output_genes,
    )

    selected, candidates = select_route_hyperparameters(
        train_source_residual,
        train_recipient_residual,
        validation_source_residual,
        recipient_aligned[validation_indices],
        route_weights[train_indices],
        selection_common,
        [paired_targets[index] for index in validation_indices],
        output_genes,
        route_gene_mask,
        args.source_rank,
        args.recipient_rank,
        ridge_grid,
        residual_scale_grid,
        route_scale_grid,
        args.top_k,
        args.seed,
    )
    print(
        "Selected "
        f"ridge_alpha={selected['ridge_alpha']} "
        f"residual_scale={selected['residual_scale']} "
        f"route_scale={selected['route_scale']} "
        f"proxy={selected['metrics']['six_member_proxy']:.6f}",
        flush=True,
    )

    source_common = weighted_common_response(
        source_aligned,
        np.maximum(source_reliability, 1e-6),
        paired_targets,
        output_genes,
    )
    recipient_common = weighted_common_response(
        recipient_aligned,
        np.maximum(recipient_reliability, 1e-6),
        paired_targets,
        output_genes,
    )
    source_residual = residualize(
        source_aligned, source_common, paired_targets, output_genes
    ) * source_reliability[:, None]
    recipient_residual = residualize(
        recipient_aligned, recipient_common, paired_targets, output_genes
    )
    full_route = fit_ridge_route(
        source_residual,
        recipient_residual,
        route_gene_mask,
        route_weights,
        args.source_rank,
        args.recipient_rank,
        selected["ridge_alpha"],
        args.seed + 10_000,
    )
    query_challenge, direct_mask, query_gene_mask = align_effects(
        query, challenge_targets, output_genes
    )
    fallback_challenge, fallback_available, fallback_gene_mask = align_effects(
        fallback, challenge_targets, output_genes
    )
    require(bool(fallback_available.all()), "Fallback alignment unexpectedly lost a target")
    query_lookup = {name: index for index, name in enumerate(query.target_names)}
    challenge_counts = np.zeros(len(challenge_targets), dtype=np.int64)
    direct_reliability = np.zeros(len(challenge_targets), dtype=np.float32)
    for index, target in enumerate(challenge_targets):
        if direct_mask[index]:
            count = int(query.target_cell_counts[query_lookup[target]])
            challenge_counts[index] = count
            direct_reliability[index] = reliability_from_counts(
                np.asarray([count]), args.reliability_tau
            )[0]

    panel_source_effects = merge_direct_and_fallback_effects(
        query_challenge,
        fallback_challenge,
        direct_mask,
        query_gene_mask,
    )
    positive_reliability = direct_reliability[direct_reliability > 0]
    fallback_common_weight = (
        float(np.median(positive_reliability)) if len(positive_reliability) else 1.0
    )
    panel_common_weights = np.where(
        direct_mask,
        np.maximum(direct_reliability, 1e-6),
        fallback_common_weight,
    ).astype(np.float32)
    shared_response = weighted_common_response(
        panel_source_effects,
        panel_common_weights,
        challenge_targets,
        output_genes,
    )
    fallback_only_gene_mask = fallback_gene_mask & ~query_gene_mask
    if bool(fallback_only_gene_mask.any()):
        fallback_only_common = weighted_common_response(
            panel_source_effects,
            np.ones(len(challenge_targets), dtype=np.float32),
            challenge_targets,
            output_genes,
        )
        shared_response[fallback_only_gene_mask] = fallback_only_common[
            fallback_only_gene_mask
        ]
    base_residual = residualize(
        panel_source_effects, shared_response, challenge_targets, output_genes
    )
    base_residual = shrink_measured_residuals(
        base_residual,
        direct_mask,
        query_gene_mask,
        direct_reliability,
    )
    routed_residual = apply_ridge_route(full_route, base_residual)
    calibrated_effects, learned_residual = compose_effects(
        shared_response,
        base_residual,
        routed_residual,
        selected["residual_scale"],
        selected["route_scale"],
    )
    require(np.isfinite(calibrated_effects).all(), "Calibrated effects are non-finite")
    fallback_mask = ~direct_mask

    for output in (args.output_npz, args.output_checkpoint, args.output_json):
        output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        effects=calibrated_effects,
        target_names=np.asarray(challenge_targets, dtype="U"),
        gene_names=np.asarray(output_genes, dtype="U"),
        target_cell_counts=challenge_counts,
        genes=np.asarray(output_genes, dtype="U"),
        targets=np.asarray(challenge_targets, dtype="U"),
        common_log_fold=shared_response,
        learned_log_fold=learned_residual,
        direct_mask=direct_mask,
        fallback_mask=fallback_mask,
        source_reliability=direct_reliability,
        route_gene_mask=route_gene_mask,
        query_gene_mask=query_gene_mask,
        fallback_gene_mask=fallback_gene_mask,
        ridge_alpha=np.asarray([selected["ridge_alpha"]], dtype=np.float64),
        residual_scale=np.asarray([selected["residual_scale"]], dtype=np.float32),
        route_scale=np.asarray([selected["route_scale"]], dtype=np.float32),
    )
    checkpoint = {
        "schema": SCHEMA,
        "route": _route_checkpoint(full_route),
        "shared_response": torch.from_numpy(shared_response),
        "selection": selected,
        "target_names": challenge_targets,
        "gene_names": output_genes,
        "paired_target_names": paired_targets,
        "direct_mask": torch.from_numpy(direct_mask),
        "fallback_mask": torch.from_numpy(fallback_mask),
        "source_reliability": torch.from_numpy(direct_reliability),
    }
    torch.save(checkpoint, args.output_checkpoint)

    input_paths = {
        "query_atlas": args.query_atlas,
        "source_essential_atlas": args.source_essential_atlas,
        "recipient_essential_atlas": args.recipient_essential_atlas,
        "fallback_prior": args.fallback_prior,
        "challenge_targets": args.challenge_targets,
    }
    if args.challenge_genes is not None:
        input_paths["challenge_genes"] = args.challenge_genes
    if args.esm2_embeddings is not None:
        input_paths["esm2_embeddings"] = args.esm2_embeddings
    report = {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "inputs": {name: describe_file(path) for name, path in input_paths.items()},
        "outputs": {
            "effect_prior": describe_file(args.output_npz),
            "checkpoint": describe_file(args.output_checkpoint),
        },
        "dimensions": {
            "query_targets": len(query.target_names),
            "paired_targets": len(paired_targets),
            "esm2_cluster_eligible_paired_targets": len(clustered_paired_indices),
            "esm2_missing_paired_targets": len(missing_embedding_targets),
            "training_targets": len(train_indices),
            "validation_targets": len(validation_indices),
            "challenge_targets": len(challenge_targets),
            "output_genes": len(output_genes),
            "route_genes": int(route_gene_mask.sum()),
            "source_pca_rank": int(full_route.source_pca.components.shape[0]),
            "recipient_pca_rank": int(full_route.recipient_pca.components.shape[0]),
            "esm2_embedding_dim": int(paired_features.shape[1]),
        },
        "input_effect_reductions": {
            "query_atlas": query.context_reduction,
            "source_essential_atlas": source.context_reduction,
            "recipient_essential_atlas": recipient.context_reduction,
            "fallback_prior": fallback.context_reduction,
        },
        "coverage": {
            "direct_targets": int(direct_mask.sum()),
            "fallback_targets": int(fallback_mask.sum()),
            "direct_target_names": [
                target
                for target, direct in zip(challenge_targets, direct_mask, strict=True)
                if direct
            ],
            "fallback_target_names": [
                target
                for target, fallback_value in zip(challenge_targets, fallback_mask, strict=True)
                if fallback_value
            ],
            "query_genes_on_output_axis": int(query_gene_mask.sum()),
            "fallback_genes_on_output_axis": int(fallback_gene_mask.sum()),
            "direct_coordinates_from_query": int(direct_mask.sum() * query_gene_mask.sum()),
            "direct_coordinates_from_fallback": int(
                direct_mask.sum() * np.sum(fallback_gene_mask & ~query_gene_mask)
            ),
        },
        "holdout": {
            "method": "whole-esm2-cluster",
            "selection_preprocessing_fit_scope": "training-targets-only",
            "final_refit_scope": "all-paired-targets-after-selection",
            "cluster_count": int(len(np.unique(cluster_labels))),
            "validation_clusters": sorted(
                np.unique(cluster_labels[local_validation]).astype(int).tolist()
            ),
            "training_target_names": [paired_targets[index] for index in train_indices],
            "validation_target_names": [paired_targets[index] for index in validation_indices],
            "targets_without_esm2_embedding": missing_embedding_targets,
        },
        "selection": {
            "selected": selected,
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
        "reliability": {
            "formula": "n_cells/(n_cells+tau)",
            "shrink_scope": "direct-target-and-query-gene-coordinates-only",
            "tau": args.reliability_tau,
            "paired_source_min": float(source_reliability.min()),
            "paired_source_median": float(np.median(source_reliability)),
            "paired_source_max": float(source_reliability.max()),
            "fallback_common_weight": fallback_common_weight,
        },
        "contract": {
            "effect_space": "log1p-cp50000-pseudobulk-delta",
            "shared_response_definition": "reliability-weighted-300-target-panel-mean",
            "shared_response_is_unrouted": True,
            "routing_applies_to": "target-specific-residual-only",
            "direct_source": "same-target-k562-gwps-residual",
            "direct_missing_gene_source": "esm2-predicted-fallback-effect",
            "missing_target_source": "esm2-predicted-fallback-residual",
        },
        "runtime": {
            "argv": sys.argv,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
            "torch": torch.__version__,
            "host": socket.gethostname(),
            "seed": args.seed,
        },
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "coverage": report["coverage"]}, indent=2))
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
