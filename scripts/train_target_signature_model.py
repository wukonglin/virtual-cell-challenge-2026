#!/usr/bin/env python3
"""Train an ESM2-conditioned downstream CRISPRi signature model.

The model separates a shared perturbation surface from a target-specific
residual.  It learns the residual in a low-rank gene-program basis and uses
pointwise, pairwise, cosine, and magnitude objectives.  Model selection is
performed on whole ESM2 clusters, which is stricter than a random target split
and better reflects the VCC zero-overlap target setting.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


SCHEMA = "vcc-target-signature-model-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("effect_atlas", type=Path)
    parser.add_argument("esm2_embeddings", type=Path)
    parser.add_argument("challenge_targets", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("output_checkpoint", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--target-column", default="target_gene")
    parser.add_argument(
        "--challenge-genes",
        type=Path,
        default=None,
        help="Optional headerless challenge gene axis; output is restricted to its overlap.",
    )
    parser.add_argument("--input-rank", type=int, default=256)
    parser.add_argument("--output-rank", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(name: str, require_cuda: bool) -> torch.device:
    device = torch.device(name)
    if require_cuda and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("A CUDA device was required but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def read_challenge_targets(path: Path, column: str) -> list[str]:
    table = pd.read_csv(path)
    require(column in table, f"Missing challenge target column {column}")
    targets = table[column].astype(str).tolist()
    require(len(targets) == len(set(targets)), "Challenge targets must be unique")
    require(len(targets) > 1, "Challenge target list is empty")
    return targets


def load_esm2(path: Path, names: list[str]) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(payload, dict), "ESM2 file must contain a target-to-tensor map")
    missing = [name for name in names if name not in payload]
    require(not missing, f"Missing ESM2 embeddings for {len(missing)} targets: {missing[:8]}")
    dimensions = {int(payload[name].numel()) for name in names}
    require(len(dimensions) == 1, "ESM2 embeddings have inconsistent dimensions")
    matrix = torch.stack([payload[name].float().reshape(-1) for name in names]).numpy()
    require(np.isfinite(matrix).all(), "ESM2 embeddings contain non-finite values")
    return matrix


def weighted_common_effect(
    effects: np.ndarray,
    weights: np.ndarray,
    target_names: list[str],
    gene_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a common response without leaking each target's self-knockdown."""

    require(effects.shape == (len(target_names), len(gene_names)), "Effect shape mismatch")
    gene_lookup = {name: index for index, name in enumerate(gene_names)}
    numerator = np.sum(effects * weights[:, None], axis=0, dtype=np.float64)
    denominator = np.full(effects.shape[1], weights.sum(dtype=np.float64), dtype=np.float64)
    own_indices = np.full(len(target_names), -1, dtype=np.int64)
    for row, target in enumerate(target_names):
        column = gene_lookup.get(target, -1)
        own_indices[row] = column
        if column >= 0:
            numerator[column] -= float(weights[row]) * float(effects[row, column])
            denominator[column] -= float(weights[row])
    require(np.all(denominator > 0), "Common-effect denominator is not positive")
    common = (numerator / denominator).astype(np.float32)
    residual = effects.astype(np.float32, copy=True) - common[None, :]
    valid = own_indices >= 0
    residual[np.flatnonzero(valid), own_indices[valid]] = 0.0
    return common, residual


def pca_fit_transform(
    train: torch.Tensor,
    other: torch.Tensor,
    rank: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    mean = train.mean(dim=0)
    centered = train - mean
    effective_rank = min(rank, centered.shape[0] - 1, centered.shape[1])
    require(effective_rank >= 2, "PCA rank is too small after shape constraints")
    _, _, vectors = torch.pca_lowrank(centered, q=effective_rank, center=False, niter=4)
    train_scores = centered @ vectors
    other_scores = (other - mean) @ vectors
    return train_scores, other_scores, mean, vectors.T.contiguous()


class SignatureNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


@dataclass(frozen=True)
class Objective:
    name: str
    pair_weight: float
    cosine_weight: float
    magnitude_weight: float


OBJECTIVES = (
    Objective("absolute", pair_weight=0.0, cosine_weight=0.10, magnitude_weight=0.05),
    Objective("balanced", pair_weight=0.50, cosine_weight=0.25, magnitude_weight=0.10),
    Objective("rank", pair_weight=1.00, cosine_weight=0.50, magnitude_weight=0.10),
)


def objective_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    score_scale: torch.Tensor,
    objective: Objective,
) -> torch.Tensor:
    absolute = F.smooth_l1_loss(prediction, target)
    physical_prediction = prediction * score_scale
    physical_target = target * score_scale
    cosine = (1.0 - F.cosine_similarity(physical_prediction, physical_target, dim=1)).mean()
    magnitude = F.smooth_l1_loss(
        torch.log1p(torch.linalg.vector_norm(physical_prediction, dim=1)),
        torch.log1p(torch.linalg.vector_norm(physical_target, dim=1)),
    )
    if objective.pair_weight > 0 and prediction.shape[0] > 1:
        permutation = torch.randperm(prediction.shape[0], device=prediction.device)
        pairwise = F.smooth_l1_loss(
            prediction - prediction[permutation],
            target - target[permutation],
        )
    else:
        pairwise = prediction.new_zeros(())
    return (
        absolute
        + objective.pair_weight * pairwise
        + objective.cosine_weight * cosine
        + objective.magnitude_weight * magnitude
    )


def train_network(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor | None,
    validation_y: torch.Tensor | None,
    score_scale: torch.Tensor,
    objective: Objective,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[SignatureNetwork, int, float]:
    torch.manual_seed(seed)
    model = SignatureNetwork(train_x.shape[1], hidden_dim, train_y.shape[1]).to(train_x.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_value = math.inf
    patience = max(20, epochs // 4)
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator(device=train_x.device).manual_seed(seed + epoch)
        order = torch.randperm(train_x.shape[0], generator=generator, device=train_x.device)
        for begin in range(0, train_x.shape[0], batch_size):
            rows = order[begin : begin + batch_size]
            prediction = model(train_x[rows])
            loss = objective_loss(prediction, train_y[rows], score_scale, objective)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            if validation_x is None or validation_y is None:
                value = float(
                    objective_loss(model(train_x), train_y, score_scale, objective).item()
                )
            else:
                value = float(
                    objective_loss(
                        model(validation_x), validation_y, score_scale, objective
                    ).item()
                )
        if value < best_value - 1e-6:
            best_value = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"objective={objective.name} epoch={epoch} validation_loss={value:.6f}",
                flush=True,
            )
        if validation_x is not None and stale >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_epoch, best_value


def metric_mask(
    target_names: list[str], gene_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {name: index for index, name in enumerate(gene_names)}
    own = np.asarray([lookup.get(name, -1) for name in target_names], dtype=np.int64)
    panel_mask = np.ones(len(gene_names), dtype=bool)
    panel_indices = own[own >= 0]
    panel_mask[panel_indices] = False
    return own, panel_mask


def proxy_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    target_names: list[str],
    gene_names: list[str],
    top_k: int = 200,
) -> dict[str, float]:
    require(prediction.shape == truth.shape, "Prediction/truth shape mismatch")
    own, panel_mask = metric_mask(target_names, gene_names)
    row_mask = np.ones_like(truth, dtype=bool)
    valid_own = own >= 0
    row_mask[np.flatnonzero(valid_own), own[valid_own]] = False

    squared_error = np.sum(np.square(prediction - truth), where=row_mask)
    squared_signal = np.sum(np.square(truth), where=row_mask)
    mse_ratio = float(squared_error / max(squared_signal, 1e-12))
    absolute_error = np.sum(np.abs(prediction - truth), where=row_mask)
    absolute_signal = np.sum(np.abs(truth), where=row_mask)
    nmae = float(absolute_error / max(absolute_signal, 1e-12))

    pred_pds = prediction[:, panel_mask].astype(np.float64, copy=False)
    true_pds = truth[:, panel_mask].astype(np.float64, copy=False)
    pred_norm = np.linalg.norm(pred_pds, axis=1, keepdims=True)
    true_norm = np.linalg.norm(true_pds, axis=1, keepdims=True)
    similarities = (pred_pds @ true_pds.T) / np.maximum(pred_norm * true_norm.T, 1e-12)
    diagonal = np.diag(similarities)
    ranks = np.sum(similarities > diagonal[:, None], axis=1)
    ties = np.sum(np.isclose(similarities, diagonal[:, None], atol=1e-12), axis=1) - 1
    denominator = max(len(target_names) - 1, 1)
    pds = float(np.mean(1.0 - (ranks + 0.5 * ties) / denominator))

    fidelities: list[float] = []
    reaches: list[float] = []
    jaccards: list[float] = []
    k = min(top_k, truth.shape[1] - 1)
    for row in range(truth.shape[0]):
        allowed = row_mask[row]
        allowed_indices = np.flatnonzero(allowed)
        true_order = allowed_indices[
            np.argpartition(np.abs(truth[row, allowed]), -k)[-k:]
        ]
        pred_order = allowed_indices[
            np.argpartition(np.abs(prediction[row, allowed]), -k)[-k:]
        ]
        fidelities.append(
            float(np.mean(np.sign(prediction[row, true_order]) == np.sign(truth[row, true_order])))
        )
        intersection = len(set(true_order.tolist()) & set(pred_order.tolist()))
        reaches.append(intersection / k)
        jaccards.append(intersection / (2 * k - intersection))

    fidelity = float(np.mean(fidelities))
    reach = float(np.mean(reaches))
    jaccard = float(np.mean(jaccards))
    scaled_proxy = float(
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
        "pds_cosine_proxy": pds,
        "expression_mse_ratio_proxy": mse_ratio,
        "nmae_proxy": nmae,
        "direction_fidelity_proxy": fidelity,
        "direction_reach_proxy": reach,
        "set_jaccard_proxy": jaccard,
        "six_member_scaled_proxy": scaled_proxy,
    }


def select_scales(
    residual_prediction: np.ndarray,
    truth: np.ndarray,
    common: np.ndarray,
    targets: list[str],
    genes: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for common_scale in (0.0, 0.25, 0.50, 0.75, 1.0, 1.25):
        for residual_scale in (0.0, 0.25, 0.50, 0.75, 1.0, 1.25):
            prediction = common_scale * common[None, :] + residual_scale * residual_prediction
            metrics = proxy_metrics(prediction, truth, targets, genes)
            candidates.append(
                {
                    "common_scale": common_scale,
                    "residual_scale": residual_scale,
                    "metrics": metrics,
                }
            )
    selected = max(
        candidates, key=lambda item: item["metrics"]["six_member_scaled_proxy"]
    )
    return selected, candidates


def make_cluster_split(
    features: np.ndarray,
    clusters: int,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(0 < validation_fraction < 0.5, "validation-fraction must be in (0, 0.5)")
    cluster_rank = min(64, features.shape[0] - 1, features.shape[1])
    cluster_features = PCA(
        n_components=cluster_rank, svd_solver="randomized", random_state=seed
    ).fit_transform(features)
    cluster_labels = MiniBatchKMeans(
        n_clusters=min(clusters, max(2, features.shape[0] // 20)),
        random_state=seed,
        n_init=10,
        batch_size=512,
    ).fit_predict(cluster_features)
    rng = np.random.default_rng(seed)
    validation_clusters: list[int] = []
    validation_count = 0
    desired = int(math.ceil(validation_fraction * len(features)))
    for cluster in rng.permutation(np.unique(cluster_labels)):
        validation_clusters.append(int(cluster))
        validation_count += int(np.sum(cluster_labels == cluster))
        if validation_count >= desired:
            break
    validation_mask = np.isin(cluster_labels, validation_clusters)
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    require(len(train_indices) > len(validation_indices) > 1, "Invalid cluster split")
    return train_indices, validation_indices, cluster_labels


def fit_representation(
    features: np.ndarray,
    effects: np.ndarray,
    counts: np.ndarray,
    target_names: list[str],
    gene_names: list[str],
    train_indices: np.ndarray,
    other_indices: np.ndarray,
    input_rank: int,
    output_rank: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor | np.ndarray]:
    train_weights = np.sqrt(counts[train_indices].astype(np.float64))
    train_weights /= np.median(train_weights)
    train_weights = np.clip(train_weights, 0.5, 2.0).astype(np.float32)
    common, train_residual = weighted_common_effect(
        effects[train_indices],
        train_weights,
        [target_names[index] for index in train_indices],
        gene_names,
    )
    other_residual = effects[other_indices].astype(np.float32) - common[None, :]
    gene_lookup = {name: index for index, name in enumerate(gene_names)}
    for row, index in enumerate(other_indices):
        column = gene_lookup.get(target_names[index], -1)
        if column >= 0:
            other_residual[row, column] = 0.0

    normalized_features = features / np.maximum(
        np.linalg.norm(features, axis=1, keepdims=True), 1e-12
    )
    train_feature_tensor = torch.from_numpy(normalized_features[train_indices]).to(
        device=device, dtype=torch.float32
    )
    other_feature_tensor = torch.from_numpy(normalized_features[other_indices]).to(
        device=device, dtype=torch.float32
    )
    train_x, other_x, input_mean, input_components = pca_fit_transform(
        train_feature_tensor, other_feature_tensor, input_rank, seed
    )
    train_residual_tensor = torch.from_numpy(train_residual).to(device=device)
    other_residual_tensor = torch.from_numpy(other_residual).to(device=device)
    train_y_raw, other_y_raw, output_center, output_components = pca_fit_transform(
        train_residual_tensor, other_residual_tensor, output_rank, seed + 1
    )
    score_scale = torch.clamp(train_y_raw.std(dim=0, unbiased=False), min=1e-4)
    return {
        "common": common,
        "train_x": train_x,
        "other_x": other_x,
        "train_y": train_y_raw / score_scale,
        "other_y": other_y_raw / score_scale,
        "score_scale": score_scale,
        "input_mean": input_mean,
        "input_components": input_components,
        "output_center": output_center,
        "output_components": output_components,
    }


def decode_residual(
    standardized_scores: torch.Tensor,
    score_scale: torch.Tensor,
    output_center: torch.Tensor,
    output_components: torch.Tensor,
) -> np.ndarray:
    with torch.inference_mode():
        values = (
            output_center[None, :]
            + (standardized_scores * score_scale) @ output_components
        )
    return values.float().cpu().numpy()


def main() -> None:
    args = parse_args()
    started = time.time()
    for path in (args.effect_atlas, args.esm2_embeddings, args.challenge_targets):
        require(path.is_file(), f"Missing input: {path}")
    for output in (args.output_npz, args.output_checkpoint, args.output_json):
        require(not output.exists(), f"Refusing to overwrite {output}")
    require(args.epochs > 0 and args.batch_size > 0, "Invalid training schedule")
    require(args.input_rank > 1 and args.output_rank > 1, "PCA ranks must exceed one")
    device = resolve_device(args.device, args.require_cuda)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    atlas = np.load(args.effect_atlas, allow_pickle=False)
    effects = atlas["effects"].astype(np.float32)
    target_names = atlas["target_names"].astype(str).tolist()
    gene_names = atlas["gene_names"].astype(str).tolist()
    counts = atlas["target_cell_counts"].astype(np.int64)
    require(effects.shape == (len(target_names), len(gene_names)), "Atlas shape mismatch")
    challenge_targets = read_challenge_targets(args.challenge_targets, args.target_column)
    if args.challenge_genes is None:
        output_gene_mask = np.ones(len(gene_names), dtype=bool)
    else:
        require(args.challenge_genes.is_file(), f"Missing gene axis: {args.challenge_genes}")
        challenge_gene_axis = pd.read_csv(args.challenge_genes, header=None).iloc[:, 0].astype(str)
        challenge_gene_set = set(challenge_gene_axis)
        output_gene_mask = np.asarray(
            [name in challenge_gene_set for name in gene_names], dtype=bool
        )
        require(bool(output_gene_mask.any()), "Source and challenge gene axes do not overlap")
    output_gene_names = [
        name for name, keep in zip(gene_names, output_gene_mask, strict=True) if keep
    ]
    all_feature_names = target_names + challenge_targets
    all_features = load_esm2(args.esm2_embeddings, all_feature_names)
    train_features = all_features[: len(target_names)]
    challenge_features = all_features[len(target_names) :]

    train_indices, validation_indices, cluster_labels = make_cluster_split(
        train_features, args.clusters, args.validation_fraction, args.seed
    )
    representation = fit_representation(
        train_features,
        effects,
        counts,
        target_names,
        gene_names,
        train_indices,
        validation_indices,
        args.input_rank,
        args.output_rank,
        device,
        args.seed,
    )

    objective_reports: list[dict[str, Any]] = []
    selected_objective: Objective | None = None
    selected_epoch = 0
    best_proxy = -math.inf
    for objective_index, objective in enumerate(OBJECTIVES):
        model, best_epoch, validation_loss = train_network(
            representation["train_x"],  # type: ignore[arg-type]
            representation["train_y"],  # type: ignore[arg-type]
            representation["other_x"],  # type: ignore[arg-type]
            representation["other_y"],  # type: ignore[arg-type]
            representation["score_scale"],  # type: ignore[arg-type]
            objective,
            args.hidden_dim,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            args.seed + 100 * objective_index,
        )
        model.eval()
        with torch.inference_mode():
            validation_scores = model(representation["other_x"])  # type: ignore[arg-type]
        validation_residual = decode_residual(
            validation_scores,
            representation["score_scale"],  # type: ignore[arg-type]
            representation["output_center"],  # type: ignore[arg-type]
            representation["output_components"],  # type: ignore[arg-type]
        )
        selected_scales, scale_candidates = select_scales(
            validation_residual,
            effects[validation_indices],
            representation["common"],  # type: ignore[arg-type]
            [target_names[index] for index in validation_indices],
            gene_names,
        )
        report = {
            "objective": asdict(objective),
            "best_epoch": best_epoch,
            "validation_loss": validation_loss,
            "selected_scales": selected_scales,
            "scale_candidates": scale_candidates,
        }
        objective_reports.append(report)
        proxy = selected_scales["metrics"]["six_member_scaled_proxy"]
        print(
            f"objective={objective.name} selected_proxy={proxy:.6f} "
            f"common_scale={selected_scales['common_scale']} "
            f"residual_scale={selected_scales['residual_scale']}",
            flush=True,
        )
        if proxy > best_proxy:
            best_proxy = proxy
            selected_objective = objective
            selected_epoch = best_epoch

    require(selected_objective is not None, "No objective was selected")
    selected_report = next(
        item for item in objective_reports if item["objective"]["name"] == selected_objective.name
    )
    common_scale = float(selected_report["selected_scales"]["common_scale"])
    residual_scale = float(selected_report["selected_scales"]["residual_scale"])

    all_indices = np.arange(len(target_names), dtype=np.int64)
    full_representation = fit_representation(
        train_features,
        effects,
        counts,
        target_names,
        gene_names,
        all_indices,
        all_indices[:1],
        args.input_rank,
        args.output_rank,
        device,
        args.seed + 10_000,
    )
    full_model, _, full_training_loss = train_network(
        full_representation["train_x"],  # type: ignore[arg-type]
        full_representation["train_y"],  # type: ignore[arg-type]
        None,
        None,
        full_representation["score_scale"],  # type: ignore[arg-type]
        selected_objective,
        args.hidden_dim,
        max(selected_epoch, 1),
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.seed + 20_000,
    )

    normalized_challenge = challenge_features / np.maximum(
        np.linalg.norm(challenge_features, axis=1, keepdims=True), 1e-12
    )
    challenge_tensor = torch.from_numpy(normalized_challenge).to(device=device, dtype=torch.float32)
    challenge_x = (
        challenge_tensor - full_representation["input_mean"]  # type: ignore[operator]
    ) @ full_representation["input_components"].T  # type: ignore[union-attr]
    full_model.eval()
    with torch.inference_mode():
        challenge_scores = full_model(challenge_x)
    challenge_residual = decode_residual(
        challenge_scores,
        full_representation["score_scale"],  # type: ignore[arg-type]
        full_representation["output_center"],  # type: ignore[arg-type]
        full_representation["output_components"],  # type: ignore[arg-type]
    )
    common_effect = full_representation["common"]  # type: ignore[assignment]
    challenge_effects = (
        common_scale * common_effect[None, :] + residual_scale * challenge_residual
    ).astype(np.float32)
    require(np.isfinite(challenge_effects).all(), "Challenge effects are non-finite")

    for output in (args.output_npz, args.output_checkpoint, args.output_json):
        output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        genes=np.asarray(output_gene_names, dtype="U"),
        targets=np.asarray(challenge_targets, dtype="U"),
        common_log_fold=np.asarray(common_effect[output_gene_mask], dtype=np.float32),
        learned_log_fold=challenge_residual[:, output_gene_mask].astype(np.float32),
        effects=challenge_effects[:, output_gene_mask],
        target_names=np.asarray(challenge_targets, dtype="U"),
        gene_names=np.asarray(output_gene_names, dtype="U"),
        common_effect=np.asarray(common_effect[output_gene_mask], dtype=np.float32),
        residual_effects=challenge_residual[:, output_gene_mask].astype(np.float32),
        common_scale=np.asarray([common_scale], dtype=np.float32),
        residual_scale=np.asarray([residual_scale], dtype=np.float32),
    )
    checkpoint = {
        "schema": SCHEMA,
        "model_state_dict": {key: value.cpu() for key, value in full_model.state_dict().items()},
        "input_mean": full_representation["input_mean"].cpu(),  # type: ignore[union-attr]
        "input_components": full_representation["input_components"].cpu(),  # type: ignore[union-attr]
        "output_center": full_representation["output_center"].cpu(),  # type: ignore[union-attr]
        "output_components": full_representation["output_components"].cpu(),  # type: ignore[union-attr]
        "score_scale": full_representation["score_scale"].cpu(),  # type: ignore[union-attr]
        "common_effect": torch.from_numpy(np.asarray(common_effect, dtype=np.float32)),
        "architecture": {
            "input_dim": int(full_representation["train_x"].shape[1]),  # type: ignore[union-attr]
            "hidden_dim": args.hidden_dim,
            "output_dim": int(full_representation["train_y"].shape[1]),  # type: ignore[union-attr]
        },
        "objective": asdict(selected_objective),
        "common_scale": common_scale,
        "residual_scale": residual_scale,
        "gene_names": gene_names,
    }
    torch.save(checkpoint, args.output_checkpoint)

    report = {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "inputs": {
            "effect_atlas": {
                "path": str(args.effect_atlas.resolve()),
                "sha256": sha256_file(args.effect_atlas),
            },
            "esm2_embeddings": {
                "path": str(args.esm2_embeddings.resolve()),
                "sha256": sha256_file(args.esm2_embeddings),
            },
            "challenge_targets": {
                "path": str(args.challenge_targets.resolve()),
                "sha256": sha256_file(args.challenge_targets),
            },
        },
        "outputs": {
            "effect_prior": {
                "path": str(args.output_npz.resolve()),
                "sha256": sha256_file(args.output_npz),
            },
            "checkpoint": {
                "path": str(args.output_checkpoint.resolve()),
                "sha256": sha256_file(args.output_checkpoint),
            },
        },
        "data": {
            "training_targets": len(target_names),
            "challenge_targets": len(challenge_targets),
            "genes": len(gene_names),
            "output_genes_on_challenge_axis": len(output_gene_names),
            "direct_training_challenge_target_overlap": len(
                set(target_names) & set(challenge_targets)
            ),
            "train_split_targets": len(train_indices),
            "validation_split_targets": len(validation_indices),
            "cluster_count": int(len(np.unique(cluster_labels))),
            "validation_clusters": sorted(
                np.unique(cluster_labels[validation_indices]).astype(int).tolist()
            ),
        },
        "selection": {
            "selected_objective": asdict(selected_objective),
            "selected_epoch": selected_epoch,
            "common_scale": common_scale,
            "residual_scale": residual_scale,
            "cross_validation_proxy": best_proxy,
            "full_training_loss": full_training_loss,
            "all_objectives": objective_reports,
        },
        "configuration": vars(args),
        "runtime": {
            "argv": sys.argv,
            "python": platform.python_version(),
            "host": socket.gethostname(),
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "torch": torch.__version__,
        },
    }
    report["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in report["configuration"].items()
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
