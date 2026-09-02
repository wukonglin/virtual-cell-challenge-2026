#!/usr/bin/env python3
"""Generate VCC 2026 raw counts from paired per-cell STATE residuals.

For every context/target group, this adapter samples real non-targeting cells,
runs STATE twice on identical cell chunks (target and non-targeting embeddings),
and transfers only the paired prediction difference to the source raw-count
composition.  Current-only challenge genes are copied exactly, and adjusted
shared-gene expectations are integerized to their original total, so every
output cell has the same library size as its paired source control.

The implementation writes small H5AD shards and concatenates them on disk.  It
never materializes the full 360,000 by 18,533 prediction matrix in memory and
never overwrites an existing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import random
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from anndata.experimental import concat_on_disk

# Reuse the checkpoint/data-contract helpers used by the effect-prior adapter.
# Importing by module name works when this file is executed as
# ``python scripts/generate_state_direct_counts.py``.
from infer_state_effect_prior import (  # noqa: E402
    CONTEXTS,
    CONTROL_LABEL,
    EXPECTED_CURRENT_GENES,
    EXPECTED_CURRENT_ONLY_GENES,
    EXPECTED_OVERLAP_GENES,
    EXPECTED_PERT_DIM,
    EXPECTED_SUPPORT_GENES,
    EXPECTED_TARGETS,
    describe_file,
    git_commit,
    load_perturbation_embeddings,
    load_state_model,
    load_var_dims,
    parse_expected_sha256,
    read_single_column,
    read_targets,
    require,
    resolve_checkpoint,
    sha256_file,
    stratified_control_indices,
)


SAFE_INT32_NNZ = 2_140_000_000
MAX_OFFICIAL_CELL_LIBRARY = 1_000_000
DEFAULT_MAX_GENES_PER_CELL = 5_900
SCHEMA = "vcc-state-direct-counts-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired-residual STATE raw-count predictions for the "
            "VCC 2026 validation contexts."
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="STATE run directory containing checkpoints and embedding metadata.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("selected.ckpt"),
        help="Validation-selected checkpoint, resolved below MODEL_DIR/checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-expected-sha256",
        type=parse_expected_sha256,
        required=True,
        help="Required pre-registered SHA-256 for the selected STATE checkpoint.",
    )
    parser.add_argument(
        "--perturbation-map-expected-sha256",
        type=parse_expected_sha256,
        required=True,
        help="Required pre-registered SHA-256 for pert_onehot_map.pt.",
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help=(
            "Checkpoint-selection manifest. Defaults to "
            "MODEL_DIR/checkpoints/selected_checkpoint.json."
        ),
    )
    parser.add_argument("--controls-dir", type=Path, default=Path("dataset/controls"))
    parser.add_argument(
        "--support-genes",
        type=Path,
        default=Path("dataset/state_support/extracted/gene_names.csv"),
        help="Headerless STATE support-axis gene list.",
    )
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=Path("artifacts/state_direct_counts_v0.h5ad"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/state_direct_counts_v0.json"),
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Parent directory for temporary H5AD blocks; defaults to system TMPDIR.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-h100", action="store_true")
    parser.add_argument(
        "--model-chunk-size",
        type=int,
        default=128,
        help="Maximum number of paired cells in a variable-length STATE set.",
    )
    parser.add_argument("--cells-per-group", type=int, default=400)
    parser.add_argument(
        "--target-limit",
        type=int,
        default=0,
        help="Non-production smoke mode; zero means all 300 official targets.",
    )
    parser.add_argument(
        "--effect-scale",
        type=float,
        default=1.0,
        help="Scale paired target-minus-control STATE deltas before bounding.",
    )
    parser.add_argument(
        "--effect-clip",
        type=float,
        default=0.60,
        help="Symmetric bound for scaled cellwise log-expression deltas.",
    )
    parser.add_argument(
        "--effect-bounding",
        choices=("tanh", "clip"),
        default="tanh",
        help=(
            "Smoothly bound deltas as bound*tanh(delta/bound), or use a hard clip. "
            "The smooth production default preserves effect ordering without a point "
            "mass at the bound."
        ),
    )
    parser.add_argument(
        "--target-strategy",
        choices=("model", "force", "blend"),
        default="force",
        help="Use the modeled target delta, force CRISPRi knockdown, or blend both.",
    )
    parser.add_argument(
        "--target-remaining-fraction",
        type=float,
        default=0.20,
        help="Forced target fold; 0.20 represents 80 percent knockdown.",
    )
    parser.add_argument(
        "--target-blend-weight",
        type=float,
        default=0.50,
        help="Weight on the forced log-fold when target-strategy is blend.",
    )
    parser.add_argument(
        "--max-genes-per-cell",
        type=int,
        default=DEFAULT_MAX_GENES_PER_CELL,
        help=(
            "Safe output row-nnz cap. Current-only genes are never removed; low-count "
            "shared genes beyond the cap donate their counts to retained shared genes."
        ),
    )
    parser.add_argument(
        "--integerization",
        choices=("largest-remainder", "multinomial"),
        default="largest-remainder",
        help=(
            "Convert adjusted shared-gene expectations to exact integers. The "
            "deterministic largest-remainder default avoids resampling noise; "
            "multinomial is available for sensitivity analyses."
        ),
    )
    parser.add_argument(
        "--groups-per-shard",
        type=int,
        default=20,
        help="Temporary target blocks concatenated into each on-disk shard.",
    )
    parser.add_argument(
        "--concat-max-loaded-elements",
        type=int,
        default=20_000_000,
        help="Memory bound passed to anndata.experimental.concat_on_disk.",
    )
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def deterministic_seed(base: int, *tokens: str) -> int:
    payload = "|".join((str(base), *tokens)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def summarize(values: np.ndarray | Iterable[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    require(array.size > 0 and np.isfinite(array).all(), "Cannot summarize invalid values")
    quantiles = np.quantile(array, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    return {
        "min": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p10": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": float(quantiles[6]),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


@dataclass
class StreamingMoments:
    """Constant-memory moments for large dense STATE delta tensors."""

    count: int = 0
    total: float = 0.0
    total_square: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        require(array.size > 0 and np.isfinite(array).all(), "Invalid streaming values")
        self.count += int(array.size)
        self.total += float(array.sum(dtype=np.float64))
        self.total_square += float(np.square(array).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    def merge(self, other: "StreamingMoments") -> None:
        if other.count == 0:
            return
        self.count += other.count
        self.total += other.total
        self.total_square += other.total_square
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)

    def report(self) -> dict[str, float | int]:
        require(self.count > 0, "No values were accumulated")
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": math.sqrt(variance),
        }


def validate_args(args: argparse.Namespace) -> None:
    require(args.model_chunk_size > 0, "model-chunk-size must be positive")
    require(args.model_chunk_size <= 128, "STATE chunks must contain at most 128 cells")
    require(args.cells_per_group > 0, "cells-per-group must be positive")
    require(args.effect_scale > 0 and np.isfinite(args.effect_scale), "Invalid effect-scale")
    require(args.effect_clip > 0 and np.isfinite(args.effect_clip), "Invalid effect-clip")
    require(
        0 < args.target_remaining_fraction < 1,
        "target-remaining-fraction must be in (0, 1)",
    )
    require(0 <= args.target_blend_weight <= 1, "target-blend-weight must be in [0, 1]")
    require(args.max_genes_per_cell > 0, "max-genes-per-cell must be positive")
    require(args.max_genes_per_cell <= EXPECTED_CURRENT_GENES, "Invalid row-nnz cap")
    require(args.groups_per_shard > 0, "groups-per-shard must be positive")
    require(args.concat_max_loaded_elements > 0, "Invalid concat memory bound")
    require(args.target_limit >= 0, "target-limit cannot be negative")
    require(args.output_h5ad.suffix == ".h5ad", "output-h5ad must end in .h5ad")
    require(args.output_json.suffix == ".json", "output-json must end in .json")
    require(args.output_h5ad.resolve() != args.output_json.resolve(), "Output paths differ")
    for output in (args.output_h5ad, args.output_json):
        require(not output.exists(), f"Refusing to overwrite existing artifact: {output}")


def validate_model(model: Any, model_chunk_size: int) -> dict[str, Any]:
    input_dim = int(getattr(model, "input_dim", -1))
    output_dim = int(getattr(model, "output_dim", -1))
    pert_dim = int(getattr(model, "pert_dim", -1))
    sentence_len = int(getattr(model, "cell_sentence_len", -1))
    require(input_dim == EXPECTED_SUPPORT_GENES, f"Checkpoint input_dim is {input_dim}")
    require(output_dim == EXPECTED_SUPPORT_GENES, f"Checkpoint output_dim is {output_dim}")
    require(pert_dim == EXPECTED_PERT_DIM, f"Checkpoint pert_dim is {pert_dim}")
    require(sentence_len == 128, f"Checkpoint cell_sentence_len is {sentence_len}")
    require(model_chunk_size <= sentence_len, "Model chunk exceeds checkpoint sentence length")
    require(getattr(model, "batch_encoder", None) is None, "batch_encoder must be disabled")
    require(str(getattr(model, "output_space", "")) == "all", "output_space must be all")
    require(
        not bool(getattr(model, "log1p_from_raw_counts", False)),
        "Adapter supplies log1p values, but checkpoint also performs log1p",
    )
    return {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "input_dim": input_dim,
        "output_dim": output_dim,
        "pert_dim": pert_dim,
        "cell_sentence_len": sentence_len,
        "output_space": str(getattr(model, "output_space", "")),
        "predict_residual": bool(getattr(model, "predict_residual", False)),
        "log1p_from_raw_counts": bool(getattr(model, "log1p_from_raw_counts", False)),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def validate_selection_manifest(
    path: Path, checkpoint: Path, checkpoint_sha256: str
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema") == "vcc-state-checkpoint-selection-v1", "Bad selection schema")
    selected = payload.get("selected", {})
    selected_checkpoint = Path(str(selected.get("checkpoint", ""))).resolve()
    require(selected_checkpoint == checkpoint.resolve(), "Checkpoint differs from selection manifest")
    require(
        int(selected.get("checkpoint_size_bytes", -1)) == checkpoint.stat().st_size,
        "Selected checkpoint size mismatch",
    )
    require(
        str(selected.get("checkpoint_sha256", "")) == checkpoint_sha256,
        "Selected checkpoint SHA-256 mismatch",
    )
    require(int(selected.get("global_step", 0)) > 0, "Selection manifest lacks a step")
    require(np.isfinite(float(selected.get("val_loss", np.nan))), "Selection lacks val_loss")
    return payload


def read_control_rows(adata: ad.AnnData, indices: np.ndarray) -> sp.csr_matrix:
    """Read backed rows while preserving the requested shuffled order."""
    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order]
    inverse = np.argsort(order, kind="stable")
    matrix = adata.X[sorted_indices]
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(np.asarray(matrix))
    matrix = matrix.tocsr()[inverse]
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    require(np.isfinite(matrix.data).all(), "Control rows contain non-finite values")
    require(np.all(matrix.data >= 0), "Control rows contain negative values")
    require(
        np.allclose(matrix.data, np.rint(matrix.data), rtol=0.0, atol=1e-6),
        "Control rows are not integer-like raw counts",
    )
    matrix.data = np.rint(matrix.data).astype(np.int32)
    libraries = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)
    require(np.all(libraries > 0), "A sampled control cell has an empty library")
    require(
        np.all(libraries <= MAX_OFFICIAL_CELL_LIBRARY),
        "A sampled control cell exceeds the official one-million-count cap",
    )
    return matrix


def make_model_input(
    raw_current: sp.csr_matrix,
    support_to_current: np.ndarray,
) -> np.ndarray:
    valid_support = support_to_current >= 0
    selected = raw_current[:, support_to_current[valid_support]]
    if sp.issparse(selected):
        selected = selected.toarray()
    model_input = np.zeros(
        (raw_current.shape[0], EXPECTED_SUPPORT_GENES), dtype=np.float32
    )
    model_input[:, valid_support] = np.asarray(selected, dtype=np.float32)
    np.log1p(model_input, out=model_input)
    require(np.isfinite(model_input).all(), "Raw-log1p transformation is non-finite")
    return model_input


def state_paired_delta(
    model: Any,
    model_input: np.ndarray,
    target_embedding: torch.Tensor,
    control_embedding: torch.Tensor,
    target: str,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    """Subtract paired model-control predictions without converting either to counts."""
    basal = torch.from_numpy(model_input).to(device=device, dtype=torch.float32)

    def predict(embedding: torch.Tensor, name: str) -> torch.Tensor:
        pert = embedding.unsqueeze(0).expand(len(model_input), -1)
        batch = {
            "ctrl_cell_emb": basal,
            "pert_emb": pert,
            "pert_name": [name] * len(model_input),
        }
        output = model.predict_step(batch, 0, padded=False)
        require(isinstance(output, dict) and "preds" in output, "STATE output lacks preds")
        prediction = output["preds"].float()
        require(
            tuple(prediction.shape) == (len(model_input), EXPECTED_SUPPORT_GENES),
            f"Unexpected STATE shape for {name}: {tuple(prediction.shape)}",
        )
        require(bool(torch.isfinite(prediction).all()), f"Non-finite STATE output for {name}")
        return prediction

    # Both passes receive the exact same basal tensor and variable-length cell set.
    control_prediction = predict(control_embedding, CONTROL_LABEL)
    target_prediction = predict(target_embedding, target)
    delta = target_prediction - control_prediction
    require(bool(torch.isfinite(delta).all()), f"Non-finite paired delta for {target}")
    qc = {
        "target_prediction_min": float(target_prediction.amin().item()),
        "target_prediction_max": float(target_prediction.amax().item()),
        "control_prediction_min": float(control_prediction.amin().item()),
        "control_prediction_max": float(control_prediction.amax().item()),
        "paired_delta_abs_max": float(delta.abs().amax().item()),
    }
    result = delta.cpu().numpy().astype(np.float32, copy=False)
    del basal, control_prediction, target_prediction, delta
    return result, qc


def apply_delta_policy(
    raw_delta: np.ndarray,
    target_support_index: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    pre_bound = raw_delta.astype(np.float32, copy=True)
    pre_bound *= np.float32(args.effect_scale)
    below_bound = int(np.count_nonzero(pre_bound < -args.effect_clip))
    above_bound = int(np.count_nonzero(pre_bound > args.effect_clip))
    if args.effect_bounding == "tanh":
        bounded = np.tanh(pre_bound / np.float32(args.effect_clip)).astype(
            np.float32, copy=False
        )
        bounded *= np.float32(args.effect_clip)
    elif args.effect_bounding == "clip":
        bounded = np.clip(
            pre_bound, -args.effect_clip, args.effect_clip
        ).astype(np.float32, copy=False)
    else:
        raise RuntimeError(f"Unsupported effect-bounding mode: {args.effect_bounding}")
    modeled_target = bounded[:, target_support_index].copy()
    near_saturation = int(
        np.count_nonzero(np.abs(bounded) >= np.float32(0.95 * args.effect_clip))
    )
    exactly_at_bound = int(
        np.count_nonzero(np.abs(bounded) == np.float32(args.effect_clip))
    )
    forced = float(math.log(args.target_remaining_fraction))
    if args.target_strategy == "force":
        bounded[:, target_support_index] = np.float32(forced)
    elif args.target_strategy == "blend":
        weight = np.float32(args.target_blend_weight)
        bounded[:, target_support_index] = (
            (np.float32(1.0) - weight) * modeled_target + weight * np.float32(forced)
        )
    final_target = bounded[:, target_support_index].copy()
    return bounded, {
        "mode": args.effect_bounding,
        "bound": float(args.effect_clip),
        "below_bound_before_bounding": below_bound,
        "above_bound_before_bounding": above_bound,
        "values": int(bounded.size),
        "outside_bound_fraction_before_bounding": float(
            (below_bound + above_bound) / bounded.size
        ),
        "near_bound_fraction_after_bounding": float(near_saturation / bounded.size),
        "exactly_at_bound_after_bounding": exactly_at_bound,
        "modeled_target_delta": summarize(modeled_target),
        "final_target_delta": summarize(final_target),
        "final_target_fold": summarize(np.exp(final_target)),
    }


def transform_raw_row(
    base_indices: np.ndarray,
    base_values: np.ndarray,
    support_delta: np.ndarray,
    current_to_support: np.ndarray,
    max_genes_per_cell: int,
    integerization: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | bool]]:
    """Transfer a log-fold vector and retain an exact integer library size.

    Challenge-only genes are copied exactly. Shared genes are weighted by the
    paired STATE fold and integerized to the source cell's original shared-gene
    count. If an nnz cap is needed, only low-count shared genes are
    removed; their molecules remain in the shared-gene integerization total.
    """
    indices = np.asarray(base_indices, dtype=np.int32)
    values = np.asarray(base_values, dtype=np.int64)
    require(len(indices) == len(values), "Malformed source CSR row")
    require(np.all(values > 0), "Source CSR row contains non-positive values")
    support_positions = current_to_support[indices]
    is_shared = support_positions >= 0
    current_only_indices = indices[~is_shared]
    current_only_values = values[~is_shared]
    shared_indices = indices[is_shared]
    shared_values = values[is_shared]
    shared_support = support_positions[is_shared]
    source_library = int(values.sum(dtype=np.int64))
    current_only_total = int(current_only_values.sum(dtype=np.int64))
    shared_total = source_library - current_only_total
    require(shared_total > 0 and len(shared_values) > 0, "Cell has no shared-gene counts")

    available_shared_nnz = max_genes_per_cell - len(current_only_indices)
    require(available_shared_nnz > 0, "Current-only genes exhaust the row-nnz cap")
    removed_shared_nnz = 0
    removed_shared_counts = 0
    if len(shared_indices) > available_shared_nnz:
        # Highest count first; current-axis index deterministically breaks ties.
        order = np.lexsort((shared_indices, -shared_values))
        keep = order[:available_shared_nnz]
        removed_shared_nnz = int(len(shared_indices) - len(keep))
        removed_shared_counts = int(shared_total - shared_values[keep].sum(dtype=np.int64))
        shared_indices = shared_indices[keep]
        shared_values = shared_values[keep]
        shared_support = shared_support[keep]

    folds = np.exp(support_delta[shared_support].astype(np.float64, copy=False))
    require(np.isfinite(folds).all() and np.all(folds > 0), "Invalid STATE fold")
    weights = shared_values.astype(np.float64) * folds
    weight_total = float(weights.sum(dtype=np.float64))
    require(np.isfinite(weight_total) and weight_total > 0, "Invalid adjusted composition")
    probabilities = weights / weight_total
    expected_shared = probabilities * shared_total
    if integerization == "largest-remainder":
        sampled_shared = np.floor(expected_shared).astype(np.int64)
        remainder = int(shared_total - sampled_shared.sum(dtype=np.int64))
        require(0 <= remainder < len(sampled_shared), "Invalid largest-remainder total")
        if remainder:
            fractions = expected_shared - sampled_shared
            # Descending fractional part, with current-axis gene index as a
            # deterministic tie breaker.
            recipients = np.lexsort((shared_indices, -fractions))[:remainder]
            sampled_shared[recipients] += 1
    elif integerization == "multinomial":
        sampled_shared = rng.multinomial(shared_total, probabilities).astype(np.int64)
    else:
        raise RuntimeError(f"Unsupported integerization method: {integerization}")
    positive = sampled_shared > 0
    output_indices = np.concatenate((current_only_indices, shared_indices[positive]))
    output_values = np.concatenate((current_only_values, sampled_shared[positive]))
    order = np.argsort(output_indices, kind="stable")
    output_indices = output_indices[order].astype(np.int32, copy=False)
    output_values = output_values[order]
    require(len(output_indices) <= max_genes_per_cell, "Output row exceeds nnz cap")
    require(int(output_values.sum(dtype=np.int64)) == source_library, "Library size changed")
    require(np.all(output_values > 0), "Output row contains non-positive values")
    require(output_values.max(initial=0) <= np.iinfo(np.int32).max, "Count exceeds int32")

    unchanged = bool(
        np.array_equal(indices, output_indices)
        and np.array_equal(values, output_values)
    )
    return output_indices, output_values.astype(np.int32), {
        "source_library": source_library,
        "output_library": int(output_values.sum(dtype=np.int64)),
        "source_nnz": int(len(indices)),
        "output_nnz": int(len(output_indices)),
        "current_only_nnz": int(len(current_only_indices)),
        "current_only_counts": current_only_total,
        "unnormalized_shared_expectation_total": weight_total,
        "composition_normalization_factor": float(shared_total / weight_total),
        "removed_shared_nnz": removed_shared_nnz,
        "removed_shared_counts": removed_shared_counts,
        "changed": not unchanged,
    }


def build_group_block(
    raw: sp.csr_matrix,
    model: Any,
    target: str,
    target_embedding: torch.Tensor,
    control_embedding: torch.Tensor,
    target_current_index: int,
    target_support_index: int,
    support_to_current: np.ndarray,
    current_to_support: np.ndarray,
    device: torch.device,
    count_rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    indptr = np.zeros(raw.shape[0] + 1, dtype=np.int64)
    raw_delta_moments = StreamingMoments()
    final_delta_moments = StreamingMoments()
    prediction_ranges: list[dict[str, float]] = []
    outside_low = 0
    outside_high = 0
    bounding_values = 0
    near_bound_fractions: list[float] = []
    exact_at_bound = 0
    pre_bound_moments = StreamingMoments()
    target_raw_deltas: list[np.ndarray] = []
    target_model_deltas: list[np.ndarray] = []
    target_final_deltas: list[np.ndarray] = []
    input_libraries: list[int] = []
    output_libraries: list[int] = []
    input_nnzs: list[int] = []
    output_nnzs: list[int] = []
    removed_counts: list[int] = []
    removed_nnzs: list[int] = []
    changed_cells = 0
    current_only_mismatches = 0
    library_mismatches = 0
    target_before = 0
    target_after = 0
    chunk_sizes: list[int] = []

    for start in range(0, raw.shape[0], args.model_chunk_size):
        stop = min(start + args.model_chunk_size, raw.shape[0])
        chunk = raw[start:stop]
        chunk_sizes.append(stop - start)
        model_input = make_model_input(chunk, support_to_current)
        with torch.inference_mode():
            raw_delta, range_qc = state_paired_delta(
                model,
                model_input,
                target_embedding,
                control_embedding,
                target,
                device,
            )
        prediction_ranges.append(range_qc)
        raw_delta_moments.update(raw_delta)
        final_delta, policy_qc = apply_delta_policy(raw_delta, target_support_index, args)
        final_delta_moments.update(final_delta)
        outside_low += int(policy_qc["below_bound_before_bounding"])
        outside_high += int(policy_qc["above_bound_before_bounding"])
        bounding_values += int(policy_qc["values"])
        near_bound_fractions.append(float(policy_qc["near_bound_fraction_after_bounding"]))
        exact_at_bound += int(policy_qc["exactly_at_bound_after_bounding"])
        pre_bound_moments.update(raw_delta * np.float32(args.effect_scale))
        raw_target = raw_delta[:, target_support_index].copy()
        target_raw_deltas.append(raw_target)
        scaled_target = raw_target * np.float32(args.effect_scale)
        if args.effect_bounding == "tanh":
            bounded_target = np.float32(args.effect_clip) * np.tanh(
                scaled_target / np.float32(args.effect_clip)
            )
        else:
            bounded_target = np.clip(
                scaled_target, -args.effect_clip, args.effect_clip
            )
        target_model_deltas.append(bounded_target.astype(np.float32, copy=False))
        target_final_deltas.append(final_delta[:, target_support_index].copy())

        for local_row in range(stop - start):
            row_start, row_stop = chunk.indptr[local_row : local_row + 2]
            base_indices = chunk.indices[row_start:row_stop]
            base_values = chunk.data[row_start:row_stop]
            output_indices, output_values, row_qc = transform_raw_row(
                base_indices,
                base_values,
                final_delta[local_row],
                current_to_support,
                args.max_genes_per_cell,
                args.integerization,
                count_rng,
            )
            data_parts.append(output_values)
            index_parts.append(output_indices)
            global_row = start + local_row
            indptr[global_row + 1] = indptr[global_row] + len(output_values)
            input_libraries.append(int(row_qc["source_library"]))
            output_libraries.append(int(row_qc["output_library"]))
            input_nnzs.append(int(row_qc["source_nnz"]))
            output_nnzs.append(int(row_qc["output_nnz"]))
            removed_counts.append(int(row_qc["removed_shared_counts"]))
            removed_nnzs.append(int(row_qc["removed_shared_nnz"]))
            changed_cells += int(bool(row_qc["changed"]))
            library_mismatches += int(row_qc["source_library"] != row_qc["output_library"])

            base_target_position = np.searchsorted(base_indices, target_current_index)
            if (
                base_target_position < len(base_indices)
                and int(base_indices[base_target_position]) == target_current_index
            ):
                target_before += int(base_values[base_target_position])
            output_target_position = np.searchsorted(output_indices, target_current_index)
            if (
                output_target_position < len(output_indices)
                and int(output_indices[output_target_position]) == target_current_index
            ):
                target_after += int(output_values[output_target_position])

            # Exact preservation is checked directly on the challenge-only subset.
            base_only = current_to_support[base_indices] < 0
            output_only = current_to_support[output_indices] < 0
            if not (
                np.array_equal(base_indices[base_only], output_indices[output_only])
                and np.array_equal(base_values[base_only], output_values[output_only])
            ):
                current_only_mismatches += 1

        del model_input, raw_delta, final_delta

    require(indptr[-1] < np.iinfo(np.int32).max, "One group exceeds int32 CSR capacity")
    matrix = sp.csr_matrix(
        (
            np.concatenate(data_parts).astype(np.int32, copy=False),
            np.concatenate(index_parts).astype(np.int32, copy=False),
            indptr.astype(np.int32),
        ),
        shape=(raw.shape[0], EXPECTED_CURRENT_GENES),
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    require(matrix.has_canonical_format, "Generated group CSR is not canonical")
    require(library_mismatches == 0, "A generated cell changed library size")
    require(current_only_mismatches == 0, "A current-only gene count changed")
    require(np.all(matrix.data > 0), "Generated group has invalid sparse values")

    target_raw = np.concatenate(target_raw_deltas)
    target_model = np.concatenate(target_model_deltas)
    target_final = np.concatenate(target_final_deltas)
    target_remaining = float(target_after / max(target_before, 1))
    qc = {
        "cells": raw.shape[0],
        "chunk_sizes": chunk_sizes,
        "paired_passes": 2 * len(chunk_sizes),
        "input_normalization": "elementwise-log1p-raw-on-support-axis",
        "delta_definition": "state(target, basal)-state(non-targeting, identical basal)",
        "raw_paired_delta": raw_delta_moments.report(),
        "final_paired_delta": final_delta_moments.report(),
        "effect_bounding": {
            "mode": args.effect_bounding,
            "bound": args.effect_clip,
            "pre_bound_delta": pre_bound_moments.report(),
            "below_bound_before_bounding": outside_low,
            "above_bound_before_bounding": outside_high,
            "values": bounding_values,
            "outside_bound_fraction_before_bounding": float(
                (outside_low + outside_high) / bounding_values
            ),
            "near_bound_fraction_after_bounding": float(
                np.average(near_bound_fractions, weights=chunk_sizes)
            ),
            "exactly_at_bound_after_bounding": exact_at_bound,
        },
        "raw_target_delta": summarize(target_raw),
        "bounded_model_target_delta": summarize(target_model),
        "final_target_delta": summarize(target_final),
        "final_target_fold": summarize(np.exp(target_final)),
        "prediction_ranges": prediction_ranges,
        "input_library": summarize(np.asarray(input_libraries)),
        "output_library": summarize(np.asarray(output_libraries)),
        "input_nnz": summarize(np.asarray(input_nnzs)),
        "output_nnz": summarize(np.asarray(output_nnzs)),
        "nnz": int(matrix.nnz),
        "changed_cell_fraction": float(changed_cells / raw.shape[0]),
        "library_size_mismatches": library_mismatches,
        "current_only_count_mismatches": current_only_mismatches,
        "integerization": args.integerization,
        "zero_entry_policy": "shared genes absent from the source cell remain zero",
        "cells_sparsity_capped": int(np.count_nonzero(np.asarray(removed_nnzs) > 0)),
        "removed_shared_nnz": int(np.sum(removed_nnzs, dtype=np.int64)),
        "redistributed_shared_counts": int(np.sum(removed_counts, dtype=np.int64)),
        "redistributed_count_fraction": float(
            np.sum(removed_counts, dtype=np.int64)
            / max(np.sum(input_libraries, dtype=np.int64), 1)
        ),
        "target_sum_before": target_before,
        "target_sum_after": target_after,
        "target_remaining_fraction": target_remaining,
    }
    return matrix, qc


def write_group_h5ad(
    path: Path,
    matrix: sp.csr_matrix,
    genes: list[str],
    context: str,
    target: str,
    selected_rows: np.ndarray,
    method: str,
    seed: int,
    compression: str,
) -> None:
    obs = pd.DataFrame(
        {
            "context": context,
            "target_gene": target,
            "source_control_row": selected_rows.astype(np.int32),
        },
        index=pd.Index(
            [f"{context}|{target}|{cell:03d}" for cell in range(len(selected_rows))],
            name="cell_id",
        ),
    )
    var = pd.DataFrame(index=pd.Index(genes, name="gene_name"))
    block = ad.AnnData(X=matrix, obs=obs, var=var)
    block.uns["method"] = method
    block.uns["seed"] = seed
    block.uns["schema"] = SCHEMA
    block.write_h5ad(path, compression=None if compression == "none" else compression)


def concatenate_paths(
    paths: list[Path], output: Path, max_loaded_elements: int
) -> None:
    require(paths, "No H5AD blocks to concatenate")
    require(not output.exists(), f"Temporary output already exists: {output}")
    concat_on_disk(
        paths,
        output,
        axis=0,
        join="inner",
        merge="same",
        uns_merge="same",
        max_loaded_elems=max_loaded_elements,
    )


def validate_backed_candidate(
    path: Path,
    genes: list[str],
    targets: list[str],
    cells_per_group: int,
    expected_nnz: int,
) -> dict[str, Any]:
    prediction = ad.read_h5ad(path, backed="r")
    try:
        expected_shape = (len(CONTEXTS) * len(targets) * cells_per_group, len(genes))
        group_sizes = prediction.obs.groupby(
            ["context", "target_gene"], observed=True, sort=True
        ).size()
        expected_groups = pd.MultiIndex.from_product(
            [CONTEXTS, targets], names=["context", "target_gene"]
        )
        expected_group_keys = {
            (str(context), str(target)) for context, target in expected_groups
        }
        observed_group_keys = {
            (str(context), str(target)) for context, target in group_sizes.index
        }
        matrix = prediction.X
        observed_nnz = int(matrix.group["data"].shape[0])
        data_dtype = np.dtype(matrix.group["data"].dtype)
        indices_dtype = np.dtype(matrix.group["indices"].dtype)
        indptr_dtype = np.dtype(matrix.group["indptr"].dtype)
        checks = {
            "shape_exact": tuple(prediction.shape) == expected_shape,
            "gene_axis_exact": list(prediction.var_names.astype(str)) == genes,
            "obs_names_unique": bool(prediction.obs_names.is_unique),
            "contexts_exact": set(prediction.obs["context"].astype(str)) == set(CONTEXTS),
            "targets_exact": set(prediction.obs["target_gene"].astype(str)) == set(targets),
            # AnnData serializes string columns as categoricals and may reorder
            # category levels lexicographically.  Group membership is strict,
            # but categorical level order is not part of the VCC contract.
            "groups_exact": observed_group_keys == expected_group_keys
            and len(group_sizes) == len(expected_groups),
            "cells_per_group_exact": bool((group_sizes == cells_per_group).all()),
            "no_controls": CONTROL_LABEL
            not in set(prediction.obs["target_gene"].astype(str)),
            "csr_encoding": str(matrix.format) == "csr",
            "integer_count_dtype": bool(np.issubdtype(data_dtype, np.integer)),
            "signed_integer_indices": bool(np.issubdtype(indices_dtype, np.signedinteger)),
            "signed_integer_indptr": bool(np.issubdtype(indptr_dtype, np.signedinteger)),
            "nnz_matches_stream": observed_nnz == expected_nnz,
            "nnz_safe_int32": observed_nnz < SAFE_INT32_NNZ,
        }
    finally:
        prediction.file.close()
    failed = sorted(name for name, passed in checks.items() if not passed)
    require(not failed, f"Candidate validation failed: {failed}")
    return {
        "checks": checks,
        "failed_checks": failed,
        "shape": list(expected_shape),
        "groups": len(group_sizes),
        "nnz": observed_nnz,
        "mean_nnz_per_cell": float(observed_nnz / expected_shape[0]),
        "data_dtype": str(data_dtype),
        "indices_dtype": str(indices_dtype),
        "indptr_dtype": str(indptr_dtype),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    started = time.time()
    args = parse_args()
    validate_args(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    method = (
        "Validation-selected STATE paired target-minus-model-control residuals "
        "transferred to real raw control counts"
    )

    model_dir = args.model_dir.resolve()
    checkpoint = resolve_checkpoint(model_dir, args.checkpoint)
    selection_path = (
        args.selection_json
        if args.selection_json is not None
        else model_dir / "checkpoints" / "selected_checkpoint.json"
    ).resolve()
    pert_map_path = model_dir / "pert_onehot_map.pt"
    var_dims_path = model_dir / "var_dims.pkl"
    config_path = model_dir / "config.yaml"
    for path in (checkpoint, selection_path, pert_map_path, var_dims_path):
        require(path.is_file(), f"Required STATE input is missing: {path}")
    checkpoint_hash = sha256_file(checkpoint)
    require(
        checkpoint_hash == args.checkpoint_expected_sha256,
        "STATE checkpoint differs from --checkpoint-expected-sha256",
    )
    selection = validate_selection_manifest(selection_path, checkpoint, checkpoint_hash)

    controls_dir = args.controls_dir.resolve()
    current_gene_path = controls_dir / "gene_names.csv"
    target_path = controls_dir / "pert_counts.csv"
    control_paths = {context: controls_dir / f"context_{context}.h5ad" for context in CONTEXTS}
    for path in (current_gene_path, target_path, args.support_genes, *control_paths.values()):
        require(path.is_file(), f"Required data input is missing: {path}")

    genes = read_single_column(current_gene_path, "gene_name")
    support_genes = read_single_column(args.support_genes.resolve())
    official_targets = read_targets(target_path)
    require(len(genes) == EXPECTED_CURRENT_GENES, "Unexpected current gene count")
    require(len(support_genes) == EXPECTED_SUPPORT_GENES, "Unexpected support gene count")
    require(len(official_targets) == EXPECTED_TARGETS, "Unexpected target count")
    if args.target_limit:
        require(1 <= args.target_limit <= EXPECTED_TARGETS, "Invalid target-limit")
        targets = official_targets[: args.target_limit]
    else:
        targets = official_targets
    full_contract = len(targets) == EXPECTED_TARGETS and args.cells_per_group == 400
    theoretical_nnz = (
        len(CONTEXTS) * len(targets) * args.cells_per_group * args.max_genes_per_cell
    )
    require(
        theoretical_nnz < SAFE_INT32_NNZ,
        f"Theoretical nnz bound {theoretical_nnz:,} is not int32-safe",
    )

    current_gene_index = {gene: index for index, gene in enumerate(genes)}
    support_gene_index = {gene: index for index, gene in enumerate(support_genes)}
    support_to_current = np.asarray(
        [current_gene_index.get(gene, -1) for gene in support_genes], dtype=np.int64
    )
    current_to_support = np.full(len(genes), -1, dtype=np.int64)
    valid_support = support_to_current >= 0
    current_to_support[support_to_current[valid_support]] = np.flatnonzero(valid_support)
    require(int(np.count_nonzero(valid_support)) == EXPECTED_OVERLAP_GENES, "Bad gene overlap")
    require(
        int(np.count_nonzero(current_to_support < 0)) == EXPECTED_CURRENT_ONLY_GENES,
        "Bad current-only gene count",
    )
    require(all(target in current_gene_index for target in targets), "Target missing from challenge axis")
    require(all(target in support_gene_index for target in targets), "Target missing from support axis")

    var_dims = load_var_dims(var_dims_path, support_genes)
    embeddings, perturbation_map_load = load_perturbation_embeddings(
        pert_map_path,
        official_targets,
        expected_sha256=args.perturbation_map_expected_sha256,
        return_descriptor=True,
    )
    device = torch.device(args.device)
    if args.require_cuda:
        require(device.type == "cuda", "--require-cuda requires a CUDA device")
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        device_name = torch.cuda.get_device_name(device)
        if args.require_h100:
            require("H100" in device_name.upper(), f"Selected GPU is not an H100: {device_name}")
    else:
        require(not args.require_h100, "--require-h100 requires CUDA")
        device_name = "CPU"
    print(f"Loading validation-selected checkpoint: {checkpoint}", flush=True)
    model, checkpoint_load = load_state_model(
        checkpoint,
        device,
        expected_sha256=args.checkpoint_expected_sha256,
        expected_gene_names=support_genes,
        return_descriptor=True,
    )
    model_info = validate_model(model, args.model_chunk_size)
    print(
        f"Loaded {model_info['parameters']:,} parameters on {device_name}; "
        f"generating {len(CONTEXTS) * len(targets)} groups",
        flush=True,
    )
    control_embedding = embeddings[CONTROL_LABEL].to(device=device, dtype=torch.float32)

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    scratch_parent = args.scratch_dir.resolve() if args.scratch_dir else None
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    final_temporary = args.output_h5ad.with_name(
        f".{args.output_h5ad.name}.{os.getpid()}.tmp.h5ad"
    )
    require(not final_temporary.exists(), f"Temporary output exists: {final_temporary}")

    group_qc: list[dict[str, Any]] = []
    context_qc: dict[str, Any] = {}
    final_shards: list[Path] = []
    total_nnz = 0
    global_delta = StreamingMoments()
    try:
        with tempfile.TemporaryDirectory(prefix="vcc_state_direct_", dir=scratch_parent) as scratch:
            scratch_path = Path(scratch)
            for context in CONTEXTS:
                control = ad.read_h5ad(control_paths[context], backed="r")
                try:
                    require(list(control.var_names.astype(str)) == genes, f"{context}: gene order mismatch")
                    required_obs = {"context", "target_gene", "ntc_id"}
                    require(required_obs.issubset(control.obs.columns), f"{context}: missing obs fields")
                    require(set(control.obs["context"].astype(str)) == {context}, f"{context}: wrong context")
                    require(
                        set(control.obs["target_gene"].astype(str)) == {CONTROL_LABEL},
                        f"{context}: non-control rows found",
                    )
                    pending_blocks: list[Path] = []
                    context_shards: list[Path] = []
                    context_nnz = 0
                    context_input_libraries: list[float] = []
                    context_output_libraries: list[float] = []
                    context_removed_counts = 0
                    for target_index, target in enumerate(targets):
                        sample_rng = np.random.default_rng(
                            deterministic_seed(args.seed, context, target, "sample")
                        )
                        count_rng = np.random.default_rng(
                            deterministic_seed(args.seed, context, target, "counts")
                        )
                        selected_rows, allocation = stratified_control_indices(
                            control.obs, args.cells_per_group, sample_rng
                        )
                        raw = read_control_rows(control, selected_rows)
                        target_embedding = embeddings[target].to(device=device, dtype=torch.float32)
                        block, qc = build_group_block(
                            raw,
                            model,
                            target,
                            target_embedding,
                            control_embedding,
                            current_gene_index[target],
                            support_gene_index[target],
                            support_to_current,
                            current_to_support,
                            device,
                            count_rng,
                            args,
                        )
                        qc.update(
                            {
                                "context": context,
                                "target_gene": target,
                                "target_ordinal": target_index,
                                "source_control_index_sha256": hashlib.sha256(
                                    selected_rows.tobytes()
                                ).hexdigest(),
                                "ntc_strata": len(allocation),
                                "ntc_allocation_min": min(allocation.values()),
                                "ntc_allocation_max": max(allocation.values()),
                                "sample_seed": deterministic_seed(
                                    args.seed, context, target, "sample"
                                ),
                                "count_seed": deterministic_seed(
                                    args.seed, context, target, "counts"
                                ),
                            }
                        )
                        group_qc.append(qc)
                        delta_summary = qc["final_paired_delta"]
                        synthetic_moments = StreamingMoments(
                            count=int(delta_summary["count"]),
                            total=float(delta_summary["mean"]) * int(delta_summary["count"]),
                            total_square=(
                                float(delta_summary["std"]) ** 2
                                + float(delta_summary["mean"]) ** 2
                            )
                            * int(delta_summary["count"]),
                            minimum=float(delta_summary["min"]),
                            maximum=float(delta_summary["max"]),
                        )
                        global_delta.merge(synthetic_moments)
                        context_input_libraries.append(float(qc["input_library"]["median"]))
                        context_output_libraries.append(float(qc["output_library"]["median"]))
                        context_removed_counts += int(qc["redistributed_shared_counts"])
                        context_nnz += int(block.nnz)
                        total_nnz += int(block.nnz)
                        require(total_nnz < SAFE_INT32_NNZ, "Streaming nnz exceeds safe limit")

                        block_path = scratch_path / f"{context}_{target_index:03d}_block.h5ad"
                        write_group_h5ad(
                            block_path,
                            block,
                            genes,
                            context,
                            target,
                            selected_rows,
                            method,
                            args.seed,
                            args.compression,
                        )
                        pending_blocks.append(block_path)
                        del raw, block, target_embedding

                        if (
                            len(pending_blocks) == args.groups_per_shard
                            or target_index + 1 == len(targets)
                        ):
                            shard_path = scratch_path / (
                                f"{context}_shard_{len(context_shards):03d}.h5ad"
                            )
                            concatenate_paths(
                                pending_blocks, shard_path, args.concat_max_loaded_elements
                            )
                            for block_path_to_remove in pending_blocks:
                                block_path_to_remove.unlink()
                            pending_blocks.clear()
                            context_shards.append(shard_path)

                        if (target_index + 1) % 10 == 0 or target_index + 1 == len(targets):
                            print(
                                f"[{context}] generated {target_index + 1}/{len(targets)} "
                                f"targets; cumulative nnz={total_nnz:,}",
                                flush=True,
                            )

                    context_qc[context] = {
                        "available_control_cells": int(control.n_obs),
                        "groups": len(targets),
                        "cells": len(targets) * args.cells_per_group,
                        "nnz": context_nnz,
                        "mean_nnz_per_cell": float(
                            context_nnz / (len(targets) * args.cells_per_group)
                        ),
                        "median_input_library_across_groups": summarize(
                            np.asarray(context_input_libraries)
                        ),
                        "median_output_library_across_groups": summarize(
                            np.asarray(context_output_libraries)
                        ),
                        "redistributed_shared_counts": context_removed_counts,
                    }
                    final_shards.extend(context_shards)
                finally:
                    control.file.close()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            concatenate_paths(final_shards, final_temporary, args.concat_max_loaded_elements)
            validation = validate_backed_candidate(
                final_temporary,
                genes,
                targets,
                args.cells_per_group,
                total_nnz,
            )
            target_failures = [
                f"{item['context']}|{item['target_gene']}"
                for item in group_qc
                if item["target_sum_before"] >= 20
                and item["target_remaining_fraction"] > 0.50
            ]
            library_failures = [
                f"{item['context']}|{item['target_gene']}"
                for item in group_qc
                if item["library_size_mismatches"] != 0
            ]
            current_only_failures = [
                f"{item['context']}|{item['target_gene']}"
                for item in group_qc
                if item["current_only_count_mismatches"] != 0
            ]
            scientific_qc = {
                "all_libraries_exact": not library_failures,
                "library_failures": library_failures,
                "all_current_only_counts_exact": not current_only_failures,
                "current_only_failures": current_only_failures,
                "target_knockdown_failures": target_failures,
                "target_knockdown_gate": {
                    "minimum_source_target_counts": 20,
                    "maximum_realized_remaining_fraction": 0.50,
                    "forced_model_space_fold": args.target_remaining_fraction,
                },
                "groups_with_target_baseline_at_least_20": int(
                    sum(item["target_sum_before"] >= 20 for item in group_qc)
                ),
                "changed_cell_fraction": summarize(
                    np.asarray([item["changed_cell_fraction"] for item in group_qc])
                ),
                "target_remaining_fraction": summarize(
                    np.asarray([item["target_remaining_fraction"] for item in group_qc])
                ),
                "delta_outside_bound_fraction_before_bounding": summarize(
                    np.asarray(
                        [
                            item["effect_bounding"][
                                "outside_bound_fraction_before_bounding"
                            ]
                            for item in group_qc
                        ]
                    )
                ),
                "delta_near_bound_fraction_after_bounding": summarize(
                    np.asarray(
                        [
                            item["effect_bounding"][
                                "near_bound_fraction_after_bounding"
                            ]
                            for item in group_qc
                        ]
                    )
                ),
                "redistributed_count_fraction": summarize(
                    np.asarray([item["redistributed_count_fraction"] for item in group_qc])
                ),
            }
            require(not library_failures, "One or more groups changed library sizes")
            require(not current_only_failures, "One or more groups changed current-only counts")
            if args.target_strategy == "force":
                require(not target_failures, "One or more groups failed target-knockdown QC")

            # The output remains temporary until every structural and scientific gate passes.
            os.replace(final_temporary, args.output_h5ad)

        elapsed = time.time() - started
        output_descriptor = describe_file(args.output_h5ad)
        report = {
            "schema": SCHEMA,
            "artifact_type": "vcc_2026_prediction_candidate",
            "created_utc_epoch": time.time(),
            "elapsed_seconds": elapsed,
            "command": [sys.executable, *sys.argv],
            "method": method,
            "full_official_contract": full_contract,
            "configuration": {
                "seed": args.seed,
                "device": str(device),
                "device_name": device_name,
                "precision": "float32",
                "input_normalization": "elementwise-log1p-raw-on-support-axis",
                "effect_reference": "paired-model-control-on-identical-cell-chunks",
                "model_chunk_size": args.model_chunk_size,
                "cells_per_group": args.cells_per_group,
                "targets": len(targets),
                "effect_scale": args.effect_scale,
                "effect_clip": args.effect_clip,
                "effect_bounding": args.effect_bounding,
                "target_strategy": args.target_strategy,
                "target_remaining_fraction": args.target_remaining_fraction,
                "target_blend_weight": args.target_blend_weight,
                "max_genes_per_cell": args.max_genes_per_cell,
                "integerization": args.integerization,
                "groups_per_shard": args.groups_per_shard,
                "concat_max_loaded_elements": args.concat_max_loaded_elements,
                "library_policy": "exact-source-library-per-cell",
                "current_only_policy": "copy-raw-counts-exactly",
                "zero_entry_policy": (
                    "reweight-observed-shared-genes-only; source zeros remain zero"
                ),
            },
            "model": model_info,
            "checkpoint_selection": {
                "global_step": int(selection["selected"]["global_step"]),
                "val_loss": float(selection["selected"]["val_loss"]),
                "rule": selection["selection_rule"],
            },
            "axes": {
                "contexts": list(CONTEXTS),
                "targets": len(targets),
                "official_targets": len(official_targets),
                "current_genes": len(genes),
                "support_genes": len(support_genes),
                "shared_genes": int(np.count_nonzero(valid_support)),
                "current_only_genes_preserved": int(np.count_nonzero(current_to_support < 0)),
                "support_only_genes_ignored": int(np.count_nonzero(~valid_support)),
                "perturbation_embedding_dimension": int(var_dims["pert_dim"]),
            },
            "validation": validation,
            "scientific_qc": scientific_qc,
            "qc": {
                "global_final_delta": global_delta.report(),
                "contexts": context_qc,
                "groups": group_qc,
            },
            "provenance": {
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "repository_commit": git_commit(Path(__file__).resolve().parents[1]),
                "state_source_commit": git_commit(Path(inspect.getfile(type(model))).resolve().parent),
                "script": describe_file(Path(__file__).resolve()),
                "checkpoint": checkpoint_load,
                "checkpoint_selection": describe_file(selection_path),
                "perturbation_map": perturbation_map_load,
                "var_dims": describe_file(var_dims_path),
                "state_config": describe_file(config_path) if config_path.is_file() else None,
                "current_gene_axis": describe_file(current_gene_path),
                "support_gene_axis": describe_file(args.support_genes.resolve()),
                "target_axis": describe_file(target_path),
                "controls": {
                    context: describe_file(path) for context, path in control_paths.items()
                },
                "output_h5ad": output_descriptor,
            },
        }
        atomic_write_json(args.output_json, report)
        print(
            f"Wrote {args.output_h5ad} ({args.output_h5ad.stat().st_size:,} bytes)",
            flush=True,
        )
        print(
            f"Wrote {args.output_json}; shape={validation['shape']}, "
            f"nnz={validation['nnz']:,}, elapsed={elapsed / 60:.1f} min",
            flush=True,
        )
    finally:
        if final_temporary.exists():
            final_temporary.unlink()


if __name__ == "__main__":
    main()
