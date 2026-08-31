#!/usr/bin/env python3
"""Infer a sparse VCC effect prior with a trained STATE checkpoint.

The script intentionally emits only the compact effect-prior contract consumed by
``generate_bayesian_submission.py``. It does not create a VCC submission and it
never overwrites an existing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import pickle
import platform
import random
import socket
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


CONTEXTS = ("A", "B", "C")
CONTROL_LABEL = "non-targeting"
EXPECTED_CURRENT_GENES = 18_533
EXPECTED_SUPPORT_GENES = 18_080
EXPECTED_OVERLAP_GENES = 18_077
EXPECTED_CURRENT_ONLY_GENES = 456
EXPECTED_TARGETS = 300
EXPECTED_PERT_DIM = 5_120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run STATE on VCC controls and write a sparse (3, 300, 18533) effect prior."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="STATE run directory containing pert_onehot_map.pt and var_dims.pkl.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("best.ckpt"),
        help="Checkpoint path, or a filename resolved below MODEL_DIR/checkpoints.",
    )
    parser.add_argument(
        "--controls-dir", type=Path, default=Path("dataset/controls")
    )
    parser.add_argument(
        "--input-normalization",
        choices=("log1p-raw", "cp10k-log1p"),
        default="log1p-raw",
        help=(
            "Transform raw challenge controls to STATE input space. The official "
            "support matrices are already log1p and are not uniformly CP10K-scaled; "
            "log1p-raw therefore matches the released training path."
        ),
    )
    parser.add_argument(
        "--support-genes",
        type=Path,
        default=Path("dataset/state_support/extracted/gene_names.csv"),
        help="Headerless STATE support-axis gene list.",
    )
    parser.add_argument(
        "--output-npz", type=Path, default=Path("artifacts/state_effect_prior_v0.npz")
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("artifacts/state_effect_prior_v0.json")
    )
    parser.add_argument("--device", default="cuda", help="Torch device, normally cuda.")
    parser.add_argument(
        "--require-cuda", action="store_true", help="Fail unless inference uses CUDA."
    )
    parser.add_argument(
        "--require-h100",
        action="store_true",
        help="Fail unless the selected CUDA device reports an H100 GPU.",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Autocast precision for model forward passes.",
    )
    parser.add_argument(
        "--set-size",
        type=int,
        default=128,
        help="Cells per STATE set; must match the checkpoint cell_sentence_len.",
    )
    parser.add_argument(
        "--sets-per-context",
        type=int,
        default=4,
        help="Independent full control sets used for every target and context.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=161,
        help="Maximum nonzero genes retained per target/context, including the target.",
    )
    parser.add_argument(
        "--min-abs-effect",
        type=float,
        default=0.01,
        help="Discard modeled effects below this absolute log-expression delta.",
    )
    parser.add_argument(
        "--effect-scale", type=float, default=1.0, help="Scale all modeled deltas."
    )
    parser.add_argument(
        "--effect-reference",
        choices=("model-control", "input-control"),
        default="model-control",
        help=(
            "Subtract STATE's non-targeting prediction to cancel reconstruction bias, "
            "or subtract the input basal mean directly."
        ),
    )
    parser.add_argument(
        "--effect-clip",
        type=float,
        default=0.60,
        help="Symmetric clip applied to modeled deltas before target handling.",
    )
    parser.add_argument(
        "--target-strategy",
        choices=("model", "force", "blend"),
        default="blend",
        help="Use STATE's target delta, force a CRISPRi delta, or blend the two.",
    )
    parser.add_argument(
        "--target-log-effect",
        type=float,
        default=float(np.log(0.20)),
        help="Forced target log effect; log(0.20) corresponds to 80%% knockdown.",
    )
    parser.add_argument(
        "--target-blend-weight",
        type=float,
        default=0.50,
        help="Weight on target-log-effect when --target-strategy=blend.",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_file:
        result["sha256"] = sha256_file(path)
    return result


def read_single_column(path: Path, preferred_column: str | None = None) -> list[str]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if preferred_column is not None and preferred_column in frame.columns:
        values = frame[preferred_column].tolist()
    elif frame.shape[1] == 1:
        values = frame.iloc[:, 0].tolist()
        # STATE's support gene file is headerless. Pandas interpreted its first gene
        # as a header, so recover that value when no canonical header was requested.
        if preferred_column is None:
            values.insert(0, str(frame.columns[0]))
    else:
        raise RuntimeError(f"Expected one column in {path}, found {frame.shape[1]}")
    values = [str(value) for value in values]
    require(values and all(values), f"Empty value found in {path}")
    require(len(values) == len(set(values)), f"Duplicate values found in {path}")
    return values


def read_targets(path: Path) -> list[str]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    require("target_gene" in frame.columns, f"Missing target_gene column in {path}")
    targets = frame["target_gene"].astype(str).tolist()
    require(len(targets) == EXPECTED_TARGETS, f"Expected 300 targets, found {len(targets)}")
    require(len(targets) == len(set(targets)), "Target list contains duplicates")
    return targets


def resolve_checkpoint(model_dir: Path, checkpoint_arg: Path) -> Path:
    candidates = []
    if checkpoint_arg.is_absolute():
        candidates.append(checkpoint_arg)
    else:
        candidates.extend(
            [checkpoint_arg, model_dir / checkpoint_arg, model_dir / "checkpoints" / checkpoint_arg]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Checkpoint not found; tried: {rendered}")


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_var_dims(path: Path, support_genes: list[str]) -> dict[str, Any]:
    with path.open("rb") as handle:
        var_dims = pickle.load(handle)
    require(isinstance(var_dims, dict), f"Expected a dictionary in {path}")
    required = ("gene_names", "input_dim", "output_dim", "pert_dim")
    missing = [key for key in required if key not in var_dims]
    require(not missing, f"var_dims.pkl lacks keys: {missing}")
    checkpoint_genes = [str(gene) for gene in var_dims["gene_names"]]
    require(
        checkpoint_genes == support_genes,
        "var_dims.pkl gene_names do not exactly match the support gene axis",
    )
    require(int(var_dims["input_dim"]) == len(support_genes), "Unexpected STATE input_dim")
    require(int(var_dims["output_dim"]) == len(support_genes), "Unexpected STATE output_dim")
    require(int(var_dims["pert_dim"]) == EXPECTED_PERT_DIM, "Unexpected STATE pert_dim")
    return var_dims


def load_perturbation_embeddings(
    path: Path, targets: list[str]
) -> dict[str, torch.Tensor]:
    raw_map = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(raw_map, dict), f"Expected a dictionary in {path}")
    embedding_map = {str(key): value for key, value in raw_map.items()}
    required_names = [CONTROL_LABEL, *targets]
    missing = [name for name in required_names if name not in embedding_map]
    require(
        not missing,
        f"Perturbation map misses {len(missing)} required entries: {missing[:10]}",
    )
    result: dict[str, torch.Tensor] = {}
    for name in required_names:
        tensor = torch.as_tensor(embedding_map[name], dtype=torch.float32).reshape(-1)
        require(
            tensor.numel() == EXPECTED_PERT_DIM,
            f"Embedding for {name} has dimension {tensor.numel()}, expected 5120",
        )
        require(bool(torch.isfinite(tensor).all()), f"Embedding for {name} is not finite")
        result[name] = tensor.contiguous()
    require(
        float(torch.linalg.vector_norm(result[CONTROL_LABEL])) == 0.0,
        "Expected the saved non-targeting perturbation embedding to be the zero vector",
    )
    return result


def stratified_control_indices(
    obs: pd.DataFrame, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, int]]:
    require(count > 0 and count <= len(obs), "Invalid number of sampled control cells")
    require("ntc_id" in obs.columns, "Control obs lacks ntc_id")
    strata = sorted(str(value) for value in obs["ntc_id"].unique())
    require(strata, "No NTC strata found")
    base, remainder = divmod(count, len(strata))
    sampled: list[int] = []
    allocation: dict[str, int] = {}
    for index, stratum in enumerate(strata):
        take = base + int(index < remainder)
        candidates = np.flatnonzero(obs["ntc_id"].astype(str).to_numpy() == stratum)
        require(take <= len(candidates), f"NTC stratum {stratum} has too few cells")
        chosen = rng.choice(candidates, size=take, replace=False)
        sampled.extend(int(value) for value in chosen)
        allocation[stratum] = take
    sampled_array = np.asarray(sampled, dtype=np.int64)
    rng.shuffle(sampled_array)
    require(len(sampled_array) == count, "Stratified sampling returned the wrong cell count")
    return sampled_array, allocation


def load_control_sets(
    path: Path,
    context: str,
    current_genes: list[str],
    support_to_current: np.ndarray,
    set_size: int,
    sets_per_context: int,
    seed: int,
    input_normalization: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    total_cells = set_size * sets_per_context
    adata = ad.read_h5ad(path, backed="r")
    try:
        require(adata.n_vars == len(current_genes), f"Unexpected gene count in {path}")
        var_names = [str(value) for value in adata.var_names]
        require(var_names == current_genes, f"Gene order in {path} does not match gene_names.csv")
        obs = adata.obs.copy()
        require("context" in obs.columns, f"Control obs lacks context in {path}")
        require(set(obs["context"].astype(str)) == {context}, f"Wrong context labels in {path}")
        require("target_gene" in obs.columns, f"Control obs lacks target_gene in {path}")
        require(
            set(obs["target_gene"].astype(str)) == {CONTROL_LABEL},
            f"Non-control rows found in {path}",
        )
        rng = np.random.default_rng(seed)
        indices, allocation = stratified_control_indices(obs, total_cells, rng)

        # Backed sparse arrays require sorted fancy indices. Restore the sampled
        # order after loading so independently shuffled 128-cell sets are retained.
        order = np.argsort(indices)
        sorted_indices = indices[order]
        inverse_order = np.argsort(order)
        raw = adata.X[sorted_indices]
        if sp.issparse(raw):
            raw = raw.tocsr()[inverse_order]
            raw_values = raw.data
        else:
            raw = np.asarray(raw)[inverse_order]
            raw_values = raw.reshape(-1)
    finally:
        adata.file.close()

    require(np.isfinite(raw_values).all(), f"Non-finite raw values in {path}")
    require(np.min(raw_values, initial=0) >= 0, f"Negative raw values in {path}")
    require(
        np.allclose(raw_values, np.rint(raw_values), atol=1e-6),
        f"Control matrix in {path} is not raw integer-like counts",
    )

    valid_support = support_to_current >= 0
    current_columns = support_to_current[valid_support]
    selected = raw[:, current_columns]
    if sp.issparse(selected):
        selected = selected.toarray()
    selected = np.asarray(selected, dtype=np.float32)
    model_input = np.zeros((total_cells, len(support_to_current)), dtype=np.float32)
    model_input[:, valid_support] = selected
    library_sizes = model_input.sum(axis=1)
    require(np.all(library_sizes > 0), f"Zero support-axis library size in {path}")
    if input_normalization == "cp10k-log1p":
        model_input *= (10_000.0 / library_sizes)[:, None]
    elif input_normalization != "log1p-raw":
        raise RuntimeError(f"Unsupported input normalization: {input_normalization}")
    np.log1p(model_input, out=model_input)
    require(np.isfinite(model_input).all(), f"Normalization produced non-finite values in {path}")

    qc = {
        "path": str(path.resolve()),
        "available_cells": int(len(obs)),
        "sampled_cells": int(total_cells),
        "sets": int(sets_per_context),
        "set_size": int(set_size),
        "ntc_strata": int(len(allocation)),
        "ntc_allocation_min": int(min(allocation.values())),
        "ntc_allocation_max": int(max(allocation.values())),
        "input_normalization": input_normalization,
        "raw_support_library_size": summarize(library_sizes),
        "model_input_expression": summarize(model_input),
        "sample_index_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }
    return model_input.reshape(sets_per_context, set_size, -1), qc


def summarize(values: np.ndarray | Iterable[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    require(array.size > 0 and np.isfinite(array).all(), "Cannot summarize invalid values")
    quantiles = np.quantile(array, [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    return {
        "min": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": float(quantiles[6]),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def summarize_optional(values: np.ndarray | Iterable[float]) -> dict[str, float] | None:
    array = np.asarray(values)
    if array.size == 0:
        return None
    return summarize(array)


def sparsify_effect(
    raw_support_effect: np.ndarray,
    target: str,
    current_gene_index: dict[str, int],
    support_to_current: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | int]]:
    current_effect = np.zeros(EXPECTED_CURRENT_GENES, dtype=np.float32)
    valid_support = support_to_current >= 0
    current_effect[support_to_current[valid_support]] = raw_support_effect[valid_support]
    current_effect *= np.float32(args.effect_scale)
    np.clip(current_effect, -args.effect_clip, args.effect_clip, out=current_effect)

    target_index = current_gene_index[target]
    modeled_target = float(current_effect[target_index])
    if args.target_strategy == "force":
        final_target = float(args.target_log_effect)
    elif args.target_strategy == "blend":
        weight = float(args.target_blend_weight)
        final_target = (1.0 - weight) * modeled_target + weight * float(args.target_log_effect)
    else:
        final_target = modeled_target
    current_effect[target_index] = np.float32(final_target)

    eligible = np.flatnonzero(np.abs(current_effect) >= args.min_abs_effect)
    target_is_nonzero = bool(current_effect[target_index] != 0.0)
    if target_is_nonzero and target_index not in eligible:
        eligible = np.append(eligible, target_index)
    if len(eligible) > args.top_k:
        if target_is_nonzero:
            others = eligible[eligible != target_index]
            slots = args.top_k - 1
            order = np.argsort(-np.abs(current_effect[others]), kind="stable")[:slots]
            keep = np.concatenate(([target_index], others[order]))
        else:
            order = np.argsort(-np.abs(current_effect[eligible]), kind="stable")[: args.top_k]
            keep = eligible[order]
    else:
        keep = eligible
    sparse_effect = np.zeros_like(current_effect)
    sparse_effect[keep] = current_effect[keep]
    require(np.count_nonzero(sparse_effect) <= args.top_k, "Top-k pruning failed")
    return sparse_effect, {
        "modeled_target_effect": modeled_target,
        "final_target_effect": float(sparse_effect[target_index]),
        "retained_effects": int(np.count_nonzero(sparse_effect)),
        "raw_support_l2": float(np.linalg.norm(raw_support_effect)),
        "raw_support_abs_max": float(np.max(np.abs(raw_support_effect))),
    }


def load_state_model(checkpoint: Path, device: torch.device) -> Any:
    try:
        from state.tx.models.state_transition import StateTransitionPerturbationModel
    except ImportError as error:
        raise RuntimeError(
            "STATE is not importable. Install the ArcInstitute/state package in this environment."
        ) from error
    model = StateTransitionPerturbationModel.load_from_checkpoint(
        str(checkpoint), map_location="cpu", weights_only=False
    )
    model = model.to(device)
    model.eval()
    return model


def validate_model(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    input_dim = int(getattr(model, "input_dim", -1))
    output_dim = int(getattr(model, "output_dim", -1))
    pert_dim = int(getattr(model, "pert_dim", -1))
    set_size = int(getattr(model, "cell_sentence_len", -1))
    require(input_dim == EXPECTED_SUPPORT_GENES, f"Checkpoint input_dim is {input_dim}")
    require(output_dim == EXPECTED_SUPPORT_GENES, f"Checkpoint output_dim is {output_dim}")
    require(pert_dim == EXPECTED_PERT_DIM, f"Checkpoint pert_dim is {pert_dim}")
    require(set_size == args.set_size, f"Checkpoint set size is {set_size}, not {args.set_size}")
    require(getattr(model, "batch_encoder", None) is None, "This adapter requires batch_encoder=false")
    output_space = str(getattr(model, "output_space", ""))
    require(output_space == "all", f"This adapter requires output_space=all, found {output_space}")
    require(
        not bool(getattr(model, "log1p_from_raw_counts", False)),
        "Checkpoint normalizes raw counts internally, but this adapter supplies log1p values",
    )
    return {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "input_dim": input_dim,
        "output_dim": output_dim,
        "pert_dim": pert_dim,
        "cell_sentence_len": set_size,
        "output_space": output_space,
        "log1p_from_raw_counts": bool(getattr(model, "log1p_from_raw_counts", False)),
        "predict_residual": bool(getattr(model, "predict_residual", False)),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def autocast_context(args: argparse.Namespace, device: torch.device):
    if args.precision == "bfloat16":
        require(device.type == "cuda", "bfloat16 mode is supported here only on CUDA")
        require(torch.cuda.is_bf16_supported(), "Selected CUDA device lacks bfloat16 support")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def infer_context(
    model: Any,
    control_sets: np.ndarray,
    targets: list[str],
    embeddings: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    context: str,
) -> tuple[np.ndarray, list[dict[str, float]], dict[str, Any]]:
    require(
        control_sets.shape == (args.sets_per_context, args.set_size, EXPECTED_SUPPORT_GENES),
        f"Unexpected control tensor shape for context {context}: {control_sets.shape}",
    )
    basal_tensors = [
        torch.from_numpy(control_set).to(device=device, dtype=torch.float32)
        for control_set in control_sets
    ]
    basal_mean = torch.stack([tensor.sum(dim=0) for tensor in basal_tensors]).sum(dim=0)
    basal_mean /= float(args.sets_per_context * args.set_size)
    effects = np.empty((len(targets), EXPECTED_SUPPORT_GENES), dtype=np.float32)
    prediction_qc: list[dict[str, float]] = []

    def predict_mean(
        pert_vector: torch.Tensor, pert_name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction_sum = torch.zeros(
            EXPECTED_SUPPORT_GENES, device=device, dtype=torch.float32
        )
        prediction_min = torch.full((), float("inf"), device=device)
        prediction_max = torch.full((), float("-inf"), device=device)
        for basal in basal_tensors:
            pert = pert_vector.unsqueeze(0).expand(args.set_size, -1)
            batch = {
                "ctrl_cell_emb": basal,
                "pert_emb": pert,
                "pert_name": [pert_name] * args.set_size,
            }
            with autocast_context(args, device):
                output = model.predict_step(batch, 0, padded=False)
            require(isinstance(output, dict) and "preds" in output, "STATE output lacks preds")
            prediction = output["preds"]
            require(
                tuple(prediction.shape) == (args.set_size, EXPECTED_SUPPORT_GENES),
                f"STATE returned shape {tuple(prediction.shape)} for {context}/{pert_name}",
            )
            prediction = prediction.float()
            prediction_sum += prediction.sum(dim=0)
            prediction_min = torch.minimum(prediction_min, prediction.amin())
            prediction_max = torch.maximum(prediction_max, prediction.amax())
            del prediction, output
        require(
            bool(torch.isfinite(prediction_sum).all()),
            f"Non-finite STATE output for {context}/{pert_name}",
        )
        prediction_mean = prediction_sum / float(args.sets_per_context * args.set_size)
        return prediction_mean, prediction_min, prediction_max

    with torch.inference_mode():
        if args.effect_reference == "model-control":
            control_vector = embeddings[CONTROL_LABEL].to(
                device=device, dtype=torch.float32
            )
            reference_mean, control_min, control_max = predict_mean(
                control_vector, CONTROL_LABEL
            )
        else:
            reference_mean = basal_mean
            control_min = basal_mean.amin()
            control_max = basal_mean.amax()

        for target_index, target in enumerate(targets):
            pert_vector = embeddings[target].to(device=device, dtype=torch.float32)
            prediction_mean, prediction_min_tensor, prediction_max_tensor = predict_mean(
                pert_vector, target
            )
            effects[target_index] = (prediction_mean - reference_mean).cpu().numpy()
            prediction_qc.append(
                {
                    "prediction_min": float(prediction_min_tensor.item()),
                    "prediction_max": float(prediction_max_tensor.item()),
                    "delta_abs_max": float(
                        torch.max(torch.abs(prediction_mean - reference_mean)).item()
                    ),
                }
            )
            if (target_index + 1) % 25 == 0 or target_index + 1 == len(targets):
                print(
                    f"[{context}] inferred {target_index + 1}/{len(targets)} targets",
                    flush=True,
                )
    reference_qc = {
        "effect_reference": args.effect_reference,
        "reference_prediction_min": float(control_min.item()),
        "reference_prediction_max": float(control_max.item()),
        "reference_minus_input": summarize(
            (reference_mean - basal_mean).detach().cpu().numpy()
        ),
    }
    return effects, prediction_qc, reference_qc


def validate_args(args: argparse.Namespace) -> None:
    require(args.set_size == 128, "STATE state_sm inference must use exact 128-cell sets")
    require(args.sets_per_context > 0, "sets-per-context must be positive")
    require(1 <= args.top_k <= 161, "top-k must be between 1 and 161")
    require(args.min_abs_effect >= 0, "min-abs-effect must be nonnegative")
    require(args.effect_scale > 0, "effect-scale must be positive")
    require(args.effect_clip > 0, "effect-clip must be positive")
    require(0 <= args.target_blend_weight <= 1, "target-blend-weight must be in [0, 1]")
    require(np.isfinite(args.target_log_effect), "target-log-effect must be finite")
    require(args.output_npz.suffix == ".npz", "output-npz must end in .npz")
    require(args.output_json.suffix == ".json", "output-json must end in .json")
    require(args.output_npz.resolve() != args.output_json.resolve(), "Output paths must differ")
    require(not args.output_npz.exists(), f"Refusing to overwrite {args.output_npz}")
    require(not args.output_json.exists(), f"Refusing to overwrite {args.output_json}")


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = args.model_dir.resolve()
    checkpoint = resolve_checkpoint(model_dir, args.checkpoint)
    pert_map_path = model_dir / "pert_onehot_map.pt"
    var_dims_path = model_dir / "var_dims.pkl"
    config_path = model_dir / "config.yaml"
    for path in (pert_map_path, var_dims_path):
        require(path.is_file(), f"Required STATE run file is missing: {path}")

    controls_dir = args.controls_dir.resolve()
    current_gene_path = controls_dir / "gene_names.csv"
    target_path = controls_dir / "pert_counts.csv"
    control_paths = {context: controls_dir / f"context_{context}.h5ad" for context in CONTEXTS}
    required_inputs = [current_gene_path, target_path, args.support_genes, *control_paths.values()]
    for path in required_inputs:
        require(path.is_file(), f"Required input is missing: {path}")

    current_genes = read_single_column(current_gene_path, "gene_name")
    support_genes = read_single_column(args.support_genes.resolve())
    targets = read_targets(target_path)
    require(len(current_genes) == EXPECTED_CURRENT_GENES, "Unexpected current gene count")
    require(len(support_genes) == EXPECTED_SUPPORT_GENES, "Unexpected support gene count")
    current_gene_index = {gene: index for index, gene in enumerate(current_genes)}
    support_gene_set = set(support_genes)
    support_to_current = np.asarray(
        [current_gene_index.get(gene, -1) for gene in support_genes], dtype=np.int64
    )
    overlap = int(np.count_nonzero(support_to_current >= 0))
    current_only_indices = np.asarray(
        [index for index, gene in enumerate(current_genes) if gene not in support_gene_set],
        dtype=np.int64,
    )
    require(overlap == EXPECTED_OVERLAP_GENES, f"Unexpected gene overlap: {overlap}")
    require(
        len(current_only_indices) == EXPECTED_CURRENT_ONLY_GENES,
        f"Unexpected current-only gene count: {len(current_only_indices)}",
    )
    require(all(target in current_gene_index for target in targets), "A target is absent from current genes")
    require(all(target in support_gene_set for target in targets), "A target is absent from support genes")

    var_dims = load_var_dims(var_dims_path, support_genes)
    embeddings = load_perturbation_embeddings(pert_map_path, targets)

    device = torch.device(args.device)
    if args.require_cuda:
        require(device.type == "cuda", "--require-cuda was set but device is not CUDA")
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is not available")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        device_name = torch.cuda.get_device_name(device)
        if args.require_h100:
            require("H100" in device_name.upper(), f"Selected GPU is not an H100: {device_name}")
    else:
        require(not args.require_h100, "--require-h100 requires a CUDA device")
        device_name = "CPU"

    print(f"Loading STATE checkpoint {checkpoint}", flush=True)
    model = load_state_model(checkpoint, device)
    model_info = validate_model(model, args)
    print(
        f"Loaded {model_info['parameters']:,} parameters on {device_name}; "
        f"all {len(targets)} target embeddings have dimension {var_dims['pert_dim']}",
        flush=True,
    )

    all_effects = np.zeros(
        (len(CONTEXTS), len(targets), len(current_genes)), dtype=np.float32
    )
    control_qc: dict[str, Any] = {}
    group_qc: list[dict[str, Any]] = []
    for context_index, context in enumerate(CONTEXTS):
        control_sets, context_control_qc = load_control_sets(
            control_paths[context],
            context,
            current_genes,
            support_to_current,
            args.set_size,
            args.sets_per_context,
            args.seed + context_index,
            args.input_normalization,
        )
        control_qc[context] = context_control_qc
        raw_context_effects, prediction_qc, reference_qc = infer_context(
            model, control_sets, targets, embeddings, device, args, context
        )
        control_qc[context]["state_reference"] = reference_qc
        del control_sets
        for target_index, target in enumerate(targets):
            sparse_effect, effect_qc = sparsify_effect(
                raw_context_effects[target_index],
                target,
                current_gene_index,
                support_to_current,
                args,
            )
            all_effects[context_index, target_index] = sparse_effect
            group_qc.append(
                {
                    "context": context,
                    "target": target,
                    **effect_qc,
                    **prediction_qc[target_index],
                }
            )
        del raw_context_effects
        if device.type == "cuda":
            torch.cuda.empty_cache()

    require(all_effects.shape == (3, 300, 18_533), "Final effect tensor has the wrong shape")
    require(np.isfinite(all_effects).all(), "Final effect tensor contains non-finite values")
    require(
        np.count_nonzero(all_effects[:, :, current_only_indices]) == 0,
        "A current-only gene received a nonzero delta",
    )
    nonzero_counts = np.count_nonzero(all_effects, axis=2)
    require(int(nonzero_counts.max()) <= args.top_k <= 161, "Effect sparsity contract failed")

    atomic_save_npz(
        args.output_npz,
        contexts=np.asarray(CONTEXTS, dtype="U1"),
        targets=np.asarray(targets, dtype="U64"),
        genes=np.asarray(current_genes, dtype="U64"),
        effects=all_effects,
    )
    elapsed = time.time() - started
    target_effects = np.asarray(
        [
            all_effects[cidx, tidx, current_gene_index[target]]
            for cidx in range(len(CONTEXTS))
            for tidx, target in enumerate(targets)
        ]
    )
    payload = {
        "schema": "vcc-state-effect-prior-v1",
        "created_utc_epoch": time.time(),
        "elapsed_seconds": elapsed,
        "command": [sys.executable, *sys.argv],
        "configuration": {
            "seed": args.seed,
            "device": str(device),
            "device_name": device_name,
            "precision": args.precision,
            "set_size": args.set_size,
            "sets_per_context": args.sets_per_context,
            "cells_per_context": args.set_size * args.sets_per_context,
            "input_normalization": args.input_normalization,
            "effect_reference": args.effect_reference,
            "top_k_including_target": args.top_k,
            "min_abs_effect": args.min_abs_effect,
            "effect_scale": args.effect_scale,
            "effect_clip": args.effect_clip,
            "target_strategy": args.target_strategy,
            "target_log_effect": args.target_log_effect,
            "target_blend_weight": args.target_blend_weight,
        },
        "model": model_info,
        "axes": {
            "contexts": list(CONTEXTS),
            "targets": len(targets),
            "current_genes": len(current_genes),
            "support_genes": len(support_genes),
            "overlap_genes": overlap,
            "current_only_zero_delta_genes": len(current_only_indices),
            "support_only_genes": int(np.count_nonzero(support_to_current < 0)),
            "esm_target_coverage": sum(target in embeddings for target in targets),
            "esm_dimension": EXPECTED_PERT_DIM,
            "control_embedding_l2": float(
                torch.linalg.vector_norm(embeddings[CONTROL_LABEL]).item()
            ),
        },
        "qc": {
            "controls": control_qc,
            "nonzero_effects_per_group": summarize(nonzero_counts),
            "target_effects": summarize(target_effects),
            "all_retained_effects": summarize_optional(all_effects[all_effects != 0]),
            "current_only_nonzero_effects": int(
                np.count_nonzero(all_effects[:, :, current_only_indices])
            ),
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
            "checkpoint": describe_file(checkpoint),
            "perturbation_map": describe_file(pert_map_path),
            "var_dims": describe_file(var_dims_path),
            "state_config": describe_file(config_path) if config_path.is_file() else None,
            "current_gene_axis": describe_file(current_gene_path),
            "support_gene_axis": describe_file(args.support_genes.resolve()),
            "target_axis": describe_file(target_path),
            "controls": {
                context: describe_file(path) for context, path in control_paths.items()
            },
            "output_npz": describe_file(args.output_npz),
        },
    }
    atomic_save_json(args.output_json, payload)
    print(f"Wrote {args.output_npz} ({args.output_npz.stat().st_size:,} bytes)", flush=True)
    print(f"Wrote {args.output_json} ({args.output_json.stat().st_size:,} bytes)", flush=True)
    print(
        f"QC: nonzero effects/group {int(nonzero_counts.min())}..{int(nonzero_counts.max())}; "
        f"current-only nonzeros=0; elapsed={elapsed / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
