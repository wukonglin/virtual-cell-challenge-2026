#!/usr/bin/env python3
"""Run a real-data CUDA/H100 smoke test for the VCC 2026 controls.

This is an infrastructure and tensor-path test, not a perturbation model.  The
autoencoder sees controls only.  The optional mini H5AD is deliberately
incomplete and applies only a target-transcript knockdown, which the official
metrics exclude.  It must never be submitted or interpreted as biology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import socket
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F


CONTEXTS = ("A", "B", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/controls"))
    parser.add_argument(
        "--output-json", type=Path, default=Path("artifacts/h100_smoke_result.json")
    )
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=Path("artifacts/h100_smoke_mini_prediction.h5ad"),
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--cells-per-context", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--mini-targets", type=int, default=8)
    parser.add_argument("--mini-cells", type=int, default=16)
    return parser.parse_args()


def reject_path_collisions(args: argparse.Namespace) -> None:
    inputs = {
        *(args.data_dir / f"context_{context}.h5ad" for context in CONTEXTS),
        args.data_dir / "gene_names.csv",
        args.data_dir / "pert_counts.csv",
    }
    resolved_inputs = {path.resolve() for path in inputs}
    resolved_outputs = [args.output_json.resolve(), args.output_h5ad.resolve()]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("--output-json and --output-h5ad must be different files")
    collisions = resolved_inputs.intersection(resolved_outputs)
    if collisions:
        raise ValueError(f"refusing to overwrite challenge input(s): {sorted(map(str, collisions))}")


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def read_csv_column(path: Path, column: str) -> list[str]:
    frame = pd.read_csv(path)
    if list(frame.columns) != [column]:
        raise ValueError(f"{path}: expected only {column!r}, found {list(frame.columns)!r}")
    return frame[column].astype(str).tolist()


def stratified_indices(obs: pd.DataFrame, n_cells: int, seed: int) -> np.ndarray:
    if n_cells > len(obs):
        raise ValueError(f"requested {n_cells} cells from only {len(obs)} controls")
    rng = np.random.default_rng(seed)
    groups = obs.groupby("ntc_id", observed=True, sort=True).indices
    keys = sorted(groups)
    base, remainder = divmod(n_cells, len(keys))
    chosen: list[int] = []
    for position, key in enumerate(keys):
        count = base + int(position < remainder)
        candidates = np.asarray(groups[key], dtype=np.int64)
        selected = rng.choice(candidates, size=count, replace=False)
        chosen.extend(selected.tolist())
    return np.sort(np.asarray(chosen, dtype=np.int64))


def dense_rows(adata: ad.AnnData, indices: np.ndarray) -> np.ndarray:
    block = adata.X[indices]
    if sparse.issparse(block):
        block = block.toarray()
    result = np.asarray(block, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {result.shape}")
    return result


def normalize_log1p(raw: np.ndarray) -> np.ndarray:
    library = raw.sum(axis=1, keepdims=True)
    if np.any(library <= 0):
        raise ValueError("zero-library control cell encountered")
    return np.log1p(raw / library * 10_000.0).astype(np.float32, copy=False)


class DenoisingAutoencoder(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int, latent_dim: int, n_contexts: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.context_embedding = nn.Embedding(n_contexts, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x) + self.context_embedding(context)
        return self.decoder(latent)


def validate_mini(
    prediction: ad.AnnData,
    genes: list[str],
    targets: list[str],
    mini_cells: int,
) -> dict[str, Any]:
    matrix = prediction.X.tocsr()
    groups = prediction.obs.groupby(["context", "target_gene"], observed=True).size()
    values = matrix.data
    checks = {
        "gene_axis_exact": list(prediction.var_names.astype(str)) == genes,
        "contexts_exact": sorted(prediction.obs["context"].astype(str).unique())
        == list(CONTEXTS),
        "target_subset_exact": sorted(prediction.obs["target_gene"].astype(str).unique())
        == sorted(targets),
        "cells_per_group_exact": bool((groups == mini_cells).all()),
        "no_controls": "non-targeting"
        not in set(prediction.obs["target_gene"].astype(str)),
        "finite": bool(np.isfinite(values).all()),
        "nonnegative": bool((values >= 0).all()),
        "integer_values": bool(np.equal(values, np.floor(values)).all()),
        "max_cell_total_le_1m": bool(
            np.asarray(matrix.sum(axis=1)).ravel().max() <= 1_000_000
        ),
        "no_explicit_zeros": not bool(np.any(values == 0)),
    }
    return {
        "shape": list(prediction.shape),
        "nnz": int(matrix.nnz),
        "groups": int(len(groups)),
        "checks": checks,
        "passed": all(checks.values()),
        "officially_submittable": False,
        "official_rejection_reason": (
            f"mini smoke artifact has {len(targets)} targets and {mini_cells} cells/group; "
            "the official contract requires all 300 targets and 400 cells/group"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.cells_per_context < len(CONTEXTS):
        raise ValueError("--cells-per-context must be positive and large enough to sample")
    if args.batch_size < 1 or args.steps < 1:
        raise ValueError("--batch-size and --steps must both be >= 1")
    if args.hidden_dim < 1 or args.latent_dim < 1:
        raise ValueError("--hidden-dim and --latent-dim must both be >= 1")
    if args.mini_targets < 1 or args.mini_cells < 1:
        raise ValueError("--mini-targets and --mini-cells must both be >= 1")
    reject_path_collisions(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this smoke test must run inside an H100 allocation")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    device_name = torch.cuda.get_device_name(device)
    if "H100" not in device_name.upper():
        raise RuntimeError(f"expected an H100 allocation, found {device_name!r}")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError(
            f"expected H100 compute capability (9, 0), found {torch.cuda.get_device_capability(device)}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the allocated CUDA device does not support BF16")

    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats(device)

    genes = read_csv_column(args.data_dir / "gene_names.csv", "gene_name")
    targets = read_csv_column(args.data_dir / "pert_counts.csv", "target_gene")
    if args.mini_targets > len(targets):
        raise ValueError(f"requested {args.mini_targets} mini targets from only {len(targets)}")
    selected_targets = targets[: args.mini_targets]
    gene_to_index = {gene: index for index, gene in enumerate(genes)}

    raw_by_context: dict[str, np.ndarray] = {}
    index_by_context: dict[str, np.ndarray] = {}
    normalized_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    loading_started = time.perf_counter()
    for context_index, context in enumerate(CONTEXTS):
        adata = ad.read_h5ad(args.data_dir / f"context_{context}.h5ad", backed="r")
        try:
            required_obs = {"target_gene", "context", "ntc_id"}
            if adata.shape != (18_400, len(genes)):
                raise ValueError(f"context {context} has unexpected shape {adata.shape}")
            if not required_obs.issubset(adata.obs.columns):
                raise ValueError(f"context {context} is missing required obs columns")
            if set(adata.obs["context"].astype(str)) != {context}:
                raise ValueError(f"context {context} file contains a different context label")
            if set(adata.obs["target_gene"].astype(str)) != {"non-targeting"}:
                raise ValueError(f"context {context} file contains non-control target labels")
            if adata.obs["ntc_id"].nunique() != 46:
                raise ValueError(f"context {context} does not contain exactly 46 NTC IDs")
            if list(adata.var_names.astype(str)) != genes:
                raise ValueError(f"context {context} gene axis does not match gene_names.csv")
            indices = stratified_indices(
                adata.obs, args.cells_per_context, args.seed + context_index
            )
            raw = dense_rows(adata, indices)
        finally:
            adata.file.close()
        raw_by_context[context] = raw
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError(f"context {context} sampled controls contain invalid counts")
        if not np.equal(raw, np.floor(raw)).all():
            raise ValueError(f"context {context} sampled controls are not whole-valued")
        if np.any(raw.sum(axis=1) <= 0):
            raise ValueError(f"context {context} sampled controls contain zero libraries")
        index_by_context[context] = indices
        normalized_blocks.append(normalize_log1p(raw))
        label_blocks.append(
            np.full(args.cells_per_context, context_index, dtype=np.int64)
        )
    loading_seconds = time.perf_counter() - loading_started

    normalized = np.concatenate(normalized_blocks, axis=0)
    labels = np.concatenate(label_blocks, axis=0)
    x_cpu = torch.from_numpy(normalized)
    labels_cpu = torch.from_numpy(labels)

    model = DenoisingAutoencoder(
        n_genes=len(genes),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_contexts=len(CONTEXTS),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    losses: list[float] = []
    step_milliseconds: list[float] = []
    gradient_norms: list[float] = []

    model.train()
    batch_indices = torch.randint(len(x_cpu), (args.batch_size,), generator=generator)
    clean = x_cpu[batch_indices].to(device, non_blocking=True)
    context_labels = labels_cpu[batch_indices].to(device, non_blocking=True)
    corruption = torch.rand(clean.shape, device=device) < 0.08
    noisy = clean.masked_fill(corruption, 0)
    first_parameter_before = next(model.parameters()).detach().clone()
    for _ in range(args.steps):

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            reconstruction = model(noisy, context_labels)
            loss = F.mse_loss(reconstruction.float(), clean.float())
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss: {loss}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float("inf"), error_if_nonfinite=True
        )
        gradient_norms.append(float(gradient_norm.detach().cpu()))
        optimizer.step()
        stop.record()
        torch.cuda.synchronize(device)
        step_milliseconds.append(float(start.elapsed_time(stop)))
        losses.append(float(loss.detach().cpu()))

    parameter_update_max = float(
        (next(model.parameters()).detach() - first_parameter_before).abs().max().cpu()
    )
    if parameter_update_max <= 0:
        raise RuntimeError("optimizer did not update the model parameters")
    if not all(norm > 0 and np.isfinite(norm) for norm in gradient_norms):
        raise RuntimeError(f"invalid gradient norms: {gradient_norms}")
    if args.steps > 1 and min(losses[1:]) >= losses[0]:
        raise RuntimeError(f"fixed-batch training loss did not improve: {losses}")

    model.eval()
    with torch.no_grad():
        centroids = []
        for context_index in range(len(CONTEXTS)):
            block = x_cpu[labels_cpu == context_index].to(device)
            centroids.append(block.mean(dim=0))
        normalized_centroids = F.normalize(torch.stack(centroids), dim=1)
        context_cosine = (normalized_centroids @ normalized_centroids.T).cpu().numpy()

        # A small BF16 GEMM proves that tensor-core matrix multiplication works.
        left = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
        right = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
        gemm_start = torch.cuda.Event(enable_timing=True)
        gemm_stop = torch.cuda.Event(enable_timing=True)
        gemm_start.record()
        product = left @ right
        gemm_stop.record()
        torch.cuda.synchronize(device)
        gemm_ms = float(gemm_start.elapsed_time(gemm_stop))
        gemm_checksum = float(product.float().mean().cpu())
        del left, right, product

    mini_blocks: list[sparse.csr_matrix] = []
    mini_obs: list[pd.DataFrame] = []
    for context_position, context in enumerate(CONTEXTS):
        raw = raw_by_context[context]
        for target_position, target in enumerate(selected_targets):
            start = (target_position * args.mini_cells) % len(raw)
            rows = np.arange(start, start + args.mini_cells) % len(raw)
            counts_gpu = torch.from_numpy(raw[rows]).to(device)
            target_index = gene_to_index[target]
            counts_gpu[:, target_index] = torch.floor(
                counts_gpu[:, target_index] * 0.2
            )
            counts = counts_gpu.to(torch.int32).cpu().numpy()
            mini_blocks.append(sparse.csr_matrix(counts, dtype=np.int32))
            mini_obs.append(
                pd.DataFrame(
                    {
                        "target_gene": [target] * args.mini_cells,
                        "context": [context] * args.mini_cells,
                    },
                    index=[
                        f"smoke_{context}_{target}_{cell:03d}"
                        for cell in range(args.mini_cells)
                    ],
                )
            )

    mini_matrix = sparse.vstack(mini_blocks, format="csr")
    mini_matrix.eliminate_zeros()
    prediction = ad.AnnData(
        X=mini_matrix,
        obs=pd.concat(mini_obs),
        var=pd.DataFrame(index=pd.Index(genes, name="gene_name")),
    )
    prediction.uns["warning"] = (
        "Infrastructure-only H100 smoke artifact. Incomplete and not biologically "
        "predictive. Never submit."
    )
    prediction.uns["method"] = (
        "Resampled controls with an 80% target-transcript reduction only; all "
        "official metrics exclude that transcript."
    )
    prediction.uns["seed"] = args.seed
    mini_validation = validate_mini(
        prediction, genes, selected_targets, args.mini_cells
    )
    if not mini_validation["passed"]:
        raise RuntimeError(f"mini prediction validation failed: {mini_validation}")
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    partial_h5ad = args.output_h5ad.with_name(args.output_h5ad.name + ".partial")
    if partial_h5ad.exists():
        partial_h5ad.unlink()
    prediction.write_h5ad(partial_h5ad, compression="gzip")
    reopened = ad.read_h5ad(partial_h5ad, backed="r")
    try:
        if reopened.shape != prediction.shape:
            raise RuntimeError("reopened mini H5AD shape differs from in-memory prediction")
        if list(reopened.var_names.astype(str)) != genes:
            raise RuntimeError("reopened mini H5AD gene order changed")
    finally:
        reopened.file.close()
    os.replace(partial_h5ad, args.output_h5ad)

    torch.cuda.synchronize(device)
    result: dict[str, Any] = {
        "status": "PASS",
        "purpose": "H100/CUDA/real-control tensor-path smoke test only",
        "biological_performance_claim": False,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "anndata": ad.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "gpu": {
            "name": device_name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(properties.total_memory),
            "multiprocessor_count": int(properties.multi_processor_count),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "data": {
            "contexts": list(CONTEXTS),
            "genes": len(genes),
            "official_targets": len(targets),
            "cells_per_context_loaded": args.cells_per_context,
            "total_cells_loaded": int(len(normalized)),
            "selected_control_row_indices": {
                context: indices.tolist()
                for context, indices in index_by_context.items()
            },
            "loading_seconds": loading_seconds,
            "context_centroid_cosine": context_cosine.tolist(),
        },
        "model_smoke": {
            "architecture": "control-only context-conditioned denoising autoencoder",
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "precision": "BF16 autocast with FP32 MSE",
            "losses": losses,
            "loss_improved": bool(min(losses[1:]) < losses[0]) if len(losses) > 1 else None,
            "gradient_norms": gradient_norms,
            "parameter_update_max": parameter_update_max,
            "step_milliseconds": step_milliseconds,
            "median_step_milliseconds_excluding_warmup": float(
                np.median(step_milliseconds[1:] if len(step_milliseconds) > 1 else step_milliseconds)
            ),
        },
        "bf16_gemm": {
            "shape": [4096, 4096],
            "milliseconds": gemm_ms,
            "mean_checksum": gemm_checksum,
        },
        "mini_prediction": {
            **mini_validation,
            "path": str(args.output_h5ad),
            "bytes": args.output_h5ad.stat().st_size,
            "sha256": sha256_file(args.output_h5ad),
            "targets": selected_targets,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
