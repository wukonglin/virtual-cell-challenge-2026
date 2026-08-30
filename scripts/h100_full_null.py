#!/usr/bin/env python3
"""Generate a complete context-informed null prediction on one H100.

The output covers all 900 context-target groups and is intended solely for
schema and pipeline validation.  Every target within a context receives the
same control-derived distribution, so it contains no target-specific biology
and must not be submitted for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
import torch


CONTEXTS = ("A", "B", "C")
MAX_NNZ = 4_750_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/controls"))
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=Path("artifacts/h100_context_null_full.h5ad"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/h100_context_null_full.json"),
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--cells-per-group", type=int, default=400)
    parser.add_argument("--genes-per-cell", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
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
    if args.output_h5ad.exists() and not args.force:
        raise FileExistsError(
            f"output already exists: {args.output_h5ad}; use --force to atomically replace it"
        )


def read_csv_column(path: Path, column: str) -> list[str]:
    frame = pd.read_csv(path)
    if list(frame.columns) != [column]:
        raise ValueError(f"{path}: expected only {column!r}, found {list(frame.columns)!r}")
    values = frame[column].astype(str).tolist()
    if len(values) != len(set(values)):
        raise ValueError(f"{path}: duplicate values")
    return values


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def stratified_indices(obs: pd.DataFrame, n_cells: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = obs.groupby("ntc_id", observed=True, sort=True).indices
    keys = sorted(groups)
    base, remainder = divmod(n_cells, len(keys))
    chosen: list[int] = []
    for position, key in enumerate(keys):
        count = base + int(position < remainder)
        candidates = np.asarray(groups[key], dtype=np.int64)
        chosen.extend(rng.choice(candidates, size=count, replace=False).tolist())
    return np.sort(np.asarray(chosen, dtype=np.int64))


def control_statistics(
    path: Path,
    context: str,
    genes: list[str],
    cells_per_group: int,
    chunk_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        required_obs = {"target_gene", "context", "ntc_id"}
        if adata.shape != (18_400, len(genes)):
            raise ValueError(f"{path}: unexpected shape {adata.shape}")
        if not required_obs.issubset(adata.obs.columns):
            raise ValueError(f"{path}: missing required obs columns")
        if set(adata.obs["context"].astype(str)) != {context}:
            raise ValueError(f"{path}: context label does not equal {context!r}")
        if set(adata.obs["target_gene"].astype(str)) != {"non-targeting"}:
            raise ValueError(f"{path}: target_gene contains non-control labels")
        if adata.obs["ntc_id"].nunique() != 46:
            raise ValueError(f"{path}: expected exactly 46 NTC IDs")
        if list(adata.var_names.astype(str)) != genes:
            raise ValueError(f"{path}: gene axis mismatch")
        row_sums = np.empty(adata.n_obs, dtype=np.int64)
        gene_sums = np.zeros(adata.n_vars, dtype=np.float64)
        total_nnz = 0
        for start in range(0, adata.n_obs, chunk_size):
            stop = min(start + chunk_size, adata.n_obs)
            block = adata.X[start:stop]
            if not sparse.issparse(block):
                block = sparse.csr_matrix(block)
            block = block.tocsr(copy=False)
            if not np.isfinite(block.data).all() or np.any(block.data < 0):
                raise ValueError(f"{path}: invalid stored counts in rows {start}:{stop}")
            if not np.equal(block.data, np.floor(block.data)).all():
                raise ValueError(f"{path}: fractional stored counts in rows {start}:{stop}")
            row_sums[start:stop] = np.rint(
                np.asarray(block.sum(axis=1)).ravel()
            ).astype(np.int64)
            gene_sums += np.asarray(block.sum(axis=0)).ravel()
            total_nnz += int(block.nnz)
        if np.any(row_sums <= 0):
            raise ValueError(f"{path}: one or more control cells have zero library size")
        selected = stratified_indices(adata.obs, cells_per_group, seed)
        libraries = row_sums[selected]
        stats = {
            "control_cells": int(adata.n_obs),
            "control_nnz": total_nnz,
            "pseudobulk_total": int(round(gene_sums.sum())),
            "selected_library_min": int(libraries.min()),
            "selected_library_median": float(np.median(libraries)),
            "selected_library_max": int(libraries.max()),
            "selected_rows": selected.tolist(),
        }
        return gene_sums, libraries, stats
    finally:
        adata.file.close()


def sample_base_matrix(
    gene_sums: np.ndarray,
    libraries: np.ndarray,
    genes_per_cell: int,
    device: torch.device,
    seed: int,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    if np.count_nonzero(gene_sums) < genes_per_cell:
        raise ValueError("not enough expressed genes for sampling without replacement")
    probabilities = torch.as_tensor(gene_sums, dtype=torch.float64, device=device)
    probabilities /= probabilities.sum()
    library_tensor = torch.as_tensor(libraries, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)

    started = time.perf_counter()
    selected = torch.multinomial(
        probabilities.expand(len(libraries), -1),
        num_samples=genes_per_cell,
        replacement=False,
        generator=generator,
    )
    weights = probabilities[selected]
    expected = weights / weights.sum(dim=1, keepdim=True) * library_tensor[:, None]
    counts = torch.floor(expected).to(torch.int64)
    fractional = expected - counts
    remainder = library_tensor.to(torch.int64) - counts.sum(dim=1)
    for row in range(len(libraries)):
        n_add = int(remainder[row].item())
        if n_add:
            positions = torch.topk(fractional[row], k=n_add, sorted=False).indices
            counts[row, positions] += 1

    selected, order = torch.sort(selected, dim=1)
    counts = torch.gather(counts, 1, order)
    torch.cuda.synchronize(device)
    sampling_seconds = time.perf_counter() - started

    selected_np = selected.cpu().numpy().astype(np.int32, copy=False)
    counts_np = counts.cpu().numpy().astype(np.int32, copy=False)
    rows = np.repeat(np.arange(len(libraries), dtype=np.int32), genes_per_cell)
    matrix = sparse.csr_matrix(
        (counts_np.ravel(), (rows, selected_np.ravel())),
        shape=(len(libraries), len(gene_sums)),
        dtype=np.int32,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    actual_libraries = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)
    if not np.array_equal(actual_libraries, libraries):
        raise RuntimeError("largest-remainder allocation did not preserve library sizes")
    return matrix, {
        "sampling_seconds": sampling_seconds,
        "nnz": int(matrix.nnz),
        "mean_nnz_per_cell": float(matrix.getnnz(axis=1).mean()),
        "library_sizes_preserved": True,
    }


def validate_full(
    prediction: ad.AnnData,
    genes: list[str],
    targets: list[str],
    cells_per_group: int,
) -> dict[str, Any]:
    matrix = prediction.X.tocsr(copy=False)
    values = matrix.data
    group_sizes = prediction.obs.groupby(
        ["context", "target_gene"], observed=True, sort=True
    ).size()
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    expected_shape = (len(CONTEXTS) * len(targets) * cells_per_group, len(genes))
    context_target_sets = {
        context: set(
            prediction.obs.loc[
                prediction.obs["context"].astype(str) == context, "target_gene"
            ].astype(str)
        )
        for context in CONTEXTS
    }
    expected_groups = pd.MultiIndex.from_product(
        [CONTEXTS, targets], names=["context", "target_gene"]
    )
    integer_values = bool(np.issubdtype(values.dtype, np.integer)) or bool(
        np.equal(values, np.floor(values)).all()
    )
    checks = {
        "shape_exact": prediction.shape == expected_shape,
        "csr": sparse.isspmatrix_csr(matrix),
        "canonical_csr": bool(matrix.has_canonical_format),
        "nnz_under_cap": int(matrix.nnz) <= MAX_NNZ,
        "gene_axis_exact": list(prediction.var_names.astype(str)) == genes,
        "var_names_unique": bool(prediction.var_names.is_unique),
        "obs_names_unique": bool(prediction.obs_names.is_unique),
        "contexts_exact": sorted(prediction.obs["context"].astype(str).unique())
        == list(CONTEXTS),
        "no_controls": "non-targeting"
        not in set(prediction.obs["target_gene"].astype(str)),
        "all_targets_in_each_context": all(
            target_set == set(targets) for target_set in context_target_sets.values()
        ),
        "groups_exact": list(group_sizes.index.to_list()) == list(expected_groups.to_list()),
        "group_sizes_sum_to_n_obs": int(group_sizes.sum()) == prediction.n_obs,
        "cells_per_group_exact": bool((group_sizes == cells_per_group).all()),
        "finite": bool(np.isfinite(values).all()),
        "nonnegative": bool((values >= 0).all()),
        "integer_values": integer_values,
        "no_explicit_zeros": not bool(np.any(values == 0)),
        "max_cell_total_le_1m": bool(row_sums.max() <= 1_000_000),
        "no_zero_library_cells": not bool(np.any(row_sums == 0)),
    }
    return {
        "shape": list(prediction.shape),
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        "groups": int(len(group_sizes)),
        "library_min": int(row_sums.min()),
        "library_median": float(np.median(row_sums)),
        "library_max": int(row_sums.max()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.genes_per_cell < 1:
        raise ValueError("--genes-per-cell must be >= 1")
    reject_path_collisions(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run inside the requested H100 allocation")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if "H100" not in device_name.upper():
        raise RuntimeError(f"expected H100, found {device_name!r}")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError(
            f"expected H100 compute capability (9, 0), found {torch.cuda.get_device_capability(device)}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the allocated CUDA device does not support BF16")

    genes = read_csv_column(args.data_dir / "gene_names.csv", "gene_name")
    targets = read_csv_column(args.data_dir / "pert_counts.csv", "target_gene")
    if len(genes) != 18_533 or len(targets) != 300:
        raise ValueError(f"unexpected panel: {len(genes)} genes, {len(targets)} targets")
    if args.cells_per_group != 400:
        raise ValueError("the official 2026 contract requires 400 cells/group")

    build_started = time.perf_counter()
    context_matrices: list[sparse.csr_matrix] = []
    context_stats: dict[str, Any] = {}
    for context_position, context in enumerate(CONTEXTS):
        gene_sums, libraries, control_stats = control_statistics(
            args.data_dir / f"context_{context}.h5ad",
            context,
            genes,
            args.cells_per_group,
            args.chunk_size,
            args.seed + context_position,
        )
        base, sampling_stats = sample_base_matrix(
            gene_sums,
            libraries,
            args.genes_per_cell,
            device,
            args.seed + 10_000 + context_position,
        )
        repeated = sparse.vstack([base] * len(targets), format="csr")
        repeated.sum_duplicates()
        repeated.eliminate_zeros()
        repeated.sort_indices()
        context_matrices.append(repeated)
        context_stats[context] = {
            **control_stats,
            **sampling_stats,
            "repeated_shape": list(repeated.shape),
            "repeated_nnz": int(repeated.nnz),
        }

    full_matrix = sparse.vstack(context_matrices, format="csr")
    del context_matrices
    full_matrix.sum_duplicates()
    full_matrix.eliminate_zeros()
    full_matrix.sort_indices()
    observations: list[pd.DataFrame] = []
    for context in CONTEXTS:
        target_values = np.repeat(np.asarray(targets, dtype=object), args.cells_per_group)
        cell_numbers = np.tile(np.arange(args.cells_per_group), len(targets))
        observations.append(
            pd.DataFrame(
                {
                    "target_gene": pd.Categorical(target_values, categories=targets),
                    "context": pd.Categorical(
                        [context] * len(target_values), categories=list(CONTEXTS)
                    ),
                },
                index=[
                    f"null_{context}_{target}_{cell:03d}"
                    for target, cell in zip(target_values, cell_numbers, strict=True)
                ],
            )
        )
    prediction = ad.AnnData(
        X=full_matrix,
        obs=pd.concat(observations),
        var=pd.DataFrame(index=pd.Index(genes, name="gene_name")),
    )
    prediction.uns["warning"] = (
        "Full schema smoke artifact with no target-specific perturbation response. "
        "Do not submit."
    )
    prediction.uns["method"] = (
        "For each context, sample 400 sparse cells from the control pseudobulk and "
        "empirical library-size distribution, then repeat that null population for "
        "all 300 targets."
    )
    prediction.uns["seed"] = args.seed
    validation = validate_full(prediction, genes, targets, args.cells_per_group)
    if not validation["passed"]:
        raise RuntimeError(f"in-memory validation failed: {validation}")

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    partial_h5ad = args.output_h5ad.with_name(args.output_h5ad.name + ".partial")
    if partial_h5ad.exists():
        partial_h5ad.unlink()
    write_started = time.perf_counter()
    prediction.write_h5ad(partial_h5ad, compression="lzf")
    write_seconds = time.perf_counter() - write_started

    reopened = ad.read_h5ad(partial_h5ad, backed="r")
    try:
        reopened_groups = reopened.obs.groupby(
            ["context", "target_gene"], observed=True, sort=True
        ).size()
        expected_groups = pd.MultiIndex.from_product(
            [CONTEXTS, targets], names=["context", "target_gene"]
        )
        reopened_checks = {
            "shape_exact": reopened.shape == prediction.shape,
            "gene_axis_exact": list(reopened.var_names.astype(str)) == genes,
            "obs_names_unique": bool(reopened.obs_names.is_unique),
            "contexts_exact": sorted(reopened.obs["context"].astype(str).unique())
            == list(CONTEXTS),
            "group_index_exact": list(reopened_groups.index.to_list())
            == list(expected_groups.to_list()),
            "group_sizes_exact": bool((reopened_groups == args.cells_per_group).all()),
            "group_sizes_sum_to_n_obs": int(reopened_groups.sum()) == reopened.n_obs,
        }
    finally:
        reopened.file.close()
    if not all(reopened_checks.values()):
        raise RuntimeError(f"reopened-file validation failed: {reopened_checks}")
    os.replace(partial_h5ad, args.output_h5ad)

    properties = torch.cuda.get_device_properties(device)
    result = {
        "status": "INTERNAL_PASS_PENDING_OFFICIAL_VCC_PREP",
        "purpose": "complete H100-backed context-null schema smoke test",
        "biological_performance_claim": False,
        "do_not_submit": True,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "account": os.environ.get("SLURM_JOB_ACCOUNT"),
        },
        "gpu": {
            "name": device_name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(properties.total_memory),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "anndata": ad.__version__,
            "numpy": np.__version__,
        },
        "parameters": {
            "seed": args.seed,
            "cells_per_group": args.cells_per_group,
            "genes_per_cell_requested": args.genes_per_cell,
            "targets": len(targets),
            "contexts": list(CONTEXTS),
        },
        "context_statistics": context_stats,
        "validation": validation,
        "reopened_checks": reopened_checks,
        "output": {
            "path": str(args.output_h5ad),
            "bytes": args.output_h5ad.stat().st_size,
            "sha256": sha256_file(args.output_h5ad),
            "write_seconds": write_seconds,
        },
        "total_build_seconds": time.perf_counter() - build_started,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
