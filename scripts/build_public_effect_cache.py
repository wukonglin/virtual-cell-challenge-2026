#!/usr/bin/env python3
"""Build batch-matched public Perturb-seq effect summaries for VCC 2026.

The Arc/STATE support matrices are already normalized continuous expression,
whereas the 2026 controls are raw counts.  This script never mixes those two
measurement scales.  It summarizes each public perturbation as a batch-matched
mean shift on the support gene axis and separately records current-context
control means in both log1p(CP10K) and raw-count space.

No challenge labels are used: the 2026 inputs contain non-targeting controls
only.  The output cache is an intermediate model-training artifact, not a VCC
submission.
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


PUBLIC_FILES = (
    "competition_train.h5",
    "k562.h5",
    "k562_gwps.h5",
    "rpe1.h5",
    "jurkat.h5",
    "hepg2.h5",
)
CONTEXTS = ("A", "B", "C")
CONTROL_LABEL = "non-targeting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--support-dir",
        type=Path,
        default=Path("dataset/state_support/extracted"),
    )
    parser.add_argument(
        "--controls-dir", type=Path, default=Path("dataset/controls")
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("artifacts/bayes_public_effects_v1.npz"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/bayes_public_effects_v1.json"),
    )
    parser.add_argument("--block-rows", type=int, default=512)
    parser.add_argument(
        "--batch-control-prior",
        type=float,
        default=100.0,
        help="global-control pseudo-cells used to shrink each batch control mean",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def grouped_sum(
    matrix: sparse.spmatrix | np.ndarray,
    codes: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Return group-by-feature sums without densifying the row dimension."""
    if matrix.shape[0] != len(codes):
        raise ValueError("matrix/codes row mismatch")
    rows = np.arange(len(codes), dtype=np.int32)
    selector = sparse.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (codes, rows)),
        shape=(n_groups, len(codes)),
    )
    result = selector @ matrix
    if sparse.issparse(result):
        result = result.toarray()
    return np.asarray(result, dtype=np.float64)


def renormalize_log_expression(
    matrix: sparse.spmatrix | np.ndarray,
) -> sparse.csr_matrix | np.ndarray:
    """Convert a log1p expression block to a common log1p(CP10K) scale.

    The Arc H1 support matrix is log1p of integer counts, while the Replogle
    matrices contain log1p of continuously normalized values.  In both cases
    expm1 recovers a non-negative abundance matrix.  Re-normalizing every row
    to 10,000 makes within-source deltas comparable without pretending that
    either input is submission-ready raw counts.
    """
    if sparse.issparse(matrix):
        result = matrix.tocsr(copy=True).astype(np.float32)
        np.expm1(result.data, out=result.data)
        totals = np.asarray(result.sum(axis=1)).ravel()
        if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
            raise ValueError("invalid recovered library size")
        result.data *= np.repeat(10_000.0 / totals, np.diff(result.indptr))
        np.log1p(result.data, out=result.data)
        return result
    result = np.asarray(matrix, dtype=np.float32).copy()
    np.expm1(result, out=result)
    totals = result.sum(axis=1, dtype=np.float64)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        raise ValueError("invalid recovered library size")
    result *= (10_000.0 / totals)[:, None]
    np.log1p(result, out=result)
    return result


def summarize_public_dataset(
    path: Path,
    expected_genes: list[str],
    block_rows: int,
    batch_control_prior: float,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    adata = ad.read_h5ad(path, backed="r")
    try:
        required = {"target_gene", "batch_var", "cell_type"}
        if not required.issubset(adata.obs.columns):
            raise ValueError(f"{path}: missing obs columns {sorted(required - set(adata.obs))}")
        genes = list(adata.var_names.astype(str))
        if genes != expected_genes:
            raise ValueError(f"{path}: support gene axis mismatch")

        obs = adata.obs[["target_gene", "batch_var", "cell_type"]].copy()
        obs["target_gene"] = obs["target_gene"].astype(str)
        obs["batch_var"] = obs["batch_var"].astype(str)
        targets = sorted(obs["target_gene"].unique())
        batches = sorted(obs["batch_var"].unique())
        if CONTROL_LABEL not in targets:
            raise ValueError(f"{path}: missing {CONTROL_LABEL!r} controls")

        target_index = {name: i for i, name in enumerate(targets)}
        batch_index = {name: i for i, name in enumerate(batches)}
        target_codes = obs["target_gene"].map(target_index).to_numpy(np.int32)
        batch_codes = obs["batch_var"].map(batch_index).to_numpy(np.int32)
        control_mask = obs["target_gene"].to_numpy() == CONTROL_LABEL

        target_counts = np.bincount(target_codes, minlength=len(targets)).astype(np.int64)
        target_sums = np.zeros((len(targets), adata.n_vars), dtype=np.float64)
        control_batch_sums = np.zeros((len(batches), adata.n_vars), dtype=np.float64)
        control_batch_counts = np.bincount(
            batch_codes[control_mask], minlength=len(batches)
        ).astype(np.int64)

        for start in range(0, adata.n_obs, block_rows):
            stop = min(start + block_rows, adata.n_obs)
            block = adata.X[start:stop]
            if sparse.issparse(block):
                block = block.tocsr(copy=False)
            else:
                block = np.asarray(block)
            if sparse.issparse(block):
                values = block.data
            else:
                values = block
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{path}: invalid expression values in rows {start}:{stop}")

            block = renormalize_log_expression(block)

            target_sums += grouped_sum(
                block, target_codes[start:stop], len(targets)
            )
            local_control = control_mask[start:stop]
            if np.any(local_control):
                control_block = block[local_control]
                control_batch_sums += grouped_sum(
                    control_block,
                    batch_codes[start:stop][local_control],
                    len(batches),
                )

        control_idx = target_index[CONTROL_LABEL]
        global_control_mean = target_sums[control_idx] / target_counts[control_idx]
        control_batch_means = np.repeat(
            global_control_mean[None, :], len(batches), axis=0
        )
        valid_batches = control_batch_counts > 0
        control_batch_means[valid_batches] = (
            control_batch_sums[valid_batches]
            + batch_control_prior * global_control_mean[None, :]
        ) / (
            control_batch_counts[valid_batches, None] + batch_control_prior
        )

        count_table = pd.crosstab(obs["target_gene"], obs["batch_var"]).reindex(
            index=targets, columns=batches, fill_value=0
        )
        target_batch_counts = count_table.to_numpy(dtype=np.float64)
        target_means = target_sums / target_counts[:, None]
        matched_control_means = (
            target_batch_counts @ control_batch_means
        ) / target_counts[:, None]
        effects = target_means - matched_control_means

        keep = [i for i, target in enumerate(targets) if target != CONTROL_LABEL]
        pert_targets = [targets[i] for i in keep]
        pert_counts = target_counts[keep]
        pert_effects = effects[keep].astype(np.float32)
        if not np.isfinite(pert_effects).all():
            raise ValueError(f"{path}: non-finite batch-matched effects")

        metadata = {
            "file": path.name,
            "shape": [int(adata.n_obs), int(adata.n_vars)],
            "cell_types": sorted(obs["cell_type"].astype(str).unique()),
            "n_batches": len(batches),
            "n_controls": int(target_counts[control_idx]),
            "n_perturbations": len(pert_targets),
            "perturbation_cells": {
                "min": int(pert_counts.min()),
                "median": float(np.median(pert_counts)),
                "max": int(pert_counts.max()),
            },
            "effect_abs_quantiles": {
                str(q): float(np.quantile(np.abs(pert_effects), q))
                for q in (0.5, 0.9, 0.99, 0.999)
            },
            "effect_scale": "mean log1p(CP10K), batch-matched",
            "batch_control_prior": batch_control_prior,
            "seconds": time.perf_counter() - started,
        }
        return (
            pert_effects,
            pert_targets,
            pert_counts,
            global_control_mean.astype(np.float32),
            metadata,
        )
    finally:
        adata.file.close()


def summarize_current_control(
    path: Path,
    context: str,
    expected_genes: list[str],
    block_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    adata = ad.read_h5ad(path, backed="r")
    try:
        if list(adata.var_names.astype(str)) != expected_genes:
            raise ValueError(f"{path}: current gene axis mismatch")
        if set(adata.obs["context"].astype(str)) != {context}:
            raise ValueError(f"{path}: unexpected context labels")
        if set(adata.obs["target_gene"].astype(str)) != {CONTROL_LABEL}:
            raise ValueError(f"{path}: contains non-control cells")

        raw_sum = np.zeros(adata.n_vars, dtype=np.float64)
        log_sum = np.zeros(adata.n_vars, dtype=np.float64)
        library_sizes = np.empty(adata.n_obs, dtype=np.int64)
        nnz = 0
        for start in range(0, adata.n_obs, block_rows):
            stop = min(start + block_rows, adata.n_obs)
            block = adata.X[start:stop]
            if not sparse.issparse(block):
                block = sparse.csr_matrix(block)
            block = block.tocsr(copy=True)
            if not np.isfinite(block.data).all() or np.any(block.data < 0):
                raise ValueError(f"{path}: invalid raw counts")
            if not np.equal(block.data, np.floor(block.data)).all():
                raise ValueError(f"{path}: fractional raw counts")
            libs = np.asarray(block.sum(axis=1)).ravel()
            if np.any(libs <= 0):
                raise ValueError(f"{path}: zero-library control cell")
            library_sizes[start:stop] = np.rint(libs).astype(np.int64)
            raw_sum += np.asarray(block.sum(axis=0)).ravel()
            block.data = block.data.astype(np.float64, copy=False)
            block.data *= np.repeat(10_000.0 / libs, np.diff(block.indptr))
            np.log1p(block.data, out=block.data)
            log_sum += np.asarray(block.sum(axis=0)).ravel()
            nnz += int(block.nnz)

        metadata = {
            "context": context,
            "shape": [int(adata.n_obs), int(adata.n_vars)],
            "nnz": nnz,
            "mean_nnz_per_cell": nnz / adata.n_obs,
            "library_size": {
                "min": int(library_sizes.min()),
                "median": float(np.median(library_sizes)),
                "max": int(library_sizes.max()),
            },
            "seconds": time.perf_counter() - started,
        }
        return (
            (log_sum / adata.n_obs).astype(np.float32),
            (raw_sum / adata.n_obs).astype(np.float32),
            metadata,
        )
    finally:
        adata.file.close()


def main() -> None:
    args = parse_args()
    if args.block_rows <= 0:
        raise ValueError("--block-rows must be positive")
    if args.batch_control_prior < 0:
        raise ValueError("--batch-control-prior must be non-negative")
    for output in (args.output_npz, args.output_json):
        if output.exists() and not args.force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    support_genes = pd.read_csv(
        args.support_dir / "gene_names.csv", header=None
    ).iloc[:, 0].astype(str).tolist()
    current_genes = pd.read_csv(args.controls_dir / "gene_names.csv")[
        "gene_name"
    ].astype(str).tolist()
    if len(support_genes) != 18_080 or len(current_genes) != 18_533:
        raise ValueError("unexpected official gene-axis length")
    if len(set(support_genes)) != len(support_genes):
        raise ValueError("duplicate support genes")
    if len(set(current_genes)) != len(current_genes):
        raise ValueError("duplicate current genes")

    all_effects: list[np.ndarray] = []
    all_targets: list[str] = []
    all_sources: list[str] = []
    all_counts: list[np.ndarray] = []
    source_control_means: list[np.ndarray] = []
    public_metadata: list[dict[str, Any]] = []

    for filename in PUBLIC_FILES:
        path = args.support_dir / filename
        effects, targets, counts, control_mean, metadata = summarize_public_dataset(
            path, support_genes, args.block_rows, args.batch_control_prior
        )
        all_effects.append(effects)
        all_targets.extend(targets)
        all_sources.extend([filename.removesuffix(".h5")] * len(targets))
        all_counts.append(counts)
        source_control_means.append(control_mean)
        public_metadata.append(metadata)
        print(json.dumps(metadata, sort_keys=True), flush=True)

    current_log_means: list[np.ndarray] = []
    current_raw_means: list[np.ndarray] = []
    current_metadata: list[dict[str, Any]] = []
    for context in CONTEXTS:
        log_mean, raw_mean, metadata = summarize_current_control(
            args.controls_dir / f"context_{context}.h5ad",
            context,
            current_genes,
            args.block_rows,
        )
        current_log_means.append(log_mean)
        current_raw_means.append(raw_mean)
        current_metadata.append(metadata)
        print(json.dumps(metadata, sort_keys=True), flush=True)

    current_index = {gene: i for i, gene in enumerate(current_genes)}
    support_to_current = np.asarray(
        [current_index.get(gene, -1) for gene in support_genes], dtype=np.int32
    )
    if int(np.sum(support_to_current >= 0)) != 18_077:
        raise ValueError("unexpected support/current gene overlap")

    arrays = {
        "effects": np.vstack(all_effects).astype(np.float32, copy=False),
        "sample_targets": np.asarray(all_targets, dtype="U64"),
        "sample_sources": np.asarray(all_sources, dtype="U32"),
        "sample_n_cells": np.concatenate(all_counts).astype(np.int32),
        "source_names": np.asarray(
            [name.removesuffix(".h5") for name in PUBLIC_FILES], dtype="U32"
        ),
        "source_control_log_means": np.vstack(source_control_means).astype(np.float32),
        "support_genes": np.asarray(support_genes, dtype="U64"),
        "current_genes": np.asarray(current_genes, dtype="U64"),
        "support_to_current": support_to_current,
        "current_contexts": np.asarray(CONTEXTS, dtype="U1"),
        "current_control_log_means": np.vstack(current_log_means).astype(np.float32),
        "current_control_raw_means": np.vstack(current_raw_means).astype(np.float32),
    }
    temporary_npz = args.output_npz.with_suffix(args.output_npz.suffix + ".tmp.npz")
    np.savez_compressed(temporary_npz, **arrays)
    os.replace(temporary_npz, args.output_npz)

    challenge_targets = pd.read_csv(args.controls_dir / "pert_counts.csv")[
        "target_gene"
    ].astype(str).tolist()
    direct_overlap = sorted(set(all_targets).intersection(challenge_targets))
    report = {
        "artifact_type": "public_perturbation_effect_cache",
        "created_at_unix": time.time(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "support_gene_count": len(support_genes),
        "current_gene_count": len(current_genes),
        "gene_overlap": int(np.sum(support_to_current >= 0)),
        "public_effect_samples": len(all_targets),
        "public_unique_targets": len(set(all_targets)),
        "challenge_targets": len(challenge_targets),
        "direct_challenge_target_overlap": len(direct_overlap),
        "direct_overlap_targets": direct_overlap,
        "public_datasets": public_metadata,
        "current_contexts": current_metadata,
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
