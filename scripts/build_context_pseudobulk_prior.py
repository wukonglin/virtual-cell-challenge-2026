#!/usr/bin/env python3
"""Convert a learned log-profile effect prior into context-specific count means.

The transformation starts from each released control context, applies the
learned effect in the same log1p(CP50K) space used to build the source atlas,
and maps the result back to the context's observed mean library depth.  The
result is model-derived; no measured perturbed profile is inserted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


SCHEMA = "vcc-context-pseudobulk-prior-v1"
CONTEXTS = ("A", "B", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("effect_prior", type=Path)
    parser.add_argument("controls_dir", type=Path)
    parser.add_argument("targets_csv", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--target-column", default="target_gene")
    parser.add_argument("--target-sum", type=float, default=50_000.0)
    parser.add_argument("--chunk-rows", type=int, default=4_096)
    parser.add_argument("--effect-clip", type=float, default=1.0)
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


def string_array(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in np.asarray(values).reshape(-1)
    ]


def control_mean(path: Path, chunk_rows: int) -> tuple[list[str], np.ndarray, int]:
    data = ad.read_h5ad(path, backed="r")
    try:
        genes = data.var_names.astype(str).tolist()
        total = np.zeros(data.n_vars, dtype=np.float64)
        for begin in range(0, data.n_obs, chunk_rows):
            end = min(begin + chunk_rows, data.n_obs)
            values = data.X[begin:end]
            if sp.issparse(values):
                total += np.asarray(values.sum(axis=0)).reshape(-1)
            else:
                array = np.asarray(values)
                require(np.isfinite(array).all() and np.min(array) >= 0, "Invalid controls")
                total += array.sum(axis=0, dtype=np.float64)
        require(total.sum() > 0, f"Control matrix has zero total: {path}")
        return genes, total / data.n_obs, data.n_obs
    finally:
        data.file.close()


def main() -> None:
    args = parse_args()
    started = time.time()
    for path in (args.effect_prior, args.targets_csv):
        require(path.is_file(), f"Missing input: {path}")
    require(args.controls_dir.is_dir(), f"Missing controls directory: {args.controls_dir}")
    require(args.target_sum > 0 and args.effect_clip > 0, "Invalid scale")
    require(args.chunk_rows > 0, "chunk-rows must be positive")
    for output in (args.output_npz, args.output_json):
        require(not output.exists(), f"Refusing to overwrite {output}")

    targets_table = pd.read_csv(args.targets_csv)
    require(args.target_column in targets_table, f"Missing column {args.target_column}")
    targets = targets_table[args.target_column].astype(str).tolist()
    require(len(targets) == len(set(targets)), "Target axis has duplicates")

    with np.load(args.effect_prior, allow_pickle=False) as archive:
        require("effects" in archive.files, "Effect prior lacks effects")
        prior_targets_key = "targets" if "targets" in archive.files else "target_names"
        prior_genes_key = "genes" if "genes" in archive.files else "gene_names"
        require(prior_targets_key in archive.files, "Effect prior lacks targets")
        require(prior_genes_key in archive.files, "Effect prior lacks genes")
        prior_targets = string_array(archive[prior_targets_key])
        prior_genes = string_array(archive[prior_genes_key])
        prior_effects = np.asarray(archive["effects"], dtype=np.float32)
    require(
        prior_effects.shape == (len(prior_targets), len(prior_genes)),
        "Effect prior shape does not match its axes",
    )
    require(np.isfinite(prior_effects).all(), "Effect prior is non-finite")
    target_lookup = {name: index for index, name in enumerate(prior_targets)}
    missing_targets = sorted(set(targets) - set(target_lookup))
    require(not missing_targets, f"Effect prior misses targets: {missing_targets[:8]}")

    context_means: list[np.ndarray] = []
    context_cells: dict[str, int] = {}
    challenge_genes: list[str] | None = None
    for context in CONTEXTS:
        path = args.controls_dir / f"context_{context}.h5ad"
        require(path.is_file(), f"Missing context control: {path}")
        genes, mean, cells = control_mean(path, args.chunk_rows)
        if challenge_genes is None:
            challenge_genes = genes
        else:
            require(genes == challenge_genes, "Control gene axes differ across contexts")
        context_means.append(mean)
        context_cells[context] = cells
    assert challenge_genes is not None

    challenge_gene_lookup = {name: index for index, name in enumerate(challenge_genes)}
    unknown_genes = sorted(set(prior_genes) - set(challenge_gene_lookup))
    require(not unknown_genes, f"Prior genes are absent from challenge axis: {unknown_genes[:8]}")
    gene_positions = np.asarray(
        [challenge_gene_lookup[name] for name in prior_genes], dtype=np.int64
    )
    target_positions = np.asarray([target_lookup[name] for name in targets], dtype=np.int64)
    aligned_effects = np.zeros((len(targets), len(challenge_genes)), dtype=np.float32)
    aligned_effects[:, gene_positions] = prior_effects[target_positions]
    np.clip(aligned_effects, -args.effect_clip, args.effect_clip, out=aligned_effects)

    pseudobulk = np.zeros(
        (len(CONTEXTS), len(targets), len(challenge_genes)), dtype=np.float32
    )
    total_log_fold = np.zeros_like(pseudobulk)
    depth_summary: dict[str, dict[str, float]] = {}
    for context_index, (context, mean) in enumerate(zip(CONTEXTS, context_means, strict=True)):
        mean_depth = float(mean.sum())
        control_cp = args.target_sum * mean / mean_depth
        target_cp = np.expm1(np.log1p(control_cp)[None, :] + aligned_effects)
        np.maximum(target_cp, 0.0, out=target_cp)
        target_cp_sums = target_cp.sum(axis=1, keepdims=True)
        require(np.all(target_cp_sums > 0), f"Zero pseudobulk total for context {context}")
        means = mean_depth * target_cp / target_cp_sums
        pseudobulk[context_index] = means.astype(np.float32)
        ratio = (means + 1e-8) / (mean[None, :] + 1e-8)
        total_log_fold[context_index] = np.clip(
            np.log(ratio), -args.effect_clip, args.effect_clip
        ).astype(np.float32)
        depths = means.sum(axis=1)
        depth_summary[context] = {
            "control_mean_depth": mean_depth,
            "predicted_min_depth": float(depths.min()),
            "predicted_median_depth": float(np.median(depths)),
            "predicted_max_depth": float(depths.max()),
        }

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    common_log_fold = total_log_fold.mean(axis=1, dtype=np.float64).astype(np.float32)
    learned_log_fold = total_log_fold - common_log_fold[:, None, :]
    np.savez_compressed(
        args.output_npz,
        genes=np.asarray(challenge_genes, dtype="U"),
        targets=np.asarray(targets, dtype="U"),
        contexts=np.asarray(CONTEXTS, dtype="U"),
        pseudobulk_mean=pseudobulk,
        common_log_fold=common_log_fold,
        learned_log_fold=learned_log_fold,
    )
    report = {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "inputs": {
            "effect_prior": {
                "path": str(args.effect_prior.resolve()),
                "sha256": sha256_file(args.effect_prior),
            },
            "targets": {
                "path": str(args.targets_csv.resolve()),
                "sha256": sha256_file(args.targets_csv),
            },
            "controls": {
                context: {
                    "path": str((args.controls_dir / f"context_{context}.h5ad").resolve()),
                    "sha256": sha256_file(args.controls_dir / f"context_{context}.h5ad"),
                    "cells": context_cells[context],
                }
                for context in CONTEXTS
            },
        },
        "output": {
            "path": str(args.output_npz.resolve()),
            "sha256": sha256_file(args.output_npz),
            "shape": list(pseudobulk.shape),
        },
        "contract": {
            "effect_space": "log1p-group-sum-cp50000-delta",
            "output_space": "expected-raw-counts-per-cell",
            "response_space": "context-specific-raw-count-log-fold",
            "model_derived": True,
            "measured_perturbed_profiles_inserted": False,
            "target_sum": args.target_sum,
            "effect_clip": args.effect_clip,
        },
        "depth": depth_summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
