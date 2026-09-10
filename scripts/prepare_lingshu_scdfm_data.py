#!/usr/bin/env python3
"""Prepare a bounded public-data cache for a Lingshu-conditioned flow model.

The STATE K562/RPE1 support files contain log1p continuous abundances, not raw
UMI counts. We apply expm1, normalize over the COMPLETE shared public/challenge
gene axis, and log1p again before selecting genes. Only the three source genes
absent from the challenge axis are excluded. No raw-count recovery is claimed.
K562 whole targets and the entire RPE1 context are held out before selecting
genes. Only K562 training treated rows determine the gene-variance ranking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse


SCHEMA = "vcc-lingshu-scdfm-public-cache-v1"
CONTROL = "non-targeting"
TARGET_SUM = 10_000.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_axis(path: Path, header: str) -> list[str]:
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if rows and rows[0] == [header]:
        rows = rows[1:]
    if not rows or any(len(row) != 1 or not row[0].strip() for row in rows):
        raise ValueError(f"Expected one nonempty column in {path}")
    values = [row[0].strip() for row in rows]
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate axis entries: {path}")
    return values


def stable_rng(seed: int, *parts: str) -> np.random.Generator:
    token = "|".join([str(seed), *parts]).encode()
    derived = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(derived)


def target_split(targets: list[str], seed: int, fraction: float = 0.2) -> tuple[list[str], list[str]]:
    if len(targets) < 2 or not 0 < fraction < 1:
        raise ValueError("At least two targets and a holdout fraction in (0,1) are required")
    ordered = np.asarray(sorted(set(targets)))
    shuffled = stable_rng(seed, "target-holdout").permutation(ordered)
    count = min(len(ordered) - 1, max(1, int(np.ceil(len(ordered) * fraction))))
    heldout = set(shuffled[:count].tolist())
    return sorted(set(ordered) - heldout), sorted(heldout)


def normalize_source_log1p(block: Any, normalization_columns: np.ndarray | None = None) -> np.ndarray:
    """Normalize on the full shared axis, never on the selected model features."""
    x = block.toarray() if sparse.issparse(block) else np.asarray(block)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("Source log1p expression must be finite, nonnegative, and 2-D")
    with np.errstate(over="ignore", invalid="ignore"):
        abundance = np.expm1(x)
        denominator_values = abundance if normalization_columns is None else abundance[:, normalization_columns]
        totals = denominator_values.sum(axis=1, dtype=np.float64)
    if not np.all(np.isfinite(totals)) or np.any(totals <= 0):
        raise ValueError("Invalid normalization-axis abundance totals")
    result = np.log1p(abundance * (TARGET_SUM / totals[:, None]))
    return result.astype(np.float32)


def _read_rows(adata: ad.AnnData, rows: np.ndarray) -> Any:
    # h5py requires strictly increasing indices; restore duplicates and order.
    unique, inverse = np.unique(rows, return_inverse=True)
    return adata.X[unique, :][inverse]


def sampled_rows(labels: np.ndarray, targets: list[str], maximum: int, seed: int, context: str) -> np.ndarray:
    parts = []
    for target in sorted(targets):
        candidates = np.flatnonzero(labels == target)
        if not len(candidates):
            raise ValueError(f"Target {target} has no cells in {context}")
        rng = stable_rng(seed, "treated", context, target)
        parts.append(np.sort(rng.choice(candidates, min(maximum, len(candidates)), replace=False)))
    return np.concatenate(parts).astype(np.int64)


def paired_controls(
    labels: np.ndarray, batches: np.ndarray, rows: np.ndarray,
    maximum: int, seed: int, context: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    controls = np.flatnonzero(labels == CONTROL)
    if not len(controls):
        raise ValueError(f"No controls in {context}")
    pools: dict[str, np.ndarray] = {}
    for batch in sorted(set(batches[controls])):
        candidates = controls[batches[controls] == batch]
        rng = stable_rng(seed, "control-pool", context, batch)
        pools[batch] = np.sort(rng.choice(candidates, min(maximum, len(candidates)), replace=False))
    fallback = np.concatenate(list(pools.values()))
    selected = []
    unmatched = 0
    # Pairing is randomized conditional sampling, not observed cell pairing.
    for row in rows:
        pool = pools.get(batches[row])
        if pool is None:
            pool = fallback
            unmatched += 1
        rng = stable_rng(seed, "paired-control", context, str(int(row)))
        selected.append(int(rng.choice(pool)))
    return np.asarray(selected, dtype=np.int64), {
        "matching": "gem_group_when_available_else_context",
        "observed_paired_cells": False,
        "control_pool_max_per_batch": maximum,
        "control_pool_unique_cells": int(len(fallback)),
        "unmatched_batch_rows": unmatched,
    }


def select_genes(
    adata: ad.AnnData, rows: np.ndarray, candidates: np.ndarray,
    gene_count: int, block_rows: int,
    normalization_columns: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(candidates) < gene_count:
        raise ValueError(f"Only {len(candidates)} shared official genes for requested {gene_count}")
    sums = np.zeros(len(candidates), dtype=np.float64)
    squares = np.zeros_like(sums)
    for start in range(0, len(rows), block_rows):
        normalized = normalize_source_log1p(_read_rows(adata, rows[start:start + block_rows]), normalization_columns)
        subset = normalized[:, candidates].astype(np.float64)
        sums += subset.sum(axis=0)
        squares += np.square(subset).sum(axis=0)
    variance = np.maximum(0, squares / len(rows) - np.square(sums / len(rows)))
    # Candidates already follow the official gene axis, which also breaks ties.
    chosen = np.argsort(-variance, kind="stable")[:gene_count]
    chosen.sort()
    return chosen, variance[chosen]


def read_selected(
    adata: ad.AnnData, rows: np.ndarray, columns: np.ndarray, block_rows: int,
    normalization_columns: np.ndarray | None = None,
) -> np.ndarray:
    output = np.empty((len(rows), len(columns)), dtype=np.float32)
    for start in range(0, len(rows), block_rows):
        end = min(start + block_rows, len(rows))
        normalized = normalize_source_log1p(_read_rows(adata, rows[start:end]), normalization_columns)
        output[start:end] = normalized[:, columns]
    return output


def describe_source(path: Path, adata: ad.AnnData) -> dict[str, Any]:
    return {
        "path": str(path.resolve()), "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path), "shape": list(adata.shape),
        "input_space": "log1p_continuous_normalized_abundance",
        "raw_counts_reconstructed": False,
        "normalization_evidence": "STATE support encoding; project build_public_effect_cache.renormalize_log_expression",
    }


def _validate_source(adata: ad.AnnData, expected_context: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    for column in ("target_gene", "cell_type"):
        if column not in adata.obs:
            raise ValueError(f"Missing {column} in {expected_context} source")
        if adata.obs[column].isna().any():
            raise ValueError(f"Missing values in source {column}")
    contexts = set(adata.obs["cell_type"].astype(str).str.lower())
    if contexts != {expected_context.lower()}:
        raise ValueError(f"Expected only {expected_context} public cells, found {sorted(contexts)}")
    genes = adata.var_names.astype(str).tolist()
    if len(set(genes)) != len(genes):
        raise ValueError("Duplicate source gene symbols")
    labels = adata.obs["target_gene"].astype(str).to_numpy()
    batches = (adata.obs["gem_group"].astype(str).to_numpy() if "gem_group" in adata.obs
               else np.full(adata.n_obs, "all"))
    return labels, batches, genes


def prepare_cache(args: argparse.Namespace) -> dict[str, Any]:
    for output in (args.output_npz, args.output_json, args.output_genes_csv):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
    if args.gene_count < 1 or args.max_cells_per_target < 1 or args.block_rows < 1:
        raise ValueError("Gene count, cell limit, and block size must be positive")
    official_genes = read_axis(args.official_genes, "gene_name")
    official_targets = read_axis(args.official_targets, "target_gene")
    k562 = ad.read_h5ad(args.k562, backed="r")
    rpe1 = ad.read_h5ad(args.rpe1, backed="r")
    try:
        kl, kb, kg = _validate_source(k562, "k562")
        rl, rb, rg = _validate_source(rpe1, "rpe1")
        if kg != rg:
            raise ValueError("K562 and RPE1 normalization gene axes must match exactly")
        k_targets = sorted(set(kl) - {CONTROL})
        r_targets = sorted(set(rl) - {CONTROL})
        train_targets, holdout_targets = target_split(k_targets, args.seed, args.target_holdout_fraction)
        train_rows = sampled_rows(kl, train_targets, args.max_cells_per_target, args.seed, "K562")
        target_rows = sampled_rows(kl, holdout_targets, args.max_cells_per_target, args.seed, "K562")
        context_rows = sampled_rows(rl, r_targets, args.max_cells_per_target, args.seed, "RPE1")
        if set(kl[train_rows]) & set(kl[target_rows]):
            raise ValueError("Whole-target holdout leaked into training")
        k_gene_map = {gene: index for index, gene in enumerate(kg)}
        r_gene_map = {gene: index for index, gene in enumerate(rg)}
        common_official_indices = np.asarray([
            index for index, gene in enumerate(official_genes)
            if gene in k_gene_map and gene in r_gene_map
        ], dtype=np.int64)
        common_genes = [official_genes[index] for index in common_official_indices]
        k_candidates = np.asarray([k_gene_map[gene] for gene in common_genes], dtype=np.int64)
        r_normalization = np.asarray([r_gene_map[gene] for gene in common_genes], dtype=np.int64)
        chosen, variances = select_genes(k562, train_rows, k_candidates, args.gene_count, args.block_rows, k_candidates)
        gene_indices = common_official_indices[chosen]
        gene_names = np.asarray([official_genes[index] for index in gene_indices], dtype=str)
        k_columns = np.asarray([k_gene_map[gene] for gene in gene_names])
        r_columns = np.asarray([r_gene_map[gene] for gene in gene_names])
        train_controls, train_matching = paired_controls(kl, kb, train_rows, args.max_cells_per_target, args.seed, "K562")
        target_controls, target_matching = paired_controls(kl, kb, target_rows, args.max_cells_per_target, args.seed, "K562")
        context_controls, context_matching = paired_controls(rl, rb, context_rows, args.max_cells_per_target, args.seed, "RPE1")
        arrays = {
            "gene_names": gene_names, "gene_indices": gene_indices,
            "normalization_gene_names": np.asarray(common_genes, dtype=str),
            "train_x": read_selected(k562, train_rows, k_columns, args.block_rows, k_candidates),
            "train_control": read_selected(k562, train_controls, k_columns, args.block_rows, k_candidates),
            "train_targets": np.asarray(kl[train_rows], dtype=str),
            "train_contexts": np.full(len(train_rows), "K562"),
            "val_x": np.concatenate([
                read_selected(k562, target_rows, k_columns, args.block_rows, k_candidates),
                read_selected(rpe1, context_rows, r_columns, args.block_rows, r_normalization),
            ]),
            "val_control": np.concatenate([
                read_selected(k562, target_controls, k_columns, args.block_rows, k_candidates),
                read_selected(rpe1, context_controls, r_columns, args.block_rows, r_normalization),
            ]),
            "val_targets": np.asarray(list(kl[target_rows]) + list(rl[context_rows]), dtype=str),
            "val_contexts": np.asarray(["K562"] * len(target_rows) + ["RPE1"] * len(context_rows)),
            "val_kind": np.asarray(["target"] * len(target_rows) + ["context"] * len(context_rows)),
            "train_source_rows": train_rows, "train_control_source_rows": train_controls,
            "val_source_rows": np.concatenate([target_rows, context_rows]),
            "val_control_source_rows": np.concatenate([target_controls, context_controls]),
        }
        metadata = {
            "schema": SCHEMA, "seed": args.seed, "challenge_treated_used": False,
            "official_gene_axis_sha256": sha256_file(args.official_genes),
            "official_target_axis_sha256": sha256_file(args.official_targets),
            "training_gene_selection": True,
            "gene_selection": {
                "method": "descending_variance_on_sampled_K562_train_treated_rows_only",
                "tie_break": "official_gene_order", "output_order": "official_gene_order",
                "count": args.gene_count, "source_intersection_count": len(common_genes),
                "selected_variances": variances.tolist(), "validation_expression_used": False,
            },
            "normalization": {
                "target_sum": TARGET_SUM, "space": "log1p_library_normalized_full_axis",
                "full_axis_scope": "shared_public_challenge_gene_axis",
                "gene_count": len(common_genes),
                "formula": "log1p(10000 * expm1(source_X) / sum_shared_public_challenge_genes(expm1(source_X)))",
                "source_input_space": "log1p_continuous_normalized_abundance",
                "raw_counts_reconstructed": False,
                "unobserved_official_genes_not_imputed": True,
                "source_genes_absent_from_official_axis": sorted(set(kg) - set(official_genes)),
                "official_genes_absent_from_source_axis": sorted(set(official_genes) - set(kg)),
            },
            "split": {
                "training_context": "K562", "heldout_context": "RPE1",
                "target_holdout_fraction": args.target_holdout_fraction,
                "training_targets": train_targets, "heldout_targets": holdout_targets,
                "heldout_targets_globally_excluded_from_training": True,
                "entire_RPE1_context_heldout": True,
                "fit_preprocessing_before_validation": True,
                "shared_control_conditioning_allowed": True,
                "max_cells_per_target": args.max_cells_per_target,
                "train_rows": len(train_rows), "target_validation_rows": len(target_rows),
                "context_validation_rows": len(context_rows),
            },
            "control_matching": {"train": train_matching, "target": target_matching, "context": context_matching},
            "sources": {"K562": describe_source(args.k562, k562), "RPE1": describe_source(args.rpe1, rpe1)},
            "software": {"numpy": np.__version__, "anndata": ad.__version__},
            "script_sha256": sha256_file(Path(__file__)),
            "interpretation": "Public-source supervised model training, zero-shot only with respect to challenge contexts and labels",
        }
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=str)
        for output in (args.output_npz, args.output_json, args.output_genes_csv):
            output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents an accidental restart from replacing a cache.
        with args.output_npz.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        embedding_genes = sorted(set(k_targets) | set(r_targets) | set(official_targets) | {CONTROL})
        with args.output_genes_csv.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["gene_name"])
            writer.writerows((gene,) for gene in embedding_genes)
        receipt = dict(metadata)
        receipt["artifacts"] = {
            "cache": {"path": str(args.output_npz.resolve()), "sha256": sha256_file(args.output_npz)},
            "embedding_genes": {"path": str(args.output_genes_csv.resolve()), "sha256": sha256_file(args.output_genes_csv), "count": len(embedding_genes)},
        }
        with args.output_json.open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return receipt
    finally:
        k562.file.close()
        rpe1.file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k562", type=Path, default=Path("dataset/state_support/extracted/k562_gwps.h5"))
    parser.add_argument("--rpe1", type=Path, default=Path("dataset/state_support/extracted/rpe1.h5"))
    parser.add_argument("--official-genes", type=Path, default=Path("dataset/controls/gene_names.csv"))
    parser.add_argument("--official-targets", type=Path, default=Path("dataset/controls/pert_counts.csv"))
    parser.add_argument("--gene-count", type=int, default=512)
    parser.add_argument("--max-cells-per-target", type=int, default=64)
    parser.add_argument("--target-holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-genes-csv", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = prepare_cache(parse_args())
    print(json.dumps({"schema": result["schema"], "split": result["split"], "artifacts": result["artifacts"]}, indent=2))
