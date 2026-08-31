#!/usr/bin/env python3
"""Build a batch-matched perturbation effect atlas from pseudobulk means.

The Replogle ``*_raw_bulk_01.h5ad`` files contain one per-cell mean expression
profile per perturbation condition, not integer single-cell counts.  This
builder reconstructs condition sums with the condition cell count, pools
multiple guide/promoter conditions by target, and matches each target's batch
composition to non-targeting controls.  If no batch annotation exists, it uses
one explicitly reported global control stratum.

The output NPZ deliberately matches ``build_raw_effect_atlas.py``:
``effects``, ``matched_control_profiles``, ``target_names``, ``gene_names``,
``target_cell_counts``, ``batch_names``, ``target_batch_counts``,
``control_batch_counts``, and ``target_sum``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


SCHEMA = "vcc-bulk-effect-atlas-v1"
GLOBAL_BATCH = "__global__"
REQUIRED_NPZ_KEYS = (
    "effects",
    "matched_control_profiles",
    "target_names",
    "gene_names",
    "target_cell_counts",
    "batch_names",
    "target_batch_counts",
    "control_batch_counts",
    "target_sum",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--target-col",
        default="auto",
        help="Target column, 'auto', or 'index' for encoded observation names.",
    )
    parser.add_argument(
        "--guide-col",
        default="auto",
        help="Guide/promoter column, 'auto', 'index', or 'none'.",
    )
    parser.add_argument(
        "--batch-col",
        default="auto",
        help="Batch column, 'auto', or 'none' for global controls.",
    )
    parser.add_argument(
        "--cell-count-col",
        default="auto",
        help="Number of cells represented by each pseudobulk row.",
    )
    parser.add_argument(
        "--control-selection-col",
        default="auto",
        help="Optional boolean field selecting valid non-targeting controls.",
    )
    parser.add_argument("--gene-col", default="auto")
    parser.add_argument("--control", default="non-targeting")
    parser.add_argument("--min-cells", type=int, default=30)
    parser.add_argument("--target-sum", type=float, default=50_000.0)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument(
        "--matrix-semantics",
        choices=("auto", "per-cell-mean", "summed-counts"),
        default="auto",
    )
    parser.add_argument(
        "--count-reconstruction-tolerance",
        type=float,
        default=0.02,
        help="Diagnostic tolerance for mean multiplied by represented cell count.",
    )
    parser.add_argument(
        "--missing-control-batch",
        choices=("global", "error"),
        default="global",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=0,
        help="Deterministic smoke-test limit; zero retains every eligible target.",
    )
    parser.add_argument(
        "--challenge-genes",
        type=Path,
        default=Path("dataset/controls/gene_names.csv"),
    )
    parser.add_argument(
        "--challenge-targets",
        type=Path,
        default=Path("dataset/controls/pert_counts.csv"),
    )
    parser.add_argument("--expected-md5", default=None)
    parser.add_argument("--expected-sha256", default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    require(args.input_h5ad.is_file(), f"Missing input: {args.input_h5ad}")
    require(args.output_npz.suffix == ".npz", "output_npz must end in .npz")
    require(args.output_json.suffix == ".json", "output_json must end in .json")
    require(args.min_cells > 0, "min-cells must be positive")
    require(args.target_sum > 0, "target-sum must be positive")
    require(args.chunk_rows > 0, "chunk-rows must be positive")
    require(args.max_targets >= 0, "max-targets cannot be negative")
    require(
        args.count_reconstruction_tolerance >= 0,
        "count-reconstruction-tolerance cannot be negative",
    )
    for output in (args.output_npz, args.output_json):
        require(not output.exists(), f"Refusing to overwrite {output}")


def file_hashes(path: Path, block_size: int = 8 * 1024 * 1024) -> dict[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            md5.update(block)
            sha256.update(block)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def parse_condition_name(name: str, control_label: str) -> tuple[str, str, str]:
    """Parse ``row_target_guide_target-id`` while preserving guide underscores."""
    rendered = str(name)
    try:
        _, remainder = rendered.split("_", 1)
    except ValueError as error:
        raise ValueError(f"Cannot parse condition name: {rendered}") from error
    control_encoding = f"{control_label}_{control_label}_{control_label}"
    if remainder == control_encoding:
        return control_label, control_label, control_label
    try:
        target, guide_and_id = remainder.split("_", 1)
        guide, target_id = guide_and_id.rsplit("_", 1)
    except ValueError as error:
        raise ValueError(f"Cannot parse condition name: {rendered}") from error
    require(target != "" and guide != "" and target_id != "", f"Malformed name: {rendered}")
    return target, guide, target_id


def _resolve_column(
    frame: pd.DataFrame,
    requested: str,
    candidates: tuple[str, ...],
    field: str,
    *,
    allow_none: bool,
) -> str | None:
    if requested in {"none", "index"}:
        require(allow_none, f"{field} cannot be disabled")
        return None
    if requested == "auto":
        return next((candidate for candidate in candidates if candidate in frame.columns), None)
    require(requested in frame.columns, f"Missing {field} column: {requested}")
    return requested


@dataclass(frozen=True)
class BulkSchema:
    targets: np.ndarray
    guides: np.ndarray
    target_ids: np.ndarray
    batches: np.ndarray
    cell_counts: np.ndarray
    controls: np.ndarray
    selected_controls: np.ndarray
    target_source: str
    guide_source: str
    batch_source: str
    cell_count_source: str
    control_selection_source: str | None


def resolve_bulk_schema(data: ad.AnnData, args: argparse.Namespace) -> BulkSchema:
    obs = data.obs
    parsed = [parse_condition_name(name, args.control) for name in data.obs_names.astype(str)]
    parsed_targets = np.asarray([item[0] for item in parsed], dtype="U")
    parsed_guides = np.asarray([item[1] for item in parsed], dtype="U")
    parsed_ids = np.asarray([item[2] for item in parsed], dtype="U")

    target_col = _resolve_column(
        obs,
        args.target_col,
        ("target_gene", "gene", "perturbation", "condition"),
        "target",
        allow_none=True,
    )
    if args.target_col == "index" or target_col is None:
        targets = parsed_targets
        target_source = "obs_names"
    else:
        targets = obs[target_col].astype(str).to_numpy(dtype="U")
        target_source = target_col

    guide_col = _resolve_column(
        obs,
        args.guide_col,
        ("guide_id", "sgID_AB", "guide", "transcript"),
        "guide",
        allow_none=True,
    )
    if args.guide_col == "none":
        guides = np.asarray(["__unspecified__"] * data.n_obs, dtype="U")
        guide_source = "none"
    elif args.guide_col == "index" or guide_col is None:
        guides = parsed_guides
        guide_source = "obs_names"
    else:
        guides = obs[guide_col].astype(str).to_numpy(dtype="U")
        guide_source = guide_col

    batch_col = _resolve_column(
        obs,
        args.batch_col,
        ("batch", "gem_group", "gemgroup", "gem_group_id"),
        "batch",
        allow_none=True,
    )
    if args.batch_col == "none" or batch_col is None:
        batches = np.asarray([GLOBAL_BATCH] * data.n_obs, dtype="U")
        batch_source = "global-fallback"
    else:
        batches = obs[batch_col].astype(str).to_numpy(dtype="U")
        batch_source = batch_col

    count_col = _resolve_column(
        obs,
        args.cell_count_col,
        ("num_cells_filtered", "n_cells", "cell_count", "num_cells"),
        "cell-count",
        allow_none=False,
    )
    require(count_col is not None, "No cell-count column was found")
    raw_counts = pd.to_numeric(obs[count_col], errors="coerce").to_numpy(dtype=np.float64)
    finite_counts = np.isfinite(raw_counts)
    require(
        np.all(np.abs(raw_counts[finite_counts] - np.rint(raw_counts[finite_counts])) < 1e-6),
        "Cell counts must be integer-like",
    )
    cell_counts = np.zeros(data.n_obs, dtype=np.int64)
    cell_counts[finite_counts] = np.rint(raw_counts[finite_counts]).astype(np.int64)

    controls = targets == args.control
    selection_col = _resolve_column(
        obs,
        args.control_selection_col,
        ("core_control", "is_control_qc", "selected_control"),
        "control-selection",
        allow_none=True,
    )
    selected_controls = controls & finite_counts & (cell_counts > 0)
    if selection_col is not None:
        selection = obs[selection_col]
        require(
            pd.api.types.is_bool_dtype(selection.dtype),
            f"Control-selection column must be boolean: {selection_col}",
        )
        selected_controls &= selection.to_numpy(dtype=bool)
    valid_targets = (~controls) & finite_counts
    require(np.all(cell_counts[valid_targets] > 0), "A target row has a non-positive cell count")
    require(bool(selected_controls.any()), "No usable non-targeting controls were found")
    require(np.all(targets != ""), "Target labels contain empty strings")
    require(np.all(guides != ""), "Guide labels contain empty strings")
    require(np.all(batches != ""), "Batch labels contain empty strings")
    return BulkSchema(
        targets=targets,
        guides=guides,
        target_ids=parsed_ids,
        batches=batches,
        cell_counts=cell_counts,
        controls=controls,
        selected_controls=selected_controls,
        target_source=target_source,
        guide_source=guide_source,
        batch_source=batch_source,
        cell_count_source=count_col,
        control_selection_source=selection_col,
    )


@dataclass(frozen=True)
class GeneCollapser:
    genes: np.ndarray
    first_positions: np.ndarray
    duplicate_positions: np.ndarray
    duplicate_destinations: np.ndarray
    duplicate_symbols: dict[str, int]

    def collapse(self, values: np.ndarray) -> np.ndarray:
        source = np.asarray(values)
        result = source[:, self.first_positions].copy()
        for source_position, destination in zip(
            self.duplicate_positions, self.duplicate_destinations, strict=True
        ):
            result[:, destination] += source[:, source_position]
        return result


def resolve_gene_axis(data: ad.AnnData, args: argparse.Namespace) -> GeneCollapser:
    if args.gene_col == "auto":
        gene_col = "gene_name" if "gene_name" in data.var.columns else None
    else:
        require(args.gene_col in data.var.columns, f"Missing gene column: {args.gene_col}")
        gene_col = args.gene_col
    raw_genes = (
        data.var_names.astype(str).to_numpy(dtype="U")
        if gene_col is None
        else data.var[gene_col].astype(str).to_numpy(dtype="U")
    )
    require(np.all(raw_genes != ""), "Gene symbols contain empty strings")
    lookup: dict[str, int] = {}
    first: list[int] = []
    duplicates: list[int] = []
    duplicate_destinations: list[int] = []
    for position, gene in enumerate(raw_genes):
        if gene not in lookup:
            lookup[gene] = len(first)
            first.append(position)
        else:
            duplicates.append(position)
            duplicate_destinations.append(lookup[gene])
    counts = pd.Series(raw_genes).value_counts()
    duplicate_symbols = {
        str(gene): int(count) for gene, count in counts[counts > 1].items()
    }
    return GeneCollapser(
        genes=np.asarray(list(lookup), dtype="U"),
        first_positions=np.asarray(first, dtype=np.int64),
        duplicate_positions=np.asarray(duplicates, dtype=np.int64),
        duplicate_destinations=np.asarray(duplicate_destinations, dtype=np.int64),
        duplicate_symbols=duplicate_symbols,
    )


def inspect_matrix(
    data: ad.AnnData,
    cell_counts: np.ndarray,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    total_values = data.n_obs * data.n_vars
    nonfinite = 0
    negative = 0
    noninteger = 0
    reconstructed_noninteger = 0
    reconstructed_values = 0
    maximum_fractional_error = 0.0
    maximum_reconstruction_error = 0.0
    minimum = np.inf
    maximum = -np.inf
    nonzero = 0
    for begin in range(0, data.n_obs, args.chunk_rows):
        end = min(begin + args.chunk_rows, data.n_obs)
        values = dense_rows(data.X, begin, end)
        nonfinite += int(np.count_nonzero(~np.isfinite(values)))
        negative += int(np.count_nonzero(values < 0))
        minimum = min(minimum, float(np.nanmin(values)))
        maximum = max(maximum, float(np.nanmax(values)))
        nonzero += int(np.count_nonzero(values))
        error = np.abs(values - np.rint(values))
        noninteger += int(np.count_nonzero(error > 1e-6))
        maximum_fractional_error = max(maximum_fractional_error, float(error.max()))
        local_counts = cell_counts[begin:end]
        valid = local_counts > 0
        if np.any(valid):
            reconstructed = values[valid] * local_counts[valid, None]
            reconstruction_error = np.abs(reconstructed - np.rint(reconstructed))
            reconstructed_noninteger += int(
                np.count_nonzero(reconstruction_error > args.count_reconstruction_tolerance)
            )
            reconstructed_values += int(reconstruction_error.size)
            maximum_reconstruction_error = max(
                maximum_reconstruction_error, float(reconstruction_error.max())
            )
    require(nonfinite == 0, "Expression matrix contains non-finite values")
    require(negative == 0, "Expression matrix contains negative values")
    if args.matrix_semantics == "auto":
        semantics = "per-cell-mean" if noninteger > 0 else "summed-counts"
    else:
        semantics = args.matrix_semantics
    if semantics == "summed-counts":
        require(noninteger == 0, "summed-counts matrix contains fractional values")
    return semantics, {
        "shape": [int(data.n_obs), int(data.n_vars)],
        "storage_type": f"{type(data.X).__module__}.{type(data.X).__name__}",
        "dtype": str(data.X.dtype),
        "minimum": minimum,
        "maximum": maximum,
        "nonzero_values": nonzero,
        "density": float(nonzero / total_values),
        "noninteger_values_at_1e-6": noninteger,
        "noninteger_fraction": float(noninteger / total_values),
        "maximum_fractional_error": maximum_fractional_error,
        "reconstructed_values": reconstructed_values,
        "reconstructed_noninteger_values": reconstructed_noninteger,
        "reconstructed_noninteger_fraction": float(
            reconstructed_noninteger / max(reconstructed_values, 1)
        ),
        "maximum_reconstruction_error": maximum_reconstruction_error,
        "resolved_semantics": semantics,
    }


def _encode(values: np.ndarray, vocabulary: list[str]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(vocabulary)}
    return np.fromiter(
        (lookup.get(str(value), -1) for value in values),
        dtype=np.int64,
        count=len(values),
    )


def dense_rows(matrix: Any, begin: int, end: int) -> np.ndarray:
    """Read a bounded dense float64 block from dense or sparse backed storage."""
    values = matrix[begin:end]
    if sp.issparse(values):
        values = values.toarray()
    result = np.asarray(values, dtype=np.float64)
    require(result.ndim == 2, "Expression chunk is not two-dimensional")
    return result


def build_effect_atlas(
    data: ad.AnnData,
    schema: BulkSchema,
    genes: GeneCollapser,
    semantics: str,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target_rows = ~schema.controls
    aggregated_cells = (
        pd.DataFrame(
            {
                "target": schema.targets[target_rows],
                "cells": schema.cell_counts[target_rows],
            }
        )
        .groupby("target", sort=True, observed=True)["cells"]
        .sum()
    )
    target_names = sorted(
        str(target)
        for target, count in aggregated_cells.items()
        if int(count) >= args.min_cells
    )
    if args.max_targets:
        target_names = target_names[: args.max_targets]
    require(target_names, "No perturbations passed the cell-count filter")

    valid_selected_targets = (schema.cell_counts > 0) & np.isin(
        schema.targets, target_names
    )
    usable = schema.selected_controls | valid_selected_targets
    batch_names = sorted(set(schema.batches[usable].tolist()))
    require(batch_names, "No batches were retained")
    target_codes = _encode(schema.targets, target_names)
    batch_codes = _encode(schema.batches, batch_names)
    n_targets = len(target_names)
    n_batches = len(batch_names)
    n_genes = len(genes.genes)

    target_cell_counts = np.zeros(n_targets, dtype=np.int64)
    target_batch_counts = np.zeros((n_targets, n_batches), dtype=np.int64)
    control_batch_counts = np.zeros(n_batches, dtype=np.int64)
    valid_target_rows = (target_codes >= 0) & (schema.cell_counts > 0)
    np.add.at(
        target_cell_counts,
        target_codes[valid_target_rows],
        schema.cell_counts[valid_target_rows],
    )
    np.add.at(
        target_batch_counts,
        (target_codes[valid_target_rows], batch_codes[valid_target_rows]),
        schema.cell_counts[valid_target_rows],
    )
    np.add.at(
        control_batch_counts,
        batch_codes[schema.selected_controls],
        schema.cell_counts[schema.selected_controls],
    )
    require(np.all(target_cell_counts >= args.min_cells), "Target count filter failed")
    require(int(control_batch_counts.sum()) > 0, "Selected controls have no cells")

    target_sums = np.zeros((n_targets, n_genes), dtype=np.float64)
    control_sums = np.zeros((n_batches, n_genes), dtype=np.float64)
    for begin in range(0, data.n_obs, args.chunk_rows):
        end = min(begin + args.chunk_rows, data.n_obs)
        values = dense_rows(data.X, begin, end)
        values = genes.collapse(values)
        if semantics == "per-cell-mean":
            values *= schema.cell_counts[begin:end, None]
        local_targets = target_codes[begin:end]
        local_target_mask = local_targets >= 0
        if np.any(local_target_mask):
            np.add.at(
                target_sums,
                local_targets[local_target_mask],
                values[local_target_mask],
            )
        local_controls = schema.selected_controls[begin:end]
        if np.any(local_controls):
            np.add.at(
                control_sums,
                batch_codes[begin:end][local_controls],
                values[local_controls],
            )

    global_control_sum = control_sums.sum(axis=0, dtype=np.float64)
    global_control_cells = int(control_batch_counts.sum(dtype=np.int64))
    global_control_mean = global_control_sum / global_control_cells
    control_means = np.empty_like(control_sums)
    missing_control_batches: list[str] = []
    for batch_index, batch in enumerate(batch_names):
        if control_batch_counts[batch_index] > 0:
            control_means[batch_index] = (
                control_sums[batch_index] / control_batch_counts[batch_index]
            )
        else:
            require(
                args.missing_control_batch == "global",
                f"Batch has target rows but no controls: {batch}",
            )
            control_means[batch_index] = global_control_mean
            missing_control_batches.append(batch)

    matched_control_sums = target_batch_counts @ control_means
    target_totals = target_sums.sum(axis=1, keepdims=True, dtype=np.float64)
    control_totals = matched_control_sums.sum(axis=1, keepdims=True, dtype=np.float64)
    require(np.all(target_totals > 0), "At least one target has zero total expression")
    require(np.all(control_totals > 0), "At least one matched control has zero expression")
    target_profiles = np.log1p(args.target_sum * target_sums / target_totals)
    control_profiles = np.log1p(
        args.target_sum * matched_control_sums / control_totals
    ).astype(np.float32)
    effects = (target_profiles - control_profiles).astype(np.float32)
    require(np.isfinite(effects).all(), "Computed effects contain non-finite values")

    guide_frame = pd.DataFrame(
        {
            "target": schema.targets[valid_target_rows],
            "guide": schema.guides[valid_target_rows],
        }
    )
    guide_counts = guide_frame.groupby("target", sort=True, observed=True)["guide"].nunique()
    retained_guide_counts = np.asarray(
        [int(guide_counts.loc[target]) for target in target_names], dtype=np.int64
    )
    payload = {
        "effects": effects,
        "matched_control_profiles": control_profiles,
        "target_names": np.asarray(target_names, dtype="U"),
        "gene_names": genes.genes.astype("U"),
        "target_cell_counts": target_cell_counts,
        "batch_names": np.asarray(batch_names, dtype="U"),
        "target_batch_counts": target_batch_counts,
        "control_batch_counts": control_batch_counts,
        "target_sum": np.asarray([args.target_sum], dtype=np.float64),
    }
    require(tuple(payload) == REQUIRED_NPZ_KEYS, "Internal NPZ contract mismatch")
    report = {
        "targets_before_cell_filter": int(aggregated_cells.size),
        "targets_retained": n_targets,
        "target_rows_retained": int(valid_target_rows.sum()),
        "target_cells_retained": int(target_cell_counts.sum()),
        "controls_available": int(schema.controls.sum()),
        "controls_selected": int(schema.selected_controls.sum()),
        "control_cells_selected": global_control_cells,
        "batches": n_batches,
        "batch_names": batch_names,
        "missing_control_batches_using_global_mean": missing_control_batches,
        "guide_conditions_retained": int(guide_frame.shape[0]),
        "unique_guide_labels": int(guide_frame["guide"].nunique()),
        "guides_per_target": {
            "min": int(retained_guide_counts.min()),
            "median": float(np.median(retained_guide_counts)),
            "max": int(retained_guide_counts.max()),
            "targets_with_multiple_guides": int(np.count_nonzero(retained_guide_counts > 1)),
        },
    }
    return payload, report


def axis_overlap(
    genes: np.ndarray,
    targets: np.ndarray,
    challenge_genes_path: Path,
    challenge_targets_path: Path,
) -> dict[str, Any] | None:
    if not challenge_genes_path.is_file() or not challenge_targets_path.is_file():
        return None
    challenge_gene_frame = pd.read_csv(challenge_genes_path, dtype=str)
    challenge_target_frame = pd.read_csv(challenge_targets_path, dtype=str)
    require("gene_name" in challenge_gene_frame, "Challenge gene axis lacks gene_name")
    require("target_gene" in challenge_target_frame, "Challenge targets lack target_gene")
    challenge_genes = set(challenge_gene_frame["gene_name"].astype(str))
    challenge_targets = set(challenge_target_frame["target_gene"].astype(str))
    gene_set = set(genes.astype(str))
    target_set = set(targets.astype(str))
    return {
        "challenge_genes": len(challenge_genes),
        "challenge_targets": len(challenge_targets),
        "expression_gene_overlap": len(gene_set & challenge_genes),
        "target_overlap": len(target_set & challenge_targets),
        "challenge_targets_on_expression_axis": len(challenge_targets & gene_set),
        "overlapping_targets": sorted(target_set & challenge_targets),
    }


def atomic_save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.time()
    input_path = args.input_h5ad.resolve()
    input_stat = input_path.stat()
    input_hashes = file_hashes(input_path)
    if args.expected_md5 is not None:
        require(
            input_hashes["md5"].lower() == args.expected_md5.lower(),
            "Input MD5 does not match --expected-md5",
        )
    if args.expected_sha256 is not None:
        require(
            input_hashes["sha256"].lower() == args.expected_sha256.lower(),
            "Input SHA-256 does not match --expected-sha256",
        )
    data = ad.read_h5ad(input_path, backed="r")
    try:
        schema = resolve_bulk_schema(data, args)
        genes = resolve_gene_axis(data, args)
        semantics, matrix_report = inspect_matrix(data, schema.cell_counts, args)
        payload, aggregation_report = build_effect_atlas(
            data, schema, genes, semantics, args
        )
    finally:
        data.file.close()

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(args.output_npz, payload)
    output_hashes = file_hashes(args.output_npz)
    overlap = axis_overlap(
        genes.genes,
        payload["target_names"],
        args.challenge_genes,
        args.challenge_targets,
    )
    report = {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "input": {
            "path": str(input_path),
            "bytes": input_stat.st_size,
            **input_hashes,
        },
        "output": {
            "path": str(args.output_npz.resolve()),
            "bytes": args.output_npz.stat().st_size,
            **output_hashes,
            "npz_keys": list(payload),
        },
        "contract": {
            "effect_space": "log1p-group-sum-cp50000-target-minus-batch-matched-control",
            "normalization_target_sum": args.target_sum,
            "minimum_cells": args.min_cells,
            "matrix_semantics": semantics,
            "target_source": schema.target_source,
            "guide_source": schema.guide_source,
            "batch_source": schema.batch_source,
            "cell_count_source": schema.cell_count_source,
            "control_selection_source": schema.control_selection_source,
            "condition_aggregation": "cell-count-weighted guide/promoter pooling",
            "control_matching": (
                "target batch-cell composition" if schema.batch_source != "global-fallback"
                else "global cell-count-weighted non-targeting mean"
            ),
            "targets": int(len(payload["target_names"])),
            "genes": int(len(payload["gene_names"])),
            "batches": int(len(payload["batch_names"])),
        },
        "source_matrix": matrix_report,
        "gene_axis": {
            "source_genes": int(matrix_report["shape"][1]),
            "unique_output_genes": int(len(genes.genes)),
            "duplicate_symbols_collapsed_by_sum": genes.duplicate_symbols,
        },
        "aggregation": aggregation_report,
        "challenge_overlap": overlap,
        "runtime": {
            "argv": sys.argv,
            "python": platform.python_version(),
            "host": socket.gethostname(),
            "numpy": np.__version__,
        },
    }
    atomic_write_json(args.output_json, report)
    print(
        f"Wrote {args.output_npz} with {len(payload['target_names']):,} targets, "
        f"{len(payload['gene_names']):,} genes, and {len(payload['batch_names']):,} batches",
        flush=True,
    )
    print(f"Wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
