#!/usr/bin/env python3
"""Build a batch-matched pseudobulk CRISPRi effect atlas from raw counts.

The output is a compact training contract for target-generalization models.  It
stores one log-normalized effect vector per perturbation, where every target is
compared with a non-targeting profile matched to its experimental-batch mix.
The script reads the expression matrix in bounded row chunks and can use a GPU
for grouped accumulation, so it does not materialize the full cell-by-gene
matrix in host memory.
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
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from build_bulk_effect_atlas import resolve_gene_axis


SCHEMA = "vcc-raw-effect-atlas-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--pert-col", default="gene")
    parser.add_argument("--batch-col", default="gem_group")
    parser.add_argument("--gene-col", default="gene_name")
    parser.add_argument("--control", default="non-targeting")
    parser.add_argument("--min-cells", type=int, default=30)
    parser.add_argument("--target-sum", type=float, default=50_000.0)
    parser.add_argument("--chunk-rows", type=int, default=2_048)
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


def encode(values: np.ndarray, vocabulary: list[str]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(vocabulary)}
    return np.fromiter(
        (lookup.get(str(value), -1) for value in values),
        dtype=np.int64,
        count=len(values),
    )


def resolve_device(name: str, require_cuda: bool) -> torch.device:
    device = torch.device(name)
    if require_cuda and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("A CUDA device was required but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main() -> None:
    args = parse_args()
    started = time.time()
    require(args.input_h5ad.is_file(), f"Missing input: {args.input_h5ad}")
    require(args.output_npz.suffix == ".npz", "output_npz must end in .npz")
    require(args.output_json.suffix == ".json", "output_json must end in .json")
    require(args.min_cells > 0, "min-cells must be positive")
    require(args.target_sum > 0, "target-sum must be positive")
    require(args.chunk_rows > 0, "chunk-rows must be positive")
    for output in (args.output_npz, args.output_json):
        require(not output.exists(), f"Refusing to overwrite {output}")
    require(
        args.output_npz.parent.resolve() == args.output_json.parent.resolve(),
        "output_npz and output_json must share one directory",
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = args.output_npz.with_name(
        f".{args.output_npz.stem}.{os.getpid()}.partial.npz"
    )
    temporary_json = args.output_json.with_name(
        f".{args.output_json.stem}.{os.getpid()}.partial.json"
    )
    for temporary_output in (temporary_npz, temporary_json):
        require(
            not temporary_output.exists(),
            f"Refusing to overwrite staging output {temporary_output}",
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device, args.require_cuda)

    data = ad.read_h5ad(args.input_h5ad, backed="r")
    try:
        require(args.pert_col in data.obs, f"Missing obs column {args.pert_col}")
        require(args.batch_col in data.obs, f"Missing obs column {args.batch_col}")
        require(args.gene_col in data.var, f"Missing var column {args.gene_col}")
        require(data.n_obs > 0 and data.n_vars > 0, "Input matrix is empty")

        perturbations = data.obs[args.pert_col].astype(str).to_numpy()
        batches_raw = data.obs[args.batch_col].astype(str).to_numpy()
        raw_gene_names = data.var[args.gene_col].astype(str).to_numpy(dtype="U")
        raw_gene_ids = data.var_names.astype(str).to_numpy(dtype="U")
        gene_collapser = resolve_gene_axis(data, args)
        genes = gene_collapser.genes
        duplicate_gene_sources = {
            gene: [
                {
                    "source_column": int(position),
                    "source_var_name": str(raw_gene_ids[position]),
                }
                for position in np.flatnonzero(raw_gene_names == gene)
            ]
            for gene in gene_collapser.duplicate_symbols
        }

        unique, counts = np.unique(perturbations, return_counts=True)
        target_names = sorted(
            name
            for name, count in zip(unique.tolist(), counts.tolist(), strict=True)
            if name != args.control and count >= args.min_cells
        )
        require(target_names, "No perturbations passed the cell-count filter")
        batch_names = sorted(set(batches_raw.tolist()))
        target_codes = encode(perturbations, target_names)
        batch_codes = encode(batches_raw, batch_names)
        require(np.all(batch_codes >= 0), "Failed to encode experimental batches")
        control_mask = perturbations == args.control
        require(int(control_mask.sum()) > 0, "No non-targeting control cells were found")

        n_targets = len(target_names)
        n_batches = len(batch_names)
        n_genes = len(genes)
        target_cell_counts = np.bincount(
            target_codes[target_codes >= 0], minlength=n_targets
        ).astype(np.int64)
        target_batch_counts = np.zeros((n_targets, n_batches), dtype=np.int64)
        np.add.at(
            target_batch_counts,
            (target_codes[target_codes >= 0], batch_codes[target_codes >= 0]),
            1,
        )
        control_batch_counts = np.bincount(
            batch_codes[control_mask], minlength=n_batches
        ).astype(np.int64)

        target_sums = torch.zeros(
            (n_targets, n_genes), dtype=torch.float64, device=device
        )
        control_sums = torch.zeros(
            (n_batches, n_genes), dtype=torch.float64, device=device
        )
        for begin in range(0, data.n_obs, args.chunk_rows):
            end = min(begin + args.chunk_rows, data.n_obs)
            values_np = np.asarray(data.X[begin:end], dtype=np.float32)
            require(np.isfinite(values_np).all(), f"Non-finite values in rows {begin}:{end}")
            require(np.min(values_np) >= 0, f"Negative values in rows {begin}:{end}")
            require(
                np.max(np.abs(values_np - np.rint(values_np))) == 0,
                "Input expression must contain raw integer counts",
            )
            values_np = gene_collapser.collapse(values_np)
            values = torch.from_numpy(values_np).to(device=device, dtype=torch.float64)

            local_targets = target_codes[begin:end]
            valid_targets = local_targets >= 0
            if np.any(valid_targets):
                positions = torch.from_numpy(np.flatnonzero(valid_targets)).to(
                    device=device, dtype=torch.long
                )
                indices = torch.from_numpy(local_targets[valid_targets]).to(
                    device=device, dtype=torch.long
                )
                target_sums.index_add_(0, indices, values.index_select(0, positions))

            local_control = control_mask[begin:end]
            if np.any(local_control):
                positions = torch.from_numpy(np.flatnonzero(local_control)).to(
                    device=device, dtype=torch.long
                )
                indices = torch.from_numpy(batch_codes[begin:end][local_control]).to(
                    device=device, dtype=torch.long
                )
                control_sums.index_add_(0, indices, values.index_select(0, positions))

            if begin == 0 or end == data.n_obs or end % (10 * args.chunk_rows) == 0:
                print(f"Accumulated {end:,}/{data.n_obs:,} cells", flush=True)

        global_control = control_sums.sum(dim=0) / float(control_mask.sum())
        control_denominator = torch.from_numpy(control_batch_counts).to(
            device=device, dtype=torch.float64
        )
        missing_control_batches = control_denominator == 0
        safe_denominator = torch.clamp(control_denominator, min=1.0)
        control_means = control_sums / safe_denominator[:, None]
        if bool(missing_control_batches.any()):
            control_means[missing_control_batches] = global_control

        target_batch_weights = torch.from_numpy(target_batch_counts).to(
            device=device, dtype=torch.float64
        )
        matched_control_sums = target_batch_weights @ control_means
        target_totals = target_sums.sum(dim=1, keepdim=True)
        control_totals = matched_control_sums.sum(dim=1, keepdim=True)
        require(bool(torch.all(target_totals > 0)), "At least one target has zero total counts")
        require(bool(torch.all(control_totals > 0)), "At least one matched control has zero counts")

        target_profiles = torch.log1p(args.target_sum * target_sums / target_totals)
        control_profiles = torch.log1p(
            args.target_sum * matched_control_sums / control_totals
        )
        effects = (target_profiles - control_profiles).to(torch.float32).cpu().numpy()
        control_profiles_np = control_profiles.to(torch.float32).cpu().numpy()
        require(np.isfinite(effects).all(), "Computed effects contain non-finite values")

        np.savez_compressed(
            temporary_npz,
            effects=effects,
            matched_control_profiles=control_profiles_np,
            target_names=np.asarray(target_names, dtype="U"),
            gene_names=genes.astype("U"),
            input_gene_names=raw_gene_names,
            input_gene_ids=raw_gene_ids,
            gene_first_positions=gene_collapser.first_positions,
            gene_duplicate_positions=gene_collapser.duplicate_positions,
            gene_duplicate_destinations=gene_collapser.duplicate_destinations,
            target_cell_counts=target_cell_counts,
            batch_names=np.asarray(batch_names, dtype="U"),
            target_batch_counts=target_batch_counts,
            control_batch_counts=control_batch_counts,
            target_sum=np.asarray([args.target_sum], dtype=np.float64),
        )
    finally:
        data.file.close()

    input_sha256 = sha256_file(args.input_h5ad)
    output_sha256 = sha256_file(temporary_npz)
    report = {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "input": {
            "path": str(args.input_h5ad.resolve()),
            "bytes": args.input_h5ad.stat().st_size,
            "sha256": input_sha256,
        },
        "output": {
            "path": str(args.output_npz.resolve()),
            "bytes": temporary_npz.stat().st_size,
            "sha256": output_sha256,
        },
        "contract": {
            "perturbation_column": args.pert_col,
            "batch_column": args.batch_col,
            "gene_column": args.gene_col,
            "control_label": args.control,
            "minimum_cells": args.min_cells,
            "normalization_target_sum": args.target_sum,
            "effect_space": "log1p-group-sum-cp50000-target-minus-batch-matched-control",
            "targets": len(target_names),
            "genes": len(genes),
            "input_gene_columns": int(len(raw_gene_names)),
            "collapsed_duplicate_columns": int(
                len(gene_collapser.duplicate_positions)
            ),
            "duplicate_symbols_collapsed_by_sum": gene_collapser.duplicate_symbols,
            "duplicate_gene_sources": duplicate_gene_sources,
            "batches": len(batch_names),
        },
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
    temporary_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    npz_committed = False
    try:
        os.replace(temporary_npz, args.output_npz)
        npz_committed = True
        os.replace(temporary_json, args.output_json)
    except BaseException:
        # The JSON is the pair's commit marker. Roll back the first rename if
        # the second rename fails so a normal error cannot strand a final NPZ.
        if npz_committed and not args.output_json.exists():
            args.output_npz.unlink(missing_ok=True)
        raise
    finally:
        temporary_npz.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
