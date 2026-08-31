#!/usr/bin/env python3
"""Convert one released STATE support HDF5 matrix to an evaluation AnnData.

The competition support archive stores dense expression and categorical
metadata in a compact cell-load-compatible HDF5 layout rather than AnnData.
This utility performs a lossless conversion for local, label-aware holdout
evaluation.  It does not alter expression values or use hidden VCC labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5", type=Path)
    parser.add_argument("output_h5ad", type=Path)
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf"),
        default="lzf",
        help="HDF5 compression for the generated AnnData matrix.",
    )
    return parser.parse_args()


def decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def main() -> None:
    args = parse_args()
    if not args.input_h5.is_file():
        raise FileNotFoundError(args.input_h5)
    if args.output_h5ad.exists():
        raise FileExistsError(args.output_h5ad)

    with h5py.File(args.input_h5, "r") as source:
        matrix = source["X"][:].astype(np.float32, copy=False)
        genes = decode(source["var/_index"][:])
        if matrix.ndim != 2 or matrix.shape[1] != len(genes):
            raise ValueError("Expression matrix and gene axis are inconsistent")
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError("Expression matrix must be finite and non-negative")

        obs_data: dict[str, object] = {}
        for name, node in source["obs"].items():
            if isinstance(node, h5py.Group) and {"categories", "codes"}.issubset(node):
                categories = decode(node["categories"][:])
                codes = node["codes"][:].astype(np.int64, copy=False)
                obs_data[name] = pd.Categorical.from_codes(codes, categories=categories)
            elif isinstance(node, h5py.Dataset):
                values = node[:]
                obs_data[name] = decode(values) if values.dtype.kind in {"O", "S", "U"} else values

    obs = pd.DataFrame(obs_data)
    if len(obs) != matrix.shape[0]:
        raise ValueError("Observation metadata and expression matrix are inconsistent")
    var = pd.DataFrame(index=pd.Index(genes, name="gene_name"))
    result = ad.AnnData(X=matrix, obs=obs, var=var)
    result.obs_names = pd.Index([f"holdout_cell_{i:06d}" for i in range(result.n_obs)])
    result.uns["log1p"] = {"base": None}

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(args.output_h5ad, compression=args.compression)
    print(
        f"Wrote {args.output_h5ad} with shape {result.shape}, "
        f"{result.obs['target_gene'].nunique()} perturbation labels, and unchanged values"
    )


if __name__ == "__main__":
    main()
