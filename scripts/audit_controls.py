#!/usr/bin/env python3
"""Audit the downloaded VCC 2026 validation control bundle.

The script is deliberately read-only with respect to the challenge inputs.  It
streams each backed CSR matrix so the numerical checks do not require loading
all three H5AD files into memory at once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse


EXPECTED_CONTEXTS = ("A", "B", "C")


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def read_one_column_csv(path: Path, expected_header: str) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [expected_header]:
            raise ValueError(
                f"{path} must contain exactly the header {expected_header!r}; "
                f"found {reader.fieldnames!r}"
            )
        values = [(row.get(expected_header) or "").strip() for row in reader]
    if any(not value for value in values):
        raise ValueError(f"{path} contains blank values")
    return values


def json_number(value: Any) -> int | float:
    scalar = value.item() if hasattr(value, "item") else value
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    return float(scalar)


def audit_context(
    path: Path,
    context: str,
    expected_genes: list[str],
    expected_cells: int,
    expected_ntc_ids: int,
    control_label: str,
    chunk_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        errors: list[str] = []
        expected_shape = (expected_cells, len(expected_genes))
        if adata.shape != expected_shape:
            errors.append(f"shape={adata.shape}, expected={expected_shape}")
        if list(adata.var_names.astype(str)) != expected_genes:
            errors.append("var_names differ from gene_names.csv or are reordered")
        if not adata.var_names.is_unique:
            errors.append("var_names are not unique")
        if not adata.obs_names.is_unique:
            errors.append("obs_names are not unique")

        required_obs = ["target_gene", "context", "ntc_id"]
        if list(adata.obs.columns) != required_obs:
            errors.append(
                f"obs columns={list(adata.obs.columns)!r}, expected={required_obs!r}"
            )

        observed_contexts = sorted(set(adata.obs["context"].astype(str)))
        observed_targets = sorted(set(adata.obs["target_gene"].astype(str)))
        ntc_counts = adata.obs["ntc_id"].astype(str).value_counts().sort_index()
        if observed_contexts != [context]:
            errors.append(f"context values={observed_contexts!r}, expected={[context]!r}")
        if observed_targets != [control_label]:
            errors.append(
                f"target_gene values={observed_targets!r}, expected={[control_label]!r}"
            )
        if len(ntc_counts) != expected_ntc_ids:
            errors.append(
                f"n_ntc_ids={len(ntc_counts)}, expected={expected_ntc_ids}"
            )
        expected_per_ntc = expected_cells // expected_ntc_ids
        if not bool((ntc_counts == expected_per_ntc).all()):
            errors.append(
                f"NTC guide counts are not uniformly {expected_per_ntc}: "
                f"min={int(ntc_counts.min())}, max={int(ntc_counts.max())}"
            )

        row_sums = np.empty(adata.n_obs, dtype=np.float64)
        row_nnz = np.empty(adata.n_obs, dtype=np.int64)
        gene_sums = np.zeros(adata.n_vars, dtype=np.float64)
        gene_nnz = np.zeros(adata.n_vars, dtype=np.int64)
        total_nnz = 0
        stored_min = math.inf
        stored_max = -math.inf
        finite = True
        nonnegative = True
        integral = True
        explicit_zeros = 0
        sorted_indices = True
        canonical_format = True
        matrix_dtype = None

        for start in range(0, adata.n_obs, chunk_size):
            stop = min(start + chunk_size, adata.n_obs)
            block = adata.X[start:stop]
            if not sparse.issparse(block):
                block = sparse.csr_matrix(block)
            block = block.tocsr(copy=False)
            data = block.data
            matrix_dtype = str(data.dtype) if matrix_dtype is None else matrix_dtype

            if data.size:
                finite = finite and bool(np.isfinite(data).all())
                nonnegative = nonnegative and bool((data >= 0).all())
                integral = integral and bool(np.equal(data, np.floor(data)).all())
                explicit_zeros += int(np.count_nonzero(data == 0))
                stored_min = min(stored_min, float(data.min()))
                stored_max = max(stored_max, float(data.max()))

            sorted_indices = sorted_indices and bool(block.has_sorted_indices)
            canonical_format = canonical_format and bool(block.has_canonical_format)
            row_sums[start:stop] = np.asarray(block.sum(axis=1)).ravel()
            row_nnz[start:stop] = np.diff(block.indptr)
            gene_sums += np.asarray(block.sum(axis=0)).ravel()
            gene_nnz += np.bincount(block.indices, minlength=adata.n_vars)
            total_nnz += int(block.nnz)

        density = total_nnz / (adata.n_obs * adata.n_vars)
        quantile_levels = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
        quantiles = np.quantile(row_sums, quantile_levels)
        top_indices = np.argsort(gene_sums)[-10:][::-1]

        numeric_checks = {
            "finite": finite,
            "nonnegative": nonnegative,
            "integral_values": integral,
            "explicit_sparse_zeros": explicit_zeros,
            "sorted_csr_indices": sorted_indices,
            "canonical_csr": canonical_format,
            "zero_library_cells": int(np.count_nonzero(row_sums == 0)),
            "cells_over_1m_counts": int(np.count_nonzero(row_sums > 1_000_000)),
        }
        for name, value in numeric_checks.items():
            if name in {"explicit_sparse_zeros", "zero_library_cells", "cells_over_1m_counts"}:
                if value != 0:
                    errors.append(f"{name}={value}")
            elif value is not True:
                errors.append(f"{name}={value}")

        result: dict[str, Any] = {
            "context": context,
            "path": str(path),
            "shape": list(adata.shape),
            "x_backend": type(adata.X).__name__,
            "x_stored_dtype": matrix_dtype,
            "obs_columns": list(adata.obs.columns),
            "var_columns": list(adata.var.columns),
            "obs_names_unique": bool(adata.obs_names.is_unique),
            "var_names_unique": bool(adata.var_names.is_unique),
            "observed_contexts": observed_contexts,
            "observed_target_genes": observed_targets,
            "n_ntc_ids": int(len(ntc_counts)),
            "ntc_cells_min": int(ntc_counts.min()),
            "ntc_cells_max": int(ntc_counts.max()),
            "nnz": int(total_nnz),
            "density": float(density),
            "sparsity": float(1 - density),
            "mean_nnz_per_cell": float(row_nnz.mean()),
            "total_counts": int(round(row_sums.sum())),
            "mean_library_size": float(row_sums.mean()),
            "median_library_size": float(np.median(row_sums)),
            "library_size_quantiles": {
                str(level): json_number(value)
                for level, value in zip(quantile_levels, quantiles, strict=True)
            },
            "stored_min": None if math.isinf(stored_min) else stored_min,
            "stored_max": None if math.isinf(stored_max) else stored_max,
            "genes_observed": int(np.count_nonzero(gene_nnz)),
            "all_zero_genes": int(np.count_nonzero(gene_nnz == 0)),
            "top_genes_by_total_count": [
                {
                    "gene": expected_genes[index],
                    "total_count": int(round(gene_sums[index])),
                }
                for index in top_indices
            ],
            "numeric_checks": numeric_checks,
            "errors": errors,
            "passed": not errors,
        }
        return result, gene_nnz
    finally:
        adata.file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("dataset/controls")
    )
    parser.add_argument(
        "--archive", type=Path, default=Path("dataset/controls.zip")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/control_audit.json")
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--deep-hash", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.data_dir / "manifest.json"
    genes_path = args.data_dir / "gene_names.csv"
    perts_path = args.data_dir / "pert_counts.csv"

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    genes = read_one_column_csv(genes_path, "gene_name")
    targets = read_one_column_csv(perts_path, "target_gene")

    errors: list[str] = []
    if manifest.get("contexts") != list(EXPECTED_CONTEXTS):
        errors.append(f"unexpected contexts: {manifest.get('contexts')!r}")
    if manifest.get("n_genes") != len(genes):
        errors.append("manifest n_genes differs from gene_names.csv")
    if manifest.get("n_constructs") != len(targets):
        errors.append("manifest n_constructs differs from pert_counts.csv")
    if len(set(genes)) != len(genes):
        errors.append("gene_names.csv contains duplicates")
    if len(set(targets)) != len(targets):
        errors.append("pert_counts.csv contains duplicates")
    missing_targets = sorted(set(targets) - set(genes))
    if missing_targets:
        errors.append(f"targets absent from gene axis: {missing_targets[:10]!r}")

    archive_result: dict[str, Any] = {
        "path": str(args.archive),
        "bytes": args.archive.stat().st_size,
        "sha256": sha256_file(args.archive),
    }
    with zipfile.ZipFile(args.archive) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
        archive_result["members"] = members
        archive_result["testzip_bad_member"] = bad_member
        archive_result["safe_flat_members"] = all(
            Path(name).name == name and not name.startswith(("/", "\\"))
            for name in members
        )
    if archive_result["testzip_bad_member"] is not None:
        errors.append(f"ZIP CRC failure: {archive_result['testzip_bad_member']}")
    if not archive_result["safe_flat_members"]:
        errors.append("ZIP contains a nested or unsafe path")

    file_hashes: dict[str, str] = {}
    if args.deep_hash:
        for path in sorted(args.data_dir.iterdir()):
            if path.is_file():
                file_hashes[path.name] = sha256_file(path)

    context_results: dict[str, Any] = {}
    gene_presence: list[np.ndarray] = []
    for context in EXPECTED_CONTEXTS:
        per_context = manifest["per_context"][context]
        result, gene_nnz = audit_context(
            args.data_dir / f"context_{context}.h5ad",
            context,
            genes,
            int(per_context["control_cells"]),
            int(per_context["n_ntc_ids"]),
            str(manifest["control_label"]),
            args.chunk_size,
        )
        context_results[context] = result
        gene_presence.append(gene_nnz > 0)
        errors.extend(f"context {context}: {message}" for message in result["errors"])

    presence = np.stack(gene_presence, axis=0)
    target_indices = np.array([genes.index(target) for target in targets], dtype=np.int64)
    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "archive": archive_result,
        "extracted_sha256": file_hashes,
        "manifest": manifest,
        "gene_axis": {
            "count": len(genes),
            "unique": len(set(genes)) == len(genes),
            "first_10": genes[:10],
            "last_10": genes[-10:],
            "order_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
        },
        "targets": {
            "count": len(targets),
            "unique": len(set(targets)) == len(targets),
            "sorted": targets == sorted(targets),
            "all_on_gene_axis": not missing_targets,
            "all_observed_in_every_context": bool(presence[:, target_indices].all()),
            "first_10": targets[:10],
            "last_10": targets[-10:],
            "order_sha256": hashlib.sha256("\n".join(targets).encode()).hexdigest(),
        },
        "contexts": context_results,
        "cross_context": {
            "genes_zero_in_at_least_one_context": int(np.count_nonzero(~presence.all(axis=0))),
            "genes_zero_in_all_contexts": int(np.count_nonzero(~presence.any(axis=0))),
            "gene_axes_identical": all(
                result["shape"][1] == len(genes) and not result["errors"]
                for result in context_results.values()
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
