#!/usr/bin/env python3
"""Freeze the shared STATE/HepG2/Jurkat gene axis for public validation."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections import Counter
from pathlib import Path

import anndata as ad
import pandas as pd

from generate_state_direct_counts import atomic_write_json
from infer_state_effect_prior import describe_file, read_single_column, require


SCHEMA = "vcc-public-validation-gene-axis-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-genes",
        type=Path,
        default=Path("dataset/state_support/extracted/gene_names.csv"),
    )
    parser.add_argument(
        "--hepg2-raw",
        type=Path,
        default=Path(
            "dataset/raw/replogle_nadig/GSE264667_hepg2_raw_singlecell_01.h5ad"
        ),
    )
    parser.add_argument(
        "--jurkat-raw",
        type=Path,
        default=Path(
            "dataset/raw/replogle_nadig/GSE264667_jurkat_raw_singlecell_01.h5ad"
        ),
    )
    parser.add_argument(
        "--panel-manifest",
        type=Path,
        default=Path("dataset/public_v51/split_manifest.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("dataset/public_v51/common_genes.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("dataset/public_v51/common_genes.json"),
    )
    parser.add_argument("--expected-genes", type=int, default=7107)
    return parser.parse_args()


def read_raw_gene_symbols(path: Path) -> tuple[set[str], dict[str, int], int]:
    require(path.is_file(), f"Missing raw public data: {path}")
    data = ad.read_h5ad(path, backed="r")
    try:
        require("gene_name" in data.var.columns, f"Missing gene_name in {path}")
        raw = data.var["gene_name"].astype(str).tolist()
    finally:
        data.file.close()
    counts = Counter(raw)
    duplicates = {gene: count for gene, count in counts.items() if count > 1}
    return set(raw), duplicates, len(raw)


def shared_axis_in_state_order(
    state_genes: list[str], hepg2_genes: set[str], jurkat_genes: set[str]
) -> list[str]:
    require(len(state_genes) == len(set(state_genes)), "STATE gene axis has duplicates")
    result = [
        gene for gene in state_genes if gene in hepg2_genes and gene in jurkat_genes
    ]
    require(len(result) == len(set(result)), "Shared gene axis has duplicates")
    return result


def atomic_write_csv(path: Path, genes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        pd.DataFrame({"gene_name": genes}).to_csv(handle, index=False)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    require(args.expected_genes > 0, "expected-genes must be positive")
    require(args.panel_manifest.is_file(), "Frozen panel manifest is missing")
    for output in (args.output_csv, args.output_json):
        require(not output.exists(), f"Refusing to overwrite frozen gene axis: {output}")
    state_genes = read_single_column(args.state_genes.resolve())
    hepg2, hepg2_duplicates, hepg2_columns = read_raw_gene_symbols(
        args.hepg2_raw.resolve()
    )
    jurkat, jurkat_duplicates, jurkat_columns = read_raw_gene_symbols(
        args.jurkat_raw.resolve()
    )
    genes = shared_axis_in_state_order(state_genes, hepg2, jurkat)
    require(
        len(genes) == args.expected_genes,
        f"Expected {args.expected_genes} shared genes, found {len(genes)}",
    )
    atomic_write_csv(args.output_csv, genes)
    report = {
        "schema": SCHEMA,
        "contract": {
            "ordering": "STATE-support-axis order",
            "genes": len(genes),
            "state_genes": len(state_genes),
            "hepg2_input_gene_columns": hepg2_columns,
            "hepg2_unique_gene_symbols": len(hepg2),
            "jurkat_input_gene_columns": jurkat_columns,
            "jurkat_unique_gene_symbols": len(jurkat),
            "hepg2_duplicate_symbols_to_sum": hepg2_duplicates,
            "jurkat_duplicate_symbols_to_sum": jurkat_duplicates,
            "expression_matrix_accessed": False,
        },
        "provenance": {
            "state_genes": describe_file(args.state_genes.resolve()),
            "hepg2_raw": describe_file(args.hepg2_raw.resolve(), hash_file=False),
            "jurkat_raw": describe_file(args.jurkat_raw.resolve(), hash_file=False),
            "panel_manifest": describe_file(args.panel_manifest.resolve()),
            "output_csv": describe_file(args.output_csv.resolve()),
        },
    }
    atomic_write_json(args.output_json, report)
    print(f"Wrote frozen public gene axis: {args.output_csv}")
    print(f"Wrote gene-axis provenance: {args.output_json}")


if __name__ == "__main__":
    main()
