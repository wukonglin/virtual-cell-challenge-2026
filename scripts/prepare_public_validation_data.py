#!/usr/bin/env python3
"""Create controls-only and sealed-truth raw-count public validation files.

The controls-only file is the only held-context input a candidate generator may
read. The sealed file contains the same controls plus treated cells for the
frozen 300-target panel and is reserved for scoring. Duplicate gene symbols are
summed rather than dropped. Native context axes are retained for STATE input;
the frozen cross-context common-gene mask is stored in ``var`` for matched-axis
scoring.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from build_bulk_effect_atlas import resolve_gene_axis
from generate_state_direct_counts import atomic_write_json
from infer_state_effect_prior import (
    describe_file,
    read_single_column,
    require,
    sha256_file,
)


SCHEMA = "vcc-public-validation-data-v1"
CONTROL_LABEL = "non-targeting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("context", choices=("HepG2", "Jurkat"))
    parser.add_argument("output_controls", type=Path)
    parser.add_argument("output_truth", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--panel-csv",
        type=Path,
        default=Path("dataset/public_v51/split_manifest.csv"),
    )
    parser.add_argument(
        "--panel-json",
        type=Path,
        default=Path("dataset/public_v51/split_manifest.json"),
    )
    parser.add_argument(
        "--common-genes",
        type=Path,
        default=Path("dataset/public_v51/common_genes.csv"),
    )
    parser.add_argument(
        "--common-genes-json",
        type=Path,
        default=Path("dataset/public_v51/common_genes.json"),
    )
    parser.add_argument(
        "--state-genes",
        type=Path,
        default=Path("dataset/state_support/extracted/gene_names.csv"),
    )
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--compression", choices=("lzf", "gzip"), default="lzf")
    parser.add_argument("--expected-panel-targets", type=int, default=300)
    parser.add_argument("--expected-common-genes", type=int, default=7107)
    return parser.parse_args()


def _validate_frozen_hash(path: Path, recorded: dict[str, object], label: str) -> None:
    require(path.resolve() == Path(str(recorded["path"])).resolve(), f"{label} path changed")
    require(path.stat().st_size == int(recorded["size_bytes"]), f"{label} size changed")
    require(sha256_file(path) == str(recorded["sha256"]), f"{label} hash changed")


def _write_h5ad_atomic(path: Path, data: ad.AnnData, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.h5ad")
    require(not temporary.exists(), f"Temporary output exists: {temporary}")
    try:
        data.write_h5ad(temporary, compression=compression)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _obs_frame(
    source: ad.AnnData,
    positions: np.ndarray,
    context: str,
    route_by_target: dict[str, str],
) -> pd.DataFrame:
    labels = source.obs["gene"].astype(str).to_numpy()[positions]
    sgids = (
        source.obs["sgID_AB"].astype(str).to_numpy()[positions]
        if "sgID_AB" in source.obs.columns
        else labels
    )
    batches = source.obs["gem_group"].astype(str).to_numpy()[positions]
    routes = [
        "control" if label == CONTROL_LABEL else route_by_target[label]
        for label in labels
    ]
    index = pd.Index(
        [f"{context}|public|{int(position):07d}" for position in positions],
        name="cell_id",
    )
    return pd.DataFrame(
        {
            "context": pd.Categorical([context] * len(positions)),
            "target_gene": pd.Categorical(labels),
            "ntc_id": pd.Categorical(sgids),
            "batch": pd.Categorical(batches),
            "validation_route": pd.Categorical(routes),
            "source_cell_index": positions.astype(np.int64),
        },
        index=index,
    )


def main() -> None:
    args = parse_args()
    require(args.input_h5ad.is_file(), f"Missing input: {args.input_h5ad}")
    require(args.chunk_rows > 0, "chunk-rows must be positive")
    require(args.expected_panel_targets > 0, "expected-panel-targets must be positive")
    require(args.expected_common_genes > 0, "expected-common-genes must be positive")
    for source in (
        args.panel_csv,
        args.panel_json,
        args.common_genes,
        args.common_genes_json,
        args.state_genes,
    ):
        require(source.is_file(), f"Missing frozen input: {source}")
    outputs = (args.output_controls, args.output_truth, args.output_json)
    require(
        len({output.resolve() for output in outputs}) == len(outputs),
        "Controls, truth, and report outputs must be distinct",
    )
    for output in outputs:
        require(not output.exists(), f"Refusing to overwrite output: {output}")
    require(args.output_controls.suffix == ".h5ad", "Controls output must be H5AD")
    require(args.output_truth.suffix == ".h5ad", "Truth output must be H5AD")
    require(args.output_json.suffix == ".json", "Report output must be JSON")

    panel_report = json.loads(args.panel_json.read_text(encoding="utf-8"))
    gene_report = json.loads(args.common_genes_json.read_text(encoding="utf-8"))
    require(panel_report.get("schema") == "vcc-public-validation-manifest-v1", "Bad panel schema")
    require(gene_report.get("schema") == "vcc-public-validation-gene-axis-v1", "Bad gene schema")
    _validate_frozen_hash(
        args.panel_csv.resolve(), panel_report["provenance"]["output_csv"], "Panel CSV"
    )
    _validate_frozen_hash(
        args.common_genes.resolve(), gene_report["provenance"]["output_csv"], "Gene CSV"
    )
    panel = pd.read_csv(args.panel_csv, dtype={"target_gene": str})
    required_panel = {"target_gene", "validation_route"}
    require(required_panel.issubset(panel.columns), "Panel CSV lacks required columns")
    require(
        len(panel) == args.expected_panel_targets and panel["target_gene"].is_unique,
        "Bad panel target axis",
    )
    selected_targets = panel["target_gene"].astype(str).tolist()
    selected_set = set(selected_targets)
    require(CONTROL_LABEL not in selected_set, "Control label cannot be a panel target")
    routes = set(panel["validation_route"].astype(str))
    require(
        bool(routes) and routes.issubset({"direct", "held_target"}),
        f"Unsupported validation route(s): {sorted(routes)}",
    )
    route_by_target = dict(
        zip(
            selected_targets,
            panel["validation_route"].astype(str).tolist(),
            strict=True,
        )
    )
    common_genes = read_single_column(args.common_genes.resolve(), "gene_name")
    state_genes = set(read_single_column(args.state_genes.resolve()))
    require(
        len(common_genes) == args.expected_common_genes,
        "Unexpected frozen common-gene count",
    )
    require(set(common_genes).issubset(state_genes), "Common axis is outside STATE support")

    source = ad.read_h5ad(args.input_h5ad, backed="r")
    try:
        for column in ("gene", "gem_group"):
            require(column in source.obs.columns, f"Missing obs column {column}")
        require("gene_name" in source.var.columns, "Missing var gene_name")
        gene_collapser = resolve_gene_axis(source, argparse.Namespace(gene_col="gene_name"))
        native_genes = gene_collapser.genes.tolist()
        native_gene_set = set(native_genes)
        require(selected_set.issubset(native_gene_set), "A panel target is absent from gene axis")
        require(set(common_genes).issubset(native_gene_set), "Common axis is not a subset")
        labels = source.obs["gene"].astype(str).to_numpy()
        available_counts = pd.Series(labels).value_counts()
        manifest_count_column = (
            "hepg2_treated_cells" if args.context == "HepG2" else "jurkat_treated_cells"
        )
        require(manifest_count_column in panel.columns, "Panel lacks context cell counts")
        for target, expected in zip(
            selected_targets, panel[manifest_count_column].astype(int), strict=True
        ):
            require(int(available_counts.get(target, 0)) == int(expected), f"{target}: count drift")

        truth_blocks: list[sp.csr_matrix] = []
        control_blocks: list[sp.csr_matrix] = []
        truth_obs: list[pd.DataFrame] = []
        control_obs: list[pd.DataFrame] = []
        for begin in range(0, source.n_obs, args.chunk_rows):
            end = min(begin + args.chunk_rows, source.n_obs)
            local_labels = labels[begin:end]
            local_truth = np.asarray(
                [(label == CONTROL_LABEL or label in selected_set) for label in local_labels],
                dtype=bool,
            )
            if not np.any(local_truth):
                continue
            values = np.asarray(source.X[begin:end], dtype=np.float32)
            require(np.isfinite(values).all() and np.all(values >= 0), "Invalid raw counts")
            require(
                np.max(np.abs(values - np.rint(values)), initial=0) == 0,
                "Source matrix is not integer-like",
            )
            values = gene_collapser.collapse(values)
            require(values.max(initial=0) <= np.iinfo(np.int32).max, "Count exceeds int32")
            local_positions = np.arange(begin, end, dtype=np.int64)
            truth_positions = local_positions[local_truth]
            truth_values = sp.csr_matrix(values[local_truth].astype(np.int32, copy=False))
            truth_values.eliminate_zeros()
            truth_values.sort_indices()
            truth_blocks.append(truth_values)
            truth_obs.append(_obs_frame(source, truth_positions, args.context, route_by_target))

            local_control = local_labels == CONTROL_LABEL
            if np.any(local_control):
                control_positions = local_positions[local_control]
                control_values = sp.csr_matrix(
                    values[local_control].astype(np.int32, copy=False)
                )
                control_values.eliminate_zeros()
                control_values.sort_indices()
                control_blocks.append(control_values)
                control_obs.append(
                    _obs_frame(source, control_positions, args.context, route_by_target)
                )
            if begin == 0 or end == source.n_obs or end % (10 * args.chunk_rows) == 0:
                print(f"Prepared {end:,}/{source.n_obs:,} source cells", flush=True)
    finally:
        source.file.close()

    require(truth_blocks and control_blocks, "No selected public cells were prepared")
    truth_matrix = sp.vstack(truth_blocks, format="csr", dtype=np.int32)
    control_matrix = sp.vstack(control_blocks, format="csr", dtype=np.int32)
    truth_metadata = pd.concat(truth_obs, axis=0)
    control_metadata = pd.concat(control_obs, axis=0)
    require(truth_matrix.shape[0] == len(truth_metadata), "Truth obs mismatch")
    require(control_matrix.shape[0] == len(control_metadata), "Control obs mismatch")
    require(truth_matrix.shape[1] == len(native_genes), "Truth gene mismatch")
    require(control_matrix.shape[1] == len(native_genes), "Control gene mismatch")
    require((control_metadata["target_gene"].astype(str) == CONTROL_LABEL).all(), "Control leak")
    truth_counts = truth_metadata["target_gene"].astype(str).value_counts()
    require(
        set(truth_counts.index) == selected_set | {CONTROL_LABEL},
        "Truth labels differ",
    )
    require(int(truth_counts[CONTROL_LABEL]) == len(control_metadata), "Control count mismatch")
    require(np.all(np.asarray(truth_matrix.sum(axis=1)).ravel() > 0), "Empty truth cell")
    require(np.all(np.asarray(control_matrix.sum(axis=1)).ravel() > 0), "Empty control cell")

    var = pd.DataFrame(index=pd.Index(native_genes, name="gene_name"))
    var["is_state_support"] = [gene in state_genes for gene in native_genes]
    common_set = set(common_genes)
    var["is_common_public_gene"] = [gene in common_set for gene in native_genes]
    require(
        int(var["is_common_public_gene"].sum()) == args.expected_common_genes,
        "Common mask mismatch",
    )
    controls = ad.AnnData(X=control_matrix, obs=control_metadata, var=var.copy())
    truth = ad.AnnData(X=truth_matrix, obs=truth_metadata, var=var.copy())
    controls.uns["public_validation"] = {
        "schema": SCHEMA,
        "role": "controls-only-generator-input",
        "sealed_treated_profiles_present": False,
    }
    truth.uns["public_validation"] = {
        "schema": SCHEMA,
        "role": "sealed-scorer-input",
        "sealed_treated_profiles_present": True,
    }
    _write_h5ad_atomic(args.output_controls, controls, args.compression)
    _write_h5ad_atomic(args.output_truth, truth, args.compression)

    report = {
        "schema": SCHEMA,
        "context": args.context,
        "firewall": {
            "generator_allowed_input": str(args.output_controls.resolve()),
            "scorer_only_truth": str(args.output_truth.resolve()),
            "controls_contains_treated_cells": False,
            "truth_contains_frozen_panel_only": True,
        },
        "axes": {
            "native_genes_after_duplicate_sum": len(native_genes),
            "state_support_overlap": int(var["is_state_support"].sum()),
            "frozen_common_genes": int(var["is_common_public_gene"].sum()),
            "panel_targets": len(selected_targets),
            "controls": int(len(control_metadata)),
            "truth_cells_including_controls": int(len(truth_metadata)),
        },
        "duplicate_symbols_collapsed_by_sum": gene_collapser.duplicate_symbols,
        "provenance": {
            "source": describe_file(args.input_h5ad.resolve()),
            "panel_csv": describe_file(args.panel_csv.resolve()),
            "panel_json": describe_file(args.panel_json.resolve()),
            "common_genes": describe_file(args.common_genes.resolve()),
            "common_genes_json": describe_file(args.common_genes_json.resolve()),
            "state_genes": describe_file(args.state_genes.resolve()),
            "controls_output": describe_file(args.output_controls.resolve()),
            "truth_output": describe_file(args.output_truth.resolve()),
        },
    }
    atomic_write_json(args.output_json, report)
    print(
        f"Wrote {args.context}: controls={controls.shape}, truth={truth.shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
