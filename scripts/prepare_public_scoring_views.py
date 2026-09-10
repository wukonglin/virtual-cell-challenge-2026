#!/usr/bin/env python3
"""Prepare matched-axis scorer views for sealed public validation.

The model prediction must contain only the frozen 300 perturbations. Observed
controls are copied from the controls-only input into the prediction view because
they are part of the challenge input, while treated truth is read only for the
separate scorer-side view. Both views use the frozen cross-context gene axis.
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

from generate_state_direct_counts import atomic_write_json
from infer_state_effect_prior import describe_file, read_single_column, require, sha256_file


SCHEMA = "vcc-public-scoring-views-v1"
DATA_SCHEMA = "vcc-public-validation-data-v1"
CONTROL_LABEL = "non-targeting"


def _validate_common_axis(path: Path) -> Path:
    """Authenticate the frozen common-gene CSV against its adjacent sidecar."""
    sidecar = path.with_suffix(".json")
    require(sidecar.is_file(), f"Missing frozen common-gene sidecar: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    require(payload.get("schema") == "vcc-public-validation-gene-axis-v1", "Bad gene axis")
    require(
        payload.get("contract", {}).get("expression_matrix_accessed") is False,
        "Frozen gene-axis selection read expression values",
    )
    recorded = payload["provenance"]["output_csv"]
    require(path.resolve() == Path(str(recorded["path"])).resolve(), "Gene CSV path changed")
    require(path.stat().st_size == int(recorded["size_bytes"]), "Gene CSV size changed")
    require(sha256_file(path) == str(recorded["sha256"]), "Gene CSV hash changed")
    return sidecar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("controls_only", type=Path)
    parser.add_argument("sealed_truth", type=Path)
    parser.add_argument("common_genes", type=Path)
    parser.add_argument("output_prediction", type=Path)
    parser.add_argument("output_truth", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--compression", choices=("lzf", "gzip"), default="lzf")
    return parser.parse_args()


def _copy_matrix_on_axis(
    source: ad.AnnData, gene_indices: np.ndarray, *, chunk_rows: int
) -> sp.csr_matrix:
    blocks: list[sp.csr_matrix] = []
    for start in range(0, source.n_obs, chunk_rows):
        stop = min(start + chunk_rows, source.n_obs)
        # Read rows before reordering columns. Dense HDF5 datasets reject a
        # non-monotonic fancy index, while the frozen axis is intentionally in
        # STATE order and need not follow the source H5AD order.
        block = source.X[start:stop]
        block = block[:, gene_indices]
        if sp.issparse(block):
            sparse = block.tocsr()
            require(
                np.isfinite(sparse.data).all() and np.all(sparse.data >= 0),
                "Invalid sparse counts",
            )
            require(
                np.max(np.abs(sparse.data - np.rint(sparse.data)), initial=0) == 0,
                "Sparse scoring input is not integer-like",
            )
            require(
                np.max(sparse.data, initial=0) <= np.iinfo(np.int32).max,
                "Sparse scoring count exceeds int32",
            )
            values = sparse.astype(np.int32)
        else:
            dense = np.asarray(block)
            require(np.isfinite(dense).all() and np.all(dense >= 0), "Invalid counts")
            require(
                np.max(np.abs(dense - np.rint(dense)), initial=0) == 0,
                "Scoring input is not integer-like",
            )
            require(
                np.max(dense, initial=0) <= np.iinfo(np.int32).max,
                "Scoring count exceeds int32",
            )
            values = sp.csr_matrix(dense.astype(np.int32, copy=False))
        values.eliminate_zeros()
        values.sort_indices()
        blocks.append(values)
    require(bool(blocks), "Cannot copy an empty scoring matrix")
    return sp.vstack(blocks, format="csr", dtype=np.int32)


def _atomic_h5ad(path: Path, data: ad.AnnData, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.h5ad")
    require(not temporary.exists(), f"Temporary output exists: {temporary}")
    try:
        data.write_h5ad(temporary, compression=compression)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _axis_indices(data: ad.AnnData, genes: list[str], label: str) -> np.ndarray:
    source_genes = data.var_names.astype(str).tolist()
    require(len(source_genes) == len(set(source_genes)), f"{label} gene axis has duplicates")
    lookup = {name: index for index, name in enumerate(source_genes)}
    missing = sorted(set(genes) - set(lookup))
    require(not missing, f"{label} lacks frozen genes: {missing[:10]}")
    return np.asarray([lookup[name] for name in genes], dtype=np.int64)


def main() -> None:
    args = parse_args()
    require(args.chunk_rows > 0, "chunk-rows must be positive")
    for path in (args.prediction, args.controls_only, args.sealed_truth, args.common_genes):
        require(path.is_file(), f"Missing input: {path}")
    require(args.output_prediction.suffix == ".h5ad", "Prediction output must be H5AD")
    require(args.output_truth.suffix == ".h5ad", "Truth output must be H5AD")
    require(args.output_json.suffix == ".json", "Report output must be JSON")
    outputs = (args.output_prediction, args.output_truth, args.output_json)
    for path in outputs:
        require(not path.exists(), f"Refusing to overwrite output: {path}")
    require(
        len({path.resolve() for path in outputs}) == len(outputs),
        "Scoring outputs must be distinct",
    )
    common_genes_json = _validate_common_axis(args.common_genes)
    genes = read_single_column(args.common_genes, "gene_name")
    require(len(genes) == len(set(genes)), "Frozen common-gene axis has duplicates")

    prediction = ad.read_h5ad(args.prediction, backed="r")
    controls = ad.read_h5ad(args.controls_only, backed="r")
    truth = ad.read_h5ad(args.sealed_truth, backed="r")
    try:
        for label, data in (("prediction", prediction), ("controls", controls), ("truth", truth)):
            require("target_gene" in data.obs.columns, f"{label} lacks target_gene")
            require("context" in data.obs.columns, f"{label} lacks context")
            require(data.obs.index.is_unique, f"{label} cell IDs are not unique")
        control_contract = controls.uns.get("public_validation", {})
        truth_contract = truth.uns.get("public_validation", {})
        control_sealed_profiles_present = control_contract.get(
            "sealed_treated_profiles_present"
        )
        truth_sealed_profiles_present = truth_contract.get(
            "sealed_treated_profiles_present"
        )
        require(
            control_contract.get("schema") == DATA_SCHEMA
            and control_contract.get("role") == "controls-only-generator-input"
            and isinstance(control_sealed_profiles_present, (bool, np.bool_))
            and not bool(control_sealed_profiles_present),
            "Controls input lacks the controls-only firewall role",
        )
        require(
            truth_contract.get("schema") == DATA_SCHEMA
            and truth_contract.get("role") == "sealed-scorer-input"
            and isinstance(truth_sealed_profiles_present, (bool, np.bool_))
            and bool(truth_sealed_profiles_present),
            "Truth input lacks the sealed-scorer role",
        )
        context_sets = {
            label: set(data.obs["context"].astype(str))
            for label, data in (
                ("prediction", prediction),
                ("controls", controls),
                ("truth", truth),
            )
        }
        require(
            all(len(values) == 1 for values in context_sets.values()),
            f"A scoring input mixes contexts: {context_sets}",
        )
        contexts = {next(iter(values)) for values in context_sets.values()}
        require(len(contexts) == 1, f"Scoring input contexts differ: {context_sets}")
        context = next(iter(contexts))
        pred_labels = set(prediction.obs["target_gene"].astype(str))
        control_labels = set(controls.obs["target_gene"].astype(str))
        truth_labels = set(truth.obs["target_gene"].astype(str))
        require(CONTROL_LABEL not in pred_labels, "Model prediction must not contain controls")
        require(control_labels == {CONTROL_LABEL}, "Controls-only input contains treated cells")
        require(len(pred_labels) == 300, "Prediction must contain exactly 300 perturbations")
        require(truth_labels == pred_labels | {CONTROL_LABEL}, "Prediction/truth targets differ")

        pred_axis = _axis_indices(prediction, genes, "Prediction")
        control_axis = _axis_indices(controls, genes, "Controls")
        truth_axis = _axis_indices(truth, genes, "Truth")
        pred_matrix = _copy_matrix_on_axis(
            prediction, pred_axis, chunk_rows=args.chunk_rows
        )
        control_matrix = _copy_matrix_on_axis(
            controls, control_axis, chunk_rows=args.chunk_rows
        )
        truth_matrix = _copy_matrix_on_axis(truth, truth_axis, chunk_rows=args.chunk_rows)
        truth_control_rows = (
            truth.obs["target_gene"].astype(str).to_numpy() == CONTROL_LABEL
        )
        truth_control_matrix = truth_matrix[truth_control_rows]
        require(
            truth_control_matrix.shape == control_matrix.shape,
            "Controls and sealed truth contain different control-cell counts",
        )
        require(
            (truth_control_matrix != control_matrix).nnz == 0,
            "Controls and sealed truth contain different control profiles",
        )
        pred_view_matrix = sp.vstack(
            [control_matrix, pred_matrix], format="csr", dtype=np.int32
        )
        pred_obs = pd.concat(
            [controls.obs.copy(), prediction.obs.copy()], axis=0, join="outer"
        )
        require(pred_obs.index.is_unique, "Prediction scoring-view cell IDs collide")
        truth_obs = truth.obs.copy()
        var = pd.DataFrame(index=pd.Index(genes, name="gene_name"))
        pred_view = ad.AnnData(pred_view_matrix, obs=pred_obs, var=var.copy())
        truth_view = ad.AnnData(truth_matrix, obs=truth_obs, var=var.copy())
        pred_view.uns["public_scoring"] = {
            "schema": SCHEMA,
            "observed_controls_appended": True,
            "treated_truth_profiles_present": False,
        }
        truth_view.uns["public_scoring"] = {
            "schema": SCHEMA,
            "observed_controls_appended": False,
            "treated_truth_profiles_present": True,
        }
    finally:
        prediction.file.close()
        controls.file.close()
        truth.file.close()

    _atomic_h5ad(args.output_prediction, pred_view, args.compression)
    _atomic_h5ad(args.output_truth, truth_view, args.compression)
    report = {
        "schema": SCHEMA,
        "context": context,
        "axes": {
            "genes": len(genes),
            "targets": len(pred_labels),
            "prediction_treated_cells": int(prediction.n_obs),
            "observed_control_cells": int(controls.n_obs),
            "prediction_view_cells": int(pred_view.n_obs),
            "truth_view_cells": int(truth_view.n_obs),
        },
        "contract": {
            "model_read_treated_truth": False,
            "prediction_controls_source": "released-controls-only-input",
            "prediction_controls_match_sealed_truth_exactly": True,
            "gene_axis": "frozen-state-hepg2-jurkat-intersection",
            "input_type": "raw-nonnegative-integer-counts",
            "frozen_common_axis_authenticated": True,
        },
        "provenance": {
            "prediction": describe_file(args.prediction),
            "controls_only": describe_file(args.controls_only),
            "sealed_truth": describe_file(args.sealed_truth),
            "common_genes": describe_file(args.common_genes),
            "common_genes_json": describe_file(common_genes_json),
            "output_prediction": {
                "path": str(args.output_prediction.resolve()),
                "size_bytes": args.output_prediction.stat().st_size,
                "sha256": sha256_file(args.output_prediction),
            },
            "output_truth": {
                "path": str(args.output_truth.resolve()),
                "size_bytes": args.output_truth.stat().st_size,
                "sha256": sha256_file(args.output_truth),
            },
        },
    }
    atomic_write_json(args.output_json, report)
    print(f"Wrote {args.output_prediction}")
    print(f"Wrote {args.output_truth}")


if __name__ == "__main__":
    main()
