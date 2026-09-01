#!/usr/bin/env python3
"""Build a sparse, reliability-gated residual prior for a STATE anchor.

The input cross-context prior contains a shared response and a target-specific
response.  This builder deliberately discards the shared response, recenters
the target-specific response over directly observed targets, gates it to genes
measured in the public query atlas, removes unscored panel-target coordinates,
and keeps only the strongest eligible genes for each context-target group.

The output follows the response-NPZ contract consumed by
``generate_state_scorer_aware_counts.py``.  It is an additive residual only;
the paired STATE prediction remains the anchor during generation.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import scipy.sparse as sp

from generate_state_direct_counts import atomic_write_json
from infer_state_effect_prior import (
    CONTEXTS,
    EXPECTED_CURRENT_GENES,
    EXPECTED_SUPPORT_GENES,
    EXPECTED_TARGETS,
    describe_file,
    read_single_column,
    read_targets,
    require,
    sha256_file,
)


SCHEMA = "vcc-state-anchored-residual-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sparse public-data residual for a paired STATE anchor."
    )
    parser.add_argument("--input-prior", type=Path, required=True)
    parser.add_argument(
        "--metadata-prior",
        type=Path,
        required=True,
        help="Prior artifact containing direct/fallback masks and source reliability.",
    )
    parser.add_argument("--controls-dir", type=Path, default=Path("dataset/controls"))
    parser.add_argument(
        "--support-genes",
        type=Path,
        default=Path("dataset/state_support/extracted/gene_names.csv"),
    )
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--min-context-detection-rate", type=float, default=0.01)
    parser.add_argument("--min-context-mean-count", type=float, default=0.01)
    parser.add_argument(
        "--direct-reliability-power",
        type=float,
        default=0.0,
        help=(
            "Optional additional reliability shrinkage. The v4 learned residual already "
            "contains count-based reliability once, so zero avoids applying it twice."
        ),
    )
    parser.add_argument(
        "--fallback-reliability",
        type=float,
        default=0.0,
        help="Fixed reliability for ESM2-only fallback targets; zero is conservative.",
    )
    parser.add_argument("--control-chunk-size", type=int, default=1024)
    parser.add_argument("--centering", choices=("mean", "median"), default="mean")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    require(args.input_prior.is_file(), f"Input prior is missing: {args.input_prior}")
    require(args.metadata_prior.is_file(), f"Metadata prior is missing: {args.metadata_prior}")
    require(args.controls_dir.is_dir(), f"Controls directory is missing: {args.controls_dir}")
    require(args.support_genes.is_file(), f"Support gene list is missing: {args.support_genes}")
    require(args.output_npz.suffix == ".npz", "output-npz must end in .npz")
    require(args.output_json.suffix == ".json", "output-json must end in .json")
    require(args.output_npz.resolve() != args.output_json.resolve(), "Output paths differ")
    require(1 <= args.top_k <= EXPECTED_CURRENT_GENES, "Invalid top-k")
    require(
        0 <= args.min_context_detection_rate <= 1,
        "min-context-detection-rate must be in [0, 1]",
    )
    require(args.min_context_mean_count >= 0, "min-context-mean-count cannot be negative")
    require(args.direct_reliability_power >= 0, "Invalid reliability power")
    require(0 <= args.fallback_reliability <= 1, "fallback-reliability must be in [0, 1]")
    require(args.control_chunk_size > 0, "control-chunk-size must be positive")
    for output in (args.output_npz, args.output_json):
        require(not output.exists(), f"Refusing to overwrite existing artifact: {output}")


def _strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in np.asarray(values).reshape(-1)
    ]


def read_context_detection(
    controls_dir: Path,
    genes: list[str],
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-context detection rate, mean count, and number of cells."""
    detection = np.zeros((len(CONTEXTS), len(genes)), dtype=np.float32)
    mean_count = np.zeros_like(detection)
    cells = np.zeros(len(CONTEXTS), dtype=np.int64)
    for context_index, context in enumerate(CONTEXTS):
        path = controls_dir / f"context_{context}.h5ad"
        require(path.is_file(), f"Control file is missing: {path}")
        control = ad.read_h5ad(path, backed="r")
        try:
            require(list(control.var_names.astype(str)) == genes, f"{context}: gene mismatch")
            cell_count = int(control.n_obs)
            require(cell_count > 0, f"{context}: no control cells")
            nonzero = np.zeros(len(genes), dtype=np.int64)
            totals = np.zeros(len(genes), dtype=np.float64)
            for start in range(0, cell_count, chunk_size):
                stop = min(start + chunk_size, cell_count)
                block = control.X[start:stop]
                if sp.issparse(block):
                    block = block.tocsr()
                    require(np.isfinite(block.data).all(), f"{context}: non-finite counts")
                    require(np.all(block.data >= 0), f"{context}: negative counts")
                    nonzero += np.asarray((block > 0).sum(axis=0)).ravel().astype(np.int64)
                    totals += np.asarray(block.sum(axis=0)).ravel().astype(np.float64)
                else:
                    dense = np.asarray(block)
                    require(np.isfinite(dense).all(), f"{context}: non-finite counts")
                    require(np.all(dense >= 0), f"{context}: negative counts")
                    nonzero += np.count_nonzero(dense > 0, axis=0)
                    totals += dense.sum(axis=0, dtype=np.float64)
            detection[context_index] = nonzero / cell_count
            mean_count[context_index] = totals / cell_count
            cells[context_index] = cell_count
        finally:
            control.file.close()
    return detection, mean_count, cells


def build_gated_residual(
    learned_log_fold: np.ndarray,
    direct_mask: np.ndarray,
    fallback_mask: np.ndarray,
    source_reliability: np.ndarray,
    context_detection: np.ndarray,
    context_mean_count: np.ndarray,
    mutable_gene_mask: np.ndarray,
    query_gene_mask: np.ndarray,
    panel_gene_mask: np.ndarray,
    *,
    top_k: int,
    min_context_detection_rate: float,
    min_context_mean_count: float,
    direct_reliability_power: float,
    fallback_reliability: float,
    centering: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Center, gate, and sparsify a target-specific response matrix."""
    learned = np.asarray(learned_log_fold, dtype=np.float32)
    direct = np.asarray(direct_mask, dtype=bool)
    fallback = np.asarray(fallback_mask, dtype=bool)
    reliability = np.asarray(source_reliability, dtype=np.float32)
    detection = np.asarray(context_detection, dtype=np.float32)
    mean_count = np.asarray(context_mean_count, dtype=np.float32)
    mutable = np.asarray(mutable_gene_mask, dtype=bool)
    query = np.asarray(query_gene_mask, dtype=bool)
    panel = np.asarray(panel_gene_mask, dtype=bool)
    require(learned.ndim in (2, 3), "learned_log_fold must have two or three dimensions")
    if learned.ndim == 2:
        targets, genes = learned.shape
        learned_by_context = np.broadcast_to(
            learned[None, :, :], (detection.shape[0], targets, genes)
        )
    else:
        contexts, targets, genes = learned.shape
        require(contexts == detection.shape[0], "Response and detection contexts differ")
        learned_by_context = learned
    require(direct.shape == (targets,), "Bad direct mask")
    require(fallback.shape == (targets,), "Bad fallback mask")
    require(np.all(direct ^ fallback), "Every target must be direct or fallback")
    require(reliability.shape == (targets,), "Bad source reliability")
    require(detection.ndim == 2 and detection.shape[1] == genes, "Bad detection matrix")
    require(mean_count.shape == detection.shape, "Bad context mean-count matrix")
    require(mutable.shape == (genes,), "Bad mutable-gene mask")
    require(query.shape == (genes,), "Bad query-gene mask")
    require(panel.shape == (genes,), "Bad panel-gene mask")
    require(np.isfinite(learned).all(), "Non-finite learned response")
    require(np.isfinite(reliability).all(), "Non-finite reliability")
    require(np.isfinite(detection).all(), "Non-finite detection rate")
    require(np.isfinite(mean_count).all() and np.all(mean_count >= 0), "Invalid mean count")
    require(np.all((reliability >= 0) & (reliability <= 1)), "Reliability is outside [0, 1]")
    require(np.any(direct), "No directly observed targets")

    if centering == "mean":
        center = learned_by_context[:, direct].mean(axis=1, dtype=np.float64).astype(
            np.float32
        )
    elif centering == "median":
        center = np.median(learned_by_context[:, direct], axis=1).astype(np.float32)
    else:
        raise RuntimeError(f"Unsupported centering mode: {centering}")
    centered = learned_by_context - center[:, None, :]

    target_reliability = np.zeros(targets, dtype=np.float32)
    target_reliability[direct] = np.power(
        reliability[direct], np.float32(direct_reliability_power)
    ).astype(np.float32)
    target_reliability[fallback] = np.float32(fallback_reliability)
    weighted = centered * target_reliability[None, :, None]

    output = np.zeros((detection.shape[0], targets, genes), dtype=np.float32)
    eligible_counts: list[int] = []
    retained_counts: list[int] = []
    for context_index in range(detection.shape[0]):
        eligible = (
            mutable
            & query
            & ~panel
            & (detection[context_index] >= np.float32(min_context_detection_rate))
            & (mean_count[context_index] >= np.float32(min_context_mean_count))
        )
        for target_index in range(targets):
            candidates = np.flatnonzero(
                eligible & (weighted[context_index, target_index] != 0)
            )
            eligible_counts.append(int(len(candidates)))
            if len(candidates) > top_k:
                order = np.lexsort(
                    (candidates, -np.abs(weighted[context_index, target_index, candidates]))
                )
                candidates = candidates[order[:top_k]]
            output[context_index, target_index, candidates] = weighted[
                context_index, target_index, candidates
            ]
            retained_counts.append(int(len(candidates)))

    require(np.isfinite(output).all(), "Non-finite gated residual")
    nonzero_per_group = np.count_nonzero(output, axis=2)
    require(np.all(nonzero_per_group <= top_k), "A residual group exceeds top-k")
    require(not np.any(output[:, :, panel]), "Panel-target coordinates were retained")
    require(not np.any(output[:, :, ~mutable]), "Protected coordinates were retained")
    require(not np.any(output[:, :, ~query]), "Unmeasured query coordinates were retained")
    return output, {
        "center": center,
        "target_reliability": target_reliability,
        "eligible_counts": np.asarray(eligible_counts, dtype=np.int64),
        "retained_counts": np.asarray(retained_counts, dtype=np.int64),
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    require(array.size > 0 and np.isfinite(array).all(), "Cannot summarize values")
    quantiles = np.quantile(array, [0.0, 0.1, 0.5, 0.9, 1.0])
    return {
        "min": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "max": float(quantiles[4]),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    controls_dir = args.controls_dir.resolve()
    genes = read_single_column(controls_dir / "gene_names.csv", "gene_name")
    targets = read_targets(controls_dir / "pert_counts.csv")
    support_genes = read_single_column(args.support_genes.resolve())
    require(len(genes) == EXPECTED_CURRENT_GENES, "Unexpected current gene count")
    require(len(targets) == EXPECTED_TARGETS, "Unexpected target count")
    require(len(support_genes) == EXPECTED_SUPPORT_GENES, "Unexpected support gene count")

    with np.load(args.input_prior.resolve(), allow_pickle=False) as archive:
        required = {"genes", "targets", "learned_log_fold"}
        require(required.issubset(archive.files), "Input prior is missing required arrays")
        prior_genes = _strings(archive["genes"])
        prior_targets = _strings(archive["targets"])
        require(prior_genes == genes, "Input prior gene axis differs from controls")
        require(prior_targets == targets, "Input prior target axis differs from controls")
        learned = np.asarray(archive["learned_log_fold"], dtype=np.float32)
        if learned.ndim == 3:
            require("contexts" in archive.files, "3D input prior requires contexts")
            require(
                _strings(archive["contexts"]) == list(CONTEXTS),
                "Input prior context axis differs from the challenge",
            )
    with np.load(args.metadata_prior.resolve(), allow_pickle=False) as archive:
        required = {
            "genes",
            "targets",
            "direct_mask",
            "fallback_mask",
            "source_reliability",
            "query_gene_mask",
        }
        require(required.issubset(archive.files), "Metadata prior is missing required arrays")
        metadata_genes = _strings(archive["genes"])
        metadata_targets = _strings(archive["targets"])
        require(metadata_genes == genes, "Metadata prior gene axis differs from controls")
        require(metadata_targets == targets, "Metadata prior target axis differs from controls")
        direct = np.asarray(archive["direct_mask"], dtype=bool)
        fallback = np.asarray(archive["fallback_mask"], dtype=bool)
        reliability = np.asarray(archive["source_reliability"], dtype=np.float32)
        query = np.asarray(archive["query_gene_mask"], dtype=bool)

    detection, mean_count, control_cells = read_context_detection(
        controls_dir, genes, chunk_size=args.control_chunk_size
    )
    support_set = set(support_genes)
    mutable = np.asarray([gene in support_set for gene in genes], dtype=bool)
    target_set = set(targets)
    panel = np.asarray([gene in target_set for gene in genes], dtype=bool)
    residual, qc = build_gated_residual(
        learned,
        direct,
        fallback,
        reliability,
        detection,
        mean_count,
        mutable,
        query,
        panel,
        top_k=args.top_k,
        min_context_detection_rate=args.min_context_detection_rate,
        min_context_mean_count=args.min_context_mean_count,
        direct_reliability_power=args.direct_reliability_power,
        fallback_reliability=args.fallback_reliability,
        centering=args.centering,
    )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{args.output_npz.name}.", suffix=".npz", dir=args.output_npz.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(
            temporary_path,
            genes=np.asarray(genes),
            targets=np.asarray(targets),
            contexts=np.asarray(CONTEXTS),
            learned_log_fold=residual,
            direct_mask=direct,
            fallback_mask=fallback,
            source_reliability=reliability,
            applied_target_reliability=qc["target_reliability"],
            context_detection_rate=detection,
            context_mean_count=mean_count,
            protected_current_only_mask=~mutable,
            query_gene_mask=query,
            excluded_panel_target_mask=panel,
        )
        os.replace(temporary_path, args.output_npz)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    nonzero = np.count_nonzero(residual, axis=2)
    nonzero_values = residual[residual != 0]
    report = {
        "schema": SCHEMA,
        "configuration": {
            "top_k": args.top_k,
            "min_context_detection_rate": args.min_context_detection_rate,
            "min_context_mean_count": args.min_context_mean_count,
            "direct_reliability_power": args.direct_reliability_power,
            "fallback_reliability": args.fallback_reliability,
            "centering": args.centering,
            "shared_response_weight": 0.0,
            "preserve_current_only_genes": True,
            "exclude_all_panel_target_genes": True,
        },
        "axes": {
            "contexts": list(CONTEXTS),
            "targets": len(targets),
            "genes": len(genes),
            "support_overlap_genes": int(mutable.sum()),
            "protected_current_only_genes": int((~mutable).sum()),
            "public_query_genes": int(query.sum()),
            "excluded_panel_target_genes": int(panel.sum()),
            "direct_targets": int(direct.sum()),
            "fallback_targets": int(fallback.sum()),
            "control_cells": {
                context: int(control_cells[index]) for index, context in enumerate(CONTEXTS)
            },
        },
        "scientific_qc": {
            "nonzero_residuals": int(np.count_nonzero(residual)),
            "nonzero_per_group": _summary(nonzero),
            "retained_effect": _summary(nonzero_values),
            "eligible_before_top_k": _summary(qc["eligible_counts"]),
            "applied_target_reliability": _summary(qc["target_reliability"]),
            "context_detection_rate": {
                context: _summary(detection[index]) for index, context in enumerate(CONTEXTS)
            },
            "context_mean_count": {
                context: _summary(mean_count[index]) for index, context in enumerate(CONTEXTS)
            },
            "fallback_residuals_are_zero": bool(not np.any(residual[:, fallback])),
            "common_response_emitted": False,
            "source_cell_reliability": (
                "already-applied-upstream"
                if args.direct_reliability_power == 0
                else "additional-power-applied"
            ),
        },
        "provenance": {
            "input_prior": describe_file(args.input_prior.resolve()),
            "metadata_prior": describe_file(args.metadata_prior.resolve()),
            "support_genes": describe_file(args.support_genes.resolve()),
            "controls": {
                context: describe_file(controls_dir / f"context_{context}.h5ad")
                for context in CONTEXTS
            },
            "output_npz": {
                "path": str(args.output_npz.resolve()),
                "size_bytes": args.output_npz.stat().st_size,
                "sha256": sha256_file(args.output_npz),
            },
        },
    }
    atomic_write_json(args.output_json, report)
    print(f"Wrote {args.output_npz}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
