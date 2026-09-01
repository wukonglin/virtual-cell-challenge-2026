#!/usr/bin/env python3
"""Build a leakage-safe K562/RPE1 residual for public STATE validation.

The builder may read the frozen panel, public K562/RPE1 effect atlases, and one
controls-only recipient context. It never reads the held recipient perturbation
profiles. Complete ESM2 clusters marked ``held_target`` are forced to zero so
the same artifact evaluates both same-target transfer and target fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from build_state_anchored_residual import _summary, build_gated_residual
from generate_state_direct_counts import atomic_write_json
from infer_state_effect_prior import describe_file, read_single_column, require, sha256_file


SCHEMA = "vcc-public-state-residual-v1"
DATA_SCHEMA = "vcc-public-validation-data-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k562-atlas", type=Path, required=True)
    parser.add_argument("--rpe1-atlas", type=Path, required=True)
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--panel-json", type=Path, required=True)
    parser.add_argument("--common-genes", type=Path, required=True)
    parser.add_argument("--common-genes-json", type=Path, required=True)
    parser.add_argument("--controls-only", type=Path, required=True)
    parser.add_argument("--context", choices=("HepG2", "Jurkat"), required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--reliability-offset", type=float, default=60.0)
    parser.add_argument("--direct-reliability-power", type=float, default=0.0)
    parser.add_argument("--min-context-detection-rate", type=float, default=0.01)
    parser.add_argument("--min-context-mean-count", type=float, default=0.01)
    parser.add_argument("--control-chunk-size", type=int, default=1024)
    return parser.parse_args()


def _strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in np.asarray(values).reshape(-1)
    ]


def validate_frozen_file(
    path: Path, recorded: dict[str, object], label: str
) -> None:
    """Require a frozen CSV to match its sidecar's exact path, size, and hash."""
    require(path.resolve() == Path(str(recorded["path"])).resolve(), f"{label} path changed")
    require(path.stat().st_size == int(recorded["size_bytes"]), f"{label} size changed")
    require(sha256_file(path) == str(recorded["sha256"]), f"{label} hash changed")


def count_reliability(cell_counts: np.ndarray, offset: float) -> np.ndarray:
    counts = np.asarray(cell_counts, dtype=np.float32)
    require(offset > 0, "Reliability offset must be positive")
    require(np.isfinite(counts).all() and np.all(counts >= 0), "Invalid cell counts")
    return counts / np.float32(counts + offset)


def read_atlas_axes(path: Path) -> tuple[set[str], set[str]]:
    """Read named source axes without materializing an atlas effect matrix."""
    with np.load(path, allow_pickle=False) as archive:
        require(
            {"target_names", "gene_names"}.issubset(archive.files),
            f"Atlas lacks named axes: {path}",
        )
        targets = _strings(archive["target_names"])
        genes = _strings(archive["gene_names"])
    require(len(targets) == len(set(targets)), "Duplicate atlas targets")
    require(len(genes) == len(set(genes)), "Duplicate atlas genes")
    return set(targets), set(genes)


def load_aligned_atlas(
    path: Path, targets: list[str], genes: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load a named target/gene slice and preserve the requested axis order."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"effects", "target_names", "gene_names", "target_cell_counts"}
        require(required.issubset(archive.files), f"Atlas lacks arrays: {path}")
        source_targets = _strings(archive["target_names"])
        source_genes = _strings(archive["gene_names"])
        require(len(source_targets) == len(set(source_targets)), "Duplicate atlas targets")
        require(len(source_genes) == len(set(source_genes)), "Duplicate atlas genes")
        target_lookup = {name: index for index, name in enumerate(source_targets)}
        gene_lookup = {name: index for index, name in enumerate(source_genes)}
        missing_targets = sorted(set(targets) - set(target_lookup))
        missing_genes = sorted(set(genes) - set(gene_lookup))
        require(not missing_targets, f"Atlas lacks panel targets: {missing_targets[:10]}")
        require(not missing_genes, f"Atlas lacks common genes: {missing_genes[:10]}")
        target_indices = np.asarray([target_lookup[name] for name in targets], dtype=np.int64)
        gene_indices = np.asarray([gene_lookup[name] for name in genes], dtype=np.int64)
        effects = np.asarray(archive["effects"][target_indices], dtype=np.float32)
        effects = effects[:, gene_indices]
        counts = np.asarray(archive["target_cell_counts"][target_indices], dtype=np.float32)
    require(effects.shape == (len(targets), len(genes)), "Bad aligned effect shape")
    require(np.isfinite(effects).all(), "Non-finite source effect")
    return effects, counts, {
        "source_targets": len(source_targets),
        "source_genes": len(source_genes),
    }


def combine_centered_sources(
    k562_effect: np.ndarray,
    rpe1_effect: np.ndarray,
    k562_reliability: np.ndarray,
    rpe1_reliability: np.ndarray,
    direct_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Center each source on development targets and reliability-blend rows."""
    k_effect = np.asarray(k562_effect, dtype=np.float32)
    r_effect = np.asarray(rpe1_effect, dtype=np.float32)
    k_rel = np.asarray(k562_reliability, dtype=np.float32)
    r_rel = np.asarray(rpe1_reliability, dtype=np.float32)
    direct = np.asarray(direct_mask, dtype=bool)
    require(k_effect.shape == r_effect.shape, "Source effect shapes differ")
    require(k_rel.shape == r_rel.shape == (k_effect.shape[0],), "Bad reliability shape")
    require(np.any(direct) and np.any(~direct), "Both direct and held targets are required")
    require(np.all((k_rel >= 0) & (k_rel <= 1)), "Bad K562 reliability")
    require(np.all((r_rel >= 0) & (r_rel <= 1)), "Bad RPE1 reliability")

    k_weights = k_rel[direct].astype(np.float64)
    r_weights = r_rel[direct].astype(np.float64)
    require(k_weights.sum() > 0 and r_weights.sum() > 0, "Zero source reliability")
    k_center = np.average(k_effect[direct], axis=0, weights=k_weights).astype(np.float32)
    r_center = np.average(r_effect[direct], axis=0, weights=r_weights).astype(np.float32)
    k_centered = k_effect - k_center
    r_centered = r_effect - r_center
    denominator = k_rel + r_rel
    require(np.all(denominator > 0), "A panel target has no reliable source")
    combined = (
        k_centered * k_rel[:, None] + r_centered * r_rel[:, None]
    ) / denominator[:, None]
    combined[~direct] = 0
    joint_reliability = np.sqrt(k_rel * r_rel).astype(np.float32)
    require(np.isfinite(combined).all(), "Non-finite combined response")
    return combined.astype(np.float32), joint_reliability, {
        "k562_center": k_center,
        "rpe1_center": r_center,
    }


def read_control_statistics(
    path: Path,
    *,
    chunk_size: int,
    expected_context: str | None = None,
    require_public_role: bool = False,
) -> tuple[list[str], np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    data = ad.read_h5ad(path, backed="r")
    try:
        require(data.n_obs > 0 and data.n_vars > 0, "Empty controls-only data")
        require("target_gene" in data.obs.columns, "Controls lack target_gene")
        require(
            set(data.obs["target_gene"].astype(str)) == {"non-targeting"},
            "Treated profiles leaked into controls-only data",
        )
        if expected_context is not None:
            require("context" in data.obs.columns, "Controls lack context")
            require(
                set(data.obs["context"].astype(str)) == {expected_context},
                "Controls context differs from requested context",
            )
        if require_public_role:
            contract = data.uns.get("public_validation", {})
            require(
                contract.get("schema") == DATA_SCHEMA
                and contract.get("role") == "controls-only-generator-input"
                and contract.get("sealed_treated_profiles_present") is False,
                "Controls lack the controls-only firewall role",
            )
        genes = data.var_names.astype(str).tolist()
        require(len(genes) == len(set(genes)), "Controls have duplicate gene symbols")
        require("is_state_support" in data.var.columns, "Controls lack STATE support mask")
        require(
            "is_common_public_gene" in data.var.columns,
            "Controls lack frozen common-gene mask",
        )
        state_mask = data.var["is_state_support"].to_numpy(dtype=bool)
        common_mask = data.var["is_common_public_gene"].to_numpy(dtype=bool)
        detected = np.zeros(data.n_vars, dtype=np.int64)
        totals = np.zeros(data.n_vars, dtype=np.float64)
        for start in range(0, data.n_obs, chunk_size):
            block = data.X[start : min(start + chunk_size, data.n_obs)]
            if sp.issparse(block):
                block = block.tocsr()
                require(np.isfinite(block.data).all() and np.all(block.data >= 0), "Bad counts")
                require(
                    np.max(np.abs(block.data - np.rint(block.data)), initial=0) == 0,
                    "Controls contain non-integer counts",
                )
                detected += np.asarray((block > 0).sum(axis=0)).ravel().astype(np.int64)
                totals += np.asarray(block.sum(axis=0)).ravel().astype(np.float64)
            else:
                dense = np.asarray(block)
                require(np.isfinite(dense).all() and np.all(dense >= 0), "Bad counts")
                require(
                    np.max(np.abs(dense - np.rint(dense)), initial=0) == 0,
                    "Controls contain non-integer counts",
                )
                detected += np.count_nonzero(dense > 0, axis=0)
                totals += dense.sum(axis=0, dtype=np.float64)
        detection = (detected / data.n_obs).astype(np.float32)
        mean_count = (totals / data.n_obs).astype(np.float32)
        return genes, detection, mean_count, int(data.n_obs), state_mask, common_mask
    finally:
        data.file.close()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    for path in (
        args.k562_atlas,
        args.rpe1_atlas,
        args.panel_csv,
        args.panel_json,
        args.common_genes,
        args.common_genes_json,
        args.controls_only,
    ):
        require(path.is_file(), f"Missing input: {path}")
    require(args.top_k > 0, "top-k must be positive")
    require(args.control_chunk_size > 0, "control-chunk-size must be positive")
    require(args.direct_reliability_power >= 0, "Invalid reliability power")
    require(
        0 <= args.min_context_detection_rate <= 1,
        "min-context-detection-rate must be in [0, 1]",
    )
    require(args.min_context_mean_count >= 0, "min-context-mean-count cannot be negative")
    require(args.output_npz.suffix == ".npz", "output-npz must end in .npz")
    require(args.output_json.suffix == ".json", "output-json must end in .json")
    require(args.output_npz.resolve() != args.output_json.resolve(), "Outputs must be distinct")
    for path in (args.output_npz, args.output_json):
        require(not path.exists(), f"Refusing to overwrite output: {path}")

    manifest_report = json.loads(args.panel_json.read_text(encoding="utf-8"))
    gene_report = json.loads(args.common_genes_json.read_text(encoding="utf-8"))
    require(manifest_report.get("schema") == "vcc-public-validation-manifest-v1", "Bad manifest")
    require(gene_report.get("schema") == "vcc-public-validation-gene-axis-v1", "Bad gene axis")
    validate_frozen_file(
        args.panel_csv, manifest_report["provenance"]["output_csv"], "Panel CSV"
    )
    validate_frozen_file(
        args.common_genes,
        gene_report["provenance"]["output_csv"],
        "Common-gene CSV",
    )
    validate_frozen_file(
        args.panel_json,
        gene_report["provenance"]["panel_manifest"],
        "Common-axis panel manifest",
    )
    panel = pd.read_csv(args.panel_csv)
    require(
        {"target_gene", "validation_route"}.issubset(panel.columns),
        "Panel lacks required columns",
    )
    targets = panel["target_gene"].astype(str).tolist()
    require(len(targets) == len(set(targets)), "Duplicate panel targets")
    selection = manifest_report.get("selection_contract", {})
    require(selection.get("effect_values_accessed") is False, "Panel selection read effects")
    require(
        gene_report.get("contract", {}).get("expression_matrix_accessed") is False,
        "Frozen gene-axis selection read expression values",
    )
    require(
        len(targets) == int(selection.get("panel_targets", -1)),
        "Panel size differs from frozen manifest",
    )
    routes = panel["validation_route"].astype(str).to_numpy()
    require(set(routes) == {"direct", "held_target"}, "Unexpected validation routes")
    direct = routes == "direct"
    fallback = ~direct
    require(
        int(direct.sum()) == int(selection.get("direct_targets", -1))
        and int(fallback.sum()) == int(selection.get("held_targets", -1)),
        "Validation-route counts differ from frozen manifest",
    )
    common_genes = read_single_column(args.common_genes, "gene_name")
    k_target_set, k_gene_set = read_atlas_axes(args.k562_atlas)
    r_target_set, r_gene_set = read_atlas_axes(args.rpe1_atlas)
    require(set(targets).issubset(k_target_set), "K562 atlas lacks a panel target")
    require(set(targets).issubset(r_target_set), "RPE1 atlas lacks a panel target")
    source_genes = [
        gene for gene in common_genes if gene in k_gene_set and gene in r_gene_set
    ]
    require(bool(source_genes), "K562/RPE1 have no genes on the frozen scoring axis")

    k_effect, k_counts, k_axes = load_aligned_atlas(args.k562_atlas, targets, source_genes)
    r_effect, r_counts, r_axes = load_aligned_atlas(args.rpe1_atlas, targets, source_genes)
    k_rel = count_reliability(k_counts, args.reliability_offset)
    r_rel = count_reliability(r_counts, args.reliability_offset)
    learned_common, joint_rel, centers = combine_centered_sources(
        k_effect, r_effect, k_rel, r_rel, direct
    )

    (
        native_genes,
        detection,
        mean_count,
        control_cells,
        state_mask,
        common_mask,
    ) = read_control_statistics(
        args.controls_only,
        chunk_size=args.control_chunk_size,
        expected_context=args.context,
        require_public_role=True,
    )
    native_lookup = {name: index for index, name in enumerate(native_genes)}
    missing_common = sorted(set(common_genes) - set(native_lookup))
    require(not missing_common, f"Controls lack common genes: {missing_common[:10]}")
    common_set = set(common_genes)
    expected_common_mask = np.asarray(
        [name in common_set for name in native_genes], dtype=bool
    )
    require(
        np.array_equal(common_mask, expected_common_mask),
        "Controls common-gene mask differs from frozen axis",
    )
    learned_native = np.zeros((len(targets), len(native_genes)), dtype=np.float32)
    source_indices = np.asarray([native_lookup[name] for name in source_genes], dtype=np.int64)
    learned_native[:, source_indices] = learned_common
    query_mask = np.zeros(len(native_genes), dtype=bool)
    query_mask[source_indices] = True
    require(not np.any(query_mask & ~common_mask), "Source query is outside frozen gene axis")
    panel_set = set(targets)
    panel_gene_mask = np.asarray([name in panel_set for name in native_genes], dtype=bool)
    residual, qc = build_gated_residual(
        learned_native,
        direct,
        fallback,
        joint_rel,
        detection[None, :],
        mean_count[None, :],
        state_mask,
        query_mask,
        panel_gene_mask,
        top_k=args.top_k,
        min_context_detection_rate=args.min_context_detection_rate,
        min_context_mean_count=args.min_context_mean_count,
        direct_reliability_power=args.direct_reliability_power,
        fallback_reliability=0.0,
        centering="mean",
    )
    require(not np.any(residual[:, fallback]), "Held-target response is nonzero")
    require(not np.any(residual[:, :, ~state_mask]), "Non-STATE residual is nonzero")
    require(not np.any(residual[:, :, panel_gene_mask]), "Panel-gene residual is nonzero")
    require(np.count_nonzero(residual) > 0, "All public residual coordinates were gated out")

    _atomic_npz(
        args.output_npz,
        effects=residual,
        learned_log_fold=residual,
        target_names=np.asarray(targets),
        gene_names=np.asarray(native_genes),
        targets=np.asarray(targets),
        genes=np.asarray(native_genes),
        contexts=np.asarray([args.context]),
        direct_mask=direct,
        fallback_mask=fallback,
        source_reliability=joint_rel,
        applied_target_reliability=qc["target_reliability"],
        context_detection_rate=detection[None, :],
        context_mean_count=mean_count[None, :],
        query_gene_mask=query_mask,
        state_support_mask=state_mask,
        excluded_panel_target_mask=panel_gene_mask,
        frozen_scoring_gene_mask=common_mask,
    )
    nonzero = np.count_nonzero(residual, axis=2)
    values = residual[residual != 0]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "context": args.context,
        "configuration": {
            "top_k": args.top_k,
            "reliability_offset": args.reliability_offset,
            "direct_reliability_power": args.direct_reliability_power,
            "min_context_detection_rate": args.min_context_detection_rate,
            "min_context_mean_count": args.min_context_mean_count,
            "shared_response_weight": 0.0,
            "exclude_all_panel_target_genes": True,
        },
        "axes": {
            "targets": len(targets),
            "direct_targets": int(direct.sum()),
            "held_targets": int(fallback.sum()),
            "native_genes": len(native_genes),
            "frozen_scoring_genes": len(common_genes),
            "common_source_genes": len(source_genes),
            "state_support_genes": int(state_mask.sum()),
            "control_cells": control_cells,
            "k562_source": k_axes,
            "rpe1_source": r_axes,
        },
        "scientific_qc": {
            "nonzero_residuals": int(np.count_nonzero(residual)),
            "nonzero_per_group": _summary(nonzero),
            "retained_effect": _summary(values),
            "joint_source_reliability": _summary(joint_rel),
            "held_target_residuals_are_zero": True,
            "truth_profiles_read": False,
            "controls_only_firewall_role_verified": True,
            "source_centering_targets": "direct-only",
            "source_center_norms": {
                "k562": float(np.linalg.norm(centers["k562_center"])),
                "rpe1": float(np.linalg.norm(centers["rpe1_center"])),
            },
        },
        "provenance": {
            "k562_atlas": describe_file(args.k562_atlas),
            "rpe1_atlas": describe_file(args.rpe1_atlas),
            "panel_csv": describe_file(args.panel_csv),
            "panel_json": describe_file(args.panel_json),
            "common_genes": describe_file(args.common_genes),
            "common_genes_json": describe_file(args.common_genes_json),
            "controls_only": describe_file(args.controls_only),
            "output_npz": {
                "path": str(args.output_npz.resolve()),
                "size_bytes": args.output_npz.stat().st_size,
                "sha256": sha256_file(args.output_npz),
            },
        },
        "contract": {
            "held_target_rows_are_zero": True,
            "recipient_treated_profiles_used": False,
            "input_paths_exclude_sealed_truth": True,
            "native_non_state_genes_are_immutable": True,
            "effect_space": "log1p-group-sum-cp50000-centered-delta",
            "count_generator_must_align_named_axes": True,
        },
    }
    atomic_write_json(args.output_json, report)
    print(f"Wrote {args.output_npz}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
