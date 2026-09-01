#!/usr/bin/env python3
"""Generate native-axis public-validation counts from a paired STATE anchor.

This adapter accepts exactly one controls-only H5AD and one frozen public-panel
manifest.  It runs the validation-selected STATE checkpoint twice on each
sampled control cell (target and non-targeting), transfers the paired log-space
difference to the native raw-count composition, and preserves each source
library exactly.  Native genes outside STATE support are copied byte-for-byte
at the sparse-row level.

An optional residual NPZ may add a precomputed, leakage-safe public-data
residual.  The generator only aligns and scales that residual; centering,
reliability gating, and Top-K selection must happen upstream.  The alpha-zero
route does not open a residual artifact.

The command line deliberately has no truth-data argument.  A controls file is
accepted only when its embedded public-validation role identifies it as a
controls-only generator input with no sealed treated profiles.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import random
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from generate_state_direct_counts import (  # noqa: E402
    DEFAULT_MAX_GENES_PER_CELL,
    SAFE_INT32_NNZ,
    StreamingMoments,
    atomic_write_json,
    concatenate_paths,
    deterministic_seed,
    make_model_input,
    read_control_rows,
    state_paired_delta,
    summarize,
    transform_raw_row,
    validate_model,
    validate_selection_manifest,
)
from infer_state_effect_prior import (  # noqa: E402
    CONTROL_LABEL,
    EXPECTED_SUPPORT_GENES,
    describe_file,
    git_commit,
    load_perturbation_embeddings,
    load_state_model,
    load_var_dims,
    read_single_column,
    require,
    resolve_checkpoint,
    sha256_file,
)


SCHEMA = "vcc-native-state-anchor-counts-v1"
PANEL_SCHEMA = "vcc-public-validation-manifest-v1"
DATA_SCHEMA = "vcc-public-validation-data-v1"
CONTROL_ROLE = "controls-only-generator-input"


@dataclass(frozen=True)
class FrozenPanel:
    """Authenticated public target axis and its leakage-safe routing labels."""

    targets: tuple[str, ...]
    routes: tuple[str, ...]
    manifest: dict[str, Any]
    manifest_path: Path
    csv_path: Path


@dataclass(frozen=True)
class GeneMaps:
    """Bidirectional mapping between STATE support and a native gene axis."""

    support_to_native: np.ndarray
    native_to_support: np.ndarray


@dataclass(frozen=True)
class ResidualOverlay:
    """A panel-aligned residual represented only on STATE support genes."""

    effects: np.ndarray | None
    direct_mask: np.ndarray
    fallback_mask: np.ndarray
    source: dict[str, Any] | None
    qc: dict[str, Any]

    def target(self, target_index: int, support_genes: int) -> np.ndarray:
        if self.effects is None:
            return np.zeros(support_genes, dtype=np.float32)
        return self.effects[target_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-h5ad", type=Path, required=True)
    parser.add_argument(
        "--panel-manifest",
        type=Path,
        required=True,
        help="Frozen vcc-public-validation-manifest-v1 JSON.",
    )
    parser.add_argument(
        "--panel-csv",
        type=Path,
        default=None,
        help="Optional relocated target CSV; its size and SHA-256 must match the manifest.",
    )
    parser.add_argument(
        "--context",
        default=None,
        help="Optional expected context. Otherwise the sole controls context is used.",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("selected.ckpt"))
    parser.add_argument("--selection-json", type=Path, default=None)
    parser.add_argument(
        "--support-genes",
        type=Path,
        default=Path("dataset/state_support/extracted/gene_names.csv"),
    )
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-h100", action="store_true")
    parser.add_argument("--model-chunk-size", type=int, default=128)
    parser.add_argument("--cells-per-group", type=int, default=400)
    parser.add_argument(
        "--target-limit",
        type=int,
        default=0,
        help="Smoke-only prefix length; zero generates the entire frozen panel.",
    )
    parser.add_argument("--state-effect-weight", type=float, default=1.0)
    parser.add_argument("--state-effect-clip", type=float, default=0.60)
    parser.add_argument(
        "--state-effect-bounding", choices=("tanh", "clip"), default="tanh"
    )
    parser.add_argument(
        "--residual-npz",
        type=Path,
        default=None,
        help=(
            "Optional upstream-centered/gated/Top-K residual with effects, "
            "target_names, gene_names, direct_mask, and fallback_mask."
        ),
    )
    parser.add_argument(
        "--residual-alpha",
        type=float,
        default=0.0,
        help="Additive residual multiplier. At zero, residual-npz is not opened.",
    )
    parser.add_argument(
        "--combined-effect-clip",
        type=float,
        default=0.80,
        help="Final symmetric clip after adding the optional residual.",
    )
    parser.add_argument(
        "--target-policy",
        choices=("off", "force"),
        default="force",
        help="Apply the model effect, or force a CRISPRi target fold after final clipping.",
    )
    parser.add_argument("--target-remaining-fraction", type=float, default=0.20)
    parser.add_argument(
        "--max-genes-per-cell", type=int, default=DEFAULT_MAX_GENES_PER_CELL
    )
    parser.add_argument(
        "--integerization",
        choices=("largest-remainder", "multinomial"),
        default="largest-remainder",
    )
    parser.add_argument(
        "--strata-column",
        default="ntc_id",
        help="Equal-allocation control stratum; absent columns fall back to uniform sampling.",
    )
    parser.add_argument("--groups-per-shard", type=int, default=20)
    parser.add_argument("--concat-max-loaded-elements", type=int, default=20_000_000)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    require(args.controls_h5ad.is_file(), f"Missing controls H5AD: {args.controls_h5ad}")
    require(args.panel_manifest.is_file(), f"Missing panel manifest: {args.panel_manifest}")
    require(args.support_genes.is_file(), f"Missing support genes: {args.support_genes}")
    require(0 < args.model_chunk_size <= 128, "model-chunk-size must be in [1, 128]")
    require(args.cells_per_group > 0, "cells-per-group must be positive")
    require(args.target_limit >= 0, "target-limit cannot be negative")
    require(np.isfinite(args.state_effect_weight), "Invalid state-effect-weight")
    require(args.state_effect_clip > 0, "state-effect-clip must be positive")
    require(np.isfinite(args.residual_alpha) and args.residual_alpha >= 0, "Invalid residual-alpha")
    require(args.combined_effect_clip > 0, "combined-effect-clip must be positive")
    require(
        0 < args.target_remaining_fraction <= 1,
        "target-remaining-fraction must be in (0, 1]",
    )
    if args.target_policy == "force":
        require(
            args.target_remaining_fraction < 1,
            "target-policy force requires a remaining fraction below one",
        )
    require(args.max_genes_per_cell > 0, "max-genes-per-cell must be positive")
    require(args.groups_per_shard > 0, "groups-per-shard must be positive")
    require(args.concat_max_loaded_elements > 0, "Invalid concat memory bound")
    require(args.output_h5ad.suffix == ".h5ad", "output-h5ad must end in .h5ad")
    require(args.output_json.suffix == ".json", "output-json must end in .json")
    require(args.output_h5ad.resolve() != args.output_json.resolve(), "Output paths differ")
    if args.residual_alpha > 0:
        require(args.residual_npz is not None, "Positive residual-alpha requires residual-npz")
    for output in (args.output_h5ad, args.output_json):
        require(not output.exists(), f"Refusing to overwrite existing artifact: {output}")


def _string_axis(values: np.ndarray, name: str) -> list[str]:
    result = [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in np.asarray(values).reshape(-1)
    ]
    require(result and all(result), f"{name} is empty")
    require(len(result) == len(set(result)), f"{name} contains duplicates")
    return result


def _validate_descriptor(path: Path, descriptor: dict[str, Any], label: str) -> None:
    require(path.is_file(), f"Missing {label}: {path}")
    require(path.stat().st_size == int(descriptor.get("size_bytes", -1)), f"{label} size changed")
    require(sha256_file(path) == str(descriptor.get("sha256", "")), f"{label} hash changed")


def load_frozen_panel(manifest_path: Path, csv_path: Path | None = None) -> FrozenPanel:
    """Authenticate and load the target axis without reading effect values."""
    resolved_manifest = manifest_path.resolve()
    payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    require(payload.get("schema") == PANEL_SCHEMA, "Bad public-panel manifest schema")
    contract = payload.get("selection_contract", {})
    require(contract.get("effect_values_accessed") is False, "Panel selection is not leakage-safe")
    descriptor = payload.get("provenance", {}).get("output_csv", {})
    require(isinstance(descriptor, dict), "Panel manifest lacks output_csv provenance")
    recorded_path = Path(str(descriptor.get("path", "")))
    resolved_csv = (csv_path if csv_path is not None else recorded_path).resolve()
    _validate_descriptor(resolved_csv, descriptor, "frozen panel CSV")

    frame = pd.read_csv(resolved_csv, dtype=str, keep_default_na=False)
    require(
        {"target_gene", "validation_route"}.issubset(frame.columns),
        "Panel CSV lacks target_gene or validation_route",
    )
    targets = frame["target_gene"].astype(str).tolist()
    routes = frame["validation_route"].astype(str).tolist()
    require(targets and all(targets), "Panel target axis is empty")
    require(len(targets) == len(set(targets)), "Panel targets contain duplicates")
    require(CONTROL_LABEL not in targets, "Panel contains the control label")
    require(set(routes).issubset({"direct", "held_target"}), "Unknown validation route")
    require(len(routes) == len(targets), "Panel route axis differs from target axis")
    require(
        len(targets) == int(contract.get("panel_targets", -1)),
        "Panel target count differs from its frozen contract",
    )
    require(
        sum(route == "held_target" for route in routes)
        == int(contract.get("held_targets", -1)),
        "Held-target count differs from its frozen contract",
    )
    require(
        sum(route == "direct" for route in routes)
        == int(contract.get("direct_targets", -1)),
        "Direct-target count differs from its frozen contract",
    )
    return FrozenPanel(tuple(targets), tuple(routes), payload, resolved_manifest, resolved_csv)


def build_gene_maps(native_genes: list[str], support_genes: list[str]) -> GeneMaps:
    """Map arbitrary unique native symbols to the frozen STATE support order."""
    require(native_genes and support_genes, "Gene axes must be non-empty")
    require(len(native_genes) == len(set(native_genes)), "Native gene axis has duplicates")
    require(len(support_genes) == len(set(support_genes)), "STATE gene axis has duplicates")
    native_index = {gene: index for index, gene in enumerate(native_genes)}
    support_to_native = np.asarray(
        [native_index.get(gene, -1) for gene in support_genes], dtype=np.int64
    )
    native_to_support = np.full(len(native_genes), -1, dtype=np.int64)
    shared = support_to_native >= 0
    native_to_support[support_to_native[shared]] = np.flatnonzero(shared)
    require(np.count_nonzero(shared) > 0, "Native axis has no STATE-supported genes")
    return GeneMaps(support_to_native=support_to_native, native_to_support=native_to_support)


def validate_controls_contract(
    controls: ad.AnnData,
    *,
    expected_context: str | None = None,
    support_genes: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Reject any H5AD that is not explicitly marked as controls-only."""
    firewall = controls.uns.get("public_validation")
    require(isinstance(firewall, dict), "Controls H5AD lacks public_validation metadata")
    require(firewall.get("schema") == DATA_SCHEMA, "Bad controls data schema")
    require(firewall.get("role") == CONTROL_ROLE, "H5AD is not a generator controls input")
    require(
        firewall.get("sealed_treated_profiles_present") is False,
        "Controls H5AD declares sealed treated profiles",
    )
    require({"context", "target_gene"}.issubset(controls.obs.columns), "Controls obs is incomplete")
    labels = set(controls.obs["target_gene"].astype(str))
    require(labels == {CONTROL_LABEL}, "Controls H5AD contains non-control rows")
    contexts = set(controls.obs["context"].astype(str))
    require(len(contexts) == 1, "Controls H5AD must contain exactly one context")
    context = next(iter(contexts))
    if expected_context is not None:
        require(context == expected_context, f"Controls context is {context}, expected {expected_context}")
    native_genes = controls.var_names.astype(str).tolist()
    require(native_genes and len(native_genes) == len(set(native_genes)), "Bad native gene axis")
    require(controls.n_obs > 0, "Controls H5AD has no cells")
    if support_genes is not None and "is_state_support" in controls.var.columns:
        support_set = set(support_genes)
        recorded = controls.var["is_state_support"].astype(bool).to_numpy()
        expected = np.asarray([gene in support_set for gene in native_genes], dtype=bool)
        require(np.array_equal(recorded, expected), "Controls STATE-support mask is stale")
    return context, native_genes


def load_residual_overlay(
    path: Path | None,
    *,
    alpha: float,
    panel: FrozenPanel,
    native_genes: list[str],
    support_genes: list[str],
    context: str | None = None,
) -> ResidualOverlay:
    """Load and strictly align a leakage-safe upstream residual artifact.

    The artifact contract is intentionally narrow: ``effects`` is ``(T,G)``
    or ``(1,T,G)``; axes are named by ``target_names`` and ``gene_names``; and
    ``direct_mask``/``fallback_mask`` partition the target axis.  Frozen held
    targets must be fallback rows and must contain exact zeros.
    """
    disabled_direct = np.asarray([route == "direct" for route in panel.routes], dtype=bool)
    disabled_fallback = ~disabled_direct
    if alpha == 0:
        return ResidualOverlay(
            effects=None,
            direct_mask=disabled_direct,
            fallback_mask=disabled_fallback,
            source=None,
            qc={
                "enabled": False,
                "alpha": 0.0,
                "artifact_read": False,
                "declared_path": str(path.resolve()) if path is not None else None,
                "generator_transform": "none",
            },
        )
    require(path is not None, "Positive residual alpha requires an artifact")
    resolved = path.resolve()
    require(resolved.is_file(), f"Residual artifact is missing: {resolved}")
    with np.load(resolved, allow_pickle=False) as archive:
        required = {"effects", "target_names", "gene_names", "direct_mask", "fallback_mask"}
        missing = sorted(required - set(archive.files))
        require(not missing, f"Residual artifact lacks keys: {missing}")
        source_targets = _string_axis(archive["target_names"], "target_names")
        source_genes = _string_axis(archive["gene_names"], "gene_names")
        if "contexts" in archive.files:
            source_contexts = _string_axis(archive["contexts"], "contexts")
            require(len(source_contexts) == 1, "Residual artifact must name one context")
            if context is not None:
                require(
                    source_contexts == [context],
                    f"Residual context is {source_contexts[0]}, expected {context}",
                )
        effects = np.asarray(archive["effects"], dtype=np.float32)
        if effects.ndim == 3:
            require(effects.shape[0] == 1, "3D residual effects must have one context")
            effects = effects[0]
        require(effects.ndim == 2, "Residual effects must have two or three dimensions")
        require(
            effects.shape == (len(source_targets), len(source_genes)),
            "Residual effects shape does not match its axes",
        )
        require(np.isfinite(effects).all(), "Residual effects contain non-finite values")
        direct_values = np.asarray(archive["direct_mask"]).reshape(-1)
        fallback_values = np.asarray(archive["fallback_mask"]).reshape(-1)
        require(
            direct_values.shape == fallback_values.shape == (len(source_targets),),
            "Residual masks have bad shapes",
        )
        require(
            np.all(np.isin(direct_values, [0, 1]))
            and np.all(np.isin(fallback_values, [0, 1])),
            "Residual masks must be binary",
        )
        source_direct = direct_values.astype(bool)
        source_fallback = fallback_values.astype(bool)
        require(np.all(source_direct ^ source_fallback), "Residual masks must be XOR")

    require(set(source_targets) == set(panel.targets), "Residual target set differs from panel")
    unknown_genes = sorted(set(source_genes) - set(native_genes))
    require(not unknown_genes, f"Residual genes are absent from native axis: {unknown_genes[:5]}")
    target_source_index = {target: index for index, target in enumerate(source_targets)}
    target_order = np.asarray([target_source_index[target] for target in panel.targets], dtype=np.int64)
    aligned_effects = effects[target_order]
    aligned_direct = source_direct[target_order]
    aligned_fallback = source_fallback[target_order]
    expected_direct = np.asarray([route == "direct" for route in panel.routes], dtype=bool)
    require(np.array_equal(aligned_direct, expected_direct), "Residual masks differ from panel routes")
    require(np.array_equal(aligned_fallback, ~expected_direct), "Fallback mask differs from panel routes")
    require(
        not np.any(aligned_effects[aligned_fallback] != 0),
        "Held/fallback residual rows must be exactly zero",
    )

    support_index = {gene: index for index, gene in enumerate(support_genes)}
    source_is_support = np.asarray([gene in support_index for gene in source_genes], dtype=bool)
    require(
        not np.any(aligned_effects[:, ~source_is_support] != 0),
        "Residual has a nonzero effect on a native non-STATE gene",
    )
    aligned_support = np.zeros((len(panel.targets), len(support_genes)), dtype=np.float32)
    if np.any(source_is_support):
        support_positions = np.asarray(
            [support_index[gene] for gene in np.asarray(source_genes)[source_is_support]],
            dtype=np.int64,
        )
        aligned_support[:, support_positions] = aligned_effects[:, source_is_support]
    nonzero = np.count_nonzero(aligned_support, axis=1)
    direct_values_flat = aligned_support[aligned_direct]
    return ResidualOverlay(
        effects=aligned_support,
        direct_mask=aligned_direct,
        fallback_mask=aligned_fallback,
        source=describe_file(resolved),
        qc={
            "enabled": True,
            "alpha": float(alpha),
            "artifact_read": True,
            "generator_transform": "name alignment plus alpha scaling only",
            "targets": len(panel.targets),
            "genes_in_artifact": len(source_genes),
            "support_genes_with_artifact_coordinates": int(np.count_nonzero(source_is_support)),
            "direct_targets": int(np.count_nonzero(aligned_direct)),
            "fallback_targets": int(np.count_nonzero(aligned_fallback)),
            "fallback_nonzero_values": int(np.count_nonzero(aligned_support[aligned_fallback])),
            "nonzero_per_target": summarize(nonzero.astype(np.float64)),
            "direct_effect_values": (
                summarize(direct_values_flat)
                if direct_values_flat.size
                else None
            ),
            "upstream_operations_not_repeated": [
                "centering",
                "reliability gating",
                "Top-K selection",
            ],
        },
    )


def combine_support_delta(
    state_delta: np.ndarray,
    residual: np.ndarray,
    *,
    state_weight: float,
    state_clip: float,
    state_bounding: str,
    residual_alpha: float,
    combined_clip: float,
    target_support_index: int,
    target_policy: str,
    target_remaining_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bound the STATE anchor, add one residual, then apply target handling."""
    state = np.asarray(state_delta, dtype=np.float32)
    overlay = np.asarray(residual, dtype=np.float32).reshape(-1)
    require(state.ndim == 2 and state.shape[1] == len(overlay), "Effect shapes differ")
    require(np.isfinite(state).all() and np.isfinite(overlay).all(), "Non-finite effect")
    require(0 <= target_support_index < state.shape[1], "Bad target support index")
    scaled = state * np.float32(state_weight)
    if state_bounding == "tanh":
        anchor = np.float32(state_clip) * np.tanh(scaled / np.float32(state_clip))
    elif state_bounding == "clip":
        anchor = np.clip(scaled, -state_clip, state_clip)
    else:
        raise RuntimeError(f"Unsupported STATE bounding: {state_bounding}")
    before_clip = anchor + np.float32(residual_alpha) * overlay[None, :]
    combined = np.clip(before_clip, -combined_clip, combined_clip).astype(np.float32, copy=False)
    modeled_target = combined[:, target_support_index].copy()
    if target_policy == "force":
        combined[:, target_support_index] = np.float32(math.log(target_remaining_fraction))
    elif target_policy != "off":
        raise RuntimeError(f"Unsupported target policy: {target_policy}")
    require(np.isfinite(combined).all(), "Combined effects are non-finite")
    return combined, {
        "values": int(combined.size),
        "state_outside_bound": int(np.count_nonzero(np.abs(scaled) > state_clip)),
        "combined_outside_bound": int(np.count_nonzero(np.abs(before_clip) > combined_clip)),
        "modeled_target_delta": summarize(modeled_target),
        "final_target_delta": summarize(combined[:, target_support_index]),
        "final_target_fold": summarize(np.exp(combined[:, target_support_index])),
    }


def render_exact_native_counts(
    raw: sp.csr_matrix,
    support_delta: np.ndarray,
    native_to_support: np.ndarray,
    *,
    max_genes_per_cell: int,
    integerization: str,
    rng: np.random.Generator,
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Render exact-library rows while checking native non-support invariance."""
    base = raw.tocsr()
    delta = np.asarray(support_delta, dtype=np.float32)
    mapping = np.asarray(native_to_support, dtype=np.int64)
    require(delta.ndim == 2 and delta.shape[0] == base.shape[0], "Bad support delta")
    require(mapping.shape == (base.shape[1],), "Bad native-to-support map")
    require(np.all((mapping < 0) | (mapping < delta.shape[1])), "Support mapping is out of range")
    require(0 < max_genes_per_cell <= base.shape[1], "Invalid native row-nnz cap")

    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    indptr = np.zeros(base.shape[0] + 1, dtype=np.int64)
    input_libraries: list[int] = []
    output_libraries: list[int] = []
    removed_counts = 0
    removed_nnz = 0
    changed = 0
    non_support_mismatches = 0
    for row_index in range(base.shape[0]):
        row_start, row_stop = base.indptr[row_index : row_index + 2]
        base_indices = base.indices[row_start:row_stop]
        base_values = base.data[row_start:row_stop]
        output_indices, output_values, row_qc = transform_raw_row(
            base_indices,
            base_values,
            delta[row_index],
            mapping,
            max_genes_per_cell,
            integerization,
            rng,
        )
        data_parts.append(output_values)
        index_parts.append(output_indices)
        indptr[row_index + 1] = indptr[row_index] + len(output_values)
        input_libraries.append(int(row_qc["source_library"]))
        output_libraries.append(int(row_qc["output_library"]))
        removed_counts += int(row_qc["removed_shared_counts"])
        removed_nnz += int(row_qc["removed_shared_nnz"])
        changed += int(bool(row_qc["changed"]))
        base_only = mapping[base_indices] < 0
        output_only = mapping[output_indices] < 0
        if not (
            np.array_equal(base_indices[base_only], output_indices[output_only])
            and np.array_equal(base_values[base_only], output_values[output_only])
        ):
            non_support_mismatches += 1

    require(indptr[-1] < np.iinfo(np.int32).max, "One group exceeds int32 CSR capacity")
    data = np.concatenate(data_parts).astype(np.int32, copy=False) if data_parts else np.empty(0, np.int32)
    indices = (
        np.concatenate(index_parts).astype(np.int32, copy=False)
        if index_parts
        else np.empty(0, np.int32)
    )
    matrix = sp.csr_matrix(
        (data, indices, indptr.astype(np.int32)), shape=base.shape, dtype=np.int32
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    require(matrix.has_canonical_format, "Generated native group is not canonical CSR")
    require(input_libraries == output_libraries, "A generated row changed library size")
    require(non_support_mismatches == 0, "A native non-support count changed")
    return matrix, {
        "input_library": summarize(np.asarray(input_libraries)),
        "output_library": summarize(np.asarray(output_libraries)),
        "input_nnz": summarize(np.diff(base.indptr)),
        "output_nnz": summarize(np.diff(matrix.indptr)),
        "changed_cell_fraction": float(changed / max(base.shape[0], 1)),
        "library_size_mismatches": 0,
        "native_non_support_count_mismatches": non_support_mismatches,
        "removed_shared_nnz": removed_nnz,
        "redistributed_shared_counts": removed_counts,
        "redistributed_count_fraction": float(
            removed_counts / max(sum(input_libraries), 1)
        ),
        "zero_entry_policy": "source-zero coordinates remain zero",
    }


def build_native_group(
    raw: sp.csr_matrix,
    model: Any,
    target: str,
    target_embedding: torch.Tensor,
    control_embedding: torch.Tensor,
    target_native_index: int,
    target_support_index: int,
    support_to_native: np.ndarray,
    native_to_support: np.ndarray,
    residual: np.ndarray,
    device: torch.device,
    count_rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Infer and render one context-target group on the native gene axis."""
    blocks: list[sp.csr_matrix] = []
    raw_state_moments = StreamingMoments()
    final_delta_moments = StreamingMoments()
    state_outside = 0
    combined_outside = 0
    effect_values = 0
    target_before = 0
    target_after = 0
    chunk_sizes: list[int] = []
    prediction_ranges: list[dict[str, float]] = []
    render_qc: list[dict[str, Any]] = []
    target_model_values: list[np.ndarray] = []
    target_final_values: list[np.ndarray] = []

    for start in range(0, raw.shape[0], args.model_chunk_size):
        stop = min(start + args.model_chunk_size, raw.shape[0])
        chunk = raw[start:stop]
        chunk_sizes.append(stop - start)
        model_input = make_model_input(chunk, support_to_native)
        with torch.inference_mode():
            state_delta, range_qc = state_paired_delta(
                model,
                model_input,
                target_embedding,
                control_embedding,
                target,
                device,
            )
        prediction_ranges.append(range_qc)
        raw_state_moments.update(state_delta)
        final_delta, effect_qc = combine_support_delta(
            state_delta,
            residual,
            state_weight=args.state_effect_weight,
            state_clip=args.state_effect_clip,
            state_bounding=args.state_effect_bounding,
            residual_alpha=args.residual_alpha,
            combined_clip=args.combined_effect_clip,
            target_support_index=target_support_index,
            target_policy=args.target_policy,
            target_remaining_fraction=args.target_remaining_fraction,
        )
        final_delta_moments.update(final_delta)
        state_outside += int(effect_qc["state_outside_bound"])
        combined_outside += int(effect_qc["combined_outside_bound"])
        effect_values += int(effect_qc["values"])
        target_model_values.append(
            np.asarray(effect_qc["modeled_target_delta"]["mean"], dtype=np.float32).reshape(1)
        )
        target_final_values.append(final_delta[:, target_support_index].copy())
        emitted, chunk_qc = render_exact_native_counts(
            chunk,
            final_delta,
            native_to_support,
            max_genes_per_cell=args.max_genes_per_cell,
            integerization=args.integerization,
            rng=count_rng,
        )
        blocks.append(emitted)
        render_qc.append(chunk_qc)
        target_before += int(chunk[:, target_native_index].sum())
        target_after += int(emitted[:, target_native_index].sum())
        del model_input, state_delta, final_delta, emitted

    matrix = sp.vstack(blocks, format="csr", dtype=np.int32)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    require(matrix.shape == raw.shape, "Generated group changed native shape")
    require(all(item["library_size_mismatches"] == 0 for item in render_qc), "Library drift")
    require(
        all(item["native_non_support_count_mismatches"] == 0 for item in render_qc),
        "Native non-support drift",
    )
    input_libraries = np.asarray(raw.sum(axis=1)).ravel().astype(np.int64)
    output_libraries = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)
    require(np.array_equal(input_libraries, output_libraries), "Group libraries changed")
    return matrix, {
        "cells": int(raw.shape[0]),
        "chunk_sizes": chunk_sizes,
        "paired_passes": 2 * len(chunk_sizes),
        "input_normalization": "elementwise-log1p-raw-on-STATE-support-axis",
        "delta_definition": "state(target, basal)-state(non-targeting, identical basal)",
        "raw_state_delta": raw_state_moments.report(),
        "final_support_delta": final_delta_moments.report(),
        "state_outside_bound": state_outside,
        "combined_outside_bound": combined_outside,
        "effect_values": effect_values,
        "prediction_ranges": prediction_ranges,
        "input_library": summarize(input_libraries),
        "output_library": summarize(output_libraries),
        "nnz": int(matrix.nnz),
        "changed_cell_fraction": float(
            np.average(
                [item["changed_cell_fraction"] for item in render_qc],
                weights=chunk_sizes,
            )
        ),
        "library_size_mismatches": 0,
        "native_non_support_count_mismatches": 0,
        "redistributed_shared_counts": int(
            sum(item["redistributed_shared_counts"] for item in render_qc)
        ),
        "redistributed_count_fraction": float(
            sum(item["redistributed_shared_counts"] for item in render_qc)
            / max(int(input_libraries.sum()), 1)
        ),
        "target_modeled_delta_chunk_means": summarize(np.concatenate(target_model_values)),
        "final_target_delta": summarize(np.concatenate(target_final_values)),
        "target_sum_before": target_before,
        "target_sum_after": target_after,
        "target_remaining_fraction": float(target_after / max(target_before, 1)),
    }


def sample_control_indices(
    obs: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
    *,
    strata_column: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Sample controls without replacement, stratifying when metadata permits."""
    require(0 < count <= len(obs), "Invalid number of sampled controls")
    if strata_column not in obs.columns:
        chosen = rng.choice(len(obs), size=count, replace=False).astype(np.int64)
        return chosen, {"__uniform__": count}
    values = obs[strata_column].astype(str).to_numpy()
    strata = sorted(set(values.tolist()))
    require(strata, "Control stratum column is empty")
    base, remainder = divmod(count, len(strata))
    sampled: list[int] = []
    allocation: dict[str, int] = {}
    for ordinal, stratum in enumerate(strata):
        take = base + int(ordinal < remainder)
        candidates = np.flatnonzero(values == stratum)
        require(take <= len(candidates), f"Control stratum {stratum} has too few cells")
        if take:
            sampled.extend(rng.choice(candidates, size=take, replace=False).astype(int).tolist())
        allocation[stratum] = take
    selected = np.asarray(sampled, dtype=np.int64)
    rng.shuffle(selected)
    require(len(selected) == count, "Stratified sampling returned the wrong cell count")
    return selected, allocation


def write_group_h5ad(
    path: Path,
    matrix: sp.csr_matrix,
    var: pd.DataFrame,
    *,
    context: str,
    target: str,
    selected_rows: np.ndarray,
    selected_names: list[str],
    seed: int,
    compression: str,
) -> None:
    obs = pd.DataFrame(
        {
            "context": context,
            "target_gene": target,
            "source_control_row": selected_rows.astype(np.int64),
            "source_control_cell_id": selected_names,
        },
        index=pd.Index(
            [f"{context}|{target}|{cell:04d}" for cell in range(len(selected_rows))],
            name="cell_id",
        ),
    )
    block = ad.AnnData(X=matrix, obs=obs, var=var.copy())
    block.uns["native_state_anchor"] = {
        "schema": SCHEMA,
        "context": context,
        "seed": seed,
        "source_role": CONTROL_ROLE,
        "sealed_treated_profiles_read": False,
    }
    block.write_h5ad(path, compression=None if compression == "none" else compression)


def validate_native_candidate(
    path: Path,
    *,
    genes: list[str],
    context: str,
    targets: list[str],
    cells_per_group: int,
    expected_nnz: int,
) -> dict[str, Any]:
    candidate = ad.read_h5ad(path, backed="r")
    try:
        expected_shape = (len(targets) * cells_per_group, len(genes))
        required_obs = {"context", "target_gene", "source_control_row"}
        require(required_obs.issubset(candidate.obs.columns), "Candidate obs is incomplete")
        group_sizes = candidate.obs.groupby("target_gene", observed=True).size()
        matrix = candidate.X
        require(
            str(getattr(matrix, "format", "")) == "csr",
            "Candidate matrix is not backed CSR",
        )
        observed_nnz = int(matrix.group["data"].shape[0])
        data_dtype = np.dtype(matrix.group["data"].dtype)
        checks = {
            "shape_exact": tuple(candidate.shape) == expected_shape,
            "native_gene_axis_exact": candidate.var_names.astype(str).tolist() == genes,
            "obs_names_unique": bool(candidate.obs_names.is_unique),
            "single_context_exact": set(candidate.obs["context"].astype(str)) == {context},
            "target_set_exact": set(candidate.obs["target_gene"].astype(str)) == set(targets),
            "groups_exact": len(group_sizes) == len(targets),
            "cells_per_group_exact": bool((group_sizes == cells_per_group).all()),
            "no_controls_emitted": CONTROL_LABEL not in set(candidate.obs["target_gene"].astype(str)),
            "csr_encoding": str(matrix.format) == "csr",
            "integer_count_dtype": bool(np.issubdtype(data_dtype, np.integer)),
            "nnz_matches_stream": observed_nnz == expected_nnz,
            "nnz_safe_int32": observed_nnz < SAFE_INT32_NNZ,
        }
    finally:
        candidate.file.close()
    failures = sorted(name for name, passed in checks.items() if not passed)
    require(not failures, f"Native candidate validation failed: {failures}")
    return {
        "checks": checks,
        "failed_checks": failures,
        "shape": list(expected_shape),
        "groups": len(group_sizes),
        "nnz": observed_nnz,
        "mean_nnz_per_cell": float(observed_nnz / expected_shape[0]),
        "data_dtype": str(data_dtype),
    }


def _device(args: argparse.Namespace) -> tuple[torch.device, str]:
    device = torch.device(args.device)
    if args.require_cuda:
        require(device.type == "cuda", "--require-cuda requires a CUDA device")
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        name = torch.cuda.get_device_name(device)
        if args.require_h100:
            require("H100" in name.upper(), f"Selected GPU is not an H100: {name}")
    else:
        require(not args.require_h100, "--require-h100 requires CUDA")
        name = "CPU"
    return device, name


def main() -> None:
    started = time.time()
    args = parse_args()
    validate_args(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False

    panel = load_frozen_panel(args.panel_manifest, args.panel_csv)
    support_path = args.support_genes.resolve()
    support_genes = read_single_column(support_path)
    require(len(support_genes) == EXPECTED_SUPPORT_GENES, "Unexpected STATE support size")

    controls_path = args.controls_h5ad.resolve()
    controls = ad.read_h5ad(controls_path, backed="r")
    final_temporary = args.output_h5ad.with_name(
        f".{args.output_h5ad.name}.{os.getpid()}.tmp.h5ad"
    )
    try:
        context, native_genes = validate_controls_contract(
            controls,
            expected_context=args.context,
            support_genes=support_genes,
        )
        require(
            args.max_genes_per_cell <= len(native_genes),
            "max-genes-per-cell exceeds the native gene count",
        )
        require(
            args.cells_per_group <= controls.n_obs,
            "Not enough controls for sampling without replacement",
        )
        gene_maps = build_gene_maps(native_genes, support_genes)
        native_index = {gene: index for index, gene in enumerate(native_genes)}
        support_index = {gene: index for index, gene in enumerate(support_genes)}
        require(
            all(target in native_index for target in panel.targets),
            "A panel target is absent from the native gene axis",
        )
        require(
            all(target in support_index for target in panel.targets),
            "A panel target is absent from STATE support",
        )
        if args.target_limit:
            require(args.target_limit <= len(panel.targets), "target-limit exceeds panel size")
            targets = list(panel.targets[: args.target_limit])
        else:
            targets = list(panel.targets)
        theoretical_nnz = len(targets) * args.cells_per_group * args.max_genes_per_cell
        require(theoretical_nnz < SAFE_INT32_NNZ, "Theoretical candidate nnz is not int32-safe")

        residual = load_residual_overlay(
            args.residual_npz,
            alpha=args.residual_alpha,
            panel=panel,
            native_genes=native_genes,
            support_genes=support_genes,
            context=context,
        )

        model_dir = args.model_dir.resolve()
        checkpoint = resolve_checkpoint(model_dir, args.checkpoint)
        selection_path = (
            args.selection_json
            if args.selection_json is not None
            else model_dir / "checkpoints" / "selected_checkpoint.json"
        ).resolve()
        pert_map_path = model_dir / "pert_onehot_map.pt"
        var_dims_path = model_dir / "var_dims.pkl"
        config_path = model_dir / "config.yaml"
        for path in (checkpoint, selection_path, pert_map_path, var_dims_path):
            require(path.is_file(), f"Missing required STATE input: {path}")
        checkpoint_hash = sha256_file(checkpoint)
        selection = validate_selection_manifest(selection_path, checkpoint, checkpoint_hash)
        var_dims = load_var_dims(var_dims_path, support_genes)
        embeddings = load_perturbation_embeddings(pert_map_path, list(panel.targets))
        device, device_name = _device(args)
        model = load_state_model(checkpoint, device)
        model_info = validate_model(model, args.model_chunk_size)
        control_embedding = embeddings[CONTROL_LABEL].to(device=device, dtype=torch.float32)

        args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        scratch_parent = args.scratch_dir.resolve() if args.scratch_dir else None
        if scratch_parent is not None:
            scratch_parent.mkdir(parents=True, exist_ok=True)
        require(not final_temporary.exists(), f"Temporary output exists: {final_temporary}")

        var = controls.var.copy()
        group_qc: list[dict[str, Any]] = []
        total_nnz = 0
        final_shards: list[Path] = []
        with tempfile.TemporaryDirectory(prefix="vcc_native_state_", dir=scratch_parent) as scratch:
            scratch_path = Path(scratch)
            pending_blocks: list[Path] = []
            for target_index, target in enumerate(targets):
                panel_target_index = panel.targets.index(target)
                sample_seed = deterministic_seed(args.seed, context, target, "sample")
                count_seed = deterministic_seed(args.seed, context, target, "counts")
                sample_rng = np.random.default_rng(sample_seed)
                count_rng = np.random.default_rng(count_seed)
                selected_rows, allocation = sample_control_indices(
                    controls.obs,
                    args.cells_per_group,
                    sample_rng,
                    strata_column=args.strata_column,
                )
                raw = read_control_rows(controls, selected_rows)
                target_embedding = embeddings[target].to(device=device, dtype=torch.float32)
                block, qc = build_native_group(
                    raw,
                    model,
                    target,
                    target_embedding,
                    control_embedding,
                    native_index[target],
                    support_index[target],
                    gene_maps.support_to_native,
                    gene_maps.native_to_support,
                    residual.target(panel_target_index, len(support_genes)),
                    device,
                    count_rng,
                    args,
                )
                qc.update(
                    {
                        "context": context,
                        "target_gene": target,
                        "target_ordinal": target_index,
                        "panel_ordinal": panel_target_index,
                        "validation_route": panel.routes[panel_target_index],
                        "residual_nonzero": int(
                            np.count_nonzero(residual.target(panel_target_index, len(support_genes)))
                        ),
                        "source_control_index_sha256": hashlib.sha256(
                            selected_rows.tobytes()
                        ).hexdigest(),
                        "sample_seed": sample_seed,
                        "count_seed": count_seed,
                        "control_strata": len(allocation),
                        "control_strata_used": int(sum(value > 0 for value in allocation.values())),
                    }
                )
                group_qc.append(qc)
                total_nnz += int(block.nnz)
                require(total_nnz < SAFE_INT32_NNZ, "Streaming nnz exceeds int32 safety bound")
                block_path = scratch_path / f"{target_index:03d}_{target}_block.h5ad"
                write_group_h5ad(
                    block_path,
                    block,
                    var,
                    context=context,
                    target=target,
                    selected_rows=selected_rows,
                    selected_names=controls.obs_names[selected_rows].astype(str).tolist(),
                    seed=args.seed,
                    compression=args.compression,
                )
                pending_blocks.append(block_path)
                del raw, block, target_embedding

                if len(pending_blocks) == args.groups_per_shard or target_index + 1 == len(targets):
                    shard_path = scratch_path / f"shard_{len(final_shards):03d}.h5ad"
                    concatenate_paths(pending_blocks, shard_path, args.concat_max_loaded_elements)
                    for block_path_to_remove in pending_blocks:
                        block_path_to_remove.unlink()
                    pending_blocks.clear()
                    final_shards.append(shard_path)
                if (target_index + 1) % 10 == 0 or target_index + 1 == len(targets):
                    print(
                        f"[{context}] generated {target_index + 1}/{len(targets)} targets; "
                        f"cumulative nnz={total_nnz:,}",
                        flush=True,
                    )

            concatenate_paths(final_shards, final_temporary, args.concat_max_loaded_elements)
            validation = validate_native_candidate(
                final_temporary,
                genes=native_genes,
                context=context,
                targets=targets,
                cells_per_group=args.cells_per_group,
                expected_nnz=total_nnz,
            )
            os.replace(final_temporary, args.output_h5ad)

        failures = [
            item["target_gene"]
            for item in group_qc
            if item["library_size_mismatches"]
            or item["native_non_support_count_mismatches"]
        ]
        require(not failures, f"Exact-count invariants failed for targets: {failures[:5]}")
        target_failures = [
            item["target_gene"]
            for item in group_qc
            if args.target_policy == "force"
            and item["target_sum_before"] >= 20
            and item["target_remaining_fraction"] > 0.50
        ]
        require(not target_failures, f"Target-knockdown QC failed: {target_failures[:5]}")

        elapsed = time.time() - started
        report = {
            "schema": SCHEMA,
            "artifact_type": "sealed_public_validation_prediction",
            "created_utc_epoch": time.time(),
            "elapsed_seconds": elapsed,
            "command": [sys.executable, *sys.argv],
            "full_frozen_panel_contract": len(targets) == len(panel.targets),
            "data_firewall": {
                "accepted_cell_data_role": CONTROL_ROLE,
                "controls_only_h5ad": True,
                "sealed_treated_profiles_read": False,
                "truth_inputs_read": [],
                "generator_cli_has_truth_argument": False,
                "panel_selection_effect_values_accessed": False,
            },
            "configuration": {
                "seed": args.seed,
                "device": str(device),
                "device_name": device_name,
                "precision": "float32",
                "context": context,
                "cells_per_group": args.cells_per_group,
                "targets": len(targets),
                "model_chunk_size": args.model_chunk_size,
                "state_effect_weight": args.state_effect_weight,
                "state_effect_clip": args.state_effect_clip,
                "state_effect_bounding": args.state_effect_bounding,
                "residual_alpha": args.residual_alpha,
                "combined_effect_clip": args.combined_effect_clip,
                "target_policy": args.target_policy,
                "target_remaining_fraction": args.target_remaining_fraction,
                "max_genes_per_cell": args.max_genes_per_cell,
                "integerization": args.integerization,
                "strata_column": args.strata_column,
                "library_policy": "exact-source-library-per-cell",
                "native_non_support_policy": "copy-sparse-indices-and-counts-exactly",
                "zero_entry_policy": "source-zero coordinates remain zero",
                "residual_policy": "upstream-centered/reliability-gated/Top-K; alpha-scale-only here",
            },
            "axes": {
                "context": context,
                "frozen_panel_targets": len(panel.targets),
                "generated_targets": len(targets),
                "native_genes": len(native_genes),
                "state_support_genes": len(support_genes),
                "shared_genes": int(np.count_nonzero(gene_maps.support_to_native >= 0)),
                "native_non_support_genes_preserved": int(
                    np.count_nonzero(gene_maps.native_to_support < 0)
                ),
                "support_only_genes_zero_filled_for_state": int(
                    np.count_nonzero(gene_maps.support_to_native < 0)
                ),
                "perturbation_embedding_dimension": int(var_dims["pert_dim"]),
            },
            "model": model_info,
            "checkpoint_selection": {
                "global_step": int(selection["selected"]["global_step"]),
                "val_loss": float(selection["selected"]["val_loss"]),
                "rule": selection["selection_rule"],
            },
            "residual": residual.qc,
            "validation": validation,
            "scientific_qc": {
                "all_libraries_exact": True,
                "all_native_non_support_counts_exact": True,
                "target_knockdown_failures": target_failures,
                "groups_with_source_target_count_at_least_20": int(
                    sum(item["target_sum_before"] >= 20 for item in group_qc)
                ),
                "target_remaining_fraction": summarize(
                    np.asarray([item["target_remaining_fraction"] for item in group_qc])
                ),
                "redistributed_count_fraction": summarize(
                    np.asarray([item["redistributed_count_fraction"] for item in group_qc])
                ),
            },
            "qc": {"groups": group_qc},
            "provenance": {
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "repository_commit": git_commit(Path(__file__).resolve().parents[1]),
                "state_source_commit": git_commit(Path(inspect.getfile(type(model))).resolve().parent),
                "script": describe_file(Path(__file__).resolve()),
                "controls_h5ad": describe_file(controls_path),
                "panel_manifest": describe_file(panel.manifest_path),
                "panel_csv": describe_file(panel.csv_path),
                "support_gene_axis": describe_file(support_path),
                "checkpoint": {
                    **describe_file(checkpoint, hash_file=False),
                    "sha256": checkpoint_hash,
                },
                "checkpoint_selection": describe_file(selection_path),
                "perturbation_map": describe_file(pert_map_path),
                "var_dims": describe_file(var_dims_path),
                "state_config": describe_file(config_path) if config_path.is_file() else None,
                "residual_artifact": residual.source,
                "output_h5ad": describe_file(args.output_h5ad),
            },
        }
        atomic_write_json(args.output_json, report)
        print(
            f"Wrote native-axis candidate {args.output_h5ad}; "
            f"shape={validation['shape']}, elapsed={elapsed / 60:.1f} min",
            flush=True,
        )
    finally:
        controls.file.close()
        if final_temporary.exists():
            final_temporary.unlink()


if __name__ == "__main__":
    main()
