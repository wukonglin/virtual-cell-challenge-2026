#!/usr/bin/env python3
"""Generate scorer-aware VCC counts from STATE and optional response priors.

This v2 adapter fixes a consequential property of the original direct-count
adapter: it never renormalizes a target-gene change over the other genes.  Each
gene receives an independent expected count and is integerized independently.
The target gene is not forced by default because VCC 2026 excludes the perturbed
gene from every metric.  An optional target clamp changes only that coordinate.

Optional NPZ artifacts can add a common log-fold response, a learned
target-specific log-fold response, and an absolute pseudobulk mean.  A
pseudobulk mean is the only mechanism that can activate a gene that is zero in
the sampled control cell.  This explicit contract prevents an accidental dense
background from being invented by a log-fold model.

Response NPZ contract
---------------------
``genes`` is required and may be a subset of the challenge axis.  Optional keys
are ``contexts``, ``targets``, ``common_log_fold``, and ``learned_log_fold``.
``common_log_fold`` has shape ``(genes,)`` or ``(contexts, genes)``.
``learned_log_fold`` has shape ``(targets, genes)`` or
``(contexts, targets, genes)``.  Missing contexts or targets receive zero.

Pseudobulk NPZ contract
-----------------------
``genes`` and ``pseudobulk_mean`` are required.  The mean has shape
``(targets, genes)`` or ``(contexts, targets, genes)`` and is expressed as
non-negative expected raw counts per cell.  Missing groups receive no mean.

The implementation writes bounded H5AD shards and never overwrites an existing
artifact.  All random operations use deterministic group-specific generators.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
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
    MAX_OFFICIAL_CELL_LIBRARY,
    SAFE_INT32_NNZ,
    StreamingMoments,
    atomic_write_json,
    concatenate_paths,
    deterministic_seed,
    make_model_input,
    read_control_rows,
    state_paired_delta,
    summarize,
    validate_backed_candidate,
    validate_model,
    validate_selection_manifest,
)
from infer_state_effect_prior import (  # noqa: E402
    CONTEXTS,
    CONTROL_LABEL,
    EXPECTED_CURRENT_GENES,
    EXPECTED_CURRENT_ONLY_GENES,
    EXPECTED_OVERLAP_GENES,
    EXPECTED_SUPPORT_GENES,
    EXPECTED_TARGETS,
    describe_file,
    git_commit,
    load_perturbation_embeddings,
    load_state_model,
    load_var_dims,
    read_single_column,
    read_targets,
    require,
    resolve_checkpoint,
    sha256_file,
    stratified_control_indices,
)


SCHEMA = "vcc-state-scorer-aware-counts-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scorer-aware raw-count predictions from paired STATE residuals."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("selected.ckpt"))
    parser.add_argument("--selection-json", type=Path, default=None)
    parser.add_argument("--controls-dir", type=Path, default=Path("dataset/controls"))
    parser.add_argument(
        "--support-genes",
        type=Path,
        default=Path("dataset/state_support/extracted/gene_names.csv"),
    )
    parser.add_argument(
        "--output-h5ad", type=Path, default=Path("artifacts/state_scorer_aware_v2.h5ad")
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("artifacts/state_scorer_aware_v2.json")
    )
    parser.add_argument("--scratch-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-h100", action="store_true")
    parser.add_argument("--model-chunk-size", type=int, default=128)
    parser.add_argument("--cells-per-group", type=int, default=400)
    parser.add_argument("--target-limit", type=int, default=0)

    parser.add_argument("--state-effect-weight", type=float, default=1.0)
    parser.add_argument("--state-effect-clip", type=float, default=0.60)
    parser.add_argument(
        "--state-effect-bounding", choices=("tanh", "clip"), default="tanh"
    )
    parser.add_argument(
        "--response-npz",
        type=Path,
        default=None,
        help="Optional aligned common and learned log-fold response artifact.",
    )
    parser.add_argument("--common-response-weight", type=float, default=0.0)
    parser.add_argument("--learned-response-weight", type=float, default=0.0)
    parser.add_argument(
        "--combined-effect-clip",
        type=float,
        default=0.80,
        help="Final symmetric log-fold bound after response addition.",
    )

    parser.add_argument(
        "--pseudobulk-npz",
        type=Path,
        default=None,
        help="Optional artifact containing absolute target pseudobulk means.",
    )
    parser.add_argument("--pseudobulk-blend-weight", type=float, default=0.0)
    parser.add_argument(
        "--zero-induction-scale",
        type=float,
        default=0.0,
        help="Scale pseudobulk mass on genes absent from a source cell.",
    )
    parser.add_argument(
        "--pseudobulk-depth-policy",
        choices=("absolute", "source-library"),
        default="source-library",
        help="Use artifact means as-is or match their total to each source library.",
    )

    parser.add_argument(
        "--target-policy",
        choices=("off", "clamp"),
        default="off",
        help="Default off; clamp changes only the perturbed-gene coordinate.",
    )
    parser.add_argument("--target-max-remaining-fraction", type=float, default=0.20)
    parser.add_argument(
        "--count-emission",
        choices=("round", "stochastic-round", "poisson", "negative-binomial"),
        default="round",
        help="Independent per-gene integer emission; no multinomial normalization.",
    )
    parser.add_argument(
        "--nb-dispersion",
        type=float,
        default=20.0,
        help="Gamma-Poisson dispersion; variance is mean + mean^2/dispersion.",
    )
    parser.add_argument(
        "--max-genes-per-cell", type=int, default=DEFAULT_MAX_GENES_PER_CELL
    )
    parser.add_argument("--groups-per-shard", type=int, default=20)
    parser.add_argument("--concat-max-loaded-elements", type=int, default=20_000_000)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    require(0 < args.model_chunk_size <= 128, "model-chunk-size must be in [1, 128]")
    require(args.cells_per_group > 0, "cells-per-group must be positive")
    require(args.target_limit >= 0, "target-limit cannot be negative")
    require(np.isfinite(args.state_effect_weight), "Invalid state-effect-weight")
    require(args.state_effect_clip > 0, "state-effect-clip must be positive")
    require(args.combined_effect_clip > 0, "combined-effect-clip must be positive")
    require(np.isfinite(args.common_response_weight), "Invalid common-response-weight")
    require(np.isfinite(args.learned_response_weight), "Invalid learned-response-weight")
    require(
        0 <= args.pseudobulk_blend_weight <= 1,
        "pseudobulk-blend-weight must be in [0, 1]",
    )
    require(args.zero_induction_scale >= 0, "zero-induction-scale cannot be negative")
    require(
        0 < args.target_max_remaining_fraction <= 1,
        "target-max-remaining-fraction must be in (0, 1]",
    )
    require(args.nb_dispersion > 0, "nb-dispersion must be positive")
    require(
        0 < args.max_genes_per_cell <= EXPECTED_CURRENT_GENES,
        "Invalid max-genes-per-cell",
    )
    require(args.groups_per_shard > 0, "groups-per-shard must be positive")
    require(args.concat_max_loaded_elements > 0, "Invalid concat memory bound")
    require(args.output_h5ad.suffix == ".h5ad", "output-h5ad must end in .h5ad")
    require(args.output_json.suffix == ".json", "output-json must end in .json")
    require(args.output_h5ad.resolve() != args.output_json.resolve(), "Output paths differ")
    if args.common_response_weight or args.learned_response_weight:
        require(args.response_npz is not None, "Response weights require --response-npz")
    if args.pseudobulk_blend_weight or args.zero_induction_scale:
        require(args.pseudobulk_npz is not None, "Pseudobulk controls require --pseudobulk-npz")
        require(
            args.pseudobulk_blend_weight > 0,
            "zero induction requires a positive pseudobulk blend weight",
        )
    for output in (args.output_h5ad, args.output_json):
        require(not output.exists(), f"Refusing to overwrite existing artifact: {output}")


def _string_axis(archive: Any, key: str, required: bool) -> list[str] | None:
    if key not in archive.files:
        require(not required, f"NPZ artifact is missing {key}")
        return None
    values = [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in np.asarray(archive[key]).reshape(-1)
    ]
    require(values and all(values), f"NPZ axis {key} is empty")
    require(len(values) == len(set(values)), f"NPZ axis {key} has duplicates")
    return values


def _axis_positions(source: list[str], destination: list[str], axis_name: str) -> np.ndarray:
    destination_index = {value: index for index, value in enumerate(destination)}
    unknown = sorted(set(source) - set(destination_index))
    require(not unknown, f"Unknown {axis_name} values in NPZ: {unknown[:5]}")
    return np.asarray([destination_index[value] for value in source], dtype=np.int64)


@dataclass(frozen=True)
class ResponseBundle:
    """Dense aligned response arrays with explicit group availability masks."""

    common_log_fold: np.ndarray | None
    learned_log_fold: np.ndarray | None
    learned_available: np.ndarray
    source: dict[str, Any] | None

    def group(
        self, context_index: int, target_index: int, genes: int
    ) -> tuple[np.ndarray, np.ndarray]:
        common = (
            np.zeros(genes, dtype=np.float32)
            if self.common_log_fold is None
            else self.common_log_fold[context_index]
        )
        learned = (
            np.zeros(genes, dtype=np.float32)
            if self.learned_log_fold is None
            or not self.learned_available[context_index, target_index]
            else self.learned_log_fold[context_index, target_index]
        )
        return common, learned


@dataclass(frozen=True)
class PseudobulkBundle:
    mean: np.ndarray | None
    available: np.ndarray
    source: dict[str, Any] | None

    def group(self, context_index: int, target_index: int) -> np.ndarray | None:
        if self.mean is None or not self.available[context_index, target_index]:
            return None
        return self.mean[context_index, target_index]


def load_response_bundle(
    path: Path | None,
    challenge_genes: list[str],
    challenge_targets: list[str],
) -> ResponseBundle:
    if path is None:
        return ResponseBundle(
            common_log_fold=None,
            learned_log_fold=None,
            learned_available=np.zeros((len(CONTEXTS), len(challenge_targets)), dtype=bool),
            source=None,
        )
    resolved = path.resolve()
    require(resolved.is_file(), f"Response artifact is missing: {resolved}")
    with np.load(resolved, allow_pickle=False) as archive:
        genes = _string_axis(archive, "genes", required=True)
        assert genes is not None
        gene_positions = _axis_positions(genes, challenge_genes, "gene")
        context_values = _string_axis(archive, "contexts", required=False)
        target_values = _string_axis(archive, "targets", required=False)
        context_positions = (
            np.arange(len(CONTEXTS), dtype=np.int64)
            if context_values is None
            else _axis_positions(context_values, list(CONTEXTS), "context")
        )
        target_positions = (
            None
            if target_values is None
            else _axis_positions(target_values, challenge_targets, "target")
        )

        common: np.ndarray | None = None
        if "common_log_fold" in archive.files:
            source_common = np.asarray(archive["common_log_fold"], dtype=np.float32)
            common = np.zeros((len(CONTEXTS), len(challenge_genes)), dtype=np.float32)
            if source_common.ndim == 1:
                require(source_common.shape == (len(genes),), "Bad common_log_fold shape")
                common[:, gene_positions] = source_common[None, :]
            elif source_common.ndim == 2:
                require(context_values is not None, "2D common_log_fold requires contexts")
                require(
                    source_common.shape == (len(context_values), len(genes)),
                    "Bad common_log_fold shape",
                )
                common[np.ix_(context_positions, gene_positions)] = source_common
            else:
                raise RuntimeError("common_log_fold must have one or two dimensions")
            require(np.isfinite(common).all(), "common_log_fold is non-finite")

        learned: np.ndarray | None = None
        available = np.zeros((len(CONTEXTS), len(challenge_targets)), dtype=bool)
        if "learned_log_fold" in archive.files:
            require(target_values is not None, "learned_log_fold requires targets")
            assert target_positions is not None
            source_learned = np.asarray(archive["learned_log_fold"], dtype=np.float32)
            learned = np.zeros(
                (len(CONTEXTS), len(challenge_targets), len(challenge_genes)),
                dtype=np.float32,
            )
            if source_learned.ndim == 2:
                require(
                    source_learned.shape == (len(target_values), len(genes)),
                    "Bad learned_log_fold shape",
                )
                for context_index in range(len(CONTEXTS)):
                    learned[context_index][
                        np.ix_(target_positions, gene_positions)
                    ] = source_learned
                    available[context_index, target_positions] = True
            elif source_learned.ndim == 3:
                require(context_values is not None, "3D learned_log_fold requires contexts")
                require(
                    source_learned.shape
                    == (len(context_values), len(target_values), len(genes)),
                    "Bad learned_log_fold shape",
                )
                for source_context, context_index in enumerate(context_positions):
                    learned[context_index][
                        np.ix_(target_positions, gene_positions)
                    ] = source_learned[source_context]
                    available[context_index, target_positions] = True
            else:
                raise RuntimeError("learned_log_fold must have two or three dimensions")
            require(np.isfinite(learned).all(), "learned_log_fold is non-finite")

        require(common is not None or learned is not None, "Response NPZ has no response arrays")
    return ResponseBundle(common, learned, available, describe_file(resolved))


def load_pseudobulk_bundle(
    path: Path | None,
    challenge_genes: list[str],
    challenge_targets: list[str],
) -> PseudobulkBundle:
    if path is None:
        return PseudobulkBundle(
            mean=None,
            available=np.zeros((len(CONTEXTS), len(challenge_targets)), dtype=bool),
            source=None,
        )
    resolved = path.resolve()
    require(resolved.is_file(), f"Pseudobulk artifact is missing: {resolved}")
    with np.load(resolved, allow_pickle=False) as archive:
        genes = _string_axis(archive, "genes", required=True)
        targets = _string_axis(archive, "targets", required=True)
        contexts = _string_axis(archive, "contexts", required=False)
        assert genes is not None and targets is not None
        gene_positions = _axis_positions(genes, challenge_genes, "gene")
        target_positions = _axis_positions(targets, challenge_targets, "target")
        context_positions = (
            np.arange(len(CONTEXTS), dtype=np.int64)
            if contexts is None
            else _axis_positions(contexts, list(CONTEXTS), "context")
        )
        require("pseudobulk_mean" in archive.files, "NPZ lacks pseudobulk_mean")
        source_mean = np.asarray(archive["pseudobulk_mean"], dtype=np.float32)
        mean = np.zeros(
            (len(CONTEXTS), len(challenge_targets), len(challenge_genes)), dtype=np.float32
        )
        available = np.zeros((len(CONTEXTS), len(challenge_targets)), dtype=bool)
        if source_mean.ndim == 2:
            require(source_mean.shape == (len(targets), len(genes)), "Bad pseudobulk_mean shape")
            for context_index in range(len(CONTEXTS)):
                mean[context_index][np.ix_(target_positions, gene_positions)] = source_mean
                available[context_index, target_positions] = True
        elif source_mean.ndim == 3:
            require(contexts is not None, "3D pseudobulk_mean requires contexts")
            require(
                source_mean.shape == (len(contexts), len(targets), len(genes)),
                "Bad pseudobulk_mean shape",
            )
            for source_context, context_index in enumerate(context_positions):
                mean[context_index][np.ix_(target_positions, gene_positions)] = source_mean[
                    source_context
                ]
                available[context_index, target_positions] = True
        else:
            raise RuntimeError("pseudobulk_mean must have two or three dimensions")
        require(np.isfinite(mean).all() and np.all(mean >= 0), "Invalid pseudobulk mean")
    return PseudobulkBundle(mean, available, describe_file(resolved))


def combine_log_fold(
    state_delta: np.ndarray,
    support_to_current: np.ndarray,
    common_log_fold: np.ndarray,
    learned_log_fold: np.ndarray,
    *,
    state_weight: float,
    state_clip: float,
    state_bounding: str,
    common_weight: float,
    learned_weight: float,
    combined_clip: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align and add model components in log-fold space."""
    scaled = state_delta.astype(np.float32, copy=True) * np.float32(state_weight)
    if state_bounding == "tanh":
        bounded = np.float32(state_clip) * np.tanh(scaled / np.float32(state_clip))
    elif state_bounding == "clip":
        bounded = np.clip(scaled, -state_clip, state_clip)
    else:
        raise RuntimeError(f"Unsupported state bounding: {state_bounding}")
    result = np.zeros((state_delta.shape[0], len(common_log_fold)), dtype=np.float32)
    valid = support_to_current >= 0
    result[:, support_to_current[valid]] = bounded[:, valid]
    result += np.float32(common_weight) * common_log_fold[None, :]
    result += np.float32(learned_weight) * learned_log_fold[None, :]
    before_final_clip = result.copy()
    np.clip(result, -combined_clip, combined_clip, out=result)
    require(np.isfinite(result).all(), "Combined log-fold is non-finite")
    return result, {
        "state_outside_bound": int(np.count_nonzero(np.abs(scaled) > state_clip)),
        "combined_outside_bound": int(np.count_nonzero(np.abs(before_final_clip) > combined_clip)),
        "values": int(result.size),
        "combined_log_fold": summarize(result),
    }


def expected_counts(
    base_counts: np.ndarray,
    log_fold: np.ndarray,
    pseudobulk_mean: np.ndarray | None,
    *,
    pseudobulk_weight: float,
    zero_induction_scale: float,
    depth_policy: str,
    target_index: int,
    target_policy: str,
    target_remaining_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct independent expectations without conserving a row total."""
    base = np.asarray(base_counts, dtype=np.float64)
    effect = np.asarray(log_fold, dtype=np.float64)
    require(base.shape == effect.shape, "Base counts and log-fold shapes differ")
    require(np.isfinite(base).all() and np.all(base >= 0), "Invalid base counts")
    expectation = base * np.exp(effect)
    induced_entries = 0
    if pseudobulk_mean is not None and pseudobulk_weight > 0:
        mean = np.asarray(pseudobulk_mean, dtype=np.float64)
        require(mean.ndim == 1 and mean.shape[0] == base.shape[1], "Bad pseudobulk vector")
        tiled = np.broadcast_to(mean[None, :], base.shape).copy()
        if depth_policy == "source-library":
            mean_total = float(mean.sum(dtype=np.float64))
            require(mean_total > 0, "Pseudobulk mean is empty")
            source_libraries = base.sum(axis=1, dtype=np.float64)
            tiled *= (source_libraries / mean_total)[:, None]
        elif depth_policy != "absolute":
            raise RuntimeError(f"Unsupported depth policy: {depth_policy}")
        observed = base > 0
        expectation[observed] = (
            (1.0 - pseudobulk_weight) * expectation[observed]
            + pseudobulk_weight * tiled[observed]
        )
        absent = ~observed
        induced = pseudobulk_weight * zero_induction_scale * tiled[absent]
        expectation[absent] = induced
        induced_entries = int(np.count_nonzero(induced > 0))
    if target_policy == "clamp":
        target_cap = base[:, target_index] * target_remaining_fraction
        expectation[:, target_index] = np.minimum(expectation[:, target_index], target_cap)
    elif target_policy != "off":
        raise RuntimeError(f"Unsupported target policy: {target_policy}")
    require(np.isfinite(expectation).all() and np.all(expectation >= 0), "Invalid expectation")
    return expectation, {
        "expected_library": summarize(expectation.sum(axis=1, dtype=np.float64)),
        "induced_zero_entries": induced_entries,
        "positive_expected_entries": int(np.count_nonzero(expectation > 0)),
    }


def emit_independent_counts(
    expectation: np.ndarray,
    rng: np.random.Generator,
    *,
    emission: str,
    nb_dispersion: float,
    max_genes_per_cell: int,
    target_index: int,
    target_policy: str,
    target_remaining_fraction: float,
    base_target_counts: np.ndarray,
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Emit each gene independently and drop excess nnz without redistribution."""
    mean = np.asarray(expectation, dtype=np.float64)
    require(mean.ndim == 2 and np.isfinite(mean).all() and np.all(mean >= 0), "Invalid mean")
    if emission == "round":
        counts = np.floor(mean + 0.5).astype(np.int64)
    elif emission == "stochastic-round":
        floor = np.floor(mean).astype(np.int64)
        counts = floor + (rng.random(mean.shape) < (mean - floor))
        counts = counts.astype(np.int64, copy=False)
    elif emission == "poisson":
        counts = rng.poisson(mean).astype(np.int64)
    elif emission == "negative-binomial":
        gamma_rate = rng.gamma(shape=nb_dispersion, scale=mean / nb_dispersion)
        counts = rng.poisson(gamma_rate).astype(np.int64)
    else:
        raise RuntimeError(f"Unsupported count emission: {emission}")

    if target_policy == "clamp":
        target_caps = np.floor(
            np.asarray(base_target_counts, dtype=np.float64) * target_remaining_fraction
        ).astype(np.int64)
        counts[:, target_index] = np.minimum(counts[:, target_index], target_caps)
    require(np.all(counts >= 0), "Emission produced negative counts")

    dropped_nnz = 0
    dropped_counts = 0
    rows: list[sp.csr_matrix] = []
    for row in counts:
        positive = np.flatnonzero(row > 0)
        if len(positive) > max_genes_per_cell:
            values = row[positive]
            order = np.lexsort((positive, -values))
            keep = positive[order[:max_genes_per_cell]]
            dropped = positive[order[max_genes_per_cell:]]
            dropped_nnz += len(dropped)
            dropped_counts += int(row[dropped].sum(dtype=np.int64))
            positive = np.sort(keep)
        values = row[positive]
        require(values.max(initial=0) <= np.iinfo(np.int32).max, "A gene count exceeds int32")
        rows.append(
            sp.csr_matrix(
                (
                    values.astype(np.int32),
                    positive.astype(np.int32),
                    np.asarray([0, len(positive)]),
                ),
                shape=(1, counts.shape[1]),
            )
        )
    matrix = sp.vstack(rows, format="csr", dtype=np.int32)
    libraries = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)
    require(np.all(libraries > 0), "Independent emission produced an empty cell")
    require(
        np.all(libraries <= MAX_OFFICIAL_CELL_LIBRARY),
        "Independent emission exceeded the official cell-library cap",
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix, {
        "output_library": summarize(libraries),
        "output_nnz": summarize(np.diff(matrix.indptr)),
        "dropped_nnz_without_redistribution": int(dropped_nnz),
        "dropped_counts_without_redistribution": int(dropped_counts),
    }


def build_group_block_v2(
    raw: sp.csr_matrix,
    model: Any,
    target: str,
    target_embedding: torch.Tensor,
    control_embedding: torch.Tensor,
    target_current_index: int,
    support_to_current: np.ndarray,
    common_log_fold: np.ndarray,
    learned_log_fold: np.ndarray,
    pseudobulk_mean: np.ndarray | None,
    device: torch.device,
    count_rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Generate one context-target block with independent count expectations."""
    blocks: list[sp.csr_matrix] = []
    raw_delta_moments = StreamingMoments()
    combined_delta_moments = StreamingMoments()
    input_libraries: list[int] = []
    output_libraries: list[float] = []
    state_outside = 0
    combined_outside = 0
    policy_values = 0
    induced_expected_entries = 0
    induced_observed_entries = 0
    dropped_nnz = 0
    dropped_counts = 0
    target_before = 0
    target_after = 0
    chunk_sizes: list[int] = []
    prediction_ranges: list[dict[str, float]] = []

    for start in range(0, raw.shape[0], args.model_chunk_size):
        stop = min(start + args.model_chunk_size, raw.shape[0])
        chunk = raw[start:stop]
        chunk_sizes.append(stop - start)
        model_input = make_model_input(chunk, support_to_current)
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
        raw_delta_moments.update(state_delta)
        log_fold, fold_qc = combine_log_fold(
            state_delta,
            support_to_current,
            common_log_fold,
            learned_log_fold,
            state_weight=args.state_effect_weight,
            state_clip=args.state_effect_clip,
            state_bounding=args.state_effect_bounding,
            common_weight=args.common_response_weight,
            learned_weight=args.learned_response_weight,
            combined_clip=args.combined_effect_clip,
        )
        combined_delta_moments.update(log_fold)
        state_outside += int(fold_qc["state_outside_bound"])
        combined_outside += int(fold_qc["combined_outside_bound"])
        policy_values += int(fold_qc["values"])

        base_dense = chunk.toarray().astype(np.float64, copy=False)
        expectation, expectation_qc = expected_counts(
            base_dense,
            log_fold,
            pseudobulk_mean,
            pseudobulk_weight=args.pseudobulk_blend_weight,
            zero_induction_scale=args.zero_induction_scale,
            depth_policy=args.pseudobulk_depth_policy,
            target_index=target_current_index,
            target_policy=args.target_policy,
            target_remaining_fraction=args.target_max_remaining_fraction,
        )
        emitted, emission_qc = emit_independent_counts(
            expectation,
            count_rng,
            emission=args.count_emission,
            nb_dispersion=args.nb_dispersion,
            max_genes_per_cell=args.max_genes_per_cell,
            target_index=target_current_index,
            target_policy=args.target_policy,
            target_remaining_fraction=args.target_max_remaining_fraction,
            base_target_counts=base_dense[:, target_current_index],
        )
        blocks.append(emitted)
        input_libraries.extend(
            np.asarray(chunk.sum(axis=1)).ravel().astype(np.int64).tolist()
        )
        output_libraries.extend(
            np.asarray(emitted.sum(axis=1)).ravel().astype(np.float64).tolist()
        )
        induced_expected_entries += int(expectation_qc["induced_zero_entries"])
        base_zero = base_dense == 0
        emitted_dense = emitted.toarray()
        induced_observed_entries += int(np.count_nonzero(base_zero & (emitted_dense > 0)))
        dropped_nnz += int(emission_qc["dropped_nnz_without_redistribution"])
        dropped_counts += int(emission_qc["dropped_counts_without_redistribution"])
        target_before += int(base_dense[:, target_current_index].sum(dtype=np.float64))
        target_after += int(emitted_dense[:, target_current_index].sum(dtype=np.int64))
        del model_input, state_delta, log_fold, base_dense, expectation, emitted_dense

    matrix = sp.vstack(blocks, format="csr", dtype=np.int32)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    require(matrix.shape == (raw.shape[0], EXPECTED_CURRENT_GENES), "Bad group shape")
    require(matrix.has_canonical_format, "Generated group CSR is not canonical")
    require(np.all(matrix.data > 0), "Generated group has invalid sparse values")
    target_remaining = float(target_after / max(target_before, 1))
    return matrix, {
        "cells": int(raw.shape[0]),
        "chunk_sizes": chunk_sizes,
        "paired_passes": 2 * len(chunk_sizes),
        "input_normalization": "elementwise-log1p-raw-on-support-axis",
        "delta_definition": "state(target, basal)-state(non-targeting, identical basal)",
        "raw_state_delta": raw_delta_moments.report(),
        "combined_log_fold": combined_delta_moments.report(),
        "state_outside_bound": state_outside,
        "combined_outside_bound": combined_outside,
        "effect_values": policy_values,
        "prediction_ranges": prediction_ranges,
        "input_library": summarize(np.asarray(input_libraries)),
        "output_library": summarize(np.asarray(output_libraries)),
        "nnz": int(matrix.nnz),
        "induced_zero_expected_entries": induced_expected_entries,
        "induced_zero_emitted_entries": induced_observed_entries,
        "dropped_nnz_without_redistribution": dropped_nnz,
        "dropped_counts_without_redistribution": dropped_counts,
        "target_sum_before": target_before,
        "target_sum_after": target_after,
        "target_remaining_fraction": target_remaining,
    }


def write_group_h5ad_v2(
    path: Path,
    matrix: sp.csr_matrix,
    genes: list[str],
    context: str,
    target: str,
    selected_rows: np.ndarray,
    seed: int,
    compression: str,
) -> None:
    obs = pd.DataFrame(
        {
            "context": context,
            "target_gene": target,
            "source_control_row": selected_rows.astype(np.int32),
        },
        index=pd.Index(
            [f"{context}|{target}|{cell:03d}" for cell in range(len(selected_rows))],
            name="cell_id",
        ),
    )
    var = pd.DataFrame(index=pd.Index(genes, name="gene_name"))
    block = ad.AnnData(X=matrix, obs=obs, var=var)
    block.uns["method"] = "scorer-aware STATE independent count adapter"
    block.uns["seed"] = seed
    block.uns["schema"] = SCHEMA
    block.write_h5ad(path, compression=None if compression == "none" else compression)


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
        require(path.is_file(), f"Required STATE input is missing: {path}")
    checkpoint_hash = sha256_file(checkpoint)
    selection = validate_selection_manifest(selection_path, checkpoint, checkpoint_hash)

    controls_dir = args.controls_dir.resolve()
    current_gene_path = controls_dir / "gene_names.csv"
    target_path = controls_dir / "pert_counts.csv"
    control_paths = {context: controls_dir / f"context_{context}.h5ad" for context in CONTEXTS}
    for path in (current_gene_path, target_path, args.support_genes, *control_paths.values()):
        require(path.is_file(), f"Required data input is missing: {path}")

    genes = read_single_column(current_gene_path, "gene_name")
    support_genes = read_single_column(args.support_genes.resolve())
    official_targets = read_targets(target_path)
    require(len(genes) == EXPECTED_CURRENT_GENES, "Unexpected current gene count")
    require(len(support_genes) == EXPECTED_SUPPORT_GENES, "Unexpected support gene count")
    require(len(official_targets) == EXPECTED_TARGETS, "Unexpected target count")
    if args.target_limit:
        require(1 <= args.target_limit <= EXPECTED_TARGETS, "Invalid target-limit")
        targets = official_targets[: args.target_limit]
    else:
        targets = official_targets
    full_contract = len(targets) == EXPECTED_TARGETS and args.cells_per_group == 400
    theoretical_nnz = (
        len(CONTEXTS) * len(targets) * args.cells_per_group * args.max_genes_per_cell
    )
    require(theoretical_nnz < SAFE_INT32_NNZ, "Theoretical nnz is not int32-safe")

    current_gene_index = {gene: index for index, gene in enumerate(genes)}
    support_gene_index = {gene: index for index, gene in enumerate(support_genes)}
    support_to_current = np.asarray(
        [current_gene_index.get(gene, -1) for gene in support_genes], dtype=np.int64
    )
    valid_support = support_to_current >= 0
    require(int(np.count_nonzero(valid_support)) == EXPECTED_OVERLAP_GENES, "Bad gene overlap")
    require(
        len(genes) - int(np.count_nonzero(valid_support)) == EXPECTED_CURRENT_ONLY_GENES,
        "Bad current-only gene count",
    )
    require(all(target in current_gene_index for target in targets), "Target missing from axis")
    require(all(target in support_gene_index for target in targets), "Target missing from support")

    response = load_response_bundle(args.response_npz, genes, official_targets)
    pseudobulk = load_pseudobulk_bundle(args.pseudobulk_npz, genes, official_targets)
    var_dims = load_var_dims(var_dims_path, support_genes)
    embeddings = load_perturbation_embeddings(pert_map_path, official_targets)
    device, device_name = _device(args)
    print(f"Loading validation-selected checkpoint: {checkpoint}", flush=True)
    model = load_state_model(checkpoint, device)
    model_info = validate_model(model, args.model_chunk_size)
    control_embedding = embeddings[CONTROL_LABEL].to(device=device, dtype=torch.float32)
    print(
        f"Loaded {model_info['parameters']:,} parameters on {device_name}; "
        f"generating {len(CONTEXTS) * len(targets)} groups",
        flush=True,
    )

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    scratch_parent = args.scratch_dir.resolve() if args.scratch_dir else None
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    final_temporary = args.output_h5ad.with_name(
        f".{args.output_h5ad.name}.{os.getpid()}.tmp.h5ad"
    )
    require(not final_temporary.exists(), f"Temporary output exists: {final_temporary}")

    group_qc: list[dict[str, Any]] = []
    final_shards: list[Path] = []
    total_nnz = 0
    try:
        with tempfile.TemporaryDirectory(prefix="vcc_state_v2_", dir=scratch_parent) as scratch:
            scratch_path = Path(scratch)
            for context_index, context in enumerate(CONTEXTS):
                control = ad.read_h5ad(control_paths[context], backed="r")
                try:
                    require(
                        list(control.var_names.astype(str)) == genes,
                        f"{context}: gene mismatch",
                    )
                    required_obs = {"context", "target_gene", "ntc_id"}
                    require(required_obs.issubset(control.obs.columns), f"{context}: missing obs")
                    pending_blocks: list[Path] = []
                    context_shards: list[Path] = []
                    for local_target_index, target in enumerate(targets):
                        official_target_index = official_targets.index(target)
                        sample_rng = np.random.default_rng(
                            deterministic_seed(args.seed, context, target, "sample")
                        )
                        count_rng = np.random.default_rng(
                            deterministic_seed(args.seed, context, target, "counts-v2")
                        )
                        selected_rows, allocation = stratified_control_indices(
                            control.obs, args.cells_per_group, sample_rng
                        )
                        raw = read_control_rows(control, selected_rows)
                        common, learned = response.group(
                            context_index, official_target_index, len(genes)
                        )
                        mean = pseudobulk.group(context_index, official_target_index)
                        target_embedding = embeddings[target].to(
                            device=device, dtype=torch.float32
                        )
                        block, qc = build_group_block_v2(
                            raw,
                            model,
                            target,
                            target_embedding,
                            control_embedding,
                            current_gene_index[target],
                            support_to_current,
                            common,
                            learned,
                            mean,
                            device,
                            count_rng,
                            args,
                        )
                        qc.update(
                            {
                                "context": context,
                                "target_gene": target,
                                "target_ordinal": official_target_index,
                                "response_available": bool(
                                    response.learned_available[
                                        context_index, official_target_index
                                    ]
                                ),
                                "pseudobulk_available": bool(
                                    pseudobulk.available[context_index, official_target_index]
                                ),
                                "source_control_index_sha256": hashlib.sha256(
                                    selected_rows.tobytes()
                                ).hexdigest(),
                                "ntc_strata": len(allocation),
                                "sample_seed": deterministic_seed(
                                    args.seed, context, target, "sample"
                                ),
                                "count_seed": deterministic_seed(
                                    args.seed, context, target, "counts-v2"
                                ),
                            }
                        )
                        group_qc.append(qc)
                        total_nnz += int(block.nnz)
                        require(total_nnz < SAFE_INT32_NNZ, "Streaming nnz exceeds safe limit")
                        block_path = scratch_path / f"{context}_{local_target_index:03d}_block.h5ad"
                        write_group_h5ad_v2(
                            block_path,
                            block,
                            genes,
                            context,
                            target,
                            selected_rows,
                            args.seed,
                            args.compression,
                        )
                        pending_blocks.append(block_path)
                        del raw, block, target_embedding

                        if (
                            len(pending_blocks) == args.groups_per_shard
                            or local_target_index + 1 == len(targets)
                        ):
                            shard_path = scratch_path / (
                                f"{context}_shard_{len(context_shards):03d}.h5ad"
                            )
                            concatenate_paths(
                                pending_blocks, shard_path, args.concat_max_loaded_elements
                            )
                            for block_to_remove in pending_blocks:
                                block_to_remove.unlink()
                            pending_blocks.clear()
                            context_shards.append(shard_path)
                        if (
                            (local_target_index + 1) % 10 == 0
                            or local_target_index + 1 == len(targets)
                        ):
                            print(
                                f"[{context}] generated {local_target_index + 1}/{len(targets)} "
                                f"targets; cumulative nnz={total_nnz:,}",
                                flush=True,
                            )
                    final_shards.extend(context_shards)
                finally:
                    control.file.close()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            concatenate_paths(final_shards, final_temporary, args.concat_max_loaded_elements)
            validation = validate_backed_candidate(
                final_temporary, genes, targets, args.cells_per_group, total_nnz
            )
            if args.target_policy == "clamp":
                target_failures = [
                    f"{item['context']}|{item['target_gene']}"
                    for item in group_qc
                    if item["target_sum_before"] >= 20
                    and item["target_remaining_fraction"]
                    > args.target_max_remaining_fraction + 1e-12
                ]
                require(not target_failures, "One or more groups failed target clamp QC")
            else:
                target_failures = []
            os.replace(final_temporary, args.output_h5ad)

        elapsed = time.time() - started
        report = {
            "schema": SCHEMA,
            "artifact_type": "vcc_2026_prediction_candidate",
            "created_utc_epoch": time.time(),
            "elapsed_seconds": elapsed,
            "command": [sys.executable, *sys.argv],
            "method": "paired STATE residuals with independent scorer-aware count emission",
            "full_official_contract": full_contract,
            "configuration": {
                "seed": args.seed,
                "device": str(device),
                "device_name": device_name,
                "model_chunk_size": args.model_chunk_size,
                "cells_per_group": args.cells_per_group,
                "targets": len(targets),
                "state_effect_weight": args.state_effect_weight,
                "state_effect_clip": args.state_effect_clip,
                "state_effect_bounding": args.state_effect_bounding,
                "common_response_weight": args.common_response_weight,
                "learned_response_weight": args.learned_response_weight,
                "combined_effect_clip": args.combined_effect_clip,
                "pseudobulk_blend_weight": args.pseudobulk_blend_weight,
                "zero_induction_scale": args.zero_induction_scale,
                "pseudobulk_depth_policy": args.pseudobulk_depth_policy,
                "target_policy": args.target_policy,
                "target_max_remaining_fraction": args.target_max_remaining_fraction,
                "count_emission": args.count_emission,
                "nb_dispersion": args.nb_dispersion,
                "max_genes_per_cell": args.max_genes_per_cell,
                "library_policy": "independent-gene-expectations; no total renormalization",
                "nnz_cap_policy": "drop-low-count-genes-without-redistribution",
                "zero_entry_policy": (
                    "source zeros remain zero unless an explicit pseudobulk mean is blended"
                ),
            },
            "scorer_contract": {
                "perturbed_gene_excluded_from_all_six_metrics": True,
                "target_force_default": "off",
                "target_clamp_changes_other_coordinates": False,
            },
            "model": model_info,
            "checkpoint_selection": {
                "global_step": int(selection["selected"]["global_step"]),
                "val_loss": float(selection["selected"]["val_loss"]),
                "rule": selection["selection_rule"],
            },
            "axes": {
                "contexts": list(CONTEXTS),
                "targets": len(targets),
                "current_genes": len(genes),
                "support_genes": len(support_genes),
                "shared_genes": int(np.count_nonzero(valid_support)),
                "perturbation_embedding_dimension": int(var_dims["pert_dim"]),
            },
            "validation": validation,
            "scientific_qc": {
                "target_clamp_failures": target_failures,
                "input_library": summarize(
                    np.asarray([item["input_library"]["mean"] for item in group_qc])
                ),
                "output_library": summarize(
                    np.asarray([item["output_library"]["mean"] for item in group_qc])
                ),
                "target_remaining_fraction": summarize(
                    np.asarray([item["target_remaining_fraction"] for item in group_qc])
                ),
                "induced_zero_emitted_entries": int(
                    sum(item["induced_zero_emitted_entries"] for item in group_qc)
                ),
                "dropped_counts_without_redistribution": int(
                    sum(item["dropped_counts_without_redistribution"] for item in group_qc)
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
                "state_source_commit": git_commit(
                    Path(inspect.getfile(type(model))).resolve().parent
                ),
                "script": describe_file(Path(__file__).resolve()),
                "checkpoint": {
                    **describe_file(checkpoint, hash_file=False),
                    "sha256": checkpoint_hash,
                },
                "checkpoint_selection": describe_file(selection_path),
                "perturbation_map": describe_file(pert_map_path),
                "var_dims": describe_file(var_dims_path),
                "state_config": describe_file(config_path) if config_path.is_file() else None,
                "current_gene_axis": describe_file(current_gene_path),
                "support_gene_axis": describe_file(args.support_genes.resolve()),
                "target_axis": describe_file(target_path),
                "controls": {
                    context: describe_file(path) for context, path in control_paths.items()
                },
                "response_artifact": response.source,
                "pseudobulk_artifact": pseudobulk.source,
                "output_h5ad": describe_file(args.output_h5ad),
            },
        }
        atomic_write_json(args.output_json, report)
        print(
            f"Wrote {args.output_h5ad}; shape={validation['shape']}, "
            f"nnz={validation['nnz']:,}, elapsed={elapsed / 60:.1f} min",
            flush=True,
        )
        print(f"Wrote {args.output_json}", flush=True)
    finally:
        if final_temporary.exists():
            final_temporary.unlink()


if __name__ == "__main__":
    main()
