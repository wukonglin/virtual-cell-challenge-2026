#!/usr/bin/env python3
"""Build or verify a truth-blind, uniform STATE seed ensemble.

The four registered reconstructions use the same generator seed, sampled
control rows, gene axis, and row order.  This utility averages their raw count
vectors cell by cell, then uses deterministic largest-remainder rounding to
restore every source-cell library exactly.  It has no treated-data argument
and does not select a source from evaluation metrics.

The output is deliberately an ensemble-specific artifact rather than a
single-checkpoint ``vcc-public-candidate-spec-v2``.  A downstream scorer must
authenticate ``ensemble_verified.json`` and must not relabel this derived
artifact as one of its source candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.experimental import concat_on_disk


PLAN_SCHEMA = "vcc-state-seed-ensemble-plan-v1"
RECEIPT_SCHEMA = "vcc-state-seed-ensemble-verification-v1"
PLAN_LINEAGE = "anvil-portable-reconstruction-v2-correlated-not-historical-p4"
METHOD_NAME = "cellwise-uniform-count-mean"
INTEGERIZATION = "per-cell-largest-remainder"
TIE_BREAK = "cyclic-gene-rotation-v1"
CONTEXT_PREFIX = {"HepG2": "hepg2", "Jurkat": "jurkat"}
TAG_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
MAX_JSON_BYTES = 16 << 20
UINT64_MASK = (1 << 64) - 1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> Path:
    require(not path.is_symlink(), f"{label} must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing {label}: {path}") from error
    require(stat.S_ISREG(resolved.stat().st_mode), f"{label} is not a regular file")
    return resolved


def regular_directory(path: Path, label: str) -> Path:
    require(not path.is_symlink(), f"{label} must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing {label}: {path}") from error
    require(resolved.is_dir(), f"{label} is not a directory: {resolved}")
    return resolved


def load_json(path: Path, label: str) -> dict[str, Any]:
    resolved = regular_file(path, label)
    require(resolved.stat().st_size <= MAX_JSON_BYTES, f"{label} is unexpectedly large")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read {label}: {resolved}") from error
    require(type(value) is dict, f"{label} root must be a JSON object")
    return value


def describe_file(path: Path, *, recorded_path: Path | None = None) -> dict[str, Any]:
    resolved = regular_file(path, "provenance file")
    return {
        "path": str((recorded_path or resolved).absolute()),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def descriptor_content(descriptor: Any, label: str) -> tuple[int, str]:
    require(type(descriptor) is dict, f"{label} descriptor is absent")
    try:
        size = int(descriptor.get("size_bytes", -1))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} descriptor size is invalid") from error
    digest = str(descriptor.get("sha256", "")).lower()
    require(size >= 0 and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            f"{label} descriptor content is invalid")
    return size, digest


def authenticate_descriptor(path: Path, descriptor: Any, label: str) -> dict[str, Any]:
    observed = describe_file(path)
    require(
        descriptor_content(observed, f"observed {label}")
        == descriptor_content(descriptor, f"recorded {label}"),
        f"{label} content identity differs",
    )
    return observed


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    require(
        set(plan)
        == {
            "axes",
            "context",
            "evaluation_policy",
            "firewall",
            "lineage",
            "method",
            "output_tag",
            "schema",
            "selection_policy",
            "source_tags",
        },
        "Seed-ensemble plan fields are not exact",
    )
    require(plan.get("schema") == PLAN_SCHEMA, "Bad seed-ensemble plan schema")
    require(plan.get("lineage") == PLAN_LINEAGE, "Bad seed-ensemble plan lineage")
    context = plan.get("context")
    require(context in CONTEXT_PREFIX, "Unsupported seed-ensemble context")
    output_tag = str(plan.get("output_tag", ""))
    require(TAG_PATTERN.fullmatch(output_tag) is not None, "Unsafe ensemble output tag")
    tags = plan.get("source_tags")
    require(
        type(tags) is list
        and len(tags) >= 2
        and all(type(tag) is str and TAG_PATTERN.fullmatch(tag) for tag in tags)
        and len(tags) == len(set(tags)),
        "Seed-ensemble source tags are invalid",
    )
    method = plan.get("method")
    require(
        type(method) is dict
        and set(method)
        == {"integerization", "name", "tie_break", "tie_break_seed", "weights"}
        and method.get("name") == METHOD_NAME
        and method.get("weights") == "uniform"
        and method.get("integerization") == INTEGERIZATION
        and method.get("tie_break") == TIE_BREAK
        and type(method.get("tie_break_seed")) is int,
        "Seed-ensemble method is not the registered uniform exact-library mean",
    )
    axes = plan.get("axes")
    require(
        type(axes) is dict
        and set(axes) == {"cells_per_target", "contexts", "genes", "targets"}
        and type(axes.get("contexts")) is int
        and type(axes.get("targets")) is int
        and type(axes.get("cells_per_target")) is int
        and type(axes.get("genes")) is int
        and axes["contexts"] == 1
        and min(axes["targets"], axes["cells_per_target"], axes["genes"]) > 0,
        "Seed-ensemble axes are invalid",
    )
    selection = plan.get("selection_policy")
    require(
        selection
        == {
            "all_registered_sources_required": True,
            "best_public_score_seed_selection_allowed": False,
            "failed_integrity_source_exclusion_allowed": False,
            "missing_source_fallback_allowed": False,
        },
        "Seed-ensemble source-selection policy is unsafe",
    )
    evaluation = plan.get("evaluation_policy")
    require(
        type(evaluation) is dict
        and set(evaluation)
        == {
            "absolute_tolerance",
            "best_individual_seed_selection_allowed",
            "fallback_tag",
            "held_target_gate",
            "individual_scores_role",
            "primary_baseline_tag",
            "primary_candidate_tag",
            "primary_gate",
            "public_score_opens_after_ensemble_materialization",
        }
        and evaluation.get("absolute_tolerance") == 1e-12
        and evaluation.get("best_individual_seed_selection_allowed") is False
        and evaluation.get("fallback_tag") == tags[0]
        and evaluation.get("held_target_gate")
        == "all_evaluable_metrics_non_regression"
        and evaluation.get("individual_scores_role")
        == "seed_variance_diagnostic_only"
        and evaluation.get("primary_baseline_tag") == tags[0]
        and evaluation.get("primary_candidate_tag") == output_tag
        and evaluation.get("primary_gate")
        == (
            "all_and_direct_mse_nmae_reach_non_regression_and_"
            "pds_or_jaccard_strict_improvement"
        )
        and evaluation.get("public_score_opens_after_ensemble_materialization")
        is True,
        "Seed-ensemble evaluation policy is unsafe or incomplete",
    )
    firewall = plan.get("firewall")
    require(
        firewall
        == {
            "challenge_leaderboard_used_for_selection": False,
            "sealed_treated_profiles_read": False,
            "source_selection_uses_metrics": False,
        },
        "Seed-ensemble truth firewall is malformed",
    )
    return plan


def _canonical_csr(matrix: Any, label: str) -> sp.csr_matrix:
    require(sp.issparse(matrix), f"{label} is not sparse")
    result = matrix.tocsr().astype(np.int64, copy=False)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    require(not result.data.size or int(result.data.min()) >= 0,
            f"{label} has negative counts")
    return result


def _csr_equal(left: sp.csr_matrix, right: sp.csr_matrix) -> bool:
    return (
        left.shape == right.shape
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and np.array_equal(left.data, right.data)
    )


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (value ^ (value >> 31)) & UINT64_MASK


def _rounding_positions(
    indices: np.ndarray,
    remainders: np.ndarray,
    count: int,
    *,
    global_row: int,
    genes: int,
    tie_break_seed: int,
) -> np.ndarray:
    """Choose largest remainders with a row-rotated, index-unique tie break."""

    require(0 <= count <= len(indices), "Invalid largest-remainder increment count")
    chosen = np.zeros(len(indices), dtype=bool)
    remaining = count
    offset = _splitmix64(tie_break_seed ^ global_row) % genes
    for remainder in range(int(remainders.max(initial=0)), 0, -1):
        positions = np.flatnonzero(remainders == remainder)
        if len(positions) <= remaining:
            chosen[positions] = True
            remaining -= len(positions)
        elif remaining:
            # Gene indices are unique within a canonical CSR row, and rotation
            # modulo the full axis is bijective, so these keys have no ties.
            keys = (indices[positions].astype(np.int64) - int(offset)) % genes
            selected = np.argpartition(keys, remaining - 1)[:remaining]
            chosen[positions[selected]] = True
            remaining = 0
        if remaining == 0:
            break
    require(remaining == 0, "Unable to satisfy exact-library rounding")
    return chosen


def mean_csr_exact_library(
    matrices: Sequence[sp.csr_matrix],
    *,
    global_row_start: int,
    tie_break_seed: int,
) -> tuple[sp.csr_matrix, dict[str, int]]:
    """Return the uniform cellwise count mean with exact source libraries."""

    require(len(matrices) >= 2, "At least two matrices are required for an ensemble")
    canonical: list[sp.csr_matrix] = []
    shape = matrices[0].shape
    for ordinal, matrix in enumerate(matrices):
        require(sp.issparse(matrix), f"Source matrix {ordinal} is not sparse")
        current = matrix.tocsr().astype(np.int64, copy=False)
        current.sum_duplicates()
        current.eliminate_zeros()
        current.sort_indices()
        require(current.shape == shape, "Source matrix shapes differ")
        require(not current.data.size or int(current.data.min()) >= 0,
                "Source matrix has negative counts")
        canonical.append(current)

    libraries = [np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.int64)
                 for matrix in canonical]
    for observed in libraries[1:]:
        require(np.array_equal(observed, libraries[0]), "Source cell libraries differ")

    summed = canonical[0].copy()
    for matrix in canonical[1:]:
        summed = summed + matrix
    summed = summed.tocsr()
    summed.sum_duplicates()
    summed.eliminate_zeros()
    summed.sort_indices()

    source_count = len(canonical)
    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    indptr = np.zeros(shape[0] + 1, dtype=np.int64)
    fractional_entries = 0
    rounding_increments = 0
    rows_with_fractional_entries = 0
    for row in range(shape[0]):
        start, stop = summed.indptr[row : row + 2]
        indices = summed.indices[start:stop]
        totals = summed.data[start:stop]
        quotients, remainders = np.divmod(totals, source_count)
        remainder_sum = int(remainders.sum(dtype=np.int64))
        require(remainder_sum % source_count == 0,
                "Source libraries do not admit exact mean integerization")
        deficit = remainder_sum // source_count
        require(int(quotients.sum(dtype=np.int64)) + deficit == int(libraries[0][row]),
                "Largest-remainder target differs from the source library")
        if deficit:
            increments = _rounding_positions(
                indices,
                remainders,
                deficit,
                global_row=global_row_start + row,
                genes=shape[1],
                tie_break_seed=tie_break_seed,
            )
            quotients = quotients + increments.astype(np.int64)
            rounding_increments += deficit
        nonzero = quotients > 0
        row_data = quotients[nonzero]
        require(not row_data.size or int(row_data.max()) <= np.iinfo(np.int32).max,
                "Ensemble count exceeds int32")
        data_parts.append(row_data.astype(np.int32, copy=False))
        index_parts.append(indices[nonzero].astype(np.int32, copy=False))
        indptr[row + 1] = indptr[row] + len(row_data)
        fractional = int(np.count_nonzero(remainders))
        fractional_entries += fractional
        rows_with_fractional_entries += int(fractional > 0)

    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.int32)
    indices = (
        np.concatenate(index_parts) if index_parts else np.empty(0, dtype=np.int32)
    )
    require(indptr[-1] <= np.iinfo(np.int32).max, "Ensemble CSR exceeds int32 capacity")
    output = sp.csr_matrix(
        (data, indices, indptr.astype(np.int32)), shape=shape, dtype=np.int32
    )
    output.sum_duplicates()
    output.eliminate_zeros()
    output.sort_indices()
    require(output.has_canonical_format, "Ensemble output is not canonical CSR")
    output_libraries = np.asarray(output.sum(axis=1)).reshape(-1).astype(np.int64)
    require(np.array_equal(output_libraries, libraries[0]),
            "Ensemble integerization changed a source-cell library")
    return output, {
        "fractional_entries": fractional_entries,
        "rounding_increments": rounding_increments,
        "rows_with_fractional_entries": rows_with_fractional_entries,
    }


@dataclass(frozen=True)
class SourceBundle:
    tag: str
    run_dir: Path
    prediction: Path
    spec: dict[str, Any]
    generation: dict[str, Any]
    verification: dict[str, Any]
    descriptors: dict[str, dict[str, Any]]


def _authenticate_source(
    project_dir: Path, public_root: Path, context: str, tag: str
) -> SourceBundle:
    public_root = regular_directory(public_root, "public root")
    prefix = CONTEXT_PREFIX[context]
    run_dir = regular_directory(
        public_root / "candidates" / prefix / tag, f"source candidate {tag}"
    )
    require(_is_relative_to(run_dir, public_root),
            f"Source candidate escapes the public root: {tag}")
    from materialize_public_candidate_transport import (  # noqa: PLC0415
        TRANSPORT_RECEIPT_NAME,
        validate_transport_receipt,
    )

    transport = validate_transport_receipt(
        run_dir=run_dir,
        project_dir=project_dir,
        expected_context=context,
        expected_output_tag=tag,
    )
    require(transport is not None, "Reconstructed ensemble source lacks transport receipt")
    paths = {
        "spec": regular_file(run_dir / "spec.json", f"{tag} spec"),
        "generation": regular_file(run_dir / "generation.json", f"{tag} generation"),
        "verification": regular_file(
            run_dir / "generation_verified.json", f"{tag} verification"
        ),
        "prediction": regular_file(run_dir / "prediction.h5ad", f"{tag} prediction"),
        "transport": regular_file(
            run_dir / TRANSPORT_RECEIPT_NAME, f"{tag} transport receipt"
        ),
    }
    spec = load_json(paths["spec"], f"{tag} spec")
    generation = load_json(paths["generation"], f"{tag} generation")
    verification = load_json(paths["verification"], f"{tag} verification")
    require(spec.get("context") == verification.get("context") == context,
            f"{tag} context differs")
    require(spec.get("output_tag") == verification.get("output_tag") == tag,
            f"{tag} output identity differs")
    require(verification.get("status") == "passed", f"{tag} verification did not pass")
    require(
        type(verification.get("checks")) is dict
        and all(value is True for value in verification["checks"].values()),
        f"{tag} verification checks are incomplete",
    )
    descriptors = {name: describe_file(path) for name, path in paths.items()}
    authenticate_descriptor(
        paths["prediction"],
        (verification.get("provenance") or {}).get("prediction_h5ad"),
        f"{tag} prediction",
    )
    require(
        (generation.get("data_firewall") or {}).get("sealed_treated_profiles_read")
        is False,
        f"{tag} does not satisfy the generator firewall",
    )
    return SourceBundle(
        tag=tag,
        run_dir=run_dir,
        prediction=paths["prediction"],
        spec=spec,
        generation=generation,
        verification=verification,
        descriptors=descriptors,
    )


def _descriptor_identity(descriptor: Any) -> tuple[int, str]:
    return descriptor_content(descriptor, "shared input")


def validate_source_contracts(sources: Sequence[SourceBundle], plan: dict[str, Any]) -> None:
    require([source.tag for source in sources] == plan["source_tags"],
            "Authenticated source order differs from the plan")
    baseline = sources[0]
    shared_inputs = (
        "controls_h5ad",
        "panel_manifest",
        "panel_csv",
        "residual_npz",
        "residual_json",
        "support_genes",
        "generator_script",
        "generate_state_direct_counts_helper",
        "infer_state_effect_prior_helper",
    )
    for source in sources:
        require(source.spec.get("configuration") == baseline.spec.get("configuration"),
                f"{source.tag} generator configuration differs")
        require(source.spec.get("axes") == baseline.spec.get("axes"),
                f"{source.tag} candidate axes differ")
        require(source.spec.get("state_source", {}).get("repository_commit")
                == baseline.spec.get("state_source", {}).get("repository_commit"),
                f"{source.tag} STATE source commit differs")
        for key in shared_inputs:
            require(
                _descriptor_identity(source.spec["inputs"][key])
                == _descriptor_identity(baseline.spec["inputs"][key]),
                f"{source.tag} shared generator input differs: {key}",
            )
        scientific = source.generation.get("scientific_qc") or {}
        require(scientific.get("all_libraries_exact") is True,
                f"{source.tag} lacks exact-library evidence")
        require(scientific.get("all_native_non_support_counts_exact") is True,
                f"{source.tag} lacks native non-support invariance")
        require(scientific.get("target_knockdown_failures") == [],
                f"{source.tag} has target-knockdown failures")

    axes = plan["axes"]
    expected_spec_axes = {
        "targets": axes["targets"],
        "native_genes": axes["genes"],
    }
    for key, value in expected_spec_axes.items():
        require(int(baseline.spec["axes"].get(key, -1)) == value,
                f"Plan/source axis differs: {key}")
    require(
        int(baseline.spec["configuration"].get("cells_per_group", -1))
        == axes["cells_per_target"],
        "Plan/source cells-per-target differs",
    )


def _string_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if isinstance(result[column].dtype, pd.CategoricalDtype) or result[column].dtype == object:
            result[column] = result[column].astype(str)
    return result


def _validate_open_sources(
    opened: Sequence[ad.AnnData], plan: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    axes = plan["axes"]
    expected_shape = (
        axes["targets"] * axes["cells_per_target"],
        axes["genes"],
    )
    first = opened[0]
    require(first.shape == expected_shape, "First source prediction shape differs from plan")
    require(str(getattr(first.X, "format", "")) == "csr", "Source prediction is not CSR")
    obs = _string_frame(first.obs)
    var = first.var.copy()
    genes = first.var_names.astype(str).tolist()
    required_obs = {"context", "target_gene", "source_control_row", "source_control_cell_id"}
    require(required_obs.issubset(obs.columns), "Source prediction obs is incomplete")
    require(set(obs["context"]) == {plan["context"]}, "Source context differs from plan")
    require(first.obs_names.is_unique and len(genes) == len(set(genes)),
            "Source axes contain duplicate identifiers")
    for ordinal, current in enumerate(opened[1:], start=1):
        require(current.shape == expected_shape, f"Source {ordinal} shape differs")
        require(str(getattr(current.X, "format", "")) == "csr",
                f"Source {ordinal} is not CSR")
        require(current.obs_names.astype(str).tolist() == first.obs_names.astype(str).tolist(),
                f"Source {ordinal} cell axis differs")
        require(current.var_names.astype(str).tolist() == genes,
                f"Source {ordinal} gene axis differs")
        require(_string_frame(current.obs).equals(obs),
                f"Source {ordinal} observation metadata differs")
    return obs, var, genes


def materialize_prediction(
    source_predictions: Sequence[Path],
    output_h5ad: Path,
    *,
    plan: dict[str, Any],
    max_loaded_elements: int = 20_000_000,
) -> dict[str, Any]:
    """Materialize an authenticated-plan prediction without reading references."""

    validate_plan(plan)
    require(len(source_predictions) == len(plan["source_tags"]),
            "Prediction count differs from the registered source count")
    require(not output_h5ad.exists(), f"Refusing to overwrite output: {output_h5ad}")
    opened = [ad.read_h5ad(path, backed="r") for path in source_predictions]
    try:
        obs, var, genes = _validate_open_sources(opened, plan)
        cells_per_target = int(plan["axes"]["cells_per_target"])
        targets = int(plan["axes"]["targets"])
        tie_break_seed = int(plan["method"]["tie_break_seed"])
        group_reports: list[dict[str, Any]] = []
        block_paths: list[Path] = []
        with tempfile.TemporaryDirectory(
            prefix="vcc_seed_ensemble_", dir=output_h5ad.parent
        ) as scratch_text:
            scratch = Path(scratch_text)
            block_uns = {
                "schema": RECEIPT_SCHEMA,
                "output_tag": plan["output_tag"],
                "method": plan["method"],
                "source_tags": plan["source_tags"],
                "sealed_treated_profiles_read": False,
            }
            for ordinal in range(targets):
                start = ordinal * cells_per_target
                stop = start + cells_per_target
                target_values = obs.iloc[start:stop]["target_gene"].unique().tolist()
                require(len(target_values) == 1, "A target group is not contiguous")
                matrices = [current.X[start:stop].tocsr() for current in opened]
                ensemble, rounding = mean_csr_exact_library(
                    matrices,
                    global_row_start=start,
                    tie_break_seed=tie_break_seed,
                )
                target = target_values[0]
                target_gene_index = genes.index(target)
                source_target_sums = [int(matrix[:, target_gene_index].sum()) for matrix in matrices]
                ensemble_target_sum = int(ensemble[:, target_gene_index].sum())
                block = ad.AnnData(
                    X=ensemble,
                    obs=opened[0].obs.iloc[start:stop].copy(),
                    var=var.copy(),
                )
                block.uns["state_seed_ensemble"] = block_uns
                block_path = scratch / f"{ordinal:03d}_{target}.h5ad"
                block.write_h5ad(block_path, compression="lzf")
                block_paths.append(block_path)
                group_reports.append(
                    {
                        "ordinal": ordinal,
                        "target_gene": target,
                        "cells": cells_per_target,
                        "nnz": int(ensemble.nnz),
                        "source_target_sums": source_target_sums,
                        "ensemble_target_sum": ensemble_target_sum,
                        **rounding,
                    }
                )
            concat_on_disk(
                block_paths,
                output_h5ad,
                axis=0,
                join="inner",
                merge="same",
                uns_merge="same",
                max_loaded_elems=max_loaded_elements,
            )
    finally:
        for current in opened:
            current.file.close()

    output = ad.read_h5ad(output_h5ad, backed="r")
    try:
        expected_shape = (
            plan["axes"]["targets"] * plan["axes"]["cells_per_target"],
            plan["axes"]["genes"],
        )
        require(output.shape == expected_shape, "Ensemble output shape differs")
        require(output.obs_names.astype(str).tolist() == obs.index.astype(str).tolist(),
                "Ensemble output cell axis differs")
        require(output.var_names.astype(str).tolist() == genes,
                "Ensemble output gene axis differs")
        require(str(getattr(output.X, "format", "")) == "csr", "Ensemble output is not CSR")
        data_dtype = np.dtype(output.X.group["data"].dtype)
        require(np.issubdtype(data_dtype, np.integer), "Ensemble output is not integer counts")
        nnz = int(output.X.group["data"].shape[0])
    finally:
        output.file.close()
    return {
        "shape": list(expected_shape),
        "groups": targets,
        "nnz": nnz,
        "data_dtype": str(data_dtype),
        "fractional_entries": int(sum(item["fractional_entries"] for item in group_reports)),
        "rounding_increments": int(sum(item["rounding_increments"] for item in group_reports)),
        "rows_with_fractional_entries": int(
            sum(item["rows_with_fractional_entries"] for item in group_reports)
        ),
        "group_qc": group_reports,
    }


def verify_prediction_exact_mean(
    source_predictions: Sequence[Path],
    output_h5ad: Path,
    *,
    plan: dict[str, Any],
) -> dict[str, int]:
    """Recompute every output block and require exact deterministic equality."""

    validate_plan(plan)
    require(len(source_predictions) == len(plan["source_tags"]),
            "Prediction count differs from the registered source count")
    opened = [ad.read_h5ad(path, backed="r") for path in source_predictions]
    output = ad.read_h5ad(output_h5ad, backed="r")
    try:
        obs, _, genes = _validate_open_sources(opened, plan)
        expected_shape = (
            plan["axes"]["targets"] * plan["axes"]["cells_per_target"],
            plan["axes"]["genes"],
        )
        require(output.shape == expected_shape, "Ensemble output shape differs from plan")
        require(str(getattr(output.X, "format", "")) == "csr",
                "Ensemble output is not CSR")
        require(output.obs_names.astype(str).tolist() == obs.index.astype(str).tolist(),
                "Ensemble output cell axis differs")
        require(output.var_names.astype(str).tolist() == genes,
                "Ensemble output gene axis differs")
        require(_string_frame(output.obs).equals(obs),
                "Ensemble output observation metadata differs")

        cells_per_target = int(plan["axes"]["cells_per_target"])
        tie_break_seed = int(plan["method"]["tie_break_seed"])
        groups = 0
        verified_nnz = 0
        for start in range(0, output.n_obs, cells_per_target):
            stop = start + cells_per_target
            expected, _ = mean_csr_exact_library(
                [current.X[start:stop].tocsr() for current in opened],
                global_row_start=start,
                tie_break_seed=tie_break_seed,
            )
            observed = _canonical_csr(
                output.X[start:stop].tocsr(), "Ensemble output block"
            )
            require(
                _csr_equal(observed, expected.astype(np.int64)),
                f"Ensemble output is not the registered deterministic mean at group {groups}",
            )
            groups += 1
            verified_nnz += int(observed.nnz)
        require(groups == plan["axes"]["targets"],
                "Ensemble exact-mean group count differs from plan")
        return {
            "groups": groups,
            "rows": int(output.n_obs),
            "nnz": verified_nnz,
        }
    finally:
        output.file.close()
        for current in opened:
            current.file.close()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def build(
    *,
    project_dir: Path,
    public_root: Path,
    plan_path: Path,
    max_loaded_elements: int,
) -> Path:
    project_dir = regular_directory(project_dir, "project directory")
    plan_path = regular_file(plan_path, "seed-ensemble plan")
    plan = validate_plan(load_json(plan_path, "seed-ensemble plan"))
    public_root = public_root.resolve(strict=True)
    artifact_root = (project_dir / "artifacts").resolve(strict=True)
    require(_is_relative_to(public_root, artifact_root),
            "Public root must be inside the project artifacts directory")
    output_dir = (
        public_root / "ensembles" / CONTEXT_PREFIX[plan["context"]] / plan["output_tag"]
    ).absolute()
    require(not output_dir.exists() and not output_dir.is_symlink(),
            f"Refusing to overwrite ensemble: {output_dir}")
    sources = [
        _authenticate_source(project_dir, public_root, plan["context"], tag)
        for tag in plan["source_tags"]
    ]
    validate_source_contracts(sources, plan)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{plan['output_tag']}.", dir=output_dir.parent))
    try:
        temporary_prediction = temporary / "prediction.h5ad"
        validation = materialize_prediction(
            [source.prediction for source in sources],
            temporary_prediction,
            plan=plan,
            max_loaded_elements=max_loaded_elements,
        )
        final_prediction = output_dir / "prediction.h5ad"
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "passed",
            "lineage": "truth-blind-derived-seed-ensemble-not-historical-p4",
            "context": plan["context"],
            "output_tag": plan["output_tag"],
            "method": plan["method"],
            "selection_policy": plan["selection_policy"],
            "evaluation_policy": plan["evaluation_policy"],
            "firewall": plan["firewall"],
            "plan": describe_file(plan_path),
            "implementation": describe_file(Path(__file__).resolve()),
            "sources": [
                {
                    "tag": source.tag,
                    "run_directory": str(source.run_dir),
                    "artifacts": source.descriptors,
                }
                for source in sources
            ],
            "output": describe_file(
                temporary_prediction, recorded_path=final_prediction
            ),
            "validation": validation,
            "contract": {
                "output_exact_deterministic_mean": True,
                "source_transport_receipts_revalidated": True,
                "shared_generator_inputs_identical": True,
                "source_cell_axes_identical": True,
                "source_gene_axes_identical": True,
                "source_cell_libraries_identical": True,
                "output_cell_libraries_exact": True,
                "uniform_weights": True,
                "metric_based_source_selection": False,
                "treated_truth_read": False,
                "leaderboard_submission_authorized": False,
            },
        }
        receipt_path = temporary / "ensemble_verified.json"
        with receipt_path.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        require(not output_dir.exists(), "Ensemble destination appeared during build")
        os.rename(temporary, output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output_dir / "ensemble_verified.json"


def verify(
    *, project_dir: Path, public_root: Path, plan_path: Path, run_dir: Path
) -> dict[str, Any]:
    project_dir = regular_directory(project_dir, "project directory")
    public_root = regular_directory(public_root, "public root")
    plan_path = regular_file(plan_path, "seed-ensemble plan")
    plan = validate_plan(load_json(plan_path, "seed-ensemble plan"))
    expected = (
        public_root
        / "ensembles"
        / CONTEXT_PREFIX[plan["context"]]
        / plan["output_tag"]
    ).resolve(strict=True)
    run_dir = regular_directory(run_dir, "ensemble run directory")
    require(run_dir == expected, "Ensemble directory differs from the plan")
    receipt = load_json(run_dir / "ensemble_verified.json", "ensemble receipt")
    require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("status") == "passed",
            "Ensemble receipt is not a passed v1 receipt")
    require(receipt.get("context") == plan["context"]
            and receipt.get("output_tag") == plan["output_tag"],
            "Ensemble receipt identity differs from the plan")
    require((receipt.get("plan") or {}).get("path") == str(plan_path),
            "Ensemble receipt plan path differs")
    authenticate_descriptor(plan_path, receipt.get("plan"), "ensemble plan")
    implementation_path = Path(__file__).resolve()
    require((receipt.get("implementation") or {}).get("path")
            == str(implementation_path), "Ensemble implementation path differs")
    authenticate_descriptor(implementation_path, receipt.get("implementation"),
                            "ensemble implementation")
    sources = [
        _authenticate_source(project_dir, public_root, plan["context"], tag)
        for tag in plan["source_tags"]
    ]
    validate_source_contracts(sources, plan)
    recorded_sources = receipt.get("sources")
    require(type(recorded_sources) is list and len(recorded_sources) == len(sources),
            "Ensemble receipt source list differs")
    for source, recorded in zip(sources, recorded_sources, strict=True):
        require(recorded.get("tag") == source.tag
                and Path(str(recorded.get("run_directory", ""))) == source.run_dir,
                f"Ensemble source identity differs: {source.tag}")
        for key, descriptor in source.descriptors.items():
            require((recorded.get("artifacts") or {}).get(key, {}).get("path")
                    == descriptor["path"],
                    f"Ensemble source path differs: {source.tag}/{key}")
            require(descriptor_content(descriptor, key)
                    == descriptor_content((recorded.get("artifacts") or {}).get(key), key),
                    f"Ensemble source descriptor differs: {source.tag}/{key}")
    output_path = regular_file(run_dir / "prediction.h5ad", "ensemble prediction")
    require((receipt.get("output") or {}).get("path") == str(output_path),
            "Ensemble output path differs")
    authenticate_descriptor(output_path, receipt.get("output"), "ensemble prediction")
    verify_prediction_exact_mean(
        [source.prediction for source in sources], output_path, plan=plan
    )
    contract = receipt.get("contract")
    require(contract == {
                "leaderboard_submission_authorized": False,
                "metric_based_source_selection": False,
                "output_cell_libraries_exact": True,
                "output_exact_deterministic_mean": True,
                "source_cell_axes_identical": True,
                "source_cell_libraries_identical": True,
                "source_gene_axes_identical": True,
                "source_transport_receipts_revalidated": True,
                "shared_generator_inputs_identical": True,
                "treated_truth_read": False,
                "uniform_weights": True,
            },
            "Ensemble receipt contract is incomplete")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--project-dir", type=Path, required=True)
        subparser.add_argument("--public-root", type=Path, required=True)
        subparser.add_argument("--plan", type=Path, required=True)
        if command == "build":
            subparser.add_argument("--max-loaded-elements", type=int, default=20_000_000)
        else:
            subparser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "build":
        require(args.max_loaded_elements > 0, "max-loaded-elements must be positive")
        receipt_path = build(
            project_dir=args.project_dir,
            public_root=args.public_root,
            plan_path=args.plan,
            max_loaded_elements=args.max_loaded_elements,
        )
        payload = load_json(receipt_path, "ensemble receipt")
    else:
        payload = verify(
            project_dir=args.project_dir,
            public_root=args.public_root,
            plan_path=args.plan,
            run_dir=args.run_dir,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
