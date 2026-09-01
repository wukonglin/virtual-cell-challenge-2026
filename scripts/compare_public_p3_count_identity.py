#!/usr/bin/env python3
"""Authenticate and compare a matched P3 alpha pair without reading truth.

The command verifies both immutable generation receipts, their candidate
specifications, generation reports, predictions, and every declared generator
input.  It then requires identical axes and source-control pairing, permits
only ``residual_alpha`` to differ in generator configuration, and streams the
two backed CSR matrices in bounded row chunks.

Rows routed as ``held_target`` by the frozen public manifest must be exactly
count-identical.  Direct rows may differ and are summarized with canonical
logical-matrix hashes and sparse difference counts.  No treated-truth path is
accepted by this CLI.  ``--output-json`` is optional and never overwrites an
existing receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from infer_state_effect_prior import describe_file, require, sha256_file
from public_candidate_v6_contract import (
    OPTIONAL_STRICT_INPUT_KEYS,
    PREDICTION_SCHEMA,
    SPEC_SCHEMA,
    STRICT_PROVENANCE_CONTRACT,
    STRICT_SPEC_INPUT_KEYS,
    VERIFICATION_SCHEMA,
    authenticate_state_source,
    validate_generation_report,
    validate_output_tag,
)


OUTPUT_SCHEMA = "vcc-public-p3-count-identity-v1"
ALLOWED_CONFIGURATION_DIFFERENCE = "residual_alpha"
REQUIRED_OBS_COLUMNS = (
    "context",
    "target_gene",
    "source_control_row",
    "source_control_cell_id",
)
REQUIRED_VERIFICATION_CHECKS = (
    "generator_inputs_authenticated",
    "generator_did_not_read_treated_truth",
    "generation_matches_locked_configuration",
    "prediction_hash_matches_report",
    "prediction_shape_and_groups_match_spec",
    "exact_library_and_non_state_invariants_passed",
    "strict_generator_provenance_authenticated",
)
REQUIRED_SPEC_INPUTS = STRICT_SPEC_INPUT_KEYS


@dataclass(frozen=True)
class Candidate:
    """Authenticated paths and metadata for one generated candidate."""

    spec_path: Path
    verification_path: Path
    prediction_path: Path
    generation_path: Path
    spec: dict[str, Any]
    verification: dict[str, Any]
    generation: dict[str, Any]
    descriptors: dict[str, Any]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    require(path.suffix == ".json", "Output receipt must end in .json")
    require(not path.exists(), f"Refusing to overwrite count-identity receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _descriptor_identity(descriptor: dict[str, Any]) -> tuple[str, int, str]:
    """Return the immutable fields used to match two file descriptors."""

    require(isinstance(descriptor, dict), "File descriptor is not an object")
    path = Path(str(descriptor.get("path", ""))).resolve()
    size = int(descriptor.get("size_bytes", -1))
    digest = str(descriptor.get("sha256", ""))
    require(size >= 0 and len(digest) == 64, f"Incomplete file descriptor: {path}")
    return str(path), size, digest


def _authenticate_file(
    path: Path,
    descriptor: dict[str, Any],
    label: str,
    cache: dict[tuple[str, int, str], dict[str, Any]],
) -> dict[str, Any]:
    """Authenticate one descriptor, deduplicating files shared by both arms."""

    identity = _descriptor_identity(descriptor)
    resolved = path.resolve()
    require(str(resolved) == identity[0], f"{label} path differs from its descriptor")
    require(resolved.is_file(), f"Missing {label}: {resolved}")
    require(resolved.stat().st_size == identity[1], f"{label} size differs from its descriptor")
    if identity in cache:
        return cache[identity]
    observed_digest = sha256_file(resolved)
    require(observed_digest == identity[2], f"{label} SHA-256 differs from its descriptor")
    observed = describe_file(resolved, hash_file=False)
    observed["sha256"] = observed_digest
    cache[identity] = observed
    return observed


def _require_exact_mapping(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    require(left == right, f"Candidate {label} differs between the two arms")


def authenticate_candidate(
    *,
    spec_path: Path,
    verification_path: Path,
    prediction_path: Path,
    cache: dict[tuple[str, int, str], dict[str, Any]],
    label: str,
) -> Candidate:
    """Fail closed unless a candidate and all recorded generator inputs verify."""

    for path, path_label in (
        (spec_path, "specification"),
        (verification_path, "generation verification"),
        (prediction_path, "prediction"),
    ):
        require(path.is_file(), f"Missing {label} {path_label}: {path}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    require(spec.get("schema") == SPEC_SCHEMA, f"Bad {label} candidate specification schema")
    validate_output_tag(str(spec.get("output_tag", "")))
    require(
        verification.get("schema") == VERIFICATION_SCHEMA
        and verification.get("status") == "passed",
        f"Bad {label} generation verification receipt",
    )
    require(
        verification.get("output_tag") == spec.get("output_tag")
        and verification.get("context") == spec.get("context"),
        f"{label} verification identity differs from its specification",
    )
    require(
        verification.get("validation_mode") == "strict-v2",
        f"{label} verification is not a strict v2 receipt",
    )
    _require_exact_mapping(
        verification.get("configuration", {}),
        spec.get("configuration", {}),
        f"{label} verification configuration",
    )
    checks = verification.get("checks", {})
    require(isinstance(checks, dict), f"{label} verification checks are absent")
    require(
        all(checks.get(name) is True for name in REQUIRED_VERIFICATION_CHECKS),
        f"{label} verification receipt lacks a required passing check",
    )
    require(
        all(value is True for value in checks.values()),
        f"{label} verification receipt contains a failed check",
    )

    provenance = verification.get("provenance", {})
    require(isinstance(provenance, dict), f"{label} verification provenance is absent")
    spec_descriptor = provenance.get("spec", {})
    prediction_descriptor = provenance.get("prediction_h5ad", {})
    generation_descriptor = provenance.get("generation_json", {})
    observed_spec = _authenticate_file(
        spec_path, spec_descriptor, f"{label} specification", cache
    )
    observed_prediction = _authenticate_file(
        prediction_path, prediction_descriptor, f"{label} prediction", cache
    )
    generation_path = Path(str(generation_descriptor.get("path", ""))).resolve()
    observed_generation = _authenticate_file(
        generation_path, generation_descriptor, f"{label} generation report", cache
    )
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    require(
        generation.get("schema") == PREDICTION_SCHEMA,
        f"Bad {label} generation report schema",
    )
    validate_generation_report(generation, spec)
    output_descriptor = generation.get("provenance", {}).get("output_h5ad", {})
    require(
        _descriptor_identity(output_descriptor) == _descriptor_identity(prediction_descriptor),
        f"{label} generation and verification prediction descriptors differ",
    )

    inputs = spec.get("inputs", {})
    require(isinstance(inputs, dict), f"{label} specification inputs are absent")
    missing = sorted(set(REQUIRED_SPEC_INPUTS) - set(inputs))
    require(not missing, f"{label} specification lacks inputs: {missing}")
    observed_inputs: dict[str, Any] = {}
    for name in REQUIRED_SPEC_INPUTS:
        descriptor = inputs[name]
        if descriptor is None:
            require(
                name in OPTIONAL_STRICT_INPUT_KEYS,
                f"{label} required specification input is null: {name}",
            )
            observed_inputs[name] = None
            continue
        require(
            isinstance(descriptor, dict),
            f"{label} specification input descriptor is invalid: {name}",
        )
        input_path = Path(str(descriptor.get("path", ""))).resolve()
        observed_inputs[name] = _authenticate_file(
            input_path, descriptor, f"{label} specification input {name}", cache
        )
    state_source = spec.get("state_source")
    require(isinstance(state_source, dict), f"{label} STATE source contract is absent")
    observed_state_source = authenticate_state_source(
        Path(str(state_source.get("path", ""))),
        expected_isolation=state_source.get("runtime_isolation"),
    )
    require(
        observed_state_source == state_source
        and provenance.get("authenticated_state_source") == state_source
        and generation.get("provenance", {}).get("state_source_commit")
        == state_source.get("repository_commit"),
        f"{label} STATE runtime provenance differs",
    )

    firewall = spec.get("firewall", {})
    require(
        firewall.get("generator_inputs_include_sealed_truth") is False
        and firewall.get("treated_profiles_read_while_planning") is False
        and firewall.get("controls_role") == "controls-only-generator-input",
        f"{label} specification does not prove the controls-only firewall",
    )
    data_firewall = generation.get("data_firewall", {})
    require(
        data_firewall.get("sealed_treated_profiles_read") is False
        and data_firewall.get("truth_inputs_read") == [],
        f"{label} generator reports treated-truth access",
    )

    return Candidate(
        spec_path=spec_path.resolve(),
        verification_path=verification_path.resolve(),
        prediction_path=prediction_path.resolve(),
        generation_path=generation_path,
        spec=spec,
        verification=verification,
        generation=generation,
        descriptors={
            "spec": observed_spec,
            "generation_verified": describe_file(verification_path),
            "generation_json": observed_generation,
            "prediction_h5ad": observed_prediction,
            **{f"input_{name}": value for name, value in observed_inputs.items()},
        },
    )


def validate_pair_contract(left: Candidate, right: Candidate) -> dict[str, Any]:
    """Require a matched alpha contrast with no other experimental drift."""

    require(left.spec["context"] == right.spec["context"], "Candidate contexts differ")
    require(
        left.spec["output_tag"] != right.spec["output_tag"],
        "Candidate output tags must be distinct",
    )
    _require_exact_mapping(left.spec.get("axes", {}), right.spec.get("axes", {}), "axes")
    for key in (
        "context_prefix",
        "checkpoint_selection",
        "residual_contract",
        "firewall",
        "provenance_contract",
        "state_source",
    ):
        _require_exact_mapping(left.spec.get(key, {}), right.spec.get(key, {}), key)

    left_inputs = left.spec.get("inputs", {})
    right_inputs = right.spec.get("inputs", {})
    require(set(left_inputs) == set(right_inputs), "Candidate specification input keys differ")
    for name in sorted(left_inputs):
        if left_inputs[name] is None or right_inputs[name] is None:
            require(
                name in OPTIONAL_STRICT_INPUT_KEYS
                and left_inputs[name] is None
                and right_inputs[name] is None,
                f"Candidate optional input presence differs between arms: {name}",
            )
            continue
        require(
            _descriptor_identity(left_inputs[name]) == _descriptor_identity(right_inputs[name]),
            f"Candidate input differs between arms: {name}",
        )

    left_config = left.spec.get("configuration", {})
    right_config = right.spec.get("configuration", {})
    require(set(left_config) == set(right_config), "Candidate configuration keys differ")
    differences = sorted(
        key for key in left_config if left_config[key] != right_config[key]
    )
    require(
        differences == [ALLOWED_CONFIGURATION_DIFFERENCE],
        "The matched pair must differ only in residual_alpha",
    )
    left_alpha = float(left_config[ALLOWED_CONFIGURATION_DIFFERENCE])
    right_alpha = float(right_config[ALLOWED_CONFIGURATION_DIFFERENCE])
    require(
        math.isfinite(left_alpha)
        and math.isfinite(right_alpha)
        and left_alpha >= 0
        and right_alpha >= 0
        and left_alpha != right_alpha,
        "The alpha contrast is invalid",
    )
    require(
        float(left_config.get("target_remaining_fraction", math.nan)) == 0.40
        and float(right_config.get("target_remaining_fraction", math.nan)) == 0.40,
        "The P3 C/D count audit requires target_remaining_fraction 0.40",
    )
    require(
        {left_alpha, right_alpha} == {0.0, 0.10},
        "The P3 C/D count audit requires residual_alpha values 0.00 and 0.10",
    )
    return {
        "context": left.spec["context"],
        "left_tag": left.spec["output_tag"],
        "right_tag": right.spec["output_tag"],
        "left_residual_alpha": left_alpha,
        "right_residual_alpha": right_alpha,
        "sole_configuration_difference": ALLOWED_CONFIGURATION_DIFFERENCE,
    }


def load_frozen_manifest(
    path: Path, left: Candidate, right: Candidate
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """Load the authenticated target routing axis, which contains no effect values."""

    descriptor = left.spec["inputs"]["panel_csv"]
    require(
        _descriptor_identity(descriptor)
        == _descriptor_identity(right.spec["inputs"]["panel_csv"]),
        "Candidate panel CSV descriptors differ",
    )
    require(path.resolve() == Path(str(descriptor["path"])).resolve(), "Manifest path differs from spec")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    require(
        {"target_gene", "validation_route"}.issubset(frame.columns),
        "Frozen manifest lacks target_gene or validation_route",
    )
    targets = frame["target_gene"].astype(str).tolist()
    routes = frame["validation_route"].astype(str).to_numpy()
    require(targets and all(targets), "Frozen target axis is empty")
    require(len(targets) == len(set(targets)), "Frozen target axis contains duplicates")
    require(set(routes) == {"direct", "held_target"}, "Frozen manifest has invalid routes")
    axes = left.spec["axes"]
    require(len(targets) == int(axes["targets"]), "Manifest target count differs from spec")
    direct = routes == "direct"
    require(int(direct.sum()) == int(axes["direct_targets"]), "Direct target count differs")
    require(int((~direct).sum()) == int(axes["held_targets"]), "Held target count differs")
    return targets, direct, {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": str(descriptor["sha256"]),
    }


def _frame_values_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if left.columns.tolist() != right.columns.tolist() or left.shape != right.shape:
        return False
    for column in left.columns:
        left_values = left[column].astype(str).to_numpy()
        right_values = right[column].astype(str).to_numpy()
        if not np.array_equal(left_values, right_values):
            return False
    return True


def validate_axes_and_pairing(
    left: ad.AnnData,
    right: ad.AnnData,
    *,
    context: str,
    targets: list[str],
    cells_per_group: int,
) -> dict[str, Any]:
    """Require exact matrix axes, annotations, and source-control row pairing."""

    expected_shape = (len(targets) * cells_per_group, left.n_vars)
    require(left.shape == right.shape == expected_shape, "Prediction matrix shapes differ")
    require(left.obs_names.is_unique and right.obs_names.is_unique, "Observation names are not unique")
    require(
        np.array_equal(left.obs_names.astype(str), right.obs_names.astype(str)),
        "Observation axes differ",
    )
    require(
        np.array_equal(left.var_names.astype(str), right.var_names.astype(str)),
        "Gene axes differ",
    )
    require(_frame_values_equal(left.var, right.var), "Gene annotations differ")
    require(_frame_values_equal(left.obs, right.obs), "Observation annotations differ")
    require(
        set(REQUIRED_OBS_COLUMNS).issubset(left.obs.columns)
        and set(REQUIRED_OBS_COLUMNS).issubset(right.obs.columns),
        "Prediction obs lacks source-control pairing columns",
    )
    require(
        set(left.obs["context"].astype(str)) == {context},
        "Prediction context differs from the matched contract",
    )
    expected_targets = np.repeat(np.asarray(targets, dtype=object), cells_per_group)
    observed_targets = left.obs["target_gene"].astype(str).to_numpy()
    require(np.array_equal(observed_targets, expected_targets), "Prediction target blocks are reordered")
    require(
        np.array_equal(
            left.obs["source_control_row"].to_numpy(),
            right.obs["source_control_row"].to_numpy(),
        )
        and np.array_equal(
            left.obs["source_control_cell_id"].astype(str).to_numpy(),
            right.obs["source_control_cell_id"].astype(str).to_numpy(),
        ),
        "Source-control pairing differs",
    )
    for matrix, label in ((left.X, "left"), (right.X, "right")):
        require(str(getattr(matrix, "format", "")) == "csr", f"{label} matrix is not backed CSR")
        dtype = np.dtype(matrix.dtype)
        require(np.issubdtype(dtype, np.integer), f"{label} matrix is not integer counts")
    return {
        "shape": list(expected_shape),
        "obs_axis_exact": True,
        "gene_axis_exact": True,
        "obs_annotations_exact": True,
        "gene_annotations_exact": True,
        "source_control_pairing_exact": True,
        "target_block_order_exact": True,
    }


def _canonical_csr(matrix: Any) -> sp.csr_matrix:
    result = sp.csr_matrix(matrix, dtype=np.int64, copy=True)
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    require(
        not result.data.size or (result.data >= 0).all(),
        "Prediction matrix contains negative counts",
    )
    return result


def _update_logical_hash(digest: Any, matrix: sp.csr_matrix) -> None:
    """Hash canonical row coordinates and values independently of HDF5 encoding."""

    for row in range(matrix.shape[0]):
        start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
        count = np.asarray([stop - start], dtype="<u8")
        indices = np.asarray(matrix.indices[start:stop], dtype="<u8")
        values = np.asarray(matrix.data[start:stop], dtype="<i8")
        digest.update(count.tobytes())
        digest.update(indices.tobytes())
        digest.update(values.tobytes())


def _empty_summary(route: str, groups: int, rows: int, genes: int) -> dict[str, Any]:
    header = f"{OUTPUT_SCHEMA}|logical-csr-v1|{route}|{rows}|{genes}".encode("utf-8")
    left_hash = hashlib.sha256(header)
    right_hash = hashlib.sha256(header)
    return {
        "route": route,
        "groups": groups,
        "rows": rows,
        "genes": genes,
        "left_hash": left_hash,
        "right_hash": right_hash,
        "differing_rows": 0,
        "differing_entries": 0,
        "sum_absolute_count_difference": 0,
        "maximum_absolute_count_difference": 0,
        "left_total_counts": 0,
        "right_total_counts": 0,
        "left_nnz": 0,
        "right_nnz": 0,
    }


def stream_compare_counts(
    left: ad.AnnData,
    right: ad.AnnData,
    *,
    direct_mask: np.ndarray,
    cells_per_group: int,
    chunk_rows: int,
) -> dict[str, dict[str, Any]]:
    """Compare backed matrices in bounded chunks and return route summaries."""

    require(chunk_rows > 0, "chunk-rows must be positive")
    require(len(direct_mask) * cells_per_group == left.n_obs, "Route axis differs from matrix rows")
    summaries = {
        "direct": _empty_summary(
            "direct", int(direct_mask.sum()), int(direct_mask.sum()) * cells_per_group, left.n_vars
        ),
        "held_target": _empty_summary(
            "held_target",
            int((~direct_mask).sum()),
            int((~direct_mask).sum()) * cells_per_group,
            left.n_vars,
        ),
    }
    for target_index, is_direct in enumerate(direct_mask.tolist()):
        route = "direct" if is_direct else "held_target"
        summary = summaries[route]
        group_start = target_index * cells_per_group
        group_stop = group_start + cells_per_group
        for start in range(group_start, group_stop, chunk_rows):
            stop = min(start + chunk_rows, group_stop)
            left_block = _canonical_csr(left.X[start:stop, :])
            right_block = _canonical_csr(right.X[start:stop, :])
            require(left_block.shape == right_block.shape, "Streamed block shapes differ")
            _update_logical_hash(summary["left_hash"], left_block)
            _update_logical_hash(summary["right_hash"], right_block)
            inequality = (left_block != right_block).tocsr()
            summary["differing_rows"] += int(np.count_nonzero(np.diff(inequality.indptr)))
            summary["differing_entries"] += int(inequality.nnz)
            delta = left_block - right_block
            delta.eliminate_zeros()
            if delta.nnz:
                absolute = np.abs(delta.data)
                summary["sum_absolute_count_difference"] += int(absolute.sum(dtype=np.int64))
                summary["maximum_absolute_count_difference"] = max(
                    summary["maximum_absolute_count_difference"], int(absolute.max())
                )
            summary["left_total_counts"] += int(left_block.sum(dtype=np.int64))
            summary["right_total_counts"] += int(right_block.sum(dtype=np.int64))
            summary["left_nnz"] += int(left_block.nnz)
            summary["right_nnz"] += int(right_block.nnz)

    finalized: dict[str, dict[str, Any]] = {}
    for route, raw in summaries.items():
        left_digest = raw.pop("left_hash").hexdigest()
        right_digest = raw.pop("right_hash").hexdigest()
        exact = (
            raw["differing_rows"] == 0
            and raw["differing_entries"] == 0
            and left_digest == right_digest
        )
        finalized[route] = {
            **raw,
            "left_logical_csr_sha256": left_digest,
            "right_logical_csr_sha256": right_digest,
            "exact_count_identity": exact,
        }
    return finalized


def compare_candidates(args: argparse.Namespace) -> dict[str, Any]:
    """Build a passing structural receipt or raise on any invariant violation."""

    cache: dict[tuple[str, int, str], dict[str, Any]] = {}
    left = authenticate_candidate(
        spec_path=args.left_spec,
        verification_path=args.left_verification,
        prediction_path=args.left_prediction,
        cache=cache,
        label="left",
    )
    right = authenticate_candidate(
        spec_path=args.right_spec,
        verification_path=args.right_verification,
        prediction_path=args.right_prediction,
        cache=cache,
        label="right",
    )
    contrast = validate_pair_contract(left, right)
    targets, direct, manifest_descriptor = load_frozen_manifest(
        args.manifest, left, right
    )

    left_data = ad.read_h5ad(left.prediction_path, backed="r")
    right_data = ad.read_h5ad(right.prediction_path, backed="r")
    try:
        axes = validate_axes_and_pairing(
            left_data,
            right_data,
            context=contrast["context"],
            targets=targets,
            cells_per_group=int(left.spec["configuration"]["cells_per_group"]),
        )
        matrices = stream_compare_counts(
            left_data,
            right_data,
            direct_mask=direct,
            cells_per_group=int(left.spec["configuration"]["cells_per_group"]),
            chunk_rows=args.chunk_rows,
        )
    finally:
        left_data.file.close()
        right_data.file.close()

    held = matrices["held_target"]
    require(
        held["exact_count_identity"]
        and held["differing_rows"] == 0
        and held["differing_entries"] == 0,
        "Held-target count matrices differ; the P3 alpha pair is invalid",
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "passed",
        "contrast": contrast,
        "manifest": manifest_descriptor,
        "axes_and_pairing": axes,
        "streaming": {
            "chunk_rows": int(args.chunk_rows),
            "both_full_matrices_loaded_simultaneously": False,
            "canonical_hash_encoding": "logical-csr-v1-rowwise-little-endian",
        },
        "count_comparison": matrices,
        "hard_gate": {
            "held_target_count_identity_required": True,
            "held_target_count_identity_passed": True,
            "direct_count_differences_are_diagnostic": True,
        },
        "data_firewall": {
            "treated_truth_read": False,
            "truth_cli_arguments_available": False,
            "routing_metadata_only": True,
        },
        "provenance": {
            "left": left.descriptors,
            "right": right.descriptors,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Frozen public target-routing CSV already authenticated by both specs.",
    )
    for side in ("left", "right"):
        parser.add_argument(
            f"--{side}-spec",
            type=Path,
            required=True,
            help=f"Immutable strict {side} vcc-public-candidate-spec-v2 JSON.",
        )
        parser.add_argument(
            f"--{side}-verification",
            type=Path,
            required=True,
            help=f"Passing {side} vcc-public-candidate-verification-v1 JSON.",
        )
        parser.add_argument(
            f"--{side}-prediction",
            type=Path,
            required=True,
            help=f"Authenticated {side} generated count H5AD.",
        )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=400,
        help="Maximum backed rows loaded from each matrix at once (default: 400).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional immutable passing receipt; existing files are never overwritten.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compare_candidates(args)
    if args.output_json is not None:
        _atomic_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
