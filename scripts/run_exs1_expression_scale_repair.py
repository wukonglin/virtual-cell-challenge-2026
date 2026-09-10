#!/usr/bin/env python3
"""Run the truth-blind EXS-1 count sweep or evaluate its sealed scores."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp

import exs1_expression_scale_core as core
from exs1_local_io import atomic_write_json, concatenate_paths
from exs1_local_io import describe_file, require, sha256_file
from exs1_score_adapter import (
    MSE_METRIC,
    NMAE_METRIC,
    _mean_summary,
    _read_results,
    _validate_cell_eval_sidecars,
)


PDS_METRIC = core.PDS_METRIC
REGISTERED_GATE = core.REGISTERED_GATE
attenuate_count_row = core.attenuate_count_row


def _library_summary(values: np.ndarray) -> dict[str, Any]:
    require(
        values.ndim == 1 and len(values) > 0,
        "Library vector is empty",
    )
    require(
        np.all(values > 0),
        "Library vector contains a non-positive value",
    )
    return {
        "cells": int(len(values)),
        "minimum": int(values.min()),
        "maximum": int(values.max()),
        "mean": float(values.mean(dtype=np.float64)),
        "total": int(values.sum(dtype=np.int64)),
    }


def _validate_transform_contract(
    prediction: ad.AnnData,
    controls: ad.AnnData,
    *,
    chunk_rows: int,
) -> dict[str, Any]:
    """Validate truth-blind axes, counts, and source-control bindings."""

    require(
        prediction.n_obs > 0 and prediction.n_vars > 0,
        "Prediction is empty",
    )
    require(
        controls.n_obs > 0 and controls.n_vars > 0,
        "Controls input is empty",
    )
    for label, data in (
        ("prediction", prediction),
        ("controls", controls),
    ):
        require(
            data.obs_names.is_unique,
            f"{label} cell IDs are not unique",
        )
        require(
            data.var_names.is_unique,
            f"{label} gene IDs are not unique",
        )
        require(
            {"context", "target_gene"}.issubset(data.obs.columns),
            f"{label} lacks context or target_gene",
        )
        require(
            str(getattr(data.X, "format", "")) == "csr",
            f"{label} matrix is not backed CSR",
        )
        require(
            np.issubdtype(np.dtype(data.X.dtype), np.integer),
            f"{label} matrix is not stored as integer counts",
        )
    require(
        set(core.REQUIRED_OBS_COLUMNS).issubset(
            prediction.obs.columns
        ),
        "Prediction lacks source-control pairing columns",
    )
    genes = prediction.var_names.astype(str).tolist()
    require(
        controls.var_names.astype(str).tolist() == genes,
        "Prediction and controls gene axes differ",
    )

    prediction_contexts = set(
        prediction.obs["context"].astype(str)
    )
    control_contexts = set(controls.obs["context"].astype(str))
    require(
        len(prediction_contexts) == len(control_contexts) == 1,
        "Transform inputs must each contain exactly one context",
    )
    require(
        prediction_contexts == control_contexts,
        "Prediction and controls contexts differ",
    )
    context = next(iter(prediction_contexts))
    control_contract = controls.uns.get("public_validation", {})
    require(
        isinstance(control_contract, dict)
        and control_contract.get("schema") == core.DATA_SCHEMA
        and control_contract.get("role")
        == "controls-only-generator-input"
        and isinstance(
            control_contract.get("sealed_treated_profiles_present"),
            (bool, np.bool_),
        )
        and not bool(
            control_contract["sealed_treated_profiles_present"]
        ),
        "Controls input lacks the controls-only firewall role",
    )
    control_labels = set(
        controls.obs["target_gene"].astype(str)
    )
    prediction_labels = prediction.obs["target_gene"].astype(str)
    require(
        control_labels == {core.CONTROL_LABEL},
        "Controls-only input contains treated cells",
    )
    require(
        core.CONTROL_LABEL not in set(prediction_labels),
        "Treated prediction unexpectedly contains controls",
    )
    require(
        prediction_labels.ne("").all(),
        "Prediction contains an empty target label",
    )

    numeric_rows = pd.to_numeric(
        prediction.obs["source_control_row"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    require(
        np.isfinite(numeric_rows).all()
        and np.equal(numeric_rows, np.floor(numeric_rows)).all(),
        "source_control_row is not a finite integer axis",
    )
    source_rows = numeric_rows.astype(np.int64)
    require(
        np.all(
            (source_rows >= 0)
            & (source_rows < controls.n_obs)
        ),
        "source_control_row is outside the controls axis",
    )
    source_ids = (
        prediction.obs["source_control_cell_id"]
        .astype(str)
        .to_numpy()
    )
    expected_ids = (
        controls.obs_names.astype(str).to_numpy()[source_rows]
    )
    require(
        np.array_equal(source_ids, expected_ids),
        "source_control_cell_id does not match source_control_row",
    )

    control_counts = core._canonical_counts(
        controls.X[:, :], "controls matrix"
    )
    control_libraries = np.asarray(
        control_counts.sum(axis=1), dtype=np.int64
    ).reshape(-1)
    require(
        np.all(control_libraries > 0),
        "Controls input contains a zero-library cell",
    )
    prediction_libraries = np.empty(
        prediction.n_obs, dtype=np.int64
    )
    for start in range(0, prediction.n_obs, chunk_rows):
        stop = min(start + chunk_rows, prediction.n_obs)
        block = core._canonical_counts(
            prediction.X[start:stop, :],
            "prediction count block",
        )
        prediction_libraries[start:stop] = np.asarray(
            block.sum(axis=1), dtype=np.int64
        ).reshape(-1)
    require(
        np.all(prediction_libraries > 0),
        "Prediction contains a zero-library cell",
    )
    require(
        int(prediction_libraries.max())
        <= np.iinfo(np.int32).max
        and int(control_libraries.max())
        <= np.iinfo(np.int32).max,
        "An input cell library exceeds the int32 output contract",
    )

    targets = list(dict.fromkeys(prediction_labels.tolist()))
    group_sizes_series = prediction.obs.groupby(
        ["context", "target_gene"],
        observed=True,
        sort=False,
    ).size()
    group_sizes = {
        f"{str(group_context)}|{str(target)}": int(count)
        for (group_context, target), count
        in group_sizes_series.items()
    }
    require(
        len(group_sizes) == len(targets),
        "Prediction target labels do not define one group each",
    )
    return {
        "context": context,
        "genes": genes,
        "targets": targets,
        "group_sizes": group_sizes,
        "source_rows": source_rows,
        "control_counts": control_counts,
        "prediction_libraries": prediction_libraries,
        "paired_control_libraries": control_libraries[source_rows],
        "axis_hashes": {
            "cell_ids_sha256": core._axis_hash(
                prediction.obs_names.astype(str), "cell_ids"
            ),
            "gene_ids_sha256": core._axis_hash(
                genes, "gene_ids"
            ),
            "target_labels_sha256": core._axis_hash(
                prediction_labels, "target_labels"
            ),
            "source_control_rows_sha256": hashlib.sha256(
                source_rows.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "source_control_cell_ids_sha256": core._axis_hash(
                source_ids, "source_control_cell_ids"
            ),
        },
    }


def _transform_block(
    prediction_block: sp.csr_matrix,
    control_block: sp.csr_matrix,
    *,
    attenuation: float,
    library_policy: str,
) -> sp.csr_matrix:
    require(
        prediction_block.shape == control_block.shape,
        "Prediction/control block shapes differ",
    )
    rows = [
        core.attenuate_count_row(
            prediction_block[row],
            control_block[row],
            attenuation=attenuation,
            library_policy=library_policy,
            n_genes=prediction_block.shape[1],
        )
        for row in range(prediction_block.shape[0])
    ]
    require(bool(rows), "Cannot transform an empty block")
    result = sp.vstack(rows, format="csr", dtype=np.int32)
    result.sort_indices()
    return result


def _write_arm(
    *,
    prediction: ad.AnnData,
    contract: dict[str, Any],
    staging_root: Path,
    final_root: Path,
    factor: Decimal,
    policy: str,
    chunk_rows: int,
    concat_max_loaded_elements: int,
    compression: str,
) -> dict[str, Any]:
    arm = core._arm_name(factor, policy)
    arm_staging = staging_root / "arms" / arm
    arm_staging.mkdir(parents=True)
    chunk_dir = staging_root / "_chunks" / arm
    chunk_dir.mkdir(parents=True)
    chunk_paths: list[Path] = []
    factor_float = float(factor)
    control_counts: sp.csr_matrix = contract["control_counts"]
    source_rows: np.ndarray = contract["source_rows"]

    for ordinal, start in enumerate(
        range(0, prediction.n_obs, chunk_rows)
    ):
        stop = min(start + chunk_rows, prediction.n_obs)
        pred_block = core._canonical_counts(
            prediction.X[start:stop, :],
            f"{arm} prediction block",
        )
        paired_controls = control_counts[
            source_rows[start:stop], :
        ]
        output_block = _transform_block(
            pred_block,
            paired_controls,
            attenuation=factor_float,
            library_policy=policy,
        )
        chunk = ad.AnnData(
            X=output_block,
            obs=prediction.obs.iloc[start:stop].copy(),
            var=prediction.var.copy(),
        )
        chunk.uns["exs1"] = {
            "schema": core.SCHEMA,
            "arm": arm,
            "attenuation": f"{factor:.4f}",
            "library_policy": policy,
            "treated_truth_profiles_read": False,
            "integerization": (
                "largest-remainder-v1; descending fractional "
                "remainder; ascending gene-index tie break"
            ),
        }
        chunk_path = (
            chunk_dir / f"chunk_{ordinal:06d}.h5ad"
        )
        chunk.write_h5ad(
            chunk_path,
            compression=(
                None if compression == "none" else compression
            ),
        )
        chunk_paths.append(chunk_path)

    output_staging = arm_staging / "prediction.h5ad"
    concatenate_paths(
        chunk_paths,
        output_staging,
        max_loaded_elements=concat_max_loaded_elements,
    )
    shutil.rmtree(chunk_dir)

    output = ad.read_h5ad(output_staging, backed="r")
    try:
        require(
            tuple(output.shape) == tuple(prediction.shape),
            f"{arm} changed matrix shape",
        )
        require(
            output.obs_names.astype(str).tolist()
            == prediction.obs_names.astype(str).tolist(),
            f"{arm} changed the cell axis",
        )
        require(
            output.var_names.astype(str).tolist()
            == prediction.var_names.astype(str).tolist(),
            f"{arm} changed the gene axis",
        )
        for column in core.REQUIRED_OBS_COLUMNS:
            require(
                output.obs[column].astype(str).tolist()
                == prediction.obs[column].astype(str).tolist(),
                (
                    f"{arm} changed observation axis "
                    f"column {column}"
                ),
            )
        require(
            str(getattr(output.X, "format", "")) == "csr"
            and np.issubdtype(
                np.dtype(output.X.dtype), np.integer
            ),
            f"{arm} is not integer CSR counts",
        )
        output_libraries = np.empty(
            output.n_obs, dtype=np.int64
        )
        observed_nnz = 0
        for start in range(0, output.n_obs, chunk_rows):
            stop = min(start + chunk_rows, output.n_obs)
            block = core._canonical_counts(
                output.X[start:stop, :],
                f"{arm} validation block",
            )
            output_libraries[start:stop] = np.asarray(
                block.sum(axis=1), dtype=np.int64
            ).reshape(-1)
            observed_nnz += int(block.nnz)
        expected_libraries = (
            contract["prediction_libraries"]
            if policy == "keep_prediction"
            else contract["paired_control_libraries"]
        )
        require(
            np.array_equal(
                output_libraries, expected_libraries
            ),
            f"{arm} violated its per-cell library policy",
        )
        logical_hash = core._logical_csr_hash(
            output, chunk_rows
        )
    finally:
        output.file.close()

    final_path = (
        final_root / "arms" / arm / "prediction.h5ad"
    )
    stat = output_staging.stat()
    descriptor = {
        "path": str(final_path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(output_staging),
        "logical_csr_sha256": logical_hash,
    }
    endpoint_check: bool | None = None
    if (
        factor == Decimal("1.0000")
        and policy == "keep_prediction"
    ):
        endpoint_check = (
            logical_hash
            == contract["prediction_logical_csr_sha256"]
        )
        require(
            endpoint_check,
            "Identity endpoint differs from prediction",
        )
    return {
        "arm": arm,
        "attenuation": f"{factor:.4f}",
        "library_policy": policy,
        "transformation": {
            "composition": (
                "q=(1-attenuation)*(paired_control/"
                "control_library)+attenuation*(prediction/"
                "prediction_library)"
            ),
            "output_library": (
                "prediction_library"
                if policy == "keep_prediction"
                else "paired_source_control_library"
            ),
            "integerization": (
                "largest-remainder-v1; descending fractional "
                "remainder; ascending gene-index tie break"
            ),
        },
        "output_prediction": descriptor,
        "qc": {
            "shape_exact": True,
            "cell_axis_exact": True,
            "target_group_axis_exact": True,
            "gene_axis_exact": True,
            "source_control_pairing_exact": True,
            "integer_nonnegative_csr_counts": True,
            "per_cell_library_policy_exact": True,
            "identity_endpoint_exact": endpoint_check,
            "nnz": observed_nnz,
            "output_libraries": _library_summary(
                output_libraries
            ),
        },
        "scores": {
            MSE_METRIC: None,
            PDS_METRIC: None,
            NMAE_METRIC: None,
        },
        "gate": {
            "passed": None,
            "reason": "not_scored",
        },
    }


def _software() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "anndata": ad.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }


def run_transform(
    args: Any, invocation: Sequence[str]
) -> Path:
    require(args.chunk_rows > 0, "chunk-rows must be positive")
    require(
        args.concat_max_loaded_elements > 0,
        "concat-max-loaded-elements must be positive",
    )
    factors = core._parse_attenuations(args.attenuation)
    raw_policies = (
        tuple(args.library_policy)
        if args.library_policy is not None
        else core.LIBRARY_POLICIES
    )
    require(
        len(raw_policies) == len(set(raw_policies)),
        "Library policies contain duplicates",
    )
    policies = tuple(
        policy
        for policy in core.LIBRARY_POLICIES
        if policy in raw_policies
    )
    require(
        bool(policies),
        "At least one library policy is required",
    )

    output_root = args.output_dir.resolve()
    require(
        not args.output_dir.is_symlink(),
        "output-dir must not be a symbolic link",
    )
    require(
        not output_root.exists(),
        f"Refusing to overwrite output directory: {output_root}",
    )
    prediction_file = core._authenticate_regular_file(
        args.prediction,
        args.prediction_expected_sha256,
        "prediction",
    )
    controls_file = core._authenticate_regular_file(
        args.controls_only,
        args.controls_expected_sha256,
        "controls-only input",
    )
    require(
        prediction_file.path != controls_file.path,
        "Prediction and controls-only inputs must be distinct",
    )

    prediction = ad.read_h5ad(
        prediction_file.path, backed="r"
    )
    controls = ad.read_h5ad(
        controls_file.path, backed="r"
    )
    staging_root: Path | None = None
    try:
        contract = _validate_transform_contract(
            prediction,
            controls,
            chunk_rows=args.chunk_rows,
        )
        contract["prediction_logical_csr_sha256"] = (
            core._logical_csr_hash(
                prediction, args.chunk_rows
            )
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{output_root.name}.exs1.",
                dir=output_root.parent,
            )
        )
        arms: dict[str, Any] = {}
        for factor in factors:
            for policy in policies:
                record = _write_arm(
                    prediction=prediction,
                    contract=contract,
                    staging_root=staging_root,
                    final_root=output_root,
                    factor=factor,
                    policy=policy,
                    chunk_rows=args.chunk_rows,
                    concat_max_loaded_elements=(
                        args.concat_max_loaded_elements
                    ),
                    compression=args.compression,
                )
                arms[record["arm"]] = record

        core._reauthenticate(
            prediction_file, "prediction"
        )
        core._reauthenticate(
            controls_file, "controls-only input"
        )
        receipt = {
            "schema": core.SCHEMA,
            "status": "transformed_unscored",
            "experiment": "EXS-1",
            "context": contract["context"],
            "method": {
                "hypothesis": (
                    "Raw expression error is repairable by effect "
                    "attenuation and per-cell library calibration "
                    "alone."
                ),
                "attenuation_grid": [
                    f"{factor:.4f}" for factor in factors
                ],
                "attenuation_grid_status": (
                    "implementation-frozen-v1"
                    if args.attenuation is None
                    else (
                        "explicit-command-line-subset-or-override"
                    )
                ),
                "library_policies": list(policies),
                "cpu_only": True,
            },
            "registered_gate": core.REGISTERED_GATE,
            "gate_result": {
                "passed": None,
                "reason": (
                    "No scorer results were supplied to the "
                    "truth-blind transform phase."
                ),
            },
            "firewall": {
                "transform_parser_accepts_treated_truth": False,
                "treated_truth_profiles_read": False,
                "controls_role_required": (
                    "controls-only-generator-input"
                ),
                "sealed_treated_profiles_present_required": False,
            },
            "axes": {
                "cells": int(prediction.n_obs),
                "genes": int(prediction.n_vars),
                "targets": len(contract["targets"]),
                "target_order_first_occurrence": (
                    contract["targets"]
                ),
                "group_sizes": contract["group_sizes"],
                **contract["axis_hashes"],
            },
            "libraries": {
                "prediction": _library_summary(
                    contract["prediction_libraries"]
                ),
                "paired_source_control": _library_summary(
                    contract["paired_control_libraries"]
                ),
            },
            "arms": arms,
            "provenance": {
                "prediction": (
                    prediction_file.descriptor()
                ),
                "controls_only": (
                    controls_file.descriptor()
                ),
                "prediction_logical_csr_sha256": contract[
                    "prediction_logical_csr_sha256"
                ],
                "script": describe_file(
                    Path(__file__).resolve()
                ),
                "core_module": describe_file(
                    Path(core.__file__).resolve()
                ),
                "invocation": list(invocation),
            },
            "software": _software(),
        }
        receipt_path = staging_root / "experiment.json"
        atomic_write_json(receipt_path, receipt)
        require(
            not output_root.exists(),
            f"Output directory appeared during run: {output_root}",
        )
        os.replace(staging_root, output_root)
        staging_root = None
    finally:
        prediction.file.close()
        controls.file.close()
        if (
            staging_root is not None
            and staging_root.exists()
        ):
            shutil.rmtree(staging_root)
    return output_root / "experiment.json"


def evaluate_registered_gate(
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Apply exactly the registered MSE/PDS/NMAE thresholds."""

    required = {MSE_METRIC, PDS_METRIC, NMAE_METRIC}
    require(
        set(metrics) == required,
        "Gate metrics are not the exact registered set",
    )
    numeric = {
        name: float(value) for name, value in metrics.items()
    }
    require(
        all(np.isfinite(value) for value in numeric.values()),
        "Gate metric is non-finite",
    )
    mse_ok = numeric[MSE_METRIC] <= 0.900
    pds_difference = abs(
        numeric[PDS_METRIC] - 0.564526198439242
    )
    pds_ok = pds_difference <= 1e-6
    nmae_ok = (
        numeric[NMAE_METRIC] <= 1.017691308794254
    )
    return {
        "checks": {
            "mse_at_most_0.900": bool(mse_ok),
            "pds_within_1e-6_of_0.564526198439242": (
                bool(pds_ok)
            ),
            "nmae_at_most_1.017691308794254": (
                bool(nmae_ok)
            ),
        },
        "pds_absolute_difference": pds_difference,
        "passed": bool(mse_ok and pds_ok and nmae_ok),
    }


def _git(
    checkout: Path, *arguments: str
) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    require(
        result.returncode == 0,
        (
            f"git {' '.join(arguments)} failed for "
            f"{checkout}: {result.stderr.strip()}"
        ),
    )
    return result.stdout.strip()


def _authenticate_scorer(checkout: Path) -> dict[str, Any]:
    canonical = checkout.resolve()
    require(
        canonical.is_dir(),
        f"Missing cell-eval2 checkout: {canonical}",
    )
    require(
        (canonical / ".git").exists(),
        "cell-eval2 checkout lacks Git metadata",
    )
    commit = _git(canonical, "rev-parse", "HEAD")
    require(
        commit == core.EXPECTED_CELL_EVAL_COMMIT,
        "cell-eval2 checkout is not at the pinned commit",
    )
    require(
        _git(
            canonical,
            "status",
            "--porcelain",
            "--untracked-files=normal",
        )
        == "",
        "cell-eval2 checkout is dirty",
    )
    pyproject = canonical / "pyproject.toml"
    require(
        pyproject.is_file(),
        "cell-eval2 pyproject.toml is missing",
    )
    text = pyproject.read_text(encoding="utf-8")
    require(
        (
            f'version = "{core.EXPECTED_CELL_EVAL_VERSION}"'
            in text
        ),
        "cell-eval2 declared version differs from pin",
    )
    return {
        "path": str(canonical),
        "git_commit": commit,
        "git_clean": True,
        "declared_version": (
            core.EXPECTED_CELL_EVAL_VERSION
        ),
        "required_pdex_version": (
            core.EXPECTED_PDEX_VERSION
        ),
        "pyproject": describe_file(pyproject),
    }


def _validate_run_meta(
    path: Path, truth_view: dict[str, Any]
) -> dict[str, Any]:
    payload = core._load_json(path, "cell-eval2 run metadata")
    require(
        payload.get("resolved_device") == "cpu",
        "Scorer did not resolve to CPU",
    )
    require(
        payload.get("resolved_de_backend") == "pdex",
        "Scorer did not use pdex",
    )
    require(
        payload.get("comparator") == "bulk_lognorm",
        "Scorer did not use bulk_lognorm",
    )
    require(
        payload.get("source_fingerprint_strict") is True,
        "Scorer strict source fingerprinting is disabled",
    )
    require(
        payload.get("input_type_real_effective") == "counts"
        and payload.get("input_type_pred_effective")
        == "counts",
        "Scorer inputs were not resolved as counts",
    )
    require(
        payload.get("cell_eval2_version")
        == core.EXPECTED_CELL_EVAL_VERSION,
        "Scorer runtime version differs from pin",
    )
    pdex_version = (
        ((payload.get("environment") or {}).get("pdex") or {})
        .get("version")
    )
    require(
        pdex_version == core.EXPECTED_PDEX_VERSION,
        "Scorer pdex runtime version differs from pin",
    )
    config_digest = payload.get("config_digest")
    require(
        isinstance(config_digest, str)
        and core.SHA256_RE.fullmatch(config_digest)
        is not None,
        "Scorer config digest is missing or malformed",
    )
    source_fingerprint = payload.get("source_fingerprint")
    require(
        isinstance(source_fingerprint, str)
        and bool(source_fingerprint),
        "Scorer source fingerprint is absent",
    )
    require(
        Path(str(payload.get("source", ""))).resolve()
        == Path(str(truth_view["path"])).resolve(),
        "Scorer truth source differs from the sealed truth view",
    )
    return {
        "descriptor": describe_file(path),
        "config_digest": config_digest,
        "source_fingerprint": source_fingerprint,
        "resolved_device": "cpu",
        "resolved_de_backend": "pdex",
        "comparator": "bulk_lognorm",
        "input_type_real_effective": "counts",
        "input_type_pred_effective": "counts",
        "cell_eval2_version": (
            core.EXPECTED_CELL_EVAL_VERSION
        ),
        "pdex_version": core.EXPECTED_PDEX_VERSION,
    }


def _authenticate_views(
    *,
    views_file: core.AuthenticatedFile,
    arm_descriptor: dict[str, Any],
    context: str,
) -> dict[str, Any]:
    payload = core._load_json(
        views_file.path, "scoring views receipt"
    )
    require(
        payload.get("schema") == core.VIEWS_SCHEMA,
        "Bad scoring views schema",
    )
    require(
        payload.get("context") == context,
        "Scoring views context differs from EXS-1",
    )
    contract = payload.get("contract")
    require(
        isinstance(contract, dict)
        and contract.get("model_read_treated_truth") is False
        and contract.get("input_type")
        == "raw-nonnegative-integer-counts",
        "Scoring views firewall or count contract differs",
    )
    provenance = payload.get("provenance")
    require(
        isinstance(provenance, dict),
        "Scoring views provenance is absent",
    )
    core._same_descriptor(
        provenance.get("prediction"),
        arm_descriptor,
        "EXS-1 source prediction",
    )
    authenticated: dict[str, Any] = {}
    for name in (
        "prediction",
        "controls_only",
        "sealed_truth",
        "common_genes",
        "common_genes_json",
        "output_prediction",
        "output_truth",
    ):
        require(
            name in provenance,
            f"Scoring views provenance lacks {name}",
        )
        authenticated[name] = core._authenticate_descriptor(
            provenance[name],
            f"scoring views {name}",
        )
    return {
        "receipt": views_file.descriptor(),
        "files": authenticated,
        "firewall": {
            "model_read_treated_truth": False,
            "input_type": "raw-nonnegative-integer-counts",
        },
    }


def _score_bundle(
    *,
    arm: str,
    results_file: core.AuthenticatedFile,
    views_file: core.AuthenticatedFile,
    arm_descriptor: dict[str, Any],
    context: str,
    targets: set[str],
) -> dict[str, Any]:
    views = _authenticate_views(
        views_file=views_file,
        arm_descriptor=arm_descriptor,
        context=context,
    )
    frame = _read_results(results_file.path, targets)
    summary = _mean_summary(
        frame, expected_targets=len(targets)
    )
    sidecars = _validate_cell_eval_sidecars(
        results_file.path,
        frame,
        summary,
        expected_targets=len(targets),
    )
    metrics: dict[str, float] = {}
    for metric in (MSE_METRIC, PDS_METRIC, NMAE_METRIC):
        value = summary.get(metric, {}).get("mean")
        require(
            value is not None and np.isfinite(float(value)),
            f"Registered metric is unavailable for {arm}: {metric}",
        )
        metrics[metric] = float(value)
    run_meta = _validate_run_meta(
        results_file.path.with_name("run_meta.json"),
        views["files"]["output_truth"],
    )
    core._reauthenticate(
        results_file, f"{arm} results"
    )
    core._reauthenticate(
        views_file, f"{arm} views receipt"
    )
    return {
        "scores": metrics,
        "gate": evaluate_registered_gate(metrics),
        "provenance": {
            "results": results_file.descriptor(),
            "views": views,
            "sidecars": sidecars,
            "run_meta": run_meta,
        },
    }


def run_evaluate(
    args: Any, invocation: Sequence[str]
) -> Path:
    require(
        args.output_json.suffix == ".json",
        "output-json must end in .json",
    )
    require(
        not args.output_json.is_symlink()
        and not args.output_json.exists(),
        f"Refusing to overwrite output: {args.output_json}",
    )
    experiment_file = core._authenticate_regular_file(
        args.experiment_json,
        args.experiment_expected_sha256,
        "EXS-1 experiment receipt",
    )
    experiment = core._load_json(
        experiment_file.path, "EXS-1 experiment receipt"
    )
    require(
        experiment.get("schema") == core.SCHEMA
        and experiment.get("experiment") == "EXS-1"
        and experiment.get("status") == "transformed_unscored",
        "Input is not an unscored EXS-1 transform receipt",
    )
    require(
        experiment.get("registered_gate")
        == core.REGISTERED_GATE,
        "EXS-1 registered gate differs from this implementation",
    )
    firewall = experiment.get("firewall")
    require(
        isinstance(firewall, dict)
        and firewall.get("treated_truth_profiles_read")
        is False
        and firewall.get(
            "transform_parser_accepts_treated_truth"
        )
        is False,
        "EXS-1 transform firewall did not pass",
    )
    arms = experiment.get("arms")
    require(
        isinstance(arms, dict) and bool(arms),
        "EXS-1 receipt has no arms",
    )
    targets_list = (
        (experiment.get("axes") or {})
        .get("target_order_first_occurrence")
    )
    require(
        isinstance(targets_list, list)
        and len(targets_list)
        == int((experiment.get("axes") or {}).get("targets", -1))
        and len(targets_list) == len(set(targets_list))
        and all(isinstance(value, str) and value for value in targets_list),
        "EXS-1 target axis is malformed",
    )
    targets = set(targets_list)
    context = str(experiment.get("context", ""))
    require(bool(context), "EXS-1 context is absent")
    scorer = _authenticate_scorer(
        args.cell_eval_checkout
    )

    supplied: dict[str, tuple[core.AuthenticatedFile, core.AuthenticatedFile]] = {}
    for (
        arm,
        raw_results,
        results_hash,
        raw_views,
        views_hash,
    ) in args.score:
        require(arm in arms, f"Unknown EXS-1 arm: {arm}")
        require(
            arm not in supplied,
            f"Duplicate score bundle for arm: {arm}",
        )
        supplied[arm] = (
            core._authenticate_regular_file(
                Path(raw_results),
                results_hash,
                f"{arm} results",
            ),
            core._authenticate_regular_file(
                Path(raw_views),
                views_hash,
                f"{arm} views receipt",
            ),
        )
    if not args.allow_partial:
        require(
            set(supplied) == set(arms),
            (
                "Complete evaluation requires exactly one score "
                "bundle for every arm"
            ),
        )

    evaluated_arms: dict[str, Any] = {}
    scorer_identities: list[tuple[str, str]] = []
    for arm, transform_record in arms.items():
        if arm not in supplied:
            evaluated_arms[arm] = {
                "scores": {
                    MSE_METRIC: None,
                    PDS_METRIC: None,
                    NMAE_METRIC: None,
                },
                "gate": {
                    "passed": None,
                    "reason": "score_bundle_not_supplied",
                },
                "provenance": None,
            }
            continue
        results_file, views_file = supplied[arm]
        evaluated = _score_bundle(
            arm=arm,
            results_file=results_file,
            views_file=views_file,
            arm_descriptor=transform_record[
                "output_prediction"
            ],
            context=context,
            targets=targets,
        )
        evaluated_arms[arm] = evaluated
        run_meta = evaluated["provenance"]["run_meta"]
        scorer_identities.append(
            (
                run_meta["config_digest"],
                run_meta["source_fingerprint"],
            )
        )
    require(
        len(set(scorer_identities)) <= 1,
        "EXS-1 arms used different scorer configurations or truth",
    )

    arm_passes = [
        record["gate"]["passed"]
        for record in evaluated_arms.values()
    ]
    if any(value is True for value in arm_passes):
        passed: bool | None = True
        status = "passed"
    elif all(value is not None for value in arm_passes):
        passed = False
        status = "failed"
    else:
        passed = None
        status = "incomplete"
    report = {
        "schema": core.EVALUATION_SCHEMA,
        "status": status,
        "experiment": "EXS-1",
        "context": context,
        "registered_gate": core.REGISTERED_GATE,
        "gate_result": {
            "passed": passed,
            "scored_arms": len(supplied),
            "total_arms": len(arms),
            "unscored_arms": [
                arm for arm in arms if arm not in supplied
            ],
        },
        "arms": evaluated_arms,
        "provenance": {
            "experiment": experiment_file.descriptor(),
            "cell_eval2": scorer,
            "script": describe_file(
                Path(__file__).resolve()
            ),
            "core_module": describe_file(
                Path(core.__file__).resolve()
            ),
            "invocation": list(invocation),
        },
        "software": _software(),
    }
    core._reauthenticate(
        experiment_file, "EXS-1 experiment receipt"
    )
    args.output_json.parent.mkdir(
        parents=True, exist_ok=True
    )
    atomic_write_json(args.output_json, report)
    return args.output_json


def main(argv: Sequence[str] | None = None) -> None:
    args = core.parse_args(argv)
    effective_argv = (
        list(argv) if argv is not None else sys.argv[1:]
    )
    invocation = [
        sys.executable,
        str(Path(__file__).resolve()),
        *effective_argv,
    ]
    if args.command == "transform":
        output = run_transform(args, invocation)
    elif args.command == "evaluate":
        output = run_evaluate(args, invocation)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
